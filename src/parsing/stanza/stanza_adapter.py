"""Convert Stanza CRAFT output to ParsedSentence."""
from __future__ import annotations

import re
import json
import importlib.metadata
import inspect
from dataclasses import replace as dc_replace

from src.parsing.syntax import BackendInfo, CorefCluster, ParsedSentence, SyntaxNode
from src.parsing.role_normalizer import RoleNormalizer
from src.system_rules import rule_frozenset, system_rule


# PROTEIN target pattern used by the primary parser backend.

_DEFAULT_TARGET_PATTERN = re.compile(
    f"({system_rule('surface_contract.masked_protein_pattern')})"
)

# Pronouns for coref evidence classification
_PRONOUN_TEXTS = rule_frozenset("reference.pronoun_identity_forms")


def _patch_stanza_nocharlm_lemma_loader() -> bool:
    """Work around Stanza 1.12.2's ``charlm_forward_file`` KeyError for no-charlm lemma models."""

    try:
        from stanza.models.lemma import trainer as lemma_trainer
    except ImportError:
        return False

    Trainer = lemma_trainer.Trainer
    if getattr(Trainer.load, "_ppi_nocharlm_safe", False):
        return False

    try:
        source = inspect.getsource(Trainer.load)
    except OSError:
        source = ""
    has_unsafe_fallback = (
        "args.get('charlm_forward_file', self.args['charlm_forward_file'])" in source
        or 'args.get("charlm_forward_file", self.args["charlm_forward_file"])' in source
    )
    if not has_unsafe_fallback:
        return False

    def _safe_load(self, filename, args, foundation_cache, lemma_classifier_args=None):
        try:
            checkpoint = lemma_trainer.torch.load(
                filename,
                lambda storage, loc: storage,
                weights_only=True,
            )
        except BaseException:
            lemma_trainer.logger.error("Cannot load model from {}".format(filename))
            raise
        self.args = checkpoint["config"]
        if args is not None:
            self.args["charlm_forward_file"] = args.get(
                "charlm_forward_file",
                self.args.get("charlm_forward_file"),
            )
            self.args["charlm_backward_file"] = args.get(
                "charlm_backward_file",
                self.args.get("charlm_backward_file"),
            )
        self.word_dict, self.composite_dict = checkpoint["dicts"]
        if not self.args["dict_only"]:
            self.model = self.build_seq2seq(self.args, None, foundation_cache)
            # strict=False: the CRAFT model may predate the runtime module.
            self.model.load_state_dict(checkpoint["model"], strict=False)
        else:
            self.model = None
        self.vocab = lemma_trainer.MultiVocab.load_state_dict(checkpoint["vocab"])
        self.contextual_lemmatizers = []
        for contextual in checkpoint.get("contextual", []):
            self.contextual_lemmatizers.append(
                lemma_trainer.LemmaClassifier.from_checkpoint(
                    contextual,
                    args=lemma_classifier_args,
                )
            )

    _safe_load._ppi_nocharlm_safe = True
    _safe_load._ppi_original_load = Trainer.load
    Trainer.load = _safe_load
    return True


# StanzaAdapter

class StanzaAdapter:
    """Stanza to ParsedSentence adapter (CRAFT, optional coref and QANom)."""

    def __init__(
        self,
        *,
        package: str = "craft",
        use_coref: bool = False,
        use_nominalization: bool = True,
        qanom_threshold: float = 0.5,
        target_pattern: str = r"PROTEIN\d+",
        use_gpu: bool = True,
        sentence_is_presegmented: bool = True,
    ) -> None:
        self.package = package
        self.use_coref = use_coref
        self.use_gpu = use_gpu
        self.sentence_is_presegmented = sentence_is_presegmented
        self.target_regex = re.compile(target_pattern)
        self._role_normalizer = RoleNormalizer()
        self._nlp = None          # lazy-loaded Stanza pipeline
        self.use_nominalization = bool(use_nominalization)
        self.qanom_threshold = float(qanom_threshold)
        self._qanom_detector = None

    # Public interface

    def parse(self, text: str, sentence_id: str = "") -> ParsedSentence:
        """Parse one marked sentence into a ParsedSentence (lazy pipeline load)."""
        nlp = self._get_nlp()
        stanza_doc = nlp(text)
        return self._convert_stanza_doc(text, sentence_id, stanza_doc)

    def parse_many(
        self,
        items: list[tuple[str, str]],
        *,
        batch_size: int = 64,
    ) -> list[ParsedSentence]:
        """Parse many sentences in one Stanza batch; output matches ``parse()``."""
        nlp = self._get_nlp()
        out: list[ParsedSentence] = []
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            texts = [text for text, _sid in chunk]
            # Stanza 1.12.2 bulk_process drops coreference; use the document path when coref is on.
            stanza_docs = (
                [nlp(text) for text in texts]
                if self.use_coref
                else nlp.bulk_process(texts)
            )
            qanom_batches = (
                self._run_qanom_many(stanza_docs)
                if self.use_nominalization
                else [{} for _ in stanza_docs]
            )
            for (text, sentence_id), stanza_doc, qanom_results in zip(
                chunk, stanza_docs, qanom_batches,
            ):
                out.append(self._convert_stanza_doc(
                    text,
                    sentence_id,
                    stanza_doc,
                    qanom_results=qanom_results,
                ))
        return out

    def _convert_stanza_doc(
        self,
        text: str,
        sentence_id: str,
        stanza_doc,
        *,
        qanom_results: dict[int, object] | None = None,
    ) -> ParsedSentence:
        """Convert an already parsed Stanza Document into ParsedSentence."""

        if qanom_results is None:
            qanom_results = self._run_qanom(stanza_doc) if self.use_nominalization else {}
        nodes, protein_map, nominalization_sources = self._build_nodes(
            stanza_doc, qanom_results,
        )

        coref_clusters: tuple[CorefCluster, ...] = ()
        if self.use_coref:
            coref_clusters = self._build_coref_clusters(stanza_doc)

        parser_tag = self.package
        if self.use_coref:
            parser_tag += "+coref"
        raw_data: dict = {
            "package": self.package,
            "nominalization_detector": "QANom" if self.use_nominalization else "off",
            "qanom_threshold": self.qanom_threshold if self.use_nominalization else None,
            "nominalization_sources": nominalization_sources,
            "stanza_native": self._build_native_metadata(stanza_doc),
            "segmentation_policy": (
                "record_as_presegmented_unit"
                if self.sentence_is_presegmented
                else "stanza_sentence_split"
            ),
            "potential_internal_sentence_boundaries": (
                self._potential_internal_sentence_boundaries(text)
            ),
        }
        return ParsedSentence(
            text=text,
            sentence_id=sentence_id,
            nodes=nodes,
            protein_map=protein_map,
            coref_clusters=coref_clusters,
            parser_name=f"stanza_{parser_tag}",
            backend_info=BackendInfo(
                backend="stanza",
                model=self.package,
                version=importlib.metadata.version("stanza"),
                capabilities=(
                    "tokenize", "sentence_split", "pos", "lemma",
                    "depparse", "enhanced_dependencies",
                ) + (("coreference",) if self.use_coref else ()),
                enrichments=(("QANom",) if self.use_nominalization else ()),
                native_format="stanza_document",
            ),
            raw_parser_data=raw_data,
        )

    # Lazy loading

    def _get_nlp(self):
        """Lazy-load Stanza pipeline (once per adapter instance)."""
        if self._nlp is not None:
            return self._nlp

        try:
            import stanza
            from stanza.utils.conll import CoNLL  # noqa: F401: ensure stanza is importable
        except ImportError as exc:
            raise ImportError(
                "Stanza is required for StanzaAdapter. "
                "Install with: pip install stanza\n"
                "Then download models: stanza.download('en', package='craft')"
            ) from exc

        # tokenize, pos, lemma, depparse; coref only when requested.
        processors = "tokenize,pos,lemma,depparse"
        if self.use_coref:
            processors += ",coref"

        # Keep the lemma-loader workaround in repo code.
        _patch_stanza_nocharlm_lemma_loader()

        # No re-download; CRAFT for syntax, default English package for coref.
        package = self.package
        if self.use_coref:
            package = {
                "tokenize": self.package,
                "pos": self.package,
                "lemma": self.package,
                "depparse": self.package,
                "coref": "default",
            }

        self._nlp = stanza.Pipeline(
            "en",
            package=package,
            processors=processors,
            download_method=None,
            logging_level="WARN",
            use_gpu=self.use_gpu,
            # Inputs are already sentences; do not split again.
            tokenize_no_ssplit=self.sentence_is_presegmented,
        )
        return self._nlp

    # QANom nominalization detection

    @staticmethod
    def _normalize_semantic_lemma(text: str, lemma: str) -> str:
        """Strip hyphen artifacts from lemmas (``-mediate`` -> ``mediate``); raw lemma is kept."""
        low_text = (text or "").lower()
        low_lemma = (lemma or text or "").lower()
        if low_text.startswith("-") and low_lemma.startswith("-"):
            stripped = low_lemma[1:]
            if stripped and any(char.isalpha() for char in stripped):
                return stripped
        return low_lemma

    def _run_qanom(self, stanza_doc) -> dict[int, object]:
        """Run QANom over Stanza nouns and return document-global results."""
        if self._qanom_detector is None:
            from src.parsing.stanza.qanom_detector import QANomDetector
            self._qanom_detector = QANomDetector(
                threshold=self.qanom_threshold,
                use_gpu=self.use_gpu,
            )

        results: dict[int, object] = {}
        offset = 0
        for sent in stanza_doc.sentences:
            words = [str(word.text or "") for word in sent.words]
            upos = [str(word.upos or "") for word in sent.words]
            for local_i, result in self._qanom_detector.detect(words, upos).items():
                results[offset + local_i] = result
            offset += len(sent.words)
        return results

    def _run_qanom_many(self, stanza_docs) -> list[dict[int, object]]:
        """Batch QANom across documents while preserving document-global indices."""
        if self._qanom_detector is None:
            from src.parsing.stanza.qanom_detector import QANomDetector
            self._qanom_detector = QANomDetector(
                threshold=self.qanom_threshold,
                use_gpu=self.use_gpu,
            )

        flat_items: list[tuple[list[str], list[str]]] = []
        layout: list[list[int]] = []
        for doc in stanza_docs:
            sentence_slots: list[int] = []
            for sent in doc.sentences:
                sentence_slots.append(len(flat_items))
                flat_items.append((
                    [str(word.text or "") for word in sent.words],
                    [str(word.upos or "") for word in sent.words],
                ))
            layout.append(sentence_slots)

        flat_results = self._qanom_detector.detect_many(flat_items)
        output: list[dict[int, object]] = []
        for doc, sentence_slots in zip(stanza_docs, layout):
            doc_results: dict[int, object] = {}
            offset = 0
            for sent, slot in zip(doc.sentences, sentence_slots):
                for local_i, result in flat_results[slot].items():
                    doc_results[offset + local_i] = result
                offset += len(sent.words)
            output.append(doc_results)
        return output

    # Node construction

    def _build_nodes(
        self,
        stanza_doc,
        qanom_results: dict[int, object] | None = None,
    ) -> tuple[tuple[SyntaxNode, ...], dict[str, tuple[int, ...]], dict[int, str]]:
        """Return (nodes, protein_map, nominalization_sources) from a Stanza Document."""
        nodes: list[SyntaxNode] = []
        protein_map: dict[str, list[int]] = {}
        nominalization_sources: dict[int, str] = {}
        qanom_results = qanom_results or {}

        sent_start = 0  # global node index of the first word in current sentence
        for sentence_index, sent in enumerate(stanza_doc.sentences):
            multiword_by_word_id: dict[int, tuple[tuple[int, ...], str]] = {}
            for token in getattr(sent, "tokens", ()) or ():
                token_words = tuple(getattr(token, "words", ()) or ())
                if len(token_words) <= 1:
                    continue
                token_id = self._native_id_tuple(getattr(token, "id", ()))
                token_text = str(getattr(token, "text", "") or "")
                for token_word in token_words:
                    word_id = getattr(token_word, "id", None)
                    if isinstance(word_id, int):
                        multiword_by_word_id[word_id] = (token_id, token_text)

            for word in sent.words:
                node_i = len(nodes)

                # Character offsets
                char_start = word.start_char if word.start_char is not None else 0
                char_end = word.end_char if word.end_char is not None else char_start + len(word.text)

                # Dependency label normalization
                raw_dep = word.deprel or ""
                dep = self._role_normalizer.normalize(raw_dep)
                raw_deps = self._native_value(getattr(word, "deps", None))
                raw_feats = self._native_value(getattr(word, "feats", None))
                raw_misc = self._native_value(getattr(word, "misc", None))

                # 1-based sentence heads -> 0-based document-global index.
                if word.head == 0:
                    head_i = -1  # root
                else:
                    head_i = sent_start + word.head - 1

                # Protein target detection (regex on word text)
                for match in self.target_regex.finditer(word.text):
                    marker = match.group(0)
                    protein_map.setdefault(marker, []).append(node_i)
                protein_indices: tuple[int, ...] = ()  # filled in second pass below

                # QANom is the sole nominalization detector.
                is_nominalized = False
                verb_form = ""
                source_method = ""

                lower_lemma = self._normalize_semantic_lemma(
                    word.text,
                    word.lemma or word.text or "",
                )

                nominalization_confidence: float | None = None

                qanom_result = qanom_results.get(node_i)
                if qanom_result is not None:
                    is_nominalized = True
                    verb_form = str(getattr(qanom_result, "verb_form", "") or "")
                    source_method = "QANom"
                    nominalization_confidence = float(
                        getattr(qanom_result, "confidence", 0.0)
                    )

                if is_nominalized:
                    nominalization_sources[node_i] = source_method

                # POS: Stanza uses UPOS (universal) in word.upos
                pos = word.upos or ""
                raw_pos = word.xpos or ""
                sentence_word_id = int(word.id) if isinstance(word.id, int) else 0
                sentence_head_id = int(word.head) if isinstance(word.head, int) else 0
                multiword_id, multiword_text = multiword_by_word_id.get(
                    sentence_word_id, ((), ""),
                )

                nodes.append(SyntaxNode(
                    i=node_i,
                    text=word.text,
                    lemma=lower_lemma,
                    raw_lemma=str(word.lemma or word.text or "").lower(),
                    pos=pos,
                    dep=dep,
                    head_i=head_i,
                    char_start=char_start,
                    char_end=char_end,
                    protein_indices=protein_indices,  # filled below
                    is_nominalized=is_nominalized,
                    verb_form=verb_form,
                    nominalization_source=source_method,
                    nominalization_confidence=nominalization_confidence,
                    raw_dep=raw_dep,
                    raw_pos=raw_pos,
                    raw_deps=raw_deps,
                    raw_feats=raw_feats,
                    raw_misc=raw_misc,
                    sentence_index=sentence_index,
                    sentence_word_id=sentence_word_id,
                    sentence_head_id=sentence_head_id,
                    multiword_token_id=multiword_id,
                    multiword_token_text=multiword_text,
                    enhanced_heads=self._enhanced_heads(
                        getattr(word, "deps", None),
                        sent_start=sent_start,
                    ),
                    native_id=self._native_id_tuple(getattr(word, "id", ())),
                ))

            sent_start += len(sent.words)  # advance global offset for next sentence

        # Second pass: fill protein_indices from protein_map
        node_to_markers: dict[int, list[str]] = {}
        for marker, node_indices in protein_map.items():
            for ni in node_indices:
                node_to_markers.setdefault(ni, []).append(marker)

        # Convert protein markers to integer indices (0-based by first occurrence)
        marker_to_idx: dict[str, int] = {}
        for marker in sorted(protein_map.keys()):
            m = re.search(r"(\d+)$", marker)
            if m:
                marker_to_idx[marker] = int(m.group(1))

        # Reconstruct nodes with protein_indices filled
        final_nodes: list[SyntaxNode] = []
        for node in nodes:
            markers_on_node = node_to_markers.get(node.i, [])
            pindices = tuple(sorted({marker_to_idx[m] for m in markers_on_node if m in marker_to_idx}))
            final_nodes.append(dc_replace(
                node,
                protein_indices=pindices,
                direct_protein_indices=pindices,
            ))

        final_protein_map = {k: tuple(sorted(set(v))) for k, v in protein_map.items()}
        return tuple(final_nodes), final_protein_map, nominalization_sources

    # Stanza-native evidence preservation

    @staticmethod
    def _native_id_tuple(value) -> tuple[int, ...]:
        """Return a JSON-stable Stanza token/word id tuple."""
        if isinstance(value, int):
            return (value,)
        if isinstance(value, (tuple, list)):
            return tuple(int(item) for item in value if isinstance(item, int))
        return ()

    @staticmethod
    def _native_value(value) -> str:
        """Serialize a native scalar/list payload without interpreting it."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(value)

    @classmethod
    def _enhanced_heads(cls, deps, *, sent_start: int) -> tuple[tuple[int, str], ...]:
        """Convert Stanza enhanced dependencies to global heads, preserving roles."""
        if not deps or deps == "_":
            return ()
        pairs: list[tuple[int, str]] = []
        if isinstance(deps, str):
            entries = deps.split("|")
            for entry in entries:
                head_text, sep, role = entry.partition(":")
                if not sep:
                    continue
                try:
                    head = int(head_text)
                except ValueError:
                    continue
                pairs.append((-1 if head == 0 else sent_start + head - 1, role))
        elif isinstance(deps, (tuple, list)):
            for entry in deps:
                if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                    continue
                try:
                    head = int(entry[0])
                except (TypeError, ValueError):
                    continue
                pairs.append((-1 if head == 0 else sent_start + head - 1, str(entry[1])))
        return tuple(pairs)

    @classmethod
    def _build_native_metadata(cls, stanza_doc) -> dict:
        """Preserve document/sentence evidence not represented by SyntaxNode."""
        sentences: list[dict] = []
        for sentence_index, sent in enumerate(getattr(stanza_doc, "sentences", ()) or ()):
            multiword_tokens: list[dict] = []
            for token in getattr(sent, "tokens", ()) or ():
                words = tuple(getattr(token, "words", ()) or ())
                if len(words) <= 1:
                    continue
                multiword_tokens.append({
                    "id": list(cls._native_id_tuple(getattr(token, "id", ()))),
                    "text": str(getattr(token, "text", "") or ""),
                    "word_ids": [
                        int(word.id) for word in words if isinstance(getattr(word, "id", None), int)
                    ],
                    "misc": cls._native_value(getattr(token, "misc", None)),
                })
            sentences.append({
                "sentence_index": sentence_index,
                "text": str(getattr(sent, "text", "") or ""),
                "comments": [
                    str(comment) for comment in (getattr(sent, "comments", ()) or ())
                ],
                "multiword_tokens": multiword_tokens,
            })
        return {
            "format": "stanza_native_evidence_v1",
            "sentence_count": len(sentences),
            "sentences": sentences,
        }

    @staticmethod
    def _potential_internal_sentence_boundaries(text: str) -> list[dict]:
        """Record possible sentence boundaries hidden by record-level parsing.

        This is review evidence only. It does not split or suppress extraction.
        """
        rows: list[dict] = []
        for match in re.finditer(r"(?<=[.!?])\s+(?=\S)", text):
            offset = match.end()
            rows.append({
                "offset": offset,
                "left_context": text[max(0, offset - 40):offset].strip(),
                "right_context": text[offset:min(len(text), offset + 40)].strip(),
            })
        return rows

    # Coreference clusters

    def _build_coref_clusters(self, stanza_doc) -> tuple[CorefCluster, ...]:
        """Extract CorefCluster evidence; pronoun vs nominal by head POS."""
        if not hasattr(stanza_doc, "coref") or not stanza_doc.coref:
            return ()

        clusters: list[CorefCluster] = []
        for cluster_id, chain in enumerate(stanza_doc.coref):
            mentions: list[tuple[int, int]] = []
            evidence_type = "nominal"
            head_text = ""

            for mention in chain.mentions:
                if mention.sentence != 0:
                    continue
                if not isinstance(mention.start_word, int) or not isinstance(mention.end_word, int):
                    continue
                
                words = stanza_doc.sentences[0].words
                if mention.start_word < len(words) and mention.end_word <= len(words):
                    first_word = words[mention.start_word]
                    last_word = words[mention.end_word - 1]
                    start = getattr(first_word, "start_char", None)
                    end = getattr(last_word, "end_char", None)
                    if start is not None and end is not None:
                        mentions.append((start, end))

                    mention_words = words[mention.start_word : mention.end_word]
                    is_pronoun = False
                    for w in mention_words:
                        w_lemma = (getattr(w, "lemma", None) or getattr(w, "text", "") or "").lower()
                        w_upos = getattr(w, "upos", "") or ""
                        if w_upos == "PRON" or w_lemma in _PRONOUN_TEXTS:
                            is_pronoun = True
                            break
                    if is_pronoun:
                        evidence_type = "pronoun"
                    
                    if not head_text:
                        head_text = chain.representative_text or ""

            # Skip singleton chains.
            if len(mentions) >= 2:
                clusters.append(CorefCluster(
                    cluster_id=cluster_id,
                    mentions=tuple(mentions),
                    evidence_type=evidence_type,
                    head_text=head_text.lower(),
                    source="stanza_coref",
                ))

        return tuple(clusters)
