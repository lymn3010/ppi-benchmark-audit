from __future__ import annotations

import json
import csv
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

from src.analysis.observation_categories import SCORE_PROPAGATION_PRECISION, category_metadata_for

_RAW_JSONL_HANDLE_ATTRS = (
    "_records_fp",
    "_candidates_fp",
    "_observations_fp",
    "_mention_graphs_fp",
    "_pair_traces_fp",
    "_projection_plans_fp",
)

# Nominal-to-verbal mapping (shared with extractor.py)
_NOMINAL_TO_VERBAL = {}
_nominal_csv = Path(__file__).parent.parent.parent / "data" / "lexicons" / "nominal_to_verbal.csv"
if _nominal_csv.exists():
    with open(_nominal_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                _NOMINAL_TO_VERBAL[row[0].strip().lower()] = row[1].strip().lower()


@dataclass
class EventDetail:
    """Rich serialization of a single extracted event."""
    event_type: str             # "InteractionalEvent", "UndirectedInteraction", "RelationalStatement"
    semantic_event_class: str   # predicate_event / relation_statement
    event_detail: str           # "S-V-O", "AppositionGroup", etc.
    realization_channel: str    # clause / event_nominal / state_nominal / appositive / lexical_path
    relation_type: str          # "IS-A", "HAS", "" etc.
    span: str                   # text span covering the event
    span_indices: list[int]     # [min_token_idx, max_token_idx]
    pred_pairs: list[list[int]] # pairs produced by this event
    predicate: Optional[dict]   # {text, lemma, token_idx, category}
    arguments: dict             # full argument structure
    observations_used: list[dict]   # [{word, lemma, category, role, token_idx}]
    pattern_key: str = ""       # canonical pattern identity for DB/audit joins
    semantic_pattern_key: str = ""
    yield_type: str = ""
    construction: str = ""
    chain_dict: Optional[dict] = None  # legacy RelationChain.to_dict() fallback


@dataclass
class ObservationOccurrence:
    """A single occurrence of a observation term in context."""
    sentence_id: str
    sentence: str
    orig_sentence: str          # with original protein names (toggleable)
    proteins: list[str]         # protein list for this sentence
    span: str                   # event span text
    span_indices: list[int]     # [min_idx, max_idx]
    event_type: str
    event_detail: str
    pred_pairs: list[list[int]]
    contribution_pairs: list[list[int]]
    gold_pairs: list[list[int]]
    is_correct: bool            # did this event produce a correct pair?
    arguments: dict             # full argument structure
    observation_role: str           # "predicate", "entities_b", "sources", etc.
    realization_channel: str = ""
    propagation_sources: tuple[str, ...] = ()
    is_nested: bool = False
    direct_core: bool = False


@dataclass
class ObservationStats:
    """Aggregated statistics for an observation term across the dataset."""
    word: str
    category: str               # "trigger_verbal", "trigger_nominal", "target_context"
    count: int = 0              # trigger: evaluable occurrences; target_context: propagated pair count
    correct: int = 0            # trigger: correct occurrences; target_context: propagated TP pairs
    count_all: int = 0          # total occurrences (including non-evaluable, all categories)
    fp_count: int = 0           # false positive pair count (evaluable, incorrect)
    fn_count: int = 0           # false negative pair count (gold pairs not predicted)
    positive: list[ObservationOccurrence] = field(default_factory=list)  # TP examples
    negative: list[ObservationOccurrence] = field(default_factory=list)  # FP examples
    no_gold: list[ObservationOccurrence] = field(default_factory=list)   # no gold pairs (can't judge)
    no_pairs: list[ObservationOccurrence] = field(default_factory=list)  # event produced no pairs
    pair_contributing_occurrences: int = 0
    nonprojecting_occurrences: int = 0
    owner_occurrences: int = 0
    descriptor_occurrences: int = 0
    nested_occurrences: int = 0
    direct_core_occurrences: int = 0

    @property
    def precision(self) -> float:
        """Category-specific precision; 0 for categories without a score."""
        return self.correct / self.count if self.count > 0 else 0.0


@dataclass
class SentenceReport:
    """Full report for a single sentence."""
    sentence_id: str
    text: str
    orig_sentence: str
    proteins: list[str]
    pred_pairs: list[list[int]]
    gold_pairs: Optional[list[list[int]]]
    label: str                  # "correct", "too_many", "too_few", "wrong_pairs", "no_gold", "no_event"
    events: list[EventDetail] = field(default_factory=list)
    record_version: str = ""
    provenance: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
    run_meta: dict = field(default_factory=dict)


class EventReportBuilder:
    """Build the run report from pipeline Records; call ``add_record`` then ``export``."""

    def __init__(self,
                 mode: str = "extract",
                 dataset: str = "",
                 config: dict = None,
                 include_orig_sentence: bool = True,
                 max_examples_per_observation: int = 50):
        """Store run mode, dataset, config and example limits."""
        self.mode = mode
        self.dataset = dataset
        self.config = config or {}
        self.include_orig_sentence = include_orig_sentence
        self.max_examples = max_examples_per_observation
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Accumulated data
        self.observations: dict[str, dict[str, ObservationStats]] = {
            "trigger_verbal": {},
            "trigger_nominal": {},
            "target_context": {},
        }
        self.sentences: dict[str, SentenceReport] = {}
        self.total_records = 0

        # Observation-only structural pattern aggregation. It cannot gate runtime.
        from src.models.data.pattern_observation_store import PatternObservationStore
        self._pattern_observations = PatternObservationStore()

        # Flat attribution log for database export (no examples cap, all occurrences)
        self._attributions: list[dict] = []

        # Streaming raw JSONL handles (None until enable_raw_jsonl() is called)
        self._raw_dir: Optional[Path] = None
        self._records_fp = None       # file handle: raw/records.jsonl
        self._candidates_fp = None    # file handle: raw/candidate_events.jsonl
        self._observations_fp = None  # file handle: raw/event_observations.jsonl
        self._mention_graphs_fp = None  # file handle: raw/mention_graphs.jsonl
        self._pair_traces_fp = None  # file handle: raw/pair_inference_traces.jsonl
        self._projection_plans_fp = None  # raw/projection_plans.jsonl


    def enable_raw_jsonl(
        self,
        raw_dir: Path,
        *,
        include_event_observations: bool = False,
    ) -> None:
        """Open the per-sentence JSONL streams for this run."""
        self.close_raw_jsonl()
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        self._raw_dir = raw_dir
        self._records_fp = open(raw_dir / "records.jsonl", "w", encoding="utf-8")
        self._candidates_fp = open(raw_dir / "candidate_events.jsonl", "w", encoding="utf-8")
        if include_event_observations:
            self._observations_fp = open(
                raw_dir / "event_observations.jsonl", "w", encoding="utf-8",
            )
            self._mention_graphs_fp = open(
                raw_dir / "mention_graphs.jsonl", "w", encoding="utf-8",
            )
            self._pair_traces_fp = open(
                raw_dir / "pair_inference_traces.jsonl", "w", encoding="utf-8",
            )
            self._projection_plans_fp = open(
                raw_dir / "projection_plans.jsonl", "w", encoding="utf-8",
            )

    def close_raw_jsonl(self) -> None:
        """Flush and close streaming JSONL handles. Called automatically by export()."""
        for attr in _RAW_JSONL_HANDLE_ATTRS:
            handle = getattr(self, attr, None)
            if handle is None:
                continue
            try:
                handle.flush()
                handle.close()
            except Exception:
                pass
            setattr(self, attr, None)

    def __del__(self) -> None:
        """Ensure handles are closed even on unexpected exit."""
        self.close_raw_jsonl()
    
    def add_record(self, record, entry_info: dict = None) -> None:
        """Accumulate observation and sentence data from one Record."""
        from src.models import Record
        if not isinstance(record, Record):
            return
            
        self.total_records += 1
        entry_info = entry_info or {}
        orig_sentence = entry_info.get("orig_sentence", "") if self.include_orig_sentence else ""
        proteins = entry_info.get("proteins", [])
        gold_pairs = record.gold_pairs
        gold_set = set(tuple(sorted(p)) for p in gold_pairs) if gold_pairs else set()
        pred_set = set(tuple(sorted(p)) for p in record.predicted_pairs)
        label = self._classify_sentence(pred_set, gold_set, gold_pairs)

        event_details = self._build_record_event_details(
            record,
            orig_sentence=orig_sentence,
            proteins=proteins,
            gold_set=gold_set,
            gold_pairs=gold_pairs,
        )
        self.sentences[record.sentence_id] = self._sentence_report_from_record(
            record,
            orig_sentence=orig_sentence,
            proteins=proteins,
            pred_set=pred_set,
            gold_set=gold_set,
            gold_pairs=gold_pairs,
            label=label,
            events=event_details,
        )
        self._stream_raw_jsonl_record(record)

    def _build_record_event_details(
        self,
        record,
        *,
        orig_sentence: str,
        proteins: list,
        gold_set: set,
        gold_pairs,
    ) -> list[EventDetail]:
        event_details = []
        for event in record.events or []:
            detail = self._build_event_detail(
                event, record.sentence_text,
                token_offsets=getattr(record, "token_offsets", None),
            )
            event_details.append(detail)
            self._attribute_observations(
                event=event,
                detail=detail,
                sentence_id=record.sentence_id,
                sentence=record.sentence_text,
                orig_sentence=orig_sentence,
                proteins=proteins,
                gold_set=gold_set,
                gold_pairs=gold_pairs,
            )
        return event_details

    @staticmethod
    def _sentence_report_from_record(
        record,
        *,
        orig_sentence: str,
        proteins: list,
        pred_set: set,
        gold_set: set,
        gold_pairs,
        label: str,
        events: list[EventDetail],
    ) -> SentenceReport:
        return SentenceReport(
            sentence_id=record.sentence_id,
            text=record.sentence_text,
            orig_sentence=orig_sentence,
            proteins=proteins,
            pred_pairs=[list(p) for p in sorted(pred_set)],
            gold_pairs=[list(p) for p in sorted(gold_set)] if gold_pairs is not None else None,
            label=label,
            events=events,
            record_version=getattr(record, "record_version", ""),
            provenance=getattr(record, "provenance", {}) or {},
            artifacts=getattr(record, "artifacts", {}) or {},
            run_meta=getattr(record, "run_meta", {}) or {},
        )

    @staticmethod
    def _write_jsonl_line(handle, payload: dict) -> None:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_jsonl_items(self, handle, items) -> None:
        """Best-effort JSONL writer; raw streams must never abort extraction."""
        if handle is None:
            return
        try:
            for item in items:
                self._write_jsonl_line(handle, item)
        except Exception:
            pass

    def _stream_raw_jsonl_record(self, record) -> None:
        """Write optional raw JSONL streams immediately for partial-run recovery."""
        artifacts = getattr(record, "artifacts", {}) or {}
        self._write_jsonl_items(
            self._records_fp,
            (record.to_json() for _ in (None,)),
        )
        self._write_jsonl_items(
            self._candidates_fp,
            (
                {"sentence_id": record.sentence_id, "candidate": candidate}
                for candidate in artifacts.get("candidate_events") or []
            ),
        )
        self._write_jsonl_items(
            self._observations_fp,
            (
                {"sentence_id": record.sentence_id, "observation": observation}
                for observation in artifacts.get("event_observations") or []
            ),
        )
        graph = artifacts.get("mention_graph")
        self._write_jsonl_items(
            self._mention_graphs_fp,
            (graph,) if graph else (),
        )
        self._write_jsonl_items(
            self._pair_traces_fp,
            artifacts.get("pair_inference_traces") or [],
        )
        self._write_jsonl_items(
            self._projection_plans_fp,
            artifacts.get("projection_plans") or [],
        )
    
    def export(
        self,
        output_dir: Path,
        run_meta: dict | None = None,
        include_database: bool | None = None,
        include_observation_csvs: bool | None = None,
        # Compatibility opt-in outputs
        include_observations_json: bool = False,
        include_observation_tsvs: bool = False,
        include_sentences_json: bool = False,
        include_metrics: bool = False,
    ) -> Path:
        """Write the event database and observation CSVs; other exports are optional."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if include_database is None:
            include_database = True
        if include_observation_csvs is None:
            include_observation_csvs = self.mode == "extract"

        # events.db lives under db/ to keep the root uncluttered
        if include_database:
            db_dir = output_dir / "db"
            db_dir.mkdir(exist_ok=True)
            self._export_database(db_dir / "events.db", run_meta)

        if include_observation_csvs:
            kw_dir = output_dir / "observations"
            kw_dir.mkdir(exist_ok=True)
            self._export_observation_csvs(kw_dir, root_dir=output_dir)

        if include_observations_json:
            self._export_observations(output_dir / "observations.json")

        if include_sentences_json:
            self._export_sentences(output_dir / "sentences.json")

        if include_metrics:
            reports_dir = output_dir / "reports"
            reports_dir.mkdir(exist_ok=True)
            self._export_metrics(reports_dir / "metrics.json")

        if include_observation_tsvs:
            self._export_observation_tsvs(output_dir)

        # Close streaming JSONL handles (flush and finalize raw/*.jsonl)
        self.close_raw_jsonl()

        return output_dir


    def _build_event_detail(self, event, sentence: str, token_offsets: list | None = None) -> EventDetail:
        """Convert a canonical CandidateEvent to a rich EventDetail."""
        from src.models.candidate import EventType

        if event.event_type is EventType.RELATION_STATEMENT:
            event_type = "RelationalStatement"
        elif event.is_directed:
            event_type = "InteractionalEvent"
        else:
            event_type = "UndirectedInteraction"
        relation_type = event.relation_subtype if event.event_type is EventType.RELATION_STATEMENT else ""
        
        # Build arguments dict with full detail
        arguments = {}
        all_token_indices = []
        observations_used = []
        
        if event.arguments:
            for role, frames in event.arguments.items():
                args_list = []
                for argument in frames:
                    arg_dict = self._argument_to_dict(argument)
                    args_list.append(arg_dict)
                    
                    # Collect token indices for span
                    all_token_indices.extend(arg_dict.get("token_idx", []))
                    for owner in arg_dict.get("owners", []):
                        all_token_indices.extend(owner.get("token_idx", []))
                    
                    # Detect observation usage
                    kw = self._detect_observation(argument, role, event)
                    if kw:
                        observations_used.append(kw)
                
                arguments[role] = args_list
        
        # Predicate
        predicate_dict = None
        if event.predicate:
            predicate = event.predicate
            lemma = self._get_lemma(predicate)
            predicate_dict = {
                "text": predicate.text,
                "lemma": lemma,
                "token_idx": list(predicate.node_indices),
            }
            all_token_indices.extend(predicate_dict["token_idx"])
            
            # Detect predicate observation
            kw = self._detect_predicate_observation(event)
            if kw:
                observations_used.append(kw)
        
        # Compute span
        span, span_indices = self._compute_span(sentence, all_token_indices, token_offsets)
        
        identity = self._event_pattern_identity(event)
        shape = identity.shape if identity else ""
        return EventDetail(
            event_type=event_type,
            semantic_event_class=(
                "relation_statement"
                if event_type == "RelationalStatement"
                else "predicate_event"
            ),
            event_detail=event.extraction_detail,
            realization_channel=getattr(event, "realization_channel", "unknown"),
            relation_type=relation_type,
            span=span,
            span_indices=span_indices,
            pred_pairs=[list(p) for p in event.projected_pairs],
            predicate=predicate_dict,
            arguments=arguments,
            observations_used=observations_used,
            pattern_key=identity.canonical_key if identity else "",
            semantic_pattern_key=identity.semantic_key if identity else "",
            yield_type=shape,
            construction=(
                getattr(event, "construction", "")
                or (identity.construction if identity else "")
                or ""
            ),
            chain_dict=None,
        )
    
    
    def _attribute_observations(self, event, detail: EventDetail,
                            sentence_id: str, sentence: str,
                            orig_sentence: str, proteins: list,
                            gold_set: set, gold_pairs) -> None:
        """Attribute event to observations and track TP/FP/no_gold."""
        event_pairs = set(event.projected_pairs)
        is_correct = len(event_pairs & gold_set) > 0 if event_pairs and gold_set else False

        # Determine if IS-A relational
        from src.models.candidate import EventType
        is_relational_is_a = (
            event.event_type is EventType.RELATION_STATEMENT
            and (event.relation_subtype or "IS-A") == "IS-A"
        )
        
        def _make_occurrence(
            observation_role: str,
            contribution_pairs: set[tuple[int, int]] | None = None,
            is_correct_override: bool | None = None,
            propagation_sources: tuple[str, ...] = (),
            direct_core: bool = False,
        ) -> ObservationOccurrence:
            local_pairs = event_pairs if contribution_pairs is None else contribution_pairs
            local_correct = (
                is_correct if is_correct_override is None
                else is_correct_override
            )
            return ObservationOccurrence(
                sentence_id=sentence_id,
                sentence=sentence,
                orig_sentence=orig_sentence,
                proteins=proteins,
                span=detail.span,
                span_indices=detail.span_indices,
                event_type=detail.event_type,
                event_detail=detail.event_detail,
                pred_pairs=detail.pred_pairs,
                contribution_pairs=[list(p) for p in sorted(local_pairs)],
                gold_pairs=[list(p) for p in sorted(gold_set)] if gold_pairs is not None else [],
                is_correct=local_correct,
                arguments=detail.arguments,
                observation_role=observation_role,
                realization_channel=detail.realization_channel,
                propagation_sources=propagation_sources,
                is_nested="nested" in detail.event_detail.lower(),
                direct_core=direct_core,
            )
        
        def _add_to_stats(
            category: str,
            lemma: str,
            occurrence: ObservationOccurrence,
            scoring_pairs: set[tuple[int, int]] | None = None,
        ):
            if lemma not in self.observations[category]:
                self.observations[category][lemma] = ObservationStats(word=lemma, category=category)
            stats = self.observations[category][lemma]
            stats.count_all += 1

            metadata = category_metadata_for(category)
            pairs_for_score = event_pairs if scoring_pairs is None else scoring_pairs
            is_evaluable = bool(pairs_for_score) and gold_pairs is not None
            if category == "target_context":
                if pairs_for_score:
                    stats.pair_contributing_occurrences += 1
                else:
                    stats.nonprojecting_occurrences += 1
                stats.owner_occurrences += int("owner" in occurrence.propagation_sources)
                stats.descriptor_occurrences += int("descriptor" in occurrence.propagation_sources)
                stats.nested_occurrences += int(occurrence.is_nested)
                stats.direct_core_occurrences += int(occurrence.direct_core)

            if is_evaluable and metadata.is_trigger_scored:
                stats.count += 1
                if occurrence.is_correct:
                    stats.correct += 1

            if is_evaluable and metadata.score_mode == SCORE_PROPAGATION_PRECISION:
                correct_pairs = pairs_for_score & gold_set
                stats.count += len(pairs_for_score)
                stats.correct += len(correct_pairs)
                stats.fp_count += len(pairs_for_score - gold_set)

            if gold_pairs is None:
                if len(stats.no_gold) < self.max_examples:
                    stats.no_gold.append(occurrence)
            elif not pairs_for_score:
                if len(stats.no_pairs) < self.max_examples:
                    stats.no_pairs.append(occurrence)
            elif occurrence.is_correct:
                if len(stats.positive) < self.max_examples:
                    stats.positive.append(occurrence)
            else:
                if len(stats.negative) < self.max_examples:
                    stats.negative.append(occurrence)

            # Flat attribution log (every occurrence).
            _pat_key = self._event_pattern_key(event)
            self._attributions.append({
                "sentence_id": occurrence.sentence_id,
                "event_detail": detail.event_detail,
                "pattern_key": _pat_key,
                "term": lemma,
                "category": category,
                "role": occurrence.observation_role,
                "is_correct": occurrence.is_correct,
                "is_evaluable": is_evaluable,
            })
        
        # 0. Track (predicate_lemma, preposition, yield_type) as run observations.
        self._track_pattern(event, event_pairs, gold_set, gold_pairs, sentence)

        # 1. Predicate: verbal -> trigger_verbal, nominalized -> trigger_nominal (parser lemma).
        if event.predicate:
            core = event.predicate
            if not is_relational_is_a:
                if core.pos == "VERB":
                    lemma = self._get_lemma(core, convert_nominal=True)
                    _add_to_stats("trigger_verbal", lemma, _make_occurrence("predicate"))
                elif core.pos in {"NOUN", "PROPN"}:
                    # Nominalized predicate: store the noun's parser lemma.
                    lemma = self._get_lemma(core)
                    _add_to_stats("trigger_nominal", lemma, _make_occurrence("predicate"))

        # 2. IS-A entities_b nominals. 3. Nouns that own proteins.
        if event.arguments:
            for role, frames in event.arguments.items():
                if not frames:
                    continue
                for frame in frames:
                    core = frame.core

                    # Trigger Nominal (IS-A relational: relational nouns like "kinase of X")
                    if is_relational_is_a and role == "entities_b":
                        lemma = self._get_lemma(core, convert_nominal=True)
                        _add_to_stats("trigger_nominal", lemma, _make_occurrence(role))
                    
                    # Score only pairs with a protein from this frame's owners/descriptors.
                    propagated = set(frame.owner_protein_indices) | set(frame.descriptor_protein_indices)
                    direct = set(core.protein_indices) if core is not None else set()
                    if core is not None and not direct and propagated:
                        lemma = self._get_lemma(core)
                        propagated_pairs = {
                            p for p in event_pairs
                            if int(p[0]) in propagated or int(p[1]) in propagated
                        }
                        propagation_sources = tuple(sorted({
                            source
                            for source, items in (
                                ("owner", frame.owners),
                                ("descriptor", frame.descriptors),
                            )
                            if any(item.protein_indices for item in items)
                        }))
                        local_correct = bool(propagated_pairs & gold_set)
                        _add_to_stats(
                            "target_context",
                            lemma,
                            _make_occurrence(
                                role,
                                contribution_pairs=propagated_pairs,
                                is_correct_override=local_correct,
                                propagation_sources=propagation_sources,
                            ),
                            scoring_pairs=propagated_pairs,
                        )
    

    def _track_pattern(self, event, event_pairs: set, gold_set: set,
                       gold_pairs, sentence: str) -> None:
        """Record one canonical event observation for analysis."""
        from src.models.candidate import EventType

        detail = getattr(event, "extraction_detail", "") or ""

        if not event.predicate or event.event_type is EventType.RELATION_STATEMENT:
            return

        example = sentence[:120].replace("\t", " ").replace("\n", " ") if sentence else ""
        # Pass gold_set when annotation exists, else None so pattern_store skips stats
        gold_pairs_set = gold_set if gold_pairs is not None else None

        if detail in ("S-V-O", "S-V-O[Unk]"):
            return

        pred_lemma = self._get_lemma(event.predicate, convert_nominal=True)

        _undirected = not event.is_directed
        if _undirected:
            self._pattern_observations.record_observation(
                pred_lemma, None, "AA", event_pairs, gold_pairs_set, example,
                construction=event.construction,
            )
            return

        if event.is_directed:
            # Extract preposition from target frames' case markers
            target_frames = event.arguments.get("targets", ())
            preposition = None
            for frame in target_frames:
                for marker in frame.case_markers:
                    prep = self._get_lemma(marker)
                    if prep:
                        preposition = prep
                        break
                if preposition:
                    break

            # Bare direct object: no preposition -- skip nmod tracking
            if preposition is None and detail in ("S-V-O", "S-V-O[Unk]", "S-V-ModTarget",
                                                   "S-V-SMod_Nom", "Possession-NomComp"):
                return

            self._pattern_observations.record_observation(
                pred_lemma, preposition, "AB", event_pairs, gold_pairs_set, example,
                construction=event.construction,
            )

    
    def _detect_observation(self, frame, role: str, event) -> Optional[dict]:
        """Classify a canonical argument as an observation carrier."""
        core = frame.core
        if core is None:
            return None
        
        # Skip protein entities (they have targets)
        if core.protein_indices:
            return None
        
        result = {
            "word": core.text,
            "lemma": self._get_lemma(core),
            "token_idx": list(core.node_indices),
            "role": role,
        }
        
        from src.models.candidate import EventType
        if event.event_type is EventType.RELATION_STATEMENT and role == "entities_b":
            result["category"] = "trigger_nominal"
            return result
        if frame.owner_protein_indices or frame.descriptor_protein_indices:
            result["category"] = "target_context"
            return result
        return None
    
    def _detect_predicate_observation(self, event) -> Optional[dict]:
        """Detect observation from predicate."""
        if not event.predicate:
            return None
        core = event.predicate
        if core.pos == "VERB":
            return {
                "word": core.text,
                "lemma": self._get_lemma(core, convert_nominal=True),
                "category": "trigger_verbal",
                "role": "predicate",
                "token_idx": list(core.node_indices),
            }
        return None
    
    
    @staticmethod
    def _compute_span(
        sentence: str,
        token_indices: list[int],
        token_offsets: list[tuple[int, int]] | None = None,
    ) -> tuple[str, list[int]]:
        """Text span from the first to last token; exact with offsets, approximate otherwise."""
        if not token_indices:
            return "", []

        min_idx = min(token_indices)
        max_idx = max(token_indices)

        if token_offsets and 0 <= min_idx < len(token_offsets) and 0 <= max_idx < len(token_offsets):
            start = token_offsets[min_idx][0]
            end   = token_offsets[max_idx][1]
            return sentence[start:end], [min_idx, max_idx]

        # Fallback (should not normally fire post-fix)
        words = sentence.split()
        if max_idx < len(words):
            span = " ".join(words[min_idx:max_idx + 1])
        else:
            span = " ".join(words[min_idx:])
        return span, [min_idx, max_idx]
    
    
    @staticmethod
    def _argument_to_dict(frame) -> dict:
        """Convert CandidateArgument to a report-friendly dictionary."""
        if frame.is_event_reference:
            return {"event_ref": frame.event_ref, "role": frame.role}
        if frame.core is None:
            return {"text": "", "token_idx": [], "targets": [], "role": frame.role}
        result = {
            "text": frame.core.text,
            "token_idx": list(frame.core.node_indices),
            "targets": list(frame.projection_protein_indices),
        }
        direct_targets = frame.core.protein_indices
        propagated_targets = tuple(sorted(
            set(frame.owner_protein_indices) | set(frame.descriptor_protein_indices)
        ))
        if direct_targets:
            result["direct_targets"] = list(direct_targets)
        if propagated_targets:
            result["propagated_targets"] = list(propagated_targets)
        if frame.core.is_negated:
            result["is_negated"] = True
        if frame.owners:
            result["owners"] = [{
                "text": o.text,
                "token_idx": list(o.node_indices),
                "targets": list(o.protein_indices),
            } for o in frame.owners]
        if frame.descriptors:
            result["descriptors"] = [{
                "text": d.text,
                "token_idx": list(d.node_indices),
            } for d in frame.descriptors]
        group = getattr(frame, "group", None)
        if group:
            result["group"] = dict(group)
        return result
    
    
    @staticmethod
    def _get_lemma(lex_item, convert_nominal: bool = False) -> str:
        """Get normalized lemma from a canonical Span."""
        lemmas = lex_item.lemmas
        lemma = lemmas[0].lower() if lemmas else lex_item.text.lower()
        if convert_nominal and lemma in _NOMINAL_TO_VERBAL:
            lemma = _NOMINAL_TO_VERBAL[lemma]
        return lemma

    @staticmethod
    def _event_pattern_identity(event):
        """Stable pattern identity derived from the canonical CandidateEvent."""
        from src.models.candidate import EventType
        from src.models.pattern_identity import PatternIdentity

        if event.event_type is EventType.RELATION_STATEMENT:
            predicate = event.predicate
            predicate_concept = (
                EventReportBuilder._get_lemma(predicate, convert_nominal=True)
                if predicate is not None
                else (event.relation_subtype or "relational").lower()
            )
            construction = event.construction
            if not construction:
                construction = (
                    "nominalized"
                    if predicate is not None and predicate.pos in {"NOUN", "PROPN"}
                    else "verbal"
                )
            return PatternIdentity(
                evidence_class="relational",
                shape="RELATIONAL",
                predicate_concept=predicate_concept,
                case_realization="",
                construction=construction,
            )

        if not event.predicate:
            return None
        prep = None
        for argument in event.arguments.get("targets", ()):
            if argument.case_markers:
                prep = EventReportBuilder._get_lemma(argument.case_markers[0])
                break
        return PatternIdentity(
            evidence_class="structural",
            # The preposition is its own key segment, so directed PP events stay AB.
            shape=("AA" if not event.is_directed else "AB"),
            predicate_concept=EventReportBuilder._get_lemma(
                event.predicate, convert_nominal=True
            ),
            case_realization=prep or "",
            construction=event.construction,
        )

    @staticmethod
    def _event_pattern_key(event) -> str | None:
        """Stable canonical pattern key without a legacy RelationChain."""
        identity = EventReportBuilder._event_pattern_identity(event)
        return identity.canonical_key if identity else None
    
    
    @staticmethod
    def _classify_sentence(pred_set: set, gold_set: set, gold_pairs) -> str:
        """Classify sentence prediction result."""
        if gold_pairs is None:
            return "no_gold"
        if not pred_set and not gold_set:
            return "correct"
        if pred_set == gold_set:
            return "correct"
        if len(pred_set) > len(gold_set):
            return "too_many"
        if len(pred_set) < len(gold_set):
            return "too_few"
        return "wrong_pairs"

    @staticmethod
    def _sorted_observation_items(category: str, stats_map: dict):
        """Return category metadata and stats sorted by the category support unit."""
        metadata = category_metadata_for(category)
        sort_key = (
            (lambda item: item[1].count)
            if metadata.uses_precision_threshold
            else (lambda item: item[1].count_all)
        )
        return metadata, sorted(stats_map.items(), key=sort_key, reverse=True)

    @staticmethod
    def _observation_score_fields(stats: ObservationStats, metadata) -> dict:
        """Count/correct/precision fields with category-specific semantics."""
        return {
            "count": stats.count if metadata.uses_precision_threshold else stats.count_all,
            "correct": stats.correct if metadata.uses_precision_threshold else None,
            "precision": (
                round(stats.precision, 4)
                if metadata.uses_precision_threshold else None
            ),
        }
    
    
    def _export_observations(self, path: Path) -> None:
        """Export observation-centric view with positive/negative examples."""
        output = {}
        
        for category, kw_dict in self.observations.items():
            cat_output = {}
            metadata, items = self._sorted_observation_items(category, kw_dict)
            for word, stats in items:
                entry = {
                    "score_mode": metadata.score_mode,
                    "observation_unit": metadata.observation_unit,
                    "success_definition": metadata.success_definition,
                    "count_all": stats.count_all,
                    "positive": [self._occurrence_to_dict(o) for o in stats.positive],
                    "negative": [self._occurrence_to_dict(o) for o in stats.negative],
                }
                entry.update(self._observation_score_fields(stats, metadata))
                if stats.no_gold:
                    entry["no_gold"] = [self._occurrence_to_dict(o) for o in stats.no_gold]
                cat_output[word] = entry
            
            output[category] = cat_output
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    
    def _occurrence_to_dict(self, occ: ObservationOccurrence) -> dict:
        """Serialize ObservationOccurrence."""
        result = {
            "sentence_id": occ.sentence_id,
            "sentence": occ.sentence,
            "span": occ.span,
            "span_indices": occ.span_indices,
            "event_type": occ.event_type,
            "event_detail": occ.event_detail,
            "pred_pairs": occ.pred_pairs,
            "contribution_pairs": occ.contribution_pairs,
            "gold_pairs": occ.gold_pairs,
            "is_correct": occ.is_correct,
            "observation_role": occ.observation_role,
            "arguments": occ.arguments,
        }
        if self.include_orig_sentence and occ.orig_sentence:
            result["orig_sentence"] = occ.orig_sentence
        if occ.proteins:
            result["proteins"] = occ.proteins
        return result
    
    
    def _export_sentences(self, path: Path) -> None:
        """Export sentence-centric view with full event detail."""
        output = {}
        
        for sid, report in self.sentences.items():
            entry = {
                "text": report.text,
                "pred_pairs": report.pred_pairs,
                "gold_pairs": report.gold_pairs,
                "label": report.label,
                "events": [self._event_detail_to_dict(e) for e in report.events],
                "record_version": report.record_version,
                "provenance": report.provenance,
                "artifacts": report.artifacts,
                "run_meta": report.run_meta,
            }
            if self.include_orig_sentence and report.orig_sentence:
                entry["orig_sentence"] = report.orig_sentence
            if report.proteins:
                entry["proteins"] = report.proteins
            output[sid] = entry
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    
    def _event_detail_to_dict(self, detail: EventDetail) -> dict:
        """Serialize EventDetail."""
        result = {
            "type": detail.event_type,
            "semantic_event_class": detail.semantic_event_class,
            "detail": detail.event_detail,
            "realization_channel": detail.realization_channel,
            "span": detail.span,
            "span_indices": detail.span_indices,
            "pred_pairs": detail.pred_pairs,
            "arguments": detail.arguments,
        }
        if detail.relation_type:
            result["relation_type"] = detail.relation_type
        if detail.predicate:
            result["predicate"] = detail.predicate
        if detail.pattern_key:
            result["pattern_key"] = detail.pattern_key
        if detail.semantic_pattern_key:
            result["semantic_pattern_key"] = detail.semantic_pattern_key
        if detail.yield_type:
            result["yield_type"] = detail.yield_type
        if detail.construction:
            result["construction"] = detail.construction
        if detail.observations_used:
            result["observations_used"] = detail.observations_used
        return result
    
    
    def _export_metrics(self, path: Path) -> None:
        """Export aggregate metrics and config."""
        output = {
            "meta": {
                "mode": self.mode,
                "dataset": self.dataset,
                "timestamp": self.timestamp,
                "config": self.config,
                "total_sentences": self.total_records,
            },
            "observation_summary": {},
            "sentence_summary": {
                "total": len(self.sentences),
                "by_label": {},
            },
        }
        
        # Observation summary per category
        for category, kw_dict in self.observations.items():
            metadata, items = self._sorted_observation_items(category, kw_dict)
            total_kw = len(kw_dict)
            total_count = sum(s.count for s in kw_dict.values())
            total_count_all = sum(s.count_all for s in kw_dict.values())
            total_correct = sum(s.correct for s in kw_dict.values())
            
            output["observation_summary"][category] = {
                "unique_terms": total_kw,
                "score_mode": metadata.score_mode,
                "observation_unit": metadata.observation_unit,
                "success_definition": metadata.success_definition,
                "total_evaluable_occurrences": total_count,
                "total_occurrences": total_count_all,
                "total_correct": total_correct,
                "overall_precision": (
                    round(total_correct / total_count, 4)
                    if metadata.uses_precision_threshold and total_count > 0
                    else None
                ),
                "top_10": [
                    {
                        "word": w,
                        **self._observation_score_fields(s, metadata),
                    }
                    for w, s in items[:10]
                ],
            }
        
        # Sentence label distribution
        label_counts = defaultdict(int)
        for report in self.sentences.values():
            label_counts[report.label] += 1
        output["sentence_summary"]["by_label"] = dict(label_counts)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    

    def _export_observation_csvs(self, output_dir: Path,
                             root_dir: Path | None = None) -> None:
        """Write observation counts and pattern summaries as CSV files."""
        for category, kw_dict in self.observations.items():
            if not kw_dict:
                continue
            _, items = self._sorted_observation_items(category, kw_dict)
            csv_path = output_dir / f"{category}.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["word", "correct", "total", "precision", "count_all"])
                for word, stats in items:
                    writer.writerow([
                        word, stats.correct, stats.count,
                        f"{stats.precision:.4f}", stats.count_all,
                    ])

        if root_dir is not None:
            # Route auxiliary outputs to dedicated subdirs
            reports_dir = root_dir / "reports"
            reports_dir.mkdir(exist_ok=True)
            self._export_observation_measure_contract(
                reports_dir / "observation_measure_contract.json"
            )
            self._export_target_context_stats(reports_dir)

            analysis_dir = root_dir / "analysis"
            analysis_dir.mkdir(exist_ok=True)
            self._export_relation_patterns_yaml(
                analysis_dir, name="relation_patterns_audit.yaml"
            )
        else:
            # older / standalone call: write beside observation CSVs
            self._export_target_context_stats(output_dir)
            self._export_relation_patterns_yaml(output_dir)

    def _export_observation_measure_contract(self, path: Path) -> None:
        """Write explicit numerator/denominator semantics beside scored CSVs."""
        contract = {}
        for category in self.observations:
            metadata = category_metadata_for(category)
            contract[category] = {
                "score_mode": metadata.score_mode,
                "observation_unit": metadata.observation_unit,
                "success_definition": metadata.success_definition,
                "numerator_column": "correct",
                "denominator_column": "total",
            }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(contract, handle, indent=2, ensure_ascii=False)

    def _export_target_context_stats(self, output_dir: Path) -> None:
        """Export a review-oriented propagation table for target_context."""
        kw_dict = self.observations.get("target_context", {})
        if not kw_dict:
            return

        csv_path = output_dir / "target_context_stats.csv"
        items = sorted(kw_dict.items(), key=lambda x: x[1].count_all, reverse=True)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "word",
                "decision",
                "observation_unit",
                "propagation_occurrences",
                "propagated_pair_count",
                "propagated_tp",
                "propagated_fp",
                "propagation_precision",
                "stored_pair_examples",
                "stored_no_gold_examples",
                "stored_no_pair_examples",
                "pair_contributing_occurrences",
                "nonprojecting_occurrences",
                "owner_occurrences",
                "descriptor_occurrences",
                "nested_occurrences",
                "notes",
            ])
            for word, stats in items:
                writer.writerow([
                    word,
                    "review",
                    category_metadata_for("target_context").observation_unit,
                    stats.count_all,
                    stats.count,
                    stats.correct,
                    stats.fp_count,
                    f"{stats.precision:.4f}",
                    len(stats.positive) + len(stats.negative),
                    len(stats.no_gold),
                    len(stats.no_pairs),
                    stats.pair_contributing_occurrences,
                    stats.nonprojecting_occurrences,
                    stats.owner_occurrences,
                    stats.descriptor_occurrences,
                    stats.nested_occurrences,
                    "",
                ])

    def _export_relation_patterns_yaml(self, output_dir: Path,
                                       name: str = "relation_patterns.yaml") -> None:
        """Export relation patterns as YAML to *output_dir*/*name*."""
        if not self._pattern_observations.has_patterns():
            return
        meta = {
            "format": "pattern_observations_v1",
            "source": f"{self.dataset}/extract",
            "date": self.timestamp[:8],
            "counts": {
                "total_patterns": self._pattern_observations.count_patterns(),
            },
        }
        self._pattern_observations.to_yaml(output_dir / name, meta=meta)


    def _export_database(self, path: Path, run_meta: dict | None = None) -> None:
        """Write all accumulated data to an SQLite events.db file."""
        from src.export.event_database import EventDatabase
        with EventDatabase(path) as db:
            db.populate(self, run_meta=run_meta)


    def _export_observation_tsvs(self, output_dir: Path) -> None:
        """Export optional observation TSVs for simple external tooling."""
        for category, kw_dict in self.observations.items():
            if not kw_dict:
                continue

            _, items = self._sorted_observation_items(category, kw_dict)

            tsv_path = output_dir / f"{category}.tsv"
            with open(tsv_path, "w", encoding="utf-8") as f:
                f.write("word\tcorrect_count\ttotal_count\tprecision\ttotal_all\texample\n")
                for word, stats in items:
                    # Get first example sentence
                    example = ""
                    all_examples = stats.positive + stats.negative + stats.no_gold
                    if all_examples:
                        example = all_examples[0].sentence[:100].replace("\t", " ").replace("\n", " ")

                    f.write(f"{word}\t{stats.correct}\t{stats.count}\t{stats.precision:.3f}\t{stats.count_all}\t{example}\n")

        # NMod-as-patient pattern exports live in relation_patterns.yaml.

    @property
    def pattern_observations(self):
        return self._pattern_observations


    def finalize(self):
        """Return the completed bundle for analysis and export."""
        return self
