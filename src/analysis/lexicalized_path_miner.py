"""Record shortest dependency paths between protein mentions (audit evidence only)."""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Any, Iterable
from src.system_rules import rule_frozenset, rule_int
from src.models.lexpath_pattern import LexPathPatternCandidate
from src.models.path_evidence import mined_path_lifecycle


# Diagnostic anchors for grouping mined paths; never used in extraction.
_ANCHOR_LEMMAS = rule_frozenset("diagnostic_priors.lexical_path_anchor_lemmas")
_METHODOLOGY_LEMMAS = rule_frozenset("diagnostic_priors.methodology_lemmas")
_EXISTENTIAL_LEMMAS = rule_frozenset("diagnostic_priors.existential_lemmas")

# Heuristic preposition set; only used to classify path edges.
_PREP_DEPS = rule_frozenset("diagnostic_priors.path_edge_relations")

_MAX_PATH_HOPS = rule_int("diagnostic_priors.max_path_hops")


def mine_path_candidates(
    doc,
    sentence_id: str,
    gold_pairs: Iterable[tuple[int, int]] | None,
    predicted_pairs: Iterable[tuple[int, int]] | None,
    candidate_relation_rows: Iterable[dict] | None,
    *,
    include_fp: bool = True,
) -> list[dict]:
    """Return review-only path candidates for one sentence; none if fewer than two proteins."""
    target_list = list(getattr(doc._, "target_list", []) or [])
    if len(target_list) < 2:
        return []

    # protein_index -> parser-node index
    token_idx_of: dict[int, int] = {}
    for i, target in enumerate(target_list):
        protein_index = int(target.get("protein_index", i))
        token_idx_of[protein_index] = int(target["token_id"])

    gold_set: set[tuple[int, int]] | None
    if gold_pairs is None:
        gold_set = None
    else:
        gold_set = {tuple(sorted(map(int, p))) for p in gold_pairs}

    pred_set: set[tuple[int, int]] = {
        tuple(sorted(map(int, p))) for p in (predicted_pairs or [])
    }

    # Canonical pair -> pattern key of the first covering candidate row.
    covered_by_structural: dict[tuple[int, int], str] = {}
    for rel in candidate_relation_rows or []:
        pair = rel.get("pair") or []
        if len(pair) != 2:
            continue
        canon = tuple(sorted((int(pair[0]), int(pair[1]))))
        if canon in covered_by_structural:
            continue
        key = rel.get("pattern_key") or ""
        covered_by_structural[canon] = key

    pairs_to_mine: list[tuple[tuple[int, int], str]] = []
    if gold_set is not None:
        for pair in sorted(gold_set):
            if pair not in covered_by_structural:
                pairs_to_mine.append((pair, "gold_positive"))
        if include_fp:
            for pair in sorted(pred_set):
                if pair in gold_set:
                    continue
                # Predicted but not gold: record it and whether a structural row exists.
                pairs_to_mine.append((pair, "gold_negative"))
    else:
        for pair in sorted(pred_set):
            pairs_to_mine.append((pair, "no_gold"))

    out: list[dict] = []
    for pair, gold_status in pairs_to_mine:
        candidate = _build_candidate(
            doc=doc,
            sentence_id=sentence_id,
            pair=pair,
            gold_status=gold_status,
            token_idx_of=token_idx_of,
            structural_pattern_key=covered_by_structural.get(pair, ""),
        )
        if candidate is not None:
            candidate["path_evidence"] = mined_path_lifecycle()
            out.append(candidate)
    return out


def mine_path_candidates_from_parsed(
    parsed,
    sentence_id: str,
    gold_pairs: Iterable[tuple[int, int]] | None,
    predicted_pairs: Iterable[tuple[int, int]] | None,
    candidate_relation_rows: Iterable[dict] | None,
    *,
    include_fp: bool = True,
) -> list[dict]:
    """Parser-neutral adapter for :func:`mine_path_candidates`."""
    return mine_path_candidates(
        _ParsedDocView(parsed),
        sentence_id=sentence_id,
        gold_pairs=gold_pairs,
        predicted_pairs=predicted_pairs,
        candidate_relation_rows=candidate_relation_rows,
        include_fp=include_fp,
    )


class _ParsedTokenView:
    def __init__(self, doc: "_ParsedDocView", node):
        self.doc = doc
        self.i = int(node.i)
        self.text = node.text
        self.lemma_ = (node.lemma or node.text).lower()
        self.pos_ = node.pos
        self.dep_ = node.dep
        self.idx = int(node.char_start)
        self._head_i = int(node.head_i)

    @property
    def head(self):
        if self._head_i < 0 or self._head_i == self.i:
            return self
        return self.doc[self._head_i]

    @property
    def children(self):
        return [t for t in self.doc.tokens if t._head_i == self.i and t.i != self.i]


class _ParsedDocView:
    def __init__(self, parsed):
        self.text = parsed.text
        self.tokens = [_ParsedTokenView(self, node) for node in parsed.nodes]
        self._ = SimpleNamespace(target_list=self._target_list(parsed))

    @staticmethod
    def _target_list(parsed) -> list[dict]:
        out: list[dict] = []
        for marker, node_indices in sorted(
            (parsed.protein_map or {}).items(),
            key=lambda item: _protein_marker_index(item[0]),
        ):
            protein_index = _protein_marker_index(marker)
            if protein_index < 0 or not node_indices:
                continue
            out.append({
                "protein_index": protein_index,
                "token_id": int(node_indices[0]),
                "text": marker,
            })
        return out

    def __getitem__(self, i):
        return self.tokens[i]

    def __len__(self):
        return len(self.tokens)


def _protein_marker_index(marker: str) -> int:
    text = str(marker or "")
    if text.startswith("PROTEIN"):
        suffix = text[len("PROTEIN"):]
        if suffix.isdigit():
            return int(suffix)
    return -1


def _build_candidate(
    doc,
    sentence_id: str,
    pair: tuple[int, int],
    gold_status: str,
    token_idx_of: dict[int, int],
    structural_pattern_key: str,
) -> dict | None:
    i, j = pair
    if i not in token_idx_of or j not in token_idx_of:
        return None
    tok_a = doc[token_idx_of[i]]
    tok_b = doc[token_idx_of[j]]

    path_tokens = _shortest_dep_path(tok_a, tok_b, max_hops=_MAX_PATH_HOPS)
    if not path_tokens:
        # Disconnected subtrees still get a row with an empty path.
        path_tokens = []

    dep_path: list[dict] = [
        {
            "i": int(tok.i),
            "lemma": (tok.lemma_ or tok.text).lower(),
            "pos": tok.pos_,
            "dep": tok.dep_,
            "text": tok.text,
        }
        for tok in path_tokens
    ]

    path_lemmas = [step["lemma"] for step in dep_path]
    path_deps = [step["dep"].lower() for step in dep_path]
    path_pos = [step["pos"] for step in dep_path]

    anchor_lemmas = sorted({lem for lem in path_lemmas if lem in _ANCHOR_LEMMAS})

    # Apex: internal path token closest to root ("effect" in "effect of A on B").
    endpoint_ids = {tok_a.i, tok_b.i}
    apex_tok = _find_apex(path_tokens, endpoint_ids)
    apex_lemma = (apex_tok.lemma_ or apex_tok.text).lower() if apex_tok is not None else ""
    apex_pos = apex_tok.pos_ if apex_tok is not None else ""
    apex_dep = apex_tok.dep_ if apex_tok is not None else ""
    external_apex = (
        _find_external_relation_head(tok_a, tok_b, endpoint_ids)
        if apex_tok is None else None
    )
    external_apex_lemma = (
        (external_apex.lemma_ or external_apex.text).lower()
        if external_apex is not None else ""
    )
    external_apex_pos = external_apex.pos_ if external_apex is not None else ""
    external_apex_dep = external_apex.dep_ if external_apex is not None else ""

    # Endpoint case preposition and dep relation (UD puts prepositions off-path).
    endpoint_edges = [
        _endpoint_edge(tok_a, path_tokens, int(pair[0])),
        _endpoint_edge(tok_b, path_tokens, int(pair[1])),
    ]
    endpoint_preps = sorted({e["prep"] for e in endpoint_edges if e["prep"]})

    # Prefer an on-path ADP, else the first endpoint case preposition.
    edge_prep = ""
    for step in dep_path:
        if step["dep"].lower() in _PREP_DEPS and step["pos"] == "ADP":
            edge_prep = step["lemma"]
            break
    if not edge_prep and endpoint_preps:
        edge_prep = endpoint_preps[0]

    compound_hyphen = _is_hyphen_compound(tok_a, tok_b)

    apex_i = apex_tok.i if apex_tok is not None else -1
    has_intermediate_noun = any(
        t.pos_ in ("NOUN", "PROPN") and t.i not in endpoint_ids and t.i != apex_i
        for t in path_tokens
    )

    construction = _classify_construction(
        path_lemmas=path_lemmas,
        path_deps=path_deps,
        path_pos=path_pos,
        compound_hyphen=compound_hyphen,
        edge_prep=edge_prep,
        apex_pos=apex_pos,
        endpoint_preps=endpoint_preps,
        has_intermediate_noun=has_intermediate_noun,
    )

    left_kind = _classify_context(tok_a, side="left")
    right_kind = _classify_context(tok_b, side="right")

    surface_window = _surface_window(doc, path_tokens, tok_a, tok_b)

    pattern_signature = _make_signature(
        anchor_lemmas=anchor_lemmas,
        edge_prep=edge_prep,
        construction=construction,
        left_kind=left_kind,
        right_kind=right_kind,
        apex_lemma=apex_lemma,
        endpoint_preps=endpoint_preps,
    )

    candidate = {
        "sentence_id": sentence_id,
        "pair": [int(pair[0]), int(pair[1])],
        "gold_status": gold_status,
        "existing_structural_pattern_key": structural_pattern_key,
        "dep_path": dep_path,
        "path_length": max(0, len(dep_path) - 1),
        "anchor_lemmas": anchor_lemmas,
        "apex_lemma": apex_lemma,
        "apex_pos": apex_pos,
        "apex_dep": apex_dep,
        "external_apex_lemma": external_apex_lemma,
        "external_apex_pos": external_apex_pos,
        "external_apex_dep": external_apex_dep,
        "endpoint_edges": endpoint_edges,
        "endpoint_preps": endpoint_preps,
        "edge_prep": edge_prep,
        "compound_hyphen": compound_hyphen,
        "construction": construction,
        "left_context_kind": left_kind,
        "right_context_kind": right_kind,
        "left_context_lemmas": _window_lemmas(doc, tok_a.i, side="left", n=3),
        "right_context_lemmas": _window_lemmas(doc, tok_b.i, side="right", n=3),
        "surface_window": surface_window,
        "pattern_signature": pattern_signature,
    }
    # Standardized path-evidence signatures (diagnostic only).
    signature_payload = LexPathPatternCandidate.from_mined_row(candidate).to_dict()
    for key in (
        "prep_signature",
        "endpoint_attachment_signature",
        "carrier_heads",
        "carrier_signature",
        "typed_intermediate_heads",
        "typed_intermediate_signature",
        "typed_intermediate_signatures",
    ):
        candidate[key] = signature_payload[key]
    # Compatibility alias for carrier_signature.
    candidate["carrier_head_signature"] = signature_payload["carrier_signature"]
    return candidate



def _token_depth(tok) -> int:
    """Number of head hops from ``tok`` to its dependency-tree root."""
    depth = 0
    cur = tok
    seen: set[int] = set()
    while cur.head.i != cur.i and cur.i not in seen:
        seen.add(cur.i)
        cur = cur.head
        depth += 1
    return depth


def _find_apex(path_tokens: list, endpoint_ids: set):
    """Return the internal path token closest to root, or ``None`` for direct protein links."""
    internal = [t for t in path_tokens if t.i not in endpoint_ids]
    if not internal:
        return None
    return min(internal, key=lambda t: (_token_depth(t), t.i))


def _find_external_relation_head(tok_a, tok_b, endpoint_ids: set):
    """Find a lexical head just outside a direct protein-protein path (diagnostic only)."""
    candidates = []
    for endpoint in (tok_a, tok_b):
        for distance, candidate in enumerate(_ancestor_candidates(endpoint), start=1):
            if candidate.i in endpoint_ids:
                continue
            if _is_relation_head_candidate(candidate):
                candidates.append((distance, candidate))
                break
    if not candidates:
        return None
    # Prefer a head seen from both endpoints, then the nearest/root-most one.
    by_i: dict[int, list[tuple[int, Any]]] = {}
    for distance, candidate in candidates:
        by_i.setdefault(candidate.i, []).append((distance, candidate))
    shared = [
        (sum(distance for distance, _candidate in rows), rows[0][1])
        for rows in by_i.values()
        if len(rows) >= 2
    ]
    if shared:
        return min(shared, key=lambda item: (item[0], _token_depth(item[1]), item[1].i))[1]
    return min(candidates, key=lambda item: (item[0], _token_depth(item[1]), item[1].i))[1]


def _ancestor_candidates(tok, *, max_hops: int = 4):
    cur = tok
    seen: set[int] = set()
    for _ in range(max_hops):
        head = cur.head
        if head is None or head.i == cur.i or head.i in seen:
            break
        seen.add(head.i)
        yield head
        cur = head


def _is_relation_head_candidate(tok) -> bool:
    lemma = (tok.lemma_ or tok.text or "").lower()
    if not lemma or lemma.startswith("protein"):
        return False
    if tok.dep_.lower() in {"case", "det", "punct", "cc", "mark"}:
        return False
    return tok.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"}


def _case_prep(tok) -> str:
    """Return the preposition attaching ``tok`` to its head (UD or prep/pobj), else ""."""
    for child in tok.children:
        if child.dep_.lower() == "case" and child.pos_ in ("ADP", "SCONJ", "PART"):
            return (child.lemma_ or child.text).lower()
    if tok.dep_.lower() == "pobj" and tok.head is not None and tok.head.pos_ == "ADP":
        return (tok.head.lemma_ or tok.head.text).lower()
    return ""


def _endpoint_edge(endpoint_tok, path_tokens: list, protein_index: int) -> dict:
    """Describe how a protein endpoint attaches to the path (neighbour lemma, prep, dep)."""
    via_lemma = ""
    if len(path_tokens) >= 2:
        if path_tokens[0].i == endpoint_tok.i:
            neighbour = path_tokens[1]
        elif path_tokens[-1].i == endpoint_tok.i:
            neighbour = path_tokens[-2]
        else:
            neighbour = endpoint_tok.head
        via_lemma = (neighbour.lemma_ or neighbour.text).lower()
    return {
        "protein_index": int(protein_index),
        "dep": (endpoint_tok.dep_ or "").lower(),
        "prep": _case_prep(endpoint_tok),
        "via_lemma": via_lemma,
    }



def _shortest_dep_path(tok_a, tok_b, *, max_hops: int) -> list:
    """BFS over (head, children) edges; returns ordered list of tokens
    from ``tok_a`` to ``tok_b`` inclusive. Returns ``[]`` when no path
    is found within ``max_hops``.
    """
    if tok_a is tok_b:
        return [tok_a]
    target = tok_b.i
    visited: set[int] = {tok_a.i}
    parents: dict[int, int] = {}
    queue: deque[tuple[int, int]] = deque([(tok_a.i, 0)])
    tok_by_i: dict[int, Any] = {tok_a.i: tok_a}
    doc = tok_a.doc
    while queue:
        cur_i, hops = queue.popleft()
        if hops >= max_hops:
            continue
        cur = doc[cur_i]
        neighbours: list[int] = []
        head_i = cur.head.i
        if head_i != cur_i:
            neighbours.append(head_i)
        for child in cur.children:
            neighbours.append(child.i)
        for nb in neighbours:
            if nb in visited:
                continue
            visited.add(nb)
            parents[nb] = cur_i
            tok_by_i[nb] = doc[nb]
            if nb == target:
                # reconstruct
                rev = [nb]
                while rev[-1] in parents:
                    rev.append(parents[rev[-1]])
                rev.reverse()
                return [doc[i] for i in rev]
            queue.append((nb, hops + 1))
    return []



def _is_hyphen_compound(tok_a, tok_b) -> bool:
    """True when the two protein tokens are adjacent and joined by a hyphen
    in the surface text after parser tokenization. We look for any ``-`` token
    between the endpoints.
    """
    if tok_a.doc is not tok_b.doc:
        return False
    lo, hi = sorted((tok_a.i, tok_b.i))
    if hi - lo > 2:
        return False
    for k in range(lo + 1, hi):
        if tok_a.doc[k].text == "-":
            return True
    return False


def _classify_construction(
    *,
    path_lemmas: list[str],
    path_deps: list[str],
    path_pos: list[str],
    compound_hyphen: bool,
    edge_prep: str,
    apex_pos: str = "",
    endpoint_preps: list[str] | None = None,
    has_intermediate_noun: bool = False,
) -> str:
    if compound_hyphen:
        return "compound_hyphen"
    endpoint_preps = endpoint_preps or []
    has_verb = any(p == "VERB" for p in path_pos)
    # Verbal apex with an intermediate nominal: nested frame ("A increases methylation of B").
    if apex_pos == "VERB" or has_verb:
        if has_intermediate_noun:
            return "nested_dobj"
        return "verbal_path"
    # Nominal apex: endpoint prepositions distinguish the path ("effect of A on B").
    if apex_pos in ("NOUN", "PROPN"):
        if endpoint_preps or edge_prep:
            return "nominal_prep_path"
        return "nominal_compound"
    # older anchor-based fallback (parser schemes that yield no apex pos).
    has_noun_anchor = any(
        lem in _ANCHOR_LEMMAS and pos in ("NOUN", "PROPN")
        for lem, pos in zip(path_lemmas, path_pos)
    )
    if has_noun_anchor:
        return "nominal_prep_path" if edge_prep else "nominal_compound"
    if edge_prep:
        return "prep_path"
    return "bare_path"


def _classify_context(token, *, side: str) -> str:
    """Look at a small left/right window in the dep tree for methodology /
    existential cues. Side controls which direction we scan in surface
    order (cheap heuristic, not a parser feature).
    """
    doc = token.doc
    if side == "left":
        start = max(0, token.i - 5)
        window = range(start, token.i)
    else:
        end = min(len(doc), token.i + 6)
        window = range(token.i + 1, end)
    lemmas = {(doc[k].lemma_ or doc[k].text).lower() for k in window}
    if lemmas & _METHODOLOGY_LEMMAS:
        return "methodology"
    if lemmas & _EXISTENTIAL_LEMMAS:
        return "existential"
    return "none"


def _window_lemmas(doc, center_i: int, *, side: str, n: int) -> list[str]:
    if side == "left":
        start = max(0, center_i - n)
        rng = range(start, center_i)
    else:
        end = min(len(doc), center_i + n + 1)
        rng = range(center_i + 1, end)
    return [(doc[k].lemma_ or doc[k].text).lower() for k in rng]


def _surface_window(doc, path_tokens: list, tok_a, tok_b) -> str:
    if path_tokens:
        positions = [tok.i for tok in path_tokens]
    else:
        positions = [tok_a.i, tok_b.i]
    lo = min(positions)
    hi = max(positions)
    # Pad +/- 2 tokens for context
    lo = max(0, lo - 2)
    hi = min(len(doc) - 1, hi + 2)
    start = doc[lo].idx
    end = doc[hi].idx + len(doc[hi].text)
    text = doc.text[start:end]
    return text.replace("\n", " ").strip()


def _make_signature(
    *,
    anchor_lemmas: list[str],
    edge_prep: str,
    construction: str,
    left_kind: str,
    right_kind: str,
    apex_lemma: str = "",
    endpoint_preps: list[str] | None = None,
) -> str:
    """Return ``lex|construction|key_lemma|preps|left_kind|right_kind``."""
    endpoint_preps = endpoint_preps or []
    key = apex_lemma or (",".join(anchor_lemmas) if anchor_lemmas else "_")
    if endpoint_preps:
        prep = "+".join(endpoint_preps)
    else:
        prep = edge_prep or "_"
    return "|".join(("lex", construction, key, prep, left_kind, right_kind))
