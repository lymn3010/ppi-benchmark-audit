"""Group related path signatures for analysis; extraction uses the original evidence."""
from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path


_DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "lexicons" / "nominal_to_verbal.csv"
)


def load_nominal_to_verbal(path: Path | None = None) -> dict[str, str]:
    """Load the cached nominal->verbal table; empty if the file is missing."""
    csv_path = (path or _DEFAULT_CSV_PATH).resolve()
    return _load_csv_cached(str(csv_path))


@lru_cache(maxsize=4)
def _load_csv_cached(path_str: str) -> dict[str, str]:
    csv_path = Path(path_str)
    if not csv_path.exists():
        return {}
    out: dict[str, str] = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                nominal = row[0].strip().lower()
                verbal = row[1].strip().lower()
                if nominal and verbal:
                    out[nominal] = verbal
    return out


def canonicalize_anchor(anchor: str, nominal_to_verbal: dict[str, str]) -> str:
    """Map ``interaction`` -> ``interact`` etc.; unmapped lemmas pass through lowercased."""
    a = (anchor or "").strip().lower()
    return nominal_to_verbal.get(a, a)


def canonicalize_anchors(
    anchors: list[str] | tuple[str, ...] | None,
    nominal_to_verbal: dict[str, str],
) -> list[str]:
    """Canonicalize and deduplicate a list of anchor lemmas. Sorted."""
    if not anchors:
        return []
    seen = {canonicalize_anchor(a, nominal_to_verbal) for a in anchors}
    return sorted(seen)


_PREP_EQUIVALENCE = {
    "among": "between/among",
    "between": "between/among",
}


def canonicalize_prep(prep: str) -> str:
    """Normalize review-equivalent prepositions without changing exact rows."""
    p = (prep or "").strip().lower()
    if not p or p == "_":
        return "_"
    return _PREP_EQUIVALENCE.get(p, p)


def canonicalize_prep_signature(prep: str | list[str] | tuple[str, ...] | None) -> str:
    """Canonicalize a single prep or a ``+``-joined endpoint-prep signature."""
    if prep is None:
        return "_"
    if isinstance(prep, (list, tuple)):
        parts = [str(p) for p in prep]
    else:
        parts = [p for p in str(prep).split("+")]
    out = [canonicalize_prep(p) for p in parts if canonicalize_prep(p) != "_"]
    return "+".join(out) if out else "_"


def _norm_dep(dep: str) -> str:
    d = (dep or "").strip().lower()
    if d == "obj":
        return "dobj"
    if d in {"obl", "pobj"}:
        return "nmod"
    if d in {"nsubj:pass", "csubjpass"}:
        return "nsubjpass"
    if d == "csubj":
        return "nsubj"
    return d or "_"


def _prep_from_row(row: dict) -> str:
    edge_prep = (row.get("edge_prep") or "").strip()
    if edge_prep:
        return canonicalize_prep_signature(edge_prep)
    endpoint_preps = row.get("endpoint_preps") or ()
    return canonicalize_prep_signature(endpoint_preps)


def _anchor_lemmas_from_row(row: dict, nominal_to_verbal: dict[str, str]) -> list[str]:
    anchors = canonicalize_anchors(row.get("anchor_lemmas") or [], nominal_to_verbal)
    if anchors:
        return anchors
    # Fall back to the path apex so anchorless families do not collapse into "_".
    apex = (
        row.get("apex_lemma")
        or row.get("external_apex_lemma")
        or ""
    ).strip()
    if not apex or re.fullmatch(r"protein\d+", apex.lower()):
        return []
    return [canonicalize_anchor(apex, nominal_to_verbal)]


def _canonical_apex(row: dict, nominal_to_verbal: dict[str, str]) -> str:
    apex = (
        row.get("apex_lemma")
        or row.get("external_apex_lemma")
        or ""
    ).strip().lower()
    if not apex or re.fullmatch(r"protein\d+", apex):
        return ""
    return canonicalize_anchor(apex, nominal_to_verbal)


def _semantic_predicate(row: dict, nominal_to_verbal: dict[str, str]) -> str:
    anchors = _anchor_lemmas_from_row(row, nominal_to_verbal)
    apex = _canonical_apex(row, nominal_to_verbal)
    construction = (row.get("construction") or "").strip()
    if construction == "nested_dobj" and anchors and apex and apex not in anchors:
        # Keep the outer controller of a nested event ("A prevents binding of B").
        return f"{apex}>{','.join(anchors)}"
    if anchors:
        return ",".join(anchors)
    return apex or "_"


def _semantic_slot_for_edge(edge: dict, *, apex_pos: str) -> str:
    """Map endpoint syntax to a coarse ARG0/ARG1-like role for review grouping."""
    dep = _norm_dep(str(edge.get("dep") or ""))
    prep = canonicalize_prep(str(edge.get("prep") or ""))
    pos = (apex_pos or "").strip().upper()

    if prep == "by" or dep == "agent":
        return "ARG0"
    if dep == "nsubj":
        return "ARG0" if pos == "VERB" else "ARG?"
    if dep == "nsubjpass":
        return "ARG1"
    if dep == "dobj":
        return "ARG1"
    if dep == "nmod":
        if prep == "of":
            return "ARG1"
        if prep in {"to", "with", "between/among", "on"}:
            return "ARG1"
        if prep != "_":
            return f"NMOD:{prep}"
        return "NMOD"
    if dep in {"compound", "amod"}:
        return "MOD"
    if dep in {"conj", "appos"}:
        return dep.upper()
    return dep.upper()


def _semantic_edge_roles(row: dict) -> list[str]:
    apex_pos = str(row.get("apex_pos") or row.get("external_apex_pos") or "")
    edges = row.get("endpoint_edges") or ()
    roles: list[str] = []
    for edge in edges:
        if isinstance(edge, dict):
            roles.append(_semantic_slot_for_edge(edge, apex_pos=apex_pos))
    if roles:
        return sorted(roles)
    return []


def _coarse_role_signature(roles: list[str]) -> str:
    if not roles:
        return "_"
    arg_roles = {"ARG0", "ARG1", "ARG?"}
    role_set = set(roles)
    if role_set <= {"ARG0", "ARG1"}:
        return "ARG_PAIR"
    if any(r in arg_roles for r in roles) and "MOD" in role_set:
        return "ARG_MOD"
    if any(r in arg_roles for r in roles) and "CONJ" in role_set:
        return "ARG_CONJ"
    if role_set <= {"CONJ", "APPOS"}:
        return "COREF_COORD"
    return "+".join(roles)


def semantic_signature(
    row: dict,
    *,
    nominal_to_verbal: dict[str, str] | None = None,
    keep_surface_family: bool = False,
    role_detail: str = "strict",
) -> str:
    """Review signature that merges active, passive and nominalized paraphrases.

    Diagnostic only; never replaces exact signatures or feeds extraction.
    """
    if nominal_to_verbal is None:
        nominal_to_verbal = load_nominal_to_verbal()
    lemma = _semantic_predicate(row, nominal_to_verbal)
    roles = _semantic_edge_roles(row)
    if roles:
        role_sig = (
            _coarse_role_signature(roles)
            if role_detail == "coarse"
            else "+".join(roles)
        )
    else:
        role_sig = _prep_from_row(row)
    construction = (row.get("construction") or "bare_path").strip() or "bare_path"
    # Keep a coarse construction family.
    if construction in {"verbal_path", "nested_dobj"}:
        family = "event_path"
    elif construction in {"nominal_prep_path", "nominal_compound"}:
        family = "nominal_path"
    else:
        family = construction
    if keep_surface_family:
        return "|".join(("lex_sem", lemma, family, role_sig))
    return "|".join(("lex_sem", lemma, role_sig))


def abstracted_signature(
    row: dict,
    *,
    nominal_to_verbal: dict[str, str] | None = None,
    keep_context: bool = False,
) -> str:
    """Return ``lex_abs|construction|anchors|prep`` (plus context kinds if ``keep_context``)."""
    if nominal_to_verbal is None:
        nominal_to_verbal = load_nominal_to_verbal()
    construction = (row.get("construction") or "").strip() or "bare_path"
    anchors = _anchor_lemmas_from_row(row, nominal_to_verbal)
    anchors_csv = ",".join(anchors) if anchors else "_"
    prep = _prep_from_row(row)
    if keep_context:
        left = (row.get("left_context_kind") or "none").strip() or "none"
        right = (row.get("right_context_kind") or "none").strip() or "none"
        return "|".join(("lex_abs", construction, anchors_csv, prep, left, right))
    return "|".join(("lex_abs", construction, anchors_csv, prep))


def aggregate_by_abstracted(
    rows: list[dict],
    *,
    nominal_to_verbal: dict[str, str] | None = None,
    keep_context: bool = False,
) -> dict[str, dict]:
    """Tally rows by ``abstracted_signature``; writes nothing."""
    if nominal_to_verbal is None:
        nominal_to_verbal = load_nominal_to_verbal()
    agg: dict[str, dict] = {}
    seen_pair: dict[str, set[tuple[str, tuple[int, int]]]] = {}
    for row in rows:
        sig = abstracted_signature(
            row, nominal_to_verbal=nominal_to_verbal, keep_context=keep_context,
        )
        bucket = agg.setdefault(sig, {
            "abstracted_signature": sig,
            "observations": 0,
            "sentence_pair_observations": 0,
            "gold_positive_hits": 0,
            "gold_negative_hits": 0,
            "no_gold_hits": 0,
            "exact_signatures": set(),
        })
        bucket["observations"] += 1
        bucket["exact_signatures"].add(row.get("pattern_signature") or "")
        sid = str(row.get("sentence_id", ""))
        pair = row.get("pair") or []
        canon = (sid, tuple(sorted(int(x) for x in pair)) if len(pair) == 2 else (-1, -1))
        seen = seen_pair.setdefault(sig, set())
        if canon not in seen:
            seen.add(canon)
            bucket["sentence_pair_observations"] += 1
            status = row.get("gold_status")
            if status == "gold_positive":
                bucket["gold_positive_hits"] += 1
            elif status == "gold_negative":
                bucket["gold_negative_hits"] += 1
            else:
                bucket["no_gold_hits"] += 1
    for bucket in agg.values():
        bucket["evaluable"] = bucket["gold_positive_hits"] + bucket["gold_negative_hits"]
        bucket["annotation_agreement"] = (
            bucket["gold_positive_hits"] / bucket["evaluable"]
            if bucket["evaluable"] else 0.0
        )
        bucket["exact_signatures"] = sorted(bucket["exact_signatures"])
    return agg


def aggregate_by_semantic(
    rows: list[dict],
    *,
    nominal_to_verbal: dict[str, str] | None = None,
    keep_surface_family: bool = False,
    role_detail: str = "strict",
) -> dict[str, dict]:
    """Group rows by ``semantic_signature`` for review-level convergence."""
    if nominal_to_verbal is None:
        nominal_to_verbal = load_nominal_to_verbal()
    agg: dict[str, dict] = {}
    for row in rows:
        sig = semantic_signature(
            row,
            nominal_to_verbal=nominal_to_verbal,
            keep_surface_family=keep_surface_family,
            role_detail=role_detail,
        )
        bucket = agg.setdefault(sig, {
            "semantic_signature": sig,
            "family_count": 0,
            "observations": 0,
            "gold_positive_hits": 0,
            "gold_negative_hits": 0,
            "no_gold_hits": 0,
            "exact_signatures": set(),
            "abstracted_signatures": set(),
        })
        bucket["family_count"] += 1
        bucket["observations"] += int(
            row.get("sentence_pair_observations")
            or row.get("observations")
            or 1
        )
        bucket["gold_positive_hits"] += int(row.get("gold_positive_hits") or 0)
        bucket["gold_negative_hits"] += int(row.get("gold_negative_hits") or 0)
        bucket["no_gold_hits"] += int(row.get("no_gold_hits") or 0)
        bucket["exact_signatures"].add(row.get("pattern_signature") or "")
        bucket["abstracted_signatures"].add(
            abstracted_signature(row, nominal_to_verbal=nominal_to_verbal)
        )
    for bucket in agg.values():
        evaluable = bucket["gold_positive_hits"] + bucket["gold_negative_hits"]
        bucket["evaluable"] = evaluable
        bucket["annotation_agreement"] = (
            bucket["gold_positive_hits"] / evaluable if evaluable else 0.0
        )
        bucket["exact_signatures"] = sorted(bucket["exact_signatures"])
        bucket["abstracted_signatures"] = sorted(bucket["abstracted_signatures"])
    return agg
