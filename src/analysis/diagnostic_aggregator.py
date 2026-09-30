"""Aggregate lexicalized-path candidates by ``pattern_signature``.

Audit evidence only; not a runtime pattern source.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml


# Review-priority thresholds only; they never change runtime patterns.
HIGH_AGREEMENT_MIN_OBSERVATIONS: int = 5
HIGH_AGREEMENT_MIN_RATE: float = 0.80
LOW_AGREEMENT_MAX_RATE: float = 0.20
REVIEW_PRIORITY_MIN_OBSERVATIONS: int = 3

LEXPATH_AUDIT_FORMAT: str = "lexicalized_path_audit_v1"

_MAX_EXAMPLES_PER_FAMILY: int = 5


@dataclass
class _Aggregate:
    pattern_signature: str
    construction: str = ""
    anchors: list[str] = field(default_factory=list)
    edge_prep: str = ""
    left_kind: str = ""
    right_kind: str = ""
    apex_lemma: str = ""
    apex_pos: str = ""
    endpoint_edges: list[dict] = field(default_factory=list)
    observations: int = 0
    sentence_pair_observations: int = 0
    gold_positive_hits: int = 0
    gold_negative_hits: int = 0
    no_gold_hits: int = 0
    structurally_explained: int = 0
    examples: list[dict] = field(default_factory=list)
    seen_sentence_pairs: set[tuple[str, tuple[int, int]]] = field(default_factory=set)

    def add(self, row: dict) -> None:
        self.observations += 1
        sid = str(row.get("sentence_id", ""))
        pair = row.get("pair") or []
        canon = (sid, tuple(sorted(int(x) for x in pair)) if len(pair) == 2 else (-1, -1))
        if canon not in self.seen_sentence_pairs:
            self.seen_sentence_pairs.add(canon)
            self.sentence_pair_observations += 1
            gs = row.get("gold_status")
            if gs == "gold_positive":
                self.gold_positive_hits += 1
            elif gs == "gold_negative":
                self.gold_negative_hits += 1
            else:
                self.no_gold_hits += 1
            if row.get("existing_structural_pattern_key"):
                self.structurally_explained += 1
        if len(self.examples) < _MAX_EXAMPLES_PER_FAMILY:
            self.examples.append({
                "sentence_id": sid,
                "pair": list(pair),
                "gold_status": row.get("gold_status"),
                "surface": row.get("surface_window", ""),
                "dep_path": row.get("dep_path", []),
                "existing_structural_pattern_key":
                    row.get("existing_structural_pattern_key", ""),
            })

    @property
    def evaluable(self) -> int:
        return self.gold_positive_hits + self.gold_negative_hits

    @property
    def annotation_agreement(self) -> float:
        return self.gold_positive_hits / self.evaluable if self.evaluable else 0.0

    @property
    def review_signal(self) -> str:
        obs = self.sentence_pair_observations
        agr = self.annotation_agreement
        if obs >= HIGH_AGREEMENT_MIN_OBSERVATIONS and self.evaluable >= HIGH_AGREEMENT_MIN_OBSERVATIONS:
            if agr >= HIGH_AGREEMENT_MIN_RATE:
                return "high_agreement"
            if agr <= LOW_AGREEMENT_MAX_RATE:
                return "low_agreement"
        if obs >= REVIEW_PRIORITY_MIN_OBSERVATIONS:
            return "mixed_or_sparse"
        return "insufficient_support"


def aggregate_path_candidates(
    rows: Iterable[dict],
) -> list[_Aggregate]:
    """Group raw path candidate rows by ``pattern_signature``.

    Deterministic: sorts the result by ``pattern_signature``.
    """
    by_sig: dict[str, _Aggregate] = {}
    for row in rows:
        sig = row.get("pattern_signature") or ""
        if not sig:
            continue
        agg = by_sig.get(sig)
        if agg is None:
            agg = _Aggregate(
                pattern_signature=sig,
                construction=row.get("construction", ""),
                anchors=list(row.get("anchor_lemmas") or []),
                edge_prep=row.get("edge_prep", ""),
                left_kind=row.get("left_context_kind", ""),
                right_kind=row.get("right_context_kind", ""),
                apex_lemma=row.get("apex_lemma", ""),
                apex_pos=row.get("apex_pos", ""),
                endpoint_edges=list(row.get("endpoint_edges") or []),
            )
            by_sig[sig] = agg
        agg.add(row)
    return sorted(by_sig.values(), key=lambda a: a.pattern_signature)


def _family_row(agg: _Aggregate) -> dict:
    """Return the final review row for one lexicalized path family."""
    return {
        "pattern_signature": agg.pattern_signature,
        "chain": _chain_from_aggregate(agg),
        "evidence_class": "lexicalized_path",
        "status": "candidate",
        "runtime": False,
        "review_status": "candidate",
        "source": "mined",
        "review_signal": agg.review_signal,
        "construction": agg.construction,
        "apex_lemma": agg.apex_lemma,
        "apex_pos": agg.apex_pos,
        "endpoint_edges": agg.endpoint_edges,
        "anchors": agg.anchors,
        "edge_prep": agg.edge_prep,
        "left_context_kind": agg.left_kind,
        "right_context_kind": agg.right_kind,
        "statistics": {
            "observations": agg.observations,
            "sentence_pair_observations": agg.sentence_pair_observations,
            "gold_positive": agg.gold_positive_hits,
            "gold_negative": agg.gold_negative_hits,
            "no_gold": agg.no_gold_hits,
            "evaluable": agg.evaluable,
            "agreement": round(agg.annotation_agreement, 4),
            "structurally_explained": agg.structurally_explained,
        },
        "examples": agg.examples,
    }


def _chain_from_aggregate(agg: _Aggregate) -> str:
    """Human-readable YAML-style path notation for the audit surface."""
    apex = agg.apex_lemma or (agg.anchors[0] if agg.anchors else "")
    if not apex:
        return "[A] ... [B]"
    source_bits: list[str] = []
    target_bits: list[str] = []
    for edge in agg.endpoint_edges or []:
        role = str(edge.get("role") or "")
        prep = str(edge.get("prep") or edge.get("case") or "")
        dep = str(edge.get("dep") or "")
        bit = f'"{prep}"' if prep else (f'"{dep}"' if dep else "")
        if role == "source":
            source_bits.append(bit)
        elif role == "target":
            target_bits.append(bit)
    left = " ".join(["[A]", *[b for b in source_bits if b]])
    right = " ".join([*[b for b in target_bits if b], "[B]"])
    return f'{left} "{apex}" {right}'.replace("  ", " ").strip()


def write_lexicalized_path_audit(
    yaml_path: Path,
    json_path: Path,
    rows: Iterable[dict],
    *,
    source_tag: str,
    timestamp: str,
) -> None:
    """Write the supporting lexicalized-path audit artifact."""
    aggregates = aggregate_path_candidates(rows)
    families = [_family_row(agg) for agg in aggregates]
    payload = {
        "meta": {
            "format": LEXPATH_AUDIT_FORMAT,
            "source": source_tag,
            "date": timestamp[:8] if timestamp else "",
            "runtime_feedable": False,
            "notes": (
                "Mined lexicalized-path families for the unified pattern "
                "audit surface. These rows are evidence, not a runtime "
                "pattern source."
            ),
        },
        "families": families,
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(payload, f, sort_keys=False, allow_unicode=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
