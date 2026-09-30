"""Existing audit aggregation and pair normalization conventions."""
from __future__ import annotations
import json
import logging
from typing import Any
from .display import _yaml_chain_for_pattern
log = logging.getLogger(__name__)
_AUDIT_EVIDENCE_PRIORITY = (
    "structural",
    "relational_statement",
    "lexicalized_path",
    "candidate_pattern_ledger",
    "event_without_final_pair",
)

def _payload_protein_indices(value: Any) -> set[int]:
    """Collect protein identities from a serialized argument payload."""
    out: set[int] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {
                "targets", "direct_targets", "propagated_targets",
                "protein_indices", "direct_protein_indices",
                "inherited_protein_indices",
            } and isinstance(item, list):
                out.update(int(index) for index in item if isinstance(index, int))
            else:
                out.update(_payload_protein_indices(item))
    elif isinstance(value, list):
        for item in value:
            out.update(_payload_protein_indices(item))
    return out


def _json_pair_items(value: Any) -> list[Any]:
    """Return pair-shaped JSON items from a serialized pair list."""
    try:
        rows = json.loads(value or "[]") or []
    except Exception:
        return []
    return [
        item for item in rows
        if isinstance(item, (list, tuple)) and len(item) == 2
    ]


def _canonical_pair(value: Any) -> tuple[int, int]:
    """Normalize a pair-shaped item to the repository's undirected pair form."""
    a, b = int(value[0]), int(value[1])
    return tuple(sorted((a, b)))


def _canonical_pair_set_from_json(value: Any) -> set[tuple[int, int]]:
    """Parse serialized pairs into canonical undirected pair tuples."""
    return {_canonical_pair(item) for item in _json_pair_items(value)}


def _empty_audit_stats() -> dict[str, int | float]:
    """Common count bucket for pattern-audit evidence summaries."""
    return {
        "total": 0,
        "gold_positive": 0,
        "gold_negative": 0,
        "agreement": 0.0,
        "event_occurrences": 0,
        "mention_observations": 0,
    }


def _merge_audit_stats(target: dict, incoming: dict) -> None:
    """Add one audit statistics bucket into another in place."""
    for field in (
        "total",
        "gold_positive",
        "gold_negative",
        "event_occurrences",
        "mention_observations",
    ):
        target[field] = int(target.get(field) or 0) + int(incoming.get(field) or 0)
    target["agreement"] = round(
        float(target["gold_positive"]) / target["total"]
        if target["total"] else 0.0,
        4,
    )


def _primary_audit_evidence(stats_by_evidence: dict) -> str:
    """Evidence bucket used for the compact headline statistics."""
    for name in _AUDIT_EVIDENCE_PRIORITY:
        if name in stats_by_evidence:
            return name
    return "event_without_final_pair"


def _table_columns(conn, table_name: str) -> set[str]:
    """Return SQLite column names for a table or view."""
    return {
        str(row["name"] if "name" in row.keys() else row[1])
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }


def _optional_sql_column(
    columns: set[str],
    name: str,
    *,
    table_alias: str = "",
    fallback: str = "''",
) -> str:
    """SQL select expression that preserves old DB compatibility."""
    if name in columns:
        prefix = f"{table_alias}." if table_alias else ""
        return f"{prefix}{name}"
    return f"{fallback} AS {name}"


def _pattern_chain_groups(variants: list[dict]) -> list[dict]:
    """Group surface realizations by review-facing YAML chain."""
    groups: dict[str, dict] = {}
    for row in variants:
        chain = _yaml_chain_for_pattern(row)
        g = groups.setdefault(chain, {
            "chain": chain,
            "correct": 0,
            "incorrect": 0,
            "total": 0,
            "observed": {},
            "source": {},
            "realizations": [],
        })
        correct = int(row.get("tp") or 0)
        incorrect = int(row.get("fp") or 0)
        total = int(row.get("total") or 0)
        construction = str(row.get("construction") or "unknown")
        source = str(row.get("evidence_class") or "unknown")

        g["correct"] += correct
        g["incorrect"] += incorrect
        g["total"] += total

        obs = g["observed"].setdefault(construction, {
            "total": 0, "correct": 0, "incorrect": 0,
        })
        obs["total"] += total
        obs["correct"] += correct
        obs["incorrect"] += incorrect

        g["source"][source] = int(g["source"].get(source, 0)) + total
        g["realizations"].append(row)

    out = []
    for g in groups.values():
        total = int(g["total"] or 0)
        g["agreement"] = (float(g["correct"]) / total) if total else 0.0
        for obs in g["observed"].values():
            obs_total = int(obs["total"] or 0)
            obs["agreement"] = (float(obs["correct"]) / obs_total) if obs_total else 0.0
        g["realizations"].sort(
            key=lambda r: (-int(r.get("total") or 0), str(r.get("pattern_key") or ""))
        )
        out.append(g)
    out.sort(key=lambda g: (-int(g["total"] or 0), str(g["chain"])))
    return out
