"""Build one audit population for spreadsheet and machine-readable outputs."""
from __future__ import annotations
import logging
from pathlib import Path
log = logging.getLogger(__name__)
from .display import (
    _anchor_from_semantic_key,
    _trigger_type_for_construction,
    _yaml_chain_for_pattern,
    _audit_chain_for_relational_statement,
    _lexpath_chain,
)
from .statistics import (
    _empty_audit_stats,
    _merge_audit_stats,
    _primary_audit_evidence,
    _pattern_chain_groups,
)

class AuditPayloadBuilder:
    def __init__(self, queries, corpus):
        self.queries = queries
        self.corpus = corpus

    def _build_pattern_audit_payload(
        self, db_path: Path, analysis_dir: Path,
    ) -> dict:
        """Build the concise trigger-grouped human audit representation."""
        families = self.queries._load_semantic_pattern_rows(db_path, min_count=1)
        variants_by_sem = self.queries._load_pattern_variants_by_semantic_key(
            db_path, min_count=1,
        )
        examples_by_key = self.queries._load_pattern_examples_by_key(db_path)
        triggers: dict[str, dict] = {}

        def ensure_trigger(name: str) -> dict:
            clean_name = name or "(no lexical trigger)"
            return triggers.setdefault(clean_name, {
                "trigger": clean_name,
                "statistics": {
                    "total": 0, "gold_positive": 0, "gold_negative": 0,
                    "agreement": 0.0,
                },
                "statistics_by_evidence": {},
                "evidence": [],
                "patterns": [],
            })

        for fam in families:
            sem_key = str(fam.get("semantic_pattern_key") or "")
            trigger = ensure_trigger(_anchor_from_semantic_key(fam))
            if "structural" not in trigger["evidence"]:
                trigger["evidence"].append("structural")
            for group in _pattern_chain_groups(variants_by_sem.get(sem_key, [])):
                realizations: list[dict] = []
                examples: list[dict] = []
                for row in group.get("realizations") or []:
                    key = str(row.get("pattern_key") or "")
                    realizations.append({
                        "key": key,
                        "source": str(row.get("evidence_class") or "structural"),
                        "construction": str(row.get("construction") or ""),
                        "statistics": {
                            "total": int(row.get("total") or 0),
                            "gold_positive": int(row.get("tp") or 0),
                            "gold_negative": int(row.get("fp") or 0),
                            "agreement": round(float(row.get("precision") or 0.0), 4),
                        },
                    })
                    for example in examples_by_key.get(key, []):
                        if len(examples) >= 3:
                            break
                        examples.append(example)
                trigger["patterns"].append({
                    "chain": group["chain"],
                    "evidence_class": "structural",
                    "trigger_types": sorted({
                        _trigger_type_for_construction(construction, "structural")
                        for construction in (group.get("observed") or {})
                    }),
                    "statistics": {
                        "total": int(group.get("total") or 0),
                        "gold_positive": int(group.get("correct") or 0),
                        "gold_negative": int(group.get("incorrect") or 0),
                        "agreement": round(float(group.get("agreement") or 0.0), 4),
                    },
                    "observed": {
                        name: {
                            "total": int(obs.get("total") or 0),
                            "gold_positive": int(obs.get("correct") or 0),
                            "gold_negative": int(obs.get("incorrect") or 0),
                            "agreement": round(float(obs.get("agreement") or 0.0), 4),
                        }
                        for name, obs in sorted((group.get("observed") or {}).items())
                    },
                    "realizations": realizations,
                    "examples": examples,
                })

        for row in self.queries._load_candidate_pattern_rows(db_path):
            trigger_name = str(row.get("lemma") or "(no lexical trigger)")
            trigger = ensure_trigger(trigger_name)
            if "candidate_pattern_ledger" not in trigger["evidence"]:
                trigger["evidence"].append("candidate_pattern_ledger")
            total = int(row.get("sentence_pair_observations") or row.get("observations") or 0)
            positive = int(row.get("gold_positive_hits") or 0)
            negative = int(row.get("gold_negative_hits") or 0)
            agreement = float(row.get("annotation_agreement") or 0.0)
            construction = str(row.get("construction") or "")
            trigger["patterns"].append({
                "chain": _yaml_chain_for_pattern(row),
                "evidence_class": "candidate_pattern_ledger",
                "trigger_types": [
                    _trigger_type_for_construction(
                        construction,
                        str(row.get("evidence_class") or "structural"),
                    )
                ],
                "statistics": {
                    "total": total,
                    "gold_positive": positive,
                    "gold_negative": negative,
                    "agreement": round(agreement, 4),
                    "mention_observations": int(row.get("observations") or 0),
                },
                "observed": {
                    construction or "candidate_pattern": {
                        "total": total,
                        "gold_positive": positive,
                        "gold_negative": negative,
                        "agreement": round(agreement, 4),
                        "mention_observations": int(row.get("observations") or 0),
                    },
                },
                "realizations": [{
                    "key": row.get("pattern_key", ""),
                    "source": str(row.get("source") or "candidate_ledger"),
                    "construction": construction,
                    "statistics": {
                        "total": total,
                        "gold_positive": positive,
                        "gold_negative": negative,
                        "agreement": round(agreement, 4),
                        "mention_observations": int(row.get("observations") or 0),
                    },
                }],
                "examples": [],
            })

        for row in self.queries._load_relational_statement_rows(db_path, min_count=1):
            trigger = ensure_trigger(str(row.get("trigger") or row.get("schema") or "RELATIONAL"))
            if "relational_statement" not in trigger["evidence"]:
                trigger["evidence"].append("relational_statement")
            stats = row.get("statistics") or {}
            trigger["patterns"].append({
                "chain": _audit_chain_for_relational_statement(
                    str(row.get("schema") or ""),
                    str(row.get("trigger") or "") if row.get("trigger_type") == "relational_nominal" else "",
                ),
                "evidence_class": "relational_statement",
                "trigger_types": [row.get("trigger_type", "relation_schema")],
                "statistics": {
                    "total": int(stats.get("total") or 0),
                    "gold_positive": int(stats.get("gold_positive") or 0),
                    "gold_negative": int(stats.get("gold_negative") or 0),
                    "agreement": round(float(stats.get("agreement_with_gold") or 0.0), 4),
                },
                "observed": {
                    str(row.get("event_detail") or "relational"): {
                        "total": int(stats.get("total") or 0),
                        "gold_positive": int(stats.get("gold_positive") or 0),
                        "gold_negative": int(stats.get("gold_negative") or 0),
                        "agreement": round(float(stats.get("agreement_with_gold") or 0.0), 4),
                    },
                },
                "realizations": [
                    {
                        "key": key,
                        "source": "relational_statement",
                        "construction": row.get("event_detail", ""),
                        "statistics": {"total": int(total)},
                    }
                    for key, total in sorted((row.get("source") or {}).items())
                ],
                "examples": [
                    {
                        "sentence_id": str(example.get("sentence_id") or ""),
                        "pair": example.get("pair") or [],
                        "status": (
                            "gold_positive"
                            if example.get("gold_positive")
                            else "gold_negative"
                        ),
                        "text": str(example.get("text") or ""),
                    }
                    for example in row.get("examples") or []
                ],
            })

        for row in self.queries._load_lexpath_candidate_rows(analysis_dir):
            trigger_name = str(row.get("apex_lemma") or "(no lexical trigger)")
            if trigger_name.lower().startswith("protein"):
                trigger_name = "(no lexical trigger)"
            trigger = ensure_trigger(trigger_name)
            if "lexicalized_path" not in trigger["evidence"]:
                trigger["evidence"].append("lexicalized_path")
            row_stats = row.get("statistics") or {}
            total = int(
                row_stats.get("sentence_pair_observations")
                or row_stats.get("observations")
                or 0
            )
            positive = int(row_stats.get("gold_positive") or 0)
            negative = int(row_stats.get("gold_negative") or 0)
            agreement = float(row_stats.get("agreement") or 0.0)
            trigger["patterns"].append({
                "chain": row.get("chain") or _lexpath_chain(row),
                "evidence_class": "lexicalized_path",
                "trigger_types": ["lexical_path"],
                "statistics": {
                    "total": total,
                    "gold_positive": positive,
                    "gold_negative": negative,
                    "agreement": round(agreement, 4),
                },
                "observed": {
                    str(row.get("construction") or "lexicalized_path"): {
                        "total": total,
                        "gold_positive": positive,
                        "gold_negative": negative,
                        "agreement": round(agreement, 4),
                    },
                },
                "realizations": [{
                    "key": row.get("pattern_signature", ""),
                    "source": "lexicalized_path",
                    "construction": row.get("construction", ""),
                    "statistics": {
                        "total": total,
                        "gold_positive": positive,
                        "gold_negative": negative,
                        "agreement": round(agreement, 4),
                    },
                }],
                "examples": [{
                    "sentence_id": str(example.get("sentence_id") or ""),
                    "pair": example.get("pair") or [],
                    "status": str(example.get("gold_status") or ""),
                    "text": str(
                        example.get("surface")
                        or example.get("sentence")
                        or example.get("text")
                        or ""
                    ),
                } for example in (row.get("examples") or [])[:3]],
            })

        for row in self.queries._load_events_without_final_pair_rows(db_path):
            trigger = ensure_trigger(str(row.get("trigger") or "(no lexical trigger)"))
            if "event_without_final_pair" not in trigger["evidence"]:
                trigger["evidence"].append("event_without_final_pair")
            pattern_key = str(row.get("pattern_key") or "")
            segments = pattern_key.split("|")
            display_row = {
                "lemma": row.get("trigger", ""),
                "shape": segments[1] if len(segments) == 5 else row.get("yield_type", ""),
                "preposition": segments[3] if len(segments) == 5 else "_",
                "construction": row.get("construction", ""),
            }
            event_occurrences = int(row.get("event_occurrences") or 0)
            trigger["patterns"].append({
                "chain": _yaml_chain_for_pattern(display_row),
                "evidence_class": "event_without_final_pair",
                "trigger_types": [
                    _trigger_type_for_construction(
                        str(row.get("construction") or ""),
                        "structural",
                    )
                ],
                "statistics": {
                    "total": 0,
                    "gold_positive": 0,
                    "gold_negative": 0,
                    "agreement": 0.0,
                    "event_occurrences": event_occurrences,
                },
                "observed": {
                    str(row.get("event_detail") or "event_without_final_pair"): {
                        "total": 0,
                        "gold_positive": 0,
                        "gold_negative": 0,
                        "agreement": 0.0,
                        "event_occurrences": event_occurrences,
                    },
                },
                "realizations": [{
                    "key": pattern_key,
                    "source": "event_without_final_pair",
                    "construction": row.get("construction", ""),
                    "statistics": {
                        "total": 0,
                        "event_occurrences": event_occurrences,
                    },
                }],
                "examples": row.get("examples") or [],
            })

        for trigger in triggers.values():
            # Group rows by readable chain but keep per-evidence statistics.
            consolidated: dict[str, dict] = {}
            for pattern in trigger["patterns"]:
                evidence = str(pattern.get("evidence_class") or "unknown")
                trigger_types = [
                    str(item) for item in (
                        pattern.get("trigger_types")
                        or ([pattern.get("trigger_type")] if pattern.get("trigger_type") else [])
                    )
                    if item
                ]
                key = str(pattern.get("chain") or "")
                pattern["evidence"] = [evidence]
                pattern["trigger_types"] = sorted(set(trigger_types))
                pattern["statistics_by_evidence"] = {
                    evidence: dict(pattern.get("statistics") or {})
                }
                pattern["observed_by_evidence"] = {
                    evidence: dict(pattern.get("observed") or {})
                }
                pattern.pop("evidence_class", None)
                pattern.pop("observed", None)
                current = consolidated.get(key)
                if current is None:
                    consolidated[key] = pattern
                    continue
                if evidence not in current["evidence"]:
                    current["evidence"].append(evidence)
                for trigger_type in trigger_types:
                    if trigger_type and trigger_type not in current["trigger_types"]:
                        current["trigger_types"].append(trigger_type)
                incoming_stats = pattern["statistics_by_evidence"][evidence]
                stats = current["statistics_by_evidence"].setdefault(
                    evidence, _empty_audit_stats(),
                )
                _merge_audit_stats(stats, incoming_stats)
                target_observed = current["observed_by_evidence"].setdefault(evidence, {})
                for name, incoming_obs in pattern["observed_by_evidence"][evidence].items():
                    obs = target_observed.setdefault(name, _empty_audit_stats())
                    _merge_audit_stats(obs, incoming_obs)
                current["realizations"].extend(pattern.get("realizations") or [])
                existing_examples = {
                    (str(ex.get("sentence_id") or ""), tuple(ex.get("pair") or []))
                    for ex in current.get("examples") or []
                }
                for example in pattern.get("examples") or []:
                    ex_key = (
                        str(example.get("sentence_id") or ""),
                        tuple(example.get("pair") or []),
                    )
                    if ex_key not in existing_examples and len(current["examples"]) < 3:
                        current["examples"].append(example)
                        existing_examples.add(ex_key)
            trigger["patterns"] = list(consolidated.values())
            for pattern in trigger["patterns"]:
                pattern["evidence"].sort()
                pattern["trigger_types"].sort()
                by_evidence = pattern["statistics_by_evidence"]
                primary = _primary_audit_evidence(by_evidence)
                pattern["statistics"] = dict(by_evidence.get(primary) or {})
            trigger["patterns"].sort(
                key=lambda row: (
                    -int((row.get("statistics") or {}).get("total") or 0),
                    str(row.get("chain") or ""),
                )
            )
            evidence_stats: dict[str, dict] = {}
            for pattern in trigger["patterns"]:
                for evidence, pstats in (
                    pattern.get("statistics_by_evidence") or {}
                ).items():
                    stats = evidence_stats.setdefault(evidence, _empty_audit_stats())
                    _merge_audit_stats(stats, pstats)
            trigger["statistics_by_evidence"] = evidence_stats
            trigger_types = sorted({
                trigger_type
                for pattern in trigger["patterns"]
                for trigger_type in (pattern.get("trigger_types") or [])
                if trigger_type
            })
            trigger["trigger_types"] = trigger_types
            primary = _primary_audit_evidence(evidence_stats)
            trigger["statistics"] = dict(evidence_stats.get(primary) or {
                "total": 0, "gold_positive": 0, "gold_negative": 0,
                "agreement": 0.0,
            })
            trigger["evidence"].sort()

        ordered = sorted(
            triggers.values(),
            key=lambda row: (
                (
                    0 if "structural" in (row.get("evidence") or [])
                    else 1 if "relational_statement" in (row.get("evidence") or [])
                    else 2
                ),
                str(row.get("trigger") or "") == "(no lexical trigger)",
                -int((row.get("statistics") or {}).get("total") or 0),
                str(row.get("trigger") or ""),
            ),
        )
        return {
            "meta": {
                "format": "pattern_audit_v2",
                "runtime_feedable": False,
                "corpus": self.corpus,
                "source_db": str(db_path),
                "reading_order": "trigger -> pattern -> observed forms -> examples",
                "carrier_term": "carrier_word",
            },
            "evidence_summary": self.queries._load_semantic_evidence_summary(db_path),
            "carrier_trigger_matrix": self.queries._load_context_trigger_matrix(db_path),
            "triggers": ordered,
        }


    def _compact_trigger_word_rows(self, audit: dict) -> list[dict]:
        grouped: dict[tuple[str, str], dict] = {}
        for trigger in audit.get("triggers") or []:
            name = str(trigger.get("trigger") or "")
            if name.startswith("("):
                continue
            for pattern in trigger.get("patterns") or []:
                types = pattern.get("trigger_types") or trigger.get("trigger_types") or ["unknown"]
                for trigger_type in types:
                    key = (name, str(trigger_type or "unknown"))
                    row = grouped.setdefault(key, {
                        "trigger": name,
                        "trigger_type": str(trigger_type or "unknown"),
                        "total": 0,
                        "gold_positive": 0,
                        "gold_negative": 0,
                        "agreement": 0.0,
                        "evidence": set(),
                        "pattern_count": 0,
                        "events_without_final_pair": 0,
                    })
                    stats = pattern.get("statistics") or {}
                    row["total"] += int(stats.get("total") or 0)
                    row["gold_positive"] += int(stats.get("gold_positive") or 0)
                    row["gold_negative"] += int(stats.get("gold_negative") or 0)
                    row["pattern_count"] += 1
                    event_stats = (
                        pattern.get("statistics_by_evidence") or {}
                    ).get("event_without_final_pair") or {}
                    row["events_without_final_pair"] += int(
                        event_stats.get("event_occurrences") or 0
                    )
                    for evidence in pattern.get("evidence") or trigger.get("evidence") or []:
                        row["evidence"].add(str(evidence))
        rows = []
        for row in grouped.values():
            row["agreement"] = (
                float(row["gold_positive"]) / row["total"]
                if row["total"] else 0.0
            )
            row["evidence"] = sorted(row["evidence"])
            rows.append(row)
        rows.sort(key=lambda row: (-int(row["total"]), row["trigger"], row["trigger_type"]))
        return rows
