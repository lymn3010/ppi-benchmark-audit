"""Audit data access, including legacy database and bundle fallbacks."""
from __future__ import annotations
import json
import logging
from collections import defaultdict
from pathlib import Path
from src.analysis.db_queries import open_db
log = logging.getLogger(__name__)
from .display import (
    _relational_nominal_from_arguments,
    _chain_from_pattern_key,
    _contextual_chain,
    _yaml_chain_for_relational_statement,
)
from .statistics import (
    _payload_protein_indices,
    _canonical_pair_set_from_json,
    _table_columns,
    _optional_sql_column,
)

class AuditQueries:
    def __init__(self, bundle=None):
        self.bundle = bundle

    def _load_pattern_rows(
        self, db_path: Path, min_count: int = 1
    ) -> list[dict]:
        """Pattern reliability rows sorted by precision desc."""
        if db_path.exists():
            try:
                from src.analysis.db_queries import pattern_reliability
                return pattern_reliability(db_path, min_count=min_count)
            except Exception as exc:
                log.debug("db pattern_reliability failed (%s); using bundle", exc)

        # Fallback: synthesize from pattern observations in the in-memory bundle.
        rows: list[dict] = []
        if not self.bundle:
            return rows
        ps = getattr(self.bundle, "_pattern_observations", None)
        if ps is None:
            return rows
        for lemma, entries in getattr(ps, "_by_lemma", {}).items():
            for e in entries:
                if e.total < min_count:
                    continue
                prec = e.pair_precision
                rows.append({
                    "pattern_key":  e.canonical_pattern_key(),
                    "lemma":        lemma,
                    "construction": getattr(e, "construction", ""),
                    "yield_type":   getattr(e, "yield_type", ""),
                    "shape":        "",
                    "tp":           e.correct,
                    "fp":           e.total - e.correct,
                    "total":        e.total,
                    "precision":    prec,
                    "ci_low":       0.0,
                    "ci_high":      1.0,
                })
        rows.sort(key=lambda r: float(r["precision"]), reverse=True)
        return rows


    def _load_semantic_pattern_rows(
        self, db_path: Path, min_count: int = 1
    ) -> list[dict]:
        """Abstract pattern-family rows sorted by agreement then support."""
        if db_path.exists():
            try:
                from src.analysis.db_queries import semantic_pattern_reliability
                return semantic_pattern_reliability(db_path, min_count=min_count)
            except Exception as exc:
                log.debug("db semantic_pattern_reliability failed (%s); grouping pattern rows", exc)

        grouped: dict[str, dict] = {}
        for row in self._load_pattern_rows(db_path, min_count=min_count):
            sem_key = row.get("semantic_pattern_key") or f"structural|{row.get('lemma', '')}"
            g = grouped.setdefault(sem_key, {
                "semantic_pattern_key": sem_key,
                "lemma": row.get("lemma", ""),
                "preposition": "_",
                "yield_type": "",
                "shape": "",
                "evidence_class": "",
                "tp": 0,
                "fp": 0,
                "total": 0,
                "constructions": set(),
            })
            g["tp"] += int(row.get("tp") or 0)
            g["fp"] += int(row.get("fp") or 0)
            g["total"] += int(row.get("total") or 0)
            if row.get("construction"):
                g["constructions"].add(str(row.get("construction")))
        out: list[dict] = []
        for g in grouped.values():
            total = int(g["total"] or 0)
            if total < min_count:
                continue
            precision = (int(g["tp"] or 0) / total) if total else 0.0
            out.append({
                **g,
                "precision": precision,
                "ci_low": 0.0,
                "ci_high": 1.0,
                "constructions": ",".join(sorted(g["constructions"])),
            })
        out.sort(key=lambda r: (float(r["precision"]), int(r["total"])), reverse=True)
        return out


    def _load_pattern_variants_by_semantic_key(
        self, db_path: Path, min_count: int = 1
    ) -> dict[str, list[dict]]:
        variants: dict[str, list[dict]] = defaultdict(list)
        rows = self._load_pattern_rows(db_path, min_count=min_count)
        for row in rows:
            sem_key = row.get("semantic_pattern_key") or f"structural|{row.get('lemma', '')}"
            variants[sem_key].append(row)
        for sem_key in variants:
            variants[sem_key].sort(
                key=lambda r: (-int(r.get("total") or 0), str(r.get("pattern_key") or ""))
            )
        return variants


    def _load_candidate_pattern_rows(self, db_path: Path) -> list[dict]:
        """Pre-inference proposal statistics from ``candidate_pattern_stats``."""
        if not db_path.exists():
            return []
        try:
            with open_db(db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT pattern_key, lemma, preposition, shape, evidence_class,
                           construction, review_status, source, observations,
                           sentence_pair_observations, gold_positive_hits,
                           gold_negative_hits, annotation_agreement
                    FROM candidate_pattern_stats
                    ORDER BY annotation_agreement DESC,
                             sentence_pair_observations DESC,
                             pattern_key
                    """
                ).fetchall()
        except Exception as exc:
            log.debug("db candidate_pattern_stats load failed (%s)", exc)
            return []

        out: list[dict] = []
        for raw in rows:
            row = dict(raw)
            evidence_class = str(row.get("evidence_class") or "structural")
            shape = str(row.get("shape") or "AB").upper()
            lemma = str(row.get("lemma") or "")
            prep = str(row.get("preposition") or "_")
            if shape in {"AB", "AB_NMOD", "AA"}:
                semantic_key = f"{evidence_class}|{lemma}"
            else:
                semantic_key = f"{evidence_class}|{shape}|{lemma}|{prep}"
            row["semantic_pattern_key"] = semantic_key
            out.append(row)
        return out


    def _load_relational_statement_rows(
        self, db_path: Path, min_count: int = 1
    ) -> list[dict]:
        """Review-only relational statements from ``candidate_relations``.

        Agreement means the pair is also gold-positive, not that other rows are errors.
        """
        if not db_path.exists():
            return []

        nominal_lookup = self._load_relational_nominal_lookup(db_path)
        try:
            with open_db(db_path) as conn:
                columns = _table_columns(conn, "candidate_relations")
                source_event_expr = _optional_sql_column(
                    columns, "source_event_id", table_alias="cr",
                )
                predicate_lemma_expr = _optional_sql_column(
                    columns, "predicate_lemma", table_alias="cr",
                )
                rows = conn.execute(
                    f"""
                    SELECT
                        cr.sentence_id,
                        {source_event_expr},
                        {predicate_lemma_expr},
                        cr.schema,
                        cr.event_detail,
                        cr.assertion_status,
                        cr.canonical_pattern_key,
                        cr.pair_a,
                        cr.pair_b,
                        s.gold_pairs,
                        s.text
                    FROM candidate_relations cr
                    JOIN sentences s ON s.sentence_id = cr.sentence_id
                    WHERE lower(COALESCE(cr.projection_mode, '')) = 'relational'
                       OR upper(COALESCE(cr.shape, '')) = 'RELATIONAL'
                    ORDER BY cr.schema, cr.event_detail, cr.sentence_id, cr.pair_a, cr.pair_b
                    """
                ).fetchall()
        except Exception as exc:
            log.debug("db relational statement load failed (%s)", exc)
            return []

        grouped: dict[tuple[str, str], dict] = {}
        seen_pairs: set[tuple[str, str, str, int, int]] = set()
        for row in rows:
            schema = str(row["schema"] or "RELATIONAL").upper()
            detail = str(row["event_detail"] or "unknown")
            predicate_trigger = str(row["predicate_lemma"] or "").strip().lower()
            if predicate_trigger in {"", schema.lower(), "is-a", "has", "part-of", "identity"}:
                predicate_trigger = ""
            nominal_trigger = predicate_trigger or nominal_lookup.get(
                (str(row["sentence_id"]), str(row["source_event_id"] or "")),
                "",
            )
            if not nominal_trigger:
                nominal_trigger = nominal_lookup.get(
                    (str(row["sentence_id"]), detail),
                    "",
                )
            pair_a = row["pair_a"]
            pair_b = row["pair_b"]
            if pair_a is None or pair_b is None:
                continue
            pair = tuple(sorted((int(pair_a), int(pair_b))))
            dedupe_key = (str(row["sentence_id"]), schema, detail, pair[0], pair[1])
            if dedupe_key in seen_pairs:
                continue
            seen_pairs.add(dedupe_key)

            gold_set = _canonical_pair_set_from_json(row["gold_pairs"])
            is_gold_positive = pair in gold_set

            group = grouped.setdefault((schema, detail, nominal_trigger), {
                "schema": schema,
                "trigger": nominal_trigger or schema,
                "trigger_type": "relational_nominal" if nominal_trigger else "relation_schema",
                "event_detail": detail,
                "chain": _yaml_chain_for_relational_statement(schema, nominal_trigger),
                "runtime_feedable": False,
                "review_action": "review_only_evidence",
                "review_hint": (
                    "Relational evidence is visible for alias/context review; "
                    "it is not a direct PPI pattern."
                ),
                "statistics": {
                    "total": 0,
                    "gold_positive": 0,
                    "gold_negative": 0,
                    "agreement_with_gold": 0.0,
                },
                "assertion_status": {},
                "source": {},
                "examples": [],
            })
            stats = group["statistics"]
            stats["total"] += 1
            if is_gold_positive:
                stats["gold_positive"] += 1
            else:
                stats["gold_negative"] += 1

            assertion = str(row["assertion_status"] or "unknown").lower()
            group["assertion_status"][assertion] = (
                int(group["assertion_status"].get(assertion, 0)) + 1
            )
            key = str(row["canonical_pattern_key"] or "")
            if key == "relational|RELATIONAL||_|verbal":
                key = f"relational|RELATIONAL|{schema.lower()}|_|verbal"
            if key:
                group["source"][key] = int(group["source"].get(key, 0)) + 1

            if len(group["examples"]) < 3:
                group["examples"].append({
                    "sentence_id": row["sentence_id"],
                    "pair": list(pair),
                    "gold_positive": bool(is_gold_positive),
                    "text": str(row["text"] or "")[:500],
                })

        out: list[dict] = []
        for group in grouped.values():
            total = int(group["statistics"]["total"])
            if total < min_count:
                continue
            group["statistics"]["agreement_with_gold"] = round(
                float(group["statistics"]["gold_positive"]) / total if total else 0.0,
                4,
            )
            group["assertion_status"] = dict(sorted(group["assertion_status"].items()))
            group["source"] = dict(sorted(group["source"].items()))
            out.append(group)

        out.sort(
            key=lambda row: (
                -int((row.get("statistics") or {}).get("total") or 0),
                str(row.get("schema") or ""),
                str(row.get("event_detail") or ""),
            )
        )
        return out


    def _load_events_without_final_pair_rows(self, db_path: Path) -> list[dict]:
        """Protein-bearing predicate events that produced no pair."""
        if not db_path.exists():
            return []
        try:
            with open_db(db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT e.sentence_id, e.event_detail, e.realization_channel,
                           e.predicate_lemma, e.pattern_key, e.yield_type,
                           e.construction, e.chain_readable, e.arguments_json,
                           s.text
                    FROM events e
                    JOIN sentences s ON s.sentence_id = e.sentence_id
                    WHERE e.semantic_event_class = 'predicate_event'
                      AND COALESCE(e.pred_pairs, '[]') = '[]'
                    ORDER BY e.predicate_lemma, e.event_detail, e.sentence_id
                    """
                ).fetchall()
        except Exception as exc:
            log.debug("db event-without-final-pair load failed (%s)", exc)
            return []

        grouped: dict[tuple[str, str, str, str], dict] = {}
        for row in rows:
            try:
                arguments = json.loads(row["arguments_json"] or "{}")
            except Exception:
                arguments = {}
            proteins = sorted(_payload_protein_indices(arguments))
            if not proteins:
                continue
            trigger = str(row["predicate_lemma"] or "(no lexical trigger)")
            detail = str(row["event_detail"] or "unknown")
            channel = str(row["realization_channel"] or "unknown")
            pattern_key = str(row["pattern_key"] or "")
            key = (trigger, pattern_key, detail, channel)
            group = grouped.setdefault(key, {
                "trigger": trigger,
                "pattern_key": pattern_key,
                "event_detail": detail,
                "realization_channel": channel,
                "yield_type": str(row["yield_type"] or ""),
                "construction": str(row["construction"] or ""),
                "chain_readable": str(row["chain_readable"] or ""),
                "event_occurrences": 0,
                "examples": [],
            })
            group["event_occurrences"] += 1
            if len(group["examples"]) < 3:
                group["examples"].append({
                    "sentence_id": str(row["sentence_id"] or ""),
                    "pair": [],
                    "status": "event_without_final_pair",
                    "event_detail": detail,
                    "protein_indices": proteins,
                    "text": str(row["text"] or "")[:800],
                })
        return sorted(
            grouped.values(),
            key=lambda row: (
                -int(row.get("event_occurrences") or 0),
                str(row.get("trigger") or ""),
                str(row.get("event_detail") or ""),
            ),
        )


    def _load_relational_nominal_lookup(self, db_path: Path) -> dict[tuple[str, str], str]:
        """Map relation statement events to their lexical nominal head if available."""
        lookup: dict[tuple[str, str], str] = {}
        if not db_path.exists():
            return lookup
        self._load_relational_nominal_lookup_from_raw(db_path, lookup)
        try:
            with open_db(db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT sentence_id, event_id, event_detail, arguments_json
                    FROM events
                    WHERE semantic_event_class = 'relation_statement'
                       OR event_type = 'RelationalStatement'
                       OR event_type = 'RELATION_STATEMENT'
                    ORDER BY sentence_id, event_id
                    """
                ).fetchall()
        except Exception as exc:
            log.debug("db relational nominal lookup failed (%s)", exc)
            return lookup
        by_detail_count: dict[tuple[str, str], int] = defaultdict(int)
        for row in rows:
            try:
                arguments = json.loads(row["arguments_json"] or "{}") or {}
            except Exception:
                arguments = {}
            trigger = _relational_nominal_from_arguments(arguments)
            if not trigger:
                continue
            sentence_id = str(row["sentence_id"] or "")
            detail = str(row["event_detail"] or "")
            event_id = str(row["event_id"] or "")
            lookup[(sentence_id, event_id)] = trigger
            # Old databases lack event ids; use detail only when it is unambiguous.
            detail_key = (sentence_id, detail)
            by_detail_count[detail_key] += 1
            existing = lookup.get(detail_key)
            if existing is None:
                lookup[detail_key] = trigger
            elif existing != trigger:
                lookup[detail_key] = ""
        return lookup


    def _load_relational_nominal_lookup_from_raw(
        self, db_path: Path, lookup: dict[tuple[str, str], str],
    ) -> None:
        """Map raw parser event ids from candidate_events.jsonl to relation anchors."""
        raw_path = db_path.parent.parent / "raw" / "candidate_events.jsonl"
        if not raw_path.exists():
            return
        by_detail_count: dict[tuple[str, str], int] = defaultdict(int)
        try:
            with raw_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    candidate = obj.get("candidate") if isinstance(obj, dict) else None
                    if not isinstance(candidate, dict):
                        candidate = obj if isinstance(obj, dict) else {}
                    if not (
                        candidate.get("semantic_event_class") == "relation_statement"
                        or str(candidate.get("event_type") or "").upper() == "RELATION_STATEMENT"
                    ):
                        continue
                    arguments = candidate.get("arguments")
                    if not isinstance(arguments, dict):
                        arguments = {}
                    trigger = _relational_nominal_from_arguments(arguments)
                    if not trigger:
                        continue
                    sentence_id = str(candidate.get("sentence_id") or obj.get("sentence_id") or "")
                    event_id = str(candidate.get("event_id") or "")
                    detail = str(candidate.get("extraction_detail") or "")
                    if sentence_id and event_id:
                        lookup[(sentence_id, event_id)] = trigger
                    if sentence_id and detail:
                        detail_key = (sentence_id, detail)
                        by_detail_count[detail_key] += 1
                        existing = lookup.get(detail_key)
                        if existing is None:
                            lookup[detail_key] = trigger
                        elif existing != trigger:
                            lookup[detail_key] = ""
        except Exception as exc:
            log.debug("raw relational nominal lookup failed (%s)", exc)


    def _load_lexpath_candidate_rows(self, analysis_dir: Path) -> list[dict]:
        """Lexicalized path rows from the supporting audit artifact."""
        path = analysis_dir / "lexicalized_path_audit.yaml"
        if not path.exists():
            return []
        try:
            import yaml
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            log.debug("lexpath audit load failed (%s)", exc)
            return []
        rows = list(payload.get("families") or [])
        rows.sort(
            key=lambda r: (
                -int((r.get("statistics") or {}).get("sentence_pair_observations")
                     or (r.get("statistics") or {}).get("observations") or 0),
                str(r.get("pattern_signature") or ""),
            )
        )
        return rows


    def _load_pattern_examples_by_key(
        self, db_path: Path, *, limit_per_key: int = 3,
    ) -> dict[str, list[dict]]:
        """Load a few sentence examples for each exact structural pattern."""
        if not db_path.exists():
            return {}
        try:
            with open_db(db_path) as conn:
                columns = _table_columns(conn, "candidate_relations")
                rows = conn.execute(
                    f"""
                    SELECT cr.canonical_pattern_key, cr.sentence_id, cr.pair_a,
                           cr.pair_b, cr.decision, cr.event_detail,
                           {_optional_sql_column(columns, "realization_channel", table_alias="cr")},
                           {_optional_sql_column(columns, "case_marker", table_alias="cr")},
                           {_optional_sql_column(columns, "assertion_status", table_alias="cr", fallback="'asserted'")},
                           {_optional_sql_column(columns, "ambiguity_reason", table_alias="cr")},
                           {_optional_sql_column(columns, "compound_propagated", table_alias="cr", fallback="0")},
                           {_optional_sql_column(columns, "owner_propagated", table_alias="cr", fallback="0")},
                           {_optional_sql_column(columns, "descriptor_propagated", table_alias="cr", fallback="0")},
                           {_optional_sql_column(columns, "source_argument_role", table_alias="cr")},
                           {_optional_sql_column(columns, "target_argument_role", table_alias="cr")},
                           {_optional_sql_column(columns, "source_parser_dep", table_alias="cr")},
                           {_optional_sql_column(columns, "target_parser_dep", table_alias="cr")},
                           s.gold_pairs, s.text
                    FROM candidate_relations cr
                    JOIN sentences s ON s.sentence_id = cr.sentence_id
                    WHERE COALESCE(cr.canonical_pattern_key, '') <> ''
                    ORDER BY cr.canonical_pattern_key, cr.sentence_id,
                             cr.pair_a, cr.pair_b
                    """
                ).fetchall()
        except Exception as exc:
            log.debug("db pattern example load failed (%s)", exc)
            return {}

        out: dict[str, list[dict]] = defaultdict(list)
        seen: set[tuple[str, str, int, int]] = set()
        for row in rows:
            key = str(row["canonical_pattern_key"] or "")
            if len(out[key]) >= limit_per_key:
                continue
            if row["pair_a"] is None or row["pair_b"] is None:
                continue
            pair = tuple(sorted((int(row["pair_a"]), int(row["pair_b"]))))
            dedupe = (key, str(row["sentence_id"]), pair[0], pair[1])
            if dedupe in seen:
                continue
            seen.add(dedupe)
            gold_set = _canonical_pair_set_from_json(row["gold_pairs"])
            out[key].append({
                "sentence_id": str(row["sentence_id"] or ""),
                "pair": list(pair),
                "status": "gold_positive" if pair in gold_set else "gold_negative",
                "decision": str(row["decision"] or ""),
                "event_detail": str(row["event_detail"] or ""),
                "realization_channel": str(row["realization_channel"] or ""),
                "case_marker": str(row["case_marker"] or ""),
                "assertion_status": str(row["assertion_status"] or "asserted"),
                "ambiguity_reason": str(row["ambiguity_reason"] or ""),
                "propagation": [
                    name
                    for name, present in (
                        ("compound", row["compound_propagated"]),
                        ("owner", row["owner_propagated"]),
                        ("descriptor", row["descriptor_propagated"]),
                    )
                    if present
                ],
                "argument_roles": [
                    str(row["source_argument_role"] or ""),
                    str(row["target_argument_role"] or ""),
                ],
                "parser_deps": [
                    str(row["source_parser_dep"] or ""),
                    str(row["target_parser_dep"] or ""),
                ],
                "text": str(row["text"] or "")[:800],
            })
        return dict(out)


    @staticmethod
    def _load_context_trigger_matrix(
        db_path: Path,
        *,
        max_patterns_per_context: int = 12,
    ) -> dict[str, list[dict]]:
        """Carrier-word x trigger-pattern agreement rows (audit only)."""
        if not db_path.exists():
            return {}
        role_expr = "''"
        try:
            with open_db(db_path) as conn:
                columns = _table_columns(conn, "observation_attributions")
            if "role" in columns:
                role_expr = "COALESCE(role, '')"
        except Exception:
            role_expr = "''"
        sql = f"""
            SELECT
                term,
                COALESCE(pattern_key, '') AS pattern_key,
                {role_expr} AS role,
                COUNT(*) AS occurrences,
                SUM(CASE WHEN is_evaluable THEN 1 ELSE 0 END) AS evaluable,
                SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) AS gold_positive
            FROM observation_attributions
            WHERE category = 'target_context'
            GROUP BY term, pattern_key, role
            HAVING occurrences > 0
            ORDER BY term ASC, evaluable DESC, occurrences DESC, pattern_key ASC, role ASC
        """
        out: dict[str, list[dict]] = defaultdict(list)
        try:
            with open_db(db_path) as conn:
                for row in conn.execute(sql):
                    term = str(row["term"] or "")
                    if not term:
                        continue
                    if len(out[term]) >= max_patterns_per_context:
                        continue
                    evaluable = int(row["evaluable"] or 0)
                    positive = int(row["gold_positive"] or 0)
                    chain = _chain_from_pattern_key(str(row["pattern_key"] or ""))
                    role = str(row["role"] or "")
                    out[term].append({
                        "pattern_key": str(row["pattern_key"] or ""),
                        "role": role,
                        "chain": chain,
                        "contextual_chain": _contextual_chain(chain, term, role),
                        "occurrences": int(row["occurrences"] or 0),
                        "evaluable": evaluable,
                        "gold_positive": positive,
                        "gold_negative": max(0, evaluable - positive),
                        "agreement": round(float(positive) / evaluable, 4) if evaluable else None,
                    })
        except Exception as exc:
            log.debug("context trigger matrix load failed (%s)", exc)
            return {}
        return dict(out)


    @staticmethod
    def _load_semantic_evidence_summary(db_path: Path) -> dict[str, int]:
        """Count non-pattern audit artifacts so active concepts stay visible."""
        records_path = db_path.parent.parent / "raw" / "records.jsonl"
        counts = {
            "candidate_events": 0,
            "candidate_relations": 0,
            "identity_equivalences": 0,
            "self_alias_candidate_pairs": 0,
            "suppressed_pairs": 0,
            "path_candidates": 0,
            "projection_plans": 0,
            "pair_inference_traces": 0,
        }
        if not records_path.exists():
            return counts
        try:
            with records_path.open(encoding="utf-8") as handle:
                for line in handle:
                    artifacts = (json.loads(line).get("artifacts") or {})
                    for name in counts:
                        counts[name] += len(artifacts.get(name) or [])
        except Exception as exc:
            log.debug("semantic evidence summary load failed (%s)", exc)
        return counts


    def _load_observation_rows(
        self, db_path: Path, min_count: int = 1
    ) -> list[dict]:
        """Lexical observation rows sorted by category then precision desc."""
        if db_path.exists():
            try:
                from src.analysis.db_queries import observation_reliability
                return observation_reliability(db_path, min_count=min_count)
            except Exception as exc:
                log.debug("db observation_reliability failed (%s); using bundle", exc)

        # Fallback from bundle.observations
        rows: list[dict] = []
        if not self.bundle:
            return rows
        for category, kw_dict in self.bundle.observations.items():
            for word, stats in kw_dict.items():
                if stats.count < min_count:
                    continue
                rows.append({
                    "term":      word,
                    "category":  category,
                    "tp":        stats.correct,
                    "fp":        stats.fp_count,
                    "fn":        stats.fn_count,
                    "total":     stats.count,
                    "precision": stats.precision,
                    "ci_low":    0.0,
                    "ci_high":   1.0,
                    "count_all": stats.count_all,
                    "pair_contributing_occurrences": stats.pair_contributing_occurrences,
                    "nonprojecting_occurrences": stats.nonprojecting_occurrences,
                    "owner_occurrences": stats.owner_occurrences,
                    "descriptor_occurrences": stats.descriptor_occurrences,
                    "nested_occurrences": stats.nested_occurrences,
                })
        rows.sort(key=lambda r: (r["category"], -float(r["precision"])))
        return rows
