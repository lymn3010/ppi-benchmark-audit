"""Export long-form CSV ledgers for statistics and figures from ``events.db`` (read-only)."""
from __future__ import annotations

import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PAIR_FIELDS = [
    "corpus",
    "sentence_id",
    "pair_a",
    "pair_b",
    "gold_label",
    "predicted_label",
    "stage",
    "endpoint_visibility",
    "application_status",
    "reason_code",
    "proposal_count",
    "pattern_keys",
    "trigger_lemmas",
    "realization_channels",
    "assertion_statuses",
    "projection_modes",
    "proposal_refs",
    "event_refs",
    "projection_plan_refs",
]

PATTERN_FIELDS = [
    "corpus",
    "sentence_id",
    "candidate_id",
    "pair_a",
    "pair_b",
    "gold_label",
    "predicted_label",
    "application_status",
    "application_reason_code",
    "decision",
    "canonical_pattern_key",
    "semantic_pattern_key",
    "predicate_lemma",
    "shape",
    "construction",
    "case_marker",
    "realization_channel",
    "event_detail",
    "source_role",
    "target_role",
    "source_argument_role",
    "target_argument_role",
    "source_parser_dep",
    "target_parser_dep",
    "assertion_status",
    "projection_mode",
    "compound_propagated",
    "owner_propagated",
    "descriptor_propagated",
    "is_nominalized",
    "parser_source",
]

LEXICAL_OBSERVATION_FIELDS = [
    "corpus",
    "sentence_id",
    "term",
    "category",
    "role",
    "pattern_key",
    "event_detail",
    "is_evaluable",
    "gold_positive",
    "gold_label",
]

CARRIER_TRIGGER_FIELDS = [
    "corpus",
    "sentence_id",
    "carrier_word",
    "pattern_key",
    "role",
    "is_evaluable",
    "gold_positive",
    "gold_label",
]

EVENT_FIELDS = [
    "corpus",
    "sentence_id",
    "event_id",
    "event_type",
    "semantic_event_class",
    "event_detail",
    "realization_channel",
    "relation_type",
    "predicate_lemma",
    "pattern_key",
    "semantic_pattern_key",
    "yield_type",
    "construction",
    "is_inferred",
    "chain_readable",
    "pred_pair_count",
    "argument_roles",
]

SUMMARY_FIELDS = ["corpus", "metric", "value"]


def export_statistics_ledgers(
    db_path: Path | str,
    output_dir: Path | str,
    *,
    corpus: str = "",
) -> dict[str, Path]:
    """Write denormalized CSV ledgers (each with ``corpus``) under *output_dir*; return paths."""
    db_path = Path(db_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pair_observations": output_dir / "pair_observations.csv",
        "pattern_observations": output_dir / "pattern_observations.csv",
        "lexical_observations": output_dir / "lexical_observations.csv",
        "carrier_trigger_observations": output_dir / "carrier_trigger_observations.csv",
        "event_observations": output_dir / "event_observations.csv",
        "corpus_summary": output_dir / "corpus_summary.csv",
        "README": output_dir / "README.md",
    }
    if not db_path.exists():
        _write_csv(paths["pair_observations"], PAIR_FIELDS, [])
        _write_csv(paths["pattern_observations"], PATTERN_FIELDS, [])
        _write_csv(paths["lexical_observations"], LEXICAL_OBSERVATION_FIELDS, [])
        _write_csv(paths["carrier_trigger_observations"], CARRIER_TRIGGER_FIELDS, [])
        _write_csv(paths["event_observations"], EVENT_FIELDS, [])
        _write_csv(paths["corpus_summary"], SUMMARY_FIELDS, [])
        _write_readme(paths["README"], db_path)
        return paths

    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        metadata = _load_metadata(conn)
        corpus_name = corpus or metadata.get("corpus") or metadata.get("dataset") or ""
        candidate_by_sentence_pair = _load_candidate_index(conn, corpus_name)
        pair_rows = list(_pair_rows(conn, corpus_name, candidate_by_sentence_pair))
        pattern_rows = list(_pattern_rows(conn, corpus_name))
        lexical_rows = list(_lexical_rows(conn, corpus_name))
        carrier_rows = [
            {
                "corpus": row["corpus"],
                "sentence_id": row["sentence_id"],
                "carrier_word": row["term"],
                "pattern_key": row["pattern_key"],
                "role": row["role"],
                "is_evaluable": row["is_evaluable"],
                "gold_positive": row["gold_positive"],
                "gold_label": row["gold_label"],
            }
            for row in lexical_rows
            if row["category"] == "target_context"
        ]
        event_rows = list(_event_rows(conn, corpus_name))
        summary_rows = list(_summary_rows(
            conn,
            corpus_name,
            pair_rows=pair_rows,
            pattern_rows=pattern_rows,
            lexical_rows=lexical_rows,
            event_rows=event_rows,
        ))
    finally:
        conn.close()

    _write_csv(paths["pair_observations"], PAIR_FIELDS, pair_rows)
    _write_csv(paths["pattern_observations"], PATTERN_FIELDS, pattern_rows)
    _write_csv(paths["lexical_observations"], LEXICAL_OBSERVATION_FIELDS, lexical_rows)
    _write_csv(paths["carrier_trigger_observations"], CARRIER_TRIGGER_FIELDS, carrier_rows)
    _write_csv(paths["event_observations"], EVENT_FIELDS, event_rows)
    _write_csv(paths["corpus_summary"], SUMMARY_FIELDS, summary_rows)
    _write_readme(paths["README"], db_path)
    return paths


def _load_metadata(conn: sqlite3.Connection) -> dict[str, str]:
    try:
        return {
            str(row["key"]): str(row["value"])
            for row in conn.execute("SELECT key, value FROM run_metadata")
        }
    except sqlite3.Error:
        return {}


def _load_candidate_index(
    conn: sqlite3.Connection,
    corpus: str,
) -> dict[tuple[str, tuple[int, int]], list[dict[str, str]]]:
    index: dict[tuple[str, tuple[int, int]], list[dict[str, str]]] = defaultdict(list)
    if not _table_exists(conn, "candidate_relations"):
        return index
    columns = _columns(conn, "candidate_relations")
    if not {"sentence_id", "pair_a", "pair_b"}.issubset(columns):
        return index
    for row in conn.execute(
        f"""
        SELECT sentence_id, pair_a, pair_b,
               {_column_expr(columns, 'candidate_id')} ,
               {_column_expr(columns, 'predicate_lemma')} ,
               {_column_expr(columns, 'realization_channel')} ,
               {_column_expr(columns, 'assertion_status')} ,
               {_column_expr(columns, 'projection_mode')} ,
               {_column_expr(columns, 'canonical_pattern_key')} ,
               {_column_expr(columns, 'payload')}
        FROM candidate_relations
        WHERE pair_a IS NOT NULL AND pair_b IS NOT NULL
        """
    ):
        pair = _normalize_pair((row["pair_a"], row["pair_b"]))
        if pair is None:
            continue
        payload = _json_loads(row["payload"], {})
        index[(str(row["sentence_id"]), pair)].append({
            "corpus": corpus,
            "candidate_id": str(row["candidate_id"] or ""),
            "predicate_lemma": str(row["predicate_lemma"] or ""),
            "realization_channel": str(row["realization_channel"] or ""),
            "assertion_status": str(row["assertion_status"] or ""),
            "projection_mode": str(row["projection_mode"] or ""),
            "pattern_key": str(
                row["canonical_pattern_key"]
                or payload.get("pattern_key")
                or ""
            ),
        })
    return index


def _pair_rows(
    conn: sqlite3.Connection,
    corpus: str,
    candidate_index: dict[tuple[str, tuple[int, int]], list[dict[str, str]]],
) -> Iterable[dict[str, Any]]:
    if not _table_exists(conn, "sentences"):
        return
    columns = _columns(conn, "sentences")
    if "sentence_id" not in columns:
        return
    select_columns = [
        "sentence_id",
        _column_expr(columns, "pred_pairs", default="'[]'"),
        _column_expr(columns, "gold_pairs", default="NULL"),
        _column_expr(columns, "artifacts", default="'{}'"),
    ]
    for row in conn.execute(
        f"SELECT {', '.join(select_columns)} FROM sentences ORDER BY sentence_id"
    ):
        sentence_id = str(row["sentence_id"])
        predicted = {
            pair for pair in (_normalize_pair(p) for p in _json_loads(row["pred_pairs"], []))
            if pair is not None
        }
        gold_raw = _json_loads(row["gold_pairs"], None)
        gold = None if gold_raw is None else {
            pair for pair in (_normalize_pair(p) for p in gold_raw)
            if pair is not None
        }
        artifacts = _json_loads(row["artifacts"], {})
        traces = artifacts.get("pair_inference_traces") or []
        if not traces:
            universe = set(predicted) | set(candidate_pair for sid, candidate_pair in candidate_index if sid == sentence_id)
            if gold is not None:
                universe |= gold
            traces = [
                {
                    "pair": list(pair),
                    "stage": "emitted" if pair in predicted else "not_traced",
                    "endpoint_visibility": "",
                    "application_status": "applied" if pair in predicted else "",
                    "reason_code": "",
                    "proposal_refs": [],
                    "event_refs": [],
                    "projection_plan_refs": [],
                    "assertion_statuses": [],
                    "projection_modes": [],
                }
                for pair in sorted(universe)
            ]
        for trace in traces:
            pair = _normalize_pair(trace.get("pair"))
            if pair is None:
                continue
            candidates = candidate_index.get((sentence_id, pair), [])
            yield {
                "corpus": corpus,
                "sentence_id": sentence_id,
                "pair_a": pair[0],
                "pair_b": pair[1],
                "gold_label": _gold_label(pair, gold),
                "predicted_label": "positive" if pair in predicted else "negative",
                "stage": str(trace.get("stage") or ""),
                "endpoint_visibility": str(trace.get("endpoint_visibility") or ""),
                "application_status": str(trace.get("application_status") or ""),
                "reason_code": str(trace.get("reason_code") or ""),
                "proposal_count": len(candidates),
                "pattern_keys": _join_unique(c["pattern_key"] for c in candidates),
                "trigger_lemmas": _join_unique(c["predicate_lemma"] for c in candidates),
                "realization_channels": _join_unique(c["realization_channel"] for c in candidates),
                "assertion_statuses": _join_unique(
                    list(trace.get("assertion_statuses") or [])
                    + [c["assertion_status"] for c in candidates]
                ),
                "projection_modes": _join_unique(
                    list(trace.get("projection_modes") or [])
                    + [c["projection_mode"] for c in candidates]
                ),
                "proposal_refs": _join_unique(trace.get("proposal_refs") or []),
                "event_refs": _join_unique(trace.get("event_refs") or []),
                "projection_plan_refs": _join_unique(trace.get("projection_plan_refs") or []),
            }


def _pattern_rows(conn: sqlite3.Connection, corpus: str) -> Iterable[dict[str, Any]]:
    if not _table_exists(conn, "candidate_relations"):
        return
    columns = _columns(conn, "candidate_relations")
    if not {"sentence_id", "pair_a", "pair_b"}.issubset(columns):
        return
    sentence_columns = _columns(conn, "sentences")
    select_columns = [
        "cr.sentence_id",
        _column_expr(columns, "candidate_id", table_alias="cr"),
        "cr.pair_a",
        "cr.pair_b",
        _column_expr(columns, "payload", table_alias="cr"),
        _column_expr(columns, "decision", table_alias="cr"),
        _column_expr(columns, "canonical_pattern_key", table_alias="cr"),
        _column_expr(columns, "predicate_lemma", table_alias="cr"),
        _column_expr(columns, "shape", table_alias="cr"),
        _column_expr(columns, "construction", table_alias="cr"),
        _column_expr(columns, "case_marker", table_alias="cr"),
        _column_expr(columns, "realization_channel", table_alias="cr"),
        _column_expr(columns, "event_detail", table_alias="cr"),
        _column_expr(columns, "source_role", table_alias="cr"),
        _column_expr(columns, "target_role", table_alias="cr"),
        _column_expr(columns, "source_argument_role", table_alias="cr"),
        _column_expr(columns, "target_argument_role", table_alias="cr"),
        _column_expr(columns, "source_parser_dep", table_alias="cr"),
        _column_expr(columns, "target_parser_dep", table_alias="cr"),
        _column_expr(columns, "assertion_status", table_alias="cr"),
        _column_expr(columns, "projection_mode", table_alias="cr"),
        _column_expr(columns, "compound_propagated", default="0", table_alias="cr"),
        _column_expr(columns, "owner_propagated", default="0", table_alias="cr"),
        _column_expr(columns, "descriptor_propagated", default="0", table_alias="cr"),
        _column_expr(columns, "is_nominalized", default="0", table_alias="cr"),
        _column_expr(columns, "parser_source", table_alias="cr"),
        _column_expr(sentence_columns, "gold_pairs", default="NULL", table_alias="s"),
        _column_expr(sentence_columns, "pred_pairs", default="'[]'", table_alias="s"),
    ]
    order_by = "cr.sentence_id, cr.id" if "id" in columns else "cr.sentence_id"
    for row in conn.execute(
        f"""
        SELECT {', '.join(select_columns)}
        FROM candidate_relations cr
        LEFT JOIN sentences s ON s.sentence_id = cr.sentence_id
        ORDER BY {order_by}
        """
    ):
        pair = _normalize_pair((row["pair_a"], row["pair_b"]))
        gold = _gold_set(row["gold_pairs"])
        predicted = _gold_set(row["pred_pairs"]) or set()
        payload = _json_loads(row["payload"], {})
        yield {
            "corpus": corpus,
            "sentence_id": str(row["sentence_id"] or ""),
            "candidate_id": str(row["candidate_id"] or ""),
            "pair_a": "" if pair is None else pair[0],
            "pair_b": "" if pair is None else pair[1],
            "gold_label": "" if pair is None else _gold_label(pair, gold),
            "predicted_label": (
                "" if pair is None
                else ("positive" if pair in predicted else "negative")
            ),
            "application_status": str(payload.get("application_status") or ""),
            "application_reason_code": str(payload.get("application_reason_code") or ""),
            "decision": str(row["decision"] or payload.get("decision") or ""),
            "canonical_pattern_key": str(row["canonical_pattern_key"] or payload.get("pattern_key") or ""),
            "semantic_pattern_key": str(payload.get("semantic_pattern_key") or ""),
            "predicate_lemma": str(row["predicate_lemma"] or ""),
            "shape": str(row["shape"] or payload.get("shape") or ""),
            "construction": str(row["construction"] or payload.get("construction") or ""),
            "case_marker": str(row["case_marker"] or ""),
            "realization_channel": str(row["realization_channel"] or ""),
            "event_detail": str(row["event_detail"] or ""),
            "source_role": str(row["source_role"] or ""),
            "target_role": str(row["target_role"] or ""),
            "source_argument_role": str(row["source_argument_role"] or ""),
            "target_argument_role": str(row["target_argument_role"] or ""),
            "source_parser_dep": str(row["source_parser_dep"] or ""),
            "target_parser_dep": str(row["target_parser_dep"] or ""),
            "assertion_status": str(row["assertion_status"] or ""),
            "projection_mode": str(row["projection_mode"] or ""),
            "compound_propagated": int(row["compound_propagated"] or 0),
            "owner_propagated": int(row["owner_propagated"] or 0),
            "descriptor_propagated": int(row["descriptor_propagated"] or 0),
            "is_nominalized": int(row["is_nominalized"] or 0),
            "parser_source": str(row["parser_source"] or ""),
        }


def _lexical_rows(conn: sqlite3.Connection, corpus: str) -> Iterable[dict[str, Any]]:
    if not _table_exists(conn, "observation_attributions"):
        return
    columns = _columns(conn, "observation_attributions")
    if not {"sentence_id", "term", "category"}.issubset(columns):
        return
    role_expr = "COALESCE(role, '')" if "role" in columns else "''"
    pattern_expr = "COALESCE(pattern_key, '')" if "pattern_key" in columns else "''"
    detail_expr = "COALESCE(event_detail, '')" if "event_detail" in columns else "''"
    evaluable_expr = "is_evaluable" if "is_evaluable" in columns else "0"
    correct_expr = "is_correct" if "is_correct" in columns else "0"
    for row in conn.execute(
        f"""
        SELECT sentence_id, term, category, {role_expr} AS role,
               {pattern_expr} AS pattern_key,
               {detail_expr} AS event_detail,
               {evaluable_expr} AS is_evaluable,
               {correct_expr} AS is_correct
        FROM observation_attributions
        ORDER BY sentence_id, category, term
        """
    ):
        is_evaluable = int(row["is_evaluable"] or 0)
        gold_positive = int(row["is_correct"] or 0)
        yield {
            "corpus": corpus,
            "sentence_id": str(row["sentence_id"] or ""),
            "term": str(row["term"] or ""),
            "category": str(row["category"] or ""),
            "role": str(row["role"] or ""),
            "pattern_key": str(row["pattern_key"] or ""),
            "event_detail": str(row["event_detail"] or ""),
            "is_evaluable": is_evaluable,
            "gold_positive": gold_positive,
            "gold_label": (
                "unlabeled" if not is_evaluable
                else ("positive" if gold_positive else "negative")
            ),
        }


def _event_rows(conn: sqlite3.Connection, corpus: str) -> Iterable[dict[str, Any]]:
    if not _table_exists(conn, "events"):
        return
    columns = _columns(conn, "events")
    if not {"sentence_id", "event_id"}.issubset(columns):
        return
    select_columns = [
        "sentence_id",
        "event_id",
        _column_expr(columns, "event_type"),
        _column_expr(columns, "semantic_event_class"),
        _column_expr(columns, "event_detail"),
        _column_expr(columns, "realization_channel"),
        _column_expr(columns, "relation_type"),
        _column_expr(columns, "predicate_lemma"),
        _column_expr(columns, "pattern_key"),
        _column_expr(columns, "semantic_pattern_key"),
        _column_expr(columns, "yield_type"),
        _column_expr(columns, "construction"),
        _column_expr(columns, "is_inferred", default="0"),
        _column_expr(columns, "chain_readable"),
        _column_expr(columns, "pred_pairs", default="'[]'"),
        _column_expr(columns, "arguments_json", default="'{}'"),
    ]
    for row in conn.execute(
        f"""
        SELECT {', '.join(select_columns)}
        FROM events
        ORDER BY sentence_id, event_id
        """
    ):
        arguments = _json_loads(row["arguments_json"], {})
        yield {
            "corpus": corpus,
            "sentence_id": str(row["sentence_id"] or ""),
            "event_id": int(row["event_id"]),
            "event_type": str(row["event_type"] or ""),
            "semantic_event_class": str(row["semantic_event_class"] or ""),
            "event_detail": str(row["event_detail"] or ""),
            "realization_channel": str(row["realization_channel"] or ""),
            "relation_type": str(row["relation_type"] or ""),
            "predicate_lemma": str(row["predicate_lemma"] or ""),
            "pattern_key": str(row["pattern_key"] or ""),
            "semantic_pattern_key": str(row["semantic_pattern_key"] or ""),
            "yield_type": str(row["yield_type"] or ""),
            "construction": str(row["construction"] or ""),
            "is_inferred": int(row["is_inferred"] or 0),
            "chain_readable": str(row["chain_readable"] or ""),
            "pred_pair_count": len(_json_loads(row["pred_pairs"], [])),
            "argument_roles": _join_unique(arguments.keys() if isinstance(arguments, dict) else []),
        }


def _summary_rows(
    conn: sqlite3.Connection,
    corpus: str,
    *,
    pair_rows: list[dict[str, Any]],
    pattern_rows: list[dict[str, Any]],
    lexical_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    yield {"corpus": corpus, "metric": "sentences", "value": _count_table(conn, "sentences")}
    yield {"corpus": corpus, "metric": "events", "value": len(event_rows)}
    yield {"corpus": corpus, "metric": "pair_observations", "value": len(pair_rows)}
    yield {"corpus": corpus, "metric": "pattern_observations", "value": len(pattern_rows)}
    yield {"corpus": corpus, "metric": "lexical_observations", "value": len(lexical_rows)}
    yield {
        "corpus": corpus,
        "metric": "carrier_trigger_observations",
        "value": sum(1 for row in lexical_rows if row["category"] == "target_context"),
    }


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _write_readme(path: Path, db_path: Path) -> None:
    path.write_text(
        "\n".join([
            "# Statistics Ledgers",
            "",
            "These CSV files are the statistics-ready counterpart to the human",
            "audit workbook. Use them for Fisher exact tests, Wilson intervals,",
            "Benjamini-Hochberg correction, bootstrap analyses, and regression",
            "models. Do not scrape Excel/YAML for statistical tests.",
            "",
            f"Source database: `{db_path}`",
            "",
            "Files:",
            "",
            "- `pair_observations.csv`: one bounded protein-pair observation per row.",
            "- `pattern_observations.csv`: one candidate relation / pattern firing per row.",
            "- `lexical_observations.csv`: trigger and carrier lexical observation rows.",
            "- `carrier_trigger_observations.csv`: carrier-word x trigger-pattern factors.",
            "- `event_observations.csv`: one extracted semantic event per row.",
            "- `corpus_summary.csv`: run-level counts for integrity checks.",
            "",
            "Corpora should be analyzed separately unless the statistical model",
            "explicitly includes corpus as a factor.",
            "",
        ]),
        encoding="utf-8",
    )


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def _column_expr(
    columns: set[str],
    name: str,
    *,
    default: str = "''",
    table_alias: str = "",
) -> str:
    prefix = f"{table_alias}." if table_alias else ""
    if name in columns:
        return f"{prefix}{name} AS {name}"
    return f"{default} AS {name}"


def _count_table(conn: sqlite3.Connection, table: str) -> int:
    if not _table_exists(conn, table):
        return 0
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _json_loads(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _normalize_pair(value: Any) -> tuple[int, int] | None:
    if value is None or len(value) != 2:
        return None
    a, b = int(value[0]), int(value[1])
    if a == b:
        return None
    return tuple(sorted((a, b)))


def _gold_set(value: Any) -> set[tuple[int, int]] | None:
    raw = _json_loads(value, None)
    if raw is None:
        return None
    return {
        pair for pair in (_normalize_pair(item) for item in raw)
        if pair is not None
    }


def _gold_label(pair: tuple[int, int], gold: set[tuple[int, int]] | None) -> str:
    if gold is None:
        return "unlabeled"
    return "positive" if pair in gold else "negative"


def _join_unique(values: Iterable[Any]) -> str:
    return "|".join(
        sorted({
            str(value)
            for value in values
            if value is not None and str(value) != ""
        })
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value
