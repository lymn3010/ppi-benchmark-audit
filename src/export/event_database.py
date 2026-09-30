"""SQLite backend for run results (events, stats, candidate relations, views)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.export.event_report import EventReportBuilder


class EventDatabase:
    """SQLite event database for one run; use as a context manager or call ``close()``."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS run_metadata (
        key   TEXT PRIMARY KEY,
        value TEXT
    );

    CREATE TABLE IF NOT EXISTS sentences (
        sentence_id    TEXT    PRIMARY KEY,
        text           TEXT    NOT NULL,
        orig_text      TEXT,
        proteins       TEXT,       -- JSON array of protein names
        pred_pairs     TEXT,       -- JSON array of [i, j] pairs
        gold_pairs     TEXT,       -- JSON array or NULL
        label          TEXT,       -- correct/too_many/too_few/wrong_pairs/no_gold/no_event
        record_version TEXT,
        provenance     TEXT,
        artifacts      TEXT,
        run_meta       TEXT
    );

    CREATE TABLE IF NOT EXISTS events (
        event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        sentence_id     TEXT    NOT NULL REFERENCES sentences(sentence_id),
        event_type      TEXT,   -- InteractionalEvent / UndirectedInteraction / RelationalStatement
        semantic_event_class TEXT, -- predicate_event / relation_statement
        event_detail    TEXT,   -- S-V-O / AppositionGroup / ...
        realization_channel TEXT, -- clause / event_nominal / state_nominal / appositive / lexical_path
        relation_type   TEXT,   -- IS-A / HAS / empty
        span            TEXT,
        span_min        INTEGER,
        span_max        INTEGER,
        pred_pairs      TEXT,   -- JSON array of [i, j] pairs
        predicate_text  TEXT,
        predicate_lemma TEXT,
        pattern_key          TEXT,   -- 5-segment surface key: evidence_class|shape|lemma|prep|construction
        semantic_pattern_key TEXT,   -- direction-agnostic key: evidence_class|lemma (symmetric interactions); evidence_class|shape|lemma|prep (NESTED/RELATIONAL)
        yield_type      TEXT,   -- AB / AA / RELATIONAL / GROUP_ACT
        construction    TEXT,   -- verbal / passive / nominalized / compound_state
        is_inferred     INTEGER DEFAULT 0,
        chain_readable  TEXT,   -- human-readable DSL notation (e.g. "[ent_a] -> bind -> 'to' [ent_b]")
        chain_steps_json TEXT,  -- JSON array of {role, label, preposition} legacy chain steps when present
        arguments_json   TEXT   -- JSON dict of {role: [ArgumentFrame.to_json()]} for all arguments
    );

    CREATE TABLE IF NOT EXISTS observation_attributions (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        sentence_id  TEXT    NOT NULL REFERENCES sentences(sentence_id),
        event_detail TEXT,
        pattern_key  TEXT,
        term         TEXT    NOT NULL,
        category     TEXT    NOT NULL,  -- trigger_verbal / trigger_nominal / ...
        role         TEXT,              -- predicate / entities_b / ...
        is_correct   INTEGER NOT NULL,  -- 1 = TP event, 0 = FP or non-evaluable
        is_evaluable INTEGER NOT NULL   -- 1 = has pairs + gold, 0 = cannot judge
    );

    CREATE TABLE IF NOT EXISTS candidate_relations (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        sentence_id       TEXT    NOT NULL REFERENCES sentences(sentence_id),
        candidate_id      TEXT    NOT NULL,
        pair_a            INTEGER,
        pair_b            INTEGER,
        source_event_id   TEXT,
        parser_source     TEXT,
        schema            TEXT,
        predicate_lemma   TEXT,
        event_type        TEXT,
        event_detail      TEXT,
        realization_channel TEXT,
        source_role       TEXT,
        target_role       TEXT,
        predicate_parser_dep TEXT,
        source_argument_role TEXT,
        target_argument_role TEXT,
        source_parser_dep  TEXT,
        target_parser_dep  TEXT,
        case_marker       TEXT,
        assertion_status  TEXT,
        projection_mode   TEXT,
        pattern_signature TEXT,
        pattern_decision  TEXT,
        decision          TEXT,
        ambiguity_reason  TEXT,
        compound_propagated INTEGER DEFAULT 0,
        owner_propagated    INTEGER DEFAULT 0,
        descriptor_propagated INTEGER DEFAULT 0,
        is_nominalized    INTEGER DEFAULT 0,
        canonical_pattern_key TEXT,
        shape             TEXT,
        construction      TEXT,
        payload           TEXT
    );

    -- Inert candidate-pattern aggregate built from candidate_relations.
    -- Columns:
    --   annotation_agreement = gold_positive / observations
    --     (same-dataset signal, not final precision)
    --   review_status defaults to 'candidate'
    -- Review-only contract: these rows must never populate runtime
    -- Pattern-observation identity fields.
    CREATE TABLE IF NOT EXISTS candidate_pattern_stats (
        pattern_key             TEXT    PRIMARY KEY,
        lemma                   TEXT    NOT NULL,
        preposition             TEXT,
        shape                   TEXT,
        evidence_class          TEXT    DEFAULT 'structural',
        construction            TEXT,
        review_status           TEXT    DEFAULT 'candidate',
        source                  TEXT    DEFAULT 'candidate_ledger',
        observations            INTEGER DEFAULT 0,
        sentence_pair_observations INTEGER DEFAULT 0,
        gold_positive_hits      INTEGER DEFAULT 0,
        gold_negative_hits      INTEGER DEFAULT 0,
        annotation_agreement    REAL    DEFAULT 0.0
    );

    CREATE TABLE IF NOT EXISTS pattern_stats (
        pattern_key          TEXT    PRIMARY KEY,
        semantic_pattern_key TEXT,
        lemma                TEXT    NOT NULL,
        preposition    TEXT,
        yield_type     TEXT,
        shape          TEXT,
        evidence_class TEXT,
        construction   TEXT,
        review_status  TEXT,
        source         TEXT,
        correct        INTEGER DEFAULT 0,
        total          INTEGER DEFAULT 0,
        -- ``precision`` is a compatibility column name. Interpret it as
        -- same-dataset annotation_agreement / pattern_agreement_rate, not as
        -- final application precision.
        precision      REAL    DEFAULT 0.0
    );

    CREATE TABLE IF NOT EXISTS observation_stats (
        word       TEXT    NOT NULL,
        category   TEXT    NOT NULL,
        correct    INTEGER DEFAULT 0,
        total      INTEGER DEFAULT 0,    -- evaluable hits (trigger cats only)
        precision  REAL    DEFAULT 0.0,  -- correct/total (trigger cats only)
        fp_count   INTEGER DEFAULT 0,
        fn_count   INTEGER DEFAULT 0,
        count_all  INTEGER DEFAULT 0,    -- total occurrences (all categories)
        pair_contributing_occurrences INTEGER DEFAULT 0,
        nonprojecting_occurrences INTEGER DEFAULT 0,
        owner_occurrences INTEGER DEFAULT 0,
        descriptor_occurrences INTEGER DEFAULT 0,
        nested_occurrences INTEGER DEFAULT 0,
        PRIMARY KEY (word, category)
    );

    CREATE INDEX IF NOT EXISTS idx_events_sentence
        ON events(sentence_id);

    CREATE INDEX IF NOT EXISTS idx_obs_sentence
        ON observation_attributions(sentence_id);

    CREATE INDEX IF NOT EXISTS idx_obs_term
        ON observation_attributions(term, category);

    CREATE INDEX IF NOT EXISTS idx_events_pattern
        ON events(pattern_key);

    CREATE INDEX IF NOT EXISTS idx_candidate_rel_sentence
        ON candidate_relations(sentence_id);

    CREATE INDEX IF NOT EXISTS idx_candidate_rel_signature
        ON candidate_relations(pattern_signature);

    CREATE VIEW IF NOT EXISTS v_pattern_reliability AS
        SELECT pattern_key, lemma, preposition, yield_type, shape,
               evidence_class, construction, review_status, source,
               correct, total,
               -- compatibility column: same-dataset annotation_agreement
               precision AS annotation_agreement,
               precision,
               total - correct AS fp
        FROM   pattern_stats
        ORDER  BY precision DESC, total DESC;

    CREATE VIEW IF NOT EXISTS v_observation_reliability AS
        SELECT word, category, correct, total, precision,
               fp_count, fn_count, count_all,
               pair_contributing_occurrences, nonprojecting_occurrences,
               owner_occurrences, descriptor_occurrences, nested_occurrences,
               total - correct AS fp
        FROM   observation_stats
        ORDER  BY precision DESC, total DESC;

    CREATE VIEW IF NOT EXISTS v_semantic_pattern_reliability AS
        SELECT
            semantic_pattern_key,
            lemma,
            preposition,
            yield_type,
            shape,
            evidence_class,
            SUM(correct)  AS correct,
            SUM(total)    AS total,
            CASE WHEN SUM(total) > 0
                 THEN CAST(SUM(correct) AS REAL) / SUM(total)
                 ELSE 0.0 END AS precision,
            GROUP_CONCAT(DISTINCT construction) AS constructions
        FROM   pattern_stats
        WHERE  semantic_pattern_key IS NOT NULL
        GROUP  BY semantic_pattern_key
        ORDER  BY precision DESC, total DESC;
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = sqlite3.connect(str(self.path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()


    def __enter__(self) -> "EventDatabase":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        """Commit pending writes and close the connection."""
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None


    def populate(
        self,
        builder: "EventReportBuilder",
        run_meta: dict | None = None,
    ) -> None:
        """Write all data collected by an EventReportBuilder into the database."""
        assert self._conn is not None, "Database is closed"
        cur = self._conn.cursor()

        self._populate_run_metadata(cur, run_meta)
        self._populate_sentences(cur, builder.sentences)

        event_rows = self._event_rows(builder.sentences)
        cur.executemany(
            "INSERT INTO events "
            "(sentence_id, event_type, semantic_event_class, event_detail, realization_channel, "
            " relation_type, span, "
            " span_min, span_max, pred_pairs, predicate_text, predicate_lemma, "
            " pattern_key, semantic_pattern_key, yield_type, construction, is_inferred, chain_readable, "
            " chain_steps_json, arguments_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            event_rows,
        )

        candidate_relation_rows = []
        # canonical_pattern_key -> {(sentence_id, pair): in_gold}.
        sentence_pair_agreement: dict[str, dict[tuple[str, tuple[int, int]], bool | None]] = {}
        # Mention-level observation totals (every CandidateRelation row).
        mention_observations: dict[str, int] = {}
        # Per-key metadata (last-write-wins for descriptive fields)
        candidate_key_meta: dict[str, dict] = {}

        for sid, report in builder.sentences.items():
            artifacts = report.artifacts or {}
            gold_pairs = report.gold_pairs
            gold_set = (
                {tuple(sorted(p)) for p in gold_pairs}
                if gold_pairs is not None else None
            )
            for rel in artifacts.get("candidate_relations", []) or []:
                pair = rel.get("pair") or []
                pair_a = int(pair[0]) if len(pair) == 2 else None
                pair_b = int(pair[1]) if len(pair) == 2 else None
                key = rel.get("pattern_key") or ""
                shape = rel.get("shape") or ""
                construction = rel.get("construction") or ""
                candidate_relation_rows.append((
                    sid,
                    rel.get("candidate_id"),
                    pair_a,
                    pair_b,
                    rel.get("source_event_id"),
                    rel.get("parser_source"),
                    rel.get("schema"),
                    rel.get("predicate_lemma"),
                    rel.get("event_type"),
                    rel.get("event_detail"),
                    rel.get("realization_channel") or "unknown",
                    rel.get("source_role"),
                    rel.get("target_role"),
                    rel.get("predicate_parser_dep"),
                    rel.get("source_argument_role"),
                    rel.get("target_argument_role"),
                    rel.get("source_parser_dep"),
                    rel.get("target_parser_dep"),
                    rel.get("case_marker"),
                    rel.get("assertion_status"),
                    rel.get("projection_mode"),
                    rel.get("pattern_signature"),
                    rel.get("pattern_decision"),
                    rel.get("decision"),
                    rel.get("ambiguity_reason"),
                    1 if rel.get("compound_propagated") else 0,
                    1 if rel.get("owner_propagated") else 0,
                    1 if rel.get("descriptor_propagated") else 0,
                    1 if rel.get("is_nominalized") else 0,
                    key,
                    shape,
                    construction,
                    json.dumps(rel),
                ))

                # Asserted, unambiguous candidates; role-nominal IS-A rows may be projected.
                assertion_status = (rel.get("assertion_status") or "asserted").lower()
                ambiguity_reason = (rel.get("ambiguity_reason") or "").strip()
                if key and assertion_status == "asserted" and not ambiguity_reason:
                    is_relational = shape.upper() == "RELATIONAL" or (
                        rel.get("projection_mode") or ""
                    ).lower() == "relational"
                    is_applied = (rel.get("decision") or "").lower() == "applied"
                    review_status = (
                        "review_only_evidence"
                        if is_relational and not is_applied
                        else "candidate"
                    )
                    source = (
                        "candidate_relation_review"
                        if is_relational and not is_applied
                        else "candidate_ledger"
                    )
                    mention_observations[key] = mention_observations.get(key, 0) + 1
                    if key not in candidate_key_meta:
                        candidate_key_meta[key] = {
                            "lemma": rel.get("predicate_lemma") or "",
                            "preposition": rel.get("case_marker") or None,
                            "shape": shape,
                            "construction": construction,
                            "evidence_class": "relational" if is_relational else "structural",
                            "review_status": review_status,
                            "source": source,
                        }
                    elif review_status == "candidate":
                        candidate_key_meta[key]["review_status"] = "candidate"
                        candidate_key_meta[key]["source"] = "candidate_ledger"
                    if pair_a is not None and pair_b is not None:
                        sp = sentence_pair_agreement.setdefault(key, {})
                        sp_key = (sid, tuple(sorted((pair_a, pair_b))))
                        if sp_key not in sp:
                            if gold_set is None:
                                sp[sp_key] = None
                            else:
                                sp[sp_key] = sp_key[1] in gold_set
        if candidate_relation_rows:
            cur.executemany(
                "INSERT INTO candidate_relations "
                "(sentence_id, candidate_id, pair_a, pair_b, source_event_id, "
                " parser_source, schema, predicate_lemma, event_type, event_detail, "
                " realization_channel, "
                " source_role, target_role, predicate_parser_dep, "
                " source_argument_role, target_argument_role, source_parser_dep, "
                " target_parser_dep, case_marker, assertion_status, "
                " projection_mode, pattern_signature, pattern_decision, decision, "
                " ambiguity_reason, compound_propagated, owner_propagated, "
                " descriptor_propagated, is_nominalized, canonical_pattern_key, shape, "
                " construction, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                candidate_relation_rows,
            )
        if candidate_key_meta:
            agg_rows = []
            for key, meta in candidate_key_meta.items():
                sp_map = sentence_pair_agreement.get(key, {})
                gold_positive = sum(1 for v in sp_map.values() if v is True)
                gold_negative = sum(1 for v in sp_map.values() if v is False)
                sentence_pair_obs = len(sp_map)
                evaluable = gold_positive + gold_negative
                agreement = gold_positive / evaluable if evaluable else 0.0
                agg_rows.append((
                    key,
                    meta["lemma"],
                    meta["preposition"],
                    meta["shape"],
                    meta.get("evidence_class", "structural"),
                    meta["construction"],
                    meta.get("review_status", "candidate"),
                    meta.get("source", "candidate_ledger"),
                    mention_observations.get(key, 0),
                    sentence_pair_obs,
                    gold_positive,
                    gold_negative,
                    round(agreement, 4),
                ))
            cur.executemany(
                "INSERT OR REPLACE INTO candidate_pattern_stats "
                "(pattern_key, lemma, preposition, shape, evidence_class, "
                " construction, review_status, source, observations, "
                " sentence_pair_observations, gold_positive_hits, "
                " gold_negative_hits, annotation_agreement) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                agg_rows,
            )
        self._populate_observation_attributions(
            cur, getattr(builder, "_attributions", [])
        )
        ps = getattr(builder, "_pattern_observations", None)
        if ps is not None:
            self._populate_pattern_stats(cur, ps)

        self._populate_observation_stats(cur, builder.observations)

        self._conn.commit()


    def _create_tables(self) -> None:
        assert self._conn is not None
        self._conn.executescript(self._SCHEMA)
        self._conn.commit()

    @staticmethod
    def _populate_run_metadata(cur: sqlite3.Cursor, run_meta: dict | None) -> None:
        if not run_meta:
            return
        cur.executemany(
            "INSERT OR REPLACE INTO run_metadata (key, value) VALUES (?, ?)",
            [
                (k, v if isinstance(v, str) else json.dumps(v))
                for k, v in run_meta.items()
            ],
        )

    def _event_rows(self, sentences: dict) -> list[tuple]:
        return [
            self._event_row(sid, ev)
            for sid, report in sentences.items()
            for ev in report.events
        ]

    @staticmethod
    def _populate_sentences(cur: sqlite3.Cursor, sentences: dict) -> None:
        sentence_rows = [
            (
                sid,
                report.text,
                report.orig_sentence or None,
                json.dumps(report.proteins) if report.proteins else None,
                json.dumps(report.pred_pairs),
                json.dumps(report.gold_pairs) if report.gold_pairs is not None else None,
                report.label,
                report.record_version or None,
                json.dumps(report.provenance) if report.provenance else None,
                json.dumps(report.artifacts) if report.artifacts else None,
                json.dumps(report.run_meta) if report.run_meta else None,
            )
            for sid, report in sentences.items()
        ]
        cur.executemany(
            "INSERT OR IGNORE INTO sentences "
            "(sentence_id, text, orig_text, proteins, pred_pairs, gold_pairs, label, "
            " record_version, provenance, artifacts, run_meta) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            sentence_rows,
        )

    @staticmethod
    def _event_row(sid: str, ev) -> tuple:
        span_min = ev.span_indices[0] if len(ev.span_indices) >= 2 else None
        span_max = ev.span_indices[1] if len(ev.span_indices) >= 2 else None
        chain = getattr(ev, "chain_dict", None) or {}
        pattern_key = getattr(ev, "pattern_key", "") or chain.get("pattern_key")
        semantic_pattern_key = (
            getattr(ev, "semantic_pattern_key", "")
            or chain.get("semantic_pattern_key")
        )
        yield_type = getattr(ev, "yield_type", "") or chain.get("yield_type")
        construction = getattr(ev, "construction", "") or chain.get("construction")
        # Legacy per-slot chain; absent for canonical events.
        chain_steps = chain.get("steps") if chain else None
        chain_steps_json = json.dumps(chain_steps) if chain_steps else None
        # arguments_json: full ArgumentFrame detail per role slot
        args_json = json.dumps(ev.arguments) if ev.arguments else None
        return (
            sid,
            ev.event_type,
            ev.semantic_event_class,
            ev.event_detail,
            ev.realization_channel,
            ev.relation_type or None,
            ev.span or None,
            span_min,
            span_max,
            json.dumps(ev.pred_pairs),
            ev.predicate["text"] if ev.predicate else None,
            ev.predicate["lemma"] if ev.predicate else None,
            pattern_key or None,
            semantic_pattern_key or None,
            yield_type or None,
            construction or None,
            int(chain.get("is_inferred", 0)) if chain else 0,
            chain.get("readable") if chain else None,
            chain_steps_json,
            args_json,
        )

    @staticmethod
    def _populate_observation_attributions(
        cur: sqlite3.Cursor,
        attributions: list[dict],
    ) -> None:
        if not attributions:
            return
        cur.executemany(
            "INSERT INTO observation_attributions "
            "(sentence_id, event_detail, pattern_key, term, category, role, "
            " is_correct, is_evaluable) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    a["sentence_id"],
                    a["event_detail"],
                    a.get("pattern_key"),
                    a["term"],
                    a["category"],
                    a["role"],
                    int(a["is_correct"]),
                    int(a["is_evaluable"]),
                )
                for a in attributions
            ],
        )

    def _populate_pattern_stats(self, cur: sqlite3.Cursor, pattern_store) -> None:
        """Populate pattern_stats keyed by ``evidence_class|shape|lemma|prep|construction``."""
        by_lemma = getattr(pattern_store, "_by_lemma", {})
        rows = []
        aggregated: dict[str, tuple] = {}
        for lemma, entries in by_lemma.items():
            for entry in entries:
                key = entry.canonical_pattern_key()
                semantic_key = entry.semantic_pattern_key()
                shape = entry.derived_shape()
                if key in aggregated:
                    # Merge repeated keys; semantic_key is [1], correct/total are [10]/[11].
                    prev = aggregated[key]
                    correct = prev[10] + entry.correct
                    total = prev[11] + entry.total
                    pair_precision = correct / total if total else 0.0
                    aggregated[key] = (
                        key, semantic_key, entry.word, entry.preposition,
                        entry.yield_type, shape,
                        entry.evidence_class or "structural",
                        entry.construction or "",
                        entry.review_status or "enabled",
                        entry.source or "default",
                        correct, total, round(pair_precision, 4),
                    )
                else:
                    aggregated[key] = (
                        key,
                        semantic_key,
                        entry.word,
                        entry.preposition,
                        entry.yield_type,
                        shape,
                        entry.evidence_class or "structural",
                        entry.construction or "",
                        entry.review_status or "enabled",
                        entry.source or "default",
                        entry.correct,
                        entry.total,
                        round(entry.pair_precision, 4),
                    )
        rows = list(aggregated.values())
        if rows:
            cur.executemany(
                "INSERT OR REPLACE INTO pattern_stats "
                "(pattern_key, semantic_pattern_key, lemma, preposition, yield_type, shape, "
                " evidence_class, construction, review_status, source, "
                " correct, total, precision) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
    def _populate_observation_stats(self, cur: sqlite3.Cursor,
                                observations: dict) -> None:
        rows = []
        for category, stats_map in observations.items():
            for word, stat in stats_map.items():
                rows.append((
                    word,
                    category,
                    stat.correct,
                    stat.count,
                    round(stat.precision, 4),
                    stat.fp_count,
                    stat.fn_count,
                    stat.count_all,
                    stat.pair_contributing_occurrences,
                    stat.nonprojecting_occurrences,
                    stat.owner_occurrences,
                    stat.descriptor_occurrences,
                    stat.nested_occurrences,
                ))
        if rows:
            cur.executemany(
                "INSERT OR REPLACE INTO observation_stats "
                "(word, category, correct, total, precision, fp_count, fn_count, count_all, "
                " pair_contributing_occurrences, nonprojecting_occurrences, owner_occurrences, "
                " descriptor_occurrences, nested_occurrences) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
