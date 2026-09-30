"""SQL queries for coverage, annotation agreement and sentence-level review."""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any



def _wilson_ci(tp: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for tp/n, clamped to [0, 1]; (0.0, 0.0) when n == 0."""
    if n == 0:
        return 0.0, 0.0
    p = tp / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (
        max(0.0, round(center - margin, 4)),
        min(1.0, round(center + margin, 4)),
    )


def _add_ci(rows: list[dict], tp_key: str = "tp", n_key: str = "total") -> list[dict]:
    """Attach Wilson 95% CI fields ``ci_low`` / ``ci_high`` to each row in-place."""
    for r in rows:
        r["ci_low"], r["ci_high"] = _wilson_ci(r.get(tp_key, 0), r.get(n_key, 0))
    return rows


# Separate policy disagreement from small-sample noise.

# Chi-square critical values at alpha = 0.05, keyed by df.
_CHI2_CRIT_005 = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070, 6: 12.592}


def _heterogeneity(per_corpus: list[dict]) -> dict:
    """Cochran's Q homogeneity test of per-corpus proportions.

    Returns zeros/False for fewer than two corpora or a pooled rate of 0 or 1.
    """
    rows = [r for r in per_corpus if r.get("total", 0) > 0]
    k = len(rows)
    if k < 2:
        return {"q_stat": 0.0, "df": 0, "i_squared": 0.0,
                "heterogeneity_significant": False}

    tp_sum = sum(r["tp"] for r in rows)
    n_sum = sum(r["total"] for r in rows)
    p_bar = tp_sum / n_sum if n_sum else 0.0
    if p_bar <= 0.0 or p_bar >= 1.0:
        return {"q_stat": 0.0, "df": k - 1, "i_squared": 0.0,
                "heterogeneity_significant": False}

    q = sum(r["total"] * (r["tp"] / r["total"] - p_bar) ** 2 for r in rows)
    q /= p_bar * (1.0 - p_bar)
    df = k - 1
    i_squared = max(0.0, (q - df) / q) if q > 0 else 0.0
    crit = _CHI2_CRIT_005.get(df)
    significant = crit is not None and q > crit
    return {
        "q_stat": round(q, 4),
        "df": df,
        "i_squared": round(i_squared, 4),
        "heterogeneity_significant": bool(significant),
    }


def _significance_fields(per_corpus: list[dict]) -> dict:
    """Add Wilson CIs to per-corpus rows in place; return CI-disjoint and Cochran's Q fields."""
    for r in per_corpus:
        r["ci_low"], r["ci_high"] = _wilson_ci(r.get("tp", 0), r.get("total", 0))

    evaluable = [r for r in per_corpus if r.get("total", 0) > 0]
    ci_disjoint = False
    if len(evaluable) >= 2:
        hi = max(evaluable, key=lambda r: r["precision"])
        lo = min(evaluable, key=lambda r: r["precision"])
        ci_disjoint = hi["ci_low"] > lo["ci_high"]

    return {"ci_disjoint": bool(ci_disjoint), **_heterogeneity(per_corpus)}



class _ClosingConnection(sqlite3.Connection):
    """sqlite3 connection whose context manager also closes the handle."""

    def __exit__(self, exc_type, exc, tb) -> bool:
        super().__exit__(exc_type, exc, tb)
        self.close()
        return False


def open_db(path: Path | str) -> sqlite3.Connection:
    """Open events.db read-only with sqlite3.Row rows; closed on context exit."""
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, factory=_ClosingConnection)
    conn.row_factory = sqlite3.Row
    return conn


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    return [dict(r) for r in rows]


def _attach_realization_channels(
    db_path: Path | str,
    rows: list[dict],
) -> list[dict]:
    """Attach per-pattern extraction-channel counts when the v8 column exists."""
    if not rows:
        return rows
    with open_db(db_path) as conn:
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(events)").fetchall()
        }
        if "realization_channel" not in columns:
            return rows
        counts: dict[str, dict[str, int]] = {}
        for row in conn.execute(
            """
            SELECT pattern_key, realization_channel, COUNT(*) AS observations
            FROM events
            WHERE pattern_key IS NOT NULL
            GROUP BY pattern_key, realization_channel
            """
        ).fetchall():
            key = str(row["pattern_key"] or "")
            channel = str(row["realization_channel"] or "unknown")
            counts.setdefault(key, {})[channel] = int(row["observations"])
    for row in rows:
        row["realization_channels"] = counts.get(str(row.get("pattern_key") or ""), {})
    return rows



def pattern_reliability(
    db_path: Path | str,
    *,
    min_count: int = 5,
) -> list[dict]:
    """Return per-pattern label agreement and confidence intervals."""
    sql = """
        SELECT
            pattern_key,
            semantic_pattern_key,
            lemma,
            preposition,
            yield_type,
            shape,
            evidence_class,
            construction,
            correct              AS tp,
            (total - correct)    AS fp,
            total,
            precision
        FROM pattern_stats
        WHERE total >= ?
        ORDER BY precision DESC, total DESC
    """
    with open_db(db_path) as conn:
        rows = conn.execute(sql, (min_count,)).fetchall()
    return _attach_realization_channels(db_path, _add_ci(_rows_to_dicts(rows)))


def semantic_pattern_reliability(
    db_path: Path | str,
    *,
    min_count: int = 5,
) -> list[dict]:
    """Return label agreement grouped by semantic predicate."""
    sql = """
        SELECT
            semantic_pattern_key,
            lemma,
            preposition,
            yield_type,
            shape,
            evidence_class,
            SUM(correct)         AS tp,
            (SUM(total) - SUM(correct)) AS fp,
            SUM(total)           AS total,
            CASE WHEN SUM(total) > 0
                 THEN CAST(SUM(correct) AS REAL) / SUM(total)
                 ELSE 0.0 END    AS precision,
            GROUP_CONCAT(DISTINCT construction) AS constructions
        FROM pattern_stats
        WHERE semantic_pattern_key IS NOT NULL
        GROUP BY semantic_pattern_key
        HAVING SUM(total) >= ?
        ORDER BY precision DESC, total DESC
    """
    with open_db(db_path) as conn:
        rows = conn.execute(sql, (min_count,)).fetchall()
    return _add_ci(_rows_to_dicts(rows))


def observation_reliability(
    db_path: Path | str,
    *,
    category: str | None = None,
    min_count: int = 1,
) -> list[dict]:
    """Per-term precision with Wilson CIs from ``v_observation_reliability``."""
    where_clauses = ["total >= ?"]
    params: list[Any] = [min_count]
    if category is not None:
        where_clauses.append("category = ?")
        params.append(category)

    optional = (
        "pair_contributing_occurrences",
        "nonprojecting_occurrences",
        "owner_occurrences",
        "descriptor_occurrences",
        "nested_occurrences",
    )
    with open_db(db_path) as conn:
        available = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(v_observation_reliability)").fetchall()
        }
    optional_select = ",\n            ".join(
        name if name in available else f"0 AS {name}"
        for name in optional
    )
    sql = f"""
        SELECT
            word        AS term,
            category,
            correct     AS tp,
            fp,
            fn_count    AS fn,
            total,
            precision,
            count_all,
            {optional_select}
        FROM v_observation_reliability
        WHERE {" AND ".join(where_clauses)}
        ORDER BY category, precision DESC, total DESC
    """
    with open_db(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return _add_ci(_rows_to_dicts(rows))


def fp_sentences(
    db_path: Path | str,
    *,
    limit: int = 500,
) -> list[dict]:
    """Return sentences with prediction-only pairs for review."""
    sql = """
        SELECT
            s.sentence_id,
            s.text          AS sentence_text,
            s.orig_text,
            s.label,
            s.pred_pairs,
            s.gold_pairs,
            GROUP_CONCAT(DISTINCT e.pattern_key)  AS pattern_keys,
            GROUP_CONCAT(DISTINCT e.event_detail) AS event_details
        FROM sentences s
        LEFT JOIN events e ON s.sentence_id = e.sentence_id
        WHERE s.label IN ('too_many', 'wrong_pairs')
        GROUP BY s.sentence_id
        ORDER BY s.sentence_id
        LIMIT ?
    """
    with open_db(db_path) as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
    return _rows_to_dicts(rows)


def fn_sentences(
    db_path: Path | str,
    *,
    limit: int = 500,
) -> list[dict]:
    """Sentences that miss a gold-positive pair; ``no_event`` rows before ``too_few``."""
    sql = """
        SELECT
            s.sentence_id,
            s.text          AS sentence_text,
            s.orig_text,
            s.label,
            s.pred_pairs,
            s.gold_pairs,
            GROUP_CONCAT(DISTINCT e.pattern_key)  AS pattern_keys,
            GROUP_CONCAT(DISTINCT e.event_detail) AS event_details
        FROM sentences s
        LEFT JOIN events e ON s.sentence_id = e.sentence_id
        WHERE s.label IN ('too_few', 'no_event')
        GROUP BY s.sentence_id
        ORDER BY
            CASE s.label WHEN 'no_event' THEN 0 ELSE 1 END,
            s.sentence_id
        LIMIT ?
    """
    with open_db(db_path) as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
    return _rows_to_dicts(rows)


def sentence_label_distribution(db_path: Path | str) -> dict:
    """Count sentences by agreement with released labels."""
    sql = """
        SELECT
            COUNT(*)                                                    AS total,
            SUM(CASE WHEN label = 'correct'     THEN 1 ELSE 0 END)     AS correct,
            SUM(CASE WHEN label = 'too_many'    THEN 1 ELSE 0 END)     AS too_many,
            SUM(CASE WHEN label = 'too_few'     THEN 1 ELSE 0 END)     AS too_few,
            SUM(CASE WHEN label = 'wrong_pairs' THEN 1 ELSE 0 END)     AS wrong_pairs,
            SUM(CASE WHEN label = 'no_gold'     THEN 1 ELSE 0 END)     AS no_gold,
            SUM(CASE WHEN label = 'no_event'    THEN 1 ELSE 0 END)     AS no_event
        FROM sentences
    """
    with open_db(db_path) as conn:
        row = conn.execute(sql).fetchone()
    return dict(row) if row else {}




def _canonical_pairs(raw: str | None) -> set[tuple[int, int]]:
    """Parse a JSON ``[[i, j], ...]`` pair list into a set of order-independent
    ``(i, j)`` tuples.  Returns an empty set for NULL / empty / malformed input.
    """
    if not raw:
        return set()
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return set()
    pairs: set[tuple[int, int]] = set()
    for p in data or []:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            i, j = p
            pairs.add((i, j) if i <= j else (j, i))
    return pairs


def coverage_rate(db_path: Path | str) -> dict:
    """Return the fraction of released-positive pairs covered by a named pattern."""
    sql = "SELECT pred_pairs, gold_pairs, label FROM sentences"
    with open_db(db_path) as conn:
        rows = conn.execute(sql).fetchall()

    gold_total = 0
    covered = 0
    uncovered_by_label: dict[str, int] = {}
    for row in rows:
        gold = _canonical_pairs(row["gold_pairs"])
        if not gold:
            continue
        pred = _canonical_pairs(row["pred_pairs"])
        hit = len(gold & pred)
        gold_total += len(gold)
        covered += hit
        missed = len(gold) - hit
        if missed:
            label = row["label"] or "unknown"
            uncovered_by_label[label] = uncovered_by_label.get(label, 0) + missed

    rate = round(covered / gold_total, 4) if gold_total else 0.0
    ci_low, ci_high = _wilson_ci(covered, gold_total)
    return {
        "gold_pairs":         gold_total,
        "covered":            covered,
        "uncovered":          gold_total - covered,
        "coverage_rate":      rate,
        "ci_low":             ci_low,
        "ci_high":            ci_high,
        "uncovered_by_label": uncovered_by_label,
    }



def cross_corpus_pattern_stats(
    db_map: dict[str, Path | str],
    *,
    min_count: int = 3,
) -> list[dict]:
    """Run ``pattern_reliability()`` per corpus; one row per (pattern_key, corpus)."""
    all_rows: list[dict] = []
    for corpus, db_path in db_map.items():
        try:
            rows = pattern_reliability(db_path, min_count=min_count)
        except Exception:
            continue  # missing or corrupt db: skip silently
        for r in rows:
            all_rows.append({**r, "corpus": corpus})

    all_rows.sort(key=lambda r: (r["pattern_key"], r["corpus"]))
    return all_rows


def cross_corpus_coverage(db_map: dict[str, Path | str]) -> list[dict]:
    """Per-corpus coverage rate, highest first; missing databases are skipped."""
    all_rows: list[dict] = []
    for corpus, db_path in db_map.items():
        try:
            cov = coverage_rate(db_path)
        except Exception:
            continue
        all_rows.append({"corpus": corpus, **cov})

    all_rows.sort(key=lambda r: -r["coverage_rate"])
    return all_rows


def cross_corpus_observation_stats(
    db_map: dict[str, Path | str],
    *,
    category: str | None = None,
    min_count: int = 3,
) -> list[dict]:
    """Per-corpus observation-term precision; one row per (term, category, corpus)."""
    all_rows: list[dict] = []
    for corpus, db_path in db_map.items():
        try:
            rows = observation_reliability(db_path, category=category, min_count=min_count)
        except Exception:
            continue
        for r in rows:
            all_rows.append({**r, "corpus": corpus})

    all_rows.sort(key=lambda r: (r["term"], r["category"], r["corpus"]))
    return all_rows


def _precision_bounds(rows: list[dict]) -> tuple[float, float]:
    precisions = [r["precision"] for r in rows]
    return max(precisions), min(precisions)


def _per_corpus_precision_rows(
    rows: list[dict],
    *,
    include_constructions: bool = False,
) -> list[dict]:
    per_corpus = []
    for row in sorted(rows, key=lambda x: x["corpus"]):
        item = {
            "corpus":    row["corpus"],
            "precision": round(row["precision"], 4),
            "tp":        row["tp"],
            "fp":        row["fp"],
            "total":     row["total"],
        }
        if include_constructions:
            item["constructions"] = row.get("constructions", "")
        per_corpus.append(item)
    return per_corpus


def annotation_inconsistency_table(
    db_map: dict[str, Path | str],
    *,
    min_count: int = 3,
    min_corpora: int = 2,
) -> list[dict]:
    """Find patterns whose positive-label rates differ across corpora."""
    flat = cross_corpus_pattern_stats(db_map, min_count=min_count)

    # Group by pattern_key
    by_pattern: dict[str, list[dict]] = {}
    for row in flat:
        by_pattern.setdefault(row["pattern_key"], []).append(row)

    result = []
    for pattern_key, rows in by_pattern.items():
        if len(rows) < min_corpora:
            continue
        p_max, p_min = _precision_bounds(rows)
        per_corpus = _per_corpus_precision_rows(rows)
        significance = _significance_fields(per_corpus)
        result.append({
            "pattern_key":    pattern_key,
            "lemma":          rows[0].get("lemma", ""),
            "construction":   rows[0].get("construction", ""),
            "inconsistency":  round(p_max - p_min, 4),
            "precision_max":  round(p_max, 4),
            "precision_min":  round(p_min, 4),
            "corpus_count":   len(rows),
            **significance,
            "per_corpus":     per_corpus,
        })

    result.sort(key=lambda r: (-r["inconsistency"], -r["corpus_count"]))
    return result


def semantic_annotation_inconsistency_table(
    db_map: dict[str, Path | str],
    *,
    min_count: int = 3,
    min_corpora: int = 2,
) -> list[dict]:
    """Compare corpus label rates using the direction-independent semantic key."""
    # Collect per-corpus semantic reliability rows
    all_rows: list[dict] = []
    for corpus, db_path in db_map.items():
        try:
            rows = semantic_pattern_reliability(db_path, min_count=min_count)
        except Exception:
            continue
        for r in rows:
            all_rows.append({**r, "corpus": corpus})

    # Group by semantic_pattern_key
    by_key: dict[str, list[dict]] = {}
    for row in all_rows:
        by_key.setdefault(row["semantic_pattern_key"], []).append(row)

    result = []
    for sem_key, rows in by_key.items():
        if len(rows) < min_corpora:
            continue
        p_max, p_min = _precision_bounds(rows)
        # Collect all construction variants seen across corpora
        constructions: set[str] = set()
        for r in rows:
            for c in (r.get("constructions") or "").split(","):
                c = c.strip()
                if c:
                    constructions.add(c)
        per_corpus = _per_corpus_precision_rows(rows, include_constructions=True)
        significance = _significance_fields(per_corpus)
        result.append({
            "semantic_pattern_key": sem_key,
            "lemma":                rows[0].get("lemma", ""),
            "inconsistency":        round(p_max - p_min, 4),
            "precision_max":        round(p_max, 4),
            "precision_min":        round(p_min, 4),
            "corpus_count":         len(rows),
            "constructions_seen":   ",".join(sorted(constructions)),
            **significance,
            "per_corpus":           per_corpus,
        })

    result.sort(key=lambda r: (-r["inconsistency"], -r["corpus_count"]))
    return result
