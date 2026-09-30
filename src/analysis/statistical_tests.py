"""Statistical summaries over the ``analysis/stats`` CSV ledgers."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

from src.models.observation_unit import ObservationUnit


@dataclass(frozen=True)
class DimensionSpec:
    """Definition of one statistics dimension in the ledger surface."""

    name: str
    file_name: str
    key_fields: tuple[str, ...]
    label_field: str = "gold_label"
    observation_unit: str = ObservationUnit.CANDIDATE_PATTERN_FIRING.value


DIMENSIONS: dict[str, DimensionSpec] = {
    "pattern": DimensionSpec(
        name="pattern",
        file_name="pattern_observations.csv",
        key_fields=("canonical_pattern_key",),
    ),
    "semantic_pattern": DimensionSpec(
        name="semantic_pattern",
        file_name="pattern_observations.csv",
        key_fields=("semantic_pattern_key",),
    ),
    "trigger": DimensionSpec(
        name="trigger",
        file_name="pattern_observations.csv",
        key_fields=("predicate_lemma",),
    ),
    "carrier": DimensionSpec(
        name="carrier",
        file_name="carrier_trigger_observations.csv",
        key_fields=("carrier_word",),
    ),
    "carrier_pattern": DimensionSpec(
        name="carrier_pattern",
        file_name="carrier_trigger_observations.csv",
        key_fields=("carrier_word", "pattern_key"),
    ),
    "pair_stage": DimensionSpec(
        name="pair_stage",
        file_name="pair_observations.csv",
        key_fields=("stage",),
        observation_unit=ObservationUnit.BOUNDED_SENTENCE_PAIR.value,
    ),
}


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return a Wilson score confidence interval for a binomial proportion."""

    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for [[a, b], [c, d]] without SciPy."""

    if min(a, b, c, d) < 0:
        raise ValueError("Fisher exact counts must be non-negative")
    row1 = a + b
    row2 = c + d
    col1 = a + c
    total = row1 + row2
    if row1 == 0 or row2 == 0 or total == 0:
        return 1.0

    min_x = max(0, col1 - row2)
    max_x = min(row1, col1)
    observed = _hypergeom_prob(a, row1, row2, col1, total)
    p_value = 0.0
    eps = 1e-12
    for x in range(min_x, max_x + 1):
        prob = _hypergeom_prob(x, row1, row2, col1, total)
        if prob <= observed + eps:
            p_value += prob
    return min(1.0, max(0.0, p_value))


def benjamini_hochberg(
    rows: Iterable[dict[str, object]],
    *,
    p_key: str = "p_value",
    q_key: str = "q_value",
) -> list[dict[str, object]]:
    """Attach Benjamini-Hochberg q-values to a sequence of row dicts."""

    materialized = [dict(row) for row in rows]
    indexed = sorted(
        [(idx, _float(row.get(p_key), default=1.0)) for idx, row in enumerate(materialized)],
        key=lambda item: item[1],
    )
    m = len(indexed)
    if m == 0:
        return materialized
    running = 1.0
    for rank_from_end, (idx, p_value) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        running = min(running, p_value * m / rank)
        materialized[idx][q_key] = _format_float(running)
    return materialized


def summarize_stats_dirs(
    stats_dirs: Iterable[Path | str],
    *,
    dimension: str,
    min_support: int = 5,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Per-corpus summaries and pairwise cross-corpus tests; corpora are never pooled."""

    spec = _dimension_spec(dimension)
    per_cell: dict[tuple[str, str], dict[str, object]] = {}
    by_key: dict[str, list[dict[str, object]]] = {}

    for stats_dir in stats_dirs:
        resolved = resolve_stats_dir(Path(stats_dir))
        for row in _read_csv(resolved / spec.file_name):
            corpus = (row.get("corpus") or resolved.parent.parent.name or "").strip()
            key = _join_key(row, spec.key_fields)
            if not corpus or not key:
                continue
            gold_label = (row.get(spec.label_field) or "").strip().lower()
            if gold_label not in {"positive", "negative"}:
                continue
            cell = per_cell.setdefault(
                (corpus, key),
                {
                    "dimension": spec.name,
                    "corpus": corpus,
                    "key": key,
                    "support": 0,
                    "gold_positive": 0,
                    "gold_negative": 0,
                },
            )
            cell["support"] = int(cell["support"]) + 1
            if gold_label == "positive":
                cell["gold_positive"] = int(cell["gold_positive"]) + 1
            else:
                cell["gold_negative"] = int(cell["gold_negative"]) + 1

    summaries = []
    for cell in sorted(per_cell.values(), key=lambda r: (str(r["corpus"]), str(r["key"]))):
        support = int(cell["support"])
        positive = int(cell["gold_positive"])
        low, high = wilson_ci(positive, support)
        row = {
            **cell,
            "cpr": _format_float(positive / support if support else 0.0),
            "wilson_low": _format_float(low),
            "wilson_high": _format_float(high),
        }
        summaries.append(row)
        by_key.setdefault(str(row["key"]), []).append(row)

    tests: list[dict[str, object]] = []
    for key, cells in sorted(by_key.items()):
        eligible = [cell for cell in cells if int(cell["support"]) >= min_support]
        for left, right in combinations(sorted(eligible, key=lambda r: str(r["corpus"])), 2):
            a = int(left["gold_positive"])
            b = int(left["gold_negative"])
            c = int(right["gold_positive"])
            d = int(right["gold_negative"])
            p_value = fisher_exact_two_sided(a, b, c, d)
            tests.append(
                {
                    "dimension": spec.name,
                    "key": key,
                    "corpus_1": left["corpus"],
                    "positive_1": a,
                    "negative_1": b,
                    "support_1": left["support"],
                    "cpr_1": left["cpr"],
                    "corpus_2": right["corpus"],
                    "positive_2": c,
                    "negative_2": d,
                    "support_2": right["support"],
                    "cpr_2": right["cpr"],
                    "cpr_delta_abs": _format_float(abs(float(left["cpr"]) - float(right["cpr"]))),
                    "p_value": _format_float(p_value),
                }
            )
    return summaries, benjamini_hochberg(tests)


def write_statistical_outputs(
    stats_dirs: Iterable[Path | str],
    output_dir: Path | str,
    *,
    dimensions: Iterable[str] | None = None,
    min_support: int = 5,
) -> dict[str, Path]:
    """Write summary/test CSVs for the requested dimensions."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = list(dimensions or DIMENSIONS)
    all_summaries: list[dict[str, object]] = []
    all_tests: list[dict[str, object]] = []
    for dimension in selected:
        summaries, tests = summarize_stats_dirs(
            stats_dirs,
            dimension=dimension,
            min_support=min_support,
        )
        all_summaries.extend(summaries)
        all_tests.extend(tests)

    summary_path = output_dir / "cpr_summary.csv"
    tests_path = output_dir / "cross_corpus_fisher_bh.csv"
    _write_csv(summary_path, SUMMARY_FIELDS, all_summaries)
    _write_csv(tests_path, TEST_FIELDS, all_tests)
    return {
        "cpr_summary": summary_path,
        "cross_corpus_fisher_bh": tests_path,
    }


SUMMARY_FIELDS = [
    "dimension",
    "corpus",
    "key",
    "support",
    "gold_positive",
    "gold_negative",
    "cpr",
    "wilson_low",
    "wilson_high",
]

TEST_FIELDS = [
    "dimension",
    "key",
    "corpus_1",
    "positive_1",
    "negative_1",
    "support_1",
    "cpr_1",
    "corpus_2",
    "positive_2",
    "negative_2",
    "support_2",
    "cpr_2",
    "cpr_delta_abs",
    "p_value",
    "q_value",
]


def resolve_stats_dir(path: Path) -> Path:
    """Resolve either a run directory or an ``analysis/stats`` directory."""

    candidates = [
        path,
        path / "analysis" / "stats",
        path / "stats",
    ]
    for candidate in candidates:
        if ((candidate / "pattern_observations.csv").exists()
                or (candidate / "corpus_summary.csv").exists()):
            return candidate
    raise FileNotFoundError(f"Could not locate analysis/stats under {path}")


def _dimension_spec(name: str) -> DimensionSpec:
    try:
        return DIMENSIONS[name]
    except KeyError as exc:
        allowed = ", ".join(sorted(DIMENSIONS))
        raise ValueError(f"Unknown dimension {name!r}; choose one of: {allowed}") from exc


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _join_key(row: dict[str, str], fields: tuple[str, ...]) -> str:
    parts = []
    for field in fields:
        value = (row.get(field) or "").strip()
        if not value:
            return ""
        parts.append(value)
    return " | ".join(parts)


def _hypergeom_prob(x: int, row1: int, row2: int, col1: int, total: int) -> float:
    return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(total, col1)


def _float(value: object, *, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _format_float(value: float) -> str:
    return f"{value:.6g}"
