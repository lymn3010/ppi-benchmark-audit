#!/usr/bin/env python3
"""Summarize physical-interaction review candidates by carrier evidence."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reproduction.config import DB_PATHS, PAPER_DIR, SOURCE_PAPER_DIR, stats_dir  # noqa: E402
from src.analysis.carrier_taxonomy import entity_boundary_carrier_heads  # noqa: E402

CORE_PHYSICAL_TRIGGERS = {
    "bind",
    "interact",
    "associate",
    "complex",
    "dimerize",
    "heterodimerize",
    "oligomerize",
}
ENTITY_BOUNDARY_CARRIERS = entity_boundary_carrier_heads()
DETAIL_FIELDS = [
    "corpus",
    "key",
    "shape",
    "trigger",
    "case_marker",
    "construction",
    "support",
    "gold_positive",
    "gold_negative",
    "minority",
    "cpr",
    "carrier_words",
    "entity_shifting_carriers",
    "carrier_row_count",
    "strict_no_carrier_rows",
    "included_no_entity_shift",
]
SUMMARY_FIELDS = [
    "corpus",
    "isolated_canonical_cells",
    "core_physical_verbal_cells",
    "core_physical_no_entity_shift_cells",
    "strict_no_carrier_rows_cells",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _parse_pattern_key(key: str) -> dict[str, str]:
    parts = str(key or "").split("|")
    return {
        "evidence_class": parts[0] if len(parts) > 0 else "",
        "shape": parts[1] if len(parts) > 1 else "",
        "trigger": parts[2] if len(parts) > 2 else "",
        "case_marker": parts[3] if len(parts) > 3 else "",
        "construction": parts[4] if len(parts) > 4 else "",
    }


def _carrier_index() -> dict[tuple[str, str], list[str]]:
    by_cell: dict[tuple[str, str], list[str]] = {}
    for corpus in DB_PATHS:
        path = stats_dir(corpus) / "carrier_trigger_observations.csv"
        for row in _read_csv(path):
            key = (row.get("corpus") or corpus, row.get("pattern_key") or "")
            if not key[1]:
                continue
            carrier = (row.get("carrier_word") or "").strip().lower()
            if carrier:
                by_cell.setdefault(key, []).append(carrier)
    return by_cell


def analyze(
    *,
    isolated_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    isolated_rows = [
        row for row in _read_csv(isolated_path)
        if row.get("dimension") == "canonical_pattern"
        and row.get("kind") == "isolated_contrary"
    ]
    carriers = _carrier_index()

    details: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    summary_by_corpus: dict[str, Counter] = {}

    for row in isolated_rows:
        corpus = row["corpus"]
        summary_by_corpus.setdefault(corpus, Counter())["isolated_canonical_cells"] += 1
        parsed = _parse_pattern_key(row["key"])
        is_core = (
            parsed["evidence_class"] == "structural"
            and parsed["construction"] == "verbal"
            and parsed["trigger"] in CORE_PHYSICAL_TRIGGERS
        )
        if is_core:
            summary_by_corpus[corpus]["core_physical_verbal_cells"] += 1
        carrier_words = sorted(set(carriers.get((corpus, row["key"]), ())))
        entity_shift = sorted(set(carrier_words) & ENTITY_BOUNDARY_CARRIERS)
        no_entity_shift = is_core and not entity_shift
        strict_no_carrier = is_core and not carrier_words
        if no_entity_shift:
            summary_by_corpus[corpus]["core_physical_no_entity_shift_cells"] += 1
        if strict_no_carrier:
            summary_by_corpus[corpus]["strict_no_carrier_rows_cells"] += 1
        if is_core:
            detail = {
                "corpus": corpus,
                "key": row["key"],
                "shape": parsed["shape"],
                "trigger": parsed["trigger"],
                "case_marker": parsed["case_marker"],
                "construction": parsed["construction"],
                "support": int(row["support"]),
                "gold_positive": int(row["gold_positive"]),
                "gold_negative": int(row["gold_negative"]),
                "minority": int(row["minority"]),
                "cpr": row["cpr"],
                "carrier_words": "+".join(carrier_words),
                "entity_shifting_carriers": "+".join(entity_shift),
                "carrier_row_count": len(carrier_words),
                "strict_no_carrier_rows": int(strict_no_carrier),
                "included_no_entity_shift": int(no_entity_shift),
            }
            details.append(detail)
            if no_entity_shift:
                selected.append(detail)

    summary_rows: list[dict[str, object]] = []
    for corpus in sorted(summary_by_corpus):
        counter = summary_by_corpus[corpus]
        summary_rows.append({
            "corpus": corpus,
            "isolated_canonical_cells": counter["isolated_canonical_cells"],
            "core_physical_verbal_cells": counter["core_physical_verbal_cells"],
            "core_physical_no_entity_shift_cells": counter["core_physical_no_entity_shift_cells"],
            "strict_no_carrier_rows_cells": counter["strict_no_carrier_rows_cells"],
        })

    totals = {
        "isolated_canonical_cells": len(isolated_rows),
        "core_physical_verbal_cells": len(details),
        "core_physical_no_entity_shift_cells": len(selected),
        "strict_no_carrier_rows_cells": sum(
            int(row["strict_no_carrier_rows"]) for row in details
        ),
        "per_corpus": summary_rows,
        "core_physical_triggers": sorted(CORE_PHYSICAL_TRIGGERS),
        # Legacy key name; the paper calls this entity-boundary carrier evidence.
        "entity_shifting_carriers": sorted(ENTITY_BOUNDARY_CARRIERS),
    }
    return totals, summary_rows, details


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isolated",
        type=Path,
        default=SOURCE_PAPER_DIR / "results/within-corpus-cells.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PAPER_DIR / "results",
    )
    args = parser.parse_args(argv)

    totals, summary_rows, detail_rows = analyze(isolated_path=args.isolated)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        args.out_dir / "within-corpus-physical-summary.csv",
        SUMMARY_FIELDS,
        summary_rows,
    )
    _write_csv(
        args.out_dir / "within-corpus-physical-cells.csv",
        DETAIL_FIELDS,
        detail_rows,
    )
    (args.out_dir / "within-corpus-physical-summary.json").write_text(
        json.dumps(totals, indent=2),
        encoding="utf-8",
    )
    print(f"[core-physical] wrote {args.out_dir}")
    print(json.dumps(totals, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
