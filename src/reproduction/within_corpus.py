#!/usr/bin/env python3
"""Find repeated constructions with mixed labels within each corpus.

Write mixed-cell counts and examples for review; do not revise corpus labels."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.statistical_tests import wilson_ci, resolve_stats_dir  # noqa: E402

# dimension -> (ledger file, key columns), using the finest keys.
DIMENSIONS = {
    "canonical_pattern": ("pattern_observations.csv", ("canonical_pattern_key",)),
    "semantic_pattern": ("pattern_observations.csv", ("semantic_pattern_key",)),
    "carrier_pattern": ("carrier_trigger_observations.csv", ("carrier_word", "pattern_key")),
}

POSITIVE = {"positive", "gold_positive", "1", "true", "True"}


def _is_positive(gold_label: str) -> bool:
    return str(gold_label).strip() in POSITIVE


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def analyze(stats_dirs: list[Path], min_support: int, isolated_frac: float):
    cells_out: list[dict] = []
    summary_out: list[dict] = []
    examples_out: list[dict] = []

    for stats_dir in stats_dirs:
        resolved = resolve_stats_dir(stats_dir)
        corpus = _infer_corpus(resolved)
        for dim, (fname, keycols) in DIMENSIONS.items():
            rows = _read_csv(resolved / fname)
            # group observations by key -> list of (sentence_id, is_pos)
            cell = defaultdict(list)
            for r in rows:
                key = "|".join(r.get(k, "") for k in keycols)
                cell[key].append((r.get("sentence_id", ""), _is_positive(r.get("gold_label", ""))))

            assessable_firings = 0          # firings in cells we can assess (support >= min_support)
            mixed_firings = 0
            assessable_cells = 0            # cells we can assess (support >= min_support)
            unanimous_cells = 0            # assessable cells whose firings all share one label
            n_isolated = n_split = 0

            for key, obs in cell.items():
                support = len(obs)
                if support < min_support:
                    continue
                assessable_firings += support
                assessable_cells += 1
                pos = sum(1 for _, p in obs if p)
                neg = support - pos
                if pos == 0 or neg == 0:
                    unanimous_cells += 1
                    continue  # unanimous = consistent at this key
                mixed_firings += support
                minority = min(pos, neg)
                cpr = pos / support
                lo, hi = wilson_ci(pos, support)
                # Isolated-contrary: minority <= isolated_frac of support (4-1 isolated, 3-2 split).
                is_isolated = (minority / support) <= isolated_frac
                kind = "isolated_contrary" if is_isolated else "split"
                if is_isolated:
                    n_isolated += 1
                else:
                    n_split += 1
                cells_out.append({
                    "corpus": corpus,
                    "dimension": dim,
                    "key": key,
                    "support": support,
                    "gold_positive": pos,
                    "gold_negative": neg,
                    "minority": minority,
                    "cpr": round(cpr, 4),
                    "wilson_low": round(lo, 4),
                    "wilson_high": round(hi, 4),
                    "label_entropy": round(_entropy(cpr), 4),
                    "kind": kind,
                })
                if is_isolated:
                    majority_pos = pos >= neg
                    majority_ids = [s for s, p in obs if p == majority_pos][:5]
                    contrary_ids = [s for s, p in obs if p != majority_pos][:5]
                    examples_out.append({
                        "corpus": corpus,
                        "dimension": dim,
                        "key": key,
                        "support": support,
                        "majority_label": "positive" if majority_pos else "negative",
                        "majority_sentence_ids": majority_ids,
                        "contrary_sentence_ids": contrary_ids,
                    })

            summary_out.append({
                "corpus": corpus,
                "dimension": dim,
                # Denominator: firings in assessable cells only.
                "evaluable_firings": assessable_firings,
                "mixed_firings": mixed_firings,
                "mixed_firing_share": round(mixed_firings / assessable_firings, 4) if assessable_firings else 0.0,
                # Cell profile: unanimous vs each kind of mixed.
                "assessable_cells": assessable_cells,
                "unanimous_cells": unanimous_cells,
                "isolated_contrary_cells": n_isolated,
                "split_cells": n_split,
            })

    return cells_out, summary_out, examples_out


def _infer_corpus(resolved: Path) -> str:
    # stats dir is .../<corpus>/<split>/<run>/analysis/stats ; corpus is 4 up
    parts = resolved.parts
    for marker in ("aimed", "bioinfer", "hprd50", "iepa", "lll"):
        if marker in parts:
            return marker
    # fall back: read corpus_summary.csv
    rows = _read_csv(resolved / "corpus_summary.csv")
    for r in rows:
        if r.get("corpus"):
            return r["corpus"]
    return resolved.parts[-4] if len(resolved.parts) >= 4 else "unknown"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_or_stats_dirs", nargs="+", type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--min-support", type=int, default=5)
    ap.add_argument("--isolated-frac", type=float, default=0.25,
                    help="minority <= ceil(support*frac) counts as isolated-contrary")
    args = ap.parse_args(argv)

    cells, summary, examples = analyze(
        args.run_or_stats_dirs, args.min_support, args.isolated_frac)

    _write_csv(args.out_dir / "within-corpus-cells.csv",
               sorted(cells, key=lambda r: (r["corpus"], r["dimension"], -r["support"])))
    _write_csv(args.out_dir / "within-corpus-summary.csv", summary)
    with (args.out_dir / "within-corpus-examples.jsonl").open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[within-corpus] wrote {args.out_dir}")
    print(f"  mixed cells: {len(cells)} | isolated-contrary examples: {len(examples)}")
    for s in summary:
        print(f"  {s['corpus']:9s} {s['dimension']:18s} "
              f"mixed_share={s['mixed_firing_share']:.3f} "
              f"isolated={s['isolated_contrary_cells']} split={s['split_cells']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
