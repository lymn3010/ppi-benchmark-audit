"""Compute cross-corpus positive rates, Wilson intervals and Fisher tests."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.statistical_tests import DIMENSIONS, write_statistical_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute CPR summaries, Wilson intervals, pairwise Fisher exact "
            "tests, and Benjamini-Hochberg q-values from analysis/stats ledgers."
        )
    )
    parser.add_argument(
        "stats_or_run_dirs",
        nargs="+",
        help="Run directories or direct analysis/stats directories.",
    )
    parser.add_argument(
        "--out-dir",
        default="output/statistical_tests/latest",
        help="Directory for cpr_summary.csv and cross_corpus_fisher_bh.csv.",
    )
    parser.add_argument(
        "--dimension",
        action="append",
        choices=sorted(DIMENSIONS),
        help=(
            "Dimension to analyze. May be supplied multiple times. "
            "Defaults to all dimensions."
        ),
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=5,
        help="Minimum support in each corpus before a pairwise test is reported.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = write_statistical_outputs(
        [Path(path) for path in args.stats_or_run_dirs],
        Path(args.out_dir),
        dimensions=args.dimension,
        min_support=args.min_support,
    )
    readable_names = {
        "cpr_summary": "pattern-rates.csv",
        "cross_corpus_fisher_bh": "fisher-tests.csv",
    }
    for name, filename in readable_names.items():
        target = Path(args.out_dir) / filename
        paths[name].replace(target)
        paths[name] = target
    print("Wrote statistical outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
