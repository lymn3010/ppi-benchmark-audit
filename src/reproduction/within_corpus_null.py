#!/usr/bin/env python3
"""Compare isolated-contrary counts with a within-corpus label-shuffle null."""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

POSITIVE = {"positive", "gold_positive", "1", "true", "True"}


def _firings(stats_dir: Path) -> list[tuple[str, bool]]:
    path = stats_dir / "analysis" / "stats" / "pattern_observations.csv"
    if not path.exists():
        path = stats_dir / "pattern_observations.csv"
    out = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.append((r["canonical_pattern_key"],
                        str(r.get("gold_label", "")).strip() in POSITIVE))
    return out


def _isolated(labelled: list[tuple[str, bool]], min_support: int, frac: float) -> int:
    cell = defaultdict(list)
    for key, pos in labelled:
        cell[key].append(pos)
    iso = 0
    for obs in cell.values():
        s = len(obs)
        if s < min_support:
            continue
        pos = sum(obs)
        neg = s - pos
        if pos and neg and min(pos, neg) / s <= frac:
            iso += 1
    return iso


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-support", type=int, default=5)
    ap.add_argument("--isolated-frac", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=None,
                    help="write a committed null-summary JSON for the paper-number gate")
    args = ap.parse_args(argv)

    per_corpus = [_firings(d) for d in args.run_dirs]
    observed = sum(_isolated(rows, args.min_support, args.isolated_frac) for rows in per_corpus)

    rng = random.Random(args.seed)
    null = []
    for _ in range(args.iters):
        total = 0
        for rows in per_corpus:
            keys = [k for k, _ in rows]
            labels = [p for _, p in rows]
            rng.shuffle(labels)
            total += _isolated(list(zip(keys, labels)), args.min_support, args.isolated_frac)
        null.append(total)
    null.sort()
    n = len(null)
    mean = sum(null) / n
    p95 = null[int(0.95 * n)]
    pval = sum(1 for x in null if x >= observed) / n

    print(f"observed isolated-contrary cells: {observed}")
    print(f"label-shuffle null (within-corpus, iters={n}, seed={args.seed}): "
          f"mean={mean:.1f}, 95th={p95}, min={null[0]}, max={null[-1]}")
    print(f"P(null >= observed) = {pval:.3f}")
    if args.out:
        import json
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "observed": observed,
            "iters": n,
            "seed": args.seed,
            "min_support": args.min_support,
            "isolated_frac": args.isolated_frac,
            "null_mean": round(mean, 2),
            "null_p95": p95,
            "null_min": null[0],
            "null_max": null[-1],
            "p_value": round(pval, 4),
        }, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
