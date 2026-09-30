#!/usr/bin/env python
"""Report separately corrected comparison families and document-bootstrap robustness."""
from __future__ import annotations

from src.reproduction.config import PAPER_DIR as EXPERIMENT_DIR

import argparse
import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.reproduction.config import CORPORA, stats_dir  # noqa: E402
from src.analysis.statistical_tests import benjamini_hochberg  # noqa: E402

TESTS_CSV = EXPERIMENT_DIR / 'results/cross-corpus/fisher-tests.csv'

# What one row of each dimension compares, and whether that unit is a construction.
DIMENSION_UNITS = {
    "pattern": ("canonical construction: evidence class, shape, trigger lemma, "
                "preposition, construction", True),
    "semantic_pattern": ("collapsed relation: evidence class and predicate for simple "
                         "pairs; shape and case retained for richer relations", True),
    "carrier_pattern": ("carrier word crossed with construction", True),
    "carrier": ("carrier word alone", False),
    "trigger": ("predicate lemma alone -- not a construction", False),
    "pair_stage": ("stage of this reader's own pipeline -- a property of the extractor, "
                   "not of the corpus", False),
}
PRIMARY = "pattern"
POS, NEG = "positive", "negative"


def breakdown() -> dict:
    rows = list(csv.DictReader(TESTS_CSV.open()))
    per: dict[str, dict] = {}
    for r in rows:
        d = r["dimension"]
        e = per.setdefault(d, {"tests": 0, "significant": 0})
        e["tests"] += 1
        if float(r["q_value"]) < 0.05:
            e["significant"] += 1
    for d, e in per.items():
        unit, is_constr = DIMENSION_UNITS.get(d, ("?", False))
        e["statistical_unit"] = unit
        e["construction_shaped"] = is_constr
        e["bh_family"] = f"{d} ({e['tests']} tests, corrected independently)"
    return per


def _load_pattern_rows() -> list[dict]:
    """One row per candidate firing, carrying corpus, canonical key, label and document."""
    out = []
    for c in CORPORA:
        f = stats_dir(c) / "pattern_observations.csv"
        for r in csv.DictReader(f.open()):
            lab = (r.get("gold_label") or "").strip().lower()
            key = (r.get("canonical_pattern_key") or "").strip()
            sid = r.get("sentence_id", "")
            if lab not in (POS, NEG) or not key:
                continue
            out.append({"corpus": c, "key": key, "y": 1 if lab == POS else 0,
                        "doc": ".".join(sid.split(".")[:2]) if sid else c})
    return out


def _count_significant(rows, min_support=5, alpha=0.05):
    """Count BH-significant pairwise Fisher tests across eligible cells."""
    cell: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        c = cell[(r["corpus"], r["key"])]
        c[0] += r["y"]
        c[1] += 1
    by_key: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for (corpus, key), (pos, tot) in cell.items():
        if tot >= min_support:
            by_key[key].append((corpus, pos, tot))
    tests = []
    for key, cells in by_key.items():
        for (c1, p1, t1), (c2, p2, t2) in combinations(sorted(cells), 2):
            _, p = fisher_exact([[p1, t1 - p1], [p2, t2 - p2]])
            tests.append({"p_value": p})
    if not tests:
        return 0, 0
    corrected = benjamini_hochberg(tests)
    return sum(float(t["q_value"]) < alpha for t in corrected), len(corrected)


def cluster_bootstrap(rows, boot: int, seed: int) -> dict:
    """Resample documents within corpus; recount significant cells each time."""
    rng = np.random.default_rng(seed)
    by_doc: dict[str, list[dict]] = defaultdict(list)
    corpus_of: dict[str, str] = {}
    for r in rows:
        by_doc[r["doc"]].append(r)
        corpus_of[r["doc"]] = r["corpus"]
    docs_by_corpus: dict[str, list[str]] = defaultdict(list)
    for d, c in corpus_of.items():
        docs_by_corpus[c].append(d)

    sig, tot, frac = [], [], []
    for _ in range(boot):
        res = []
        for c, docs in docs_by_corpus.items():
            for d in rng.integers(0, len(docs), size=len(docs)):
                res.extend(by_doc[docs[d]])
        s, t = _count_significant(res)
        if t:
            sig.append(s)
            tot.append(t)
            frac.append(s / t)
    sig, tot, frac = np.array(sig), np.array(tot), np.array(frac)
    return {
        "n_resamples": int(sig.size),
        "significant": {"mean": round(float(sig.mean()), 2),
                        "ci_low": int(np.percentile(sig, 2.5)),
                        "ci_high": int(np.percentile(sig, 97.5))},
        "tests": {"mean": round(float(tot.mean()), 2),
                  "ci_low": int(np.percentile(tot, 2.5)),
                  "ci_high": int(np.percentile(tot, 97.5))},
        "significant_fraction": {"mean": round(float(frac.mean()), 4),
                                 "ci_low": round(float(np.percentile(frac, 2.5)), 4),
                                 "ci_high": round(float(np.percentile(frac, 97.5)), 4)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--out", type=Path,
                    default=EXPERIMENT_DIR / 'results/construction-families.json')
    args = ap.parse_args()

    per = breakdown()
    tot_t = sum(e["tests"] for e in per.values())
    tot_s = sum(e["significant"] for e in per.values())
    print(f"{'dimension':18} {'tests':>6} {'sig':>5}  construction?  unit")
    for d in sorted(per, key=lambda k: -per[k]["tests"]):
        e = per[d]
        print(f"{d:18} {e['tests']:6d} {e['significant']:5d}  "
              f"{'yes' if e['construction_shaped'] else 'NO ':13} {e['statistical_unit'][:52]}")
    print(f"{'TOTAL (6 families)':18} {tot_t:6d} {tot_s:5d}")
    print(f"\nprimary family = {PRIMARY}: {per[PRIMARY]['significant']} of {per[PRIMARY]['tests']}")

    rows = _load_pattern_rows()
    recomputed_sig, recomputed_tests = _count_significant(rows)
    print(f"recomputed from ledgers: {recomputed_sig} of {recomputed_tests} "
          f"(committed table: {per[PRIMARY]['significant']} of {per[PRIMARY]['tests']})")

    print(f"\ndocument cluster bootstrap on the primary family ({args.boot} resamples)...")
    boot = cluster_bootstrap(rows, args.boot, args.seed)
    print(f"  significant: {boot['significant']['mean']} "
          f"[{boot['significant']['ci_low']}, {boot['significant']['ci_high']}]")
    print(f"  of tests:    {boot['tests']['mean']} "
          f"[{boot['tests']['ci_low']}, {boot['tests']['ci_high']}]")
    print(f"  fraction:    {boot['significant_fraction']['mean']} "
          f"[{boot['significant_fraction']['ci_low']}, {boot['significant_fraction']['ci_high']}]")

    out = {
        "per_dimension": per,
        "total_tests_all_families": tot_t,
        "total_significant_all_families": tot_s,
        "primary_family": PRIMARY,
        "primary_significant": per[PRIMARY]["significant"],
        "primary_tests": per[PRIMARY]["tests"],
        "primary_recomputed_from_ledgers": {"significant": recomputed_sig,
                                            "tests": recomputed_tests},
        "primary_document_cluster_bootstrap": boot,
        "note": ("BH is applied per dimension in src/analysis/statistical_tests.py, so the "
                 "submitted '70 of 266' is a sum over six independently corrected families, "
                 "two of which (trigger, pair_stage) are not constructions at all."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
