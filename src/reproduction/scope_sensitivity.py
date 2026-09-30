#!/usr/bin/env python
"""Repeat the main analyses under different reader-evidence inclusion rules."""
from __future__ import annotations

from src.reproduction.config import PAPER_DIR as EXPERIMENT_DIR

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.reproduction.config import CORPORA  # noqa: E402
from src.reproduction.construction_families import _count_significant  # noqa: E402
from src.reproduction.residual_bootstrap import bootstrap as document_bootstrap  # noqa: E402
from src.reproduction.residual_corpus import _load  # noqa: E402

from src.reproduction import matched_gap_nulls as mgn

COVERAGE = EXPERIMENT_DIR / 'results/coverage.json'

# Inclusion rules, widest first: rule -> kept ``decision`` values.
RULES: dict[str, set[str] | None] = {
    "as_reported": None,
    "drop_unselected": {"applied", "review_only", "not_applied"},
    "applied_or_review_only": {"applied", "review_only"},
    "applied_only": {"applied"},
}


def _filter(rows: list[dict], keep: set[str] | None) -> list[dict]:
    return rows if keep is None else [r for r in rows if r["decision"] in keep]


def _primary_family(rows: list[dict], min_support: int = 5) -> dict:
    """Significant count and mean |difference| for the primary family."""
    from collections import defaultdict
    from itertools import combinations

    family_rows = [{"corpus": r["corpus"], "key": r["canonical_key"], "y": r["y"], "doc": r["doc"]}
                   for r in rows if r["canonical_key"]]
    significant, tests = _count_significant(family_rows, min_support=min_support)

    cell: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for r in family_rows:
        c = cell[(r["corpus"], r["key"])]
        c[0] += r["y"]
        c[1] += 1
    by_key: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for (corpus, key), (pos, tot) in cell.items():
        if tot >= min_support:
            by_key[key].append((corpus, pos, tot))
    deltas = [abs(p1 / t1 - p2 / t2)
              for cells in by_key.values()
              for (_c1, p1, t1), (_c2, p2, t2) in combinations(sorted(cells), 2)]
    return {"significant": significant, "tests": tests,
            "significant_share": round(significant / tests, 3) if tests else None,
            "mean_abs_rate_difference": round(float(np.mean(deltas)), 3) if deltas else None}


def _matched_pair_contrast(rows_match: list[dict], min_rows: int = 5) -> dict:
    """Signed AIMed -> BioInfer contrast over matched constructions (manuscript: 0.267/30)."""
    y = np.array([r["y"] for r in rows_match])
    strata = mgn._pair_index(rows_match, (), min_rows, ("aimed", "bioinfer"))
    mean, sd, n, _mean_abs = mgn._pair_stats(strata, y)
    if n == 0:
        return {"n_strata": 0, "mean_contrast": None, "spread": None}
    return {"n_strata": n, "mean_contrast": round(mean, 3), "spread": round(sd, 3)}


def _residual(rows: list[dict], *, b: int) -> dict:
    """Raw and within-corpus lift with the document-cluster bootstrap interval."""
    boot = document_bootstrap(rows, b=b, seed=0)
    raw, within = boot["raw_corpus_lift"], boot["within_corpus_lift"]
    return {
        "raw_corpus_lift": {"mean": raw["mean"],
                            "ci": [raw["ci_low_2.5"], raw["ci_high_97.5"]]},
        "within_corpus_lift": {"mean": within["mean"],
                               "ci": [within["ci_low_2.5"], within["ci_high_97.5"]],
                               "excludes_zero": bool(
                                   within["ci_low_2.5"] > 0 or within["ci_high_97.5"] < 0)},
    }


def evidence_tier_composition() -> dict:
    """Per-corpus evidence-tier shares of released positives (descriptive)."""
    data = json.loads(COVERAGE.read_text(encoding="utf-8"))
    committed_tiers = ("clause_SVO", "event_nominal", "complex_state")
    out = {}
    for corpus, v in data["per_corpus"].items():
        total = v["_gold_total"]
        committed = sum(v[t] for t in committed_tiers)
        out[corpus] = {
            "released_positives": total,
            "committed": committed,
            "committed_share": round(committed / total, 3) if total else None,
            "held_evidence": v.get("held_evidence", 0),
            "lexpath": v.get("lexpath", 0),
            "out_of_scope": v.get("out_of_scope", 0),
        }
    shares = [v["committed_share"] for v in out.values() if v["committed_share"] is not None]
    return {"per_corpus": out,
            "committed_share_range": [min(shares), max(shares)],
            "note": "Positives-only tiers. Reported per corpus as Reviewer 1 asked; not used "
                    "to subset the statistical population, which would shift each corpus's "
                    "positive rate by construction."}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--b", type=int, default=2000, help="bootstrap replicates per rule")
    args = ap.parse_args()

    # Predictive model caps predicates; matching analysis does not.
    rows = _load(cap_predicates=True)
    rows_match = _load(cap_predicates=False)
    # _load drops the canonical key, which the primary family needs; re-read it alongside.
    import csv

    from src.reproduction.config import stats_dir

    keys: list[str] = []
    for c in CORPORA:
        for r in csv.DictReader((stats_dir(c) / "pattern_observations.csv").open()):
            if str(r.get("gold_label", "")).strip() in {"positive", "gold_positive", "1", "negative"}:
                keys.append((r.get("canonical_pattern_key") or "").strip())
    if len(keys) != len(rows):
        raise AssertionError(f"canonical key alignment: {len(keys)} keys against {len(rows)} rows")
    for r, k in zip(rows, keys):
        r["canonical_key"] = k
    for r, k in zip(rows_match, keys):
        r["canonical_key"] = k

    results = {}
    for name, keep in RULES.items():
        sub, sub_match = _filter(rows, keep), _filter(rows_match, keep)
        by_corpus = Counter(r["corpus"] for r in sub)
        results[name] = {
            "kept_decisions": sorted(keep) if keep else "all",
            "n_observations": len(sub),
            "share_of_full": round(len(sub) / len(rows), 3),
            "per_corpus_n": {c: by_corpus.get(c, 0) for c in CORPORA},
            "per_corpus_positive_rate": {
                c: round(float(np.mean([r["y"] for r in sub if r["corpus"] == c])), 3)
                for c in CORPORA if by_corpus.get(c, 0)},
            "primary_family": _primary_family(sub),
            "residual": _residual(sub, b=args.b),
            "matched_pair_contrast": _matched_pair_contrast(sub_match),
        }
        r = results[name]
        print(f"{name:24} n={r['n_observations']:5}  primary "
              f"{r['primary_family']['significant']:3}/{r['primary_family']['tests']:<3} "
              f"raw {r['residual']['raw_corpus_lift']['mean']:+.3f}  within "
              f"{r['residual']['within_corpus_lift']['mean']:+.3f} "
              f"{r['residual']['within_corpus_lift']['ci']}  meanAbs {r['primary_family']['mean_abs_rate_difference']}  matched "
              f"{r['matched_pair_contrast']['mean_contrast']}"
              f"/{r['matched_pair_contrast']['n_strata']}")

    payload = {
        "decision_axis": {
            "field": "decision in ledgers/<corpus>/pattern-observations.csv",
            "values": dict(Counter(r["decision"] for r in rows)),
            "note": "The reader's own candidate-selection outcome, label-blind and defined "
                    "for positives and negatives alike. No current analysis filters on it, "
                    "so 'as_reported' is the widest rule and reproduces the manuscript.",
        },
        "evidence_tiers": evidence_tier_composition(),
        "inclusion_rules": results,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out}")
    else:
        print()
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
