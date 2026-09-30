#!/usr/bin/env python
"""Measure whether corpus identity improves held-out label prediction."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.reproduction.config import DB_PATHS, CORPORA, stats_dir  # noqa: E402
from src.analysis.carrier_taxonomy import carrier_head_classes  # noqa: E402

warnings.filterwarnings("ignore")

POS = {"positive", "gold_positive", "1"}
SEM_COLS = [
    "assertion", "construction", "shape", "case", "srole", "trole", "pred",
    "carrier_any", "carrier_roles", "carrier_classes", "carrier_words",
]

def _carrier_index(corpus: str) -> dict[tuple[str, str], dict[str, str]]:
    """Carrier controls keyed by (sentence_id, canonical_pattern_key); ignores gold labels."""
    path = stats_dir(corpus) / "carrier_trigger_observations.csv"
    grouped: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: {"words": set(), "roles": set(), "classes": set()}
    )
    for r in csv.DictReader(path.open()):
        sid = r.get("sentence_id", "")
        key = r.get("pattern_key", "")
        word = (r.get("carrier_word") or "").strip().lower()
        if not sid or not key or not word:
            continue
        g = grouped[(sid, key)]
        g["words"].add(word)
        g["roles"].add((r.get("role") or "unknown").strip() or "unknown")
        g["classes"].update(carrier_head_classes(word))
    out: dict[tuple[str, str], dict[str, str]] = {}
    for key, vals in grouped.items():
        out[key] = {
            "carrier_any": "yes",
            "carrier_roles": "+".join(sorted(vals["roles"])) or "none",
            "carrier_classes": "+".join(sorted(vals["classes"])) or "none",
            "carrier_words": "+".join(sorted(vals["words"])) or "none",
        }
    return out


def _load(cap_predicates: bool = True) -> list[dict]:
    """Load the analysis population from the frozen ledgers.

    Matching analyses must pass ``cap_predicates=False``.
    """
    rows: list[dict] = []
    for c in CORPORA:
        f = stats_dir(c) / "pattern_observations.csv"
        carriers = _carrier_index(c)
        for r in csv.DictReader(f.open()):
            sid = r.get("sentence_id", "")
            carrier = carriers.get((sid, r.get("canonical_pattern_key", "")), {
                "carrier_any": "no",
                "carrier_roles": "none",
                "carrier_classes": "none",
                "carrier_words": "none",
            })
            rows.append({
                "corpus": c,
                "y": 1 if str(r.get("gold_label", "")).strip() in POS else 0,
                "doc": ".".join(sid.split(".")[:2]) if sid else c,  # AIMed.d115.s970 -> AIMed.d115
                # Reader decision, used only by scope_sensitivity.py.
                "decision": (r.get("decision") or "").strip(),
                "assertion": r.get("assertion_status", "") or "na",
                "construction": r.get("construction", "") or "na",
                "shape": r.get("shape", "") or "na",
                "case": r.get("case_marker", "") or "na",
                "srole": r.get("source_role", "") or "na",
                "trole": r.get("target_role", "") or "na",
                "pred": r.get("predicate_lemma", "") or "na",
                **carrier,
            })
    # cap predicate cardinality so the model is parsimonious (avoid separation/overfit)
    if cap_predicates:
        top = {p for p, _ in Counter(r["pred"] for r in rows).most_common(25)}
        for r in rows:
            r["pred"] = r["pred"] if r["pred"] in top else "OTHER"
    return rows


def _foldwise(rows, cols, y, splitter, *, per_fold=False, corpora=None):
    """Fit encoder and model per fold; return mean held-out (logloss, auc, n_features).

    ``per_fold`` adds paired fold vectors; ``corpora`` adds within-corpus AUC.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import roc_auc_score, log_loss
    from scipy.sparse import hstack, csr_matrix
    # Prevalence control: training-fold positive rate of the corpus.
    rate_col = "corpus_rate" in cols
    cat_cols = [c for c in cols if c != "corpus_rate"]

    ll, au, nf, strat = [], [], [], []
    for tr, te in splitter:
        if len(set(y[te])) < 2:
            continue
        enc = OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=True)
        xtr = enc.fit_transform([[rows[i][c] for c in cat_cols] for i in tr])
        xte = enc.transform([[rows[i][c] for c in cat_cols] for i in te])
        if rate_col:
            num = defaultdict(lambda: [0, 0])
            for i in tr:
                num[rows[i]["corpus"]][0] += y[i]
                num[rows[i]["corpus"]][1] += 1
            overall = float(y[tr].mean())
            rate = {c: v[0] / v[1] for c, v in num.items()}
            def _r(i):
                return [rate.get(rows[i]["corpus"], overall)]
            xtr = hstack([xtr, csr_matrix([_r(i) for i in tr])]).tocsr()
            xte = hstack([xte, csr_matrix([_r(i) for i in te])]).tocsr()
        m = LogisticRegression(max_iter=2000, C=1.0).fit(xtr, y[tr])
        p = m.predict_proba(xte)[:, 1]
        ll.append(log_loss(y[te], p, labels=[0, 1]))
        au.append(roc_auc_score(y[te], p))
        nf.append(xtr.shape[1])
        if corpora is not None:
            num = den = 0.0
            for c in sorted({corpora[i] for i in te}):  # sorted: stable accumulation order
                sel = [k for k, i in enumerate(te) if corpora[i] == c]
                yy = y[te][sel]
                if len(set(yy)) < 2:
                    continue  # a corpus with one class in this fold carries no ranking info
                num += roc_auc_score(yy, p[sel]) * len(sel)
                den += len(sel)
            strat.append(num / den if den else float("nan"))
    if per_fold:
        return {
            "logloss": np.array(ll), "auc": np.array(au),
            "n_features": int(np.mean(nf)),
            "auc_within_corpus": np.array(strat) if corpora is not None else None,
        }
    return float(np.mean(ll)), float(np.mean(au)), int(np.mean(nf))


def _pm(v: np.ndarray) -> dict:
    """mean, standard error and a normal 95% interval for a per-fold vector."""
    v = np.asarray([x for x in v if not np.isnan(x)], dtype=float)
    if v.size == 0:
        return {}
    se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
    return {"mean": round(float(v.mean()), 4), "se": round(se, 4),
            "ci_low": round(float(v.mean() - 1.96 * se), 4),
            "ci_high": round(float(v.mean() + 1.96 * se), 4),
            "n_folds": int(v.size)}


def out_of_fold(rows: list[dict]) -> dict:
    """Held-out cues-only vs cues+corpus, with base-rate and within-corpus controls."""
    from sklearn.model_selection import StratifiedKFold

    y = np.array([r["y"] for r in rows])
    idx = np.arange(len(rows))
    corpora = [r["corpus"] for r in rows]

    def cv(cols):
        return _foldwise(rows, cols, y,
                         StratifiedKFold(5, shuffle=True, random_state=0).split(idx, y),
                         per_fold=True, corpora=corpora)

    f1 = cv(SEM_COLS)
    f2 = cv(SEM_COLS + ["corpus"])
    f3 = cv(SEM_COLS + ["corpus_rate"])

    d_auc = f2["auc"] - f1["auc"]
    d_ll = f1["logloss"] - f2["logloss"]
    d_within = f2["auc_within_corpus"] - f1["auc_within_corpus"]
    d_rate = f3["auc"] - f1["auc"]
    return {
        "m1_semantics": {"logloss": _pm(f1["logloss"]), "auc": _pm(f1["auc"]),
                         "auc_within_corpus": _pm(f1["auc_within_corpus"]),
                         "n_features": f1["n_features"]},
        "m2_plus_corpus": {"logloss": _pm(f2["logloss"]), "auc": _pm(f2["auc"]),
                           "auc_within_corpus": _pm(f2["auc_within_corpus"]),
                           "n_features": f2["n_features"]},
        "m3_corpus_base_rate": {"auc": _pm(f3["auc"]), "n_features": f3["n_features"]},
        "corpus_auc_improvement_paired": _pm(d_auc),
        "corpus_logloss_improvement_paired": _pm(d_ll),
        "corpus_auc_improvement_within_corpus_paired": _pm(d_within),
        "base_rate_only_auc_improvement_paired": _pm(d_rate),
        # kept as plain scalars so existing consumers (figure script, number gate) still work
        "corpus_auc_improvement": round(float(d_auc.mean()), 4),
        "corpus_logloss_improvement": round(float(d_ll.mean()), 4),
    }


def document_bootstrap(rows: list[dict], *, b: int = 2000, seed: int = 0) -> dict:
    """Document-cluster bootstrap of the raw and within-corpus AUC lift."""
    from src.reproduction.residual_bootstrap import bootstrap as _document_bootstrap_impl
    return _document_bootstrap_impl(rows, b=b, seed=seed)


def robustness(rows: list[dict]) -> dict:
    """Corpus lift under corpus-subset and document-clustered CV variants."""
    from sklearn.model_selection import StratifiedKFold, GroupKFold

    def auc(rs, cols, groups=None):
        y = np.array([r["y"] for r in rs])
        idx = np.arange(len(rs))
        split = (GroupKFold(5).split(idx, y, groups) if groups is not None
                 else StratifiedKFold(5, shuffle=True, random_state=0).split(idx, y))
        return round(_foldwise(rs, cols, y, split)[1], 3)

    def cell(rs, cols, groups=None):
        y = np.array([r["y"] for r in rs])
        idx = np.arange(len(rs))
        split = (GroupKFold(5).split(idx, y, groups) if groups is not None
                 else StratifiedKFold(5, shuffle=True, random_state=0).split(idx, y))
        return _foldwise(rs, cols, y, split, per_fold=True)["auc"]

    def lift(rs, groups=None):
        """Paired cues -> cues+corpus comparison evaluated on one set of folds."""
        a = cell(rs, SEM_COLS, groups)
        b = cell(rs, SEM_COLS + ["corpus"], groups)
        return {"semantics_auc": _pm(a), "semantics_plus_corpus_auc": _pm(b),
                "corpus_lift_paired": _pm(b - a)}

    big = [r for r in rows if r["corpus"] in ("aimed", "bioinfer")]
    docs = [r["doc"] for r in big]
    all_docs = [r["doc"] for r in rows]
    return {
        "corpus_only_baseline_auc": _pm(cell(rows, ["corpus"])),
        "full_sample": lift(rows),
        "full_sample_document_clustered": lift(rows, groups=all_docs),
        "big_corpora_only(aimed+bioinfer)": lift(big),
        "document_clustered_cv(aimed+bioinfer)": lift(big, groups=docs),
        # legacy flat keys, kept so the figure script and number gate keep resolving
        "semantics_auc": auc(rows, SEM_COLS),
        "semantics_plus_corpus_auc": auc(rows, SEM_COLS + ["corpus"]),
    }


def matched_stratum(rows: list[dict], min_rows: int = 5, big: float = 0.30) -> dict:
    strat: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        s = (r["pred"], r["construction"], r["assertion"])
        strat[s][r["corpus"]][0] += r["y"]
        strat[s][r["corpus"]][1] += 1
    gaps, examples = [], []
    for s, bycorp in strat.items():
        present = [(c, v[0] / v[1], v[1]) for c, v in bycorp.items() if v[1] >= min_rows]
        if len(present) >= 2:
            gap = max(r for _, r, _ in present) - min(r for _, r, _ in present)
            gaps.append(gap)
            if gap >= big and max(n for *_, n in present) >= 20:
                examples.append({
                    "stratum": "|".join(s),
                    "rates": {c: round(rt, 2) for c, rt, _ in present},
                    # Raw (positive, total) per corpus for exact fractions in the text.
                    "counts": {c: [round(rt * n), n] for c, rt, n in present},
                })
    g = np.array(gaps)
    return {
        "testable_strata": len(gaps),
        "mean_corpus_gap": round(float(g.mean()), 3),
        "median_corpus_gap": round(float(np.median(g)), 3),
        "frac_gap_ge_0.30": round(float((g >= big).mean()), 3),
        "n_strata_full_agreement": int((g == 0).sum()),
        "examples": sorted(examples, key=lambda e: -max(e["rates"].values()))[:8],
    }


def carrier_robustness(rows: list[dict], rows_match: list[dict] | None = None) -> dict:
    """Corpus lift without carrier features, with carrier word only, or class only."""
    from sklearn.model_selection import StratifiedKFold

    y = np.array([r["y"] for r in rows])
    idx = np.arange(len(rows))

    def lift(cols):
        s = StratifiedKFold(5, shuffle=True, random_state=0)
        a1 = _foldwise(rows, cols, y, s.split(idx, y))[1]
        a2 = _foldwise(rows, cols + ["corpus"], y, s.split(idx, y))[1]
        return round(a2 - a1, 4)

    carrier_cols = ["carrier_any", "carrier_roles", "carrier_classes", "carrier_words"]
    noncarr = [c for c in SEM_COLS if c not in carrier_cols]

    def matched_gap(extra: tuple[str, ...] = (), min_rows: int = 5):
        """Mean per-corpus positive-rate spread over strata in >= 2 corpora."""
        strat: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for r in (rows_match if rows_match is not None else rows):
            key = (r["pred"], r["construction"], r["assertion"]) + tuple(r[f] for f in extra)
            strat[key][r["corpus"]][0] += r["y"]
            strat[key][r["corpus"]][1] += 1
        gaps = []
        for _, bycorp in strat.items():
            present = [(c, v[0] / v[1], v[1]) for c, v in bycorp.items() if v[1] >= min_rows]
            if len(present) >= 2:
                gaps.append(max(r for _, r, _ in present) - min(r for _, r, _ in present))
        return {"strata": len(gaps), "mean_gap": round(float(np.mean(gaps)), 3)}

    n_carrier = sum(1 for r in rows if r["carrier_any"] == "yes")
    n_other = sum(1 for r in rows if r["carrier_any"] == "yes"
                  and r["carrier_classes"] == "other_carrier")
    return {
        "oof_lift_full_semantics": lift(SEM_COLS),
        "oof_lift_drop_carrier": lift(noncarr),
        "oof_lift_carrier_words_only": lift(noncarr + ["carrier_words"]),
        "oof_lift_carrier_classes_only": lift(noncarr + ["carrier_classes"]),
        "matched_stratum_no_carrier": matched_gap(),
        "matched_stratum_plus_carrier_class": matched_gap(("carrier_classes",)),
        "matched_stratum_plus_carrier_word": matched_gap(("carrier_words",)),
        "matched_stratum_plus_carrier_word_roles": matched_gap(("carrier_words", "srole", "trole")),
        "codebook_coverage": {
            "carrier_bearing_rows": n_carrier,
            "classified_rows": n_carrier - n_other,
            "other_carrier_rows": n_other,
            "classified_share": round((n_carrier - n_other) / n_carrier, 3) if n_carrier else None,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    # Predictive models cap predicates; matching analyses must not.
    rows = _load(cap_predicates=True)
    rows_match = _load(cap_predicates=False)
    result = {
        "n_observations": len(rows),
        "positive_rate": round(float(np.mean([r["y"] for r in rows])), 3),
        "per_corpus_n": {c: sum(1 for r in rows if r["corpus"] == c) for c in CORPORA},
        "per_corpus_positive_rate": {
            c: round(float(np.mean([r["y"] for r in rows if r["corpus"] == c])), 4)
            for c in CORPORA},
        "out_of_fold": out_of_fold(rows),
        "document_bootstrap": document_bootstrap(rows),
        "robustness": robustness(rows),
        "matched_stratum": matched_stratum(rows_match),
        "matched_stratum_legacy_capped_predicates": matched_stratum(rows),
        "carrier_robustness": carrier_robustness(rows, rows_match),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    oof = result["out_of_fold"]
    print(f"\nHELD-OUT: adding corpus improves AUC {oof['m1_semantics']['auc']} -> "
          f"{oof['m2_plus_corpus']['auc']} ({oof['corpus_auc_improvement']:+}); a spurious/overfit "
          "factor could not improve held-out prediction.")
    ms = result["matched_stratum"]
    print(f"MATCHED: {ms['testable_strata']} matched semantic strata across >=2 corpora; mean "
          f"per-corpus positive-rate gap {ms['mean_corpus_gap']}; "
          f"{ms['n_strata_full_agreement']} show full agreement.")
    db = result["document_bootstrap"]["within_corpus_lift"]
    print(f"DOCUMENT BOOTSTRAP: within-corpus lift on document-grouped folds "
          f"{db['mean']:+} [{db['ci_low_2.5']:+}, {db['ci_high_97.5']:+}] over "
          f"{db['n_replicates']} replicates -- supersedes the StratifiedKFold-based "
          f"corpus_auc_improvement_within_corpus_paired above for the manuscript's claim.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
