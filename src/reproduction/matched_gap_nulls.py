#!/usr/bin/env python
"""Estimate matched-construction gaps, clustered nulls and paired carrier controls."""
from __future__ import annotations

from src.reproduction.config import PAPER_DIR as EXPERIMENT_DIR

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.reproduction import residual_corpus as rpt

# Matching key; shape and case match the residual model's cues.
BASE_FIELDS = ("pred", "construction", "assertion", "shape", "case")

# The manuscript's carrier ladder. Keys are the labels used in the paper and the gate.
LEVELS: dict[str, tuple[str, ...]] = {
    "pattern_only": (),
    "plus_carrier_class": ("carrier_classes",),
    "plus_carrier_word": ("carrier_words",),
    "plus_carrier_word_roles": ("carrier_words", "srole", "trole"),
}


def _keys(row: dict, extra: tuple[str, ...]) -> tuple[tuple, tuple]:
    """Return (base_key, full_key) for one observation at a matching level."""
    base = tuple(row[f] for f in BASE_FIELDS)
    return base, base + tuple(row[f] for f in extra)


def _index(rows: list[dict], extra: tuple[str, ...], min_rows: int):
    """Return ``(strata, base_keys)`` for strata with ``min_rows`` in at least two corpora."""
    grouped: dict[tuple, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    base_of: dict[tuple, tuple] = {}
    for i, r in enumerate(rows):
        base, full = _keys(r, extra)
        base_of[full] = base
        grouped[full][r["corpus"]].append(i)

    strata = []
    base_keys = set()
    for full, bycorp in grouped.items():
        present = {c: idx for c, idx in bycorp.items() if len(idx) >= min_rows}
        if len(present) >= 2:
            strata.append((full, base_of[full], present))
            base_keys.add(base_of[full])
    return strata, base_keys


def _mean_gap(strata, y: np.ndarray, restrict: set | None = None) -> tuple[float, int]:
    """Mean over strata of (max - min) per-corpus positive rate."""
    gaps = []
    for _full, base, present in strata:
        if restrict is not None and base not in restrict:
            continue
        rates = [float(y[idx].mean()) for idx in present.values()]
        gaps.append(max(rates) - min(rates))
    if not gaps:
        return float("nan"), 0
    return float(np.mean(gaps)), len(gaps)


def _paired_by_base(strata, y: np.ndarray, common_base: set) -> dict[tuple, float]:
    """Average sub-strata so each level has one value per shared base key (paired ladder)."""
    acc: dict[tuple, list[float]] = defaultdict(list)
    for _full, base, present in strata:
        if base not in common_base:
            continue
        rates = [float(y[idx].mean()) for idx in present.values()]
        acc[base].append(max(rates) - min(rates))
    return {b: float(np.mean(v)) for b, v in acc.items()}


def _null_b(strata, y, rng, iters, restrict=None):
    """Permute labels within each stratum across the present corpora's observations."""
    # Precompute, per stratum, the flat index array and the corpus segment boundaries.
    packed = []
    for _full, base, present in strata:
        if restrict is not None and base not in restrict:
            continue
        idx = np.concatenate([np.asarray(v) for v in present.values()])
        sizes = np.array([len(v) for v in present.values()])
        packed.append((idx, np.cumsum(sizes)[:-1]))
    if not packed:
        return np.array([])

    out = np.empty(iters)
    for it in range(iters):
        gaps = []
        for idx, cuts in packed:
            vals = y[idx].copy()
            rng.shuffle(vals)
            rates = [seg.mean() for seg in np.split(vals, cuts)]
            gaps.append(max(rates) - min(rates))
        out[it] = float(np.mean(gaps))
    return out


def _null_a(rows, strata, y, rng, iters, restrict=None, C=10.0):
    """Parametric bootstrap under an additive corpus + stratum model (no interaction)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder

    used = [(full, base, present) for full, base, present in strata
            if restrict is None or base in restrict]
    if not used:
        return np.array([]), {}

    # Design matrix over exactly the observations that enter the statistic.
    member_idx, corpus_of, stratum_of = [], [], []
    for full, _base, present in used:
        for c, idx in present.items():
            member_idx.extend(idx)
            corpus_of.extend([c] * len(idx))
            stratum_of.extend([str(full)] * len(idx))
    member_idx = np.asarray(member_idx)

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X = enc.fit_transform(np.column_stack([corpus_of, stratum_of]))
    yy = y[member_idx]
    model = LogisticRegression(max_iter=5000, C=C).fit(X, yy)
    p_fit = model.predict_proba(X)[:, 1]

    # Precompute design-matrix positions per stratum.
    pos_in_member = {int(g): k for k, g in enumerate(member_idx)}
    packed = [[np.fromiter((pos_in_member[int(i)] for i in idx), dtype=int, count=len(idx))
               for idx in present.values()]
              for _full, _base, present in used]

    out = np.empty(iters)
    for it in range(iters):
        draw = (rng.random(len(p_fit)) < p_fit).astype(int)
        gaps = []
        for segs in packed:
            rates = [draw[s].mean() for s in segs]
            gaps.append(max(rates) - min(rates))
        out[it] = float(np.mean(gaps))
    fit_info = {
        "n_observations_in_model": int(len(yy)),
        "n_strata_in_model": len(used),
        "regularisation_C": C,
        "note": ("corpus intercepts are estimated on data that still contains the "
                 "interaction, so this null is conservative"),
    }
    return out, fit_info


def _cluster_bootstrap(rows, extra, min_rows, rng, boot):
    """Bootstrap the mean gap by resampling documents within each corpus."""
    by_doc: dict[str, list[int]] = defaultdict(list)
    corpus_of_doc: dict[str, str] = {}
    for i, r in enumerate(rows):
        by_doc[r["doc"]].append(i)
        corpus_of_doc[r["doc"]] = r["corpus"]
    docs_by_corpus: dict[str, list[str]] = defaultdict(list)
    for d, c in corpus_of_doc.items():
        docs_by_corpus[c].append(d)

    out, n_strata = [], []
    for _ in range(boot):
        resampled = []
        for c, docs in docs_by_corpus.items():
            pick = rng.integers(0, len(docs), size=len(docs))
            for d in pick:
                resampled.extend(rows[i] for i in by_doc[docs[d]])
        y = np.array([r["y"] for r in resampled])
        strata, _ = _index(resampled, extra, min_rows)
        gap, n = _mean_gap(strata, y)
        if n:
            out.append(gap)
            n_strata.append(n)
    return np.asarray(out), np.asarray(n_strata)


# Signed contrast d = rate(c2) - rate(c1); sd of d is the tested statistic.

def _pair_index(rows, extra, min_rows, pair):
    c1, c2 = pair
    grouped: dict[tuple, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        if r["corpus"] in pair:
            _base, full = _keys(r, extra)
            grouped[full][r["corpus"]].append(i)
    out = []
    for full, bycorp in grouped.items():
        a, b = bycorp.get(c1, []), bycorp.get(c2, [])
        if len(a) >= min_rows and len(b) >= min_rows:
            out.append((full, np.asarray(a), np.asarray(b)))
    return out


def _pair_stats(strata, y) -> tuple[float, float, int, float]:
    d = np.array([y[b].mean() - y[a].mean() for _f, a, b in strata])
    if d.size == 0:
        return float("nan"), float("nan"), 0, float("nan")
    sd = float(d.std(ddof=1)) if d.size > 1 else 0.0
    return float(d.mean()), sd, int(d.size), float(np.mean(np.abs(d)))


def _pair_nulls(strata, y, rng, iters, C=10.0):
    """Null B and Null A for the signed pairwise contrast, returning (mean d, sd d) draws."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder

    # Null B: permute labels within each stratum across the two corpora's observations.
    b_mean, b_sd = np.empty(iters), np.empty(iters)
    packed = [(np.concatenate([a, b]), len(a)) for _f, a, b in strata]
    for it in range(iters):
        d = np.empty(len(packed))
        for k, (idx, na) in enumerate(packed):
            vals = y[idx].copy()
            rng.shuffle(vals)
            d[k] = vals[na:].mean() - vals[:na].mean()
        b_mean[it], b_sd[it] = d.mean(), d.std(ddof=1)

    # Null A: additive corpus + stratum logit on exactly the pair's observations.
    member, corpus_of, stratum_of = [], [], []
    for full, a, b in strata:
        for idx, tag in ((a, "c1"), (b, "c2")):
            member.extend(idx.tolist())
            corpus_of.extend([tag] * len(idx))
            stratum_of.extend([str(full)] * len(idx))
    member = np.asarray(member)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X = enc.fit_transform(np.column_stack([corpus_of, stratum_of]))
    model = LogisticRegression(max_iter=5000, C=C).fit(X, y[member])
    p_fit = model.predict_proba(X)[:, 1]
    pos = {int(g): k for k, g in enumerate(member)}
    seg = [(np.fromiter((pos[int(i)] for i in a), dtype=int, count=len(a)),
            np.fromiter((pos[int(i)] for i in b), dtype=int, count=len(b)))
           for _f, a, b in strata]

    a_mean, a_sd = np.empty(iters), np.empty(iters)
    for it in range(iters):
        draw = (rng.random(len(p_fit)) < p_fit).astype(int)
        d = np.array([draw[sb].mean() - draw[sa].mean() for sa, sb in seg])
        a_mean[it], a_sd[it] = d.mean(), d.std(ddof=1)
    return (b_mean, b_sd), (a_mean, a_sd)


def residual_document_icc(rows, y) -> tuple[float, float]:
    """Return ``(icc, sigma)``: residual document ICC and the matching logit-scale SD."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder

    strat = [str(tuple(r[f] for f in BASE_FIELDS)) for r in rows]
    X = OneHotEncoder(handle_unknown="ignore", sparse_output=True).fit_transform(
        np.column_stack([[r["corpus"] for r in rows], strat]))
    p = LogisticRegression(max_iter=5000, C=1e6).fit(X, y).predict_proba(X)[:, 1]
    resid = (y - p) / np.sqrt(np.clip(p * (1 - p), 1e-9, None))
    groups = defaultdict(list)
    for r, e in zip(rows, resid):
        groups[r["doc"]].append(e)
    g = [np.asarray(v) for v in groups.values() if len(v) > 1]
    if not g:
        return 0.0, 0.0
    k, n = len(g), sum(len(x) for x in g)
    gm = resid.mean()
    msb = sum(len(x) * (x.mean() - gm) ** 2 for x in g) / (k - 1)
    msw = sum(((x - x.mean()) ** 2).sum() for x in g) / (n - k)
    n0 = (n - sum(len(x) ** 2 for x in g) / n) / (k - 1)
    icc = float(max(0.0, (msb - msw) / (msb + (n0 - 1) * msw)))
    sigma = float(np.sqrt(icc / (1 - icc) * np.pi ** 2 / 3)) if icc < 1 else 0.0
    return icc, sigma


def _pair_null_a_clustered(rows, y, extra, min_rows, pair, rng, iters, C=1e6, sigma=0.0):
    """Document-clustered Null A: resample documents within corpus, refit, simulate labels.

    ``C`` is effectively unpenalized so corpus offsets are not shrunk.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder

    by_doc: dict[str, list[int]] = defaultdict(list)
    corpus_of_doc: dict[str, str] = {}
    for i, r in enumerate(rows):
        if r["corpus"] in pair:
            by_doc[r["doc"]].append(i)
            corpus_of_doc[r["doc"]] = r["corpus"]
    docs_by_corpus: dict[str, list[str]] = defaultdict(list)
    for d, c in corpus_of_doc.items():
        docs_by_corpus[c].append(d)

    c1, c2 = pair
    out_sd, out_mean, n_ok = [], [], 0
    for _ in range(iters):
        res = []
        for c, docs in docs_by_corpus.items():
            for d in rng.integers(0, len(docs), size=len(docs)):
                res.extend(rows[i] for i in by_doc[docs[d]])
        yy = np.array([r["y"] for r in res])
        st = _pair_index(res, extra, min_rows, pair)
        if len(st) < 2:
            continue
        member, corpus_of, stratum_of = [], [], []
        for full, a, b in st:
            for idx, tag in ((a, c1), (b, c2)):
                member.extend(idx.tolist())
                corpus_of.extend([tag] * len(idx))
                stratum_of.extend([str(full)] * len(idx))
        member = np.asarray(member)
        X = OneHotEncoder(handle_unknown="ignore", sparse_output=True).fit_transform(
            np.column_stack([corpus_of, stratum_of]))
        ytr = yy[member]
        if len(set(ytr.tolist())) < 2:
            continue
        p_fit = LogisticRegression(max_iter=5000, C=C).fit(X, ytr).predict_proba(X)[:, 1]
        pos = {int(g): k for k, g in enumerate(member)}
        if sigma > 0:
            # Document random intercept (logit scale) carries intra-document correlation.
            docs_m = [res[int(i)]["doc"] for i in member]
            # sorted() keeps the draw order reproducible across processes.
            u = {d: rng.normal(0.0, sigma) for d in sorted(set(docs_m))}
            lo = np.log(np.clip(p_fit, 1e-9, 1 - 1e-9) / np.clip(1 - p_fit, 1e-9, 1))
            lo = lo + np.array([u[d] for d in docs_m])
            p_fit = 1.0 / (1.0 + np.exp(-lo))
        draw = (rng.random(len(p_fit)) < p_fit).astype(int)
        d = np.array([
            draw[np.fromiter((pos[int(i)] for i in b), dtype=int, count=len(b))].mean()
            - draw[np.fromiter((pos[int(i)] for i in a), dtype=int, count=len(a))].mean()
            for _f, a, b in st])
        out_sd.append(float(d.std(ddof=1)))
        out_mean.append(float(d.mean()))
        n_ok += 1
    info = {
        "regularisation_C": C,
        "unpenalized": C >= 1e5,
        "refit_per_draw": True,
        "documents_resampled_within_corpus": True,
        "document_random_intercept_sigma": sigma,
        "n_usable_resamples": n_ok,
    }
    return np.asarray(out_mean), np.asarray(out_sd), info


def _summary(observed: float, null: np.ndarray) -> dict:
    if null.size == 0:
        return {}
    exceed = int(np.sum(null >= observed))
    return {
        "null_mean": round(float(null.mean()), 4),
        "null_sd": round(float(null.std(ddof=1)), 4),
        "null_p95": round(float(np.percentile(null, 95)), 4),
        "excess_over_null": round(float(observed - null.mean()), 4),
        # One-sided p with +1 correction; report the exceedance count too.
        "n_exceedances": exceed,
        "n_draws": int(null.size),
        "p_value": round(float((exceed + 1) / (null.size + 1)), 5),
        "at_monte_carlo_floor": exceed == 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=10000, help="permutation / simulation draws")
    ap.add_argument("--boot", type=int, default=2000, help="document bootstrap resamples")
    ap.add_argument("--clustered-iters", type=int, default=2000,
                    help="draws for the document-clustered Null A (refits per draw)")
    ap.add_argument("--min-rows", type=int, default=5, help="per-corpus support to be present")
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--pair", default="aimed,bioinfer",
                    help="ordered corpus pair for the signed contrast")
    ap.add_argument("--cap-predicates", action="store_true",
                    help="reproduce the submitted analysis, which collapsed all predicates "
                         "outside the top 25 into OTHER; invalid for exact matching and off "
                         "by default")
    ap.add_argument("--out", type=Path,
                    default=EXPERIMENT_DIR / 'results/matched-gap-nulls.json')
    args = ap.parse_args()

    rows = rpt._load(cap_predicates=args.cap_predicates)
    y = np.array([r["y"] for r in rows])
    n_other = sum(1 for r in rows if r["pred"] == "OTHER")
    print(f"loaded {len(rows)} observations, positive rate {y.mean():.3f}, "
          f"predicate cap={'on' if args.cap_predicates else 'off'}, OTHER rows={n_other}")

    # Base-key sets per level, so the ladder can also be reported paired.
    per_level_base: dict[str, set] = {}
    for name, extra in LEVELS.items():
        _, bk = _index(rows, extra, args.min_rows)
        per_level_base[name] = bk
    common_base = set.intersection(*per_level_base.values())
    print(f"common base strata across all four matching levels: {len(common_base)}")

    doc_icc, doc_sigma = residual_document_icc(rows, y)
    print(f"residual document ICC={doc_icc:.4f} -> random-intercept sigma={doc_sigma:.3f}")

    results: dict = {
        "predicate_cap_applied": args.cap_predicates,
        "n_other_predicate_rows": n_other,
        "corpus_pair": args.pair,
        "n_observations": len(rows),
        "positive_rate": round(float(y.mean()), 4),
        "per_corpus_positive_rate": {
            c: round(float(np.mean([r["y"] for r in rows if r["corpus"] == c])), 4)
            for c in sorted({r["corpus"] for r in rows})
        },
        "min_rows_per_corpus": args.min_rows,
        "iters": args.iters,
        "boot": args.boot,
        "seed": args.seed,
        "n_common_base_strata": len(common_base),
        "residual_document_icc": round(doc_icc, 4),
        "document_random_intercept_sigma": round(doc_sigma, 4),
        "levels": {},
    }

    paired_by_level: dict[str, dict[tuple, float]] = {}

    for name, extra in LEVELS.items():
        strata, _ = _index(rows, extra, args.min_rows)
        obs_all, n_all = _mean_gap(strata, y)
        paired = _paired_by_base(strata, y, common_base)
        paired_by_level[name] = paired
        obs_com, n_com = float(np.mean(list(paired.values()))), len(paired)

        rng = np.random.default_rng(args.seed)
        nb = _null_b(strata, y, rng, args.iters)
        na, fit_info = _null_a(rows, strata, y, rng, args.iters)
        boot_vals, boot_n = _cluster_bootstrap(rows, extra, args.min_rows, rng, args.boot)

        entry = {
            "observed_mean_gap": round(obs_all, 4),
            "n_strata": n_all,
            "common_base_restricted": {
                "observed_mean_gap": round(obs_com, 4),
                "n_strata": n_com,
            },
            "null_b_label_permutation": _summary(obs_all, nb),
            "null_a_additive_offset": {**_summary(obs_all, na), **fit_info},
            "document_cluster_bootstrap": {
                "mean": round(float(boot_vals.mean()), 4),
                "ci_low": round(float(np.percentile(boot_vals, 2.5)), 4),
                "ci_high": round(float(np.percentile(boot_vals, 97.5)), 4),
                "median_n_strata": int(np.median(boot_n)),
                "n_resamples": int(boot_vals.size),
            },
        }
        # Signed contrast for the fixed corpus pair -- the primary inferential statistic.
        pair = tuple(args.pair.split(","))
        pstrata = _pair_index(rows, extra, args.min_rows, pair)
        if len(pstrata) >= 2:
            pm, psd, pn, pabs = _pair_stats(pstrata, y)
            (nb_m, nb_s), (na_m, na_s) = _pair_nulls(pstrata, y, rng, args.iters)
            cl_m, cl_s, cl_info = _pair_null_a_clustered(
                rows, y, extra, args.min_rows, pair, rng, args.clustered_iters,
                sigma=doc_sigma)
            # Sensitivity of the iid null to ridge strength.
            sens = {}
            for c_val in (1.0, 10.0, 1e6):
                _, (_, s_) = _pair_nulls(pstrata, y,
                                         np.random.default_rng(args.seed + 7), 400, C=c_val)
                sens[f"C={c_val:g}"] = {"null_sd_mean": round(float(s_.mean()), 4),
                                        "p_value": _summary(psd, s_)["p_value"]}
            entry["signed_pair_contrast"] = {
                "pair": f"{pair[1]} minus {pair[0]}",
                "n_strata": pn,
                "mean_contrast": round(pm, 4),
                "mean_abs_contrast": round(pabs, 4),
                "sd_contrast": round(psd, 4),
                "mean_vs_null_a_iid": _summary(pm, na_m),
                "sd_vs_null_a_iid": _summary(psd, na_s),
                "sd_vs_null_a_document_clustered": {**_summary(psd, cl_s), **cl_info},
                "mean_vs_null_a_document_clustered": _summary(pm, cl_m),
                "sd_vs_null_b_descriptive": _summary(psd, nb_s),
                "null_a_penalty_sensitivity": sens,
                "note": ("mean_contrast under an additive model is just the global offset, so "
                         "it is not evidence of construction-specific treatment; the "
                         "interaction evidence is sd_contrast exceeding the DOCUMENT-CLUSTERED "
                         "Null A. The iid Null A is anticonservative under intra-document "
                         "correlation and is reported only as a sensitivity."),
            }

        results["levels"][name] = entry

        print(f"\n[{name}] strata={n_all} observed={obs_all:.3f}")
        print(f"  null B (label permutation)   mean={entry['null_b_label_permutation']['null_mean']:.3f} "
              f"p={entry['null_b_label_permutation']['p_value']}")
        print(f"  null A (additive offset)     mean={entry['null_a_additive_offset']['null_mean']:.3f} "
              f"p={entry['null_a_additive_offset']['p_value']}")
        print(f"  doc cluster bootstrap        {entry['document_cluster_bootstrap']['ci_low']:.3f}"
              f"-{entry['document_cluster_bootstrap']['ci_high']:.3f}")
        print(f"  paired (common base keys)    n={n_com} gap={obs_com:.3f}")
        sp = entry.get("signed_pair_contrast")
        if sp:
            print(f"  signed {sp['pair']:26} n={sp['n_strata']} mean={sp['mean_contrast']:+.3f} "
                  f"sd={sp['sd_contrast']:.3f}")
            t = sp["sd_vs_null_a_iid"]
            print(f"    interaction sd vs nullA iid       null={t['null_mean']:.3f} "
                  f"p={t['p_value']} ({t['n_exceedances']}/{t['n_draws']})")
            t = sp["sd_vs_null_a_document_clustered"]
            print(f"    interaction sd vs nullA CLUSTERED null={t['null_mean']:.3f} "
                  f"p={t['p_value']} ({t['n_exceedances']}/{t['n_draws']})  <-- primary")

    # Paired ladder: one value per shared base key at every level.
    order = list(LEVELS)
    keys = sorted(common_base)
    mat = np.array([[paired_by_level[lv][k] for k in keys] for lv in order])
    rng = np.random.default_rng(args.seed + 1)
    boot_keys = rng.integers(0, len(keys), size=(args.boot, len(keys)))
    ladder = {}
    for i, lv in enumerate(order):
        draws = mat[i][boot_keys].mean(axis=1)
        ladder[lv] = {
            "mean_gap": round(float(mat[i].mean()), 4),
            "ci_low": round(float(np.percentile(draws, 2.5)), 4),
            "ci_high": round(float(np.percentile(draws, 97.5)), 4),
        }
    deltas = {}
    for i in range(1, len(order)):
        d = mat[i] - mat[0]
        draws = d[boot_keys].mean(axis=1)
        deltas[f"{order[i]}_minus_{order[0]}"] = {
            "mean_delta": round(float(d.mean()), 4),
            "ci_low": round(float(np.percentile(draws, 2.5)), 4),
            "ci_high": round(float(np.percentile(draws, 97.5)), 4),
        }
    results["paired_ladder_common_base"] = {
        "n_base_keys": len(keys),
        "note": ("each level contributes one value per base key, averaged over that key's "
                 "carrier sub-strata, so the four levels are compared on identical units; "
                 "intervals are a bootstrap over base keys"),
        "levels": ladder,
        "deltas_vs_pattern_only": deltas,
    }
    print("\n[paired ladder over %d common base keys]" % len(keys))
    for lv in order:
        e = ladder[lv]
        print(f"  {lv:26} {e['mean_gap']:.3f} ({e['ci_low']:.3f}-{e['ci_high']:.3f})")
    for k, e in deltas.items():
        print(f"  delta {k:44} {e['mean_delta']:+.3f} ({e['ci_low']:+.3f}, {e['ci_high']:+.3f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
