#!/usr/bin/env python
"""Document bootstrap of the corpus-effect AUC difference (no refit per replicate)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.reproduction.residual_corpus import SEM_COLS, _load  # noqa: E402


def _oof_predictions(rows: list[dict], cols: list[str], groups: list[str]) -> np.ndarray:
    """One document-grouped 5-fold CV, fold-local fit; returns held-out predicted
    probability per row, aligned with ``rows``."""
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder

    y = np.array([r["y"] for r in rows])
    idx = np.arange(len(rows))
    pred = np.full(len(rows), np.nan)
    for tr, te in GroupKFold(5).split(idx, y, groups):
        enc = OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=True)
        xtr = enc.fit_transform([[rows[i][c] for c in cols] for i in tr])
        xte = enc.transform([[rows[i][c] for c in cols] for i in te])
        m = LogisticRegression(max_iter=2000, C=1.0).fit(xtr, y[tr])
        pred[te] = m.predict_proba(xte)[:, 1]
    if np.isnan(pred).any():
        raise AssertionError("every row must receive exactly one held-out prediction")
    return pred


def _weighted_auc(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    mask = w > 0
    if mask.sum() == 0:
        return None
    yy, ww = y[mask], w[mask]
    if (yy * ww).sum() == 0 or ((1 - yy) * ww).sum() == 0:
        return None  # a degenerate replicate: only one class drawn
    return float(roc_auc_score(yy, p[mask], sample_weight=ww))


def bootstrap(rows: list[dict], *, b: int, seed: int) -> dict:
    corpora = [r["corpus"] for r in rows]
    docs = [r["doc"] for r in rows]
    y = np.array([r["y"] for r in rows])

    pred_m1 = _oof_predictions(rows, SEM_COLS, docs)
    pred_m2 = _oof_predictions(rows, SEM_COLS + ["corpus"], docs)

    doc_rows: dict[str, list[int]] = defaultdict(list)
    doc_corpus: dict[str, str] = {}
    for i, r in enumerate(rows):
        doc_rows[r["doc"]].append(i)
        doc_corpus[r["doc"]] = r["corpus"]
    docs_by_corpus: dict[str, list[str]] = defaultdict(list)
    for d, c in doc_corpus.items():
        docs_by_corpus[c].append(d)
    for c in docs_by_corpus:
        docs_by_corpus[c].sort()  # deterministic draw order for a fixed seed

    n_rows = len(rows)
    corpus_arr = np.array(corpora)
    rng = np.random.default_rng(seed)

    raw_lifts: list[float] = []
    within_lifts: list[float] = []
    per_corpus_lifts: dict[str, list[float]] = defaultdict(list)

    for _ in range(b):
        w = np.zeros(n_rows)
        for c, docs_c in docs_by_corpus.items():
            draw = rng.choice(docs_c, size=len(docs_c), replace=True)
            for doc, cnt in Counter(draw).items():
                for i in doc_rows[doc]:
                    w[i] = cnt

        raw1 = _weighted_auc(y, pred_m1, w)
        raw2 = _weighted_auc(y, pred_m2, w)
        if raw1 is not None and raw2 is not None:
            raw_lifts.append(raw2 - raw1)

        num, den = 0.0, 0.0
        for c in docs_by_corpus:
            sel = corpus_arr == c
            a1 = _weighted_auc(y[sel], pred_m1[sel], w[sel])
            a2 = _weighted_auc(y[sel], pred_m2[sel], w[sel])
            wsum = float(w[sel].sum())
            if a1 is None or a2 is None or wsum == 0:
                continue
            per_corpus_lifts[c].append(a2 - a1)
            num += (a2 - a1) * wsum
            den += wsum
        if den > 0:
            within_lifts.append(num / den)

    def _summary(vals: list[float]) -> dict:
        v = np.asarray(vals, dtype=float)
        return {
            "n_replicates": int(v.size),
            "mean": round(float(v.mean()), 4),
            "ci_low_2.5": round(float(np.percentile(v, 2.5)), 4),
            "ci_high_97.5": round(float(np.percentile(v, 97.5)), 4),
        }

    return {
        "method": "document cluster bootstrap over held-out predictions from one "
                  "document-grouped 5-fold CV; resampling is within corpus, preserving "
                  "each corpus's document count",
        "requested_replicates": b,
        "seed": seed,
        "n_documents_by_corpus": {c: len(v) for c, v in docs_by_corpus.items()},
        "raw_corpus_lift": _summary(raw_lifts),
        "within_corpus_lift": _summary(within_lifts),
        "within_corpus_lift_by_corpus": {c: _summary(v) for c, v in per_corpus_lifts.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = _load(cap_predicates=True)
    result = bootstrap(rows, b=args.b, seed=args.seed)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raw = result["raw_corpus_lift"]
    within = result["within_corpus_lift"]
    print(f"\nDOCUMENT BOOTSTRAP ({raw['n_replicates']} usable replicates of "
          f"{args.b} requested): raw lift {raw['mean']:+.3f} "
          f"[{raw['ci_low_2.5']:+.3f}, {raw['ci_high_97.5']:+.3f}]; "
          f"within-corpus lift {within['mean']:+.3f} "
          f"[{within['ci_low_2.5']:+.3f}, {within['ci_high_97.5']:+.3f}]")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
