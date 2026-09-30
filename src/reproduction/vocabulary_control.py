#!/usr/bin/env python
"""Control for sentence vocabulary (fold-local, protein markers excluded)."""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.reproduction.config import CORPORA, stats_dir  # noqa: E402

from src.reproduction import residual_corpus as rpt

_PROT = re.compile(r"PROTEIN\d+")
_NAME = {"aimed": "AIMed", "bioinfer": "BioInfer", "hprd50": "HPRD50", "iepa": "IEPA", "lll": "LLL"}


def _sentence_index() -> dict:
    sent: dict = {}
    for c in CORPORA:
        for cand in {_NAME[c], c, c.upper()}:
            hits = glob.glob(str(ROOT / f"data/datasets/{cand}/full.json"))
            if hits:
                data = json.load(open(hits[0]))["data"]
                for sid, rec in data.items():
                    sent[(c, sid)] = rec.get("sentence", "")
                break
    return sent


def _load_rows_with_text() -> list[dict]:
    """Re-materialize the residual-test rows (same iteration + predicate-cap logic as
    rpt._load) but also attach the masked sentence text for the lexical-content control."""
    import csv
    sent = _sentence_index()
    rows2: list[dict] = []
    for c in CORPORA:
        f = stats_dir(c) / "pattern_observations.csv"
        carriers = rpt._carrier_index(c)
        for row in csv.DictReader(f.open()):
            sid = row.get("sentence_id", "")
            base = {
                "corpus": c,
                "y": 1 if str(row.get("gold_label", "")).strip() in rpt.POS else 0,
                "assertion": row.get("assertion_status", "") or "na",
                "construction": row.get("construction", "") or "na",
                "shape": row.get("shape", "") or "na",
                "case": row.get("case_marker", "") or "na",
                "srole": row.get("source_role", "") or "na",
                "trole": row.get("target_role", "") or "na",
                "pred": row.get("predicate_lemma", "") or "na",
                "doc": ".".join(sid.split(".")[:2]) if sid else c,  # AIMed.d115.s970 -> AIMed.d115
                "text": _PROT.sub(" ", sent.get((c, sid), "")),
            }
            carrier = carriers.get((sid, row.get("canonical_pattern_key", "")), {
                "carrier_any": "no", "carrier_roles": "none",
                "carrier_classes": "none", "carrier_words": "none"})
            base.update(carrier)
            rows2.append(base)
    from collections import Counter
    top = {p for p, _ in Counter(r["pred"] for r in rows2).most_common(25)}
    for r in rows2:
        r["pred"] = r["pred"] if r["pred"] in top else "OTHER"
    return rows2


def _held_out_auc(rows, cue_cols, y, splitter, *, with_text=False, with_corpus=False,
                  max_features=800, per_fold=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import roc_auc_score
    from scipy.sparse import hstack
    cols = list(cue_cols) + (["corpus"] if with_corpus else [])
    aucs = []
    for tr, te in splitter:
        if len(set(y[te])) < 2:
            continue
        enc = OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=True)
        xtr = enc.fit_transform([[rows[i][c] for c in cols] for i in tr])
        xte = enc.transform([[rows[i][c] for c in cols] for i in te])
        if with_text:
            vec = CountVectorizer(binary=True, min_df=10, max_features=max_features,
                                  stop_words="english")
            ttr = vec.fit_transform([rows[i]["text"] for i in tr])
            tte = vec.transform([rows[i]["text"] for i in te])
            xtr, xte = hstack([xtr, ttr]).tocsr(), hstack([xte, tte]).tocsr()
        m = LogisticRegression(max_iter=3000, C=1.0).fit(xtr, y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(xte)[:, 1]))
    return np.asarray(aucs) if per_fold else float(np.mean(aucs))


def _cell(rows, cue, y, split_factory, *, with_text, max_features=800):
    """One complete design cell: cues (optionally + text) with and without corpus, paired
    on identical folds, returning the lift with fold-level uncertainty."""
    a = _held_out_auc(rows, cue, y, split_factory(), with_text=with_text,
                      max_features=max_features, per_fold=True)
    b = _held_out_auc(rows, cue, y, split_factory(), with_text=with_text, with_corpus=True,
                      max_features=max_features, per_fold=True)
    return {"auc_without_corpus": rpt._pm(a), "auc_with_corpus": rpt._pm(b),
            "corpus_lift_paired": rpt._pm(b - a)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from sklearn.model_selection import StratifiedKFold
    rows = _load_rows_with_text()
    non_empty = sum(1 for r in rows if r["text"].strip())
    y = np.array([r["y"] for r in rows])
    idx = np.arange(len(rows))

    def split():
        return StratifiedKFold(5, shuffle=True, random_state=0).split(idx, y)

    cue = rpt.SEM_COLS
    auc_cue = _held_out_auc(rows, cue, y, split())
    auc_cue_corpus = _held_out_auc(rows, cue, y, split(), with_corpus=True)
    auc_content = _held_out_auc(rows, cue, y, split(), with_text=True)
    auc_content_corpus = _held_out_auc(rows, cue, y, split(), with_text=True, with_corpus=True)

    # Sensitivity: vocabulary cap sweep and document-clustered folds.
    feature_sweep = {}
    for mf in (400, 800, 1500):
        a0 = _held_out_auc(rows, cue, y, split(), with_text=True, max_features=mf)
        a1 = _held_out_auc(rows, cue, y, split(), with_text=True, with_corpus=True, max_features=mf)
        feature_sweep[str(mf)] = round(a1 - a0, 4)

    from sklearn.model_selection import GroupKFold
    big = [r for r in rows if r["corpus"] in ("aimed", "bioinfer")]
    gy = np.array([r["y"] for r in big])
    gidx = np.arange(len(big))
    groups = [r["doc"] for r in big]
    gsplit = list(GroupKFold(5).split(gidx, gy, groups))
    a0 = _held_out_auc(big, cue, gy, iter(gsplit), with_text=True)
    a1 = _held_out_auc(big, cue, gy, iter(gsplit), with_text=True, with_corpus=True)
    doc_content_lift = round(a1 - a0, 4)

    # Corpus subset x fold construction, with and without the vocabulary control.
    all_docs = [r["doc"] for r in rows]
    design = {}
    for subset_name, rs, ys, gs in (
        ("full_sample", rows, y, all_docs),
        ("two_largest(aimed+bioinfer)", big, gy, groups),
    ):
        aidx = np.arange(len(rs))
        for clustering, factory in (
            ("unclustered", lambda rs=rs, ys=ys, aidx=aidx: StratifiedKFold(
                5, shuffle=True, random_state=0).split(aidx, ys)),
            ("document_clustered", lambda rs=rs, ys=ys, aidx=aidx, gs=gs: GroupKFold(
                5).split(aidx, ys, gs)),
        ):
            for control, wt in (("cues_only", False), ("cues_plus_bag_of_words", True)):
                design[f"{subset_name}|{clustering}|{control}"] = _cell(
                    rs, cue, ys, factory, with_text=wt)

    result = {
        "complete_design_2x2x2": design,
        "n_observations": len(rows),
        "n_with_sentence_text": non_empty,
        "auc_cues": round(auc_cue, 4),
        "auc_cues_plus_corpus": round(auc_cue_corpus, 4),
        "corpus_lift_over_cues": round(auc_cue_corpus - auc_cue, 4),
        "auc_cues_plus_content": round(auc_content, 4),
        "auc_cues_plus_content_plus_corpus": round(auc_content_corpus, 4),
        "corpus_lift_over_cues_plus_content": round(auc_content_corpus - auc_content, 4),
        "content_control_feature_sweep_corpus_lift": feature_sweep,
        "content_control_doc_clustered_corpus_lift(aimed+bioinfer)": doc_content_lift,
        "interpretation": (
            "If corpus_lift_over_cues_plus_content is still clearly positive, the source "
            "corpus predicts the released label beyond both the recorded cues AND the "
            "sentence's lexical content (a subdomain/assay-vocabulary proxy), so the residual "
            "is not merely observable biological topic."
        ),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
