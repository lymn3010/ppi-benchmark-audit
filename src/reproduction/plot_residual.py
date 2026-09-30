#!/usr/bin/env python3
"""Plot corpus-label prediction, prevalence controls and descriptive contrasts."""
from __future__ import annotations

from src.reproduction.config import PAPER_DIR as EXPERIMENT_DIR

import json
import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
SRC = EXPERIMENT_DIR / 'results/residual-corpus.json'
OUT = EXPERIMENT_DIR / 'figures'
CORPUS_ORDER = ["aimed", "bioinfer", "hprd50", "iepa", "lll"]
CL = {"aimed": "AIMed", "bioinfer": "BioInfer", "hprd50": "HPRD50", "iepa": "IEPA", "lll": "LLL"}
CCOL = {"aimed": "#4C72B0", "bioinfer": "#DD8452", "hprd50": "#55A868",
        "iepa": "#C44E52", "lll": "#8172B3"}
# Example strata; keep in sync with the supplement's table.
SHOW = ["complex|compound_state|asserted", "inhibit|verbal|asserted",
        "bind|nominalized|asserted", "link|verbal|asserted"]
# Reader-facing names; internal keys must not reach the figure.
SHOW_LABEL = {
    "complex|compound_state|asserted": "complex\nmembership",
    "inhibit|verbal|asserted": "inhibits\n(clause)",
    "bind|nominalized|asserted": "binding\n(nominal)",
    "link|verbal|asserted": "links\n(clause)",
}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.25, "axes.titlesize": 9.0,
        "axes.labelsize": 8.35, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _m(x):
    """Read a value that may be a scalar or a {mean, se, ...} block."""
    return float(x["mean"] if isinstance(x, dict) else x)


def _se(x):
    return float(x.get("se", 0.0)) if isinstance(x, dict) else 0.0


def main() -> None:
    d = json.load(SRC.open())
    r = d["robustness"]
    oof = d["out_of_fold"]
    dbo = d["document_bootstrap"]["within_corpus_lift"]
    within_corpus_block = {
        "mean": dbo["mean"], "ci_low": dbo["ci_low_2.5"], "ci_high": dbo["ci_high_97.5"],
    }
    ex = {e["stratum"]: e for e in d["matched_stratum"]["examples"]}
    missing = [s for s in SHOW if s not in ex]
    if missing:
        raise SystemExit(
            f"SHOW references strata absent from matched_stratum.examples: {missing}. "
            f"Available: {sorted(ex)}. Update SHOW in this script."
        )
    _style()
    fig, (axA, axL, axB) = plt.subplots(
        1,
        3,
        figsize=(7.2, 3.05),
        gridspec_kw={"width_ratios": [0.9, 1.0, 1.85], "wspace": 0.52},
        constrained_layout=False,
    )

    labels = ["corpus\nonly", "cues", "cues +\ncorpus"]
    src = [r["corpus_only_baseline_auc"], oof["m1_semantics"]["auc"], oof["m2_plus_corpus"]["auc"]]
    vals, errs = [_m(v) for v in src], [1.96 * _se(v) for v in src]
    cols = ["#BBBBBB", "#7BA7D7", "#1F4E79"]
    axA.bar(range(3), vals, yerr=errs, color=cols, width=0.64, edgecolor="white",
            error_kw={"ecolor": "#333333", "elinewidth": 0.9, "capsize": 2.5})
    for i, v in enumerate(vals):
        axA.text(i, v + errs[i] + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=8.2)
    axA.set_xticks(range(3), labels, fontsize=7.6)
    axA.set_ylim(0.5, 0.89)
    axA.set_ylabel("Held-out AUC")
    axA.set_title("Corpus predicts the\nreleased label", pad=10)

    # Within-corpus bar uses the document-cluster bootstrap.
    dec = [("corpus\nindicator", oof["corpus_auc_improvement_paired"], "#1F4E79"),
           ("base rate\nonly", oof["base_rate_only_auc_improvement_paired"], "#8FA9C4"),
           ("within\ncorpus", within_corpus_block, "#C44E52")]
    for i, (lab, blk, col) in enumerate(dec):
        m, lo, hi = _m(blk), blk["ci_low"], blk["ci_high"]
        axL.bar(i, m, color=col, width=0.62, edgecolor="white")
        axL.errorbar(i, m, yerr=[[m - lo], [hi - m]], ecolor="#333333", elinewidth=0.9,
                     capsize=2.5, fmt="none")
        axL.text(i, hi + 0.003, f"{m:+.3f}", ha="center", va="bottom", fontsize=8.0)
    axL.set_xticks(range(3), [x[0] for x in dec], fontsize=7.6)
    axL.axhline(0, color="#888888", linewidth=0.7)
    axL.set_ylim(-0.012, 0.088)
    axL.set_ylabel("AUC gain over cues")
    axL.set_title("...but almost all of it\nis the corpus base rate", pad=10)

    n = len(SHOW)
    group_x = [i * 0.84 for i in range(n)]
    w = 0.155
    for si, strat in enumerate(SHOW):
        rates = ex[strat]["rates"]
        counts = ex[strat].get("counts", {})
        present = [c for c in CORPUS_ORDER if c in rates]
        for ci, c in enumerate(present):
            off = (ci - (len(present) - 1) / 2) * w
            axB.bar(
                group_x[si] + off,
                rates[c],
                width=w,
                color=CCOL[c],
                edgecolor="white",
                linewidth=0.4,
            )
            # Show support n on every bar.
            if c in counts:
                axB.text(group_x[si] + off, rates[c] + 0.015, f"{counts[c][1]}",
                         ha="center", va="bottom", fontsize=5.6, color="#444444",
                         rotation=90)
    axB.set_xticks(
        group_x,
        [SHOW_LABEL[s] for s in SHOW],
        fontsize=7.6,
    )
    axB.set_xlim(group_x[0] - 0.48, group_x[-1] + 0.48)
    axB.yaxis.set_major_formatter(PercentFormatter(1.0))
    axB.set_ylim(0, 1.12)
    axB.set_ylabel("Released-positive rate")
    axB.yaxis.labelpad = 8
    axB.set_title("Same visible construction, different labeling\n"
                  "(bar labels: observations per corpus)", pad=8, fontsize=8.2)
    handles = [Patch(color=CCOL[c], label=CL[c]) for c in CORPUS_ORDER]
    axB.legend(
        handles=handles,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.21),
        columnspacing=0.75,
        handlelength=0.95,
        fontsize=7.8,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.78, bottom=0.30)
    for fmt in ("pdf", "png"):
        fig.savefig(OUT / f"fig4_residual_corpus_effect.{fmt}", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("wrote", OUT / "fig4_residual_corpus_effect.pdf")


if __name__ == "__main__":
    main()
