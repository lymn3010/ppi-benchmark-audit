#!/usr/bin/env python3
"""Verify numerical claims and the paper figure against the released materials."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
from .config import MANIFEST, SOURCE_PAPER_DIR

PAPER_DIR = SOURCE_PAPER_DIR


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _val(x) -> float:
    """Return a scalar, or the mean from a {mean, se, ci_low, ci_high} block."""
    return float(x["mean"] if isinstance(x, dict) else x)


def _assert_close(name: str, got, expected: float, *, tol: float = 0.0006) -> None:
    got = _val(got)
    if not math.isclose(got, expected, abs_tol=tol):
        raise AssertionError(f"{name}: got {got}, expected {expected} (+/- {tol})")


def check_within() -> None:
    rows = _read_csv(PAPER_DIR / "results/within-corpus-summary.csv")
    canonical = {
        r["corpus"]: r for r in rows if r["dimension"] == "canonical_pattern"
    }
    _assert_close("AIMed canonical mixed share", float(canonical["aimed"]["mixed_firing_share"]), 0.8887)
    _assert_close("BioInfer canonical mixed share", float(canonical["bioinfer"]["mixed_firing_share"]), 0.841)
    isolated = sum(int(r["isolated_contrary_cells"]) for r in canonical.values())
    if isolated != 54:
        raise AssertionError(f"canonical isolated-contrary cells: got {isolated}, expected 54")
    # Pin each corpus so offsetting changes cannot keep the total unchanged.
    per_iso = {c: int(r["isolated_contrary_cells"]) for c, r in canonical.items()}
    expected_iso = {"aimed": 27, "bioinfer": 22, "hprd50": 2, "iepa": 3, "lll": 0}
    if per_iso != expected_iso:
        raise AssertionError(f"per-corpus isolated-contrary: got {per_iso}, expected {expected_iso}")
    # Cell profile: assessable = unanimous + isolated-contrary + split.
    assessable = sum(int(r["assessable_cells"]) for r in canonical.values())
    unanimous = sum(int(r["unanimous_cells"]) for r in canonical.values())
    split = sum(int(r["split_cells"]) for r in canonical.values())
    if (assessable, unanimous, split) != (169, 62, 53):
        raise AssertionError(
            f"within-corpus cell profile: got assessable={assessable}, unanimous={unanimous}, "
            f"split={split}; expected 169/62/53"
        )
    _assert_close("HPRD50 canonical mixed share", float(canonical["hprd50"]["mixed_firing_share"]), 0.1325)
    _assert_close("IEPA canonical mixed share", float(canonical["iepa"]["mixed_firing_share"]), 0.60)
    _assert_close("LLL canonical mixed share", float(canonical["lll"]["mixed_firing_share"]), 0.0)


def check_cross_corpus() -> None:
    rows = _read_csv(PAPER_DIR / "results/cross-corpus/fisher-tests.csv")
    sig = sum(float(r["q_value"]) < 0.05 for r in rows)
    if len(rows) != 266 or sig != 70:
        raise AssertionError(f"cross-corpus Fisher/BH: got {sig}/{len(rows)}, expected 70/266")




def check_coverage() -> None:
    path = PAPER_DIR / "results/coverage.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    total = data["total"]
    committed = total["clause_SVO"] + total["event_nominal"] + total["complex_state"]
    traceable = committed + total["held_evidence"] + total["lexpath"]
    if (committed, traceable, total["_gold_total"]) != (1683, 4131, 4187):
        raise AssertionError(
            "evidence tiers: got "
            f"committed={committed}, traceable={traceable}, total={total['_gold_total']}; "
            "expected committed=1683, traceable=4131, total=4187"
        )
    # Pin each tier so offsetting changes cannot keep the committed total unchanged.
    tiers = {k: total[k] for k in ("clause_SVO", "event_nominal", "complex_state",
                                   "held_evidence", "lexpath")}
    expected_tiers = {"clause_SVO": 1303, "event_nominal": 253, "complex_state": 127,
                      "held_evidence": 302, "lexpath": 2146}
    if tiers != expected_tiers:
        raise AssertionError(f"evidence-tier counts: got {tiers}, expected {expected_tiers}")
    expected_missing = {"aimed": 3, "bioinfer": 47, "hprd50": 0, "iepa": 5, "lll": 1}
    if {c: v["out_of_scope"] for c, v in data["per_corpus"].items()} != expected_missing:
        raise AssertionError("per-corpus untraced coverage changed")
    records = data.get("untraced_pairs", [])
    keys = {(r["corpus"], r["sentence_id"], tuple(r["pair"])) for r in records}
    if total["out_of_scope"] != 56 or len(keys) != 56 or len(records) != 56:
        raise AssertionError("coverage must identify all 56 untraced positive pairs")
    if {c: sum(r["corpus"] == c for r in records) for c in expected_missing} != expected_missing:
        raise AssertionError("coverage identifiers disagree with the per-corpus counts")
    # Per-corpus committed coverage range: committed pairs / released positives.
    shares = []
    per_committed = 0
    for corpus, v in data["per_corpus"].items():
        c = v["clause_SVO"] + v["event_nominal"] + v["complex_state"]
        per_committed += c
        if v["_gold_total"]:
            shares.append(round(100 * c / v["_gold_total"]))
    if per_committed != 1683:
        raise AssertionError(f"per-corpus committed sum: got {per_committed}, expected 1683")
    if (min(shares), max(shares)) != (33, 63):
        raise AssertionError(
            f"per-corpus committed coverage range: got {min(shares)}--{max(shares)}%, expected 33--63%"
        )
    # Held positives that rest only on an apposition.
    appositive = sum(v.get("_held_appositive", 0) for v in data["per_corpus"].values())
    if appositive != 130:
        raise AssertionError(f"apposition-held positives: got {appositive}, expected 130")


def _example_rates(data: dict, stratum: str) -> dict[str, float]:
    for row in data["matched_stratum"]["examples"]:
        if row["stratum"] == stratum:
            return row["rates"]
    raise AssertionError(f"residual matched-stratum example missing: {stratum}")


def _example_counts(data: dict, stratum: str) -> dict[str, list[int]]:
    for row in data["matched_stratum"]["examples"]:
        if row["stratum"] == stratum:
            return row["counts"]
    raise AssertionError(f"residual matched-stratum example missing: {stratum}")


def check_residual_policy() -> None:
    """Check the residual corpus-effect numbers stated in the manuscript."""
    path = PAPER_DIR / "results/residual-corpus.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["n_observations"] != 4184:
        raise AssertionError(f"residual n_observations: got {data['n_observations']}, expected 4184")

    oof = data["out_of_fold"]
    _assert_close("residual semantics-only AUC", oof["m1_semantics"]["auc"], 0.7376)
    _assert_close("residual semantics+corpus AUC", oof["m2_plus_corpus"]["auc"], 0.8036)
    _assert_close("residual corpus AUC improvement", oof["corpus_auc_improvement"], 0.066)

    # Prevalence controls: the raw corpus lift should be reproduced by the base rate alone.
    _assert_close("base-rate-only lift", oof["base_rate_only_auc_improvement_paired"], 0.0651)
    # Ungrouped-fold lift; not reported (documents can span train and test).
    _assert_close("within-corpus lift, StratifiedKFold (superseded, not cited in the paper)",
                  oof["corpus_auc_improvement_within_corpus_paired"], 0.0092)

    # Reported lift: document-cluster bootstrap on a document-grouped split.
    dbo = data["document_bootstrap"]["within_corpus_lift"]
    _assert_close("within-corpus lift, document-grouped bootstrap mean", dbo["mean"], 0.0021, tol=0.0015)
    if not (dbo["ci_low_2.5"] < 0 < dbo["ci_high_97.5"]):
        raise AssertionError(
            f"document-grouped within-corpus lift CI {dbo['ci_low_2.5']}..{dbo['ci_high_97.5']} "
            "no longer straddles zero; Section 5.1's 'no residual survives' claim would need rewriting"
        )
    raw = _val(oof["corpus_auc_improvement_paired"])
    if not _val(dbo["mean"]) < raw / 5:
        raise AssertionError(
            f"document-grouped within-corpus lift {dbo['mean']} is no longer far below the raw "
            f"lift {raw}; the base-rate interpretation in Section 5.1 would need rewriting")

    robust = data["robustness"]
    _assert_close("residual corpus-only baseline AUC", robust["corpus_only_baseline_auc"], 0.664)
    big = robust["big_corpora_only(aimed+bioinfer)"]
    _assert_close("residual big-corpora AUC improvement", big["corpus_lift_paired"], 0.0619)
    doc = robust["document_clustered_cv(aimed+bioinfer)"]
    _assert_close("residual document-clustered AUC improvement", doc["corpus_lift_paired"], 0.0683)

    # The corpus lift must survive every carrier treatment.
    car = data["carrier_robustness"]
    _assert_close("residual lift dropping carrier", car["oof_lift_drop_carrier"], 0.0658)
    _assert_close("residual lift carrier-words only", car["oof_lift_carrier_words_only"], 0.0669)
    _assert_close("residual lift carrier-classes only", car["oof_lift_carrier_classes_only"], 0.0637)
    # Carrier ladder on exact predicates; the capped values are retained for comparison.
    _assert_close("matched gap plus carrier class",
                  car["matched_stratum_plus_carrier_class"]["mean_gap"], 0.363)
    _assert_close("matched gap plus carrier word+roles",
                  car["matched_stratum_plus_carrier_word_roles"]["mean_gap"], 0.337)
    _assert_close("carrier codebook classified share",
                  car["codebook_coverage"]["classified_share"], 0.746)

    matched = data["matched_stratum"]
    if matched["testable_strata"] != 36:
        raise AssertionError(f"residual testable strata: got {matched['testable_strata']}, expected 36")
    _assert_close("residual mean corpus gap", matched["mean_corpus_gap"], 0.394)
    legacy = data["matched_stratum_legacy_capped_predicates"]
    if legacy["testable_strata"] != 39:
        raise AssertionError(
            f"legacy capped strata: got {legacy['testable_strata']}, expected 39 "
            "(capped-predicate comparison)")
    _assert_close("legacy capped mean corpus gap", legacy["mean_corpus_gap"], 0.415)
    if matched["n_strata_full_agreement"] != 1:
        raise AssertionError(
            "residual full-agreement matched strata: "
            f"got {matched['n_strata_full_agreement']}, expected 1"
        )
    link = _example_rates(data, "link|verbal|asserted")
    _assert_close("residual link AIMed rate", link["aimed"], 0.0)
    _assert_close("residual link BioInfer rate", link["bioinfer"], 0.96)
    inhibit = _example_rates(data, "inhibit|verbal|asserted")
    _assert_close("residual inhibit AIMed rate", inhibit["aimed"], 0.2)
    _assert_close("residual inhibit LLL rate", inhibit["lll"], 1.0)

    # Exact counts behind the rounded rates above.
    link_counts = _example_counts(data, "link|verbal|asserted")
    if link_counts["aimed"] != [0, 5] or link_counts["bioinfer"] != [23, 24]:
        raise AssertionError(f"residual link counts: got {link_counts}, expected aimed=0/5, bioinfer=23/24")
    inhibit_counts = _example_counts(data, "inhibit|verbal|asserted")
    if inhibit_counts["aimed"] != [3, 15] or inhibit_counts["lll"] != [7, 7]:
        raise AssertionError(
            f"residual inhibit counts: got {inhibit_counts}, expected aimed=3/15, lll=7/7"
        )


def check_domain_control() -> None:
    """Check the domain-control finding with inequalities: content control absorbs much of the
    corpus effect, but a residual remains.
    """
    path = PAPER_DIR / "results/vocabulary-control.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["n_observations"] != 4184:
        raise AssertionError(f"domain-control n_observations: got {data['n_observations']}, expected 4184")
    raw = data["corpus_lift_over_cues"]
    residual = data["corpus_lift_over_cues_plus_content"]
    doc = data["content_control_doc_clustered_corpus_lift(aimed+bioinfer)"]
    if not (raw > residual > 0):
        raise AssertionError(
            f"domain control: expected raw corpus lift ({raw}) > content-controlled "
            f"residual ({residual}) > 0; the content control should shrink but not erase it"
        )
    if not (isinstance(doc, (int, float)) and doc > 0):
        raise AssertionError(f"domain control: document-clustered residual should stay positive, got {doc}")
    # Also pin the reported decimals, with tolerance for minor scikit-learn noise.
    _assert_close("domain-control raw corpus lift", raw, 0.0663, tol=0.01)
    _assert_close("domain-control content-controlled residual (random split)", residual, 0.0228, tol=0.01)
    _assert_close("domain-control content-controlled residual (doc-clustered, aimed+bioinfer)",
                  doc, 0.0528, tol=0.012)


def _sig_keys(path: Path) -> set:
    return {
        (r["dimension"], r["key"], tuple(sorted((r["corpus_1"], r["corpus_2"]))))
        for r in _read_csv(path)
        if float(r["q_value"]) < 0.05
    }


def _isolated_contrary_from_cpr(path: Path) -> int:
    """Recompute within-corpus isolated-contrary construction cells from a cpr_summary.csv:
    pattern-dimension cells with support>=5 whose minority label share is in (0, 0.25]."""
    n = 0
    for r in _read_csv(path):
        if r["dimension"] != "pattern":
            continue
        support = int(r["support"])
        if support < 5:
            continue
        minority = min(int(r["gold_positive"]), int(r["gold_negative"]))
        if 0 < minority / support <= 0.25:
            n += 1
    return n


def check_cross_examples() -> None:
    """Pin the counts and q-value magnitudes of the six example cross-corpus cells."""
    rows = _read_csv(PAPER_DIR / "results/cross-corpus/fisher-tests.csv")
    by = {}
    for r in rows:
        by[(r["dimension"], r["key"], tuple(sorted((r["corpus_1"], r["corpus_2"]))))] = r
    # (dimension, key, (corpusA, corpusB)) -> (posA, supA, posB, supB, stated_q)
    expected = [
        ("trigger", "complex", ("aimed", "bioinfer"), 51, 160, 119, 157, 4e-11),
        ("carrier", "complex", ("aimed", "bioinfer"), 30, 72, 43, 52, 2e-4),
        ("trigger", "require", ("aimed", "bioinfer"), 6, 30, 15, 16, 2e-5),
        ("trigger", "link", ("aimed", "bioinfer"), 0, 6, 26, 27, 7e-5),
        ("trigger", "stimulate", ("aimed", "iepa"), 3, 15, 19, 19, 2e-5),
        ("trigger", "inhibit", ("aimed", "lll"), 3, 45, 7, 7, 2e-5),
    ]
    for dim, key, pair, posA, supA, posB, supB, q_stated in expected:
        r = by.get((dim, key, pair))
        if r is None:
            raise AssertionError(f"cross-example missing: {dim}|{key}|{pair}")
        # orient the CSV row to (corpusA, corpusB) = alphabetical pair order used in `expected`
        if (r["corpus_1"], r["corpus_2"]) == pair:
            got = (int(r["positive_1"]), int(r["support_1"]), int(r["positive_2"]), int(r["support_2"]))
        else:
            got = (int(r["positive_2"]), int(r["support_2"]), int(r["positive_1"]), int(r["support_1"]))
        if got != (posA, supA, posB, supB):
            raise AssertionError(
                f"cross-example {dim}|{key}|{pair}: got {got}, expected {(posA, supA, posB, supB)}"
            )
        q = float(r["q_value"])
        # match the manuscript's 1-significant-figure rounding of q (same exponent + leading digit)
        exp = math.floor(math.log10(q))
        exp_stated = math.floor(math.log10(q_stated))
        if exp != exp_stated or round(q / 10**exp) != round(q_stated / 10**exp_stated):
            raise AssertionError(
                f"cross-example {dim}|{key}|{pair}: q={q:.2e} != stated {q_stated:.0e}"
            )


def check_cross_localization() -> None:
    """AIMed and BioInfer disagree on the most constructions, and AIMed differs
    significantly from every other corpus."""
    rows = _read_csv(PAPER_DIR / "results/cross-corpus/fisher-tests.csv")
    from collections import Counter
    pair_counts: Counter = Counter()
    aimed_partners = set()
    for r in rows:
        if float(r["q_value"]) >= 0.05:
            continue
        pair = tuple(sorted((r["corpus_1"], r["corpus_2"])))
        pair_counts[pair] += 1
        if "aimed" in pair:
            aimed_partners.add(pair[0] if pair[1] == "aimed" else pair[1])
    top_pair, top_n = pair_counts.most_common(1)[0]
    if top_pair != ("aimed", "bioinfer") or top_n != 32:
        raise AssertionError(
            f"cross-corpus localization: most-divergent pair got {top_pair}={top_n}, expected (aimed,bioinfer)=32"
        )
    if aimed_partners != {"bioinfer", "hprd50", "iepa", "lll"}:
        raise AssertionError(
            f"AIMed should differ significantly from every other corpus; got partners {sorted(aimed_partners)}"
        )


_CORE_PHYSICAL_TRIGGERS = {
    "bind", "interact", "associate", "complex", "dimerize", "heterodimerize", "oligomerize",
}


def check_core_physical_within() -> None:
    """Pin mixed and isolated-contrary cells for core-physical versus other triggers."""
    rows = _read_csv(PAPER_DIR / "results/cross-corpus/pattern-rates.csv")
    core_mixed = core_iso = bnd_mixed = bnd_iso = 0
    for r in rows:
        if r["dimension"] != "pattern":
            continue
        support = int(r["support"])
        pos = int(r["gold_positive"])
        neg = int(r["gold_negative"])
        if support < 5 or pos == 0 or neg == 0:
            continue
        is_isolated = (min(pos, neg) / support) <= 0.25
        parts = r["key"].split("|")
        trigger = parts[2] if len(parts) > 2 else ""
        is_core = trigger in _CORE_PHYSICAL_TRIGGERS
        if is_core:
            core_mixed += 1
            core_iso += is_isolated
        else:
            bnd_mixed += 1
            bnd_iso += is_isolated
    if (core_mixed, core_iso, bnd_mixed, bnd_iso) != (34, 20, 73, 34):
        raise AssertionError(
            f"core-physical within-corpus breakdown: got core {core_iso}/{core_mixed}, "
            f"boundary {bnd_iso}/{bnd_mixed}; expected core 20/34, boundary 34/73"
        )


def check_corpus_counts() -> None:
    """Pin sentence and released-positive counts for each corpus."""
    expected = {
        "AIMed": (1943, 991), "BioInfer": (1100, 2534), "HPRD50": (145, 163),
        "IEPA": (486, 335), "LLL": (77, 164),
    }
    for name, (n_sent, n_pos) in expected.items():
        path = ROOT / f"data/datasets/{name}/full.json"
        data = json.loads(path.read_text(encoding="utf-8"))["data"]
        got_sent = len(data)
        got_pos = sum(1 for rec in data.values()
                      for v in rec.get("relations", {}).values() if v == "PPI")
        if (got_sent, got_pos) != (n_sent, n_pos):
            raise AssertionError(
                f"corpus counts {name}: got {got_sent} sentences / {got_pos} positives, "
                f"expected {n_sent}/{n_pos}"
            )


def check_shuffle_null() -> None:
    """Pin the within-corpus label-shuffle null from its saved result."""
    path = PAPER_DIR / "results/within-corpus-null.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["observed"] != 54:
        raise AssertionError(f"shuffle null observed: got {data['observed']}, expected 54")
    if round(data["null_mean"]) != 42:
        raise AssertionError(f"shuffle null mean: got {data['null_mean']}, expected ~42")
    _assert_close("shuffle null p-value", data["p_value"], 0.023, tol=0.015)


def check_reader_validation_sample() -> None:
    """Pin the validation sample composition from the answer key that defines it."""
    path = PAPER_DIR / "validation/validation-items.csv"
    rows = _read_csv(path)
    if len(rows) != 163:
        raise AssertionError(f"validation sample size: got {len(rows)}, expected 163")
    from collections import Counter

    by_corpus = Counter(row["corpus"] for row in rows)
    expected = {"aimed": 46, "bioinfer": 44, "hprd50": 32, "iepa": 22, "lll": 19}
    if dict(by_corpus) != expected:
        raise AssertionError(f"validation sample per corpus: got {dict(by_corpus)}, expected {expected}")
    significant = sum(1 for row in rows if row["cell_significant"].lower() == "true")
    if (significant, len(rows) - significant) != (65, 98):
        raise AssertionError(
            f"validation significant/control split: got {significant}/{len(rows) - significant}, expected 65/98")


def check_scope_sensitivity() -> None:
    """Check Supplement S2: the widest rule reproduces 14/54 and 0.267 exactly; narrower rules
    keep effect sizes and add no within-corpus AUC residual.
    """
    path = PAPER_DIR / "results/scope-sensitivity.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rules = data["inclusion_rules"]

    base = rules["as_reported"]
    if base["n_observations"] != 4184:
        raise AssertionError(f"widest rule n: got {base['n_observations']}, expected 4184")
    if (base["primary_family"]["significant"], base["primary_family"]["tests"]) != (14, 54):
        raise AssertionError(
            f"widest rule primary family: got {base['primary_family']}, expected 14/54; "
            "the sensitivity analysis no longer reproduces the manuscript")
    _assert_close("widest rule matched contrast", base["matched_pair_contrast"]["mean_contrast"], 0.267)
    if base["matched_pair_contrast"]["n_strata"] != 30:
        raise AssertionError(f"widest rule matched strata: got {base['matched_pair_contrast']['n_strata']}, expected 30")

    strict = rules["applied_only"]
    if strict["n_observations"] != 2781:
        raise AssertionError(f"strictest rule n: got {strict['n_observations']}, expected 2781")
    if (strict["primary_family"]["significant"], strict["primary_family"]["tests"]) != (7, 41):
        raise AssertionError(f"strictest rule primary family: got {strict['primary_family']}, expected 7/41")
    _assert_close("strictest rule matched contrast",
                  strict["matched_pair_contrast"]["mean_contrast"], 0.239)

    # Every rule: effect size holds while the significance count falls.
    for name, entry in rules.items():
        delta = entry["primary_family"]["mean_abs_rate_difference"]
        if not 0.28 <= delta <= 0.34:
            raise AssertionError(
                f"{name}: mean |rate difference| {delta} left the 0.28--0.34 band the "
                "supplement states; Table S2's reading would need rewriting")
        contrast = entry["matched_pair_contrast"]["mean_contrast"]
        if not 0.20 <= contrast <= 0.30:
            raise AssertionError(
                f"{name}: matched AIMed-BioInfer contrast {contrast} left the 0.20--0.30 band")
        if entry["residual"]["within_corpus_lift"]["excludes_zero"]:
            raise AssertionError(
                f"{name}: within-corpus lift interval now excludes zero; the supplement says "
                "no inclusion rule produces an AUC residual")


def check_human_ratings() -> None:
    """Pin the independent annotator's descriptive fractions and definite-response counts.
    These are sample descriptions, not equivalence tests or agreement statistics."""
    item_rows = _read_csv(PAPER_DIR / "validation/validation-items.csv")
    key = {row["item_id"]: row for row in item_rows}

    # Recompute the single independent annotator's descriptive sample fractions.
    package = PAPER_DIR / "validation"
    public = _read_csv(package / "validation-ratings-2026-09-12.csv")
    expected_fields = {"item_id", "annotator_id", "relation_asserted", "trigger_correct", "carrier_correct", "endpoints_correct"}
    decode = {"yes": "是", "no": "否", "uncertain": "不確定"}
    if any(set(r) != expected_fields or r["annotator_id"] != "annotator_01" for r in public):
        raise AssertionError("Human ratings violate the anonymous comment-free schema")
    rows = [{"item_id": r["item_id"], **{k: decode[r[k]] for k in expected_fields - {"item_id", "annotator_id"}}} for r in public]
    if len({r["item_id"] for r in rows}) != len(rows) or {r["item_id"] for r in rows} != set(key):
        raise AssertionError("Human ratings do not match the fixed sample IDs")
    ratings = {"annotator_01": {r["item_id"]: r for r in rows}}
    if len(ratings["annotator_01"]) != 163:
        raise AssertionError(f"rated observations: got {len(ratings['annotator_01'])}, expected 163")
    for field, expected in (("relation_asserted", 0.765), ("trigger_correct", 0.712),
                            ("carrier_correct", 0.607), ("endpoints_correct", 0.248)):
        answers = [r[field] for r in ratings["annotator_01"].values()]
        definite = answers.count("是") + answers.count("否")
        _assert_close(f"primary-pass {field}", answers.count("是") / definite, expected, tol=0.002)
    expected_definite = {"relation_asserted": 162, "trigger_correct": 163,
                         "carrier_correct": 163, "endpoints_correct": 129}
    for field, expected in expected_definite.items():
        answers = [r[field] for r in ratings["annotator_01"].values()]
        definite = answers.count("是") + answers.count("否")
        if definite != expected:
            raise AssertionError(f"{field}: {definite} definite responses, expected {expected}")
    endpoints = [r["endpoints_correct"] for r in ratings["annotator_01"].values()]
    if endpoints.count("是") + endpoints.count("否") != 129:
        raise AssertionError("endpoint denominator changed; Section 5.4 states 129 observations")

    # No AIMed-BioInfer comparison may reach significance; recompute rather than read a summary.
    from statistics import NormalDist

    largest = 0.0
    for field in ("relation_asserted", "trigger_correct", "carrier_correct", "endpoints_correct"):
        counts = {}
        for corpus in ("aimed", "bioinfer"):
            answers = [ratings["annotator_01"][i][field] for i in key if key[i]["corpus"] == corpus]
            counts[corpus] = (answers.count("是"), answers.count("是") + answers.count("否"))
        (yes_a, n_a), (yes_b, n_b) = counts["aimed"], counts["bioinfer"]
        rate_a, rate_b = yes_a / n_a, yes_b / n_b
        pooled = (yes_a + yes_b) / (n_a + n_b)
        se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
        p_value = 2 * (1 - NormalDist().cdf(abs(rate_a - rate_b) / se))
        if p_value < 0.05:
            raise AssertionError(
                f"corpus comparison on {field} is now significant (p={p_value:.3f}); "
                "Section 5.4 states that none is and would need rewriting")
        largest = max(largest, abs(rate_a - rate_b))
    _assert_close("largest corpus difference", largest, 0.105, tol=0.002)


def check_parser_robustness() -> None:
    """Check retained overlap and within-corpus counts for the joint parser ablation."""
    sn = PAPER_DIR / "results/ablation"
    head = _sig_keys(PAPER_DIR / "results/cross-corpus/fisher-tests.csv")
    noq = _sig_keys(sn / "fisher-tests.csv")
    if len(head & noq) != 61:
        raise AssertionError(f"headline divergences surviving the joint parser ablation: got {len(head & noq)}, expected 61")
    sn_iso = _isolated_contrary_from_cpr(sn / "pattern-rates.csv")
    if sn_iso != 51:
        raise AssertionError(f"Stanza-noQANom isolated-contrary: got {sn_iso}, expected 51")


def check_figures() -> None:
    """The statistics workflow must regenerate the figure used in the paper."""
    for suffix in ("pdf", "png"):
        path = PAPER_DIR / f"figures/fig4_residual_corpus_effect.{suffix}"
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"paper figure is missing or empty: {path}; rerun src/reproduction/plot_residual.py")


def check_family_breakdown() -> None:
    """Check that the six BH families stay separate, including the primary family."""
    path = PAPER_DIR / "results/construction-families.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data["primary_significant"], data["primary_tests"]) != (14, 54):
        raise AssertionError(
            f"primary family: got {data['primary_significant']}/{data['primary_tests']}, "
            "expected 14/54 (canonical constructions)")
    if (data["total_significant_all_families"], data["total_tests_all_families"]) != (70, 266):
        raise AssertionError("six-family total drifted from 70/266")
    rec = data["primary_recomputed_from_ledgers"]
    if (rec["significant"], rec["tests"]) != (14, 54):
        raise AssertionError(f"primary family does not reproduce from ledgers: {rec}")
    expected = {"pattern": (54, 14), "semantic_pattern": (59, 18), "trigger": (59, 19),
                "carrier": (25, 5), "carrier_pattern": (1, 1), "pair_stage": (68, 13)}
    got = {d: (e["tests"], e["significant"]) for d, e in data["per_dimension"].items()}
    if got != expected:
        raise AssertionError(f"per-dimension breakdown: got {got}, expected {expected}")


def check_matched_gap_nulls() -> None:
    """Check that the matched gap exceeds the document-clustered additive-offset null."""
    path = PAPER_DIR / "results/matched-gap-nulls.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    lvl = data["levels"]["pattern_only"]
    _assert_close("strict matched-construction mean gap", lvl["observed_mean_gap"], 0.3643)
    if lvl["n_strata"] != 35:
        raise AssertionError("The strict matched-construction analysis requires 35 strata")
    sp = lvl["signed_pair_contrast"]
    _assert_close("signed contrast mean", sp["mean_contrast"], 0.267, tol=0.002)
    _assert_close("signed contrast sd", sp["sd_contrast"], 0.277, tol=0.002)
    clustered = sp["sd_vs_null_a_document_clustered"]
    if clustered["n_draws"] != 4000:
        raise AssertionError("The paper's clustered null requires 4000 draws")
    if clustered["p_value"] >= 0.05:
        raise AssertionError(
            f"interaction test no longer significant under the document-clustered null: "
            f"p={clustered['p_value']}; Section 5.1's construction-specific claim fails")
    if _val(sp["sd_contrast"]) <= clustered["null_mean"]:
        raise AssertionError("observed spread no longer exceeds the clustered null mean")
    # Null B must stay well below the observed value.
    if lvl["null_b_label_permutation"]["null_mean"] >= lvl["observed_mean_gap"]:
        raise AssertionError("label-permutation baseline now exceeds the observed gap")
    # The document ICC justifies the clustered null.
    if not 0.02 < data["residual_document_icc"] < 0.30:
        raise AssertionError(f"residual document ICC out of expected range: "
                             f"{data['residual_document_icc']}")
    # Every carrier-ladder step must straddle zero.
    for name, d in data["paired_ladder_common_base"]["deltas_vs_pattern_only"].items():
        if not (d["ci_low"] <= 0 <= d["ci_high"]):
            raise AssertionError(
                f"paired carrier-ladder delta {name} no longer straddles zero: {d}; "
                "the 'carrier does not explain the gap' claim would need rewriting")


def main() -> int:
    global PAPER_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=PAPER_DIR,
                        help="Folder with the saved paper results to check")
    PAPER_DIR = parser.parse_args().paper_dir.resolve()
    check_corpus_counts()
    check_within()
    check_core_physical_within()
    check_shuffle_null()
    check_cross_corpus()
    check_cross_examples()
    check_cross_localization()
    check_coverage()
    check_residual_policy()
    check_domain_control()
    check_family_breakdown()
    check_matched_gap_nulls()
    check_reader_validation_sample()
    check_human_ratings()
    check_scope_sensitivity()
    check_parser_robustness()
    # The public snapshot ships no figures; a statistics rebuild regenerates them.
    if MANIFEST.get("ships_figures", True) or (PAPER_DIR / "figures").is_dir():
        check_figures()
    print("APBC paper-number gate passed: numerical claims, coverage, ratings and paper figure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
