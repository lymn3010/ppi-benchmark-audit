#!/usr/bin/env python3
"""Count released-positive pairs by supporting evidence type."""
from __future__ import annotations

from src.reproduction.config import PAPER_DIR as EXPERIMENT_DIR

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reproduction.config import CORPORA, DB_PATHS  # noqa: E402


TIERS = ["clause_SVO", "event_nominal", "complex_state",
         "held_evidence", "lexpath", "out_of_scope"]


def _latest_db(corpus: str) -> Path:
    runs = sorted((ROOT / "output" / corpus / "full").glob("*/db/events.db"))
    if not runs:
        raise FileNotFoundError(f"no full-split event db found for {corpus}")
    return runs[-1]


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _parse_db_overrides(values: list[str] | None) -> dict[str, Path]:
    if not values:
        return {}
    dbs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--db must be corpus=path, got {value!r}")
        corpus, raw_path = value.split("=", 1)
        corpus = corpus.strip().lower()
        if corpus not in CORPORA:
            raise ValueError(f"unknown corpus in --db: {corpus!r}; expected one of {CORPORA}")
        path = Path(raw_path)
        dbs[corpus] = path if path.is_absolute() else ROOT / path
    missing = sorted(set(CORPORA) - set(dbs))
    if missing:
        raise ValueError(f"--db overrides must provide every corpus; missing {missing}")
    return dbs


def _db_paths(source: str, overrides: dict[str, Path] | None = None) -> dict[str, Path]:
    if overrides:
        return overrides
    if source == "latest":
        return {c: _latest_db(c) for c in CORPORA}
    if source == "frozen":
        return DB_PATHS
    raise ValueError(f"unknown db source: {source}")


def run(
    db_source: str = "frozen",
    *,
    db_overrides: dict[str, Path] | None = None,
    db_source_label: str | None = None,
) -> dict:
    untraced_pairs = []
    db_paths = _db_paths(db_source, db_overrides)
    source_label = db_source_label or ("explicit" if db_overrides else db_source)
    per_corpus: dict[str, dict] = {}
    for c in CORPORA:
        db_path = db_paths[c]
        if not db_path.exists():
            raise FileNotFoundError(f"missing event db for {c}: {db_path}")
        con = sqlite3.connect(db_path); con.row_factory = sqlite3.Row
        gold: dict[str, set] = {}
        lex = set()
        diagnostics = set()
        for r in con.execute("SELECT sentence_id, gold_pairs, artifacts FROM sentences"):
            gold[r["sentence_id"]] = {frozenset(p) for p in json.loads(r["gold_pairs"] or "[]")}
            for candidate in json.loads(r["artifacts"] or "{}").get("path_candidates", []):
                key = (r["sentence_id"], frozenset(candidate["pair"]))
                diagnostics.add(key)
                # An empty path is a diagnostic failure, not connected path evidence.
                if candidate.get("dep_path"):
                    lex.add(key)
        # Per gold pair: committed channels, any row, and appositive evidence.
        committed: dict[tuple, set] = defaultdict(set)
        seen_any: set = set()
        held_appositive: set = set()
        for r in con.execute(
            "SELECT sentence_id,pair_a,pair_b,realization_channel,decision FROM candidate_relations"
        ):
            pair = frozenset((r["pair_a"], r["pair_b"]))
            if pair not in gold.get(r["sentence_id"], set()):
                continue
            key = (r["sentence_id"], pair)
            ch = r["realization_channel"] or ""
            seen_any.add(key)
            if r["decision"] == "applied":
                committed[key].add(ch)
            elif ch == "appositive":
                held_appositive.add(key)

        counts = {t: 0 for t in TIERS}
        held_appos_n = 0
        all_gold = [(sid, p) for sid, ps in gold.items() for p in ps]
        for key in all_gold:
            chans = committed.get(key, set())
            if "clause" in chans:
                counts["clause_SVO"] += 1
            elif "event_nominal" in chans:
                counts["event_nominal"] += 1
            elif "state_nominal" in chans:
                counts["complex_state"] += 1
            elif chans:                      # committed via another channel -> structural
                counts["clause_SVO"] += 1
            elif key in seen_any:            # seen but ProjectionPlanner did not commit
                counts["held_evidence"] += 1
                if key in held_appositive:
                    held_appos_n += 1
            elif key in lex:   # only reachable via inert lexpath
                counts["lexpath"] += 1
            else:
                counts["out_of_scope"] += 1
                untraced_pairs.append({"corpus": c, "sentence_id": key[0],
                                       "pair": sorted(key[1]),
                                       "reason": "empty_dependency_path" if key in diagnostics else "no_record"})
        counts["_gold_total"] = len(all_gold)
        counts["_held_appositive"] = held_appos_n
        per_corpus[c] = counts
        con.close()

    total = {t: sum(per_corpus[c][t] for c in CORPORA) for t in TIERS}
    total["_gold_total"] = sum(per_corpus[c]["_gold_total"] for c in CORPORA)
    return {
        "schema": "ppi_coverage_v2",
        "path_criterion": "nonempty dep_path in saved sentence artifacts; diagnostic rows alone do not count",
        "untraced_pairs": sorted(untraced_pairs, key=lambda r: (r["corpus"], r["sentence_id"], r["pair"])),
        "db_source": source_label,
        "db_paths": {c: _display_path(db_paths[c]) for c in CORPORA},
        "per_corpus": per_corpus,
        "total": total,
    }


def _fmt(res: dict) -> str:
    out = ["# Coverage by evidence type (APBC corpora)\n",
           "Each released-positive pair is assigned to its single highest evidence tier; "
           "per corpus, never pooled.\n",
           f"DB source: `{res.get('db_source', 'unknown')}`.\n",
           "| corpus | released positives | clause/SVO | event-nominal | complex/state | "
           "held-as-evidence | dependency-path only | out-of-scope |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def row(name, d):
        g = d["_gold_total"]
        def cell(t):
            return f"{d[t]} ({d[t]/g*100:.0f}%)" if g else "0"
        return (f"| {name} | {g} | {cell('clause_SVO')} | {cell('event_nominal')} | "
                f"{cell('complex_state')} | {cell('held_evidence')} | "
                f"{cell('lexpath')} | {cell('out_of_scope')} |")
    for c in CORPORA:
        out.append(row(c, res["per_corpus"][c]))
    out.append(row("**TOTAL**", res["total"]))
    t = res["total"]; g = t["_gold_total"]
    committed = t["clause_SVO"] + t["event_nominal"] + t["complex_state"]
    appos = sum(res["per_corpus"][c].get("_held_appositive", 0) for c in CORPORA)
    out.append(f"\n- **Committed (clause+event-nominal+complex): "
               f"{committed} = {committed/g*100:.0f}%** of released positives -- coverage of the released-positive set "
               f"(clause/SVO {t['clause_SVO']/g*100:.0f}%, event-nominal "
               f"{t['event_nominal']/g*100:.0f}%, complex/state {t['complex_state']/g*100:.0f}%).")
    out.append(f"- Held-as-evidence (event/relation seen but ProjectionPlanner deliberately did "
               f"NOT commit; of which {appos} are appositive IS-A relation-statements): "
               f"{t['held_evidence']} = {t['held_evidence']/g*100:.0f}%, a deliberate abstention "
               f"(membership/identity/ambiguous ≠ auto-PPI).")
    out.append(f"- Dependency-path evidence only (no committed/held structural row, only an "
               f"inert lexicalized dependency path): {t['lexpath']} = {t['lexpath']/g*100:.0f}% "
               f"-- gold-conditioned diagnostic evidence, NOT predictions; these rows still "
               f"have a dependency path.")
    out.append(f"- Out of scope (no structural row and no lexpath path): {t['out_of_scope']} = "
               f"{t['out_of_scope']/g*100:.0f}%.")
    reach = committed + t['held_evidence'] + t['lexpath']
    out.append(f"- **System produces some traceable evidence for {reach}/{g} = "
               f"{reach/g*100:.0f}%** of released positives; {t['out_of_scope']/g*100:.0f}% lack structural or connected-path evidence and are "
               f"unreached.")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db-source",
        choices=["latest", "frozen"],
        default="frozen",
        help=(
            "frozen uses src.reproduction.config.DB_PATHS; latest is a drift "
            "probe over the newest output/<corpus>/full run."
        ),
    )
    ap.add_argument(
        "--db",
        action="append",
        default=None,
        metavar="CORPUS=PATH",
        help="Explicit events.db override. Provide all five corpora to avoid mixed sources.",
    )
    ap.add_argument(
        "--db-source-label",
        default=None,
        help="Manifest label used when --db overrides are supplied.",
    )
    args = ap.parse_args()
    db_overrides = _parse_db_overrides(args.db)
    res = run(
        db_source=args.db_source,
        db_overrides=db_overrides or None,
        db_source_label=args.db_source_label,
    )
    dest = EXPERIMENT_DIR / 'results'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "coverage.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = _fmt(res)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
