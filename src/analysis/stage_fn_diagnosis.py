"""Locate the stage where a released-positive pair lost support.

This is a post-hoc comparison with corpus labels; it does not change inference."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

# Controlled stage vocabulary (design doc Section 9.2)

# Terminal "good" outcomes (application runs only).
STAGE_EMITTED = "emitted"                              # pair predicted, gold-positive
STAGE_EMITTED_DISAGREE = "emitted_gold_disagreement"  # pair predicted, gold-negative (FP)

# FN stages, ordered from latest (closest to emission) to earliest.
STAGE_PROPOSAL_NOT_APPLIED = "proposal_not_applied"   # proposal exists; exact gate not yet traced
STAGE_ASSERTION_SUPPRESSED = "assertion_suppressed"   # proposal exists but is not asserted
STAGE_RELATIONAL_REVIEW_ONLY = "relational_review_only"  # relational evidence, no PPI proposal
STAGE_PAIR_NOT_PROPOSED = "pair_not_proposed"         # both endpoints in one event, no proposal
STAGE_ARGUMENT_NOT_BOUND = "argument_not_bound"       # endpoints in separate events or only one
STAGE_EVENT_NOT_FOUND = "event_not_found"             # parse ok but no event holds either endpoint
STAGE_PARSER_UNUSABLE = "parser_unusable"             # no usable parse structure

FN_STAGES = (
    STAGE_PROPOSAL_NOT_APPLIED,
    STAGE_ASSERTION_SUPPRESSED,
    STAGE_RELATIONAL_REVIEW_ONLY,
    STAGE_PAIR_NOT_PROPOSED,
    STAGE_ARGUMENT_NOT_BOUND,
    STAGE_EVENT_NOT_FOUND,
    STAGE_PARSER_UNUSABLE,
)

# Endpoint-visibility vocabulary (design doc Section 2.2)

VIS_PROPOSED = "pair_candidate_exists"     # a materialized pair proposal already binds both
VIS_SAME_EVENT = "same_event_no_pair"      # both endpoints in one event, no proposal
VIS_SEPARATE = "separate_events"           # endpoints appear in different events
VIS_ONE = "one_endpoint"                   # only one endpoint appears in any event
VIS_NEITHER = "neither_endpoint"           # neither endpoint appears in any event

VISIBILITY_ORDER = (VIS_PROPOSED, VIS_SAME_EVENT, VIS_SEPARATE, VIS_ONE, VIS_NEITHER)

# The runtime emission policy permits only the explicit "asserted" state.
_ASSERTED = "asserted"

# projection_mode value that means relational (IS-A/HAS/etc.) review evidence.
_RELATIONAL_PROJECTION = "relational"


# Data structures

@dataclass(frozen=True)
class ProposalMeta:
    """The materialized pair-proposal facts for one gold pair in one sentence."""
    exists: bool = False
    asserted: bool = True
    relational: bool = False


@dataclass
class PairDiagnosis:
    sentence_id: str
    pair: tuple[int, int]
    is_gold_positive: bool
    emitted: bool
    stage: str
    endpoint_visibility: str


@dataclass
class CorpusDiagnosis:
    corpus: str
    db_path: str
    parser_backend: str = ""
    n_sentences: int = 0
    n_gold_pairs: int = 0
    n_emitted_gold: int = 0
    stage_counts: Counter = field(default_factory=Counter)
    visibility_counts: Counter = field(default_factory=Counter)
    pairs: list[PairDiagnosis] = field(default_factory=list)

    @property
    def n_fn(self) -> int:
        return sum(self.stage_counts[s] for s in FN_STAGES)

    def to_summary(self) -> dict:
        return {
            "corpus": self.corpus,
            "db_path": self.db_path,
            "parser_backend": self.parser_backend,
            "n_sentences": self.n_sentences,
            "n_gold_pairs": self.n_gold_pairs,
            "n_emitted_gold": self.n_emitted_gold,
            "n_fn": self.n_fn,
            "stage_counts": {s: self.stage_counts.get(s, 0) for s in
                             (STAGE_EMITTED, STAGE_EMITTED_DISAGREE, *FN_STAGES)},
            "visibility_counts": {v: self.visibility_counts.get(v, 0)
                                  for v in VISIBILITY_ORDER},
        }


# Classifier core (DB-free, unit-testable)

def classify_pair(
    pair: tuple[int, int],
    *,
    proposal: ProposalMeta,
    same_event_pairs: set[tuple[int, int]],
    event_endpoints: set[int],
    emitted_pairs: set[tuple[int, int]],
    is_gold_positive: bool,
    parser_ok: bool,
) -> tuple[str, str]:
    """Classify one sorted protein pair as (stage, endpoint_visibility)."""
    a, b = pair

    if proposal.exists:
        visibility = VIS_PROPOSED
    elif pair in same_event_pairs:
        visibility = VIS_SAME_EVENT
    elif a in event_endpoints and b in event_endpoints:
        visibility = VIS_SEPARATE
    elif a in event_endpoints or b in event_endpoints:
        visibility = VIS_ONE
    else:
        visibility = VIS_NEITHER

    if pair in emitted_pairs:
        stage = STAGE_EMITTED if is_gold_positive else STAGE_EMITTED_DISAGREE
        return stage, visibility

    if proposal.exists:
        if not proposal.asserted:
            stage = STAGE_ASSERTION_SUPPRESSED
        elif proposal.relational:
            stage = STAGE_RELATIONAL_REVIEW_ONLY
        else:
            stage = STAGE_PROPOSAL_NOT_APPLIED
        return stage, visibility

    # No materialized proposal: locate the earliest structural failure.
    if pair in same_event_pairs:
        stage = STAGE_PAIR_NOT_PROPOSED
    elif (a in event_endpoints and b in event_endpoints) or \
         (a in event_endpoints or b in event_endpoints):
        stage = STAGE_ARGUMENT_NOT_BOUND
    elif parser_ok:
        stage = STAGE_EVENT_NOT_FOUND
    else:
        stage = STAGE_PARSER_UNUSABLE
    return stage, visibility


# DB extraction helpers

def _sorted_pair(a, b) -> tuple[int, int]:
    a, b = int(a), int(b)
    return (a, b) if a <= b else (b, a)


def _event_protein_sets(candidate_events: Iterable[dict]) -> list[set[int]]:
    """Return the protein indices each candidate event makes visible."""
    out: list[set[int]] = []
    for ev in candidate_events or []:
        pids: set[int] = set()
        pred = ev.get("predicate") or {}
        pids.update(int(x) for x in (pred.get("protein_indices") or []))
        args = ev.get("arguments") or {}
        for arg_list in args.values():
            if not isinstance(arg_list, list):
                continue
            for arg in arg_list:
                if not isinstance(arg, dict):
                    continue
                core = arg.get("core") or {}
                pids.update(int(x) for x in (core.get("protein_indices") or []))
                for owner in (arg.get("owners") or []):
                    if isinstance(owner, dict):
                        pids.update(int(x) for x in (owner.get("protein_indices") or []))
        out.append(pids)
    return out


def _same_event_pairs(event_sets: list[set[int]]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for s in event_sets:
        members = sorted(s)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add((members[i], members[j]))
    return pairs


def _load_proposals(conn: sqlite3.Connection) -> dict[str, dict[tuple[int, int], ProposalMeta]]:
    """Map sentence_id -> {sorted pair -> ProposalMeta} from candidate_relations."""
    out: dict[str, dict[tuple[int, int], ProposalMeta]] = {}
    cur = conn.execute(
        "SELECT sentence_id, pair_a, pair_b, assertion_status, projection_mode "
        "FROM candidate_relations"
    )
    for sid, pa, pb, assertion, projection in cur:
        if pa is None or pb is None:
            continue
        pair = _sorted_pair(pa, pb)
        asserted = (assertion or _ASSERTED) == _ASSERTED
        relational = (projection or "") == _RELATIONAL_PROJECTION
        bucket = out.setdefault(sid, {})
        prev = bucket.get(pair)
        # Asserted if any proposal is asserted; relational only if all are.
        if prev is None:
            bucket[pair] = ProposalMeta(exists=True, asserted=asserted, relational=relational)
        else:
            bucket[pair] = ProposalMeta(
                exists=True,
                asserted=prev.asserted or asserted,
                relational=prev.relational and relational,
            )
    return out


# Public DB entry point

def diagnose_db(db_path: str, corpus: Optional[str] = None) -> CorpusDiagnosis:
    """Run the stage-aware diagnosis over one finished run's events.db."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        meta = {k: v for k, v in conn.execute("SELECT key, value FROM run_metadata")}
        corpus = corpus or meta.get("dataset") or meta.get("corpus") or db_path
        result = CorpusDiagnosis(corpus=corpus, db_path=db_path)
        proposals = _load_proposals(conn)

        for row in conn.execute(
            "SELECT sentence_id, gold_pairs, pred_pairs, artifacts FROM sentences"
        ):
            sid = row["sentence_id"]
            gold = {_sorted_pair(*p) for p in json.loads(row["gold_pairs"] or "[]") if len(p) == 2}
            pred = {_sorted_pair(*p) for p in json.loads(row["pred_pairs"] or "[]") if len(p) == 2}
            if not gold:
                continue
            result.n_sentences += 1

            artifacts = json.loads(row["artifacts"] or "{}")
            if not result.parser_backend:
                result.parser_backend = (artifacts.get("parse") or {}).get("parser_backend", "")
            event_sets = _event_protein_sets(artifacts.get("candidate_events") or [])
            same_ev = _same_event_pairs(event_sets)
            all_endpoints = set().union(*event_sets) if event_sets else set()
            token_offsets = (artifacts.get("parse") or {}).get("token_offsets") or []
            parser_ok = len(token_offsets) > 0
            sent_proposals = proposals.get(sid, {})

            for pair in gold:
                result.n_gold_pairs += 1
                stage, vis = classify_pair(
                    pair,
                    proposal=sent_proposals.get(pair, ProposalMeta()),
                    same_event_pairs=same_ev,
                    event_endpoints=all_endpoints,
                    emitted_pairs=pred,
                    is_gold_positive=True,
                    parser_ok=parser_ok,
                )
                if stage == STAGE_EMITTED:
                    result.n_emitted_gold += 1
                result.stage_counts[stage] += 1
                result.visibility_counts[vis] += 1
                result.pairs.append(PairDiagnosis(
                    sentence_id=sid, pair=pair, is_gold_positive=True,
                    emitted=(pair in pred), stage=stage, endpoint_visibility=vis,
                ))
        return result
    finally:
        conn.close()


# CLI

def _format_table(results: list[CorpusDiagnosis]) -> str:
    lines: list[str] = []
    lines.append("\n=== Endpoint visibility (design doc Section 2.2) ===")
    header = f"{'corpus':<10} {'gold':>6} " + " ".join(f"{v:>20}" for v in VISIBILITY_ORDER)
    lines.append(header)
    for r in results:
        cells = " ".join(f"{r.visibility_counts.get(v, 0):>20}" for v in VISIBILITY_ORDER)
        lines.append(f"{r.corpus:<10} {r.n_gold_pairs:>6} {cells}")

    lines.append("\n=== Pipeline stage (design doc Section 9.2) ===")
    stages = (STAGE_EMITTED, *FN_STAGES)
    header = f"{'corpus':<10} {'gold':>6} " + " ".join(f"{s[:18]:>19}" for s in stages)
    lines.append(header)
    for r in results:
        cells = " ".join(f"{r.stage_counts.get(s, 0):>19}" for s in stages)
        lines.append(f"{r.corpus:<10} {r.n_gold_pairs:>6} {cells}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Stage-aware gold-pair reach diagnosis over events.db."
    )
    p.add_argument("db", nargs="+", help="One or more events.db paths.")
    p.add_argument("--json-out", default=None, help="Write per-pair + summary JSON here.")
    args = p.parse_args(argv)

    results = [diagnose_db(db) for db in args.db]
    print(_format_table(results))

    if args.json_out:
        payload = {
            "summaries": [r.to_summary() for r in results],
            "pairs": [
                {
                    "corpus": r.corpus, "sentence_id": pd.sentence_id,
                    "pair": list(pd.pair), "emitted": pd.emitted,
                    "stage": pd.stage, "endpoint_visibility": pd.endpoint_visibility,
                }
                for r in results for pd in r.pairs
            ],
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
        print(f"\n[stage_fn_diagnosis] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
