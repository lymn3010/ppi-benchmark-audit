"""Build honest pair-level inference traces from existing additive evidence."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from src.models.observation import PairInferenceTrace

from .stage_fn_diagnosis import (
    ProposalMeta,
    STAGE_ASSERTION_SUPPRESSED,
    STAGE_EMITTED,
    STAGE_EMITTED_DISAGREE,
    STAGE_PROPOSAL_NOT_APPLIED,
    STAGE_RELATIONAL_REVIEW_ONLY,
    _event_protein_sets,
    _same_event_pairs,
    classify_pair,
)


_ASSERTED = "asserted"


def _pair(value) -> tuple[int, int] | None:
    if value is None or len(value) != 2:
        return None
    a, b = int(value[0]), int(value[1])
    if a == b:
        return None
    return tuple(sorted((a, b)))


def _application_status(stage: str, proposal_exists: bool) -> str:
    if stage in {STAGE_EMITTED, STAGE_EMITTED_DISAGREE}:
        return "applied"
    if stage == STAGE_ASSERTION_SUPPRESSED:
        return "blocked_assertion"
    if stage == STAGE_RELATIONAL_REVIEW_ONLY:
        return "review_only"
    if proposal_exists:
        return "not_applied_reason_unobserved"
    return "not_generated"


def build_pair_inference_traces(
    *,
    sentence_id: str,
    candidate_events: Iterable[dict],
    candidate_relations: Iterable[dict],
    predicted_pairs: Iterable[tuple[int, int]],
    gold_pairs: Iterable[tuple[int, int]] | None,
    parser_ok: bool,
    projection_plans: Iterable[dict] = (),
) -> list[PairInferenceTrace]:
    """Trace every bounded pair visible in events, proposals, predictions, or gold."""
    events = list(candidate_events or [])
    relations = list(candidate_relations or [])
    plans = list(projection_plans or [])
    event_sets = _event_protein_sets(events)
    same_event_pairs = _same_event_pairs(event_sets)
    event_endpoints = set().union(*event_sets) if event_sets else set()
    predicted = {pair for value in predicted_pairs or [] if (pair := _pair(value))}
    gold = None if gold_pairs is None else {
        pair for value in gold_pairs if (pair := _pair(value))
    }

    proposal_rows: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in relations:
        pair = _pair(row.get("pair"))
        if pair is not None:
            proposal_rows[pair].append(row)

    plans_by_source: dict[str, list[dict]] = defaultdict(list)
    for row in plans:
        source_event_id = str(row.get("source_event_id") or "")
        if source_event_id:
            plans_by_source[source_event_id].append(row)

    universe = set(proposal_rows) | predicted | same_event_pairs
    if gold is not None:
        universe.update(gold)

    traces: list[PairInferenceTrace] = []
    for pair in sorted(universe):
        rows = proposal_rows.get(pair, [])
        assertions = tuple(sorted({
            str(row.get("assertion_status") or "asserted")
            for row in rows
        }))
        modes = tuple(sorted({
            str(row.get("projection_mode") or "")
            for row in rows
            if row.get("projection_mode")
        }))
        proposal = ProposalMeta(
            exists=bool(rows),
            asserted=any(status == _ASSERTED for status in assertions),
            relational=bool(rows) and all(mode == "relational" for mode in modes),
        )
        gold_positive = True if gold is None else pair in gold
        stage, visibility = classify_pair(
            pair,
            proposal=proposal,
            same_event_pairs=same_event_pairs,
            event_endpoints=event_endpoints,
            emitted_pairs=predicted,
            is_gold_positive=gold_positive,
            parser_ok=parser_ok,
        )
        event_refs = tuple(sorted({
            str(event.get("event_id") or "")
            for event, proteins in zip(events, event_sets)
            if set(pair) & proteins and event.get("event_id")
        }))
        proposal_source_refs = {
            str(row.get("source_event_id") or "")
            for row in rows
            if row.get("source_event_id")
        }
        related_plans = [
            plan
            for source_event_id in sorted(set(event_refs) | proposal_source_refs)
            for plan in plans_by_source.get(source_event_id, [])
        ]
        projection_plan_refs = tuple(sorted({
            str(plan.get("id"))
            for plan in related_plans
            if plan.get("id")
        }))
        application_status = _application_status(stage, proposal.exists)
        reason_code = stage
        application_reason_observed = stage in {
            STAGE_EMITTED,
            STAGE_EMITTED_DISAGREE,
            STAGE_ASSERTION_SUPPRESSED,
            STAGE_RELATIONAL_REVIEW_ONLY,
        }
        exact_application_reasons: tuple[str, ...] = ()
        if stage == STAGE_PROPOSAL_NOT_APPLIED and related_plans:
            statuses = {
                str(plan.get("application_status") or "")
                for plan in related_plans
            }
            if statuses == {"not_applied"}:
                exact_application_reasons = tuple(sorted({
                    str(plan.get("reason_code") or "")
                    for plan in related_plans
                    if plan.get("reason_code")
                }))
                if exact_application_reasons:
                    application_status = "not_applied"
                    reason_code = (
                        exact_application_reasons[0]
                        if len(exact_application_reasons) == 1
                        else "multiple_application_reasons"
                    )
                    application_reason_observed = True

        traces.append(PairInferenceTrace(
            id=f"{sentence_id}:pair:{pair[0]}-{pair[1]}",
            sentence_id=sentence_id,
            pair=pair,
            stage=stage,
            endpoint_visibility=visibility,
            application_status=application_status,
            gold_status=(
                "unlabeled"
                if gold is None
                else ("positive" if pair in gold else "negative")
            ),
            emitted=pair in predicted,
            proposal_refs=tuple(sorted(
                str(row.get("candidate_id"))
                for row in rows
                if row.get("candidate_id")
            )),
            event_refs=event_refs,
            projection_plan_refs=projection_plan_refs,
            assertion_statuses=assertions,
            projection_modes=modes,
            reason_code=reason_code,
            provenance={
                "schema": "pair_inference_trace_v1",
                "projection_reason_observed": application_reason_observed,
                "projection_reason_codes": list(exact_application_reasons),
            },
        ))
    return traces
