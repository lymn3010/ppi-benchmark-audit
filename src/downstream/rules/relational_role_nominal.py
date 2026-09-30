"""Projection rule for role-nominal IS-A statements.

Only ``A is a ligand for B`` / ``A is an activator of B`` with an of/for endpoint.
"""
from __future__ import annotations

from itertools import product

from src.downstream.reference_policy import CanonicalIdentityMap
from src.models.candidate import AssertionStatus, CandidateArgument, CandidateEvent, EventType
from src.models.semantic_vocabulary import ProjectionRuleId, SemanticSlot, projection_rule_id
from src.runtime_resources import RuntimeResources


RULE_ID = str(projection_rule_id(ProjectionRuleId.APPLIED_RELATIONAL_ROLE_NOMINAL))
TRIGGER_LEXICON = "relational_role_projection_trigger"
ALLOWED_CASE_MARKERS = frozenset({"of", "for"})


def project_relational_role_nominal(
    event: CandidateEvent,
    *,
    identity_map: CanonicalIdentityMap,
    resources: RuntimeResources | None,
) -> tuple[tuple[int, int], ...]:
    """Return pairs licensed by ``projection.applied_relational_role_nominal@1``.

    Cross-product of ``entities_a`` and the of/for endpoints of an asserted IS-A event.
    """

    if event.event_type is not EventType.RELATION_STATEMENT:
        return ()
    if (event.relation_subtype or "").upper() != "IS-A":
        return ()
    if resources is not None and not resources.rule_enabled(RULE_ID):
        return ()

    trigger_keys = _trigger_keys(event)
    if not trigger_keys:
        return ()
    if not _trigger_supported(trigger_keys, resources):
        return ()

    source_args = tuple(
        arg
        for arg in event.arguments.get(SemanticSlot.ENTITIES_A.value, ())
        if _argument_allows_projection(arg)
    )
    target_args = tuple(
        arg
        for arg in event.arguments.get(SemanticSlot.ENTITIES_B.value, ())
        if _argument_allows_projection(arg)
        and _has_allowed_case_marker(arg)
        and _argument_matches_trigger(arg, trigger_keys)
    )
    if not source_args or not target_args:
        return ()

    source_variants = [
        variant
        for arg in source_args
        for variant in identity_map.expand_argument_variants(arg)
        if variant
    ]
    target_variants = [
        variant
        for arg in target_args
        for variant in identity_map.expand_argument_variants(arg)
        if variant
    ]

    pairs: set[tuple[int, int]] = set()
    for source, target in product(source_variants, target_variants):
        for left, right in product(source, target):
            if int(left) == int(right):
                continue
            pairs.add(tuple(sorted((int(left), int(right)))))
    return tuple(sorted(pairs))


def _trigger_keys(event: CandidateEvent) -> frozenset[str]:
    if event.predicate is None:
        return frozenset()
    keys = {
        *(str(value or "").strip().lower() for value in event.predicate.lemmas),
        *(str(value or "").strip().lower() for value in event.predicate.match_keys),
        *(str(value or "").strip().lower() for value in event.predicate.tokens),
    }
    return frozenset(value for value in keys if value)


def _trigger_supported(keys: frozenset[str], resources: RuntimeResources | None) -> bool:
    if resources is None:
        return False
    return resources.match_lexicon(TRIGGER_LEXICON, keys)


def _argument_allows_projection(argument: CandidateArgument) -> bool:
    if argument.core is None or argument.is_event_reference:
        return False
    if argument.core.is_negated:
        return False
    return argument.assertion_status == AssertionStatus.ASSERTED.value


def _has_allowed_case_marker(argument: CandidateArgument) -> bool:
    if _case_markers(argument) & ALLOWED_CASE_MARKERS:
        return True
    role = str(argument.role or "").strip().lower()
    if ":" in role and role.split(":", 1)[1] in ALLOWED_CASE_MARKERS:
        return True
    group_case = str((argument.group or {}).get("case_marker") or "").strip().lower()
    return group_case in ALLOWED_CASE_MARKERS


def _case_markers(argument: CandidateArgument) -> set[str]:
    values: set[str] = set()
    for marker in argument.case_markers:
        values.update(str(value or "").strip().lower() for value in marker.lemmas)
        values.update(str(value or "").strip().lower() for value in marker.tokens)
    return {value for value in values if value}


def _argument_matches_trigger(
    argument: CandidateArgument,
    trigger_keys: frozenset[str],
) -> bool:
    if argument.core is None:
        return False
    values = {
        *(str(value or "").strip().lower() for value in argument.core.lemmas),
        *(str(value or "").strip().lower() for value in argument.core.match_keys),
        *(str(value or "").strip().lower() for value in argument.core.tokens),
    }
    return bool(values & trigger_keys)


__all__ = ["RULE_ID", "project_relational_role_nominal"]
