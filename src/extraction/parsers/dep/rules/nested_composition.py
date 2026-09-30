"""Nested-event composition: a matrix argument controlling a missing inner endpoint."""
from __future__ import annotations

from dataclasses import replace

from src.extraction.rules import RuleApplication, append_rule_application
from src.models.candidate import CandidateArgument, CandidateEvent
from src.system_rules import rule_frozenset

from ..roles import DEP_DOBJ


SUBJECT_CONTROL_RULE_ID = "extraction.dep_nested_subject_control@1"
ADJUNCT_CONTROL_RULE_ID = "extraction.dep_nested_adjunct_control@1"

_SUBJECT_CONTROLLERS = rule_frozenset(
    "event_composition.nested_subject_controller_predicates"
)
_SUBJECT_CONTROLLER_PREPOSITIONS = rule_frozenset(
    "event_composition.nested_subject_controller_prepositions"
)
_ADJUNCT_CONTROLLER_PREPOSITIONS = rule_frozenset(
    "event_composition.nested_controller_prepositions"
)
_RELATION_COMPLEMENTS = rule_frozenset(
    "prepositions.nominal_relation_complements"
)


def bind_nested_controller(
    inner: CandidateEvent,
    reference: CandidateArgument,
    sources: tuple[CandidateArgument, ...],
    *,
    outer_lemma: str,
    resources=None,
) -> CandidateEvent:
    """Fill a controlled missing inner anchor; inner participants must be with/to complements."""
    if not sources or inner.is_directed:
        return inner
    participants = tuple(inner.arguments.get("participants", ()))
    if not participants:
        return inner

    participant_preps = [
        str(argument.role or "").partition(":")[2].lower()
        for argument in participants
    ]
    _inherit_group_complement_prepositions(participants, participant_preps)
    if not participant_preps or any(
        prep not in _RELATION_COMPLEMENTS for prep in participant_preps
    ):
        return inner

    reference_role = str(reference.role or "").lower()
    reference_prep = reference_role.partition(":")[2]
    subject_control = (
        outer_lemma in _SUBJECT_CONTROLLERS
        and (
            reference_role == DEP_DOBJ
            or reference_prep in _SUBJECT_CONTROLLER_PREPOSITIONS
        )
    )
    adjunct_control = reference_prep in _ADJUNCT_CONTROLLER_PREPOSITIONS
    if not (subject_control or adjunct_control):
        return inner

    rule_id = SUBJECT_CONTROL_RULE_ID if subject_control else ADJUNCT_CONTROL_RULE_ID
    if resources is not None and not resources.rule_enabled(rule_id):
        return inner
    controlled = replace(
        inner,
        extraction_detail=f"{inner.extraction_detail}-controlled",
        arguments={"sources": sources, "targets": participants},
        is_directed=True,
    )
    return append_rule_application(
        controlled,
        RuleApplication(
            rule_id=rule_id,
            stage="event_composition",
            reason="matrix_argument_controls_missing_nominal_relation_anchor",
            evidence={
                "outer_lemma": outer_lemma,
                "reference_role": reference_role,
                "source_referents": sorted({
                    index for source in sources
                    for index in source.projection_protein_indices
                }),
                "complement_referents": sorted({
                    index for participant in participants
                    for index in participant.projection_protein_indices
                }),
            },
        ),
    )


def _inherit_group_complement_prepositions(
    participants: tuple[CandidateArgument, ...],
    participant_preps: list[str],
) -> None:
    """Propagate ``with/to`` across a parser-flattened coordination group."""
    for index, (argument, prep) in enumerate(zip(participants, participant_preps)):
        if prep:
            continue
        argument_indices = set(argument.projection_protein_indices)
        for sibling, sibling_prep in zip(participants, participant_preps):
            if sibling_prep not in _RELATION_COMPLEMENTS:
                continue
            sibling_indices = set(sibling.projection_protein_indices)
            same_group = bool(argument.group) and (
                argument.group.get("all_protein_indices")
                == (sibling.group or {}).get("all_protein_indices")
            )
            if same_group or (argument_indices and argument_indices <= sibling_indices):
                participant_preps[index] = sibling_prep
                break


__all__ = [
    "ADJUNCT_CONTROL_RULE_ID",
    "SUBJECT_CONTROL_RULE_ID",
    "bind_nested_controller",
]
