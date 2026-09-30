"""Serializable, parser-neutral candidate event model."""
from .candidate_event import (
    AssertionStatus,
    EventType,
    SemanticEventClass,
    RealizationChannel,
    Span,
    ReferentLink,
    CandidateArgument,
    CandidateEvent,
    INTERACTION_DIRECTED_SLOTS,
    INTERACTION_UNDIRECTED_SLOTS,
    RELATIONAL_SLOTS,
    STATE_SLOTS,
)
from .candidate_relation import CandidateRelation, CandidateRelationLedger

__all__ = [
    "EventType",
    "SemanticEventClass",
    "RealizationChannel",
    "AssertionStatus",
    "Span",
    "ReferentLink",
    "CandidateArgument",
    "CandidateEvent",
    "CandidateRelation",
    "CandidateRelationLedger",
    "INTERACTION_DIRECTED_SLOTS",
    "INTERACTION_UNDIRECTED_SLOTS",
    "RELATIONAL_SLOTS",
    "STATE_SLOTS",
]
