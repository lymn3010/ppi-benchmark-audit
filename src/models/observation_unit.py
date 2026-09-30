"""Vocabulary for the denominator represented by one statistical row/count."""
from __future__ import annotations

from enum import Enum


class ObservationUnit(str, Enum):
    """Do not compare proportions whose units differ without an explicit model."""

    EVENT_OCCURRENCE = "event_occurrence"
    PROPAGATED_PAIR = "propagated_pair"
    CANDIDATE_PATTERN_FIRING = "candidate_pattern_firing"
    BOUNDED_SENTENCE_PAIR = "bounded_sentence_pair"
    SEMANTIC_EVENT = "semantic_event"


__all__ = ["ObservationUnit"]
