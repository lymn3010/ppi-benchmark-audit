"""Measurement contracts for lexical observation reports."""
from __future__ import annotations

from dataclasses import dataclass

from src.models.observation_unit import ObservationUnit

SCORE_TRIGGER_PRECISION = "trigger_precision"
SCORE_PROPAGATION_PRECISION = "propagation_precision"


@dataclass(frozen=True)
class ObservationCategory:
    name: str
    score_mode: str
    observation_unit: str
    success_definition: str
    reload_count_column: str = "correct"
    uses_precision_threshold: bool = True

    @property
    def is_trigger_scored(self) -> bool:
        return self.score_mode == SCORE_TRIGGER_PRECISION


_CATEGORIES = {
    "trigger_verbal": ObservationCategory(
        "trigger_verbal", SCORE_TRIGGER_PRECISION,
        ObservationUnit.EVENT_OCCURRENCE.value,
        "event occurrence emits at least one gold pair",
    ),
    "trigger_nominal": ObservationCategory(
        "trigger_nominal", SCORE_TRIGGER_PRECISION,
        ObservationUnit.EVENT_OCCURRENCE.value,
        "event occurrence emits at least one gold pair",
    ),
    "target_context": ObservationCategory(
        "target_context", SCORE_PROPAGATION_PRECISION,
        ObservationUnit.PROPAGATED_PAIR.value,
        "propagated pair is present in sentence gold pairs",
    ),
}


def category_metadata_for(category: str) -> ObservationCategory:
    try:
        return _CATEGORIES[category]
    except KeyError as exc:
        raise ValueError(f"Unknown observation category: {category}") from exc


__all__ = [
    "ObservationCategory", "SCORE_PROPAGATION_PRECISION",
    "SCORE_TRIGGER_PRECISION", "category_metadata_for",
]
