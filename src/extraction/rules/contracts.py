"""Stable IDs and serializable traces for applied extraction rules."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from typing import Any

from src.models.candidate import CandidateEvent
from src.models.semantic_vocabulary import RuleId


RULE_TRACE_KEY = "extraction_rule_trace"
_DECISIONS = frozenset({"applied", "blocked", "not_applicable"})


@dataclass(frozen=True)
class RuleApplication:
    """One explicit decision made by a versioned extraction rule."""

    rule_id: str
    stage: str
    decision: str = "applied"
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        RuleId.parse(self.rule_id)
        if not self.stage.strip():
            raise ValueError("Extraction rule application requires a stage")
        if self.decision not in _DECISIONS:
            raise ValueError(f"Unsupported extraction-rule decision: {self.decision}")
        try:
            json.dumps(self.evidence, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("Extraction-rule evidence must be JSON-serializable") from exc

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "rule_id": self.rule_id,
            "stage": self.stage,
            "decision": self.decision,
        }
        if self.reason:
            result["reason"] = self.reason
        if self.evidence:
            result["evidence"] = dict(self.evidence)
        return result


def append_rule_application(
    event: CandidateEvent,
    application: RuleApplication,
) -> CandidateEvent:
    """Return an event with one additive, JSON-safe rule trace entry."""
    custom_values = dict(event.custom_values)
    trace = list(custom_values.get(RULE_TRACE_KEY, ()))
    trace.append(application.to_dict())
    custom_values[RULE_TRACE_KEY] = trace
    return replace(event, custom_values=custom_values)


__all__ = ["RULE_TRACE_KEY", "RuleApplication", "append_rule_application"]
