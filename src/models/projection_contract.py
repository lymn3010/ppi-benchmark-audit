"""Canonical, serializable contracts for pair projection planning and decisions."""
from __future__ import annotations

from dataclasses import dataclass, field

from .artifact_contract import CURRENT_ARTIFACT_CONTRACT


def _pair(value: tuple[int, int] | list[int]) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError("Protein pair must contain exactly two indices")
    first, second = int(value[0]), int(value[1])
    if first < 0 or second < 0 or first == second:
        raise ValueError("Protein pair must contain two distinct non-negative indices")
    return (first, second) if first < second else (second, first)


@dataclass(frozen=True)
class PairProposal:
    id: str
    pair: tuple[int, int]
    source_event_id: str
    projection_mode: str
    rule_id: str
    semantic_slots: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    provenance: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair", _pair(self.pair))

    def to_dict(self) -> dict:
        result = {
            "id": self.id,
            "pair": list(self.pair),
            "source_event_id": self.source_event_id,
            "projection_mode": self.projection_mode,
            "rule_id": self.rule_id,
            "semantic_slots": list(self.semantic_slots),
            "evidence_refs": list(self.evidence_refs),
        }
        if self.provenance:
            result["provenance"] = dict(self.provenance)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "PairProposal":
        return cls(
            id=str(data["id"]),
            pair=tuple(data["pair"]),
            source_event_id=str(data.get("source_event_id", "")),
            projection_mode=str(data.get("projection_mode", "")),
            rule_id=str(data.get("rule_id", "")),
            semantic_slots=tuple(data.get("semantic_slots", ())),
            evidence_refs=tuple(data.get("evidence_refs", ())),
            provenance=dict(data.get("provenance", {})),
        )


@dataclass(frozen=True)
class PairDecision:
    proposal_ref: str
    pair: tuple[int, int]
    status: str
    reason_code: str
    emitted: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair", _pair(self.pair))

    def to_dict(self) -> dict:
        return {
            "proposal_ref": self.proposal_ref,
            "pair": list(self.pair),
            "status": self.status,
            "reason_code": self.reason_code,
            "emitted": self.emitted,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PairDecision":
        return cls(
            proposal_ref=str(data["proposal_ref"]),
            pair=tuple(data["pair"]),
            status=str(data.get("status", "")),
            reason_code=str(data.get("reason_code", "")),
            emitted=bool(data.get("emitted", False)),
        )


@dataclass(frozen=True)
class ProjectionPlan:
    id: str
    sentence_id: str
    source_event_id: str
    event_index: int
    event_type: str
    application_status: str
    reason_code: str
    rule_id: str
    projection_mode: str
    proposals: tuple[PairProposal, ...] = ()
    decisions: tuple[PairDecision, ...] = ()
    final_pairs: tuple[tuple[int, int], ...] = ()
    provenance: dict = field(default_factory=dict, compare=False, hash=False)
    schema_version: str = CURRENT_ARTIFACT_CONTRACT.projection_plan_version

    def __post_init__(self) -> None:
        if not CURRENT_ARTIFACT_CONTRACT.supports_projection_plan(self.schema_version):
            raise ValueError(f"Unsupported ProjectionPlan version: {self.schema_version}")
        object.__setattr__(self, "final_pairs", tuple(sorted({_pair(p) for p in self.final_pairs})))

    def to_dict(self) -> dict:
        result = {
            "schema": "projection_plan",
            "schema_version": self.schema_version,
            "id": self.id,
            "sentence_id": self.sentence_id,
            "source_event_id": self.source_event_id,
            "event_index": self.event_index,
            "event_type": self.event_type,
            "application_status": self.application_status,
            "reason_code": self.reason_code,
            "rule_id": self.rule_id,
            "projection_mode": self.projection_mode,
            "proposals": [proposal.to_dict() for proposal in self.proposals],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "final_pairs": [list(pair) for pair in self.final_pairs],
        }
        if self.provenance:
            result["provenance"] = dict(self.provenance)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectionPlan":
        return cls(
            id=str(data["id"]),
            sentence_id=str(data.get("sentence_id", "")),
            source_event_id=str(data.get("source_event_id", "")),
            event_index=int(data.get("event_index", -1)),
            event_type=str(data.get("event_type", "")),
            application_status=str(data.get("application_status", "")),
            reason_code=str(data.get("reason_code", "")),
            rule_id=str(data.get("rule_id", "")),
            projection_mode=str(data.get("projection_mode", "")),
            proposals=tuple(PairProposal.from_dict(row) for row in data.get("proposals", ())),
            decisions=tuple(PairDecision.from_dict(row) for row in data.get("decisions", ())),
            final_pairs=tuple(tuple(row) for row in data.get("final_pairs", ())),
            provenance=dict(data.get("provenance", {})),
            schema_version=str(data.get("schema_version", CURRENT_ARTIFACT_CONTRACT.projection_plan_version)),
        )


__all__ = [
    "PairDecision",
    "PairProposal",
    "ProjectionPlan",
]
