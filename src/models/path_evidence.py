"""Shared lifecycle contract for mined and runtime lexical-path evidence."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PathEvidenceStage(str, Enum):
    MINED = "mined"
    REVIEWED = "reviewed"
    ENABLED = "enabled"
    APPLIED = "applied"


@dataclass(frozen=True)
class PathEvidenceLifecycle:
    source: str
    review_status: str
    stage: PathEvidenceStage
    runtime_eligible: bool

    def to_dict(self) -> dict:
        return {
            "schema": "path_evidence_lifecycle_v1",
            "source": self.source,
            "review_status": self.review_status,
            "stage": self.stage.value,
            "runtime_eligible": self.runtime_eligible,
        }


def classify_path_evidence(source: object, review_status: object) -> PathEvidenceLifecycle:
    """Apply the sole gate from catalog evidence to runtime matching."""
    source_text = str(source or "mined").strip().lower()
    status_text = str(review_status or "candidate").strip().lower()
    eligible = source_text == "reviewed" and status_text == "enabled"
    if eligible:
        stage = PathEvidenceStage.ENABLED
    elif source_text == "reviewed":
        stage = PathEvidenceStage.REVIEWED
    else:
        stage = PathEvidenceStage.MINED
    return PathEvidenceLifecycle(source_text, status_text, stage, eligible)


def mined_path_lifecycle() -> dict:
    return classify_path_evidence("mined", "candidate").to_dict()


__all__ = [
    "PathEvidenceLifecycle",
    "PathEvidenceStage",
    "classify_path_evidence",
    "mined_path_lifecycle",
]
