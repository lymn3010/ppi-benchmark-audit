"""Version authority for serialized run and sentence artifacts."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactContract:
    contract_version: str
    record_version: str
    run_manifest_version: str
    candidate_event_version: str
    event_observation_version: str
    projection_plan_version: str

    def to_dict(self) -> dict[str, str]:
        return {
            "contract_version": self.contract_version,
            "record_version": self.record_version,
            "run_manifest_version": self.run_manifest_version,
            "candidate_event_version": self.candidate_event_version,
            "event_observation_version": self.event_observation_version,
            "projection_plan_version": self.projection_plan_version,
        }

    def supports_record(self, version: object) -> bool:
        return _major(version or self.record_version) == _major(self.record_version)

    def supports_contract(self, version: object) -> bool:
        return _major(version or self.contract_version) == _major(self.contract_version)

    def supports_projection_plan(self, version: object) -> bool:
        return _major(version or self.projection_plan_version) == _major(
            self.projection_plan_version
        )


def _major(version: object) -> str:
    return str(version or "").split(".", 1)[0]


CURRENT_ARTIFACT_CONTRACT = ArtifactContract(
    contract_version="2.0",
    record_version="4.0",
    run_manifest_version="2.0",
    candidate_event_version="2.0",
    event_observation_version="1.0",
    projection_plan_version="2.0",
)


__all__ = ["ArtifactContract", "CURRENT_ARTIFACT_CONTRACT"]
