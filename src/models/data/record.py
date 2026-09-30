from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.models.candidate import CandidateEvent
from src.models.artifact_contract import CURRENT_ARTIFACT_CONTRACT


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_pairs(pairs: list[tuple[int, int]] | list[list[int]] | None) -> list[list[int]]:
    if not pairs:
        return []

    normalized = {
        tuple(sorted((int(pair[0]), int(pair[1]))))
        for pair in pairs
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    }
    return [list(pair) for pair in sorted(normalized)]


def _normalize_offsets(offsets: list[tuple[int, int]] | list[list[int]] | None) -> list[tuple[int, int]]:
    if not offsets:
        return []

    normalized: list[tuple[int, int]] = []
    for item in offsets:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        start, end = int(item[0]), int(item[1])
        if start < 0 or end < start:
            continue
        normalized.append((start, end))
    return normalized


@dataclass
class Record:
    """Serializable sentence-level contract between pipeline stages."""

    sentence_id: str
    sentence_text: str
    target_count: int
    events: list[CandidateEvent] = field(default_factory=list)
    events_json: list[dict[str, Any]] = field(default_factory=list)
    predicted_pairs: list[tuple[int, int]] = field(default_factory=list)
    gold_pairs: list[tuple[int, int]] | None = None
    token_offsets: list[tuple[int, int]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    run_meta: dict[str, Any] = field(default_factory=dict)
    record_version: str = CURRENT_ARTIFACT_CONTRACT.record_version

    def __post_init__(self) -> None:
        if self.target_count < 0:
            raise ValueError("target_count must be >= 0")

        self.predicted_pairs = [tuple(pair) for pair in _normalize_pairs(self.predicted_pairs)]
        if self.gold_pairs is not None:
            self.gold_pairs = [tuple(pair) for pair in _normalize_pairs(self.gold_pairs)]
        self.token_offsets = _normalize_offsets(self.token_offsets)

        for a, b in self.predicted_pairs:
            if a >= self.target_count or b >= self.target_count:
                raise ValueError("predicted_pairs contains index out of range for target_count")
        if self.gold_pairs is not None:
            for a, b in self.gold_pairs:
                if a >= self.target_count or b >= self.target_count:
                    raise ValueError("gold_pairs contains index out of range for target_count")

        if self.events and not self.events_json:
            self.events_json = [event.to_dict() for event in self.events]

        self.provenance = {
            "sentence_id": self.sentence_id,
            "sentence_text": self.sentence_text,
            "target_count": self.target_count,
            **(self.provenance or {}),
        }
        self.artifacts = dict(self.artifacts or {})
        self.artifacts.setdefault("parse", {})
        self.artifacts["parse"].setdefault("token_offsets", [list(pair) for pair in self.token_offsets])
        self.events_json = list(self.events_json or [])

        self.run_meta = dict(self.run_meta or {})
        self.run_meta.setdefault("created_at", _utc_now())
        self.run_meta.setdefault("updated_at", self.run_meta["created_at"])

    def update_run_meta(self, *, mode: str, config: dict[str, Any], timestamp: str | None = None) -> None:
        ts = timestamp or _utc_now()
        self.run_meta["mode"] = mode
        self.run_meta["config"] = config
        self.run_meta["updated_at"] = ts

    def to_json(self) -> dict[str, Any]:
        return {
            "record_version": self.record_version,
            "artifact_contract": CURRENT_ARTIFACT_CONTRACT.to_dict(),
            "sentence_id": self.sentence_id,
            "sentence_text": self.sentence_text,
            "target_count": self.target_count,
            "token_offsets": [list(pair) for pair in self.token_offsets],
            "predicted_pairs": _normalize_pairs(self.predicted_pairs),
            "gold_pairs": _normalize_pairs(self.gold_pairs) if self.gold_pairs is not None else None,
            "events": (
                self.events_json
                if self.events_json
                else [event.to_dict() for event in self.events]
            ),
            "provenance": self.provenance,
            "artifacts": self.artifacts,
            "run_meta": self.run_meta,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "Record":
        if not isinstance(payload, dict):
            raise ValueError("Record payload must be a dict")

        required_keys = {
            "sentence_id",
            "sentence_text",
            "target_count",
            "token_offsets",
            "predicted_pairs",
            "events",
            "provenance",
            "artifacts",
            "run_meta",
        }
        missing = sorted(required_keys - set(payload))
        if missing:
            raise ValueError(f"Record payload missing required keys: {', '.join(missing)}")

        record_version = str(payload.get("record_version", ""))
        if record_version and not CURRENT_ARTIFACT_CONTRACT.supports_record(record_version):
            raise ValueError(f"Unsupported Record version: {record_version}")
        declared_contract = payload.get("artifact_contract") or {}
        if declared_contract and not isinstance(declared_contract, dict):
            raise ValueError("Record artifact_contract must be a dict")
        if declared_contract:
            if not CURRENT_ARTIFACT_CONTRACT.supports_contract(
                declared_contract.get("contract_version")
            ):
                raise ValueError("Unsupported artifact contract version")
            declared_record_version = str(declared_contract.get("record_version") or "")
            if declared_record_version and declared_record_version != record_version:
                raise ValueError(
                    "Record version conflicts with artifact_contract.record_version"
                )

        event_rows = list(payload.get("events") or [])
        canonical_events = [
            CandidateEvent.from_dict(row)
            for row in event_rows
            if isinstance(row, dict) and "event_type" in row and "event_id" in row
        ]
        return cls(
            sentence_id=str(payload["sentence_id"]),
            sentence_text=str(payload["sentence_text"]),
            target_count=int(payload["target_count"]),
            events=canonical_events,
            events_json=event_rows,
            predicted_pairs=[tuple(pair) for pair in payload.get("predicted_pairs", [])],
            gold_pairs=[tuple(pair) for pair in payload.get("gold_pairs", [])] if payload.get("gold_pairs") is not None else None,
            token_offsets=[tuple(pair) for pair in payload.get("token_offsets", [])],
            provenance=dict(payload.get("provenance") or {}),
            artifacts=dict(payload.get("artifacts") or {}),
            run_meta=dict(payload.get("run_meta") or {}),
            record_version=record_version or CURRENT_ARTIFACT_CONTRACT.record_version,
        )
