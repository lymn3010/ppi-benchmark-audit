"""Run cache key: data, effective configuration, code phases and catalog source."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


IDENTITY_SCHEMA_VERSION = 1
_HASHABLE_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".json", ".txt", ".csv", ".tsv"})


def canonical_json(value: Any) -> str:
    """Stable JSON representation used by all identity fingerprints."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_file(path: str | Path) -> str:
    """Full SHA256 of one file, or an explicit marker when it is absent."""
    source = Path(path)
    if not source.is_file():
        return "missing"
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_path(path: str | Path) -> str:
    """Full deterministic SHA256 for a file or supported-text directory tree."""
    source = Path(path)
    digest = hashlib.sha256()
    if source.is_file():
        digest.update(source.name.encode("utf-8"))
        digest.update(source.read_bytes())
    elif source.is_dir():
        for child in sorted(p for p in source.rglob("*") if p.is_file()):
            if child.suffix.lower() not in _HASHABLE_SUFFIXES:
                continue
            digest.update(child.relative_to(source).as_posix().encode("utf-8"))
            digest.update(child.read_bytes())
    else:
        digest.update(b"missing")
        digest.update(str(source).encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class RunIdentity:
    """All semantic inputs that determine whether a run may be reused."""

    mode: str
    dataset: str
    split: str
    dataset_sha256: str
    config: Mapping[str, Any]
    phase_hashes: Mapping[str, str]
    catalog_source: Mapping[str, Any] | None = None
    schema_version: int = IDENTITY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "dataset": self.dataset,
            "split": self.split,
            "dataset_sha256": self.dataset_sha256,
            "config": dict(self.config),
            "phase_hashes": dict(self.phase_hashes),
            "catalog_source": dict(self.catalog_source) if self.catalog_source else None,
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()[:12]


__all__ = [
    "IDENTITY_SCHEMA_VERSION",
    "RunIdentity",
    "canonical_json",
    "fingerprint_path",
    "sha256_file",
]
