"""Hash each pipeline stage so saved runs identify the code they used."""
from __future__ import annotations

import hashlib
from pathlib import Path

# Project root is three levels up from this file (src/pipeline/phase_hash.py)
_PROJECT_ROOT = Path(__file__).parent.parent.parent

# Phase label -> project-relative paths that define it.
PHASE_SOURCES: dict[str, list[Path]] = {
    "layer0_parse": [
        _PROJECT_ROOT / "src" / "parsing",
    ],
    "layer1_struct": [
        _PROJECT_ROOT / "src" / "extraction",
    ],
    "layer2_resources": [
        _PROJECT_ROOT / "src" / "lexicons",
        _PROJECT_ROOT / "src" / "catalogs",
        _PROJECT_ROOT / "src" / "runtime_resources.py",
    ],
    "layer3_pipeline": [
        _PROJECT_ROOT / "src" / "pipeline",
    ],
    "layer4_export": [
        _PROJECT_ROOT / "src" / "export",
        _PROJECT_ROOT / "src" / "analysis",
    ],
    "linguistic_data": [
        _PROJECT_ROOT / "data" / "lexicons",
    ],
    "reviewed_catalogs": [
        _PROJECT_ROOT / "data" / "catalogs",
    ],
    "semantic_models": [
        _PROJECT_ROOT / "src" / "models",
    ],
    "reference_projection": [
        _PROJECT_ROOT / "src" / "downstream",
    ],
    "system_rules_data": [
        _PROJECT_ROOT / "data" / "system_rules.yaml",
    ],
}

# Only hash these file types (skip .pyc, __pycache__, large binaries, etc.)
_HASHABLE_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".json", ".txt", ".csv"})


def _hash_file(path: Path, h: "hashlib._Hash") -> None:
    """Feed one file's content (and relative path) into the hasher."""
    h.update(path.name.encode())
    h.update(path.read_bytes())


def _hash_path(path: Path) -> str:
    """SHA256 of a single file or a directory tree; returns 12-char hex prefix."""
    h = hashlib.sha256()
    if path.is_file():
        if path.suffix in _HASHABLE_SUFFIXES:
            _hash_file(path, h)
    elif path.is_dir():
        # Sort for determinism across OS
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix in _HASHABLE_SUFFIXES:
                # Include relative path so renames change the hash
                h.update(child.relative_to(path).as_posix().encode())
                _hash_file(child, h)
    return h.hexdigest()[:12]


def _hash_paths(paths: list[Path]) -> str:
    """Combine hashes from multiple paths into one 12-char fingerprint."""
    h = hashlib.sha256()
    for path in paths:
        # Hash each sub-path independently so adding/removing a path changes result
        h.update(_hash_path(path).encode())
    return h.hexdigest()[:12]


def compute_phase_hashes() -> dict[str, str]:
    """Return source fingerprints; unreadable files raise an I/O error."""
    return {phase: _hash_paths(paths) for phase, paths in PHASE_SOURCES.items()}
