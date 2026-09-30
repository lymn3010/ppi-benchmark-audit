"""Index completed runs and find reusable results by configuration hash."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_FILENAME = "_registry.json"



def _registry_path(registry_root: Path) -> Path:
    return Path(registry_root) / _FILENAME


def _load(registry_root: Path) -> list[dict]:
    path = _registry_path(registry_root)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []  # corrupted or empty: treat as empty, never crash


def _save(registry_root: Path, entries: list[dict]) -> None:
    path = _registry_path(registry_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )



def config_hash(config: dict) -> str:
    """12-char SHA256 of a canonicalized config dict.

    Two configs that produce the same JSON (key-sorted) get the same hash.
    """
    canon = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canon.encode()).hexdigest()[:12]


def append_run(registry_root: Path, entry: dict) -> None:
    """Append one run entry to the registry.

    Safe to call after every successful run.  Never raises.
    """
    try:
        entries = _load(registry_root)
        entries.append(entry)
        _save(registry_root, entries)
    except Exception:
        pass  # registry write failure must never abort a successful run


def find_cached_run(
    registry_root: Path,
    dataset: str,
    split: str,
    mode: str,
    cfg_hash: str,
) -> dict | None:
    """Most recent existing run matching dataset, split, mode and config_hash, else None."""
    try:
        entries = _load(registry_root)
        matches = [
            e for e in entries
            if e.get("dataset") == dataset
            and e.get("split") == split
            and e.get("mode") == mode
            and e.get("config_hash") == cfg_hash
            and _run_exists(e)
        ]
        return matches[-1] if matches else None
    except Exception:
        return None


def get_latest_run(
    registry_root: Path,
    dataset: str,
    split: str,
    mode: str,
) -> dict | None:
    """Return the most recent completed run for a given dataset/split/mode.

    Ignores config; just finds the latest by append order.
    """
    try:
        entries = _load(registry_root)
        matches = [
            e for e in entries
            if e.get("dataset") == dataset
            and e.get("split") == split
            and e.get("mode") == mode
            and _run_exists(e)
        ]
        return matches[-1] if matches else None
    except Exception:
        return None


def list_runs(
    registry_root: Path,
    *,
    dataset: str | None = None,
    split: str | None = None,
    mode: str | None = None,
    only_existing: bool = True,
) -> list[dict]:
    """Return registry entries; ``only_existing`` skips missing run directories."""
    try:
        entries = _load(registry_root)
        result = []
        for e in entries:
            if dataset is not None and e.get("dataset") != dataset:
                continue
            if split is not None and e.get("split") != split:
                continue
            if mode is not None and e.get("mode") != mode:
                continue
            if only_existing and not _run_exists(e):
                continue
            result.append(e)
        return result
    except Exception:
        return []



def _run_exists(entry: dict) -> bool:
    """True if the run directory and its run.json both exist."""
    path_str = entry.get("path", "")
    if not path_str:
        return False
    run_dir = Path(path_str)
    return run_dir.is_dir() and (run_dir / "run.json").exists()
