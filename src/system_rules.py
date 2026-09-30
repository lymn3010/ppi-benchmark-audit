"""Typed access to ``data/system_rules.yaml`` (corpus-independent rules only)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


SYSTEM_RULES_PATH = Path(__file__).resolve().parents[1] / "data" / "system_rules.yaml"


@lru_cache(maxsize=1)
def load_system_rules() -> dict[str, Any]:
    with SYSTEM_RULES_PATH.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if payload.get("meta", {}).get("format") != "system_rules_v1":
        raise ValueError(f"Unsupported system rule registry: {SYSTEM_RULES_PATH}")
    return payload


def system_rule(path: str) -> Any:
    value: Any = load_system_rules()
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing system rule: {path}")
        value = value[key]
    return value


def rule_tuple(path: str) -> tuple[str, ...]:
    value = system_rule(path)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"System rule {path} must be a list of strings")
    return tuple(value)


def rule_frozenset(path: str) -> frozenset[str]:
    return frozenset(rule_tuple(path))


def rule_int(path: str) -> int:
    value = system_rule(path)
    if not isinstance(value, int):
        raise TypeError(f"System rule {path} must be an integer")
    return value
