"""Explicit extraction resources; reviewed catalogs are intentionally absent."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from src.system_rules import SYSTEM_RULES_PATH, system_rule

from src.lexicons import LinguisticLexicon, load_linguistic_lexicon


@dataclass(frozen=True)
class RuntimeResources:
    lexicon: LinguisticLexicon
    rule_toggles: dict

    def match_lexicon(self, category: str, values, *, strictly: bool = False) -> bool:
        return self.lexicon.matches(category, values, all_values=strictly)

    def map_lexicon(self, category: str, value: str) -> str | None:
        return self.lexicon.map(category, value)

    def rule_enabled(self, rule_id: str) -> bool:
        """Return whether a stable runtime rule id is enabled for this run."""
        payload = self.rule_toggles or {}
        disabled_ids = {str(value) for value in payload.get("disabled_rule_ids") or ()}
        if rule_id in disabled_ids:
            return False
        disabled_namespaces = tuple(
            str(value).rstrip(".")
            for value in payload.get("disabled_namespaces") or ()
            if str(value).strip()
        )
        return not any(
            rule_id == namespace or rule_id.startswith(f"{namespace}.")
            for namespace in disabled_namespaces
        )

    def rule_toggle_manifest(self) -> dict:
        payload = dict(self.rule_toggles or {})
        return {
            "schema": payload.get("schema", "runtime_rule_toggles_v1"),
            "source": payload.get("_source", ""),
            "default": payload.get("default", "enabled"),
            "disabled_rule_ids": list(payload.get("disabled_rule_ids") or ()),
            "disabled_namespaces": list(payload.get("disabled_namespaces") or ()),
        }

def load_runtime_resources() -> RuntimeResources:
    root = Path(__file__).resolve().parents[1]
    toggles = deepcopy(system_rule("runtime_rule_toggles"))
    toggles["_source"] = SYSTEM_RULES_PATH.as_posix() + "#runtime_rule_toggles"
    return RuntimeResources(
        lexicon=load_linguistic_lexicon(root / "data" / "lexicons"),
        rule_toggles=toggles,
    )


__all__ = ["RuntimeResources", "load_runtime_resources"]
