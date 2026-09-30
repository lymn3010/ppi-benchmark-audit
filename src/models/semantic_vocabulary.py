"""Typed semantic vocabulary shared across event, projection, and artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Construction(str, Enum):
    """Known surface/construction labels; unknown labels remain lossless strings."""

    UNKNOWN = "unknown"
    VERBAL = "verbal"
    PASSIVE = "passive"
    NOMINALIZED = "nominalized"
    COMPOUND_STATE = "compound_state"
    NESTED = "nested"
    HYPHEN_PARTICIPLE = "hyphen_participle"
    VERBAL_PATH = "verbal_path"
    NOMINAL_COMPOUND = "nominal_compound"
    NOMINAL_PREP_PATH = "nominal_prep_path"
    NESTED_DOBJ = "nested_dobj"
    BARE_PATH = "bare_path"

    @classmethod
    def normalize(cls, value: object, *, default: str = "") -> str:
        """Normalize spelling without erasing a forward-compatible label."""
        text = (
            value.value
            if isinstance(value, cls)
            else str(value or default).strip().lower()
        )
        return text

    @classmethod
    def is_known(cls, value: object) -> bool:
        text = cls.normalize(value)
        return text in {item.value for item in cls}


class SemanticSlot(str, Enum):
    """Canonical event-argument slots, separate from dependency roles."""

    SOURCES = "sources"
    TARGETS = "targets"
    PARTICIPANTS = "participants"
    ENTITIES_A = "entities_a"
    ENTITIES_B = "entities_b"
    CONTEXTS = "contexts"
    EVIDENCE = "evidence"
    UNKNOWN = "unknown"

    @classmethod
    def normalize(cls, value: object) -> str:
        text = value.value if isinstance(value, cls) else str(value or "").strip().lower()
        return text if text in {item.value for item in cls} else cls.UNKNOWN.value


class ProjectionRuleId(str, Enum):
    """Stable compatibility reason codes at the projection boundary."""

    BLOCKED_ASSERTION = "blocked_assertion"
    BLOCKED_PREDICATE_NEGATION = "blocked_predicate_negation"
    BLOCKED_PATTERN_SUPPORT = "blocked_pattern_support"
    BLOCKED_NESTED_WITHOUT_GROUP_SEMANTICS = "blocked_nested_without_group_semantics"
    APPLIED_GROUP_INTERNAL_TOPOLOGY = "applied_group_internal_topology"
    APPLIED_RELATION_PROJECTION = "applied_relation_projection"
    APPLIED_RELATIONAL_ROLE_NOMINAL = "applied_relational_role_nominal"
    RELATIONAL_REVIEW_ONLY = "relational_review_only"
    NO_PROJECTABLE_PAIRS = "no_projectable_pairs"


class ProjectionStatus(str, Enum):
    APPLIED = "applied"
    NOT_APPLIED = "not_applied"
    REVIEW_ONLY = "review_only"


@dataclass(frozen=True)
class RuleId:
    """Namespaced, versioned rule identity independent of display strings."""

    namespace: str
    name: str
    version: int = 1

    def __post_init__(self) -> None:
        namespace = self.namespace.strip().lower()
        name = self.name.strip().lower()
        if not namespace or not name:
            raise ValueError("RuleId requires non-empty namespace and name")
        if self.version < 1:
            raise ValueError("RuleId.version must be >= 1")
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "name", name)

    def __str__(self) -> str:
        return f"{self.namespace}.{self.name}@{self.version}"

    @classmethod
    def parse(cls, value: str) -> "RuleId":
        body, separator, raw_version = str(value).rpartition("@")
        if not separator:
            body, raw_version = str(value), "1"
        namespace, separator, name = body.partition(".")
        if not separator:
            raise ValueError(f"Invalid RuleId: {value!r}")
        return cls(namespace=namespace, name=name, version=int(raw_version))


def projection_rule_id(reason: str | ProjectionRuleId) -> RuleId:
    name = reason.value if isinstance(reason, ProjectionRuleId) else str(reason)
    return RuleId("projection", name, 1)


__all__ = [
    "Construction",
    "ProjectionRuleId",
    "ProjectionStatus",
    "RuleId",
    "SemanticSlot",
    "projection_rule_id",
]
