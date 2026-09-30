"""Rules for subject-only reciprocal predicate realizations."""
from __future__ import annotations


RECIPROCAL_PURPOSE_SUBJECT_RULE_ID = "extraction.dep_reciprocal_subject_purpose@1"
SUBJECT_ONLY_RECIPROCITY_GATE_RULE_ID = (
    "extraction.dep_subject_only_reciprocity_gate@1"
)


def allow_subject_only_participants(
    *,
    lexically_undirected: bool,
    has_purpose_clause: bool,
) -> bool:
    """License subject-only participant projection only for reciprocal predicates."""
    return lexically_undirected


__all__ = [
    "RECIPROCAL_PURPOSE_SUBJECT_RULE_ID",
    "SUBJECT_ONLY_RECIPROCITY_GATE_RULE_ID",
    "allow_subject_only_participants",
]
