"""Structural assertion-scope boundaries for dependency events."""
from __future__ import annotations


ADVCL_SCOPE_BARRIER_RULE_ID = "extraction.dep_advcl_assertion_barrier@1"
NESTED_NOMINAL_LOCAL_SCOPE_RULE_ID = "extraction.dep_nested_nominal_local_assertion@1"


def include_parent_in_negation_search(
    *,
    dependency: str,
    parent_pos: str,
    has_explicit_subject: bool,
) -> bool:
    """Whether a parent's negation scopes over this token.

    Clausal subjects and finite/adverbial clauses do not inherit; adjectival control does.
    """
    if dependency in {"nsubj", "nsubjpass", "nsubj:pass"}:
        return False
    if (
        dependency == "advcl"
        and parent_pos.upper() != "ADJ"
        and has_explicit_subject
    ):
        return False
    return True


__all__ = [
    "ADVCL_SCOPE_BARRIER_RULE_ID",
    "NESTED_NOMINAL_LOCAL_SCOPE_RULE_ID",
    "include_parent_in_negation_search",
]
