"""Compound-owner repair rules for dependency argument construction."""
from __future__ import annotations

from src.parsing.syntax import ParsedSentence, SyntaxNode

from ..roles import DEP_COMPOUND


TRANSITIVE_COMPOUND_OWNER_RULE_ID = "extraction.dep_transitive_compound_owner@1"
_PROTEIN_COMPOUND_POS = frozenset({"NOUN", "PROPN"})


def _is_protein_compound_node(node: SyntaxNode) -> bool:
    from ..node_helpers import node_protein_indices

    return bool(node_protein_indices(node)) and node.pos in _PROTEIN_COMPOUND_POS


def _has_coordinate_separator_between(
    left: SyntaxNode,
    right: SyntaxNode,
    parsed: ParsedSentence,
) -> bool:
    """True when the protein markers are joined by slash-style coordination."""
    if left.char_end <= right.char_start:
        gap = parsed.text[left.char_end:right.char_start]
    elif right.char_end <= left.char_start:
        gap = parsed.text[right.char_end:left.char_start]
    else:
        gap = ""
    return "/" in gap


def transitive_compound_protein_members(
    member: SyntaxNode,
    parsed: ParsedSentence,
    *,
    max_depth: int = 8,
) -> tuple[SyntaxNode, ...]:
    """Protein compounds nested under a compound owner (``P3 -> P4 -> operon``).

    Only compound edges; slashes are coordination, not ownership.
    """
    from ..node_helpers import children_by_dep_n

    if not _is_protein_compound_node(member):
        return ()
    seen: set[int] = set()
    pending: list[tuple[SyntaxNode, SyntaxNode, int]] = []
    for child in children_by_dep_n(member, parsed, DEP_COMPOUND):
        pending.append((member, child, 1))

    out: list[SyntaxNode] = []
    while pending:
        parent, node, depth = pending.pop(0)
        if node.i in seen or depth > max_depth:
            continue
        if (
            not _is_protein_compound_node(node)
            or _has_coordinate_separator_between(parent, node, parsed)
        ):
            continue
        seen.add(node.i)
        out.append(node)
        for child in children_by_dep_n(node, parsed, DEP_COMPOUND):
            pending.append((node, child, depth + 1))
    return tuple(out)


__all__ = [
    "TRANSITIVE_COMPOUND_OWNER_RULE_ID",
    "transitive_compound_protein_members",
]
