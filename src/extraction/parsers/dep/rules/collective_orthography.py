"""Orthographic member topology scoped by a collective-state head."""
from __future__ import annotations

from src.parsing.syntax import ParsedSentence


COLLECTIVE_SLASH_MEMBER_RULE_ID = (
    "extraction.dep_collective_slash_member_pair@1"
)

_COLLECTIVE_ANCESTOR_DEPS = frozenset({
    "amod", "appos", "compound", "conj", "dep",
})


def collective_slash_member_pairs(
    parsed: ParsedSentence,
    collective_head_i: int,
) -> tuple[tuple[int, int], ...]:
    """Adjacent ``PROTEINx/PROTEINy`` pairs owned by one collective NP (no all-vs-all)."""
    nodes = parsed.nodes
    pairs: list[tuple[int, int]] = []
    for index in range(len(nodes) - 2):
        left, separator, right = nodes[index:index + 3]
        if not left.protein_indices or separator.text != "/" or not right.protein_indices:
            continue
        # ``P0-P1/P2`` means P0-P1 or P0-P2, not a P1-P2 edge.
        if (
            (
                index >= 2
                and nodes[index - 1].text in {"-", "–", "—"}
                and nodes[index - 2].protein_indices
            )
            or (
                index >= 1
                and nodes[index - 1].protein_indices
                and nodes[index - 1].text.rstrip().endswith(("-", "–", "—"))
            )
        ):
            continue
        if not _reaches_collective_head(parsed, left.i, collective_head_i):
            continue
        if not _reaches_collective_head(parsed, right.i, collective_head_i):
            continue
        pairs.append((left.i, right.i))
    return tuple(dict.fromkeys(pairs))


def _reaches_collective_head(
    parsed: ParsedSentence,
    node_i: int,
    collective_head_i: int,
) -> bool:
    current = parsed.node(node_i)
    seen: set[int] = set()
    for _ in range(8):
        if current.i == collective_head_i:
            return True
        # Do not climb through another coordinated lexical head.
        if (
            current.pos in {"NOUN", "PROPN"}
            and not current.protein_indices
            and (current.lemma or "").lower() != "protein"
        ):
            return False
        if current.i in seen or current.dep not in _COLLECTIVE_ANCESTOR_DEPS:
            return False
        seen.add(current.i)
        if current.head_i < 0 or current.head_i >= len(parsed.nodes):
            return False
        current = parsed.node(current.head_i)
    return False


__all__ = ["COLLECTIVE_SLASH_MEMBER_RULE_ID", "collective_slash_member_pairs"]
