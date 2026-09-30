"""Per-sentence cache of semantic groups from ``argument_group_n``."""
from __future__ import annotations

from src.parsing.syntax import ParsedSentence, SyntaxNode
from .semantic_groups import SemanticGroup, argument_group_n
from .node_helpers import conj_chain_n


class GroupIndex:
    """Memoized view of the sentence's semantic groups, keyed by head node.

    Operates purely on ParsedSentence/SyntaxNode.
    """

    def __init__(self, parsed: ParsedSentence) -> None:
        self._parsed = parsed
        self._by_head: dict[int, SemanticGroup] = {}
        self._members_by_head: dict[int, tuple[SyntaxNode, ...]] = {}

    def group_for(self, head: SyntaxNode) -> SemanticGroup:
        """The SemanticGroup anchored at ``head``, computed once and reused."""
        group = self._by_head.get(head.i)
        if group is None:
            group = argument_group_n(head, self._parsed)
            self._by_head[head.i] = group
        return group

    def conj_members(self, head: SyntaxNode) -> tuple[SyntaxNode, ...]:
        """Coordinate members reachable from ``head`` via ``conj``."""
        members = self._members_by_head.get(head.i)
        if members is None:
            members = tuple(conj_chain_n(head, self._parsed))
            self._members_by_head[head.i] = members
        return members
