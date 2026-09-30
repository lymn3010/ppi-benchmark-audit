"""Protein reachability through nominal dependency structure."""
from __future__ import annotations
from src.parsing.syntax import ParsedSentence, SyntaxNode
from .roles import BINARY_CASE_MARKERS

def has_binary_case_marker( head: SyntaxNode, parsed: ParsedSentence) -> bool:
    from .node_helpers import case_lemma_n
    from .roles import DEP_NMOD
    for c in parsed.children(head):
        if c.dep != DEP_NMOD:
            continue
        if case_lemma_n(c, parsed) in BINARY_CASE_MARKERS:
            return True
    return False


def nominal_has_protein(nmod: SyntaxNode, parsed: ParsedSentence) -> bool:
    """True if a protein is reachable through nominal/group structure (neutral version)."""
    from .node_helpers import node_protein_indices
    from .roles import SAME_NP_TRAVERSAL_DEPS
    seen: set[int] = set()
    pending = [nmod]
    while pending:
        token = pending.pop()
        if token.i in seen:
            continue
        seen.add(token.i)
        if node_protein_indices(token):
            return True
        pending.extend(
            child for child in parsed.children(token)
            if child.dep in SAME_NP_TRAVERSAL_DEPS
        )
    return False


def nominal_has_direct_protein(nmod: SyntaxNode, parsed: ParsedSentence) -> bool:
    """True if a protein is reachable without crossing an nmod edge."""
    from .node_helpers import node_protein_indices
    from .roles import SAME_NP_TRAVERSAL_DEPS, DEP_NMOD
    seen: set[int] = set()
    pending = [nmod]
    while pending:
        token = pending.pop()
        if token.i in seen:
            continue
        seen.add(token.i)
        if node_protein_indices(token):
            return True
        pending.extend(
            child for child in parsed.children(token)
            if child.dep in SAME_NP_TRAVERSAL_DEPS and child.dep != DEP_NMOD
        )
    return False
