"""Conservative parser-neutral reference rule guards."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.parsing import ParsedSentence, SyntaxNode


IS_A_NOMINAL_REFERENCE_RULE_ID = "reference.is_a_nominal_antecedent@1"
IS_A_NOMINAL_REFERENCE_KIND = "is_a_nominal_reference"

NOMINAL_REFERENCE_DETERMINERS = frozenset({"this", "that", "the", "these", "those"})
NOMINAL_DECLARATION_POS = frozenset({"NOUN", "PROPN"})
IS_A_SUBJECT_DEPS = frozenset({"nsubj", "nsubjpass"})


def nominal_reference_key(node: SyntaxNode) -> str:
    """Return the lexical key used to pair an IS-A nominal with a later NP."""
    return (node.lemma or node.text or "").strip().lower()


def is_nominal_declaration_head(node: SyntaxNode) -> bool:
    """Whether a node can be the nominal side of a referable IS-A declaration."""
    return node.pos in NOMINAL_DECLARATION_POS and bool(nominal_reference_key(node))


def has_nominal_reference_determiner(
    parsed: ParsedSentence,
    node: SyntaxNode,
) -> bool:
    """Require an explicit definite/demonstrative marker on nominal anaphora."""
    return any(
        child.dep == "det"
        and (child.lemma or child.text or "").strip().lower()
        in NOMINAL_REFERENCE_DETERMINERS
        for child in parsed.children(node)
    )


def is_nominal_reference_candidate(
    parsed: ParsedSentence,
    node: SyntaxNode,
) -> bool:
    """Guard for a later nominal mention that may inherit IS-A referent ids."""
    if node.protein_indices or not is_nominal_declaration_head(node):
        return False
    if has_local_protein_anchor(parsed, node):
        return False
    return has_nominal_reference_determiner(parsed, node)


def has_local_protein_anchor(parsed: ParsedSentence, node: SyntaxNode) -> bool:
    """Whether the nominal already names local protein entities in its own NP."""
    for child in parsed.children(node):
        if child.dep in {"det", "case", "punct", "cc", "mark"}:
            continue
        if child.direct_protein_indices or child.protein_indices:
            return True
        if any(
            grandchild.direct_protein_indices or grandchild.protein_indices
            for grandchild in parsed.children(child)
            if grandchild.dep not in {"case", "punct", "cc", "mark"}
        ):
            return True
    if node.dep == "conj" and node.head_i >= 0:
        parent = parsed.node(node.head_i)
        if parent.direct_protein_indices or parent.protein_indices:
            return True
    return False


def is_entity_role_nominal(node: SyntaxNode) -> bool:
    """Whether a nominal can plausibly denote a protein/entity role."""
    key = nominal_reference_key(node)
    singular = key[:-1] if key.endswith("s") else key
    return singular in _reference_entity_nominals()


def is_transparent_owner_head(node: SyntaxNode) -> bool:
    """Whether an apposition host can pass its ``of`` owners as antecedents."""
    return nominal_reference_key(node) in _reference_transparent_owner_heads()


@lru_cache(maxsize=1)
def _reference_entity_nominals() -> frozenset[str]:
    return _read_lexicon("reference_entity_nominal.txt")


@lru_cache(maxsize=1)
def _reference_transparent_owner_heads() -> frozenset[str]:
    return _read_lexicon("reference_transparent_owner_head.txt")


def _read_lexicon(file_name: str) -> frozenset[str]:
    path = Path(__file__).resolve().parents[2] / "data" / "lexicons" / file_name
    with path.open(encoding="utf-8") as handle:
        return frozenset(
            line.strip().lower()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        )


__all__ = [
    "IS_A_NOMINAL_REFERENCE_KIND",
    "IS_A_NOMINAL_REFERENCE_RULE_ID",
    "IS_A_SUBJECT_DEPS",
    "has_local_protein_anchor",
    "is_nominal_declaration_head",
    "is_nominal_reference_candidate",
    "is_entity_role_nominal",
    "is_transparent_owner_head",
    "nominal_reference_key",
]
