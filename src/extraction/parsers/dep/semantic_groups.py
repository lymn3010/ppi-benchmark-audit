"""Build coordination and ownership groups; ProjectionPlanner decides pairs."""

from __future__ import annotations
from src.parsing.syntax import ParsedSentence, SyntaxNode

from dataclasses import dataclass

from src.models.candidate.group_alignment import align_group_provenance

from .morphology import looks_complex
from .roles import DEP_APPOS, DEP_COMPOUND, DEP_CONJ, DEP_NMOD, SAME_NP_PROTEIN_DEPS


# Group kinds (plain strings, parser-neutral).
KIND_COORDINATION = "coordination"   # "P0 and P1", "[P0 P1] and [P2 P3]"
KIND_COMPOUND = "compound"           # "P0 receptor" -> single entity, P0 is owner
KIND_APPOSITION = "apposition"       # "P0, the P1 kinase" -> coreferent (alias)
KIND_ACCOMPANIMENT = "accompaniment" # "P0 together with P1" -> joint co-agents


@dataclass(frozen=True)
class GroupMember:
    """One coordinate slot of a SemanticGroup.

    Head proteins carry the role; same-NP modifier proteins are kept as evidence only.
    """
    head_index: int
    head_protein_indices: tuple[int, ...]
    modifier_protein_indices: tuple[int, ...] = ()

    @property
    def protein_indices(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.head_protein_indices)
                            | set(self.modifier_protein_indices)))

    @property
    def role_protein_indices(self) -> tuple[int, ...]:
        """Proteins that take this member's role: the head protein, else its modifiers."""
        if self.head_protein_indices:
            return self.head_protein_indices
        return self.protein_indices

    def to_dict(self) -> dict:
        return {
            "head_index": self.head_index,
            "head_protein_indices": list(self.head_protein_indices),
            "modifier_protein_indices": list(self.modifier_protein_indices),
            "protein_indices": list(self.protein_indices),
            "role_protein_indices": list(self.role_protein_indices),
        }


@dataclass(frozen=True)
class OwnerRef:
    """One ownership link of an argument head, separate from coordination.

    Sources: compound, complex_conj, nmod:<prep>, apposition, hyphen_modifier.
    Accompaniment (``together with``) is never ownership.
    """
    head_index: int
    protein_indices: tuple[int, ...]
    source: str

    def to_dict(self) -> dict:
        return {
            "head_index": self.head_index,
            "protein_indices": list(self.protein_indices),
            "source": self.source,
        }


@dataclass(frozen=True)
class SemanticGroup:
    """Coordinate slots for one argument position; ``ownership`` is provenance only."""
    kind: str
    head_index: int
    members: tuple[GroupMember, ...]
    is_alternative: bool = False     # "or"/"/" coordination -> no positional reading
    coreferent: bool = False         # alias/apposition -> members are one entity
    case_marker: str = ""            # introducing preposition, provenance only
    ownership: tuple[OwnerRef, ...] = ()  # distinct OWNERSHIP facet (provenance only)

    @property
    def slot_count(self) -> int:
        return len(self.members)

    @property
    def all_protein_indices(self) -> tuple[int, ...]:
        out: set[int] = set()
        for m in self.members:
            out.update(m.protein_indices)
        return tuple(sorted(out))

    @property
    def role_protein_indices(self) -> tuple[int, ...]:
        out: set[int] = set()
        for m in self.members:
            out.update(m.role_protein_indices)
        return tuple(sorted(out))

    @property
    def member_internal_pairs(self) -> tuple[tuple[int, int], ...]:
        """Same-slot protein pairs kept as topology evidence; ``align`` does not emit them."""
        pairs: set[tuple[int, int]] = set()
        for m in self.members:
            proteins = list(m.protein_indices)
            for i, a in enumerate(proteins):
                for b in proteins[i + 1:]:
                    pairs.add((a, b) if a < b else (b, a))
        return tuple(sorted(pairs))

    def to_dict(self) -> dict:
        """Serializable provenance shape (for the future CandidateArgument field)."""
        return {
            "kind": self.kind,
            "head_index": self.head_index,
            "members": [m.to_dict() for m in self.members],
            "role_protein_indices": list(self.role_protein_indices),
            "all_protein_indices": list(self.all_protein_indices),
            "member_internal_pairs": [list(p) for p in self.member_internal_pairs],
            "is_alternative": self.is_alternative,
            "coreferent": self.coreferent,
            "case_marker": self.case_marker,
            "ownership": [o.to_dict() for o in self.ownership],
        }


@dataclass(frozen=True)
class AlignmentResult:
    """Alignment result: committed ``aligned`` pairs and exported ``cross`` alternatives."""
    aligned: tuple[tuple[int, int], ...]
    cross: tuple[tuple[int, int], ...]
    mode: str = ""                   # "diagonal" | "full"
    slots: tuple[int, int] = (0, 0)  # (source_slot_count, target_slot_count)

    @property
    def is_diagonal(self) -> bool:
        return self.mode == "diagonal"


# Alignment.

def align(source: SemanticGroup, target: SemanticGroup) -> AlignmentResult:
    """Align source and target groups.

    Diagonal only for equal arity >= 2, no or/alias, and internal slot topology;
    otherwise the full cross product as ``aligned``.
    """
    result = align_group_provenance(source.to_dict(), target.to_dict())
    if result is None:
        return AlignmentResult(aligned=(), cross=(), mode="", slots=(0, 0))
    return AlignmentResult(
        aligned=result.aligned,
        cross=result.cross,
        mode=result.mode,
        slots=result.slots,
    )


# Parser-neutral member and group builders.

def _member_protein_parts_n(member: SyntaxNode, parsed: ParsedSentence) -> tuple[tuple[int, ...], tuple[int, ...]]:
    from .node_helpers import node_protein_indices, conj_chain_n
    head_targets = tuple(sorted(set(node_protein_indices(member))))
    modifiers: set[int] = set()
    if member.dep == DEP_CONJ and looks_complex((member.lemma or member.text or "").lower()):
        head_head = parsed.head(member)
        if head_head is not None:
            modifiers.update(node_protein_indices(head_head))
    for child in parsed.children(member):
        if child.dep in SAME_NP_PROTEIN_DEPS:
            for same_np in conj_chain_n(child, parsed):
                modifiers.update(node_protein_indices(same_np))
    return head_targets, tuple(sorted(modifiers - set(head_targets)))


def _member_n(member: SyntaxNode, parsed: ParsedSentence) -> GroupMember:
    head_proteins, modifier_proteins = _member_protein_parts_n(member, parsed)
    return GroupMember(
        head_index=member.i,
        head_protein_indices=head_proteins,
        modifier_protein_indices=modifier_proteins,
    )


def build_coordination_group_n(anchor: SyntaxNode, parsed: ParsedSentence) -> SemanticGroup:
    from .node_helpers import conj_chain_n, conj_chain_has_alternative_n, conj_negation_excluded_n
    # Stanza marks "and not with" as cc; drop those negated conjuncts.
    members_tokens = sorted(
        (t for t in conj_chain_n(anchor, parsed)
         if not (t.dep == DEP_CONJ and conj_negation_excluded_n(t, parsed))),
        key=lambda t: t.i,
    )
    members = tuple(_member_n(t, parsed) for t in members_tokens)
    return SemanticGroup(
        kind=KIND_COORDINATION,
        head_index=anchor.i,
        members=members,
        is_alternative=conj_chain_has_alternative_n(anchor, parsed),
    )


def build_coordinated_compound_group_n(
    head: SyntaxNode,
    anchor: SyntaxNode,
    parsed: ParsedSentence,
) -> SemanticGroup:
    """Recover two slots by surface order when UD flattens ``P0 P1 and P2 P3 complexes``."""
    from .node_helpers import (
        children_by_dep_n,
        conj_chain_has_alternative_n,
        conj_chain_n,
        node_protein_indices,
    )

    anchors = sorted(conj_chain_n(anchor, parsed), key=lambda node: node.i)
    if len(anchors) < 2:
        return build_coordination_group_n(anchor, parsed)

    members = [_member_n(node, parsed) for node in anchors]
    represented = {
        protein
        for member in members
        for protein in member.protein_indices
    }
    extras = sorted(
        (
            node for node in children_by_dep_n(head, parsed, DEP_COMPOUND)
            if set(node_protein_indices(node)) - represented
        ),
        key=lambda node: node.i,
    )
    boundaries = [node.i for node in anchors[1:]]
    for node in extras:
        slot_index = sum(node.i >= boundary for boundary in boundaries)
        slot_index = min(slot_index, len(members) - 1)
        member = members[slot_index]
        extra_proteins = set(node_protein_indices(node))
        if node.i > member.head_index:
            modifiers = set(member.protein_indices) | set(member.modifier_protein_indices)
            members[slot_index] = GroupMember(
                head_index=node.i,
                head_protein_indices=tuple(sorted(extra_proteins)),
                modifier_protein_indices=tuple(sorted(modifiers - extra_proteins)),
            )
        else:
            modifiers = set(member.modifier_protein_indices) | extra_proteins
            members[slot_index] = GroupMember(
                head_index=member.head_index,
                head_protein_indices=member.head_protein_indices,
                modifier_protein_indices=tuple(
                    sorted(modifiers - set(member.head_protein_indices))
                ),
            )

    return SemanticGroup(
        kind=KIND_COORDINATION,
        head_index=head.i,
        members=tuple(members),
        is_alternative=conj_chain_has_alternative_n(anchor, parsed),
    )


def build_accompaniment_group_n(head: SyntaxNode, parsed: ParsedSentence) -> SemanticGroup:
    from .node_helpers import accompaniment_nmods_n, conj_chain_n
    proteins: set[int] = set(_member_n(head, parsed).role_protein_indices)
    for nmod in accompaniment_nmods_n(head, parsed):
        for accomp in conj_chain_n(nmod, parsed):
            proteins.update(_member_n(accomp, parsed).role_protein_indices)
    return SemanticGroup(
        kind=KIND_ACCOMPANIMENT,
        head_index=head.i,
        members=(GroupMember(head_index=head.i,
                             head_protein_indices=tuple(sorted(proteins))),),
    )


def build_apposition_group_n(head: SyntaxNode, parsed: ParsedSentence) -> SemanticGroup:
    from .node_helpers import children_by_dep_n
    members = [_member_n(head, parsed)]
    for appos in children_by_dep_n(head, parsed, DEP_APPOS):
        members.append(_member_n(appos, parsed))
    return SemanticGroup(
        kind=KIND_APPOSITION,
        head_index=head.i,
        members=tuple(members),
        coreferent=True,
    )


def argument_group_n(head: SyntaxNode, parsed: ParsedSentence) -> SemanticGroup:
    from .node_helpers import children_by_dep_n, conj_chain_n
    self_chain = conj_chain_n(head, parsed)
    if len(self_chain) >= 2:
        return build_coordination_group_n(head, parsed)

    # Keep coordinated compound slots inside a plural complex head.
    for comp in children_by_dep_n(head, parsed, DEP_COMPOUND):
        chain = conj_chain_n(comp, parsed)
        if len(chain) >= 2:
            if looks_complex((head.lemma or head.text or "").lower()) and not (
                (head.text or "").lower().endswith("s")
            ):
                from .node_helpers import node_protein_indices
                proteins = tuple(sorted({
                    protein
                    for member in chain
                    for protein in node_protein_indices(member)
                }))
                if len(proteins) >= 2:
                    return SemanticGroup(
                        kind=KIND_COORDINATION,
                        head_index=head.i,
                        members=(GroupMember(
                            head_index=head.i,
                            head_protein_indices=proteins,
                        ),),
                    )
            return build_coordinated_compound_group_n(head, comp, parsed)

    if looks_complex((head.lemma or head.text or "").lower()):
        return SemanticGroup(
            kind=KIND_COORDINATION,
            head_index=head.i,
            members=(_member_n(head, parsed),),
        )

    return SemanticGroup(
        kind=KIND_COORDINATION,
        head_index=head.i,
        members=(_member_n(head, parsed),),
    )
