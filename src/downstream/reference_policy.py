"""Canonical identity policy over parser-neutral CandidateEvents."""
from __future__ import annotations

from dataclasses import dataclass, field

from itertools import product

from src.models.candidate import CandidateArgument, CandidateEvent, EventType, Span
from src.models.semantic_vocabulary import SemanticSlot
from src.system_rules import rule_frozenset


@dataclass
class CanonicalIdentityMap:
    """Union-find identity classes plus declaration provenance."""

    parent: dict[int, int] = field(default_factory=dict)
    declarations: list[dict] = field(default_factory=list)
    exact_mappings: dict[tuple, set[tuple]] = field(default_factory=dict)
    signature_indices: dict[tuple, tuple[int, ...]] = field(default_factory=dict)
    global_mappings: dict[int, set[tuple]] = field(default_factory=dict)

    def find(self, value: int) -> int:
        value = int(value)
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(
        self,
        first: int,
        second: int,
        *,
        provenance: str = "",
        source_event_id: str = "",
    ) -> None:
        first, second = int(first), int(second)
        root_a, root_b = self.find(first), self.find(second)
        if root_a != root_b:
            low, high = sorted((root_a, root_b))
            self.parent[high] = low
        declaration = {
            "id": (
                f"{source_event_id}:identity:{min(first, second)}:{max(first, second)}"
                if source_event_id else f"identity:{min(first, second)}:{max(first, second)}"
            ),
            "first": first,
            "second": second,
            "provenance": provenance,
            "source_event_id": source_event_id,
        }
        if declaration not in self.declarations:
            self.declarations.append(declaration)

    def equivalents(self, value: int) -> tuple[int, ...]:
        root = self.find(value)
        return tuple(sorted(index for index in self.parent if self.find(index) == root))

    def expand(self, values) -> tuple[int, ...]:
        expanded: set[int] = set()
        for value in values:
            expanded.update(self.equivalents(int(value)))
        return tuple(sorted(expanded))

    @property
    def classes(self) -> tuple[tuple[int, ...], ...]:
        groups: dict[int, set[int]] = {}
        for value in list(self.parent):
            groups.setdefault(self.find(value), set()).add(value)
        return tuple(sorted(tuple(sorted(group)) for group in groups.values() if len(group) > 1))

    def to_dict(self) -> dict:
        return {
            "schema": "canonical_identity_map_v1",
            "classes": [list(group) for group in self.classes],
            "declarations": list(self.declarations),
        }

    def evidence_refs_for(self, values) -> tuple[str, ...]:
        """Return identity declarations relevant to projected protein indices."""
        roots = {self.find(int(value)) for value in values}
        refs = {
            str(declaration["id"])
            for declaration in self.declarations
            if self.find(int(declaration["first"])) in roots
            or self.find(int(declaration["second"])) in roots
        }
        return tuple(sorted(refs))

    @staticmethod
    def span_signature(span: Span) -> tuple:
        return tuple(
            (
                int(offset[0]),
                tuple(int(index) for index in span.protein_indices) if position == 0 else (),
            )
            for position, offset in enumerate(span.char_offsets)
        )

    def add_span_equivalence(
        self,
        first: Span,
        second: Span,
        *,
        provenance: str,
        source_event_id: str = "",
    ) -> None:
        first_signature = self.span_signature(first)
        second_signature = self.span_signature(second)
        self.signature_indices[first_signature] = tuple(first.protein_indices)
        self.signature_indices[second_signature] = tuple(second.protein_indices)
        self.exact_mappings.setdefault(first_signature, set()).add(second_signature)
        self.exact_mappings.setdefault(second_signature, set()).add(first_signature)
        if len(first_signature) == 1:
            for index in first.protein_indices:
                self.global_mappings.setdefault(int(index), set()).add(second_signature)
        if len(second_signature) == 1:
            for index in second.protein_indices:
                self.global_mappings.setdefault(int(index), set()).add(first_signature)
        for first_index in first.protein_indices:
            for second_index in second.protein_indices:
                if first_index != second_index:
                    self.union(
                        first_index,
                        second_index,
                        provenance=provenance,
                        source_event_id=source_event_id,
                    )

    def expand_span_variants(self, span: Span | None) -> tuple[tuple[int, ...], ...]:
        if span is None:
            return ((),)
        start = self.span_signature(span)
        self.signature_indices.setdefault(start, tuple(span.protein_indices))
        seen = {start}
        pending = [start]
        while pending:
            signature = pending.pop()
            neighbors = set(self.exact_mappings.get(signature, ()))
            if len(signature) == 1:
                for index in self.signature_indices.get(signature, ()):
                    neighbors.update(self.global_mappings.get(int(index), ()))
            for neighbor in neighbors:
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
        variants = {
            tuple(sorted(self.signature_indices.get(signature, ())))
            for signature in seen
        }
        return tuple(sorted(variants)) or ((),)

    def expand_argument_variants(
        self,
        argument: CandidateArgument,
    ) -> tuple[tuple[int, ...], ...]:
        if argument.is_event_reference:
            return ((),)
        core_variants = self.expand_span_variants(argument.core)
        owner_variants = [
            self.expand_span_variants(owner)
            for owner in argument.owners
            if not owner.is_negated
        ]
        descriptor_variants = [
            self.expand_span_variants(descriptor)
            for descriptor in argument.descriptors
        ]
        dimensions = [core_variants, *owner_variants, *descriptor_variants]
        variants: set[tuple[int, ...]] = set()
        for combination in product(*dimensions):
            variants.add(tuple(sorted({index for values in combination for index in values})))
        return tuple(sorted(variants)) or ((),)

    @classmethod
    def from_candidate_events(cls, events: list[CandidateEvent]) -> "CanonicalIdentityMap":
        identity_map = cls()
        safe_details = rule_frozenset("reference.identity_safe_extraction_details")
        for event in events:
            if event.event_type is not EventType.RELATION_STATEMENT:
                continue
            if (event.relation_subtype or "IS-A") != "IS-A":
                continue
            if event.extraction_detail not in safe_details:
                continue
            left = event.arguments.get(SemanticSlot.ENTITIES_A.value, ())
            right = event.arguments.get(SemanticSlot.ENTITIES_B.value, ())
            for first in left:
                for second in right:
                    if first.core is None or second.core is None:
                        continue
                    if not first.core.protein_indices or not second.core.protein_indices:
                        continue
                    identity_map.add_span_equivalence(
                        first.core,
                        second.core,
                        provenance=event.extraction_detail,
                        source_event_id=event.event_id,
                    )
        return identity_map


__all__ = ["CanonicalIdentityMap"]
