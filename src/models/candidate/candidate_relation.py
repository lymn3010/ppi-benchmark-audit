from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterable

from .candidate_event import AssertionStatus, CandidateArgument, CandidateEvent, EventType
from ..pattern_identity import PatternIdentity


@dataclass(frozen=True)
class CandidateRelation:
    """Export-only pair proposed by a CandidateEvent; ``decision`` defaults to review_only."""

    candidate_id: str
    sentence_id: str
    pair: tuple[int, int] | None
    source_event_id: str
    parser_source: str
    schema: str
    predicate_lemma: str
    event_type: str
    event_detail: str
    source_role: str
    target_role: str
    case_marker: str
    assertion_status: str
    projection_mode: str
    pattern_signature: str
    pattern_decision: str = "LOW_SUPPORT"
    decision: str = "review_only"
    ambiguity_reason: str = ""
    is_nominalized: bool = False
    realization_channel: str = "unknown"
    # A protein entered via a compound modifier ("P1 receptor"), not as a direct argument.
    compound_propagated: bool = False
    owner_propagated: bool = False
    descriptor_propagated: bool = False
    predicate_parser_dep: str = ""
    source_argument_role: str = ""
    target_argument_role: str = ""
    source_parser_dep: str = ""
    target_parser_dep: str = ""


    def derived_shape(self) -> str:
        """Canonical shape: AB, AA, NESTED or RELATIONAL."""
        pm = (self.projection_mode or "").lower()
        if pm == "nested_review":
            return "NESTED"
        if pm == "relational":
            return "RELATIONAL"
        if pm == "undirected":
            return "AA"
        # Directed rows are AB; the case marker is in the prep segment.
        return "AB"

    def derived_construction(self) -> str:
        """Compute the construction label from event_detail / is_nominalized."""
        if (self.projection_mode or "").lower() == "nested_review":
            return "nested"
        detail = (self.event_detail or "").lower()
        if "passive" in detail:
            return "passive"
        if (self.event_type or "").upper() == "INTERACTION_STATE":
            return "compound_state"
        if self.is_nominalized:
            return "nominalized"
        return "verbal"

    def canonical_pattern_key(self) -> str:
        """Stable key joinable against ``pattern_stats.pattern_key``.

        Format: ``{evidence_class}|{shape}|{lemma}|{prep_or_'_'}|{construction}``
        """
        return self._pattern_identity().canonical_key

    def semantic_pattern_key(self) -> str:
        """Direction-agnostic key joinable to ``pattern_stats.semantic_pattern_key``."""
        return self._pattern_identity().semantic_key

    def _pattern_identity(self) -> PatternIdentity:
        is_relational = (self.projection_mode or "").lower() == "relational"
        lemma = self.predicate_lemma
        if is_relational and not lemma:
            lemma = self.schema or "relational"
        return PatternIdentity(
            evidence_class="relational" if is_relational else "structural",
            shape=self.derived_shape(),
            predicate_concept=lemma,
            case_realization="" if is_relational else self.case_marker,
            construction=self.derived_construction(),
        )

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "sentence_id": self.sentence_id,
            "pair": list(self.pair) if self.pair is not None else None,
            "source_event_id": self.source_event_id,
            "parser_source": self.parser_source,
            "schema": self.schema,
            "predicate_lemma": self.predicate_lemma,
            "event_type": self.event_type,
            "event_detail": self.event_detail,
            "realization_channel": self.realization_channel,
            "source_role": self.source_role,
            "target_role": self.target_role,
            "case_marker": self.case_marker,
            "assertion_status": self.assertion_status,
            "projection_mode": self.projection_mode,
            "pattern_signature": self.pattern_signature,
            "pattern_decision": self.pattern_decision,
            "decision": self.decision,
            "ambiguity_reason": self.ambiguity_reason,
            "compound_propagated": self.compound_propagated,
            "owner_propagated": self.owner_propagated,
            "descriptor_propagated": self.descriptor_propagated,
            "predicate_parser_dep": self.predicate_parser_dep,
            "source_argument_role": self.source_argument_role,
            "target_argument_role": self.target_argument_role,
            "source_parser_dep": self.source_parser_dep,
            "target_parser_dep": self.target_parser_dep,
            "is_nominalized": self.is_nominalized,
            "pattern_key": self.canonical_pattern_key(),
            "semantic_pattern_key": self.semantic_pattern_key(),
            "shape": self.derived_shape(),
            "construction": self.derived_construction(),
        }


class CandidateRelationLedger:
    """Build export-only candidate relation evidence from CandidateEvents."""

    def build(
        self,
        candidates: Iterable[CandidateEvent],
        *,
        sentence_id: str = "",
    ) -> list[CandidateRelation]:
        events = list(candidates)
        by_id = {event.event_id: event for event in events}
        rows: list[CandidateRelation] = []
        for event in events:
            for index, row in enumerate(self._event_rows(event, sentence_id, by_id)):
                rows.append(
                    CandidateRelation(
                        candidate_id=f"{event.event_id}:rel:{index}",
                        **row,
                    )
                )
        return rows

    def _event_rows(
        self,
        event: CandidateEvent,
        sentence_id: str,
        by_id: dict[str, CandidateEvent],
    ) -> list[dict]:
        roles = self._role_pairs(event)
        rows: list[dict] = []
        # Deduplicate protein pairs across argument combinations.
        seen_pairs: set[tuple[tuple[int, int], str]] = set()
        event_case_marker = self._event_case_marker(event)
        for source_role, target_role, source_args, target_args, projection_mode in roles:
            for source_arg, target_arg in self._argument_pairs(source_args, target_args, projection_mode):
                # The pattern preposition comes from the event, not from an owner PP.
                if projection_mode == "directed":
                    case_marker = self._first_case_marker(target_arg)
                elif projection_mode == "undirected":
                    case_marker = event_case_marker
                else:
                    case_marker = ""
                assertion_status = self._assertion_status(event, source_arg, target_arg)
                compound_idx = (
                    set(source_arg.compound_owner_indices)
                    | set(target_arg.compound_owner_indices)
                )
                owner_idx = (
                    set(source_arg.owner_protein_indices)
                    | set(target_arg.owner_protein_indices)
                )
                descriptor_idx = (
                    set(source_arg.descriptor_protein_indices)
                    | set(target_arg.descriptor_protein_indices)
                )
                for pair in self._pair_candidates(source_arg, target_arg, projection_mode):
                    dedup_key = (pair, projection_mode)
                    if dedup_key in seen_pairs:
                        continue
                    seen_pairs.add(dedup_key)
                    rows.append({
                        "sentence_id": sentence_id,
                        "pair": pair,
                        "source_event_id": event.event_id,
                        "parser_source": event.parser_source,
                        "schema": self._schema(event, case_marker),
                        "predicate_lemma": self._predicate_lemma(event),
                        "event_type": event.event_type.name,
                        "event_detail": event.extraction_detail,
                        "realization_channel": event.realization_channel,
                        "source_role": source_role,
                        "target_role": target_role,
                        "case_marker": case_marker,
                        "assertion_status": assertion_status,
                        "projection_mode": projection_mode,
                        "pattern_signature": self._pattern_signature(
                            event,
                            source_role,
                            target_role,
                            case_marker,
                            projection_mode,
                            source_arg,
                            target_arg,
                        ),
                        "ambiguity_reason": self._ambiguity_reason(assertion_status),
                        "is_nominalized": bool(
                            event.predicate is not None and event.predicate.is_nominalized
                        ),
                        "compound_propagated": bool(set(pair) & compound_idx),
                        "owner_propagated": bool(set(pair) & owner_idx),
                        "descriptor_propagated": bool(set(pair) & descriptor_idx),
                        "predicate_parser_dep": (
                            event.predicate.parser_dep if event.predicate is not None else ""
                        ),
                        "source_argument_role": source_arg.role,
                        "target_argument_role": target_arg.role,
                        "source_parser_dep": self._argument_parser_dep(source_arg),
                        "target_parser_dep": self._argument_parser_dep(target_arg),
                    })
        rows.extend(self._nested_review_rows(event, sentence_id, by_id))
        return rows

    def _nested_review_rows(
        self,
        event: CandidateEvent,
        sentence_id: str,
        by_id: dict[str, CandidateEvent],
    ) -> list[dict]:
        if not event.has_nested_arg():
            return []

        rows: list[dict] = []
        for slot, event_ref_arg in event.iter_args():
            if not event_ref_arg.is_event_reference:
                continue
            inner = by_id.get(event_ref_arg.event_ref)
            if inner is None:
                continue

            outer_args = self._outer_args_for_nested_ref(event, slot)
            inner_args = tuple(
                arg
                for _, arg in inner.iter_args()
                if not arg.is_event_reference
            )
            case_marker = self._prep_from_role(event_ref_arg.role)
            for outer_arg, inner_arg in product(outer_args, inner_args):
                compound_idx = (
                    set(outer_arg.compound_owner_indices)
                    | set(inner_arg.compound_owner_indices)
                )
                owner_idx = (
                    set(outer_arg.owner_protein_indices)
                    | set(inner_arg.owner_protein_indices)
                )
                descriptor_idx = (
                    set(outer_arg.descriptor_protein_indices)
                    | set(inner_arg.descriptor_protein_indices)
                )
                for pair in self._pair_candidates(outer_arg, inner_arg, "nested_review"):
                    assertion = self._combine_assertion_statuses(
                        self._assertion_status(event, outer_arg, inner_arg),
                        self._span_assertion_status(inner.predicate),
                    )
                    rows.append({
                        "sentence_id": sentence_id,
                        "pair": pair,
                        "source_event_id": event.event_id,
                        "parser_source": event.parser_source,
                        "schema": "Nested",
                        "predicate_lemma": self._predicate_lemma(event),
                        "event_type": event.event_type.name,
                        "event_detail": event.extraction_detail,
                        "realization_channel": event.realization_channel,
                        "source_role": self._nested_outer_role(event, slot),
                        "target_role": f"{slot}:event_ref:{inner.event_id}",
                        "case_marker": case_marker,
                        "assertion_status": assertion,
                        "projection_mode": "nested_review",
                        "pattern_signature": self._nested_pattern_signature(
                            event,
                            inner,
                            slot,
                            case_marker,
                            assertion,
                        ),
                        "ambiguity_reason": self._ambiguity_reason(assertion),
                        "is_nominalized": bool(
                            event.predicate is not None and event.predicate.is_nominalized
                        ),
                        "compound_propagated": bool(set(pair) & compound_idx),
                        "owner_propagated": bool(set(pair) & owner_idx),
                        "descriptor_propagated": bool(set(pair) & descriptor_idx),
                        "predicate_parser_dep": (
                            event.predicate.parser_dep if event.predicate is not None else ""
                        ),
                        "source_argument_role": outer_arg.role,
                        "target_argument_role": inner_arg.role,
                        "source_parser_dep": self._argument_parser_dep(outer_arg),
                        "target_parser_dep": self._argument_parser_dep(inner_arg),
                    })
        return rows

    @staticmethod
    def _outer_args_for_nested_ref(
        event: CandidateEvent,
        event_ref_slot: str,
    ) -> tuple[CandidateArgument, ...]:
        if event.is_directed:
            opposite = "sources" if event_ref_slot == "targets" else "targets"
            args = tuple(
                arg for arg in event.arguments.get(opposite, ())
                if not arg.is_event_reference
            )
            if args:
                return args
        return tuple(
            arg
            for slot, slot_args in event.arguments.items()
            if slot != event_ref_slot
            for arg in slot_args
            if not arg.is_event_reference
        ) or tuple(
            arg
            for arg in event.arguments.get(event_ref_slot, ())
            if not arg.is_event_reference
        )

    @staticmethod
    def _nested_outer_role(event: CandidateEvent, event_ref_slot: str) -> str:
        if event.is_directed:
            opposite = "sources" if event_ref_slot == "targets" else "targets"
            if event.arguments.get(opposite):
                return opposite
        return "outer_arguments"

    @staticmethod
    def _role_pairs(event: CandidateEvent) -> list[tuple[str, str, tuple[CandidateArgument, ...], tuple[CandidateArgument, ...], str]]:
        args = event.arguments
        if event.event_type is EventType.INTERACTION and event.is_directed:
            return [("sources", "targets", args.get("sources", ()), args.get("targets", ()), "directed")]
        if event.event_type in {EventType.INTERACTION, EventType.INTERACTION_STATE}:
            participants = args.get("participants", ())
            return [("participants", "participants", participants, participants, "undirected")]
        if event.event_type is EventType.RELATION_STATEMENT:
            return [("entities_a", "entities_b", args.get("entities_a", ()), args.get("entities_b", ()), "relational")]
        return []

    @staticmethod
    def _argument_pairs(
        left: tuple[CandidateArgument, ...],
        right: tuple[CandidateArgument, ...],
        projection_mode: str,
    ) -> Iterable[tuple[CandidateArgument, CandidateArgument]]:
        if projection_mode == "undirected" and left is right:
            # Pairs across participant arguments, plus pairs within each multi-protein argument.
            yield from combinations(left, 2)
            for arg in left:
                if len(arg.all_protein_indices) >= 2:
                    yield (arg, arg)
            return
        yield from product(left, right)

    @staticmethod
    def _pair_candidates(
        left: CandidateArgument,
        right: CandidateArgument,
        projection_mode: str,
    ) -> Iterable[tuple[int, int]]:
        if left.is_event_reference or right.is_event_reference:
            return
        left_targets = left.all_protein_indices
        right_targets = right.all_protein_indices
        seen: set[tuple[int, int]] = set()
        for a, b in product(left_targets, right_targets):
            if a == b:
                continue
            pair = tuple(sorted((int(a), int(b))))
            if pair in seen:
                continue
            seen.add(pair)
            yield pair

    @staticmethod
    def _predicate_lemma(event: CandidateEvent) -> str:
        if event.predicate is None or not event.predicate.lemmas:
            return ""
        # Canonicalize nominalized predicates so keys join RelationChain keys.
        from ._nominal_canon import canonicalize_predicate_lemma
        return canonicalize_predicate_lemma(event.predicate.lemmas[0])

    @staticmethod
    def _argument_parser_dep(arg: CandidateArgument) -> str:
        if arg.core is None:
            return ""
        return arg.core.parser_dep or arg.core.dep

    @staticmethod
    def _case_marker(arg: CandidateArgument) -> str:
        markers: list[str] = []
        for marker in arg.case_markers:
            if marker.lemmas:
                markers.append(marker.lemmas[0])
            elif marker.tokens:
                markers.append(marker.tokens[0].lower())
        return "|".join(sorted(set(m for m in markers if m)))

    @staticmethod
    def _first_case_marker(arg: CandidateArgument) -> str:
        # Use the event-role PP, not PPs inside the mention ("domain of P1").
        role_prep = CandidateRelationLedger._prep_from_role(arg.role)
        if role_prep:
            return role_prep
        # A non-PP role means markers belong to the mention interior.
        if arg.role:
            return ""
        for marker in arg.case_markers:
            if marker.lemmas:
                return marker.lemmas[0]
            if marker.tokens:
                return marker.tokens[0].lower()
        return ""

    @classmethod
    def _event_case_marker(cls, event: CandidateEvent) -> str:
        """Return the event-level relation marker used by ``RelationChain``."""
        if event.event_type is not EventType.INTERACTION:
            return ""
        slots = ("targets",) if event.is_directed else ("participants",)
        for slot in slots:
            for arg in event.arguments.get(slot, ()):
                marker = cls._first_case_marker(arg)
                if marker:
                    return marker
        return ""

    @staticmethod
    def _prep_from_role(role: str) -> str:
        if ":" not in (role or ""):
            return ""
        return role.split(":", 1)[1].lower()

    @staticmethod
    def _assertion_status(
        event: CandidateEvent,
        left: CandidateArgument,
        right: CandidateArgument,
    ) -> str:
        return CandidateRelationLedger._combine_assertion_statuses(
            CandidateRelationLedger._span_assertion_status(event.predicate),
            left.assertion_status,
            right.assertion_status,
        )

    @staticmethod
    def _span_assertion_status(span) -> str:
        if span is None:
            return AssertionStatus.ASSERTED.value
        return AssertionStatus.normalize(
            getattr(span, "assertion_status", AssertionStatus.ASSERTED.value),
            is_negated=getattr(span, "is_negated", False),
        )

    @staticmethod
    def _combine_assertion_statuses(*statuses: str) -> str:
        normalized = [AssertionStatus.normalize(status) for status in statuses if status]
        if AssertionStatus.NEGATED.value in normalized:
            return AssertionStatus.NEGATED.value
        for status in normalized:
            if status != AssertionStatus.ASSERTED.value:
                return status
        return AssertionStatus.ASSERTED.value

    @staticmethod
    def _ambiguity_reason(assertion_status: str) -> str:
        if assertion_status in ("", AssertionStatus.ASSERTED.value):
            return ""
        return f"assertion_{assertion_status}"

    @staticmethod
    def _schema(event: CandidateEvent, case_marker: str) -> str:
        detail = (event.extraction_detail or "").lower()
        if event.event_type is EventType.RELATION_STATEMENT:
            return event.relation_subtype or "RelationalAssertion"
        if event.event_type is EventType.INTERACTION_STATE:
            return "InteractionState"
        if event.has_nested_arg():
            return "Nested"
        if "passive" in detail:
            return "PassivePrep" if case_marker else "Passive"
        if case_marker:
            return "PrepObject"
        if event.is_directed:
            return "DirectObject"
        return "Undirected"

    def _pattern_signature(
        self,
        event: CandidateEvent,
        source_role: str,
        target_role: str,
        case_marker: str,
        projection_mode: str,
        source_arg: CandidateArgument,
        target_arg: CandidateArgument,
    ) -> str:
        flags = []
        if event.predicate is not None and event.predicate.is_nominalized:
            flags.append("nominalized")
        if event.has_nested_arg():
            flags.append("nested")
        if "passive" in (event.extraction_detail or "").lower():
            flags.append("passive")
        assertion = self._assertion_status(event, source_arg, target_arg)
        parts = [
            event.parser_source or "",
            self._predicate_lemma(event),
            event.event_type.name,
            event.extraction_detail or "",
            self._schema(event, case_marker),
            source_role,
            target_role,
            case_marker or "_",
            projection_mode,
            assertion,
            "+".join(flags) if flags else "_",
        ]
        return "|".join(parts)

    def _nested_pattern_signature(
        self,
        outer: CandidateEvent,
        inner: CandidateEvent,
        event_ref_slot: str,
        case_marker: str,
        assertion_status: str,
    ) -> str:
        parts = [
            outer.parser_source or "",
            self._predicate_lemma(outer),
            "NESTED",
            outer.extraction_detail or "",
            event_ref_slot,
            case_marker or "_",
            self._predicate_lemma(inner),
            inner.event_type.name,
            inner.extraction_detail or "",
            assertion_status,
        ]
        return "|".join(parts)
