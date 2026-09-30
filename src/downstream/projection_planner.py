"""Canonical projection planner over parser-neutral CandidateEvents."""
from __future__ import annotations

from itertools import combinations, product

from src.lexicons.morphology import looks_complex as _looks_complex_lemma
from src.models.candidate import AssertionStatus, CandidateArgument, CandidateEvent, EventType
from src.models.projection_contract import (
    PairDecision,
    PairProposal,
    ProjectionPlan,
)
from src.models.semantic_vocabulary import (
    ProjectionRuleId,
    ProjectionStatus,
    SemanticSlot,
    projection_rule_id,
)
from src.system_rules import rule_frozenset, system_rule
from src.downstream.reference_policy import CanonicalIdentityMap
from src.downstream.rules.relational_role_nominal import project_relational_role_nominal
from src.extraction.rules import RULE_TRACE_KEY, RuleApplication
from src.models.candidate.group_alignment import align_group_provenance


_EXEMPLAR_PREPS = rule_frozenset("semantics.exemplar_prepositions")
_GROUP_RELATIONAL_CASE_MARKERS = rule_frozenset("prepositions.group_relational")
_EVENT_SOURCE_MARKERS = system_rule("event_source_markers")


class ProjectionPlanner:
    """Build the canonical typed projection plan for each CandidateEvent."""

    def __init__(self, resources=None) -> None:
        self._resources = resources

    def plan_all(
        self,
        candidates: list[CandidateEvent],
        *,
        sentence_id: str,
        identity_map: CanonicalIdentityMap | None = None,
    ) -> list[ProjectionPlan]:
        identity_map = identity_map or CanonicalIdentityMap.from_candidate_events(candidates)
        plans: list[ProjectionPlan] = []
        for event_index, event in enumerate(candidates):
            plan = self.plan_event(
                event,
                sentence_id=sentence_id,
                event_index=event_index,
                identity_map=identity_map,
            )
            plans.append(plan)
        return plans

    def plan_event(
        self,
        event: CandidateEvent,
        *,
        sentence_id: str,
        event_index: int,
        identity_map: CanonicalIdentityMap | None = None,
    ) -> ProjectionPlan:
        identity_map = identity_map or CanonicalIdentityMap.from_candidate_events([event])
        computed_pairs, computed_reason, projection_mode = self._project(event, identity_map)

        if computed_pairs:
            final_pairs = computed_pairs
            status = ProjectionStatus.APPLIED.value
            reason = computed_reason
        elif computed_reason == ProjectionRuleId.RELATIONAL_REVIEW_ONLY.value:
            final_pairs = ()
            status = ProjectionStatus.REVIEW_ONLY.value
            reason = computed_reason
        else:
            final_pairs = ()
            status = ProjectionStatus.NOT_APPLIED.value
            reason = computed_reason or ProjectionRuleId.NO_PROJECTABLE_PAIRS.value

        proposals: list[PairProposal] = []
        decisions: list[PairDecision] = []
        for index, pair in enumerate(computed_pairs):
            proposal = PairProposal(
                id=f"{sentence_id}:{event.event_id}:proposal:{index}",
                pair=pair,
                source_event_id=event.event_id,
                projection_mode=projection_mode,
                rule_id=str(projection_rule_id(computed_reason)),
                semantic_slots=self._semantic_slots(event),
                evidence_refs=(event.event_id, *identity_map.evidence_refs_for(pair)),
                provenance={"origin": "candidate_event"},
            )
            proposals.append(proposal)
            decisions.append(PairDecision(
                proposal_ref=proposal.id,
                pair=pair,
                status=ProjectionStatus.APPLIED.value,
                reason_code=reason,
                emitted=True,
            ))

        return ProjectionPlan(
            id=f"{sentence_id}:{event.event_id}:projection-plan",
            sentence_id=sentence_id,
            source_event_id=event.event_id,
            event_index=event_index,
            event_type=event.event_type.name,
            application_status=status,
            reason_code=reason,
            rule_id=str(projection_rule_id(reason)),
            projection_mode=projection_mode,
            proposals=tuple(proposals),
            decisions=tuple(decisions),
            final_pairs=final_pairs,
            provenance={
                "authority": "canonical_candidate_event",
                "runtime_effect": "final_pairs",
                "identity_classes": [list(group) for group in identity_map.classes],
            },
        )

    @staticmethod
    def apply_plans(events: list[CandidateEvent], plans: list[ProjectionPlan]) -> None:
        """Attach canonical final pairs to their source CandidateEvents."""
        by_source = {plan.source_event_id: plan for plan in plans}
        for event in events:
            plan = by_source.get(event.event_id)
            if plan is not None:
                event.projected_pairs = tuple(plan.final_pairs)

    def _project(
        self,
        event: CandidateEvent,
        identity_map: CanonicalIdentityMap,
    ) -> tuple[tuple[tuple[int, int], ...], str, str]:
        if not self._predicate_asserted(event):
            return (), ProjectionRuleId.BLOCKED_ASSERTION.value, self._mode(event)
        if event.predicate is not None and event.predicate.is_negated:
            return (), ProjectionRuleId.BLOCKED_PREDICATE_NEGATION.value, self._mode(event)

        if event.event_type is EventType.RELATION_STATEMENT:
            pairs = project_relational_role_nominal(
                event,
                identity_map=identity_map,
                resources=self._resources,
            )
            if pairs:
                self._append_projection_trace(event, pairs)
                return (
                    pairs,
                    ProjectionRuleId.APPLIED_RELATIONAL_ROLE_NOMINAL.value,
                    "relational_role_nominal",
                )
            return (), ProjectionRuleId.RELATIONAL_REVIEW_ONLY.value, "relational"
        if event.event_type not in {EventType.INTERACTION, EventType.INTERACTION_STATE}:
            return (), ProjectionRuleId.NO_PROJECTABLE_PAIRS.value, "unknown"

        if event.is_directed:
            if not self._event_pattern_supported(event):
                return (), ProjectionRuleId.BLOCKED_PATTERN_SUPPORT.value, "directed"
            pairs = self._directed_pairs(event, identity_map)
            return (
                pairs,
                ProjectionRuleId.APPLIED_RELATION_PROJECTION.value if pairs else ProjectionRuleId.NO_PROJECTABLE_PAIRS.value,
                "directed",
            )

        is_nested_inner = (event.extraction_detail or "").endswith(
            _EVENT_SOURCE_MARKERS["nested_inner_suffix"]
        )
        compound_internal = self._predicate_compound_internal_pairs(event)
        if is_nested_inner and compound_internal:
            return (
                compound_internal,
                ProjectionRuleId.APPLIED_GROUP_INTERNAL_TOPOLOGY.value,
                "predicate_compound_internal_pairs",
            )
        if is_nested_inner and not self._explicit_group_relation_supported(event):
            return (), ProjectionRuleId.BLOCKED_NESTED_WITHOUT_GROUP_SEMANTICS.value, "undirected"
        if not self._event_pattern_supported(event):
            return (), ProjectionRuleId.BLOCKED_PATTERN_SUPPORT.value, "undirected"

        participants = [
            argument
            for argument in event.arguments.get(SemanticSlot.PARTICIPANTS.value, ())
            if self._argument_allows_projection(argument)
        ]
        participant_frames = [
            (argument, variant)
            for argument in participants
            for variant in identity_map.expand_argument_variants(argument)
        ]
        visible_frame_indices = {
            index
            for _, variant in participant_frames
            for index in variant
        }
        internal = self._shared_internal_group_pairs(
            participants,
            visible=visible_frame_indices,
        )
        if internal:
            return internal, ProjectionRuleId.APPLIED_GROUP_INTERNAL_TOPOLOGY.value, "shared_group_internal_pairs"

        result: set[tuple[int, int]] = set()
        for index, (first_argument, first_indices) in enumerate(participant_frames):
            for second_argument, second_indices in participant_frames[index + 1:]:
                if self._is_exemplar(first_argument) and self._is_exemplar(second_argument):
                    continue
                result.update(self._cross_pairs(
                    first_indices,
                    second_indices,
                ))
        for participant, indices in participant_frames:
            if self._is_exemplar(participant):
                continue
            result.update(combinations(indices, 2))
        pairs = self._normalize_pairs(result)
        return (
            pairs,
            ProjectionRuleId.APPLIED_RELATION_PROJECTION.value if pairs else ProjectionRuleId.NO_PROJECTABLE_PAIRS.value,
            "undirected",
        )

    @staticmethod
    def _append_projection_trace(
        event: CandidateEvent,
        pairs: tuple[tuple[int, int], ...],
    ) -> None:
        if not pairs:
            return
        custom = dict(event.custom_values)
        trace = list(custom.get(RULE_TRACE_KEY, ()))
        trace.append(RuleApplication(
            rule_id=str(projection_rule_id(ProjectionRuleId.APPLIED_RELATIONAL_ROLE_NOMINAL)),
            stage="pair_projection",
            decision="applied",
            reason="role_nominal_is_a_of_for_endpoint",
            evidence={
                "relation_subtype": event.relation_subtype,
                "predicate_lemmas": list(event.predicate.lemmas) if event.predicate else [],
                "pairs": [list(pair) for pair in pairs],
            },
        ).to_dict())
        custom[RULE_TRACE_KEY] = trace
        event.custom_values = custom

    def _directed_pairs(
        self,
        event: CandidateEvent,
        identity_map: CanonicalIdentityMap,
    ) -> tuple[tuple[int, int], ...]:
        sources = [
            argument for argument in event.arguments.get(SemanticSlot.SOURCES.value, ())
            if self._argument_allows_projection(argument)
        ]
        targets = [
            argument for argument in event.arguments.get(SemanticSlot.TARGETS.value, ())
            if self._argument_allows_projection(argument)
        ]
        source_indices = [
            variant
            for argument in sources
            for variant in identity_map.expand_argument_variants(argument)
        ]
        target_indices = [
            variant
            for argument in targets
            for variant in identity_map.expand_argument_variants(argument)
        ]
        result: set[tuple[int, int]] = set()
        if any(target_indices):
            for source_argument in sources:
                for target_argument in targets:
                    alignment = align_group_provenance(
                        source_argument.group,
                        target_argument.group,
                    ) if (
                        self._argument_represents_whole_group(source_argument)
                        and self._argument_represents_whole_group(target_argument)
                    ) else None
                    if alignment is not None and alignment.mode == "diagonal":
                        for first, second in alignment.aligned:
                            for expanded_first in identity_map.equivalents(first):
                                for expanded_second in identity_map.equivalents(second):
                                    result.update(self._cross_pairs(
                                        (expanded_first,), (expanded_second,),
                                    ))
                        return self._normalize_pairs(result)
            for source, target in product(source_indices, target_indices):
                result.update(self._cross_pairs(source, target))
        else:
            has_group_object = any(
                self._nominal_is_group(argument)
                for argument in targets
            )
            if has_group_object:
                flattened = tuple(index for values in source_indices for index in values)
                result.update(combinations(flattened, 2))
        return self._normalize_pairs(result)

    @staticmethod
    def _argument_represents_whole_group(argument: CandidateArgument) -> bool:
        """True only when the argument denotes every member of its group."""
        if not argument.group:
            return False
        group_indices = {
            int(index) for index in argument.group.get("all_protein_indices", ())
        }
        return bool(group_indices) and set(argument.projection_protein_indices) == group_indices

    def _event_pattern_supported(self, event: CandidateEvent) -> bool:
        detail = event.extraction_detail or ""
        if (
            detail.startswith(_EVENT_SOURCE_MARKERS["compound_adj_prefix"])
            or detail.startswith(_EVENT_SOURCE_MARKERS["lexical_path_prefix"])
            or detail.startswith(_EVENT_SOURCE_MARKERS["hyphen_participle_prefix"])
            or detail.startswith(_EVENT_SOURCE_MARKERS["agentive_process_projection_prefix"])
            or self._explicit_group_relation_supported(event)
        ):
            return True
        predicate = event.predicate
        return predicate is not None and predicate.pos in {"VERB", "NOUN", "PROPN"}

    def _nominal_is_group(self, argument: CandidateArgument) -> bool:
        if argument.core is None:
            return False
        lemmas = set(argument.core.lemmas)
        return (
            any(_looks_complex_lemma(lemma) for lemma in lemmas)
        )

    def _argument_allows_projection(self, argument: CandidateArgument) -> bool:
        if argument.core is None or argument.is_event_reference:
            return False
        if argument.core.is_negated:
            return False
        return not bool(
            self._resources
            and self._resources.match_lexicon("negation", argument.core.match_keys)
        )

    @staticmethod
    def _predicate_asserted(event: CandidateEvent) -> bool:
        if event.predicate is None:
            return True
        return AssertionStatus.normalize(event.predicate.assertion_status) == AssertionStatus.ASSERTED.value

    @staticmethod
    def _cross_pairs(first, second) -> set[tuple[int, int]]:
        return {
            tuple(sorted((int(a), int(b))))
            for a, b in product(first, second)
            if int(a) != int(b)
        }

    @staticmethod
    def _normalize_pairs(values) -> tuple[tuple[int, int], ...]:
        return tuple(sorted({
            tuple(sorted((int(pair[0]), int(pair[1]))))
            for pair in values or ()
            if len(pair) == 2 and int(pair[0]) != int(pair[1])
        }))

    @staticmethod
    def _mode(event: CandidateEvent) -> str:
        if event.event_type is EventType.RELATION_STATEMENT:
            return "relational"
        return "directed" if event.is_directed else "undirected"

    @staticmethod
    def _semantic_slots(event: CandidateEvent) -> tuple[str, ...]:
        return tuple(sorted(SemanticSlot.normalize(slot) for slot in event.arguments))

    @staticmethod
    def _role_prep(argument: CandidateArgument) -> str:
        role = str(argument.role or "")
        if ":" in role:
            return role.split(":", 1)[1].strip().lower()
        for marker in argument.case_markers:
            if marker.lemmas:
                return marker.lemmas[0].lower()
        return ""

    @classmethod
    def _explicit_group_relation_supported(cls, event: CandidateEvent) -> bool:
        participants = event.arguments.get(SemanticSlot.PARTICIPANTS.value, ())
        prepositions = [cls._role_prep(argument) for argument in participants]
        # Require one explicit between/among marker across conjuncts and no conflicting one.
        return (
            bool(participants)
            and any(prep in _GROUP_RELATIONAL_CASE_MARKERS for prep in prepositions)
            and all(
                not prep or prep in _GROUP_RELATIONAL_CASE_MARKERS
                for prep in prepositions
            )
        )

    @classmethod
    def _is_exemplar(cls, argument: CandidateArgument) -> bool:
        role_prep = cls._role_prep(argument)
        if role_prep in _EXEMPLAR_PREPS:
            return True
        for marker in argument.case_markers:
            forms = {
                *(token.lower() for token in marker.tokens if token),
                *(lemma.lower() for lemma in marker.lemmas if lemma),
            }
            if forms & _EXEMPLAR_PREPS:
                return True
        return False

    @staticmethod
    def _predicate_compound_internal_pairs(
        event: CandidateEvent,
    ) -> tuple[tuple[int, int], ...]:
        """Pairs named by a compound of a reciprocal event nominal (``P0/P1 interaction``).

        ``interaction with P0/P1`` is excluded: role must be compound.
        """
        participants = event.arguments.get(SemanticSlot.PARTICIPANTS.value, ())
        event_visible = {
            index
            for argument in participants
            for index in argument.projection_protein_indices
        }
        # Only an exact binary compound; separator topology is not preserved.
        if len(event_visible) != 2:
            return ()

        result: set[tuple[int, int]] = set()
        for argument in participants:
            if argument.role != "compound" or not argument.group:
                continue
            visible = set(argument.projection_protein_indices)
            for value in argument.group.get("member_internal_pairs", ()):
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    continue
                pair = tuple(sorted((int(value[0]), int(value[1]))))
                if pair[0] != pair[1] and set(pair) <= visible:
                    result.add(pair)
        return tuple(sorted(result))

    @staticmethod
    def _shared_internal_group_pairs(
        participants,
        *,
        visible: set[int] | None = None,
    ) -> tuple[tuple[int, int], ...]:
        if len(participants) < 2:
            return ()
        visible = visible or {
            int(index)
            for participant in participants
            for index in participant.projection_protein_indices
        }
        if not visible:
            return ()
        covering: set[tuple[tuple[int, int], ...]] = set()
        for participant in participants:
            group = participant.group or {}
            if not group or group.get("coreferent") or group.get("is_alternative"):
                continue
            declared = {int(index) for index in group.get("all_protein_indices") or ()}
            if declared != visible:
                continue
            pairs = {
                tuple(sorted((int(raw[0]), int(raw[1]))))
                for raw in group.get("member_internal_pairs") or ()
                if isinstance(raw, (list, tuple)) and len(raw) == 2 and int(raw[0]) != int(raw[1])
            }
            covered = {index for pair in pairs for index in pair}
            if len(pairs) >= 2 and covered == visible:
                covering.add(tuple(sorted(pairs)))
        return next(iter(covering)) if len(covering) == 1 else ()


__all__ = ["ProjectionPlanner"]
