"""State-nominal channel: NPs naming a joint state (``complex``, ``heterodimer``)."""
from __future__ import annotations

from typing import Optional

from src.models.candidate import CandidateArgument, CandidateEvent, EventType
from src.extraction.rules import RuleApplication, append_rule_application
from src.system_rules import rule_frozenset

from .morphology import looks_agentive, looks_complex
from .rules import COLLECTIVE_SLASH_MEMBER_RULE_ID, collective_slash_member_pairs

# PART-OF membership predicates ("complex containing P0, P1").
_MEMBERSHIP_PREDICATES = rule_frozenset("semantics.membership_predicates")

_MEMBERSHIP_ACL_DEPS = rule_frozenset("dependency.categories.membership_acl")


from .channel import EventChannel


class StateNominalEventChannel(EventChannel):
    """Extract compound-complex and compound-adjective state-nominal events."""

    def __init__(self, resources=None, context=None):
        super().__init__(resources, context)
        from .nominalization_events import NominalizationEventChannel
        self.nominals = NominalizationEventChannel(resources, context)

    def _extract_compound_complex_from_parsed(
        self, parsed, next_id, sentence_text: str,
    ) -> list[CandidateEvent]:
        from .node_helpers import (
            children_by_dep_n, node_protein_indices, conj_chain_n,
            match_keys_for_node, single_token_span_n,
        )
        events: list[CandidateEvent] = []
        membership_acl_emitted: set[int] = set()
        complex_state_signatures: set[tuple[int, tuple[int, ...]]] = set()

        # One event per slash pair ("(P0/P1 and P2/P3) oligomers").
        slash_rule_enabled = (
            self.resources is None
            or self.resources.rule_enabled(COLLECTIVE_SLASH_MEMBER_RULE_ID)
        )
        if slash_rule_enabled:
            for collective_head in parsed.nodes:
                is_complex, resolved_head = self._resolve_complex_head_n(
                    collective_head, parsed,
                )
                if (
                    not is_complex
                    or resolved_head is None
                    or resolved_head.i != collective_head.i
                ):
                    continue
                for left_i, right_i in collective_slash_member_pairs(
                    parsed, collective_head.i,
                ):
                    left = parsed.node(left_i)
                    right = parsed.node(right_i)
                    event = CandidateEvent(
                        event_type=EventType.INTERACTION_STATE,
                        event_id=next_id(),
                        parser_source=self.parser_name,
                        extraction_detail="DEP-CollectiveSlashPair",
                        construction="compound_state",
                        sentence_text=sentence_text,
                        predicate=single_token_span_n(
                            collective_head, force_nominal=True,
                        ),
                        arguments={
                            "participants": (
                                CandidateArgument(
                                    core=single_token_span_n(left),
                                    role="orthographic_member",
                                ),
                                CandidateArgument(
                                    core=single_token_span_n(right),
                                    role="orthographic_member",
                                ),
                            ),
                        },
                        is_directed=False,
                    )
                    events.append(append_rule_application(event, RuleApplication(
                        rule_id=COLLECTIVE_SLASH_MEMBER_RULE_ID,
                        stage="group_topology",
                        reason="slash_pair_scoped_by_collective_state_head",
                        evidence={
                            "collective_head_i": collective_head.i,
                            "left_node_i": left.i,
                            "right_node_i": right.i,
                            "separator": "/",
                        },
                    )))

        for token in parsed.nodes:
            if token.dep not in _MEMBERSHIP_ACL_DEPS:
                continue
            if (token.lemma or "").lower() not in _MEMBERSHIP_PREDICATES:
                continue
            complex_head = parsed.head(token)
            if complex_head is None or complex_head.i == token.i:
                continue

            def _is_complex_noun_local_n(t) -> bool:
                if t.pos not in {"NOUN", "PROPN"}:
                    return False
                lem = (t.lemma or "").lower()
                if looks_agentive(lem):
                    return False
                # Membership is evidence even for non-complex heads.
                return True

            if not _is_complex_noun_local_n(complex_head):
                continue
            if self.nominals._is_process_nominalization_n(complex_head, parsed):
                continue

            protein_spans = []
            for child in parsed.children(token):
                if child.dep in ("dobj", "nmod", "nmod:of"):
                    for member in self.arguments._coordinate_members_n(child, parsed):
                        if node_protein_indices(member):
                            protein_spans.append(single_token_span_n(member))
                elif child.dep == "conj":
                    for member in self.arguments._coordinate_members_n(child, parsed):
                        if node_protein_indices(member):
                            protein_spans.append(single_token_span_n(member))
            if len(protein_spans) < 2:
                continue

            seen_offsets: set[tuple[int, int]] = set()
            deduped_spans = []
            for sp in protein_spans:
                key = sp.head_offset
                if key not in seen_offsets:
                    seen_offsets.add(key)
                    deduped_spans.append(sp)
            if len(deduped_spans) < 2:
                continue

            pred_span = single_token_span_n(complex_head, force_nominal=True)
            participant = CandidateArgument(
                core=single_token_span_n(complex_head, force_nominal=True),
                owners=tuple(deduped_spans),
                role="compound_state",
            )
            intrinsic_group = looks_complex((complex_head.lemma or "").lower())
            events.append(CandidateEvent(
                event_type=EventType.INTERACTION_STATE,
                event_id=next_id(),
                parser_source=self.parser_name,
                extraction_detail=(
                    "DEP-Complex" if intrinsic_group
                    else "DEP-collective-membership-evidence-only"
                ),
                construction="compound_state",
                sentence_text=sentence_text,
                predicate=pred_span,
                arguments={
                    "participants" if intrinsic_group else "evidence": (participant,)
                },
                is_directed=False,
            ))
            membership_acl_emitted.add(complex_head.i)

        for token in parsed.nodes:
            is_complex, head_token = self._resolve_complex_head_n(token, parsed)
            if not is_complex or head_token is None:
                continue
            if head_token.i in membership_acl_emitted and token.i == head_token.i:
                continue

            pred_span = single_token_span_n(head_token, force_nominal=True)

            if self.nominals._is_process_nominalization_n(head_token, parsed):
                continue

            participant = self.arguments._build_argument_from_parsed(token, parsed)
            if len(participant.all_protein_indices) < 2:
                continue
            signature = (head_token.i, tuple(participant.all_protein_indices))
            if signature in complex_state_signatures:
                continue
            complex_state_signatures.add(signature)

            events.append(CandidateEvent(
                event_type=EventType.INTERACTION_STATE,
                event_id=next_id(),
                parser_source=self.parser_name,
                extraction_detail="DEP-Complex",
                construction="compound_state",
                sentence_text=sentence_text,
                predicate=pred_span,
                arguments={
                    "participants": (participant,),
                },
                is_directed=False,
            ))
        return events

    def _extract_compound_adj_ppi_from_parsed(
        self, parsed, next_id, sentence_text: str,
    ) -> list[CandidateEvent]:
        from .node_helpers import (
            children_by_dep_n, node_protein_indices, case_lemma_n,
            single_token_span_n,
        )
        from .roles import DEP_AMOD, DEP_NMOD_NPMOD, DEP_NMOD_OF, DEP_COMPOUND, DEP_NMOD, DEP_APPOS
        from .roles import PREP_OF
        events: list[CandidateEvent] = []
        for head in parsed.nodes:
            if head.pos not in ("NOUN", "PROPN"):
                continue

            npmod_prot_pairs: list[tuple] = []
            for amod in parsed.children(head):
                if amod.dep != DEP_AMOD:
                    continue
                for npmod in parsed.children(amod):
                    if npmod.dep != DEP_NMOD_NPMOD:
                        continue
                    for cc in self.arguments._coordinate_members_n(npmod, parsed):
                        if node_protein_indices(cc):
                            npmod_prot_pairs.append((cc, amod))
            if not npmod_prot_pairs:
                continue

            target_prots = []
            if node_protein_indices(head):
                target_prots.append(head)
            for c in parsed.children(head):
                if c.dep == DEP_COMPOUND:
                    for cc in self.arguments._coordinate_members_n(c, parsed):
                        if node_protein_indices(cc):
                            target_prots.append(cc)
                elif c.dep in (DEP_NMOD, DEP_NMOD_OF):
                    prep = case_lemma_n(c, parsed)
                    if prep == PREP_OF or c.dep == DEP_NMOD_OF:
                        for cc in self.arguments._coordinate_members_n(c, parsed):
                            if node_protein_indices(cc):
                                target_prots.append(cc)
                    elif prep in ("in", "for", "among", "within"):
                        for cc in self.arguments._coordinate_members_n(c, parsed):
                            if node_protein_indices(cc):
                                target_prots.append(cc)
                elif c.dep == DEP_APPOS:
                    for cc in self.arguments._coordinate_members_n(c, parsed):
                        if node_protein_indices(cc):
                            target_prots.append(cc)
            if not target_prots:
                continue

            for agent, amod in npmod_prot_pairs:
                agent_arg = CandidateArgument(
                    core=single_token_span_n(agent), role="nmod:npmod",
                )
                for tgt in target_prots:
                    if tgt.i == agent.i:
                        continue
                    tgt_arg = CandidateArgument(
                        core=single_token_span_n(tgt), role=tgt.dep or "target",
                    )
                    pred_span = single_token_span_n(head, force_nominal=True)
                    events.append(CandidateEvent(
                        event_type=EventType.INTERACTION,
                        event_id=next_id(),
                        parser_source=self.parser_name,
                        extraction_detail=f"DEP-compound-adj-{(amod.lemma or amod.text).lower()}",
                        construction="compound_state",
                        sentence_text=sentence_text,
                        predicate=pred_span,
                        arguments={
                            "sources": (agent_arg,),
                            "targets": (tgt_arg,),
                        },
                        is_directed=True,
                    ))
        return events

    def _resolve_complex_head_n(self, token, parsed) -> tuple[bool, Optional[object]]:
        from .node_helpers import match_keys_for_node
        from .morphology import looks_agentive, looks_complex

        def _is_complex_noun_n(t) -> bool:
            if t.pos not in {"NOUN", "PROPN"}:
                return False
            lem = (t.lemma or "").lower()
            if looks_agentive(lem):
                return False
            if self.arguments._matches_group_pattern(match_keys_for_node(t)):
                return True
            if self.arguments._matches_collective_pattern(match_keys_for_node(t)):
                return True
            return looks_complex(lem)

        if _is_complex_noun_n(token):
            return True, token
        head = parsed.head(token)
        if head is None:
            return False, None
        if _is_complex_noun_n(head):
            return True, head
        if (head.dep in _MEMBERSHIP_ACL_DEPS
                and (head.lemma or "").lower() in _MEMBERSHIP_PREDICATES):
            grandparent = parsed.head(head)
            if grandparent is not None and grandparent.i != head.i and _is_complex_noun_n(grandparent):
                return True, grandparent
        return False, None


StateNominalEventMixin = StateNominalEventChannel  # Compatibility import.
