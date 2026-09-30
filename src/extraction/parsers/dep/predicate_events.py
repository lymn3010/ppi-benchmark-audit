
from __future__ import annotations
from src.parsing.syntax import ParsedSentence, SyntaxNode
from .rules.predicate_subjects import bind_predicate_subjects
from .nominal_paths import nominal_has_protein

from dataclasses import replace as dc_replace

from src.models.candidate import CandidateArgument, CandidateEvent
from src.extraction.rules import RuleApplication, append_rule_application
from src.system_rules import rule_frozenset

from .rules import (
    ADVCL_SCOPE_BARRIER_RULE_ID,
    RECIPROCAL_PURPOSE_SUBJECT_RULE_ID,
    SUBJECT_ONLY_RECIPROCITY_GATE_RULE_ID,
    ACL_CONJ_SUBJECT_INHERITANCE_RULE_ID,
    allow_subject_only_participants,
    bind_nested_controller,
)

from .roles import BINARY_CASE_MARKERS, DEP_ADVCL, DEP_COMPOUND, SAME_NP_PROTEIN_DEPS

_SUBJECT_INHERIT_POS = rule_frozenset("syntax.subject_inherit_pos")
_PARTICIPLE_XPOS = rule_frozenset("syntax.participle_xpos")
_NOMINAL_RELATION_COMPLEMENTS = rule_frozenset(
    "prepositions.nominal_relation_complements"
)
_EVIDENTIAL_SCOPE_PREDICATES = rule_frozenset(
    "event_composition.evidential_scope_predicates"
)


from .channel import EventChannel


class PredicateEventChannel(EventChannel):
    """Record verbal relation evidence in dependency traversal order."""

    _BINARY_CASE_MARKERS = BINARY_CASE_MARKERS

    def __init__(self, resources=None, context=None):
        super().__init__(resources, context)
        from .rules.nested_event import NestedEventBuilder
        self.nested = NestedEventBuilder(resources, context)

    def _maybe_build_nested_inner_n(self, *args, **kwargs):
        return self.nested._maybe_build_nested_inner_n(*args, **kwargs)


    def _extract_predicate_events_from_parsed(
        self,
        parsed: ParsedSentence,
        next_id,
        sentence_text: str,
        consumed_inner_heads: set[int],
    ) -> list[CandidateEvent]:
        from src.parsing.syntax import SyntaxNode
        from .node_helpers import (
            children_by_dep_n,
            conj_chain_n,
            case_lemma_n,
            case_phrase_n,
            case_token_n,
            match_keys_for_node,
            single_token_span_n,
            neg_flag_n,
            conj_negation_excluded_n,
            conj_root_n,
        )
        from src.models.candidate import CandidateArgument, CandidateEvent, EventType
        from .roles import (
            BINARY_CASE_MARKERS,
            DEP_CONJ,
            DEP_DOBJ,
            DEP_NMOD,
            DEP_NSUBJ,
            DEP_NSUBJPASS,
            PREP_BY,
            PREP_AS,
            PREP_LIKE,
            CONTEXTUAL_CASE_MARKERS,
        )

        events: list[CandidateEvent] = []

        for node in parsed.nodes:
            if node.pos != "VERB":
                continue
            if node.i in consumed_inner_heads:
                continue

            nmod_children = children_by_dep_n(node, parsed, DEP_NMOD)
            subj_children = children_by_dep_n(node, parsed, DEP_NSUBJ)
            pass_subj_children = children_by_dep_n(node, parsed, DEP_NSUBJPASS)
            dobj_children = children_by_dep_n(node, parsed, DEP_DOBJ)

            subj_children, pass_subj_children, acl_conj_subject_evidence = bind_predicate_subjects(
                node, parsed, subj_children, pass_subj_children, self.resources,
            )
            node_head = parsed.head(node)

            agent_children: list[SyntaxNode] = []
            patient_nmod: list[SyntaxNode] = []
            context_nmod: list[SyntaxNode] = []
            coagent_nmod: list[SyntaxNode] = []

            verb_keys = match_keys_for_node(node)

            for nmod in nmod_children:
                prep = case_phrase_n(nmod, parsed)
                if prep == PREP_BY:
                    agent_children.append(nmod)
                elif prep == PREP_AS:
                    pass
                elif prep == PREP_LIKE:
                    # Comparative "like P1" signals similarity, not a binding partner.
                    pass
                elif prep in CONTEXTUAL_CASE_MARKERS:
                    # Setting PPs ("in P1 cells") are context; eventive PPs are nested arguments.
                    if nmod.is_nominalized and nominal_has_protein(nmod, parsed):
                        patient_nmod.append(nmod)
                    elif nominal_has_protein(nmod, parsed):
                        context_nmod.append(nmod)
                elif nominal_has_protein(nmod, parsed):
                    patient_nmod.append(nmod)
                elif (prep in BINARY_CASE_MARKERS | frozenset({"to"})
                      and nmod.pos in ("NOUN", "PROPN")):
                    # Non-protein patient: coordinated subjects are co-agents, not interactors.
                    coagent_nmod.append(nmod)

            lexically_undirected = self._is_undirected(verb_keys)
            is_undirected = lexically_undirected
            has_purpose_clause = self._has_purpose_clause_n(node, parsed)
            reciprocal_purpose_rule_applied = False

            sources: list[CandidateArgument] = []
            targets: list[CandidateArgument] = []
            participants: list[CandidateArgument] = []
            passive_subject_args: list[CandidateArgument] = []
            patient_nmod_args: list[CandidateArgument] = []
            context_args: list[CandidateArgument] = []
            nested_pairs: list[tuple[CandidateArgument, CandidateEvent]] = []

            def _source_head_for_emission(s_head: SyntaxNode) -> SyntaxNode:
                if conj_negation_excluded_n(s_head, parsed):
                    root = conj_root_n(s_head, parsed)
                    if root is not s_head:
                        return root
                return s_head

            if pass_subj_children:
                for s_head in pass_subj_children:
                    for s in self._entity_chain_with_accompaniment_n(s_head, parsed):
                        nested = self._maybe_build_nested_inner_n(
                            s, next_id, sentence_text, consumed_inner_heads, node, parsed,
                        )
                        if nested is not None:
                            nested_pairs.append((CandidateArgument(
                                core=None, event_ref=nested.event_id, role="nsubjpass",
                            ), nested))
                        else:
                            arg = self.arguments._build_argument_from_parsed(s, parsed)
                            targets.append(arg)
                            passive_subject_args.append(arg)
                for a_head in agent_children:
                    for a in conj_chain_n(a_head, parsed):
                        sources.append(self.arguments._build_argument_from_parsed(a, parsed))
            else:
                for s_head in subj_children:
                    s_head = _source_head_for_emission(s_head)
                    sources.extend(
                        self._entity_chain_arguments_with_accompaniment_n(s_head, parsed)
                    )
                for o_head in dobj_children:
                    for o in conj_chain_n(o_head, parsed):
                        nested = self._maybe_build_nested_inner_n(
                            o, next_id, sentence_text, consumed_inner_heads, node, parsed,
                        )
                        if nested is not None:
                            nested_pairs.append((CandidateArgument(
                                core=None, event_ref=nested.event_id, role="dobj",
                            ), nested))
                        else:
                            targets.append(self.arguments._build_argument_from_parsed(o, parsed))

            for nmod_head in patient_nmod:
                if nmod_head.i in consumed_inner_heads:
                    continue
                head_case_tok = case_token_n(nmod_head, parsed)
                for nmod in conj_chain_n(nmod_head, parsed):
                    # Skip conjuncts excluded by "but not with X".
                    if nmod.dep == DEP_CONJ and conj_negation_excluded_n(nmod, parsed):
                        continue
                    nested = self._maybe_build_nested_inner_n(

                        nmod, next_id, sentence_text, consumed_inner_heads, node, parsed,
                    )
                    if nested is not None:
                        nested_pairs.append((CandidateArgument(
                            core=None, event_ref=nested.event_id,
                            role=f"nmod:{case_lemma_n(nmod_head, parsed)}",
                        ), nested))
                    else:
                        own_case_tok = case_token_n(nmod, parsed) or head_case_tok
                        arg = self.arguments._build_argument_from_parsed(
                            nmod, parsed, own_case_token=own_case_tok,
                        )
                        # Only the event-role PP keys the pattern (``to``, not ``of``).
                        direct_prep = case_phrase_n(nmod, parsed) or case_phrase_n(
                            nmod_head, parsed,
                        )
                        if direct_prep:
                            arg = dc_replace(arg, role=f"{DEP_NMOD}:{direct_prep}")
                        targets.append(arg)
                        patient_nmod_args.append(arg)

            for nmod_head in context_nmod:
                head_case_tok = case_token_n(nmod_head, parsed)
                context_args.append(dc_replace(
                    self.arguments._build_argument_from_parsed(
                        nmod_head, parsed, own_case_token=head_case_tok,
                    ),
                    role=f"context:{case_phrase_n(nmod_head, parsed)}",
                ))

            had_object_control_target = bool(has_purpose_clause and targets)
            if had_object_control_target:
                context_args.extend(
                    dc_replace(arg, role="control:object")
                    for arg in targets
                )
                targets = []
                patient_nmod_args = []

            # Subject-only pairs need a reciprocal predicate (interact, bind, co-purify).
            subject_only_active_count = len(sources)
            subject_only_passive_count = len(passive_subject_args)
            subject_only_reciprocity_candidate = bool(
                (
                    subject_only_active_count >= 2
                    and not targets
                    and not nested_pairs
                    and not coagent_nmod
                    and not had_object_control_target
                )
                or (
                    pass_subj_children
                    and subject_only_passive_count >= 2
                    and not sources
                    and not patient_nmod
                    and not nested_pairs
                )
            )
            subject_only_reciprocal = bool(
                lexically_undirected
                or (
                    self.resources
                    and self.resources.match_lexicon(
                        "subject_only_reciprocal_predicate", verb_keys,
                    )
                )
            )
            if subject_only_reciprocity_candidate and subject_only_reciprocal:
                is_undirected = True

            if not is_undirected and pass_subj_children and patient_nmod \
               and not sources and not nested_pairs:
                target_heads = {
                    arg.core.head_offset
                    for arg in targets
                    if arg.core is not None and arg.all_protein_indices
                }
                if len(target_heads) >= 2:
                    is_undirected = True

            if is_undirected:
                if pass_subj_children and patient_nmod_args and not sources:
                    # Passive with patient nmod: subject <-> nmod pairs only, no same-side cliques.
                    source_side: list[CandidateArgument] = list(passive_subject_args)
                    patient_side: list[CandidateArgument] = list(patient_nmod_args)
                else:
                    source_side = list(sources)
                    patient_side = list(targets)

                def _dedup(xs: list[CandidateArgument]) -> list[CandidateArgument]:
                    seen_keys: set[tuple[int, int]] = set()
                    out: list[CandidateArgument] = []
                    for x in xs:
                        k = x.core.head_offset if x.core else (-1, -1)
                        if k in seen_keys:
                            continue
                        seen_keys.add(k)
                        out.append(x)
                    return out

                sources_dedup = _dedup(source_side)
                patients_dedup = _dedup(patient_side)

                if sources_dedup and patients_dedup:
                    sources = sources_dedup
                    targets = patients_dedup
                    is_undirected = False
                else:
                    passive_with_nonprotein_nmod = (
                        not sources_dedup
                        and any(
                            case_lemma_n(n, parsed) in BINARY_CASE_MARKERS
                            and not nominal_has_protein(n, parsed)
                            for n in nmod_children
                        )
                    )
                    # Undirected verb with a non-protein patient: subjects are co-agents.
                    coagent_suppression = bool(
                        coagent_nmod and len(sources_dedup) >= 2
                    )
                    participant_reciprocity_licensed = bool(
                        lexically_undirected
                        or (
                            subject_only_reciprocity_candidate
                            and subject_only_reciprocal
                        )
                    )
                    allow_participants = allow_subject_only_participants(
                        lexically_undirected=participant_reciprocity_licensed,
                        has_purpose_clause=has_purpose_clause,
                    )
                    if allow_participants \
                       and not passive_with_nonprotein_nmod \
                       and not coagent_suppression:
                        participants = _dedup(sources_dedup + patients_dedup)
                        reciprocal_purpose_rule_applied = bool(
                            participant_reciprocity_licensed
                            and has_purpose_clause
                            and participants
                        )
                    sources = []
                    targets = []

            nested_pairs = [
                (
                    reference,
                    bind_nested_controller(
                        inner,
                        reference,
                        tuple(sources or passive_subject_args),
                        outer_lemma=(node.lemma or node.text or "").lower(),
                        resources=self.resources,
                    ),
                )
                for reference, inner in nested_pairs
            ]
            for _, inner in nested_pairs:
                events.append(inner)
            targets.extend(ref for ref, _ in nested_pairs)

            has_any = bool(sources or targets or participants or context_args)
            if not has_any:
                continue

            assertion_status = self.arguments._token_assertion_status_n(node, parsed)
            pred_span = single_token_span_n(
                node,
                is_negated=assertion_status == "negated",
                assertion_status=assertion_status,
            )
            arguments: dict[str, tuple[CandidateArgument, ...]] = {}
            if participants:
                arguments["participants"] = tuple(participants)
            if sources:
                arguments["sources"] = tuple(sources)
            if targets:
                arguments["targets"] = tuple(targets)
            if context_args:
                arguments["contexts"] = tuple(context_args)

            is_passive_event = bool(pass_subj_children)

            # Evidential report verbs (demonstrate, show) do not project subject x object pairs.
            if (node.lemma or node.text or "").lower() in _EVIDENTIAL_SCOPE_PREDICATES:
                continue

            detail = self._predicate_detail_n(
                node,
                has_nested=bool(nested_pairs),
                is_passive=is_passive_event,
                is_undirected=is_undirected,
            )

            event = CandidateEvent(
                event_type=EventType.INTERACTION,
                event_id=next_id(),
                parser_source=self.parser_name,
                extraction_detail=detail,
                construction="passive" if is_passive_event else "verbal",
                sentence_text=sentence_text,
                predicate=pred_span,
                arguments=arguments,
                is_directed=not is_undirected,
            )
            if subject_only_reciprocity_candidate:
                event = append_rule_application(event, RuleApplication(
                    rule_id=SUBJECT_ONLY_RECIPROCITY_GATE_RULE_ID,
                    stage="event_argument_binding",
                    decision="applied" if subject_only_reciprocal else "blocked",
                    reason=(
                        "lexically_reciprocal_subjects_become_participants"
                        if subject_only_reciprocal
                        else "ordinary_coordinated_subjects_remain_coarguments"
                    ),
                    evidence={
                        "predicate": (node.lemma or node.text or "").lower(),
                        "active_subject_arguments": subject_only_active_count,
                        "passive_subject_arguments": subject_only_passive_count,
                        "has_purpose_clause": has_purpose_clause,
                        "had_object_control_target": had_object_control_target,
                    },
                ))
            if reciprocal_purpose_rule_applied:
                event = append_rule_application(event, RuleApplication(
                    rule_id=RECIPROCAL_PURPOSE_SUBJECT_RULE_ID,
                    stage="event_argument_binding",
                    reason="reciprocal_subject_survives_purpose_clause",
                    evidence={"predicate": (node.lemma or node.text or "").lower()},
                ))
            if acl_conj_subject_evidence is not None:
                event = append_rule_application(event, RuleApplication(
                    rule_id=ACL_CONJ_SUBJECT_INHERITANCE_RULE_ID,
                    stage="event_argument_binding",
                    reason="conjunct_predicate_inherits_subject_from_acl_nominal_head",
                    evidence=acl_conj_subject_evidence,
                ))
            if (
                node.dep == DEP_ADVCL
                and node_head is not None
                and node_head.pos != "ADJ"
                and neg_flag_n(node_head, parsed)
                and assertion_status == "asserted"
            ):
                event = append_rule_application(event, RuleApplication(
                    rule_id=ADVCL_SCOPE_BARRIER_RULE_ID,
                    stage="assertion_scope",
                    reason="matrix_negation_does_not_cross_advcl",
                    evidence={"matrix_head": node_head.i, "predicate_head": node.i},
                ))
            events.append(event)

        return events

    def _predicate_detail_n(
        self, node: SyntaxNode, has_nested: bool, is_passive: bool, is_undirected: bool,
    ) -> str:
        base = f"DEP-{(node.lemma or node.text).lower()}"
        suffix = ""
        if is_undirected and is_passive:
            suffix = "-passive-undirected"
        elif is_undirected:
            suffix = "-undirected"
        elif is_passive:

            suffix = "-passive"
        if has_nested:
            suffix = f"{suffix}-nested" if suffix else "-nested"
        return f"{base}{suffix}"

    def _has_purpose_clause_n(self, node: SyntaxNode, parsed: ParsedSentence) -> bool:
        from .roles import DEP_XCOMP, DEP_MARK, PREP_TO, DEP_ADVCL
        for child in parsed.children(node):
            if child.dep == DEP_XCOMP:
                tag = child.raw_pos or ""
                is_infinitival = tag in ("VB", "VBP")
                has_to_mark = any(
                    m.dep == DEP_MARK and (m.text or "").lower() == PREP_TO
                    for m in parsed.children(child)
                )
                if is_infinitival or has_to_mark:
                    return True
            if child.dep == DEP_ADVCL:
                for mark in parsed.children(child):
                    if mark.dep == DEP_MARK and (mark.text or "").lower() == PREP_TO:
                        return True
        return False


    def _entity_chain_with_accompaniment_n(self, head: SyntaxNode, parsed: ParsedSentence) -> list[SyntaxNode]:
        from .node_helpers import accompaniment_nmods_n, conj_chain_n
        absorbing_complex = self._absorbing_complex_conj_n(head, parsed)
        out: list[SyntaxNode] = []
        seen: set[int] = set()
        for member in self.arguments._coordinate_members_n(head, parsed):
            if absorbing_complex is not None and member.i == head.i:
                continue
            if member.i not in seen:
                out.append(member)
                seen.add(member.i)
            for nmod in accompaniment_nmods_n(member, parsed):
                for accomp in conj_chain_n(nmod, parsed):

                    if accomp.i in seen:
                        continue
                    out.append(accomp)
                    seen.add(accomp.i)
        return out

    def _entity_chain_arguments_with_accompaniment_n(
        self, head: SyntaxNode, parsed: ParsedSentence,
    ) -> list[CandidateArgument]:
        """Build source-side arguments, keeping ``P0 and P1 receptor`` as P0 plus ``P1 receptor``.

        Group nouns (``P0 and P1 complex``) stay one item with members P0/P1.
        """
        from .node_helpers import children_by_dep_n, conj_chain_n
        from .morphology import looks_complex

        def split_compound_members(n: SyntaxNode) -> tuple[list[SyntaxNode], frozenset[int]]:
            lemma = (n.lemma or n.text or "").lower()
            if n.is_nominalized or looks_complex(lemma):
                return [], frozenset()
            for comp in children_by_dep_n(n, parsed, DEP_COMPOUND):
                chain = [m for m in conj_chain_n(comp, parsed) if m.protein_indices]
                if len(chain) >= 2:
                    return [chain[0]], frozenset(m.i for m in chain[:1])
            return [], frozenset()

        args: list[CandidateArgument] = []
        seen_heads: set[int] = set()
        split_happened = False
        for member in self._entity_chain_with_accompaniment_n(head, parsed):
            promoted, exclude_from_head = split_compound_members(member)
            split_happened = split_happened or bool(promoted)
            member_build = self.arguments.build_argument_from_parsed(
                member,
                parsed,
                exclude_owner_node_indices=exclude_from_head,
            )
            root_group = member_build.group
            for promoted_member in promoted:
                if promoted_member.i in seen_heads:
                    continue
                promoted_arg = self.arguments._build_argument_from_parsed(promoted_member, parsed)
                args.append(dc_replace(
                    promoted_arg,
                    descriptors=member_build.candidate.descriptors,
                    group=root_group.to_dict(),
                ))
                seen_heads.add(promoted_member.i)
            if member.i not in seen_heads:
                args.append(member_build.candidate)
                seen_heads.add(member.i)
        if split_happened:
            from .semantic_groups import GroupMember, SemanticGroup, KIND_COORDINATION
            def direct_arg_indices(arg: CandidateArgument) -> tuple[int, ...]:
                proteins: set[int] = set()
                if arg.core is not None:
                    proteins.update(arg.core.protein_indices)
                for owner in arg.owners:
                    proteins.update(owner.protein_indices)
                for descriptor in arg.descriptors:
                    proteins.update(descriptor.protein_indices)
                return tuple(sorted(proteins))

            members = tuple(
                GroupMember(
                    head_index=(
                        arg.core.node_indices[0]
                        if arg.core is not None and arg.core.node_indices
                        else -1
                    ),
                    head_protein_indices=direct_arg_indices(arg),
                )
                for arg in args
                if direct_arg_indices(arg)
            )
            if members:
                group = SemanticGroup(
                    kind=KIND_COORDINATION,
                    head_index=head.i,
                    members=members,
                ).to_dict()
                args = [dc_replace(arg, group=group) for arg in args]
        return args

    @staticmethod
    def _absorbing_complex_conj_n(head: SyntaxNode, parsed: ParsedSentence) -> SyntaxNode | None:
        from .node_helpers import node_protein_indices, conj_chain_n
        from .morphology import looks_complex
        if not node_protein_indices(head):
            return None
        for member in conj_chain_n(head, parsed):
            if member.i == head.i:
                continue
            lemma = (member.lemma or member.text or "").lower()
            if not looks_complex(lemma):
                continue
            if any(
                node_protein_indices(child)
                for child in parsed.children(member)
                if child.dep in SAME_NP_PROTEIN_DEPS
            ):
                return member
        return None


    def _is_undirected(self, keys: set[str]) -> bool:
        return bool(
            self.resources
            and self.resources.match_lexicon("reciprocal_predicate", keys)
        )


PredicateEventMixin = PredicateEventChannel  # Compatibility import.
