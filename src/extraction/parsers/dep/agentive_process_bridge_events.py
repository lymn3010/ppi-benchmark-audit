"""Find agentive participles attached to process nouns and collect bridge evidence."""
from __future__ import annotations

from src.models.candidate import CandidateArgument, Span

from .node_helpers import (
    hyphen_adjacent_protein_children_n,
    merged_participle_clean_lemma_n,
    merged_participle_clean_surface_n,
    node_protein_indices,
    single_token_span_n,
)
from .roles import DEP_AMOD, DEP_CASE, DEP_COMPOUND, DEP_NMOD, PREP_OF
from .rules.agentive_process_bridge import (
    AGENTIVE_PROCESS_BRIDGE_RULE_ID,
    AGENTIVE_PROCESS_BRIDGE_PROJECTION_RULE_ID,
    build_agentive_process_bridge_events,
    is_agentive_bridge_predicate,
    is_agentive_process_nominal,
)


from .channel import EventChannel


class AgentiveProcessBridgeEventChannel(EventChannel):
    """Extract agentive-process bridge evidence and optional projection events."""

    def _extract_agentive_process_bridge_from_parsed(
        self, parsed, next_id, sentence_text: str,
    ):
        if self.resources and not self.resources.rule_enabled(AGENTIVE_PROCESS_BRIDGE_RULE_ID):
            return []
        events = []
        for amod in parsed.nodes:
            if amod.dep != DEP_AMOD:
                continue
            resolved = self._resolve_agentive_controller(amod, parsed)
            if resolved is None:
                continue
            controller_node, bridge_lemma, bridge_predicate = resolved
            if not is_agentive_bridge_predicate(bridge_lemma):
                continue

            head = parsed.head(amod)
            if head is None or node_protein_indices(head):
                continue
            process_lemma = (head.lemma or head.text or "").lower()
            if not (head.is_nominalized or head.verb_form or is_agentive_process_nominal(process_lemma)):
                continue

            controller_index = node_protein_indices(controller_node)[0]
            affected = self._resolve_affected_protein(head, parsed, controller_index)
            if affected is None:
                continue
            affected_node, attachment = affected

            controller_arg = CandidateArgument(
                core=single_token_span_n(controller_node),
                role="agentive_controller",
            )
            affected_arg = CandidateArgument(
                core=single_token_span_n(affected_node),
                role=attachment,
            )
            process_predicate = single_token_span_n(head, force_nominal=True)
            surface = parsed.text[
                min(amod.char_start, controller_node.char_start):
                max(head.char_end, affected_node.char_end)
            ]
            events.extend(build_agentive_process_bridge_events(
                bridge_lemma=bridge_lemma,
                process_lemma=process_lemma,
                controller=controller_arg,
                affected=affected_arg,
                bridge_predicate=bridge_predicate,
                process_predicate=process_predicate,
                construction_surface=surface,
                affected_attachment=attachment,
                emit_projection_event=(
                    self.resources is None
                    or self.resources.rule_enabled(AGENTIVE_PROCESS_BRIDGE_PROJECTION_RULE_ID)
                ),
                next_id=next_id,
                sentence_text=sentence_text,
                parser_source=self.parser_name,
            ))
        return events

    def _resolve_agentive_controller(self, amod, parsed):
        """Return (controller, bridge_lemma, clean_predicate_span) or None.

        Handles merged ``PROTEIN0-mediated`` and split ``PROTEIN0 -induced`` tokens.
        """
        if node_protein_indices(amod):
            clean_surface = merged_participle_clean_surface_n(amod)
            clean_lemma = merged_participle_clean_lemma_n(amod)
            if clean_surface is None or clean_lemma is None:
                return None
            predicate = self._clean_participle_span(amod, clean_surface, clean_lemma)
            return amod, clean_lemma, predicate

        markers = hyphen_adjacent_protein_children_n(amod, parsed)
        if not markers:
            return None
        controller_node = markers[0]
        bridge_lemma = self._split_participle_lemma(amod)
        if not bridge_lemma:
            return None
        surface = (amod.text or "").lstrip("-").lower() or bridge_lemma
        predicate = self._clean_participle_span(amod, surface, bridge_lemma)
        return controller_node, bridge_lemma, predicate

    @staticmethod
    def _split_participle_lemma(amod) -> str:
        raw = (amod.lemma or amod.text or "").lower().lstrip("-")
        final = raw.rsplit("-", 1)[-1]
        if final and "-" not in final:
            return final
        surface = (amod.text or "").lower().lstrip("-").rsplit("-", 1)[-1]
        from src.models.candidate._nominal_canon import mapped_verbal_from_surface

        mapped = mapped_verbal_from_surface(surface)
        if mapped:
            return mapped
        try:
            import lemminflect

            forms = tuple(lemminflect.getLemma(surface, upos="VERB") or ())
        except Exception:
            forms = ()
        return forms[0].lower() if forms else surface

    @staticmethod
    def _clean_participle_span(amod, clean_surface: str, clean_lemma: str) -> Span:
        return Span(
            tokens=(clean_surface,),
            lemmas=(clean_lemma,),
            match_keys=tuple(sorted({clean_surface, clean_lemma})),
            protein_indices=(),
            char_offsets=((amod.char_start, amod.char_end),),
            token_lemmas=(clean_lemma,),
            head_node_index=amod.i,
            pos=amod.pos or "",
            dep=amod.dep or "",
            assertion_status="asserted",
            node_indices=(amod.i,),
            parser_dep=amod.raw_dep or amod.dep,
            parser_pos=amod.raw_pos or amod.pos,
            parser_features=amod.raw_feats,
            parser_misc=amod.raw_misc,
            enhanced_heads=amod.enhanced_heads,
        )

    def _resolve_affected_protein(self, head, parsed, controller_index: int):
        """Return (affected_node, attachment): compound (``P1 expression``) or ``of`` complement."""
        compound = None
        nmod_of = None
        for child in parsed.children(head):
            indices = node_protein_indices(child)
            if not indices or controller_index in indices:
                continue
            if child.dep == DEP_COMPOUND and compound is None:
                compound = child
            elif child.dep == DEP_NMOD and nmod_of is None and self._is_of_complement(child, parsed):
                nmod_of = child
        if compound is not None:
            return compound, "compound"
        if nmod_of is not None:
            return nmod_of, "nmod:of"
        return None

    @staticmethod
    def _is_of_complement(nmod, parsed) -> bool:
        for case in parsed.children(nmod):
            if case.dep == DEP_CASE and (case.lemma or case.text or "").lower() == PREP_OF:
                return True
        return False


__all__ = ["AgentiveProcessBridgeEventMixin"]


AgentiveProcessBridgeEventMixin = AgentiveProcessBridgeEventChannel  # Compatibility import.
