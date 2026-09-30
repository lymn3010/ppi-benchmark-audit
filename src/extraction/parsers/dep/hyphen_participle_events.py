"""Hyphen-participle event channel (own voice and token-merging rules)."""
from __future__ import annotations

from src.models.candidate import CandidateArgument, CandidateEvent, EventType, Span
from src.system_rules import rule_frozenset

from .node_helpers import (
    hyphen_adjacent_protein_children_n,
    merged_participle_clean_lemma_n,
    merged_participle_clean_surface_n,
    node_protein_indices,
    single_token_span_n,
)
from .roles import DEP_AMOD, DEP_APPOS

_PARTICIPLE_XPOS = rule_frozenset("syntax.participle_xpos")


from .channel import EventChannel


class HyphenParticipleEventChannel(EventChannel):
    """Extract reduced-relative events encoded as hyphen compounds."""

    def _extract_hyphen_participle_ppi_from_parsed(
        self, parsed, next_id, sentence_text: str,
    ) -> list[CandidateEvent]:
        """Emit directed events with direction determined by participle voice."""
        events: list[CandidateEvent] = []
        for amod in parsed.nodes:
            if amod.dep != DEP_AMOD:
                continue
            amod_protein_indices = node_protein_indices(amod)
            if not amod_protein_indices:
                if amod.pos != "VERB" or (amod.raw_pos or "") not in _PARTICIPLE_XPOS:
                    continue
            head = parsed.head(amod)
            if head is None:
                continue
            head_target = head
            if not node_protein_indices(head_target):
                antecedent = parsed.head(head) if head.dep == DEP_APPOS else None
                if antecedent is None or not node_protein_indices(antecedent):
                    continue
                head_target = antecedent

            clean_surface_for_voice = (
                merged_participle_clean_surface_n(amod) if amod_protein_indices else None
            )
            is_active = (
                (amod.raw_pos or "") == "VBG"
                or (
                    (amod.raw_pos or "") not in _PARTICIPLE_XPOS
                    and bool(clean_surface_for_voice)
                    and clean_surface_for_voice.rsplit("-", 1)[-1].endswith("ing")
                )
            )
            voice = "active" if is_active else "passive"
            head_arg = CandidateArgument(
                core=single_token_span_n(head_target), role=head.dep or "target",
            )

            if amod_protein_indices:
                if amod_protein_indices == node_protein_indices(head_target):
                    continue
                clean_surface = merged_participle_clean_surface_n(amod)
                clean_lemma = merged_participle_clean_lemma_n(amod)
                if clean_surface is None or clean_lemma is None:
                    continue
                pred_span = Span(
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
                mod_arg = CandidateArgument(
                    core=single_token_span_n(amod),
                    role="amod:patient" if is_active else "amod:agent",
                )
                sources, targets = (
                    ((head_arg,), (mod_arg,)) if is_active else ((mod_arg,), (head_arg,))
                )
                events.append(CandidateEvent(
                    event_type=EventType.INTERACTION,
                    event_id=next_id(),
                    parser_source=self.parser_name,
                    extraction_detail=f"DEP-hyphen-participle-merged-{voice}-{clean_lemma}",
                    construction="hyphen_participle",
                    sentence_text=sentence_text,
                    predicate=pred_span,
                    arguments={"sources": sources, "targets": targets},
                    is_directed=True,
                ))
                continue

            modifiers = hyphen_adjacent_protein_children_n(amod, parsed)
            if not modifiers:
                continue
            lemma = (amod.lemma or amod.text or "").lower()
            pred_span = single_token_span_n(amod)
            for mod in modifiers:
                if mod.i == head.i:
                    continue
                mod_arg = CandidateArgument(
                    core=single_token_span_n(mod),
                    role="amod:patient" if is_active else "amod:agent",
                )
                sources, targets = (
                    ((head_arg,), (mod_arg,)) if is_active else ((mod_arg,), (head_arg,))
                )
                events.append(CandidateEvent(
                    event_type=EventType.INTERACTION,
                    event_id=next_id(),
                    parser_source=self.parser_name,
                    extraction_detail=f"DEP-hyphen-participle-{voice}-{lemma}",
                    construction="hyphen_participle",
                    sentence_text=sentence_text,
                    predicate=pred_span,
                    arguments={"sources": sources, "targets": targets},
                    is_directed=True,
                ))
        return events


__all__ = ["HyphenParticipleEventMixin"]


HyphenParticipleEventMixin = HyphenParticipleEventChannel  # Compatibility import.
