from __future__ import annotations

from collections import deque
from dataclasses import replace as dc_replace

from src.models.candidate import CandidateArgument, CandidateEvent, EventType
from src.models.candidate._nominal_canon import canonicalize_predicate_lemma
from src.parsing.syntax import ParsedSentence, SyntaxNode

from .node_helpers import (
    children_by_dep_n,
    node_protein_indices,
    conj_chain_n,
    case_phrase_n,
    case_token_n,
    match_keys_for_node,
    single_token_span_n,
    conj_chain_has_alternative_n,
    conj_chain_aa_anchor_n,
)
from .roles import (
    DEP_APPOS,
    DEP_AMOD,
    DEP_COMPOUND,
    DEP_COP,
    DEP_NMOD,
    DEP_NSUBJ,
    DEP_NSUBJPASS,
    CONTEXTUAL_CASE_MARKERS,
    BINARY_CASE_MARKERS,
    NOMINAL_SOURCE_DEPS,
    PREP_BY,
    SAME_NP_TRAVERSAL_DEPS,
)


from .channel import EventChannel


class NominalizationEventChannel(EventChannel):
    """Extract process-nominalization events such as 'binding of P0 by P1'."""


    def _extract_nominalization_events_from_parsed(
        self,
        parsed: ParsedSentence,
        next_id,
        sentence_text: str,
        consumed_inner_heads: set[int],
    ) -> list[CandidateEvent]:
        events: list[CandidateEvent] = []
        for token in parsed.nodes:
            if token.i in consumed_inner_heads:
                continue
            if token.pos != "NOUN":
                continue
            if not self._is_process_nominalization_n(token, parsed):
                continue
            if any(c.dep == DEP_COP for c in parsed.children(token)):
                continue
            token_head = parsed.head(token)
            if token_head is not None and token_head.pos == "VERB":
                if token.dep not in NOMINAL_SOURCE_DEPS:
                    continue

            sources: list[CandidateArgument] = []
            targets: list[CandidateArgument] = []
            contexts: list[CandidateArgument] = []
            prep_buckets: dict[str, list[CandidateArgument]] = {}

            def _emit(tgt_token: SyntaxNode, prep: str, role_prefix: str = DEP_NMOD) -> None:
                # Use the shared argument builder so assertion and case markers are kept.
                own_case_tok = case_token_n(tgt_token, parsed)
                arg = dc_replace(
                    self.arguments._build_argument_from_parsed(
                        tgt_token, parsed, own_case_token=own_case_tok,
                    ),
                    role=f"{role_prefix}:{prep}" if prep else role_prefix,
                )
                prep_buckets.setdefault(prep, []).append(arg)

            chain_has_alternative = False
            aa_anchor_indices: set[int] = set()

            for nmod_head in children_by_dep_n(token, parsed, DEP_NMOD):
                prep = case_phrase_n(nmod_head, parsed)
                if conj_chain_has_alternative_n(nmod_head, parsed):
                    chain_has_alternative = True
                    anchor_tok = conj_chain_aa_anchor_n(nmod_head, parsed)
                    if anchor_tok is not None:
                        aa_anchor_indices.update(node_protein_indices(anchor_tok))
                        for cc in parsed.children(anchor_tok):
                            if cc.dep in (DEP_COMPOUND, DEP_APPOS):
                                aa_anchor_indices.update(node_protein_indices(cc))
                for nmod in self.arguments._coordinate_members_n(nmod_head, parsed):
                    if (
                        nmod.i != nmod_head.i
                        and not node_protein_indices(nmod)
                        and self._is_process_nominalization_n(nmod, parsed)
                    ):
                        # A coordinated process nominal starts another event, not another endpoint.
                        continue
                    if node_protein_indices(nmod):
                        _emit(nmod, prep)
                    else:
                        # Descend same-NP deps to reach nested nmod proteins.
                        _bfs_seen: set[int] = {nmod.i}
                        _bfs_queue: deque = deque([nmod])
                        while _bfs_queue:
                            _bfs_node = _bfs_queue.popleft()
                            for _bfs_child in parsed.children(_bfs_node):
                                if _bfs_child.dep not in SAME_NP_TRAVERSAL_DEPS:
                                    continue
                                if (
                                    not node_protein_indices(_bfs_child)
                                    and self._is_process_nominalization_n(
                                        _bfs_child, parsed,
                                    )
                                ):
                                    continue
                                if conj_chain_has_alternative_n(_bfs_child, parsed):
                                    chain_has_alternative = True
                                    anchor_tok = conj_chain_aa_anchor_n(
                                        _bfs_child, parsed,
                                    )
                                    if anchor_tok is not None:
                                        aa_anchor_indices.update(
                                            node_protein_indices(anchor_tok)
                                        )
                                for c in conj_chain_n(_bfs_child, parsed):
                                    if c.i in _bfs_seen:
                                        continue
                                    _bfs_seen.add(c.i)
                                    if node_protein_indices(c):
                                        _emit(c, prep)
                                    # Also descend from protein nodes.
                                    _bfs_queue.append(c)

            compound_bucket: list[CandidateArgument] = []
            for comp in children_by_dep_n(token, parsed, DEP_COMPOUND, DEP_AMOD):
                for c in self.arguments._coordinate_members_n(comp, parsed):
                    pids = node_protein_indices(c)
                    if not pids:
                        continue
                    if c.dep == DEP_AMOD and len(pids) < 2:
                        continue
                    assertion_status = self.arguments._token_assertion_status_n(c, parsed)
                    core = single_token_span_n(
                        c,
                        is_negated=assertion_status == "negated",
                        assertion_status=assertion_status,
                    )
                    # A merged "PROTEIN0-PROTEIN1" modifier counts as two compound participants.
                    if len(pids) > 1:
                        for pid in pids:
                            compound_bucket.append(CandidateArgument(
                                core=dc_replace(core, protein_indices=(pid,)),
                                role=DEP_COMPOUND,
                            ))
                    else:
                        compound_bucket.append(CandidateArgument(
                            core=core, role=DEP_COMPOUND,
                        ))
            if compound_bucket:
                prep_buckets.setdefault("", []).extend(compound_bucket)

            if PREP_BY in prep_buckets:
                sources.extend(prep_buckets.pop(PREP_BY))
            for contextual_prep in tuple(prep_buckets):
                if contextual_prep in CONTEXTUAL_CASE_MARKERS:
                    contexts.extend(prep_buckets.pop(contextual_prep))
            non_by_preps = [p for p in prep_buckets.keys()]
            if len(non_by_preps) >= 2 and not sources:
                # No "by" agent: the first preposition in sentence order is the source.
                first = non_by_preps[0]
                sources.extend(prep_buckets[first])
                for p in non_by_preps[1:]:
                    targets.extend(prep_buckets[p])
            else:
                for p in non_by_preps:
                    targets.extend(prep_buckets[p])

            if not sources and len(targets) < 2:
                continue

            self_is_negation_nom = self.arguments._kw_match(
                "negation", {(token.lemma or token.text).lower()}
            )
            nom_negated = self_is_negation_nom or self.arguments._token_has_negation_cue_n(token, parsed)
            pred_span = single_token_span_n(
                token,
                force_nominal=True,
                is_negated=nom_negated,
            )
            arguments: dict[str, tuple[CandidateArgument, ...]] = {}
            if contexts:
                # Location PPs are evidence but not endpoints ("interaction of A and B in C").
                arguments["contexts"] = tuple(contexts)
            lemma = (token.lemma or token.text).lower()
            is_subj = token.dep in (DEP_NSUBJ, DEP_NSUBJPASS)

            if sources and targets:
                arguments["sources"] = tuple(sources)
                arguments["targets"] = tuple(targets)
                is_directed = True
                detail = f"DEP-nom-{lemma}-subject" if is_subj else f"DEP-nom-{lemma}"
            elif len(targets) >= 2 and not sources:
                reciprocal_supported = False
                binary_prep_supported = any(
                    arg.role.split(":", 1)[-1] in BINARY_CASE_MARKERS
                    for arg in targets
                )
                group_pattern_supported = self.arguments._matches_group_pattern(
                    match_keys_for_node(token)
                )
                if (
                    token.nominalization_source == "QANom"
                    and not (
                        reciprocal_supported
                        or binary_prep_supported
                        or group_pattern_supported
                    )
                ):
                    # Eventiveness alone licenses no pairs; unknown roles get no slot.
                    arguments["evidence"] = tuple(targets)
                    is_directed = False
                    detail = (f"DEP-nom-{lemma}-subject-evidence-only" if is_subj
                              else f"DEP-nom-{lemma}-evidence-only")
                elif chain_has_alternative:
                    anchor_set = aa_anchor_indices
                    anchor_arg = next(
                        (t for t in targets
                         if t.core is not None
                         and any(p in anchor_set for p in t.core.protein_indices)),
                        targets[0],
                    )
                    rest = [t for t in targets if t is not anchor_arg]
                    # Keep the full argument provenance from the shared builder.
                    arguments["sources"] = (anchor_arg,)
                    arguments["targets"] = tuple(rest)
                    is_directed = True
                    detail = (f"DEP-nom-{lemma}-subject-alt" if is_subj
                              else f"DEP-nom-{lemma}-alt")
                else:
                    arguments["participants"] = tuple(targets)
                    is_directed = False
                    detail = (f"DEP-nom-{lemma}-subject-undirected" if is_subj
                              else f"DEP-nom-{lemma}-undirected")
            else:
                continue

            events.append(CandidateEvent(
                event_type=EventType.INTERACTION,
                event_id=next_id(),
                parser_source=self.parser_name,
                extraction_detail=detail,
                construction="nominalized",
                sentence_text=sentence_text,
                predicate=pred_span,
                arguments=arguments,
                is_directed=is_directed,
            ))
            consumed_inner_heads.add(token.i)
        return events

    def _is_process_nominalization_n(self, token: SyntaxNode, parsed: ParsedSentence) -> bool:
        """Accept QANom nominalizations or a reciprocal-event NP pattern."""
        if token.is_nominalized:
            return True
        return self.arguments._matches_group_pattern(match_keys_for_node(token))


NominalizationEventMixin = NominalizationEventChannel  # Compatibility import.
