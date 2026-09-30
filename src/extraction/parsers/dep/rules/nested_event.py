"""Existing nested-event construction, preserving rule IDs and evidence order."""

from __future__ import annotations
from src.parsing.syntax import ParsedSentence, SyntaxNode
from ..nominal_paths import nominal_has_protein, has_binary_case_marker

from dataclasses import replace as dc_replace
from typing import Optional

from src.models.candidate import CandidateEvent
from src.extraction.rules import RULE_TRACE_KEY, RuleApplication
from src.system_rules import rule_frozenset

from ..rules import NESTED_NOMINAL_LOCAL_SCOPE_RULE_ID

from ..roles import (
    BINARY_CASE_MARKERS,
    DEP_CONJ,
    DEP_DOBJ,
    PREP_BY,
    SAME_NP_PROTEIN_DEPS,
    SAME_NP_TRAVERSAL_DEPS,
)

_NOMINAL_RELATION_COMPLEMENTS = rule_frozenset(
    "prepositions.nominal_relation_complements"
)
_EVIDENTIAL_SCOPE_PREDICATES = rule_frozenset(
    "event_composition.evidential_scope_predicates"
)


from ..channel import EventChannel


class NestedEventBuilder(EventChannel):
    _BINARY_CASE_MARKERS = BINARY_CASE_MARKERS

    def _maybe_build_nested_inner_n(
        self,
        pobj: SyntaxNode,
        next_id,
        sentence_text: str,
        consumed_inner_heads: set[int],
        outer_predicate,
        parsed: ParsedSentence,
    ) -> Optional[CandidateEvent]:
        from ..node_helpers import (
            match_keys_for_node, node_protein_indices, single_token_span_n,
            conj_chain_n, conj_chain_has_alternative_n, conj_chain_aa_anchor_n,
            conj_root_n, case_lemma_n, case_token_n,
        )
        from ..morphology import looks_agentive, looks_complex
        from src.models.candidate import CandidateArgument, CandidateEvent, EventType
        from ..roles import DEP_COMPOUND, DEP_NMOD, DEP_CONJ

        if pobj.i in consumed_inner_heads:
            return None
        lemma = (pobj.lemma or "").lower()
        keys = match_keys_for_node(pobj)

        if not pobj.is_nominalized and looks_agentive(lemma):
            return None

        pattern_supported = self.arguments._matches_group_pattern(keys)
        morph_supported = pobj.is_nominalized or looks_complex(lemma)

        if not (pattern_supported or morph_supported):
            return None
        if not pattern_supported and not has_binary_case_marker(pobj, parsed):
            return None

        protein_tokens = self._collect_inner_proteins_n(pobj, parsed)
        # Reattach a flattened ``with/to`` complement only for reciprocal nominals.
        detached_complement_heads: set[int] = set()
        if pattern_supported and pobj.dep == DEP_DOBJ:
            matrix = parsed.head(pobj)
            if matrix is not None:
                for sibling in parsed.children(matrix):
                    if sibling.i == pobj.i or sibling.dep != DEP_NMOD:
                        continue
                    if case_lemma_n(sibling, parsed) not in _NOMINAL_RELATION_COMPLEMENTS:
                        continue
                    if sibling.i < pobj.i or not nominal_has_protein(sibling, parsed):
                        continue
                    detached_complement_heads.add(sibling.i)
                    protein_tokens.extend(conj_chain_n(sibling, parsed))
        unique_proteins = set()
        for t in protein_tokens:
            unique_proteins.update(node_protein_indices(t))
        if len(unique_proteins) < 2:
            return None

        consumed_inner_heads.add(pobj.i)
        consumed_inner_heads.update(detached_complement_heads)
        nested_assertion = self.arguments._token_assertion_status_n(
            pobj,
            parsed,
            inherit_parent_negation=False,
        )
        outer_lemma = (
            outer_predicate.lemma or outer_predicate.text or ""
        ).lower()
        outer_assertion = self.arguments._token_assertion_status_n(outer_predicate, parsed)
        if (
            nested_assertion == "asserted"
            and outer_lemma in _EVIDENTIAL_SCOPE_PREDICATES
            and outer_assertion != "asserted"
        ):
            nested_assertion = outer_assertion
        nested_negated = nested_assertion == "negated"
        pred_span = single_token_span_n(
            pobj,
            force_nominal=True,
            is_negated=nested_negated,
            assertion_status=nested_assertion,
        )
        nested_scope_values = {
            RULE_TRACE_KEY: [RuleApplication(
                rule_id=NESTED_NOMINAL_LOCAL_SCOPE_RULE_ID,
                stage="assertion_scope",
                reason="nested_nominal_uses_local_scope_before_explicit_evidential_policy",
                evidence={
                    "nested_head": pobj.i,
                    "matrix_head": outer_predicate.i,
                    "matrix_lemma": outer_lemma,
                },
            ).to_dict()]
        }

        chain_has_alternative = False
        aa_anchor_indices: set[int] = set()
        inferred_anchor_indices: set[int] = set()

        # Recover nested alternatives ("[A or B] loop and C"); anchor on one outside protein.
        alternative_indices: set[int] = set()
        for token in protein_tokens:
            root = conj_root_n(token, parsed)
            if conj_chain_has_alternative_n(root, parsed):
                alternative_indices.update(node_protein_indices(token))
        if alternative_indices:
            chain_has_alternative = True
            outside_alternative = unique_proteins - alternative_indices
            if len(outside_alternative) == 1:
                inferred_anchor_indices.update(outside_alternative)

        for child in parsed.children(pobj):
            if child.dep not in (DEP_COMPOUND, DEP_NMOD, DEP_CONJ):
                continue
            if not conj_chain_has_alternative_n(child, parsed):
                continue
            chain_has_alternative = True
            anchor_tok = conj_chain_aa_anchor_n(child, parsed)
            if anchor_tok is not None:
                aa_anchor_indices.update(node_protein_indices(anchor_tok))
                for cc in parsed.children(anchor_tok):
                    if cc.dep in SAME_NP_PROTEIN_DEPS:
                        aa_anchor_indices.update(node_protein_indices(cc))
        if not aa_anchor_indices:
            aa_anchor_indices.update(inferred_anchor_indices)

        # Keep ``by`` direction on embedded event nominals.
        by_tokens = [t for t in protein_tokens if case_lemma_n(t, parsed) == PREP_BY]
        non_by_tokens = [t for t in protein_tokens if t not in by_tokens]
        def _nested_prep(token: SyntaxNode) -> str:
            prep = case_lemma_n(token, parsed)
            if prep:
                return prep
            root = conj_root_n(token, parsed)
            if root.i != token.i:
                prep = case_lemma_n(root, parsed)
                if prep:
                    return prep
            # The case marker may sit on a lexical head ("to [P1 and P2 proteins]").
            argument_head = self._nested_argument_head_n(token, pobj, parsed)
            if argument_head.i != token.i:
                return case_lemma_n(argument_head, parsed)
            return ""

        def _nested_arg(token: SyntaxNode, role: str) -> CandidateArgument:
            argument_head = self._nested_argument_head_n(token, pobj, parsed)
            argument = self.arguments._build_argument_from_parsed(
                argument_head,
                parsed,
                own_case_token=case_token_n(token, parsed)
                or case_token_n(conj_root_n(token, parsed), parsed),
            )
            # A flattened complement keeps the nested nominal's assertion.
            if argument.core is not None:
                local_status = self.arguments._token_assertion_status_n(
                    argument_head,
                    parsed,
                    inherit_parent_negation=False,
                )
                argument = dc_replace(
                    argument,
                    core=dc_replace(
                        argument.core,
                        assertion_status=local_status,
                        is_negated=local_status == "negated",
                    ),
                )
            return dc_replace(
                argument,
                role=role,
                group=self._nested_inner_group_provenance_n(token, parsed),
            )

        if by_tokens and non_by_tokens:
            return CandidateEvent(
                event_type=EventType.INTERACTION,
                event_id=next_id(),
                parser_source=self.parser_name,
                extraction_detail=f"DEP-{lemma}-nestedinner-directed",
                construction="nominalized",
                sentence_text=sentence_text,
                predicate=pred_span,
                arguments={
                    "sources": tuple(_nested_arg(t, f"{DEP_NMOD}:{PREP_BY}") for t in by_tokens),
                    "targets": tuple(
                        _nested_arg(t, f"{DEP_NMOD}:{_nested_prep(t)}")
                        for t in non_by_tokens
                    ),
                },
                custom_values=nested_scope_values,
                is_directed=True,
            )

        # Keep distinct roles ("colocalization of A with B and C" is A -> {B, C}).
        prep_buckets: dict[str, list[SyntaxNode]] = {}
        for protein_token in protein_tokens:
            prep_buckets.setdefault(_nested_prep(protein_token), []).append(
                protein_token
            )
        nonempty_preps = [prep for prep in prep_buckets if prep]
        has_compound_anchor_and_complement = bool(
            len(nonempty_preps) == 1
            and prep_buckets.get("")
            and all(
                token.dep == DEP_COMPOUND
                and parsed.head(token) is not None
                and parsed.head(token).i == pobj.i
                for token in prep_buckets[""]
            )
        )
        if len(nonempty_preps) >= 2 or has_compound_anchor_and_complement:
            if has_compound_anchor_and_complement:
                first_prep = ""
                source_tokens = prep_buckets[""]
                target_tokens = [
                    token
                    for prep in nonempty_preps
                    for token in prep_buckets[prep]
                ]
            else:
                first_prep = nonempty_preps[0]
                source_tokens = prep_buckets[first_prep]
                target_tokens = [
                    token
                    for prep in nonempty_preps[1:]
                    for token in prep_buckets[prep]
                ]
                target_tokens.extend(prep_buckets.get("", ()))
            if source_tokens and target_tokens:
                return CandidateEvent(
                    event_type=EventType.INTERACTION,

                    event_id=next_id(),
                    parser_source=self.parser_name,
                    extraction_detail=f"DEP-{lemma}-nestedinner-directed",
                    construction="nominalized",
                    sentence_text=sentence_text,
                    predicate=pred_span,
                    arguments={
                        "sources": tuple(
                            _nested_arg(t, f"{DEP_NMOD}:{first_prep}")
                            for t in source_tokens
                        ),
                        "targets": tuple(
                            _nested_arg(t, f"{DEP_NMOD}:{_nested_prep(t)}")
                            for t in target_tokens
                        ),
                    },
                    custom_values=nested_scope_values,
                    is_directed=True,
                )

        if chain_has_alternative and len(protein_tokens) >= 2:
            if not aa_anchor_indices:
                # ``A or B`` are alternatives: no source anchor between them.
                alternatives = tuple(
                    dc_replace(
                        self.arguments._build_argument_from_parsed(
                            t,
                            parsed,
                            own_case_token=case_token_n(t, parsed),
                        ),
                        role="alternative",
                        group=self._nested_inner_group_provenance_n(t, parsed),
                    )
                    for t in protein_tokens
                )
                return CandidateEvent(
                    event_type=EventType.INTERACTION,
                    event_id=next_id(),
                    parser_source=self.parser_name,
                    extraction_detail=f"DEP-{lemma}-nestedinner-alt-unanchored",
                    construction="nominalized",
                    sentence_text=sentence_text,
                    predicate=pred_span,
                    arguments={"sources": (), "targets": alternatives},
                    custom_values=nested_scope_values,
                    is_directed=True,
                )

            anchor_tok = next(
                (t for t in protein_tokens
                 if any(idx in aa_anchor_indices for idx in node_protein_indices(t))),
                protein_tokens[0],
            )
            rest_toks = [t for t in protein_tokens if t is not anchor_tok]
            sources_arg = (_nested_arg(anchor_tok, anchor_tok.dep or ""),)
            targets_args = tuple(
                _nested_arg(t, t.dep or "")
                for t in rest_toks
            )
            return CandidateEvent(
                event_type=EventType.INTERACTION,
                event_id=next_id(),
                parser_source=self.parser_name,
                extraction_detail=f"DEP-{lemma}-nestedinner-alt",
                construction="nominalized",
                sentence_text=sentence_text,
                predicate=pred_span,
                arguments={"sources": sources_arg, "targets": targets_args},
                custom_values=nested_scope_values,
                is_directed=True,
            )

        participants = tuple(
            _nested_arg(
                t,
                f"{DEP_NMOD}:{_nested_prep(t)}"
                if _nested_prep(t) else (t.dep or ""),
            )
            for t in protein_tokens
        )
        return CandidateEvent(
            event_type=EventType.INTERACTION,
            event_id=next_id(),
            parser_source=self.parser_name,
            extraction_detail=f"DEP-{lemma}-nestedinner",
            construction="nominalized",
            sentence_text=sentence_text,
            predicate=pred_span,
            arguments={"participants": participants},
            custom_values=nested_scope_values,
            is_directed=False,
        )


    def _nested_argument_head_n(
        self,
        protein_token: SyntaxNode,
        inner_event_head: SyntaxNode,
        parsed: ParsedSentence,
    ) -> SyntaxNode:
        """Walk from a nested-event protein to its lexical argument head within the NP."""
        current = protein_token
        # Do not cross ``conj`` upward.
        traversable = SAME_NP_TRAVERSAL_DEPS - {DEP_CONJ}
        while current.i != inner_event_head.i and current.dep in traversable:
            parent = parsed.head(current)
            if parent is None or parent.i == inner_event_head.i:
                break
            if parent.pos not in ("NOUN", "PROPN"):
                break
            current = parent
        return current


    def _collect_inner_proteins_n(self, head: SyntaxNode, parsed: ParsedSentence) -> list[SyntaxNode]:
        from ..node_helpers import node_protein_indices
        from ..roles import DEP_COMPOUND, DEP_NMOD, DEP_CONJ
        seen: set[int] = set()
        out: list[SyntaxNode] = []

        def visit(node: SyntaxNode, depth: int = 0) -> None:
            if depth > 3:
                return
            if node.i in seen:
                return
            seen.add(node.i)
            if node_protein_indices(node):
                out.append(node)
            for c in parsed.children(node):
                if c.dep in SAME_NP_TRAVERSAL_DEPS:
                    visit(c, depth + 1)

        for c in parsed.children(head):
            if c.dep in (DEP_COMPOUND, DEP_NMOD, DEP_CONJ):
                visit(c, 1)
        return out


    def _nested_inner_group_provenance_n(self, token: SyntaxNode, parsed: ParsedSentence) -> dict:
        from ..node_helpers import node_protein_indices
        from ..roles import DEP_CONJ
        anchor = token
        if token.dep in SAME_NP_PROTEIN_DEPS and token.head_i != -1:
            head = parsed.head(token)
            if head is not None:
                anchor = head
        elif token.dep == DEP_CONJ and token.head_i != -1:
            head = parsed.head(token)
            if head is not None:
                head_head = parsed.head(head)
                if head.dep in SAME_NP_PROTEIN_DEPS \
                        and head_head is not None \
                        and not node_protein_indices(head_head):
                    anchor = head_head
                else:
                    anchor = head
        if anchor.dep == DEP_CONJ and anchor.head_i != -1:
            head = parsed.head(anchor)
            if head is not None:
                anchor = head
        return self.arguments._argument_group_provenance_n(anchor, parsed)
