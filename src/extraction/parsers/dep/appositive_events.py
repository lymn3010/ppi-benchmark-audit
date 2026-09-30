"""Appositive channel: apposition, alias and identity (IS-A / HAS / PART-OF).

Examples: ``P0 (P1)``, ``P0 as P1``, ``P0, a kinase``.
"""
from __future__ import annotations

import re

from src.models.candidate import CandidateArgument, CandidateEvent, EventType
from src.system_rules import system_rule

from .roles import (
    DEP_APPOS,
    DEP_COP,
    DEP_CONJ,
    DEP_DEP,
    DEP_NMOD,
    DEP_NSUBJ,
    DEP_NSUBJPASS,
    PREP_AS,
    PREP_OF,
    RELATIVE_CLAUSE_DEPS,
)

_STRICT_ALIAS_PHRASE_RE = re.compile(
    system_rule("reference.strict_alias_phrase_pattern"),
    re.IGNORECASE,
)
_LOOSE_IDENTITY_WINDOW_RE = re.compile(
    system_rule("reference.loose_identity_window_pattern"),
    re.IGNORECASE,
)


from .channel import EventChannel


class AppositiveEventChannel(EventChannel):
    """Extract apposition, alias, and identity relational evidence."""

    def _extract_apposition_from_parsed(
        self, parsed, next_id, sentence_text: str,
    ) -> list[CandidateEvent]:
        from .node_helpers import node_protein_indices, single_token_span_n
        events: list[CandidateEvent] = []
        for token in parsed.nodes:
            head = parsed.head(token)
            if head is None:
                continue
            if token.dep == DEP_APPOS:
                detail = "DEP-Apposition"
            elif token.dep == DEP_DEP and self._is_uncertain_appositive_dep(token, head, parsed):
                detail = "DEP-Apposition"
            else:
                continue
            if not (node_protein_indices(head) or node_protein_indices(token)):
                continue
            left, right = (head, token) if head.char_start <= token.char_start else (token, head)
            if node_protein_indices(left) and self._is_loose_identity_candidate_window(
                parsed.text[left.char_end:right.char_start]
            ):
                continue

            head_arg = self.arguments._build_argument_from_parsed(head, parsed)
            appos_arg = self.arguments._build_argument_from_parsed(token, parsed)

            # Keep a relational role noun ("A, an activator of B") as the predicate.
            role_predicate = single_token_span_n(token) if not node_protein_indices(token) else None

            events.append(CandidateEvent(
                event_type=EventType.RELATION_STATEMENT,
                event_id=next_id(),
                parser_source=self.parser_name,
                extraction_detail=detail,
                sentence_text=sentence_text,
                predicate=role_predicate,
                arguments={
                    "entities_a": (head_arg,),
                    "entities_b": (appos_arg,),
                },
                is_directed=False,
                relation_subtype="IS-A",
            ))
        return events

    def _is_uncertain_appositive_dep(self, token, head, parsed) -> bool:
        """True when a generic ``dep`` edge links two adjacent, uncoordinated nominals."""
        from .node_helpers import conj_chain_n
        if token.pos not in ("NOUN", "PROPN") or head.pos not in ("NOUN", "PROPN"):
            return False
        if token.dep == DEP_CONJ or len(conj_chain_n(token, parsed)) > 1:
            return False
        if head.dep == DEP_CONJ or len(conj_chain_n(head, parsed)) > 1:
            return False
        left, right = (head, token) if head.char_start <= token.char_start else (token, head)
        window = parsed.text[left.char_end:right.char_start]
        if window.strip() == "":
            return True
        return self._is_loose_identity_candidate_window(window)

    def _extract_parenthetical_alias_from_parsed(
        self, parsed, next_id, sentence_text: str,
    ) -> list[CandidateEvent]:
        """``PROTEIN_A (PROTEIN_B)`` parenthetical alias -> IS-A relational event."""
        from .node_helpers import parenthetical_alias_antecedent_n
        events: list[CandidateEvent] = []
        seen: set[tuple[int, int]] = set()
        for token in parsed.nodes:
            antecedent = parenthetical_alias_antecedent_n(token, parsed)
            if antecedent is None:
                continue
            key = (antecedent.i, token.i)
            if key in seen:
                continue
            seen.add(key)

            ent_a = self.arguments._build_argument_from_parsed(antecedent, parsed)
            ent_b = self.arguments._build_argument_from_parsed(token, parsed)

            events.append(CandidateEvent(
                event_type=EventType.RELATION_STATEMENT,
                event_id=next_id(),
                parser_source=self.parser_name,
                extraction_detail="DEP-ParentheticalAlias",
                sentence_text=sentence_text,
                predicate=None,
                arguments={
                    "entities_a": (ent_a,),
                    "entities_b": (ent_b,),
                },
                is_directed=False,
                relation_subtype="IS-A",
            ))
        return events

    def _extract_alias_phrase_from_parsed(
        self, parsed, next_id, sentence_text: str,
    ) -> list[CandidateEvent]:
        """Strict alias phrases (``P0 (also known as P1)``) -> IS-A evidence."""
        events: list[CandidateEvent] = []
        protein_nodes = [n for n in parsed.nodes if getattr(n, "protein_indices", ())]
        seen: set[tuple[int, int]] = set()

        for alias in protein_nodes:
            antecedents = [n for n in protein_nodes if n.char_end <= alias.char_start]
            if not antecedents:
                continue
            antecedent = max(antecedents, key=lambda n: n.char_end)
            if set(antecedent.protein_indices) == set(alias.protein_indices):
                continue

            between = parsed.text[antecedent.char_end:alias.char_start]
            if not self._is_strict_alias_phrase_window(between):
                continue

            key = (antecedent.i, alias.i)
            if key in seen:
                continue
            seen.add(key)

            ent_a = self.arguments._build_argument_from_parsed(antecedent, parsed)
            ent_b = self.arguments._build_argument_from_parsed(alias, parsed)
            events.append(CandidateEvent(
                event_type=EventType.RELATION_STATEMENT,
                event_id=next_id(),
                parser_source=self.parser_name,
                extraction_detail="DEP-AliasPhrase",
                sentence_text=sentence_text,
                predicate=None,
                arguments={
                    "entities_a": (ent_a,),
                    "entities_b": (ent_b,),
                },
                is_directed=False,
                relation_subtype="IS-A",
            ))
        return events

    def _extract_loose_identity_candidate_from_parsed(
        self, parsed, next_id, sentence_text: str,
    ) -> list[CandidateEvent]:
        """Loose ``P0, a P1`` identity candidates (review only; not used for identity)."""
        events: list[CandidateEvent] = []
        protein_nodes = [n for n in parsed.nodes if getattr(n, "protein_indices", ())]
        seen: set[tuple[int, int]] = set()

        for candidate in protein_nodes:
            antecedents = [n for n in protein_nodes if n.char_end <= candidate.char_start]
            if not antecedents:
                continue
            antecedent = max(antecedents, key=lambda n: n.char_end)
            if set(antecedent.protein_indices) == set(candidate.protein_indices):
                continue

            between = parsed.text[antecedent.char_end:candidate.char_start]
            if not self._is_loose_identity_candidate_window(between):
                continue

            key = (antecedent.i, candidate.i)
            if key in seen:
                continue
            seen.add(key)

            ent_a = self.arguments._build_argument_from_parsed(antecedent, parsed)
            ent_b = self.arguments._build_argument_from_parsed(candidate, parsed)
            events.append(CandidateEvent(
                event_type=EventType.RELATION_STATEMENT,
                event_id=next_id(),
                parser_source=self.parser_name,
                extraction_detail="DEP-LooseIdentityCandidate",
                sentence_text=sentence_text,
                predicate=None,
                arguments={
                    "entities_a": (ent_a,),
                    "entities_b": (ent_b,),
                },
                is_directed=False,
                relation_subtype="IDENTITY",
                custom_values={
                    "review_mode": "loose_identity_candidate",
                    "runtime_alias_expansion": False,
                },
            ))
        return events

    @staticmethod
    def _is_strict_alias_phrase_window(text: str) -> bool:
        cleaned = (
            text.replace("(", " ")
            .replace(")", " ")
            .replace(",", " ")
            .replace(";", " ")
        )
        words = " ".join(cleaned.lower().split())
        return bool(_STRICT_ALIAS_PHRASE_RE.fullmatch(words))

    @staticmethod
    def _is_loose_identity_candidate_window(text: str) -> bool:
        if "(" in text or ")" in text or ";" in text or ":" in text:
            return False
        return bool(_LOOSE_IDENTITY_WINDOW_RE.fullmatch(text))

    def _extract_identity_from_parsed(
        self, parsed, next_id, sentence_text: str,
    ) -> list[CandidateEvent]:
        from .node_helpers import children_by_dep_n, case_lemma_n, node_protein_indices
        events: list[CandidateEvent] = []
        for token in parsed.nodes:
            for nmod in children_by_dep_n(token, parsed, DEP_NMOD):
                if case_lemma_n(nmod, parsed) != PREP_AS:
                    continue
                if not (node_protein_indices(token) or node_protein_indices(nmod)):
                    continue
                ent_a = self.arguments._build_argument_from_parsed(token, parsed)
                ent_b = self.arguments._build_argument_from_parsed(nmod, parsed)
                events.append(CandidateEvent(
                    event_type=EventType.RELATION_STATEMENT,
                    event_id=next_id(),
                    parser_source=self.parser_name,
                    extraction_detail="DEP-Identity",
                    sentence_text=sentence_text,
                    predicate=None,
                    arguments={
                        "entities_a": (ent_a,),
                        "entities_b": (ent_b,),
                    },
                    is_directed=False,
                    relation_subtype="IS-A",
                ))
        return events

    def _extract_copula_is_a_from_parsed(
        self, parsed, next_id, sentence_text: str,
    ) -> list[CandidateEvent]:
        from .node_helpers import node_protein_indices, single_token_span_n
        events: list[CandidateEvent] = []
        for head in parsed.nodes:
            if head.pos not in ("NOUN", "PROPN"):
                continue
            children_by_label: dict[str, list] = {}
            for c in parsed.children(head):
                children_by_label.setdefault(c.dep, []).append(c)
            if DEP_COP not in children_by_label:
                continue
            subj_heads = children_by_label.get(DEP_NSUBJ, []) + children_by_label.get(DEP_NSUBJPASS, [])
            if not subj_heads:
                continue

            if head.dep in RELATIVE_CLAUSE_DEPS:
                head_head = parsed.head(head)
                if head_head is not None and head_head.pos in ("NOUN", "PROPN"):
                    pron_only = (
                        len(subj_heads) == 1 and subj_heads[0].pos == "PRON"
                    ) or not subj_heads
                    if pron_only:
                        subj_heads = [head_head]

            entities_a: list[CandidateArgument] = []
            for s_head in subj_heads:
                for s in self.arguments._coordinate_members_n(s_head, parsed):
                    entities_a.append(self.arguments._build_argument_from_parsed(s, parsed))
            if not entities_a:
                continue

            entity_b = self.arguments._build_argument_from_parsed(head, parsed, force_possession=True)
            has_b_targets = bool(entity_b.core and entity_b.core.protein_indices) or \
                             any(o.protein_indices for o in entity_b.owners)
            if not has_b_targets:
                continue

            # Keep a role noun ("A is an activator of B") as predicate; protein heads stay None.
            role_predicate = single_token_span_n(head) if not node_protein_indices(head) else None

            events.append(CandidateEvent(
                event_type=EventType.RELATION_STATEMENT,
                event_id=next_id(),
                parser_source=self.parser_name,
                extraction_detail="DEP-CopulaIsA",
                sentence_text=sentence_text,
                predicate=role_predicate,
                arguments={
                    "entities_a": tuple(entities_a),
                    "entities_b": (entity_b,),
                },
                is_directed=False,
                relation_subtype="IS-A",
            ))
        return events

    def _extract_nominal_alias_genitive_from_parsed(
        self, parsed, next_id, sentence_text: str,
    ) -> list[CandidateEvent]:
        from .node_helpers import children_by_dep_n, case_lemma_n, node_protein_indices, match_keys_for_node, single_token_span_n
        events: list[CandidateEvent] = []
        for token in parsed.nodes:
            if token.pos not in ("NOUN", "PROPN"):
                continue
            alias_keys = match_keys_for_node(token)
            if not self.arguments._kw_match("relational_nominal_alias", alias_keys):
                continue
            for nmod in children_by_dep_n(token, parsed, DEP_NMOD):
                if case_lemma_n(nmod, parsed) != PREP_OF:
                    continue
                nmod_targets = node_protein_indices(nmod)
                head_targets = node_protein_indices(token)
                if not (nmod_targets or head_targets):
                    continue
                ent_a = self.arguments._build_argument_from_parsed(token, parsed)
                ent_b = self.arguments._build_argument_from_parsed(nmod, parsed, force_possession=True)
                # "A *homolog* of B": keep the relational noun as the predicate.
                role_predicate = single_token_span_n(token) if not node_protein_indices(token) else None
                events.append(CandidateEvent(
                    event_type=EventType.RELATION_STATEMENT,
                    event_id=next_id(),
                    parser_source=self.parser_name,
                    extraction_detail="DEP-NominalAlias",
                    sentence_text=sentence_text,
                    predicate=role_predicate,
                    arguments={
                        "entities_a": (ent_a,),
                        "entities_b": (ent_b,),
                    },
                    is_directed=False,
                    relation_subtype="IS-A",
                ))
        return events


AppositiveEventMixin = AppositiveEventChannel  # Compatibility import.
