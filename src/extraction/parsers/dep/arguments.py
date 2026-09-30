from __future__ import annotations
from src.parsing.syntax import ParsedSentence, SyntaxNode

from collections import deque
from dataclasses import dataclass, replace

from src.models.candidate import CandidateArgument, Span

from .morphology import looks_complex
from src.system_rules import rule_frozenset
from .semantic_groups import OwnerRef, SemanticGroup, argument_group_n
from .rules.assertion_scope import include_parent_in_negation_search
from .rules.compound_ownership import (
    TRANSITIVE_COMPOUND_OWNER_RULE_ID,
    transitive_compound_protein_members,
)
from .roles import (
    DEP_ADVCL,
    DEP_ACL,
    DEP_ACL_RELCL,
    DEP_APPOS,
    DEP_ATTR,
    DEP_CASE,
    DEP_CCOMP,
    DEP_COMPOUND,
    DEP_CONJ,
    DEP_DEP,
    DEP_DOBJ,
    DEP_NEG,
    DEP_NMOD,
    DEP_NSUBJ,
    DEP_NSUBJPASS,
    DEP_XCOMP,
    NEG_NOMINAL_MOD_DEPS,
    NEG_SCOPE_DEPS,
    SAME_NP_OR_LOOSE_DEPS,
    SAME_NP_TRAVERSAL_DEPS,
    PREP_OF,
)

# Subjects are themes of their head, not in scope of its negation.
_IDENTITY_DESIGNATION_PREDICATES = rule_frozenset(
    "reference.identity_designation_predicates"
)
_PARTICIPLE_XPOS = rule_frozenset("syntax.participle_xpos")


@dataclass(frozen=True)
class ArgumentBuild:
    """Argument output in two views from one build: ``candidate`` and ``group``."""
    candidate: CandidateArgument
    group: SemanticGroup


class ArgumentBuilder:
    """Build CandidateArgument objects from parser-neutral dependency heads."""

    def __init__(self, resources=None, context=None):
        self.resources = resources
        self._groups = context.groups if context is not None else None

    _NEG_CONTROL_INHERIT_DEPS = rule_frozenset("dependency.categories.neg_control_inherit")
    _NEG_SCOPE_DEPS = NEG_SCOPE_DEPS
    _NEG_NOMINAL_MOD_DEPS = NEG_NOMINAL_MOD_DEPS


    def build_argument_from_parsed(
        self, head: SyntaxNode, parsed: ParsedSentence, *, force_possession: bool = False,
        own_case_token: "SyntaxNode | None" = None,
        exclude_owner_node_indices: frozenset[int] = frozenset(),
    ) -> ArgumentBuild:
        reference_build = self._build_reference_argument_from_parsed(
            head, parsed, own_case_token=own_case_token,
        )
        if reference_build is not None:
            return reference_build
        membership = self._argument_group_n(head, parsed)
        candidate, ownership = self._construct_candidate_from_parsed(
            head, parsed, force_possession=force_possession,
            own_case_token=own_case_token,
            exclude_owner_node_indices=exclude_owner_node_indices,
        )
        group = replace(membership, ownership=tuple(ownership))
        candidate = replace(candidate, group=group.to_dict())
        return ArgumentBuild(candidate=candidate, group=group)

    def _build_reference_argument_from_parsed(
        self,
        head: SyntaxNode,
        parsed: ParsedSentence,
        *,
        own_case_token: "SyntaxNode | None" = None,
    ) -> "ArgumentBuild | None":
        """Build a safe identity pronoun's argument from its unique lexical antecedent."""
        antecedent_i = head.inherited_reference_head_i
        if antecedent_i < 0 or antecedent_i == head.i:
            return None
        if antecedent_i >= len(parsed.nodes):
            return None
        antecedent = parsed.node(antecedent_i)
        # Only a direct, singular, identity-safe antecedent.
        membership = self._argument_group_n(antecedent, parsed)
        candidate, ownership = self._construct_candidate_from_parsed(antecedent, parsed)

        reference_kinds = set(head.inherited_reference_kinds)
        if candidate.core is not None:
            reference_kinds.update(candidate.core.reference_kinds)
            candidate = replace(
                candidate,
                core=replace(
                    candidate.core,
                    reference_kinds=tuple(sorted(reference_kinds)),
                ),
            )

        from .node_helpers import case_token_n, single_token_span_n
        current_case = own_case_token or case_token_n(head, parsed)
        if current_case is not None:
            candidate = replace(
                candidate,
                case_markers=(single_token_span_n(current_case),),
            )
        candidate = replace(candidate, role=head.dep or candidate.role)
        group = replace(membership, ownership=tuple(ownership))
        candidate = replace(candidate, group=group.to_dict())
        return ArgumentBuild(candidate=candidate, group=group)

    def _build_argument_from_parsed(
        self, head: SyntaxNode, parsed: ParsedSentence, *, force_possession: bool = False,
        own_case_token: "SyntaxNode | None" = None,
        exclude_owner_node_indices: frozenset[int] = frozenset(),
    ) -> CandidateArgument:
        return self.build_argument_from_parsed(
            head, parsed, force_possession=force_possession,
            own_case_token=own_case_token,
            exclude_owner_node_indices=exclude_owner_node_indices,
        ).candidate

    def _construct_candidate_from_parsed(
        self, head: SyntaxNode, parsed: ParsedSentence, *, force_possession: bool = False,
        own_case_token: "SyntaxNode | None" = None,
        exclude_owner_node_indices: frozenset[int] = frozenset(),
    ) -> "tuple[CandidateArgument, list[OwnerRef]]":
        from .node_helpers import (
            children_by_dep_n, node_protein_indices, conj_chain_n,
            case_token_n, case_lemma_n, is_accompaniment_nmod_n,
            descriptor_tokens_n, single_token_span_n, compound_span_n,
            hyphen_adjacent_protein_children_n, conj_negation_excluded_n,
        )
        original_role = head.dep or ""
        head = self._promote_quantified_group_head_n(head, parsed)

        compound_owners: list[Span] = []
        compound_text_only: list[SyntaxNode] = []
        ownership: list[OwnerRef] = []
        resources = getattr(self, "resources", None)
        transitive_compound_enabled = (
            resources is None
            or resources.rule_enabled(TRANSITIVE_COMPOUND_OWNER_RULE_ID)
        )
        if self._absorbs_conj_head_as_complex_owner_n(head, parsed):
            head_head = parsed.head(head)
            if head_head is not None:
                compound_owners.append(single_token_span_n(
                    head_head,
                    assertion_status=self._token_assertion_status_n(
                        head_head, parsed, include_negation=False,
                    ),
                ))
                ownership.append(OwnerRef(
                    head_head.i, node_protein_indices(head_head), "complex_conj"))
        for c in children_by_dep_n(head, parsed, DEP_COMPOUND):
            compound_members = conj_chain_n(c, parsed)
            target_members = [m for m in compound_members if node_protein_indices(m)]
            if target_members:
                for member in target_members:
                    if member.i in exclude_owner_node_indices:
                        continue
                    compound_owners.append(single_token_span_n(
                        member,
                        assertion_status=self._token_assertion_status_n(
                            member, parsed, include_negation=False,
                        ),
                    ))
                    ownership.append(OwnerRef(
                        member.i, node_protein_indices(member), "compound"))
                    if transitive_compound_enabled:
                        for nested_member in transitive_compound_protein_members(
                            member, parsed,
                        ):
                            if nested_member.i in exclude_owner_node_indices:
                                continue
                            compound_owners.append(single_token_span_n(
                                nested_member,
                                assertion_status=self._token_assertion_status_n(
                                    nested_member, parsed, include_negation=False,
                                ),
                            ))
                            ownership.append(OwnerRef(
                                nested_member.i,
                                node_protein_indices(nested_member),
                                "compound_transitive",
                            ))
            else:
                found_deeper = False
                # A process-nominal compound ("P0 binding domain") is not an owner of the head.
                if c.is_nominalized:
                    compound_text_only.append(c)
                else:
                    for grandchild in parsed.children(c):
                        if grandchild.dep not in (DEP_COMPOUND, DEP_APPOS):
                            continue
                        if not node_protein_indices(grandchild):
                            continue
                        compound_owners.append(single_token_span_n(
                            grandchild,
                            assertion_status=self._token_assertion_status_n(
                                grandchild, parsed, include_negation=False,
                            ),
                        ))
                        ownership.append(OwnerRef(
                            grandchild.i, node_protein_indices(grandchild), "compound"))
                        found_deeper = True
                    if not found_deeper:
                        compound_text_only.append(c)
        head_is_target = bool(node_protein_indices(head))
        for c in children_by_dep_n(head, parsed, DEP_APPOS):
            if head_is_target:
                continue
            # Expand the conj chain of each appositive to capture whole protein lists.
            for member in conj_chain_n(c, parsed):
                if not node_protein_indices(member):
                    continue
                compound_owners.append(single_token_span_n(
                    member,
                    assertion_status=self._token_assertion_status_n(member, parsed, include_negation=False),
                ))
                ownership.append(OwnerRef(member.i, node_protein_indices(member), "apposition"))

        if head.pos in ("NOUN", "PROPN") and not head_is_target:
            for acl in (
                children_by_dep_n(head, parsed, DEP_ACL)
                + children_by_dep_n(head, parsed, DEP_ACL_RELCL)
            ):
                if (acl.lemma or acl.text or "").lower() not in _IDENTITY_DESIGNATION_PREDICATES:
                    continue
                for dep in (DEP_XCOMP, DEP_DOBJ, DEP_ATTR):
                    for complement in children_by_dep_n(acl, parsed, dep):
                        for member in conj_chain_n(complement, parsed):
                            if not node_protein_indices(member):
                                continue
                            compound_owners.append(single_token_span_n(
                                member,
                                assertion_status=self._token_assertion_status_n(
                                    member, parsed, include_negation=False,
                                ),
                            ))
                            ownership.append(OwnerRef(
                                member.i,
                                node_protein_indices(member),
                                "identity_designation",
                            ))

        # Stanza may attach the first appositive member with dep; follow its conj chain.
        if head.pos in ("NOUN", "PROPN") and not head_is_target:
            for c in children_by_dep_n(head, parsed, DEP_DEP):
                for member in conj_chain_n(c, parsed):
                    if not node_protein_indices(member):
                        continue
                    compound_owners.append(single_token_span_n(
                        member,
                        assertion_status=self._token_assertion_status_n(
                            member, parsed, include_negation=False,
                        ),
                    ))
                    ownership.append(OwnerRef(
                        member.i, node_protein_indices(member), "loose_apposition"))

        nmod_owners: list[Span] = []
        case_markers: list[Span] = []
        is_process_nom = (not force_possession) and head.is_nominalized

        for nmod in children_by_dep_n(head, parsed, DEP_NMOD):
            if is_accompaniment_nmod_n(nmod, parsed):
                continue
            _ct = case_token_n(nmod, parsed)
            nmod_source = "nmod:" + ((_ct.lemma or _ct.text or "").lower() if _ct is not None else "_")
            if (
                nmod_source == f"nmod:{PREP_OF}"
                and not force_possession
                and self._is_relational_nominal_alias_head_n(head)
            ):
                continue
            for conj in conj_chain_n(nmod, parsed):
                # Walk every nmod NP, including protein-headed ones.
                _bfs_seen: set[int] = set()
                _bfs_queue: deque = deque([conj])
                while _bfs_queue:
                    member = _bfs_queue.popleft()
                    if member.i in _bfs_seen:
                        continue
                    _bfs_seen.add(member.i)
                    # Skip conjuncts excluded by "but not".
                    if member.dep == DEP_CONJ and conj_negation_excluded_n(member, parsed):
                        continue
                    if node_protein_indices(member):
                        nmod_owners.append(single_token_span_n(
                            member,
                            assertion_status=self._token_assertion_status_n(
                                member, parsed, include_negation=False,
                            ),
                        ))
                        ownership.append(OwnerRef(
                            member.i, node_protein_indices(member), nmod_source))
                    for child in parsed.children(member):
                        if child.dep not in SAME_NP_TRAVERSAL_DEPS:
                            continue
                        _bfs_queue.extend(conj_chain_n(child, parsed))
            case_tok = case_token_n(nmod, parsed)
            if case_tok is not None and not is_process_nom:
                case_markers.append(single_token_span_n(case_tok))

        if own_case_token is not None:
            case_markers.append(single_token_span_n(own_case_token))

        descriptor_nodes = descriptor_tokens_n(head, parsed)
        descriptors: list[Span] = [
            single_token_span_n(
                amod,
                assertion_status=self._token_assertion_status_n(amod, parsed, include_negation=False),
            )
            for amod in descriptor_nodes
        ]

        # A split marker in "P0-tagged P1" needs explicit owner evidence.
        hyphen_owners: list[Span] = []
        for amod in descriptor_nodes:
            # Participial modifiers are hyphen-participle events, not owners.
            if amod.pos == "VERB" and (amod.raw_pos or "") in _PARTICIPLE_XPOS:
                continue
            for marker in hyphen_adjacent_protein_children_n(amod, parsed):
                hyphen_owners.append(single_token_span_n(
                    marker,
                    assertion_status=self._token_assertion_status_n(
                        marker, parsed, include_negation=False,
                    ),
                ))
                ownership.append(OwnerRef(
                    marker.i, node_protein_indices(marker), "hyphen_modifier"))

        if compound_text_only:
            core_span = compound_span_n(head, compound_text_only)
        else:
            core_status = self._token_assertion_status_n(head, parsed)
            core_span = single_token_span_n(
                head,
                is_negated=core_status == "negated",
                assertion_status=core_status,
            )

        owners = self._dedupe_owner_spans(
            tuple(compound_owners + nmod_owners + hyphen_owners)
        )
        ownership = self._dedupe_owner_refs(ownership)

        candidate = CandidateArgument(
            core=core_span,
            owners=owners,
            descriptors=tuple(descriptors),
            case_markers=tuple(case_markers),
            event_ref="",
            role=original_role,
            group={},
        )
        return candidate, ownership

    def _is_relational_nominal_alias_head_n(self, head: SyntaxNode) -> bool:
        from .node_helpers import match_keys_for_node

        try:
            return bool(self._kw_match("relational_nominal_alias", match_keys_for_node(head)))
        except AttributeError:
            return False

    @staticmethod
    def _dedupe_owner_spans(spans: tuple[Span, ...]) -> tuple[Span, ...]:
        """Remove exact duplicate owner spans, keeping traversal order."""
        seen: set[tuple] = set()
        out: list[Span] = []
        for span in spans:
            key = (
                tuple(span.tokens),
                tuple(span.protein_indices),
                tuple(span.char_offsets),
                span.dep,
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(span)
        return tuple(out)

    @staticmethod
    def _dedupe_owner_refs(refs: list[OwnerRef]) -> list[OwnerRef]:
        seen: set[tuple] = set()
        out: list[OwnerRef] = []
        for ref in refs:
            key = (ref.head_index, tuple(ref.protein_indices), ref.source)
            if key in seen:
                continue
            seen.add(key)
            out.append(ref)
        return out

    def _absorbs_conj_head_as_complex_owner_n(self, head: SyntaxNode, parsed: ParsedSentence) -> bool:
        from .node_helpers import node_protein_indices
        if head.dep != DEP_CONJ or head.head_i == -1 or head.head_i == head.i:
            return False
        head_head = parsed.head(head)
        if head_head is None or not node_protein_indices(head_head):
            return False
        lemma = (head.lemma or head.text or "").lower()
        if not looks_complex(lemma):
            return False
        return any(
            node_protein_indices(c)
            for c in parsed.children(head)
            if c.dep in SAME_NP_OR_LOOSE_DEPS
        )

    def _group_for_n(self, head: SyntaxNode, parsed: ParsedSentence):
        groups = getattr(self, "_groups", None)
        if groups is not None:
            return groups.group_for(head)
        return argument_group_n(head, parsed)

    def _coordinate_members_n(self, head: SyntaxNode, parsed: ParsedSentence):
        groups = getattr(self, "_groups", None)
        if groups is not None:
            return groups.conj_members(head)
        from .node_helpers import conj_chain_n
        return conj_chain_n(head, parsed)

    def _argument_group_n(self, head: SyntaxNode, parsed: ParsedSentence) -> SemanticGroup:
        from .node_helpers import children_by_dep_n, is_accompaniment_nmod_n
        core_group = self._group_for_n(head, parsed)
        if core_group.role_protein_indices:
            return core_group

        for nmod in children_by_dep_n(head, parsed, DEP_NMOD):
            if is_accompaniment_nmod_n(nmod, parsed):
                continue
            owner_group = self._group_for_n(nmod, parsed)
            if owner_group.role_protein_indices:
                return owner_group

        return core_group

    def _argument_group_provenance_n(self, head: SyntaxNode, parsed: ParsedSentence) -> dict:
        return self._argument_group_n(head, parsed).to_dict()

    @staticmethod
    def _promote_quantified_group_head_n(head: SyntaxNode, parsed: ParsedSentence) -> SyntaxNode:
        from .node_helpers import children_by_dep_n
        lemma = (head.lemma or head.text or "").lower()
        if lemma not in {"all", "both", "each"}:
            return head
        for child in parsed.children(head):
            if child.dep == DEP_NMOD:
                return child
        return head

    def _token_assertion_status_n(
        self,
        token: SyntaxNode,
        parsed: ParsedSentence,
        *,
        include_negation: bool = True,
        inherit_parent_negation: bool = True,
    ) -> str:
        from .node_helpers import neg_flag_n
        if include_negation and (
            neg_flag_n(token, parsed)
            or self._token_has_negation_cue_n(
                token,
                parsed,
                inherit_parent_negation=inherit_parent_negation,
            )
        ):
            return "negated"
        if self._token_has_uncertainty_cue_n(token, parsed):
            return "uncertain"
        return "asserted"

    def _token_has_negation_cue_n(
        self,
        token: SyntaxNode,
        parsed: ParsedSentence,
        *,
        inherit_parent_negation: bool = True,
    ) -> bool:
        from .node_helpers import (
            conj_negation_excluded_n, correlative_negation_head_n,
            conj_contrast_resets_head_negation_n, neg_flag_n,
        )
        if conj_negation_excluded_n(token, parsed):
            return True
        if correlative_negation_head_n(token, parsed):
            return True
        head = parsed.head(token)
        if token.dep == DEP_CONJ and head is not None and correlative_negation_head_n(head, parsed):
            return True

        if any(child.dep == DEP_NEG for child in parsed.children(token)):
            return True

        if head is not None and token.dep in (
            DEP_NMOD, "acl", "acl:relcl", "relcl", DEP_XCOMP, DEP_CCOMP, DEP_ADVCL,
        ):
            head_key = (head.lemma or head.text or "").lower()
            # Only a noun/adjective head passes negation to an nmod.
            if token.dep == DEP_NMOD and (head.pos or "").upper() in ("VERB", "AUX"):
                pass
            elif head_key and self._kw_match("negation", {head_key}):
                return True

        # A subject is not in the scope of its head verb's negation.
        parent_scope_allowed = bool(
            inherit_parent_negation
            and head is not None
            and include_parent_in_negation_search(
                dependency=token.dep,
                parent_pos=head.pos or "",
                has_explicit_subject=any(
                    child.dep in {DEP_NSUBJ, DEP_NSUBJPASS, "nsubj:pass"}
                    for child in parsed.children(token)
                ),
            )
        )
        search_nodes = (
            (token, head)
            if parent_scope_allowed
            and not conj_contrast_resets_head_negation_n(token, parsed)
            else (token,)
        )
        for node in search_nodes:
            for child in parsed.children(node):
                if child.dep not in self._NEG_SCOPE_DEPS:
                    continue
                key = (child.lemma or child.text or "").lower()
                if self._kw_match("assertion_uncertainty", {key}):
                    continue
                if self._kw_match("negation", {key}):
                    return True

        for child in parsed.children(token):
            if child.dep not in self._NEG_NOMINAL_MOD_DEPS:
                continue
            key = (child.lemma or child.text or "").lower()
            if self._kw_match("assertion_uncertainty", {key}):
                continue
            if self._kw_match("negation", {key}):
                return True

        for child in parsed.children(token):
            if child.dep != DEP_CASE:
                continue
            key = (child.lemma or child.text or "").lower()
            if self._kw_match("assertion_uncertainty", {key}):
                continue
            if self._kw_match("negation", {key}):
                return True

        if (
            inherit_parent_negation
            and head is not None
            and token.dep in self._NEG_CONTROL_INHERIT_DEPS
        ):
            if token.dep == DEP_ADVCL and (head.pos or "") != "ADJ":
                pass
            else:
                head_key = (head.lemma or head.text or "").lower()
                if head_key and self._kw_match("negation", {head_key}):
                    return True

        for child in parsed.children(token):
            if child.dep not in (DEP_DOBJ, DEP_ATTR):
                continue
            for grand in parsed.children(child):
                if grand.dep not in self._NEG_NOMINAL_MOD_DEPS:
                    continue
                key = (grand.lemma or grand.text or "").lower()
                if self._kw_match("assertion_uncertainty", {key}):
                    continue
                if self._kw_match("negation", {key}):
                    return True

        return False

    # Shared lexical categories; reviewed catalogs are never consulted.

    def _kw_match(self, category: str, keys) -> bool:
        if self.resources is None:
            return False
        return self.resources.match_lexicon(category, set(keys))

    def _matches_group_pattern(self, keys) -> bool:
        return self._kw_match("reciprocal_event_nominal", keys)

    def _matches_collective_pattern(self, keys) -> bool:
        return self._kw_match("collective_state_nominal", keys)

    def _token_has_uncertainty_cue_n(self, token: SyntaxNode, parsed: ParsedSentence) -> bool:
        # Modal auxiliaries are routine hedging, not uncertainty.
        # Relative clauses keep their own assertion status.
        _RELCL_SCOPE_BARRIER = "acl:relcl"
        uncertainty_scope_deps = NEG_SCOPE_DEPS - {"aux", "auxpass"}
        current = token
        depth = 0
        while depth < 4:
            for child in parsed.children(current):
                if child.dep not in uncertainty_scope_deps:
                    continue
                key = (child.lemma or child.text or "").lower()
                if self._kw_match("assertion_uncertainty", {key}):
                    return True
            # Stop at relative-clause boundary; its assertion scope is independent.
            if current.dep == _RELCL_SCOPE_BARRIER:
                break
            head = parsed.head(current)
            if head is None:
                break
            current = head
            depth += 1
        return False


ArgumentBuilderMixin = ArgumentBuilder  # Compatibility import.
