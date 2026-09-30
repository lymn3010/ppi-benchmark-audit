"""Pre-event reference propagation over the parser-neutral syntax graph."""
from __future__ import annotations

from dataclasses import replace

from src.downstream.reference_rules import (
    IS_A_NOMINAL_REFERENCE_KIND,
    IS_A_NOMINAL_REFERENCE_RULE_ID,
    IS_A_SUBJECT_DEPS,
    has_nominal_reference_determiner,
    is_entity_role_nominal,
    is_nominal_declaration_head,
    is_nominal_reference_candidate,
    is_transparent_owner_head,
    nominal_reference_key,
)
from src.extraction.rules import RuleApplication
from src.parsing import ParsedSentence, SyntaxNode
from src.system_rules import rule_frozenset, system_rule


PRONOUN_NUMBER_AGREEMENT_RULE_ID = "reference.pronoun_number_agreement@1"


class ParsedReferenceResolver:
    """Apply typed reference operations before event extraction.

    Only pronouns and explicit demonstrative sets inherit protein indices.
    """

    def __init__(self, resources=None) -> None:
        self.resources = resources

    def resolve(self, parsed: ParsedSentence) -> ParsedSentence:
        parsed = self._pronoun_coreference(parsed)
        parsed = self._is_a_nominal_references(parsed)
        return self._demonstrative_groups(parsed)

    @staticmethod
    def _pronoun_coreference(parsed: ParsedSentence) -> ParsedSentence:
        if not parsed.coref_clusters:
            return parsed
        overrides: dict[int, tuple[int, ...]] = {}
        reference_mentions: dict[int, tuple[int, tuple[int, ...]]] = {}
        rule_trace: list[dict] = []
        explicit_groups = {
            group for _end, group in ParsedReferenceResolver._explicit_coordination_groups(parsed)
        }
        for cluster in parsed.coref_clusters:
            if cluster.evidence_type != "pronoun":
                continue
            antecedent_pids: list[int] = []
            antecedent_spans: list[tuple[SyntaxNode, ...]] = []
            pronouns: list[int] = []
            for start, end in cluster.mentions:
                nodes = tuple(
                    node for node in parsed.nodes
                    if node.char_start >= start and node.char_end <= end
                )
                if not nodes:
                    continue
                mention_pids = [pid for node in nodes for pid in node.protein_indices]
                if mention_pids:
                    antecedent_spans.append(nodes)
                    antecedent_pids.extend(
                        pid for pid in mention_pids if pid not in antecedent_pids
                    )
                    continue
                for node in nodes:
                    identity_pronoun = node.pos == "PRON"
                    standalone_determiner = node.pos == "DET" and len(nodes) == 1
                    if (identity_pronoun or standalone_determiner) and not node.protein_indices:
                        pronouns.append(node.i)
            if not antecedent_pids or not pronouns:
                continue
            for node_i in pronouns:
                pronoun = parsed.node(node_i)
                normalized_pids = tuple(sorted(antecedent_pids))
                allows_multiple = ParsedReferenceResolver._allows_multiple_antecedents(
                    pronoun,
                    normalized_pids,
                    explicit_groups,
                )
                application = RuleApplication(
                    rule_id=PRONOUN_NUMBER_AGREEMENT_RULE_ID,
                    stage="reference_resolution",
                    decision="applied" if allows_multiple else "blocked",
                    reason=(
                        "pronoun_number_compatible_with_antecedent_cardinality"
                        if allows_multiple
                        else "singular_or_clausal_pronoun_cannot_inherit_multiple_proteins"
                    ),
                    evidence={
                        "cluster_id": cluster.cluster_id,
                        "pronoun_node_i": node_i,
                        "pronoun": (pronoun.lemma or pronoun.text or "").lower(),
                        "antecedent_protein_indices": list(normalized_pids),
                    },
                )
                rule_trace.append(application.to_dict())
                if not allows_multiple:
                    continue
                overrides[node_i] = tuple(sorted(antecedent_pids))
                if len(antecedent_spans) == 1 and len(antecedent_pids) == 1:
                    mention_nodes = antecedent_spans[0]
                    mention_ids = {node.i for node in mention_nodes}
                    heads = [
                        node for node in mention_nodes
                        if node.head_i not in mention_ids or node.head_i == node.i
                    ]
                    if len(heads) == 1:
                        reference_mentions[node_i] = (
                            heads[0].i,
                            tuple(node.i for node in mention_nodes),
                        )
        raw_parser_data = dict(parsed.raw_parser_data)
        if rule_trace:
            raw_parser_data["reference_rule_trace"] = [
                *(raw_parser_data.get("reference_rule_trace") or ()),
                *rule_trace,
            ]
        if not overrides:
            return replace(parsed, raw_parser_data=raw_parser_data) if rule_trace else parsed
        nodes = tuple(
            replace(
                node,
                protein_indices=overrides[node.i],
                inherited_protein_indices=overrides[node.i],
                inherited_reference_kinds=("pronoun_coreference",),
                inherited_reference_head_i=reference_mentions.get(node.i, (-1, ()))[0],
                inherited_reference_node_indices=reference_mentions.get(node.i, (-1, ()))[1],
            )
            if node.i in overrides and not node.protein_indices else node
            for node in parsed.nodes
        )
        return replace(parsed, nodes=nodes, raw_parser_data=raw_parser_data)

    @staticmethod
    def _allows_multiple_antecedents(
        pronoun: SyntaxNode,
        antecedent_pids: tuple[int, ...],
        explicit_groups: set[tuple[int, ...]],
    ) -> bool:
        if len(antecedent_pids) <= 1:
            return True
        lemma = (pronoun.lemma or pronoun.text or "").lower()
        if lemma in rule_frozenset("reference.plural_coreference_pronouns"):
            return True
        # Relative ``which`` inherits only an explicit coordinated NP, not a whole clause.
        return lemma == "which" and antecedent_pids in explicit_groups

    def _demonstrative_groups(self, parsed: ParsedSentence) -> ParsedSentence:
        groups = self._explicit_coordination_groups(parsed)
        if not groups:
            return parsed
        overrides: dict[int, tuple[int, ...]] = {}
        for node in parsed.nodes:
            if node.protein_indices or node.pos not in {"NOUN", "PROPN"}:
                continue
            count = self._demonstrative_count(parsed, node)
            if count is None:
                continue
            candidates = [
                group for group in groups
                if group[0] < node.i and len(group[1]) == count
            ]
            if candidates:
                overrides[node.i] = max(candidates, key=lambda group: group[0])[1]
        if not overrides:
            return parsed
        nodes = tuple(
            replace(
                node,
                protein_indices=overrides[node.i],
                inherited_protein_indices=overrides[node.i],
                inherited_reference_kinds=("demonstrative_group_reference",),
            )
            if node.i in overrides else node
            for node in parsed.nodes
        )
        return replace(parsed, nodes=nodes)

    def _is_a_nominal_references(self, parsed: ParsedSentence) -> ParsedSentence:
        if self.resources and not self.resources.rule_enabled(IS_A_NOMINAL_REFERENCE_RULE_ID):
            return parsed
        declarations = ParsedReferenceResolver._is_a_nominal_declarations(parsed)
        if not declarations:
            return parsed

        overrides: dict[int, tuple[int, ...]] = {}
        rule_trace: list[dict] = []
        by_key: dict[str, list[dict]] = {}
        for declaration in declarations:
            by_key.setdefault(str(declaration["key"]), []).append(declaration)

        for node in parsed.nodes:
            if not is_nominal_reference_candidate(parsed, node):
                continue
            key = nominal_reference_key(node)
            candidates = [
                declaration for declaration in by_key.get(key, ())
                if int(declaration["nominal_node_i"]) < node.i
            ]
            if not candidates:
                continue
            unique_pid_sets = {
                tuple(declaration["protein_indices"])
                for declaration in candidates
            }
            allows_cardinality = all(
                ParsedReferenceResolver._nominal_reference_allows_cardinality(
                    parsed, node, pids,
                )
                for pids in unique_pid_sets
            )
            applied = len(unique_pid_sets) == 1 and allows_cardinality
            reason = "unique_prior_is_a_nominal_antecedent" if applied else (
                "reference_marker_cardinality_incompatible_with_antecedent"
                if not allows_cardinality
                else "ambiguous_prior_is_a_nominal_antecedents"
            )
            trace_declarations = [
                {
                    "source": declaration["source"],
                    "nominal_node_i": declaration["nominal_node_i"],
                    "antecedent_node_indices": list(declaration["antecedent_node_indices"]),
                    "protein_indices": list(declaration["protein_indices"]),
                }
                for declaration in candidates
            ]
            application = RuleApplication(
                rule_id=IS_A_NOMINAL_REFERENCE_RULE_ID,
                stage="reference_resolution",
                decision="applied" if applied else "blocked",
                reason=reason,
                evidence={
                    "reference_node_i": node.i,
                    "reference_lemma": key,
                    "candidate_declarations": trace_declarations,
                },
            )
            rule_trace.append(application.to_dict())
            if applied:
                overrides[node.i] = next(iter(unique_pid_sets))

        raw_parser_data = dict(parsed.raw_parser_data)
        if rule_trace:
            raw_parser_data["reference_rule_trace"] = [
                *(raw_parser_data.get("reference_rule_trace") or ()),
                *rule_trace,
            ]
        if not overrides:
            return replace(parsed, raw_parser_data=raw_parser_data) if rule_trace else parsed
        nodes = tuple(
            replace(
                node,
                protein_indices=overrides[node.i],
                inherited_protein_indices=overrides[node.i],
                inherited_reference_kinds=(IS_A_NOMINAL_REFERENCE_KIND,),
            )
            if node.i in overrides else node
            for node in parsed.nodes
        )
        return replace(parsed, nodes=nodes, raw_parser_data=raw_parser_data)

    @staticmethod
    def _is_a_nominal_declarations(parsed: ParsedSentence) -> list[dict]:
        declarations: list[dict] = []
        for head in parsed.nodes:
            if (
                is_nominal_declaration_head(head)
                and not head.protein_indices
                and any(child.dep == "cop" for child in parsed.children(head))
            ):
                subjects = [
                    child for child in parsed.children(head)
                    if child.dep in IS_A_SUBJECT_DEPS
                ]
                for subject in subjects:
                    pids, antecedent_nodes = (
                        ParsedReferenceResolver._coordination_protein_indices(
                            parsed, subject,
                        )
                    )
                    if pids:
                        declarations.append({
                            "source": "copula_is_a",
                            "key": nominal_reference_key(head),
                            "nominal_node_i": head.i,
                            "antecedent_node_indices": antecedent_nodes,
                            "protein_indices": pids,
                        })

            if head.dep != "appos" or head.head_i < 0:
                continue
            parent = parsed.node(head.head_i)
            declarations.extend(
                ParsedReferenceResolver._appositive_nominal_declarations(
                    parsed, parent, head,
                )
            )
        return declarations

    @staticmethod
    def _appositive_nominal_declarations(
        parsed: ParsedSentence,
        first: SyntaxNode,
        second: SyntaxNode,
    ) -> list[dict]:
        first_pids, first_nodes = ParsedReferenceResolver._coordination_protein_indices(
            parsed, first,
        )
        second_pids, second_nodes = ParsedReferenceResolver._coordination_protein_indices(
            parsed, second,
        )
        declarations: list[dict] = []
        if first_pids and not second_pids and is_nominal_declaration_head(second):
            declarations.append({
                "source": "appositive_is_a",
                "key": nominal_reference_key(second),
                "nominal_node_i": second.i,
                "antecedent_node_indices": first_nodes,
                "protein_indices": first_pids,
            })
        if second_pids and not first_pids and is_nominal_declaration_head(first):
            declarations.append({
                "source": "appositive_is_a",
                "key": nominal_reference_key(first),
                "nominal_node_i": first.i,
                "antecedent_node_indices": second_nodes,
                "protein_indices": second_pids,
            })
        if (
            not first_pids
            and not second_pids
            and is_transparent_owner_head(first)
            and is_entity_role_nominal(second)
            and has_nominal_reference_determiner(parsed, second)
        ):
            owner_pids, owner_nodes = ParsedReferenceResolver._of_owner_protein_indices(
                parsed, first,
            )
            if owner_pids:
                declarations.append({
                    "source": "transparent_appositive_owner_is_a",
                    "key": nominal_reference_key(second),
                    "nominal_node_i": second.i,
                    "antecedent_node_indices": owner_nodes,
                    "protein_indices": owner_pids,
                })
        return declarations

    @staticmethod
    def _coordination_protein_indices(
        parsed: ParsedSentence,
        node: SyntaxNode,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        members = {node.i}
        if node.dep == "conj" and node.head_i >= 0:
            members.add(node.head_i)
            parent = parsed.node(node.head_i)
            members.update(
                child.i for child in parsed.children(parent)
                if child.dep == "conj"
            )
        members.update(
            child.i for child in parsed.children(node)
            if child.dep == "conj"
        )
        selected = [parsed.node(index) for index in sorted(members)]
        pids = tuple(sorted({
            pid for member in selected
            for pid in (member.direct_protein_indices or member.protein_indices)
        }))
        antecedent_nodes = tuple(
            member.i for member in selected
            if member.direct_protein_indices or member.protein_indices
        )
        return pids, antecedent_nodes

    @staticmethod
    def _of_owner_protein_indices(
        parsed: ParsedSentence,
        head: SyntaxNode,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        pids: set[int] = set()
        antecedent_nodes: set[int] = set()
        for child in parsed.children(head):
            if not child.dep.startswith("nmod"):
                continue
            has_of_case = any(
                case_child.dep == "case"
                and (case_child.lemma or case_child.text or "").strip().lower() == "of"
                for case_child in parsed.children(child)
            )
            if not has_of_case:
                continue
            child_pids, child_nodes = (
                ParsedReferenceResolver._coordination_protein_indices(parsed, child)
            )
            pids.update(child_pids)
            antecedent_nodes.update(child_nodes)
        return tuple(sorted(pids)), tuple(sorted(antecedent_nodes))

    @staticmethod
    def _nominal_reference_allows_cardinality(
        parsed: ParsedSentence,
        node: SyntaxNode,
        protein_indices: tuple[int, ...],
    ) -> bool:
        if len(protein_indices) <= 1:
            return True
        determiner_lemmas = {
            (child.lemma or child.text or "").strip().lower()
            for child in parsed.children(node)
            if child.dep == "det"
        }
        if determiner_lemmas & {"these", "those"}:
            return True
        text = (node.text or "").strip().lower()
        lemma = (node.lemma or "").strip().lower()
        return bool(lemma and text.endswith("s") and text != lemma)

    @staticmethod
    def _explicit_coordination_groups(
        parsed: ParsedSentence,
    ) -> list[tuple[int, tuple[int, ...]]]:
        groups: list[tuple[int, tuple[int, ...]]] = []
        for head in parsed.nodes:
            head_pids = head.direct_protein_indices or head.protein_indices
            if not head_pids:
                continue
            members = [head, *[
                child for child in parsed.children(head)
                if child.dep == "conj"
                and (child.direct_protein_indices or child.protein_indices)
            ]]
            if len(members) < 2:
                continue
            pids = tuple(sorted({
                pid for member in members
                for pid in (member.direct_protein_indices or member.protein_indices)
            }))
            if len(pids) == len(members):
                groups.append((max(member.i for member in members), pids))
        return groups

    @staticmethod
    def _demonstrative_count(parsed: ParsedSentence, node: SyntaxNode) -> int | None:
        determiners = rule_frozenset("reference.demonstrative_group_determiners")
        if not any(
            child.dep == "det" and child.lemma.lower() in determiners
            for child in parsed.children(node)
        ):
            return None
        numbers: list[int] = []
        words = system_rule("reference.cardinal_words")
        for child in parsed.children(node):
            if child.dep != "nummod" and child.pos != "NUM":
                continue
            value = (child.lemma or child.text or "").strip().lower()
            cardinal = int(value) if value.isdigit() else words.get(value)
            if cardinal is not None:
                numbers.append(int(cardinal))
        return numbers[0] if len(numbers) == 1 else None


__all__ = [
    "IS_A_NOMINAL_REFERENCE_RULE_ID",
    "PRONOUN_NUMBER_AGREEMENT_RULE_ID",
    "ParsedReferenceResolver",
]
