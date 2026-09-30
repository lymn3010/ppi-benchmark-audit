"""Parser-neutral syntax and reduced semantic-tree artifacts."""
from __future__ import annotations

from typing import Any

from src.extraction.parsers.dep.node_helpers import (
    accompaniment_nmods_n,
    case_lemma_n,
    case_phrase_n,
    children_by_dep_n,
    node_protein_indices,
)
from src.extraction.parsers.dep.roles import DEP_APPOS
from src.extraction.parsers.dep.semantic_groups import (
    KIND_ACCOMPANIMENT,
    KIND_APPOSITION,
    argument_group_n,
    build_accompaniment_group_n,
    build_apposition_group_n,
)
from src.parsing.syntax import ParsedSentence, SyntaxNode


RAW_SYNTAX_TREE_FORMAT = "parser_neutral_syntax_tree_v2"
SEMANTIC_REDUCED_TREE_FORMAT = "parser_neutral_semantic_tree_v1"

_FUNCTION_DEPS = frozenset({
    "det", "punct", "aux", "auxpass", "cop", "mark", "case", "cc",
    "predet", "preconj", "expl", "meta",
})


def serialize_parsed_trees(
    parsed: ParsedSentence,
    *,
    candidate_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the common review trees emitted by every parser backend."""
    return {
        "raw_syntax_tree": serialize_raw_syntax_tree(parsed),
        "semantic_reduced_tree": serialize_semantic_tree(
            parsed,
            candidate_events=candidate_events,
        ),
    }


def serialize_raw_syntax_tree(parsed: ParsedSentence) -> dict[str, Any]:
    """Serialize normalized syntax plus the complete retained parser evidence."""
    return {
        "format": RAW_SYNTAX_TREE_FORMAT,
        "parser_backend": parsed.parser_name,
        "native_metadata": (parsed.raw_parser_data or {}).get("stanza_native", {}),
        "token_count": len(parsed),
        "roots": [node.i for node in parsed if node.is_root],
        "nodes": [
            {
                "id": node.i,
                "text": node.text,
                "lemma": node.lemma,
                "pos": node.pos,
                "dep": node.dep,
                "raw_pos": node.raw_pos,
                "raw_dep": node.raw_dep,
                "raw_deps": node.raw_deps,
                "raw_feats": node.raw_feats,
                "raw_misc": node.raw_misc,
                "sentence_index": node.sentence_index,
                "sentence_word_id": node.sentence_word_id,
                "sentence_head_id": node.sentence_head_id,
                "multiword_token_id": list(node.multiword_token_id),
                "multiword_token_text": node.multiword_token_text,
                "enhanced_heads": [
                    {"head": head_i, "raw_dep": dep}
                    for head_i, dep in node.enhanced_heads
                ],
                "head": node.head_i,
                "children": [child.i for child in parsed.children(node)],
                "offset": [node.char_start, node.char_end],
                "protein_indices": list(node.protein_indices),
                "direct_protein_indices": list(node.direct_protein_indices),
                "inherited_protein_indices": list(node.inherited_protein_indices),
                "is_nominalized": node.is_nominalized,
                "verb_form": node.verb_form,
                "nominalization_source": node.nominalization_source,
                "nominalization_confidence": node.nominalization_confidence,
            }
            for node in parsed
        ],
        "edges": [
            {
                "source": node.head_i,
                "target": node.i,
                "dep": node.dep,
                "raw_dep": node.raw_dep,
                "normalization_applied": node.dep != node.raw_dep,
                "case_marker": case_lemma_n(node, parsed),
                "case_phrase": case_phrase_n(node, parsed),
            }
            for node in parsed
            if not node.is_root
        ],
    }


def serialize_semantic_tree(
    parsed: ParsedSentence,
    *,
    candidate_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Collapse meaningful groups while preserving lexical tree structure."""
    assertion_by_offset = _assertions_by_offset(candidate_events or [])
    roots = [node for node in parsed if node.is_root]
    if not roots and parsed.nodes:
        roots = [parsed.nodes[0]]
    return {
        "format": SEMANTIC_REDUCED_TREE_FORMAT,
        "parser_backend": parsed.parser_name,
        "token_count": len(parsed),
        "roots": [node.i for node in roots],
        "nodes": [
            _reduce(
                node,
                parsed,
                role=node.dep or "ROOT",
                seen=set(),
                assertion_by_offset=assertion_by_offset,
            )
            for node in roots
        ],
    }


def _reduce(
    node: SyntaxNode,
    parsed: ParsedSentence,
    *,
    role: str,
    seen: set[int],
    assertion_by_offset: dict[tuple[int, int], str],
) -> dict[str, Any]:
    if node.i in seen:
        return _lexical_node(node, parsed, role, [], assertion_by_offset)
    seen.add(node.i)

    group = _meaningful_group(node, parsed)
    if group is not None:
        member_indices = {member.head_index for member in group.members}
        members = []
        for member in group.members:
            member_node = parsed.node(member.head_index)
            seen.add(member_node.i)
            members.append({
                "node_id": member_node.i,
                "kind": "member",
                "text": member_node.text,
                "role_protein_indices": list(member.role_protein_indices),
                "head_protein_indices": list(member.head_protein_indices),
                "modifier_protein_indices": list(member.modifier_protein_indices),
                "children": [
                    _reduce(
                        child,
                        parsed,
                        role=child.dep,
                        seen=seen,
                        assertion_by_offset=assertion_by_offset,
                    )
                    for child in parsed.children(member_node)
                    if _is_content_child(child) and child.i not in member_indices
                ],
            })
        return {
            "node_id": node.i,
            "kind": group.kind,
            "text": f"[{group.kind.upper()}]",
            "anchor_text": node.text,
            "role": role,
            "case_marker": case_lemma_n(node, parsed),
            "is_alternative": group.is_alternative,
            "coreferent": group.coreferent,
            "role_protein_indices": list(group.role_protein_indices),
            "all_protein_indices": list(group.all_protein_indices),
            "member_internal_pairs": [list(pair) for pair in group.member_internal_pairs],
            "ownership": [owner.to_dict() for owner in group.ownership],
            "members": members,
            "effective_children": [
                _reduce(
                    child,
                    parsed,
                    role=child.dep,
                    seen=seen,
                    assertion_by_offset=assertion_by_offset,
                )
                for child in parsed.children(node)
                if child.i not in member_indices
                and child.dep != "conj"
                and _is_content_child(child)
            ],
        }

    children = [
        _reduce(
            child,
            parsed,
            role=child.dep,
            seen=seen,
            assertion_by_offset=assertion_by_offset,
        )
        for child in parsed.children(node)
        if _is_content_child(child)
    ]
    return _lexical_node(node, parsed, role, children, assertion_by_offset)


def _meaningful_group(node: SyntaxNode, parsed: ParsedSentence):
    appos = children_by_dep_n(node, parsed, DEP_APPOS)
    if appos and node_protein_indices(node):
        group = build_apposition_group_n(node, parsed)
    elif accompaniment_nmods_n(node, parsed):
        group = build_accompaniment_group_n(node, parsed)
    else:
        group = argument_group_n(node, parsed)
    if (
        group.slot_count >= 2
        or group.member_internal_pairs
        or group.coreferent
        or group.kind in {KIND_APPOSITION, KIND_ACCOMPANIMENT}
    ):
        return group
    return None


def _lexical_node(
    node: SyntaxNode,
    parsed: ParsedSentence,
    role: str,
    children: list[dict[str, Any]],
    assertion_by_offset: dict[tuple[int, int], str],
) -> dict[str, Any]:
    return {
        "node_id": node.i,
        "kind": "lexical",
        "text": node.text,
        "lemma": node.lemma,
        "pos": node.pos,
        "dep": node.dep,
        "raw_dep": node.raw_dep,
        "raw_pos": node.raw_pos,
        "raw_feats": node.raw_feats,
        "enhanced_heads": [
            {"head": head_i, "raw_dep": dep}
            for head_i, dep in node.enhanced_heads
        ],
        "role": role,
        "targets": list(node.protein_indices),
        "case_marker": case_lemma_n(node, parsed),
        "assertion_status": assertion_by_offset.get((node.char_start, node.char_end), ""),
        "is_nominalized": node.is_nominalized,
        "verb_form": node.verb_form,
        "nominalization_source": node.nominalization_source,
        "nominalization_confidence": node.nominalization_confidence,
        "children": children,
    }


def _is_content_child(node: SyntaxNode) -> bool:
    return node.dep not in _FUNCTION_DEPS


def _assertions_by_offset(events: list[dict[str, Any]]) -> dict[tuple[int, int], str]:
    out: dict[tuple[int, int], str] = {}
    for event in events:
        for span in _iter_spans(event):
            status = span.get("assertion_status") or ""
            for offset in span.get("char_offsets") or []:
                if status and isinstance(offset, (list, tuple)) and len(offset) >= 2:
                    out[(int(offset[0]), int(offset[1]))] = status
    return out


def _iter_spans(value: Any):
    if isinstance(value, dict):
        if {"tokens", "char_offsets", "assertion_status"} <= set(value):
            yield value
        for child in value.values():
            yield from _iter_spans(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_spans(child)
