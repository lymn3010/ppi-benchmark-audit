"""Build the parser-fact MentionGraph before event extraction."""
from __future__ import annotations

import re

from src.models.candidate import Span
from src.parsing.syntax import ParsedSentence, SyntaxNode

from .models import Mention, MentionGraph, MentionGraphLink


_PROTEIN_MARKER = re.compile(r"PROTEIN\d+", re.IGNORECASE)


def parser_mention_id(parsed: ParsedSentence, node: SyntaxNode) -> str:
    """Return the stable graph id for one parser node."""
    sentence_id = parsed.sentence_id or "sentence"
    return f"{sentence_id}:node:{node.i}:{node.char_start}-{node.char_end}"


def parser_mention_ref_for_span(parsed: ParsedSentence, span: Span | None) -> str:
    """Resolve a compatibility Span back to its first parser-fact anchor."""
    refs = parser_mention_refs_for_span(parsed, span)
    return refs[0] if refs else ""


def parser_mention_refs_for_span(
    parsed: ParsedSentence,
    span: Span | None,
) -> tuple[str, ...]:
    """Resolve every token offset in a compatibility Span to graph anchors."""
    if span is None or not span.char_offsets:
        return ()
    offsets = set(span.char_offsets)
    return tuple(
        parser_mention_id(parsed, node)
        for node in parsed.nodes
        if (node.char_start, node.char_end) in offsets
    )


def _span_for_node(node: SyntaxNode) -> Span:
    keys = {node.lemma.lower(), node.text.lower()}
    if node.verb_form:
        keys.add(node.verb_form.lower())
    return Span(
        tokens=(node.text,),
        lemmas=(node.lemma.lower(),),
        match_keys=tuple(sorted(key for key in keys if key)),
        protein_indices=tuple(sorted(set(node.protein_indices))),
        char_offsets=((node.char_start, node.char_end),),
        token_lemmas=(node.lemma.lower(),),
        head_node_index=node.i,
        pos=node.pos,
        dep=node.dep,
        is_nominalized=node.is_nominalized,
        verb_form=node.verb_form,
        node_indices=(node.i,),
        parser_dep=node.raw_dep or node.dep,
        parser_pos=node.raw_pos or node.pos,
        parser_features=node.raw_feats,
        parser_misc=node.raw_misc,
        enhanced_heads=node.enhanced_heads,
        reference_kinds=node.inherited_reference_kinds,
    )


def _mention_kind(node: SyntaxNode) -> str:
    if _PROTEIN_MARKER.fullmatch(node.text):
        return "direct_protein_marker"
    if node.protein_indices:
        return "reference_bearing_anchor"
    return "lexical_anchor"


def _case_phrase(node: SyntaxNode, parsed: ParsedSentence) -> str:
    markers = sorted(
        (
            child
            for child in parsed.children(node)
            if child.dep == "case"
        ),
        key=lambda child: child.i,
    )
    return " ".join((child.lemma or child.text).lower() for child in markers)


def _reference_clusters(parsed: ParsedSentence) -> tuple[dict, ...]:
    rows: list[dict] = []
    for cluster in parsed.coref_clusters:
        mentions = []
        for start, end in cluster.mentions:
            node_refs = [
                parser_mention_id(parsed, node)
                for node in parsed.nodes
                if node.char_start < end and node.char_end > start
            ]
            mentions.append({
                "char_start": int(start),
                "char_end": int(end),
                "node_refs": node_refs,
            })
        rows.append({
            "cluster_id": int(cluster.cluster_id),
            "evidence_type": cluster.evidence_type,
            "head_text": cluster.head_text,
            "mentions": mentions,
        })
    return tuple(rows)


def build_mention_graph(parsed: ParsedSentence) -> MentionGraph:
    """Serialize parser nodes and edges without selecting semantic bindings."""
    mentions: list[Mention] = []
    for node in parsed.nodes:
        span = _span_for_node(node)
        is_direct_marker = bool(_PROTEIN_MARKER.fullmatch(node.text))
        direct_referents = tuple(sorted(set(node.direct_protein_indices)))
        inherited_referents = tuple(sorted(set(node.inherited_protein_indices)))
        # Fallback for fixtures and old parser facts without identity provenance.
        if not direct_referents and not inherited_referents and node.protein_indices:
            if is_direct_marker:
                direct_referents = tuple(sorted(set(node.protein_indices)))
            else:
                inherited_referents = tuple(sorted(set(node.protein_indices)))
        topology = {
            "node_index": node.i,
            "mention_kind": _mention_kind(node),
            "raw_pos": node.raw_pos,
            "inherited_reference_kinds": list(node.inherited_reference_kinds),
            "inherited_reference_head_i": node.inherited_reference_head_i,
            "inherited_reference_node_indices": list(node.inherited_reference_node_indices),
        }
        mentions.append(Mention(
            id=parser_mention_id(parsed, node),
            head=span,
            extent_spans=(span,),
            direct_referents=direct_referents,
            inherited_referents=inherited_referents,
            topology=topology,
        ))

    links: list[MentionGraphLink] = []
    for node in parsed.nodes:
        head = parsed.head(node)
        if head is not None:
            links.append(MentionGraphLink(
                id=f"{parsed.sentence_id or 'sentence'}:dep:{head.i}->{node.i}",
                source_mention_ref=parser_mention_id(parsed, head),
                target_mention_ref=parser_mention_id(parsed, node),
                observed_relation=node.dep,
                raw_relation=node.raw_dep,
                case=_case_phrase(node, parsed),
            ))
        if 0 <= node.inherited_reference_head_i < len(parsed.nodes):
            antecedent = parsed.node(node.inherited_reference_head_i)
            kind = (
                node.inherited_reference_kinds[0]
                if node.inherited_reference_kinds
                else "reference"
            )
            links.append(MentionGraphLink(
                id=f"{parsed.sentence_id or 'sentence'}:reference:{node.i}->{antecedent.i}",
                source_mention_ref=parser_mention_id(parsed, node),
                target_mention_ref=parser_mention_id(parsed, antecedent),
                observed_relation=f"reference:{kind}",
                raw_relation="coreference_cluster",
            ))

    return MentionGraph(
        sentence_id=parsed.sentence_id,
        parser_source=parsed.parser_name,
        mentions=tuple(mentions),
        links=tuple(links),
        reference_clusters=_reference_clusters(parsed),
        provenance={
            "schema": "parser_fact_mention_graph_v1",
            "interpretation": "selected_identity_references_only",
            "node_count": len(mentions),
            "edge_count": len(links),
        },
    )
