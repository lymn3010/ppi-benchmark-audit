"""Pure CandidateEvent -> EventObservation projection."""
from __future__ import annotations

from src.models.candidate import CandidateArgument, CandidateEvent, Span

from .models import (
    BindingProposal,
    EventObservation,
    Mention,
    MentionLink,
    ObservationArgument,
)


_OWNER_INTERPRETATIONS = {
    "compound": "compound_modifier_represents_head",
    "complex_conj": "coordinate_member_represents_argument",
    "apposition": "apposition_identity",
    "hyphen_modifier": "hyphen_modifier_represents_argument",
}


def _owner_interpretation(source: str) -> str:
    if source.startswith("nmod:"):
        prep = source.split(":", 1)[1] or "_"
        return f"nmod_argument_with_case_{prep}"
    return _OWNER_INTERPRETATIONS.get(source, "linked_referent_represents_argument")


def _owner_relation(source: str, owner: Span) -> tuple[str, str]:
    if source.startswith("nmod:"):
        return owner.dep or "nmod", source.split(":", 1)[1]
    return owner.dep or source, ""


def _owner_sources(arg: CandidateArgument) -> list[str]:
    ownership = arg.group.get("ownership", []) if arg.group else []
    sources = [
        str(row.get("source", ""))
        for row in ownership
        if isinstance(row, dict)
    ]
    return [
        sources[index] if index < len(sources) and sources[index] else _fallback_owner_source(owner)
        for index, owner in enumerate(arg.owners)
    ]


def _fallback_owner_source(owner: Span) -> str:
    del owner
    return "untyped_owner_without_provenance"


def _mention_for_argument(
    event_id: str,
    argument_id: str,
    arg: CandidateArgument,
    parsed=None,
) -> tuple[Mention, list[BindingProposal]]:
    mention_id = f"{argument_id}:mention"
    links: list[MentionLink] = []
    proposals: list[BindingProposal] = []

    for referent in sorted(set(arg.core.protein_indices if arg.core else ())):
        proposals.append(BindingProposal(
            id=f"{argument_id}:binding:direct:{referent}",
            argument_ref=argument_id,
            referent=int(referent),
            operation="bind_direct_referent",
            interpretation="direct_identity",
        ))

    owner_sources = _owner_sources(arg)
    for index, owner in enumerate(arg.owners):
        source = owner_sources[index]
        relation, case = _owner_relation(source, owner)
        link_id = f"{mention_id}:owner:{index}"
        interpretation = _owner_interpretation(source)
        links.append(MentionLink(
            id=link_id,
            dependent=owner,
            observed_relation=relation,
            case=case,
            interpretation_proposals=(interpretation,),
            compatibility_kind="owner",
            compatibility_index=index,
            owner_source=source,
        ))
        for referent in sorted(set(owner.protein_indices)):
            proposals.append(BindingProposal(
                id=f"{argument_id}:binding:owner:{index}:{referent}",
                argument_ref=argument_id,
                referent=int(referent),
                operation="bind_linked_referent",
                interpretation=interpretation,
                evidence_links=(link_id,),
            ))

    for index, descriptor in enumerate(arg.descriptors):
        link_id = f"{mention_id}:descriptor:{index}"
        links.append(MentionLink(
            id=link_id,
            dependent=descriptor,
            observed_relation=descriptor.dep or "descriptor",
            interpretation_proposals=("descriptor_modifier",),
            compatibility_kind="descriptor",
            compatibility_index=index,
        ))
        for referent in sorted(set(descriptor.protein_indices)):
            proposals.append(BindingProposal(
                id=f"{argument_id}:binding:descriptor:{index}:{referent}",
                argument_ref=argument_id,
                referent=int(referent),
                operation="bind_linked_referent",
                interpretation="descriptor_referent_represents_argument",
                evidence_links=(link_id,),
            ))

    for index, marker in enumerate(arg.case_markers):
        links.append(MentionLink(
            id=f"{mention_id}:case:{index}",
            dependent=marker,
            observed_relation="case",
            case=(marker.lemmas[0] if marker.lemmas else marker.text.lower()),
            compatibility_kind="case_marker",
            compatibility_index=index,
        ))

    core_spans = (arg.core,) if arg.core is not None else ()
    extent = tuple(core_spans + arg.owners + arg.descriptors)
    topology = dict(arg.group)
    if parsed is not None:
        from .mention_graph import parser_mention_refs_for_span

        topology["parser_mention_refs"] = [
            ref
            for span in (arg.core,) + arg.owners + arg.descriptors + arg.case_markers
            for ref in parser_mention_refs_for_span(parsed, span)
        ]
    return Mention(
        id=mention_id,
        head=arg.core,
        extent_spans=extent,
        direct_referents=tuple(sorted(set(arg.core.protein_indices if arg.core else ()))),
        links=tuple(links),
        topology=topology,
    ), proposals


def project_observation(event: CandidateEvent, parsed=None) -> EventObservation:
    """Project one already-built CandidateEvent and optionally link parser mentions."""

    mentions: list[Mention] = []
    arguments: list[ObservationArgument] = []
    proposals: list[BindingProposal] = []

    predicate_ref = ""
    if event.predicate is not None:
        topology = {}
        if parsed is not None:
            from .mention_graph import parser_mention_refs_for_span

            refs = parser_mention_refs_for_span(parsed, event.predicate)
            if refs:
                topology["parser_mention_refs"] = list(refs)
        predicate_ref = f"{event.event_id}:predicate"
        mentions.append(Mention(
            id=predicate_ref,
            head=event.predicate,
            extent_spans=(event.predicate,),
            direct_referents=tuple(sorted(set(event.predicate.protein_indices))),
            topology=topology,
        ))

    for slot, slot_args in event.arguments.items():
        for index, arg in enumerate(slot_args):
            argument_id = f"{event.event_id}:{slot}:{index}"
            mention_ref = ""
            if not arg.is_event_reference:
                mention, argument_proposals = _mention_for_argument(
                    event.event_id, argument_id, arg, parsed,
                )
                mentions.append(mention)
                proposals.extend(argument_proposals)
                mention_ref = mention.id
            arguments.append(ObservationArgument(
                id=argument_id,
                slot=slot,
                filler_mention_ref=mention_ref,
                event_ref=arg.event_ref,
                observed_role=arg.role,
                topology_ref=f"{mention_ref}:topology" if mention_ref and arg.group else "",
                assertion=arg.assertion_status,
            ))

    return EventObservation(
        id=event.event_id,
        event_type=event.event_type.name,
        semantic_event_class=event.semantic_event_class,
        parser_source=event.parser_source,
        extraction_detail=event.extraction_detail,
        construction=event.construction,
        realization_channel=event.realization_channel,
        assertion=event.predicate.assertion_status if event.predicate else "asserted",
        predicate_mention_ref=predicate_ref,
        arguments=tuple(arguments),
        mentions=tuple(mentions),
        binding_proposals=tuple(proposals),
        relation_subtype=event.relation_subtype,
        provenance={
            "projection": "candidate_event_compatibility_v1",
            "parser_mentions_linked": parsed is not None,
        },
    )


def derive_owners(observation: EventObservation, argument_id: str) -> tuple[Span, ...]:
    """Reconstruct the CandidateArgument.owners compatibility view."""
    argument = observation.argument_by_id(argument_id)
    if argument is None or not argument.filler_mention_ref:
        return ()
    mention = observation.mention_by_id(argument.filler_mention_ref)
    if mention is None:
        return ()
    links = sorted(
        (link for link in mention.links if link.compatibility_kind == "owner"),
        key=lambda link: link.compatibility_index,
    )
    return tuple(link.dependent for link in links)


def derive_descriptors(observation: EventObservation, argument_id: str) -> tuple[Span, ...]:
    """Reconstruct the CandidateArgument.descriptors compatibility view."""
    argument = observation.argument_by_id(argument_id)
    if argument is None or not argument.filler_mention_ref:
        return ()
    mention = observation.mention_by_id(argument.filler_mention_ref)
    if mention is None:
        return ()
    links = sorted(
        (link for link in mention.links if link.compatibility_kind == "descriptor"),
        key=lambda link: link.compatibility_index,
    )
    return tuple(link.dependent for link in links)


def derive_referent_links(
    observation: EventObservation,
    argument_id: str,
) -> tuple[MentionLink, ...]:
    """Return all protein-bearing mention links for an argument."""
    argument = observation.argument_by_id(argument_id)
    if argument is None or not argument.filler_mention_ref:
        return ()
    mention = observation.mention_by_id(argument.filler_mention_ref)
    if mention is None:
        return ()
    links = [
        link
        for link in mention.links
        if link.compatibility_kind in {"owner", "descriptor"}
    ]
    return tuple(sorted(
        links,
        key=lambda link: (link.compatibility_kind, link.compatibility_index),
    ))


def derive_owner_provenance(
    observation: EventObservation,
    argument_id: str,
) -> tuple[tuple[tuple[int, ...], str], ...]:
    """Reconstruct source-equivalent OwnerRef provenance."""
    argument = observation.argument_by_id(argument_id)
    if argument is None or not argument.filler_mention_ref:
        return ()
    mention = observation.mention_by_id(argument.filler_mention_ref)
    if mention is None:
        return ()
    links = sorted(
        (link for link in mention.links if link.compatibility_kind == "owner"),
        key=lambda link: link.compatibility_index,
    )
    return tuple((link.dependent.protein_indices, link.owner_source) for link in links)


def derive_all_protein_indices(
    observation: EventObservation,
    argument_id: str,
) -> tuple[int, ...]:
    """Reconstruct CandidateArgument.all_protein_indices visibility."""
    argument = observation.argument_by_id(argument_id)
    if argument is None or not argument.filler_mention_ref:
        return ()
    mention = observation.mention_by_id(argument.filler_mention_ref)
    if mention is None:
        return ()
    values = set(mention.direct_referents)
    for link in derive_referent_links(observation, argument_id):
        values.update(int(i) for i in link.dependent.protein_indices)
    return tuple(sorted(values))
