"""Parser-neutral observation models for mentions and proposed referent bindings."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.models.candidate import Span


@dataclass(frozen=True)
class MentionLink:
    id: str
    dependent: Span
    observed_relation: str
    case: str = ""
    interpretation_proposals: tuple[str, ...] = ()
    compatibility_kind: str = ""
    compatibility_index: int = -1
    owner_source: str = ""

    def to_dict(self) -> dict:
        result = {
            "id": self.id,
            "dependent": self.dependent.to_dict(),
            "observed": {"relation": self.observed_relation},
            "interpretation_proposals": list(self.interpretation_proposals),
        }
        if self.case:
            result["observed"]["case"] = self.case
        if self.compatibility_kind:
            result["compatibility"] = {
                "kind": self.compatibility_kind,
                "index": self.compatibility_index,
            }
        if self.owner_source:
            result["owner_source"] = self.owner_source
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "MentionLink":
        observed = data.get("observed", {})
        compatibility = data.get("compatibility", {})
        return cls(
            id=str(data["id"]),
            dependent=Span.from_dict(data["dependent"]),
            observed_relation=str(observed.get("relation", "")),
            case=str(observed.get("case", "")),
            interpretation_proposals=tuple(data.get("interpretation_proposals", [])),
            compatibility_kind=str(compatibility.get("kind", "")),
            compatibility_index=int(compatibility.get("index", -1)),
            owner_source=str(data.get("owner_source", "")),
        )
@dataclass(frozen=True)
class MentionGraphLink:
    """One parser-observed edge between sentence-level mention anchors."""

    id: str
    source_mention_ref: str
    target_mention_ref: str
    observed_relation: str
    raw_relation: str = ""
    case: str = ""

    def to_dict(self) -> dict:
        result = {
            "id": self.id,
            "source_mention_ref": self.source_mention_ref,
            "target_mention_ref": self.target_mention_ref,
            "observed_relation": self.observed_relation,
        }
        if self.raw_relation:
            result["raw_relation"] = self.raw_relation
        if self.case:
            result["case"] = self.case
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "MentionGraphLink":
        return cls(
            id=str(data["id"]),
            source_mention_ref=str(data["source_mention_ref"]),
            target_mention_ref=str(data["target_mention_ref"]),
            observed_relation=str(data.get("observed_relation", "")),
            raw_relation=str(data.get("raw_relation", "")),
            case=str(data.get("case", "")),
        )
@dataclass(frozen=True)
class Mention:
    id: str
    head: Span | None = None
    extent_spans: tuple[Span, ...] = ()
    direct_referents: tuple[int, ...] = ()
    inherited_referents: tuple[int, ...] = ()
    links: tuple[MentionLink, ...] = ()
    topology: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def observed_referents(self) -> tuple[int, ...]:
        values = set(self.direct_referents)
        values.update(self.inherited_referents)
        for link in self.links:
            values.update(int(i) for i in link.dependent.protein_indices)
        return tuple(sorted(values))

    def to_dict(self) -> dict:
        result = {
            "id": self.id,
            "head": self.head.to_dict() if self.head is not None else None,
            "extent_spans": [span.to_dict() for span in self.extent_spans],
            "direct_referents": list(self.direct_referents),
            "inherited_referents": list(self.inherited_referents),
            "observed_referents": list(self.observed_referents),
            "links": [link.to_dict() for link in self.links],
        }
        if self.topology:
            result["topology"] = dict(self.topology)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Mention":
        return cls(
            id=str(data["id"]),
            head=Span.from_dict(data["head"]) if data.get("head") else None,
            extent_spans=tuple(Span.from_dict(span) for span in data.get("extent_spans", [])),
            direct_referents=tuple(int(i) for i in data.get("direct_referents", [])),
            inherited_referents=tuple(int(i) for i in data.get("inherited_referents", [])),
            links=tuple(MentionLink.from_dict(link) for link in data.get("links", [])),
            topology=dict(data.get("topology", {})),
        )
@dataclass(frozen=True)
class MentionGraph:
    """Lossless parser-fact graph built before event extraction."""

    sentence_id: str
    parser_source: str
    mentions: tuple[Mention, ...] = ()
    links: tuple[MentionGraphLink, ...] = ()
    reference_clusters: tuple[dict, ...] = ()
    provenance: dict = field(default_factory=dict, compare=False, hash=False)

    def to_dict(self) -> dict:
        result = {
            "sentence_id": self.sentence_id,
            "parser_source": self.parser_source,
            "mentions": [mention.to_dict() for mention in self.mentions],
            "links": [link.to_dict() for link in self.links],
            "reference_clusters": [dict(cluster) for cluster in self.reference_clusters],
        }
        if self.provenance:
            result["provenance"] = dict(self.provenance)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "MentionGraph":
        return cls(
            sentence_id=str(data.get("sentence_id", "")),
            parser_source=str(data.get("parser_source", "")),
            mentions=tuple(Mention.from_dict(row) for row in data.get("mentions", [])),
            links=tuple(MentionGraphLink.from_dict(row) for row in data.get("links", [])),
            reference_clusters=tuple(
                dict(cluster) for cluster in data.get("reference_clusters", [])
            ),
            provenance=dict(data.get("provenance", {})),
        )


@dataclass(frozen=True)
class PairInferenceTrace:
    """Sentence-local trace for one bounded protein-pair evaluation."""

    id: str
    sentence_id: str
    pair: tuple[int, int]
    stage: str
    endpoint_visibility: str
    application_status: str
    gold_status: str
    emitted: bool
    proposal_refs: tuple[str, ...] = ()
    event_refs: tuple[str, ...] = ()
    projection_plan_refs: tuple[str, ...] = ()
    assertion_statuses: tuple[str, ...] = ()
    projection_modes: tuple[str, ...] = ()
    reason_code: str = ""
    provenance: dict = field(default_factory=dict, compare=False, hash=False)

    def to_dict(self) -> dict:
        result = {
            "id": self.id,
            "sentence_id": self.sentence_id,
            "pair": list(self.pair),
            "stage": self.stage,
            "endpoint_visibility": self.endpoint_visibility,
            "application_status": self.application_status,
            "gold_status": self.gold_status,
            "emitted": self.emitted,
            "proposal_refs": list(self.proposal_refs),
            "event_refs": list(self.event_refs),
            "projection_plan_refs": list(self.projection_plan_refs),
            "assertion_statuses": list(self.assertion_statuses),
            "projection_modes": list(self.projection_modes),
            "reason_code": self.reason_code or self.stage,
        }
        if self.provenance:
            result["provenance"] = dict(self.provenance)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "PairInferenceTrace":
        return cls(
            id=str(data["id"]),
            sentence_id=str(data.get("sentence_id", "")),
            pair=tuple(int(i) for i in data["pair"]),
            stage=str(data.get("stage", "")),
            endpoint_visibility=str(data.get("endpoint_visibility", "")),
            application_status=str(data.get("application_status", "")),
            gold_status=str(data.get("gold_status", "unlabeled")),
            emitted=bool(data.get("emitted", False)),
            proposal_refs=tuple(data.get("proposal_refs", [])),
            event_refs=tuple(data.get("event_refs", [])),
            projection_plan_refs=tuple(
                data.get("projection_plan_refs", data.get("application_refs", []))
            ),
            assertion_statuses=tuple(data.get("assertion_statuses", [])),
            projection_modes=tuple(data.get("projection_modes", [])),
            reason_code=str(data.get("reason_code", "")),
            provenance=dict(data.get("provenance", {})),
        )


@dataclass(frozen=True)
class ObservationArgument:
    id: str
    slot: str
    filler_mention_ref: str = ""
    event_ref: str = ""
    observed_role: str = ""
    normalized_role_proposals: tuple[str, ...] = ()
    topology_ref: str = ""
    assertion: str = "asserted"

    def to_dict(self) -> dict:
        filler = {}
        if self.filler_mention_ref:
            filler["mention_ref"] = self.filler_mention_ref
        if self.event_ref:
            filler["event_ref"] = self.event_ref
        return {
            "id": self.id,
            "slot": self.slot,
            "filler": filler,
            "observed_role": self.observed_role,
            "normalized_role_proposals": list(self.normalized_role_proposals),
            "topology_ref": self.topology_ref,
            "assertion": self.assertion,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ObservationArgument":
        filler = data.get("filler", {})
        return cls(
            id=str(data["id"]),
            slot=str(data.get("slot", "")),
            filler_mention_ref=str(filler.get("mention_ref", "")),
            event_ref=str(filler.get("event_ref", "")),
            observed_role=str(data.get("observed_role", "")),
            normalized_role_proposals=tuple(data.get("normalized_role_proposals", [])),
            topology_ref=str(data.get("topology_ref", "")),
            assertion=str(data.get("assertion", "asserted")),
        )


@dataclass(frozen=True)
class BindingProposal:
    id: str
    argument_ref: str
    referent: int
    operation: str
    interpretation: str
    evidence_links: tuple[str, ...] = ()
    status: str = "proposed"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "argument_ref": self.argument_ref,
            "referent": self.referent,
            "operation": self.operation,
            "interpretation": self.interpretation,
            "evidence_links": list(self.evidence_links),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BindingProposal":
        return cls(
            id=str(data["id"]),
            argument_ref=str(data["argument_ref"]),
            referent=int(data["referent"]),
            operation=str(data["operation"]),
            interpretation=str(data["interpretation"]),
            evidence_links=tuple(data.get("evidence_links", [])),
            status=str(data.get("status", "proposed")),
        )


@dataclass(frozen=True)
class EventObservation:
    id: str
    event_type: str
    parser_source: str
    extraction_detail: str
    construction: str
    assertion: str
    semantic_event_class: str = "predicate_event"
    realization_channel: str = "unknown"
    predicate_mention_ref: str = ""
    arguments: tuple[ObservationArgument, ...] = ()
    mentions: tuple[Mention, ...] = ()
    binding_proposals: tuple[BindingProposal, ...] = ()
    relation_subtype: str = ""
    provenance: dict = field(default_factory=dict, compare=False, hash=False)

    def mention_by_id(self, mention_id: str) -> Mention | None:
        return next((mention for mention in self.mentions if mention.id == mention_id), None)

    def argument_by_id(self, argument_id: str) -> ObservationArgument | None:
        return next((arg for arg in self.arguments if arg.id == argument_id), None)

    def to_dict(self) -> dict:
        result = {
            "id": self.id,
            "event_type": self.event_type,
            "semantic_event_class": self.semantic_event_class,
            "parser_source": self.parser_source,
            "extraction_detail": self.extraction_detail,
            "construction": self.construction,
            "realization_channel": self.realization_channel,
            "assertion": self.assertion,
            "predicate": {"mention_ref": self.predicate_mention_ref} if self.predicate_mention_ref else None,
            "arguments": [argument.to_dict() for argument in self.arguments],
            "mentions": [mention.to_dict() for mention in self.mentions],
            "binding_proposals": [proposal.to_dict() for proposal in self.binding_proposals],
            "relation_subtype": self.relation_subtype,
        }
        if self.provenance:
            result["provenance"] = dict(self.provenance)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "EventObservation":
        predicate = data.get("predicate") or {}
        return cls(
            id=str(data["id"]),
            event_type=str(data.get("event_type", "")),
            semantic_event_class=str(data.get("semantic_event_class", "predicate_event")),
            parser_source=str(data.get("parser_source", "")),
            extraction_detail=str(data.get("extraction_detail", "")),
            construction=str(data.get("construction", "")),
            assertion=str(data.get("assertion", "asserted")),
            realization_channel=str(data.get("realization_channel", "unknown")),
            predicate_mention_ref=str(predicate.get("mention_ref", "")),
            arguments=tuple(ObservationArgument.from_dict(arg) for arg in data.get("arguments", [])),
            mentions=tuple(Mention.from_dict(mention) for mention in data.get("mentions", [])),
            binding_proposals=tuple(
                BindingProposal.from_dict(proposal)
                for proposal in data.get("binding_proposals", [])
            ),
            relation_subtype=str(data.get("relation_subtype", "")),
            provenance=dict(data.get("provenance", {})),
        )
