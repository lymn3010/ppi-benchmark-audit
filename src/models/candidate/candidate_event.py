"""Parser-independent events, arguments and groups."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# Event type

class EventType(Enum):
    INTERACTION = auto()        # "A verb B" + nested predicates that hold event refs
    RELATION_STATEMENT = auto() # "A is B" / "A is B's X"
    INTERACTION_STATE = auto()  # "A+B complex"

    @classmethod
    def from_name(cls, name: str) -> "EventType":
        # Artifact migration aliases.
        aliases = {
            "RELATIONAL_ASSERTION": "RELATION_STATEMENT",
        }
        normalized = aliases.get(str(name).strip().upper(), str(name).strip().upper())
        return cls[normalized]


class SemanticEventClass(str, Enum):
    """The two semantic event classes reported by the v8 methodology."""

    PREDICATE_EVENT = "predicate_event"
    RELATION_STATEMENT = "relation_statement"


class RealizationChannel(str, Enum):
    """Surface realization used to discover an event."""

    CLAUSE = "clause"
    EVENT_NOMINAL = "event_nominal"
    STATE_NOMINAL = "state_nominal"
    APPOSITIVE = "appositive"
    LEXICAL_PATH = "lexical_path"
    UNKNOWN = "unknown"

    @classmethod
    def normalize(cls, value: str | "RealizationChannel" | None) -> str:
        text = value.value if isinstance(value, cls) else str(value or "").strip().lower()
        return text if text in {member.value for member in cls} else cls.UNKNOWN.value


class AssertionStatus(str, Enum):
    """Assertion state for an extracted predicate or argument span."""

    ASSERTED = "asserted"
    NEGATED = "negated"
    UNCERTAIN = "uncertain"
    HYPOTHETICAL = "hypothetical"
    QUESTION = "question"
    PURPOSE = "purpose"

    @classmethod
    def normalize(cls, value: str | "AssertionStatus" | None, *, is_negated: bool = False) -> str:
        if is_negated:
            return cls.NEGATED.value
        text = (value.value if isinstance(value, cls) else value or "").strip().lower()
        if not text:
            return cls.ASSERTED.value
        known = {status.value for status in cls}
        return text if text in known else cls.UNCERTAIN.value



@dataclass(frozen=True)
class Span:
    """Parser-independent surface span with lemmas, forms and char offsets."""
    tokens: tuple[str, ...]                              # surface forms, in order
    lemmas: tuple[str, ...]                              # lowercased lemmas (sorted, unique)
    match_keys: tuple[str, ...]                          # lemmas + verb forms + surfaces
    protein_indices: tuple[int, ...]                     # PROTEIN target indices, e.g. (0, 1)
    char_offsets: tuple[tuple[int, int], ...]            # per-token (start, end)
    pos: str = ""                                        # POS of head token
    dep: str = ""                                        # dep label of head token (for adapter)
    is_negated: bool = False
    assertion_status: str = AssertionStatus.ASSERTED.value
    is_nominalized: bool = False
    verb_form: str = ""                                  # verbal lemma when nominalized
    # Original parser analysis, kept for audit.
    node_indices: tuple[int, ...] = ()
    parser_dep: str = ""
    parser_pos: str = ""
    parser_features: str = ""
    parser_misc: str = ""
    enhanced_heads: tuple[tuple[int, str], ...] = ()
    reference_kinds: tuple[str, ...] = ()
    token_lemmas: tuple[str, ...] = ()                   # one lemma per surface token, in token order
    head_node_index: int = -1                            # parser-neutral node id of the syntactic head

    def __post_init__(self) -> None:
        if self.token_lemmas and len(self.token_lemmas) != len(self.tokens):
            raise ValueError("Span.token_lemmas must align one-to-one with Span.tokens")
        if not self.token_lemmas:
            # Old JSON: fall back to lowercased surface for multiword spans.
            if len(self.tokens) == 1 and self.lemmas:
                ordered_lemmas = (self.lemmas[0],)
            else:
                ordered_lemmas = tuple(token.lower() for token in self.tokens)
            object.__setattr__(self, "token_lemmas", ordered_lemmas)
        object.__setattr__(
            self,
            "assertion_status",
            AssertionStatus.normalize(self.assertion_status, is_negated=self.is_negated),
        )

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    @property
    def head_offset(self) -> tuple[int, int]:
        """Character offset of the syntactic head, not merely the first token."""
        index = self.head_token_index
        return self.char_offsets[index] if index < len(self.char_offsets) else (-1, -1)

    @property
    def head_token_index(self) -> int:
        """Position of ``head_node_index`` in the surface-ordered token arrays."""
        if self.head_node_index >= 0 and self.head_node_index in self.node_indices:
            return self.node_indices.index(self.head_node_index)
        return 0

    def to_dict(self) -> dict:
        return {
            "tokens": list(self.tokens),
            "lemmas": list(self.lemmas),
            "match_keys": list(self.match_keys),
            "protein_indices": list(self.protein_indices),
            "char_offsets": [list(o) for o in self.char_offsets],
            "token_lemmas": list(self.token_lemmas),
            "head_node_index": self.head_node_index,
            "pos": self.pos,
            "dep": self.dep,
            "is_negated": self.is_negated,
            "assertion_status": self.assertion_status,
            "is_nominalized": self.is_nominalized,
            "verb_form": self.verb_form,
            "node_indices": list(self.node_indices),
            "parser_dep": self.parser_dep,
            "parser_pos": self.parser_pos,
            "parser_features": self.parser_features,
            "parser_misc": self.parser_misc,
            "enhanced_heads": [list(pair) for pair in self.enhanced_heads],
            "reference_kinds": list(self.reference_kinds),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Span":
        return cls(
            tokens=tuple(d["tokens"]),
            lemmas=tuple(d["lemmas"]),
            match_keys=tuple(d["match_keys"]),
            protein_indices=tuple(d["protein_indices"]),
            char_offsets=tuple(tuple(o) for o in d["char_offsets"]),
            token_lemmas=tuple(d.get("token_lemmas", ())),
            head_node_index=int(d.get("head_node_index", -1)),
            pos=d.get("pos", ""),
            dep=d.get("dep", ""),
            is_negated=d.get("is_negated", False),
            assertion_status=d.get("assertion_status", AssertionStatus.ASSERTED.value),
            is_nominalized=d.get("is_nominalized", False),
            verb_form=d.get("verb_form", ""),
            node_indices=tuple(d.get("node_indices", ())),
            parser_dep=d.get("parser_dep", ""),
            parser_pos=d.get("parser_pos", ""),
            parser_features=d.get("parser_features", ""),
            parser_misc=d.get("parser_misc", ""),
            enhanced_heads=tuple(
                (int(pair[0]), str(pair[1]))
                for pair in d.get("enhanced_heads", ())
                if isinstance(pair, (tuple, list)) and len(pair) >= 2
            ),
            reference_kinds=tuple(str(v) for v in d.get("reference_kinds", ())),
        )


@dataclass(frozen=True)
class ReferentLink:
    """Protein-bearing link inside a lexical argument (view of owners/descriptors)."""

    link_type: str
    span: Span
    syntax: str = ""

    @property
    def protein_indices(self) -> tuple[int, ...]:
        return self.span.protein_indices



@dataclass(frozen=True)
class CandidateArgument:
    """Candidate event argument: entity form (core + owners) or ``event_ref`` form."""
    core: Optional[Span] = None
    owners: tuple[Span, ...] = ()
    descriptors: tuple[Span, ...] = ()
    case_markers: tuple[Span, ...] = ()      # prepositions ("in", "for", "with")
    event_ref: str = ""                       # event_id of nested CandidateEvent
    role: str = ""                            # parser hint: "ARG0", "nsubj", "nmod:in", ...
    group: dict = field(default_factory=dict, compare=False, hash=False)
    # Optional topology provenance; does not affect emission.

    @property
    def is_event_reference(self) -> bool:
        return bool(self.event_ref)

    @property
    def all_protein_indices(self) -> tuple[int, ...]:
        """All protein indices visible on the argument (audit view), per coordination member."""
        if self.is_event_reference:
            return ()
        indices: set[int] = set()
        if self.core is not None:
            indices.update(self.core.protein_indices)
        for owner in self.owners:
            indices.update(owner.protein_indices)
        for descriptor in self.descriptors:
            indices.update(descriptor.protein_indices)
        if self.group:
            core_prots = set(self.core.protein_indices) if self.core is not None else set()
            owner_prots: set[int] = set()
            for owner in self.owners:
                owner_prots.update(owner.protein_indices)
            anchor_set = core_prots | owner_prots
            if anchor_set:
                for member in self.group.get("members", ()):
                    member_indices = set(int(i) for i in member.get("protein_indices", ()))
                    if member_indices & anchor_set:
                        indices.update(member_indices)
        return tuple(sorted(indices))

    @property
    def projection_protein_indices(self) -> tuple[int, ...]:
        """Endpoints for projection: core, non-negated owners and descriptors."""
        if self.is_event_reference:
            return ()
        indices: set[int] = set()
        if self.core is not None:
            indices.update(self.core.protein_indices)
        for owner in self.owners:
            if not owner.is_negated:
                indices.update(owner.protein_indices)
        for descriptor in self.descriptors:
            indices.update(descriptor.protein_indices)
        return tuple(sorted(indices))

    @property
    def referent_links(self) -> tuple[ReferentLink, ...]:
        """Protein-bearing lexical links; does not affect serialization or projection."""
        if self.is_event_reference:
            return ()
        links: list[ReferentLink] = []
        for owner in self.owners:
            links.append(ReferentLink(
                link_type="owner",
                span=owner,
                syntax=owner.dep or "",
            ))
        for descriptor in self.descriptors:
            links.append(ReferentLink(
                link_type="descriptor",
                span=descriptor,
                syntax=descriptor.dep or "",
            ))
        return tuple(links)

    @property
    def compound_owner_indices(self) -> tuple[int, ...]:
        """Protein indices that entered only via a compound modifier (``P1 receptor``)."""
        if self.is_event_reference:
            return ()
        compound: set[int] = set()
        other: set[int] = set()
        if self.core is not None:
            core_is_compound = (
                (self.core.dep or "") == "compound"
                or (self.role or "") == "compound"
            )
            if core_is_compound:
                compound.update(self.core.protein_indices)
            else:
                other.update(self.core.protein_indices)
        for owner in self.owners:
            if (owner.dep or "") == "compound":
                compound.update(owner.protein_indices)
            else:
                other.update(owner.protein_indices)
        return tuple(sorted(compound - other))

    @property
    def owner_protein_indices(self) -> tuple[int, ...]:
        """Protein indices from owner spans (export provenance)."""
        if self.is_event_reference:
            return ()
        out: set[int] = set()
        for owner in self.owners:
            out.update(owner.protein_indices)
        return tuple(sorted(out))

    @property
    def descriptor_protein_indices(self) -> tuple[int, ...]:
        """Protein indices from descriptor spans (``anti-P0 antibody``)."""
        if self.is_event_reference:
            return ()
        out: set[int] = set()
        for descriptor in self.descriptors:
            out.update(descriptor.protein_indices)
        return tuple(sorted(out))

    @property
    def is_negated(self) -> bool:
        """An arg is negated if its core is negated (event_ref args inherit from inner event)."""
        if self.is_event_reference:
            return False
        return self.core.is_negated if self.core is not None else False

    @property
    def assertion_status(self) -> str:
        """Assertion status from the core span; event_ref args inherit elsewhere."""
        if self.is_event_reference or self.core is None:
            return AssertionStatus.ASSERTED.value
        return self.core.assertion_status

    def to_dict(self) -> dict:
        result = {
            "core": self.core.to_dict() if self.core is not None else None,
            "owners": [s.to_dict() for s in self.owners],
            "descriptors": [s.to_dict() for s in self.descriptors],
            "case_markers": [s.to_dict() for s in self.case_markers],
            "event_ref": self.event_ref,
            "role": self.role,
        }
        if self.group:
            result["group"] = dict(self.group)
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateArgument":
        return cls(
            core=Span.from_dict(d["core"]) if d.get("core") else None,
            owners=tuple(Span.from_dict(s) for s in d.get("owners", [])),
            descriptors=tuple(Span.from_dict(s) for s in d.get("descriptors", [])),
            case_markers=tuple(Span.from_dict(s) for s in d.get("case_markers", [])),
            event_ref=d.get("event_ref", ""),
            role=d.get("role", ""),
            group=dict(d.get("group", {})),
        )


# CandidateEvent -- the whole event (predicate + role-keyed arguments).

# Role slot names per event type
INTERACTION_DIRECTED_SLOTS = ("sources", "targets")
INTERACTION_UNDIRECTED_SLOTS = ("participants",)
RELATIONAL_SLOTS = ("entities_a", "entities_b")
STATE_SLOTS = ("participants",)


@dataclass
class CandidateEvent:
    """Parser-neutral event produced by a realization channel."""
    event_type: EventType
    event_id: str                                                # "{parser}_{sent_hash}_{seq}"
    group_id: str = ""                                            # shared across coord expansions
    parser_source: str = ""                                       # "stanza_craft", ...
    extraction_detail: str = ""                                   # detailed rule provenance
    construction: str = ""                                        # "verbal"|"passive"|"nominalized"|"compound_state"
    realization_channel: str = RealizationChannel.UNKNOWN.value
    sentence_text: str = ""                                       # source sentence

    predicate: Optional[Span] = None

    # Role-keyed arguments; see *_SLOTS. Any argument may use event_ref.
    arguments: dict[str, tuple[CandidateArgument, ...]] = field(default_factory=dict)

    is_directed: bool = True
    relation_subtype: str = ""  # "IS-A", "HAS", "PART-OF" only for RELATION_STATEMENT

    custom_values: dict = field(default_factory=dict)             # parser-opaque metadata
    projected_pairs: tuple[tuple[int, int], ...] = ()             # canonical projection output

    def __post_init__(self) -> None:
        from src.models.semantic_vocabulary import Construction, SemanticSlot

        self.construction = Construction.normalize(self.construction)
        self.realization_channel = RealizationChannel.normalize(self.realization_channel)
        unknown_slots = [
            slot for slot in self.arguments
            if SemanticSlot.normalize(slot) == SemanticSlot.UNKNOWN.value
        ]
        if unknown_slots:
            raise ValueError(f"CandidateEvent has unknown semantic slots: {unknown_slots}")
        self.projected_pairs = tuple(sorted({
            tuple(sorted((int(pair[0]), int(pair[1]))))
            for pair in self.projected_pairs
            if len(pair) == 2 and int(pair[0]) != int(pair[1])
        }))

    @property
    def pred_relations(self) -> list[tuple[int, int]]:
        """Read-only compatibility spelling for report/evaluation consumers."""
        return list(self.projected_pairs)

    @property
    def semantic_event_class(self) -> str:
        """Return the stable v8 event class without rewriting frame internals."""
        if self.event_type is EventType.RELATION_STATEMENT:
            return SemanticEventClass.RELATION_STATEMENT.value
        return SemanticEventClass.PREDICATE_EVENT.value

    def has_nested_arg(self) -> bool:
        return any(
            a.is_event_reference
            for args in self.arguments.values()
            for a in args
        )

    def referenced_event_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        for args in self.arguments.values():
            for a in args:
                if a.is_event_reference:
                    ids.append(a.event_ref)
        return tuple(ids)

    def iter_args(self):
        for slot, args in self.arguments.items():
            for a in args:
                yield slot, a

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.name,
            "semantic_event_class": self.semantic_event_class,
            "event_id": self.event_id,
            "group_id": self.group_id,
            "parser_source": self.parser_source,
            "extraction_detail": self.extraction_detail,
            "construction": self.construction,
            "realization_channel": self.realization_channel,
            "sentence_text": self.sentence_text,
            "is_directed": self.is_directed,
            "relation_subtype": self.relation_subtype,
            "predicate": self.predicate.to_dict() if self.predicate is not None else None,
            "arguments": {
                slot: [a.to_dict() for a in args]
                for slot, args in self.arguments.items()
            },
            "custom_values": dict(self.custom_values),
            "projected_pairs": [list(pair) for pair in self.projected_pairs],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateEvent":
        return cls(
            event_type=EventType.from_name(d["event_type"]),
            event_id=d["event_id"],
            group_id=d.get("group_id", ""),
            parser_source=d.get("parser_source", ""),
            extraction_detail=d.get("extraction_detail", ""),
            construction=d.get("construction", ""),
            realization_channel=d.get("realization_channel", RealizationChannel.UNKNOWN.value),
            sentence_text=d.get("sentence_text", ""),
            is_directed=d.get("is_directed", True),
            relation_subtype=d.get("relation_subtype", ""),
            predicate=Span.from_dict(d["predicate"]) if d.get("predicate") else None,
            arguments={
                slot: tuple(CandidateArgument.from_dict(a) for a in args)
                for slot, args in d.get("arguments", {}).items()
            },
            custom_values=dict(d.get("custom_values", {})),
            projected_pairs=tuple(tuple(pair) for pair in d.get("projected_pairs", ())),
        )
