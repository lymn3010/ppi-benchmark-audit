"""Typed lexicalized-path pattern evidence (not pair-emitting)."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re
from typing import Any


_PROTEIN_RE = re.compile(r"^protein\d+$")
_LEXICON_DIR = Path(__file__).resolve().parents[2] / "data" / "lexicons"
_INTERMEDIATE_CATEGORY_ORDER = (
    "bridge_predicate",
    "process_nominal",
    "physical_relation_nominal",
    "role_nominal",
    "carrier_noun",
    "group_composition_head",
    "metric_measurement",
    "location_context",
    "unknown_intermediate",
)


def _clean(value: object, default: str = "_") -> str:
    text = str(value or "").strip().lower()
    return text or default


def _pair(value: object) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("LexPathPatternCandidate.pair requires two indices")
    first, second = int(value[0]), int(value[1])
    if first < 0 or second < 0 or first == second:
        raise ValueError("LexPathPatternCandidate.pair requires distinct non-negative indices")
    return (first, second) if first < second else (second, first)


@dataclass(frozen=True)
class LexPathEndpointAttachment:
    """How one protein endpoint attaches to the mined dependency path."""

    protein_index: int
    dep: str = "_"
    prep: str = "_"
    via_lemma: str = "_"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LexPathEndpointAttachment":
        return cls(
            protein_index=int(payload.get("protein_index", -1)),
            dep=_clean(payload.get("dep")),
            prep=_clean(payload.get("prep")),
            via_lemma=_clean(payload.get("via_lemma")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protein_index": self.protein_index,
            "dep": self.dep,
            "prep": "" if self.prep == "_" else self.prep,
            "via_lemma": "" if self.via_lemma == "_" else self.via_lemma,
        }

    @property
    def signature(self) -> str:
        return f"{self.dep}:{self.prep}:{self.via_lemma}"


@dataclass(frozen=True)
class LexPathIntermediateHead:
    """Typed intermediate head on a lexicalized path; may have several categories."""

    head: str
    categories: tuple[str, ...] = ("unknown_intermediate",)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LexPathIntermediateHead":
        raw_categories = payload.get("categories") or payload.get("types") or ()
        categories = tuple(
            category
            for category in (_clean(item) for item in raw_categories)
            if category != "_"
        )
        return cls(
            head=_clean(payload.get("head")),
            categories=_ordered_categories(categories) or ("unknown_intermediate",),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "head": self.head,
            "categories": list(self.categories),
        }

    @property
    def signature(self) -> str:
        return f"{self.head}:{','.join(self.categories)}"


@dataclass(frozen=True)
class LexPathPatternCandidate:
    """Serializable lexicalized-path pattern evidence; carrier heads are data-derived."""

    sentence_id: str
    pair: tuple[int, int]
    semantic_signature: str
    construction: str
    predicate_chain: str = "_"
    prep_signature: str = "_"
    endpoint_attachments: tuple[LexPathEndpointAttachment, ...] = ()
    carrier_heads: tuple[str, ...] = ()
    typed_intermediate_heads: tuple[LexPathIntermediateHead, ...] = ()
    stage: str = ""
    gold_status: str = ""
    surface_window: str = ""
    raw: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @classmethod
    def from_mined_row(cls, row: dict[str, Any]) -> "LexPathPatternCandidate":
        semantic_signature = str(row.get("semantic_signature") or row.get("pattern_signature") or "")
        if not semantic_signature:
            raise ValueError("LexPathPatternCandidate requires semantic_signature or pattern_signature")
        attachments = tuple(
            LexPathEndpointAttachment.from_dict(edge)
            for edge in row.get("endpoint_edges", ()) or ()
            if isinstance(edge, dict)
        )
        prep_signature = _prep_signature(row, attachments)
        carrier_heads = tuple(_carrier_heads(row, attachments))
        typed_intermediate_heads = tuple(_typed_intermediate_heads(row, carrier_heads))
        predicate_chain = str(row.get("predicate_chain") or _predicate_from_semantic(semantic_signature) or "_")
        return cls(
            sentence_id=str(row.get("sentence_id") or ""),
            pair=_pair(row.get("pair")),
            semantic_signature=semantic_signature,
            construction=_clean(row.get("construction")),
            predicate_chain=predicate_chain,
            prep_signature=prep_signature,
            endpoint_attachments=attachments,
            carrier_heads=carrier_heads,
            typed_intermediate_heads=typed_intermediate_heads,
            stage=str(row.get("stage") or ""),
            gold_status=str(row.get("gold_status") or ""),
            surface_window=str(row.get("surface_window") or ""),
            raw=dict(row),
        )

    @property
    def endpoint_attachment_signature(self) -> str:
        if not self.endpoint_attachments:
            return "_"
        return "|".join(sorted(edge.signature for edge in self.endpoint_attachments))

    @property
    def carrier_signature(self) -> str:
        return "+".join(self.carrier_heads) if self.carrier_heads else "_"

    @property
    def typed_intermediate_signature(self) -> str:
        if not self.typed_intermediate_heads:
            return "_"
        return "|".join(head.signature for head in self.typed_intermediate_heads)

    @property
    def typed_intermediate_signatures(self) -> dict[str, str]:
        grouped: dict[str, list[str]] = {category: [] for category in _INTERMEDIATE_CATEGORY_ORDER}
        for item in self.typed_intermediate_heads:
            for category in item.categories:
                grouped.setdefault(category, []).append(item.head)
        out: dict[str, str] = {}
        for category in _INTERMEDIATE_CATEGORY_ORDER:
            values = sorted(set(grouped.get(category, ())))
            if values:
                out[category] = "+".join(values)
        for category in sorted(set(grouped) - set(_INTERMEDIATE_CATEGORY_ORDER)):
            values = sorted(set(grouped.get(category, ())))
            if values:
                out[category] = "+".join(values)
        return out

    def intermediate_signature_for(self, category: str) -> str:
        return self.typed_intermediate_signatures.get(_clean(category), "_")

    @property
    def semantic_key(self) -> str:
        return self.semantic_signature

    @property
    def construction_key(self) -> str:
        return f"{self.semantic_key}|construction={self.construction}"

    @property
    def pp_key(self) -> str:
        return f"{self.construction_key}|prep={self.prep_signature}"

    @property
    def attachment_key(self) -> str:
        return f"{self.pp_key}|attach={self.endpoint_attachment_signature}"

    @property
    def carrier_key(self) -> str:
        return f"{self.attachment_key}|carrier={self.carrier_signature}"

    @property
    def typed_intermediate_key(self) -> str:
        return f"{self.carrier_key}|typed_intermediates={self.typed_intermediate_signature}"

    @property
    def predicate_chain_key(self) -> str:
        return (
            f"{self.predicate_chain}|construction={self.construction}|"
            f"prep={self.prep_signature}|carrier={self.carrier_signature}|"
            f"typed_intermediates={self.typed_intermediate_signature}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "lexpath_pattern_candidate_v1",
            "sentence_id": self.sentence_id,
            "pair": list(self.pair),
            "semantic_signature": self.semantic_signature,
            "construction": self.construction,
            "predicate_chain": self.predicate_chain,
            "prep_signature": self.prep_signature,
            "endpoint_attachments": [edge.to_dict() for edge in self.endpoint_attachments],
            "endpoint_attachment_signature": self.endpoint_attachment_signature,
            "carrier_heads": list(self.carrier_heads),
            "carrier_signature": self.carrier_signature,
            "typed_intermediate_heads": [
                head.to_dict() for head in self.typed_intermediate_heads
            ],
            "typed_intermediate_signature": self.typed_intermediate_signature,
            "typed_intermediate_signatures": self.typed_intermediate_signatures,
            "stage": self.stage,
            "gold_status": self.gold_status,
            "surface_window": self.surface_window,
            "keys": {
                "semantic": self.semantic_key,
                "construction": self.construction_key,
                "pp": self.pp_key,
                "attachment": self.attachment_key,
                "carrier": self.carrier_key,
                "typed_intermediate": self.typed_intermediate_key,
                "predicate_chain": self.predicate_chain_key,
            },
        }


def _predicate_from_semantic(signature: str) -> str:
    parts = str(signature or "").split("|")
    return parts[1] if len(parts) >= 2 else "_"


def _prep_signature(
    row: dict[str, Any],
    attachments: tuple[LexPathEndpointAttachment, ...],
) -> str:
    explicit = str(row.get("prep_signature") or "").strip().lower()
    if explicit:
        return explicit
    preps = row.get("endpoint_preps") or ()
    if preps:
        values = sorted({_clean(prep) for prep in preps if _clean(prep) != "_"})
        return "+".join(values) if values else "_"
    values = sorted({edge.prep for edge in attachments if edge.prep != "_"})
    if values:
        return "+".join(values)
    edge_prep = _clean(row.get("edge_prep"))
    return edge_prep


def _carrier_heads(
    row: dict[str, Any],
    attachments: tuple[LexPathEndpointAttachment, ...],
) -> list[str]:
    explicit = str(row.get("carrier_head_signature") or "").strip().lower()
    if explicit and explicit != "_":
        return [part for part in explicit.split("+") if part]
    apexes = {
        _clean(row.get("apex_lemma")),
        _clean(row.get("external_apex_lemma")),
    }
    out: set[str] = set()
    for edge in attachments:
        via = edge.via_lemma
        if via and via != "_" and via not in apexes and not _PROTEIN_RE.match(via):
            out.add(via)
    for step in row.get("dep_path", ()) or ():
        if not isinstance(step, dict):
            continue
        lemma = _clean(step.get("lemma") or step.get("text"))
        pos = str(step.get("pos") or "")
        if lemma in apexes or lemma == "_" or _PROTEIN_RE.match(lemma):
            continue
        if pos in {"NOUN", "PROPN"}:
            out.add(lemma)
    return sorted(out)


def _typed_intermediate_heads(
    row: dict[str, Any],
    carrier_heads: tuple[str, ...],
) -> list[LexPathIntermediateHead]:
    explicit = row.get("typed_intermediate_heads")
    if isinstance(explicit, list):
        out = [
            LexPathIntermediateHead.from_dict(item)
            for item in explicit
            if isinstance(item, dict)
        ]
        if out:
            return sorted(out, key=lambda item: item.head)
    return [
        LexPathIntermediateHead(head=head, categories=_classify_intermediate_head(head))
        for head in carrier_heads
    ]


def _classify_intermediate_head(head: str) -> tuple[str, ...]:
    lemma = _clean(head)
    if lemma == "_" or _PROTEIN_RE.match(lemma):
        return ("unknown_intermediate",)
    categories: list[str] = []
    if lemma in _lexicon("lexpath_bridge_predicate.txt"):
        categories.append("bridge_predicate")
    if lemma in _nominal_to_verbal():
        categories.append("process_nominal")
    if lemma in _lexicon("lexpath_process_nominal.txt"):
        categories.append("process_nominal")
    if lemma in _lexicon("reciprocal_event_nominal.txt"):
        categories.append("physical_relation_nominal")
    if lemma in _lexicon("lexpath_role_nominal.txt"):
        categories.append("role_nominal")
    if lemma in _lexicon("lexpath_carrier_noun.txt"):
        categories.append("carrier_noun")
    if lemma in _lexicon("collective_state_nominal.txt"):
        categories.append("group_composition_head")
    if lemma in _lexicon("lexpath_group_composition_head.txt"):
        categories.append("group_composition_head")
    if lemma in _lexicon("lexpath_metric_measurement_head.txt"):
        categories.append("metric_measurement")
    if lemma in _lexicon("lexpath_location_context_head.txt"):
        categories.append("location_context")
    return _ordered_categories(categories) or ("unknown_intermediate",)


def _ordered_categories(categories: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    seen = {_clean(category) for category in categories if _clean(category) != "_"}
    if not seen:
        return ()
    ordered = [category for category in _INTERMEDIATE_CATEGORY_ORDER if category in seen]
    ordered.extend(sorted(seen - set(_INTERMEDIATE_CATEGORY_ORDER)))
    return tuple(ordered)


@lru_cache(maxsize=32)
def _lexicon(name: str) -> frozenset[str]:
    path = _LEXICON_DIR / name
    if not path.exists():
        return frozenset()
    return frozenset(
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


@lru_cache(maxsize=1)
def _nominal_to_verbal() -> dict[str, str]:
    path = _LEXICON_DIR / "nominal_to_verbal.csv"
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 2:
                continue
            nominal = row[0].strip().lower()
            verbal = row[1].strip().lower()
            if nominal and verbal:
                out[nominal] = verbal
    return out


__all__ = [
    "LexPathEndpointAttachment",
    "LexPathIntermediateHead",
    "LexPathPatternCandidate",
]
