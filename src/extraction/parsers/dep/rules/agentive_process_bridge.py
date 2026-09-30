"""Represent ``A-mediated B expression`` as nested events.

Does not assert a direct A–B interaction; that projection is a separate, disabled rule.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.extraction.rules import RuleApplication, append_rule_application
from src.models.candidate import (
    CandidateArgument,
    CandidateEvent,
    EventType,
    RealizationChannel,
    Span,
)


AGENTIVE_PROCESS_BRIDGE_RULE_ID = "extraction.dep_agentive_process_bridge@1"
AGENTIVE_PROCESS_BRIDGE_PROJECTION_RULE_ID = (
    "extraction.dep_agentive_process_bridge_projection@1"
)

# Detail prefixes are stable provenance keys consumed by tests and audits.
OUTER_DETAIL_PREFIX = "DEP-agentive-process-bridge-"
INNER_DETAIL_PREFIX = "DEP-agentive-process-"
INNER_DETAIL_SUFFIX = "-inner"
PROJECTION_DETAIL_PREFIX = "DEP-agentive-process-projection-"

_CONSTRUCTION = "agentive_process_bridge"
_LEXICON_DIR = Path(__file__).resolve().parents[5] / "data" / "lexicons"


@lru_cache(maxsize=4)
def _lexicon(name: str) -> frozenset[str]:
    path = _LEXICON_DIR / name
    if not path.exists():
        return frozenset()
    return frozenset(
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def is_agentive_bridge_predicate(lemma: str) -> bool:
    """True for the focused agentive/causative control participles."""
    return bool(lemma) and lemma.strip().lower() in _lexicon("agentive_bridge_predicate.txt")


def is_agentive_process_nominal(lemma: str) -> bool:
    """Lexical fallback for process heads the nominalization detector missed."""
    return bool(lemma) and lemma.strip().lower() in _lexicon("agentive_process_nominal.txt")


def build_agentive_process_bridge_events(
    *,
    bridge_lemma: str,
    process_lemma: str,
    controller: CandidateArgument,
    affected: CandidateArgument,
    bridge_predicate: Span,
    process_predicate: Span,
    construction_surface: str,
    affected_attachment: str,
    emit_projection_event: bool = False,
    next_id,
    sentence_text: str,
    parser_source: str,
) -> list[CandidateEvent]:
    """Build inert inner/outer events for one bridge; projects no pair."""
    inner_id = next_id()
    inner = CandidateEvent(
        event_type=EventType.INTERACTION,
        event_id=inner_id,
        parser_source=parser_source,
        extraction_detail=f"{INNER_DETAIL_PREFIX}{process_lemma}{INNER_DETAIL_SUFFIX}",
        construction=_CONSTRUCTION,
        sentence_text=sentence_text,
        predicate=process_predicate,
        arguments={"participants": (affected,)},
        is_directed=False,
    )

    controller_indices = sorted(controller.projection_protein_indices)
    affected_indices = sorted(affected.projection_protein_indices)
    outer = CandidateEvent(
        event_type=EventType.INTERACTION,
        event_id=next_id(),
        parser_source=parser_source,
        extraction_detail=f"{OUTER_DETAIL_PREFIX}{bridge_lemma}",
        construction=_CONSTRUCTION,
        sentence_text=sentence_text,
        predicate=bridge_predicate,
        arguments={
            "sources": (controller,),
            "targets": (CandidateArgument(event_ref=inner_id, role="agentive_process"),),
        },
        is_directed=True,
    )
    outer = append_rule_application(
        outer,
        RuleApplication(
            rule_id=AGENTIVE_PROCESS_BRIDGE_RULE_ID,
            stage="event_composition",
            reason="agentive_participle_controls_process_of_affected_entity",
            evidence={
                "bridge_lemma": bridge_lemma,
                "process_lemma": process_lemma,
                "controller_referents": controller_indices,
                "affected_referents": affected_indices,
                "affected_attachment": affected_attachment,
                "construction_surface": construction_surface,
                "inner_event_id": inner_id,
                "is_direct_ppi": False,
            },
        ),
    )
    events = [inner, outer]
    if not emit_projection_event:
        return events

    projection = CandidateEvent(
        event_type=EventType.INTERACTION,
        event_id=next_id(),
        parser_source=parser_source,
        extraction_detail=f"{PROJECTION_DETAIL_PREFIX}{bridge_lemma}",
        construction=_CONSTRUCTION,
        realization_channel=RealizationChannel.EVENT_NOMINAL.value,
        sentence_text=sentence_text,
        predicate=bridge_predicate,
        arguments={
            "sources": (controller,),
            "targets": (affected,),
        },
        is_directed=True,
    )
    events.append(
        append_rule_application(
            projection,
            RuleApplication(
                rule_id=AGENTIVE_PROCESS_BRIDGE_PROJECTION_RULE_ID,
                stage="pair_projection_evidence",
                reason="agentive_participle_projects_controller_to_affected_process_entity",
                evidence={
                    "bridge_lemma": bridge_lemma,
                    "process_lemma": process_lemma,
                    "controller_referents": controller_indices,
                    "affected_referents": affected_indices,
                    "affected_attachment": affected_attachment,
                    "construction_surface": construction_surface,
                    "base_bridge_event_id": outer.event_id,
                    "inner_event_id": inner_id,
                    "policy_projection": "controller_to_affected_process_entity",
                },
            ),
        )
    )
    return events


__all__ = [
    "AGENTIVE_PROCESS_BRIDGE_RULE_ID",
    "AGENTIVE_PROCESS_BRIDGE_PROJECTION_RULE_ID",
    "INNER_DETAIL_PREFIX",
    "INNER_DETAIL_SUFFIX",
    "OUTER_DETAIL_PREFIX",
    "PROJECTION_DETAIL_PREFIX",
    "build_agentive_process_bridge_events",
    "is_agentive_bridge_predicate",
    "is_agentive_process_nominal",
]
