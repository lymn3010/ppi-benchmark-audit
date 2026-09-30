"""Ordered event-extraction channels (realizations, not event types)."""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Iterable

from src.models.candidate import CandidateEvent, RealizationChannel
from src.extraction.rules import RuleApplication, append_rule_application

from .rules import TRANSITIVE_COMPOUND_OWNER_RULE_ID


class EventChannels:
    """Dispatch dependency evidence through the ordered realization channels."""

    def __init__(self, context):
        from .predicate_events import PredicateEventChannel
        from .nominalization_events import NominalizationEventChannel
        from .state_nominal_events import StateNominalEventChannel
        from .appositive_events import AppositiveEventChannel
        from .hyphen_participle_events import HyphenParticipleEventChannel
        from .agentive_process_bridge_events import AgentiveProcessBridgeEventChannel
        self.context = context
        self.predicate = PredicateEventChannel(context.resources, context)
        self.nominal = NominalizationEventChannel(context.resources, context)
        self.state = StateNominalEventChannel(context.resources, context)
        self.appositive = AppositiveEventChannel(context.resources, context)
        self.hyphen = HyphenParticipleEventChannel(context.resources, context)
        self.bridge = AgentiveProcessBridgeEventChannel(context.resources, context)

    def extract_events_by_channel(self, parsed, next_id) -> list[CandidateEvent]:
        events: list[CandidateEvent] = []
        consumed_inner_heads = self.context.consumed_inner_heads

        events.extend(self._extract_clause_channel(
            parsed, next_id, consumed_inner_heads,
        ))
        events.extend(self._extract_event_nominal_channel(
            parsed, next_id, consumed_inner_heads,
        ))
        events.extend(self._extract_state_nominal_channel(parsed, next_id))
        events.extend(self._extract_appositive_channel(parsed, next_id))
        return self._append_argument_binding_rule_traces(events)

    @staticmethod
    def _tag_channel(
        events: Iterable[CandidateEvent],
        channel: RealizationChannel | Callable[[CandidateEvent], RealizationChannel],
    ) -> list[CandidateEvent]:
        """Attach realization provenance without changing event semantics."""
        return [
            replace(
                event,
                realization_channel=(
                    channel(event).value if callable(channel) else channel.value
                ),
            )
            for event in events
        ]

    def _extract_clause_channel(
        self, parsed, next_id, consumed_inner_heads: set[int],
    ) -> list[CandidateEvent]:
        events = self.predicate._extract_predicate_events_from_parsed(
            parsed, next_id, parsed.text, consumed_inner_heads,
        )
        events += self.hyphen._extract_hyphen_participle_ppi_from_parsed(
            parsed, next_id, parsed.text,
        )
        events += self.appositive._extract_copula_is_a_from_parsed(
            parsed, next_id, parsed.text,
        )
        events += self.bridge._extract_agentive_process_bridge_from_parsed(
            parsed, next_id, parsed.text,
        )
        return self._tag_channel(events, RealizationChannel.CLAUSE)

    def _extract_event_nominal_channel(
        self, parsed, next_id, consumed_inner_heads: set[int],
    ) -> list[CandidateEvent]:
        """Extract nominalized predicates, including ``interaction between A and B``."""
        events = self.nominal._extract_nominalization_events_from_parsed(
            parsed, next_id, parsed.text, consumed_inner_heads,
        )
        return self._tag_channel(events, RealizationChannel.EVENT_NOMINAL)

    def _extract_state_nominal_channel(self, parsed, next_id) -> list[CandidateEvent]:
        """Extract state NPs whose head names a joint state (``complex``, ``dimer``)."""
        events = self.state._extract_compound_complex_from_parsed(parsed, next_id, parsed.text)
        events += self.state._extract_compound_adj_ppi_from_parsed(parsed, next_id, parsed.text)
        return self._tag_channel(events, RealizationChannel.STATE_NOMINAL)

    def _extract_appositive_channel(self, parsed, next_id) -> list[CandidateEvent]:
        """Extract apposition, alias and identity constructions (copula goes to ``clause``)."""
        events: list[CandidateEvent] = []
        events.extend(self.appositive._extract_apposition_from_parsed(parsed, next_id, parsed.text))
        events.extend(self.appositive._extract_parenthetical_alias_from_parsed(parsed, next_id, parsed.text))
        events.extend(self.appositive._extract_alias_phrase_from_parsed(parsed, next_id, parsed.text))
        events.extend(self.appositive._extract_loose_identity_candidate_from_parsed(parsed, next_id, parsed.text))
        events.extend(self.appositive._extract_identity_from_parsed(parsed, next_id, parsed.text))
        events.extend(self.appositive._extract_nominal_alias_genitive_from_parsed(parsed, next_id, parsed.text))
        return self._tag_channel(events, RealizationChannel.APPOSITIVE)

    @staticmethod
    def _append_argument_binding_rule_traces(
        events: list[CandidateEvent],
    ) -> list[CandidateEvent]:
        return [
            EventChannels._append_transitive_compound_owner_trace(event)
            for event in events
        ]

    @staticmethod
    def _append_transitive_compound_owner_trace(
        event: CandidateEvent,
    ) -> CandidateEvent:
        evidence: list[dict] = []
        seen: set[tuple] = set()
        for slot, arguments in event.arguments.items():
            for argument in arguments:
                for owner in (argument.group or {}).get("ownership", ()):
                    if owner.get("source") != "compound_transitive":
                        continue
                    key = (
                        slot,
                        argument.role,
                        int(owner.get("head_index", -1)),
                        tuple(int(i) for i in owner.get("protein_indices", ())),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    evidence.append({
                        "slot": slot,
                        "argument_role": argument.role,
                        "owner_head": key[2],
                        "protein_indices": list(key[3]),
                    })
        if not evidence:
            return event
        return append_rule_application(
            event,
            RuleApplication(
                rule_id=TRANSITIVE_COMPOUND_OWNER_RULE_ID,
                stage="event_argument_binding",
                reason="compound_owner_chain_followed_transitively",
                evidence={"owners": evidence},
            ),
        )


EventChannelMixin = EventChannels  # Compatibility import.
