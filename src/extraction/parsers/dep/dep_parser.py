"""Dependency-based reader facade with sentence-local extraction state."""
from src.models.candidate import CandidateEvent
from src.parsing import ParsedSentence
from ..base import EventExtractor
from .context import ExtractionContext
from .event_channels import EventChannels


class SemanticEventExtractor(EventExtractor):
    def __init__(self, resources=None):
        self.resources = resources

    @property
    def parser_name(self):
        return "dependency"

    def extract(self, parsed: ParsedSentence, sentence_id: str = "") -> list[CandidateEvent]:
        if not isinstance(parsed, ParsedSentence):
            raise TypeError(f"extract expects a ParsedSentence; got {type(parsed).__name__}")
        context = ExtractionContext(parsed, self.resources)
        return EventChannels(context).extract_events_by_channel(parsed, context.next_id)

    def observe(self, parsed: ParsedSentence, sentence_id: str = ""):
        from src.models.observation import project_observation
        events = self.extract(parsed, sentence_id)
        return events, [project_observation(event, parsed) for event in events]
