"""Parser-neutral event extraction from ``ParsedSentence`` input."""

from src.extraction.parsers import (
    EventExtractor,
    SemanticEventExtractor,
)

__all__ = [
    "EventExtractor",
    "SemanticEventExtractor",
]
