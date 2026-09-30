"""Semantic event-extractor namespace."""
from .base import EventExtractor
from .dep import SemanticEventExtractor

__all__ = [
    "EventExtractor",
    "SemanticEventExtractor",
]
