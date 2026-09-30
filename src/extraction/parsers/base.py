"""Semantic event-extractor interface.

Extractors map a ParsedSentence to serializable ``CandidateEvent`` objects; no pairs.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from src.models.candidate import CandidateEvent
from src.parsing import ParsedSentence


class EventExtractor(ABC):

    @property
    @abstractmethod
    def parser_name(self) -> str:
        """Short identifier for the extraction implementation."""
        ...

    @abstractmethod
    def extract(
        self, parsed: ParsedSentence, sentence_id: str = "",
    ) -> list[CandidateEvent]:
        """Extract semantic events from parser-neutral syntax."""
        ...

    @staticmethod
    def _hash_sentence(text: str) -> str:
        """Short stable hash for event_id generation."""
        return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:8]
