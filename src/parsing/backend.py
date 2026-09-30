"""Parser-backend contract for the v8 runtime."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .syntax import ParsedSentence


@runtime_checkable
class ParserBackend(Protocol):
    """A parser backend must normalize its output before returning."""

    def parse(self, text: str, sentence_id: str = "") -> ParsedSentence:
        """Parse one sentence into the shared parser-neutral representation."""
        ...


@runtime_checkable
class BatchParserBackend(ParserBackend, Protocol):
    """Optional high-throughput extension implemented by capable backends."""

    def parse_many(
        self,
        items: list[tuple[str, str]],
        *,
        batch_size: int = 64,
    ) -> list[ParsedSentence]:
        ...
