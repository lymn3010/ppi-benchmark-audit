"""Sentence-local state shared by realization channels."""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
from src.parsing import ParsedSentence
from src.runtime_resources import RuntimeResources
from .group_index import GroupIndex


@dataclass
class ExtractionContext:
    parsed: ParsedSentence
    resources: RuntimeResources | None = None
    consumed_inner_heads: set[int] = field(default_factory=set)
    groups: GroupIndex = field(init=False)
    _prefix: str = field(init=False)
    _counter: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.groups = GroupIndex(self.parsed)
        self._prefix = f"{self.parsed.parser_name or 'dependency'}_{hashlib.md5(self.parsed.text.encode('utf-8', errors='replace')).hexdigest()[:8]}"

    def next_id(self) -> str:
        value = f"{self._prefix}_{self._counter}"
        self._counter += 1
        return value
