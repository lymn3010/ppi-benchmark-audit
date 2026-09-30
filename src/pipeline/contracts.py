"""Input metadata shared by reader stages and record assembly."""
from typing import TypedDict


class EntryData(TypedDict, total=False):
    id: str
    text: str
    gold_pairs: list[tuple[int, int]] | None
    proteins: list[str]
