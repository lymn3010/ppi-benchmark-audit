"""Shared canonical and semantic identity for every pattern-bearing model."""
from __future__ import annotations

from dataclasses import dataclass

from .semantic_vocabulary import Construction


@dataclass(frozen=True)
class PatternIdentity:
    """Canonical and semantic pattern identity derived from a model's shape."""

    evidence_class: str
    shape: str
    predicate_concept: str
    case_realization: str = ""
    construction: str = "verbal"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "construction",
            Construction.normalize(self.construction, default=Construction.VERBAL.value),
        )

    @property
    def canonical_key(self) -> str:
        return "|".join((
            self.evidence_class or "structural",
            self.shape or "AB",
            (self.predicate_concept or "").lower(),
            self.case_realization or "_",
            (self.construction or "verbal").lower(),
        ))

    @property
    def semantic_key(self) -> str:
        evidence_class = self.evidence_class or "structural"
        shape = self.shape or "AB"
        lemma = (self.predicate_concept or "").lower()
        if shape in ("AB", "AB_NMOD", "AA"):
            return "|".join((evidence_class, lemma))
        return "|".join((
            evidence_class,
            shape,
            lemma,
            self.case_realization or "_",
        ))


__all__ = ["PatternIdentity"]
