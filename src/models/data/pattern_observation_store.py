"""Observation-only aggregation of structural event patterns."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from ..pattern_identity import PatternIdentity


@dataclass
class PatternObservation:
    word: str
    yield_type: str
    preposition: Optional[str]
    construction: str = "verbal"
    correct: int = 0
    total: int = 0
    pair_precision: float = 0.0
    examples: list[str] = field(default_factory=list)
    evidence_class: str = "structural"
    review_status: str = "observation_only"
    source: str = "wide_net_run"

    def derived_shape(self) -> str:
        return "AA" if self.yield_type.upper() == "AA" else "AB"

    def _identity(self) -> PatternIdentity:
        return PatternIdentity(
            evidence_class=self.evidence_class,
            shape=self.derived_shape(),
            predicate_concept=self.word,
            case_realization=self.preposition or "",
            construction=self.construction or "verbal",
        )

    def canonical_pattern_key(self) -> str:
        return self._identity().canonical_key

    def semantic_pattern_key(self) -> str:
        return self._identity().semantic_key

    def to_dict(self) -> dict:
        return {
            "pattern_key": self.canonical_pattern_key(),
            "semantic_pattern_key": self.semantic_pattern_key(),
            "evidence_class": self.evidence_class,
            "shape": self.derived_shape(),
            "predicate_concept": self.word,
            "case_realization": self.preposition,
            "construction": self.construction,
            "yield_type": self.yield_type,
            "observation_unit": "projected_pair",
            "correct": self.correct,
            "total": self.total,
            "agreement": round(self.pair_precision, 4),
            "review_status": self.review_status,
            "source": self.source,
            "examples": list(self.examples),
        }


class PatternObservationStore:
    """Aggregate projected-pair agreement without providing inference APIs."""

    def __init__(self) -> None:
        self._by_lemma: dict[str, list[PatternObservation]] = {}

    def record_observation(
        self,
        word: str,
        preposition: Optional[str],
        yield_type: str,
        pred_pairs: set,
        gold_pairs: set | None,
        example: str = "",
        construction: str = "",
    ) -> None:
        if not pred_pairs:
            return
        lemma = str(word).strip().lower()
        prep = str(preposition).strip().lower() if preposition else None
        construction = str(construction or "verbal").strip().lower()
        yield_type = str(yield_type).strip().upper()
        key = (prep, yield_type, construction)

        existing = next(
            (
                item for item in self._by_lemma.get(lemma, ())
                if (item.preposition, item.yield_type, item.construction) == key
            ),
            None,
        )
        if existing is None:
            existing = PatternObservation(
                word=lemma,
                preposition=prep,
                yield_type=yield_type,
                construction=construction,
            )
            self._by_lemma.setdefault(lemma, []).append(existing)

        if gold_pairs is not None:
            predicted = {tuple(sorted(map(int, pair))) for pair in pred_pairs}
            gold = {tuple(sorted(map(int, pair))) for pair in gold_pairs}
            existing.total += len(predicted)
            existing.correct += len(predicted & gold)
            existing.pair_precision = (
                existing.correct / existing.total if existing.total else 0.0
            )
        if example and example not in existing.examples and len(existing.examples) < 3:
            existing.examples.append(example)

    def entries(self) -> tuple[PatternObservation, ...]:
        return tuple(
            item
            for lemma in sorted(self._by_lemma)
            for item in sorted(
                self._by_lemma[lemma],
                key=lambda row: row.canonical_pattern_key(),
            )
        )

    def has_patterns(self) -> bool:
        return bool(self._by_lemma)

    def count_patterns(self) -> int:
        return sum(len(rows) for rows in self._by_lemma.values())

    def to_yaml(self, path: Path, meta: dict | None = None) -> None:
        payload = {
            "schema": "pattern_observations_v1",
            "meta": dict(meta or {}),
            "observations": [entry.to_dict() for entry in self.entries()],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


__all__ = ["PatternObservation", "PatternObservationStore"]
