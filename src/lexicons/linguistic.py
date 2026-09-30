"""Corpus-independent lexical facts used during semantic extraction."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.system_rules import rule_frozenset


@dataclass(frozen=True)
class LinguisticLexicon:
    sets: dict[str, frozenset[str]]
    mappings: dict[str, dict[str, str]]

    def matches(self, category: str, values, *, all_values: bool = False) -> bool:
        vocabulary = self.sets.get(category, frozenset())
        normalized = {str(value or "").strip().lower() for value in values}
        return (
            all(value in vocabulary for value in normalized)
            if all_values else any(value in vocabulary for value in normalized)
        )

    def map(self, category: str, value: str) -> str | None:
        return self.mappings.get(category, {}).get(str(value or "").strip().lower())


def load_linguistic_lexicon(path: Path | str | None = None) -> LinguisticLexicon:
    base = Path(path) if path is not None else Path(__file__).resolve().parents[2] / "data" / "lexicons"
    sets = {
        "linking_verb": rule_frozenset("coordination.linking_verbs"),
        "conj_alternative": rule_frozenset("coordination.alternative_markers"),
        "conj_negation": rule_frozenset("coordination.negation_markers"),
        "pronoun_referential": rule_frozenset("reference.pronoun_referential_forms"),
        "pronoun_ignored": rule_frozenset("reference.pronoun_ignored_forms"),
        "negation": _read_words(base / "negation.txt"),
        "assertion_uncertainty": _read_words(base / "assertion_uncertainty.txt"),
        "relational_is_a": _read_words(base / "relational_is_a.txt"),
        "relational_has": _read_words(base / "relational_has.txt"),
        "relational_part_of": _read_words(base / "relational_part_of.txt"),
        "relational_nominal_alias": _read_words(base / "relational_nominal_alias.txt"),
        "relational_role_projection_trigger": _read_words(
            base / "relational_role_projection_trigger.txt"
        ),
        "reciprocal_event_nominal": _read_words(base / "reciprocal_event_nominal.txt"),
        "reciprocal_predicate": _read_words(base / "reciprocal_predicate.txt"),
        "subject_only_reciprocal_predicate": _read_words(
            base / "subject_only_reciprocal_predicate.txt"
        ),
        "collective_state_nominal": _read_words(base / "collective_state_nominal.txt"),
    }
    nominal_map: dict[str, str] = {}
    with open(base / "nominal_to_verbal.csv", newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) == 2 and row[0].strip() and row[1].strip():
                nominal_map[row[0].strip().lower()] = row[1].strip().lower()
    return LinguisticLexicon(sets=sets, mappings={"nominal_to_verbal": nominal_map})


def _read_words(path: Path) -> frozenset[str]:
    with open(path, encoding="utf-8") as handle:
        return frozenset(line.strip().lower() for line in handle if line.strip())


__all__ = ["LinguisticLexicon", "load_linguistic_lexicon"]
