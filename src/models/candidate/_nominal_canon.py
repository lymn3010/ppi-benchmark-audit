"""Canonicalize nominalized predicates via ``data/lexicons/nominal_to_verbal.csv``."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


_CSV_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "lexicons" / "nominal_to_verbal.csv"
)


def _load_mapping(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                nominal = row[0].strip().lower()
                verbal = row[1].strip().lower()
                if nominal and verbal:
                    out[nominal] = verbal
    return out


_NOMINAL_TO_VERBAL: dict[str, str] = _load_mapping(_CSV_PATH)


def mapped_verbal_form(lemma: str) -> str:
    """Return the reviewed exact verbal form, or ``""`` when unmapped."""
    if not lemma:
        return ""
    return _NOMINAL_TO_VERBAL.get(lemma.lower(), "")


def mapped_verbal_from_surface(surface: str) -> str:
    """Resolve a nominal or regular participle of a verb already in the map."""
    low = (surface or "").lower()
    mapped = mapped_verbal_form(low)
    if mapped:
        return mapped
    for verb in set(_NOMINAL_TO_VERBAL.values()):
        forms = {verb}
        forms.add(f"{verb}d" if verb.endswith("e") else f"{verb}ed")
        forms.add(f"{verb[:-1]}ing" if verb.endswith("e") else f"{verb}ing")
        if verb.endswith("y"):
            forms.add(f"{verb[:-1]}ied")
        if low in forms:
            return verb
    return ""


def canonicalize_predicate_lemma(lemma: str) -> str:
    """Map a lemma to its canonical verb; unmapped input is lowercased, empty gives ""."""
    if not lemma:
        return ""
    low = lemma.lower()
    return mapped_verbal_form(low) or low


def canonicalize_first(lemmas: Iterable[str] | None) -> str:
    """Convenience wrapper: take the first non-empty lemma and
    canonicalize it. Returns "" when the iterable is empty / all blank.
    """
    if not lemmas:
        return ""
    for lemma in lemmas:
        if lemma:
            return canonicalize_predicate_lemma(lemma)
    return ""
