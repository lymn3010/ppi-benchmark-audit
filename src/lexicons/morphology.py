"""Structural suffix hints for event extraction; exact mappings live in the lexicon."""

from src.system_rules import rule_frozenset, rule_tuple


# Agentive before process: role nouns (kinase, receptor) are not events.
AGENTIVE_NOUN_SUFFIXES = rule_tuple("morphology.agentive_noun_suffixes")
COMPLEX_NOUN_SUFFIXES = rule_tuple("morphology.complex_noun_suffixes")
NON_GROUP_COMPLEX_LEMMAS = rule_frozenset("morphology.non_group_complex_lemmas")
NON_AGENTIVE_LEMMAS = rule_frozenset("morphology.non_agentive_lemmas")

# Capacity nouns ("ability to bind"): the owner is the implicit actor.
CAPACITY_NOUNS = rule_frozenset("morphology.capacity_nouns")


def looks_complex(lemma: str) -> bool:
    """Return True for complex/aggregate-looking nouns."""
    if not lemma or len(lemma) < 4:
        return False
    lem = lemma.lower()
    if lem in NON_GROUP_COMPLEX_LEMMAS:
        return False
    return lem.endswith(COMPLEX_NOUN_SUFFIXES)


def looks_agentive(lemma: str) -> bool:
    """Return True for role-holder-looking nouns."""
    if not lemma or len(lemma) < 5:
        return False
    if lemma.lower() in NON_AGENTIVE_LEMMAS:
        return False
    if looks_complex(lemma):
        return False
    return lemma.endswith(AGENTIVE_NOUN_SUFFIXES)
