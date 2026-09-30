"""Normalized dependency-role and case constants for DEP rules."""

from src.system_rules import rule_frozenset, system_rule

_LABELS = system_rule("dependency.labels")

DEP_ACL = _LABELS["acl"]
DEP_ACL_RELCL = _LABELS["acl_relcl"]
DEP_ADVCL = _LABELS["advcl"]
DEP_ADVMOD = _LABELS["advmod"]
DEP_AMOD = _LABELS["amod"]
DEP_APPOS = _LABELS["appos"]
DEP_ATTR = _LABELS["attr"]
DEP_AUX = _LABELS["aux"]
DEP_AUXPASS = _LABELS["auxpass"]
DEP_CASE = _LABELS["case"]
DEP_CCOMP = _LABELS["ccomp"]
DEP_CC = _LABELS["cc"]
DEP_CC_PRECONJ = _LABELS["cc_preconj"]
DEP_COMPOUND = _LABELS["compound"]
DEP_CONJ = _LABELS["conj"]
DEP_COP = _LABELS["cop"]
DEP_DATIVE = _LABELS["dative"]
DEP_DEP = _LABELS["dep"]
DEP_DET = _LABELS["det"]
DEP_DOBJ = _LABELS["dobj"]
DEP_FIXED = _LABELS["fixed"]
DEP_MARK = _LABELS["mark"]
DEP_MWE = _LABELS["mwe"]
DEP_NEG = _LABELS["neg"]
DEP_NMOD = _LABELS["nmod"]
DEP_NMOD_NPMOD = _LABELS["nmod_npmod"]
DEP_NMOD_OF = _LABELS["nmod_of"]
DEP_NSUBJ = _LABELS["nsubj"]
DEP_NSUBJPASS = _LABELS["nsubjpass"]
DEP_PRECONJ = _LABELS["preconj"]
DEP_RELATED_RELCL = _LABELS["related_relcl"]
DEP_ROOT = _LABELS["root"]
DEP_XCOMP = _LABELS["xcomp"]


# Role categories used by multiple extraction paths.
COORDINATION_DEPS = rule_frozenset("dependency.categories.coordination")
COORD_MARKER_DEPS = rule_frozenset("dependency.categories.coordination_markers")
SAME_NP_PROTEIN_DEPS = rule_frozenset("dependency.categories.same_np_protein")
SAME_NP_OR_LOOSE_DEPS = rule_frozenset("dependency.categories.same_np_or_loose")
SAME_NP_TRAVERSAL_DEPS = rule_frozenset("dependency.categories.same_np_traversal")
NOMINAL_SOURCE_DEPS = rule_frozenset("dependency.categories.nominal_sources")
CLAUSAL_INHERIT_DEPS = rule_frozenset("dependency.categories.clausal_inherit")
RELATIVE_CLAUSE_DEPS = rule_frozenset("dependency.categories.relative_clause")
NEG_SCOPE_DEPS = rule_frozenset("dependency.categories.neg_scope")
NEG_NOMINAL_MOD_DEPS = rule_frozenset("dependency.categories.neg_nominal_mod")
PREDICATE_NEG_DEPS = rule_frozenset("dependency.categories.predicate_neg")
DESCRIPTOR_DEPS = rule_frozenset("dependency.categories.descriptors")
CASE_MODIFIER_DEPS = rule_frozenset("dependency.categories.case_modifiers")


# Typed case/preposition categories (not suppression lists).
_PREPOSITIONS = system_rule("prepositions.named")
PREP_BY = _PREPOSITIONS["by"]
PREP_OF = _PREPOSITIONS["of"]
PREP_AS = _PREPOSITIONS["as"]
PREP_TO = _PREPOSITIONS["to"]
PREP_WITH = _PREPOSITIONS["with"]
PREP_BETWEEN = _PREPOSITIONS["between"]
PREP_AMONG = _PREPOSITIONS["among"]
PREP_TOGETHER_WITH = _PREPOSITIONS["together_with"]
PREP_LIKE = _PREPOSITIONS["like"]

BINARY_CASE_MARKERS = rule_frozenset("prepositions.binary")
CONTEXTUAL_CASE_MARKERS = rule_frozenset("prepositions.contextual")
ACCOMPANIMENT_CASE_PHRASES = rule_frozenset("prepositions.accompaniment_phrases")

