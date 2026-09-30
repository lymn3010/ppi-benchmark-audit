"""Subject-inheritance rules for gapped predicate dependents."""
from __future__ import annotations

from src.parsing.syntax import ParsedSentence, SyntaxNode

from ..roles import DEP_ACL, DEP_ACL_RELCL, DEP_CONJ, DEP_NSUBJ, DEP_NSUBJPASS


ACL_CONJ_SUBJECT_INHERITANCE_RULE_ID = (
    "extraction.dep_acl_conj_subject_inheritance@1"
)
ACL_DEPS = frozenset({DEP_ACL, DEP_ACL_RELCL, "relcl"})


def inherit_subjects_from_acl_conj_head(
    predicate: SyntaxNode,
    parsed: ParsedSentence,
) -> tuple[tuple[SyntaxNode, ...], tuple[SyntaxNode, ...], dict]:
    """Inherit a nominal head's subject through ``acl`` + verbal ``conj``."""
    from ..node_helpers import children_by_dep_n

    if predicate.dep != DEP_CONJ:
        return (), (), {}
    acl_head = parsed.head(predicate)
    if acl_head is None or acl_head.pos != "VERB" or acl_head.dep not in ACL_DEPS:
        return (), (), {}
    nominal_head = parsed.head(acl_head)
    if nominal_head is None or nominal_head.pos not in {"NOUN", "PROPN"}:
        return (), (), {}

    inherited_subj = tuple(children_by_dep_n(nominal_head, parsed, DEP_NSUBJ))
    inherited_pass = tuple(children_by_dep_n(nominal_head, parsed, DEP_NSUBJPASS))
    if not inherited_subj and not inherited_pass:
        return (), (), {}

    evidence = {
        "predicate_head": predicate.i,
        "acl_head": acl_head.i,
        "nominal_head": nominal_head.i,
        "active_subject_heads": [node.i for node in inherited_subj],
        "passive_subject_heads": [node.i for node in inherited_pass],
    }
    return inherited_subj, inherited_pass, evidence


__all__ = [
    "ACL_CONJ_SUBJECT_INHERITANCE_RULE_ID",
    "inherit_subjects_from_acl_conj_head",
]
