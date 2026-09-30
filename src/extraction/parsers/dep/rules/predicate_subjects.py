"""Subject attachment and inheritance in original evaluation order."""

from __future__ import annotations
from src.parsing.syntax import SyntaxNode


from src.system_rules import rule_frozenset

from ..rules import ACL_CONJ_SUBJECT_INHERITANCE_RULE_ID, inherit_subjects_from_acl_conj_head

from ..morphology import CAPACITY_NOUNS
from ..roles import CLAUSAL_INHERIT_DEPS, DEP_ACL, DEP_ADVCL, DEP_APPOS, DEP_COMPOUND, DEP_CONJ, DEP_DEP, DEP_DOBJ, DEP_MARK, DEP_NSUBJ, DEP_NSUBJPASS, DEP_XCOMP, PREP_TO, RELATIVE_CLAUSE_DEPS

_SUBJECT_INHERIT_POS = rule_frozenset("syntax.subject_inherit_pos")
_PARTICIPLE_XPOS = rule_frozenset("syntax.participle_xpos")
_NOMINAL_RELATION_COMPLEMENTS = rule_frozenset(
    "prepositions.nominal_relation_complements"
)
_EVIDENTIAL_SCOPE_PREDICATES = rule_frozenset(
    "event_composition.evidential_scope_predicates"
)


from ..node_helpers import children_by_dep_n, node_protein_indices, copula_predicate_nominal_subject_n, neg_flag_n
from ..roles import DEP_AUXPASS

def bind_predicate_subjects(node, parsed, subj_children, pass_subj_children, resources=None):
    node_head = parsed.head(node)
    acl_conj_subject_evidence: dict | None = None

    # Controlled verbs inherit the matrix subject unless the adjective is negated.
    _adj_head_negated = (
        node_head is not None
        and node_head.pos == "ADJ"
        and neg_flag_n(node_head, parsed)
    )
    if not subj_children and not pass_subj_children \
       and node.dep in CLAUSAL_INHERIT_DEPS \
       and node_head is not None and node_head.pos in _SUBJECT_INHERIT_POS \
       and not _adj_head_negated:
        head_dobj = children_by_dep_n(node_head, parsed, DEP_DOBJ)
        has_to_mark = any(
            m.dep == DEP_MARK and (m.text or "").lower() == PREP_TO
            for m in parsed.children(node)
        )
        if node.dep in (DEP_XCOMP, DEP_ADVCL) and head_dobj and has_to_mark:
            subj_children = head_dobj
        else:
            inherited_subj = children_by_dep_n(node_head, parsed, DEP_NSUBJ)
            inherited_pass = children_by_dep_n(node_head, parsed, DEP_NSUBJPASS)
            node_head_head = parsed.head(node_head)
            if not inherited_subj and not inherited_pass \
               and node_head.dep in CLAUSAL_INHERIT_DEPS \
               and node_head_head is not None \
               and node_head_head.pos in frozenset({"VERB", "ADJ", "NUM"}):
                inherited_subj = children_by_dep_n(
                    node_head_head, parsed, DEP_NSUBJ,
                )
                inherited_pass = children_by_dep_n(
                    node_head_head, parsed, DEP_NSUBJPASS,
                )
            subj_children = inherited_subj + inherited_pass

            # "X binds and is activated by Y": X is the patient of the auxpass conjunct.
            if (
                node.dep == DEP_CONJ
                and subj_children
                and children_by_dep_n(node, parsed, DEP_AUXPASS)
            ):
                pass_subj_children = subj_children
                subj_children = []

    if not subj_children and not pass_subj_children:
        acl_subject_rule_enabled = (
            resources is None
            or resources.rule_enabled(ACL_CONJ_SUBJECT_INHERITANCE_RULE_ID)
        )
        if acl_subject_rule_enabled:
            inherited_subj, inherited_pass, evidence = (
                inherit_subjects_from_acl_conj_head(node, parsed)
            )
            if inherited_subj or inherited_pass:
                subj_children = list(inherited_subj)
                pass_subj_children = list(inherited_pass)
                acl_conj_subject_evidence = evidence

    # Relative/reduced clauses inherit their modified noun as subject
    if node.dep in RELATIVE_CLAUSE_DEPS \
       and node_head is not None and node_head.pos in ("NOUN", "PROPN"):
        def _has_proteins(n: SyntaxNode) -> bool:
            if node_protein_indices(n):
                return True
            for c in parsed.children(n):

                # An appositive protein completes the antecedent ("a phosphatase, P3, which").
                if c.dep in (DEP_COMPOUND, DEP_APPOS) and node_protein_indices(c):
                    return True
            return False

        def _is_referential_placeholder(n: SyntaxNode) -> bool:
            key = (n.lemma or n.text or "").lower()
            return (
                not _has_proteins(n)
                and (
                    n.pos in ("PRON", "DET")
                    or key in {"which", "that", "who", "whom", "whose"}
                )
            )

        subj_is_placeholder = (
            len(subj_children) == 1
            and _is_referential_placeholder(subj_children[0])
        )
        # Passive relative clauses bind the antecedent as patient.
        pass_is_placeholder = (
            len(pass_subj_children) == 1
            and _is_referential_placeholder(pass_subj_children[0])
        )
        pron_only_subj = subj_is_placeholder or pass_is_placeholder \
            or (not subj_children and not pass_subj_children)
        if pron_only_subj:
            anchor = node_head

            safety = 0
            anchor_head = parsed.head(anchor)
            while not _has_proteins(anchor) \
                  and anchor_head is not None \
                  and anchor.dep in (DEP_APPOS, DEP_DEP, DEP_CONJ) \
                  and safety < 4:
                anchor = anchor_head
                anchor_head = parsed.head(anchor)
                safety += 1

            _anchor_lemma = (anchor.lemma or anchor.text or "").lower()
            if _anchor_lemma in CAPACITY_NOUNS:
                for _poss in parsed.children(anchor):
                    if _poss.dep == "nmod:poss" and node_protein_indices(_poss):
                        anchor = _poss
                        break

            # "P1 is a kinase that phosphorylates P2": recover P1 via the copula.
            if not _has_proteins(anchor):
                _cop_subj = copula_predicate_nominal_subject_n(anchor, parsed)
                if _cop_subj is not None:
                    anchor = _cop_subj

            is_passive_reduction = (
                node.dep == DEP_ACL
                and node.raw_pos in ("VBN",)
                and not children_by_dep_n(node, parsed, DEP_NSUBJ)
            )
            if is_passive_reduction or pass_is_placeholder:
                pass_subj_children = [anchor]
                subj_children = []
            else:
                subj_children = [anchor]

    return subj_children, pass_subj_children, acl_conj_subject_evidence
