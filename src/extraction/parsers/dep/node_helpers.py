"""Read-only structural helpers over ``SyntaxNode`` / ``ParsedSentence``."""
from __future__ import annotations

import re

from typing import Optional

from src.models.candidate import Span
from src.parsing.syntax import ParsedSentence, SyntaxNode

from .roles import (
    ACCOMPANIMENT_CASE_PHRASES,
    CASE_MODIFIER_DEPS,
    COORD_MARKER_DEPS,
    DEP_ADVMOD,
    DEP_CASE,
    DEP_CONJ,
    DEP_COP,
    DEP_NEG,
    DEP_NMOD,
    DEP_NSUBJ,
    DEP_NSUBJPASS,
    DESCRIPTOR_DEPS,
    PREP_TOGETHER_WITH,
    PREP_WITH,
)


# Basic accessors

def node_protein_indices(node: SyntaxNode) -> tuple[int, ...]:
    """Return protein target indices for node (replaces token_targets)."""
    return node.protein_indices


def hyphen_adjacent_protein_children_n(
    descriptor: SyntaxNode, parsed: ParsedSentence,
) -> tuple[SyntaxNode, ...]:
    """Protein children joined to ``descriptor`` only by a surface hyphen.

    Covers split-hyphen and glued-hyphen tokenizations; empty if ``descriptor`` has a protein.
    """
    if descriptor.protein_indices:
        return ()
    out: list[SyntaxNode] = []
    for child in parsed.children(descriptor):
        if not child.protein_indices:
            continue
        if child.char_start <= descriptor.char_start:
            lo, hi = child, descriptor
        else:
            lo, hi = descriptor, child
        gap = parsed.text[lo.char_end:hi.char_start]
        if gap == "-":
            out.append(child)  # spaCy-style: hyphen is its own punct token between the spans
        elif gap == "" and ((lo.text or "").endswith("-") or (hi.text or "").startswith("-")):
            # Stanza glues the hyphen onto a span, leaving an empty gap.
            out.append(child)
    return tuple(out)


def parenthetical_alias_antecedent_n(
    node: SyntaxNode, parsed: ParsedSentence,
) -> SyntaxNode | None:
    """Antecedent ``PROTEIN_A`` when ``node`` is exactly ``PROTEIN_A ( node )``, else None.

    Surface punctuation test; the result is identity (IS-A), not ownership.
    """
    if not node.protein_indices:
        return None
    # Positional order tolerates non-contiguous indices.
    nodes = sorted(parsed.nodes, key=lambda n: n.i)
    pos_map = {n.i: idx for idx, n in enumerate(nodes)}
    pos = pos_map.get(node.i)
    if pos is None or pos < 2 or pos + 1 >= len(nodes):
        return None
    open_paren = nodes[pos - 1]
    antecedent = nodes[pos - 2]
    close_paren = nodes[pos + 1]
    if open_paren.text != "(" or close_paren.text != ")":
        return None
    if not antecedent.protein_indices:
        return None
    if antecedent.protein_indices == node.protein_indices:
        return None
    return antecedent


# Literal ``PROTEIN<digits>-`` prefix: Stanza merged marker and participle.
_MERGED_PROTEIN_PREFIX_RE = re.compile(r"^PROTEIN\d+-(.+)$")


def merged_participle_clean_surface_n(node: SyntaxNode) -> Optional[str]:
    """Participle surface without its ``PROTEINn-`` prefix, or None.

    Uses surface text because Stanza lemmas of merged tokens are unreliable.
    """
    if not node.protein_indices:
        return None
    m = _MERGED_PROTEIN_PREFIX_RE.match(node.text or "")
    if m is None:
        return None
    remainder = m.group(1)
    if not remainder:
        return None
    return remainder.lower()


def merged_participle_clean_lemma_n(node: SyntaxNode) -> Optional[str]:
    """Verb lemma of a merged ``PROTEINn-participle`` token.

    Nominal map, then LemmInflect; retries the last segment of multi-hyphen tokens.
    """
    surface = merged_participle_clean_surface_n(node)
    if surface is None:
        return None
    from src.models.candidate._nominal_canon import mapped_verbal_from_surface

    final_surface = surface.rsplit("-", 1)[-1]
    mapped = mapped_verbal_from_surface(final_surface)
    if mapped:
        return mapped
    try:
        import lemminflect
        forms = tuple(lemminflect.getLemma(final_surface, upos="VERB") or ())
    except Exception:
        forms = ()
    return forms[0].lower() if forms else final_surface


def copula_predicate_nominal_subject_n(
    anchor: SyntaxNode, parsed: ParsedSentence,
) -> SyntaxNode | None:
    """Protein subject of a copular predicate nominal (``P1 is a kinase that ...``), or None."""
    if node_protein_indices(anchor):
        return None
    if not any(c.dep == DEP_COP for c in parsed.children(anchor)):
        return None
    for child in parsed.children(anchor):
        if child.dep in (DEP_NSUBJ, DEP_NSUBJPASS) and node_protein_indices(child):
            return child
    return None


def match_keys_for_node(node: SyntaxNode) -> set[str]:
    """Union of lemma, nominal verb form, and surface form for lexical matching.

    Mirrors match_keys_for_token from tokens.py.
    """
    keys: set[str] = set()
    if node.lemma:
        keys.add(node.lemma.lower())
    if node.text:
        keys.add(node.text.lower())
    if node.verb_form:
        keys.add(node.verb_form.lower())
    return keys


# Child traversal

def children_by_dep_n(
    node: SyntaxNode, parsed: ParsedSentence, *deps: str,
) -> list[SyntaxNode]:
    """Children whose dep label is in deps (replaces children_by_dep)."""
    return [c for c in parsed.children(node) if c.dep in deps]


def children_by_role_n(
    node: SyntaxNode, parsed: ParsedSentence, deps: frozenset[str],
) -> list[SyntaxNode]:
    """Children whose dep label belongs to a normalized role set."""
    return [c for c in parsed.children(node) if c.dep in deps]


# Conjunction chain helpers

def conj_chain_n(
    node: SyntaxNode, parsed: ParsedSentence, skip_negated: bool = True,
) -> list[SyntaxNode]:
    """Return node plus all conj descendants; ``skip_negated`` drops members with ``neg``."""
    def _is_negated(n: SyntaxNode) -> bool:
        return any(c.dep == DEP_NEG for c in parsed.children(n))

    out: list[SyntaxNode] = []
    seen: set[int] = {node.i}
    if not (skip_negated and _is_negated(node)):
        out.append(node)
    stack: list[SyntaxNode] = [node]
    while stack:
        t = stack.pop()
        for c in parsed.children(t):
            if c.dep == DEP_CONJ and c.i not in seen:
                seen.add(c.i)
                stack.append(c)
                if not (skip_negated and _is_negated(c)):
                    out.append(c)
    return out


def conj_root_n(node: SyntaxNode, parsed: ParsedSentence) -> SyntaxNode:
    """Return the highest head reached by walking conj links upward."""
    root = node
    seen = {node.i}
    while True:
        if root.dep != DEP_CONJ:
            break
        head = parsed.head(root)
        if head is None or head.i in seen:
            break
        seen.add(head.i)
        root = head
    return root


def _conj_chain_members_with_cc_n(
    head: SyntaxNode, parsed: ParsedSentence,
) -> list[tuple[SyntaxNode, str]]:
    """Return [(member, incoming_cc_lemma)] for every conj descendant of head.

    Mirrors _conj_chain_members_with_cc from tokens.py.
    """
    members: list[tuple[SyntaxNode, str]] = [(head, "")]
    seen: set[int] = {head.i}
    stack: list[SyntaxNode] = [head]
    while stack:
        parent = stack.pop()
        for child in parsed.children(parent):
            if child.dep != DEP_CONJ or child.i in seen:
                continue
            seen.add(child.i)
            stack.append(child)
            cc_lem = _incoming_coord_marker_lemma_n(parent, child, parsed)
            members.append((child, cc_lem))
    members.sort(key=lambda m: m[0].i)
    return members


def _incoming_coord_marker_lemma_n(
    parent: SyntaxNode,
    child: SyntaxNode,
    parsed: ParsedSentence,
) -> str:
    """Coordination marker of a conjunct, whether ``cc`` attaches to head or conjunct."""
    lo = min(parent.i, child.i)
    hi = max(parent.i, child.i)
    candidates: list[SyntaxNode] = []
    for owner in (parent, child):
        for marker in parsed.children(owner):
            if marker.dep not in COORD_MARKER_DEPS:
                continue
            if lo < marker.i < hi:
                candidates.append(marker)
    if not candidates:
        return ""
    marker = max(candidates, key=lambda node: node.i)
    return (marker.lemma or marker.text or "").lower()


from src.system_rules import rule_frozenset

_ALTERNATIVE_CC_LEMMAS = rule_frozenset("coordination.alternative_markers")
_CONJUNCT_EXCLUDE_LEMMAS = rule_frozenset("coordination.conjunct_exclusion_markers")
_CONTRASTIVE_CONJ_LEMMAS = rule_frozenset("coordination.contrastive_markers")
_CORRELATIVE_NEG_LEMMAS = rule_frozenset("coordination.correlative_negative_markers")
_SAME_SIDE_CC_LEMMAS = rule_frozenset("coordination.same_side_markers")


def conj_chain_has_alternative_n(head: SyntaxNode, parsed: ParsedSentence) -> bool:
    """True if the conj chain rooted at head contains any 'or'/'/' cc."""
    seen: set[int] = {head.i}
    stack: list[SyntaxNode] = [head]
    while stack:
        parent = stack.pop()
        for child in parsed.children(parent):
            if child.dep != DEP_CONJ or child.i in seen:
                continue
            seen.add(child.i)
            stack.append(child)
            if _incoming_coord_marker_lemma_n(parent, child, parsed) in _ALTERNATIVE_CC_LEMMAS:
                return True
    return False


def conj_chain_aa_anchor_n(
    head: SyntaxNode, parsed: ParsedSentence,
) -> Optional[SyntaxNode]:
    """Pick the AA-anchor of an alternative-coordination chain.

    Mirrors conj_chain_aa_anchor from tokens.py.
    """
    members = _conj_chain_members_with_cc_n(head, parsed)
    if len(members) < 2:
        return None
    side2_start: Optional[int] = None
    for i in range(1, len(members)):
        cc = members[i][1]
        if cc in _SAME_SIDE_CC_LEMMAS:
            side2_start = i
            break
    if side2_start is None:
        return None
    side1 = [tok for tok, _ in members[:side2_start]]
    side2 = [tok for tok, _ in members[side2_start:]]
    if len(side1) == 1:
        return side1[0]
    if len(side2) == 1:
        return side2[0]
    return side1[0]


def _conj_negation_excluded_direct_n(node: SyntaxNode, parsed: ParsedSentence) -> bool:
    """Checks only node-local exclusion cues: no sibling propagation."""
    for c in parsed.children(node):
        if c.dep in COORD_MARKER_DEPS and (c.lemma or c.text or "").lower() in _CONJUNCT_EXCLUDE_LEMMAS:
            return True
    parent = parsed.head(node)
    if parent is None:
        return False
    lo = min(parent.i, node.i)
    hi = max(parent.i, node.i)
    for c in parsed.children(parent):
        if c.dep not in COORD_MARKER_DEPS:
            continue
        if not (lo < c.i < hi):
            continue
        if (c.lemma or c.text or "").lower() in _CONJUNCT_EXCLUDE_LEMMAS:
            return True
    return False


def conj_negation_excluded_n(node: SyntaxNode, parsed: ParsedSentence) -> bool:
    """True if node is in a ``but not`` excluded conjunct group.

    Checks the node, preceding conj siblings and the parent conjunct.
    """
    if node.dep != DEP_CONJ:
        return False
    # Check 1 & 2: node-local exclusion cue (direct child or parent cc between).
    if _conj_negation_excluded_direct_n(node, parsed):
        return True
    parent = parsed.head(node)
    if parent is None:
        return False
    # Check 3: sibling propagation: a preceding conj sibling is directly excluded.
    for sibling in parsed.children(parent):
        if sibling.dep != DEP_CONJ:
            continue
        if sibling.i >= node.i:
            continue
        if _conj_negation_excluded_direct_n(sibling, parsed):
            return True
    # Check 4: the parent conjunct is itself excluded (recursive).
    if parent.dep == DEP_CONJ and conj_negation_excluded_n(parent, parsed):
        return True
    return False


def conj_contrast_resets_head_negation_n(
    node: SyntaxNode, parsed: ParsedSentence,
) -> bool:
    """True when node is a contrastive conj whose head's negation should not be inherited."""
    if node.dep != DEP_CONJ:
        return False
    parent = parsed.head(node)
    if parent is None:
        return False
    lo = min(parent.i, node.i)
    hi = max(parent.i, node.i)
    for owner in (parent, node):
        for c in parsed.children(owner):
            if c.dep not in COORD_MARKER_DEPS:
                continue
            if not (lo < c.i < hi):
                continue
            if (c.lemma or c.text or "").lower() in _CONTRASTIVE_CONJ_LEMMAS:
                return True
    return False


def correlative_negation_head_n(node: SyntaxNode, parsed: ParsedSentence) -> bool:
    """True if node has a cc:preconj/preconj/cc child whose lemma is 'neither'."""
    for c in parsed.children(node):
        if c.dep not in COORD_MARKER_DEPS:
            continue
        if (c.lemma or c.text or "").lower() in _CORRELATIVE_NEG_LEMMAS:
            return True
    return False


# Case / preposition helpers

def case_token_n(
    nmod_node: SyntaxNode, parsed: ParsedSentence,
) -> Optional[SyntaxNode]:
    """Return the case child of nmod_node, or None."""
    for c in parsed.children(nmod_node):
        if c.dep == DEP_CASE:
            return c
    return None


def case_lemma_n(nmod_node: SyntaxNode, parsed: ParsedSentence) -> str:
    """Return lowercased preposition of an nmod node via its case child."""
    case = case_token_n(nmod_node, parsed)
    if case is None:
        return ""
    return (case.lemma or case.text or "").lower()


def case_phrase_n(nmod_node: SyntaxNode, parsed: ParsedSentence) -> str:
    """Return a normalized case phrase for multi-token case markers."""
    case_nodes = [
        child for child in parsed.children(nmod_node)
        if child.dep == DEP_CASE
    ]
    if not case_nodes:
        return ""
    phrase_nodes = list(case_nodes)
    for case in case_nodes:
        phrase_nodes.extend(
            child for child in parsed.children(case)
            if child.dep in CASE_MODIFIER_DEPS
        )
    phrase_nodes = sorted({node.i: node for node in phrase_nodes}.values(), key=lambda n: n.i)
    phrase = " ".join(
        (node.lemma or node.text or "").lower()
        for node in phrase_nodes
        if (node.lemma or node.text or "")
    )
    if phrase:
        # Stanza CRAFT: PROTEIN0 --advmod--> together --nmod:with--> PROTEIN1
        parent = parsed.head(nmod_node)
        if phrase == PREP_WITH and parent is not None:
            parent_lemma = (parent.lemma or parent.text or "").lower()
            if parent_lemma == "together":
                return PREP_TOGETHER_WITH
        return phrase

    case = case_nodes[0]
    base = (case.lemma or case.text or "").lower()
    return base


def is_accompaniment_nmod_n(nmod_node: SyntaxNode, parsed: ParsedSentence) -> bool:
    """True when an nmod expresses co-participation rather than possession."""
    return case_phrase_n(nmod_node, parsed) in ACCOMPANIMENT_CASE_PHRASES


def accompaniment_nmods_n(head: SyntaxNode, parsed: ParsedSentence) -> list[SyntaxNode]:
    """Accompaniment nmods (``together with``) in UD and Stanza CRAFT shapes."""
    out: list[SyntaxNode] = []
    seen: set[int] = set()

    def add_if_accompaniment(node: SyntaxNode) -> None:
        if node.i in seen:
            return
        if is_accompaniment_nmod_n(node, parsed):
            out.append(node)
            seen.add(node.i)

    for child in parsed.children(head):
        if child.dep == DEP_NMOD:
            add_if_accompaniment(child)
            continue
        child_lemma = (child.lemma or child.text or "").lower()
        if child.dep == DEP_ADVMOD and child_lemma == "together":
            for grandchild in parsed.children(child):
                if grandchild.dep == DEP_NMOD:
                    add_if_accompaniment(grandchild)
    return out


# Negation helpers

def neg_flag_n(node: SyntaxNode, parsed: ParsedSentence) -> bool:
    """True if the node carries a neg or advmod=not child."""
    for c in parsed.children(node):
        if c.dep == DEP_NEG:
            return True
        if c.dep == DEP_ADVMOD and (c.lemma or "").lower() == "not":
            return True
    return False


# Descriptor helpers

def descriptor_tokens_n(
    node: SyntaxNode, parsed: ParsedSentence,
    *, inherit_from_conj_root: bool = True,
) -> list[SyntaxNode]:
    """Descriptor (amod) children for a lexical argument node.

    Mirrors descriptor_tokens from tokens.py.
    """
    local = children_by_role_n(node, parsed, DESCRIPTOR_DEPS)
    if local or not inherit_from_conj_root:
        return local
    root = conj_root_n(node, parsed)
    if root.i == node.i:
        return local
    return children_by_role_n(root, parsed, DESCRIPTOR_DEPS)


# Span builders

def single_token_span_n(
    node: SyntaxNode,
    *,
    is_negated: bool = False,
    assertion_status: str = "",
    force_nominal: bool = False,
) -> Span:
    """Build a single-token Span from a SyntaxNode.

    Mirrors single_token_span from tokens.py.
    """
    char_start = node.char_start
    char_end = node.char_end
    is_nom = node.is_nominalized or force_nominal
    vf = node.verb_form or ""
    match_keys = match_keys_for_node(node)
    return Span(
        tokens=(node.text,),
        lemmas=((node.lemma or node.text).lower(),),
        match_keys=tuple(sorted(match_keys)),
        protein_indices=node.protein_indices,
        char_offsets=((char_start, char_end),),
        token_lemmas=((node.lemma or node.text).lower(),),
        head_node_index=node.i,
        pos=node.pos or "",
        dep=node.dep or "",
        is_negated=is_negated,
        assertion_status=assertion_status or ("negated" if is_negated else "asserted"),
        is_nominalized=is_nom,
        verb_form=vf,
        node_indices=(node.i,),
        parser_dep=node.raw_dep or node.dep or "",
        parser_pos=node.raw_pos or node.pos or "",
        parser_features=node.raw_feats,
        parser_misc=node.raw_misc,
        enhanced_heads=node.enhanced_heads,
        reference_kinds=node.inherited_reference_kinds,
    )


def compound_span_n(
    head: SyntaxNode, members: list[SyntaxNode],
) -> Span:
    """Build a Span covering a head plus additional compound members.

    Mirrors compound_span from tokens.py.
    """
    all_members = sorted([head] + members, key=lambda n: n.char_start)
    tokens = tuple(n.text for n in all_members)
    token_lemmas = tuple((n.lemma or n.text).lower() for n in all_members)
    lemmas: set[str] = set()
    match_keys: set[str] = set()
    offsets: list[tuple[int, int]] = []
    proteins: set[int] = set()
    reference_kinds: set[str] = set()
    for n in all_members:
        if n.lemma:
            lemmas.add(n.lemma.lower())
        match_keys |= match_keys_for_node(n)
        offsets.append((n.char_start, n.char_end))
        proteins.update(n.protein_indices)
        reference_kinds.update(n.inherited_reference_kinds)
    return Span(
        tokens=tokens,
        lemmas=tuple(sorted(lemmas)),
        match_keys=tuple(sorted(match_keys)),
        protein_indices=tuple(sorted(proteins)),
        char_offsets=tuple(offsets),
        token_lemmas=token_lemmas,
        head_node_index=head.i,
        pos=head.pos or "",
        dep=head.dep or "",
        is_negated=False,
        is_nominalized=head.is_nominalized,
        verb_form=head.verb_form or "",
        node_indices=tuple(n.i for n in all_members),
        parser_dep=head.raw_dep or head.dep or "",
        parser_pos=head.raw_pos or head.pos or "",
        parser_features=head.raw_feats,
        parser_misc=head.raw_misc,
        enhanced_heads=head.enhanced_heads,
        reference_kinds=tuple(sorted(reference_kinds)),
    )
