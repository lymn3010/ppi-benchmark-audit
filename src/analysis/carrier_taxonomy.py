"""Classify intermediate heads for carrier analysis (audit only).

Sources: data/system_rules.yaml, lexpath_audit.
"""
from __future__ import annotations

import re

from src.system_rules import rule_frozenset


_TOKENIZATION_ARTIFACT_RE = re.compile(r"protein\d.*protein\d")

_CATEGORY_RULES = (
    ("role_entity_carrier", "lexpath_audit.carrier_effect_role_entity_heads"),
    ("part_entity_carrier", "lexpath_audit.carrier_effect_part_entity_heads"),
    ("entity_shifting_carrier", "lexpath_audit.carrier_effect_entity_shift_heads"),
    ("relation_nominal", "lexpath_audit.carrier_effect_relation_heads"),
    ("process_layer", "lexpath_audit.carrier_effect_process_heads"),
    ("attribute_layer", "lexpath_audit.carrier_effect_attribute_heads"),
    ("metric_or_expression_layer", "lexpath_audit.carrier_effect_metric_heads"),
    ("pathway_location_layer", "lexpath_audit.carrier_effect_pathway_heads"),
    ("assay_reagent_layer", "lexpath_audit.carrier_effect_assay_heads"),
    ("expression_variant_layer", "lexpath_audit.carrier_effect_expression_variant_heads"),
    ("generic_entity_head", "lexpath_audit.carrier_effect_generic_head"),
)
# Heads that can make the marked protein not the endpoint itself.
_ENTITY_BOUNDARY_CLASSES = frozenset(
    {"role_entity_carrier", "part_entity_carrier", "entity_shifting_carrier"}
)


def carrier_head_classes(head: str) -> tuple[str, ...]:
    """Return all diagnostic classes for one carrier head (multi-label by design)."""
    lemma = _clean_head(head)
    if lemma in {"", "_", "none"}:
        return ("none",)
    classes = [
        label
        for label, rule_path in _CATEGORY_RULES
        if lemma in rule_frozenset(rule_path)
    ]
    return tuple(classes) or ("other_carrier",)


def carrier_signature_heads(signature: str) -> tuple[str, ...]:
    """Split a mined carrier signature into normalized heads."""
    heads = tuple(
        _clean_head(part)
        for raw in re.split(r"[+,]", str(signature or ""))
        for part in (raw.strip(),)
        if _clean_head(part) not in {"", "_", "none"}
    )
    return tuple(dict.fromkeys(heads))


def carrier_signature_classes(signature: str) -> tuple[str, ...]:
    """Return ordered diagnostic classes for a multi-head carrier signature."""
    classes: list[str] = []
    for head in carrier_signature_heads(signature):
        for label in carrier_head_classes(head):
            if label != "none" and label not in classes:
                classes.append(label)
    return tuple(classes) or ("none",)


def entity_boundary_carrier_heads() -> frozenset[str]:
    """Heads that can make the marked protein not be the endpoint itself."""
    heads: set[str] = set()
    for label, rule_path in _CATEGORY_RULES:
        if label in _ENTITY_BOUNDARY_CLASSES:
            heads.update(rule_frozenset(rule_path))
    return frozenset(heads)


def carrier_effect_guess(signature: str) -> str:
    """Single coarse label used only for human-audit sampling stratification."""
    heads = carrier_signature_heads(signature)
    if not heads:
        return "none"
    if any(_looks_like_tokenization_artifact(head) for head in heads):
        return "tokenization_artifact"
    classes = set(carrier_signature_classes(signature))
    if "relation_nominal" in classes:
        return "is_the_relation"
    if classes & _ENTITY_BOUNDARY_CLASSES:
        return "entity_boundary"
    if "assay_reagent_layer" in classes:
        return "assay_layer"
    if "expression_variant_layer" in classes:
        return "expression_or_variant"
    if "pathway_location_layer" in classes:
        return "blocks_to_pathway"
    if classes & {"metric_or_expression_layer", "attribute_layer"}:
        return "blocks_to_metric"
    if "process_layer" in classes:
        return "process_layer"
    if "generic_entity_head" in classes:
        return "generic_head"
    return "unknown"


def _clean_head(head: str) -> str:
    return str(head or "").strip().lower()


def _looks_like_tokenization_artifact(head: str) -> bool:
    return "-" in head or (
        head.startswith("protein") and head.endswith("-")
    ) or bool(_TOKENIZATION_ARTIFACT_RE.search(head))
