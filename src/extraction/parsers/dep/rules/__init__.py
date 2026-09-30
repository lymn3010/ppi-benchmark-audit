"""Dependency-parser extraction rules with explicit contracts."""

from .agentive_process_bridge import (
    AGENTIVE_PROCESS_BRIDGE_RULE_ID,
    AGENTIVE_PROCESS_BRIDGE_PROJECTION_RULE_ID,
    build_agentive_process_bridge_events,
    is_agentive_bridge_predicate,
    is_agentive_process_nominal,
)
from .nested_composition import bind_nested_controller
from .collective_orthography import (
    COLLECTIVE_SLASH_MEMBER_RULE_ID,
    collective_slash_member_pairs,
)
from .assertion_scope import (
    ADVCL_SCOPE_BARRIER_RULE_ID,
    NESTED_NOMINAL_LOCAL_SCOPE_RULE_ID,
    include_parent_in_negation_search,
)
from .reciprocal_subjects import (
    RECIPROCAL_PURPOSE_SUBJECT_RULE_ID,
    SUBJECT_ONLY_RECIPROCITY_GATE_RULE_ID,
    allow_subject_only_participants,
)
from .compound_ownership import (
    TRANSITIVE_COMPOUND_OWNER_RULE_ID,
    transitive_compound_protein_members,
)
from .subject_inheritance import (
    ACL_CONJ_SUBJECT_INHERITANCE_RULE_ID,
    inherit_subjects_from_acl_conj_head,
)

__all__ = [
    "ADVCL_SCOPE_BARRIER_RULE_ID",
    "AGENTIVE_PROCESS_BRIDGE_RULE_ID",
    "AGENTIVE_PROCESS_BRIDGE_PROJECTION_RULE_ID",
    "COLLECTIVE_SLASH_MEMBER_RULE_ID",
    "build_agentive_process_bridge_events",
    "is_agentive_bridge_predicate",
    "is_agentive_process_nominal",
    "NESTED_NOMINAL_LOCAL_SCOPE_RULE_ID",
    "RECIPROCAL_PURPOSE_SUBJECT_RULE_ID",
    "SUBJECT_ONLY_RECIPROCITY_GATE_RULE_ID",
    "TRANSITIVE_COMPOUND_OWNER_RULE_ID",
    "ACL_CONJ_SUBJECT_INHERITANCE_RULE_ID",
    "allow_subject_only_participants",
    "bind_nested_controller",
    "collective_slash_member_pairs",
    "include_parent_in_negation_search",
    "inherit_subjects_from_acl_conj_head",
    "transitive_compound_protein_members",
]
