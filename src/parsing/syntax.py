"""Tokens, dependency links and coreference evidence shared by the reader."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# SyntaxNode: parser-neutral token

@dataclass
class SyntaxNode:
    """One token with index and half-open char offsets; head_i is -1 at root."""
    i: int
    text: str
    lemma: str
    pos: str                         # universal POS
    dep: str                         # normalized internal role
    head_i: int                      # -1 for root
    char_start: int
    char_end: int
    # Enrichments filled by the adapter
    protein_indices: tuple[int, ...] = ()
    is_nominalized: bool = False
    verb_form: str = ""
    nominalization_source: str = ""
    nominalization_confidence: float | None = None
    # Raw labels for audit / disagreement tracking
    raw_dep: str = ""
    raw_lemma: str = ""
    raw_pos: str = ""
    raw_deps: str = ""
    raw_feats: str = ""
    raw_misc: str = ""
    sentence_index: int = 0
    sentence_word_id: int = 0
    sentence_head_id: int = 0
    multiword_token_id: tuple[int, ...] = ()
    multiword_token_text: str = ""
    enhanced_heads: tuple[tuple[int, str], ...] = ()
    native_id: tuple[int, ...] = ()
    # Direct and inherited protein ids; protein_indices is their union.
    direct_protein_indices: tuple[int, ...] = ()
    inherited_protein_indices: tuple[int, ...] = ()
    inherited_reference_kinds: tuple[str, ...] = ()
    # Antecedent mention of an accepted identity reference.
    inherited_reference_head_i: int = -1
    inherited_reference_node_indices: tuple[int, ...] = ()

    @property
    def is_root(self) -> bool:
        return self.head_i == -1 or self.head_i == self.i


# CorefCluster: coreference evidence (never collapses into protein identity)

@dataclass
class CorefCluster:
    """Coreference evidence with half-open spans; does not merge proteins."""
    cluster_id: int
    mentions: tuple[tuple[int, int], ...]  # (char_start, char_end) per mention
    evidence_type: str = "nominal"         # "pronoun" | "nominal" | "lexical"
    head_text: str = ""
    source: str = ""
    confidence: float | None = None
    mention_head_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class BackendInfo:
    """Identity and declared capabilities of the producing parser backend."""

    backend: str = ""
    model: str = ""
    version: str = ""
    capabilities: tuple[str, ...] = ()
    enrichments: tuple[str, ...] = ()
    native_format: str = ""


# ParsedSentence: parser-neutral sentence

@dataclass
class ParsedSentence:
    """Reader input: sorted tokens, dependencies and protein-marker positions."""
    text: str
    sentence_id: str
    nodes: tuple[SyntaxNode, ...]
    protein_map: dict[str, tuple[int, ...]] = field(default_factory=dict)
    coref_clusters: tuple[CorefCluster, ...] = ()
    parser_name: str = ""
    raw_parser_data: dict = field(default_factory=dict)
    backend_info: BackendInfo = field(default_factory=BackendInfo)

    # Structural accessors.

    def node(self, i: int) -> SyntaxNode:
        """Return the node at sentence-local index i."""
        return self.nodes[i]

    def head(self, node: SyntaxNode) -> Optional[SyntaxNode]:
        """Return the syntactic head of node, or None if root."""
        if node.is_root:
            return None
        return self.nodes[node.head_i]

    def children(self, node: SyntaxNode) -> list[SyntaxNode]:
        """Return all direct syntactic children of node, in document order."""
        return [n for n in self.nodes if n.head_i == node.i and n.i != node.i]

    def children_by_dep(self, node: SyntaxNode, *deps: str) -> list[SyntaxNode]:
        """Children whose normalized dep label is in deps."""
        return [c for c in self.children(node) if c.dep in deps]

    def children_by_raw_dep(self, node: SyntaxNode, *deps: str) -> list[SyntaxNode]:
        """Children whose original parser dependency label is in deps."""
        return [c for c in self.children(node) if c.raw_dep in deps]

    def enhanced_heads(self, node: SyntaxNode) -> list[tuple[Optional[SyntaxNode], str]]:
        """Return parser-native enhanced heads without changing the basic tree."""
        out: list[tuple[Optional[SyntaxNode], str]] = []
        for head_i, role in node.enhanced_heads:
            out.append((None if head_i < 0 else self.node(head_i), role))
        return out

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)
