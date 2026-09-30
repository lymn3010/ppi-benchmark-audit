"""Parser-neutral representation layer; Stanza adapter in ``src/parsing/stanza``."""
from .syntax import BackendInfo, CorefCluster, ParsedSentence, SyntaxNode
from .backend import BatchParserBackend, ParserBackend
from .role_normalizer import RoleNormalizer, normalize_ud_role
from .factory import build_parser_backend

__all__ = [
    "BackendInfo",
    "BatchParserBackend",
    "ParserBackend",
    "ParsedSentence",
    "SyntaxNode",
    "CorefCluster",
    "RoleNormalizer",
    "normalize_ud_role",
    "build_parser_backend",
]
