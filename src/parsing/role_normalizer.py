"""Map UD dependency labels to reader roles while preserving the raw labels."""
from __future__ import annotations


# Role mapping table: UD label → internal label

_UD_TO_INTERNAL: dict[str, str] = {
    # Root
    "root": "ROOT",
    # Core arguments (UD v2 collapse of Stanford deps)
    "obj": "dobj",
    "iobj": "dative",
    # Obliques
    "obl": "nmod",          # prepositional role preserved via case child
    "obl:agent": "nmod",  # by-agent of passive → treated as oblique nominal (nmod)
    # Passive morphology
    "nsubj:pass": "nsubjpass",
    "aux:pass": "auxpass",
    # Coordination (CRAFT sometimes uses preconj instead of cc:preconj)
    "preconj": "cc:preconj",
    # Relative clause.
    "acl:relcl": "acl:relcl",  # pass-through (already matches internal)
    # Fixed/MWE (already match)
    "fixed": "fixed",
    "flat": "flat",
}

# Labels already in internal form (documentation only).
_PASS_THROUGH = {
    "nsubj", "nmod", "nmod:of", "nmod:npmod",
    "compound", "amod", "det", "case", "mark",
    "cc", "cc:preconj", "conj", "appos",
    "aux", "cop", "neg", "advmod",
    "acl", "relcl", "advcl", "xcomp", "ccomp",
    "dep", "dobj", "nsubjpass", "auxpass",
    "attr", "dative", "ROOT",
}


class RoleNormalizer:
    """Normalize parser dependency labels to internal roles (``obj`` -> ``dobj``)."""

    def __init__(self, extra: dict[str, str] | None = None) -> None:
        self._map = dict(_UD_TO_INTERNAL)
        if extra:
            self._map.update(extra)

    def normalize(self, label: str) -> str:
        """Return the internal label for the given parser dep label."""
        return self._map.get(label, label)

    def normalize_all(self, labels: list[str]) -> list[str]:
        """Normalize a list of dep labels in-place semantics (new list)."""
        return [self.normalize(lbl) for lbl in labels]

    def mapping(self) -> dict[str, str]:
        """Return the explicit parser-label mapping used by this normalizer."""
        return dict(self._map)


# Module-level convenience function

_default_normalizer = RoleNormalizer()


def normalize_ud_role(label: str) -> str:
    """Normalize a UD label with the default map; unknown labels pass through."""
    return _default_normalizer.normalize(label)


def default_ud_role_mapping() -> dict[str, str]:
    """Explicit UD labels whose names change at the parser boundary."""
    return dict(_UD_TO_INTERNAL)


def documented_pass_through_roles() -> tuple[str, ...]:
    """Labels documented as already compatible with the internal vocabulary."""
    identity_mapped = {
        source for source, target in _UD_TO_INTERNAL.items()
        if source == target
    }
    return tuple(sorted(_PASS_THROUGH | identity_mapped))
