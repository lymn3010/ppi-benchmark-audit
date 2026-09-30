"""Present structural and lexical-path evidence in a shared, read-only form."""
from __future__ import annotations

from dataclasses import dataclass, field

_POS_DECISION = "applied"


@dataclass(frozen=True)
class PathFields:
    """Normalized projection fields, emitted by either adapter below."""

    event_type: str = ""        # INTERACTION | INTERACTION_STATE | RELATION_STATEMENT | ""
    topology: str = ""          # symmetric | directed | dep_path | ""
    anchor: str = ""            # raw lemma, e.g. "bind" / "complex" / "" (not normalized)
    construction: str = ""      # verbal | passive | nominalized | compound_state | nested_dobj | ...
    prep: str = ""              # normalized connector ("of"/"with"/"to"/"between"/"")
    carrier_head: str = ""      # explodeable attribute ONLY (never in the key); single head
    source_method: str = ""     # typed_rule | dep_mined
    unmapped_reason: str = ""   # "" if clean; else "no_anchor" | "multi_carrier"
    extra: dict = field(default_factory=dict)


def path_key(f: PathFields) -> tuple[str, str, str, str]:
    """Low-cardinality canonical aggregation VIEW key (shared by both sources).

    Carrier head is deliberately excluded (it is an explodeable attribute).
    """
    return (f.event_type or "?", f.topology or "?", f.anchor or "_", f.construction or "?")


def is_mapped(f: PathFields) -> bool:
    """True if the row is a clean construction (structural, or lexpath with one anchor)."""
    if f.source_method == "typed_rule":
        return True
    return bool(f.anchor) and f.anchor not in ("", "_") and not f.unmapped_reason


def render_path(f: PathFields) -> str:
    """Deterministic display string for a path."""
    a, b = "P0", "P1"
    anchor = f.anchor or "?"
    prep = f.prep if f.prep not in ("", "_", "none") else ""
    carrier = f.carrier_head or ""
    et = (f.event_type or "").upper()

    if et == "INTERACTION_STATE":
        return f'{{{a}, {b}}} "{anchor}"'
    if et == "RELATION_STATEMENT":
        if carrier:
            return f'{a} [IS-A] "{carrier}" {prep or "of"} {b}'
        if anchor.lower() in {"is-a", "identity", "has", "part-of"}:
            return f'{a} [IS-A] {b}'
        return f'{a} [IS-A] "{anchor}" {b}'
    # INTERACTION (typed) or untyped dep-mined path
    if carrier:
        rhs = f'("{carrier}" {prep or "of"} {b})'
    elif prep:
        rhs = f'{prep} {b}'
    else:
        rhs = b
    return f'{a} "{anchor}" {rhs}'.replace("  ", " ").strip()


# --- adapters: each KNOWS one source's field names, emits the shared PathFields ---

def from_candidate_relation(row: dict) -> PathFields:
    """Structural candidate_relation row -> PathFields.

    Reads canonical_pattern_key = evidence_class|shape|lemma|prep|construction.
    """
    parts = str(row.get("canonical_pattern_key") or "").split("|")
    lemma = parts[2] if len(parts) > 2 else (row.get("predicate_lemma") or "")
    prep = parts[3] if len(parts) > 3 and parts[3] != "_" else ""
    construction = parts[4] if len(parts) > 4 else (row.get("construction") or "")
    return PathFields(
        event_type=(row.get("event_type") or "INTERACTION").upper(),
        topology="symmetric",
        anchor=lemma,
        construction=construction,
        prep=prep,
        carrier_head=(row.get("case_marker") or "") if row.get("owner_propagated") else "",
        source_method="typed_rule",
    )


def from_lexpath_row(row: dict) -> PathFields:
    """Lexpath row -> PathFields; ``unmapped_reason`` is no_anchor or multi_carrier."""
    anchor = str(row.get("apex_lemma") or "")
    carrier_sig = str(row.get("carrier_head_signature") or row.get("carrier_signature") or "")
    multi = "+" in carrier_sig
    # single clean head only; multi-head (+) carriers carry no single attribute
    carrier = carrier_sig if (carrier_sig and not multi and carrier_sig != "_") else ""
    prep_sig = str(row.get("prep_signature") or "")
    prep = prep_sig.split("|")[0] if prep_sig and prep_sig not in ("none", "", "_") else ""
    reason = "no_anchor" if not anchor else ("multi_carrier" if multi else "")
    return PathFields(
        event_type="",
        topology="dep_path",
        anchor=anchor,
        construction=str(row.get("construction") or ""),
        prep=prep,
        carrier_head=carrier,
        source_method="dep_mined",
        unmapped_reason=reason,
    )
