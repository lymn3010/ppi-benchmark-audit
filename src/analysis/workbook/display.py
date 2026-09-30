"""Reader-facing relation and entity-boundary descriptions."""
from __future__ import annotations
import logging
import re
from typing import Any
log = logging.getLogger(__name__)
def _anchor_from_semantic_key(row: dict) -> str:
    key = str(row.get("semantic_pattern_key") or "")
    parts = key.split("|")
    if len(parts) >= 2 and parts[1]:
        return parts[1].lower()
    return str(row.get("lemma") or "").lower()


def _trigger_type_for_construction(construction: str, evidence_class: str = "") -> str:
    """Human-facing trigger type used by compact audit outputs."""
    evidence = str(evidence_class or "").lower()
    if evidence == "relational_statement":
        return "relational_nominal"
    if evidence == "lexicalized_path_candidate":
        return "lexical_path"
    text = str(construction or "").lower()
    if text in {"nominalized", "compound_state", "state_nominal"}:
        return "nominal"
    if text in {"verbal", "passive", "hyphen_participle"}:
        return "verbal"
    return "nominal" if "nom" in text else "verbal"


def _argument_nominal_head_text(argument: dict) -> str:
    """Best-effort nominal head of a relation-statement argument (display only)."""
    core = argument.get("core") if isinstance(argument.get("core"), dict) else {}
    if isinstance(core, dict) and core.get("protein_indices"):
        return ""
    text = str(argument.get("text") or "").strip()
    if not text:
        tokens = core.get("tokens") if isinstance(core, dict) else None
        if isinstance(tokens, list):
            text = " ".join(str(token) for token in tokens).strip()
    if not text:
        return ""
    if re.search(r"\bprotein\d+\b", text, flags=re.IGNORECASE):
        return ""
    words = [
        w.strip("()[]{}.,;:'\"").lower()
        for w in text.replace("/", " ").replace("-", " ").split()
    ]
    words = [
        w for w in words
        if w and not re.fullmatch(r"protein\d+", w) and any(ch.isalpha() for ch in w)
    ]
    if not words:
        return ""
    return words[-1]


def _relational_nominal_from_arguments(arguments: dict) -> str:
    """Return the core nominal for IS-A/HAS-style relation statements if present."""
    for role in ("entities_b", "entities_a"):
        for argument in arguments.get(role) or []:
            if not isinstance(argument, dict):
                continue
            head = _argument_nominal_head_text(argument)
            if head:
                return head
    return ""


def _yaml_chain_for_pattern(row: dict) -> str:
    lemma = str(row.get("lemma") or _anchor_from_semantic_key(row) or "_")
    shape = str(row.get("shape") or "")
    prep = str(row.get("preposition") or "_")
    construction = str(row.get("construction") or "")
    if shape == "AA":
        return f'[A+] "{lemma}"'
    if construction == "passive":
        marker = prep if prep not in ("", "_", "None") else "by"
        return f'[B] "{lemma}" "{marker}" [A]'
    if shape == "AB" and prep not in ("", "_", "None"):
        return f'[A] "{lemma}" "{prep}" [B]'
    if shape == "NESTED":
        return f'[[event]] "{lemma}" [B]'
    if shape == "RELATIONAL":
        return f'[A] REL:"{lemma}" [B]'
    return f'[A] "{lemma}" [B]'


def _chain_from_pattern_key(pattern_key: str) -> str:
    """Best-effort YAML-style chain from a canonical pattern key."""
    parts = str(pattern_key or "").split("|")
    if len(parts) != 5:
        return str(pattern_key or "")
    evidence, shape, lemma, prep, construction = parts
    if evidence == "relational":
        normalized = lemma.strip().lower()
        if normalized in {"is-a", "has", "part-of", "identity"}:
            return _yaml_chain_for_relational_statement(normalized)
        return _yaml_chain_for_relational_statement("IS-A", lemma)
    return _yaml_chain_for_pattern({
        "lemma": lemma,
        "shape": shape,
        "preposition": "" if prep == "_" else prep,
        "construction": construction,
    })


def _contextual_chain(chain: str, context_word: str, role: str = "") -> str:
    """Pattern display with its lexical carrier; the carrier stays out of the pattern key."""
    chain = str(chain or "")
    context_word = str(context_word or "").strip().replace('"', '\\"')
    role = str(role or "").strip().lower()
    if not context_word:
        return chain
    if any(marker in chain for marker in (" IS-A ", " HAS ", " PART-OF ", " IDENTITY? ")):
        index = chain.find("[A]")
        if index >= 0:
            return (
                f'{chain[:index]}'
                f'[A "{context_word}"]'
                f'{chain[index + len("[A]"):]}'
            )
    if role in {"sources", "source", "entities_a", "entity_a", "participants"}:
        index = chain.find("[A]")
        if index >= 0:
            return (
                f'{chain[:index]}'
                f'[A "{context_word}"]'
                f'{chain[index + len("[A]"):]}'
            )
    if role in {"targets", "target", "entities_b", "entity_b"}:
        index = chain.find("[B]")
        if index >= 0:
            return (
                f'{chain[:index]}'
                f'[B "{context_word}"]'
                f'{chain[index + len("[B]"):]}'
            )
    for marker in ("[B]", "[A+]", "[A]"):
        index = chain.rfind(marker)
        if index >= 0:
            return (
                f'{chain[:index]}'
                f'{marker[:-1]} "{context_word}"]'
                f'{chain[index + len(marker):]}'
            )
    return f'{chain} ["{context_word}"]'


def _context_trigger_summary_text(rows: list[dict]) -> str:
    """Compact carrier x trigger summary for the workbook."""
    parts = []
    for row in rows[:6]:
        total = int(row.get("evaluable") or row.get("occurrences") or 0)
        pos = int(row.get("gold_positive") or 0)
        neg = int(row.get("gold_negative") or 0)
        agreement = row.get("agreement")
        agreement_text = "-" if agreement is None else f"{float(agreement):.3f}"
        parts.append(
            f"{row.get('contextual_chain') or row.get('chain') or row.get('pattern_key')}: "
            f"{total} ({pos}P/{neg}N, {agreement_text})"
        )
    return "\n".join(parts)


def _carrier_propagation_breakdown(row: dict) -> str:
    """Human-readable carrier provenance without spreading it over many columns."""
    pieces = []
    for label, key in (
        ("owner", "owner_occurrences"),
        ("descriptor", "descriptor_occurrences"),
        ("nested", "nested_occurrences"),
    ):
        count = int(row.get(key) or 0)
        if count:
            pieces.append(f"{label}={count}")
    return ", ".join(pieces) if pieces else "-"


def _yaml_chain_for_relational_statement(schema: str, trigger: str = "") -> str:
    """Display chain for review-only relational evidence (IS-A, HAS, PART-OF)."""
    label = (schema or "RELATIONAL").upper()
    trigger = str(trigger or "").strip().replace('"', '\\"')
    if label == "IDENTITY":
        return "[A] IDENTITY? [B]"
    if trigger and label in {"IS-A", "HAS", "PART-OF"}:
        return f'[A] {label} "{trigger}" [B]'
    if label in {"IS-A", "HAS", "PART-OF"}:
        return f"[A] {label} [B]"
    if trigger:
        return f'[A] REL:{label} "{trigger}" [B]'
    return f"[A] REL:{label} [B]"


def _audit_chain_for_relational_statement(schema: str, trigger: str = "") -> str:
    """Readable relation-statement pattern with lexical nominal anchor when present."""
    return _yaml_chain_for_relational_statement(schema, trigger)


def _lexpath_chain(row: dict) -> str:
    """Compact YAML-style display chain for mined lexicalized paths."""
    apex = str(row.get("apex_lemma") or "")
    if apex.lower().startswith("protein"):
        apex = ""
    edges = sorted(
        row.get("endpoint_edges") or [],
        key=lambda e: int(e.get("protein_index") or 0),
    )
    preps = [_yaml_scalar_text(e.get("prep") or "") for e in edges]
    if apex and apex != "_":
        if len(preps) >= 2 and preps[0] and preps[1]:
            return f'[A] "{preps[0]}" "{apex}" "{preps[1]}" [B]'
        if preps and preps[0]:
            return f'[A] "{preps[0]}" "{apex}" [B]'
        return f'[A] "{apex}" [B]'

    residual = _lexpath_residual_lemmas(row)
    if residual:
        return "[A] " + " ".join(f'"{lemma}"' for lemma in residual[:4]) + " [B]"
    if len(preps) >= 2 and preps[0] and preps[1]:
        return f'[A] "{preps[0]}" "{preps[1]}" [B]'
    if preps and preps[0]:
        return f'[A] "{preps[0]}" [B]'
    return "[A] [residual_path] [B]"


def _lexpath_residual_lemmas(row: dict) -> list[str]:
    examples = row.get("examples") or []
    if not examples:
        return []
    path = examples[0].get("dep_path") or []
    lemmas = []
    for step in path:
        lemma = str(step.get("lemma") or step.get("text") or "").lower()
        if not lemma or lemma.startswith("protein"):
            continue
        if lemma not in lemmas:
            lemmas.append(lemma)
    return lemmas


def _yaml_scalar_text(value: Any) -> str:
    # Quote on/off/yes/no so YAML 1.1 keeps them as strings.
    if value is True:
        return "on"
    if value is False:
        return "off"
    return str(value or "")


_PUBLIC_LABELS = {
    'candidate_pattern_ledger': 'Relation proposal',
    'structural': 'Event structure',
    'relational_statement': 'Reference statement',
    'lexicalized_path': 'Dependency path',
    'event_without_final_pair': 'Event without projected pair',
    'relation_schema': 'Reference construction',
    'lexical_path': 'Dependency path',
    'gold_positive': 'Released positive',
    'gold_negative': 'Released negative',
    'prediction_only': 'Not released positive',
    'unlabeled': 'Unlabeled',
}


def public_label(value):
    return _PUBLIC_LABELS.get(str(value), str(value).replace('_', ' '))


def public_labels(values):
    return ', '.join(public_label(value) for value in values)
