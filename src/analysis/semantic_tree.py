"""anytree wrapper and renderer for the reduced semantic tree."""
from __future__ import annotations

from typing import Any

from anytree import Node, RenderTree
from anytree.render import AsciiStyle, ContRoundStyle


RESET = "\033[0m"
BOLD = "\033[1m"
_PROTEIN = "\033[38;5;80m"    # cyan: protein mentions (entities)
_ACTION = "\033[38;5;214m"    # orange: predicates / actions
_GROUP = f"{BOLD}\033[38;5;33m"  # bold blue: group nodes
_TAG = "\033[35m"             # pink: semantic tags (<T>, role)
_NEG = "\033[31m"             # red: negation / alternative
_GRAY = "\033[90m"            # gray: case markers / function detail


def tree_from_payload(root_payload: dict[str, Any], parent: Node | None = None,
                      *, relation: str = "", color: bool = False) -> Node:
    """Build an anytree node (with payload) from one reduced-tree node dict."""
    node = Node(
        _label(root_payload, relation, color=color),
        parent=parent,
        kind=root_payload.get("kind", ""),
        node_id=root_payload.get("node_id"),
        relation=relation,
        payload=root_payload,
    )
    for shared in root_payload.get("effective_children", []) or []:
        tree_from_payload(shared, node, relation="shared", color=color)
    for member in root_payload.get("members", []) or []:
        tree_from_payload(member, node, relation="member", color=color)
    for child in root_payload.get("children", []) or []:
        tree_from_payload(child, node, relation="child", color=color)
    return node


def render_reduced_tree(payload: dict[str, Any], *, plain: bool = False) -> str:
    """Render the reduced tree; ``plain=True`` gives ASCII without colour."""
    roots = payload.get("nodes") or []
    if not roots:
        return "(empty reduced tree)"
    style = AsciiStyle() if plain else ContRoundStyle()
    color = not plain
    lines: list[str] = []
    for root_payload in roots:
        root = tree_from_payload(root_payload, color=color)
        for pre, _fill, node in RenderTree(root, style=style):
            lines.append(f"{pre}{node.name}")
    return "\n".join(lines)


def _c(text: str, code: str, color: bool) -> str:
    return f"{code}{text}{RESET}" if color else text


def _label(n: dict[str, Any], relation: str, *, color: bool) -> str:
    kind = n.get("kind", "")
    prefix = _c(f"{relation}: ", _GRAY, color) if relation else ""

    if kind == "lexical":
        is_protein = bool(n.get("targets"))
        is_action = (n.get("pos") == "VERB")
        text_color = _PROTEIN if is_protein else (_ACTION if is_action else "")
        bits = [_c(str(n.get("text", "")), text_color, color) if text_color else str(n.get("text", ""))]
        pos, dep = n.get("pos", ""), n.get("dep", "")
        if pos or dep:
            bits.append(_c(f"({pos}:{dep})", _GRAY, color))
        if n.get("targets"):
            bits.append(_c(f"<T={n['targets']}>", _TAG, color))
        if n.get("case_marker"):
            bits.append(_c(f"case={n['case_marker']}", _GRAY, color))
        return prefix + " ".join(bits)

    if kind == "member":
        roles = n.get("role_protein_indices", [])
        mods = n.get("modifier_protein_indices", [])
        label = _c(f"member role={roles}", _PROTEIN, color)
        if mods:
            label += " " + _c(f"mod={mods}", _GRAY, color)
        return prefix + label

    # group node
    text = n.get("text") or "[GROUP]"
    head = _c(f"{text} {kind} role={n.get('role_protein_indices', [])}", _GROUP, color)
    extra = []
    if n.get("all_protein_indices") != n.get("role_protein_indices"):
        extra.append(_c(f"all={n.get('all_protein_indices', [])}", _GRAY, color))
    if n.get("member_internal_pairs"):
        extra.append(_c(f"internal={n['member_internal_pairs']}", _GRAY, color))
    sh = n.get("shared_head") or {}
    if sh.get("text"):
        extra.append(_c(f"shared_head={sh['text']}", _TAG, color))
    if n.get("is_alternative"):
        extra.append(_c("ALT", _NEG, color))
    if n.get("coreferent"):
        extra.append(_c("COREF", _TAG, color))
    return prefix + head + ((" " + " ".join(extra)) if extra else "")
