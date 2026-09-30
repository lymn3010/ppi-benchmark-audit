"""Align serialized argument-group provenance without DEP internals."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence


@dataclass(frozen=True)
class GroupAlignment:
    aligned: tuple[tuple[int, int], ...]
    cross: tuple[tuple[int, int], ...]
    mode: str = ""
    slots: tuple[int, int] = (0, 0)

    @property
    def is_diagonal(self) -> bool:
        return self.mode == "diagonal"


def align_group_provenance(
    source_group: Mapping | None,
    target_group: Mapping | None,
) -> GroupAlignment | None:
    """Full or diagonal alignment of role-bearing proteins; None without group provenance."""
    if not source_group or not target_group:
        return None

    s_slots = _members(source_group)
    t_slots = _members(target_group)
    if not s_slots or not t_slots:
        return None

    has_subgroup = (
        any(len(_all_proteins(m)) > len(_role_proteins(m)) for m in s_slots)
        or any(len(_all_proteins(m)) > len(_role_proteins(m)) for m in t_slots)
        or any(len(_role_proteins(m)) > 1 for m in s_slots)
        or any(len(_role_proteins(m)) > 1 for m in t_slots)
    )
    use_diagonal = (
        len(s_slots) == len(t_slots) >= 2
        and not bool(source_group.get("is_alternative"))
        and not bool(target_group.get("is_alternative"))
        and not bool(source_group.get("coreferent"))
        and not bool(target_group.get("coreferent"))
        and has_subgroup
    )

    aligned: set[tuple[int, int]] = set()
    cross: set[tuple[int, int]] = set()

    for i, s in enumerate(s_slots):
        for j, t in enumerate(t_slots):
            for a, b in product(_role_proteins(s), _role_proteins(t)):
                if a == b:
                    continue
                pair = (a, b) if a < b else (b, a)
                if not use_diagonal:
                    aligned.add(pair)
                elif i == j:
                    aligned.add(pair)
                else:
                    cross.add(pair)

    cross -= aligned
    return GroupAlignment(
        aligned=tuple(sorted(aligned)),
        cross=tuple(sorted(cross)),
        mode="diagonal" if use_diagonal else "full",
        slots=(len(s_slots), len(t_slots)),
    )


def _members(group: Mapping) -> tuple[Mapping, ...]:
    raw = group.get("members") or ()
    return tuple(m for m in raw if isinstance(m, Mapping))


def _role_proteins(member: Mapping) -> tuple[int, ...]:
    values = member.get("role_protein_indices")
    if values is None:
        values = member.get("protein_indices") or ()
    return _int_tuple(values)


def _all_proteins(member: Mapping) -> tuple[int, ...]:
    return _int_tuple(member.get("protein_indices") or ())


def _int_tuple(values: Sequence | object) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ()
    out: set[int] = set()
    for value in values:
        try:
            out.add(int(value))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(out))
