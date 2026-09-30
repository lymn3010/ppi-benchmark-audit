#!/usr/bin/env python3
"""Rebuild the five PPI JSON datasets from ``data/datasets/<Corpus>/source_xml``.

charOffset is end-exclusive; entities are indexed by position; overlapping entities
share one placeholder group. Default is a read-only check; ``--write`` regenerates.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "datasets"
CLASSIC_CORPORA = ("AIMed", "BioInfer", "HPRD50", "IEPA", "LLL")
SPLITS = ("train", "test")
PROTEIN_RE = re.compile(r"\bPROTEIN(\d+)\b")


@dataclass(frozen=True)
class Entity:
    xml_id: str
    text: str
    spans: tuple[tuple[int, int], ...]
    xml_order: int

    @property
    def first_start(self) -> int:
        return self.spans[0][0]

    @property
    def cover(self) -> tuple[int, int]:
        return self.spans[0]


@dataclass(frozen=True)
class MaskGroup:
    entities: tuple[Entity, ...]
    start: int
    end: int


def _parse_char_offsets(raw: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        start_s, end_s = part.split("-", 1)
        start, end = int(start_s), int(end_s)
        if start < 0 or end < start:
            raise ValueError(f"Invalid charOffset range: {raw!r}")
        spans.append((start, end))
    if not spans:
        raise ValueError(f"Missing charOffset range: {raw!r}")
    return tuple(spans)


def _spans_overlap(a: Entity, b: Entity) -> bool:
    return any(sa < eb and sb < ea for sa, ea in a.spans for sb, eb in b.spans)


def _mask_groups(entities: list[Entity]) -> list[MaskGroup]:
    """Return connected overlap groups ordered by surface position."""
    parent = list(range(len(entities)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i, j in combinations(range(len(entities)), 2):
        if _spans_overlap(entities[i], entities[j]):
            union(i, j)

    buckets: dict[int, list[Entity]] = {}
    for idx, entity in enumerate(entities):
        buckets.setdefault(find(idx), []).append(entity)

    groups: list[MaskGroup] = []
    for bucket in buckets.values():
        ordered = tuple(sorted(bucket, key=lambda e: (e.first_start, e.xml_order)))
        groups.append(_make_group(ordered, all_entities=entities))
    return _coalesce_replacement_groups(groups)


def _full_cover(entity: Entity) -> tuple[int, int]:
    starts = [start for start, _end in entity.spans]
    ends = [end for _start, end in entity.spans]
    return min(starts), max(ends)


def _entity_inside_cover(entity: Entity, cover: tuple[int, int]) -> bool:
    start, end = cover
    return any(start <= s and e <= end for s, e in entity.spans)


def _make_group(
    entities: tuple[Entity, ...],
    *,
    all_entities: list[Entity] | None = None,
) -> MaskGroup:
    ordered = tuple(sorted(entities, key=lambda e: (e.first_start, e.xml_order)))
    if len(ordered) == 1:
        entity = ordered[0]
        full_start, full_end = _full_cover(entity)
        embedded_other = False
        if len(entity.spans) > 1 and all_entities:
            embedded_other = any(
                other.xml_id != entity.xml_id
                and not _spans_overlap(entity, other)
                and _entity_inside_cover(other, (full_start, full_end))
                for other in all_entities
            )
        start, end = entity.cover if embedded_other else (full_start, full_end)
    else:
        starts = [start for entity in ordered for start, _end in entity.spans]
        ends = [end for entity in ordered for _start, end in entity.spans]
        start, end = min(starts), max(ends)
    return MaskGroup(ordered, start, end)


def _coalesce_replacement_groups(groups: list[MaskGroup]) -> list[MaskGroup]:
    """Merge abbreviation singletons that fall inside a multi-entity cover."""
    sorted_groups = sorted(groups, key=lambda g: (g.start, g.end, g.entities[0].xml_order))
    merged: list[MaskGroup] = []
    for group in sorted_groups:
        if not merged or group.start >= merged[-1].end:
            merged.append(group)
            continue
        previous = merged[-1]
        if len(previous.entities) > 1 or len(group.entities) > 1:
            merged[-1] = _make_group(previous.entities + group.entities)
            continue
        # Overlapping singletons were already grouped above.
        raise ValueError(
            f"Overlapping singleton replacement covers: "
            f"{previous.start}-{previous.end} and {group.start}-{group.end}"
        )
    return merged


def _mask_sentence(text: str, groups: list[MaskGroup]) -> tuple[str, list[str], dict[str, int]]:
    parts: list[str] = []
    proteins: list[str] = []
    entity_to_index: dict[str, int] = {}
    cursor = 0
    for group in groups:
        if group.start < cursor:
            raise ValueError(f"Overlapping replacement covers at {group.start}-{group.end}")
        parts.append(text[cursor:group.start])
        placeholders: list[str] = []
        for entity in group.entities:
            index = len(proteins)
            proteins.append(entity.text)
            entity_to_index[entity.xml_id] = index
            placeholders.append(f"PROTEIN{index}")
        replacement = " ".join(placeholders)
        if group.start > 0 and text[group.start - 1].isalnum():
            replacement = " " + replacement
        if group.end < len(text) and text[group.end:group.end + 1].isalnum():
            replacement = replacement + " "
        parts.append(replacement)
        cursor = group.end
    parts.append(text[cursor:])
    return "".join(parts), proteins, entity_to_index


def _relations(sentence_el: ET.Element, entity_to_index: dict[str, int]) -> OrderedDict[str, str]:
    rels: dict[tuple[int, int], str] = {}
    for interaction in sentence_el.iter("interaction"):
        e1 = interaction.attrib.get("e1")
        e2 = interaction.attrib.get("e2")
        if e1 not in entity_to_index or e2 not in entity_to_index:
            raise ValueError(
                f"Interaction references unknown entity: {interaction.attrib!r}"
            )
        a, b = entity_to_index[e1], entity_to_index[e2]
        if a == b:
            continue
        pair = tuple(sorted((a, b)))
        rels[pair] = interaction.attrib.get("type") or "PPI"
    return OrderedDict((str(list(pair)), rels[pair]) for pair in sorted(rels))


def _record_from_sentence(sentence_el: ET.Element) -> tuple[str, dict]:
    sid = sentence_el.attrib["id"]
    text = sentence_el.attrib["text"]
    entities = [
        Entity(
            xml_id=entity.attrib["id"],
            text=entity.attrib.get("text", ""),
            spans=_parse_char_offsets(entity.attrib.get("charOffset", "")),
            xml_order=order,
        )
        for order, entity in enumerate(sentence_el.iter("entity"))
    ]
    groups = _mask_groups(entities)
    masked, proteins, entity_to_index = _mask_sentence(text, groups)
    return sid, {
        "sentence": masked,
        "orig_sentence": text,
        "proteins": proteins,
        "relations": _relations(sentence_el, entity_to_index),
    }


def _xml_path(corpus_dir: Path, corpus: str, split: str) -> Path:
    matches = sorted((corpus_dir / "source_xml").glob(f"{corpus}-{split}.xml"))
    if not matches:
        raise FileNotFoundError(f"Missing source XML for {corpus} {split}")
    return matches[0]


def _build_split(corpus_dir: Path, corpus: str, split: str) -> OrderedDict[str, dict]:
    root = ET.parse(_xml_path(corpus_dir, corpus, split)).getroot()
    rows: OrderedDict[str, dict] = OrderedDict()
    for sentence_el in root.iter("sentence"):
        sid, record = _record_from_sentence(sentence_el)
        if sid in rows:
            raise ValueError(f"Duplicate sentence id in XML: {sid}")
        rows[sid] = record
    return rows


def _metadata(corpus: str, split_names: Iterable[str], data: OrderedDict[str, dict], created_at: str) -> dict:
    split_names = list(split_names)
    total_pairs = 0
    sentences_with_relation = 0
    pairs_with_relation = 0
    for record in data.values():
        n = len(record["proteins"])
        total_pairs += n * (n - 1) // 2
        rel_count = len(record["relations"])
        pairs_with_relation += rel_count
        if rel_count:
            sentences_with_relation += 1
    return {
        "created_at": created_at,
        "source": (
            f"{corpus} {split_names[0]} (rebuilt from canonical XML, 2026-06-24)"
            if len(split_names) == 1
            else f"{corpus} {split_names} combined (rebuilt from canonical XML, 2026-06-24)"
        ),
        "total_sentences": len(data),
        "sentences_with_relation": sentences_with_relation,
        "total_pairs": total_pairs,
        "pairs_with_relation": pairs_with_relation,
    }


def _load_existing_created_at(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("metadata", {}).get("created_at")
    except Exception:
        return None


def _package(corpus: str, split_names: Iterable[str], data: OrderedDict[str, dict], path: Path, created_at: str | None) -> dict:
    timestamp = (
        created_at
        or _load_existing_created_at(path)
        or datetime.now(timezone.utc).isoformat()
    )
    return {
        "metadata": _metadata(corpus, split_names, data, timestamp),
        "data": data,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _normalise_for_compare(payload: dict) -> dict:
    copied = json.loads(json.dumps(payload, ensure_ascii=False))
    copied.get("metadata", {}).pop("created_at", None)
    return copied


def _compare_payload(current: dict, generated: dict, *, strict_surface: bool) -> tuple[list[str], int]:
    """Return structural mismatches and number of masked-surface differences."""
    structural: list[str] = []
    current_norm = _normalise_for_compare(current)
    generated_norm = _normalise_for_compare(generated)
    if current_norm.get("metadata") != generated_norm.get("metadata"):
        structural.append(
            f"metadata differs: {current_norm.get('metadata')} != {generated_norm.get('metadata')}"
        )
    current_data = current_norm.get("data") or {}
    generated_data = generated_norm.get("data") or {}
    if list(current_data) != list(generated_data):
        structural.append("sentence id order differs")
        return structural, 0
    surface_diffs = 0
    for sid, current_row in current_data.items():
        generated_row = generated_data.get(sid)
        if generated_row is None:
            structural.append(f"{sid}: missing generated row")
            continue
        for field in ("orig_sentence", "proteins", "relations"):
            if current_row.get(field) != generated_row.get(field):
                structural.append(f"{sid}: {field} differs")
        if current_row.get("sentence") != generated_row.get("sentence"):
            surface_diffs += 1
            if strict_surface:
                structural.append(f"{sid}: masked sentence surface differs")
    return structural, surface_diffs


def _protein_boundary_problems(data: OrderedDict[str, dict]) -> list[tuple[str, list[int], list[int]]]:
    problems: list[tuple[str, list[int], list[int]]] = []
    for sid, record in data.items():
        seen = {int(m.group(1)) for m in PROTEIN_RE.finditer(record["sentence"])}
        expected = set(range(len(record["proteins"])))
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        if missing or extra:
            problems.append((sid, missing, extra))
    return problems


def rebuild_corpus(corpus: str, *, root: Path, created_at: str | None) -> dict[str, dict]:
    corpus_dir = root / corpus
    split_data = {
        split: _build_split(corpus_dir, corpus, split)
        for split in SPLITS
    }
    full_data: OrderedDict[str, dict] = OrderedDict()
    for split in SPLITS:
        full_data.update(split_data[split])

    return {
        "train": _package(corpus, ["train"], split_data["train"], corpus_dir / "train.json", created_at),
        "test": _package(corpus, ["test"], split_data["test"], corpus_dir / "test.json", created_at),
        "full": _package(corpus, ["train", "test"], full_data, corpus_dir / "full.json", created_at),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--corpora", nargs="+", default=list(CLASSIC_CORPORA))
    parser.add_argument("--write", action="store_true", help="overwrite dataset JSON files")
    parser.add_argument(
        "--created-at",
        default=None,
        help="metadata.created_at value for all outputs; default preserves existing value if present",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on any mismatch in check mode",
    )
    parser.add_argument(
        "--strict-surface",
        action="store_true",
        help="also require generated masked sentence strings to match current JSON exactly",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mismatches: list[str] = []
    for corpus in args.corpora:
        if corpus not in CLASSIC_CORPORA:
            raise ValueError(f"{corpus!r} is not one of {CLASSIC_CORPORA}")
        built = rebuild_corpus(corpus, root=args.root, created_at=args.created_at)
        print(f"\n[{corpus}]")
        for split, payload in built.items():
            path = args.root / corpus / f"{split}.json"
            data = payload["data"]
            boundary_problems = _protein_boundary_problems(data)
            rels = sum(len(row["relations"]) for row in data.values())
            print(
                f"  {split}: sentences={len(data)} relations={rels} "
                f"boundary_problems={len(boundary_problems)}"
            )
            if boundary_problems:
                sample = boundary_problems[:3]
                mismatches.append(f"{corpus}/{split}: protein boundary problems {sample}")
            if args.write:
                _write_json(path, payload)
                continue
            if path.exists():
                current = json.loads(path.read_text(encoding="utf-8"))
                structural, surface_diffs = _compare_payload(
                    current,
                    payload,
                    strict_surface=args.strict_surface,
                )
                if surface_diffs and not args.strict_surface:
                    print(f"    legacy_surface_differences={surface_diffs}")
                for issue in structural:
                    mismatches.append(f"{corpus}/{split}: {issue}")
            else:
                mismatches.append(f"{corpus}/{split}: missing current JSON {path}")

    if mismatches:
        print("\nMISMATCHES:")
        for issue in mismatches[:50]:
            print(f"  - {issue}")
        if args.strict or not args.write:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
