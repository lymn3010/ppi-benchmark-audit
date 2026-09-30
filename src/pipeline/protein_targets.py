"""Protein index alignment and existing self-alias filtering policy."""
from __future__ import annotations

import re
from dataclasses import replace as dc_replace
from src.parsing import ParsedSentence
from src.models.candidate import CandidateEvent
from .contracts import EntryData

class ProteinTargets:
    @staticmethod
    def _normalize_parsed_targets(
        parsed: ParsedSentence,
        entry_data: EntryData,
    ) -> ParsedSentence:
        """Align protein indices with the dataset protein list; unchanged if the list is empty."""
        proteins = entry_data.get("proteins") or []
        if not proteins:
            return parsed

        protein_count = len(proteins)
        target_re = re.compile(r"PROTEIN(\d+)")
        # Collect (node_i, numeric_id) in document order
        matches: list[tuple[int, int]] = []
        for node in parsed.nodes:
            for m in target_re.finditer(node.text):
                matches.append((node.i, int(m.group(1))))

        if not matches:
            return parsed

        # Build corrected protein_indices per node
        corrected: dict[int, list[int]] = {}
        first_node_by_target: dict[int, int] = {}
        for occurrence_index, (node_i, numeric_id) in enumerate(matches):
            if 0 <= numeric_id < protein_count:
                target_id = numeric_id
            elif occurrence_index < protein_count:
                target_id = occurrence_index
            else:
                continue
            corrected.setdefault(node_i, [])
            if target_id not in corrected[node_i]:
                corrected[node_i].append(target_id)
            first_node_by_target.setdefault(target_id, node_i)

        # Rebuild nodes with corrected protein_indices
        new_nodes = tuple(
            dc_replace(
                n,
                protein_indices=tuple(sorted(corrected.get(n.i, n.protein_indices))),
                direct_protein_indices=tuple(sorted(corrected.get(n.i, n.protein_indices))),
                inherited_protein_indices=(),
            )
            for n in parsed.nodes
        )

        # Rebuild protein_map from corrected assignments
        new_protein_map: dict[str, tuple[int, ...]] = {
            f"PROTEIN{i}": (first_node_by_target[i],)
            for i in range(protein_count)
            if i in first_node_by_target
        }

        return dc_replace(parsed, nodes=new_nodes, protein_map=new_protein_map)


    @classmethod
    def _protein_entities_from_parsed(
        cls,
        parsed: ParsedSentence,
        entry_data: EntryData,
    ) -> list[dict]:
        proteins = entry_data.get("proteins") or []
        if proteins:
            names = [str(p) for p in proteins]
        else:
            names = sorted(parsed.protein_map.keys())
        canonical = [cls._canonical_protein_name(n) for n in names]
        return [
            {"index": i, "text": name, "canonical_id": canonical[i] if i < len(canonical) else ""}
            for i, name in enumerate(names)
        ]


    @classmethod
    def _self_alias_candidate_pairs_from_parsed(
        cls,
        parsed: ParsedSentence,
        entry_data: EntryData,
    ) -> list[dict]:
        proteins = entry_data.get("proteins") or []
        if not proteins:
            return []
        canonical = [cls._canonical_protein_name(str(p)) for p in proteins]
        out: list[dict] = []
        for a in range(len(canonical)):
            for b in range(a + 1, len(canonical)):
                if cls._is_self_alias_pair(a, b, canonical):
                    out.append({
                        "pair": [a, b],
                        "reason": "same_canonical_protein",
                        "canonical_id": canonical[a],
                    })
        return out


    @classmethod
    def _suppress_self_alias_pairs_n(
        cls,
        events: list[CandidateEvent],
        parsed: ParsedSentence,
        entry_data: EntryData,
    ) -> list[dict]:
        canonical = cls._protein_canonical_ids_n(parsed, entry_data)
        if not canonical:
            return []

        suppressed: list[dict] = []
        for event in events:
            kept: set[tuple[int, int]] = set()
            for pair in event.pred_relations or []:
                if len(pair) != 2:
                    continue
                a, b = int(pair[0]), int(pair[1])
                if cls._is_self_alias_pair(a, b, canonical):
                    suppressed.append({
                        "pair": [a, b],
                        "reason": "same_canonical_protein",
                        "canonical_id": canonical[a],
                        "event_detail": getattr(event, "extraction_detail", "") or "",
                    })
                    continue
                kept.add(tuple(sorted((a, b))))
            event.projected_pairs = tuple(sorted(kept))
        return suppressed


    @classmethod
    def _protein_canonical_ids_n(
        cls,
        parsed: ParsedSentence,
        entry_data: EntryData,
    ) -> list[str]:
        return [cls._canonical_protein_name(name) for name in cls._protein_names_n(parsed, entry_data)]


    @staticmethod
    def _protein_names_n(parsed: ParsedSentence, entry_data: EntryData) -> list[str]:
        proteins = entry_data.get("proteins") or []
        if proteins:
            return [str(p) for p in proteins]
        return sorted((parsed.protein_map or {}).keys())


    @staticmethod
    def _canonical_protein_name(name: str) -> str:
        cleaned = re.sub(r"\s+", "", (name or "").strip().lower())
        return cleaned


    @staticmethod
    def _is_self_alias_pair(a: int, b: int, canonical: list[str]) -> bool:
        if a == b or a >= len(canonical) or b >= len(canonical):
            return False
        return bool(canonical[a] and canonical[a] == canonical[b])
