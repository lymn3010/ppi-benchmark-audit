#!/usr/bin/env python3
"""Verify dataset JSON against the preserved source XML (ids, text, relations, masks)."""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "datasets"
CLASSIC_CORPORA = ("AIMed", "BioInfer", "HPRD50", "IEPA", "LLL")
PROTEIN_RE = re.compile(r"\bPROTEIN(\d+)\b")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relation_pair(key: str) -> tuple[int, int]:
    value = ast.literal_eval(key)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Bad relation key: {key!r}")
    a, b = int(value[0]), int(value[1])
    return a, b


def _json_relation_count(data: dict[str, dict]) -> int:
    return sum(len(row.get("relations") or {}) for row in data.values())


def _xml_summary(corpus_dir: Path) -> tuple[dict[str, str], int]:
    sentences: dict[str, str] = {}
    interactions = 0
    for xml_path in sorted((corpus_dir / "source_xml").glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        for sentence_el in root.iter("sentence"):
            sid = sentence_el.attrib["id"]
            sentences[sid] = sentence_el.attrib.get("text", "")
            interactions += sum(1 for _ in sentence_el.iter("interaction"))
    return sentences, interactions


def _boundary_and_relation_issues(data: dict[str, dict]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for sid, row in data.items():
        proteins = row.get("proteins") or []
        text = row.get("sentence") or ""
        seen = {int(match.group(1)) for match in PROTEIN_RE.finditer(text)}
        expected = set(range(len(proteins)))
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        if missing or extra:
            issues.append({
                "sentence_id": sid,
                "kind": "protein_mask_boundary",
                "missing": missing,
                "extra": extra,
                "n_proteins": len(proteins),
            })
        for key in (row.get("relations") or {}):
            try:
                a, b = _relation_pair(key)
            except Exception as exc:
                issues.append({
                    "sentence_id": sid,
                    "kind": "bad_relation_key",
                    "key": key,
                    "error": str(exc),
                })
                continue
            if not (0 <= a < len(proteins) and 0 <= b < len(proteins)):
                issues.append({
                    "sentence_id": sid,
                    "kind": "relation_index_out_of_range",
                    "key": key,
                    "n_proteins": len(proteins),
                })
    return issues


def _verify_classic(corpus: str) -> dict[str, Any]:
    corpus_dir = DATASET_ROOT / corpus
    full = _load(corpus_dir / "full.json")
    data = full["data"]
    xml_sentences, xml_interactions = _xml_summary(corpus_dir)
    issues: list[dict[str, Any]] = []

    json_ids = set(data)
    xml_ids = set(xml_sentences)
    missing = sorted(xml_ids - json_ids)
    extra = sorted(json_ids - xml_ids)
    if missing or extra:
        issues.append({
            "kind": "sentence_id_set_mismatch",
            "missing_json_ids": missing[:20],
            "extra_json_ids": extra[:20],
            "n_missing": len(missing),
            "n_extra": len(extra),
        })
    for sid, xml_text in xml_sentences.items():
        if sid in data and data[sid].get("orig_sentence") != xml_text:
            issues.append({
                "kind": "orig_sentence_mismatch",
                "sentence_id": sid,
                "xml": xml_text,
                "json": data[sid].get("orig_sentence"),
            })
            break

    json_relations = _json_relation_count(data)
    if json_relations != xml_interactions:
        issues.append({
            "kind": "interaction_count_mismatch",
            "xml_interactions": xml_interactions,
            "json_relations": json_relations,
        })
    issues.extend(_boundary_and_relation_issues(data))

    return {
        "corpus": corpus,
        "mode": "source_xml",
        "sentences_xml": len(xml_sentences),
        "sentences_json": len(data),
        "interactions_xml": xml_interactions,
        "relations_json": json_relations,
        "issues": issues,
        "ok": not issues,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpora", nargs="+", default=list(CLASSIC_CORPORA))
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = []
    for corpus in args.corpora:
        if corpus not in CLASSIC_CORPORA:
            raise ValueError(f"Unknown corpus {corpus!r}; expected {CLASSIC_CORPORA}")
        report = _verify_classic(corpus)
        reports.append(report)
        print(
            f"{corpus}: ok={report['ok']} "
            f"issues={len(report['issues'])} mode={report['mode']}"
        )
        for issue in report["issues"][:5]:
            print(f"  - {issue}")

    payload = {"reports": reports, "ok": all(report["ok"] for report in reports)}
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {args.output_json}")

    return 0 if all(report["ok"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
