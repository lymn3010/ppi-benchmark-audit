"""Cache of ParsedSentence objects so repeated runs skip the parser."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from src.parsing.syntax import BackendInfo, CorefCluster, ParsedSentence, SyntaxNode
from src.system_rules import rule_frozenset, system_rule


PARSER_CACHE_SCHEMA_VERSION = "ppi_parsed_sentence_cache_v4"


def resolve_parsed_sentence_cache(dataset_path: Path, parser) -> tuple[Path, Path]:
    """Return ``(records_jsonl, manifest_json)`` for this dataset/parser."""
    folder = Path(dataset_path).parent
    dataset_name = folder.name
    root_value = os.environ.get("PPI_PARSER_CACHE_ROOT")
    root = Path(root_value) if root_value else folder / ".parser_cache"
    profile = _parser_profile(parser)
    cache_dir = root / dataset_name / profile
    return cache_dir / "parsed_sentences.jsonl", cache_dir / "manifest.json"


def load_parsed_sentence_cache(dataset_path: Path, parser) -> dict[str, ParsedSentence]:
    """Load a validated parser-neutral cache, or return an empty dict."""
    records_path, manifest_path = resolve_parsed_sentence_cache(dataset_path, parser)
    if not records_path.exists() or not manifest_path.exists():
        return {}

    ok, reasons = validate_parsed_sentence_manifest(dataset_path, parser, manifest_path)
    if not ok:
        print(f"[ParsedCache] Ignoring stale cache {records_path}: {'; '.join(reasons)}")
        return {}

    out: dict[str, ParsedSentence] = {}
    try:
        with records_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                parsed = parsed_sentence_from_dict(row["parsed_sentence"])
                if parsed.sentence_id:
                    out[parsed.sentence_id] = parsed
    except Exception as exc:
        print(f"[ParsedCache] Ignoring unreadable cache {records_path}: {exc}")
        return {}

    expected_count = int(
        json.loads(manifest_path.read_text(encoding="utf-8")).get("record_count", -1)
    )
    if len(out) != expected_count:
        print(
            f"[ParsedCache] Ignoring incomplete cache {records_path}: "
            f"loaded {len(out)} unique records, expected {expected_count}"
        )
        return {}

    print(f"[ParsedCache] Loaded {len(out)} ParsedSentence records from {records_path}")
    return out


def write_parsed_sentence_cache(
    *,
    dataset_name: str,
    source_json: Path,
    parser,
    records: list[ParsedSentence],
    output_records: Path,
    stats: dict[str, Any] | None = None,
) -> Path:
    """Write records JSONL plus a validating manifest."""
    output_records.parent.mkdir(parents=True, exist_ok=True)
    with output_records.open("w", encoding="utf-8") as fh:
        for parsed in records:
            fh.write(json.dumps({
                "id": parsed.sentence_id,
                "parsed_sentence": parsed_sentence_to_dict(parsed),
            }, ensure_ascii=False, sort_keys=True))
            fh.write("\n")

    manifest = build_parsed_sentence_manifest(
        dataset_name=dataset_name,
        source_json=source_json,
        parser=parser,
        records_path=output_records,
        record_count=len(records),
        stats=stats or {},
    )
    manifest_path = output_records.with_name("manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def build_parsed_sentence_manifest(
    *,
    dataset_name: str,
    source_json: Path,
    parser,
    records_path: Path,
    record_count: int,
    stats: dict[str, Any],
) -> dict[str, Any]:
    source_json = Path(source_json)
    return {
        "schema_version": PARSER_CACHE_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_name,
        "source_json": str(source_json.resolve()),
        "source_parse_sha256": _parse_source_sha256(source_json),
        "source_sha256": _sha256_file(source_json),
        "records_file": records_path.name,
        "records_sha256": _sha256_file(records_path),
        "record_count": int(record_count),
        "parser_profile": _parser_profile(parser),
        "parser_config": _parser_config(parser),
        "code_fingerprint": _code_fingerprint(),
        "git_commit": _git_commit(),
        "stats": stats,
    }


def validate_parsed_sentence_manifest(
    dataset_path: Path,
    parser,
    manifest_path: Path,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"manifest_unreadable:{type(exc).__name__}"]

    if manifest.get("schema_version") != PARSER_CACHE_SCHEMA_VERSION:
        reasons.append("schema_version")
    if manifest.get("parser_profile") != _parser_profile(parser):
        reasons.append("parser_profile")
    if manifest.get("parser_config") != _parser_config(parser):
        reasons.append("parser_config")

    source_json = Path(dataset_path).parent / "full.json"
    if source_json.exists():
        if "source_parse_sha256" in manifest:
            if manifest.get("source_parse_sha256") != _parse_source_sha256(source_json):
                reasons.append("source_parse_sha256")
        elif manifest.get("source_sha256") != _sha256_file(source_json):
            reasons.append("source_sha256")
    else:
        reasons.append("source_json_missing")

    current_fp = _code_fingerprint()
    if manifest.get("code_fingerprint") != current_fp:
        reasons.append("code_fingerprint")

    records_path = Path(manifest_path).with_name(str(manifest.get("records_file", "parsed_sentences.jsonl")))
    if not records_path.exists():
        reasons.append("records_missing")
    else:
        if manifest.get("records_sha256") != _sha256_file(records_path):
            reasons.append("records_sha256")
        try:
            actual_count = sum(
                1
                for line in records_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except (OSError, UnicodeError):
            reasons.append("records_unreadable")
        else:
            if manifest.get("record_count") != actual_count:
                reasons.append("record_count")

    return not reasons, reasons


def parsed_sentence_to_dict(parsed: ParsedSentence) -> dict[str, Any]:
    return {
        "text": parsed.text,
        "sentence_id": parsed.sentence_id,
        "nodes": [asdict(node) for node in parsed.nodes],
        "protein_map": {k: list(v) for k, v in parsed.protein_map.items()},
        "coref_clusters": [asdict(cluster) for cluster in parsed.coref_clusters],
        "parser_name": parsed.parser_name,
        "raw_parser_data": parsed.raw_parser_data,
        "backend_info": asdict(parsed.backend_info),
    }


def parsed_sentence_from_dict(payload: dict[str, Any]) -> ParsedSentence:
    nodes = tuple(
        SyntaxNode(
            **{
                **row,
                "protein_indices": tuple(row.get("protein_indices", ())),
                "direct_protein_indices": tuple(row.get("direct_protein_indices", ())),
                "inherited_protein_indices": tuple(row.get("inherited_protein_indices", ())),
                "inherited_reference_kinds": tuple(row.get("inherited_reference_kinds", ())),
                "inherited_reference_node_indices": tuple(
                    row.get("inherited_reference_node_indices", ())
                ),
                "multiword_token_id": tuple(row.get("multiword_token_id", ())),
                "native_id": tuple(row.get("native_id", ())),
                "enhanced_heads": tuple(
                    (int(pair[0]), str(pair[1]))
                    for pair in row.get("enhanced_heads", ())
                    if isinstance(pair, (tuple, list)) and len(pair) >= 2
                ),
            }
        )
        for row in payload.get("nodes", [])
    )
    clusters = tuple(
        CorefCluster(
            cluster_id=int(row.get("cluster_id", 0)),
            mentions=tuple(tuple(pair) for pair in row.get("mentions", ())),
            evidence_type=row.get("evidence_type", "nominal"),
            head_text=row.get("head_text", ""),
            source=row.get("source", ""),
            confidence=row.get("confidence"),
            mention_head_indices=tuple(row.get("mention_head_indices", ())),
        )
        for row in payload.get("coref_clusters", [])
    )
    raw_parser_data = dict(payload.get("raw_parser_data", {}))
    nominalization_sources = raw_parser_data.get("nominalization_sources")
    if isinstance(nominalization_sources, dict):
        raw_parser_data["nominalization_sources"] = {
            int(key) if str(key).lstrip("-").isdigit() else key: value
            for key, value in nominalization_sources.items()
        }
    backend_payload = dict(payload.get("backend_info", {}))
    backend_payload["capabilities"] = tuple(backend_payload.get("capabilities", ()))
    backend_payload["enrichments"] = tuple(backend_payload.get("enrichments", ()))
    return ParsedSentence(
        text=payload.get("text", ""),
        sentence_id=payload.get("sentence_id", ""),
        nodes=nodes,
        protein_map={k: tuple(v) for k, v in payload.get("protein_map", {}).items()},
        coref_clusters=clusters,
        parser_name=payload.get("parser_name", ""),
        raw_parser_data=raw_parser_data,
        backend_info=BackendInfo(**backend_payload),
    )


def _parser_profile(parser) -> str:
    name = type(parser).__name__.lower()
    if name == "stanzaadapter":
        package = getattr(parser, "package", "craft")
        nominalization = (
            f"qanom{getattr(parser, 'qanom_threshold', 0.5):g}"
            if getattr(parser, "use_nominalization", False)
            else "nominaloff"
        )
        coref = "coref" if getattr(parser, "use_coref", False) else "nocoref"
        gpu = "gpu" if getattr(parser, "use_gpu", True) else "cpu"
        segmentation = (
            "nosplit" if getattr(parser, "sentence_is_presegmented", True)
            else "ssplit"
        )
        return f"stanza_{package}_{nominalization}_{coref}_{gpu}_{segmentation}"
    return name


def _parser_config(parser) -> dict[str, Any]:
    if type(parser).__name__.lower() == "stanzaadapter":
        config = {
            "backend": "stanza",
            "stanza_version": _package_version("stanza"),
            "package": getattr(parser, "package", "craft"),
            "use_coref": bool(getattr(parser, "use_coref", False)),
            "coref_package": (
                "udcoref_xlm-roberta-lora"
                if getattr(parser, "use_coref", False)
                else None
            ),
            "nominalization_detector": (
                "QANom" if getattr(parser, "use_nominalization", False) else "off"
            ),
            "qanom_threshold": getattr(parser, "qanom_threshold", None),
            "qanom_model": (
                _qanom_model_identity()
                if getattr(parser, "use_nominalization", False)
                else None
            ),
            "torch_version": (
                _package_version("torch")
                if getattr(parser, "use_nominalization", False)
                or getattr(parser, "use_coref", False)
                else ""
            ),
            "transformers_version": (
                _package_version("transformers")
                if getattr(parser, "use_nominalization", False)
                or getattr(parser, "use_coref", False)
                else ""
            ),
            "use_gpu": bool(getattr(parser, "use_gpu", True)),
            "sentence_is_presegmented": bool(
                getattr(parser, "sentence_is_presegmented", True)
            ),
            "target_pattern": getattr(
                getattr(parser, "target_regex", None), "pattern", ""
            ),
            "stanza_models": _stanza_model_identity(parser),
        }
        return config
    return {"backend": type(parser).__name__}


def _code_fingerprint() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = [
        "src/parsing/syntax.py",
        "src/parsing/role_normalizer.py",
        "src/parsing/stanza/stanza_adapter.py",
        "src/parsing/stanza/qanom_detector.py",
        "src/extraction/parsers/dep/morphology.py",
        "src/models/candidate/_nominal_canon.py",
        "src/system_rules.py",
        "src/utils/parsed_sentence_cache.py",
        "data/lexicons/nominal_to_verbal.csv",
    ]
    fingerprints = {
        rel: _sha256_file(root / rel)
        for rel in paths
        if (root / rel).exists()
    }
    # Only parser-boundary rules belong in the cache identity.
    parser_rules = {
        "masked_protein_pattern": system_rule(
            "surface_contract.masked_protein_pattern"
        ),
        "pronoun_identity_forms": sorted(
            rule_frozenset("reference.pronoun_identity_forms")
        ),
    }
    fingerprints["system_rules:parser_contract"] = hashlib.sha256(
        json.dumps(parser_rules, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return fingerprints


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _qanom_model_identity() -> dict[str, str]:
    from src.parsing.stanza.qanom_detector import QANomDetector

    model_name = QANomDetector.MODEL_NAME
    revision = ""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        model_folder = "models--" + model_name.replace("/", "--")
        revision_path = Path(HF_HUB_CACHE) / model_folder / "refs" / "main"
        if revision_path.exists():
            revision = revision_path.read_text(encoding="utf-8").strip()
    except (ImportError, OSError, UnicodeError):
        pass
    return {"name": model_name, "cached_revision": revision}


def _stanza_model_identity(parser) -> dict[str, Any]:
    """Model identity from Stanza's declared checksum and file size (no model load)."""
    try:
        from stanza.resources.common import DEFAULT_MODEL_DIR

        root = Path(DEFAULT_MODEL_DIR)
        resources_path = root / "resources.json"
        resources = json.loads(resources_path.read_text(encoding="utf-8"))
        english = resources["en"]
    except (ImportError, KeyError, OSError, TypeError, json.JSONDecodeError):
        return {}

    package_name = getattr(parser, "package", "craft")
    selected = dict(english.get("packages", {}).get(package_name, {}))
    selected = {
        processor: model
        for processor, model in selected.items()
        if processor in {"tokenize", "pos", "lemma", "depparse"}
    }
    if getattr(parser, "use_coref", False):
        selected["coref"] = "udcoref_xlm-roberta-lora"

    models: dict[str, Any] = {}
    for processor, model in sorted(selected.items()):
        metadata = english.get(processor, {}).get(model, {})
        model_path = root / "en" / processor / f"{model}.pt"
        models[processor] = {
            "package": model,
            "declared_md5": metadata.get("md5", ""),
            "file_size": model_path.stat().st_size if model_path.exists() else None,
        }
    return {
        "resources_sha256": _sha256_file(resources_path),
        "models": models,
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_source_sha256(path: Path) -> str:
    """Fingerprint only the dataset fields that affect parser output."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        raise ValueError(f"Dataset JSON must contain a dict: {path}")
    parse_rows = [
        {
            "id": str(entry_id),
            "text": (entry.get("sentence") or entry.get("text") or "")
            if isinstance(entry, dict)
            else "",
        }
        for entry_id, entry in data.items()
    ]
    return hashlib.sha256(
        json.dumps(parse_rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return ""
