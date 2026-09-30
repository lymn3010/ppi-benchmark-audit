from __future__ import annotations

import json
import logging
import platform
import re
import sys
from datetime import datetime
from importlib import metadata as _importlib_metadata
from pathlib import Path

from src.analysis import compute_metrics, print_metrics
from src.analysis.diagnostic_aggregator import write_lexicalized_path_audit
from src.analysis.review_workbook import ReviewWorkbookBuilder
from src.datasets import get_dataset_path
from src.export import EventReportBuilder
from src.models import Record
from src.runtime_resources import load_runtime_resources
from src.models.artifact_contract import CURRENT_ARTIFACT_CONTRACT

from .config import RunConfig
from .pipeline import Pipeline, init_default_pipeline
from .dataset_runner import prepare_entry, run_pipeline_on_dataset, write_workflow_stats
from . import phase_hash as _phase_hash
from . import run_registry as _registry
from .run_identity import RunIdentity, sha256_file

# Conventional output root; registry always lives here regardless of custom output_dir.
_DEFAULT_OUTPUT_ROOT = Path("output")


def _run_config_payload(cfg: RunConfig, *, is_sampled: bool) -> dict:
    """Complete effective configuration that can affect output artifacts."""
    payload = {
        "limit": int(cfg.limit),
        "sample": {
            "requested_size": cfg.sample_size,
            "effective": bool(is_sampled),
            "size": int(cfg.sample_size) if is_sampled and cfg.sample_size is not None else None,
            "seed": int(cfg.sample_seed) if is_sampled else None,
        },
        "coref": bool(cfg.use_coref),
        "nominalization": bool(cfg.use_nominalization),
        "parser_cache": bool(cfg.use_parser_cache),
        "gpu": bool(cfg.use_gpu),
        "parser": cfg.parser_name,
        "stanza_package": cfg.stanza_package,
        "qanom_threshold": float(cfg.qanom_threshold),
        "suppress_self_alias_pairs": bool(cfg.suppress_self_alias_pairs),
        "mine_lexicalized_paths": bool(cfg.mine_lexicalized_paths),
        "emit_event_observations": bool(cfg.emit_event_observations),
        "runtime_rule_toggles": load_runtime_resources().rule_toggle_manifest(),
        "permissive": True,
    }
    return payload


def run_dataset(
    cfg: RunConfig | None = None,
    *,
    output_dir: Path | None = None,
) -> Path | None:
    """Read a corpus and write the evidence database, ledgers and workbook.

    Reuses an identical cached run unless ``force_rerun``.
    """
    cfg = cfg or RunConfig(split="full")
    tag = f"{cfg.dataset}_{cfg.split}"

    print(f"\n{'=' * 60}")
    print(f"  RUN: {tag}")
    print(f"{'=' * 60}\n")

    try:
        dataset_path = get_dataset_path(cfg.dataset, cfg.split)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return None

    # ── Determine the effective sample before constructing run identity ──────
    sample_size = getattr(cfg, "sample_size", None)
    # Sampled only when 0 < sample_size < corpus size (as in dataset_runner).
    corpus_size = None
    if sample_size is not None and sample_size > 0:
        try:
            from src.utils import load_dataset_json
            corpus_size = len(load_dataset_json(dataset_path))
        except Exception:
            corpus_size = None
    is_sampled = (
        sample_size is not None and corpus_size is not None
        and 0 < sample_size < corpus_size
    )
    run_config = _run_config_payload(cfg, is_sampled=is_sampled)
    phase_hashes = _phase_hash.compute_phase_hashes()
    run_identity = RunIdentity(
        mode="run",
        dataset=cfg.dataset,
        split=cfg.split,
        dataset_sha256=sha256_file(dataset_path),
        config=run_config,
        phase_hashes=phase_hashes,
    )
    cfg_hash = run_identity.fingerprint

    # ── Registry cache check (skip if caller supplied a custom output_dir) ────
    if output_dir is None and not cfg.force_rerun:
        cached = _registry.find_cached_run(
            _DEFAULT_OUTPUT_ROOT, cfg.dataset, cfg.split, "run", cfg_hash,
        )
        if cached:
            print(f"[Registry] Reusing cached run ({cfg_hash}): {cached['path']}\n")
            return Path(cached["path"])

    pipeline = _build_pipeline(cfg, permissive=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    split_tag = cfg.split
    if is_sampled:
        # Mark sample output so it is never quoted as a corpus result.
        split_tag = f"{cfg.split}_sample{int(sample_size)}_seed{int(getattr(cfg, 'sample_seed', 42))}"
    output_dir = Path(output_dir) if output_dir else _DEFAULT_OUTPUT_ROOT / cfg.dataset / split_tag / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir / "run.log")

    builder = EventReportBuilder(
        mode="extract",
        dataset=tag,
        config=run_config,
        include_orig_sentence=True,
    )
    builder.enable_raw_jsonl(
        output_dir / "raw",
        include_event_observations=bool(getattr(cfg, "emit_event_observations", True)),
    )

    results, stats = run_pipeline_on_dataset(
        mode="run",
        dataset=cfg.dataset,
        split=cfg.split,
        dataset_path=dataset_path,
        pipeline=pipeline,
        entry_preparer=prepare_entry,
        limit=cfg.limit if cfg.limit > 0 else -1,
        sample_size=int(sample_size) if is_sampled else None,
        sample_seed=int(getattr(cfg, "sample_seed", 42)),
        use_parser_cache=cfg.use_parser_cache,
        event_report_builder=builder,
        record_run_meta=run_config,
        progress_desc="Dataset analysis",
    )

    # Compute raw counts from permissive extraction -- NOT strict F1
    metrics = compute_metrics(results)

    # Export observation files + events.db (builder.export routes subdirs automatically)
    run_meta = {
        "mode": "run",
        "dataset": cfg.dataset,
        "split": cfg.split,
        "split_tag": split_tag,
        "is_sampled": is_sampled,
        "dataset_tag": tag,
        "timestamp": timestamp,
        "run_identity": run_identity.to_dict(),
        **run_config,
    }
    bundle   = builder.finalize()
    builder.export(output_dir, run_meta=run_meta, include_observation_csvs=True)

    # Lexicalized-path support artifact -> analysis/
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    _export_lexpath_audit(
        builder, analysis_dir, source_tag=f"{tag}/run", timestamp=timestamp,
    )

    observation_counts = {category: len(values) for category, values in builder.observations.items()}
    observation_counts["pattern_observations"] = builder.pattern_observations.count_patterns()

    extraction_diagnostics = {
        "predicted_pair_count":    metrics["predicted_pairs"],
        "gold_pair_count":         metrics["gold_pairs"],
        "confusion_matrix":        metrics["confusion_matrix"],
        "event_type_distribution": metrics["event_type_performance"],
    }

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    payload: dict = {
        "run": {
            "mode":         "run",
            "dataset":      cfg.dataset,
            "split":        cfg.split,
            "dataset_tag":  tag,
            "timestamp":    timestamp,
            "config":       run_config,
            "processed":    stats.processed_records,
            "duration_s":   stats.duration_seconds,
        },
        "extraction_diagnostics": extraction_diagnostics,
        "observation_counts":         observation_counts,
    }
    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    # Workbook uses the complete permissive DB.
    wb = ReviewWorkbookBuilder(bundle, corpus=cfg.dataset, metrics=metrics)
    wb.export(analysis_dir / "analysis.xlsx")

    # Workflow stats -> reports/
    write_workflow_stats(
        reports_dir,
        stats,
        extra={
            "config":                 run_config,
            "extraction_diagnostics": extraction_diagnostics,
            "observation_counts":         observation_counts,
        },
    )

    _write_run_json(
        output_dir,
        mode="run", dataset=cfg.dataset, split=cfg.split,
        timestamp=timestamp, config=run_config,
        config_hash=cfg_hash, phase_hashes=phase_hashes,
        run_identity=run_identity.to_dict(),
    )
    _write_artifact_manifest(output_dir)
    _registry.append_run(_DEFAULT_OUTPUT_ROOT, {
        "run_id":       f"{tag}_{timestamp}",
        "mode":         "run",
        "dataset":      cfg.dataset,
        "split":        cfg.split,
        "config_hash":  cfg_hash,
        "phase_hashes": phase_hashes,
        "run_identity": run_identity.to_dict(),
        "path":         str(output_dir),
        "timestamp":    timestamp,
        "duration_s":   stats.duration_seconds,
        "metrics_f1":   None,
    })

    cm = extraction_diagnostics["confusion_matrix"]
    print(f"  [READER AUDIT] projected={extraction_diagnostics['predicted_pair_count']} "
          f"released_positive={extraction_diagnostics['gold_pair_count']} "
          f"TP={cm['tp']} FP={cm['fp']} FN={cm['fn']}")
    print(f"\nObservations recorded:")
    for category, count in observation_counts.items():
        print(f"  {category}: {count} unique")
    print(f"\nOutput: {output_dir}\n")
    return output_dir


def run_sentence(text: str, cfg: RunConfig | None = None,
                 *, discovery: bool = False) -> Record | None:
    """Inspect a single sentence through the pipeline."""

    cfg = cfg or RunConfig()
    pipeline = _build_pipeline(cfg, permissive=discovery)

    pipeline.debug_errors = True
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")
    record = pipeline.process_sentence(text, verbose=True)

    if record:
        record.update_run_meta(
            mode="sentence",
            config={
                "coref": cfg.use_coref,
                "nominalization": cfg.use_nominalization,
                "parser": cfg.parser_name,
            },
        )

    # Reduced semantic tree (display only).
    if record:
        parse = (getattr(record, "artifacts", None) or {}).get("parse", {})
        payload = parse.get("semantic_reduced_tree")
        if payload:
            try:
                from src.analysis.semantic_tree import render_reduced_tree
                print("\n" + " Reduced semantic tree ".center(60, "-"))
                print(render_reduced_tree(payload))
            except Exception as exc:  # never let an inspection view break the run
                print(f"[reduced tree unavailable: {type(exc).__name__}: {exc}]")

    if record and record.predicted_pairs:
        print(f"\nPredicted pairs: {record.predicted_pairs}")
    elif record:
        print("\nNo pairs predicted.")
    return record


# Packages whose versions affect parser or extraction output.
_PROVENANCE_PACKAGES = (
    "stanza",
    "torch",
    "transformers",
    "tokenizers",
    "lemminflect",
)


def _capture_environment() -> dict:
    """Record interpreter and key package versions (fail-soft)."""
    packages: dict[str, str | None] = {}
    for name in _PROVENANCE_PACKAGES:
        try:
            packages[name] = _importlib_metadata.version(name)
        except Exception:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": packages,
    }


def _write_run_json(
    output_dir: Path,
    *,
    mode: str,
    dataset: str,
    split: str,
    timestamp: str,
    config: dict,
    config_hash: str = "",
    phase_hashes: dict | None = None,
    run_identity: dict | None = None,
) -> None:
    """Write run.json with config hash, phase hashes and environment."""
    manifest = {
        "manifest_version": CURRENT_ARTIFACT_CONTRACT.run_manifest_version,
        "artifact_contract": CURRENT_ARTIFACT_CONTRACT.to_dict(),
        "mode": mode,
        "dataset": dataset,
        "split": split,
        "timestamp": timestamp,
        "config": config,
        "config_hash": config_hash,
        "phase_hashes": phase_hashes or {},
        "run_identity": run_identity,
        "environment": _capture_environment(),
        "paths": {
            "db": "db/events.db",
            "records_jsonl": "raw/records.jsonl",
            "candidate_events_jsonl": "raw/candidate_events.jsonl",
            "metrics": "reports/metrics.json",
            "artifact_manifest": "reports/artifact_manifest.json",
            "workflow_stats": "reports/workflow_stats.json",
            "observations_dir": "observations/",
            "analysis_dir": "analysis/",
            "log": "run.log",
        },
    }
    if config.get("emit_event_observations"):
        manifest["paths"]["event_observations_jsonl"] = "raw/event_observations.jsonl"
        manifest["paths"]["mention_graphs_jsonl"] = "raw/mention_graphs.jsonl"
        manifest["paths"]["pair_inference_traces_jsonl"] = "raw/pair_inference_traces.jsonl"
        manifest["paths"]["projection_plans_jsonl"] = "raw/projection_plans.jsonl"
    with open(output_dir / "run.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)


_REQUIRED_RUN_ARTIFACTS = frozenset({
    "db",
    "records_jsonl",
    "candidate_events_jsonl",
    "metrics",
    "workflow_stats",
})


def _write_artifact_manifest(output_dir: Path) -> dict:
    """Inventory run artifacts; missing required ones mark it incomplete."""
    run_json = output_dir / "run.json"
    run_payload = json.loads(run_json.read_text(encoding="utf-8"))
    artifacts: list[dict] = []
    for name, relative in sorted(run_payload.get("paths", {}).items()):
        if name == "artifact_manifest" or str(relative).endswith("/"):
            continue
        path = output_dir / str(relative)
        exists = path.is_file()
        artifacts.append({
            "name": name,
            "path": str(relative).replace("\\", "/"),
            "required": name in _REQUIRED_RUN_ARTIFACTS,
            "status": "present" if exists else "missing_optional",
            "size_bytes": path.stat().st_size if exists else 0,
        })
    for artifact in artifacts:
        if artifact["required"] and artifact["status"] != "present":
            artifact["status"] = "missing_required"
    payload = {
        "schema": "artifact_inventory_v1",
        "artifact_contract": CURRENT_ARTIFACT_CONTRACT.to_dict(),
        "complete": not any(row["status"] == "missing_required" for row in artifacts),
        "artifacts": artifacts,
    }
    target = output_dir / "reports" / "artifact_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def _export_lexpath_audit(
    builder: EventReportBuilder,
    output_dir: Path,
    *,
    source_tag: str,
    timestamp: str,
) -> None:
    """Aggregate mined lexicalized paths into the supporting audit artifact."""
    rows: list[dict] = []
    for _sid, report in builder.sentences.items():
        artifacts = report.artifacts or {}
        for cand in artifacts.get("path_candidates", []) or []:
            rows.append(cand)
    if not rows:
        return
    write_lexicalized_path_audit(
        output_dir / "lexicalized_path_audit.yaml",
        output_dir / "lexicalized_path_audit.json",
        rows,
        source_tag=source_tag,
        timestamp=timestamp,
    )


def _build_pipeline(cfg: RunConfig, *, permissive: bool) -> Pipeline:
    """Build the shared pipeline for dataset or sentence processing."""

    mine_paths = getattr(cfg, "mine_lexicalized_paths", True)
    suppress_alias = getattr(cfg, "suppress_self_alias_pairs", False)
    emit_observations = getattr(cfg, "emit_event_observations", True)
    stanza_pkg = getattr(cfg, "stanza_package", "craft")
    qanom_threshold = float(getattr(cfg, "qanom_threshold", 0.5))

    if permissive:
        # Discovery pass: corpus-independent resources only.
        resources = load_runtime_resources()

        from src.parsing import build_parser_backend
        parser = build_parser_backend(
            cfg.parser_name,
            stanza_package=stanza_pkg,
            use_coref=cfg.use_coref,
            use_nominalization=cfg.use_nominalization,
            qanom_threshold=qanom_threshold,
            use_gpu=getattr(cfg, "use_gpu", True),
        )

        return Pipeline(
            parser=parser,
            resources=resources,
            parser_name=cfg.parser_name,
            mine_lexicalized_paths=mine_paths,
            suppress_self_alias_pairs=suppress_alias,
            emit_event_observations=emit_observations,
        )

    return init_default_pipeline(
        use_coref_resolver=cfg.use_coref,
        use_nominal_detector=cfg.use_nominalization,
        resources=load_runtime_resources(),
        parser_name=cfg.parser_name,
        mine_lexicalized_paths=mine_paths,
        suppress_self_alias_pairs=suppress_alias,
        emit_event_observations=emit_observations,
        stanza_package=stanza_pkg,
        qanom_threshold=qanom_threshold,
        use_gpu=getattr(cfg, "use_gpu", True),
    )


class _CleanFileHandler(logging.FileHandler):
    """File handler that strips ANSI escape codes before writing logs."""

    _ANSI = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        try:
            self.stream.write(self._ANSI.sub("", message) + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def _setup_logging(log_path: Path) -> None:
    """Configure terminal and file logging for one workflow run."""

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = _CleanFileHandler(str(log_path), encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
