from __future__ import annotations

import gc
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from tqdm import tqdm

from src.models import Record
from src.utils import (
    extract_gold_pairs,
    load_dataset_json,
    load_parsed_sentence_cache,
    parse_gold_pairs,
    parse_relation_keys,
)

if TYPE_CHECKING:
    from .pipeline import Pipeline


@dataclass
class PreparedEntry:
    """Normalized dataset entry ready for pipeline processing."""

    text: str
    gold_pairs: list[tuple[int, int]] | None = None
    entry_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowStats:
    """Shared run statistics for infer and extract workflows."""

    mode: str
    dataset: str
    split: str
    dataset_path: str
    started_at: str
    finished_at: str = ""
    duration_seconds: float = 0.0

    total_entries: int = 0
    attempted_entries: int = 0
    processed_records: int = 0
    skipped_invalid_entries: int = 0
    failed_entries: int = 0

    cache_hits: int = 0
    records_with_events: int = 0
    records_without_events: int = 0
    total_events: int = 0
    total_predicted_pairs: int = 0

    skip_reasons: dict[str, int] = field(default_factory=dict)
    fail_reasons: dict[str, int] = field(default_factory=dict)

    # Sampling provenance; None for full runs.
    sample_size: int | None = None
    sample_seed: int | None = None
    sample_entry_ids: list[str] | None = None

    _started_dt: datetime = field(default_factory=datetime.now, repr=False, compare=False)

    @classmethod
    def start(cls, mode: str, dataset: str, split: str, dataset_path: Path) -> "WorkflowStats":
        now = datetime.now()
        return cls(
            mode=mode,
            dataset=dataset,
            split=split,
            dataset_path=str(dataset_path),
            started_at=now.strftime("%Y-%m-%d %H:%M:%S"),
            _started_dt=now,
        )

    def finish(self) -> None:
        finished_dt = datetime.now()
        self.finished_at = finished_dt.strftime("%Y-%m-%d %H:%M:%S")
        self.duration_seconds = round((finished_dt - self._started_dt).total_seconds(), 3)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("_started_dt", None)
        return payload


def prepare_entry(entry_id: str, entry: dict[str, Any]) -> PreparedEntry:
    """Normalize a raw dataset entry across supported dataset shapes."""

    text = entry.get("sentence") or entry.get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("invalid_text")

    gold_pairs = None
    relations = entry.get("relations")
    if isinstance(relations, dict):
        try:
            parsed = parse_relation_keys(relations)
            gold_pairs = extract_gold_pairs(parsed)
        except (TypeError, ValueError):
            gold_pairs = list(parse_gold_pairs(relations))
    elif isinstance(relations, list):
        gold_pairs = list(parse_gold_pairs(relations))

    return PreparedEntry(
        text=text,
        gold_pairs=gold_pairs,
        entry_info={
            "orig_sentence": entry.get("orig_sentence", ""),
            "proteins": entry.get("proteins", []),
        },
    )


def run_pipeline_on_dataset(
    *,
    mode: str,
    dataset: str,
    split: str,
    dataset_path: Path,
    pipeline: "Pipeline",
    entry_preparer: Callable[[str, dict[str, Any]], PreparedEntry],
    limit: int = -1,
    sample_size: int | None = None,
    sample_seed: int = 42,
    use_parser_cache: bool = True,
    event_report_builder: Any | None = None,
    record_run_meta: dict[str, Any] | None = None,
    on_record: Callable[[str, dict[str, Any], PreparedEntry, Record], None] | None = None,
    progress_desc: str = "Processing",
    dataset_entries: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[Record], WorkflowStats]:
    """Run the pipeline over a dataset and collect workflow statistics."""

    dataset_dict = (
        dict(dataset_entries) if dataset_entries is not None
        else load_dataset_json(dataset_path)
    )

    # Optional seeded sample in corpus order (validation only).
    sample_entry_ids: list[str] | None = None
    if sample_size is not None and 0 < sample_size < len(dataset_dict):
        all_ids = list(dataset_dict.keys())
        rng = random.Random(sample_seed)
        chosen = set(rng.sample(all_ids, sample_size))
        sample_entry_ids = [eid for eid in all_ids if eid in chosen]
        dataset_dict = {eid: dataset_dict[eid] for eid in sample_entry_ids}

    stats = WorkflowStats.start(mode=mode, dataset=dataset, split=split, dataset_path=dataset_path)
    if sample_entry_ids is not None:
        stats.sample_size = len(sample_entry_ids)
        stats.sample_seed = sample_seed
        stats.sample_entry_ids = sample_entry_ids
    stats.total_entries = len(dataset_dict)

    # Only the parser-neutral Stanza cache is used.
    parsed_cache = (
        load_parsed_sentence_cache(dataset_path, pipeline.parser)
        if use_parser_cache
        else {}
    )
    results: list[Record] = []

    total = len(dataset_dict) if limit < 1 else min(limit, len(dataset_dict))
    for index, (entry_id, entry) in enumerate(
        tqdm(dataset_dict.items(), total=total, desc=progress_desc)
    ):
        if 0 < limit <= index:
            break

        stats.attempted_entries += 1

        if not isinstance(entry, dict):
            _bump(stats.skip_reasons, "entry_not_dict")
            stats.skipped_invalid_entries += 1
            continue

        try:
            prepared = entry_preparer(entry_id, entry)
        except ValueError as exc:
            _bump(stats.skip_reasons, str(exc))
            stats.skipped_invalid_entries += 1
            continue
        except Exception as exc:
            _bump(stats.fail_reasons, f"prepare:{type(exc).__name__}")
            stats.failed_entries += 1
            continue

        if not isinstance(prepared, PreparedEntry):
            _bump(stats.fail_reasons, "prepare_return_type")
            stats.failed_entries += 1
            continue

        text = prepared.text.strip() if isinstance(prepared.text, str) else ""
        if not text:
            _bump(stats.skip_reasons, "missing_text")
            stats.skipped_invalid_entries += 1
            continue

        entry_data = {
            "id": entry_id,
            "text": text,
            "gold_pairs": prepared.gold_pairs,
            "proteins": prepared.entry_info.get("proteins", []),
        }

        try:
            if entry_id in parsed_cache:
                result = pipeline.process_parsed(parsed_cache[entry_id], entry_data=entry_data, verbose=False)
                stats.cache_hits += 1
            else:
                result = pipeline.process_sentence(text, entry_data=entry_data, verbose=False)
        except Exception as exc:
            _bump(stats.fail_reasons, f"process:{type(exc).__name__}")
            stats.failed_entries += 1
            continue

        if not isinstance(result, Record):
            _bump(stats.fail_reasons, "empty_record")
            stats.failed_entries += 1
            continue

        result.update_run_meta(mode=mode, config=record_run_meta or {})

        if event_report_builder is not None:
            event_report_builder.add_record(result, entry_info=prepared.entry_info)

        if on_record is not None:
            on_record(entry_id, entry, prepared, result)

        results.append(result)
        stats.processed_records += 1

        event_count = len(result.events) if result.events else 0
        if event_count:
            stats.records_with_events += 1
        else:
            stats.records_without_events += 1

        stats.total_events += event_count
        stats.total_predicted_pairs += len(result.predicted_pairs)

        if stats.processed_records and stats.processed_records % 1000 == 0:
            gc.collect()

    stats.finish()
    return results, stats


def write_workflow_stats(
    output_dir: Path,
    stats: WorkflowStats,
    extra: dict[str, Any] | None = None,
    filename: str = "workflow_stats.json",
) -> Path:
    """Write workflow statistics to the run output directory."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = stats.to_dict()
    if extra:
        payload["extra"] = extra

    path = output_dir / filename
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    return path


def _bump(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1
