"""Export the public audit workbook and its shared evidence payload."""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from openpyxl import Workbook
    from src.export.event_report import EventReportBuilder
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING
log = logging.getLogger(__name__)
from copy import deepcopy
from .workbook.style import _HAS_OPENPYXL
try:
    import openpyxl
except ImportError:
    openpyxl = None
from .workbook.queries import AuditQueries
from .workbook.payload import AuditPayloadBuilder
from .workbook.render import WorkbookRenderer
from .workbook.display import (
    _anchor_from_semantic_key as _anchor_from_semantic_key,
    _trigger_type_for_construction as _trigger_type_for_construction,
    _argument_nominal_head_text as _argument_nominal_head_text,
    _relational_nominal_from_arguments as _relational_nominal_from_arguments,
    _yaml_chain_for_pattern as _yaml_chain_for_pattern,
    _chain_from_pattern_key as _chain_from_pattern_key,
    _contextual_chain as _contextual_chain,
    _context_trigger_summary_text as _context_trigger_summary_text,
    _carrier_propagation_breakdown as _carrier_propagation_breakdown,
    _yaml_chain_for_relational_statement as _yaml_chain_for_relational_statement,
    _audit_chain_for_relational_statement as _audit_chain_for_relational_statement,
    _lexpath_chain as _lexpath_chain,
    _lexpath_residual_lemmas as _lexpath_residual_lemmas,
    _yaml_scalar_text as _yaml_scalar_text,
)
from .workbook.statistics import (
    _payload_protein_indices as _payload_protein_indices,
    _json_pair_items as _json_pair_items,
    _canonical_pair as _canonical_pair,
    _canonical_pair_set_from_json as _canonical_pair_set_from_json,
    _empty_audit_stats as _empty_audit_stats,
    _merge_audit_stats as _merge_audit_stats,
    _primary_audit_evidence as _primary_audit_evidence,
    _table_columns as _table_columns,
    _optional_sql_column as _optional_sql_column,
    _pattern_chain_groups as _pattern_chain_groups,
)

class ReviewWorkbookBuilder:
    def __init__(
        self,
        bundle: "EventReportBuilder",
        corpus: str,
        metrics: dict,
    ) -> None:
        self.bundle   = bundle
        self.corpus   = corpus or ""
        self.metrics  = metrics or {}
        self.queries = AuditQueries(bundle)
        self.payload_builder = AuditPayloadBuilder(self.queries, self.corpus)
        self.renderer = WorkbookRenderer(self.corpus)
        self._audit_payload = None


    def export(self, path: Path | str) -> Path:
        """Write ``analysis.xlsx`` to *path* and return it; no-op without openpyxl."""
        path = Path(path)
        self._audit_payload = None
        # events.db sits two directories up from analysis/analysis.xlsx
        db_path = path.parent.parent / "db" / "events.db"

        if not _HAS_OPENPYXL:
            log.warning(
                "openpyxl not available -- skipping XLSX export to %s", path
            )
            self._export_pattern_audit_yaml(path.parent / "pattern_audit.yaml", db_path)
            self._export_statistics_ledgers(path.parent / "stats", db_path)
            return path

        try:
            self._export_compact_pattern_audit_workbook(path, db_path, path.parent)
            self._export_pattern_audit_yaml(path.parent / "pattern_audit.yaml", db_path)
            self._export_statistics_ledgers(path.parent / "stats", db_path)
        except Exception:
            log.exception("Error building analysis.xlsx")
        log.info("[ReviewWorkbook] written -> %s", path)
        return path


    def _export_statistics_ledgers(self, output_dir: Path, db_path: Path) -> None:
        """Write machine-oriented CSV ledgers beside the human workbook."""
        try:
            from src.analysis.stats_ledgers import export_statistics_ledgers

            export_statistics_ledgers(db_path, output_dir, corpus=self.corpus)
        except Exception:
            log.exception("Error building statistics ledgers")


    def _export_compact_pattern_audit_workbook(
        self, path: Path, db_path: Path, analysis_dir: Path,
    ) -> Path:
        """Write the five-sheet public audit workbook."""
        audit = self._audit_for(db_path, analysis_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        trigger_rows = self._compact_trigger_word_rows(audit)
        context_rows = audit["entity_boundaries"]
        context_trigger_matrix = audit.get("carrier_trigger_matrix") or {}

        wb: "Workbook" = openpyxl.Workbook()
        wb.remove(wb.active)
        self._build_compact_summary_sheet(wb, audit, trigger_rows, context_rows)
        self._build_compact_trigger_sheet(wb, self._relation_pattern_rows(audit))
        self._build_compact_context_sheet(wb, context_rows, context_trigger_matrix)
        self._build_compact_pattern_sentence_sheet(wb, audit)
        self.renderer.finish(wb, audit)
        path.parent.mkdir(parents=True, exist_ok=True)
        wb.save(str(path))
        log.info("[ReviewWorkbook] compact audit written -> %s", path)
        return path


    def _export_pattern_audit_yaml(self, path: Path, db_path: Path) -> None:
        """Write the concise human-facing trigger/pattern audit YAML."""
        if not db_path.exists():
            return
        try:
            import yaml
        except Exception as exc:  # pragma: no cover
            log.warning("pattern_audit.yaml export skipped: yaml unavailable (%s)", exc)
            return
        payload = deepcopy(self._audit_for(db_path, path.parent))
        # Compact YAML; exact keys stay in the Excel rows.
        for trigger in payload.get("triggers") or []:
            compact_patterns = []
            for pattern in trigger.get("patterns") or []:
                sources: dict[str, int] = {}
                for realization in pattern.pop("realizations", []) or []:
                    source = str(realization.get("source") or "unknown")
                    total = int((realization.get("statistics") or {}).get("total") or 0)
                    sources[source] = int(sources.get(source, 0)) + total
                compact_patterns.append({
                    "chain": pattern.get("chain", ""),
                    "trigger_types": pattern.get("trigger_types") or [],
                    "evidence": pattern.get("evidence") or [],
                    "statistics": pattern.get("statistics") or {},
                    "statistics_by_evidence": pattern.get("statistics_by_evidence") or {},
                    "observed_by_evidence": pattern.get("observed_by_evidence") or {},
                    "sources": dict(sorted(sources.items())),
                    "examples": pattern.get("examples") or [],
                })
            trigger["patterns"] = compact_patterns
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(
                payload, allow_unicode=True, sort_keys=False, width=110,
                default_flow_style=False,
            ),
            encoding="utf-8",
        )


    def _load_pattern_rows(self, *args, **kwargs):
        return self.queries._load_pattern_rows(*args, **kwargs)

    def _load_semantic_pattern_rows(self, *args, **kwargs):
        return self.queries._load_semantic_pattern_rows(*args, **kwargs)

    def _load_pattern_variants_by_semantic_key(self, *args, **kwargs):
        return self.queries._load_pattern_variants_by_semantic_key(*args, **kwargs)

    def _load_candidate_pattern_rows(self, *args, **kwargs):
        return self.queries._load_candidate_pattern_rows(*args, **kwargs)

    def _load_relational_statement_rows(self, *args, **kwargs):
        return self.queries._load_relational_statement_rows(*args, **kwargs)

    def _load_events_without_final_pair_rows(self, *args, **kwargs):
        return self.queries._load_events_without_final_pair_rows(*args, **kwargs)

    def _load_relational_nominal_lookup(self, *args, **kwargs):
        return self.queries._load_relational_nominal_lookup(*args, **kwargs)

    def _load_relational_nominal_lookup_from_raw(self, *args, **kwargs):
        return self.queries._load_relational_nominal_lookup_from_raw(*args, **kwargs)

    def _load_lexpath_candidate_rows(self, *args, **kwargs):
        return self.queries._load_lexpath_candidate_rows(*args, **kwargs)

    def _load_pattern_examples_by_key(self, *args, **kwargs):
        return self.queries._load_pattern_examples_by_key(*args, **kwargs)

    _load_context_trigger_matrix = staticmethod(AuditQueries._load_context_trigger_matrix)

    _load_semantic_evidence_summary = staticmethod(AuditQueries._load_semantic_evidence_summary)

    def _load_observation_rows(self, *args, **kwargs):
        return self.queries._load_observation_rows(*args, **kwargs)

    def _build_pattern_audit_payload(self, *args, **kwargs):
        return self.payload_builder._build_pattern_audit_payload(*args, **kwargs)

    def _compact_trigger_word_rows(self, *args, **kwargs):
        return self.payload_builder._compact_trigger_word_rows(*args, **kwargs)

    def _build_compact_summary_sheet(self, *args, **kwargs):
        return self.renderer._build_compact_summary_sheet(*args, **kwargs)

    def _build_compact_trigger_sheet(self, *args, **kwargs):
        return self.renderer._build_compact_trigger_sheet(*args, **kwargs)

    def _build_compact_context_sheet(self, *args, **kwargs):
        return self.renderer._build_compact_context_sheet(*args, **kwargs)

    def _build_compact_pattern_sentence_sheet(self, *args, **kwargs):
        return self.renderer._build_compact_pattern_sentence_sheet(*args, **kwargs)

    @staticmethod
    def _relation_pattern_rows(audit):
        rows = []
        for trigger in audit.get("triggers") or []:
            for pattern in trigger.get("patterns") or []:
                stats = pattern.get("statistics") or {}
                rows.append({
                    "trigger": trigger.get("trigger", ""),
                    "trigger_type": ", ".join(pattern.get("trigger_types") or []),
                    "total": stats.get("total", 0),
                    "gold_positive": stats.get("gold_positive", 0),
                    "gold_negative": stats.get("gold_negative", 0),
                    "agreement": stats.get("agreement", 0.0),
                    "events_without_final_pair": sum(int(v.get("event_occurrences") or 0) for v in (pattern.get("observed_by_evidence") or {}).get("event_without_final_pair", {}).values()),
                    "evidence": pattern.get("evidence") or [],
                    "pattern_count": pattern.get("chain", ""),
                })
        return rows

    def _audit_for(self, db_path, analysis_dir):
        if self._audit_payload is None:
            audit = self._build_pattern_audit_payload(db_path, analysis_dir)
            # Carrier-only mentions remain visible even without a projected pair.
            rows = [row for row in self._load_observation_rows(db_path, min_count=0)
                    if row.get("category") == "target_context"]
            rows.sort(key=lambda row: (-int(row.get("count_all") or 0), str(row.get("term") or "")))
            audit["entity_boundaries"] = rows
            self._audit_payload = audit
        return self._audit_payload
