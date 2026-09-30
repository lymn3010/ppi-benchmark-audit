"""Excel views of saved audit evidence; no SQL or inference here."""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from openpyxl import Workbook
import logging
log = logging.getLogger(__name__)
from .display import _context_trigger_summary_text, _carrier_propagation_breakdown
from .style import (
    _f,
    _fill,
    _align,
    _prec_color,
    _write_header_row,
    _write_section_header,
    _write_kv_row,
    _write_data_row,
    _set_col_widths,
    _set_autofilter,
    _tab_color,
)
from .style import _C, _BORDER, _FMT_INT, _FMT_PREC

from .display import public_label, public_labels


class WorkbookRenderer:
    def __init__(self, corpus):
        self.corpus = corpus

    def _build_compact_summary_sheet(
        self,
        wb: "Workbook",
        audit: dict,
        trigger_rows: list[dict],
        context_rows: list[dict],
    ) -> None:
        ws = wb.create_sheet("Summary")
        _tab_color(ws, "1F4E79")
        n_cols = 6
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=n_cols)
        title = ws.cell(row=1, column=1, value=f"Relation pattern and entity boundary audit - {self.corpus}")
        title.font = _f(14, bold=True, color="FFFFFF")
        title.fill = _fill(_C["hdr_dark"])
        title.alignment = _align("center", "center")
        title.border = _BORDER
        row = 3
        meta = audit.get("meta") or {}
        totals = [
            ("Corpus", self.corpus),
            ("Source DB", meta.get("source_db", "")),
            ("Trigger words", len(trigger_rows)),
            ("Carrier words", len(context_rows)),
            ("Readable patterns", sum(len(t.get("patterns") or []) for t in audit.get("triggers") or [])),
            ("Reading order", "Summary → patterns / boundaries → sentence evidence"),
        ]
        for label, value in totals:
            _write_kv_row(ws, row, n_cols, label, value)
            row += 1

        evidence_summary = audit.get("evidence_summary") or {}
        if evidence_summary:
            row += 1
            _write_section_header(ws, row, n_cols, "Semantic Evidence Inventory", "mid")
            row += 1
            for key, label in (
                ("candidate_events", "Candidate events"),
                ("candidate_relations", "Candidate relation proposals"),
                ("identity_equivalences", "Applied identity-equivalence edges"),
                ("self_alias_candidate_pairs", "Self-alias review candidates"),
                ("suppressed_pairs", "Explicitly suppressed pairs"),
                ("path_candidates", "Lexicalized-path candidates"),
                ("projection_plans", "Canonical projection plans"),
                ("pair_inference_traces", "Pair inference traces"),
            ):
                _write_kv_row(ws, row, n_cols, label, int(evidence_summary.get(key) or 0))
                row += 1

        row += 1
        _write_section_header(ws, row, n_cols, "Top relation triggers", "mid")
        row += 1
        _write_header_row(
            ws, row,
            ["Trigger", "Pair Observations", "Released +", "Not released +", "Released-positive rate", "Events Without Final Pair"],
        )
        row += 1
        for item in trigger_rows[:15]:
            _write_data_row(
                ws, row,
                [
                    item["trigger"], item["total"], item["gold_positive"],
                    item["gold_negative"], item["agreement"],
                    item["events_without_final_pair"],
                ],
                fill_hex=_prec_color(float(item["agreement"])),
                formats=[None, _FMT_INT, _FMT_INT, _FMT_INT, _FMT_PREC, _FMT_INT],
                aligns=["left", "right", "right", "right", "right", "right"],
            )
            row += 1

        row += 1
        _write_section_header(ws, row, n_cols, "Top carriers", "mid")
        row += 1
        _write_header_row(ws, row, ["Carrier", "Total", "Released +", "Not released +", "Released-positive rate", "All Mentions"])
        row += 1
        for item in context_rows[:15]:
            agreement = float(item.get("precision") or 0.0)
            _write_data_row(
                ws, row,
                [
                    item.get("term", ""), int(item.get("total") or 0),
                    int(item.get("tp") or 0), int(item.get("fp") or 0),
                    agreement, int(item.get("count_all") or 0),
                ],
                fill_hex=_prec_color(agreement),
                formats=[None, _FMT_INT, _FMT_INT, _FMT_INT, _FMT_PREC, _FMT_INT],
                aligns=["left", "right", "right", "right", "right", "right"],
            )
            row += 1
        _set_col_widths(ws, [34, 20, 16, 20, 26, 28])
        ws.freeze_panes = "A3"


    def _build_compact_trigger_sheet(self, wb: "Workbook", rows: list[dict]) -> None:
        ws = wb.create_sheet("Relation patterns")
        _tab_color(ws, "70AD47")
        headers = [
            "#", "Trigger", "Trigger Type", "Pair Observations", "Released +", "Not released +",
            "Released-positive rate", "Events Without Final Pair", "Evidence", "Relation pattern",
        ]
        _write_header_row(ws, 1, headers)
        for idx, row in enumerate(rows, 2):
            agreement = float(row.get("agreement") or 0.0)
            _write_data_row(
                ws, idx,
                [
                    idx - 1, row.get("trigger", ""), public_label(row.get("trigger_type", "")),
                    int(row.get("total") or 0),
                    int(row.get("gold_positive") or 0),
                    int(row.get("gold_negative") or 0), agreement,
                    int(row.get("events_without_final_pair") or 0),
                    public_labels(row.get("evidence") or []),
                    row.get("pattern_count", ""),
                ],
                fill_hex=_prec_color(agreement),
                formats=[None, None, None, _FMT_INT, _FMT_INT, _FMT_INT, _FMT_PREC, _FMT_INT, None, _FMT_INT],
                aligns=["right", "left", "left", "right", "right", "right", "right", "right", "left", "right"],
            )
        _set_col_widths(ws, [6, 24, 20, 20, 16, 18, 24, 25, 34, 65])
        _set_autofilter(ws, len(headers), 1)
        ws.freeze_panes = "B2"


    def _build_compact_context_sheet(
        self,
        wb: "Workbook",
        rows: list[dict],
        context_trigger_matrix: dict[str, list[dict]] | None = None,
    ) -> None:
        ws = wb.create_sheet("Entity boundaries")
        _tab_color(ws, "FFC000")
        headers = [
            "#", "Carrier Word", "Pair Observations", "Released +", "Not released +",
            "Released-positive rate", "Carrier Mentions", "Pair-Contributing Mentions",
            "Evidence-Only Mentions", "Propagation Breakdown", "Top Trigger Patterns",
        ]
        _write_header_row(ws, 1, headers)
        context_trigger_matrix = context_trigger_matrix or {}
        for idx, row in enumerate(rows, 2):
            agreement = float(row.get("precision") or 0.0)
            paired = context_trigger_matrix.get(str(row.get("term") or ""), [])
            _write_data_row(
                ws, idx,
                [
                    idx - 1, row.get("term", ""), int(row.get("total") or 0),
                    int(row.get("tp") or 0), int(row.get("fp") or 0), agreement,
                    int(row.get("count_all") or 0),
                    int(row.get("pair_contributing_occurrences") or 0),
                    int(row.get("nonprojecting_occurrences") or 0),
                    _carrier_propagation_breakdown(row),
                    _context_trigger_summary_text(paired),
                ],
                fill_hex=_prec_color(agreement),
                formats=[
                    None, None, _FMT_INT, _FMT_INT, _FMT_INT, _FMT_PREC,
                    _FMT_INT, _FMT_INT, _FMT_INT, None, None,
                ],
                aligns=[
                    "right", "left", "right", "right", "right", "right",
                    "right", "right", "right", "left", "left",
                ],
            )
        _set_col_widths(ws, [6, 24, 20, 16, 18, 24, 20, 24, 24, 38, 62])
        _set_autofilter(ws, len(headers), 1)
        ws.freeze_panes = "B2"


    def _build_compact_pattern_sentence_sheet(self, wb: "Workbook", audit: dict) -> None:
        ws = wb.create_sheet("Sentence evidence")
        _tab_color(ws, "00A6A6")
        ws.sheet_properties.outlinePr.summaryBelow = False
        headers = [
            "Level", "Trigger", "Trigger Type", "Pattern", "Evidence", "Total", "Released +",
            "Not released +", "Released-positive rate", "Breakdown", "Sentence / Example",
        ]
        _write_header_row(ws, 1, headers, height=24)
        _set_col_widths(ws, [12, 20, 20, 46, 30, 14, 16, 18, 24, 44, 90])
        row_i = 2
        for trigger in audit.get("triggers") or []:
            stats = trigger.get("statistics") or {}
            trigger_row = row_i
            _write_data_row(
                ws, row_i,
                [
                    "TRIGGER", trigger.get("trigger", ""),
                    public_labels(trigger.get("trigger_types") or []),
                    "",
                    public_labels(trigger.get("evidence") or []),
                    int(stats.get("total") or 0),
                    int(stats.get("gold_positive") or 0),
                    int(stats.get("gold_negative") or 0),
                    float(stats.get("agreement") or 0.0),
                    "\n".join(
                        f"{name}: {int(item.get('total') or 0)} ({float(item.get('agreement') or 0.0):.3f})"
                        for name, item in sorted((trigger.get("statistics_by_evidence") or {}).items())
                    ),
                    "",
                ],
                fill_hex=_C["hdr_light"],
                formats=[None, None, None, None, None, _FMT_INT, _FMT_INT, _FMT_INT, _FMT_PREC],
                aligns=["left", "left", "left", "left", "left", "right", "right", "right", "right"],
                height=26,
            )
            for cell in ws[trigger_row]:
                cell.font = _f(10, bold=True)
            row_i += 1

            for pattern in trigger.get("patterns") or []:
                pstats = pattern.get("statistics") or {}
                pattern_row = row_i
                _write_data_row(
                    ws, row_i,
                    [
                        "PATTERN", trigger.get("trigger", ""),
                        public_labels(pattern.get("trigger_types") or []),
                        pattern.get("chain", ""), public_labels(pattern.get("evidence") or []),
                        int(pstats.get("total") or 0),
                        int(pstats.get("gold_positive") or 0),
                        int(pstats.get("gold_negative") or 0),
                        float(pstats.get("agreement") or 0.0),
                        "\n".join(
                            f"{evidence}/{kind}: pairs={int(item.get('total') or 0)}; "
                            f"events={int(item.get('event_occurrences') or 0)}; "
                            f"released-positive rate={float(item.get('agreement') or 0.0):.3f}"
                            for evidence, observed in sorted((pattern.get("observed_by_evidence") or {}).items())
                            for kind, item in sorted(observed.items())
                        ),
                        "",
                    ],
                    fill_hex=_prec_color(float(pstats.get("agreement") or 0.0)),
                    formats=[None, None, None, None, None, _FMT_INT, _FMT_INT, _FMT_INT, _FMT_PREC],
                    height=34,
                )
                ws.row_dimensions[row_i].outlineLevel = 1
                ws.row_dimensions[row_i].hidden = False
                ws.row_dimensions[row_i].collapsed = True
                row_i += 1
                for example in pattern.get("examples") or []:
                    _write_data_row(
                        ws, row_i,
                        [
                            "SENTENCE", trigger.get("trigger", ""),
                            public_labels(pattern.get("trigger_types") or []),
                            pattern.get("chain", ""), public_label(example.get("status", "")),
                            "", "", "", "",
                            (
                                f"{example.get('sentence_id', '')} pair={example.get('pair', [])}\n"
                                f"assertion={example.get('assertion_status', '')}; "
                                f"propagation={','.join(example.get('propagation') or [])}; "
                                f"ambiguity={example.get('ambiguity_reason', '')}; "
                                f"proteins={example.get('protein_indices', [])}"
                            ),
                            example.get("text", ""),
                        ],
                        fill_hex=_C["white"],
                        height=42,
                    )
                    ws.row_dimensions[row_i].outlineLevel = 2
                    ws.row_dimensions[row_i].hidden = False
                    row_i += 1
                ws.row_dimensions[pattern_row].collapsed = True
            ws.row_dimensions[trigger_row].collapsed = True
        _set_autofilter(ws, len(headers), 1)
        ws.freeze_panes = "D2"


    def finish(self, wb, audit):
        """Link summary, patterns and sentence evidence, and document provenance."""
        from math import ceil
        from openpyxl.utils import get_column_letter
        ws = wb.create_sheet("Reproduction")
        _write_header_row(ws, 1, ["Item", "Value"])
        rows = [
            ("Report", "Dependency-based reader and corpus annotation audit"),
            ("Source database", (audit.get("meta") or {}).get("source_db", "")),
            ("Corpus", self.corpus),
            ("Machine-readable evidence", "audit.json; pattern_audit.yaml; stats/*.csv"),
            ("Released-positive rate", "Fraction of recorded pair observations with a released-positive label; not reader accuracy."),
            ("Interpretation", "Reader evidence and released labels are separate. Disagreement does not establish annotation error."),
            ("Scope", "This workbook describes the current reader run. Reproduce commands separately verify the paper experiment."),
            ("Saved-result verification", "python -m src reproduce verify"),
            ("Recompute statistics", "python -m src reproduce statistics --output output/reproduction/new-run"),
            ("Full reader rebuild", "python -m src reproduce full --output output/reproduction/new-full-run"),
        ]
        for index, row in enumerate(rows, 2):
            _write_data_row(ws, index, list(row), height=44)
        _set_col_widths(ws, [30, 105])
        ws.freeze_panes = "B2"
        _set_autofilter(ws, 2)
        summary = wb["Summary"]
        row = summary.max_row + 3
        for title in ["Relation patterns", "Entity boundaries", "Sentence evidence", "Reproduction"]:
            cell = summary.cell(row, 1, title)
            cell.hyperlink = f"#'{title}'!A1"
            cell.font = _f(10, color="176B9B")
            row += 1
        # Links use exact sentence rows, with no inferred pair decisions.
        evidence = wb["Sentence evidence"]
        first_by_chain = {}
        for cells in evidence.iter_rows(min_row=2):
            if cells[0].value == "PATTERN":
                first_by_chain.setdefault((cells[1].value, cells[3].value), cells[0].row)
        patterns = wb["Relation patterns"]
        for cells in patterns.iter_rows(min_row=2):
            target = first_by_chain.get((cells[1].value, cells[9].value))
            if target:
                cells[9].hyperlink = f"#'Sentence evidence'!A{target}"
                cells[9].font = _f(10, color="176B9B")
        for sheet in wb:
            sheet.sheet_view.showGridLines = False
            sheet.sheet_properties.pageSetUpPr.fitToPage = True
            sheet.page_setup.orientation = "landscape"
            sheet.page_setup.paperSize = sheet.PAPERSIZE_A3
            sheet.page_setup.fitToWidth = 1
            sheet.page_setup.fitToHeight = 0
            sheet.print_title_rows = "1:1"
            from copy import copy
            merged_widths = {}
            for merged in sheet.merged_cells.ranges:
                merged_widths[(merged.min_row, merged.min_col)] = sum(
                    sheet.column_dimensions[get_column_letter(column)].width or 12
                    for column in range(merged.min_col, merged.max_col + 1)
                )
            for cells in sheet.iter_rows():
                line_count = 1
                for cell in cells:
                    if cell.__class__.__name__ == 'MergedCell':
                        continue
                    alignment = copy(cell.alignment)
                    alignment.wrap_text = True
                    cell.alignment = alignment
                    width = merged_widths.get((cell.row, cell.column), sheet.column_dimensions[cell.column_letter].width or 12)
                    count = sum(max(1, ceil(len(part) / max(8, width - 2))) for part in str(cell.value or '').split('\n'))
                    line_count = max(line_count, count)
                sheet.row_dimensions[cells[0].row].height = max(sheet.row_dimensions[cells[0].row].height or 20, min(409, line_count * 15 + 10))
