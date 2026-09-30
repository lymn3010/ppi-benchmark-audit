"""Shared workbook typography, spacing and table formatting."""
from __future__ import annotations
import logging
from typing import Any, TYPE_CHECKING
log = logging.getLogger(__name__)

try:
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    _HAS_OPENPYXL = True
except ImportError:  # pragma: no cover
    _HAS_OPENPYXL = False

if TYPE_CHECKING:
    from openpyxl.worksheet.worksheet import Worksheet



_FONT_NAME = "Calibri"

# Hex colour palette (no leading #, as openpyxl expects)
_C: dict[str, str] = {
    # Sheet headers
    "hdr_dark":   "1F4E79",   # dark navy  -- frozen column/row headers
    "hdr_mid":    "2E75B6",   # medium blue -- section group titles
    "hdr_light":  "DEEAF1",   # pale blue  -- metadata label cells
    # Editable surface
    "edit":       "EBF3FB",   # light sky  -- Review Decision / Notes columns
    # Data row alternation
    "alt":        "F2F2F2",   # off-white  -- even rows when no prec-color applies
    "white":      "FFFFFF",
    # Precision tiers (applied to the entire data row)
    "prec_hi":    "E4EFF7",   # green   >= 0.80
    "prec_mid":   "F0F5F9",   # amber   0.50 – 0.79
    "prec_lo":    "FAFCFE",   # salmon  < 0.50
}

if _HAS_OPENPYXL:
    _SIDE   = Side(style="thin", color="BFBFBF")
    _BORDER = Border(left=_SIDE, right=_SIDE, top=_SIDE, bottom=_SIDE)
else:  # pragma: no cover - import-only fallback when XLSX support is absent
    _SIDE = None
    _BORDER = None

_FMT_PREC = "0.0000"
_FMT_INT  = "#,##0"
_FMT_PCT  = "0.0%"

_AUDIT_EVIDENCE_PRIORITY = (
    "structural",
    "relational_statement",
    "lexicalized_path",
    "candidate_pattern_ledger",
    "event_without_final_pair",
)



def _f(
    size: int = 10,
    bold: bool = False,
    color: str = "000000",
    italic: bool = False,
) -> "Font":
    return Font(name=_FONT_NAME, size=size, bold=bold, color=color,
                italic=italic)


def _fill(hex_color: str) -> "PatternFill":
    return PatternFill("solid", fgColor=hex_color)


def _align(
    h: str = "left",
    v: str = "center",
    wrap: bool = False,
) -> "Alignment":
    return Alignment(horizontal=h, vertical=v, wrap_text=wrap)


def _prec_color(prec: float) -> str:
    """Precision value → fill hex colour."""
    if prec >= 0.80:
        return _C["prec_hi"]
    if prec >= 0.50:
        return _C["prec_mid"]
    return _C["prec_lo"]


def _write_header_row(
    ws: "Worksheet",
    row: int,
    cols: list[str],
    height: float = 22,
) -> None:
    """Frozen header row: dark-navy background, white bold Calibri."""
    for c, text in enumerate(cols, start=1):
        cell = ws.cell(row=row, column=c, value=text)
        cell.font      = _f(10, bold=True, color="FFFFFF")
        cell.fill      = _fill(_C["hdr_dark"])
        cell.alignment = _align("center", "center")
        cell.border    = _BORDER
    ws.row_dimensions[row].height = height


def _write_section_header(
    ws: "Worksheet",
    row: int,
    n_cols: int,
    text: str,
    level: str = "mid",
    height: float = 18,
) -> None:
    """Full-width merged section header with medium-blue background."""
    color = _C["hdr_dark"] if level == "dark" else _C["hdr_mid"]
    size  = 11 if level == "dark" else 10
    cell  = ws.cell(row=row, column=1, value=text)
    cell.font      = _f(size, bold=True, color="FFFFFF")
    cell.fill      = _fill(color)
    cell.alignment = _align("left", "center")
    cell.border    = _BORDER
    if n_cols > 1:
        ws.merge_cells(
            start_row=row, start_column=1,
            end_row=row,   end_column=n_cols,
        )
    ws.row_dimensions[row].height = height


def _write_kv_row(
    ws: "Worksheet",
    row: int,
    n_cols: int,
    label: str,
    value: Any,
    height: float = 16,
) -> None:
    """Two-column metadata row: label (light-blue) | value (spans rest)."""
    lbl_cell = ws.cell(row=row, column=1, value=label)
    lbl_cell.font      = _f(10, bold=True)
    lbl_cell.fill      = _fill(_C["hdr_light"])
    lbl_cell.alignment = _align("left", "center")
    lbl_cell.border    = _BORDER

    val_cell = ws.cell(row=row, column=2, value=str(value))
    val_cell.font      = _f(10)
    val_cell.alignment = _align("left", "center")
    val_cell.border    = _BORDER
    if n_cols > 2:
        ws.merge_cells(start_row=row, start_column=2,
                       end_row=row, end_column=n_cols)
    ws.row_dimensions[row].height = height


def _write_data_row(
    ws: "Worksheet",
    row: int,
    values: list[Any],
    fill_hex: str = _C["white"],
    formats: list[str | None] | None = None,
    height: float | None = None,
    aligns: list[str] | None = None,
) -> None:
    """Write one data row with uniform background and optional formats."""
    for c, val in enumerate(values, start=1):
        cell = ws.cell(row=row, column=c, value=val)
        cell.font   = _f(10)
        cell.fill   = _fill(fill_hex)
        cell.border = _BORDER
        h_align = (aligns[c - 1] if aligns and c <= len(aligns) else "left")
        cell.alignment = _align(h_align, "center")
        if formats and c <= len(formats) and formats[c - 1]:
            cell.number_format = formats[c - 1]
    if height is not None:
        ws.row_dimensions[row].height = height


def _set_col_widths(ws: "Worksheet", widths: list[float]) -> None:
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


def _set_autofilter(ws: "Worksheet", n_cols: int, header_row: int = 1) -> None:
    ws.auto_filter.ref = (
        f"A{header_row}:{get_column_letter(n_cols)}{max(header_row, ws.max_row)}"
    )


def _tab_color(ws: "Worksheet", hex_color: str) -> None:
    ws.sheet_properties.tabColor = hex_color
