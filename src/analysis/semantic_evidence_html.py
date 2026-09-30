"""Read-only HTML view of semantic evidence in ``events.db``."""
from __future__ import annotations

import argparse
import html
import json
import sqlite3
from pathlib import Path
from typing import Any

from src.models.candidate.group_alignment import align_group_provenance


def export_semantic_evidence_html(
    db_path: str | Path,
    output_path: str | Path,
    *,
    limit: int = 250,
    sentence_id: str | None = None,
) -> Path:
    """Write a compact review HTML file for group/topology evidence."""
    db_path = Path(db_path)
    output_path = Path(output_path)
    rows = load_semantic_evidence(db_path, limit=limit, sentence_id=sentence_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_semantic_evidence_html(rows, db_path), encoding="utf-8")
    return output_path


def load_semantic_evidence(
    db_path: str | Path,
    *,
    limit: int = 250,
    sentence_id: str | None = None,
) -> list[dict[str, Any]]:
    """Load event rows that carry group provenance from an events.db file."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    where = ["e.arguments_json LIKE '%\"group\"%'"]
    params: list[Any] = []
    if sentence_id:
        where.append("e.sentence_id = ?")
        params.append(sentence_id)
    requested_limit = max(1, int(limit))
    # Many groups are empty; read more rows and keep those with topology.
    params.append(requested_limit * 5)

    sql = f"""
        SELECT
            e.event_id,
            e.sentence_id,
            s.text,
            s.orig_text,
            s.proteins,
            s.pred_pairs AS sentence_pred_pairs,
            s.gold_pairs,
            s.label,
            e.event_type,
            e.event_detail,
            e.pred_pairs AS event_pred_pairs,
            e.predicate_text,
            e.predicate_lemma,
            e.pattern_key,
            e.semantic_pattern_key,
            e.chain_readable,
            e.arguments_json
        FROM events e
        JOIN sentences s ON s.sentence_id = e.sentence_id
        WHERE {" AND ".join(where)}
        ORDER BY e.sentence_id, e.event_id
        LIMIT ?
    """

    con = sqlite3.connect(str(db_path))
    try:
        con.row_factory = sqlite3.Row
        event_rows = [dict(r) for r in con.execute(sql, params).fetchall()]
        sentence_ids = sorted({r["sentence_id"] for r in event_rows})
        candidate_by_sentence = _load_candidate_relations(con, sentence_ids)
    finally:
        con.close()

    out: list[dict[str, Any]] = []
    for row in event_rows:
        arguments = _loads(row.get("arguments_json"), {})
        if not _has_useful_group(arguments):
            continue
        enriched = dict(row)
        enriched["proteins"] = _loads(row.get("proteins"), [])
        enriched["sentence_pred_pairs"] = _loads(row.get("sentence_pred_pairs"), [])
        enriched["gold_pairs"] = _loads(row.get("gold_pairs"), [])
        enriched["event_pred_pairs"] = _loads(row.get("event_pred_pairs"), [])
        enriched["arguments"] = arguments
        enriched["group_alignments"] = _group_alignments(arguments)
        enriched["candidate_relations"] = candidate_by_sentence.get(row["sentence_id"], [])
        out.append(enriched)
        if len(out) >= requested_limit:
            break
    return out


def render_semantic_evidence_html(rows: list[dict[str, Any]], db_path: Path) -> str:
    title = "Semantic Evidence Review"
    body = "\n".join(_render_event(row) for row in rows)
    if not body:
        body = '<p class="empty">No event rows with group provenance were found.</p>'
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f7f7f4;
      --ink: #202124;
      --muted: #666b73;
      --line: #d8d8d2;
      --panel: #ffffff;
      --accent: #245f73;
      --soft: #edf4f6;
      --warn: #8a5b10;
      --bad: #8b2f2f;
      --code: #f2f2ee;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      padding: 24px 28px 14px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{ margin: 0 0 6px; font-size: 22px; font-weight: 650; }}
    .meta {{ color: var(--muted); font-size: 13px; }}
    main {{ padding: 18px 28px 36px; max-width: 1280px; margin: 0 auto; }}
    details.event {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 0 0 12px;
      overflow: hidden;
    }}
    details.event[open] {{ box-shadow: 0 1px 8px rgba(0,0,0,.04); }}
    summary {{
      cursor: pointer;
      padding: 12px 14px;
      list-style: none;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
    }}
    summary::-webkit-details-marker {{ display: none; }}
    .sid {{ font-weight: 650; color: var(--accent); }}
    .sentence {{ margin-top: 4px; color: var(--ink); }}
    .chips {{ display: flex; gap: 6px; flex-wrap: wrap; justify-content: flex-end; }}
    .chip {{
      border: 1px solid var(--line);
      background: var(--soft);
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      color: #264653;
      white-space: nowrap;
    }}
    .content {{ border-top: 1px solid var(--line); padding: 14px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    h2 {{ font-size: 14px; margin: 0 0 8px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; }}
    th, td {{
      border: 1px solid var(--line);
      padding: 6px 8px;
      text-align: left;
      vertical-align: top;
      font-size: 13px;
    }}
    th {{ background: #f0f2ef; font-weight: 650; }}
    code, pre {{
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      background: var(--code);
      border-radius: 4px;
    }}
    code {{ padding: 1px 4px; }}
    pre {{ margin: 0; padding: 8px; white-space: pre-wrap; overflow-wrap: anywhere; }}
    .section {{ margin-bottom: 14px; }}
    .warn {{ color: var(--warn); font-weight: 600; }}
    .bad {{ color: var(--bad); font-weight: 600; }}
    .empty {{ color: var(--muted); }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="meta">Source: <code>{_esc(str(db_path))}</code> - Rows: {len(rows)}</div>
  </header>
  <main>
    {body}
  </main>
</body>
</html>
"""


def _load_candidate_relations(
    con: sqlite3.Connection,
    sentence_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    if not sentence_ids:
        return {}
    placeholders = ",".join("?" for _ in sentence_ids)
    sql = f"""
        SELECT
            sentence_id,
            pair_a,
            pair_b,
            predicate_lemma,
            event_detail,
            source_role,
            target_role,
            case_marker,
            assertion_status,
            projection_mode,
            pattern_signature,
            canonical_pattern_key,
            ambiguity_reason,
            payload
        FROM candidate_relations
        WHERE sentence_id IN ({placeholders})
        ORDER BY sentence_id, id
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    try:
        rows = con.execute(sql, sentence_ids).fetchall()
    except sqlite3.OperationalError:
        return grouped
    for row in rows:
        data = dict(row)
        data["payload"] = _loads(data.get("payload"), {})
        grouped.setdefault(data["sentence_id"], []).append(data)
    return grouped


def _group_alignments(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    sources = _frames_with_group(arguments.get("sources", []))
    targets = _frames_with_group(arguments.get("targets", []))
    alignments: list[dict[str, Any]] = []
    for si, src in enumerate(sources):
        for ti, tgt in enumerate(targets):
            result = align_group_provenance(src.get("group"), tgt.get("group"))
            if result is None:
                continue
            alignments.append({
                "source_index": si,
                "target_index": ti,
                "source_text": src.get("text", ""),
                "target_text": tgt.get("text", ""),
                "mode": result.mode,
                "slots": list(result.slots),
                "aligned": [list(p) for p in result.aligned],
                "cross": [list(p) for p in result.cross],
            })
    return alignments


def _frames_with_group(frames: Any) -> list[dict[str, Any]]:
    if not isinstance(frames, list):
        return []
    return [f for f in frames if isinstance(f, dict) and f.get("group")]


def _has_useful_group(arguments: Any) -> bool:
    if not isinstance(arguments, dict):
        return False
    for frames in arguments.values():
        if not isinstance(frames, list):
            continue
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            group = frame.get("group")
            if not isinstance(group, dict):
                continue
            if group.get("role_protein_indices") or group.get("all_protein_indices"):
                return True
            if group.get("member_internal_pairs"):
                return True
            members = group.get("members") or []
            if isinstance(members, list) and len(members) > 1:
                return True
    return False


def _render_event(row: dict[str, Any]) -> str:
    event_id = _esc(str(row.get("event_id", "")))
    sid = _esc(str(row.get("sentence_id", "")))
    label = _esc(str(row.get("label", "")))
    detail = _esc(str(row.get("event_detail", "")))
    pattern = _esc(str(row.get("pattern_key") or ""))
    semantic_pattern = _esc(str(row.get("semantic_pattern_key") or ""))
    sentence = _esc(str(row.get("text", "")))
    chips = "".join(
        f'<span class="chip">{chip}</span>'
        for chip in [f"event {event_id}", detail, label]
        if chip
    )
    return f"""
<details class="event">
  <summary>
    <div>
      <div class="sid">{sid}</div>
      <div class="sentence">{sentence}</div>
    </div>
    <div class="chips">{chips}</div>
  </summary>
  <div class="content">
    <div class="section">
      <h2>Pattern</h2>
      <table>
        <tr><th>pattern_key</th><td><code>{pattern}</code></td></tr>
        <tr><th>semantic_pattern_key</th><td><code>{semantic_pattern}</code></td></tr>
        <tr><th>event pairs</th><td><code>{_esc(json.dumps(row.get("event_pred_pairs", [])))}</code></td></tr>
        <tr><th>sentence gold</th><td><code>{_esc(json.dumps(row.get("gold_pairs", [])))}</code></td></tr>
      </table>
    </div>
    <div class="grid">
      <div>{_render_arguments(row.get("arguments", {}))}</div>
      <div>{_render_alignments(row.get("group_alignments", []))}</div>
    </div>
    <div class="section">
      <h2>Candidate Relations</h2>
      {_render_candidate_relations(row.get("candidate_relations", []))}
    </div>
  </div>
</details>
"""


def _render_arguments(arguments: dict[str, Any]) -> str:
    rows: list[str] = []
    if isinstance(arguments, dict):
        for role, frames in arguments.items():
            if not isinstance(frames, list):
                continue
            for i, frame in enumerate(frames):
                group = frame.get("group") if isinstance(frame, dict) else None
                rows.append(
                    "<tr>"
                    f"<td>{_esc(str(role))}[{i}]</td>"
                    f"<td>{_esc(str(frame.get('text', '')))}</td>"
                    f"<td><code>{_esc(json.dumps(frame.get('targets', [])))}</code></td>"
                    f"<td>{_render_group(group)}</td>"
                    "</tr>"
                )
    if not rows:
        rows.append('<tr><td colspan="4" class="empty">No argument group evidence.</td></tr>')
    return f"""
<div class="section">
  <h2>Arguments</h2>
  <table>
    <tr><th>role</th><th>text</th><th>targets</th><th>group provenance</th></tr>
    {''.join(rows)}
  </table>
</div>
"""


def _render_group(group: Any) -> str:
    if not isinstance(group, dict) or not group:
        return '<span class="empty">none</span>'
    members = group.get("members") or []
    member_bits = []
    for i, member in enumerate(members):
        member_bits.append(
            f"slot {i}: role={member.get('role_protein_indices', [])}, "
            f"all={member.get('protein_indices', [])}, "
            f"modifier={member.get('modifier_protein_indices', [])}"
        )
    lines = [
        f"kind: {group.get('kind', '')}",
        f"role proteins: {group.get('role_protein_indices', [])}",
        f"all proteins: {group.get('all_protein_indices', [])}",
        f"internal pairs: {group.get('member_internal_pairs', [])}",
        *member_bits,
    ]
    return "<pre>" + _esc("\n".join(lines)) + "</pre>"


def _render_alignments(alignments: list[dict[str, Any]]) -> str:
    rows = []
    for item in alignments:
        cross = item.get("cross", [])
        cross_class = "bad" if cross else ""
        rows.append(
            "<tr>"
            f"<td>{_esc(item.get('source_text', ''))} -> {_esc(item.get('target_text', ''))}</td>"
            f"<td><code>{_esc(str(item.get('mode', '')))}</code></td>"
            f"<td><code>{_esc(json.dumps(item.get('aligned', [])))}</code></td>"
            f"<td class=\"{cross_class}\"><code>{_esc(json.dumps(cross))}</code></td>"
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="4" class="empty">No source/target group alignment.</td></tr>')
    return f"""
<div class="section">
  <h2>Possible Group Alignment</h2>
  <table>
    <tr><th>source -> target</th><th>mode</th><th>aligned</th><th>cross evidence</th></tr>
    {''.join(rows)}
  </table>
</div>
"""


def _render_candidate_relations(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="empty">No candidate relation rows for this sentence.</p>'
    out = []
    for rel in rows[:80]:
        ambiguity = rel.get("ambiguity_reason") or ""
        cls = "warn" if ambiguity else ""
        out.append(
            "<tr>"
            f"<td><code>[{rel.get('pair_a')}, {rel.get('pair_b')}]</code></td>"
            f"<td>{_esc(str(rel.get('event_detail') or ''))}</td>"
            f"<td><code>{_esc(str(rel.get('canonical_pattern_key') or rel.get('pattern_signature') or ''))}</code></td>"
            f"<td>{_esc(str(rel.get('assertion_status') or ''))}</td>"
            f"<td class=\"{cls}\">{_esc(str(ambiguity))}</td>"
            "</tr>"
        )
    return f"""
<table>
  <tr><th>pair</th><th>event</th><th>pattern</th><th>assertion</th><th>ambiguity</th></tr>
  {''.join(out)}
</table>
"""


def _loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _esc(value: str) -> str:
    return html.escape(value, quote=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export group/topology semantic evidence from events.db to HTML.",
    )
    parser.add_argument("db_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--sentence-id", default=None)
    args = parser.parse_args(argv)
    path = export_semantic_evidence_html(
        args.db_path,
        args.output_path,
        limit=args.limit,
        sentence_id=args.sentence_id,
    )
    print(f"[SemanticEvidenceHTML] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
