from __future__ import annotations

import csv
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image

from .run_campaign import load_campaign_meta, resolve_campaign_manifest_path, save_campaign_meta
from .schemas import file_sha256, iter_jsonl, read_json, utc_now_iso, write_json, write_jsonl


REVIEW_GRANULARITY = "frame_bundle"
REVIEW_POLICY = "exclude_rejects"


def _bundle_id(row: Dict[str, Any]) -> str:
    scene = str(row.get("scene") or "unknown").strip().lower() or "unknown"
    seq = str(row.get("seq") or "unknown").strip() or "unknown"
    frame = str(row.get("frame") or "unknown").strip() or "unknown"
    return f"{scene}||{seq}||{frame}"


def _bundle_tuple(bundle_id: str) -> Tuple[str, str, str]:
    parts = str(bundle_id).split("||")
    if len(parts) != 3:
        return ("unknown", "unknown", "unknown")
    return (parts[0], parts[1], parts[2])


def _resolve_input_path(campaign_dir: Path, raw: str) -> Path:
    p = Path(str(raw or "").strip())
    if p.is_absolute():
        return p
    return (campaign_dir / p).resolve()


def _as_rel_or_abs(path: Path, anchor_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(anchor_dir.resolve()))
    except Exception:
        return str(path.resolve())


def _normalize_decision(value: Any) -> str:
    low = str(value or "").strip().lower()
    if low in {"approve", "approved", "accept", "accepted", "yes", "y", "1", "true"}:
        return "approve"
    if low in {"reject", "rejected", "drop", "no", "n", "0", "false"}:
        return "reject"
    if low in {"undecided", "", "none", "skip", "hold", "maybe"}:
        return "undecided"
    return "undecided"


def _thumbnail_name(img_path: Path) -> str:
    digest = hashlib.sha1(str(img_path.resolve()).encode("utf-8")).hexdigest()
    return f"{digest}.jpg"


def _make_thumbnail(image_path: Path, thumbs_dir: Path, long_edge: int) -> Optional[Path]:
    if not image_path.exists():
        return None
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    out = thumbs_dir / _thumbnail_name(image_path)
    if out.exists():
        return out
    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            size = max(64, int(long_edge))
            resampling = getattr(Image, "Resampling", Image)
            im.thumbnail((size, size), resampling.LANCZOS)
            im.save(out, format="JPEG", quality=85)
        return out
    except Exception:
        return None


def _detect_debug_tasks_html(campaign_dir: Path, bundle_id: str, rows: Iterable[Dict[str, Any]]) -> Optional[Path]:
    scene, seq, frame = _bundle_tuple(bundle_id)
    debug_leaf = f"frame_{scene}_{seq}_{frame}/tasks.html"

    # First, look under source benchmark run debug directories.
    for row in rows:
        source_benchmark = str(row.get("source_benchmark") or "").strip()
        if not source_benchmark:
            continue
        run_dir = Path(source_benchmark).resolve().parent
        candidate = run_dir / "debug" / debug_leaf
        if candidate.exists():
            return candidate
    return None


def _load_manifest_rows(manifest_path: Path) -> List[Dict[str, Any]]:
    rows = list(iter_jsonl(manifest_path))
    if not rows:
        raise ValueError(f"Manifest has no rows: {manifest_path}")
    return rows


def _build_review_items(
    campaign_dir: Path,
    review_dir: Path,
    rows: List[Dict[str, Any]],
    thumb_long_edge: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_bundle_id(row)].append(row)

    thumbs_dir = review_dir / "thumbs"
    out: List[Dict[str, Any]] = []

    def sort_key(bundle_key: str) -> Tuple[str, str, str]:
        return _bundle_tuple(bundle_key)

    for bid in sorted(grouped.keys(), key=sort_key):
        bundle_rows = sorted(grouped[bid], key=lambda r: int(r.get("task_id") or 0))
        scene, seq, frame = _bundle_tuple(bid)

        unique_images: Dict[str, Dict[str, Any]] = {}
        for row in bundle_rows:
            for img_raw in row.get("image_paths") or []:
                img_abs = _resolve_input_path(campaign_dir, str(img_raw))
                img_key = str(img_abs.resolve())
                if img_key in unique_images:
                    continue

                thumb_abs = _make_thumbnail(img_abs, thumbs_dir, long_edge=thumb_long_edge)
                image_href = _as_rel_or_abs(img_abs, review_dir)
                thumb_href = _as_rel_or_abs(thumb_abs, review_dir) if thumb_abs else image_href

                unique_images[img_key] = {
                    "image_path": str(img_abs),
                    "image_href": image_href,
                    "thumb_path": str(thumb_abs) if thumb_abs else None,
                    "thumb_href": thumb_href,
                    "exists": bool(img_abs.exists()),
                }

        debug_html = _detect_debug_tasks_html(campaign_dir, bid, bundle_rows)

        samples: List[Dict[str, Any]] = []
        for row in bundle_rows:
            samples.append(
                {
                    "sample_uid": row.get("sample_uid"),
                    "task_id": row.get("task_id"),
                    "task_type": row.get("task_type"),
                    "question": row.get("question"),
                    "groundtruth_answer": row.get("groundtruth_answer"),
                    "review_flags": list(row.get("review_flags") or []),
                }
            )

        out.append(
            {
                "bundle_id": bid,
                "scene": scene,
                "seq": seq,
                "frame": frame,
                "image_paths": [v["image_path"] for v in unique_images.values()],
                "thumb_paths": [v["thumb_path"] for v in unique_images.values() if v.get("thumb_path")],
                "images": list(unique_images.values()),
                "debug_tasks_html": _as_rel_or_abs(debug_html, review_dir) if debug_html else None,
                "samples": samples,
            }
        )
    return out


def _json_for_html(data: Any) -> str:
    # Prevent accidental </script> breakouts.
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def _write_review_html(review_dir: Path, items: List[Dict[str, Any]], page_size: int) -> Path:
    page_size = max(25, int(page_size))
    scenes = sorted({str(x.get("scene") or "unknown") for x in items})

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Gaze-VQA Review</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --card: #ffffff;
      --text: #1d232a;
      --muted: #5e6b7a;
      --line: #d9e0e6;
      --ok: #1f8f4b;
      --bad: #b33a3a;
      --hold: #7b6d0b;
      --accent: #0f5ea8;
    }}
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: var(--bg); color: var(--text); margin: 0; }}
    .top {{ position: sticky; top: 0; z-index: 10; background: #fff; border-bottom: 1px solid var(--line); padding: 10px 14px; }}
    .row {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    .row input, .row select, .row button {{ padding: 6px 8px; font-size: 13px; }}
    .row button {{ border: 1px solid var(--line); background: #fff; cursor: pointer; border-radius: 6px; }}
    .row button:hover {{ border-color: #b7c4d0; }}
    #cards {{ padding: 12px; display: grid; grid-template-columns: 1fr; gap: 12px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 10px; }}
    .card.selected {{ border-color: var(--accent); box-shadow: 0 0 0 2px rgba(15,94,168,0.15); }}
    .meta {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 6px; }}
    .meta .left {{ font-weight: 600; }}
    .meta .right {{ color: var(--muted); font-size: 12px; }}
    .images {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }}
    .img-wrap {{ border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: #fff; }}
    .img-wrap img {{ width: 240px; height: 180px; object-fit: cover; display: block; }}
    .img-cap {{ font-size: 11px; color: var(--muted); padding: 4px 6px; max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 8px; }}
    th, td {{ border: 1px solid var(--line); padding: 6px; font-size: 12px; vertical-align: top; }}
    th {{ background: #fafbfd; text-align: left; }}
    .flags {{ color: #7a4d00; font-weight: 600; }}
    .decision {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
    .pill {{ border: 1px solid var(--line); border-radius: 999px; padding: 5px 10px; cursor: pointer; font-size: 12px; }}
    .pill.on.approve {{ border-color: var(--ok); color: var(--ok); font-weight: 700; }}
    .pill.on.reject {{ border-color: var(--bad); color: var(--bad); font-weight: 700; }}
    .pill.on.undecided {{ border-color: var(--hold); color: var(--hold); font-weight: 700; }}
    .stats {{ color: var(--muted); font-size: 12px; }}
  </style>
</head>
<body>
  <div class="top">
    <div class="row" style="margin-bottom:8px;">
      <strong>Gaze-VQA Bundle Reviewer</strong>
      <span class="stats" id="stats"></span>
    </div>
    <div class="row">
      <label>Scene
        <select id="sceneFilter">
          <option value="">All</option>
        </select>
      </label>
      <label>Search
        <input id="searchInput" type="text" placeholder="question / answer / bundle id" />
      </label>
      <label><input id="flaggedOnly" type="checkbox" /> review_flags only</label>
      <button id="prevBtn" type="button">Prev Page</button>
      <button id="nextBtn" type="button">Next Page</button>
      <span class="stats" id="pageInfo"></span>
      <button id="exportCsvBtn" type="button">Export CSV</button>
      <button id="exportJsonBtn" type="button">Export JSON</button>
      <label style="border:1px solid var(--line); border-radius:6px; padding:6px 8px; background:#fff; cursor:pointer;">Import decisions
        <input id="importInput" type="file" accept=".csv,.json" style="display:none" />
      </label>
    </div>
    <div class="stats" style="margin-top:6px;">Shortcuts: A=approve, R=reject, U=undecided, N=next card</div>
  </div>
  <div id="cards"></div>

  <script id="review-data" type="application/json">{_json_for_html(items)}</script>
  <script>
    const items = JSON.parse(document.getElementById('review-data').textContent || '[]');
    const pageSize = {int(page_size)};
    let page = 1;
    let selectedBundle = null;
    const decisions = Object.create(null);

    for (const it of items) {{
      decisions[it.bundle_id] = {{
        bundle_id: it.bundle_id,
        decision: 'undecided',
        note: '',
        reviewer: '',
        updated_at: ''
      }};
    }}

    function esc(s) {{
      return String(s ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    }}

    function setDecision(bundleId, decision) {{
      const rec = decisions[bundleId] || {{ bundle_id: bundleId }};
      rec.decision = decision;
      rec.updated_at = new Date().toISOString();
      decisions[bundleId] = rec;
      render();
    }}

    function setNote(bundleId, note) {{
      const rec = decisions[bundleId] || {{ bundle_id: bundleId, decision: 'undecided' }};
      rec.note = note;
      rec.updated_at = new Date().toISOString();
      decisions[bundleId] = rec;
    }}

    function currentFiltered() {{
      const scene = document.getElementById('sceneFilter').value;
      const search = (document.getElementById('searchInput').value || '').trim().toLowerCase();
      const flaggedOnly = document.getElementById('flaggedOnly').checked;

      return items.filter(it => {{
        if (scene && it.scene !== scene) return false;
        if (flaggedOnly) {{
          const hasFlags = (it.samples || []).some(s => (s.review_flags || []).length > 0);
          if (!hasFlags) return false;
        }}
        if (!search) return true;

        const hay = [
          it.bundle_id,
          it.scene,
          it.seq,
          it.frame,
          ...((it.samples || []).map(s => `${{s.task_type}} ${{s.question}} ${{s.groundtruth_answer}} ${{(s.review_flags||[]).join(' ')}}`))
        ].join(' ').toLowerCase();
        return hay.includes(search);
      }});
    }}

    function pageRows(rows) {{
      const pages = Math.max(1, Math.ceil(rows.length / pageSize));
      if (page > pages) page = pages;
      if (page < 1) page = 1;
      const s = (page - 1) * pageSize;
      const e = s + pageSize;
      return {{ pages, rows: rows.slice(s, e) }};
    }}

    function render() {{
      const rows = currentFiltered();
      const cardRoot = document.getElementById('cards');
      const stats = document.getElementById('stats');
      const pageInfo = document.getElementById('pageInfo');
      const paged = pageRows(rows);

      stats.textContent = `bundles=${{rows.length}} / total=${{items.length}}`;
      pageInfo.textContent = `page ${{page}} / ${{paged.pages}}`;

      cardRoot.innerHTML = '';
      for (const it of paged.rows) {{
        const d = decisions[it.bundle_id] || {{ decision: 'undecided', note: '' }};
        const card = document.createElement('div');
        card.className = 'card' + (selectedBundle === it.bundle_id ? ' selected' : '');
        card.onclick = () => {{ selectedBundle = it.bundle_id; render(); }};

        const imagesHtml = (it.images || []).map(img => {{
          const imageHref = encodeURI(String(img.image_href || img.image_path || ''));
          const thumbHref = encodeURI(String(img.thumb_href || imageHref));
          const label = esc(String(img.image_path || ''));
          return `<a class="img-wrap" href="${{imageHref}}" target="_blank" rel="noopener"><img src="${{thumbHref}}" loading="lazy"/><div class="img-cap">${{label}}</div></a>`;
        }}).join('');

        const tasksHtml = (it.samples || []).map(s => {{
          const flags = (s.review_flags || []).join(', ');
          return `<tr>
            <td>${{esc(s.task_id)}}</td>
            <td>${{esc(s.task_type)}}</td>
            <td>${{esc(s.question)}}</td>
            <td>${{esc(s.groundtruth_answer)}}</td>
            <td class="flags">${{esc(flags)}}</td>
          </tr>`;
        }}).join('');

        const debugHtml = it.debug_tasks_html
          ? `<a href="${{encodeURI(String(it.debug_tasks_html))}}" target="_blank" rel="noopener">open tasks.html</a>`
          : `<span class="stats">debug html: unavailable</span>`;

        const decisionClass = (name) => 'pill ' + ((d.decision === name) ? ('on ' + name) : '');

        card.innerHTML = `
          <div class="meta">
            <div class="left">${{esc(it.bundle_id)}} (scene=${{esc(it.scene)}}, seq=${{esc(it.seq)}}, frame=${{esc(it.frame)}})</div>
            <div class="right">${{debugHtml}}</div>
          </div>
          <div class="images">${{imagesHtml}}</div>
          <table>
            <thead><tr><th>Task</th><th>Type</th><th>Question</th><th>GT answer</th><th>Flags</th></tr></thead>
            <tbody>${{tasksHtml}}</tbody>
          </table>
          <div class="decision">
            <button type="button" class="${{decisionClass('approve')}}" data-action="approve">Approve</button>
            <button type="button" class="${{decisionClass('reject')}}" data-action="reject">Reject</button>
            <button type="button" class="${{decisionClass('undecided')}}" data-action="undecided">Undecided</button>
            <input type="text" placeholder="optional note" value="${{esc(d.note || '')}}" style="min-width:320px;" />
          </div>
        `;

        card.querySelectorAll('button[data-action]').forEach(btn => {{
          btn.addEventListener('click', (ev) => {{
            ev.stopPropagation();
            setDecision(it.bundle_id, btn.dataset.action);
          }});
        }});

        const noteInput = card.querySelector('input[type="text"]');
        noteInput.addEventListener('input', (ev) => setNote(it.bundle_id, ev.target.value || ''));

        cardRoot.appendChild(card);
      }}
    }}

    function exportJson() {{
      const rows = Object.values(decisions);
      const blob = new Blob([JSON.stringify(rows, null, 2)], {{ type: 'application/json' }});
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'review_decisions.json';
      a.click();
      URL.revokeObjectURL(a.href);
    }}

    function csvEscape(val) {{
      const s = String(val ?? '');
      if (s.includes(',') || s.includes('"') || s.includes('\\n')) {{
        return '"' + s.replaceAll('"', '""') + '"';
      }}
      return s;
    }}

    function exportCsv() {{
      const cols = ['bundle_id', 'decision', 'note', 'reviewer', 'updated_at'];
      const rows = [cols.join(',')];
      for (const d of Object.values(decisions)) {{
        rows.push(cols.map(c => csvEscape(d[c] ?? '')).join(','));
      }}
      const blob = new Blob([rows.join('\\n') + '\\n'], {{ type: 'text/csv' }});
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'review_decisions.csv';
      a.click();
      URL.revokeObjectURL(a.href);
    }}

    function parseCsv(text) {{
      const lines = text.split(/\\r?\\n/).filter(Boolean);
      if (lines.length < 2) return [];
      const header = lines[0].split(',').map(s => s.trim());
      const out = [];
      for (const ln of lines.slice(1)) {{
        const cols = ln.split(',');
        const row = {{}};
        for (let i = 0; i < header.length; i++) row[header[i]] = (cols[i] || '').trim();
        out.push(row);
      }}
      return out;
    }}

    function importDecisions(rows) {{
      for (const r of rows || []) {{
        const bid = String(r.bundle_id || '').trim();
        if (!bid || !(bid in decisions)) continue;
        const dec = String(r.decision || 'undecided').trim().toLowerCase();
        decisions[bid] = {{
          bundle_id: bid,
          decision: (dec === 'approve' || dec === 'reject') ? dec : 'undecided',
          note: String(r.note || ''),
          reviewer: String(r.reviewer || ''),
          updated_at: String(r.updated_at || ''),
        }};
      }}
      render();
    }}

    document.getElementById('sceneFilter').innerHTML += {json.dumps(''.join([f'<option value="{html.escape(s)}">{html.escape(s)}</option>' for s in scenes]))};

    document.getElementById('sceneFilter').addEventListener('change', () => {{ page = 1; render(); }});
    document.getElementById('searchInput').addEventListener('input', () => {{ page = 1; render(); }});
    document.getElementById('flaggedOnly').addEventListener('change', () => {{ page = 1; render(); }});
    document.getElementById('prevBtn').addEventListener('click', () => {{ page -= 1; render(); }});
    document.getElementById('nextBtn').addEventListener('click', () => {{ page += 1; render(); }});
    document.getElementById('exportCsvBtn').addEventListener('click', exportCsv);
    document.getElementById('exportJsonBtn').addEventListener('click', exportJson);

    document.getElementById('importInput').addEventListener('change', async (ev) => {{
      const file = ev.target.files && ev.target.files[0];
      if (!file) return;
      const text = await file.text();
      if (file.name.toLowerCase().endsWith('.json')) {{
        const payload = JSON.parse(text);
        importDecisions(Array.isArray(payload) ? payload : (payload.decisions || []));
      }} else {{
        importDecisions(parseCsv(text));
      }}
      ev.target.value = '';
    }});

    document.addEventListener('keydown', (ev) => {{
      const key = (ev.key || '').toLowerCase();
      if (!['a', 'r', 'u', 'n'].includes(key)) return;
      const rows = currentFiltered();
      if (!rows.length) return;
      const ids = rows.map(r => r.bundle_id);
      if (!selectedBundle || !ids.includes(selectedBundle)) selectedBundle = ids[0];
      const idx = ids.indexOf(selectedBundle);

      if (key === 'a') setDecision(selectedBundle, 'approve');
      if (key === 'r') setDecision(selectedBundle, 'reject');
      if (key === 'u') setDecision(selectedBundle, 'undecided');
      if (key === 'n') {{
        const next = ids[Math.min(ids.length - 1, idx + 1)];
        selectedBundle = next;
        render();
      }}
    }});

    render();
  </script>
</body>
</html>
"""

    out_path = review_dir / "index.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    return out_path


def build_review(
    campaign_dir: Path,
    page_size: int = 200,
    thumb_long_edge: int = 480,
) -> Dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    meta = load_campaign_meta(campaign_dir)
    manifest_path = resolve_campaign_manifest_path(campaign_dir, meta=meta)
    rows = _load_manifest_rows(manifest_path)

    review_dir = campaign_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    items = _build_review_items(
        campaign_dir=campaign_dir,
        review_dir=review_dir,
        rows=rows,
        thumb_long_edge=thumb_long_edge,
    )
    items_path = review_dir / "review_items.json"
    write_json(items_path, items)

    template_path = review_dir / "decisions_template.csv"
    with template_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["bundle_id", "decision", "note", "reviewer", "updated_at"],
        )
        w.writeheader()
        for item in items:
            w.writerow(
                {
                    "bundle_id": item.get("bundle_id"),
                    "decision": "undecided",
                    "note": "",
                    "reviewer": "",
                    "updated_at": "",
                }
            )

    html_path = _write_review_html(review_dir=review_dir, items=items, page_size=page_size)

    return {
        "campaign_dir": str(campaign_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "review_index_html": str(html_path),
        "review_items_json": str(items_path),
        "decisions_template_csv": str(template_path),
        "thumbs_dir": str(review_dir / "thumbs"),
        "bundle_count": len(items),
        "sample_count": len(rows),
    }


def _load_decisions(path: Path) -> Dict[str, Dict[str, Any]]:
    path = path.resolve()
    out: Dict[str, Dict[str, Any]] = {}

    def set_row(row: Dict[str, Any]) -> None:
        bid = str(row.get("bundle_id") or "").strip()
        if not bid:
            return
        out[bid] = {
            "bundle_id": bid,
            "decision": _normalize_decision(row.get("decision")),
            "note": str(row.get("note") or ""),
            "reviewer": str(row.get("reviewer") or ""),
            "updated_at": str(row.get("updated_at") or ""),
        }

    if path.suffix.lower() == ".json":
        data = read_json(path)
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    set_row(row)
        elif isinstance(data, dict):
            if isinstance(data.get("decisions"), list):
                for row in data.get("decisions"):
                    if isinstance(row, dict):
                        set_row(row)
            else:
                for bid, val in data.items():
                    if isinstance(val, dict):
                        row = dict(val)
                        row.setdefault("bundle_id", bid)
                        set_row(row)
                    else:
                        set_row({"bundle_id": bid, "decision": val})
        return out

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if isinstance(row, dict):
                set_row(row)
    return out


def apply_review(
    campaign_dir: Path,
    decisions_path: Path,
    policy: str = REVIEW_POLICY,
    granularity: str = REVIEW_GRANULARITY,
    set_active: bool = True,
) -> Dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    decisions_path = decisions_path.resolve()
    if policy != REVIEW_POLICY:
        raise ValueError(f"Unsupported policy: {policy}")
    if granularity != REVIEW_GRANULARITY:
        raise ValueError(f"Unsupported granularity: {granularity}")
    if not decisions_path.exists():
        raise FileNotFoundError(f"Decisions file not found: {decisions_path}")

    meta = load_campaign_meta(campaign_dir)
    manifest_path = resolve_campaign_manifest_path(campaign_dir, meta=meta)
    rows = _load_manifest_rows(manifest_path)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_bundle_id(row)].append(row)

    decisions = _load_decisions(decisions_path)

    reviewed_rows: List[Dict[str, Any]] = []
    approved_bundles = 0
    rejected_bundles = 0
    undecided_bundles = 0
    missing_decision_bundles = 0

    kept_bundles = 0
    removed_bundles = 0
    kept_samples = 0
    removed_samples = 0

    for bid in sorted(grouped.keys()):
        bundle_rows = grouped[bid]
        if bid in decisions:
            decision = _normalize_decision(decisions[bid].get("decision"))
        else:
            decision = "undecided"
            missing_decision_bundles += 1

        if decision == "approve":
            approved_bundles += 1
        elif decision == "reject":
            rejected_bundles += 1
        else:
            undecided_bundles += 1

        if decision == "reject":
            removed_bundles += 1
            removed_samples += len(bundle_rows)
            continue

        reviewed_rows.extend(bundle_rows)
        kept_bundles += 1
        kept_samples += len(bundle_rows)

    out_manifest = campaign_dir / "gt" / "gt_manifest_v1_reviewed.jsonl"
    write_jsonl(out_manifest, reviewed_rows)

    bundle_counts = {
        "total": len(grouped),
        "approved": approved_bundles,
        "rejected": rejected_bundles,
        "undecided": undecided_bundles,
        "missing_decision_rows": missing_decision_bundles,
        "kept": kept_bundles,
        "removed": removed_bundles,
    }
    sample_counts = {
        "total": len(rows),
        "kept": kept_samples,
        "removed": removed_samples,
    }

    report = {
        "campaign_dir": str(campaign_dir),
        "applied_at": utc_now_iso(),
        "granularity": granularity,
        "policy": policy,
        "manifest_in": str(manifest_path),
        "manifest_in_sha256": file_sha256(manifest_path),
        "manifest_out": str(out_manifest),
        "manifest_out_sha256": file_sha256(out_manifest),
        "decisions_path": str(decisions_path),
        "decisions_sha256": file_sha256(decisions_path),
        "bundle_counts": bundle_counts,
        "sample_counts": sample_counts,
        "set_active": bool(set_active),
    }

    report_path = campaign_dir / "gt" / "review_apply_report.json"
    write_json(report_path, report)

    if set_active:
        current_active = str(meta.get("active_manifest_path") or "").strip()
        if not str(meta.get("base_manifest_path") or "").strip():
            meta["base_manifest_path"] = current_active or str(manifest_path)

        meta["active_manifest_path"] = str(out_manifest)
        meta["active_manifest_sha256"] = file_sha256(out_manifest)
        review = dict(meta.get("review") or {})
        review.update(
            {
                "granularity": granularity,
                "policy": policy,
                "decisions_path": str(decisions_path),
                "decisions_sha256": file_sha256(decisions_path),
                "applied_at": utc_now_iso(),
                "bundle_counts": bundle_counts,
                "sample_counts": sample_counts,
                "report_path": str(report_path),
            }
        )
        meta["review"] = review
        save_campaign_meta(campaign_dir, meta)

    return {
        "reviewed_manifest": str(out_manifest),
        "review_apply_report": str(report_path),
        "active_manifest_path": str(meta.get("active_manifest_path") or out_manifest),
        "bundle_counts": bundle_counts,
        "sample_counts": sample_counts,
    }


def review_status(campaign_dir: Path) -> Dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    meta = load_campaign_meta(campaign_dir)
    active_manifest = resolve_campaign_manifest_path(campaign_dir, meta=meta)
    rows = list(iter_jsonl(active_manifest))

    out = {
        "campaign_dir": str(campaign_dir),
        "base_manifest_path": str(meta.get("base_manifest_path") or ""),
        "active_manifest_path": str(active_manifest),
        "active_manifest_sha256": file_sha256(active_manifest),
        "active_sample_count": len(rows),
        "review": dict(meta.get("review") or {}),
    }
    return out
