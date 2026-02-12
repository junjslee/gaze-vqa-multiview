import json
import os
import html as html_lib
from pathlib import Path
from PIL import Image

from . import state as st
from .vlm import safe_reasoning


def _resize(im: Image.Image) -> Image.Image:
    return im.resize(st.RESIZE_WH)


def make_id(*parts):
    import hashlib
    key = "|".join([str(p) for p in parts])
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def log_debug(obj: dict):
    with open(st.DEBUG_MANIFEST, "a") as f:
        f.write(json.dumps(obj) + "\n")


def _short_reject_report():
    items = sorted(st.REJECT_STATS.items(), key=lambda x: -x[1])
    items = [(k, v) for k, v in items if v > 0]
    return items[:8]


def _short_last_samples(samples, k=3):
    out = []
    for s in samples[-k:]:
        out.append({
            "task": s.get("task_id"),
            "type": s.get("task_type"),
            "scene": s.get("scene"),
            "ts": s.get("timestamp"),
            "cams": s.get("input_cams"),
            "answer": (s.get("answer")[:80] + "...") if isinstance(s.get("answer"), str) and len(s.get("answer")) > 80 else s.get("answer")
        })
    return out


def audit_every_n_accepts(samples, counts, every=5, force=False):
    total = counts["task1"] + counts["task2"] + counts["task3"] + counts["task4"]
    if (not force) and (total % every != 0):
        return

    st.logger.info("=" * 88)
    st.logger.info(f"PROGRESS AUDIT @ total_accepts={total}")
    st.logger.info(f"Counts: T1={counts['task1']}  T2={counts['task2']}  T3={counts['task3']}  T4={counts['task4']}")
    st.logger.info(f"Frames seen={st.FRAME_STATS['frames_seen']}  frames_with_min_views={st.FRAME_STATS['frames_with_min_views']}")

    top_rejects = _short_reject_report()
    if top_rejects:
        st.logger.info("Top reject reasons (so far): " + ", ".join([f"{k}={v}" for k, v in top_rejects]))
    else:
        st.logger.info("Top reject reasons (so far): none")

    peek = _short_last_samples(samples, k=3)
    st.logger.info("Last accepted samples (peek):")
    for p in peek:
        st.logger.info(json.dumps(p, indent=2))

    st.logger.info(f"[DIRS] raw_images_saved = {len(list(st.RAW_IMG_DIR.glob('*.jpg')))}")
    st.logger.info(f"[DIRS] debug_items = {len(list(st.DEBUG_DIR.glob('*')))}")

    st.logger.info("=" * 88)


def write_snapshot_if_needed(samples, counts, every=5):
    total = counts["task1"] + counts["task2"] + counts["task3"] + counts["task4"]
    if total % every != 0:
        return
    snap_path = st.RUN_DIR / "snapshot_progress.json"
    with open(snap_path, "w") as f:
        json.dump({
            "counts": counts,
            "reject_stats": st.REJECT_STATS,
            "frame_stats": st.FRAME_STATS,
            "last_samples": _short_last_samples(samples, k=5),
        }, f, indent=2)
    st.logger.info(f"[AUDIT] wrote snapshot: {snap_path}")


def write_frame_debug(split, seq, frame_id, tasks):
    """
    Write a per-frame debug summary with task JSON and a simple HTML viewer.
    """
    if not st.SAVE_DEBUG:
        return
    frame_dir = st.DEBUG_DIR / f"frame_{split}_{seq}_{frame_id}"
    frame_dir.mkdir(exist_ok=True)

    out = {
        "split": split,
        "seq": seq,
        "frame_id": frame_id,
        "tasks": tasks,
    }
    json_path = frame_dir / "tasks.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    # Build a lightweight HTML viewer
    def img_tag(p):
        try:
            rel = os.path.relpath(Path(p), start=frame_dir)
        except Exception:
            rel = str(p)
        rel_esc = html_lib.escape(rel, quote=True)
        return (
            f'<div style="display:inline-block; text-align:center; margin:6px;">'
            f'<a href="{rel_esc}" target="_blank">open</a><br>'
            f'<img src="{rel_esc}" style="max-width: 420px; margin: 6px;">'
            f'</div>'
        )

    html_lines = [
        "<html><head><meta charset='utf-8'><title>Frame Debug</title></head><body>",
        f"<h2>Frame: {html_lib.escape(str(split))}/{html_lib.escape(str(seq))} {html_lib.escape(str(frame_id))}</h2>",
        "<p><i>Note: This view lists only tasks that were accepted (written into benchmark_gazevqa.json).</i></p>",
    ]
    for t in tasks:
        html_lines.append(f"<h3>Task {html_lib.escape(str(t.get('task_id')))}</h3>")
        html_lines.append(f"<p><b>Accepted:</b> {html_lib.escape(str(t.get('_accepted')))}</p>")
        html_lines.append(f"<p><b>Question:</b> {html_lib.escape(str(t.get('question', '')))}</p>")
        html_lines.append(f"<p><b>Answer:</b> {html_lib.escape(str(t.get('answer', '')))}</p>")
        html_lines.append(f"<p><b>Reasoning:</b> {html_lib.escape(str(t.get('reasoning', '')))}</p>")
        meta = t.get("meta", {}) if isinstance(t.get("meta"), dict) else {}
        if t.get("task_id") == 1 and meta:
            html_lines.append(f"<p><b>Task1 anchor camera (mask-only view):</b> {html_lib.escape(str(meta.get('camera_id', '')))}</p>")
            html_lines.append(f"<p><b>Task1 anchor label:</b> {html_lib.escape(str(meta.get('anchor_canonical_object', '')))}</p>")
            html_lines.append(f"<p><b>Task1 final canonical label:</b> {html_lib.escape(str(meta.get('canonical_object', '')))}</p>")
            html_lines.append(f"<p><b>Task1 segmentation_try:</b> {html_lib.escape(str(meta.get('segmentation_try', '')))}</p>")
            html_lines.append(f"<p><b>Task1 canonical_mode:</b> {html_lib.escape(str(meta.get('canonical_mode', '')))}</p>")
            html_lines.append(f"<p><b>Task1 ray_available:</b> {html_lib.escape(str(meta.get('ray_available', '')))}</p>")
            mv_weights = meta.get("multiview_weight_map")
            if isinstance(mv_weights, dict) and mv_weights:
                html_lines.append(
                    f"<p><b>Task1 multiview weights:</b> {html_lib.escape(json.dumps(mv_weights, sort_keys=True))}</p>"
                )
            mv_coords = meta.get("multiview_coords_scaled")
            if isinstance(mv_coords, dict) and mv_coords:
                html_lines.append(
                    f"<p><b>Task1 multiview coords (scaled):</b> {html_lib.escape(json.dumps(mv_coords, sort_keys=True))}</p>"
                )
            label_flow = meta.get("label_flow")
            if isinstance(label_flow, list) and label_flow:
                html_lines.append(
                    f"<p><b>Task1 label_flow:</b> {html_lib.escape(json.dumps(label_flow, ensure_ascii=False))}</p>"
                )
            anchor_scores = meta.get("anchor_candidate_scores")
            if isinstance(anchor_scores, dict) and anchor_scores:
                ordered = sorted(
                    anchor_scores.items(),
                    key=lambda kv: float((kv[1] or {}).get("score", -1.0)),
                    reverse=True,
                )
                rank_txt = ", ".join(
                    f"{cam}:{float((info or {}).get('score', -1.0)):.3f}"
                    for cam, info in ordered
                )
                html_lines.append(f"<p><b>Task1 anchor ranking:</b> {html_lib.escape(rank_txt)}</p>")
        imgs = t.get("input_images", [])
        if imgs:
            html_lines.append("<div>")
            for ent in imgs:
                p = ent.get("image")
                if p:
                    html_lines.append(img_tag(p))
            html_lines.append("</div>")
        dbg_imgs = t.get("_debug_images", [])
        if dbg_imgs:
            html_lines.append("<div><b>Debug images:</b><br>")
            for p in dbg_imgs:
                html_lines.append(img_tag(p))
            html_lines.append("</div>")
    html_lines.append("</body></html>")

    html_path = frame_dir / "tasks.html"
    with open(html_path, "w") as f:
        f.write("\n".join(html_lines))
