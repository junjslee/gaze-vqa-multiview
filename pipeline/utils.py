import json
import os
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
        return (
            f'<div style="display:inline-block; text-align:center; margin:6px;">'
            f'<a href="{rel}" target="_blank">open</a><br>'
            f'<img src="{rel}" style="max-width: 420px; margin: 6px;">'
            f'</div>'
        )

    html = [
        "<html><head><meta charset='utf-8'><title>Frame Debug</title></head><body>",
        f"<h2>Frame: {split}/{seq} {frame_id}</h2>",
        "<p><i>Note: This view lists only tasks that were accepted (written into benchmark_gazevqa.json).</i></p>",
    ]
    for t in tasks:
        html.append(f"<h3>Task {t.get('task_id')}</h3>")
        html.append(f"<p><b>Accepted:</b> {t.get('_accepted')}</p>")
        html.append(f"<p><b>Question:</b> {t.get('question','')}</p>")
        html.append(f"<p><b>Answer:</b> {t.get('answer','')}</p>")
        html.append(f"<p><b>Reasoning:</b> {t.get('reasoning','')}</p>")
        imgs = t.get("input_images", [])
        if imgs:
            html.append("<div>")
            for ent in imgs:
                p = ent.get("image")
                if p:
                    html.append(img_tag(p))
            html.append("</div>")
        dbg_imgs = t.get("_debug_images", [])
        if dbg_imgs:
            html.append("<div><b>Debug images:</b><br>")
            for p in dbg_imgs:
                html.append(img_tag(p))
            html.append("</div>")
    html.append("</body></html>")

    html_path = frame_dir / "tasks.html"
    with open(html_path, "w") as f:
        f.write("\n".join(html))
