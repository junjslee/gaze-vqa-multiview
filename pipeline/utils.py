import json
import os
import time
import html as html_lib
from pathlib import Path
from collections import Counter
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


def _atomic_write_json(path: Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def checkpoint_bundle_key(split, seq, frame_id):
    return f"{split}/{seq}/{frame_id}"


def append_checkpoint_samples(samples, bundle_key):
    if not samples:
        return
    st.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(st.CHECKPOINT_ACCEPTED_SAMPLES_JSONL, "a") as f:
        for s in samples:
            rec = {
                "bundle_key": str(bundle_key),
                "task_id": int((s or {}).get("task_id") or 0) if isinstance(s, dict) else 0,
                "sample": s,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_checkpoint_bundle(bundle_key, split=None, seq=None, frame_id=None):
    st.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    rec = {
        "bundle_key": str(bundle_key),
        "split": split,
        "seq": seq,
        "frame_id": frame_id,
        "ts": int(time.time()),
    }
    with open(st.CHECKPOINT_ACCEPTED_BUNDLES_JSONL, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _recompute_counts_from_samples(samples, accepted_bundle_keys):
    counts = {"task1": 0, "task2": 0, "task3": 0, "task4": 0}
    for s in samples:
        if not isinstance(s, dict):
            continue
        tid = int(s.get("task_id") or 0)
        if tid in (1, 2, 3, 4):
            counts[f"task{tid}"] += 1
    if st.REQUIRE_ALL_TASKS_PER_FRAME:
        counts["bundle"] = int(len(accepted_bundle_keys))
    return counts


def write_resume_state(counts, sample_count, accepted_bundle_count, last_bundle_key=None, checkpoint_events=0):
    payload = {
        "run_name": st.RUN_NAME,
        "run_dir": str(st.RUN_DIR),
        "updated_at_epoch": int(time.time()),
        "counts": counts,
        "sample_count": int(sample_count),
        "accepted_bundle_count": int(accepted_bundle_count),
        "last_bundle_key": last_bundle_key,
        "checkpoint_events": int(checkpoint_events),
        "reject_stats": st.REJECT_STATS,
        "frame_stats": st.FRAME_STATS,
        "gemini_error_stats": st.GEMINI_ERROR_STATS,
    }
    _atomic_write_json(st.CHECKPOINT_STATE_JSON, payload)


def load_checkpoint_state():
    samples = []
    accepted_bundle_keys = set()
    state_payload = {}

    if st.CHECKPOINT_ACCEPTED_BUNDLES_JSONL.exists():
        with open(st.CHECKPOINT_ACCEPTED_BUNDLES_JSONL, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict):
                    bk = str(rec.get("bundle_key") or "").strip()
                else:
                    bk = str(rec).strip()
                if bk:
                    accepted_bundle_keys.add(bk)

    if st.CHECKPOINT_ACCEPTED_SAMPLES_JSONL.exists():
        with open(st.CHECKPOINT_ACCEPTED_SAMPLES_JSONL, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                sample = None
                if isinstance(rec, dict) and isinstance(rec.get("sample"), dict):
                    sample = rec.get("sample")
                    bk = str(rec.get("bundle_key") or "").strip()
                    if bk:
                        accepted_bundle_keys.add(bk)
                elif isinstance(rec, dict) and int(rec.get("task_id") or 0) in (1, 2, 3, 4):
                    sample = rec
                if isinstance(sample, dict):
                    samples.append(sample)

    if st.CHECKPOINT_STATE_JSON.exists():
        try:
            with open(st.CHECKPOINT_STATE_JSON, "r") as f:
                state_payload = json.load(f) or {}
        except Exception:
            state_payload = {}

    counts = state_payload.get("counts") if isinstance(state_payload, dict) else None
    if not isinstance(counts, dict):
        counts = _recompute_counts_from_samples(samples, accepted_bundle_keys)
    else:
        base = _recompute_counts_from_samples(samples, accepted_bundle_keys)
        for k, v in base.items():
            counts[k] = int(max(int(counts.get(k) or 0), int(v or 0)))
        if st.REQUIRE_ALL_TASKS_PER_FRAME:
            counts["bundle"] = int(max(int(counts.get("bundle") or 0), len(accepted_bundle_keys)))

    return {
        "samples": samples,
        "accepted_bundle_keys": accepted_bundle_keys,
        "counts": counts,
        "reject_stats": state_payload.get("reject_stats") if isinstance(state_payload, dict) else None,
        "frame_stats": state_payload.get("frame_stats") if isinstance(state_payload, dict) else None,
        "checkpoint_events": int(state_payload.get("checkpoint_events") or 0) if isinstance(state_payload, dict) else 0,
        "last_bundle_key": state_payload.get("last_bundle_key") if isinstance(state_payload, dict) else None,
    }


def write_run_status(status, reason="", extra=None):
    payload = {
        "status": str(status),
        "reason": str(reason or ""),
        "run_name": st.RUN_NAME,
        "run_dir": str(st.RUN_DIR),
        "timestamp_epoch": int(time.time()),
    }
    if isinstance(extra, dict) and extra:
        payload["extra"] = extra
    _atomic_write_json(st.RUN_STATUS_JSON, payload)


def _safe_inc_map(d, key, inc=1):
    k = str(key or "unknown")
    d[k] = int(d.get(k, 0)) + int(inc)


def log_gemini_error(obj: dict):
    rec = dict(obj or {})
    rec["timestamp_epoch"] = int(time.time())
    rec["frame_key"] = str(rec.get("frame_key") or st.CURRENT_FRAME_KEY or "")
    task_id = int(rec.get("task_id") or 0)
    stage = str(rec.get("stage") or "unknown")
    err_type = str(rec.get("error_type") or "unknown")
    model_name = str(rec.get("model") or rec.get("model_requested") or "")

    st.LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(st.GEMINI_ERROR_LOG_JSONL, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    st.REJECT_STATS["gemini_error_events"] = int(st.REJECT_STATS.get("gemini_error_events", 0)) + 1
    if task_id == 1:
        st.REJECT_STATS["t1_gemini_error"] = int(st.REJECT_STATS.get("t1_gemini_error", 0)) + 1
    elif task_id == 4:
        st.REJECT_STATS["t4_gemini_error"] = int(st.REJECT_STATS.get("t4_gemini_error", 0)) + 1

    st.GEMINI_ERROR_STATS["total"] = int(st.GEMINI_ERROR_STATS.get("total", 0)) + 1
    _safe_inc_map(st.GEMINI_ERROR_STATS.setdefault("by_task", {}), task_id if task_id else "unknown")
    _safe_inc_map(st.GEMINI_ERROR_STATS.setdefault("by_stage", {}), stage)
    _safe_inc_map(st.GEMINI_ERROR_STATS.setdefault("by_type", {}), err_type)
    _safe_inc_map(st.GEMINI_ERROR_STATS.setdefault("by_model", {}), model_name or "unknown")

    msg = str(rec.get("message") or rec.get("parse_reason") or "")[:240]
    st.logger.warning(
        "[GEMINI-ERROR] task=%s stage=%s type=%s model=%s frame=%s msg=%s",
        task_id if task_id else "unknown",
        stage,
        err_type,
        model_name or "unknown",
        rec.get("frame_key") or "unknown",
        msg,
    )


def seed_gemini_error_stats_from_log():
    st.GEMINI_ERROR_STATS = {
        "total": 0,
        "by_task": {},
        "by_stage": {},
        "by_type": {},
        "by_model": {},
    }
    if not st.GEMINI_ERROR_LOG_JSONL.exists():
        return
    with open(st.GEMINI_ERROR_LOG_JSONL, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            task_id = rec.get("task_id", "unknown")
            stage = rec.get("stage", "unknown")
            err_type = rec.get("error_type", "unknown")
            model_name = rec.get("model") or rec.get("model_requested") or "unknown"
            st.GEMINI_ERROR_STATS["total"] += 1
            _safe_inc_map(st.GEMINI_ERROR_STATS.setdefault("by_task", {}), task_id)
            _safe_inc_map(st.GEMINI_ERROR_STATS.setdefault("by_stage", {}), stage)
            _safe_inc_map(st.GEMINI_ERROR_STATS.setdefault("by_type", {}), err_type)
            _safe_inc_map(st.GEMINI_ERROR_STATS.setdefault("by_model", {}), model_name)


def write_gemini_error_summary(path=None):
    out_path = Path(path or st.GEMINI_ERROR_SUMMARY_JSON)
    out = {
        "run_name": st.RUN_NAME,
        "run_dir": str(st.RUN_DIR),
        "generated_at_epoch": int(time.time()),
        "stats": st.GEMINI_ERROR_STATS,
        "log_path": str(st.GEMINI_ERROR_LOG_JSONL),
    }
    _atomic_write_json(out_path, out)
    return str(out_path)


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


def _ratio(num, den):
    den = int(den or 0)
    if den <= 0:
        return None
    return float(num) / float(den)


def summarize_teacher_verifier(samples):
    t1_samples = 0
    t1_applied = 0
    t1_fallback = 0
    t1_parse_attempts = 0
    t1_parse_success = 0
    t1_partial_retries = 0
    t1_finish = Counter()
    t1_modes = Counter()
    t1_models = Counter()
    t1_model_fallback = Counter()

    t4_samples = 0
    t4_parse_success = 0
    t4_partial_retries = 0
    t4_actions = Counter()
    t4_presence = Counter()
    t4_person_presence = Counter()
    t4_finish = Counter()
    t4_modes = Counter()
    t4_models = Counter()
    t4_model_fallback = Counter()

    for s in samples:
        meta = s.get("meta") if isinstance(s, dict) else None
        if not isinstance(meta, dict):
            continue
        if int(s.get("task_id") or 0) == 1:
            tf = meta.get("teacher_final")
            if not isinstance(tf, dict) or not bool(tf.get("enabled")):
                continue
            t1_samples += 1
            if bool(tf.get("applied")):
                t1_applied += 1
            if str(tf.get("final_source") or "") == "student_fallback":
                t1_fallback += 1
            for key in ("pass1", "pass2"):
                rec = tf.get(key)
                if not isinstance(rec, dict) or not rec:
                    continue
                t1_parse_attempts += 1
                if bool(rec.get("parse_ok")):
                    t1_parse_success += 1
                if int(rec.get("retry_count") or 0) > 0:
                    t1_partial_retries += 1
                fr = str(rec.get("finish_reason") or "").strip().upper()
                if fr:
                    t1_finish[fr] += 1
                mode = str(rec.get("gemini_mode") or "").strip().lower()
                if mode:
                    t1_modes[mode] += 1
                mu = str(rec.get("model") or "").strip()
                mr = str(rec.get("model_requested") or "").strip()
                if mu:
                    t1_models[mu] += 1
                if mu and mr and mu.lower() != mr.lower():
                    t1_model_fallback[f"{mr} -> {mu}"] += 1

        if int(s.get("task_id") or 0) == 4:
            gv = meta.get("task4_gemini_verifier")
            if not isinstance(gv, dict) or not bool(gv.get("enabled")):
                continue
            t4_samples += 1
            if bool(gv.get("parse_ok")) and str(gv.get("prediction") or "").upper() in {"YES", "NO"}:
                t4_parse_success += 1
            if int(gv.get("retry_count") or 0) > 0:
                t4_partial_retries += 1
            act = str(gv.get("action") or "").strip().lower()
            if act:
                t4_actions[act] += 1
            pres = str(gv.get("presence") or "").strip().upper()
            if pres:
                t4_presence[pres] += 1
            person_pres = str(gv.get("person_presence") or "").strip().upper()
            if person_pres:
                t4_person_presence[person_pres] += 1
            fr = str(gv.get("finish_reason") or "").strip().upper()
            if fr:
                t4_finish[fr] += 1
            mode = str(gv.get("gemini_mode") or "").strip().lower()
            if mode:
                t4_modes[mode] += 1
            mu = str(gv.get("model") or "").strip()
            mr = str(gv.get("model_requested") or "").strip()
            if mu:
                t4_models[mu] += 1
            if mu and mr and mu.lower() != mr.lower():
                t4_model_fallback[f"{mr} -> {mu}"] += 1

    return {
        "task1_teacher": {
            "samples": t1_samples,
            "applied": t1_applied,
            "applied_rate": _ratio(t1_applied, t1_samples),
            "student_fallback": t1_fallback,
            "parse_attempts": t1_parse_attempts,
            "parse_success": t1_parse_success,
            "parse_success_rate": _ratio(t1_parse_success, t1_parse_attempts),
            "partial_retry_calls": t1_partial_retries,
            "finish_reasons": dict(t1_finish),
            "gemini_modes": dict(t1_modes),
            "models_used": dict(t1_models),
            "model_fallback_usage": dict(t1_model_fallback),
        },
        "task4_verifier": {
            "samples": t4_samples,
            "parse_success": t4_parse_success,
            "parse_success_rate": _ratio(t4_parse_success, t4_samples),
            "partial_retry_calls": t4_partial_retries,
            "actions": dict(t4_actions),
            "presence": dict(t4_presence),
            "person_presence": dict(t4_person_presence),
            "finish_reasons": dict(t4_finish),
            "gemini_modes": dict(t4_modes),
            "models_used": dict(t4_models),
            "model_fallback_usage": dict(t4_model_fallback),
        },
    }


def write_gemini_teacher_verifier_report(samples, path=None):
    report = summarize_teacher_verifier(samples)
    out_path = Path(path or st.GEMINI_TEACHER_VERIFIER_REPORT_JSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return str(out_path)


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
    tv = summarize_teacher_verifier(samples)
    t1 = tv.get("task1_teacher", {})
    t4 = tv.get("task4_verifier", {})
    if int(t1.get("samples") or 0) > 0:
        st.logger.info(
            "Task1 teacher stats: "
            f"applied={t1.get('applied')}/{t1.get('samples')} "
            f"({(t1.get('applied_rate') or 0.0):.3f}), "
            f"parse={t1.get('parse_success')}/{t1.get('parse_attempts')} "
            f"({(t1.get('parse_success_rate') or 0.0):.3f}), "
            f"partial_retry_calls={t1.get('partial_retry_calls')}"
        )
    if int(t4.get("samples") or 0) > 0:
        st.logger.info(
            "Task4 verifier stats: "
            f"parse={t4.get('parse_success')}/{t4.get('samples')} "
            f"({(t4.get('parse_success_rate') or 0.0):.3f}), "
            f"actions={json.dumps(t4.get('actions', {}), sort_keys=True)}"
        )

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
    snap_path = st.SNAPSHOT_JSON
    with open(snap_path, "w") as f:
        json.dump({
            "counts": counts,
            "reject_stats": st.REJECT_STATS,
            "frame_stats": st.FRAME_STATS,
            "gemini_errors": st.GEMINI_ERROR_STATS,
            "checkpoint": st.CHECKPOINT_STATE,
            "gemini_teacher_verifier": summarize_teacher_verifier(samples),
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
            tf = meta.get("teacher_final")
            if isinstance(tf, dict) and tf:
                html_lines.append(f"<p><b>Task1 teacher applied:</b> {html_lib.escape(str(tf.get('applied')))}</p>")
                html_lines.append(f"<p><b>Task1 teacher source:</b> {html_lib.escape(str(tf.get('final_source')))}</p>")
                html_lines.append(f"<p><b>Task1 teacher calls:</b> {html_lib.escape(str(tf.get('call_count')))}</p>")
                for pkey in ("pass1", "pass2"):
                    prec = tf.get(pkey)
                    if isinstance(prec, dict) and prec:
                        raw_snip = str(prec.get("raw") or "")
                        if len(raw_snip) > 180:
                            raw_snip = raw_snip[:180] + "..."
                        compact = {
                            "final_label": prec.get("final_label"),
                            "confidence": prec.get("confidence"),
                            "parse_ok": prec.get("parse_ok"),
                            "parse_status": prec.get("parse_status"),
                            "parse_reason": prec.get("parse_reason"),
                            "finish_reason": prec.get("finish_reason"),
                            "retry_count": prec.get("retry_count"),
                            "model_requested": prec.get("model_requested"),
                            "model": prec.get("model"),
                            "gemini_api_version": prec.get("gemini_api_version"),
                            "gemini_mode": prec.get("gemini_mode"),
                            "raw_snippet": raw_snip,
                        }
                        html_lines.append(
                            f"<p><b>Task1 teacher {pkey}:</b> {html_lib.escape(json.dumps(compact, ensure_ascii=False))}</p>"
                        )
        if t.get("task_id") == 4 and meta:
            gv = meta.get("task4_gemini_verifier")
            if isinstance(gv, dict) and gv:
                raw_snip = str(gv.get("raw") or "")
                if len(raw_snip) > 180:
                    raw_snip = raw_snip[:180] + "..."
                compact = {
                    "prediction": gv.get("prediction"),
                    "presence": gv.get("presence"),
                    "person_presence": gv.get("person_presence"),
                    "confidence": gv.get("confidence"),
                    "action": gv.get("action"),
                    "parse_ok": gv.get("parse_ok"),
                    "parse_status": gv.get("parse_status"),
                    "parse_reason": gv.get("parse_reason"),
                    "finish_reason": gv.get("finish_reason"),
                    "retry_count": gv.get("retry_count"),
                    "model_requested": gv.get("model_requested"),
                    "model": gv.get("model"),
                    "gemini_api_version": gv.get("gemini_api_version"),
                    "gemini_mode": gv.get("gemini_mode"),
                    "raw_snippet": raw_snip,
                }
                html_lines.append(
                    f"<p><b>Task4 Gemini verifier:</b> {html_lib.escape(json.dumps(compact, ensure_ascii=False))}</p>"
                )
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
