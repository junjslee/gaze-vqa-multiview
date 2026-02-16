# task1.py
import json
import re
import random
import traceback
import math
import numpy as np
from PIL import Image, ImageDraw

from . import state as st
from .annotations import parse_visibility, has_coord, get_body_bbox, get_head_bbox, scale_annotations_for_resized_image
from .io_utils import zip_try_image_path, zip_read_image, save_raw_cam_images_parallel
from .sam2_utils import (
    segment_object_at_gaze,
    segment_object_on_crop,
    overlay_mask_on_image,
    overlay_mask_on_image_neutral,
    draw_dot_on_crop,
    mask_person_overlap_ratio,
)
from .vlm import vlm_generate, strict_noun_phrase, _first_two_sentences, clean_label, choose_by_letter
from . import prompts
from .utils import _resize, make_id, log_debug


def draw_gaze_ray_overlay(im, anno_scaled):
    im = im.copy()
    if not isinstance(anno_scaled, dict):
        return im

    vis = parse_visibility(anno_scaled)
    coord = anno_scaled.get("coordinate", None)
    if not (isinstance(coord, (list, tuple)) and len(coord) == 2):
        return im

    gx, gy = float(coord[0]), float(coord[1])

    ec = None
    eyes = anno_scaled.get("eye", None)
    if isinstance(eyes, list) and len(eyes) > 0:
        xs = [e[0] for e in eyes if isinstance(e, (list, tuple)) and len(e) >= 2]
        ys = [e[1] for e in eyes if isinstance(e, (list, tuple)) and len(e) >= 2]
        if xs and ys:
            ec = (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))

    hc = None
    hb = get_head_bbox(anno_scaled)
    if hb is not None:
        x, y, w, h = hb
        hc = (float(x + w / 2.0), float(y + h / 2.0))

    sx, sy = ec if ec is not None else (hc if hc is not None else (gx, gy))

    W, H = im.size

    def clamp_to_border(x0, y0, x1, y1):
        dx, dy = (x1 - x0), (y1 - y0)
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return (max(0, min(W - 1, x1)), max(0, min(H - 1, y1)))
        ts = []
        for xb in (0, W - 1):
            if abs(dx) > 1e-6:
                t = (xb - x0) / dx
                yb = y0 + t * dy
                if t >= 0 and 0 <= yb <= H - 1:
                    ts.append((t, xb, yb))
        for yb in (0, H - 1):
            if abs(dy) > 1e-6:
                t = (yb - y0) / dy
                xb = x0 + t * dx
                if t >= 0 and 0 <= xb <= W - 1:
                    ts.append((t, xb, yb))
        if not ts:
            return (max(0, min(W - 1, x1)), max(0, min(H - 1, y1)))
        ts.sort(key=lambda z: z[0])
        _, xb, yb = ts[0]
        return (float(xb), float(yb))

    if vis is False:
        ex, ey = clamp_to_border(sx, sy, gx, gy)
        draw_arrow = False
    else:
        dx, dy = (gx - sx), (gy - sy)
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            ex, ey = gx, gy
            draw_arrow = False
        else:
            ux, uy = dx / norm, dy / norm
            offset = max(0.0, float(getattr(st, "GAZE_ARROW_OFFSET_PX", 8)))
            offset = min(offset, max(0.0, norm - 1.0))
            ex, ey = gx - (ux * offset), gy - (uy * offset)
            draw_arrow = True

    draw = ImageDraw.Draw(im)
    draw.line([(sx, sy), (ex, ey)], fill=st.GAZE_COLOR, width=st.GAZE_LINE_W)
    if draw_arrow:
        dx, dy = (ex - sx), (ey - sy)
        norm = math.hypot(dx, dy)
        if norm >= 1e-6:
            ux, uy = dx / norm, dy / norm
            px, py = -uy, ux
            arrow_len = max(4.0, float(getattr(st, "GAZE_ARROW_LEN", 12)))
            arrow_half_w = max(2.0, float(getattr(st, "GAZE_ARROW_HALF_W", 5)))
            base_x = ex - (ux * arrow_len)
            base_y = ey - (uy * arrow_len)
            left = (base_x + (px * arrow_half_w), base_y + (py * arrow_half_w))
            right = (base_x - (px * arrow_half_w), base_y - (py * arrow_half_w))
            draw.polygon([(ex, ey), left, right], fill=st.GAZE_COLOR)

    return im


def _draw_person_bbox_overlay(im, body_bbox_xywh, color=(0, 255, 255), width=3):
    if im is None or body_bbox_xywh is None:
        return im
    out = im.copy()
    try:
        x, y, w, h = [float(v) for v in body_bbox_xywh]
        draw = ImageDraw.Draw(out)
        draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
    except Exception:
        return out
    return out


def build_ray_label_prompt_image(raw_img_pil, anno_scaled, body_bbox_xywh=None, target_mask_u8=None):
    """
    Build a cue-rich image for ray-label VQA:
    person cue (bbox), target mask, gaze ray + arrow marker.
    """
    cue = raw_img_pil.copy()
    if target_mask_u8 is not None:
        try:
            # Neutral dim overlay to avoid injecting red hue into object appearance.
            cue = overlay_mask_on_image_neutral(
                cue,
                (target_mask_u8 > 0).astype(np.uint8),
                dim_outside=0.55,
            )
        except Exception:
            pass
    cue = _draw_person_bbox_overlay(cue, body_bbox_xywh, color=(0, 255, 255), width=3)
    cue = draw_gaze_ray_overlay(cue, anno_scaled)
    return cue


def generate_target_description(ray_img_pil, person_desc, anchor_cam, scene_type=None):
    prompt = prompts.prompt_target_description_ray(person_desc, anchor_cam, scene_type=scene_type)
    raw = vlm_generate([ray_img_pil], prompt, max_new_tokens=120)
    desc = _first_two_sentences(raw)
    if not desc:
        desc = "The gaze appears to land on a specific object in the scene."
    return desc


def _filter_object_phrase(phrase):
    if not phrase:
        return ""
    low = phrase.strip().lower()
    banned = ("dot", "line", "ray", "marker", "overlay", "circle", "point", "pointer", "arrow", "arrowhead")
    if any(b in low for b in banned):
        return ""
    return phrase


def _clean_person_desc(s):
    if not s:
        return "person"
    low = str(s).strip().lower()
    if "no person visible" in low or "no person is visible" in low:
        return "person"
    low = re.sub(r"^(a|the)\s+", "", low)
    low = re.sub(r"^(person|man|woman|boy|girl)\s+(is\s+)?(wearing|with)\s+", "", low)
    low = re.sub(r"^person\s+", "", low)
    low = re.sub(r"^(wearing|with)\s+", "", low)
    low = re.sub(r"\b(and|with)\s*$", "", low).strip()
    return low if low else "person"


def _is_generic_label(label):
    if not label:
        return True
    low = str(label).strip().lower()
    toks = low.split()
    if toks and toks[0] in {"person", "man", "woman", "boy", "girl", "people"}:
        return True
    if low in st.BAD_OBJECTS or low in st.BAD_GENERIC_PHRASES:
        return True
    if low in ("furniture", "wooden furniture", "decor", "room decor"):
        return True
    if len(toks) <= 2 and any(t in st.BAD_GENERIC_WORDS for t in toks):
        return True
    return False


def _prefer_specific_label(canon, mask_label, ray_label, dot_label):
    if canon and not _is_generic_label(canon):
        return canon, None
    for cand in (mask_label, ray_label, dot_label):
        if cand and not _is_generic_label(cand):
            return cand, "prefer_specific"
    return canon, None


def _label_specificity_score(label):
    if not label:
        return -1.0
    toks = str(label).split()
    score = float(len(toks))
    if _is_generic_label(label):
        score -= 2.0
    if _has_on_phrase(label):
        score += 0.5
    return score


def _pick_most_specific_label(labels):
    best = None
    best_score = -1e9
    for lab in labels or []:
        lab = _sanitize_label(lab)
        if not lab:
            continue
        score = _label_specificity_score(lab)
        if score > best_score or (score == best_score and len(lab) > len(best or "")):
            best = lab
            best_score = score
    if best:
        return best
    return labels[0] if labels else None


_BAD_NOUN_TOKENS = {
    "is", "are", "was", "were", "be", "being", "been", "standing", "sitting", "wearing",
    "near", "next", "to", "left", "right", "behind", "infront", "front", "back",
    "on", "under", "above", "below", "with", "and", "or", "the"
}


def _has_on_phrase(s):
    return " on " in str(s).lower() if s else False


def _labels_token_subset_match(a, b):
    if not a or not b:
        return False
    if _has_on_phrase(a) or _has_on_phrase(b):
        return False
    a_clean = _strip_after_conjunction(a)
    b_clean = _strip_after_conjunction(b)
    toks_a = [t for t in str(a_clean).lower().split() if t]
    toks_b = [t for t in str(b_clean).lower().split() if t]
    if not toks_a or not toks_b:
        return False
    set_a = set(toks_a)
    set_b = set(toks_b)
    return set_a.issubset(set_b) or set_b.issubset(set_a)


def _labels_relaxed_match(a, b):
    a_s = _sanitize_label(a)
    b_s = _sanitize_label(b)
    if not a_s or not b_s:
        return False
    if a_s == b_s or _labels_token_subset_match(a_s, b_s):
        return True
    a_root = a_s.split(" on ", 1)[0].strip()
    b_root = b_s.split(" on ", 1)[0].strip()
    if not a_root or not b_root:
        return False
    return a_root == b_root or _labels_token_subset_match(a_root, b_root)


def _squash_on_phrase(s):
    if not s or " on " not in str(s).lower():
        return s
    parts = [p.strip() for p in str(s).split(" on ") if p.strip()]
    if len(parts) <= 2:
        return s
    return f"{parts[0]} on {parts[1]}"


def _strip_after_conjunction(s):
    if not s:
        return s
    low = str(s).lower()
    for sep in (" or ", " and "):
        if sep in low:
            return str(s)[:low.index(sep)].strip()
    return s


def _is_bleeding_label(s):
    if not s:
        return False
    low = str(s).lower().strip()
    if " or " in low or " and " in low:
        return True
    if low.endswith(" on") or low.endswith(" on "):
        return True
    return False


def _sanitize_label(s):
    s = _strip_after_conjunction(s)
    s = _squash_on_phrase(s)
    if not s:
        return s
    toks = str(s).strip().split()
    while toks and toks[-1].lower() in {"in", "on", "at", "with", "of", "to", "from", "near", "by", "for"}:
        toks = toks[:-1]
    s = " ".join(toks).strip()
    if not s:
        return None
    return s


def _is_ambiguous_label(label):
    if not label:
        return True
    if _is_generic_label(label):
        return True
    if _is_bleeding_label(label):
        return True
    return False


def _is_person_like_label(label):
    if not label:
        return False
    low = str(label).strip().lower()
    return bool(re.search(r"\b(person|man|woman|boy|girl|human|people)\b", low))


def _is_support_surface_label(label):
    if not label:
        return False
    low = str(label).strip().lower()
    surface_terms = {
        "table", "desk", "counter", "shelf", "cabinet", "bench", "stool",
        "stand", "rack", "piano", "whiteboard", "board",
    }
    toks = set(low.split())
    return any(t in surface_terms for t in toks)


def _should_override_anchor_with_mv(anchor_label, anchor_mode, mv_canon, mv_labels, anchor_cam):
    if not mv_canon:
        return False, "mv_missing"
    if not anchor_label:
        return True, "no_anchor"

    anchor_mode_l = str(anchor_mode or "").lower()
    anchor_locked = (not _is_ambiguous_label(anchor_label)) and ("mask_small_skip" not in anchor_mode_l)

    if anchor_label == mv_canon or _labels_token_subset_match(anchor_label, mv_canon):
        return True, "anchor_match"

    agree_total = 0
    agree_strong = 0
    non_anchor_total = 0

    for cam, ent in (mv_labels or {}).items():
        if cam == anchor_cam:
            continue
        lab = _sanitize_label(ent.get("label"))
        if not lab:
            continue
        non_anchor_total += 1
        mode_l = str(ent.get("mode") or "").lower()
        strong = "mask_small_skip" not in mode_l
        if lab == mv_canon or _labels_token_subset_match(lab, mv_canon):
            agree_total += 1
            if strong:
                agree_strong += 1

    if not anchor_locked:
        if agree_total >= 1 and non_anchor_total >= 1:
            return True, "anchor_uncertain_mv_support"
        return False, "anchor_uncertain_no_support"

    if agree_strong >= 2:
        return True, "mv_strong_consensus"
    return False, "mv_consensus_weak"


def _task1_informative_label_map(label_map):
    out = {}
    for k, v in (label_map or {}).items():
        s = _sanitize_label(v)
        s = _filter_object_phrase(s)
        s = _sanitize_label(s)
        if s:
            out[str(k)] = s
    return out


def _task1_has_semantic_disagreement(label_map):
    vals = [v for v in (label_map or {}).values() if v and not _is_generic_label(v)]
    if len(vals) < 2:
        return False
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if not _labels_relaxed_match(vals[i], vals[j]):
                return True
    return False


def _task1_should_run_semantic_arbiter(current_label, label_map):
    cur = _sanitize_label(current_label)
    if _is_ambiguous_label(cur):
        return True, "current_ambiguous"
    if _task1_has_semantic_disagreement(label_map):
        return True, "cue_disagreement"
    return False, "no_trigger"


def _parse_task1_semantic_arbiter_output(raw):
    txt = (raw or "").strip()
    final_label = None
    decision = "UNSURE"
    confidence = "LOW"
    rationale = None

    if txt:
        m = re.search(r"FINAL_LABEL\s*:\s*(.+)", txt, flags=re.I)
        if m:
            final_label = m.group(1).strip()
        m = re.search(r"DECISION\s*:\s*([A-Z_]+)", txt, flags=re.I)
        if m:
            decision = m.group(1).strip().upper()
        m = re.search(r"CONFIDENCE\s*:\s*([A-Z]+)", txt, flags=re.I)
        if m:
            confidence = m.group(1).strip().upper()
        m = re.search(r"RATIONALE\s*:\s*(.+)", txt, flags=re.I)
        if m:
            rationale = _first_two_sentences(m.group(1).strip())

    if not final_label:
        final_label = strict_noun_phrase(txt, max_words=4)
    if not final_label:
        final_label = clean_label(txt, max_words=4)
    final_label = _filter_object_phrase(final_label)
    final_label = _sanitize_label(final_label)
    if final_label and _is_generic_label(final_label):
        final_label = None

    if confidence not in {"HIGH", "MEDIUM", "LOW"}:
        confidence = "LOW"
    if not decision:
        decision = "UNSURE"

    return {
        "final_label": final_label,
        "decision": decision,
        "confidence": confidence,
        "rationale": rationale,
        "raw": txt,
    }


def _run_task1_semantic_arbiter(
    person_desc,
    anchor_cam,
    scene_type,
    anchor_resized,
    ray_label_prompt_pil,
    masked_crop,
    dot_mask_crop,
    candidate_labels,
    mask_area_ratio=None,
    ray_available=True,
):
    prompt = prompts.prompt_task1_semantic_arbiter(
        person_desc,
        anchor_cam,
        candidate_labels,
        scene_type=scene_type,
        mask_area_ratio=mask_area_ratio,
        ray_available=ray_available,
    )
    imgs = [masked_crop, dot_mask_crop, ray_label_prompt_pil, anchor_resized]
    imgs = [im for im in imgs if im is not None]
    raw = vlm_generate(imgs, prompt, max_new_tokens=90)
    parsed = _parse_task1_semantic_arbiter_output(raw)
    parsed["images_used"] = len(imgs)
    return parsed


def _conf_rank(conf):
    return {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(str(conf or "LOW").upper(), 0)


def _extract_json_obj(raw):
    txt = (raw or "").strip()
    if not txt:
        return None
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.I)
        txt = re.sub(r"\s*```$", "", txt)
        txt = txt.strip()
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _extract_partial_json_value(txt, keys):
    s = str(txt or "")
    for key in keys:
        # Accept both fully closed and truncated string values.
        m = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"\n\r}}]*)', s, flags=re.I)
        if m:
            v = m.group(1).strip()
            if v:
                return v
    return None


def _task1_teacher_scene_setting(scene_setting):
    ss = str(scene_setting or "").strip().upper()
    if ss not in {"COMMON_AREA", "OFFICE", "STORAGE_AREA", "KITCHEN", "LAB", "SHOP", "OTHER", "UNCLEAR"}:
        ss = "UNCLEAR"
    return ss


def _parse_task1_teacher_output_json(raw):
    txt = (raw or "").strip()
    out = {
        "final_label": None,
        "qwen_verdict": "UNCLEAR",
        "confidence": "LOW",
        "scene_setting": "UNCLEAR",
        "target_local_context": None,
        "rationale": None,
        "parse_ok": False,
        "parse_status": "invalid",
        "parse_partial": False,
        "parse_reason": None,
        "raw": txt,
    }
    obj = _extract_json_obj(txt)
    recovered_partial = False
    if isinstance(obj, dict):
        out["final_label"] = obj.get("final_label") or obj.get("label")
        out["qwen_verdict"] = obj.get("qwen_verdict") or obj.get("verdict") or out["qwen_verdict"]
        out["confidence"] = obj.get("confidence") or out["confidence"]
        out["scene_setting"] = obj.get("scene_setting") or out["scene_setting"]
        out["target_local_context"] = obj.get("target_local_context")
        out["rationale"] = obj.get("rationale")
    else:
        # Recover fields from partial/truncated JSON fragments first.
        pl = _extract_partial_json_value(txt, ["final_label", "label"])
        if pl:
            out["final_label"] = pl
            recovered_partial = True
        pv = _extract_partial_json_value(txt, ["qwen_verdict", "verdict"])
        if pv:
            out["qwen_verdict"] = pv
            recovered_partial = True
        pc = _extract_partial_json_value(txt, ["confidence"])
        if pc:
            out["confidence"] = pc
            recovered_partial = True
        m = re.search(r"FINAL_LABEL\s*:\s*(.+)", txt, flags=re.I)
        if m:
            out["final_label"] = m.group(1).strip()
        m = re.search(r"QWEN_VERDICT\s*:\s*([A-Z_]+)", txt, flags=re.I)
        if m:
            out["qwen_verdict"] = m.group(1).strip().upper()
        m = re.search(r"CONFIDENCE\s*:\s*([A-Z]+)", txt, flags=re.I)
        if m:
            out["confidence"] = m.group(1).strip().upper()
        m = re.search(r"SCENE_SETTING\s*:\s*([A-Z_]+)", txt, flags=re.I)
        if m:
            out["scene_setting"] = m.group(1).strip().upper()
        m = re.search(r"TARGET_LOCAL_CONTEXT\s*:\s*(.+)", txt, flags=re.I)
        if m:
            out["target_local_context"] = _first_two_sentences(m.group(1).strip())
        m = re.search(r"RATIONALE\s*:\s*(.+)", txt, flags=re.I)
        if m:
            out["rationale"] = _first_two_sentences(m.group(1).strip())

    if not out["final_label"]:
        out["final_label"] = strict_noun_phrase(txt, max_words=4)
    if not out["final_label"]:
        out["final_label"] = clean_label(txt, max_words=4)
    out["final_label"] = _sanitize_label(_filter_object_phrase(out["final_label"]))
    if out["final_label"] and _is_generic_label(out["final_label"]):
        out["final_label"] = None

    qv = str(out.get("qwen_verdict") or "UNCLEAR").strip().upper()
    if qv not in {"CORRECT", "INCORRECT", "UNCLEAR"}:
        qv = "UNCLEAR"
    out["qwen_verdict"] = qv

    conf = str(out.get("confidence") or "LOW").strip().upper()
    if conf not in {"LOW", "MEDIUM", "HIGH"}:
        conf = "LOW"
    out["confidence"] = conf

    out["scene_setting"] = _task1_teacher_scene_setting(out.get("scene_setting"))
    if out.get("target_local_context"):
        out["target_local_context"] = _first_two_sentences(str(out["target_local_context"]).strip())
    else:
        out["target_local_context"] = None
    if out.get("rationale"):
        out["rationale"] = _first_two_sentences(str(out["rationale"]).strip())
    else:
        out["rationale"] = None
    if isinstance(obj, dict):
        out["parse_status"] = "ok"
        out["parse_ok"] = bool(out["final_label"])
        out["parse_partial"] = False
        out["parse_reason"] = "json_object" if out["final_label"] else "json_missing_label"
    elif not txt:
        out["parse_status"] = "empty"
        out["parse_ok"] = False
        out["parse_partial"] = False
        out["parse_reason"] = "empty_text"
    elif recovered_partial:
        out["parse_status"] = "partial_recovered"
        out["parse_ok"] = bool(out["final_label"])
        out["parse_partial"] = True
        out["parse_reason"] = "partial_json_fields"
    else:
        out["parse_status"] = "invalid"
        out["parse_ok"] = bool(out["final_label"])
        out["parse_partial"] = False
        out["parse_reason"] = "non_json_fallback"
    return out


def _task1_label_root(label):
    s = _sanitize_label(label)
    if not s:
        return None
    return _sanitize_label(s.split(" on ", 1)[0].strip())


def _task1_semantic_equivalent(label_a, label_b):
    if _labels_relaxed_match(label_a, label_b):
        return True
    ra = _task1_label_root(label_a)
    rb = _task1_label_root(label_b)
    if not ra or not rb:
        return False
    if _labels_relaxed_match(ra, rb):
        return True
    alias = {
        "couch": "sofa",
        "sofa": "sofa",
        "desk": "table",
        "table": "table",
        "cupboard": "cabinet",
        "cabinet": "cabinet",
    }
    ta = [alias.get(t, t) for t in ra.split()]
    tb = [alias.get(t, t) for t in rb.split()]
    sa = set(ta)
    sb = set(tb)
    if not sa or not sb:
        return False
    return sa.issubset(sb) or sb.issubset(sa)


def _task1_teacher_gen_cfg():
    cfg = {
        "temperature": float(getattr(st, "TASK1_TEACHER_TEMPERATURE", 0.2)),
    }
    if str(getattr(st, "TASK1_TEACHER_PROVIDER", "")).strip().lower() == "gemini":
        cfg.update({
            "thinking_level": str(getattr(st, "TASK1_TEACHER_GEMINI_THINKING_LEVEL", "medium")).strip().lower(),
            "thinking_budget": int(getattr(st, "TASK1_TEACHER_GEMINI_THINKING_BUDGET", 64)),
            "media_resolution": str(getattr(st, "TASK1_TEACHER_GEMINI_MEDIA_RESOLUTION", "high")).strip().lower(),
            "structured_json": bool(getattr(st, "TASK1_TEACHER_STRUCTURED_JSON", True)),
            "json_schema": {
                "type": "OBJECT",
                "properties": {
                    "final_label": {"type": "STRING"},
                    "qwen_verdict": {"type": "STRING", "enum": ["CORRECT", "INCORRECT", "UNCLEAR"]},
                    "confidence": {"type": "STRING", "enum": ["HIGH", "MEDIUM", "LOW"]},
                },
                "required": ["final_label", "qwen_verdict", "confidence"],
            },
            "retry_on_partial_json": bool(getattr(st, "TASK1_TEACHER_RETRY_ON_PARTIAL_JSON", True)),
            "retry_token_multiplier": float(getattr(st, "TASK1_TEACHER_RETRY_TOKEN_MULTIPLIER", 2.0)),
        })
    return cfg


def _run_task1_teacher_pass1(
    person_desc,
    anchor_cam,
    scene_type,
    ray_context_pil,
    student_label,
    candidate_labels,
    mask_area_ratio=None,
):
    prompt = prompts.prompt_task1_teacher_pass1(
        person_desc=person_desc,
        anchor_cam=anchor_cam,
        student_label=student_label,
        candidate_labels=candidate_labels,
        scene_type=scene_type,
        mask_area_ratio=mask_area_ratio,
    )
    imgs = [ray_context_pil] if ray_context_pil is not None else []
    provider = str(getattr(st, "TASK1_TEACHER_PROVIDER", "gemini")).strip().lower()
    model = str(getattr(st, "TASK1_TEACHER_MODEL", "")).strip() or None
    resp = vlm_generate(
        imgs,
        prompt,
        max_new_tokens=int(getattr(st, "TASK1_TEACHER_MAX_NEW_TOKENS", 512)),
        provider=provider,
        model_id=model,
        generation_cfg=_task1_teacher_gen_cfg(),
        return_meta=True,
    )
    if isinstance(resp, dict):
        raw = str(resp.get("text") or "")
        model_used = str(resp.get("model_used") or (model or "provider_default"))
        api_version_used = str(resp.get("api_version_used") or "")
        mode_used = str(resp.get("mode_used") or "default")
        sig_count = int(resp.get("thought_signature_count") or 0)
        history_turn = resp.get("history_turn")
        model_requested = str(resp.get("model_requested") or (model or "provider_default"))
        finish_reason = str(resp.get("finish_reason") or "")
        retry_count = int(resp.get("retry_count") or 0)
        token_budget_used = int(resp.get("token_budget_used") or 0)
        schema_enabled = bool(resp.get("schema_enabled"))
        parse_health = resp.get("parse_health")
    else:
        raw = str(resp or "")
        model_used = model or "provider_default"
        api_version_used = ""
        mode_used = "default"
        sig_count = 0
        history_turn = None
        model_requested = model or "provider_default"
        finish_reason = ""
        retry_count = 0
        token_budget_used = 0
        schema_enabled = False
        parse_health = None
    out = _parse_task1_teacher_output_json(raw)
    out["provider"] = provider
    out["model"] = model_used
    out["model_requested"] = model_requested
    out["images_used"] = len(imgs)
    out["finish_reason"] = finish_reason
    out["retry_count"] = retry_count
    out["token_budget_used"] = token_budget_used
    out["schema_enabled"] = schema_enabled
    out["parse_health"] = parse_health
    if provider == "gemini":
        out["gemini_api_version"] = api_version_used
        out["gemini_mode"] = mode_used
        out["gemini_thought_signature_count"] = sig_count
        if isinstance(history_turn, list) and history_turn:
            out["_gemini_history_turn"] = history_turn
    return out


def _run_task1_teacher_pass2_conflict(
    person_desc,
    anchor_cam,
    scene_type,
    ray_context_pil,
    student_label,
    teacher_label_pass1,
    candidate_labels,
    prior_history_turn=None,
):
    prompt = prompts.prompt_task1_teacher_pass2_conflict(
        person_desc=person_desc,
        anchor_cam=anchor_cam,
        student_label=student_label,
        teacher_label_pass1=teacher_label_pass1,
        candidate_labels=candidate_labels,
        scene_type=scene_type,
    )
    imgs = [ray_context_pil] if ray_context_pil is not None else []
    provider = str(getattr(st, "TASK1_TEACHER_PROVIDER", "gemini")).strip().lower()
    model = str(getattr(st, "TASK1_TEACHER_MODEL", "")).strip() or None
    gen_cfg = _task1_teacher_gen_cfg()
    if provider == "gemini" and isinstance(prior_history_turn, list) and prior_history_turn:
        gen_cfg = dict(gen_cfg)
        gen_cfg["history_contents"] = prior_history_turn
    resp = vlm_generate(
        imgs,
        prompt,
        max_new_tokens=int(getattr(st, "TASK1_TEACHER_MAX_NEW_TOKENS", 512)),
        provider=provider,
        model_id=model,
        generation_cfg=gen_cfg,
        return_meta=True,
    )
    if isinstance(resp, dict):
        raw = str(resp.get("text") or "")
        model_used = str(resp.get("model_used") or (model or "provider_default"))
        api_version_used = str(resp.get("api_version_used") or "")
        mode_used = str(resp.get("mode_used") or "default")
        sig_count = int(resp.get("thought_signature_count") or 0)
        model_requested = str(resp.get("model_requested") or (model or "provider_default"))
        finish_reason = str(resp.get("finish_reason") or "")
        retry_count = int(resp.get("retry_count") or 0)
        token_budget_used = int(resp.get("token_budget_used") or 0)
        schema_enabled = bool(resp.get("schema_enabled"))
        parse_health = resp.get("parse_health")
    else:
        raw = str(resp or "")
        model_used = model or "provider_default"
        api_version_used = ""
        mode_used = "default"
        sig_count = 0
        model_requested = model or "provider_default"
        finish_reason = ""
        retry_count = 0
        token_budget_used = 0
        schema_enabled = False
        parse_health = None
    out = _parse_task1_teacher_output_json(raw)
    out["provider"] = provider
    out["model"] = model_used
    out["model_requested"] = model_requested
    out["images_used"] = len(imgs)
    out["finish_reason"] = finish_reason
    out["retry_count"] = retry_count
    out["token_budget_used"] = token_budget_used
    out["schema_enabled"] = schema_enabled
    out["parse_health"] = parse_health
    if provider == "gemini":
        out["gemini_api_version"] = api_version_used
        out["gemini_mode"] = mode_used
        out["gemini_thought_signature_count"] = sig_count
    return out


def _task1_should_run_teacher_second_call(student_label, teacher_label):
    if not bool(getattr(st, "TASK1_TEACHER_SECOND_CALL_ON_MISMATCH", True)):
        return False
    if not student_label or not teacher_label:
        return False
    return not _task1_semantic_equivalent(student_label, teacher_label)


def _task1_teacher_label_valid(label, confidence):
    lab = _sanitize_label(label)
    if not lab or _is_generic_label(lab):
        return False
    conf = str(confidence or "LOW").upper()
    min_conf = str(getattr(st, "TASK1_TEACHER_MIN_CONF", "MEDIUM")).upper()
    return _conf_rank(conf) >= _conf_rank(min_conf)


def _task1_should_run_hybrid_guardrail(current_label, label_map):
    cur = _sanitize_label(current_label)
    if _is_ambiguous_label(cur):
        return True, "current_ambiguous"
    if _task1_has_semantic_disagreement(label_map):
        return True, "cue_disagreement"
    return True, "always_on"


def _parse_task1_hybrid_guardrail_output(raw):
    txt = (raw or "").strip()
    out = {
        "final_label": None,
        "decision": "UNSURE",
        "confidence": "LOW",
        "scene_setting": "UNCLEAR",
        "target_local_context": None,
        "scene_fit_current": "UNCLEAR",
        "scene_fit_proposed": "UNCLEAR",
        "rationale": None,
        "raw": txt,
    }
    if txt:
        m = re.search(r"FINAL_LABEL\s*:\s*(.+)", txt, flags=re.I)
        if m:
            out["final_label"] = m.group(1).strip()
        m = re.search(r"DECISION\s*:\s*([A-Z_]+)", txt, flags=re.I)
        if m:
            out["decision"] = m.group(1).strip().upper()
        m = re.search(r"CONFIDENCE\s*:\s*([A-Z]+)", txt, flags=re.I)
        if m:
            out["confidence"] = m.group(1).strip().upper()
        m = re.search(r"SCENE_SETTING\s*:\s*([A-Z_]+)", txt, flags=re.I)
        if m:
            out["scene_setting"] = m.group(1).strip().upper()
        m = re.search(r"TARGET_LOCAL_CONTEXT\s*:\s*(.+)", txt, flags=re.I)
        if m:
            out["target_local_context"] = _first_two_sentences(m.group(1).strip())
        m = re.search(r"SCENE_FIT_CURRENT\s*:\s*([A-Z]+)", txt, flags=re.I)
        if m:
            out["scene_fit_current"] = m.group(1).strip().upper()
        m = re.search(r"SCENE_FIT_PROPOSED\s*:\s*([A-Z]+)", txt, flags=re.I)
        if m:
            out["scene_fit_proposed"] = m.group(1).strip().upper()
        m = re.search(r"RATIONALE\s*:\s*(.+)", txt, flags=re.I)
        if m:
            out["rationale"] = _first_two_sentences(m.group(1).strip())

    if not out["final_label"]:
        out["final_label"] = strict_noun_phrase(txt, max_words=4)
    if not out["final_label"]:
        out["final_label"] = clean_label(txt, max_words=4)
    out["final_label"] = _sanitize_label(_filter_object_phrase(out["final_label"]))
    if out["final_label"] and _is_generic_label(out["final_label"]):
        out["final_label"] = None

    if out["confidence"] not in {"LOW", "MEDIUM", "HIGH"}:
        out["confidence"] = "LOW"
    if out["scene_setting"] not in {
        "COMMON_AREA", "OFFICE", "STORAGE_AREA", "KITCHEN", "LAB", "SHOP", "OTHER", "UNCLEAR"
    }:
        out["scene_setting"] = "UNCLEAR"
    if not str(out.get("target_local_context") or "").strip():
        out["target_local_context"] = None
    if out["scene_fit_current"] not in {"YES", "NO", "UNCLEAR"}:
        out["scene_fit_current"] = "UNCLEAR"
    if out["scene_fit_proposed"] not in {"YES", "NO", "UNCLEAR"}:
        out["scene_fit_proposed"] = "UNCLEAR"
    if not out["decision"]:
        out["decision"] = "UNSURE"
    return out


def _build_task1_hybrid_guardrail_prompt(
    person_desc,
    anchor_cam,
    current_label,
    candidate_labels,
    scene_type=None,
    mask_area_ratio=None,
    scene_check=True,
):
    who = prompts.person_ref(person_desc)
    lines = []
    for key, val in (candidate_labels or {}).items():
        if val:
            lines.append(f"- {key}: {_sanitize_label(val)}")
    cand_block = "\n".join(lines) if lines else "- none"
    mar_txt = "N/A" if mask_area_ratio is None else f"{float(mask_area_ratio):.4f}"
    cur_txt = _sanitize_label(current_label) or "none"
    scene_line = (
        "Include scene-setting plausibility judgement for current/proposed labels."
        if scene_check else
        "Focus on cue consistency; scene-setting plausibility fields may be UNCLEAR."
    )
    return (
        "You are a strict gaze-target arbiter for dataset quality.\n"
        "Inputs are multi-cue candidates from mask, dot, ray, and multiview synthesis.\n"
        "Do not confuse cue markers (arrow/ring/crosshair) with real objects; if the true object is a tomato, output tomato/red tomato.\n"
        "Interpret images in order:\n"
        "1) full frame with gaze-ray cue 2) full raw frame.\n"
        f"Target person: {who}. Anchor camera: {anchor_cam}. Scene: {scene_type}.\n"
        f"Current label: {cur_txt}\n"
        f"Mask area ratio: {mar_txt}\n"
        "Candidate labels:\n"
        f"{cand_block}\n"
        f"{scene_line}\n"
        "First infer broad scene setting and local context near the gaze ray silently.\n"
        "Then infer semantic plausibility silently.\n"
        "Do NOT output internal chain-of-thought.\n"
        "Use SCENE_SETTING from this set: COMMON_AREA, OFFICE, STORAGE_AREA, KITCHEN, LAB, SHOP, OTHER, UNCLEAR.\n"
        "TARGET_LOCAL_CONTEXT should be a short phrase (3-8 words) describing nearby context around the gaze target.\n"
        "Then choose FINAL_LABEL by combining visual cues and candidate consistency.\n"
        "Return ONLY these lines:\n"
        "FINAL_LABEL: <1-4 words>\n"
        "DECISION: <KEEP|SWITCH_MASK|SWITCH_DOT|SWITCH_RAY|SWITCH_MV|REFINE|UNSURE>\n"
        "CONFIDENCE: <HIGH|MEDIUM|LOW>\n"
        "SCENE_SETTING: <COMMON_AREA|OFFICE|STORAGE_AREA|KITCHEN|LAB|SHOP|OTHER|UNCLEAR>\n"
        "TARGET_LOCAL_CONTEXT: <3-8 words>\n"
        "SCENE_FIT_CURRENT: <YES|NO|UNCLEAR>\n"
        "SCENE_FIT_PROPOSED: <YES|NO|UNCLEAR>\n"
        "RATIONALE: <max 16 words>"
    )


def _run_task1_hybrid_guardrail(
    person_desc,
    anchor_cam,
    scene_type,
    anchor_resized,
    ray_context_pil,
    masked_crop,
    dot_mask_crop,
    current_label,
    candidate_labels,
    mask_area_ratio=None,
):
    prompt = _build_task1_hybrid_guardrail_prompt(
        person_desc=person_desc,
        anchor_cam=anchor_cam,
        current_label=current_label,
        candidate_labels=candidate_labels,
        scene_type=scene_type,
        mask_area_ratio=mask_area_ratio,
        scene_check=bool(getattr(st, "TASK1_GUARDRAIL_SCENE_CHECK", True)),
    )
    # Guardrail now only sees global scene context with gaze cue + raw frame.
    imgs = [ray_context_pil, anchor_resized]
    imgs = [im for im in imgs if im is not None]
    guard_provider = str(getattr(st, "TASK1_GUARDRAIL_PROVIDER", "gemini")).strip().lower()
    guard_model = str(getattr(st, "TASK1_GUARDRAIL_MODEL", "")).strip() or None
    gen_cfg = None
    if guard_provider == "gemini":
        gen_cfg = {
            "temperature": 0.0,
            "top_p": float(getattr(st, "TASK1_GUARDRAIL_GEMINI_TOP_P", 0.85)),
            "top_k": int(getattr(st, "TASK1_GUARDRAIL_GEMINI_TOP_K", 32)),
            "thinking_budget": int(getattr(st, "TASK1_GUARDRAIL_GEMINI_THINKING_BUDGET", 48)),
        }
    raw = vlm_generate(
        imgs,
        prompt,
        max_new_tokens=int(getattr(st, "TASK1_GUARDRAIL_MAX_NEW_TOKENS", 96)),
        provider=guard_provider,
        model_id=guard_model,
        generation_cfg=gen_cfg,
    )
    parsed = _parse_task1_hybrid_guardrail_output(raw)
    parsed["images_used"] = len(imgs)
    parsed["provider"] = guard_provider
    parsed["model"] = guard_model or "provider_default"
    return parsed


def _parse_task1_qwen_guided_refine_output(raw):
    txt = (raw or "").strip()
    out = {"final_label": None, "confidence": "LOW", "rationale": None, "raw": txt}
    if txt:
        m = re.search(r"FINAL_LABEL\s*:\s*(.+)", txt, flags=re.I)
        if m:
            out["final_label"] = m.group(1).strip()
        m = re.search(r"CONFIDENCE\s*:\s*([A-Z]+)", txt, flags=re.I)
        if m:
            out["confidence"] = m.group(1).strip().upper()
        m = re.search(r"RATIONALE\s*:\s*(.+)", txt, flags=re.I)
        if m:
            out["rationale"] = _first_two_sentences(m.group(1).strip())
    if not out["final_label"]:
        out["final_label"] = strict_noun_phrase(txt, max_words=4)
    if not out["final_label"]:
        out["final_label"] = clean_label(txt, max_words=4)
    out["final_label"] = _sanitize_label(_filter_object_phrase(out["final_label"]))
    if out["final_label"] and _is_generic_label(out["final_label"]):
        out["final_label"] = None
    if out["confidence"] not in {"LOW", "MEDIUM", "HIGH"}:
        out["confidence"] = "LOW"
    return out


def _run_task1_qwen_guided_refine(
    current_label,
    guardrail_label,
    candidate_labels,
    anchor_resized,
    ray_label_prompt_pil,
    masked_crop,
    dot_mask_crop,
    guardrail_rationale=None,
    guardrail_scene_setting=None,
    guardrail_target_local_context=None,
):
    allow = []
    for v in (candidate_labels or {}).values():
        s = _sanitize_label(v)
        if s and s not in allow:
            allow.append(s)
    cur = _sanitize_label(current_label)
    grd = _sanitize_label(guardrail_label)
    if cur and cur not in allow:
        allow.append(cur)
    if grd and grd not in allow:
        allow.append(grd)
    allow_txt = ", ".join(allow[:10]) if allow else "none"
    prompt = (
        "You are selecting a final gaze-target label from constrained candidates.\n"
        "Prefer scene-consistent, concrete object identity and avoid generic terms.\n"
        f"Current label: {cur or 'none'}\n"
        f"Guardrail suggestion: {grd or 'none'}\n"
    )
    scene_setting = str(guardrail_scene_setting or "").strip().upper()
    if scene_setting and scene_setting != "UNCLEAR":
        prompt += f"Broad scene setting (from guardrail): {scene_setting}\n"
    local_ctx = _first_two_sentences(guardrail_target_local_context or "")
    if local_ctx:
        prompt += f"Target local context (from guardrail): {local_ctx}\n"
    hint = _first_two_sentences(guardrail_rationale or "")
    if hint:
        prompt += f"Scene hint from guardrail: {hint}\n"
    prompt += (
        f"Allowed labels: {allow_txt}\n"
        "Output ONLY:\n"
        "FINAL_LABEL: <one label from Allowed labels>\n"
        "CONFIDENCE: <HIGH|MEDIUM|LOW>\n"
        "RATIONALE: <max 14 words>"
    )
    imgs = [masked_crop, dot_mask_crop, ray_label_prompt_pil, anchor_resized]
    imgs = [im for im in imgs if im is not None]
    raw = vlm_generate(
        imgs,
        prompt,
        max_new_tokens=56,
        provider="qwen",
        model_id=str(getattr(st, "QWEN_MODEL_ID", "")).strip() or None,
        generation_cfg={"temperature": 0.0},
    )
    parsed = _parse_task1_qwen_guided_refine_output(raw)
    parsed["allowed_labels"] = allow
    parsed["images_used"] = len(imgs)
    return parsed


def _task1_label_cue_agreement(label, dot_label=None, ray_label=None):
    """
    Count lightweight cue agreements between a candidate mask label and
    point/ray-derived labels. Higher is better.
    """
    cand = _sanitize_label(label)
    if not cand:
        return 0
    hits = 0
    for cue in (dot_label, ray_label):
        cue_s = _sanitize_label(cue)
        if not cue_s:
            continue
        if cand == cue_s or _labels_token_subset_match(cand, cue_s):
            hits += 1
    return hits


def _task1_should_accept_attempt(seg_name, mask_area_ratio, cue_hits, label=None):
    # In loose-sweep -> strict order, strict is the terminal fallback.
    if seg_name == "strict":
        return True
    if str(seg_name).startswith("loose"):
        return False

    min_area = float(getattr(st, "TASK1_EARLY_ACCEPT_MIN_AREA_RATIO", 0.02))
    if mask_area_ratio is None or float(mask_area_ratio) < min_area:
        return False

    if label and _is_ambiguous_label(label):
        return False

    require_agree = bool(getattr(st, "TASK1_EARLY_ACCEPT_REQUIRE_CUE_AGREEMENT", True))
    if require_agree and cue_hits <= 0:
        return False
    return True


def _is_loose_stage(seg_name):
    return str(seg_name).startswith("loose")


def _pick_loose_baseline(current, candidate):
    if current is None:
        return candidate
    if candidate is None:
        return current

    cur_lab = _sanitize_label(current.get("canonical_object") or current.get("label"))
    new_lab = _sanitize_label(candidate.get("canonical_object") or candidate.get("label"))
    cur_bad = _is_ambiguous_label(cur_lab) or _is_generic_label(cur_lab)
    new_bad = _is_ambiguous_label(new_lab) or _is_generic_label(new_lab)
    if cur_bad and (not new_bad):
        return candidate
    if new_bad and (not cur_bad):
        return current

    cur_area = float(current.get("mask_area_ratio") or 0.0)
    new_area = float(candidate.get("mask_area_ratio") or 0.0)
    if new_area > cur_area * 1.05:
        return candidate
    if cur_area > new_area * 1.05:
        return current

    if _label_specificity_score(new_lab) > _label_specificity_score(cur_lab):
        return candidate
    return current


def _strict_refines_loose(strict_label, loose_label):
    strict_s = _sanitize_label(strict_label)
    loose_s = _sanitize_label(loose_label)
    if not strict_s:
        return False
    if not loose_s:
        return True
    if strict_s == loose_s:
        return True
    if _is_generic_label(strict_s) or _is_ambiguous_label(strict_s):
        return False
    if _labels_token_subset_match(strict_s, loose_s):
        return True
    if _label_specificity_score(strict_s) >= (_label_specificity_score(loose_s) + 0.7):
        return True
    return False


def _choose_mask_label(mask_label, refined_label, mask_area_ratio):
    """
    For larger masks, trust the primary mask label first to avoid tiny refined
    crops overriding the main object (e.g., couch -> small subpart/object).
    """
    mask_s = _sanitize_label(mask_label)
    refined_s = _sanitize_label(refined_label)

    if mask_area_ratio is not None:
        try:
            area = float(mask_area_ratio)
        except Exception:
            area = None
        if area is not None and area > max(0.015, float(st.TASK1_SMALL_OBJ_AREA_RATIO) * 1.8):
            return mask_s or refined_s

    if mask_s and refined_s and _labels_token_subset_match(mask_s, refined_s):
        return mask_s
    return _pick_most_specific_label([mask_s, refined_s])


def _on_relation_plausible(label, scene_type=None):
    if not label or " on " not in str(label).lower():
        return True
    prompt = prompts.prompt_on_relation_plausibility(label, scene_type=scene_type)
    raw = vlm_generate(None, prompt, max_new_tokens=4)
    if not raw:
        return True
    return raw.strip().lower().startswith("y")


def _append_label_flow(flow, stage, before, after, note=None):
    if flow is None:
        return
    b = _sanitize_label(before)
    a = _sanitize_label(after)
    if b == a:
        return
    flow.append({
        "stage": stage,
        "before": b,
        "after": a,
        "note": note,
    })


def _compose_positional_label(small_label, surface_label, max_words=6):
    small = clean_label(small_label, max_words=3)
    surface = clean_label(surface_label, max_words=3)
    if not small or not surface:
        return None
    if _has_on_phrase(small) or _has_on_phrase(surface):
        return None
    if small.strip().lower() == surface.strip().lower():
        return None
    phrase = f"{small} on {surface}"
    toks = phrase.split()
    if len(toks) > max_words:
        phrase = " ".join(toks[:max_words])
    return _filter_object_phrase(phrase)


def _make_collage(images, cols=3, bg=(0, 0, 0)):
    imgs = [im for im in images if im is not None]
    if not imgs:
        return None
    max_w = max(im.width for im in imgs)
    max_h = max(im.height for im in imgs)
    rows = int(math.ceil(len(imgs) / float(cols)))
    canvas = Image.new("RGB", (cols * max_w, rows * max_h), bg)
    for i, im in enumerate(imgs):
        im2 = im.resize((max_w, max_h))
        x = (i % cols) * max_w
        y = (i // cols) * max_h
        canvas.paste(im2, (x, y))
    return canvas


def _mask_to_pil(mask_u8):
    if mask_u8 is None:
        return None
    try:
        arr = (np.asarray(mask_u8) > 0).astype(np.uint8) * 255
        return Image.fromarray(arr, mode="L").convert("RGB")
    except Exception:
        return None


def _expand_bbox_xyxy(bb, full_wh, expand_ratio=2.0):
    if bb is None or full_wh is None:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bb]
    except Exception:
        return None
    W, H = int(full_wh[0]), int(full_wh[1])
    if W <= 1 or H <= 1:
        return None
    bw = max(1.0, x2 - x1 + 1.0)
    bh = max(1.0, y2 - y1 + 1.0)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    nw = max(2.0, bw * float(expand_ratio))
    nh = max(2.0, bh * float(expand_ratio))
    nx1 = int(max(0, min(W - 1, round(cx - 0.5 * nw))))
    ny1 = int(max(0, min(H - 1, round(cy - 0.5 * nh))))
    nx2 = int(max(0, min(W - 1, round(cx + 0.5 * nw))))
    ny2 = int(max(0, min(H - 1, round(cy + 0.5 * nh))))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return (nx1, ny1, nx2, ny2)


def _build_context_dot_crop(full_img_pil, bb, point_xy_scaled, expand_ratio=2.0):
    if full_img_pil is None or bb is None:
        return None
    ctx_bb = _expand_bbox_xyxy(bb, full_img_pil.size, expand_ratio=expand_ratio)
    if ctx_bb is None:
        return None
    x1, y1, x2, y2 = ctx_bb
    crop = full_img_pil.crop((x1, y1, x2 + 1, y2 + 1))
    return draw_dot_on_crop(
        crop, point_xy_scaled, ctx_bb, alpha=0.7, full_wh=full_img_pil.size, color=st.GAZE_COLOR
    )


def _should_use_small_mask_context(mask_area_ratio, mask_label):
    if not bool(getattr(st, "TASK1_SMALL_MASK_USE_CONTEXT", True)):
        return False
    area_trigger = float(getattr(st, "TASK1_SMALL_MASK_CONTEXT_AREA_RATIO", 0.03))
    area = float(mask_area_ratio) if mask_area_ratio is not None else 0.0
    if area > 0 and area <= area_trigger:
        return True
    return _is_ambiguous_label(mask_label)


def _salvage_noun_phrase(raw, max_words=4):
    s = clean_label(raw, max_words=10)
    if not s:
        return None
    toks = s.split()
    low = [t.lower() for t in toks]
    cut = None
    for i, t in enumerate(low):
        if t in _BAD_NOUN_TOKENS:
            cut = i
            break
    if cut is not None:
        toks = toks[:cut]
    if not toks:
        return None
    if len(toks) > max_words:
        toks = toks[:max_words]
    return " ".join(toks)


def distill_object_phrase(target_description: str, scene_type=None):
    prompt = prompts.prompt_distill_object_phrase(target_description, scene_type=scene_type)
    raw = vlm_generate(None, prompt, max_new_tokens=20)
    if raw and raw.strip().lower().startswith("none"):
        return ""
    phrase = strict_noun_phrase(raw, max_words=4)
    if not phrase:
        phrase = _salvage_noun_phrase(raw, max_words=4)
    if not phrase:
        phrase = clean_label(raw, max_words=4)
    return _filter_object_phrase(phrase)


def describe_masked_object(masked_crop_pil, masked_crop_dot_pil=None, overlay_crop_pil=None, scene_type=None):
    prompt = prompts.prompt_masked_object(scene_type=scene_type)
    # Use ONLY the cropped mask for primary labeling to avoid overlay color bias.
    imgs = [masked_crop_pil]
    raw = vlm_generate(imgs, prompt, max_new_tokens=24)
    phrase = strict_noun_phrase(raw, max_words=4)
    if not phrase:
        phrase = _salvage_noun_phrase(raw, max_words=4)
    if not phrase:
        phrase = clean_label(raw, max_words=4)
    return _filter_object_phrase(phrase)


def describe_masked_object_contextual(masked_crop_pil, context_dot_pil=None, full_scene_pil=None, scene_type=None):
    if masked_crop_pil is None:
        return None
    imgs = [masked_crop_pil]
    if context_dot_pil is not None:
        imgs.append(context_dot_pil)
    if full_scene_pil is not None:
        imgs.append(full_scene_pil)
    prompt = (
        f"Identify the object at the marked gaze target in a {scene_type} scene. "
        "Image 1 is the segmented crop, image 2 is a larger context with a target dot, "
        "and image 3 is the full scene. If image 1 shows only a small part, return the full object identity. "
        "Answer with one concise noun phrase (1-4 words), no explanation."
    )
    raw = vlm_generate(imgs, prompt, max_new_tokens=24)
    phrase = strict_noun_phrase(raw, max_words=4)
    if not phrase:
        phrase = _salvage_noun_phrase(raw, max_words=4)
    if not phrase:
        phrase = clean_label(raw, max_words=4)
    return _filter_object_phrase(phrase)


def describe_masked_object_detailed(masked_crop_pil, scene_type=None):
    if masked_crop_pil is None:
        return None
    prompt = prompts.prompt_masked_object_detailed(scene_type=scene_type)
    raw = vlm_generate([masked_crop_pil], prompt, max_new_tokens=28)
    phrase = strict_noun_phrase(raw, max_words=5)
    if not phrase:
        phrase = _salvage_noun_phrase(raw, max_words=5)
    if not phrase:
        phrase = clean_label(raw, max_words=5)
    return _filter_object_phrase(phrase)


def judge_same_object_phrase(label_a, label_b, scene_type=None):
    if not label_a or not label_b:
        return False, None, "missing"
    prompt = prompts.prompt_judge_same_object_phrase(label_a, label_b, scene_type=scene_type)
    raw = vlm_generate(None, prompt, max_new_tokens=24)
    t = raw.strip().lower()
    if t.startswith("yes"):
        m = re.search(r"canonical\s*:\s*(.+)", raw, flags=re.I)
        canon = m.group(1).strip() if m else ""
        canon = strict_noun_phrase(canon, max_words=4)
        canon = _filter_object_phrase(canon)
        return True, canon, raw
    return False, None, raw


def canonicalize_triple_cue(ray_label, mask_label, dot_label, ray_desc=None, dot_desc=None, scene_type=None):
    labels = {"ray": ray_label, "mask": mask_label, "dot": dot_label}
    avail = {k: v for k, v in labels.items() if v}
    if len(avail) < 2:
        fallback = avail.get("mask") or avail.get("ray") or avail.get("dot")
        return (fallback, "fallback_single") if fallback else (None, "missing_label")

    pairs = [
        ("mask", "ray"),
        ("mask", "dot"),
        ("ray", "dot"),
    ]
    for a_name, b_name in pairs:
        if a_name not in avail or b_name not in avail:
            continue
        ok, canon, _ = judge_same_object_phrase(avail[a_name], avail[b_name], scene_type=scene_type)
        canon = _filter_object_phrase(canon)
        if ok and canon:
            return canon, f"judge_{a_name}_{b_name}"

    if len(avail) == 3:
        prompt = prompts.prompt_reconcile_triple(
            ray_label, mask_label, dot_label, ray_desc=ray_desc, dot_desc=dot_desc, scene_type=scene_type
        )
        raw = vlm_generate(None, prompt, max_new_tokens=16)
        canon2 = strict_noun_phrase(raw, max_words=4)
        canon2 = _filter_object_phrase(canon2)
        if canon2:
            return canon2, "judge_reconcile_all"

    fallback = avail.get("mask") or avail.get("ray") or avail.get("dot")
    return (fallback, "fallback_no_consensus") if fallback else (None, "no_consensus")


def _dot_tiebreak_allowed(mask_label, dot_label, ray_fallback=None):
    dot = _sanitize_label(dot_label)
    if not dot or _is_generic_label(dot) or _is_bleeding_label(dot):
        return False
    mask = _sanitize_label(mask_label)
    mask_needs_help = (
        (not mask)
        or _is_generic_label(mask)
        or _is_ambiguous_label(mask)
        or _is_bleeding_label(mask)
    )
    if not mask_needs_help:
        return False
    ray = _sanitize_label(ray_fallback)
    if not ray:
        return False
    return _labels_relaxed_match(dot, ray) or _labels_token_subset_match(dot, ray)


def canonicalize_mask_overlay(mask_label, dot_label, mask_area_ratio=None, scene_type=None, ray_fallback=None):
    mask_label_for_canon = mask_label
    mask_small_skip = False
    if (
        mask_area_ratio is not None
        and mask_area_ratio <= st.TASK1_SMALL_OBJ_AREA_RATIO
        and mask_label
        and dot_label
        and mask_label != dot_label
    ):
        # For tiny masks, skip mask label only when mask cue itself is generic/ambiguous.
        # This avoids overriding clear large-object masks (e.g., sofa/couch) with dot-only text.
        dot_ray_agree = _labels_relaxed_match(dot_label, ray_fallback)
        if (
            _is_generic_label(mask_label)
            or _is_bleeding_label(mask_label)
            or (_has_on_phrase(dot_label) and dot_ray_agree)
            or (_has_on_phrase(dot_label) and _is_support_surface_label(mask_label))
        ):
            mask_label_for_canon = None
            mask_small_skip = True

    label = None
    mode = None

    # Mask-first policy: use mask cue when it is concrete enough.
    if mask_label_for_canon and not (
        _is_generic_label(mask_label_for_canon)
        or _is_ambiguous_label(mask_label_for_canon)
        or _is_bleeding_label(mask_label_for_canon)
    ):
        label = _sanitize_label(mask_label_for_canon)
        mode = "mask_primary"

    # Dot is tie-break only when mask is weak and dot agrees with ray context.
    if (not label) and _dot_tiebreak_allowed(mask_label_for_canon, dot_label, ray_fallback=ray_fallback):
        label = _sanitize_label(dot_label)
        mode = "dot_tiebreak_ray"

    if mask_small_skip:
        mode = f"{mode}+mask_small_skip" if mode else "mask_small_skip"

    if not label and ray_fallback:
        label = ray_fallback
        mode = "fallback_ray_only"

    return label, mode, mask_small_skip


def _camera_rank_weight(
    cam,
    anchor_cam,
    anchor_order,
    *,
    closest_cam=None,
    bbox_gaze_dist_norm=None,
):
    weight = 2.0 if cam == anchor_cam else 1.0
    if closest_cam is None or cam != closest_cam:
        return weight
    try:
        dist_norm = float(bbox_gaze_dist_norm)
    except Exception:
        return weight
    if not math.isfinite(dist_norm):
        return weight
    close_thresh = float(getattr(st, "TASK1_MV_CLOSEST_DIST_NORM_THRESH", 0.20))
    downscale = float(getattr(st, "TASK1_MV_CLOSEST_WEIGHT_DOWNSCALE", 0.75))
    downscale = max(0.5, min(1.0, downscale))
    if dist_norm <= close_thresh:
        weight *= downscale
    return weight


def _synthesize_multiview_labels(mv_labels, scene_type=None, anchor_cam=None, anchor_order=None):
    """
    Weighted multiview synthesis:
    - Anchor camera gets strongest prior.
    - Remaining cameras use equal weight.
    - Closest bbox<->gaze view gets a small downweight when very close.
    """
    items = []
    closest_cam = None
    closest_dist_norm = None
    if isinstance(mv_labels, dict):
        for cam, ent in mv_labels.items():
            if not isinstance(ent, dict):
                continue
            try:
                dist_norm = float(ent.get("bbox_gaze_dist_norm"))
            except Exception:
                continue
            if not math.isfinite(dist_norm):
                continue
            if closest_dist_norm is None or dist_norm < closest_dist_norm:
                closest_dist_norm = dist_norm
                closest_cam = cam
        for cam, ent in mv_labels.items():
            if not isinstance(ent, dict):
                continue
            lab = ent.get("label")
            if not lab:
                continue
            w = _camera_rank_weight(
                cam,
                anchor_cam,
                anchor_order,
                closest_cam=closest_cam,
                bbox_gaze_dist_norm=ent.get("bbox_gaze_dist_norm"),
            )
            items.append((cam, lab, w))
    else:
        for i, lab in enumerate(mv_labels or []):
            if lab:
                items.append((f"idx{i}", lab, 1.0))

    if not items:
        return None, "mv_missing", {}
    if len(items) == 1:
        return items[0][1], "mv_single", {items[0][0]: items[0][2]}

    clusters = []
    for cam, lab, w in items:
        placed = False
        for c in clusters:
            rep = c["rep"]
            if _labels_token_subset_match(lab, rep):
                c["labels"].append(lab)
                c["cams"].append(cam)
                c["weight"] += w
                placed = True
                break
            ok, canon, _ = judge_same_object_phrase(lab, rep, scene_type=scene_type)
            canon = _filter_object_phrase(canon)
            if ok:
                c["labels"].append(lab)
                c["cams"].append(cam)
                c["weight"] += w
                if canon:
                    c["canon"] = canon
                placed = True
                break
        if not placed:
            clusters.append({
                "rep": lab,
                "labels": [lab],
                "cams": [cam],
                "canon": lab,
                "weight": float(w),
            })

    best = None
    best_w = -1e9
    best_spec = -1e9
    for c in clusters:
        cand = _pick_most_specific_label(c["labels"]) or c.get("canon") or c.get("rep")
        spec = _label_specificity_score(cand)
        w = float(c.get("weight", 0.0))
        if (w > best_w) or (abs(w - best_w) <= 1e-6 and spec > best_spec):
            best_w = w
            best_spec = spec
            best = cand

    weight_meta = {cam: float(w) for cam, _, w in items}
    if best is None:
        best = _pick_most_specific_label([lab for _, lab, _ in items])
    return best, "mv_weighted", weight_meta


def _mask_quality_score(mask_area_ratio):
    if mask_area_ratio is None:
        return 0.5
    r = float(mask_area_ratio)
    lo = float(st.TASK1_MASK_MIN_AREA_RATIO)
    hi = float(st.TASK1_MASK_MAX_AREA_RATIO)
    if lo <= r <= hi:
        return 1.0
    if r < lo:
        return max(0.0, min(1.0, r / max(lo, 1e-6)))
    return max(0.0, min(1.0, hi / max(r, 1e-6)))


def _confidence_score_task1(mv_labels, mv_canon, anchor_cam=None, mask_area_ratio=None, pose_check=None):
    labels = [v.get("label") for v in (mv_labels or {}).values() if v.get("label")]
    if not labels or not mv_canon:
        return 0.0, {
            "consensus": 0.0,
            "specificity": 0.0,
            "mask_quality": _mask_quality_score(mask_area_ratio),
            "cue_agreement": 0.0,
            "view_factor": 0.0,
        }

    agree = 0
    for lab in labels:
        if lab == mv_canon or _labels_token_subset_match(lab, mv_canon):
            agree += 1
    consensus = agree / max(1, len(labels))

    spec_raw = _label_specificity_score(mv_canon)
    specificity = max(0.0, min(1.0, (spec_raw + 1.5) / 6.0))

    mask_quality = _mask_quality_score(mask_area_ratio)

    cue_agree = 0.0
    anchor = (mv_labels or {}).get(anchor_cam) if anchor_cam else None
    if anchor:
        cue_labels = [
            anchor.get("mask_label"),
            anchor.get("mask_overlay_label"),
            anchor.get("mask_refined_label"),
            anchor.get("dot_label_full"),
            anchor.get("dot_overlay_label"),
            anchor.get("ray_label"),
        ]
        cue_labels = [c for c in cue_labels if c]
        if cue_labels:
            cue_hits = sum(
                1 for c in cue_labels
                if c == mv_canon or _labels_token_subset_match(c, mv_canon)
            )
            cue_agree = cue_hits / max(1, len(cue_labels))

    view_factor = min(1.0, len(labels) / 2.0)

    conf = (
        0.45 * consensus
        + 0.25 * specificity
        + 0.20 * mask_quality
        + 0.10 * cue_agree
    )
    conf *= view_factor

    if pose_check == "NO":
        conf *= 0.6

    conf = max(0.0, min(1.0, conf))
    return conf, {
        "consensus": consensus,
        "specificity": specificity,
        "mask_quality": mask_quality,
        "cue_agreement": cue_agree,
        "view_factor": view_factor,
    }


# =============================================================================
# Person descriptor
# =============================================================================

def build_person_descriptor(anchor_raw_pil_resized, body_bbox_xywh_scaled=None, scene_type=None):
    if body_bbox_xywh_scaled is None:
        prompt = prompts.prompt_person_descriptor(scene_type=scene_type)
        raw = vlm_generate([anchor_raw_pil_resized], prompt, max_new_tokens=40)
        s = clean_label(raw, max_words=10)
        return _clean_person_desc(s)

    x, y, w, h = body_bbox_xywh_scaled
    W, H = anchor_raw_pil_resized.size
    x1 = int(max(0, x - 0.10 * w))
    y1 = int(max(0, y - 0.10 * h))
    x2 = int(min(W - 1, x + 1.10 * w))
    y2 = int(min(H - 1, y + 1.10 * h))
    crop = anchor_raw_pil_resized.crop((x1, y1, x2, y2))

    prompt = prompts.prompt_person_descriptor(scene_type=scene_type)
    raw = vlm_generate([crop], prompt, max_new_tokens=40)
    s = clean_label(raw, max_words=10)
    return _clean_person_desc(s)


def _task1_segmentation_attempts():
    base = {}
    loose_1 = {
        "use_tight_box": True,
        "point_box_size": max(220, int(round(st.TASK1_POINT_BOX_SIZE * 2.0))),
        "pad_around_mask": int(round(st.TASK1_PAD_AROUND_MASK * 1.7)),
        "pad_around_mask_ratio": max(st.TASK1_PAD_AROUND_MASK_RATIO, 0.14),
        "pad_around_mask_max": max(int(st.TASK1_PAD_AROUND_MASK_MAX), 90),
        "dilate_mask": True,
        "dilate_iter": max(st.TASK1_DILATE_ITER, 4),
        "mask_min_area_ratio": max(1e-5, st.TASK1_MASK_MIN_AREA_RATIO),
        "mask_max_area_ratio": 0.95,
        "min_soft_conf_around_gaze": 0.0,
        "soft_mask_threshold": 0.0,
        "reject_if_mask_overlaps_person": True,
        "allow_box_fallback": True,
    }
    loose_2 = {
        **loose_1,
        "point_box_size": max(320, int(round(st.TASK1_POINT_BOX_SIZE * 2.8))),
        "pad_around_mask": int(round(st.TASK1_PAD_AROUND_MASK * 2.4)),
        "pad_around_mask_ratio": max(st.TASK1_PAD_AROUND_MASK_RATIO, 0.18),
        "pad_around_mask_max": max(int(st.TASK1_PAD_AROUND_MASK_MAX), 120),
        "dilate_iter": max(st.TASK1_DILATE_ITER, 5),
    }
    loose_3 = {
        **loose_1,
        "point_box_size": max(400, int(round(st.TASK1_POINT_BOX_SIZE * 3.5))),
        "pad_around_mask": int(round(st.TASK1_PAD_AROUND_MASK * 3.0)),
        "pad_around_mask_ratio": max(st.TASK1_PAD_AROUND_MASK_RATIO, 0.22),
        "pad_around_mask_max": max(int(st.TASK1_PAD_AROUND_MASK_MAX), 170),
        "dilate_iter": max(st.TASK1_DILATE_ITER, 6),
    }
    # Sweep loose masks from broad -> broadest, then run strict refinement.
    return [("loose_1", loose_1), ("loose_2", loose_2), ("loose_3", loose_3), ("strict", base)]


# =============================================================================
# Task1 builder
# =============================================================================

def _point_to_bbox_distance(point_xy, bbox_xywh):
    if not point_xy or not bbox_xywh:
        return None
    px, py = float(point_xy[0]), float(point_xy[1])
    x, y, w, h = [float(v) for v in bbox_xywh]
    x1, y1 = x, y
    x2, y2 = x + max(0.0, w), y + max(0.0, h)
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


def _point_to_bbox_distance_norm(point_xy, bbox_xywh):
    dist_px = _point_to_bbox_distance(point_xy, bbox_xywh)
    if dist_px is None or not bbox_xywh:
        return None
    try:
        bw = float(bbox_xywh[2])
        bh = float(bbox_xywh[3])
    except Exception:
        return None
    diag = max(1.0, math.hypot(max(1.0, bw), max(1.0, bh)))
    return float(dist_px / diag)


def _anchor_cam_score(anno):
    if not isinstance(anno, dict):
        return -1.0, {
            "distance_px": None,
            "distance_norm": None,
            "body_bbox_present": False,
            "distance_bucket": "unknown",
            "bucket_priority": 9,
        }
    coord = anno.get("coordinate", None)
    body = get_body_bbox(anno)
    if not (isinstance(coord, (list, tuple)) and len(coord) == 2) or body is None:
        return -1.0, {
            "distance_px": None,
            "distance_norm": None,
            "body_bbox_present": bool(body is not None),
            "distance_bucket": "unknown",
            "bucket_priority": 9,
        }

    dist = _point_to_bbox_distance(coord, body)
    if dist is None:
        return -1.0, {
            "distance_px": None,
            "distance_norm": None,
            "body_bbox_present": True,
            "distance_bucket": "unknown",
            "bucket_priority": 9,
        }

    bw, bh = float(body[2]), float(body[3])
    diag = max(1.0, math.hypot(max(1.0, bw), max(1.0, bh)))
    dist_norm = float(dist / diag)
    target = float(getattr(st, "TASK1_ANCHOR_DIST_TARGET", 0.7))
    sigma = float(getattr(st, "TASK1_ANCHOR_DIST_SIGMA", 0.35))
    medium_band = float(getattr(st, "TASK1_ANCHOR_MEDIUM_BAND", 0.18))
    medium_band = max(1e-6, medium_band)
    sigma = max(1e-6, sigma)
    diff = float(dist_norm - target)
    score = 1.0 - min(1.0, abs(diff) / sigma)

    if abs(diff) <= medium_band:
        bucket = "medium"
        bucket_priority = 0
    elif diff > 0:
        bucket = "large"
        bucket_priority = 1
    else:
        bucket = "small"
        bucket_priority = 2

    return score, {
        "distance_px": float(dist),
        "distance_norm": float(dist_norm),
        "distance_diff": float(diff),
        "distance_target": float(target),
        "distance_sigma": float(sigma),
        "distance_bucket": bucket,
        "bucket_priority": int(bucket_priority),
        "medium_band": float(medium_band),
        "body_bbox_present": True,
    }


def _has_valid_body_bbox(bbox_xywh):
    if not (isinstance(bbox_xywh, (list, tuple)) and len(bbox_xywh) == 4):
        return False
    try:
        _, _, w, h = [float(v) for v in bbox_xywh]
    except Exception:
        return False
    return (w > 1.0) and (h > 1.0)


def _bbox_area_xywh(bbox_xywh):
    if not (isinstance(bbox_xywh, (list, tuple)) and len(bbox_xywh) == 4):
        return 0.0
    try:
        _, _, w, h = [float(v) for v in bbox_xywh]
    except Exception:
        return 0.0
    return max(0.0, w) * max(0.0, h)


def _should_trigger_large_mask_refine(mask_u8, body_bbox_xywh):
    if not bool(getattr(st, "TASK1_LARGE_MASK_REFINE", False)):
        return False, 0.0, 0.0
    if mask_u8 is None or body_bbox_xywh is None:
        return False, 0.0, 0.0
    person_area_px = _bbox_area_xywh(body_bbox_xywh)
    if person_area_px <= 1.0:
        return False, 0.0, person_area_px
    try:
        mask_area_px = float(np.asarray(mask_u8, dtype=np.uint8).sum())
    except Exception:
        return False, 0.0, person_area_px
    trigger_ratio = max(0.0, float(getattr(st, "TASK1_LARGE_MASK_REFINE_TRIGGER_RATIO", 1.0)))
    trigger = mask_area_px >= (trigger_ratio * person_area_px)
    return bool(trigger), float(mask_area_px), float(person_area_px)


def _task1_large_mask_refine_cfg():
    point_scale = float(getattr(st, "TASK1_LARGE_MASK_REFINE_POINT_SCALE", 0.45))
    pad_scale = float(getattr(st, "TASK1_LARGE_MASK_REFINE_PAD_SCALE", 0.2))
    point_scale = max(0.05, min(1.0, point_scale))
    pad_scale = max(0.05, min(1.0, pad_scale))
    return {
        "use_tight_box": True,
        "point_box_size": max(8, int(st.TASK1_POINT_BOX_SIZE * point_scale)),
        "pad_around_mask": max(2, int(st.TASK1_PAD_AROUND_MASK * pad_scale)),
        "pad_around_mask_ratio": max(0.01, st.TASK1_PAD_AROUND_MASK_RATIO * pad_scale),
        "pad_around_mask_max": max(8, int(st.TASK1_PAD_AROUND_MASK_MAX * pad_scale)),
        # Extra-tight variant should not inflate mask back outward.
        "dilate_mask": False,
        "allow_box_fallback": True,
    }


def list_anchor_cam_candidates(cams, per_cam, zf, split, seq, frame_id):
    """
    All cams that have coordinate and visibility is not explicitly False AND image exists.
    Ranked by distance class priority:
    1) medium distance
    2) large distance
    3) small distance
    with score tie-break inside each class.
    """
    scored = []
    score_meta = {}
    for idx, c in enumerate(cams):
        a = per_cam.get(c, {})
        vis = parse_visibility(a)
        if has_coord(a) and (vis is None or vis is True):
            if zip_try_image_path(zf, split, seq, c, frame_id) is not None:
                score, details = _anchor_cam_score(a)
                pri = int(details.get("bucket_priority", 9))
                scored.append((c, pri, score, idx))
                score_meta[c] = {"score": float(score), **details}

    scored.sort(key=lambda t: (t[1], -t[2], t[3]))
    out = [c for (c, _, _, _) in scored]
    return out, score_meta


def build_task1(zf, split, seq, frame_id, cams, per_cam, task1_index):
    anchor_cands, anchor_score_meta = list_anchor_cam_candidates(cams, per_cam, zf, split, seq, frame_id)
    if not anchor_cands:
        st.REJECT_STATS["t1_no_anchor"] += 1
        return None
    if len(anchor_cands) > 1:
        rank_txt = ", ".join(
            f"{cam}:{anchor_score_meta.get(cam, {}).get('distance_bucket', 'unknown')}:{anchor_score_meta.get(cam, {}).get('score', -1.0):.3f}"
            for cam in anchor_cands
        )
        st.logger.info(f"[Task1] anchor_order {split}/{seq} frame={frame_id}: {rank_txt}")

    input_images = save_raw_cam_images_parallel(zf, split, seq, cams, frame_id)
    if len(input_images) < 2:
        st.REJECT_STATS["t1_no_images"] += 1
        return None
    save_debug_this = st.SAVE_DEBUG and (task1_index % max(1, st.DEBUG_EVERY_N_TASK1) == 0)

    last_fail_reason = None
    last_fail_code = None
    for anchor_idx, anchor_cam in enumerate(anchor_cands):
        try:
            last_fail_reason = None
            last_fail_code = None
            force_next_anchor = False
            anno_orig = per_cam[anchor_cam]
            zp = zip_try_image_path(zf, split, seq, anchor_cam, frame_id)
            if zp is None:
                continue

            anchor_orig = zip_read_image(zf, zp)
            orig_wh = anchor_orig.size
            anchor_resized = _resize(anchor_orig)
            new_wh = anchor_resized.size

            anno_scaled, audit_geom = scale_annotations_for_resized_image(anno_orig, orig_wh, new_wh)
            if audit_geom is None:
                st.REJECT_STATS["t1_no_coord"] += 1
                last_fail_code = "no_coord"
                last_fail_reason = "geom_audit_none"
                continue

            coord_scaled = anno_scaled.get("coordinate", None)
            if coord_scaled is None:
                st.REJECT_STATS["t1_no_coord"] += 1
                last_fail_code = "no_coord"
                last_fail_reason = "no_coord_scaled"
                continue
            body_bbox_scaled = get_body_bbox(anno_scaled)
            ray_available = _has_valid_body_bbox(body_bbox_scaled)

            person_desc = build_person_descriptor(anchor_resized, body_bbox_scaled, scene_type=split)

            ray_pil = draw_gaze_ray_overlay(anchor_resized, anno_scaled) if ray_available else anchor_resized.copy()
            ray_label_prompt_pil = ray_pil

            target_description = None
            phrase_from_ray = None
            phrase_from_dot = None  # dot-on-mask label (no full-image dot prompt)

            seg_used = None
            masked_crop = full_mask = bb = soft_mask = None
            phrase_from_mask = None
            phrase_from_mask_overlay = None
            phrase_from_mask_context = None
            overlay_crop = None
            dot_overlay_crop = None
            dot_mask_crop = None
            phrase_from_dot_overlay = None
            refined_masked_crop = None
            phrase_from_mask_refined = None
            phrase_from_mask_refined_large = None
            masked_crop_soft = None
            masked_crop_for_vlm = None
            canonical_object = None
            canon_mode = None
            mask_area_ratio = None
            positional_label = None
            mask_small_skip = False
            label_flow = []
            dot_label_for_canon = None
            large_mask_refine_triggered = False
            large_mask_refine_used = False
            large_mask_refine_frac = None
            last_seg_candidate = None
            loose_baseline_candidate = None
            semantic_arbiter = {
                "enabled": bool(st.TASK1_SEMANTIC_ARBITER),
                "triggered": False,
                "applied": False,
            }
            hybrid_guardrail = {
                "enabled": bool(getattr(st, "TASK1_HYBRID_GUARDRAIL", False)),
                "mode": "guide_only",
                "triggered": False,
                "applied": False,
                "applied_by": "none",
                "proposed_label": None,
                "proposed_passed_gate": False,
                "blocked_reason": None,
            }
            teacher_mode = bool(getattr(st, "TASK1_TEACHER_FINAL", False)) and not bool(
                getattr(st, "TASK1_LEGACY_PIPELINE", False)
            )
            teacher_final = {
                "enabled": teacher_mode,
                "triggered": False,
                "applied": False,
                "call_count": 0,
                "student_label": None,
                "teacher_label_pass1": None,
                "teacher_label_pass2": None,
                "semantic_match_qwen_teacher": None,
                "qwen_verdict": "UNCLEAR",
                "teacher_confidence": "LOW",
                "final_source": "student_fallback",
                "final_label": None,
                "reason": None,
                "rationale": None,
            }

            for seg_name, seg_cfg in _task1_segmentation_attempts():
                masked_crop, full_mask, bb, _, soft_mask, seg_debug = segment_object_at_gaze(
                    zf, split, seq, anchor_cam, frame_id, coord_scaled,
                    body_bbox_xywh_scaled=body_bbox_scaled,
                    cfg=seg_cfg
                )
                if masked_crop is None:
                    if seg_debug and seg_debug.get("last_reject_reason") == "overlap_reject":
                        last_fail_code = "sam2_overlap_reject"
                        last_fail_reason = f"sam2_overlap_reject:{seg_name}"
                        st.logger.info(
                            f"[Task1] {split}/{seq} frame={frame_id} cam={anchor_cam} fail={last_fail_reason}"
                        )
                        # If this view overlaps the person mask, skip this anchor and try other cameras.
                        force_next_anchor = True
                        if st.SAVE_DEBUG:
                            if st.DUMP_OVERLAP_DEBUG:
                                stem2 = f"t1_overlap_{split}_{seq}_{frame_id}_{anchor_cam}_{seg_name}"
                                (st.DEBUG_DIR / stem2).mkdir(exist_ok=True)
                                if seg_debug.get("overlap_reject_mask") is not None:
                                    ov_overlay = overlay_mask_on_image(
                                        anchor_resized, seg_debug["overlap_reject_mask"]
                                    )
                                    ov_overlay.save(st.DEBUG_DIR / stem2 / "overlap_mask_overlay.jpg", quality=95)
                                anchor_resized.save(st.DEBUG_DIR / stem2 / "anchor_raw_resized.jpg", quality=95)
                                ray_pil.save(st.DEBUG_DIR / stem2 / "anchor_ray.jpg", quality=95)
                                # Draw body bbox + gaze point for scale sanity checks
                                dbg = anchor_resized.copy()
                                try:
                                    draw = ImageDraw.Draw(dbg)
                                    if body_bbox_scaled is not None:
                                        x, y, w, h = body_bbox_scaled
                                        draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 255), width=3)
                                    if isinstance(coord_scaled, (list, tuple)) and len(coord_scaled) == 2:
                                        gx, gy = float(coord_scaled[0]), float(coord_scaled[1])
                                        r = st.GAZE_DOT_R + 2
                                        draw.ellipse([gx - r, gy - r, gx + r, gy + r], outline=(255, 255, 0), width=3)
                                except Exception:
                                    pass
                                dbg.save(st.DEBUG_DIR / stem2 / "anchor_body_gaze.jpg", quality=95)
                                log_debug({
                                    "task": 1,
                                    "split": split,
                                    "seq": seq,
                                    "frame_id": frame_id,
                                    "anchor_cam": anchor_cam,
                                    "segmentation_try": seg_name,
                                    "fail_reason": last_fail_reason,
                                    "fail_code": last_fail_code,
                                    "dump": "overlap_debug",
                                })
                            stem = f"t1_fail_{split}_{seq}_{frame_id}_{anchor_cam}_{seg_name}"
                            (st.DEBUG_DIR / stem).mkdir(exist_ok=True)
                            if seg_debug.get("overlap_reject_mask") is not None:
                                rej_overlay = overlay_mask_on_image(
                                    anchor_resized, seg_debug["overlap_reject_mask"]
                                )
                            else:
                                rej_overlay = None
                            collage = _make_collage([anchor_resized, ray_pil, rej_overlay])
                            if collage is not None:
                                collage.save(st.DEBUG_DIR / stem / "collage.jpg", quality=95)
                            log_debug({
                                "task": 1,
                                "split": split,
                                "seq": seq,
                                "frame_id": frame_id,
                                "anchor_cam": anchor_cam,
                                "segmentation_try": seg_name,
                                "fail_reason": last_fail_reason,
                                "fail_code": last_fail_code,
                            })
                    else:
                        last_fail_code = "sam2_mask_fail"
                        last_fail_reason = f"sam2_mask_fail:{seg_name}"
                        st.logger.info(
                            f"[Task1] {split}/{seq} frame={frame_id} cam={anchor_cam} fail={last_fail_reason}"
                        )
                        if st.SAVE_DEBUG:
                            log_debug({
                                "task": 1,
                                "split": split,
                                "seq": seq,
                                "frame_id": frame_id,
                                "anchor_cam": anchor_cam,
                                "segmentation_try": seg_name,
                                "fail_reason": last_fail_reason,
                                "fail_code": last_fail_code,
                            })
                    if force_next_anchor:
                        break
                    continue

                if full_mask is not None:
                    mask_area_ratio = float(full_mask.sum()) / float(max(1, full_mask.size))

                # Reset per-attempt fields so retries do not leak stale values.
                phrase_from_mask = None
                phrase_from_mask_overlay = None
                phrase_from_mask_context = None
                phrase_from_dot = None
                overlay_crop = None
                dot_overlay_crop = None
                dot_mask_crop = None
                phrase_from_dot_overlay = None
                refined_masked_crop = None
                phrase_from_mask_refined = None
                phrase_from_mask_refined_large = None
                positional_label = None
                large_mask_refine_triggered = False
                large_mask_refine_used = False
                large_mask_refine_frac = None

                if ray_available and target_description is None:
                    ray_label_prompt_pil = build_ray_label_prompt_image(
                        anchor_resized, anno_scaled, body_bbox_xywh=body_bbox_scaled, target_mask_u8=full_mask
                    )
                    target_description = generate_target_description(
                        ray_label_prompt_pil, person_desc, anchor_cam, scene_type=split
                    )
                    phrase_from_ray = distill_object_phrase(target_description, scene_type=split)
                    st.logger.info(
                        f"[Task1] ray_desc={target_description} | ray_phrase={phrase_from_ray}"
                    )
                # Keep label cues mask-only (no alpha overlay tinting).
                overlay_crop = masked_crop
                if bb is not None:
                    dot_mask_crop = draw_dot_on_crop(
                        masked_crop, coord_scaled, bb,
                        alpha=0.6, full_wh=anchor_resized.size, color=st.GAZE_COLOR
                    )
                dot_overlay_crop = dot_mask_crop
                masked_crop_for_vlm = masked_crop
                phrase_from_mask = describe_masked_object(
                    masked_crop_for_vlm, None, None, scene_type=split
                )
                st.logger.info(
                    f"[Task1] mask_phrase={phrase_from_mask}"
                )
                # Backward-compatible metadata field; now mask-only cue.
                phrase_from_mask_overlay = phrase_from_mask
                if dot_overlay_crop is not None:
                    phrase_from_dot_overlay = describe_masked_object(dot_overlay_crop, None, None, scene_type=split)
                    phrase_from_dot = phrase_from_dot_overlay

                # Context rescue for tiny/ambiguous masks: use enlarged local context + full scene.
                if _should_use_small_mask_context(mask_area_ratio, phrase_from_mask) and bb is not None:
                    context_dot_crop = _build_context_dot_crop(
                        anchor_resized,
                        bb,
                        coord_scaled,
                        expand_ratio=float(getattr(st, "TASK1_SMALL_MASK_CONTEXT_EXPAND_RATIO", 2.3)),
                    )
                    phrase_from_mask_context = describe_masked_object_contextual(
                        masked_crop,
                        context_dot_pil=context_dot_crop,
                        full_scene_pil=anchor_resized,
                        scene_type=split,
                    )
                    if phrase_from_mask_context and not _is_generic_label(phrase_from_mask_context):
                        phrase_from_mask = phrase_from_mask_context
                        phrase_from_mask_overlay = phrase_from_mask_context

                # Refine small-object mask within the bbox crop
                if bb is not None and coord_scaled is not None:
                    x1, y1, x2, y2 = bb
                    raw_crop = anchor_resized.crop((x1, y1, x2 + 1, y2 + 1))
                    rx, ry = float(coord_scaled[0]) - x1, float(coord_scaled[1]) - y1
                    refine_cfg = {
                        "point_box_size": max(10, int(st.TASK1_POINT_BOX_SIZE * 0.6)),
                        "pad_around_mask": max(5, int(st.TASK1_PAD_AROUND_MASK * 0.3)),
                        "pad_around_mask_ratio": max(0.02, st.TASK1_PAD_AROUND_MASK_RATIO * 0.3),
                        "pad_around_mask_max": max(20, int(st.TASK1_PAD_AROUND_MASK_MAX * 0.3)),
                    }
                    ref_crop, _, _, _ = segment_object_on_crop(
                        raw_crop, (rx, ry), cfg=refine_cfg
                    )
                    if ref_crop is not None:
                        refined_masked_crop = ref_crop
                        phrase_from_mask_refined = describe_masked_object(
                            refined_masked_crop, None, None, scene_type=split
                        )

                    # Oversized-mask rescue: if mask is at least person-sized, run extra tighter refine.
                    do_large_refine, mask_area_px, person_area_px = _should_trigger_large_mask_refine(
                        full_mask, body_bbox_scaled
                    )
                    if do_large_refine:
                        large_mask_refine_triggered = True
                        tight_cfg = _task1_large_mask_refine_cfg()
                        tight_crop, tight_mask, _, _ = segment_object_on_crop(
                            raw_crop, (rx, ry), cfg=tight_cfg
                        )
                        if tight_crop is not None and tight_mask is not None:
                            tight_area_px = float(np.asarray(tight_mask, dtype=np.uint8).sum())
                            if mask_area_px > 0.0:
                                large_mask_refine_frac = float(tight_area_px / mask_area_px)
                            else:
                                large_mask_refine_frac = None
                            max_frac = float(getattr(st, "TASK1_LARGE_MASK_REFINE_MAX_FRAC", 0.85))
                            if (
                                large_mask_refine_frac is not None
                                and large_mask_refine_frac > 0.0
                                and large_mask_refine_frac <= max_frac
                            ):
                                phrase_from_mask_refined_large = describe_masked_object(
                                    tight_crop, None, None, scene_type=split
                                )
                                if phrase_from_mask_refined_large and not _is_generic_label(phrase_from_mask_refined_large):
                                    large_mask_refine_used = True
                                    if _is_ambiguous_label(phrase_from_mask):
                                        phrase_from_mask = phrase_from_mask_refined_large
                                        phrase_from_mask_overlay = phrase_from_mask_refined_large
                if not phrase_from_mask and not phrase_from_dot_overlay and not phrase_from_ray:
                    last_fail_code = "phrase_missing"
                    last_fail_reason = f"phrase_missing:{seg_name}"
                    st.logger.info(
                        f"[Task1] {split}/{seq} frame={frame_id} cam={anchor_cam} fail={last_fail_reason}"
                    )
                    if st.SAVE_DEBUG:
                        log_debug({
                            "task": 1,
                            "split": split,
                            "seq": seq,
                            "frame_id": frame_id,
                            "anchor_cam": anchor_cam,
                            "segmentation_try": seg_name,
                            "fail_reason": last_fail_reason,
                            "fail_code": last_fail_code,
                            "phrase_from_ray": phrase_from_ray,
                            "phrase_from_mask": phrase_from_mask,
                            "target_description": target_description,
                        })
                    continue

                # Additional guard: if mask label itself is person-like and overlaps body,
                # skip this anchor and move to another camera.
                if (
                    full_mask is not None
                    and body_bbox_scaled is not None
                    and any(
                        _is_person_like_label(x)
                        for x in (phrase_from_mask, phrase_from_mask_overlay, phrase_from_mask_refined)
                        if x
                    )
                ):
                    overlap = mask_person_overlap_ratio((full_mask > 0).astype(np.uint8), body_bbox_scaled)
                    if overlap >= max(0.2, st.TASK1_PERSON_OVERLAP_THRESHOLD * 0.5):
                        last_fail_code = "sam2_overlap_reject"
                        last_fail_reason = f"mask_person_overlap:{seg_name}:{overlap:.3f}"
                        force_next_anchor = True
                        st.logger.info(
                            f"[Task1] {split}/{seq} frame={frame_id} cam={anchor_cam} fail={last_fail_reason}"
                        )
                        break

                mask_label_for_canon = _choose_mask_label(
                    phrase_from_mask, phrase_from_mask_refined, mask_area_ratio
                )
                dot_label_for_canon = _pick_most_specific_label(
                    [phrase_from_dot, phrase_from_dot_overlay]
                )
                if phrase_from_mask_refined_large:
                    if (not mask_label_for_canon) or _is_ambiguous_label(mask_label_for_canon):
                        mask_label_for_canon = _sanitize_label(phrase_from_mask_refined_large)
                    elif dot_label_for_canon and _labels_relaxed_match(phrase_from_mask_refined_large, dot_label_for_canon):
                        if not _labels_relaxed_match(mask_label_for_canon, dot_label_for_canon):
                            mask_label_for_canon = _sanitize_label(phrase_from_mask_refined_large)
                attempt_canonical_object, attempt_canon_mode, mask_small_skip = canonicalize_mask_overlay(
                    mask_label_for_canon,
                    dot_label_for_canon,
                    mask_area_ratio=mask_area_ratio,
                    scene_type=split,
                    ray_fallback=phrase_from_ray
                )

                if (
                    _is_loose_stage(seg_name)
                    and (not attempt_canonical_object or _is_ambiguous_label(attempt_canonical_object))
                    and _dot_tiebreak_allowed(
                        attempt_canonical_object,
                        dot_label_for_canon,
                        ray_fallback=phrase_from_ray,
                    )
                ):
                    attempt_canonical_object = _sanitize_label(dot_label_for_canon)
                    attempt_canon_mode = (
                        f"{attempt_canon_mode}+dot_tiebreak_ray"
                        if attempt_canon_mode else "dot_tiebreak_ray"
                    )

                if (
                    mask_area_ratio is not None
                    and mask_area_ratio <= st.TASK1_SMALL_OBJ_AREA_RATIO
                    and mask_label_for_canon
                    and dot_label_for_canon
                    and not mask_small_skip
                ):
                    small_label = mask_label_for_canon
                    if not _is_generic_label(small_label) and not _is_generic_label(dot_label_for_canon):
                        positional_label = _compose_positional_label(small_label, dot_label_for_canon)
                        if positional_label:
                            attempt_canonical_object = positional_label
                            attempt_canon_mode = (
                                f"{attempt_canon_mode}+positional" if attempt_canon_mode else "positional"
                            )

                if not attempt_canonical_object:
                    continue

                attempt_canonical_object = _squash_on_phrase(attempt_canonical_object)
                cue_hits = _task1_label_cue_agreement(
                    attempt_canonical_object,
                    dot_label=dot_label_for_canon or phrase_from_dot,
                    ray_label=phrase_from_ray,
                )
                candidate = {
                    "seg_used": seg_name,
                    "masked_crop": masked_crop,
                    "full_mask": full_mask,
                    "bb": bb,
                    "soft_mask": soft_mask,
                    "phrase_from_mask": phrase_from_mask,
                    "phrase_from_mask_overlay": phrase_from_mask_overlay,
                    "phrase_from_mask_context": phrase_from_mask_context,
                    "overlay_crop": overlay_crop,
                    "dot_overlay_crop": dot_overlay_crop,
                    "dot_mask_crop": dot_mask_crop,
                    "phrase_from_dot": phrase_from_dot,
                    "phrase_from_dot_overlay": phrase_from_dot_overlay,
                    "refined_masked_crop": refined_masked_crop,
                    "phrase_from_mask_refined": phrase_from_mask_refined,
                    "phrase_from_mask_refined_large": phrase_from_mask_refined_large,
                    "large_mask_refine_triggered": large_mask_refine_triggered,
                    "large_mask_refine_used": large_mask_refine_used,
                    "large_mask_refine_frac": large_mask_refine_frac,
                    "canonical_object": attempt_canonical_object,
                    "canon_mode": attempt_canon_mode,
                    "mask_area_ratio": mask_area_ratio,
                    "positional_label": positional_label,
                    "dot_label_for_canon": dot_label_for_canon,
                }
                last_seg_candidate = candidate
                if _is_loose_stage(seg_name):
                    loose_baseline_candidate = _pick_loose_baseline(loose_baseline_candidate, candidate)

                if _task1_should_accept_attempt(
                    seg_name, mask_area_ratio, cue_hits, label=attempt_canonical_object
                ):
                    chosen = candidate
                    if seg_name == "strict" and loose_baseline_candidate is not None:
                        if not _strict_refines_loose(
                            candidate.get("canonical_object"),
                            loose_baseline_candidate.get("canonical_object"),
                        ):
                            chosen = dict(loose_baseline_candidate)
                            prev_mode = str(chosen.get("canon_mode") or "").strip()
                            chosen["canon_mode"] = (
                                f"{prev_mode}+prefer_loose_baseline"
                                if prev_mode else "prefer_loose_baseline"
                            )
                    seg_used = chosen["seg_used"]
                    masked_crop = chosen["masked_crop"]
                    full_mask = chosen["full_mask"]
                    bb = chosen["bb"]
                    soft_mask = chosen["soft_mask"]
                    phrase_from_mask = chosen["phrase_from_mask"]
                    phrase_from_mask_overlay = chosen["phrase_from_mask_overlay"]
                    phrase_from_mask_context = chosen["phrase_from_mask_context"]
                    overlay_crop = chosen["overlay_crop"]
                    dot_overlay_crop = chosen["dot_overlay_crop"]
                    dot_mask_crop = chosen["dot_mask_crop"]
                    phrase_from_dot = chosen["phrase_from_dot"]
                    phrase_from_dot_overlay = chosen["phrase_from_dot_overlay"]
                    refined_masked_crop = chosen["refined_masked_crop"]
                    phrase_from_mask_refined = chosen["phrase_from_mask_refined"]
                    phrase_from_mask_refined_large = chosen["phrase_from_mask_refined_large"]
                    large_mask_refine_triggered = chosen["large_mask_refine_triggered"]
                    large_mask_refine_used = chosen["large_mask_refine_used"]
                    large_mask_refine_frac = chosen["large_mask_refine_frac"]
                    canonical_object = chosen["canonical_object"]
                    canon_mode = chosen["canon_mode"]
                    mask_area_ratio = chosen["mask_area_ratio"]
                    positional_label = chosen["positional_label"]
                    dot_label_for_canon = chosen["dot_label_for_canon"]
                    break

            if canonical_object is None and loose_baseline_candidate is not None:
                last_seg_candidate = loose_baseline_candidate
            if canonical_object is None and last_seg_candidate is not None:
                seg_used = last_seg_candidate["seg_used"]
                masked_crop = last_seg_candidate["masked_crop"]
                full_mask = last_seg_candidate["full_mask"]
                bb = last_seg_candidate["bb"]
                soft_mask = last_seg_candidate["soft_mask"]
                phrase_from_mask = last_seg_candidate["phrase_from_mask"]
                phrase_from_mask_overlay = last_seg_candidate["phrase_from_mask_overlay"]
                phrase_from_mask_context = last_seg_candidate["phrase_from_mask_context"]
                overlay_crop = last_seg_candidate["overlay_crop"]
                dot_overlay_crop = last_seg_candidate["dot_overlay_crop"]
                dot_mask_crop = last_seg_candidate["dot_mask_crop"]
                phrase_from_dot = last_seg_candidate["phrase_from_dot"]
                phrase_from_dot_overlay = last_seg_candidate["phrase_from_dot_overlay"]
                refined_masked_crop = last_seg_candidate["refined_masked_crop"]
                phrase_from_mask_refined = last_seg_candidate["phrase_from_mask_refined"]
                phrase_from_mask_refined_large = last_seg_candidate["phrase_from_mask_refined_large"]
                large_mask_refine_triggered = last_seg_candidate["large_mask_refine_triggered"]
                large_mask_refine_used = last_seg_candidate["large_mask_refine_used"]
                large_mask_refine_frac = last_seg_candidate["large_mask_refine_frac"]
                canonical_object = last_seg_candidate["canonical_object"]
                canon_mode = last_seg_candidate["canon_mode"]
                mask_area_ratio = last_seg_candidate["mask_area_ratio"]
                positional_label = last_seg_candidate["positional_label"]
                dot_label_for_canon = last_seg_candidate["dot_label_for_canon"]

            if force_next_anchor:
                continue

            if not canonical_object:
                last_fail_code = "canonical_fail"
                last_fail_reason = "no_semantic_consensus"
                st.logger.info(
                    f"[Task1] {split}/{seq} frame={frame_id} cam={anchor_cam} fail={last_fail_reason}"
                )
                if st.SAVE_DEBUG:
                    log_debug({
                        "task": 1,
                        "split": split,
                        "seq": seq,
                        "frame_id": frame_id,
                        "anchor_cam": anchor_cam,
                        "segmentation_try": seg_name,
                        "fail_reason": last_fail_reason,
                        "fail_code": last_fail_code,
                            "phrase_from_ray": phrase_from_ray,
                            "phrase_from_mask": phrase_from_mask,
                            "phrase_from_dot": phrase_from_dot,
                            "target_description": target_description,
                        })
                continue

            preferred, prefer_mode = _prefer_specific_label(
                canonical_object, phrase_from_mask, None, dot_label_for_canon or phrase_from_dot
            )
            if preferred and preferred != canonical_object:
                _append_label_flow(label_flow, "prefer_specific", canonical_object, preferred, prefer_mode)
                canonical_object = preferred
                if prefer_mode:
                    canon_mode = f"{canon_mode}+{prefer_mode}"
            canon_before_sanitize = canonical_object
            canonical_object = _sanitize_label(canonical_object)
            _append_label_flow(label_flow, "sanitize", canon_before_sanitize, canonical_object, "sanitize_label")
            if canonical_object and " on " in canonical_object:
                if not _on_relation_plausible(canonical_object, scene_type=split):
                    left = canonical_object.split(" on ", 1)[0].strip()
                    if left and not _is_generic_label(left):
                        _append_label_flow(
                            label_flow, "on_plausibility", canonical_object, left, "on_plausibility_fallback"
                        )
                        canonical_object = left
                        canon_mode = f"{canon_mode}+on_plausibility_fallback" if canon_mode else "on_plausibility_fallback"
            if _is_bleeding_label(canonical_object):
                fallback = None
                for cand in (phrase_from_mask, phrase_from_dot):
                    cand = _sanitize_label(cand)
                    if cand and not _is_generic_label(cand) and not _is_bleeding_label(cand):
                        fallback = cand
                        break
                if fallback:
                    _append_label_flow(label_flow, "bleed_fallback", canonical_object, fallback, "bleed_fallback")
                    canonical_object = fallback
                    canon_mode = f"{canon_mode}+bleed_fallback" if canon_mode else "bleed_fallback"

            # If label is ambiguous, try another anchor camera if available
            if (
                _is_ambiguous_label(canonical_object)
                and anchor_idx < (len(anchor_cands) - 1)
                and (not st.TASK1_SEMANTIC_ARBITER)
            ):
                st.logger.info(
                    f"[Task1] ambiguous label '{canonical_object}' in {anchor_cam}; trying another anchor."
                )
                continue

            raw_multi_images = [x["image"] for x in input_images[:st.MAX_VIEWS]]

            reasoning = None

            # Multi-view synthesis (separate from anchor labeling)
            mv_labels = {}
            mv_visible_cams = []
            mv_nonvisible_cams = []
            mv_unknown_cams = []
            mv_canon = None
            mv_canon_mode = None
            mv_match_anchor = None
            mv_weight_map = {}
            mv_debug_payload = {}
            mv_coords_scaled = {}
            anchor_label = canonical_object
            anchor_canon_mode = canon_mode
            anchor_bbox_gaze_dist_px = _point_to_bbox_distance(coord_scaled, body_bbox_scaled)
            anchor_bbox_gaze_dist_norm = _point_to_bbox_distance_norm(coord_scaled, body_bbox_scaled)

            if anchor_label:
                mv_labels[anchor_cam] = {
                    "label": anchor_label,
                    "mode": anchor_canon_mode,
                    "mask_area_ratio": mask_area_ratio,
                    "ray_label": phrase_from_ray,
                    "dot_label_full": phrase_from_dot,
                    "mask_label": phrase_from_mask,
                    "mask_context_label": phrase_from_mask_context,
                    "mask_only_label": phrase_from_mask,
                    "dot_mask_label": phrase_from_dot_overlay,
                    "mask_overlay_label": phrase_from_mask_overlay,
                    "dot_overlay_label": phrase_from_dot_overlay,
                    "coord_scaled": [float(coord_scaled[0]), float(coord_scaled[1])],
                    "mask_refined_label": phrase_from_mask_refined,
                    "mask_refined_large_label": phrase_from_mask_refined_large,
                    "large_mask_refine_triggered": large_mask_refine_triggered,
                    "large_mask_refine_used": large_mask_refine_used,
                    "large_mask_refine_frac": large_mask_refine_frac,
                    "positional_label": positional_label,
                    "ray_available": ray_available,
                    "bbox_gaze_dist_px": anchor_bbox_gaze_dist_px,
                    "bbox_gaze_dist_norm": anchor_bbox_gaze_dist_norm,
                }
                mv_coords_scaled[anchor_cam] = [float(coord_scaled[0]), float(coord_scaled[1])]

            for cam in cams:
                if cam == anchor_cam:
                    continue
                anno_orig = per_cam.get(cam, {})
                vis = parse_visibility(anno_orig)
                if vis is True:
                    mv_visible_cams.append(cam)
                elif vis is False:
                    mv_nonvisible_cams.append(cam)
                else:
                    mv_unknown_cams.append(cam)

                zp_mv = zip_try_image_path(zf, split, seq, cam, frame_id)
                if zp_mv is None:
                    continue

                mv_orig = zip_read_image(zf, zp_mv)
                mv_resized = _resize(mv_orig)
                anno_scaled, _ = scale_annotations_for_resized_image(anno_orig, mv_orig.size, mv_resized.size)
                if not isinstance(anno_scaled, dict) or not has_coord(anno_scaled):
                    continue
                coord_mv = anno_scaled.get("coordinate", None)
                if not (isinstance(coord_mv, (list, tuple)) and len(coord_mv) == 2):
                    continue
                mv_coords_scaled[cam] = [float(coord_mv[0]), float(coord_mv[1])]

                mv_body = get_body_bbox(anno_scaled)
                mv_ray_available = _has_valid_body_bbox(mv_body)
                mv_bbox_gaze_dist_px = _point_to_bbox_distance(coord_mv, mv_body)
                mv_bbox_gaze_dist_norm = _point_to_bbox_distance_norm(coord_mv, mv_body)
                mv_ray_label = None
                mv_dot_label_full = None
                mv_ray_desc = None

                mv_label = None
                mv_mode = None
                mv_mask_area = None
                mv_mask_label = None
                mv_mask_overlay_label = None
                mv_mask_context_label = None
                mv_dot_overlay_label = None
                mv_dot_mask_crop = None
                mv_refined_label = None
                mv_refined_large_label = None
                mv_bb = None
                mv_soft = None
                mv_mask_small_skip = False
                mv_large_refine_triggered = False
                mv_large_refine_used = False
                mv_large_refine_frac = None
                mv_coord_scaled = [float(coord_mv[0]), float(coord_mv[1])]

                if mv_ray_available:
                    mv_ray_pil = draw_gaze_ray_overlay(mv_resized, anno_scaled)
                else:
                    mv_ray_pil = mv_resized.copy()
                mv_ray_prompt_pil = mv_ray_pil

                if save_debug_this:
                    mv_debug_payload[cam] = {
                        "ray_image": mv_ray_pil,
                        "resized_image": mv_resized,
                        "mask_image": None,
                        "mask_crop_image": None,
                        "dot_mask_crop": None,
                    }

                last_mv_candidate = None
                mv_loose_baseline_candidate = None
                for seg_name, seg_cfg in _task1_segmentation_attempts():
                    mv_crop, mv_mask, mv_bb, _, mv_soft, _ = segment_object_at_gaze(
                        zf, split, seq, cam, frame_id, coord_mv,
                        body_bbox_xywh_scaled=mv_body,
                        cfg=seg_cfg
                    )
                    if mv_crop is None or mv_mask is None:
                        continue
                    mv_mask_label = None
                    mv_mask_overlay_label = None
                    mv_mask_context_label = None
                    mv_dot_overlay_label = None
                    mv_refined_label = None
                    mv_refined_large_label = None
                    mv_large_refine_triggered = False
                    mv_large_refine_used = False
                    mv_large_refine_frac = None
                    mv_mask_area = float(mv_mask.sum()) / float(max(1, mv_mask.size))
                    mv_mask_label = describe_masked_object(mv_crop, None, None, scene_type=split)
                    mv_mask_overlay_label = mv_mask_label

                    mv_dot_mask_crop = None
                    if mv_bb is not None:
                        mv_dot_mask_crop = draw_dot_on_crop(
                            mv_crop, coord_mv, mv_bb,
                            alpha=0.6, full_wh=mv_resized.size, color=st.GAZE_COLOR
                        )
                    if mv_dot_mask_crop is not None:
                        mv_dot_overlay_label = describe_masked_object(
                            mv_dot_mask_crop, None, None, scene_type=split
                        )
                        mv_dot_label_full = mv_dot_overlay_label

                    if _should_use_small_mask_context(mv_mask_area, mv_mask_label) and mv_bb is not None:
                        mv_context_dot_crop = _build_context_dot_crop(
                            mv_resized,
                            mv_bb,
                            coord_mv,
                            expand_ratio=float(getattr(st, "TASK1_SMALL_MASK_CONTEXT_EXPAND_RATIO", 2.3)),
                        )
                        mv_mask_context_label = describe_masked_object_contextual(
                            mv_crop,
                            context_dot_pil=mv_context_dot_crop,
                            full_scene_pil=mv_resized,
                            scene_type=split,
                        )
                        if mv_mask_context_label and not _is_generic_label(mv_mask_context_label):
                            mv_mask_label = mv_mask_context_label
                            mv_mask_overlay_label = mv_mask_context_label

                    if mv_bb is not None and coord_mv is not None:
                        x1, y1, x2, y2 = mv_bb
                        mv_raw_crop = mv_resized.crop((x1, y1, x2 + 1, y2 + 1))
                        rx, ry = float(coord_mv[0]) - x1, float(coord_mv[1]) - y1
                        refine_cfg = {
                            "point_box_size": max(80, int(st.TASK1_POINT_BOX_SIZE * 0.6)),
                            "pad_around_mask": max(8, int(st.TASK1_PAD_AROUND_MASK * 0.3)),
                            "pad_around_mask_ratio": max(0.02, st.TASK1_PAD_AROUND_MASK_RATIO * 0.5),
                            "pad_around_mask_max": max(60, int(st.TASK1_PAD_AROUND_MASK_MAX * 0.5)),
                        }
                        mv_ref_crop, _, _, _ = segment_object_on_crop(
                            mv_raw_crop, (rx, ry), cfg=refine_cfg
                        )
                        if mv_ref_crop is not None:
                            mv_refined_label = describe_masked_object(
                                mv_ref_crop, None, None, scene_type=split
                            )
                        do_mv_large_refine, mv_mask_area_px, mv_person_area_px = _should_trigger_large_mask_refine(
                            mv_mask, mv_body
                        )
                        if do_mv_large_refine:
                            mv_large_refine_triggered = True
                            mv_tight_cfg = _task1_large_mask_refine_cfg()
                            mv_tight_crop, mv_tight_mask, _, _ = segment_object_on_crop(
                                mv_raw_crop, (rx, ry), cfg=mv_tight_cfg
                            )
                            if mv_tight_crop is not None and mv_tight_mask is not None:
                                mv_tight_area_px = float(np.asarray(mv_tight_mask, dtype=np.uint8).sum())
                                if mv_mask_area_px > 0.0:
                                    mv_large_refine_frac = float(mv_tight_area_px / mv_mask_area_px)
                                else:
                                    mv_large_refine_frac = None
                                mv_max_frac = float(getattr(st, "TASK1_LARGE_MASK_REFINE_MAX_FRAC", 0.85))
                                if (
                                    mv_large_refine_frac is not None
                                    and mv_large_refine_frac > 0.0
                                    and mv_large_refine_frac <= mv_max_frac
                                ):
                                    mv_refined_large_label = describe_masked_object(
                                        mv_tight_crop, None, None, scene_type=split
                                    )
                                    if mv_refined_large_label and not _is_generic_label(mv_refined_large_label):
                                        mv_large_refine_used = True
                                        if _is_ambiguous_label(mv_mask_label):
                                            mv_mask_label = mv_refined_large_label
                                            mv_mask_overlay_label = mv_refined_large_label
                    mv_mask_for_canon = _choose_mask_label(
                        mv_mask_label, mv_refined_label, mv_mask_area
                    )
                    mv_dot_for_canon = _pick_most_specific_label(
                        [mv_dot_label_full, mv_dot_overlay_label]
                    )
                    if mv_refined_large_label:
                        if (not mv_mask_for_canon) or _is_ambiguous_label(mv_mask_for_canon):
                            mv_mask_for_canon = _sanitize_label(mv_refined_large_label)
                        elif mv_dot_for_canon and _labels_relaxed_match(mv_refined_large_label, mv_dot_for_canon):
                            if not _labels_relaxed_match(mv_mask_for_canon, mv_dot_for_canon):
                                mv_mask_for_canon = _sanitize_label(mv_refined_large_label)

                    mv_label_attempt, mv_mode_attempt, mv_mask_small_skip = canonicalize_mask_overlay(
                        mv_mask_for_canon,
                        mv_dot_for_canon,
                        mask_area_ratio=mv_mask_area,
                        scene_type=split,
                        ray_fallback=mv_ray_label
                    )

                    if mv_ray_available and mv_ray_label is None:
                        mv_ray_prompt_pil = build_ray_label_prompt_image(
                            mv_resized, anno_scaled, body_bbox_xywh=mv_body, target_mask_u8=mv_mask
                        )
                        mv_ray_desc = generate_target_description(
                            mv_ray_prompt_pil, person_desc, cam, scene_type=split
                        )
                        mv_ray_label = distill_object_phrase(mv_ray_desc, scene_type=split)
                        mv_label_attempt, mv_mode_attempt, mv_mask_small_skip = canonicalize_mask_overlay(
                            mv_mask_for_canon,
                            mv_dot_for_canon,
                            mask_area_ratio=mv_mask_area,
                            scene_type=split,
                            ray_fallback=mv_ray_label
                        )

                    if (
                        _is_loose_stage(seg_name)
                        and (not mv_label_attempt or _is_ambiguous_label(mv_label_attempt))
                        and _dot_tiebreak_allowed(
                            mv_label_attempt,
                            mv_dot_for_canon,
                            ray_fallback=mv_ray_label,
                        )
                    ):
                        mv_label_attempt = _sanitize_label(mv_dot_for_canon)
                        mv_mode_attempt = (
                            f"{mv_mode_attempt}+dot_tiebreak_ray"
                            if mv_mode_attempt else "dot_tiebreak_ray"
                        )

                    mv_positional_label = None
                    if (
                        mv_mask_area is not None
                        and mv_mask_area <= st.TASK1_SMALL_OBJ_AREA_RATIO
                        and mv_mask_for_canon
                        and mv_dot_for_canon
                        and not mv_mask_small_skip
                    ):
                        if not _is_generic_label(mv_mask_for_canon) and not _is_generic_label(mv_dot_for_canon):
                            mv_positional_label = _compose_positional_label(mv_mask_for_canon, mv_dot_for_canon)
                            if mv_positional_label:
                                mv_label_attempt = mv_positional_label
                                mv_mode_attempt = f"{mv_mode_attempt}+positional" if mv_mode_attempt else "positional"

                    if not mv_label_attempt:
                        continue

                    mv_label_attempt = _squash_on_phrase(mv_label_attempt)
                    cue_hits = _task1_label_cue_agreement(
                        mv_label_attempt,
                        dot_label=mv_dot_for_canon or mv_dot_label_full,
                        ray_label=mv_ray_label,
                    )
                    candidate = {
                        "label": mv_label_attempt,
                        "mode": mv_mode_attempt,
                        "mask_area_ratio": mv_mask_area,
                        "ray_label": mv_ray_label,
                        "dot_label_full": mv_dot_label_full,
                        "mask_label": mv_mask_label,
                        "mask_context_label": mv_mask_context_label,
                        "mask_overlay_label": mv_mask_overlay_label,
                        "dot_overlay_label": mv_dot_overlay_label,
                        "dot_mask_crop": mv_dot_mask_crop,
                        "mask_refined_label": mv_refined_label,
                        "mask_refined_large_label": mv_refined_large_label,
                        "large_mask_refine_triggered": mv_large_refine_triggered,
                        "large_mask_refine_used": mv_large_refine_used,
                        "large_mask_refine_frac": mv_large_refine_frac,
                        "positional_label": mv_positional_label,
                        "ray_available": mv_ray_available,
                        "coord_scaled": mv_coord_scaled,
                        "ray_image": mv_ray_prompt_pil,
                        "mask_image": mv_mask,
                        "mask_crop_image": mv_crop,
                        "resized_image": mv_resized,
                    }
                    last_mv_candidate = candidate
                    if _is_loose_stage(seg_name):
                        mv_loose_baseline_candidate = _pick_loose_baseline(
                            mv_loose_baseline_candidate, candidate
                        )

                    if _task1_should_accept_attempt(
                        seg_name, mv_mask_area, cue_hits, label=mv_label_attempt
                    ):
                        chosen_mv = candidate
                        if seg_name == "strict" and mv_loose_baseline_candidate is not None:
                            if not _strict_refines_loose(
                                candidate.get("canonical_object") or candidate.get("label"),
                                mv_loose_baseline_candidate.get("canonical_object")
                                or mv_loose_baseline_candidate.get("label"),
                            ):
                                chosen_mv = dict(mv_loose_baseline_candidate)
                                prev_mode = str(chosen_mv.get("mode") or "").strip()
                                chosen_mv["mode"] = (
                                    f"{prev_mode}+prefer_loose_baseline"
                                    if prev_mode else "prefer_loose_baseline"
                                )
                        last_mv_candidate = chosen_mv
                        break

                if last_mv_candidate is None and mv_loose_baseline_candidate is not None:
                    last_mv_candidate = mv_loose_baseline_candidate
                if last_mv_candidate is not None:
                    mv_label = last_mv_candidate["label"]
                    mv_mode = last_mv_candidate["mode"]
                    mv_mask_area = last_mv_candidate["mask_area_ratio"]
                    mv_ray_label = last_mv_candidate["ray_label"]
                    mv_dot_label_full = last_mv_candidate["dot_label_full"]
                    mv_mask_label = last_mv_candidate["mask_label"]
                    mv_mask_overlay_label = last_mv_candidate["mask_overlay_label"]
                    mv_mask_context_label = last_mv_candidate["mask_context_label"]
                    mv_dot_overlay_label = last_mv_candidate["dot_overlay_label"]
                    mv_dot_mask_crop = last_mv_candidate["dot_mask_crop"]
                    mv_refined_label = last_mv_candidate["mask_refined_label"]
                    mv_refined_large_label = last_mv_candidate["mask_refined_large_label"]
                    mv_large_refine_triggered = last_mv_candidate["large_mask_refine_triggered"]
                    mv_large_refine_used = last_mv_candidate["large_mask_refine_used"]
                    mv_large_refine_frac = last_mv_candidate["large_mask_refine_frac"]
                    mv_positional_label = last_mv_candidate["positional_label"]

                if mv_label and last_mv_candidate is not None:
                    mv_labels[cam] = {
                        "label": mv_label,
                        "mode": mv_mode,
                        "mask_area_ratio": mv_mask_area,
                        "ray_label": mv_ray_label,
                        "dot_label_full": mv_dot_label_full,
                        "mask_label": mv_mask_label,
                        "mask_context_label": mv_mask_context_label,
                        "mask_only_label": mv_mask_label,
                        "dot_mask_label": mv_dot_overlay_label,
                        "mask_overlay_label": mv_mask_overlay_label,
                        "dot_overlay_label": mv_dot_overlay_label,
                        "coord_scaled": mv_coord_scaled,
                        "mask_refined_label": mv_refined_label,
                        "mask_refined_large_label": mv_refined_large_label,
                        "large_mask_refine_triggered": mv_large_refine_triggered,
                        "large_mask_refine_used": mv_large_refine_used,
                        "large_mask_refine_frac": mv_large_refine_frac,
                        "positional_label": mv_positional_label,
                        "ray_available": mv_ray_available,
                        "bbox_gaze_dist_px": mv_bbox_gaze_dist_px,
                        "bbox_gaze_dist_norm": mv_bbox_gaze_dist_norm,
                    }
                    if save_debug_this:
                        mv_debug_payload[cam] = {
                            "ray_image": last_mv_candidate["ray_image"],
                            "resized_image": last_mv_candidate["resized_image"],
                            "mask_image": last_mv_candidate["mask_image"],
                            "mask_crop_image": last_mv_candidate["mask_crop_image"],
                            "dot_mask_crop": mv_dot_mask_crop,
                        }

            if mv_labels:
                mv_canon, mv_canon_mode, mv_weight_map = _synthesize_multiview_labels(
                    mv_labels, scene_type=split, anchor_cam=anchor_cam, anchor_order=anchor_cands
                )
                if mv_canon:
                    can_override, override_mode = _should_override_anchor_with_mv(
                        anchor_label, anchor_canon_mode, mv_canon, mv_labels, anchor_cam
                    )
                    if can_override:
                        _append_label_flow(label_flow, "multiview_override", canonical_object, mv_canon, override_mode)
                        canonical_object = mv_canon
                        mv_mode_tag = mv_canon_mode or "mv"
                        mv_mode_tag = f"{mv_mode_tag}+{override_mode}" if override_mode else mv_mode_tag
                        canon_mode = f"{canon_mode}+mv_{mv_mode_tag}" if canon_mode else f"mv_{mv_mode_tag}"
                    else:
                        if mv_canon_mode:
                            canon_mode = (
                                f"{canon_mode}+mv_{mv_canon_mode}+keep_anchor_{override_mode}"
                                if canon_mode else f"mv_{mv_canon_mode}+keep_anchor_{override_mode}"
                            )
                    if anchor_label:
                        ok, canon, _ = judge_same_object_phrase(anchor_label, mv_canon, scene_type=split)
                        canon = _filter_object_phrase(canon)
                        mv_match_anchor = True if (ok and canon) else False

            guardrail_candidates = _task1_informative_label_map({
                "current": canonical_object,
                "anchor": anchor_label,
                "multiview": mv_canon,
                "mask": phrase_from_mask,
                "mask_context": phrase_from_mask_context,
                "mask_refined": phrase_from_mask_refined,
                "mask_refined_large": phrase_from_mask_refined_large,
                "dot": dot_label_for_canon or phrase_from_dot,
                "ray": phrase_from_ray,
            })
            if teacher_mode:
                teacher_final["student_label"] = _sanitize_label(canonical_object)
                teacher_final["candidates"] = guardrail_candidates
                if st.ARGS.skip_vlm or (not bool(getattr(st, "TASK1_TEACHER_FORCE_CALL", True))):
                    teacher_final["reason"] = "skip_vlm" if st.ARGS.skip_vlm else "teacher_force_call_disabled"
                    teacher_final["final_source"] = "student_fallback"
                    st.REJECT_STATS["t1_teacher_fallback"] += 1
                else:
                    teacher_final["triggered"] = True
                    try:
                        tpass1 = _run_task1_teacher_pass1(
                            person_desc=person_desc,
                            anchor_cam=anchor_cam,
                            scene_type=split,
                            ray_context_pil=ray_pil,
                            student_label=canonical_object,
                            candidate_labels=guardrail_candidates,
                            mask_area_ratio=mask_area_ratio,
                        )
                    except Exception as e:
                        st.logger.warning(f"[Task1] teacher pass1 failed; fallback to student label: {e}")
                        tpass1 = {
                            "final_label": None,
                            "confidence": "LOW",
                            "qwen_verdict": "UNCLEAR",
                            "rationale": None,
                            "parse_ok": False,
                            "parse_status": "request_error",
                            "parse_partial": False,
                            "parse_reason": str(e),
                            "retry_count": 0,
                            "finish_reason": "",
                            "token_budget_used": 0,
                        }
                    tpass1_history = None
                    if isinstance(tpass1, dict):
                        tpass1_history = tpass1.pop("_gemini_history_turn", None)
                    teacher_final["pass1"] = tpass1
                    teacher_final["call_count"] = 1
                    if not bool(tpass1.get("parse_ok")):
                        st.REJECT_STATS["t1_teacher_parse_fail"] += 1
                    if int(tpass1.get("retry_count") or 0) > 0:
                        st.REJECT_STATS["t1_teacher_partial_retry"] += 1
                    t1_label = _sanitize_label(tpass1.get("final_label"))
                    t1_conf = str(tpass1.get("confidence") or "LOW").upper()
                    teacher_final["teacher_label_pass1"] = t1_label
                    teacher_final["qwen_verdict"] = str(tpass1.get("qwen_verdict") or "UNCLEAR").upper()
                    teacher_final["teacher_confidence"] = t1_conf
                    teacher_final["rationale"] = tpass1.get("rationale")

                    student_label = _sanitize_label(canonical_object)
                    mismatch = _task1_should_run_teacher_second_call(student_label, t1_label)
                    if student_label and t1_label:
                        teacher_final["semantic_match_qwen_teacher"] = not mismatch
                    if mismatch:
                        st.REJECT_STATS["t1_teacher_conflict"] += 1

                    selected = student_label
                    selected_source = "student_fallback"
                    selected_reason = "teacher_invalid_or_low_conf"

                    if _task1_teacher_label_valid(t1_label, t1_conf):
                        selected = t1_label
                        selected_source = "teacher_pass1"
                        selected_reason = "teacher_pass1_valid"

                    if (
                        mismatch
                        and int(getattr(st, "TASK1_TEACHER_MAX_CALLS", 2)) > 1
                        and bool(getattr(st, "TASK1_TEACHER_SECOND_CALL_ON_MISMATCH", True))
                    ):
                        st.REJECT_STATS["t1_teacher_second_call"] += 1
                        try:
                            tpass2 = _run_task1_teacher_pass2_conflict(
                                person_desc=person_desc,
                                anchor_cam=anchor_cam,
                                scene_type=split,
                                ray_context_pil=ray_pil,
                                student_label=student_label,
                                teacher_label_pass1=t1_label,
                                candidate_labels=guardrail_candidates,
                                prior_history_turn=tpass1_history,
                            )
                        except Exception as e:
                            st.logger.warning(f"[Task1] teacher pass2 failed; keeping prior selection: {e}")
                            tpass2 = {
                                "final_label": None,
                                "confidence": "LOW",
                                "qwen_verdict": "UNCLEAR",
                                "rationale": None,
                                "parse_ok": False,
                                "parse_status": "request_error",
                                "parse_partial": False,
                                "parse_reason": str(e),
                                "retry_count": 0,
                                "finish_reason": "",
                                "token_budget_used": 0,
                            }
                        if isinstance(tpass2, dict):
                            tpass2.pop("_gemini_history_turn", None)
                        teacher_final["pass2"] = tpass2
                        teacher_final["call_count"] = 2
                        if not bool(tpass2.get("parse_ok")):
                            st.REJECT_STATS["t1_teacher_parse_fail"] += 1
                        if int(tpass2.get("retry_count") or 0) > 0:
                            st.REJECT_STATS["t1_teacher_partial_retry"] += 1
                        t2_label = _sanitize_label(tpass2.get("final_label"))
                        t2_conf = str(tpass2.get("confidence") or "LOW").upper()
                        teacher_final["teacher_label_pass2"] = t2_label
                        if t2_label and _task1_teacher_label_valid(t2_label, t2_conf):
                            selected = t2_label
                            selected_source = "teacher_pass2"
                            selected_reason = "teacher_pass2_conflict_resolved"
                            teacher_final["qwen_verdict"] = str(tpass2.get("qwen_verdict") or "UNCLEAR").upper()
                            teacher_final["teacher_confidence"] = t2_conf
                            teacher_final["rationale"] = tpass2.get("rationale")

                    teacher_final["final_source"] = selected_source
                    teacher_final["final_label"] = selected
                    teacher_final["reason"] = selected_reason
                    if selected_source == "student_fallback":
                        st.REJECT_STATS["t1_teacher_fallback"] += 1

                    if selected and selected != canonical_object:
                        _append_label_flow(
                            label_flow,
                            "teacher_final",
                            canonical_object,
                            selected,
                            selected_reason,
                        )
                        canonical_object = selected
                        teacher_final["applied"] = True
                        canon_mode = f"{canon_mode}+teacher_final" if canon_mode else "teacher_final"

            if (not teacher_mode) and st.TASK1_SEMANTIC_ARBITER:
                if st.ARGS.skip_vlm:
                    semantic_arbiter.update({
                        "trigger_reason": "skip_vlm",
                        "triggered": False,
                        "applied": False,
                    })
                else:
                    arb_candidates = _task1_informative_label_map({
                        "current": canonical_object,
                        "anchor": anchor_label,
                        "multiview": mv_canon,
                        "mask": phrase_from_mask,
                        "mask_context": phrase_from_mask_context,
                        "mask_refined": phrase_from_mask_refined,
                        "mask_refined_large": phrase_from_mask_refined_large,
                        "dot": dot_label_for_canon or phrase_from_dot,
                        "ray": phrase_from_ray,
                    })
                    run_arbiter, trigger_reason = _task1_should_run_semantic_arbiter(
                        canonical_object, arb_candidates
                    )
                    semantic_arbiter.update({
                        "trigger_reason": trigger_reason,
                        "triggered": bool(run_arbiter),
                        "candidates": arb_candidates,
                    })
                    if run_arbiter:
                        arb = _run_task1_semantic_arbiter(
                            person_desc=person_desc,
                            anchor_cam=anchor_cam,
                            scene_type=split,
                            anchor_resized=anchor_resized,
                            ray_label_prompt_pil=ray_label_prompt_pil,
                            masked_crop=masked_crop,
                            dot_mask_crop=dot_mask_crop,
                            candidate_labels=arb_candidates,
                            mask_area_ratio=mask_area_ratio,
                            ray_available=ray_available,
                        )
                        semantic_arbiter.update(arb)
                        proposed = _sanitize_label(arb.get("final_label"))
                        conf = str(arb.get("confidence") or "LOW").upper()
                        conf_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
                        min_conf = str(getattr(st, "TASK1_SEMANTIC_ARBITER_MIN_CONF", "MEDIUM")).upper()
                        enough_conf = conf_rank.get(conf, 0) >= conf_rank.get(min_conf, 1)
                        if proposed and (enough_conf or _is_ambiguous_label(canonical_object) or (not canonical_object)):
                            if proposed != canonical_object:
                                _append_label_flow(
                                    label_flow, "semantic_arbiter", canonical_object, proposed, arb.get("decision")
                                )
                            canonical_object = proposed
                            semantic_arbiter["applied"] = True
                            arb_mode = str(arb.get("decision") or "refine").strip().lower()
                            canon_mode = f"{canon_mode}+arbiter_{arb_mode}" if canon_mode else f"arbiter_{arb_mode}"

            if (not teacher_mode) and st.TASK1_HYBRID_GUARDRAIL:
                if st.ARGS.skip_vlm:
                    hybrid_guardrail.update({
                        "trigger_reason": "skip_vlm",
                        "triggered": False,
                        "applied": False,
                        "applied_by": "none",
                        "proposed_passed_gate": False,
                        "blocked_reason": "skip_vlm",
                        "candidates": guardrail_candidates,
                    })
                else:
                    run_guard, guard_reason = _task1_should_run_hybrid_guardrail(
                        canonical_object, guardrail_candidates
                    )
                    hybrid_guardrail.update({
                        "trigger_reason": guard_reason,
                        "triggered": bool(run_guard),
                        "candidates": guardrail_candidates,
                    })
                    if run_guard:
                        grd = _run_task1_hybrid_guardrail(
                            person_desc=person_desc,
                            anchor_cam=anchor_cam,
                            scene_type=split,
                            anchor_resized=anchor_resized,
                            ray_context_pil=ray_pil,
                            masked_crop=masked_crop,
                            dot_mask_crop=dot_mask_crop,
                            current_label=canonical_object,
                            candidate_labels=guardrail_candidates,
                            mask_area_ratio=mask_area_ratio,
                        )
                        hybrid_guardrail.update(grd)
                        proposed = _sanitize_label(grd.get("final_label"))
                        hybrid_guardrail["proposed_label"] = proposed
                        conf = str(grd.get("confidence") or "LOW").upper()
                        min_conf = str(getattr(st, "TASK1_GUARDRAIL_MIN_CONF", "MEDIUM")).upper()
                        enough_conf = _conf_rank(conf) >= _conf_rank(min_conf)
                        ambiguous_now = _is_ambiguous_label(canonical_object)

                        scene_ok = True
                        if bool(getattr(st, "TASK1_GUARDRAIL_SCENE_CHECK", True)):
                            fit_cur = str(grd.get("scene_fit_current") or "UNCLEAR").upper()
                            fit_new = str(grd.get("scene_fit_proposed") or "UNCLEAR").upper()
                            scene_ok = not (fit_cur == "YES" and fit_new == "NO")

                        cur_hits = _task1_label_cue_agreement(
                            canonical_object,
                            dot_label=dot_label_for_canon or phrase_from_dot,
                            ray_label=phrase_from_ray,
                        )
                        new_hits = _task1_label_cue_agreement(
                            proposed,
                            dot_label=dot_label_for_canon or phrase_from_dot,
                            ray_label=phrase_from_ray,
                        )
                        improves_cues = new_hits > cur_hits
                        decision = str(grd.get("decision") or "").strip().upper()
                        switch_intent = decision.startswith("SWITCH") or decision == "REFINE"
                        proposal_passed_gate = bool(
                            proposed
                            and scene_ok
                            and (enough_conf or ambiguous_now or (not canonical_object))
                            and (improves_cues or switch_intent or ambiguous_now)
                        )
                        hybrid_guardrail["proposed_passed_gate"] = bool(proposal_passed_gate)
                        hybrid_guardrail["blocked_reason"] = None if proposal_passed_gate else "proposal_gate_fail"

                        selected = canonical_object
                        if proposal_passed_gate:
                            qwen_refine_enabled = bool(getattr(st, "TASK1_GUARDRAIL_QWEN_REFINE", True))
                            provider_is_qwen = str(getattr(st, "VLM_PROVIDER", "")).lower() == "qwen"
                            if not qwen_refine_enabled:
                                hybrid_guardrail["blocked_reason"] = "qwen_refine_disabled"
                            elif not provider_is_qwen:
                                hybrid_guardrail["blocked_reason"] = "qwen_missing"
                            else:
                                qref = _run_task1_qwen_guided_refine(
                                    current_label=canonical_object,
                                    guardrail_label=proposed,
                                    candidate_labels=guardrail_candidates,
                                    anchor_resized=anchor_resized,
                                    ray_label_prompt_pil=ray_label_prompt_pil,
                                    masked_crop=masked_crop,
                                    dot_mask_crop=dot_mask_crop,
                                    guardrail_rationale=grd.get("rationale"),
                                    guardrail_scene_setting=grd.get("scene_setting"),
                                    guardrail_target_local_context=grd.get("target_local_context"),
                                )
                                hybrid_guardrail["qwen_refine"] = qref
                                qlab = _sanitize_label(qref.get("final_label"))
                                allowed = qref.get("allowed_labels") or []
                                qmatch = any(_labels_relaxed_match(qlab, a) for a in allowed) if qlab else False
                                conf_ok = _conf_rank(qref.get("confidence")) >= _conf_rank("MEDIUM") or ambiguous_now
                                if not qlab:
                                    hybrid_guardrail["blocked_reason"] = "qwen_missing_label"
                                elif not qmatch:
                                    hybrid_guardrail["blocked_reason"] = "qwen_no_match"
                                elif not conf_ok:
                                    hybrid_guardrail["blocked_reason"] = "qwen_low_conf"
                                else:
                                    selected = qlab

                        if proposal_passed_gate and selected and selected != canonical_object:
                            _append_label_flow(
                                label_flow,
                                "hybrid_guardrail",
                                canonical_object,
                                selected,
                                grd.get("decision"),
                            )
                            canonical_object = selected
                            hybrid_guardrail["applied"] = True
                            hybrid_guardrail["applied_by"] = "qwen_refine"
                            hybrid_guardrail["blocked_reason"] = None
                            grd_mode = str(grd.get("decision") or "refine").strip().lower()
                            canon_mode = (
                                f"{canon_mode}+guardrail_guide_{grd_mode}"
                                if canon_mode else f"guardrail_guide_{grd_mode}"
                            )

            if _is_ambiguous_label(canonical_object) and anchor_idx < (len(anchor_cands) - 1):
                st.logger.info(
                    f"[Task1] ambiguous label '{canonical_object}' in {anchor_cam} after synthesis; trying another anchor."
                )
                continue

            if reasoning is None:
                if st.TASK1_REASONING_MODE == "gt":
                    gp = coord_scaled if isinstance(coord_scaled, (list, tuple)) and len(coord_scaled) == 2 else None
                    gp_txt = f"gaze point ({gp[0]:.1f}, {gp[1]:.1f})" if gp else "gaze point"
                    mar_txt = f"mask area ratio={mask_area_ratio:.4f}" if mask_area_ratio is not None else "mask area ratio=N/A"
                    if anchor_label and canonical_object and anchor_label != canonical_object:
                        if ray_available:
                            reasoning = (
                                f"In {anchor_cam}, the {gp_txt} falls on the segmented mask labeled '{anchor_label}'. "
                                f"Across visible cameras, the majority label is '{canonical_object}'. "
                                f"{mar_txt}."
                            )
                        else:
                            reasoning = (
                                f"In {anchor_cam}, no valid person bbox is available for a gaze ray, so labeling uses "
                                f"dot/mask evidence. The segmented mask is '{anchor_label}', and cross-view synthesis "
                                f"selects '{canonical_object}'. {mar_txt}."
                            )
                    else:
                        if ray_available:
                            reasoning = (
                                f"In {anchor_cam}, the {gp_txt} falls on the segmented mask labeled '{canonical_object}'. "
                                f"The gaze ray from head/eye aligns with that region, and the label is assigned from the mask. "
                                f"{mar_txt}."
                            )
                        else:
                            reasoning = (
                                f"In {anchor_cam}, no valid person bbox is available for a gaze ray, so labeling uses "
                                f"dot/mask evidence around the gaze point for '{canonical_object}'. {mar_txt}."
                            )
                else:
                    prompt = prompts.prompt_task1_reasoning_rich(person_desc, canonical_object, scene_type=split)
                    raw_reason = vlm_generate(
                        [ray_label_prompt_pil, masked_crop, anchor_resized],
                        prompt,
                        max_new_tokens=90
                    )
                    from .vlm import safe_reasoning
                    reasoning = safe_reasoning(
                        raw_reason,
                        f"Across the camera views, the {person_desc} appears to direct attention toward the {canonical_object}."
                    )
            if semantic_arbiter.get("applied") and semantic_arbiter.get("rationale"):
                rationale = _first_two_sentences(semantic_arbiter.get("rationale"))
                if rationale:
                    reasoning = _first_two_sentences(
                        f"{reasoning} Semantic verifier: {rationale}."
                    ) or reasoning
            if hybrid_guardrail.get("applied") and hybrid_guardrail.get("rationale"):
                rationale = _first_two_sentences(hybrid_guardrail.get("rationale"))
                if rationale:
                    guardrail_src = "Scene guardrail (guide-only)"
                    if hybrid_guardrail.get("applied_by") == "qwen_refine":
                        guardrail_src = "Scene guardrail (guide-only, Qwen-applied)"
                    reasoning = _first_two_sentences(
                        f"{reasoning} {guardrail_src}: {rationale}."
                    ) or reasoning
            if teacher_final.get("applied") and teacher_final.get("rationale"):
                rationale = _first_two_sentences(teacher_final.get("rationale"))
                if rationale:
                    reasoning = _first_two_sentences(
                        f"{reasoning} Teacher verifier: {rationale}."
                    ) or reasoning

            pose_check = None
            if st.TASK1_POSE_CHECK and ray_available and (not st.ARGS.skip_vlm):
                prompt_pose = prompts.prompt_task1_pose_check(person_desc, canonical_object, scene_type=split)
                verdict, _ = choose_by_letter(
                    [anchor_resized],
                    prompt_pose,
                    {"A": "YES", "B": "NO", "C": "UNCLEAR"}
                )
                pose_check = verdict
                if verdict == "NO":
                    st.REJECT_STATS["t1_pose_mismatch"] += 1

            confidence, conf_components = _confidence_score_task1(
                mv_labels, canonical_object, anchor_cam=anchor_cam,
                mask_area_ratio=mask_area_ratio, pose_check=pose_check
            )
            if st.TASK1_CONF_THRESHOLD and confidence < st.TASK1_CONF_THRESHOLD:
                st.REJECT_STATS["t1_conf_below_threshold"] += 1
                st.logger.info(
                    f"[Task1] {split}/{seq} frame={frame_id} cam={anchor_cam} "
                    f"fail=conf_below_threshold ({confidence:.3f} < {st.TASK1_CONF_THRESHOLD:.3f})"
                )
                return None

            obj_id = make_id("t1", split, seq, frame_id, anchor_cam, canonical_object)

            if save_debug_this:
                stem = f"t1_{task1_index:03d}_{split}_{seq}_{frame_id}_{anchor_cam}"
                (st.DEBUG_DIR / stem).mkdir(exist_ok=True)

                anchor_resized.save(st.DEBUG_DIR / stem / "anchor_raw_resized.jpg", quality=95)
                ray_pil.save(st.DEBUG_DIR / stem / "anchor_ray.jpg", quality=95)
                ray_label_prompt_pil.save(st.DEBUG_DIR / stem / "anchor_ray_label_prompt.jpg", quality=95)
                masked_crop.save(st.DEBUG_DIR / stem / "masked_crop.jpg", quality=95)
                if dot_mask_crop is not None:
                    dot_mask_crop.save(st.DEBUG_DIR / stem / "anchor_dot_mask_crop.jpg", quality=95)
                if overlay_crop is not None:
                    overlay_crop.save(st.DEBUG_DIR / stem / "masked_mask_crop.jpg", quality=95)
                if full_mask is not None:
                    mask_only = _mask_to_pil(full_mask)
                    if mask_only is not None:
                        mask_only.save(st.DEBUG_DIR / stem / "mask_only.jpg", quality=95)
                collage = _make_collage([
                    anchor_resized, ray_pil,
                    _mask_to_pil(full_mask), masked_crop_soft or masked_crop
                ])
                if collage is not None:
                    collage.save(st.DEBUG_DIR / stem / "collage.jpg", quality=95)

                # Save Task1 per-camera ray/mask debug views to inspect multiview consistency.
                for cam in sorted(mv_debug_payload.keys()):
                    payload = mv_debug_payload.get(cam) or {}
                    ray_im = payload.get("ray_image")
                    if ray_im is not None:
                        ray_im.save(st.DEBUG_DIR / stem / f"mv_{cam}_ray.jpg", quality=95)
                    raw_im = payload.get("resized_image")
                    if raw_im is not None:
                        raw_im.save(st.DEBUG_DIR / stem / f"mv_{cam}_raw.jpg", quality=95)
                    mask_u8 = payload.get("mask_image")
                    if mask_u8 is not None:
                        mv_mask_only = _mask_to_pil(mask_u8)
                        if mv_mask_only is not None:
                            mv_mask_only.save(st.DEBUG_DIR / stem / f"mv_{cam}_mask_only.jpg", quality=95)
                    crop_im = payload.get("mask_crop_image")
                    if crop_im is not None:
                        crop_im.save(st.DEBUG_DIR / stem / f"mv_{cam}_masked_crop.jpg", quality=95)
                    dot_im = payload.get("dot_mask_crop")
                    if dot_im is not None:
                        dot_im.save(st.DEBUG_DIR / stem / f"mv_{cam}_dot_mask_crop.jpg", quality=95)

                log_debug({
                    "task": 1,
                    "stem": stem,
                    "split": split,
                    "seq": seq,
                    "frame_id": frame_id,
                    "anchor_cam": anchor_cam,
                    "anchor_candidates": anchor_cands,
                    "anchor_candidate_scores": anchor_score_meta,
                    "ray_available": ray_available,
                    "person_desc": person_desc,
                    "geom_audit": audit_geom,
                    "mask_bbox": bb,
                    "mask_soft_used": bool(soft_mask is not None),
                    "mask_area_ratio": mask_area_ratio,
                    "target_description": target_description,
                    "phrase_from_ray": phrase_from_ray,
                    "phrase_from_mask": phrase_from_mask,
                    "phrase_from_mask_context": phrase_from_mask_context,
                    "phrase_from_mask_only": phrase_from_mask,
                    "phrase_from_dot": phrase_from_dot,
                    "phrase_from_dot_mask": phrase_from_dot_overlay,
                    "phrase_from_dot_overlay": phrase_from_dot_overlay,
                    "phrase_from_mask_overlay": phrase_from_mask_overlay,
                    "phrase_from_mask_refined": phrase_from_mask_refined,
                    "phrase_from_mask_refined_large": phrase_from_mask_refined_large,
                    "large_mask_refine_triggered": large_mask_refine_triggered,
                    "large_mask_refine_used": large_mask_refine_used,
                    "large_mask_refine_frac": large_mask_refine_frac,
                    "positional_label": positional_label,
                    "canonical_object": canonical_object,
                    "pose_check": pose_check,
                    "confidence": confidence,
                    "confidence_components": conf_components,
                    "multiview_weight_map": mv_weight_map,
                    "multiview_coords_scaled": mv_coords_scaled,
                    "label_flow": label_flow,
                    "canonical_mode": canon_mode,
                    "semantic_arbiter": semantic_arbiter,
                    "hybrid_guardrail": hybrid_guardrail,
                    "teacher_final": teacher_final,
                    "segmentation_try": seg_used,
                })

            question = prompts.prompt_task1_question(person_desc, scene_type=split)
            answer = canonical_object

            return {
                "task_id": 1,
                "question": question,
                "answer": answer,
                "reasoning": reasoning,
                "meta": {
                    "camera_id": anchor_cam,
                    "object_id": obj_id,
                    "person_desc": person_desc,
                    "target_description": target_description,
                    "object_phrase": phrase_from_ray,
                    "object_phrase_dot": phrase_from_dot,
                    "object_phrase_dot_mask": phrase_from_dot_overlay,
                    "object_phrase_dot_overlay": phrase_from_dot_overlay,
                    "object_phrase_mask": phrase_from_mask,
                    "object_phrase_mask_context": phrase_from_mask_context,
                    "object_phrase_mask_only": phrase_from_mask,
                    "object_phrase_mask_overlay": phrase_from_mask_overlay,
                    "object_phrase_mask_refined": phrase_from_mask_refined,
                    "object_phrase_mask_refined_large": phrase_from_mask_refined_large,
                    "canonical_object": canonical_object,
                    "confidence": confidence,
                    "confidence_components": conf_components,
                    "ray_available": ray_available,
                    "ray_label_prompt": "person_bbox+target_mask+gaze_ray",
                    "anchor_canonical_object": anchor_label,
                    "multiview_labels": mv_labels,
                    "multiview_coords_scaled": mv_coords_scaled,
                    "multiview_visible_cams": mv_visible_cams,
                    "multiview_nonvisible_cams": mv_nonvisible_cams,
                    "multiview_unknown_cams": mv_unknown_cams,
                    "multiview_canonical": mv_canon,
                    "multiview_canonical_mode": mv_canon_mode,
                    "multiview_weight_map": mv_weight_map,
                    "multiview_match_anchor": mv_match_anchor,
                    "anchor_candidates": anchor_cands,
                    "anchor_candidate_scores": anchor_score_meta,
                    "anchor_fallback_used": (anchor_cam != anchor_cands[0]),
                    "segmentation_try": seg_used,
                    "canonical_mode": canon_mode,
                    "label_flow": label_flow,
                    "mask_area_ratio": mask_area_ratio,
                    "large_mask_refine_triggered": large_mask_refine_triggered,
                    "large_mask_refine_used": large_mask_refine_used,
                    "large_mask_refine_frac": large_mask_refine_frac,
                    "positional_label": positional_label,
                    "semantic_arbiter": semantic_arbiter,
                    "hybrid_guardrail": hybrid_guardrail,
                    "teacher_final": teacher_final,
                    **audit_geom
                },
                "input_cams": cams,
                "input_images": input_images,
                "scene": split.lower(),
                "timestamp": frame_id,
                "task_type": "gaze_target_recognition"
            }

        except Exception:
            st.REJECT_STATS["exceptions"] += 1
            st.logger.error("Task1 exception while trying anchor candidate:\n" + traceback.format_exc())
            continue

    if last_fail_code == "phrase_missing":
        st.REJECT_STATS["t1_phrase_missing"] += 1
    elif last_fail_code == "canonical_fail":
        st.REJECT_STATS["t1_canonical_fail"] += 1
    elif last_fail_code == "sam2_overlap_reject":
        st.REJECT_STATS["t1_sam2_overlap_reject"] += 1
    st.REJECT_STATS["t1_all_anchor_candidates_failed"] += 1
    if last_fail_reason:
        st.logger.info(
            f"[Task1] fail split={split} seq={seq} frame={frame_id} reason={last_fail_reason}"
        )
    if st.SAVE_DEBUG:
        log_debug({
            "task": 1,
            "split": split,
            "seq": seq,
            "frame_id": frame_id,
            "anchor_candidates": anchor_cands,
            "fail_reason": last_fail_reason,
            "fail_code": last_fail_code
        })
    return None
