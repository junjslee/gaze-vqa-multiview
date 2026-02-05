# task1.py
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
    _crop_overlay_from_mask,
    draw_dot_on_crop,
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
        draw_dot = False
    else:
        ex, ey = gx, gy
        draw_dot = True

    draw = ImageDraw.Draw(im)
    draw.line([(sx, sy), (ex, ey)], fill=st.GAZE_COLOR, width=st.GAZE_LINE_W)
    if draw_dot:
        r = st.GAZE_DOT_R
        draw.ellipse([ex - r, ey - r, ex + r, ey + r], fill=st.GAZE_COLOR)

    return im


def draw_gaze_dot_overlay(im, anno_scaled):
    im = im.copy()
    if not isinstance(anno_scaled, dict):
        return im

    coord = anno_scaled.get("coordinate", None)
    if not (isinstance(coord, (list, tuple)) and len(coord) == 2):
        return im

    gx, gy = float(coord[0]), float(coord[1])
    draw = ImageDraw.Draw(im)
    r = st.GAZE_DOT_R
    draw.ellipse([gx - r, gy - r, gx + r, gy + r], fill=st.GAZE_COLOR)
    return im


def generate_target_description(ray_img_pil, person_desc, anchor_cam, scene_type=None):
    prompt = prompts.prompt_target_description_ray(person_desc, anchor_cam, scene_type=scene_type)
    raw = vlm_generate([ray_img_pil], prompt, max_new_tokens=120)
    desc = _first_two_sentences(raw)
    if not desc:
        desc = "The gaze appears to land on a specific object in the scene."
    return desc


def generate_target_description_dot(dot_img_pil, person_desc, anchor_cam, scene_type=None):
    prompt = prompts.prompt_target_description_dot(person_desc, anchor_cam, scene_type=scene_type)
    raw = vlm_generate([dot_img_pil], prompt, max_new_tokens=120)
    desc = _first_two_sentences(raw)
    if not desc:
        desc = "The dot appears to mark a specific object in the scene."
    return desc


def _filter_object_phrase(phrase):
    if not phrase:
        return ""
    low = phrase.strip().lower()
    banned = ("dot", "line", "ray", "marker", "overlay", "circle", "point", "pointer")
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


def _on_relation_plausible(label, scene_type=None):
    if not label or " on " not in str(label).lower():
        return True
    prompt = prompts.prompt_on_relation_plausibility(label, scene_type=scene_type)
    raw = vlm_generate(None, prompt, max_new_tokens=4)
    if not raw:
        return True
    return raw.strip().lower().startswith("y")


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


def describe_overlay_object(overlay_crop_pil, scene_type=None):
    if overlay_crop_pil is None:
        return None
    prompt = prompts.prompt_masked_object(scene_type=scene_type)
    raw = vlm_generate([overlay_crop_pil], prompt, max_new_tokens=24)
    phrase = strict_noun_phrase(raw, max_words=4)
    if not phrase:
        phrase = _salvage_noun_phrase(raw, max_words=4)
    if not phrase:
        phrase = clean_label(raw, max_words=4)
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
        mask_label_for_canon = None
        mask_small_skip = True

    label, mode = canonicalize_triple_cue(
        None, mask_label_for_canon, dot_label,
        ray_desc=None, dot_desc=None, scene_type=scene_type
    )
    if mask_small_skip:
        mode = f"{mode}+mask_small_skip" if mode else "mask_small_skip"
    if not label and ray_fallback:
        label = ray_fallback
        mode = "fallback_ray_only"
    return label, mode, mask_small_skip


def _synthesize_multiview_labels(labels, scene_type=None):
    uniq = [l for l in labels if l]
    if not uniq:
        return None, "mv_missing"
    if len(uniq) == 1:
        return uniq[0], "mv_single"

    clusters = []
    for lab in uniq:
        placed = False
        for c in clusters:
            if _labels_token_subset_match(lab, c["rep"]):
                c["labels"].append(lab)
                placed = True
                break
            ok, canon, _ = judge_same_object_phrase(lab, c["rep"], scene_type=scene_type)
            canon = _filter_object_phrase(canon)
            if ok:
                c["labels"].append(lab)
                if canon:
                    c["canon"] = canon
                placed = True
                break
        if not placed:
            clusters.append({"rep": lab, "labels": [lab], "canon": lab})

    max_size = max(len(c["labels"]) for c in clusters)
    if max_size <= 1:
        # fallback: majority of exact strings
        from collections import Counter
        top = Counter(uniq).most_common(1)[0][0]
        return top, "mv_majority"

    best = None
    best_score = -1e9
    for c in clusters:
        if len(c["labels"]) != max_size:
            continue
        best_label = _pick_most_specific_label(c["labels"])
        score = _label_specificity_score(best_label)
        if score > best_score:
            best_score = score
            best = best_label
    best = best or _pick_most_specific_label(uniq)
    return best, "mv_cluster"


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
    relaxed = {
        "use_tight_box": True,
        "point_box_size": max(75, st.TASK1_POINT_BOX_SIZE),
        "pad_around_mask": int(round(st.TASK1_PAD_AROUND_MASK * 1.4)),
        "dilate_mask": True,
        "dilate_iter": max(st.TASK1_DILATE_ITER, 3),
        "mask_min_area_ratio": max(1e-5, st.TASK1_MASK_MIN_AREA_RATIO),
        "mask_max_area_ratio": min(0.95, st.TASK1_MASK_MAX_AREA_RATIO + 0.1),
        "min_soft_conf_around_gaze": max(0.0, st.TASK1_MIN_SOFT_CONF_AROUND_GAZE * 0.5),
        "soft_mask_threshold": max(0.0, st.TASK1_SOFT_MASK_THRESHOLD),
        "reject_if_mask_overlaps_person": False,
    }
    loose = {
        "use_tight_box": True,
        "point_box_size": max(200, st.TASK1_POINT_BOX_SIZE),
        "pad_around_mask": int(round(st.TASK1_PAD_AROUND_MASK * 1.8)),
        "dilate_mask": True,
        "dilate_iter": max(st.TASK1_DILATE_ITER, 4),
        "mask_min_area_ratio": max(1e-5, st.TASK1_MASK_MIN_AREA_RATIO),
        "mask_max_area_ratio": 0.9,
        "min_soft_conf_around_gaze": 0.0,
        "soft_mask_threshold": 0.0,
        "reject_if_mask_overlaps_person": False,
        "allow_box_fallback": True,
    }
    return [("strict", base), ("relaxed", relaxed), ("loose", loose)]


# =============================================================================
# Task1 builder
# =============================================================================

def list_anchor_cam_candidates(cams, per_cam, zf, split, seq, frame_id):
    """All cams that have coordinate and visibility is not explicitly False AND image exists."""
    out = []
    for c in cams:
        a = per_cam.get(c, {})
        vis = parse_visibility(a)
        if has_coord(a) and (vis is None or vis is True):
            if zip_try_image_path(zf, split, seq, c, frame_id) is not None:
                out.append(c)
    return out


def build_task1(zf, split, seq, frame_id, cams, per_cam, task1_index):
    anchor_cands = list_anchor_cam_candidates(cams, per_cam, zf, split, seq, frame_id)
    if not anchor_cands:
        st.REJECT_STATS["t1_no_anchor"] += 1
        return None

    input_images = save_raw_cam_images_parallel(zf, split, seq, cams, frame_id)
    if len(input_images) < 2:
        st.REJECT_STATS["t1_no_images"] += 1
        return None

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
            gaze_outside_body = False
            body_bbox_scaled = get_body_bbox(anno_scaled)
            if body_bbox_scaled is not None and isinstance(coord_scaled, (list, tuple)) and len(coord_scaled) == 2:
                gx, gy = float(coord_scaled[0]), float(coord_scaled[1])
                x, y, w, h = body_bbox_scaled
                if not (x <= gx <= (x + w) and y <= gy <= (y + h)):
                    gaze_outside_body = True

            person_desc = build_person_descriptor(anchor_resized, body_bbox_scaled, scene_type=split)

            ray_pil = draw_gaze_ray_overlay(anchor_resized, anno_scaled)
            dot_pil = draw_gaze_dot_overlay(anchor_resized, anno_scaled)

            target_description = None
            phrase_from_ray = None
            dot_description = None
            phrase_from_dot = None  # dot-only label from full image

            seg_used = None
            masked_crop = full_mask = bb = soft_mask = None
            phrase_from_mask = None
            phrase_from_mask_overlay = None
            overlay_crop = None
            dot_overlay_crop = None
            phrase_from_dot_overlay = None
            refined_masked_crop = None
            phrase_from_mask_refined = None
            masked_crop_soft = None
            masked_crop_for_vlm = None
            canonical_object = None
            canon_mode = None
            mask_area_ratio = None
            positional_label = None
            mask_small_skip = False

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
                        if gaze_outside_body:
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

                if target_description is None:
                    target_description = generate_target_description(
                        ray_pil, person_desc, anchor_cam, scene_type=split
                    )
                    phrase_from_ray = distill_object_phrase(target_description, scene_type=split)
                    st.logger.info(
                        f"[Task1] ray_desc={target_description} | ray_phrase={phrase_from_ray}"
                    )
                if dot_description is None:
                    dot_description = generate_target_description_dot(
                        dot_pil, person_desc, anchor_cam, scene_type=split
                    )
                    phrase_from_dot = distill_object_phrase(dot_description, scene_type=split)
                    st.logger.info(
                        f"[Task1] dot_desc={dot_description} | dot_phrase={phrase_from_dot}"
                    )

                overlay_crop = _crop_overlay_from_mask(
                    anchor_resized, full_mask, soft_mask, bb,
                    alpha=st.TASK1_MASK_OVERLAY_ALPHA, neutral=True, use_soft_mask=False
                )
                if overlay_crop is not None:
                    dot_overlay_crop = draw_dot_on_crop(
                        overlay_crop, coord_scaled, bb,
                        alpha=0.6, full_wh=anchor_resized.size, color=(255, 255, 255)
                    )
                masked_crop_for_vlm = masked_crop
                phrase_from_mask = describe_masked_object(
                    masked_crop_for_vlm, None, overlay_crop, scene_type=split
                )
                st.logger.info(
                    f"[Task1] mask_phrase={phrase_from_mask}"
                )
                # Overlay cues (always, if available)
                if overlay_crop is not None:
                    phrase_from_mask_overlay = describe_overlay_object(overlay_crop, scene_type=split)
                if dot_overlay_crop is not None:
                    phrase_from_dot_overlay = describe_overlay_object(dot_overlay_crop, scene_type=split)

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
                if not phrase_from_mask and not phrase_from_dot and not phrase_from_ray:
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

                mask_label_for_canon = _pick_most_specific_label(
                    [phrase_from_mask, phrase_from_mask_refined, phrase_from_mask_overlay]
                )
                dot_label_for_canon = _pick_most_specific_label(
                    [phrase_from_dot, phrase_from_dot_overlay]
                )
                canonical_object, canon_mode, mask_small_skip = canonicalize_mask_overlay(
                    mask_label_for_canon,
                    dot_label_for_canon,
                    mask_area_ratio=mask_area_ratio,
                    scene_type=split,
                    ray_fallback=phrase_from_ray
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
                            canonical_object = positional_label
                            canon_mode = f"{canon_mode}+positional" if canon_mode else "positional"

                if canonical_object:
                    canonical_object = _squash_on_phrase(canonical_object)
                    seg_used = seg_name
                    break

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
                canonical_object = preferred
                if prefer_mode:
                    canon_mode = f"{canon_mode}+{prefer_mode}"
            canonical_object = _sanitize_label(canonical_object)
            if canonical_object and " on " in canonical_object:
                if not _on_relation_plausible(canonical_object, scene_type=split):
                    left = canonical_object.split(" on ", 1)[0].strip()
                    if left and not _is_generic_label(left):
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
                    canonical_object = fallback
                    canon_mode = f"{canon_mode}+bleed_fallback" if canon_mode else "bleed_fallback"

            seg_used = seg_name

            # If label is ambiguous, try another anchor camera if available
            if _is_ambiguous_label(canonical_object) and anchor_idx < (len(anchor_cands) - 1):
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
            anchor_label = canonical_object
            anchor_canon_mode = canon_mode

            if anchor_label:
                mv_labels[anchor_cam] = {
                    "label": anchor_label,
                    "mode": anchor_canon_mode,
                    "mask_area_ratio": mask_area_ratio,
                    "ray_label": phrase_from_ray,
                    "dot_label_full": phrase_from_dot,
                    "mask_label": phrase_from_mask,
                    "mask_overlay_label": phrase_from_mask_overlay,
                    "dot_overlay_label": phrase_from_dot_overlay,
                    "mask_refined_label": phrase_from_mask_refined,
                    "positional_label": positional_label,
                }

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

                mv_body = get_body_bbox(anno_scaled)
                mv_ray_label = None
                mv_dot_label_full = None
                mv_ray_desc = None
                mv_dot_desc = None

                mv_label = None
                mv_mode = None
                mv_mask_area = None
                mv_mask_label = None
                mv_mask_overlay_label = None
                mv_dot_overlay_label = None
                mv_refined_label = None
                mv_bb = None
                mv_soft = None
                mv_mask_small_skip = False

                mv_ray_pil = draw_gaze_ray_overlay(mv_resized, anno_scaled)
                mv_ray_desc = generate_target_description(
                    mv_ray_pil, person_desc, cam, scene_type=split
                )
                mv_ray_label = distill_object_phrase(mv_ray_desc, scene_type=split)

                mv_dot_pil = draw_gaze_dot_overlay(mv_resized, anno_scaled)
                mv_dot_desc = generate_target_description_dot(
                    mv_dot_pil, person_desc, cam, scene_type=split
                )
                mv_dot_label_full = distill_object_phrase(mv_dot_desc, scene_type=split)

                for seg_name, seg_cfg in _task1_segmentation_attempts():
                    mv_crop, mv_mask, mv_bb, _, mv_soft, _ = segment_object_at_gaze(
                        zf, split, seq, cam, frame_id, coord_mv,
                        body_bbox_xywh_scaled=mv_body,
                        cfg=seg_cfg
                    )
                    if mv_crop is None or mv_mask is None:
                        continue
                    mv_mask_area = float(mv_mask.sum()) / float(max(1, mv_mask.size))
                    mv_overlay = _crop_overlay_from_mask(
                        mv_resized, mv_mask, mv_soft, mv_bb,
                        alpha=st.TASK1_MASK_OVERLAY_ALPHA, neutral=True, use_soft_mask=False
                    )
                    mv_mask_label = describe_masked_object(mv_crop, None, mv_overlay, scene_type=split)
                    if mv_overlay is not None:
                        mv_mask_overlay_label = describe_overlay_object(mv_overlay, scene_type=split)

                    mv_dot_overlay = None
                    if mv_overlay is not None:
                        mv_dot_overlay = draw_dot_on_crop(
                            mv_overlay, coord_mv, mv_bb,
                            alpha=0.6, full_wh=mv_resized.size, color=(255, 255, 255)
                        )
                    if mv_dot_overlay is not None:
                        mv_dot_overlay_label = describe_overlay_object(mv_dot_overlay, scene_type=split)

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
                    break

                mv_mask_for_canon = _pick_most_specific_label(
                    [mv_mask_label, mv_refined_label, mv_mask_overlay_label]
                )
                mv_dot_for_canon = _pick_most_specific_label(
                    [mv_dot_label_full, mv_dot_overlay_label]
                )

                mv_label, mv_mode, mv_mask_small_skip = canonicalize_mask_overlay(
                    mv_mask_for_canon,
                    mv_dot_for_canon,
                    mask_area_ratio=mv_mask_area,
                    scene_type=split,
                    ray_fallback=mv_ray_label
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
                            mv_label = mv_positional_label
                            mv_mode = f"{mv_mode}+positional" if mv_mode else "positional"

                if mv_label:
                    mv_labels[cam] = {
                        "label": mv_label,
                        "mode": mv_mode,
                        "mask_area_ratio": mv_mask_area,
                        "ray_label": mv_ray_label,
                        "dot_label_full": mv_dot_label_full,
                        "mask_label": mv_mask_label,
                        "mask_overlay_label": mv_mask_overlay_label,
                        "dot_overlay_label": mv_dot_overlay_label,
                        "mask_refined_label": mv_refined_label,
                        "positional_label": mv_positional_label,
                    }

            labels_for_mv = [v["label"] for v in mv_labels.values()]
            if labels_for_mv:
                mv_canon, mv_canon_mode = _synthesize_multiview_labels(
                    labels_for_mv, scene_type=split
                )
                if mv_canon:
                    canonical_object = mv_canon
                    if mv_canon_mode:
                        canon_mode = f"{canon_mode}+mv_{mv_canon_mode}" if canon_mode else f"mv_{mv_canon_mode}"
                    if anchor_label:
                        ok, canon, _ = judge_same_object_phrase(anchor_label, mv_canon, scene_type=split)
                        canon = _filter_object_phrase(canon)
                        mv_match_anchor = True if (ok and canon) else False

            if reasoning is None:
                if st.TASK1_REASONING_MODE == "gt":
                    gp = coord_scaled if isinstance(coord_scaled, (list, tuple)) and len(coord_scaled) == 2 else None
                    gp_txt = f"gaze point ({gp[0]:.1f}, {gp[1]:.1f})" if gp else "gaze point"
                    mar_txt = f"mask area ratio={mask_area_ratio:.4f}" if mask_area_ratio is not None else "mask area ratio=N/A"
                    if anchor_label and canonical_object and anchor_label != canonical_object:
                        reasoning = (
                            f"In {anchor_cam}, the {gp_txt} falls on the segmented mask labeled '{anchor_label}'. "
                            f"Across visible cameras, the majority label is '{canonical_object}'. "
                            f"{mar_txt}."
                        )
                    else:
                        reasoning = (
                            f"In {anchor_cam}, the {gp_txt} falls on the segmented mask labeled '{canonical_object}'. "
                            f"The gaze ray from head/eye aligns with that region, and the label is assigned from the mask. "
                            f"{mar_txt}."
                        )
                else:
                    prompt = prompts.prompt_task1_reasoning_rich(person_desc, canonical_object, scene_type=split)
                    raw_reason = vlm_generate(
                        [ray_pil, overlay_crop or masked_crop, anchor_resized],
                        prompt,
                        max_new_tokens=90
                    )
                    from .vlm import safe_reasoning
                    reasoning = safe_reasoning(
                        raw_reason,
                        f"Across the camera views, the {person_desc} appears to direct attention toward the {canonical_object}."
                    )

            pose_check = None
            if st.TASK1_POSE_CHECK and (not st.ARGS.skip_vlm):
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

            if st.SAVE_DEBUG and (task1_index % max(1, st.DEBUG_EVERY_N_TASK1) == 0):
                stem = f"t1_{task1_index:03d}_{split}_{seq}_{frame_id}_{anchor_cam}"
                (st.DEBUG_DIR / stem).mkdir(exist_ok=True)

                anchor_resized.save(st.DEBUG_DIR / stem / "anchor_raw_resized.jpg", quality=95)
                ray_pil.save(st.DEBUG_DIR / stem / "anchor_ray.jpg", quality=95)
                masked_crop.save(st.DEBUG_DIR / stem / "masked_crop.jpg", quality=95)
                if overlay_crop is not None:
                    overlay_crop.save(st.DEBUG_DIR / stem / "masked_overlay_crop.jpg", quality=95)
                if full_mask is not None:
                    overlay = overlay_mask_on_image(anchor_resized, (full_mask > 0).astype(np.uint8))
                    overlay.save(st.DEBUG_DIR / stem / "mask_overlay.jpg", quality=95)
                collage = _make_collage([
                    anchor_resized, ray_pil,
                    overlay_crop, masked_crop_soft or masked_crop
                ])
                if collage is not None:
                    collage.save(st.DEBUG_DIR / stem / "collage.jpg", quality=95)

                log_debug({
                    "task": 1,
                    "stem": stem,
                    "split": split,
                    "seq": seq,
                    "frame_id": frame_id,
                    "anchor_cam": anchor_cam,
                    "anchor_candidates": anchor_cands,
                    "person_desc": person_desc,
                    "geom_audit": audit_geom,
                    "mask_bbox": bb,
                    "mask_soft_used": bool(soft_mask is not None),
                    "mask_area_ratio": mask_area_ratio,
                    "target_description": target_description,
                    "phrase_from_ray": phrase_from_ray,
                    "phrase_from_mask": phrase_from_mask,
                    "phrase_from_dot": phrase_from_dot,
                    "phrase_from_dot_overlay": phrase_from_dot_overlay,
                    "phrase_from_mask_overlay": phrase_from_mask_overlay,
                    "phrase_from_mask_refined": phrase_from_mask_refined,
                    "positional_label": positional_label,
                    "canonical_object": canonical_object,
                    "pose_check": pose_check,
                    "confidence": confidence,
                    "confidence_components": conf_components,
                    "canonical_mode": canon_mode,
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
                    "object_phrase_dot_overlay": phrase_from_dot_overlay,
                    "object_phrase_mask": phrase_from_mask,
                    "object_phrase_mask_overlay": phrase_from_mask_overlay,
                    "object_phrase_mask_refined": phrase_from_mask_refined,
                    "canonical_object": canonical_object,
                    "confidence": confidence,
                    "confidence_components": conf_components,
                    "anchor_canonical_object": anchor_label,
                    "multiview_labels": mv_labels,
                    "multiview_visible_cams": mv_visible_cams,
                    "multiview_nonvisible_cams": mv_nonvisible_cams,
                    "multiview_unknown_cams": mv_unknown_cams,
                    "multiview_canonical": mv_canon,
                    "multiview_canonical_mode": mv_canon_mode,
                    "multiview_match_anchor": mv_match_anchor,
                    "anchor_candidates": anchor_cands,
                    "anchor_fallback_used": (anchor_cam != anchor_cands[0]),
                    "segmentation_try": seg_used,
                    "canonical_mode": canon_mode,
                    "mask_area_ratio": mask_area_ratio,
                    "positional_label": positional_label,
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
