# task4.py
from collections import Counter
import random
import numpy as np
from PIL import ImageDraw
from pathlib import Path

from . import state as st
from .annotations import parse_visibility, get_body_bbox, get_head_bbox, scale_annotations_for_resized_image
from .io_utils import save_raw_cam_images_parallel, zip_try_image_path, zip_read_image
from .sam2_utils import ensure_sam2, segment_object_at_gaze_precomputed, _crop_soft_masked, _crop_overlay_from_mask
from .task1 import (
    describe_masked_object,
    describe_masked_object_detailed,
    generate_target_description,
    distill_object_phrase,
    canonicalize_triple_cue,
    judge_same_object_phrase,
    _filter_object_phrase,
    _synthesize_multiview_labels,
)
from .task3 import object_visible_yesno_vlm
from .vlm import vlm_generate, safe_reasoning, choose_by_letter
from . import prompts
from .utils import _resize, make_id, log_debug


def verify_reasoning_with_pixels(images, answer_yesno, explanation, obj_phrase, person_desc, scene_type=None):
    prompt = prompts.prompt_task4_verify(answer_yesno, obj_phrase, person_desc, explanation, scene_type=scene_type)
    raw = vlm_generate(images, prompt, max_new_tokens=8).strip().upper()
    return raw.startswith("PASS"), raw


def _reasoning_has_cues(text):
    if not text:
        return False
    low = text.lower()
    spatial = any(k in low for k in ["left", "right", "front", "behind", "in front", "back", "occluded", "out of view", "outside"])
    view = any(k in low for k in ["field of view", "line of sight", "fov", "gaze", "head", "torso", "shoulder", "body"])
    return spatial and view


def _reasoning_contradicts(answer_yesno, text):
    if not text:
        return True
    low = text.lower()
    neg_cues = any(k in low for k in ["behind", "outside", "out of view", "occluded", "not in front"])
    pos_cues = any(k in low for k in ["in front", "within", "forward field of view", "line of sight"])
    if answer_yesno == "NO" and pos_cues and neg_cues:
        return True
    if answer_yesno == "NO" and pos_cues and not neg_cues:
        # still contradictory to NO
        return True
    if answer_yesno == "YES" and neg_cues:
        return True
    return False


def generate_reasoning_forced(images, answer_yesno, obj_phrase, person_desc, gaze_target=None, scene_type=None):
    """
    Note: Here 'accessibility' is interpreted as: visible/accessible from the person's current line-of-sight.
    If gaze_target is provided, we ground the explanation to that reference.
    """
    prompt, fb = prompts.prompt_task4_reasoning(
        answer_yesno, obj_phrase, person_desc, gaze_target=gaze_target, scene_type=scene_type
    )
    raw = vlm_generate(images, prompt, max_new_tokens=90)
    reasoning = safe_reasoning(raw, fb)
    low = reasoning.lower() if reasoning else ""
    if answer_yesno == "NO" and ("visible" in low or "in front" in low):
        reasoning = fb
    if answer_yesno == "YES" and ("not visible" in low or "behind" in low or "outside" in low):
        reasoning = fb
    if not _reasoning_has_cues(reasoning):
        reasoning = fb
    if _reasoning_contradicts(answer_yesno, reasoning):
        reasoning = fb
    return reasoning


def _soft_verify_reasoning(answer_yesno, explanation):
    if not explanation:
        return False
    low = explanation.lower()
    if answer_yesno == "YES":
        positive = any(k in low for k in ["visible", "in front", "line of sight", "can see"])
        negative = any(k in low for k in ["not visible", "behind", "outside", "out of view", "blocked"])
        return positive and not negative
    if answer_yesno == "NO":
        positive = any(k in low for k in ["not visible", "behind", "outside", "out of view", "blocked"])
        negative = any(k in low for k in ["visible", "in front", "can see", "line of sight"])
        return positive and not negative
    return False


def reasoning_from_gt(answer_yesno, obj_phrase, person_desc, query_cam):
    from . import prompts
    who = prompts.person_ref(person_desc)
    if answer_yesno == "YES":
        return (
            f"In {query_cam}, the {obj_phrase} is within the sight of {who} in this view, "
            "so it is physically visible to them."
        )
    return (
        f"In {query_cam}, the {obj_phrase} is outside the sight of {who} in this view "
        "(e.g., behind, occluded, or out of view), so they cannot physically see it."
    )


def _pick_task4_views(input_images, prefer_cam=None):
    """
    Choose 2-3 views, but prefer including prefer_cam (Task1 anchor cam) if provided.
    """
    imgs = [x for x in input_images if x.get("image")]
    if len(imgs) < 2:
        return None

    if prefer_cam is not None:
        imgs_sorted = sorted(imgs, key=lambda x: 0 if x["cam"] == prefer_cam else 1)
    else:
        imgs_sorted = imgs[:]

    k = 2 if len(imgs_sorted) == 2 else random.choice([2, 3])
    chosen = imgs_sorted[:min(k, len(imgs_sorted))]
    return chosen


def _pick_task4_query_cam(cams, per_cam, zf, split, seq, frame_id, prefer_cam=None):
    candidates = []
    for c in cams:
        if zip_try_image_path(zf, split, seq, c, frame_id) is None:
            continue
        v = parse_visibility(per_cam.get(c, {}))
        if st.TASK4_REQUIRE_PERSON_VISIBLE:
            if v is True:
                candidates.append(c)
        else:
            if v is True or v is False:
                candidates.append(c)

    if prefer_cam in candidates:
        return prefer_cam
    if candidates:
        return random.choice(candidates)
    return None


def _pick_task4_query_cam_any(cams, zf, split, seq, frame_id, prefer_cam=None):
    candidates = []
    for c in cams:
        if zip_try_image_path(zf, split, seq, c, frame_id) is None:
            continue
        candidates.append(c)
    if prefer_cam in candidates:
        return prefer_cam
    if candidates:
        return random.choice(candidates)
    return None


def _pick_task4_query_cam_visible_obj(cams, per_cam, zf, split, seq, frame_id, obj_label, prefer_cam=None, max_checks=4):
    candidates = []
    for c in cams:
        if zip_try_image_path(zf, split, seq, c, frame_id) is None:
            continue
        if st.TASK4_REQUIRE_PERSON_VISIBLE:
            v = parse_visibility(per_cam.get(c, {}))
            if v is not True:
                continue
        candidates.append(c)
    if not candidates:
        return None
    if prefer_cam in candidates:
        candidates.remove(prefer_cam)
        candidates.insert(0, prefer_cam)
    else:
        random.shuffle(candidates)
    prompt = prompts.prompt_object_visible_yesno(obj_label, scene_type=split)
    checks = 0
    for cam in candidates:
        zp = zip_try_image_path(zf, split, seq, cam, frame_id)
        if zp is None:
            continue
        img = zip_read_image(zf, zp)
        verdict, _ = choose_by_letter(
            [_resize(img)],
            prompt,
            {"A": "YES", "B": "NO"}
        )
        checks += 1
        if verdict == "YES":
            return cam
        if checks >= max_checks:
            break
    return None


def _verify_object_visible_in_cam(zf, split, seq, frame_id, cam, obj_label):
    zp = zip_try_image_path(zf, split, seq, cam, frame_id)
    if zp is None:
        return False
    img = zip_read_image(zf, zp)
    prompt = prompts.prompt_object_visible_yesno(obj_label, scene_type=split)
    verdict, _ = choose_by_letter(
        [_resize(img)],
        prompt,
        {"A": "YES", "B": "NO"}
    )
    return verdict == "YES"


def _draw_proxy_ray_overlay(im, anno_scaled, target_xy):
    im = im.copy()
    if not isinstance(anno_scaled, dict) or target_xy is None:
        return im
    tx, ty = float(target_xy[0]), float(target_xy[1])

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

    sx, sy = ec if ec is not None else (hc if hc is not None else (tx, ty))
    draw = ImageDraw.Draw(im)
    draw.line([sx, sy, tx, ty], fill=st.GAZE_COLOR, width=st.GAZE_LINE_W)
    r = st.GAZE_DOT_R
    draw.ellipse([tx - r, ty - r, tx + r, ty + r], fill=st.GAZE_COLOR)
    return im


def _task4_segmentation_attempts():
    """
    Task4 distractor points are NOT gaze points, so we use a more forgiving SAM2 setup
    than Task1 strict mode (larger point boxes; lower soft-conf requirements), while
    keeping person-overlap rejection enabled (we don't want the person as a distractor).
    """
    strict = {
        "use_tight_box": True,
        "point_box_size": max(90, st.TASK1_POINT_BOX_SIZE),
        "pad_around_mask": int(round(st.TASK1_PAD_AROUND_MASK * 1.2)),
        "dilate_mask": True,
        "dilate_iter": max(st.TASK1_DILATE_ITER, 2),
        "mask_min_area_ratio": max(1e-5, st.TASK1_MASK_MIN_AREA_RATIO),
        "mask_max_area_ratio": 0.98,
        "min_soft_conf_around_gaze": 0.0,
        "soft_mask_threshold": max(0.0, st.TASK1_SOFT_MASK_THRESHOLD),
        "reject_if_mask_overlaps_person": True,
        "allow_box_fallback": True,
    }
    relaxed = {
        "use_tight_box": True,
        "point_box_size": max(180, st.TASK1_POINT_BOX_SIZE),
        "pad_around_mask": int(round(st.TASK1_PAD_AROUND_MASK * 1.4)),
        "dilate_mask": True,
        "dilate_iter": max(st.TASK1_DILATE_ITER, 3),
        "mask_min_area_ratio": max(1e-5, st.TASK1_MASK_MIN_AREA_RATIO),
        "mask_max_area_ratio": 0.98,
        "min_soft_conf_around_gaze": 0.0,
        "soft_mask_threshold": max(0.0, st.TASK1_SOFT_MASK_THRESHOLD * 0.5),
        "reject_if_mask_overlaps_person": True,
        "allow_box_fallback": True,
    }
    loose = {
        "use_tight_box": True,
        "point_box_size": max(280, st.TASK1_POINT_BOX_SIZE),
        "pad_around_mask": int(round(st.TASK1_PAD_AROUND_MASK * 1.7)),
        "dilate_mask": True,
        "dilate_iter": max(st.TASK1_DILATE_ITER, 4),
        "mask_min_area_ratio": max(1e-5, st.TASK1_MASK_MIN_AREA_RATIO),
        "mask_max_area_ratio": 0.98,
        "min_soft_conf_around_gaze": 0.0,
        "soft_mask_threshold": 0.0,
        "reject_if_mask_overlaps_person": True,
        "allow_box_fallback": True,
    }
    return [("strict", strict), ("relaxed", relaxed), ("loose", loose)]


def _sample_distractor_object_phrase_sam2(
    zf, split, seq, frame_id,
    anchor_cam,
    gaze_xy_resized,
    per_cam,
    proposal_try=None,
    proposal_max=None,
):
    """
    Find a distractor object by sampling SAM2 near a point far from the gaze point in the anchor view.
    We avoid person-overlap and avoid reusing the gaze target.
    """
    zp = zip_try_image_path(zf, split, seq, anchor_cam, frame_id)
    if zp is None:
        return None

    anchor_orig = zip_read_image(zf, zp)
    anchor_resized = _resize(anchor_orig)
    W, H = anchor_resized.size

    gx, gy = float(gaze_xy_resized[0]), float(gaze_xy_resized[1])
    min_d = st.TASK4_DISTRACTOR_MIN_DIST_RATIO * max(W, H)
    max_d = st.TASK4_DISTRACTOR_MAX_DIST_RATIO * max(W, H)

    def _in_band(pt):
        dx = float(pt[0]) - gx
        dy = float(pt[1]) - gy
        d = (dx * dx + dy * dy) ** 0.5
        return min_d <= d <= max_d

    anno_orig = per_cam.get(anchor_cam, {})
    anno_scaled, _ = scale_annotations_for_resized_image(anno_orig, anchor_orig.size, anchor_resized.size)
    body_bbox_scaled = get_body_bbox(anno_scaled) if anno_scaled else None

    candidates = [
        (W - gx, H - gy),
        (W * 0.10, H * 0.10),
        (W * 0.90, H * 0.10),
        (W * 0.10, H * 0.90),
        (W * 0.90, H * 0.90),
        (W * 0.50, H * 0.10),
        (W * 0.50, H * 0.90),
        (W * 0.10, H * 0.50),
        (W * 0.90, H * 0.50),
    ]

    candidates = [pt for pt in candidates if _in_band(pt)]

    for _ in range(40):
        rx = random.uniform(0.05 * W, 0.95 * W)
        ry = random.uniform(0.05 * H, 0.95 * H)
        if _in_band((rx, ry)):
            candidates.append((rx, ry))

    best_label = None
    best_score = -1.0
    best_area_ratio = None
    best_bbox = None
    best_center = None

    # Track why candidates are rejected to diagnose t4_object_proposal_fail.
    counts = Counter()
    top_verify_crop_no = Counter()
    top_verify_full_no = Counter()
    top_tangible_no = Counter()

    # Reuse embeddings for many point queries on the same anchor image.
    predictor = ensure_sam2()
    anchor_np = np.array(anchor_resized)
    predictor.set_image(anchor_np)
    seg_attempts = _task4_segmentation_attempts()

    for (px, py) in candidates[:18]:
        counts["cand_total"] += 1
        seg_best = None
        seg_best_score = -1.0
        for _, seg_cfg in seg_attempts:
            counts["sam2_try"] += 1
            masked_crop, mask_u8, bb, _, soft_mask, seg_debug = segment_object_at_gaze_precomputed(
                predictor,
                anchor_resized,
                anchor_np,
                point_xy_scaled=(px, py),
                body_bbox_xywh_scaled=body_bbox_scaled,
                cfg=seg_cfg,
                update_reject_stats=False,
            )
            if masked_crop is None:
                reason = None
                if isinstance(seg_debug, dict):
                    reason = seg_debug.get("last_reject_reason")
                counts[f"sam2_fail_{reason or 'none'}"] += 1
                continue
            if mask_u8 is None or bb is None:
                counts["sam2_missing_mask_or_bb"] += 1
                continue

            counts["sam2_ok"] += 1
            mask_area_ratio = float(mask_u8.sum()) / float(max(1, mask_u8.size))
            if mask_area_ratio < st.TASK4_DISTRACTOR_MIN_AREA_RATIO:
                counts["area_too_small"] += 1
                continue
            if mask_area_ratio > st.TASK4_DISTRACTOR_MAX_AREA_RATIO:
                counts["area_too_large"] += 1
                continue

            x1, y1, x2, y2 = bb
            bw = int(x2 - x1 + 1)
            bh = int(y2 - y1 + 1)
            if min(bw, bh) < st.TASK4_DISTRACTOR_MIN_BBOX_PX:
                counts["bbox_too_small"] += 1
                continue

            score = float(mask_area_ratio)
            if score > seg_best_score:
                seg_best = (masked_crop, mask_u8, bb, soft_mask, mask_area_ratio, bw, bh)
                seg_best_score = score

        if seg_best is None:
            continue

        masked_crop, mask_u8, bb, soft_mask, mask_area_ratio, bw, bh = seg_best

        masked_crop_soft = _crop_soft_masked(anchor_resized, soft_mask, bb)
        masked_crop_for_vlm = masked_crop_soft if masked_crop_soft is not None else masked_crop
        lab = describe_masked_object_detailed(masked_crop_for_vlm, scene_type=split)
        lab = lab.strip() if isinstance(lab, str) else lab
        if not lab:
            counts["label_empty"] += 1
            continue
        low = lab.lower()
        if low in st.BAD_OBJECTS or any(w in st.BAD_GENERIC_WORDS for w in low.split()):
            counts["label_generic_or_bad"] += 1
            continue
        if low in st.TASK4_BAD_DISTRACTOR_WORDS or any(w in st.TASK4_BAD_DISTRACTOR_WORDS for w in low.split()):
            counts["label_bad_distractor_words"] += 1
            continue
        if st.TASK4_DISTRACTOR_VERIFY_LABEL:
            prompt = prompts.prompt_object_visible_yesno(lab, scene_type=split)
            verdict, _ = choose_by_letter(
                [masked_crop_for_vlm],
                prompt,
                {"A": "YES", "B": "NO"}
            )
            if verdict != "YES":
                counts["verify_crop_no"] += 1
                top_verify_crop_no[low] += 1
                continue
            # Second-stage verifier on the full image to reject odd labels
            verdict_full, _ = choose_by_letter(
                [anchor_resized],
                prompt,
                {"A": "YES", "B": "NO"}
            )
            if verdict_full != "YES":
                counts["verify_full_no"] += 1
                top_verify_full_no[low] += 1
                continue
        if st.TASK4_DISTRACTOR_TANGIBLE_CHECK:
            prompt_t = prompts.prompt_object_tangible_yesno(lab, scene_type=split)
            verdict_t, _ = choose_by_letter(
                [masked_crop_for_vlm],
                prompt_t,
                {"A": "YES", "B": "NO"}
            )
            if verdict_t != "YES":
                counts["tangible_no"] += 1
                top_tangible_no[low] += 1
                st.REJECT_STATS["t4_object_not_tangible"] += 1
                continue

        counts["candidate_pass"] += 1
        score = mask_area_ratio
        if score > best_score:
            best_label = lab
            best_score = score
            best_area_ratio = mask_area_ratio
            best_bbox = (int(bw), int(bh))
            ys, xs = np.where(mask_u8 > 0)
            if len(xs) > 0:
                best_center = (float(xs.mean()), float(ys.mean()))

    if best_label:
        return best_label, best_area_ratio, best_bbox, best_center

    if st.SAVE_DEBUG:
        def _compact(c: Counter):
            return {k: int(v) for k, v in c.items() if int(v) > 0}

        try_str = f" try={proposal_try}/{proposal_max}" if proposal_try and proposal_max else ""
        st.logger.info(
            f"[Task4] distractor_proposal_fail{try_str} split={split} seq={seq} frame={frame_id} cam={anchor_cam} "
            f"band_px=[{min_d:.1f},{max_d:.1f}] counts={_compact(counts)} "
            f"top_verify_crop_no={top_verify_crop_no.most_common(3)} "
            f"top_verify_full_no={top_verify_full_no.most_common(3)} "
            f"top_tangible_no={top_tangible_no.most_common(3)}"
        )

    return None


def build_task4_accessibility_grounded_to_task1(
    zf, split, seq, frame_id,
    cams, per_cam,
    person_desc,
    t1_sample=None
):
    input_images = save_raw_cam_images_parallel(zf, split, seq, cams, frame_id)
    if len(input_images) < 1:
        st.REJECT_STATS["t4_not_enough_views"] += 1
        return None

    anchor_cam = None
    gaze_target = None

    if t1_sample is not None:
        anchor_cam = t1_sample["meta"].get("camera_id", None)
        gaze_target = t1_sample["meta"].get("canonical_object", None)

    if (anchor_cam is None) or (gaze_target is None):
        st.REJECT_STATS["t4_object_proposal_fail"] += 1
        return None

    # Decide queried object
    obj_phrase = gaze_target
    obj_source = "gaze_target"
    dist_area_ratio = None
    dist_bbox = None
    dist_center = None
    proxy_meta = {"proxy_ray_label": None, "proxy_ray_desc": None, "proxy_mode": None}
    mv_meta = {
        "multiview_labels": {},
        "multiview_canonical": None,
        "multiview_canonical_mode": None,
        "multiview_match_anchor": None,
    }
    if not st.TASK4_USE_GAZE_TARGET:
        gaze_xy = None
        if t1_sample is not None:
            gp = t1_sample.get("meta", {}).get("gaze_point_resized")
            if isinstance(gp, (list, tuple)) and len(gp) == 2:
                gaze_xy = (float(gp[0]), float(gp[1]))
        if gaze_xy is None:
            st.REJECT_STATS["t4_object_proposal_fail"] += 1
            return None

        dist = None
        max_tries = max(1, st.TASK4_OBJECT_PROPOSAL_MAX_TRIES)
        for attempt in range(1, max_tries + 1):
            dist = _sample_distractor_object_phrase_sam2(
                zf, split, seq, frame_id,
                anchor_cam=anchor_cam,
                gaze_xy_resized=gaze_xy,
                per_cam=per_cam,
                proposal_try=attempt,
                proposal_max=max_tries,
            )
            if dist:
                break
        if not dist:
            st.REJECT_STATS["t4_object_proposal_fail"] += 1
            return None
        obj_phrase, dist_area_ratio, dist_bbox, dist_center = dist
        obj_source = "distractor"

        if st.TASK4_PROXY_MULTIVIEW:
            # Proxy-ray label from anchor view using mask centroid
            zp_anchor = zip_try_image_path(zf, split, seq, anchor_cam, frame_id)
            if zp_anchor is not None and dist_center is not None:
                anchor_orig = zip_read_image(zf, zp_anchor)
                anchor_resized = _resize(anchor_orig)
                anno_orig = per_cam.get(anchor_cam, {})
                anno_scaled, _ = scale_annotations_for_resized_image(anno_orig, anchor_orig.size, anchor_resized.size)
                proxy_ray = _draw_proxy_ray_overlay(anchor_resized, anno_scaled, dist_center)
                proxy_desc = generate_target_description(proxy_ray, person_desc, anchor_cam, scene_type=split)
                proxy_ray_label = _filter_object_phrase(distill_object_phrase(proxy_desc, scene_type=split))
                proxy_label = None
                proxy_mode = None
                if proxy_ray_label:
                    proxy_label, proxy_mode = canonicalize_triple_cue(
                        proxy_ray_label, obj_phrase, None,
                        ray_desc=proxy_desc, dot_desc=None, scene_type=split
                    )
                    if proxy_label:
                        obj_phrase = proxy_label
                else:
                    proxy_desc = None
                    proxy_ray_label = None

                proxy_meta = {
                    "proxy_ray_label": proxy_ray_label,
                    "proxy_ray_desc": proxy_desc,
                    "proxy_mode": proxy_mode,
                }
            else:
                proxy_meta = {
                    "proxy_ray_label": None,
                    "proxy_ray_desc": None,
                    "proxy_mode": None,
                }

            # Multi-view synthesis: ask other cams to describe the same object if visible
            mv_labels = {}
            mv_canon = None
            mv_canon_mode = None
            mv_match_anchor = None
            cams_checked = 0
            for cam in cams:
                if cam == anchor_cam:
                    continue
                if cams_checked >= st.TASK4_MV_MAX_CAMS:
                    break
                if zip_try_image_path(zf, split, seq, cam, frame_id) is None:
                    continue
                cams_checked += 1
                if not _verify_object_visible_in_cam(zf, split, seq, frame_id, cam, obj_phrase):
                    continue
                prompt_mv = prompts.prompt_object_in_scene_desc(obj_phrase, scene_type=split)
                img = zip_read_image(zf, zip_try_image_path(zf, split, seq, cam, frame_id))
                raw = vlm_generate([_resize(img)], prompt_mv, max_new_tokens=28)
                if raw and raw.strip().upper().startswith("NONE"):
                    continue
                mv_phrase = _filter_object_phrase(distill_object_phrase(raw, scene_type=split))
                if mv_phrase:
                    mv_labels[cam] = mv_phrase

            if mv_labels:
                mv_canon, mv_canon_mode = _synthesize_multiview_labels(
                    list(mv_labels.values()), scene_type=split
                )
                if mv_canon:
                    ok, canon, _ = judge_same_object_phrase(obj_phrase, mv_canon, scene_type=split)
                    canon = _filter_object_phrase(canon)
                    if ok and canon:
                        obj_phrase = canon
                        mv_match_anchor = True
                    else:
                        mv_match_anchor = False
            mv_meta = {
                "multiview_labels": mv_labels,
                "multiview_canonical": mv_canon,
                "multiview_canonical_mode": mv_canon_mode,
                "multiview_match_anchor": mv_match_anchor,
            }
        else:
            proxy_meta = {
                "proxy_ray_label": None,
                "proxy_ray_desc": None,
                "proxy_mode": None,
            }
            mv_meta = {
                "multiview_labels": {},
                "multiview_canonical": None,
                "multiview_canonical_mode": None,
                "multiview_match_anchor": None,
            }

    # Choose query cam
    if st.TASK4_USE_GAZE_TARGET:
        query_cam = _pick_task4_query_cam(cams, per_cam, zf, split, seq, frame_id, prefer_cam=None)
    else:
        query_cam = _pick_task4_query_cam_visible_obj(
            cams, per_cam, zf, split, seq, frame_id,
            obj_label=obj_phrase,
            prefer_cam=None,
            max_checks=4
        )
        if query_cam is None:
            st.REJECT_STATS["t4_object_not_in_view"] += 1
            return None
    if query_cam is None:
        st.REJECT_STATS["t4_no_visibility"] += 1
        return None
    if not st.TASK4_USE_GAZE_TARGET:
        if not _verify_object_visible_in_cam(zf, split, seq, frame_id, query_cam, obj_phrase):
            st.REJECT_STATS["t4_object_not_in_view"] += 1
            return None

    # Visibility source
    v = None
    visibility_source = "gt"
    if st.TASK4_USE_GAZE_TARGET:
        v = parse_visibility(per_cam.get(query_cam, {}))
        if v is None:
            st.REJECT_STATS["t4_no_visibility"] += 1
            return None
        answer = "YES" if v is True else "NO"
    else:
        # Pseudo-GT: VLM accessibility (line-of-sight) on query cam
        by_cam = {x["cam"]: x["image"] for x in input_images if x.get("image")}
        if query_cam not in by_cam:
            st.REJECT_STATS["t4_not_enough_views"] += 1
            return None
        prompt = prompts.prompt_task4_accessibility_yesno(obj_phrase, person_desc, scene_type=split)
        pred, _ = choose_by_letter(
            [by_cam[query_cam]],
            prompt,
            {"A": "YES", "B": "NO"}
        )
        answer = pred
        v = True if answer == "YES" else False
        visibility_source = "pseudo_vlm"

    by_cam = {x["cam"]: x["image"] for x in input_images if x.get("image")}
    if query_cam not in by_cam:
        st.REJECT_STATS["t4_not_enough_views"] += 1
        return None

    use_imgs = [by_cam[query_cam]]
    used_cams = [query_cam]

    question = prompts.prompt_task4_question(query_cam, obj_phrase, person_desc, scene_type=split)

    reasoning = None
    verify_reason = None
    # If distractor + pseudo visibility, run verifier even in GT reasoning mode
    if (not st.TASK4_USE_GAZE_TARGET) and st.TASK4_REQUIRE_VERIFIER_PASS:
        ok = False
        judge_raw = "FAIL"
        for _ in range(max(1, st.TASK4_VERIFIER_MAX_TRIES)):
            verify_reason = generate_reasoning_forced(
                use_imgs,
                answer_yesno=answer,
                obj_phrase=obj_phrase,
                person_desc=person_desc,
                gaze_target=None,
                scene_type=split
            )
            ok, judge_raw = verify_reasoning_with_pixels(
                use_imgs, answer, verify_reason, obj_phrase, person_desc, scene_type=split
            )
            if not ok and _soft_verify_reasoning(answer, verify_reason):
                ok = True
                judge_raw = "PASS_SOFT"
            if ok:
                break
        if not ok:
            st.REJECT_STATS["t4_verifier_fail"] += 1
            log_debug({
                "task": 4,
                "split": split,
                "seq": seq,
                "frame_id": frame_id,
                "person_desc": person_desc,
                "anchor_cam": anchor_cam,
                "gaze_target": gaze_target,
                "query_cam": query_cam,
                "visibility_gt": v,
                "queried_object": obj_phrase,
                "answer": answer,
                "reasoning": verify_reason,
                "verification": judge_raw,
                "fail_reason": "verifier_fail",
            })
            return None
    if st.REASONING_MODE == "gt":
        judge_raw = "SKIP"
        if st.TASK4_FORCE_VLM_REASONING:
            if verify_reason:
                reasoning = verify_reason
            else:
                reasoning = generate_reasoning_forced(
                    use_imgs,
                    answer_yesno=answer,
                    obj_phrase=obj_phrase,
                    person_desc=person_desc,
                    gaze_target=gaze_target,
                    scene_type=split
                )
        else:
            reasoning = reasoning_from_gt(answer, obj_phrase, person_desc, query_cam)
    elif st.TASK4_REQUIRE_VERIFIER_PASS:
        ok = False
        judge_raw = "FAIL"
        for _ in range(max(1, st.TASK4_VERIFIER_MAX_TRIES)):
            reasoning = generate_reasoning_forced(
                use_imgs,
                answer_yesno=answer,
                obj_phrase=obj_phrase,
                person_desc=person_desc,
                gaze_target=gaze_target,
                scene_type=split
            )
            ok, judge_raw = verify_reasoning_with_pixels(
                use_imgs, answer, reasoning, obj_phrase, person_desc, scene_type=split
            )
            if not ok and _soft_verify_reasoning(answer, reasoning):
                ok = True
                judge_raw = "PASS_SOFT"
            if ok:
                break
        if not ok:
            st.REJECT_STATS["t4_verifier_fail"] += 1
            log_debug({
                "task": 4,
                "split": split,
                "seq": seq,
                "frame_id": frame_id,
                "person_desc": person_desc,
                "anchor_cam": anchor_cam,
                "gaze_target": gaze_target,
                "query_cam": query_cam,
                "visibility_gt": v,
                "queried_object": obj_phrase,
                "answer": answer,
                "reasoning": reasoning,
                "verification": judge_raw,
                "fail_reason": "verifier_fail",
            })
            return None
    else:
        judge_raw = "SKIP"
        reasoning = generate_reasoning_forced(
            use_imgs,
            answer_yesno=answer,
            obj_phrase=obj_phrase,
            person_desc=person_desc,
            gaze_target=gaze_target,
            scene_type=split
        )

    answer_text = answer
    obj_id = make_id("t4", split, seq, frame_id, obj_phrase, answer, ",".join([Path(p).name for p in use_imgs]))

    if st.SAVE_DEBUG and random.random() < 0.18:
        log_debug({
            "task": 4,
            "split": split,
            "seq": seq,
            "frame_id": frame_id,
            "person_desc": person_desc,
            "anchor_cam": anchor_cam,
            "gaze_target": gaze_target,
            "query_cam": query_cam,
            "visibility_gt": v,
            "queried_object": obj_phrase,
            "answer": answer,
            "reasoning": reasoning,
            "verification": judge_raw
        })

    verified = bool(st.TASK4_REQUIRE_VERIFIER_PASS) and (not st.TASK4_USE_GAZE_TARGET or st.REASONING_MODE != "gt")

    return {
        "task_id": 4,
        "question": question,
        "answer": answer_text,
        "reasoning": reasoning,
        "meta": {
            "camera_id": ",".join(used_cams),
            "object_id": obj_id,
            "person_desc": person_desc,
            "queried_object": obj_phrase,
            "distractor_mask_area_ratio": dist_area_ratio if obj_source == "distractor" else None,
            "distractor_bbox_px": dist_bbox if obj_source == "distractor" else None,
            "distractor_mask_center": dist_center if obj_source == "distractor" else None,
            "proxy_ray_label": proxy_meta["proxy_ray_label"] if obj_source == "distractor" else None,
            "proxy_ray_desc": proxy_meta["proxy_ray_desc"] if obj_source == "distractor" else None,
            "proxy_mode": proxy_meta["proxy_mode"] if obj_source == "distractor" else None,
            "multiview_labels": mv_meta["multiview_labels"] if obj_source == "distractor" else {},
            "multiview_canonical": mv_meta["multiview_canonical"] if obj_source == "distractor" else None,
            "multiview_canonical_mode": mv_meta["multiview_canonical_mode"] if obj_source == "distractor" else None,
            "multiview_match_anchor": mv_meta["multiview_match_anchor"] if obj_source == "distractor" else None,
            "grounded_to_task1": True,
            "task1_anchor_cam": anchor_cam,
            "task1_gaze_target": gaze_target,
            "visibility_gt": v,
            "visibility_source": visibility_source,
            "queried_object_source": obj_source,
            "verified_reasoning": verified
        },
        "input_cams": used_cams,
        "input_images": [{"cam": query_cam, "image": by_cam[query_cam]}],
        "scene": split.lower(),
        "timestamp": frame_id,
        "task_type": "viewpoint_based_accessibility"
    }
