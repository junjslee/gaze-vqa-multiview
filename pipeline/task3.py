# task3.py
from . import state as st
from .annotations import parse_visibility
from .vlm import choose_by_letter
from . import prompts
from .io_utils import save_raw_cam_image


def object_visible_yesno_vlm(image_path, obj_label, scene_type=None):
    pred, raw = choose_by_letter(
        [image_path],
        prompts.prompt_object_visible_yesno(obj_label, scene_type=scene_type),
        {"A": "YES", "B": "NO"}
    )
    return pred, raw


def build_task3_binary_per_view(zf, split, seq, frame_id, cams_subset, obj_label, person_desc, per_cam, use_gt=True):
    input_images = []
    per_view = {}

    for cam in cams_subset:
        p = save_raw_cam_image(zf, split, seq, cam, frame_id)
        if not p:
            continue
        input_images.append({"cam": cam, "image": p})

        if use_gt and st.TASK3_USE_GT_VISIBILITY_IF_PRESENT:
            v = parse_visibility(per_cam.get(cam, {}))
            if v is True:
                per_view[cam] = "YES"
            elif v is False:
                per_view[cam] = "NO"
            else:
                per_view[cam], _ = object_visible_yesno_vlm(p, obj_label, scene_type=split)
        else:
            per_view[cam], _ = object_visible_yesno_vlm(p, obj_label, scene_type=split)

    used_cams = [cam for cam in cams_subset if cam in per_view]

    fallback_used = False
    if len(per_view) < st.TASK3_NUM_VIEWS_MIN:
        st.REJECT_STATS["t3_not_enough_views"] += 1
        if len(per_view) >= 2:
            fallback_used = True
            st.logger.info(
                f"[Task3] fallback to 2 views split={split} seq={seq} frame={frame_id} "
                f"views={used_cams}"
            )
        else:
            return None

    any_yes = any(v == "YES" for v in per_view.values())
    all_yes = bool(per_view) and all(v == "YES" for v in per_view.values())

    if not any_yes:
        answer = f"Gaze target: {obj_label}. " + st.TASK3_OUTSIDE_SUMMARY.format(person_desc=person_desc)
    else:
        parts = [f"{cam}: {per_view[cam]}" for cam in used_cams]
        answer = f"Gaze target: {obj_label}. " + ". ".join(parts) + "."

    question = prompts.prompt_task3_question(person_desc, obj_label, scene_type=split)

    if not any_yes:
        reasoning = (
            f"Gaze target is {obj_label}. "
            "None of the provided views contain the target object, so the gaze target lies outside these views."
        )
    elif all_yes:
        reasoning = (
            f"Gaze target is {obj_label}. "
            "The target object is visible in all provided views."
        )
    else:
        reasoning = (
            f"Gaze target is {obj_label}. "
            "The target object is visible in some provided views but not others."
        )

    from .utils import make_id
    obj_id = make_id("t3", split, seq, frame_id, obj_label, ",".join(used_cams))

    return {
        "task_id": 3,
        "question": question,
        "answer": answer,
        "reasoning": reasoning,
        "meta": {
            "camera_id": ",".join(used_cams),
            "object_id": obj_id,
            "binary_per_view": per_view,
            "fallback_min_views_used": fallback_used,
            "views_used": used_cams,
        },
        "input_cams": used_cams,
        "input_images": input_images,
        "scene": split.lower(),
        "timestamp": frame_id,
        "task_type": "cross_view_visibility_estimation"
    }
