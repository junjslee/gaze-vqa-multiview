# main.py
import json
import random
import traceback
import math
import zipfile
import copy
from tqdm import tqdm
import numpy as np

from . import state as st
from .io_utils import ensure_mvgt_zip, zip_list_sequences, zip_read_json, zip_try_image_path, _interleave_sequences_by_split
from .dataset_utils import normalize_annotations, detect_cameras, cam_anno
from .sam2_utils import ensure_sam2
from .vlm import load_qwen
from .task1 import build_task1, build_person_descriptor
from .task2 import build_task2_relative_camera_rotation, load_triangulate_map
from .task3 import build_task3_binary_per_view
from .task4 import build_task4_accessibility_grounded_to_task1
from .annotations import scale_annotations_for_resized_image, get_body_bbox, has_coord
from .utils import audit_every_n_accepts, write_snapshot_if_needed, write_frame_debug
from .io_utils import save_raw_cam_images_parallel, zip_read_image
from .utils import _resize


def quotas_met(counts):
    if st.REQUIRE_ALL_TASKS_PER_FRAME:
        return counts.get("bundle", 0) >= st.BUNDLE_TARGET
    return (
        counts["task1"] >= st.TARGET_TASK1 and
        counts["task2"] >= st.TARGET_TASK2 and
        counts["task3"] >= st.TARGET_TASK3 and
        counts["task4"] >= st.TARGET_TASK4
    )


def generate_benchmark_single():
    ensure_mvgt_zip()
    ensure_sam2()
    if not st.ARGS.skip_vlm:
        load_qwen()

    if st.SAVE_DEBUG and st.DEBUG_MANIFEST.exists():
        try:
            st.DEBUG_MANIFEST.unlink()
        except Exception:
            pass

    counts = {"task1": 0, "task2": 0, "task3": 0, "task4": 0}
    if st.REQUIRE_ALL_TASKS_PER_FRAME:
        counts["bundle"] = 0
    samples = []

    partial_json = st.RUN_DIR / "partial_rank0.json"

    with zipfile.ZipFile(st.LOCAL_ZIP_PATH, "r") as zf:
        seqs = zip_list_sequences(zf)

        rng = random.Random(st.SEED)
        rng.shuffle(seqs)
        if st.ARGS.splits:
            want = {s.strip().lower() for s in st.ARGS.splits if s.strip()}
            seqs = [s for s in seqs if s[0].lower() in want]
            st.logger.info(f"Filtered to splits={sorted(want)}; sequences={len(seqs)}")
        if not st.ARGS.no_interleave_splits:
            seqs = _interleave_sequences_by_split(seqs, rng)
            st.logger.info(f"Found {len(seqs)} sequences in ZIP (interleaved by split).")
        else:
            st.logger.info(f"Found {len(seqs)} sequences in ZIP (shuffled).")

        if st.ARGS.scan_dataset_stats:
            st.logger.info("Scanning dataset stats (frames / gaze points)...")
            total_frames = 0
            frames_with_gaze = 0
            total_gaze_points = 0
            for split, seq, anno_path in tqdm(seqs, desc="dataset stats"):
                raw = zip_read_json(zf, anno_path)
                frames = normalize_annotations(raw)
                if not frames:
                    continue
                total_frames += len(frames)
                for fr in frames:
                    frame_data = fr["data"]
                    cams = detect_cameras(frame_data)
                    if not cams:
                        continue
                    per_cam = {c: cam_anno(frame_data, c) for c in cams}
                    has_any = False
                    for c in cams:
                        if has_coord(per_cam.get(c, {})):
                            total_gaze_points += 1
                            has_any = True
                    if has_any:
                        frames_with_gaze += 1
            st.FRAME_STATS["dataset_total_frames"] = total_frames
            st.FRAME_STATS["dataset_frames_with_gaze"] = frames_with_gaze
            st.FRAME_STATS["dataset_total_gaze_points"] = total_gaze_points
            st.FRAME_STATS["dataset_total_sequences"] = len(seqs)
            st.DATASET_STATS = {
                "total_sequences": len(seqs),
                "total_frames": total_frames,
                "frames_with_gaze": frames_with_gaze,
                "total_gaze_points": total_gaze_points,
            }
            st.logger.info(
                f"Dataset stats: sequences={len(seqs)} frames={total_frames} "
                f"frames_with_gaze={frames_with_gaze} gaze_points={total_gaze_points}"
            )

        splits_all = sorted({s.lower() for s, _, _ in seqs})
        balance_splits = bool(st.ARGS.balance_splits) and (not st.ARGS.no_balance_splits)
        split_caps = {}
        split_counts = {s: {"task1": 0, "task2": 0, "task3": 0, "task4": 0} for s in splits_all}
        if st.REQUIRE_ALL_TASKS_PER_FRAME:
            for s in split_counts:
                split_counts[s]["bundle"] = 0
        if balance_splits and splits_all:
            if st.REQUIRE_ALL_TASKS_PER_FRAME:
                split_caps = {
                    "bundle": 0 if st.BUNDLE_TARGET <= 0 else max(1, math.ceil(st.BUNDLE_TARGET / len(splits_all))),
                    "task1": 0 if st.TARGET_TASK1 <= 0 else max(1, math.ceil(st.TARGET_TASK1 / len(splits_all))),
                    "task2": 0 if st.TARGET_TASK2 <= 0 else max(1, math.ceil(st.TARGET_TASK2 / len(splits_all))),
                    "task3": 0 if st.TARGET_TASK3 <= 0 else max(1, math.ceil(st.TARGET_TASK3 / len(splits_all))),
                    "task4": 0 if st.TARGET_TASK4 <= 0 else max(1, math.ceil(st.TARGET_TASK4 / len(splits_all))),
                }
            else:
                split_caps = {
                    "task1": 0 if st.TARGET_TASK1 <= 0 else max(1, math.ceil(st.TARGET_TASK1 / len(splits_all))),
                    "task2": 0 if st.TARGET_TASK2 <= 0 else max(1, math.ceil(st.TARGET_TASK2 / len(splits_all))),
                    "task3": 0 if st.TARGET_TASK3 <= 0 else max(1, math.ceil(st.TARGET_TASK3 / len(splits_all))),
                    "task4": 0 if st.TARGET_TASK4 <= 0 else max(1, math.ceil(st.TARGET_TASK4 / len(splits_all))),
                }
            st.logger.info(f"Split balance enabled: splits={splits_all} caps={split_caps}")

        def can_take_split(split_l, task_key):
            if not balance_splits or not splits_all:
                return True
            cap = split_caps.get(task_key, 0)
            if cap <= 0:
                return True
            if split_counts.get(split_l, {}).get(task_key, 0) < cap:
                return True
            need_more = any(split_counts[s][task_key] < cap for s in splits_all)
            return not need_more

        def _collect_task1_debug_images(split, seq, frame_id, anchor_cam):
            if (not st.SAVE_DEBUG) or (not anchor_cam):
                return []
            pattern = f"t1_*_{split}_{seq}_{frame_id}_{anchor_cam}"
            matches = sorted(st.DEBUG_DIR.glob(pattern))
            if not matches:
                return []
            dbg_dir = matches[-1]
            ordered = [
                "anchor_raw_resized.jpg",
                "anchor_ray.jpg",
                "masked_crop.jpg",
                "masked_overlay_crop.jpg",
                "mask_overlay.jpg",
                "collage.jpg",
            ]
            imgs = []
            for name in ordered:
                p = dbg_dir / name
                if p.exists():
                    imgs.append(str(p))
            return imgs

        def _prepare_frame_task(task, accepted=False, debug_images=None):
            if task is None:
                return None
            t_copy = copy.deepcopy(task)
            t_copy["_accepted"] = bool(accepted)
            if debug_images:
                t_copy["_debug_images"] = debug_images
            return t_copy

        stop_frames = False
        for seq_idx, (split, seq, anno_path) in enumerate(tqdm(seqs, desc="scanning sequences")):
            if quotas_met(counts):
                break
            if st.ARGS.max_sequences and seq_idx >= st.ARGS.max_sequences:
                st.logger.info(f"Reached max_sequences={st.ARGS.max_sequences}. Stopping.")
                break

            tri_map = load_triangulate_map(zf, split, seq)

            raw = zip_read_json(zf, anno_path)
            frames = normalize_annotations(raw)
            if not frames:
                continue

            random.shuffle(frames)

            for fr in frames:
                if quotas_met(counts):
                    break
                if st.ARGS.max_frames and st.FRAME_STATS["frames_seen"] >= st.ARGS.max_frames:
                    st.logger.info(f"Reached max_frames={st.ARGS.max_frames}. Stopping.")
                    stop_frames = True
                    break

                st.FRAME_STATS["frames_seen"] += 1

                frame_id = fr["frame_id"]
                frame_data = fr["data"]
                cams = detect_cameras(frame_data)
                if not cams:
                    continue

                real_imgs = sum(1 for c in cams if zip_try_image_path(zf, split, seq, c, frame_id) is not None)
                if real_imgs < 2:
                    continue

                st.FRAME_STATS["frames_with_min_views"] += 1

                per_cam = {c: cam_anno(frame_data, c) for c in cams}
                split_l = split.lower()
                t1 = None
                t2 = None
                t3 = None
                t4 = None
                t1_accepted = False
                t2_accepted = False
                t3_accepted = False
                t4_accepted = False
                frame_debug_tasks = []

                if st.FRAME_STATS["frames_seen"] == st.ARGS.early_warn_frames:
                    t1_rate = counts["task1"] / max(1, st.FRAME_STATS["frames_with_min_views"])
                    if t1_rate < st.ARGS.min_task1_accept_rate:
                        st.logger.warning(
                            f"EARLY WARNING: Task1 accept rate is LOW: {t1_rate:.4f} "
                            f"after {st.FRAME_STATS['frames_with_min_views']} usable frames."
                        )

                if st.REQUIRE_ALL_TASKS_PER_FRAME:
                    if not can_take_split(split_l, "bundle"):
                        continue
                    # Task1
                    try:
                        t1 = build_task1(zf, split, seq, frame_id, cams, per_cam, counts["task1"])
                    except Exception:
                        st.REJECT_STATS["exceptions"] += 1
                        st.logger.error("Task1 exception:\n" + traceback.format_exc())
                        t1 = None
                    if t1 is None:
                        st.REJECT_STATS["frame_missing_task1"] += 1
                        st.REJECT_STATS["frame_incomplete_bundle"] += 1
                        continue

                    # Task2
                    try:
                        t2 = build_task2_relative_camera_rotation(zf, split, seq, frame_id, cams, tri_map, counts["task2"])
                    except Exception:
                        st.REJECT_STATS["exceptions"] += 1
                        st.logger.error("Task2 exception:\n" + traceback.format_exc())
                        t2 = None
                    if t2 is None:
                        st.REJECT_STATS["frame_missing_task2"] += 1
                        st.REJECT_STATS["frame_incomplete_bundle"] += 1
                        continue

                    # Task3 (requires Task1)
                    try:
                        obj_label = t1["meta"]["canonical_object"]
                        person_desc = t1["meta"]["person_desc"]

                        kmax = min(st.TASK3_NUM_VIEWS_MAX, len(cams))
                        kmin = min(st.TASK3_NUM_VIEWS_MIN, len(cams))
                        if kmax < 3:
                            raise RuntimeError("Not enough cams for Task3 >=3 views.")
                        k = kmin if kmin == kmax else random.randint(kmin, kmax)
                        cams_subset = random.sample(cams, k=k)

                        t3 = build_task3_binary_per_view(
                            zf, split, seq, frame_id,
                            cams_subset=cams_subset,
                            obj_label=obj_label,
                            person_desc=person_desc,
                            per_cam=per_cam,
                            use_gt=True,
                        )
                    except Exception:
                        st.REJECT_STATS["exceptions"] += 1
                        st.logger.error("Task3 exception:\n" + traceback.format_exc())
                        t3 = None
                    if t3 is None:
                        st.REJECT_STATS["frame_missing_task3"] += 1
                        st.REJECT_STATS["frame_incomplete_bundle"] += 1
                        continue

                    # Task4 grounded to Task1
                    t4 = None
                    try:
                        person_desc = t1["meta"]["person_desc"]
                        t4 = build_task4_accessibility_grounded_to_task1(
                            zf, split, seq, frame_id, cams, per_cam,
                            person_desc=person_desc,
                            t1_sample=t1
                        )
                    except Exception:
                        st.REJECT_STATS["exceptions"] += 1
                        st.logger.error("Task4 exception:\n" + traceback.format_exc())
                        t4 = None
                    if t4 is None:
                        st.REJECT_STATS["frame_missing_task4"] += 1
                        st.REJECT_STATS["frame_incomplete_bundle"] += 1
                        continue

                    # Accept full bundle
                    samples.extend([t1, t2, t3, t4])
                    counts["task1"] += 1
                    counts["task2"] += 1
                    counts["task3"] += 1
                    counts["task4"] += 1
                    counts["bundle"] += 1
                    t1_accepted = t2_accepted = t3_accepted = t4_accepted = True
                    if split_l in split_counts:
                        split_counts[split_l]["task1"] += 1
                        split_counts[split_l]["task2"] += 1
                        split_counts[split_l]["task3"] += 1
                        split_counts[split_l]["task4"] += 1
                        split_counts[split_l]["bundle"] += 1
                    st.logger.info(
                        f"✅ ACCEPT FRAME BUNDLE ({counts['bundle']}/{st.BUNDLE_TARGET})"
                    )
                    audit_every_n_accepts(samples, counts, every=5)
                    write_snapshot_if_needed(samples, counts, every=5)
                else:
                    # Task1
                    need_t1_for_deps = (counts["task3"] < st.TARGET_TASK3) or (counts["task4"] < st.TARGET_TASK4)
                    append_t1 = counts["task1"] < st.TARGET_TASK1 and can_take_split(split_l, "task1")
                    allow_t1 = append_t1 or (
                        need_t1_for_deps and (can_take_split(split_l, "task3") or can_take_split(split_l, "task4"))
                    )
                    if allow_t1:
                        try:
                            t1 = build_task1(zf, split, seq, frame_id, cams, per_cam, counts["task1"])
                        except Exception:
                            st.REJECT_STATS["exceptions"] += 1
                            st.logger.error("Task1 exception:\n" + traceback.format_exc())
                            t1 = None
                        if t1 is not None and append_t1:
                            samples.append(t1)
                            counts["task1"] += 1
                            t1_accepted = True
                            if split_l in split_counts:
                                split_counts[split_l]["task1"] += 1
                            st.logger.info(f"✅ ACCEPT Task1 ({counts['task1']}/{st.TARGET_TASK1})")
                            audit_every_n_accepts(samples, counts, every=5)
                            write_snapshot_if_needed(samples, counts, every=5)

                    # Task2 (independent)
                    if counts["task2"] < st.TARGET_TASK2 and can_take_split(split_l, "task2"):
                        try:
                            t2 = build_task2_relative_camera_rotation(zf, split, seq, frame_id, cams, tri_map, counts["task2"])
                        except Exception:
                            st.REJECT_STATS["exceptions"] += 1
                            st.logger.error("Task2 exception:\n" + traceback.format_exc())
                            t2 = None
                        if t2 is not None:
                            samples.append(t2)
                            counts["task2"] += 1
                            t2_accepted = True
                            if split_l in split_counts:
                                split_counts[split_l]["task2"] += 1
                            st.logger.info(f"✅ ACCEPT Task2 ({counts['task2']}/{st.TARGET_TASK2})")
                            audit_every_n_accepts(samples, counts, every=5)
                            write_snapshot_if_needed(samples, counts, every=5)

                    # Task3 requires Task1 (because it inherits object label)
                    if t1 is not None and counts["task3"] < st.TARGET_TASK3 and can_take_split(split_l, "task3"):
                        try:
                            obj_label = t1["meta"]["canonical_object"]
                            person_desc = t1["meta"]["person_desc"]

                            kmax = min(st.TASK3_NUM_VIEWS_MAX, len(cams))
                            kmin = min(st.TASK3_NUM_VIEWS_MIN, len(cams))
                            if kmax < 3:
                                raise RuntimeError("Not enough cams for Task3 >=3 views.")
                            k = kmin if kmin == kmax else random.randint(kmin, kmax)
                            cams_subset = random.sample(cams, k=k)

                            t3 = build_task3_binary_per_view(
                                zf, split, seq, frame_id,
                                cams_subset=cams_subset,
                                obj_label=obj_label,
                                person_desc=person_desc,
                                per_cam=per_cam,
                                use_gt=True,
                            )
                        except Exception:
                            st.REJECT_STATS["exceptions"] += 1
                            st.logger.error("Task3 exception:\n" + traceback.format_exc())
                            t3 = None
                        if t3 is not None:
                            samples.append(t3)
                            counts["task3"] += 1
                            t3_accepted = True
                            if split_l in split_counts:
                                split_counts[split_l]["task3"] += 1
                            st.logger.info(f"✅ ACCEPT Task3 ({counts['task3']}/{st.TARGET_TASK3})")
                            audit_every_n_accepts(samples, counts, every=5)
                            write_snapshot_if_needed(samples, counts, every=5)

                    # Task4 grounded to Task1; only generated when Task1 accepted.
                    if counts["task4"] < st.TARGET_TASK4 and can_take_split(split_l, "task4"):
                        t4 = None
                        try:
                            if t1 is not None:
                                person_desc = t1["meta"]["person_desc"]
                            else:
                                anchor_cam = None
                                for c in cams:
                                    if zip_try_image_path(zf, split, seq, c, frame_id) is not None:
                                        anchor_cam = c
                                        break
                                if anchor_cam is None:
                                    person_desc = "person"
                                else:
                                    zp = zip_try_image_path(zf, split, seq, anchor_cam, frame_id)
                                    anchor_orig = zip_read_image(zf, zp)
                                    anchor_resized = _resize(anchor_orig)

                                    anno_orig = per_cam.get(anchor_cam, {})
                                    anno_scaled, _ = scale_annotations_for_resized_image(anno_orig, anchor_orig.size, anchor_resized.size)
                                    body_bbox_scaled = get_body_bbox(anno_scaled) if anno_scaled else None
                                    person_desc = build_person_descriptor(anchor_resized, body_bbox_scaled, scene_type=split)

                            if t1 is not None and counts["task4"] < st.TARGET_TASK4:
                                t4 = build_task4_accessibility_grounded_to_task1(
                                    zf, split, seq, frame_id, cams, per_cam,
                                    person_desc=person_desc,
                                    t1_sample=t1
                                )
                            else:
                                t4 = None
                        except Exception:
                            st.REJECT_STATS["exceptions"] += 1
                            st.logger.error("Task4 exception:\n" + traceback.format_exc())
                            t4 = None

                        if t4 is not None:
                            samples.append(t4)
                            counts["task4"] += 1
                            t4_accepted = True
                            if split_l in split_counts:
                                split_counts[split_l]["task4"] += 1
                            st.logger.info(f"✅ ACCEPT Task4 ({counts['task4']}/{st.TARGET_TASK4})")
                            audit_every_n_accepts(samples, counts, every=5)
                            write_snapshot_if_needed(samples, counts, every=5)

                # Frame-level HTML debug should reflect only what lands in benchmark_gazevqa.json (accepted samples).
                if st.SAVE_DEBUG:
                    if t1 is not None and t1_accepted:
                        anchor_cam = t1.get("meta", {}).get("camera_id")
                        dbg_imgs = _collect_task1_debug_images(split, seq, frame_id, anchor_cam)
                        frame_debug_tasks.append(_prepare_frame_task(t1, True, dbg_imgs))
                    if t2 is not None and t2_accepted:
                        frame_debug_tasks.append(_prepare_frame_task(t2, True))
                    if t3 is not None and t3_accepted:
                        frame_debug_tasks.append(_prepare_frame_task(t3, True))
                    if t4 is not None and t4_accepted:
                        frame_debug_tasks.append(_prepare_frame_task(t4, True))
                    if frame_debug_tasks:
                        write_frame_debug(split, seq, frame_id, frame_debug_tasks)

            if stop_frames:
                break

    partial = {
        "meta": {
            "dataset": "MVGT",
            "protocol": "Haozhen-style Gaze-VQA benchmark v7.4 (Single-process)",
            "qwen_model": None if st.ARGS.skip_vlm else st.QWEN_MODEL_ID,
            "input_images_dir": str(st.RAW_IMG_DIR),
            "debug_dir": str(st.DEBUG_DIR) if st.SAVE_DEBUG else None,
            "targets": {"task1": st.TARGET_TASK1, "task2": st.TARGET_TASK2, "task3": st.TARGET_TASK3, "task4": st.TARGET_TASK4},
            "bundle_target": st.BUNDLE_TARGET if st.REQUIRE_ALL_TASKS_PER_FRAME else None,
            "task3_views": {"min": st.TASK3_NUM_VIEWS_MIN, "max": st.TASK3_NUM_VIEWS_MAX},
            "calib_dfs_enabled": True,
            "load_intri": bool(st.ARGS.load_intri),
            "require_all_tasks_per_frame": bool(st.REQUIRE_ALL_TASKS_PER_FRAME),
            "task1_conf_threshold": float(st.TASK1_CONF_THRESHOLD),
            "dataset_stats": st.DATASET_STATS if st.DATASET_STATS else None,
        },
        "counts": counts,
        "reject_stats": st.REJECT_STATS,
        "frame_stats": st.FRAME_STATS,
        "samples": samples
    }

    with open(partial_json, "w") as f:
        json.dump(partial, f, indent=2)

    with open(st.BENCH_JSON, "w") as f:
        json.dump(partial, f, indent=2)

    st.logger.info("=== DONE ===")
    st.logger.info(f"Saved: {st.BENCH_JSON}")
    st.logger.info(f"Counts: {counts}")
    if st.DATASET_STATS:
        st.logger.info(
            "Dataset totals: "
            f"sequences={st.DATASET_STATS.get('total_sequences')} "
            f"frames={st.DATASET_STATS.get('total_frames')} "
            f"frames_with_gaze={st.DATASET_STATS.get('frames_with_gaze')} "
            f"gaze_points={st.DATASET_STATS.get('total_gaze_points')}"
        )

    st.logger.info("=== FINAL REJECTION BREAKDOWN ===")
    if st.TASK2_DIST_STATS["med_dist_vals"]:
        arr = np.array(st.TASK2_DIST_STATS["med_dist_vals"], dtype=np.float64)
        st.logger.info("=== TASK2 median_dist summary ===")
        st.logger.info(f"seen={st.TASK2_DIST_STATS['seen']} accepted={st.TASK2_DIST_STATS['accepted']} rejected={st.TASK2_DIST_STATS['rejected']}")
        st.logger.info(f"median_dist: min={arr.min():.4g} p25={np.percentile(arr,25):.4g} "
                       f"median={np.median(arr):.4g} p75={np.percentile(arr,75):.4g} max={arr.max():.4g}")
    for k, v in sorted(st.REJECT_STATS.items(), key=lambda x: -x[1]):
        if v > 0:
            st.logger.info(f"{k}: {v}")

    return st.BENCH_JSON


def main():
    try:
        generate_benchmark_single()
    except Exception:
        st.REJECT_STATS["exceptions"] += 1
        st.logger.error("FATAL ERROR:\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
