# task2.py
import json
import math
import random
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

from . import state as st
from .io_utils import zip_read_json
from .vlm import vlm_generate, _first_two_sentences
from . import prompts
from .utils import make_id


# =============================================================================
# 16) Triangulate_3d loader
# =============================================================================

def find_triangulate_3d_json(zf, split, seq):
    hits = []
    for p in zf.namelist():
        low = p.lower()
        if "triangulate_3d" in low and split.lower() in low and seq.lower() in low:
            hits.append(p)
    if hits:
        hits.sort()
        return hits[0]

    for p in zf.namelist():
        low = p.lower()
        if low.endswith(".json") and "triangulate_3d" in low and seq.lower() in low:
            hits.append(p)
    if hits:
        hits.sort()
        return hits[0]

    for p in zf.namelist():
        low = p.lower()
        if low.endswith(".json") and "triangulate_3d" in low:
            hits.append(p)
    if hits:
        hits.sort()
        return hits[0]

    return None


def load_triangulate_map(zf, split, seq):
    path = find_triangulate_3d_json(zf, split, seq)
    if path is None:
        st.logger.warning(f"⚠️ Triangulate_3d JSON not found for {split}/{seq}")
        return None

    try:
        raw = zip_read_json(zf, path)
    except Exception as e:
        st.logger.warning(f"⚠️ Failed to read triangulate JSON: {path} err={e}")
        return None

    if isinstance(raw, dict) and "frames" in raw:
        raw = raw["frames"]

    tri = {}
    if isinstance(raw, list):
        for e in raw:
            if not isinstance(e, dict):
                continue
            fid = e.get("frame_id", None)
            if fid is None:
                fname = e.get("filename", None)
                if fname:
                    base = str(fname).split("/")[-1]
                    stem = base.split(".")[0]
                    try:
                        fid = int(str(stem).lstrip("0") or "0")
                    except Exception:
                        fid = None
            if fid is None:
                continue
            try:
                fi = int(str(fid).lstrip("0") or "0")
            except Exception:
                continue
            eye = e.get("eye3d", None) or e.get("eye", None)
            if isinstance(eye, (list, tuple)) and len(eye) == 3:
                eye = np.array([float(eye[0]), float(eye[1]), float(eye[2])], dtype=np.float64)
            else:
                eye = None
            tri[f"{fi:04d}"] = {
                "eye3d": eye,
                "src": path,
                "eye_err": e.get("eye_err", None),
                "target_err": e.get("target_err", None),
            }
    elif isinstance(raw, dict):
        for k, v in raw.items():
            if not isinstance(v, dict):
                continue
            try:
                fi = int(str(k).lstrip("0") or "0")
            except Exception:
                continue
            eye = v.get("eye3d", None) or v.get("eye", None)
            if isinstance(eye, (list, tuple)) and len(eye) == 3:
                eye = np.array([float(eye[0]), float(eye[1]), float(eye[2])], dtype=np.float64)
            else:
                eye = None
            tri[f"{fi:04d}"] = {
                "eye3d": eye,
                "src": path,
                "eye_err": v.get("eye_err", None),
                "target_err": v.get("target_err", None),
            }

    st.logger.info(f"✅ Triangulate_3d loaded for {split}/{seq} from: {path}  (frames={len(tri)})")
    return tri


# =============================================================================
# 17) Calibration DFS: find extri/intri yml/yaml under Calibration/** and load centers
# =============================================================================

_CALIB_CACHE = {}  # key: (split, seq) -> dict with extri_path/intri_path/centers/info/intri_data


def _calib_candidate_score(path: str, want: str) -> Tuple[int, int, int]:
    """
    want ∈ {"extri", "intri"}
    score tuple for sorting:
      (1) exact basename match (extri.yml / intri.yml) gets best
      (2) shorter path depth is preferred
      (3) alphabetical fallback
    """
    p = Path(path)
    base = p.name.lower()
    if want == "extri":
        exact = 0 if base in ("extri.yml", "extri.yaml") else 1
    else:
        exact = 0 if base in ("intri.yml", "intri.yaml") else 1
    depth = len(p.parts)
    return (exact, depth, 0)


def find_calibration_yml_paths(zf: zipfile.ZipFile, split: str, seq: str):
    """
    DFS search under:
      Data/<split>/<seq>/Calibration/
    to find any extri/intri yml/yaml files.
    Returns:
      extri_path, intri_path (either can be None)
    """
    calib_prefix = f"Data/{split}/{seq}/Calibration/"
    calib_prefix_low = calib_prefix.lower()

    extri_cands = []
    intri_cands = []

    for p in zf.namelist():
        low = p.lower()
        if not low.startswith(calib_prefix_low):
            continue
        base = Path(p).name.lower()

        if "extri" in base or base.startswith("rot_") or "extri" in low:
            extri_cands.append(p)
        if "intri" in base or "intri" in low or "intr" in base:
            intri_cands.append(p)

    extri = None
    intri = None

    if extri_cands:
        extri_cands.sort(key=lambda x: _calib_candidate_score(x, "extri"))
        extri = extri_cands[0]

    if intri_cands:
        intri_cands.sort(key=lambda x: _calib_candidate_score(x, "intri"))
        intri = intri_cands[0]

    return extri, intri


def _opencv_fs_read_matrix(fs, key):
    try:
        node = fs.getNode(key)
        if node.empty():
            return None
        mat = node.mat()
        if mat is None:
            return None
        return np.array(mat, dtype=np.float64)
    except Exception:
        return None


def _rig_plausibility_score(centers: Dict[str, np.ndarray]):
    cams = list(centers.keys())
    if len(cams) < 2:
        return 1e18, {"reason": "too_few_cams"}

    ds = []
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            d = float(np.linalg.norm(centers[cams[i]] - centers[cams[j]]))
            ds.append(d)

    if not ds:
        return 1e18, {"reason": "no_pairs"}

    ds = np.array(ds, dtype=np.float64)
    dmin = float(ds.min())
    dmax = float(ds.max())
    dmean = float(ds.mean())
    dstd = float(ds.std())

    collapse_pen = 1e6 if dmin < 1e-6 else 0.0
    huge_pen = 1e6 if dmax > 1e6 else 0.0
    flat_pen = 1e4 if dstd < 1e-6 else 0.0

    score = collapse_pen + huge_pen + flat_pen + (1.0 / max(dmean, 1e-6)) + (1.0 / max(dstd, 1e-6))
    stats = {"dmin": dmin, "dmax": dmax, "dmean": dmean, "dstd": dstd,
             "collapse_pen": collapse_pen, "huge_pen": huge_pen, "flat_pen": flat_pen}
    return float(score), stats


def load_camera_centers_from_extri_yml(zf: zipfile.ZipFile, split: str, seq: str):
    """
    Reads recursively found extri.yml/.yaml under:
      Data/<split>/<seq>/Calibration/**

    Returns:
      centers_map: {Cam1: np.array([x,y,z]), ...}
      info: dict
      extri_path: str|None
    """
    cache_key = (split, seq)
    if cache_key in _CALIB_CACHE and "centers" in _CALIB_CACHE[cache_key]:
        return _CALIB_CACHE[cache_key]["centers"], _CALIB_CACHE[cache_key]["extri_info"], _CALIB_CACHE[cache_key].get("extri_path")

    extri_path, intri_path = find_calibration_yml_paths(zf, split, seq)
    if cache_key not in _CALIB_CACHE:
        _CALIB_CACHE[cache_key] = {}
    _CALIB_CACHE[cache_key]["extri_path"] = extri_path
    _CALIB_CACHE[cache_key]["intri_path"] = intri_path

    if extri_path is None:
        info = {"mode": "yml_extri_not_found", "src": None}
        _CALIB_CACHE[cache_key]["centers"] = None
        _CALIB_CACHE[cache_key]["extri_info"] = info
        return None, info, None

    try:
        import cv2
    except Exception as e:
        info = {"mode": "cv2_missing", "src": extri_path, "err": repr(e)}
        _CALIB_CACHE[cache_key]["centers"] = None
        _CALIB_CACHE[cache_key]["extri_info"] = info
        return None, info, extri_path

    try:
        data = zf.read(extri_path)
    except Exception as e:
        info = {"mode": "yml_read_failed", "src": extri_path, "err": repr(e)}
        _CALIB_CACHE[cache_key]["centers"] = None
        _CALIB_CACHE[cache_key]["extri_info"] = info
        return None, info, extri_path

    with tempfile.NamedTemporaryFile(suffix=".yml", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()

        fs = cv2.FileStorage(tmp.name, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            info = {"mode": "filestorage_open_failed", "src": extri_path}
            _CALIB_CACHE[cache_key]["centers"] = None
            _CALIB_CACHE[cache_key]["extri_info"] = info
            return None, info, extri_path

        centers_world2cam = {}
        centers_cam2world = {}
        missing = []

        for cam in st.CANON_CAMS:
            R = _opencv_fs_read_matrix(fs, f"Rot_{cam}")
            t = _opencv_fs_read_matrix(fs, f"T_{cam}")

            if R is None or t is None:
                rvec = _opencv_fs_read_matrix(fs, f"R_{cam}")
                if rvec is not None and rvec.size == 3 and t is not None:
                    rvec = rvec.reshape(3, 1).astype(np.float64)
                    R, _ = cv2.Rodrigues(rvec)

            if R is None or t is None:
                missing.append(cam)
                continue

            R = np.array(R, dtype=np.float64).reshape(3, 3)
            t = np.array(t, dtype=np.float64).reshape(3)

            C_A = -R.T @ t
            centers_world2cam[cam] = C_A

            C_B = t.copy()
            centers_cam2world[cam] = C_B

        score_A, stats_A = _rig_plausibility_score(centers_world2cam)
        score_B, stats_B = _rig_plausibility_score(centers_cam2world)

        if score_A <= score_B:
            centers = centers_world2cam
            chosen_mode = "yml_extri_world_to_cam (C=-R^T t)"
            chosen_stats = stats_A
        else:
            centers = centers_cam2world
            chosen_mode = "yml_extri_cam_to_world (C=t)"
            chosen_stats = stats_B

        score_gap_ratio = min(score_A, score_B) / max(score_A, score_B) if max(score_A, score_B) > 0 else 0.0
        if score_gap_ratio > 0.85:
            st.logger.warning(
                f"[CALIB-DFS] split={split} seq={seq} plausibility ambiguous: "
                f"score_gap_ratio={score_gap_ratio:.3f} A_score={score_A:.4g} B_score={score_B:.4g}"
            )

        st.logger.info(
            f"[CALIB-DFS] split={split} seq={seq} extri={extri_path} chose={chosen_mode} "
            f"A_score={score_A:.4g} B_score={score_B:.4g} stats={chosen_stats}"
        )

        fs.release()

    if centers is None or len(centers) < 2:
        info = {"mode": "yml_extri_insufficient", "src": extri_path, "count": 0, "missing": missing}
        _CALIB_CACHE[cache_key]["centers"] = None
        _CALIB_CACHE[cache_key]["extri_info"] = info
        return None, info, extri_path

    info = {
        "mode": chosen_mode,
        "src": extri_path,
        "count": len(centers),
        "missing": missing[:],
        "plausibility": chosen_stats,
        "plausibility_gap_ratio": score_gap_ratio,
        "alt_candidate": {
            "A_world2cam_score": score_A,
            "B_cam2world_score": score_B,
            "A_world2cam_stats": stats_A,
            "B_cam2world_stats": stats_B,
        }
    }

    _CALIB_CACHE[cache_key]["centers"] = centers
    _CALIB_CACHE[cache_key]["extri_info"] = info
    return centers, info, extri_path


def load_intrinsics_from_intri_yml(zf: zipfile.ZipFile, split: str, seq: str):
    """
    Loads intrinsics (if present) for traceability.
    Typical fields may include K matrices and distortion per camera.
    Returns:
      intri_data: dict|None
      intri_info: dict
      intri_path: str|None
    """
    if not st.ARGS.load_intri:
        return None, {"mode": "disabled"}, None

    cache_key = (split, seq)
    if cache_key in _CALIB_CACHE and "intri_data" in _CALIB_CACHE[cache_key]:
        return _CALIB_CACHE[cache_key]["intri_data"], _CALIB_CACHE[cache_key]["intri_info"], _CALIB_CACHE[cache_key].get("intri_path")

    if cache_key not in _CALIB_CACHE or "intri_path" not in _CALIB_CACHE[cache_key]:
        _ = load_camera_centers_from_extri_yml(zf, split, seq)

    intri_path = _CALIB_CACHE.get(cache_key, {}).get("intri_path", None)
    if intri_path is None:
        info = {"mode": "yml_intri_not_found", "src": None}
        _CALIB_CACHE[cache_key]["intri_data"] = None
        _CALIB_CACHE[cache_key]["intri_info"] = info
        return None, info, None

    try:
        import cv2
    except Exception as e:
        info = {"mode": "cv2_missing", "src": intri_path, "err": repr(e)}
        _CALIB_CACHE[cache_key]["intri_data"] = None
        _CALIB_CACHE[cache_key]["intri_info"] = info
        return None, info, intri_path

    try:
        data = zf.read(intri_path)
    except Exception as e:
        info = {"mode": "yml_read_failed", "src": intri_path, "err": repr(e)}
        _CALIB_CACHE[cache_key]["intri_data"] = None
        _CALIB_CACHE[cache_key]["intri_info"] = info
        return None, info, intri_path

    intri_data = {"K": {}, "dist": {}}

    with tempfile.NamedTemporaryFile(suffix=".yml", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()

        fs = cv2.FileStorage(tmp.name, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            info = {"mode": "filestorage_open_failed", "src": intri_path}
            _CALIB_CACHE[cache_key]["intri_data"] = None
            _CALIB_CACHE[cache_key]["intri_info"] = info
            return None, info, intri_path

        for cam in st.CANON_CAMS:
            K = _opencv_fs_read_matrix(fs, f"K_{cam}")
            if K is None:
                K = _opencv_fs_read_matrix(fs, f"K_{cam.lower()}")
            if K is not None:
                K = np.array(K, dtype=np.float64).reshape(3, 3)
                intri_data["K"][cam] = K.tolist()

            d = _opencv_fs_read_matrix(fs, f"dist_{cam}")
            if d is None:
                d = _opencv_fs_read_matrix(fs, f"dist_{cam.lower()}")
            if d is not None:
                d = np.array(d, dtype=np.float64).reshape(-1)
                intri_data["dist"][cam] = d.tolist()

        fs.release()

    info = {
        "mode": "yml_intri_loaded",
        "src": intri_path,
        "count_K": len(intri_data["K"]),
        "count_dist": len(intri_data["dist"]),
    }

    _CALIB_CACHE[cache_key]["intri_data"] = intri_data
    _CALIB_CACHE[cache_key]["intri_info"] = info
    st.logger.info(f"[CALIB-DFS] split={split} seq={seq} intri={intri_path} K_count={info['count_K']} dist_count={info['count_dist']}")
    return intri_data, info, intri_path


def _extract_extrinsic_from_block(block):
    if not isinstance(block, dict):
        return None
    for k in ["extrinsic", "extrinsics", "T", "Rt", "rt", "matrix"]:
        if k in block:
            E = block[k]
            if isinstance(E, list) and len(E) == 16:
                E = np.array(E, dtype=np.float64).reshape(4, 4)
                return E
            if isinstance(E, list) and len(E) == 4 and all(isinstance(x, list) for x in E):
                E = np.array(E, dtype=np.float64)
                return E
    return None


def _estimate_centers_from_extrinsic_json(raw_json: dict):
    """
    Parse camera centers from JSON extrinsics (if any). We try to infer whether
    the extrinsics are world->cam or cam->world using a plausibility heuristic.
    """
    if not isinstance(raw_json, dict):
        return None, {"mode": "invalid_json"}

    extrinsics = {}
    for cam in st.CANON_CAMS:
        if cam in raw_json:
            blk = raw_json.get(cam, {})
            E = _extract_extrinsic_from_block(blk)
            if E is not None:
                extrinsics[cam] = E

    if not extrinsics:
        # Try to find a list of extrinsics
        if "extrinsics" in raw_json and isinstance(raw_json["extrinsics"], list):
            extrinsics = {}
            for i, blk in enumerate(raw_json["extrinsics"]):
                E = _extract_extrinsic_from_block(blk)
                if E is not None and i < len(st.CANON_CAMS):
                    extrinsics[st.CANON_CAMS[i]] = E

    if not extrinsics:
        return None, {"mode": "no_extrinsics"}

    centers_world2cam = {}
    centers_cam2world = {}
    for cam, E in extrinsics.items():
        E = np.array(E, dtype=np.float64)
        if E.shape != (4, 4):
            continue
        R = E[:3, :3]
        t = E[:3, 3]
        C_A = -R.T @ t
        centers_world2cam[cam] = C_A
        C_B = t.copy()
        centers_cam2world[cam] = C_B

    s1, _ = _rig_plausibility_score(centers_cam2world)
    s2, _ = _rig_plausibility_score(centers_world2cam)
    if s1 <= s2:
        return centers_cam2world, {"mode": "extrinsic_cam_to_world_assumed", "score": s1, "count": len(centers_cam2world)}
    return centers_world2cam, {"mode": "extrinsic_world_to_cam_assumed", "score": s2, "count": len(centers_world2cam)}


def find_camera_params_json(zf, split, seq):
    KEYS = ("calib", "calibration", "extri", "extr", "camera", "param", "pose")
    hits = []
    for p in zf.namelist():
        low = p.lower()
        if p.endswith(".json") and (split.lower() in low) and (seq.lower() in low) and any(k in low for k in KEYS):
            hits.append(p)
    if hits:
        hits.sort()
        return hits[0]
    return None


def load_camera_centers_map(zf, split, seq):
    centers, extri_info, extri_path = load_camera_centers_from_extri_yml(zf, split, seq)
    intri_data, intri_info, intri_path = load_intrinsics_from_intri_yml(zf, split, seq)

    if centers is not None:
        out_info = {
            "src": extri_path,
            "mode": extri_info.get("mode"),
            "extri_info": extri_info,
            "intri_path": intri_path,
            "intri_info": intri_info,
        }
        return centers, out_info, intri_data

    path = find_camera_params_json(zf, split, seq)
    if path is None:
        return None, {"src": None, "mode": "not_found_or_parse_failed", "extri_info": extri_info, "intri_info": intri_info}, intri_data

    try:
        data = zip_read_json(zf, path)
    except Exception:
        return None, {"src": path, "mode": "read_failed", "extri_info": extri_info, "intri_info": intri_info}, intri_data

    centers2, info2 = _estimate_centers_from_extrinsic_json(data)
    if centers2 is None:
        return None, {"src": path, "mode": "parse_failed", "extri_info": extri_info, "intri_info": intri_info}, intri_data

    info2["src"] = path
    info2["extri_info"] = extri_info
    info2["intri_path"] = intri_path
    info2["intri_info"] = intri_info
    return centers2, info2, intri_data


# =============================================================================
# 18) Task2: Relative rotation pseudo-GT
# =============================================================================

def azimuth_deg(v, plane="xz"):
    if plane == "xy":
        x, z = float(v[0]), float(v[1])
    elif plane == "yz":
        x, z = float(v[1]), float(v[2])
    else:
        x, z = float(v[0]), float(v[2])
    ang = math.degrees(math.atan2(z, x))
    return ang


def wrap_angle_deg(a):
    while a <= -180:
        a += 360
    while a > 180:
        a -= 360
    return a


def compute_relative_rotation_deg(cam_center_1, cam_center_2, person_center, plane="xz"):
    v1 = cam_center_1 - person_center
    v2 = cam_center_2 - person_center
    a1 = azimuth_deg(v1, plane=plane)
    a2 = azimuth_deg(v2, plane=plane)
    diff = wrap_angle_deg(a2 - a1)
    if abs(diff) < 1e-6:
        direction = "no rotation"
    else:
        direction = "counterclockwise" if diff > 0 else "clockwise"
    deg = abs(diff)
    return float(deg), direction, float(diff)


def _task2_axis_plane_diag(centers_map, person_center):
    cams = [c for c in st.CANON_CAMS if c in centers_map]
    if len(cams) < st.TASK2_AXIS_DIAG_MIN_CAMS:
        return {"mode": "insufficient_cams", "count": len(cams), "cams": cams}

    out = {"mode": "ok", "cams": cams, "planes": {}}
    n = len(cams)
    for plane in ("xz", "xy", "yz"):
        angles = []
        for c in cams:
            v = centers_map[c] - person_center
            angles.append(azimuth_deg(v, plane=plane))

        adj_diffs = []
        for i in range(n):
            a1 = angles[i]
            a2 = angles[(i + 1) % n]
            adj_diffs.append(abs(wrap_angle_deg(a2 - a1)))

        opp_diffs = []
        if n >= 4 and n % 2 == 0:
            for i in range(n // 2):
                a1 = angles[i]
                a2 = angles[i + n // 2]
                opp_diffs.append(abs(wrap_angle_deg(a2 - a1)))

        adj_mean = float(np.mean(adj_diffs)) if adj_diffs else None
        opp_mean = float(np.mean(opp_diffs)) if opp_diffs else None
        ratio = None
        if adj_mean is not None and opp_mean is not None:
            ratio = float(adj_mean / max(opp_mean, 1e-6))

        out["planes"][plane] = {
            "adjacent_mean_deg": adj_mean,
            "opposite_mean_deg": opp_mean,
            "adjacent_over_opposite": ratio,
        }

    return out


def qualitative_body_orientation_desc(image_path, cam_name, scene_type=None):
    prompt = prompts.prompt_task2_body_orientation(cam_name, scene_type=scene_type)
    raw = vlm_generate([image_path], prompt, max_new_tokens=40)
    return _first_two_sentences(raw) or "Body orientation unclear."


def build_task2_relative_camera_rotation(zf, split, seq, frame_id, cams, tri_map, task2_index):
    if tri_map is None or frame_id not in tri_map:
        st.REJECT_STATS["t2_no_tri"] += 1
        return None

    ent = tri_map[frame_id]
    eye3d = ent.get("eye3d", None)
    if eye3d is None or not isinstance(eye3d, np.ndarray) or eye3d.shape != (3,):
        st.REJECT_STATS["t2_no_tri"] += 1
        return None

    centers_map, cam_info, intri_data = load_camera_centers_map(zf, split, seq)

    if centers_map is None:
        st.REJECT_STATS["t2_no_cam_centers"] += 1
        return None

    dist_cams = [c for c in cams if c in centers_map]
    med_dist = None
    if dist_cams:
        dists = [float(np.linalg.norm(centers_map[c] - eye3d)) for c in dist_cams]
        med_dist = float(np.median(dists)) if dists else None

    st.TASK2_DIST_STATS["seen"] += 1
    if med_dist is not None:
        st.TASK2_DIST_STATS["med_dist_vals"].append(med_dist)

    if (med_dist is None) or (med_dist < st.TASK2_FRAME_DIST_MIN) or (med_dist > st.TASK2_FRAME_DIST_MAX):
        st.logger.warning(
            f"[Task2] frame consistency check failed split={split} seq={seq} frame={frame_id} "
            f"median_dist={med_dist}"
        )
        st.REJECT_STATS["t2_bad_median_dist"] += 1
        st.TASK2_DIST_STATS["rejected"] += 1
        if med_dist is not None:
            st.TASK2_DIST_STATS["med_dist_reject_vals"].append(med_dist)
        return None

    avail = []
    for c in cams:
        if c not in centers_map:
            continue
        from .io_utils import save_raw_cam_image
        p = save_raw_cam_image(zf, split, seq, c, frame_id)
        if p:
            avail.append((c, p))
    if len(avail) < 2:
        st.REJECT_STATS["t2_no_images"] += 1
        return None

    random.shuffle(avail)
    (cam1, p1), (cam2, p2) = avail[0], avail[1]

    C1 = centers_map[cam1]
    C2 = centers_map[cam2]
    P = eye3d

    deg, direction, signed = compute_relative_rotation_deg(C1, C2, P, plane=st.TASK2_AZIMUTH_PLANE)
    a1 = azimuth_deg(C1 - P, plane=st.TASK2_AZIMUTH_PLANE)
    a2 = azimuth_deg(C2 - P, plane=st.TASK2_AZIMUTH_PLANE)
    if not (-1e-3 <= deg <= 180.0 + 1e-3):
        st.logger.warning(
            f"[Task2] unexpected deg={deg:.3f} split={split} seq={seq} frame={frame_id} "
            f"cam1={cam1} cam2={cam2}"
        )
    axis_diag = _task2_axis_plane_diag(centers_map, P) if st.TASK2_AXIS_DIAG else None

    cam1_desc = qualitative_body_orientation_desc(p1, cam1, scene_type=split)
    cam2_desc = qualitative_body_orientation_desc(p2, cam2, scene_type=split)

    question = prompts.prompt_task2_question(cam1, cam2, scene_type=split)

    direction_label = "no rotation" if deg < 1e-6 else direction
    answer = (
        f"Camera 1 view: {cam1_desc}\n"
        f"Camera 2 view: {cam2_desc}\n"
        f"Relative rotation: ~{int(round(deg))}° {direction_label}"
    )

    same_pose = cam1_desc.strip().lower() == cam2_desc.strip().lower()
    if st.REASONING_MODE == "gt":
        reasoning = (
            f"Compute camera-center azimuths around the 3D eye point: "
            f"a1={a1:.2f} deg, a2={a2:.2f} deg. "
            f"diff=wrap(a2-a1)={signed:.2f} deg, so rotation is ~{int(round(deg))} deg {direction_label}. "
            "This is camera rotation around the subject, not a head-turn constraint."
        )
    else:
        if same_pose:
            pose_note = (
                "The body-facing descriptions are similar, so the numeric angle is derived from camera-center geometry "
                "around the 3D eye point rather than pose change."
            )
        else:
            pose_note = (
                "The body-facing descriptions provide a qualitative check, while the numeric angle is derived from "
                "camera-center geometry around the 3D eye point."
            )
        reasoning = (
            f"Camera 1 shows the person as: {cam1_desc}. "
            f"Camera 2 shows the person as: {cam2_desc}. "
            f"{pose_note} "
            f"Relative rotation is ~{int(round(deg))}° {direction_label}."
        )

    obj_id = make_id("t2", split, seq, frame_id, cam1, cam2, int(round(deg)), direction_label)

    if st.SAVE_DEBUG and (task2_index % 8 == 0):
        from .utils import log_debug
        log_debug({
            "task": 2,
            "split": split,
            "seq": seq,
            "frame_id": frame_id,
            "cam1": cam1,
            "cam2": cam2,
            "tri_src": ent.get("src"),
            "cam_params_src": cam_info.get("src"),
            "cam_parse_mode": cam_info.get("mode"),
            "intri_path": cam_info.get("intri_path"),
            "intri_info": cam_info.get("intri_info"),
            "azimuth_plane": st.TASK2_AZIMUTH_PLANE,
            "axis_plane_diag": axis_diag,
            "person_pivot_eye3d": eye3d.tolist(),
            "deg": deg,
            "direction_label": direction_label,
            "direction": direction,
            "signed_diff": signed
        })

    st.TASK2_DIST_STATS["accepted"] += 1
    return {
        "task_id": 2,
        "question": question,
        "answer": answer,
        "reasoning": reasoning,
        "meta": {
            "camera_pair": f"{cam1},{cam2}",
            "object_id": obj_id,
            "eye_err": ent.get("eye_err", None),
            "target_err": ent.get("target_err", None),
            "triangulate_src": ent.get("src"),
            "cam_params_src": cam_info.get("src"),
            "cam_params_parse_mode": cam_info.get("mode"),
            "intri_path": cam_info.get("intri_path"),
            "intri_info": cam_info.get("intri_info"),
            "intri_loaded": bool(intri_data is not None),
            "rotation_direction_label": direction_label,
            "azimuth_plane": st.TASK2_AZIMUTH_PLANE,
            "axis_plane_diag": axis_diag,
            "relative_rotation_deg": float(deg),
            "relative_rotation_direction": direction,
            "relative_rotation_signed_deg": float(signed),
            "azimuth_cam1_deg": float(a1),
            "azimuth_cam2_deg": float(a2),
        },
        "input_cams": [cam1, cam2],
        "input_images": [{"cam": cam1, "image": p1}, {"cam": cam2, "image": p2}],
        "scene": split.lower(),
        "timestamp": frame_id,
        "task_type": "relative_orientation_reasoning"
    }
