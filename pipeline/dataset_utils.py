import re

from . import state as st


def _frame_id_from_filename(name):
    if not name:
        return None
    base = name.split("/")[-1]
    stem = base.split(".")[0]
    if not stem:
        return None
    try:
        return int(str(stem).lstrip("0") or "0")
    except Exception:
        return None


def normalize_annotations(raw):
    if raw is None:
        return []

    frames = []
    if isinstance(raw, list):
        for v in raw:
            if not isinstance(v, dict):
                continue
            fid = v.get("frame_id", None)
            if fid is None:
                fid = _frame_id_from_filename(v.get("filename", None))
            if fid is None:
                continue
            try:
                fi = int(str(fid).lstrip("0") or "0")
            except Exception:
                continue
            frames.append({"frame_id": f"{fi:04d}", "frame_idx": fi, "data": v})
        frames.sort(key=lambda x: x["frame_idx"])
        return frames

    if isinstance(raw, dict):
        for k, v in raw.items():
            if k in ("frames", "data") and isinstance(v, dict):
                raw = v
                break

        for k, v in raw.items():
            if not isinstance(v, dict):
                continue
            try:
                fi = int(str(k).lstrip("0") or "0")
            except Exception:
                continue
            frames.append({"frame_id": f"{fi:04d}", "frame_idx": fi, "data": v})
        frames.sort(key=lambda x: x["frame_idx"])
        return frames

    return frames


def detect_cameras(frame_dict):
    cams = []
    for c in st.CANON_CAMS:
        if c in frame_dict:
            cams.append(c)
    return cams


def cam_anno(frame_data, cam):
    v = frame_data.get(cam, {})
    return v if isinstance(v, dict) else {}
