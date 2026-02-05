import numpy as np


def parse_visibility(a):
    v = a.get("visibility", None) if isinstance(a, dict) else None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return True if int(v) == 1 else (False if int(v) == 0 else None)
    if isinstance(v, str):
        t = v.strip().lower()
        if t in ("true", "t", "yes", "y", "1"):
            return True
        if t in ("false", "f", "no", "n", "0"):
            return False
    return None


def has_coord(a):
    c = a.get("coordinate", None) if isinstance(a, dict) else None
    return isinstance(c, (list, tuple)) and len(c) == 2


def get_body_bbox(a):
    bb = a.get("body", None)
    if isinstance(bb, (list, tuple)) and len(bb) == 4:
        return [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]
    return None


def get_head_bbox(a):
    hb = a.get("head", None)
    if isinstance(hb, (list, tuple)) and len(hb) == 4:
        return [float(hb[0]), float(hb[1]), float(hb[2]), float(hb[3])]
    return None


# =============================================================================
# Scale annotations after resize
# =============================================================================

def scale_xy(xy, sx, sy):
    return [float(xy[0]) * sx, float(xy[1]) * sy]


def scale_bbox_xywh(bb, sx, sy):
    x, y, w, h = bb
    return [float(x) * sx, float(y) * sy, float(w) * sx, float(h) * sy]


def scale_annotations_for_resized_image(anno: dict, orig_wh, new_wh):
    if not isinstance(anno, dict):
        return anno, None
    ow, oh = orig_wh
    nw, nh = new_wh
    if ow <= 0 or oh <= 0:
        return anno, None

    sx = float(nw) / float(ow)
    sy = float(nh) / float(oh)

    out = dict(anno)

    audit = {
        "orig_size": [int(ow), int(oh)],
        "resized_size": [int(nw), int(nh)],
        "gaze_point_orig": None,
        "gaze_point_resized": None,
    }

    if isinstance(out.get("coordinate", None), (list, tuple)) and len(out["coordinate"]) == 2:
        audit["gaze_point_orig"] = [float(out["coordinate"][0]), float(out["coordinate"][1])]
        out["coordinate"] = scale_xy(out["coordinate"], sx, sy)
        # clamp to resized bounds
        out["coordinate"][0] = max(0.0, min(float(out["coordinate"][0]), float(nw - 1)))
        out["coordinate"][1] = max(0.0, min(float(out["coordinate"][1]), float(nh - 1)))
        audit["gaze_point_resized"] = [float(out["coordinate"][0]), float(out["coordinate"][1])]

    eyes = out.get("eye", None)
    if isinstance(eyes, list) and eyes:
        new_eyes = []
        for e in eyes:
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                new_eyes.append(scale_xy(e[:2], sx, sy))
        if new_eyes:
            out["eye"] = new_eyes

    if isinstance(out.get("body", None), (list, tuple)) and len(out["body"]) == 4:
        out["body"] = scale_bbox_xywh(out["body"], sx, sy)
        x, y, w, h = [float(v) for v in out["body"]]
        x = max(0.0, min(x, float(nw - 1)))
        y = max(0.0, min(y, float(nh - 1)))
        w = max(0.0, min(w, float(nw - 1) - x))
        h = max(0.0, min(h, float(nh - 1) - y))
        out["body"] = [x, y, w, h]

    if isinstance(out.get("head", None), (list, tuple)) and len(out["head"]) == 4:
        out["head"] = scale_bbox_xywh(out["head"], sx, sy)
        x, y, w, h = [float(v) for v in out["head"]]
        x = max(0.0, min(x, float(nw - 1)))
        y = max(0.0, min(y, float(nh - 1)))
        w = max(0.0, min(w, float(nw - 1) - x))
        h = max(0.0, min(h, float(nh - 1) - y))
        out["head"] = [x, y, w, h]

    return out, audit
