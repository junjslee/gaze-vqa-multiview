# sam2_utils.py
import os
import sys
import importlib
import re
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

from . import state as st
from .io_utils import _run, zip_try_image_path, zip_read_image

_sam2_predictor = None


def resolve_sam2_cfg(repo_dir: str, basename: str):
    repo = Path(repo_dir).resolve()
    pkg_root = repo / "sam2"
    candidates = [
        repo / "configs" / "sam2" / basename,
        pkg_root / "configs" / "sam2" / basename,
        repo / "configs" / "sam2" / basename.replace(".yaml", ".yml"),
        pkg_root / "configs" / "sam2" / basename.replace(".yaml", ".yml"),
    ]
    for c in candidates:
        if c.exists():
            try:
                rel = c.relative_to(pkg_root)
                return str(c), str(rel).replace("\\", "/")
            except Exception:
                rel = c.relative_to(repo)
                return str(c), str(rel).replace("\\", "/")

    hits = list(repo.rglob(basename))
    if not hits:
        hits = list(repo.rglob("sam2_hiera*.yaml")) + list(repo.rglob("sam2_hiera*.yml"))
    if not hits:
        raise FileNotFoundError(f"SAM2 config '{basename}' not found in {repo}")

    best = hits[0]
    try:
        rel = best.relative_to(pkg_root)
        return str(best), str(rel).replace("\\", "/")
    except Exception:
        rel = best.relative_to(repo)
        return str(best), str(rel).replace("\\", "/")


def ensure_sam2_repo():
    repo = Path(st.SAM2_REPO_DIR)
    repo.parent.mkdir(parents=True, exist_ok=True)
    if repo.exists() and (repo / "sam2").exists():
        st.logger.info(f"✅ SAM2 repo exists: {repo}")
        return
    if repo.exists():
        st.logger.warning(f"⚠️ SAM2 repo path exists but wrong. Removing: {repo}")
        _run(["bash", "-lc", f"rm -rf '{str(repo)}'"])
    st.logger.info(f"⬇️ Cloning SAM2 repo into: {repo}")
    _run(["bash", "-lc", f"git clone -q https://github.com/facebookresearch/sam2.git '{str(repo)}'"])
    st.logger.info("✅ SAM2 repo cloned.")


def ensure_sam2_checkpoint():
    ckpt = Path(st.SAM2_CHECKPOINT)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if ckpt.exists():
        st.logger.info(f"✅ SAM2 checkpoint exists: {ckpt}")
        return

    st.logger.info(f"⬇️ Downloading SAM2 checkpoint to: {ckpt}")
    _run(["bash", "-lc", "python3 -m pip -q install 'huggingface_hub>=0.34.0,<1.0'"], check=True)

    py = f"""
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id="facebook/sam2-hiera-large",
                    filename="sam2_hiera_large.pt",
                    local_dir="{str(ckpt.parent)}")
print("Downloaded to:", p)
"""
    _run(["bash", "-lc", f"python3 - << 'PY'\n{py}\nPY"])

    downloaded = ckpt.parent / "sam2_hiera_large.pt"
    if downloaded.exists() and str(downloaded) != str(ckpt):
        downloaded.rename(ckpt)

    if not ckpt.exists():
        raise FileNotFoundError(f"SAM2 checkpoint missing: {ckpt}")
    st.logger.info(f"✅ SAM2 checkpoint ready: {ckpt}")


def force_import_sam2_from_repo():
    repo = str(Path(st.SAM2_REPO_DIR).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    for k in list(sys.modules.keys()):
        if k == "sam2" or k.startswith("sam2."):
            del sys.modules[k]
    import sam2
    st.logger.info(f"sam2 imported from: {getattr(sam2, '__file__', '<?>')}")
    return sam2


def _ensure_torchvision_compat():
    # Avoid torchvision import crashes when torch/torchvision are mismatched in the container.
    try:
        import torch
    except Exception:
        return

    lib = getattr(torch, "library", None)
    if lib is not None and not hasattr(lib, "register_fake"):
        def _register_fake(*args, **kwargs):
            def _decorator(fn):
                return fn
            return _decorator
        lib.register_fake = _register_fake

    try:
        import torchvision  # noqa: F401
        return
    except Exception as e:
        st.logger.warning(f"torchvision import failed; attempting compat install: {e}")

    # Attempt to install a torchvision version that matches the local torch.
    try:
        ver = getattr(torch, "__version__", "")
        m = re.match(r"(\\d+)\\.(\\d+)", ver)
        major, minor = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        tv_map = {
            (2, 5): "0.20.1",
            (2, 4): "0.19.1",
            (2, 3): "0.18.1",
            (2, 2): "0.17.2",
            (2, 1): "0.16.2",
            (2, 0): "0.15.2",
        }
        tv_ver = tv_map.get((major, minor))
        if not tv_ver:
            st.logger.warning(f"No torchvision pin for torch {ver}; skipping auto-fix.")
            return

        cuda = getattr(torch.version, "cuda", None)
        if cuda:
            cu = cuda.replace(".", "")
            index = f"--index-url https://download.pytorch.org/whl/cu{cu}"
        else:
            index = "--index-url https://download.pytorch.org/whl/cpu"

        st.logger.info(f"⬇️ Installing torchvision=={tv_ver} ({index})")
        _run(["bash", "-lc", f"python3 -m pip -q install --no-deps torchvision=={tv_ver} {index}"], check=False)
        importlib.invalidate_caches()
        import torchvision  # noqa: F401
    except Exception as e:
        st.logger.warning(f"torchvision compat install failed: {e}")


def ensure_sam2():
    global _sam2_predictor
    if _sam2_predictor is not None:
        return _sam2_predictor

    _run(["bash", "-lc", "python3 -m pip -q install hydra-core omegaconf iopath fvcore opencv-python"], check=True)

    import torch
    _ensure_torchvision_compat()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ensure_sam2_repo()
    ensure_sam2_checkpoint()
    force_import_sam2_from_repo()

    cfg_abs, cfg_hydra = resolve_sam2_cfg(st.SAM2_REPO_DIR, st.SAM2_CFG_BASENAME)
    if cfg_hydra.startswith("sam2/"):
        cfg_hydra = cfg_hydra[len("sam2/"):]
    cfg_hydra = cfg_hydra.replace("\\", "/")

    st.logger.info("✅ SAM2 config resolved:")
    st.logger.info(f"   abs  : {cfg_abs}")
    st.logger.info(f"   hydra: {cfg_hydra}")

    importlib.import_module("sam2.modeling.backbones.hieradet")

    old = os.getcwd()
    os.chdir(st.SAM2_REPO_DIR)
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam2_model = build_sam2(cfg_hydra, st.SAM2_CHECKPOINT, device=device)
        _sam2_predictor = SAM2ImagePredictor(sam2_model)
    finally:
        os.chdir(old)

    st.logger.info("✅ SAM2 ready.")
    return _sam2_predictor


# =============================================================================
# Mask helpers
# =============================================================================

def point_to_box(point_xy, box_size=120):
    x, y = point_xy
    half = box_size / 2.0
    return [x - half, y - half, x + half, y + half]


def clip_box(box, W, H):
    x1, y1, x2, y2 = box
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    return [x1, y1, x2, y2]


def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def dilate_mask(mask_u8, iterations=1):
    try:
        import cv2
        k = np.ones((3, 3), np.uint8)
        m = (mask_u8 > 0).astype(np.uint8) * 255
        m = cv2.dilate(m, k, iterations=iterations)
        return (m > 0).astype(np.uint8)
    except Exception:
        return (mask_u8 > 0).astype(np.uint8)


def mask_person_overlap_ratio(mask_u8, body_bbox_xywh):
    if mask_u8 is None or body_bbox_xywh is None:
        return 0.0
    x, y, w, h = body_bbox_xywh
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    H, W = mask_u8.shape[:2]
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    # wrong
    # region = mask_u8[y1:y2 + 1, x1:x2 + 1]
    # return float((region > 0).mean())
    inter = mask_u8[y1:y2+1, x1:x2+1].sum()
    den = mask_u8.sum() + 1e-6
    return float(inter / den)


def overlay_mask_on_image(img_pil, mask_u8, alpha=0.45):
    img = np.array(img_pil).astype(np.float32)
    mask = (mask_u8 > 0).astype(np.float32)
    overlay = img.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + 255 * mask, 0, 255)
    out = img * (1 - alpha * mask[..., None]) + overlay * (alpha * mask[..., None])
    return Image.fromarray(out.astype(np.uint8))


def overlay_soft_mask_on_image(img_pil, mask_f, alpha=0.45):
    img = np.array(img_pil).astype(np.float32)
    mask = np.clip(mask_f, 0.0, 1.0).astype(np.float32)
    overlay = img.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + 255 * mask, 0, 255)
    out = img * (1 - alpha * mask[..., None]) + overlay * (alpha * mask[..., None])
    return Image.fromarray(out.astype(np.uint8))


def preprocess_segmentation_image_np(img_np):
    """
    Optional Task1 preprocessing before SAM2 embedding.
    Supports:
    - LAB-L CLAHE
    - LAB-L bilateral denoising
    - LAB L/b edge boost blended into L
    Keeps geometry unchanged and returns RGB uint8 (or grayscale for grayscale input).
    """
    if img_np is None:
        return img_np
    use_clahe = bool(getattr(st, "TASK1_SEG_USE_CLAHE", False))
    use_bilateral = bool(getattr(st, "TASK1_SEG_USE_BILATERAL", False))
    use_edge_boost = bool(getattr(st, "TASK1_SEG_USE_LB_EDGE_BOOST", False))
    if not (use_clahe or use_bilateral or use_edge_boost):
        return img_np
    try:
        import cv2
        clip = float(getattr(st, "TASK1_SEG_CLAHE_CLIP", 2.0))
        tile = max(2, int(getattr(st, "TASK1_SEG_CLAHE_TILE", 8)))
        bilateral_d = max(1, int(getattr(st, "TASK1_SEG_BILATERAL_D", 9)))
        if bilateral_d % 2 == 0:
            bilateral_d += 1
        bilateral_sigma_color = max(1e-6, float(getattr(st, "TASK1_SEG_BILATERAL_SIGMA_COLOR", 75.0)))
        bilateral_sigma_space = max(1e-6, float(getattr(st, "TASK1_SEG_BILATERAL_SIGMA_SPACE", 75.0)))

        l_low = int(getattr(st, "TASK1_SEG_EDGE_L_LOW", 30))
        l_high = int(getattr(st, "TASK1_SEG_EDGE_L_HIGH", 100))
        b_low = int(getattr(st, "TASK1_SEG_EDGE_B_LOW", 20))
        b_high = int(getattr(st, "TASK1_SEG_EDGE_B_HIGH", 50))
        l_low = max(0, min(255, l_low))
        l_high = max(0, min(255, l_high))
        b_low = max(0, min(255, b_low))
        b_high = max(0, min(255, b_high))
        if l_high < l_low:
            l_low, l_high = l_high, l_low
        if b_high < b_low:
            b_low, b_high = b_high, b_low
        edge_alpha = float(getattr(st, "TASK1_SEG_EDGE_BOOST_ALPHA", 0.2))
        edge_alpha = max(0.0, min(1.0, edge_alpha))
        edge_dilate_iter = max(0, int(getattr(st, "TASK1_SEG_EDGE_DILATE_ITER", 1)))

        arr = np.asarray(img_np, dtype=np.uint8)
        if arr.ndim == 2:
            gray = arr
            if use_clahe:
                clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
                gray = clahe.apply(gray)
            if use_bilateral:
                gray = cv2.bilateralFilter(gray, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
            if use_edge_boost:
                edges = cv2.Canny(gray, l_low, l_high)
                if edge_dilate_iter > 0:
                    kernel = np.ones((3, 3), dtype=np.uint8)
                    edges = cv2.dilate(edges, kernel, iterations=edge_dilate_iter)
                if edge_alpha > 0.0:
                    gray = cv2.addWeighted(gray, 1.0, edges, edge_alpha, 0.0)
            return gray.astype(np.uint8)

        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
            l = clahe.apply(l)

        if use_bilateral:
            l = cv2.bilateralFilter(l, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)

        if use_edge_boost:
            edges_l = cv2.Canny(l, l_low, l_high)
            edges_b = cv2.Canny(b, b_low, b_high)
            edges = cv2.bitwise_or(edges_l, edges_b)
            if edge_dilate_iter > 0:
                kernel = np.ones((3, 3), dtype=np.uint8)
                edges = cv2.dilate(edges, kernel, iterations=edge_dilate_iter)
            if edge_alpha > 0.0:
                l = cv2.addWeighted(l, 1.0, edges, edge_alpha, 0.0)

        lab2 = cv2.merge((l, a, b))
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
        return out.astype(np.uint8)
    except Exception:
        return img_np


def overlay_mask_on_image_neutral(img_pil, mask_u8, dim_outside=0.55):
    img = np.array(img_pil).astype(np.float32)
    mask = (mask_u8 > 0).astype(np.float32)
    dim_outside = float(np.clip(dim_outside, 0.0, 1.0))
    out = img * (dim_outside + (1.0 - dim_outside) * mask[..., None])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def overlay_soft_mask_on_image_neutral(img_pil, mask_f, dim_outside=0.55):
    img = np.array(img_pil).astype(np.float32)
    mask = np.clip(mask_f, 0.0, 1.0).astype(np.float32)
    dim_outside = float(np.clip(dim_outside, 0.0, 1.0))
    out = img * (dim_outside + (1.0 - dim_outside) * mask[..., None])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _resize_soft_mask(mask_f, W, H):
    if mask_f is None:
        return None
    if mask_f.shape[0] == H and mask_f.shape[1] == W:
        return mask_f
    try:
        import cv2
        return cv2.resize(mask_f, (W, H), interpolation=cv2.INTER_LINEAR)
    except Exception:
        pil = Image.fromarray((np.clip(mask_f, 0.0, 1.0) * 255).astype(np.uint8))
        pil = pil.resize((W, H), resample=Image.BILINEAR)
        return np.array(pil).astype(np.float32) / 255.0


# def _mask_keep_component_at_point(mask_u8, point_xy):
#     if mask_u8 is None:
#         return None
#     H, W = mask_u8.shape[:2]
#     x = int(round(point_xy[0]))
#     y = int(round(point_xy[1]))
#     if x < 0 or y < 0 or x >= W or y >= H:
#         return mask_u8
#     if mask_u8[y, x] == 0:
#         return mask_u8

#     try:
#         import cv2
#         num, labels = cv2.connectedComponents(mask_u8, connectivity=4)
#         if num <= 1:
#             return mask_u8
#         comp_id = labels[y, x]
#         return (labels == comp_id).astype(np.uint8)
#     except Exception:
#         visited = np.zeros((H, W), dtype=np.uint8)
#         out = np.zeros((H, W), dtype=np.uint8)
#         stack = [(y, x)]
#         visited[y, x] = 1
#         while stack:
#             cy, cx = stack.pop()
#             out[cy, cx] = 1
#             for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
#                 if 0 <= ny < H and 0 <= nx < W and visited[ny, nx] == 0 and mask_u8[ny, nx]:
#                     visited[ny, nx] = 1
#                     stack.append((ny, nx))
#         return out
def _mask_keep_component_at_point(mask_u8, point_xy):
    if mask_u8 is None:
        return None
    H, W = mask_u8.shape[:2]
    x = int(round(point_xy[0]))
    y = int(round(point_xy[1]))

    # If gaze is out of bounds -> reject
    if x < 0 or y < 0 or x >= W or y >= H:
        return np.zeros_like(mask_u8, dtype=np.uint8)

    # If gaze is not inside mask -> reject
    if mask_u8[y, x] == 0:
        return np.zeros_like(mask_u8, dtype=np.uint8)

    try:
        import cv2
        mask_u8 = (mask_u8 > 0).astype(np.uint8)
        num, labels = cv2.connectedComponents(mask_u8, connectivity=4)
        comp_id = labels[y, x]
        return (labels == comp_id).astype(np.uint8)
    except Exception:
        # fallback flood fill
        visited = np.zeros((H, W), dtype=np.uint8)
        out = np.zeros((H, W), dtype=np.uint8)
        stack = [(y, x)]
        visited[y, x] = 1
        while stack:
            cy, cx = stack.pop()
            out[cy, cx] = 1
            for ny, nx in ((cy-1,cx), (cy+1,cx), (cy,cx-1), (cy,cx+1)):
                if 0 <= ny < H and 0 <= nx < W and visited[ny, nx] == 0 and mask_u8[ny, nx]:
                    visited[ny, nx] = 1
                    stack.append((ny, nx))
        return out


def _mean_conf_around_point(soft_mask, point_xy, radius):
    if soft_mask is None:
        return None
    H, W = soft_mask.shape[:2]
    x = int(round(point_xy[0]))
    y = int(round(point_xy[1]))
    if x < 0 or y < 0 or x >= W or y >= H:
        return None
    r = int(max(1, radius))
    x1 = max(0, x - r)
    x2 = min(W - 1, x + r)
    y1 = max(0, y - r)
    y2 = min(H - 1, y + r)
    region = soft_mask[y1:y2 + 1, x1:x2 + 1]
    return float(np.mean(region)) if region.size else None


def _crop_overlay_from_mask(img_pil, mask_u8, soft_mask, bb, alpha=0.45, neutral=False, use_soft_mask=True):
    if img_pil is None or bb is None:
        return None
    x1, y1, x2, y2 = bb
    crop = img_pil.crop((x1, y1, x2 + 1, y2 + 1))
    if neutral:
        dim_outside = max(0.05, 1.0 - float(alpha))
        if use_soft_mask and soft_mask is not None:
            mask_crop = soft_mask[y1:y2 + 1, x1:x2 + 1]
            return overlay_soft_mask_on_image_neutral(crop, mask_crop, dim_outside=dim_outside)
        if mask_u8 is None:
            return crop
        mask_crop = (mask_u8[y1:y2 + 1, x1:x2 + 1] > 0).astype(np.uint8)
        return overlay_mask_on_image_neutral(crop, mask_crop, dim_outside=dim_outside)
    if use_soft_mask and soft_mask is not None:
        mask_crop = soft_mask[y1:y2 + 1, x1:x2 + 1]
        return overlay_soft_mask_on_image(crop, mask_crop, alpha=alpha)
    if mask_u8 is None:
        return crop
    mask_crop = (mask_u8[y1:y2 + 1, x1:x2 + 1] > 0).astype(np.uint8)
    return overlay_mask_on_image(crop, mask_crop, alpha=alpha)


def _crop_soft_masked(img_pil, soft_mask, bb):
    if img_pil is None or soft_mask is None or bb is None:
        return None
    x1, y1, x2, y2 = bb
    crop = np.array(img_pil)[y1:y2 + 1, x1:x2 + 1]
    mask_crop = np.clip(soft_mask[y1:y2 + 1, x1:x2 + 1], 0.0, 1.0)
    out = (crop * mask_crop[..., None]).astype(np.uint8)
    return Image.fromarray(out)


def _crop_masked(img_np, mask_u8, pad=16, pad_ratio=0.0, pad_max=None):
    H, W = mask_u8.shape[:2]
    bb = mask_to_bbox(mask_u8)
    if bb is None:
        return None, None
    x1, y1, x2, y2 = bb
    if pad_ratio and pad_ratio > 0:
        max_dim = max(1, int(x2 - x1 + 1), int(y2 - y1 + 1))
        dyn_pad = int(round(max_dim * float(pad_ratio)))
        pad = max(int(pad), dyn_pad)
    if pad_max is not None:
        pad = min(int(pad), int(pad_max))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(W - 1, x2 + pad)
    y2 = min(H - 1, y2 + pad)

    crop = img_np[y1:y2 + 1, x1:x2 + 1]
    crop_mask = mask_u8[y1:y2 + 1, x1:x2 + 1]
    masked = crop.copy()
    masked[crop_mask == 0] = 0
    return Image.fromarray(masked), (x1, y1, x2, y2)


# def draw_dot_on_crop(crop_pil, point_xy_scaled, crop_bb, alpha=0.5):
#     if crop_pil is None or crop_bb is None or point_xy_scaled is None:
#         return crop_pil
#     x1, y1, x2, y2 = crop_bb
#     gx, gy = float(point_xy_scaled[0]), float(point_xy_scaled[1])
#     gx = np.clip(gx, 0, W-1)
#     gy = np.clip(gy, 0, H-1)
#     cx, cy = gx - x1, gy - y1
#     W, H = crop_pil.size
#     if cx < 0 or cy < 0 or cx >= W or cy >= H:
#         return crop_pil

#     base = crop_pil.convert("RGBA")
#     overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
#     draw = ImageDraw.Draw(overlay)
#     r = max(2, int(st.GAZE_DOT_R * 0.75))
#     color = (st.GAZE_COLOR[0], st.GAZE_COLOR[1], st.GAZE_COLOR[2], int(255 * alpha))
#     draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
#     return Image.alpha_composite(base, overlay).convert("RGB")
def draw_dot_on_crop(crop_pil, point_xy_scaled, crop_bb, alpha=0.5, full_wh=None, color=None):
    """
    crop_pil: cropped object image (already cropped)
    point_xy_scaled: gaze point in FULL resized-image coordinates
    crop_bb: (x1,y1,x2,y2) in FULL resized-image coordinates
    full_wh: (W_full, H_full) optional, used for clipping gaze point
    """
    if crop_pil is None or crop_bb is None or point_xy_scaled is None:
        return crop_pil

    x1, y1, x2, y2 = crop_bb
    gx, gy = float(point_xy_scaled[0]), float(point_xy_scaled[1])

    if full_wh is not None:
        W_full, H_full = full_wh
        gx = np.clip(gx, 0, W_full - 1)
        gy = np.clip(gy, 0, H_full - 1)

    # gaze point relative to crop
    cx, cy = gx - x1, gy - y1

    Wc, Hc = crop_pil.size
    if cx < 0 or cy < 0 or cx >= Wc or cy >= Hc:
        return crop_pil

    base = crop_pil.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    r = max(2, int(st.GAZE_DOT_R * 0.75))
    if color is None:
        color = st.GAZE_COLOR
    color = (color[0], color[1], color[2], int(255 * alpha))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    return Image.alpha_composite(base, overlay).convert("RGB")


def segment_object_on_crop(crop_pil, point_xy, cfg=None):
    """
    Run SAM2 on a cropped image with a point in crop coordinates.
    Returns masked_crop, mask_u8, bb, soft_mask.
    """
    if crop_pil is None or point_xy is None:
        return None, None, None, None
    predictor = ensure_sam2()
    img_np = np.array(crop_pil)
    H, W = img_np.shape[0], img_np.shape[1]
    if H <= 1 or W <= 1:
        return None, None, None, None

    gx, gy = float(point_xy[0]), float(point_xy[1])
    if gx < 0 or gy < 0 or gx >= W or gy >= H:
        return None, None, None, None

    predictor.set_image(preprocess_segmentation_image_np(img_np))

    use_tight_box = _cfg_get(cfg, "use_tight_box", st.TASK1_USE_TIGHT_BOX)
    point_box_size = _cfg_get(cfg, "point_box_size", st.TASK1_POINT_BOX_SIZE)
    pad_around_mask = _cfg_get(cfg, "pad_around_mask", st.TASK1_PAD_AROUND_MASK)
    pad_ratio = _cfg_get(cfg, "pad_around_mask_ratio", st.TASK1_PAD_AROUND_MASK_RATIO)
    pad_max = _cfg_get(cfg, "pad_around_mask_max", st.TASK1_PAD_AROUND_MASK_MAX)
    dilate_mask_on = _cfg_get(cfg, "dilate_mask", st.TASK1_DILATE_MASK)
    dilate_iter = _cfg_get(cfg, "dilate_iter", st.TASK1_DILATE_ITER)
    mask_min_area_ratio = _cfg_get(cfg, "mask_min_area_ratio", st.TASK1_MASK_MIN_AREA_RATIO)
    mask_max_area_ratio = _cfg_get(cfg, "mask_max_area_ratio", st.TASK1_MASK_MAX_AREA_RATIO)
    gaze_conf_radius = _cfg_get(cfg, "gaze_conf_radius", st.TASK1_GAZE_CONF_RADIUS)
    min_soft_conf = _cfg_get(cfg, "min_soft_conf_around_gaze", st.TASK1_MIN_SOFT_CONF_AROUND_GAZE)
    soft_mask_threshold = _cfg_get(cfg, "soft_mask_threshold", st.TASK1_SOFT_MASK_THRESHOLD)
    allow_box_fallback = _cfg_get(cfg, "allow_box_fallback", True)

    if use_tight_box:
        box_size = point_box_size
    else:
        box_size = max(point_box_size, 220)

    box = clip_box(point_to_box((gx, gy), box_size=box_size), W, H)
    point_coords = np.array([[gx, gy]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)
    input_box = np.array(box, dtype=np.float32)[None, :]

    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=input_box,
        multimask_output=True,
    )

    if masks is None or len(masks) == 0:
        return None, None, None, None

    soft_masks = None
    try:
        if logits is not None and len(logits) == len(masks):
            soft_masks = _sigmoid(np.array(logits, dtype=np.float32))
    except Exception:
        soft_masks = None

    order = list(np.argsort(-np.array(scores)))
    for idx in order[:3]:
        m = masks[int(idx)].astype(np.uint8)
        if soft_masks is not None:
            soft_m = _resize_soft_mask(soft_masks[int(idx)], W, H)
        else:
            soft_m = None

        if soft_m is not None and soft_mask_threshold > 0:
            m_soft = (soft_m >= soft_mask_threshold).astype(np.uint8)
            m_soft = _mask_keep_component_at_point(m_soft, (gx, gy))
            if m_soft.sum() > 0:
                area_ratio_soft = float(m_soft.sum()) / float(max(1, H * W))
                if mask_min_area_ratio <= area_ratio_soft <= mask_max_area_ratio:
                    m = m_soft

        if dilate_mask_on:
            m = dilate_mask(m, iterations=dilate_iter)

        m = _mask_keep_component_at_point(m, (gx, gy))
        if soft_m is not None:
            soft_m = soft_m * m.astype(np.float32)

        area_ratio = float(m.sum()) / float(max(1, H * W))
        if area_ratio < mask_min_area_ratio or area_ratio > mask_max_area_ratio:
            continue

        if soft_m is not None and min_soft_conf > 0:
            conf = _mean_conf_around_point(soft_m, (gx, gy), gaze_conf_radius)
            if conf is not None and conf < min_soft_conf:
                continue

        masked_crop, bb = _crop_masked(
            img_np, m,
            pad=pad_around_mask,
            pad_ratio=pad_ratio,
            pad_max=pad_max,
        )
        if masked_crop is None or bb is None:
            continue
        return masked_crop, m, bb, soft_m

    if not allow_box_fallback:
        return None, None, None, None

    masks2, scores2, logits2 = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=None,
        multimask_output=True,
    )
    if masks2 is None or len(masks2) == 0:
        return None, None, None, None

    soft_masks2 = None
    try:
        if logits2 is not None and len(logits2) == len(masks2):
            soft_masks2 = _sigmoid(np.array(logits2, dtype=np.float32))
    except Exception:
        soft_masks2 = None

    order2 = list(np.argsort(-np.array(scores2)))
    for idx in order2[:3]:
        m = masks2[int(idx)].astype(np.uint8)
        if soft_masks2 is not None:
            soft_m = _resize_soft_mask(soft_masks2[int(idx)], W, H)
        else:
            soft_m = None

        if soft_m is not None and soft_mask_threshold > 0:
            m_soft = (soft_m >= soft_mask_threshold).astype(np.uint8)
            m_soft = _mask_keep_component_at_point(m_soft, (gx, gy))
            if m_soft.sum() > 0:
                area_ratio_soft = float(m_soft.sum()) / float(max(1, H * W))
                if mask_min_area_ratio <= area_ratio_soft <= mask_max_area_ratio:
                    m = m_soft

        m = _mask_keep_component_at_point(m, (gx, gy))
        if soft_m is not None:
            soft_m = soft_m * m.astype(np.float32)

        area_ratio = float(m.sum()) / float(max(1, H * W))
        if area_ratio < mask_min_area_ratio or area_ratio > mask_max_area_ratio:
            continue

        if soft_m is not None and min_soft_conf > 0:
            conf = _mean_conf_around_point(soft_m, (gx, gy), gaze_conf_radius)
            if conf is not None and conf < min_soft_conf:
                continue

        masked_crop, bb = _crop_masked(
            img_np, m,
            pad=pad_around_mask,
            pad_ratio=pad_ratio,
            pad_max=pad_max,
        )
        if masked_crop is None or bb is None:
            continue
        return masked_crop, m, bb, soft_m

    return None, None, None, None


def _cfg_get(cfg, key, default):
    if not cfg:
        return default
    return cfg.get(key, default)


def segment_object_at_gaze_precomputed(
    predictor,
    img_pil_resized,
    img_np,
    point_xy_scaled,
    body_bbox_xywh_scaled=None,
    cfg=None,
    update_reject_stats=False,
):
    """
    Like `segment_object_at_gaze`, but assumes the caller already:
      1) resized/loaded the image into `img_pil_resized` / `img_np`
      2) called `predictor.set_image(img_np)` for this image

    This is useful when sampling many points on the same image (e.g., Task4 distractors)
    without recomputing SAM2 image embeddings repeatedly.
    """
    debug = {
        "last_reject_reason": None,
        "overlap_reject_mask": None,
        "overlap_reject_bb": None,
        "overlap_reject_soft": None,
    }

    if predictor is None or img_np is None or point_xy_scaled is None:
        debug["last_reject_reason"] = "bad_input"
        return None, None, None, img_pil_resized, None, debug

    H, W = img_np.shape[0], img_np.shape[1]
    if H <= 1 or W <= 1:
        debug["last_reject_reason"] = "bad_image"
        return None, None, None, img_pil_resized, None, debug

    gx, gy = float(point_xy_scaled[0]), float(point_xy_scaled[1])
    use_tight_box = _cfg_get(cfg, "use_tight_box", st.TASK1_USE_TIGHT_BOX)
    point_box_size = _cfg_get(cfg, "point_box_size", st.TASK1_POINT_BOX_SIZE)
    pad_around_mask = _cfg_get(cfg, "pad_around_mask", st.TASK1_PAD_AROUND_MASK)
    dilate_mask_on = _cfg_get(cfg, "dilate_mask", st.TASK1_DILATE_MASK)
    dilate_iter = _cfg_get(cfg, "dilate_iter", st.TASK1_DILATE_ITER)
    mask_min_area_ratio = _cfg_get(cfg, "mask_min_area_ratio", st.TASK1_MASK_MIN_AREA_RATIO)
    mask_max_area_ratio = _cfg_get(cfg, "mask_max_area_ratio", st.TASK1_MASK_MAX_AREA_RATIO)
    gaze_conf_radius = _cfg_get(cfg, "gaze_conf_radius", st.TASK1_GAZE_CONF_RADIUS)
    min_soft_conf = _cfg_get(cfg, "min_soft_conf_around_gaze", st.TASK1_MIN_SOFT_CONF_AROUND_GAZE)
    soft_mask_threshold = _cfg_get(cfg, "soft_mask_threshold", st.TASK1_SOFT_MASK_THRESHOLD)
    reject_overlap = _cfg_get(cfg, "reject_if_mask_overlaps_person", st.TASK1_REJECT_IF_MASK_OVERLAPS_PERSON)
    overlap_thresh = _cfg_get(cfg, "person_overlap_threshold", st.TASK1_PERSON_OVERLAP_THRESHOLD)
    allow_box_fallback = _cfg_get(cfg, "allow_box_fallback", True)

    if use_tight_box:
        box_size = point_box_size
    else:
        box_size = max(point_box_size, 180)

    box = clip_box(point_to_box((gx, gy), box_size=box_size), W, H)

    point_coords = np.array([[gx, gy]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)
    input_box = np.array(box, dtype=np.float32)[None, :]

    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=input_box,
        multimask_output=True,
    )

    if masks is None or len(masks) == 0:
        if update_reject_stats:
            st.REJECT_STATS["t1_sam2_no_mask"] += 1
            st.REJECT_STATS["t1_sam2_total_fail"] += 1
        debug["last_reject_reason"] = "no_mask"
        return None, None, None, img_pil_resized, None, debug

    soft_masks = None
    try:
        if logits is not None and len(logits) == len(masks):
            soft_masks = _sigmoid(np.array(logits, dtype=np.float32))
    except Exception:
        soft_masks = None

    order = list(np.argsort(-np.array(scores)))

    for idx in order[:3]:
        m = masks[int(idx)].astype(np.uint8)
        if soft_masks is not None:
            soft_m = _resize_soft_mask(soft_masks[int(idx)], W, H)
        else:
            soft_m = None

        if soft_m is not None and soft_mask_threshold > 0:
            m_soft = (soft_m >= soft_mask_threshold).astype(np.uint8)
            m_soft = _mask_keep_component_at_point(m_soft, (gx, gy))
            if m_soft.sum() > 0:
                area_ratio_soft = float(m_soft.sum()) / float(max(1, H * W))
                if mask_min_area_ratio <= area_ratio_soft <= mask_max_area_ratio:
                    m = m_soft

        if dilate_mask_on:
            m = dilate_mask(m, iterations=dilate_iter)

        m = _mask_keep_component_at_point(m, (gx, gy))
        if soft_m is not None:
            soft_m = soft_m * m.astype(np.float32)

        area_ratio = float(m.sum()) / float(max(1, H * W))
        if area_ratio < mask_min_area_ratio or area_ratio > mask_max_area_ratio:
            continue

        if soft_m is not None and min_soft_conf > 0:
            conf = _mean_conf_around_point(soft_m, (gx, gy), gaze_conf_radius)
            if conf is not None and conf < min_soft_conf:
                continue

        if reject_overlap and body_bbox_xywh_scaled is not None:
            overlap = mask_person_overlap_ratio(m, body_bbox_xywh_scaled)
            if overlap >= overlap_thresh:
                debug["last_reject_reason"] = "overlap_reject"
                if debug["overlap_reject_mask"] is None:
                    debug["overlap_reject_mask"] = m.copy()
                    debug["overlap_reject_bb"] = mask_to_bbox(m)
                    debug["overlap_reject_soft"] = soft_m.copy() if soft_m is not None else None
                continue

        masked_crop, bb = _crop_masked(
            img_np, m,
            pad=pad_around_mask,
            pad_ratio=_cfg_get(cfg, "pad_around_mask_ratio", st.TASK1_PAD_AROUND_MASK_RATIO),
            pad_max=_cfg_get(cfg, "pad_around_mask_max", st.TASK1_PAD_AROUND_MASK_MAX),
        )
        if masked_crop is None or bb is None:
            if update_reject_stats:
                st.REJECT_STATS["t1_sam2_empty_crop"] += 1
            continue
        return masked_crop, m, bb, img_pil_resized, soft_m, debug

    if not allow_box_fallback:
        if update_reject_stats:
            st.REJECT_STATS["t1_sam2_total_fail"] += 1
        debug["last_reject_reason"] = debug["last_reject_reason"] or "total_fail"
        return None, None, None, img_pil_resized, None, debug

    masks2, scores2, logits2 = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=None,
        multimask_output=True,
    )

    if masks2 is not None and len(masks2) > 0:
        soft_masks2 = None
        try:
            if logits2 is not None and len(logits2) == len(masks2):
                soft_masks2 = _sigmoid(np.array(logits2, dtype=np.float32))
        except Exception:
            soft_masks2 = None

        order2 = list(np.argsort(-np.array(scores2)))
        for idx in order2[:3]:
            m = masks2[int(idx)].astype(np.uint8)
            if soft_masks2 is not None:
                soft_m = _resize_soft_mask(soft_masks2[int(idx)], W, H)
            else:
                soft_m = None

            if soft_m is not None and soft_mask_threshold > 0:
                m_soft = (soft_m >= soft_mask_threshold).astype(np.uint8)
                m_soft = _mask_keep_component_at_point(m_soft, (gx, gy))
                if m_soft.sum() > 0:
                    area_ratio_soft = float(m_soft.sum()) / float(max(1, H * W))
                    if mask_min_area_ratio <= area_ratio_soft <= mask_max_area_ratio:
                        m = m_soft

            m = _mask_keep_component_at_point(m, (gx, gy))
            if soft_m is not None:
                soft_m = soft_m * m.astype(np.float32)

            area_ratio = float(m.sum()) / float(max(1, H * W))
            if area_ratio < mask_min_area_ratio or area_ratio > mask_max_area_ratio:
                continue

            if soft_m is not None and min_soft_conf > 0:
                conf = _mean_conf_around_point(soft_m, (gx, gy), gaze_conf_radius)
                if conf is not None and conf < min_soft_conf:
                    continue

            if reject_overlap and body_bbox_xywh_scaled is not None:
                overlap = mask_person_overlap_ratio(m, body_bbox_xywh_scaled)
                if overlap >= overlap_thresh:
                    debug["last_reject_reason"] = "overlap_reject"
                    if debug["overlap_reject_mask"] is None:
                        debug["overlap_reject_mask"] = m.copy()
                        debug["overlap_reject_bb"] = mask_to_bbox(m)
                        debug["overlap_reject_soft"] = soft_m.copy() if soft_m is not None else None
                    continue

            masked_crop, bb = _crop_masked(
                img_np, m,
                pad=pad_around_mask,
                pad_ratio=_cfg_get(cfg, "pad_around_mask_ratio", st.TASK1_PAD_AROUND_MASK_RATIO),
                pad_max=_cfg_get(cfg, "pad_around_mask_max", st.TASK1_PAD_AROUND_MASK_MAX),
            )
            if masked_crop is None or bb is None:
                if update_reject_stats:
                    st.REJECT_STATS["t1_sam2_empty_crop"] += 1
                continue
            return masked_crop, m, bb, img_pil_resized, soft_m, debug

    if update_reject_stats:
        st.REJECT_STATS["t1_sam2_total_fail"] += 1
    debug["last_reject_reason"] = debug["last_reject_reason"] or "total_fail"
    return None, None, None, img_pil_resized, None, debug


def segment_object_at_gaze(zf, split, seq, cam, frame_id, point_xy_scaled, body_bbox_xywh_scaled=None, cfg=None):
    predictor = ensure_sam2()
    debug = {
        "last_reject_reason": None,
        "overlap_reject_mask": None,
        "overlap_reject_bb": None,
        "overlap_reject_soft": None,
    }

    zp = zip_try_image_path(zf, split, seq, cam, frame_id)
    if zp is None:
        return None, None, None, None, None, debug

    img_pil = zip_read_image(zf, zp)
    img_pil = img_pil.resize(st.RESIZE_WH)
    img_np = np.array(img_pil)
    H, W = img_np.shape[0], img_np.shape[1]

    predictor.set_image(preprocess_segmentation_image_np(img_np))

    gx, gy = float(point_xy_scaled[0]), float(point_xy_scaled[1])
    use_tight_box = _cfg_get(cfg, "use_tight_box", st.TASK1_USE_TIGHT_BOX)
    point_box_size = _cfg_get(cfg, "point_box_size", st.TASK1_POINT_BOX_SIZE)
    pad_around_mask = _cfg_get(cfg, "pad_around_mask", st.TASK1_PAD_AROUND_MASK)
    dilate_mask_on = _cfg_get(cfg, "dilate_mask", st.TASK1_DILATE_MASK)
    dilate_iter = _cfg_get(cfg, "dilate_iter", st.TASK1_DILATE_ITER)
    mask_min_area_ratio = _cfg_get(cfg, "mask_min_area_ratio", st.TASK1_MASK_MIN_AREA_RATIO)
    mask_max_area_ratio = _cfg_get(cfg, "mask_max_area_ratio", st.TASK1_MASK_MAX_AREA_RATIO)
    gaze_conf_radius = _cfg_get(cfg, "gaze_conf_radius", st.TASK1_GAZE_CONF_RADIUS)
    min_soft_conf = _cfg_get(cfg, "min_soft_conf_around_gaze", st.TASK1_MIN_SOFT_CONF_AROUND_GAZE)
    soft_mask_threshold = _cfg_get(cfg, "soft_mask_threshold", st.TASK1_SOFT_MASK_THRESHOLD)
    reject_overlap = _cfg_get(cfg, "reject_if_mask_overlaps_person", st.TASK1_REJECT_IF_MASK_OVERLAPS_PERSON)
    overlap_thresh = _cfg_get(cfg, "person_overlap_threshold", st.TASK1_PERSON_OVERLAP_THRESHOLD)
    allow_box_fallback = _cfg_get(cfg, "allow_box_fallback", True)

    if use_tight_box:
        box_size = point_box_size
    else:
        box_size = max(point_box_size, 180)

    box = clip_box(point_to_box((gx, gy), box_size=box_size), W, H)

    point_coords = np.array([[gx, gy]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)
    input_box = np.array(box, dtype=np.float32)[None, :]

    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=input_box,
        multimask_output=True,
    )

    if masks is None or len(masks) == 0:
        st.REJECT_STATS["t1_sam2_no_mask"] += 1
        st.REJECT_STATS["t1_sam2_total_fail"] += 1
        debug["last_reject_reason"] = "no_mask"
        return None, None, None, img_pil, None, debug

    soft_masks = None
    try:
        if logits is not None and len(logits) == len(masks):
            soft_masks = _sigmoid(np.array(logits, dtype=np.float32))
    except Exception:
        soft_masks = None

    order = list(np.argsort(-np.array(scores)))

    for idx in order[:3]:
        m = masks[int(idx)].astype(np.uint8)
        if soft_masks is not None:
            soft_m = _resize_soft_mask(soft_masks[int(idx)], W, H)
        else:
            soft_m = None
        if soft_m is not None and soft_mask_threshold > 0:
            m_soft = (soft_m >= soft_mask_threshold).astype(np.uint8)
            m_soft = _mask_keep_component_at_point(m_soft, (gx, gy))
            if m_soft.sum() > 0:
                area_ratio_soft = float(m_soft.sum()) / float(max(1, H * W))
                if mask_min_area_ratio <= area_ratio_soft <= mask_max_area_ratio:
                    m = m_soft

        if dilate_mask_on:
            m = dilate_mask(m, iterations=dilate_iter)

        m = _mask_keep_component_at_point(m, (gx, gy))
        if soft_m is not None:
            soft_m = soft_m * m.astype(np.float32)

        area_ratio = float(m.sum()) / float(max(1, H * W))
        if area_ratio < mask_min_area_ratio or area_ratio > mask_max_area_ratio:
            continue

        if soft_m is not None and min_soft_conf > 0:
            conf = _mean_conf_around_point(soft_m, (gx, gy), gaze_conf_radius)
            if conf is not None and conf < min_soft_conf:
                continue

        if reject_overlap and body_bbox_xywh_scaled is not None:
            overlap = mask_person_overlap_ratio(m, body_bbox_xywh_scaled)
            if overlap >= overlap_thresh:
                debug["last_reject_reason"] = "overlap_reject"
                if debug["overlap_reject_mask"] is None:
                    debug["overlap_reject_mask"] = m.copy()
                    debug["overlap_reject_bb"] = mask_to_bbox(m)
                    debug["overlap_reject_soft"] = soft_m.copy() if soft_m is not None else None
                continue

        masked_crop, bb = _crop_masked(
            img_np, m,
            pad=pad_around_mask,
            pad_ratio=_cfg_get(cfg, "pad_around_mask_ratio", st.TASK1_PAD_AROUND_MASK_RATIO),
            pad_max=_cfg_get(cfg, "pad_around_mask_max", st.TASK1_PAD_AROUND_MASK_MAX),
        )
        if masked_crop is None or bb is None:
            st.REJECT_STATS["t1_sam2_empty_crop"] += 1
            continue
        if masked_crop is not None and bb is not None:
            return masked_crop, m, bb, img_pil, soft_m, debug

    # fallback: box=None
    if not allow_box_fallback:
        st.REJECT_STATS["t1_sam2_total_fail"] += 1
        debug["last_reject_reason"] = debug["last_reject_reason"] or "total_fail"
        return None, None, None, img_pil, None, debug

    masks2, scores2, logits2 = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=None,
        multimask_output=True,
    )

    if masks2 is not None and len(masks2) > 0:
        soft_masks2 = None
        try:
            if logits2 is not None and len(logits2) == len(masks2):
                soft_masks2 = _sigmoid(np.array(logits2, dtype=np.float32))
        except Exception:
            soft_masks2 = None

        order2 = list(np.argsort(-np.array(scores2)))
        for idx in order2[:3]:
            m = masks2[int(idx)].astype(np.uint8)
            if soft_masks2 is not None:
                soft_m = _resize_soft_mask(soft_masks2[int(idx)], W, H)
            else:
                soft_m = None
            if soft_m is not None and soft_mask_threshold > 0:
                m_soft = (soft_m >= soft_mask_threshold).astype(np.uint8)
                m_soft = _mask_keep_component_at_point(m_soft, (gx, gy))
                if m_soft.sum() > 0:
                    area_ratio_soft = float(m_soft.sum()) / float(max(1, H * W))
                    if mask_min_area_ratio <= area_ratio_soft <= mask_max_area_ratio:
                        m = m_soft

            m = _mask_keep_component_at_point(m, (gx, gy))
            if soft_m is not None:
                soft_m = soft_m * m.astype(np.float32)

            area_ratio = float(m.sum()) / float(max(1, H * W))
            if area_ratio < mask_min_area_ratio or area_ratio > mask_max_area_ratio:
                continue

            if soft_m is not None and min_soft_conf > 0:
                conf = _mean_conf_around_point(soft_m, (gx, gy), gaze_conf_radius)
                if conf is not None and conf < min_soft_conf:
                    continue

            if reject_overlap and body_bbox_xywh_scaled is not None:
                overlap = mask_person_overlap_ratio(m, body_bbox_xywh_scaled)
                if overlap >= overlap_thresh:
                    debug["last_reject_reason"] = "overlap_reject"
                    if debug["overlap_reject_mask"] is None:
                        debug["overlap_reject_mask"] = m.copy()
                        debug["overlap_reject_bb"] = mask_to_bbox(m)
                        debug["overlap_reject_soft"] = soft_m.copy() if soft_m is not None else None
                    continue
            masked_crop, bb = _crop_masked(
                img_np, m,
                pad=pad_around_mask,
                pad_ratio=_cfg_get(cfg, "pad_around_mask_ratio", st.TASK1_PAD_AROUND_MASK_RATIO),
                pad_max=_cfg_get(cfg, "pad_around_mask_max", st.TASK1_PAD_AROUND_MASK_MAX),
            )
            if masked_crop is None or bb is None:
                st.REJECT_STATS["t1_sam2_empty_crop"] += 1
                continue
            if masked_crop is not None and bb is not None:
                return masked_crop, m, bb, img_pil, soft_m, debug

    st.REJECT_STATS["t1_sam2_total_fail"] += 1
    debug["last_reject_reason"] = debug["last_reject_reason"] or "total_fail"
    return None, None, None, img_pil, None, debug
