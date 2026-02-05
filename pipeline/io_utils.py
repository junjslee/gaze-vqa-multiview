import io
import json
import zipfile
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

from . import state as st


def _run(cmd, check=True):
    st.logger.info(">> " + " ".join(cmd))
    subprocess.run(cmd, check=check)


def ensure_mvgt_zip():
    if not st.LOCAL_ZIP_PATH:
        raise RuntimeError("--mvgt_zip not provided.")

    p = Path(st.LOCAL_ZIP_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.exists() and zipfile.is_zipfile(str(p)):
        st.logger.info(f"✅ MVGT zip exists and is valid: {p}")
        return

    if p.exists():
        st.logger.warning(f"⚠️ MVGT zip exists but invalid. Deleting: {p}")
        try:
            p.unlink()
        except Exception:
            pass

    st.logger.info(f"⬇️ Downloading MVGT zip to: {p}")
    _run(["bash", "-lc", f"curl -L -C - --retry 8 --retry-delay 2 '{st.DATASET_URL}' -o '{str(p)}'"])

    if not p.exists() or not zipfile.is_zipfile(str(p)):
        raise RuntimeError(f"MVGT zip download failed or corrupted: {p}")

    st.logger.info(f"✅ MVGT zip downloaded: {p}")


def zip_list_sequences(zf: zipfile.ZipFile):
    seqs = []
    for p in zf.namelist():
        if not (p.startswith("Annotations/") and p.endswith(".json")):
            continue
        parts = p.split("/")
        if len(parts) != 3:
            continue
        split = parts[1]
        seq = Path(parts[2]).stem
        seqs.append((split, seq, p))
    seqs.sort()
    return seqs


def _interleave_sequences_by_split(seqs, rng):
    buckets = {}
    for split, seq, anno_path in seqs:
        buckets.setdefault(split, []).append((split, seq, anno_path))
    for split in buckets:
        rng.shuffle(buckets[split])
    splits = sorted(buckets.keys())

    out = []
    while True:
        any_added = False
        for split in splits:
            if buckets[split]:
                out.append(buckets[split].pop(0))
                any_added = True
        if not any_added:
            break
    return out


def zip_try_image_path(zf, split, seq, cam, frame_id):
    paths = [
        f"Data/{split}/{seq}/Images/{cam}/{frame_id}.JPG",
        f"Data/{split}/{seq}/Images/{cam}/{frame_id}.jpg",
        f"Data/{split}/{seq}/Images/{cam}/{frame_id}.jpeg",
        f"Data/{split}/{seq}/Images/{cam}/{frame_id}.JPEG",
    ]
    for p in paths:
        if p in zf.namelist():
            return p
    return None


def zip_read_image(zf: zipfile.ZipFile, zip_path: str) -> Image.Image:
    data = zf.read(zip_path)
    im = Image.open(io.BytesIO(data))
    return im.convert("RGB")


def zip_read_json(zf: zipfile.ZipFile, zip_path: str):
    data = zf.read(zip_path)
    return json.loads(data.decode("utf-8"))


def save_raw_cam_image(zf, split, seq, cam, frame_id):
    out = st.RAW_IMG_DIR / f"{split}_{seq}_{frame_id}_{cam}_raw.jpg"
    if out.exists():
        return str(out)
    zp = zip_try_image_path(zf, split, seq, cam, frame_id)
    if zp is None:
        return None
    try:
        im = zip_read_image(zf, zp)
        im.save(out, quality=95)
        return str(out)
    except Exception:
        return None


def save_raw_cam_images_parallel(zf, split, seq, cams, frame_id):
    out = []
    with ThreadPoolExecutor(max_workers=st.THREAD_IO) as ex:
        futs = {ex.submit(save_raw_cam_image, zf, split, seq, cam, frame_id): cam for cam in cams}
        for fut in as_completed(futs):
            cam = futs[fut]
            p = fut.result()
            if p:
                out.append({"cam": cam, "image": p})
    return out
