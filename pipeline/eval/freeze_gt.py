from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schemas import EvalSample, InvalidSample, file_sha256, read_json, utc_now_iso, write_json, write_jsonl


_TASK_TYPES = {
    1: "gaze_target_recognition",
    2: "relative_orientation_reasoning",
    3: "cross_view_visibility_estimation",
    4: "viewpoint_based_accessibility",
}

_PREPS = {"in", "on", "at", "with", "of", "to", "from", "near", "by", "for"}
_HUMAN = {"person", "man", "woman", "boy", "girl", "people"}


def resolve_latest_benchmark(repo_root: Path) -> Path:
    run_dir = repo_root / "runs"
    candidates = [p for p in run_dir.rglob("benchmark_gazevqa.json") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No benchmark_gazevqa.json found under {run_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_scene_seq_frame_from_image(image_path: str) -> Tuple[str, str, str]:
    stem = Path(image_path).name
    stem = stem[:-4] if stem.lower().endswith(".jpg") else stem
    parts = stem.split("_")
    # Expect at least: scene + seq(>=1 token) + frame + CamX + raw
    if len(parts) >= 5 and parts[-1].lower() == "raw" and re.match(r"^cam\d+$", parts[-2], flags=re.IGNORECASE):
        scene = parts[0]
        frame = parts[-3]
        seq = "_".join(parts[1:-3]) if len(parts[1:-3]) > 0 else "unknown"
        return scene.lower(), seq, frame
    return "unknown", "unknown", "unknown"


def _hash_uid(*parts: str) -> str:
    text = "||".join(str(p).strip() for p in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


def _review_flags(task_id: int, question: str, answer: str) -> List[str]:
    flags: List[str] = []
    ql = (question or "").strip().lower()
    al = (answer or "").strip().lower()

    if not ql:
        flags.append("missing_question")
    if not al:
        flags.append("missing_answer")
    if len(al) <= 3 and al:
        flags.append("short_answer")

    toks = al.split()
    if toks and toks[-1] in _PREPS:
        flags.append("trailing_preposition")

    if task_id == 1:
        if "person with wearing" in ql or ql.endswith(" with") or ql.endswith(" wearing"):
            flags.append("weird_task1_question")
        if toks and toks[0] in _HUMAN:
            flags.append("task1_human_label")
        if len(toks) > 6:
            flags.append("task1_long_label")
    return flags


def _extract_images(sample: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    images = sample.get("input_images") or []
    if not isinstance(images, list):
        return [], []
    paths: List[str] = []
    cams: List[str] = []
    for item in images:
        if isinstance(item, dict):
            img = str(item.get("image") or "").strip()
            cam = str(item.get("cam") or "").strip()
            if img:
                paths.append(img)
            if cam:
                cams.append(cam)
    return paths, cams


def freeze_gt(
    benchmark_path: Path,
    out_dir: Path,
    strict_image_exists: bool = True,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = read_json(benchmark_path)
    samples = raw.get("samples") if isinstance(raw, dict) else None
    if not isinstance(samples, list):
        raise ValueError(f"Expected object with list 'samples' in {benchmark_path}")

    valid_rows: List[Dict[str, Any]] = []
    invalid_rows: List[Dict[str, Any]] = []
    review_rows: List[Dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        if not isinstance(sample, dict):
            invalid_rows.append(
                InvalidSample(
                    index=idx,
                    reason="sample_not_object",
                    raw_task_id=None,
                    raw_task_type=None,
                    raw_question=None,
                ).to_dict()
            )
            continue

        task_id = sample.get("task_id")
        task_type = str(sample.get("task_type") or _TASK_TYPES.get(task_id, "unknown")).strip()
        question = str(sample.get("question") or "").strip()
        answer = str(sample.get("answer") or "").strip()
        image_paths, camera_ids = _extract_images(sample)

        if not isinstance(task_id, int):
            invalid_rows.append(
                InvalidSample(
                    index=idx,
                    reason="invalid_task_id",
                    raw_task_id=task_id,
                    raw_task_type=task_type,
                    raw_question=question,
                ).to_dict()
            )
            continue

        if not image_paths:
            invalid_rows.append(
                InvalidSample(
                    index=idx,
                    reason="missing_input_images",
                    raw_task_id=task_id,
                    raw_task_type=task_type,
                    raw_question=question,
                ).to_dict()
            )
            continue

        missing_paths = [p for p in image_paths if not Path(p).exists()]
        if strict_image_exists and missing_paths:
            invalid_rows.append(
                InvalidSample(
                    index=idx,
                    reason=f"missing_image_files:{len(missing_paths)}",
                    raw_task_id=task_id,
                    raw_task_type=task_type,
                    raw_question=question,
                ).to_dict()
            )
            continue

        scene = str(sample.get("scene") or "").strip().lower()
        seq = "unknown"
        frame = "unknown"
        if image_paths:
            p_scene, p_seq, p_frame = parse_scene_seq_frame_from_image(image_paths[0])
            if not scene:
                scene = p_scene
            seq = p_seq
            frame = p_frame
        if not scene:
            scene = "unknown"

        uid = _hash_uid(
            str(benchmark_path.resolve()),
            str(task_id),
            task_type,
            scene,
            seq,
            frame,
            question,
        )
        flags = _review_flags(task_id=task_id, question=question, answer=answer)

        row = EvalSample(
            sample_uid=uid,
            task_id=task_id,
            task_type=task_type,
            scene=scene,
            seq=seq,
            frame=frame,
            question=question,
            groundtruth_answer=answer,
            image_paths=image_paths,
            camera_ids=camera_ids if camera_ids else list(sample.get("input_cams") or []),
            source_benchmark=str(benchmark_path),
            review_flags=flags,
        ).to_dict()
        valid_rows.append(row)

        if flags:
            review_rows.append(
                {
                    "sample_uid": uid,
                    "task_id": task_id,
                    "task_type": task_type,
                    "scene": scene,
                    "seq": seq,
                    "frame": frame,
                    "question": question,
                    "groundtruth_answer": answer,
                    "review_flags": ";".join(flags),
                }
            )

    valid_path = out_dir / "gt_manifest_v1.jsonl"
    invalid_path = out_dir / "gt_manifest_v1_invalid.jsonl"
    review_path = out_dir / "gt_manifest_v1_review_queue.csv"
    meta_path = out_dir / "gt_manifest_v1_meta.json"

    write_jsonl(valid_path, valid_rows)
    write_jsonl(invalid_path, invalid_rows)

    with review_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_uid",
                "task_id",
                "task_type",
                "scene",
                "seq",
                "frame",
                "question",
                "groundtruth_answer",
                "review_flags",
            ],
        )
        writer.writeheader()
        for row in review_rows:
            writer.writerow(row)

    meta = {
        "created_at": utc_now_iso(),
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": file_sha256(benchmark_path),
        "valid_count": len(valid_rows),
        "invalid_count": len(invalid_rows),
        "review_queue_count": len(review_rows),
        "strict_image_exists": bool(strict_image_exists),
        "manifest_sha256": file_sha256(valid_path),
    }
    write_json(meta_path, meta)
    return {
        "manifest": valid_path,
        "invalid_manifest": invalid_path,
        "review_queue": review_path,
        "meta": meta_path,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze GT manifest from benchmark_gazevqa.json")
    p.add_argument("--benchmark_path", type=Path, default=None, help="Path to benchmark_gazevqa.json")
    p.add_argument("--out_dir", type=Path, required=True, help="Output directory for frozen GT artifacts")
    p.add_argument(
        "--allow_missing_images",
        action="store_true",
        help="Do not invalidate rows whose image paths do not exist on disk.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    benchmark_path = args.benchmark_path or resolve_latest_benchmark(repo_root)
    out = freeze_gt(
        benchmark_path=benchmark_path,
        out_dir=args.out_dir,
        strict_image_exists=not args.allow_missing_images,
    )
    print(f"[DONE] Frozen GT from {benchmark_path}")
    for k, v in out.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()

