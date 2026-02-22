from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from gaze_vqa.pipeline.eval.infer_runner import run_inference_for_model
from gaze_vqa.pipeline.eval.model_registry import get_model_spec
from gaze_vqa.pipeline.eval.review_workflow import apply_review, build_review
from gaze_vqa.pipeline.eval.schemas import file_sha256, iter_jsonl, write_json, write_jsonl


def _write_image(path: Path, color=(255, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (32, 24), color=color)
    im.save(path, format="JPEG")


def _campaign_meta(campaign: Path, manifest_path: Path) -> dict:
    return {
        "campaign_name": campaign.name,
        "created_at": "2026-02-16T00:00:00+00:00",
        "benchmark_path": str(campaign / "benchmark_gazevqa.json"),
        "benchmark_sha256": "x",
        "gt_manifest_path": str(manifest_path),
        "gt_manifest_sha256": file_sha256(manifest_path),
        "base_manifest_path": str(manifest_path),
        "active_manifest_path": str(manifest_path),
        "active_manifest_sha256": file_sha256(manifest_path),
        "prompt_version": "eval_prompt_v1",
        "model_map": {},
        "container_sif": "dummy.sif",
        "slurm": {},
        "review": {
            "granularity": "frame_bundle",
            "policy": "exclude_rejects",
            "bundle_counts": {},
        },
    }


def test_build_review_outputs_html_and_items(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    gt_dir = campaign / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    run_dir = tmp_path / "runs" / "runA"
    debug_dir = run_dir / "debug" / "frame_commons_03_27_1_0047"
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "tasks.html").write_text("<html>debug</html>", encoding="utf-8")
    benchmark = run_dir / "benchmark_gazevqa.json"
    benchmark.write_text(json.dumps({"samples": []}), encoding="utf-8")

    img1 = tmp_path / "imgs" / "Commons_03_27_1_0047_Cam1_raw.jpg"
    img2 = tmp_path / "imgs" / "Commons_03_27_1_0047_Cam2_raw.jpg"
    _write_image(img1, color=(255, 0, 0))
    _write_image(img2, color=(0, 255, 0))

    manifest = gt_dir / "gt_manifest_v1.jsonl"
    rows = [
        {
            "sample_uid": "s1",
            "task_id": 1,
            "task_type": "gaze_target_recognition",
            "scene": "commons",
            "seq": "03_27_1",
            "frame": "0047",
            "question": "Q1",
            "groundtruth_answer": "chair",
            "image_paths": [str(img1), str(img2)],
            "camera_ids": ["Cam1", "Cam2"],
            "source_benchmark": str(benchmark),
            "review_flags": ["short_answer"],
        },
        {
            "sample_uid": "s2",
            "task_id": 4,
            "task_type": "viewpoint_based_accessibility",
            "scene": "commons",
            "seq": "03_27_1",
            "frame": "0047",
            "question": "Q4",
            "groundtruth_answer": "YES",
            "image_paths": [str(img1)],
            "camera_ids": ["Cam1"],
            "source_benchmark": str(benchmark),
            "review_flags": [],
        },
    ]
    write_jsonl(manifest, rows)
    write_json(campaign / "campaign_meta.json", _campaign_meta(campaign, manifest))

    out = build_review(campaign, page_size=200, thumb_long_edge=96)

    assert Path(out["review_index_html"]).exists()
    assert Path(out["review_items_json"]).exists()
    assert Path(out["decisions_template_csv"]).exists()
    assert out["bundle_count"] == 1

    review_items = json.loads(Path(out["review_items_json"]).read_text(encoding="utf-8"))
    assert len(review_items) == 1
    item = review_items[0]
    assert item["bundle_id"] == "commons||03_27_1||0047"
    assert item["debug_tasks_html"]
    assert len(item["images"]) == 2


def test_apply_review_exclude_rejects_and_activate(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    gt_dir = campaign / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    img1 = tmp_path / "imgs" / "Commons_03_27_1_0047_Cam1_raw.jpg"
    img2 = tmp_path / "imgs" / "Commons_03_27_1_0051_Cam1_raw.jpg"
    _write_image(img1)
    _write_image(img2)

    manifest = gt_dir / "gt_manifest_v1.jsonl"
    rows = [
        {
            "sample_uid": "s1",
            "task_id": 1,
            "task_type": "gaze_target_recognition",
            "scene": "commons",
            "seq": "03_27_1",
            "frame": "0047",
            "question": "Q1",
            "groundtruth_answer": "chair",
            "image_paths": [str(img1)],
            "camera_ids": ["Cam1"],
            "source_benchmark": str(tmp_path / "bench.json"),
            "review_flags": [],
        },
        {
            "sample_uid": "s2",
            "task_id": 1,
            "task_type": "gaze_target_recognition",
            "scene": "commons",
            "seq": "03_27_1",
            "frame": "0051",
            "question": "Q2",
            "groundtruth_answer": "table",
            "image_paths": [str(img2)],
            "camera_ids": ["Cam1"],
            "source_benchmark": str(tmp_path / "bench.json"),
            "review_flags": [],
        },
    ]
    write_jsonl(manifest, rows)
    write_json(campaign / "campaign_meta.json", _campaign_meta(campaign, manifest))

    decisions = campaign / "review" / "decisions.csv"
    decisions.parent.mkdir(parents=True, exist_ok=True)
    with decisions.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bundle_id", "decision", "note", "reviewer", "updated_at"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "bundle_id": "commons||03_27_1||0047",
                "decision": "reject",
                "note": "bad label",
                "reviewer": "r1",
                "updated_at": "",
            }
        )

    out = apply_review(campaign, decisions_path=decisions, set_active=True)
    reviewed = Path(out["reviewed_manifest"])
    assert reviewed.exists()

    kept = list(iter_jsonl(reviewed))
    assert len(kept) == 1
    assert kept[0]["sample_uid"] == "s2"

    meta = json.loads((campaign / "campaign_meta.json").read_text(encoding="utf-8"))
    assert meta["active_manifest_path"] == str(reviewed)
    assert meta["review"]["bundle_counts"]["removed"] == 1


def test_inference_uses_active_manifest_if_present(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    gt_dir = campaign / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    base_manifest = gt_dir / "gt_manifest_v1.jsonl"
    active_manifest = gt_dir / "gt_manifest_v1_reviewed.jsonl"

    write_jsonl(
        base_manifest,
        [
            {
                "sample_uid": "base1",
                "task_id": 1,
                "task_type": "gaze_target_recognition",
                "scene": "commons",
                "seq": "03_27_1",
                "frame": "0001",
                "question": "Q",
                "groundtruth_answer": "A",
                "image_paths": [],
                "camera_ids": [],
                "source_benchmark": "x",
                "review_flags": [],
            }
        ],
    )
    write_jsonl(
        active_manifest,
        [
            {
                "sample_uid": "act1",
                "task_id": 1,
                "task_type": "gaze_target_recognition",
                "scene": "commons",
                "seq": "03_27_1",
                "frame": "0002",
                "question": "Q",
                "groundtruth_answer": "A",
                "image_paths": [],
                "camera_ids": [],
                "source_benchmark": "x",
                "review_flags": [],
            },
            {
                "sample_uid": "act2",
                "task_id": 1,
                "task_type": "gaze_target_recognition",
                "scene": "commons",
                "seq": "03_27_1",
                "frame": "0003",
                "question": "Q",
                "groundtruth_answer": "A",
                "image_paths": [],
                "camera_ids": [],
                "source_benchmark": "x",
                "review_flags": [],
            },
        ],
    )

    meta = _campaign_meta(campaign, base_manifest)
    meta["active_manifest_path"] = str(active_manifest)
    meta["active_manifest_sha256"] = file_sha256(active_manifest)
    write_json(campaign / "campaign_meta.json", meta)

    spec = get_model_spec("gpt41")
    out = run_inference_for_model(
        campaign_dir=campaign,
        model_spec=spec,
        runtime={"openai_api_key": ""},
    )

    assert out["unavailable_reason"] == "missing_openai_api_key"
    assert out["total_manifest_samples"] == 2
    assert out["manifest_path"] == str(active_manifest)
