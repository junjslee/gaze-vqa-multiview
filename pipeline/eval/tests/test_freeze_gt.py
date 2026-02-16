from __future__ import annotations

import json
from pathlib import Path

from gaze_vqa.pipeline.eval.freeze_gt import freeze_gt
from gaze_vqa.pipeline.eval.schemas import iter_jsonl


def test_freeze_gt_conversion(tmp_path: Path) -> None:
    img_dir = tmp_path / "imgs"
    img_dir.mkdir(parents=True, exist_ok=True)
    img1 = img_dir / "Commons_03_27_1_0047_Cam1_raw.jpg"
    img1.write_bytes(b"fake")

    benchmark = tmp_path / "benchmark_gazevqa.json"
    data = {
        "samples": [
            {
                "task_id": 1,
                "task_type": "gaze_target_recognition",
                "question": "What is the person looking at?",
                "answer": "chair",
                "scene": "commons",
                "input_cams": ["Cam1"],
                "input_images": [{"cam": "Cam1", "image": str(img1)}],
            }
        ]
    }
    benchmark.write_text(json.dumps(data))

    out = freeze_gt(benchmark_path=benchmark, out_dir=tmp_path / "gt", strict_image_exists=True)
    manifest = list(iter_jsonl(out["manifest"]))
    assert len(manifest) == 1
    row = manifest[0]
    assert row["task_id"] == 1
    assert row["scene"] == "commons"
    assert row["frame"] == "0047"
    assert row["groundtruth_answer"] == "chair"

