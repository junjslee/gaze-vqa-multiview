from __future__ import annotations

import json
from pathlib import Path

from gaze_vqa.pipeline.eval.freeze_gt import freeze_gt
from gaze_vqa.pipeline.eval.schemas import iter_jsonl


def _run_freeze(tmp_path: Path, out_name: str) -> str:
    img = tmp_path / "Commons_03_27_1_0047_Cam1_raw.jpg"
    if not img.exists():
        img.write_bytes(b"x")
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "task_id": 2,
                        "task_type": "relative_orientation_reasoning",
                        "question": "Q",
                        "answer": "A",
                        "input_images": [{"cam": "Cam1", "image": str(img)}],
                    }
                ]
            }
        )
    )
    out = freeze_gt(benchmark_path=benchmark, out_dir=tmp_path / out_name, strict_image_exists=True)
    row = next(iter_jsonl(out["manifest"]))
    return row["sample_uid"]


def test_sample_uid_deterministic(tmp_path: Path) -> None:
    uid1 = _run_freeze(tmp_path, "gt1")
    uid2 = _run_freeze(tmp_path, "gt2")
    assert uid1 == uid2

