from __future__ import annotations

import json
from pathlib import Path

from gaze_vqa.pipeline.eval.aggregate_tables import aggregate_campaign


def test_aggregate_tables_outputs(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    (campaign / "gt").mkdir(parents=True, exist_ok=True)
    (campaign / "predictions").mkdir(parents=True, exist_ok=True)
    (campaign / "judge").mkdir(parents=True, exist_ok=True)

    (campaign / "campaign_meta.json").write_text(
        json.dumps(
            {
                "model_map": {
                    "m1": {
                        "key": "m1",
                        "label": "Model1",
                        "model_id": "m1-id",
                        "engine": "api",
                        "group": "proprietary",
                        "track": "A",
                    }
                }
            }
        )
    )
    (campaign / "gt" / "gt_manifest_v1.jsonl").write_text(
        json.dumps(
            {
                "sample_uid": "s1",
                "task_type": "gaze_target_recognition",
                "task_id": 1,
                "scene": "commons",
                "question": "Q1",
                "groundtruth_answer": "chair",
            }
        )
        + "\n"
    )
    (campaign / "predictions" / "m1.jsonl").write_text(
        json.dumps(
            {
                "sample_uid": "s1",
                "model_key": "m1",
                "model_id": "m1-id",
                "inference_answer": "chair",
                "error": "",
            }
        )
        + "\n"
    )
    (campaign / "judge" / "m1.jsonl").write_text(
        json.dumps({"sample_uid": "s1", "gemini_judge": "correct"}) + "\n"
    )

    out = aggregate_campaign(campaign)
    assert Path(out["table1_accuracy_csv"]).exists()
    assert Path(out["table2_semantic_csv"]).exists()
    assert Path(out["leaderboard_full_csv"]).exists()
    assert Path(out["summary_json"]).exists()

