from __future__ import annotations

import csv
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
    assert Path(out["table1_accuracy_no_overlap_csv"]).exists()
    assert Path(out["table2_semantic_no_overlap_csv"]).exists()
    assert Path(out["leaderboard_full_no_overlap_csv"]).exists()
    assert Path(out["summary_json"]).exists()

    summary = json.loads(Path(out["summary_json"]).read_text(encoding="utf-8"))
    assert summary["no_overlap_excluded_model_keys"] == ["gemini30flash"]
    assert summary["no_overlap_table1_rows"] == 1


def test_no_overlap_default_excludes_gemini30flash(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    (campaign / "gt").mkdir(parents=True, exist_ok=True)
    (campaign / "predictions").mkdir(parents=True, exist_ok=True)
    (campaign / "judge").mkdir(parents=True, exist_ok=True)

    (campaign / "campaign_meta.json").write_text(
        json.dumps(
            {
                "model_map": {
                    "gemini30flash": {
                        "key": "gemini30flash",
                        "label": "Gemini 3 Flash",
                        "model_id": "gemini-3-flash-preview",
                        "engine": "gemini_api",
                        "group": "proprietary",
                        "track": "A",
                    },
                    "m1": {
                        "key": "m1",
                        "label": "Model1",
                        "model_id": "m1-id",
                        "engine": "api",
                        "group": "proprietary",
                        "track": "A",
                    },
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
    for model_key, model_id in [("gemini30flash", "gemini-3-flash-preview"), ("m1", "m1-id")]:
        (campaign / "predictions" / f"{model_key}.jsonl").write_text(
            json.dumps(
                {
                    "sample_uid": "s1",
                    "model_key": model_key,
                    "model_id": model_id,
                    "inference_answer": "chair",
                    "error": "",
                }
            )
            + "\n"
        )
        (campaign / "judge" / f"{model_key}.jsonl").write_text(
            json.dumps({"sample_uid": "s1", "gemini_judge": "correct"}) + "\n"
        )

    out = aggregate_campaign(campaign)
    with Path(out["table1_accuracy_no_overlap_csv"]).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["model_key"] == "m1"
