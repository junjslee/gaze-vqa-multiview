from __future__ import annotations

import json
from pathlib import Path

import pytest

from gaze_vqa.pipeline.eval.run_campaign import merge_campaigns


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_campaign(
    root: Path,
    name: str,
    *,
    manifest_rows: list[dict],
    model_map: dict,
    prediction_rows_by_model: dict[str, list[dict]],
    judge_rows_by_model: dict[str, list[dict]],
) -> Path:
    campaign = root / name
    gt_manifest = campaign / "gt" / "gt_manifest_v1.jsonl"
    _write_jsonl(gt_manifest, manifest_rows)

    for key, rows in prediction_rows_by_model.items():
        _write_jsonl(campaign / "predictions" / f"{key}.jsonl", rows)
    for key, rows in judge_rows_by_model.items():
        _write_jsonl(campaign / "judge" / f"{key}.jsonl", rows)

    (campaign / "campaign_meta.json").write_text(
        json.dumps(
            {
                "campaign_name": name,
                "model_map": model_map,
                "gt_manifest_path": str(gt_manifest),
                "active_manifest_path": str(gt_manifest),
            }
        ),
        encoding="utf-8",
    )
    return campaign


def _manifest_row(uid: str) -> dict:
    return {
        "sample_uid": uid,
        "task_type": "gaze_target_recognition",
        "task_id": 1,
        "scene": "commons",
        "question": "Q",
        "groundtruth_answer": "chair",
        "image_paths": [],
        "camera_ids": [],
    }


def _pred_row(uid: str, model_key: str, model_id: str) -> dict:
    return {
        "sample_uid": uid,
        "model_key": model_key,
        "model_id": model_id,
        "inference_answer": "chair",
        "error": "",
    }


def _judge_row(uid: str) -> dict:
    return {"sample_uid": uid, "gemini_judge": "correct"}


def test_merge_campaigns_success(tmp_path: Path) -> None:
    model_map_a = {
        "m_prop": {
            "key": "m_prop",
            "label": "Model Prop",
            "model_id": "model-prop",
            "engine": "openai_api",
            "group": "proprietary",
            "track": "A",
        }
    }
    model_map_b = {
        "m_oss": {
            "key": "m_oss",
            "label": "Model OSS",
            "model_id": "model-oss",
            "engine": "sglang",
            "group": "open-source",
            "track": "B",
        }
    }
    manifest = [_manifest_row("s1")]

    c1 = _make_campaign(
        tmp_path,
        "campaign_prop",
        manifest_rows=manifest,
        model_map=model_map_a,
        prediction_rows_by_model={"m_prop": [_pred_row("s1", "m_prop", "model-prop")]},
        judge_rows_by_model={"m_prop": [_judge_row("s1")]},
    )
    c2 = _make_campaign(
        tmp_path,
        "campaign_oss",
        manifest_rows=manifest,
        model_map=model_map_b,
        prediction_rows_by_model={"m_oss": [_pred_row("s1", "m_oss", "model-oss")]},
        judge_rows_by_model={"m_oss": [_judge_row("s1")]},
    )

    out_dir = tmp_path / "campaign_merged"
    out = merge_campaigns([c1, c2], out_dir, strict_manifest_match=True)

    assert Path(out["table1_accuracy_csv"]).exists()
    assert Path(out["table2_semantic_csv"]).exists()
    assert Path(out["leaderboard_full_csv"]).exists()
    assert Path(out["table1_accuracy_no_overlap_csv"]).exists()
    assert Path(out["table2_semantic_no_overlap_csv"]).exists()
    assert Path(out["leaderboard_full_no_overlap_csv"]).exists()
    assert Path(out["summary_json"]).exists()
    assert Path(out["merge_meta_json"]).exists()

    summary = json.loads(Path(out["summary_json"]).read_text(encoding="utf-8"))
    assert summary["table1_rows"] == 2
    assert summary["no_overlap_table1_rows"] == 2
    assert set(summary["missing_models"]) == set()


def test_merge_campaigns_fails_on_manifest_mismatch(tmp_path: Path) -> None:
    model_map = {
        "m1": {
            "key": "m1",
            "label": "Model1",
            "model_id": "m1-id",
            "engine": "openai_api",
            "group": "proprietary",
            "track": "A",
        }
    }

    c1 = _make_campaign(
        tmp_path,
        "c1",
        manifest_rows=[_manifest_row("s1")],
        model_map=model_map,
        prediction_rows_by_model={"m1": [_pred_row("s1", "m1", "m1-id")]},
        judge_rows_by_model={"m1": [_judge_row("s1")]},
    )
    c2 = _make_campaign(
        tmp_path,
        "c2",
        manifest_rows=[_manifest_row("s2")],
        model_map=model_map,
        prediction_rows_by_model={"m1": [_pred_row("s2", "m1", "m1-id")]},
        judge_rows_by_model={"m1": [_judge_row("s2")]},
    )

    with pytest.raises(ValueError, match="Manifest mismatch"):
        merge_campaigns([c1, c2], tmp_path / "out", strict_manifest_match=True)


def test_merge_campaigns_fails_on_conflicting_model_map(tmp_path: Path) -> None:
    manifest = [_manifest_row("s1")]
    c1 = _make_campaign(
        tmp_path,
        "c1",
        manifest_rows=manifest,
        model_map={
            "dup": {
                "key": "dup",
                "label": "Model V1",
                "model_id": "dup-v1",
                "engine": "openai_api",
                "group": "proprietary",
                "track": "A",
            }
        },
        prediction_rows_by_model={"dup": [_pred_row("s1", "dup", "dup-v1")]},
        judge_rows_by_model={"dup": [_judge_row("s1")]},
    )
    c2 = _make_campaign(
        tmp_path,
        "c2",
        manifest_rows=manifest,
        model_map={
            "dup": {
                "key": "dup",
                "label": "Model V2",
                "model_id": "dup-v2",
                "engine": "openai_api",
                "group": "proprietary",
                "track": "A",
            }
        },
        prediction_rows_by_model={"dup": [_pred_row("s1", "dup", "dup-v2")]},
        judge_rows_by_model={"dup": [_judge_row("s1")]},
    )

    with pytest.raises(ValueError, match="Conflicting model_map"):
        merge_campaigns([c1, c2], tmp_path / "out", strict_manifest_match=True)
