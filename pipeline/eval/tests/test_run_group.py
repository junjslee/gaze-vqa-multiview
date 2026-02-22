from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from gaze_vqa.pipeline.eval.model_registry import model_keys_for_group
from gaze_vqa.pipeline.eval.run_campaign import run_group


def test_run_group_routes_keys_and_runtime(monkeypatch) -> None:
    seen: List[Dict[str, Any]] = []
    runtime = {"request_interval": 0.5, "retry": 2}

    def _fake_run_model(
        campaign_dir: Path,
        model_key: str,
        runtime: Dict[str, Any] | None = None,
        model_override_file: Path | None = None,
        reset_predictions: bool = False,
    ) -> Dict[str, Any]:
        seen.append(
            {
                "campaign_dir": campaign_dir,
                "model_key": model_key,
                "runtime": dict(runtime or {}),
                "model_override_file": model_override_file,
                "reset_predictions": bool(reset_predictions),
            }
        )
        return {
            "model_key": model_key,
            "newly_processed": 0,
            "errors_in_new_rows": 0,
        }

    monkeypatch.setattr("gaze_vqa.pipeline.eval.run_campaign.run_model", _fake_run_model)

    campaign_dir = Path("/tmp/dummy-campaign")
    out = run_group(
        campaign_dir=campaign_dir,
        group="oss",
        runtime=runtime,
        reset_predictions=True,
    )

    expected_keys = model_keys_for_group("oss")
    assert [row["model_key"] for row in out] == expected_keys
    assert [row["model_key"] for row in seen] == expected_keys
    assert all(row["campaign_dir"] == campaign_dir for row in seen)
    assert all(row["runtime"] == runtime for row in seen)
    assert all(row["reset_predictions"] is True for row in seen)
