from __future__ import annotations

from pathlib import Path

from gaze_vqa.pipeline.eval.run_campaign import run_model


def test_run_model_runtime_env_fallback_for_blank_api_keys(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-from-env")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-from-env")

    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    # Keep campaign metadata minimal; run_model only checks that it exists.
    (campaign_dir / "campaign_meta.json").write_text("{}", encoding="utf-8")

    seen = {}

    def _fake_run_inference_for_model(
        campaign_dir: Path,
        model_spec,
        runtime,
        reset_predictions: bool = False,
    ):
        seen["runtime"] = dict(runtime)
        seen["model_key"] = model_spec.key
        return {"ok": True}

    monkeypatch.setattr(
        "gaze_vqa.pipeline.eval.run_campaign.run_inference_for_model",
        _fake_run_inference_for_model,
    )

    run_model(
        campaign_dir=campaign_dir,
        model_key="gemini25flash",
        runtime={"openai_api_key": "", "gemini_api_key": ""},
    )

    assert seen["model_key"] == "gemini25flash"
    assert seen["runtime"]["openai_api_key"] == "openai-from-env"
    assert seen["runtime"]["gemini_api_key"] == "gemini-from-env"
