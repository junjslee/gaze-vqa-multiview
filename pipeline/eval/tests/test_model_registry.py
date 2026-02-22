from __future__ import annotations

from gaze_vqa.pipeline.eval.model_registry import (
    apply_model_overrides,
    get_model_spec,
    list_model_specs,
    model_keys_for_group,
    model_keys_for_track,
)


def test_track_model_lists() -> None:
    a = model_keys_for_track("A")
    b = model_keys_for_track("B")
    assert "gpt41" in a
    assert "gemini30flash" in a
    assert "gemini30pro" in a
    assert "geminiflashlatest" not in a
    assert "gemini31propreview" not in a
    assert "llava_video_72b" in b
    assert "llava_video_7b" in a and "llava_video_7b" in b


def test_model_override() -> None:
    spec = get_model_spec("gpt41")
    patched = apply_model_overrides(spec, {"gpt41": {"model_id": "gpt-4.1-mini"}})
    assert patched.model_id == "gpt-4.1-mini"
    assert patched.key == spec.key


def test_gemini_key_model_id_alignment_defaults() -> None:
    assert get_model_spec("gemini20flash").model_id == "gemini-2.0-flash"
    assert get_model_spec("gemini25flash").model_id == "gemini-2.5-flash"
    assert get_model_spec("gemini30flash").model_id == "gemini-3-flash-preview"
    assert get_model_spec("gemini25pro").model_id == "gemini-2.5-pro"
    assert get_model_spec("gemini30pro").model_id == "gemini-3-pro-preview"


def test_legacy_gemini_aliases_resolve() -> None:
    assert get_model_spec("geminiflashlatest").model_id == "gemini-2.5-flash"
    assert get_model_spec("gemini31propreview").model_id == "gemini-3.1-pro-preview"


def test_group_model_lists() -> None:
    proprietary = set(model_keys_for_group("proprietary"))
    oss = set(model_keys_for_group("oss"))
    all_keys = set(model_keys_for_group("all"))
    assert {
        "gpt41",
        "gpt4o",
        "gemini20flash",
        "gemini25flash",
        "gemini30flash",
        "gemini25pro",
        "gemini30pro",
    } <= proprietary
    assert "geminiflashlatest" not in proprietary
    assert "gemini31propreview" not in proprietary
    assert "gpt41" not in oss
    assert "llava_video_72b" in oss
    assert all_keys == proprietary | oss


def test_oss_models_have_explicit_fallbacks() -> None:
    for spec in list_model_specs():
        if spec.group in {"open-source", "expansion"}:
            assert spec.engine in {"sglang", "vllm"}
            assert spec.engine_fallback in {"sglang", "vllm"}
            assert spec.engine_fallback != spec.engine
