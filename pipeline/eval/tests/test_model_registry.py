from __future__ import annotations

from gaze_vqa.pipeline.eval.model_registry import (
    apply_model_overrides,
    get_model_spec,
    model_keys_for_track,
)


def test_track_model_lists() -> None:
    a = model_keys_for_track("A")
    b = model_keys_for_track("B")
    assert "gpt41" in a
    assert "llava_video_72b" in b
    assert "llava_video_7b" in a and "llava_video_7b" in b


def test_model_override() -> None:
    spec = get_model_spec("gpt41")
    patched = apply_model_overrides(spec, {"gpt41": {"model_id": "gpt-4.1-mini"}})
    assert patched.model_id == "gpt-4.1-mini"
    assert patched.key == spec.key

