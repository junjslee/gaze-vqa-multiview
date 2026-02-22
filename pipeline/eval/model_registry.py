from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    group: str  # proprietary | open-source | expansion
    track: str  # A | B | A+B
    model_id: str
    engine: str  # openai_api | gemini_api | sglang | vllm
    engine_fallback: str = ""
    tensor_parallel_size: int = 1
    gpus: int = 1
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)
    include_in_default_lists: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _env_model_id(default: str, *env_names: str) -> str:
    for name in env_names:
        value = str(os.getenv(name, "")).strip()
        if value:
            return value
    return default


_MODEL_SPECS: List[ModelSpec] = [
    # Track A: proprietary APIs
    ModelSpec(
        key="gpt41",
        label="GPT-4.1",
        group="proprietary",
        track="A",
        model_id="gpt-4.1",
        engine="openai_api",
    ),
    ModelSpec(
        key="gpt4o",
        label="GPT-4o",
        group="proprietary",
        track="A",
        model_id="gpt-4o",
        engine="openai_api",
    ),
    ModelSpec(
        key="gemini20flash",
        label="Gemini-2.0-Flash",
        group="proprietary",
        track="A",
        model_id=_env_model_id(
            "gemini-2.0-flash",
            "EVAL_GEMINI20FLASH_MODEL_ID",
        ),
        engine="gemini_api",
    ),
    ModelSpec(
        key="gemini25flash",
        label="Gemini-2.5-Flash",
        group="proprietary",
        track="A",
        model_id=_env_model_id(
            "gemini-2.5-flash",
            "EVAL_GEMINI25FLASH_MODEL_ID",
        ),
        engine="gemini_api",
    ),
    ModelSpec(
        key="gemini30flash",
        label="Gemini-3-Flash-Preview",
        group="proprietary",
        track="A",
        model_id=_env_model_id(
            "gemini-3-flash-preview",
            "EVAL_GEMINI30FLASH_MODEL_ID",
        ),
        engine="gemini_api",
    ),
    ModelSpec(
        key="gemini25pro",
        label="Gemini-2.5-Pro",
        group="proprietary",
        track="A",
        model_id=_env_model_id(
            "gemini-2.5-pro",
            "EVAL_GEMINI25PRO_MODEL_ID",
        ),
        engine="gemini_api",
    ),
    ModelSpec(
        key="gemini30pro",
        label="Gemini-3-Pro-Preview",
        group="proprietary",
        track="A",
        model_id=_env_model_id(
            "gemini-3-pro-preview",
            "EVAL_GEMINI30PRO_MODEL_ID",
        ),
        engine="gemini_api",
    ),
    # Legacy aliases: keep callable, but exclude from default track/group execution.
    ModelSpec(
        key="geminiflashlatest",
        label="Gemini-Flash-Latest (legacy alias)",
        group="proprietary",
        track="A",
        model_id=_env_model_id(
            "gemini-2.5-flash",
            "EVAL_GEMINIFLASHLATEST_MODEL_ID",
            "EVAL_GEMINI25FLASH_MODEL_ID",
        ),
        engine="gemini_api",
        include_in_default_lists=False,
    ),
    ModelSpec(
        key="gemini31propreview",
        label="Gemini-3.1-Pro-Preview (legacy alias)",
        group="proprietary",
        track="A",
        model_id=_env_model_id(
            "gemini-3.1-pro-preview",
            "EVAL_GEMINI31PRO_MODEL_ID",
        ),
        engine="gemini_api",
        include_in_default_lists=False,
    ),
    # Track A: open source (paper comparable)
    ModelSpec(
        key="qwen3vl_alias",
        label="Qwen3-VL (paper alias)",
        group="open-source",
        track="A",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        engine="sglang",
        engine_fallback="vllm",
        extra={"prompt_mode": "image_multi"},
    ),
    ModelSpec(
        key="internvl25",
        label="InternVL2.5",
        group="open-source",
        track="A",
        model_id="OpenGVLab/InternVL2_5-8B",
        engine="vllm",
        engine_fallback="sglang",
        extra={"prompt_mode": "image_multi"},
    ),
    ModelSpec(
        key="deepseekvl2",
        label="DeepSeek-VL2",
        group="open-source",
        track="A",
        model_id="deepseek-ai/deepseek-vl2",
        engine="sglang",
        engine_fallback="vllm",
        extra={"prompt_mode": "image_multi"},
    ),
    ModelSpec(
        key="llava_video_7b",
        label="LLaVA-Video-7B",
        group="open-source",
        track="A+B",
        model_id="lmms-lab/LLaVA-Video-7B-Qwen2",
        engine="sglang",
        engine_fallback="vllm",
        extra={"prompt_mode": "video_synth"},
    ),
    ModelSpec(
        key="llava_ov_7b",
        label="LLaVA-OneVision-7B",
        group="open-source",
        track="A+B",
        model_id="lmms-lab/llava-onevision-qwen2-7b-ov",
        engine="sglang",
        engine_fallback="vllm",
        extra={"prompt_mode": "image_multi"},
    ),
    # Track B expansion
    ModelSpec(
        key="llava_video_72b",
        label="LLaVA-Video-72B",
        group="expansion",
        track="B",
        model_id="lmms-lab/LLaVA-Video-72B-Qwen2",
        engine="sglang",
        engine_fallback="vllm",
        tensor_parallel_size=4,
        gpus=4,
        extra={"prompt_mode": "video_synth"},
    ),
    ModelSpec(
        key="llava_ov_72b",
        label="LLaVA-OneVision-72B",
        group="expansion",
        track="B",
        model_id="lmms-lab/llava-onevision-qwen2-72b-ov-sft",
        engine="sglang",
        engine_fallback="vllm",
        tensor_parallel_size=4,
        gpus=4,
        extra={"prompt_mode": "image_multi"},
    ),
    ModelSpec(
        key="llava_next_8b",
        label="LLaVA-NeXT-8B",
        group="expansion",
        track="B",
        model_id="lmms-lab/llama3-llava-next-8b",
        engine="sglang",
        engine_fallback="vllm",
        extra={"prompt_mode": "image_multi", "chat_template": "chatml-llava"},
    ),
    ModelSpec(
        key="llava_next_32b",
        label="LLaVA-NeXT-32B",
        group="expansion",
        track="B",
        model_id="lmms-lab/llava-next-qwen-32b",
        engine="sglang",
        engine_fallback="vllm",
        tensor_parallel_size=2,
        gpus=2,
        extra={"prompt_mode": "image_multi", "chat_template": "chatml-llava"},
    ),
]


def list_model_specs() -> List[ModelSpec]:
    return list(_MODEL_SPECS)


def get_model_spec(model_key: str) -> ModelSpec:
    for spec in _MODEL_SPECS:
        if spec.key == model_key:
            return spec
    raise KeyError(f"Unknown model key: {model_key}")


def model_map() -> Dict[str, Dict[str, Any]]:
    return {spec.key: spec.to_dict() for spec in _MODEL_SPECS}


def model_keys_for_track(track: str) -> List[str]:
    track = track.upper().strip()
    if track not in {"A", "B"}:
        raise ValueError(f"Unsupported track: {track}")
    keys: List[str] = []
    for spec in _MODEL_SPECS:
        if not spec.include_in_default_lists:
            continue
        if spec.track == track or spec.track == "A+B":
            keys.append(spec.key)
    return keys


def model_keys_for_group(group: str) -> List[str]:
    g = group.strip().lower()
    if g not in {"proprietary", "oss", "all"}:
        raise ValueError(f"Unsupported group: {group}")
    defaults = [spec for spec in _MODEL_SPECS if spec.include_in_default_lists]
    if g == "all":
        return [spec.key for spec in defaults]
    if g == "proprietary":
        return [spec.key for spec in defaults if spec.group == "proprietary"]
    return [spec.key for spec in defaults if spec.group in {"open-source", "expansion"}]


def apply_model_overrides(spec: ModelSpec, overrides: Dict[str, Any]) -> ModelSpec:
    if not overrides:
        return spec
    if spec.key not in overrides:
        return spec
    patch = dict(overrides.get(spec.key) or {})
    if not patch:
        return spec

    values = spec.to_dict()
    values.update(patch)
    if "extra" in patch and isinstance(patch["extra"], dict):
        merged_extra = dict(spec.extra)
        merged_extra.update(patch["extra"])
        values["extra"] = merged_extra
    return ModelSpec(**values)
