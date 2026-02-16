from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .engines.sglang_client import SGLangClient
from .engines.vllm_client import VLLMClient
from .model_registry import ModelSpec
from .providers.gemini_api import infer_gemini_multimodal
from .providers.openai_api import infer_openai_multimodal
from .schemas import PredictionRecord, append_jsonl, iter_jsonl, read_json, utc_now_iso, write_json

SYSTEM_PROMPT = (
    "You are an expert assistant for multi-camera gaze and accessibility reasoning. "
    "Use only the provided images and camera IDs to answer. Reference camera IDs explicitly. "
    "If information is insufficient, say 'I am not sure' and explain briefly."
)

PROMPT_TEMPLATE = """Image order and camera IDs:
{camera_section}

Task type: {task_type}
Question: {question}

Respond in no more than three sentences."""


def _guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    return "application/octet-stream"


def _image_to_data_url(path: Path) -> str:
    with path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{_guess_mime(path)};base64,{b64}"


def _build_prompt(sample: Dict[str, Any]) -> str:
    camera_ids = sample.get("camera_ids") or []
    frame = sample.get("frame", "unknown")
    section = "\n".join(
        f"- Image {i + 1}: {cam} (frame {frame})"
        for i, cam in enumerate(camera_ids)
    )
    if not section:
        paths = sample.get("image_paths") or []
        section = "\n".join(f"- Image {i + 1}: {Path(p).name}" for i, p in enumerate(paths))
    return PROMPT_TEMPLATE.format(
        camera_section=section,
        task_type=sample.get("task_type", "unknown"),
        question=sample.get("question", "N/A"),
    )


def _build_chat_messages(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    prompt = _build_prompt(sample)
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in sample.get("image_paths") or []:
        p = Path(img)
        if not p.exists():
            continue
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(p)}})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def _processed_ids(predictions_path: Path) -> set[str]:
    done = set()
    for row in iter_jsonl(predictions_path):
        uid = row.get("sample_uid")
        if uid:
            done.add(str(uid))
    return done


def _choose_engine_client(engine: str, base_url: str, api_key: str, timeout_s: float):
    if engine == "sglang":
        return SGLangClient(base_url=base_url, api_key=api_key, timeout_s=timeout_s)
    if engine == "vllm":
        return VLLMClient(base_url=base_url, api_key=api_key, timeout_s=timeout_s)
    raise ValueError(f"Unsupported engine client type: {engine}")


def _engine_infer(
    client: Any,
    model_id: str,
    sample: Dict[str, Any],
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, int], str]:
    try:
        text, tokens, _raw = client.chat_completion(
            model=model_id,
            messages=_build_chat_messages(sample),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return text, tokens, ""
    except Exception as exc:
        return "", {"tokens_in": 0, "tokens_out": 0}, f"{type(exc).__name__}: {exc}"


def _api_infer(
    spec: ModelSpec,
    sample: Dict[str, Any],
    runtime: Dict[str, Any],
) -> Tuple[str, Dict[str, int], str]:
    kwargs = dict(
        model=spec.model_id,
        prompt=_build_prompt(sample),
        image_paths=list(sample.get("image_paths") or []),
        temperature=float(spec.temperature),
        top_p=float(spec.top_p),
        max_output_tokens=int(spec.max_new_tokens),
        retry=int(runtime.get("retry", 3)),
        retry_backoff=float(runtime.get("retry_backoff", 2.0)),
        system_prompt=SYSTEM_PROMPT,
    )
    if spec.engine == "openai_api":
        kwargs["api_key"] = runtime.get("openai_api_key", "")
        kwargs["timeout_s"] = float(runtime.get("api_timeout_s", 180.0))
        return infer_openai_multimodal(**kwargs)
    if spec.engine == "gemini_api":
        kwargs["api_key"] = runtime.get("gemini_api_key", "")
        return infer_gemini_multimodal(**kwargs)
    raise ValueError(f"Unsupported API engine: {spec.engine}")


def run_inference_for_model(
    campaign_dir: Path,
    model_spec: ModelSpec,
    runtime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    runtime = runtime or {}
    manifest_path = campaign_dir / "gt" / "gt_manifest_v1.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing frozen GT manifest: {manifest_path}")

    predictions_dir = campaign_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pred_path = predictions_dir / f"{model_spec.key}.jsonl"
    run_meta_path = predictions_dir / f"{model_spec.key}.meta.json"

    manifest = _load_manifest(manifest_path)
    processed = _processed_ids(pred_path)

    sglang_url = str(runtime.get("sglang_base_url", "http://127.0.0.1:30000")).strip()
    vllm_url = str(runtime.get("vllm_base_url", "http://127.0.0.1:8000")).strip()
    server_api_key = str(runtime.get("server_api_key", "")).strip()
    timeout_s = float(runtime.get("server_timeout_s", 180.0))

    primary_client = None
    fallback_client = None
    if model_spec.engine in {"sglang", "vllm"}:
        primary_url = sglang_url if model_spec.engine == "sglang" else vllm_url
        primary_client = _choose_engine_client(
            engine=model_spec.engine,
            base_url=primary_url,
            api_key=server_api_key,
            timeout_s=timeout_s,
        )
        if model_spec.engine_fallback in {"sglang", "vllm"}:
            fallback_url = sglang_url if model_spec.engine_fallback == "sglang" else vllm_url
            fallback_client = _choose_engine_client(
                engine=model_spec.engine_fallback,
                base_url=fallback_url,
                api_key=server_api_key,
                timeout_s=timeout_s,
            )

    # Preflight availability checks to avoid long per-sample timeouts.
    unavailable_reason = ""
    if model_spec.engine == "openai_api" and not str(runtime.get("openai_api_key", "")).strip():
        unavailable_reason = "missing_openai_api_key"
    elif model_spec.engine == "gemini_api" and not str(runtime.get("gemini_api_key", "")).strip():
        unavailable_reason = "missing_gemini_api_key"
    elif model_spec.engine in {"sglang", "vllm"}:
        primary_ok = bool(primary_client and primary_client.healthcheck())
        fallback_ok = bool(fallback_client and fallback_client.healthcheck())
        if not primary_ok and not fallback_ok:
            unavailable_reason = f"unhealthy_server:{model_spec.engine}"
    if unavailable_reason:
        meta = {
            "model_key": model_spec.key,
            "model_label": model_spec.label,
            "model_id": model_spec.model_id,
            "engine": model_spec.engine,
            "engine_fallback": model_spec.engine_fallback,
            "track": model_spec.track,
            "group": model_spec.group,
            "total_manifest_samples": len(manifest),
            "predictions_file": str(pred_path),
            "newly_processed": 0,
            "errors_in_new_rows": 0,
            "elapsed_s": 0.0,
            "runtime": runtime,
            "unavailable_reason": unavailable_reason,
            "updated_at": utc_now_iso(),
        }
        write_json(run_meta_path, meta)
        return meta

    total = len(manifest)
    newly_processed = 0
    error_count = 0
    start = time.time()

    for sample in manifest:
        uid = str(sample.get("sample_uid") or "")
        if not uid or uid in processed:
            continue

        t0 = time.time()
        answer = ""
        error = ""
        tokens = {"tokens_in": 0, "tokens_out": 0}

        try:
            if model_spec.engine in {"openai_api", "gemini_api"}:
                answer, tokens, error = _api_infer(model_spec, sample, runtime)
            else:
                answer, tokens, error = _engine_infer(
                    client=primary_client,
                    model_id=model_spec.model_id,
                    sample=sample,
                    temperature=model_spec.temperature,
                    top_p=model_spec.top_p,
                    max_tokens=model_spec.max_new_tokens,
                )
                if error and fallback_client is not None:
                    fb_answer, fb_tokens, fb_error = _engine_infer(
                        client=fallback_client,
                        model_id=model_spec.model_id,
                        sample=sample,
                        temperature=model_spec.temperature,
                        top_p=model_spec.top_p,
                        max_tokens=model_spec.max_new_tokens,
                    )
                    if not fb_error:
                        answer, tokens, error = fb_answer, fb_tokens, ""
                    else:
                        error = f"{error} | fallback:{fb_error}"
        except Exception as exc:
            answer = ""
            tokens = {"tokens_in": 0, "tokens_out": 0}
            error = f"{type(exc).__name__}: {exc}"

        if not answer and not error:
            error = "empty_answer"

        row = PredictionRecord(
            sample_uid=uid,
            model_key=model_spec.key,
            model_id=model_spec.model_id,
            inference_answer=answer,
            error=error,
            latency_s=round(time.time() - t0, 4),
            tokens_in=int(tokens.get("tokens_in", 0)),
            tokens_out=int(tokens.get("tokens_out", 0)),
            timestamp=utc_now_iso(),
        ).to_dict()
        append_jsonl(pred_path, row)
        processed.add(uid)
        newly_processed += 1
        if error:
            error_count += 1

    elapsed = time.time() - start
    meta = {
        "model_key": model_spec.key,
        "model_label": model_spec.label,
        "model_id": model_spec.model_id,
        "engine": model_spec.engine,
        "engine_fallback": model_spec.engine_fallback,
        "track": model_spec.track,
        "group": model_spec.group,
        "total_manifest_samples": total,
        "predictions_file": str(pred_path),
        "newly_processed": newly_processed,
        "errors_in_new_rows": error_count,
        "elapsed_s": round(elapsed, 3),
        "runtime": runtime,
        "updated_at": utc_now_iso(),
    }
    write_json(run_meta_path, meta)
    return meta


def merge_predictions_for_samples(
    samples: Iterable[Dict[str, Any]],
    pred_rows: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_uid = {str(s["sample_uid"]): dict(s) for s in samples}
    merged: List[Dict[str, Any]] = []
    for p in pred_rows:
        uid = str(p.get("sample_uid") or "")
        s = by_uid.get(uid)
        if not s:
            continue
        row = {
            "sample_uid": uid,
            "task_id": s.get("task_id"),
            "task_type": s.get("task_type"),
            "scene": s.get("scene"),
            "question": s.get("question"),
            "groundtruth_answer": s.get("groundtruth_answer"),
            "inference_answer": p.get("inference_answer", ""),
            "model_key": p.get("model_key", ""),
            "model_id": p.get("model_id", ""),
            "error": p.get("error", ""),
        }
        merged.append(row)
    return merged


def build_judge_input(campaign_dir: Path, model_key: str) -> Path:
    manifest_path = campaign_dir / "gt" / "gt_manifest_v1.jsonl"
    pred_path = campaign_dir / "predictions" / f"{model_key}.jsonl"
    out_path = campaign_dir / "judge" / f"{model_key}.judge_input.json"

    samples = list(iter_jsonl(manifest_path))
    preds = list(iter_jsonl(pred_path))
    merged = merge_predictions_for_samples(samples=samples, pred_rows=preds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, merged)
    return out_path
