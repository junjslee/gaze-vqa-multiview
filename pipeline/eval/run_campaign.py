from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .aggregate_tables import aggregate_campaign
from .freeze_gt import freeze_gt, resolve_latest_benchmark
from .infer_runner import build_judge_input, run_inference_for_model
from .judge_gemini import judge_json
from .model_registry import apply_model_overrides, get_model_spec, model_keys_for_track, model_map
from .schemas import CampaignMeta, file_sha256, read_json, utc_now_iso, write_json


DEFAULT_PROMPT_VERSION = "eval_prompt_v1"


def _campaigns_root(repo_root: Path) -> Path:
    return repo_root / "runs" / "eval_campaigns"


def create_campaign(
    repo_root: Path,
    benchmark_path: Optional[Path] = None,
    campaign_name: str = "",
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    container_sif: str = "/work/nvme/bfga/jlee65/jun_fm.sif",
    notes: str = "",
) -> Path:
    repo_root = repo_root.resolve()
    benchmark = benchmark_path.resolve() if benchmark_path else resolve_latest_benchmark(repo_root)
    if not benchmark.exists():
        raise FileNotFoundError(f"Benchmark not found: {benchmark}")

    if not campaign_name.strip():
        campaign_name = f"campaign_{utc_now_iso().replace(':', '').replace('-', '')}"
    campaign_dir = _campaigns_root(repo_root) / campaign_name
    campaign_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = campaign_dir / "gt"
    gt = freeze_gt(benchmark_path=benchmark, out_dir=gt_dir, strict_image_exists=True)

    meta = CampaignMeta(
        campaign_name=campaign_name,
        created_at=utc_now_iso(),
        benchmark_path=str(benchmark),
        benchmark_sha256=file_sha256(benchmark),
        gt_manifest_path=str(gt["manifest"]),
        gt_manifest_sha256=file_sha256(gt["manifest"]),
        prompt_version=prompt_version,
        model_map=model_map(),
        container_sif=container_sif,
        slurm={},
        notes=notes or None,
    ).to_dict()
    write_json(campaign_dir / "campaign_meta.json", meta)
    return campaign_dir


def load_campaign_meta(campaign_dir: Path) -> Dict[str, Any]:
    meta_path = campaign_dir / "campaign_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing campaign metadata: {meta_path}")
    return read_json(meta_path)


def _model_overrides_from_file(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Model override file not found: {path}")
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError("Model override file must be a JSON object keyed by model_key.")
    return data


def run_model(
    campaign_dir: Path,
    model_key: str,
    runtime: Optional[Dict[str, Any]] = None,
    model_override_file: Optional[Path] = None,
) -> Dict[str, Any]:
    _ = load_campaign_meta(campaign_dir)  # ensure campaign exists
    runtime = dict(runtime or {})
    overrides = _model_overrides_from_file(model_override_file)
    spec = apply_model_overrides(get_model_spec(model_key), overrides=overrides)

    merged_runtime = {
        "retry": int(runtime.get("retry", 3)),
        "retry_backoff": float(runtime.get("retry_backoff", 2.0)),
        "server_timeout_s": float(runtime.get("server_timeout_s", 180.0)),
        "sglang_base_url": str(runtime.get("sglang_base_url", "http://127.0.0.1:30000")),
        "vllm_base_url": str(runtime.get("vllm_base_url", "http://127.0.0.1:8000")),
        "server_api_key": str(runtime.get("server_api_key", "")),
        "openai_api_key": str(runtime.get("openai_api_key", os.getenv("OPENAI_API_KEY", ""))),
        "gemini_api_key": str(runtime.get("gemini_api_key", os.getenv("GEMINI_API_KEY", ""))),
        "api_timeout_s": float(runtime.get("api_timeout_s", 180.0)),
    }
    return run_inference_for_model(campaign_dir=campaign_dir, model_spec=spec, runtime=merged_runtime)


def run_track(
    campaign_dir: Path,
    track: str,
    runtime: Optional[Dict[str, Any]] = None,
    model_override_file: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for key in model_keys_for_track(track):
        results.append(
            run_model(
                campaign_dir=campaign_dir,
                model_key=key,
                runtime=runtime,
                model_override_file=model_override_file,
            )
        )
    return results


def run_judge(
    campaign_dir: Path,
    model_keys: Optional[List[str]] = None,
    judge_model: str = "gemini-2.5-flash",
    api_key: str = "",
    retry: int = 3,
    retry_backoff: float = 2.0,
    request_interval: float = 0.0,
) -> List[Dict[str, Any]]:
    if model_keys is None:
        pred_dir = campaign_dir / "predictions"
        model_keys = [p.stem for p in sorted(pred_dir.glob("*.jsonl"))]
    out: List[Dict[str, Any]] = []
    for model_key in model_keys:
        judge_input = build_judge_input(campaign_dir=campaign_dir, model_key=model_key)
        judge_out = campaign_dir / "judge" / f"{model_key}.jsonl"
        meta = judge_json(
            input_path=judge_input,
            output_path=judge_out,
            model=judge_model,
            api_key=api_key,
            retry=retry,
            retry_backoff=retry_backoff,
            request_interval=request_interval,
            skip_existing=True,
        )
        out.append(meta)
    return out


def run_aggregate(campaign_dir: Path) -> Dict[str, Any]:
    return aggregate_campaign(campaign_dir)

