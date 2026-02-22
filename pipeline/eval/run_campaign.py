from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from .aggregate_tables import aggregate_campaign
from .freeze_gt import freeze_gt, resolve_latest_benchmark
from .infer_runner import build_judge_input, run_inference_for_model
from .judge_gemini import judge_json
from .model_registry import (
    apply_model_overrides,
    get_model_spec,
    model_keys_for_group,
    model_keys_for_track,
    model_map,
)
from .schemas import CampaignMeta, file_sha256, read_json, utc_now_iso, write_json


DEFAULT_PROMPT_VERSION = "eval_prompt_v1"


def _campaigns_root(repo_root: Path) -> Path:
    return repo_root / "runs" / "eval_campaigns"


def _campaign_meta_path(campaign_dir: Path) -> Path:
    return campaign_dir / "campaign_meta.json"


def _resolve_manifest_candidate(campaign_dir: Path, raw_path: str) -> Path:
    p = Path(str(raw_path))
    if p.is_absolute():
        return p.resolve()
    return (campaign_dir / p).resolve()


def resolve_campaign_manifest_path(campaign_dir: Path, meta: Optional[Dict[str, Any]] = None) -> Path:
    campaign_dir = campaign_dir.resolve()
    meta = dict(meta or load_campaign_meta(campaign_dir))
    for key in ("active_manifest_path", "gt_manifest_path"):
        raw = str(meta.get(key) or "").strip()
        if not raw:
            continue
        p = _resolve_manifest_candidate(campaign_dir, raw)
        if p.exists():
            return p
    fallback = campaign_dir / "gt" / "gt_manifest_v1.jsonl"
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(f"Missing GT manifest under campaign: {campaign_dir}")


def save_campaign_meta(campaign_dir: Path, meta: Dict[str, Any]) -> Path:
    campaign_dir = campaign_dir.resolve()
    out = _campaign_meta_path(campaign_dir)
    write_json(out, meta)
    return out


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
        base_manifest_path=str(gt["manifest"]),
        active_manifest_path=str(gt["manifest"]),
        active_manifest_sha256=file_sha256(gt["manifest"]),
        prompt_version=prompt_version,
        model_map=model_map(),
        container_sif=container_sif,
        slurm={},
        review={
            "granularity": "frame_bundle",
            "policy": "exclude_rejects",
            "applied_at": None,
            "decisions_path": None,
            "decisions_sha256": None,
            "bundle_counts": {},
        },
        notes=notes or None,
    ).to_dict()
    save_campaign_meta(campaign_dir, meta)
    return campaign_dir


def load_campaign_meta(campaign_dir: Path) -> Dict[str, Any]:
    meta_path = _campaign_meta_path(campaign_dir.resolve())
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
    reset_predictions: bool = False,
) -> Dict[str, Any]:
    _ = load_campaign_meta(campaign_dir)  # ensure campaign exists
    runtime = dict(runtime or {})
    overrides = _model_overrides_from_file(model_override_file)
    spec = apply_model_overrides(get_model_spec(model_key), overrides=overrides)

    merged_runtime = {
        "retry": int(runtime.get("retry", 3)),
        "retry_backoff": float(runtime.get("retry_backoff", 2.0)),
        "request_interval": float(runtime.get("request_interval", 0.0)),
        "abort_on_fatal_api_error": bool(runtime.get("abort_on_fatal_api_error", True)),
        "server_timeout_s": float(runtime.get("server_timeout_s", 180.0)),
        "sglang_base_url": str(runtime.get("sglang_base_url", "http://127.0.0.1:30000")),
        "vllm_base_url": str(runtime.get("vllm_base_url", "http://127.0.0.1:8000")),
        "server_api_key": str(runtime.get("server_api_key", "")),
        # Treat empty CLI values as unset so env vars still work in batch jobs.
        "openai_api_key": str(runtime.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")),
        "gemini_api_key": str(runtime.get("gemini_api_key") or os.getenv("GEMINI_API_KEY", "")),
        "api_timeout_s": float(runtime.get("api_timeout_s", 180.0)),
    }
    return run_inference_for_model(
        campaign_dir=campaign_dir,
        model_spec=spec,
        runtime=merged_runtime,
        reset_predictions=bool(reset_predictions),
    )


def run_track(
    campaign_dir: Path,
    track: str,
    runtime: Optional[Dict[str, Any]] = None,
    model_override_file: Optional[Path] = None,
    reset_predictions: bool = False,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for key in model_keys_for_track(track):
        results.append(
            run_model(
                campaign_dir=campaign_dir,
                model_key=key,
                runtime=runtime,
                model_override_file=model_override_file,
                reset_predictions=bool(reset_predictions),
            )
        )
    return results


def run_group(
    campaign_dir: Path,
    group: str,
    runtime: Optional[Dict[str, Any]] = None,
    model_override_file: Optional[Path] = None,
    reset_predictions: bool = False,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for key in model_keys_for_group(group):
        results.append(
            run_model(
                campaign_dir=campaign_dir,
                model_key=key,
                runtime=runtime,
                model_override_file=model_override_file,
                reset_predictions=bool(reset_predictions),
            )
        )
    return results


def run_judge(
    campaign_dir: Path,
    model_keys: Optional[List[str]] = None,
    judge_model: str = "gemini-3.1-pro-preview",
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


def merge_campaigns(
    source_campaign_dirs: List[Path],
    output_campaign_dir: Path,
    strict_manifest_match: bool = True,
) -> Dict[str, Any]:
    source_dirs = [p.resolve() for p in source_campaign_dirs]
    if len(source_dirs) < 2:
        raise ValueError("merge_campaigns requires at least two source campaigns.")
    if len(set(source_dirs)) != len(source_dirs):
        raise ValueError("merge_campaigns received duplicate source campaign paths.")

    output_campaign_dir = output_campaign_dir.resolve()
    if output_campaign_dir in source_dirs:
        raise ValueError("Output campaign path must differ from source campaign paths.")
    output_campaign_dir.mkdir(parents=True, exist_ok=True)

    source_meta: Dict[Path, Dict[str, Any]] = {}
    manifest_by_campaign: Dict[Path, Path] = {}
    manifest_sha_by_campaign: Dict[Path, str] = {}
    for campaign in source_dirs:
        meta = load_campaign_meta(campaign)
        manifest = resolve_campaign_manifest_path(campaign_dir=campaign, meta=meta)
        source_meta[campaign] = meta
        manifest_by_campaign[campaign] = manifest
        manifest_sha_by_campaign[campaign] = file_sha256(manifest)

    unique_manifest_shas = sorted(set(manifest_sha_by_campaign.values()))
    if strict_manifest_match and len(unique_manifest_shas) != 1:
        details = ", ".join(f"{c}:{manifest_sha_by_campaign[c]}" for c in source_dirs)
        raise ValueError(f"Manifest mismatch across campaigns under strict mode: {details}")

    merged_model_map: Dict[str, Dict[str, Any]] = {}
    model_source_defs: Dict[str, Path] = {}
    for campaign in source_dirs:
        mm = dict(source_meta[campaign].get("model_map") or {})
        for model_key, model_spec in mm.items():
            existing = merged_model_map.get(model_key)
            if existing is not None and existing != model_spec:
                prev_source = model_source_defs.get(model_key)
                raise ValueError(
                    f"Conflicting model_map entries for '{model_key}' between "
                    f"{prev_source or 'unknown'} and {campaign}."
                )
            if existing is None:
                merged_model_map[model_key] = dict(model_spec)
                model_source_defs[model_key] = campaign

    model_sources: Dict[str, Path] = {}
    for campaign in source_dirs:
        pred_dir = campaign / "predictions"
        for pred_path in sorted(pred_dir.glob("*.jsonl")):
            model_key = pred_path.stem
            prior = model_sources.get(model_key)
            if prior is not None and prior != campaign:
                raise ValueError(
                    f"Duplicate prediction sources for model_key '{model_key}': {prior} and {campaign}."
                )
            model_sources[model_key] = campaign

    if not model_sources:
        raise ValueError("No prediction files found across source campaigns.")

    ref_source = source_dirs[0]
    ref_manifest = manifest_by_campaign[ref_source]
    ref_manifest_sha = manifest_sha_by_campaign[ref_source]

    gt_dir = output_campaign_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    merged_manifest = gt_dir / "gt_manifest_v1.jsonl"
    shutil.copy2(ref_manifest, merged_manifest)

    base_meta = dict(source_meta[ref_source])
    base_meta["campaign_name"] = output_campaign_dir.name
    base_meta["created_at"] = utc_now_iso()
    base_meta["model_map"] = merged_model_map
    base_meta["gt_manifest_path"] = str(merged_manifest.resolve())
    base_meta["gt_manifest_sha256"] = ref_manifest_sha
    base_meta["base_manifest_path"] = str(merged_manifest.resolve())
    base_meta["active_manifest_path"] = str(merged_manifest.resolve())
    base_meta["active_manifest_sha256"] = ref_manifest_sha
    base_meta["notes"] = (
        f"Merged from source campaigns at {utc_now_iso()}: "
        + ", ".join(str(p) for p in source_dirs)
    )
    base_meta["merge"] = {
        "source_campaigns": [str(p) for p in source_dirs],
        "strict_manifest_match": bool(strict_manifest_match),
        "manifest_sha256_by_campaign": {str(k): v for k, v in manifest_sha_by_campaign.items()},
    }
    save_campaign_meta(output_campaign_dir, base_meta)

    agg = aggregate_campaign(
        campaign_dir=output_campaign_dir,
        model_sources=model_sources,
        model_meta_override=merged_model_map,
        manifest_path_override=merged_manifest,
        summary_extra={
            "merged_sources": [str(p) for p in source_dirs],
            "strict_manifest_match": bool(strict_manifest_match),
            "manifest_sha256": ref_manifest_sha,
            "no_overlap_excluded_model_keys": ["gemini30flash"],
        },
    )

    merge_meta_path = output_campaign_dir / "reports" / "merge_meta.json"
    write_json(
        merge_meta_path,
        {
            "generated_at": utc_now_iso(),
            "output_campaign_dir": str(output_campaign_dir),
            "source_campaigns": [str(p) for p in source_dirs],
            "strict_manifest_match": bool(strict_manifest_match),
            "manifest_sha256_by_campaign": {str(k): v for k, v in manifest_sha_by_campaign.items()},
            "resolved_model_sources": {k: str(v) for k, v in sorted(model_sources.items())},
            "no_overlap_excluded_model_keys": ["gemini30flash"],
        },
    )
    agg["merge_meta_json"] = merge_meta_path
    agg["output_campaign_dir"] = output_campaign_dir
    return agg
