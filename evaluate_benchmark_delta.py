#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaze_vqa.pipeline.eval.run_campaign import (
    create_campaign,
    run_aggregate,
    run_judge,
    run_model,
    run_track,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _campaign_path(repo_root: Path, campaign: str) -> Path:
    p = Path(campaign)
    if p.exists():
        return p.resolve()
    return (repo_root / "runs" / "eval_campaigns" / campaign).resolve()


def _runtime_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    runtime = {
        "retry": args.retry,
        "retry_backoff": args.retry_backoff,
        "server_timeout_s": args.server_timeout_s,
        "sglang_base_url": args.sglang_base_url,
        "vllm_base_url": args.vllm_base_url,
        "server_api_key": args.server_api_key,
        "openai_api_key": args.openai_api_key,
        "gemini_api_key": args.gemini_api_key,
        "api_timeout_s": args.api_timeout_s,
    }
    return runtime


def _add_runtime_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--retry", type=int, default=3)
    p.add_argument("--retry_backoff", type=float, default=2.0)
    p.add_argument("--server_timeout_s", type=float, default=180.0)
    p.add_argument("--api_timeout_s", type=float, default=180.0)
    p.add_argument("--sglang_base_url", type=str, default="http://127.0.0.1:30000")
    p.add_argument("--vllm_base_url", type=str, default="http://127.0.0.1:8000")
    p.add_argument("--server_api_key", type=str, default="")
    p.add_argument("--openai_api_key", type=str, default="")
    p.add_argument("--gemini_api_key", type=str, default="")
    p.add_argument("--model_override_file", type=Path, default=None)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gaze-VQA evaluation launcher for Delta.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_freeze = sub.add_parser("freeze-gt", help="Freeze GT manifest from benchmark_gazevqa.json")
    p_freeze.add_argument("--benchmark_path", type=Path, default=None)
    p_freeze.add_argument("--campaign", type=str, default="")
    p_freeze.add_argument("--prompt_version", type=str, default="eval_prompt_v1")
    p_freeze.add_argument("--container_sif", type=str, default="/work/nvme/bfga/jlee65/jun_fm.sif")
    p_freeze.add_argument("--notes", type=str, default="")

    p_model = sub.add_parser("run-model", help="Run inference for one model in a campaign")
    p_model.add_argument("--campaign", type=str, required=True)
    p_model.add_argument("--model_key", type=str, required=True)
    _add_runtime_flags(p_model)

    p_track = sub.add_parser("run-track", help="Run all models in a track (A or B)")
    p_track.add_argument("--campaign", type=str, required=True)
    p_track.add_argument("--track", type=str, required=True, choices=["A", "B", "a", "b"])
    _add_runtime_flags(p_track)

    p_judge = sub.add_parser("judge", help="Run Gemini judge across model outputs in campaign")
    p_judge.add_argument("--campaign", type=str, required=True)
    p_judge.add_argument("--model_keys", type=str, nargs="*", default=None)
    p_judge.add_argument("--judge_model", type=str, default="gemini-2.5-flash")
    p_judge.add_argument("--gemini_api_key", type=str, default="")
    p_judge.add_argument("--retry", type=int, default=3)
    p_judge.add_argument("--retry_backoff", type=float, default=2.0)
    p_judge.add_argument("--request_interval", type=float, default=0.0)

    p_agg = sub.add_parser("aggregate", help="Build table1/table2/leaderboard outputs")
    p_agg.add_argument("--campaign", type=str, required=True)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = _repo_root()

    if args.cmd == "freeze-gt":
        campaign_dir = create_campaign(
            repo_root=repo_root,
            benchmark_path=args.benchmark_path,
            campaign_name=args.campaign,
            prompt_version=args.prompt_version,
            container_sif=args.container_sif,
            notes=args.notes,
        )
        print(f"[DONE] Campaign created: {campaign_dir}")
        return

    campaign_dir = _campaign_path(repo_root, args.campaign)

    if args.cmd == "run-model":
        out = run_model(
            campaign_dir=campaign_dir,
            model_key=args.model_key,
            runtime=_runtime_from_args(args),
            model_override_file=args.model_override_file,
        )
        print("[DONE] Model inference completed.")
        for k, v in out.items():
            print(f"  - {k}: {v}")
        return

    if args.cmd == "run-track":
        outs = run_track(
            campaign_dir=campaign_dir,
            track=args.track.upper(),
            runtime=_runtime_from_args(args),
            model_override_file=args.model_override_file,
        )
        print(f"[DONE] Track {args.track.upper()} inference completed for {len(outs)} models.")
        for row in outs:
            print(f"  - {row.get('model_key')}: newly_processed={row.get('newly_processed')} errors={row.get('errors_in_new_rows')}")
        return

    if args.cmd == "judge":
        outs = run_judge(
            campaign_dir=campaign_dir,
            model_keys=args.model_keys,
            judge_model=args.judge_model,
            api_key=args.gemini_api_key,
            retry=args.retry,
            retry_backoff=args.retry_backoff,
            request_interval=args.request_interval,
        )
        print(f"[DONE] Judge completed for {len(outs)} model outputs.")
        for row in outs:
            print(f"  - {row.get('output_path')}: judged_new={row.get('judged_new')} errors={row.get('errors')}")
        return

    if args.cmd == "aggregate":
        out = run_aggregate(campaign_dir=campaign_dir)
        print("[DONE] Aggregation completed.")
        for k, v in out.items():
            if k != "summary":
                print(f"  - {k}: {v}")
        return

    raise ValueError(f"Unsupported command: {args.cmd}")


if __name__ == "__main__":
    main()
