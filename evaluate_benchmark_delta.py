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
    merge_campaigns,
    run_aggregate,
    run_group,
    run_judge,
    run_model,
    run_track,
)
from gaze_vqa.pipeline.eval.review_workflow import (
    apply_review,
    build_review,
    review_status,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _campaign_path(repo_root: Path, campaign: str) -> Path:
    p = Path(campaign)
    if p.exists():
        return p.resolve()
    return (repo_root / "runs" / "eval_campaigns" / campaign).resolve()


def _campaign_output_path(repo_root: Path, campaign: str) -> Path:
    p = Path(campaign)
    if p.is_absolute() or p.parent != Path("."):
        return p.resolve()
    return (repo_root / "runs" / "eval_campaigns" / campaign).resolve()


def _runtime_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    runtime = {
        "retry": args.retry,
        "retry_backoff": args.retry_backoff,
        "request_interval": args.request_interval,
        "abort_on_fatal_api_error": (not args.no_abort_on_fatal_api_error),
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
    p.add_argument("--request_interval", type=float, default=0.0)
    p.add_argument("--no_abort_on_fatal_api_error", action="store_true", default=False)
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
    p_model.add_argument(
        "--reset_predictions",
        action="store_true",
        default=False,
        help="Delete existing predictions/meta for this model before running.",
    )
    _add_runtime_flags(p_model)

    p_track = sub.add_parser("run-track", help="Run all models in a track (A or B)")
    p_track.add_argument("--campaign", type=str, required=True)
    p_track.add_argument("--track", type=str, required=True, choices=["A", "B", "a", "b"])
    p_track.add_argument(
        "--reset_predictions",
        action="store_true",
        default=False,
        help="Delete existing predictions/meta for each model in this track before running.",
    )
    _add_runtime_flags(p_track)

    p_group = sub.add_parser("run-group", help="Run all models in a group (proprietary|oss|all)")
    p_group.add_argument("--campaign", type=str, required=True)
    p_group.add_argument("--group", type=str, required=True, choices=["proprietary", "oss", "all"])
    p_group.add_argument(
        "--reset_predictions",
        action="store_true",
        default=False,
        help="Delete existing predictions/meta for each model in this group before running.",
    )
    _add_runtime_flags(p_group)

    p_judge = sub.add_parser("judge", help="Run Gemini judge across model outputs in campaign")
    p_judge.add_argument("--campaign", type=str, required=True)
    p_judge.add_argument("--model_keys", type=str, nargs="*", default=None)
    p_judge.add_argument("--judge_model", type=str, default="gemini-3.1-pro-preview")
    p_judge.add_argument("--gemini_api_key", type=str, default="")
    p_judge.add_argument("--retry", type=int, default=3)
    p_judge.add_argument("--retry_backoff", type=float, default=2.0)
    p_judge.add_argument("--request_interval", type=float, default=0.0)

    p_agg = sub.add_parser("aggregate", help="Build table1/table2/leaderboard outputs")
    p_agg.add_argument("--campaign", type=str, required=True)

    p_merge = sub.add_parser("merge-campaigns", help="Merge reports from multiple campaigns into one output campaign")
    p_merge.add_argument("--source_campaigns", type=str, nargs="+", required=True)
    p_merge.add_argument("--output_campaign", type=str, required=True)
    p_merge.add_argument("--strict_manifest_match", dest="strict_manifest_match", action="store_true", default=True)
    p_merge.add_argument("--no_strict_manifest_match", dest="strict_manifest_match", action="store_false")

    p_build_review = sub.add_parser("build-review", help="Build static HTML reviewer for frozen/active GT manifest")
    p_build_review.add_argument("--campaign", type=str, required=True)
    p_build_review.add_argument("--page_size", type=int, default=200)
    p_build_review.add_argument("--thumb_long_edge", type=int, default=480)

    p_apply_review = sub.add_parser("apply-review", help="Apply review decisions and materialize reviewed GT manifest")
    p_apply_review.add_argument("--campaign", type=str, required=True)
    p_apply_review.add_argument("--decisions", type=Path, required=True)
    p_apply_review.add_argument("--policy", type=str, default="exclude_rejects", choices=["exclude_rejects"])
    p_apply_review.add_argument("--granularity", type=str, default="frame_bundle", choices=["frame_bundle"])
    p_apply_review.add_argument("--no_set_active", action="store_true", default=False)

    p_status = sub.add_parser("review-status", help="Show review and active-manifest status for a campaign")
    p_status.add_argument("--campaign", type=str, required=True)

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

    campaign_dir: Optional[Path] = None
    if hasattr(args, "campaign"):
        campaign_dir = _campaign_path(repo_root, args.campaign)

    if args.cmd == "run-model":
        assert campaign_dir is not None
        out = run_model(
            campaign_dir=campaign_dir,
            model_key=args.model_key,
            runtime=_runtime_from_args(args),
            model_override_file=args.model_override_file,
            reset_predictions=bool(args.reset_predictions),
        )
        print("[DONE] Model inference completed.")
        for k, v in out.items():
            print(f"  - {k}: {v}")
        return

    if args.cmd == "run-track":
        assert campaign_dir is not None
        outs = run_track(
            campaign_dir=campaign_dir,
            track=args.track.upper(),
            runtime=_runtime_from_args(args),
            model_override_file=args.model_override_file,
            reset_predictions=bool(args.reset_predictions),
        )
        print(f"[DONE] Track {args.track.upper()} inference completed for {len(outs)} models.")
        for row in outs:
            print(f"  - {row.get('model_key')}: newly_processed={row.get('newly_processed')} errors={row.get('errors_in_new_rows')}")
        return

    if args.cmd == "run-group":
        assert campaign_dir is not None
        outs = run_group(
            campaign_dir=campaign_dir,
            group=args.group,
            runtime=_runtime_from_args(args),
            model_override_file=args.model_override_file,
            reset_predictions=bool(args.reset_predictions),
        )
        print(f"[DONE] Group {args.group} inference completed for {len(outs)} models.")
        for row in outs:
            print(f"  - {row.get('model_key')}: newly_processed={row.get('newly_processed')} errors={row.get('errors_in_new_rows')}")
        return

    if args.cmd == "judge":
        assert campaign_dir is not None
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
        assert campaign_dir is not None
        out = run_aggregate(campaign_dir=campaign_dir)
        print("[DONE] Aggregation completed.")
        for k, v in out.items():
            if k != "summary":
                print(f"  - {k}: {v}")
        return

    if args.cmd == "merge-campaigns":
        source_dirs = [_campaign_path(repo_root, c) for c in args.source_campaigns]
        output_dir = _campaign_output_path(repo_root, args.output_campaign)
        out = merge_campaigns(
            source_campaign_dirs=source_dirs,
            output_campaign_dir=output_dir,
            strict_manifest_match=bool(args.strict_manifest_match),
        )
        print("[DONE] Campaign merge completed.")
        for k, v in out.items():
            if k != "summary":
                print(f"  - {k}: {v}")
        return

    if args.cmd == "build-review":
        assert campaign_dir is not None
        out = build_review(
            campaign_dir=campaign_dir,
            page_size=args.page_size,
            thumb_long_edge=args.thumb_long_edge,
        )
        print("[DONE] Review assets generated.")
        for k, v in out.items():
            print(f"  - {k}: {v}")
        return

    if args.cmd == "apply-review":
        assert campaign_dir is not None
        out = apply_review(
            campaign_dir=campaign_dir,
            decisions_path=args.decisions,
            policy=args.policy,
            granularity=args.granularity,
            set_active=not args.no_set_active,
        )
        print("[DONE] Review decisions applied.")
        for k, v in out.items():
            print(f"  - {k}: {v}")
        return

    if args.cmd == "review-status":
        assert campaign_dir is not None
        out = review_status(campaign_dir=campaign_dir)
        print("[DONE] Review status loaded.")
        for k, v in out.items():
            print(f"  - {k}: {v}")
        return

    raise ValueError(f"Unsupported command: {args.cmd}")


if __name__ == "__main__":
    main()
