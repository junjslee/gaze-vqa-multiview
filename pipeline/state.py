import os
import json
import argparse
import time
import random
import logging
import sys
import re
from pathlib import Path
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def _strip_wrapped_quotes(s):
    s = str(s).strip()
    if len(s) >= 2 and ((s[0] == "'" and s[-1] == "'") or (s[0] == '"' and s[-1] == '"')):
        return s[1:-1]
    return s


def _load_env_file(path: Path):
    try:
        text = path.read_text()
    except Exception:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        val = val.strip()
        if val and val[0] in ("'", '"'):
            val = _strip_wrapped_quotes(val)
        elif " #" in val:
            val = val.split(" #", 1)[0].strip()
        os.environ.setdefault(key, val)


def _bootstrap_env():
    # Load .env before argparse defaults are materialized.
    candidates = []
    env_file = os.environ.get("GAZEVQA_ENV_FILE", "").strip()
    if env_file:
        candidates.append(Path(env_file).expanduser())
    candidates.append(Path.cwd() / ".env")
    candidates.append(Path(__file__).resolve().parents[1] / ".env")

    seen = set()
    for p in candidates:
        rp = str(p.resolve()) if p.exists() else str(p)
        if rp in seen:
            continue
        seen.add(rp)
        if p.exists():
            _load_env_file(p)


_bootstrap_env()


def parse_args():
    """Parse CLI flags grouped by purpose for easier modification."""
    p = argparse.ArgumentParser(
        description="MVGT → Gaze-VQA benchmark builder (Haozhen-style v7.4) [Single-process]"
    )

    # -------------------------------------------------------------------------
    # Run paths / naming
    # -------------------------------------------------------------------------
    g_paths = p.add_argument_group("Run paths / naming")
    # Root output directory. Run artifacts go to out_root/runs/<run_name>/.
    g_paths.add_argument("--out_root", type=str,
                         default=os.environ.get("GAZEVQA_OUT_ROOT", "/work/nvme/bfga/jlee65/gaze_vqa"),
                         help="Project root. Run artifacts go to out_root/runs/<run_name>/")
    # Optional run name. If empty, a timestamped name is used.
    g_paths.add_argument("--run_name", type=str, default="",
                         help="Optional run name. Default: auto timestamp.")

    # -------------------------------------------------------------------------
    # Dataset inputs
    # -------------------------------------------------------------------------
    g_data = p.add_argument_group("Dataset inputs")
    # MVGT zip path (required).
    g_data.add_argument("--mvgt_zip", type=str,
                        default=os.environ.get("MVGT_ZIP_PATH", "/work/nvme/bfga/jlee65/gaze_vqa/MVGT_Dataset.zip"),
                        help="Path to MVGT_Dataset.zip (required).")
    # Fallback download URL if zip is missing.
    g_data.add_argument("--mvgt_url", type=str,
                        default=os.environ.get(
                            "MVGT_URL",
                            "https://www3.cs.stonybrook.edu/~cvl/content/datasets/MVGT/MVGT_Dataset.zip",
                        ),
                        help="MVGT dataset URL for fallback download if mvgt_zip is missing.")

    # -------------------------------------------------------------------------
    # External models (SAM2 + VLM)
    # -------------------------------------------------------------------------
    g_models = p.add_argument_group("External models (SAM2 + VLM)")
    # SAM2 repo path (cloned if missing).
    g_models.add_argument("--sam2_repo", type=str,
                          default=os.environ.get(
                              "SAM2_REPO_DIR", "/work/nvme/bfga/jlee65/gaze_vqa/third_party/sam2_repo"
                          ),
                          help="SAM2 repo dir.")
    # SAM2 checkpoint path (downloaded if missing).
    g_models.add_argument("--sam2_ckpt", type=str,
                          default=os.environ.get(
                              "SAM2_CHECKPOINT", "/work/nvme/bfga/jlee65/gaze_vqa/checkpoints/sam2_hiera_large.pt"
                          ),
                          help="SAM2 checkpoint path.")
    # Qwen model id for VLM.
    g_models.add_argument("--qwen_model", type=str,
                          default=os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct"),
                          help="VLM model id for Qwen (e.g., Qwen/Qwen2.5-VL-7B-Instruct).")
    # VLM provider routing.
    g_models.add_argument("--vlm_provider", type=str,
                          default=os.environ.get("VLM_PROVIDER", "qwen").strip().lower(),
                          choices=["qwen", "openai", "gemini"],
                          help="Provider used for all VLM calls: qwen, openai, or gemini.")
    # Optional provider model override. If empty, provider-specific defaults are used.
    g_models.add_argument("--vlm_model", type=str,
                          default=os.environ.get("VLM_MODEL", ""),
                          help="Model name for selected provider (optional override).")
    g_models.add_argument("--openai_base_url", type=str,
                          default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                          help="OpenAI API base URL.")
    g_models.add_argument("--gemini_api_base", type=str,
                          default=os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com"),
                          help="Gemini API base URL.")
    g_models.add_argument("--gemini_thinking_budget", type=int,
                          default=int(os.environ.get("GEMINI_THINKING_BUDGET", "0")),
                          help="Gemini thinking budget (tokens). Use 0 to disable thinking for short-label stability.")
    g_models.add_argument("--vlm_timeout_s", type=float,
                          default=float(os.environ.get("VLM_TIMEOUT_S", "90")),
                          help="Timeout (seconds) for external API VLM calls.")
    # Optional pricing overrides used by usage-cost reporting.
    g_models.add_argument("--openai_price_input_per_1m", type=float,
                          default=float(os.environ.get("OPENAI_PRICE_INPUT_PER_1M", "2.5")),
                          help="OpenAI input token price (USD per 1M tokens) for usage report.")
    g_models.add_argument("--openai_price_cached_input_per_1m", type=float,
                          default=float(os.environ.get("OPENAI_PRICE_CACHED_INPUT_PER_1M", "1.25")),
                          help="OpenAI cached-input token price (USD per 1M tokens) for usage report.")
    g_models.add_argument("--openai_price_output_per_1m", type=float,
                          default=float(os.environ.get("OPENAI_PRICE_OUTPUT_PER_1M", "10.0")),
                          help="OpenAI output token price (USD per 1M tokens) for usage report.")
    g_models.add_argument("--gemini_price_input_per_1m", type=float,
                          default=float(os.environ.get("GEMINI_PRICE_INPUT_PER_1M", "0.30")),
                          help="Gemini input token price (USD per 1M tokens) for usage report.")
    g_models.add_argument("--gemini_price_output_per_1m", type=float,
                          default=float(os.environ.get("GEMINI_PRICE_OUTPUT_PER_1M", "2.50")),
                          help="Gemini output token price (USD per 1M tokens) for usage report.")
    # Disable VLM entirely (debug only).
    g_models.add_argument("--skip_vlm", action="store_true",
                          help="Skip loading VLM and any VLM calls (debug only; low quality / empty samples).")

    # -------------------------------------------------------------------------
    # Targets / quotas
    # -------------------------------------------------------------------------
    g_targets = p.add_argument_group("Targets / quotas")
    # Targets for each task.
    g_targets.add_argument("--targets", type=int, nargs=4, default=[60, 60, 60, 60],
                           help="Targets for tasks 1..4 (e.g., --targets 60 60 60 60).")

    # -------------------------------------------------------------------------
    # Run scope / gating
    # -------------------------------------------------------------------------
    g_scope = p.add_argument_group("Run scope / gating")
    # Cap frames processed (0 = no cap).
    g_scope.add_argument("--max_frames", type=int, default=200,
                         help="Optional cap on frames processed (0 = no cap).")
    # Cap sequences processed (0 = no cap).
    g_scope.add_argument("--max_sequences", type=int, default=0,
                         help="Optional cap on sequences processed (0 = no cap).")
    # Convenience flag: run full dataset.
    g_scope.add_argument("--full_dataset", action="store_true", default=False,
                         help="Process the full dataset (sets max_frames=0 and max_sequences=0).")
    # Require all 4 tasks per accepted frame.
    g_scope.add_argument("--require_all_tasks", action="store_true", default=True,
                         help="Require all four tasks per accepted frame (default: on).")
    # Allow partial acceptance (overrides require_all_tasks).
    g_scope.add_argument("--allow_partial_tasks", action="store_true", default=False,
                         help="Allow partial task acceptance per frame (overrides require_all_tasks).")
    # Task1 confidence threshold; reject if below (0 disables).
    g_scope.add_argument("--task1_conf_threshold", type=float, default=0.0,
                         help="Reject Task1 if confidence < threshold (0 disables).")
    # Scan dataset stats before processing.
    g_scope.add_argument("--scan_dataset_stats", action="store_true", default=True,
                         help="Scan full dataset to report total frames/points (default: on).")
    g_scope.add_argument("--no_scan_dataset_stats", action="store_true", default=False,
                         help="Disable full dataset stats scan.")

    # -------------------------------------------------------------------------
    # Performance / memory
    # -------------------------------------------------------------------------
    g_perf = p.add_argument_group("Performance / memory")
    # Resize images for all processing.
    g_perf.add_argument("--resize_wh", type=int, nargs=2, default=[1000, 750], help="Resize W H")
    # Max views passed to VLM verifier.
    g_perf.add_argument("--max_views", type=int, default=6, help="Max number of views to feed to VLM verifier.")
    # Thread count for IO.
    g_perf.add_argument("--thread_io", type=int, default=1, help="Threads for image save/read IO.")
    # CUDA cache empty cadence.
    g_perf.add_argument("--empty_cache_every", type=int, default=60, help="torch.cuda.empty_cache() cadence.")

    # -------------------------------------------------------------------------
    # Debug / logging
    # -------------------------------------------------------------------------
    g_debug = p.add_argument_group("Debug / logging")
    # Save per-frame debug artifacts (images + HTML).
    g_debug.add_argument("--save_debug", action="store_true", default=False,
                         help="Save debug artifacts (overlays, masks, crops).")
    # Save debug for every N Task1 accepts.
    g_debug.add_argument("--debug_every_n_task1", type=int, default=3,
                         help="Save debug for every N Task1 accepts.")
    # Early warning if Task1 accept rate is low.
    g_debug.add_argument("--early_warn_frames", type=int, default=250,
                         help="After this many frames, warn if Task1 accept rate is too low.")
    g_debug.add_argument("--min_task1_accept_rate", type=float, default=0.01,
                         help="If accept rate < this, print big warning.")
    # Log prompts and VLM outputs to stdout.
    g_debug.add_argument("--log_prompts", action="store_true", default=False,
                         help="Log VLM prompts and outputs to stdout.")
    # Extra debug dumps for SAM2 overlap rejects.
    g_debug.add_argument("--dump_overlap_debug", action="store_true", default=False,
                         help="When save_debug is enabled, dump debug images for SAM2 overlap rejects.")

    # -------------------------------------------------------------------------
    # Reasoning mode / task-specific toggles
    # -------------------------------------------------------------------------
    g_task = p.add_argument_group("Reasoning mode / task toggles")
    # Global reasoning mode (Task2/4 use this).
    g_task.add_argument("--reasoning_mode", type=str, default="gt",
                        choices=["gt", "vlm"],
                        help="Reasoning source: gt (deterministic from annotations/geometry) or vlm (generate with Qwen).")
    # Task1 reasoning mode (separate from global).
    g_task.add_argument("--task1_reasoning_mode", type=str, default="gt",
                        choices=["gt", "vlm"],
                        help="Task1 reasoning style: gt (template) or vlm (spatial/cognitive).")
    # Task1 semantic label arbiter (disagreement-only CoT-style verifier/refiner).
    g_task.add_argument("--task1_semantic_arbiter", action="store_true", default=False,
                        help="Enable Task1 semantic arbiter for disagreement/borderline label refinement.")
    # Short alias for convenience in runs.
    g_task.add_argument("--cot", action="store_true", default=False,
                        help="Alias for --task1_semantic_arbiter.")
    # Task1 hybrid guardrail: keep base labeling provider and run a secondary arbiter model on hard cases.
    g_task.add_argument("--task1_hybrid_guardrail", action="store_true", default=False,
                        help="Enable Task1 hybrid guardrail (scene-aware arbiter + optional constrained Qwen refine).")
    g_task.add_argument("--task1_guardrail_provider", type=str, default="gemini",
                        choices=["gemini", "openai", "qwen"],
                        help="Provider used only for Task1 hybrid guardrail arbitration.")
    g_task.add_argument("--task1_guardrail_model", type=str, default="",
                        help="Optional model override for Task1 guardrail provider.")
    g_task.add_argument("--task1_guardrail_min_conf", type=str, default="MEDIUM",
                        choices=["LOW", "MEDIUM", "HIGH"],
                        help="Minimum guardrail confidence before allowing label switch (unless current label is ambiguous).")
    g_task.add_argument("--task1_guardrail_max_new_tokens", type=int, default=96,
                        help="Token budget for Task1 guardrail arbitration responses.")
    g_task.add_argument("--task1_guardrail_disable_scene_check", action="store_true", default=False,
                        help="Disable scene-setting plausibility signal in Task1 hybrid guardrail decisions.")
    g_task.add_argument("--task1_guardrail_disable_qwen_refine", action="store_true", default=False,
                        help="Disable constrained Qwen follow-up after guardrail proposal.")
    g_task.add_argument("--task1_guardrail_gemini_thinking_budget", type=int, default=48,
                        help="Gemini thinking budget used by Task1 guardrail calls.")
    g_task.add_argument("--task1_guardrail_gemini_top_p", type=float, default=0.85,
                        help="Gemini topP used by Task1 guardrail calls.")
    g_task.add_argument("--task1_guardrail_gemini_top_k", type=int, default=32,
                        help="Gemini topK used by Task1 guardrail calls.")
    # Task1 teacher-final pipeline (Qwen student -> Gemini teacher).
    g_task.add_argument("--task1_teacher_final", action="store_true", default=True,
                        help="Enable Task1 teacher-final pipeline (Gemini adjudicates final Task1 label).")
    g_task.add_argument("--task1_disable_teacher_final", action="store_true", default=False,
                        help="Disable Task1 teacher-final pipeline.")
    g_task.add_argument("--task1_teacher_provider", type=str, default="gemini",
                        choices=["gemini", "openai", "qwen"],
                        help="Provider used by Task1 teacher-final adjudication.")
    g_task.add_argument("--task1_teacher_model", type=str, default="gemini-3-flash-preview",
                        help="Model id for Task1 teacher-final adjudication.")
    g_task.add_argument("--task1_teacher_force_call", action="store_true", default=True,
                        help="Always run Task1 teacher pass-1 once per sample.")
    g_task.add_argument("--task1_disable_teacher_force_call", action="store_true", default=False,
                        help="Disable mandatory Task1 teacher pass-1.")
    g_task.add_argument("--task1_teacher_max_calls", type=int, default=2,
                        help="Maximum number of Task1 teacher calls per sample.")
    g_task.add_argument("--task1_teacher_second_call_on_mismatch", action="store_true", default=True,
                        help="Run Task1 teacher pass-2 only when Qwen and teacher labels semantically mismatch.")
    g_task.add_argument("--task1_disable_teacher_second_call", action="store_true", default=False,
                        help="Disable Task1 teacher pass-2 on semantic mismatch.")
    g_task.add_argument("--task1_teacher_min_conf", type=str, default="MEDIUM",
                        choices=["LOW", "MEDIUM", "HIGH"],
                        help="Minimum confidence for applying teacher-selected Task1 labels.")
    g_task.add_argument("--task1_teacher_temperature", type=float, default=0.2,
                        help="Temperature used in Task1 teacher calls.")
    g_task.add_argument("--task1_teacher_max_new_tokens", type=int, default=512,
                        help="Token budget for Task1 teacher responses.")
    g_task.add_argument("--task1_teacher_gemini_thinking_level", type=str, default="medium",
                        choices=["minimal", "low", "medium", "high"],
                        help="Gemini thinking level for Task1 teacher calls.")
    g_task.add_argument("--task1_teacher_gemini_thinking_budget", type=int, default=64,
                        help="Gemini thinking budget for Task1 teacher calls.")
    g_task.add_argument("--task1_teacher_gemini_media_resolution", type=str, default="high",
                        choices=["low", "medium", "high", "ultra_high"],
                        help="Gemini media resolution for Task1 teacher calls (ultra_high is per-part Gemini 3 only).")
    g_task.add_argument("--task1_teacher_structured_json", action="store_true", default=True,
                        help="Request structured JSON output for Task1 teacher Gemini calls.")
    g_task.add_argument("--task1_disable_teacher_structured_json", action="store_true", default=False,
                        help="Disable structured JSON mode for Task1 teacher calls.")
    g_task.add_argument("--task1_teacher_retry_on_partial_json", action="store_true", default=True,
                        help="Retry Task1 teacher Gemini calls with larger token budget when JSON is partial/invalid.")
    g_task.add_argument("--task1_disable_teacher_retry_on_partial_json", action="store_true", default=False,
                        help="Disable partial-JSON retry behavior for Task1 teacher calls.")
    g_task.add_argument("--task1_teacher_retry_token_multiplier", type=float, default=2.0,
                        help="Multiplier for retry maxOutputTokens on partial Task1 teacher JSON responses.")
    g_task.add_argument("--task1_legacy_pipeline", action="store_true", default=False,
                        help="Use legacy Task1 late-stage arbitration path instead of teacher-final.")
    # Task1 segmentation preprocessing beyond CLAHE (optional).
    g_task.add_argument("--task1_seg_bilateral", action="store_true", default=False,
                        help="Apply bilateral filtering on LAB-L channel before SAM2.")
    g_task.add_argument("--task1_seg_bilateral_d", type=int, default=9,
                        help="Bilateral filter neighborhood diameter.")
    g_task.add_argument("--task1_seg_bilateral_sigma_color", type=float, default=75.0,
                        help="Bilateral sigmaColor.")
    g_task.add_argument("--task1_seg_bilateral_sigma_space", type=float, default=75.0,
                        help="Bilateral sigmaSpace.")
    g_task.add_argument("--task1_seg_lb_edge_boost", action="store_true", default=False,
                        help="Boost LAB-L using combined Canny edges from L and b channels.")
    g_task.add_argument("--task1_seg_edge_l_low", type=int, default=30,
                        help="Canny low threshold for L channel.")
    g_task.add_argument("--task1_seg_edge_l_high", type=int, default=100,
                        help="Canny high threshold for L channel.")
    g_task.add_argument("--task1_seg_edge_b_low", type=int, default=20,
                        help="Canny low threshold for b channel.")
    g_task.add_argument("--task1_seg_edge_b_high", type=int, default=50,
                        help="Canny high threshold for b channel.")
    g_task.add_argument("--task1_seg_edge_boost_alpha", type=float, default=0.2,
                        help="Blend weight for edge boost onto LAB-L (0.0-1.0).")
    g_task.add_argument("--task1_seg_edge_dilate_iter", type=int, default=1,
                        help="Optional edge dilation iterations before boost.")
    g_task.add_argument("--task1_large_mask_refine", action="store_true", default=False,
                        help="If Task1 mask area is larger than person bbox area, run an extra tighter refinement cue.")
    g_task.add_argument("--task1_large_mask_refine_trigger_ratio", type=float, default=1.0,
                        help="Trigger extra refine when mask_area_px >= trigger_ratio * person_bbox_area_px.")
    g_task.add_argument("--task1_large_mask_refine_max_frac", type=float, default=0.85,
                        help="Accept extra refine only if refined mask area <= this fraction of original mask area.")
    g_task.add_argument("--task1_large_mask_refine_point_scale", type=float, default=0.45,
                        help="Point-box scale (vs TASK1_POINT_BOX_SIZE) for extra tighter refine.")
    g_task.add_argument("--task1_large_mask_refine_pad_scale", type=float, default=0.2,
                        help="Padding scale (vs TASK1_PAD_AROUND_MASK) for extra tighter refine.")
    g_task.add_argument(
        "--task1_seg_preset",
        type=str,
        default="clahe_only",
        choices=["clahe_only", "clahe_bilateral", "clahe_bilateral_edge"],
        help="Simple Task1 preprocessing preset (keeps CLAHE on by default).",
    )
    # Task4: query the gaze target itself (legacy).
    g_task.add_argument("--task4_use_gaze_target", action="store_true", default=False,
                        help="If set, Task4 queries the gaze target (legacy). Default: use distractor object.")
    # Task4: proxy-ray + multiview label synthesis.
    g_task.add_argument("--task4_proxy_multiview", action="store_true", default=False,
                        help="Enable Task4 proxy-ray + multiview label synthesis for queried object.")
    # Task4 Gemini visibility verifier.
    g_task.add_argument("--task4_gemini_verifier", action="store_true", default=True,
                        help="Enable Gemini visibility verifier for Task4 queried objects.")
    g_task.add_argument("--task4_disable_gemini_verifier", action="store_true", default=False,
                        help="Disable Task4 Gemini visibility verifier.")
    g_task.add_argument("--task4_gemini_verifier_model", type=str, default="gemini-3-flash-preview",
                        help="Model id used by Task4 Gemini verifier.")
    g_task.add_argument("--task4_gemini_verifier_min_flip_conf", type=str, default="HIGH",
                        choices=["LOW", "MEDIUM", "HIGH"],
                        help="Minimum Gemini confidence to flip Task4 answer on disagreement.")
    g_task.add_argument("--task4_gemini_verifier_reject_on_uncertain_conflict", action="store_true", default=True,
                        help="Reject Task4 sample on Gemini disagreement below flip confidence.")
    g_task.add_argument("--task4_disable_gemini_reject_on_uncertain_conflict", action="store_true", default=False,
                        help="Keep Task4 sample when verifier disagreement confidence is below flip threshold.")
    g_task.add_argument("--task4_gemini_verifier_max_new_tokens", type=int, default=192,
                        help="Token budget for Task4 Gemini verifier responses.")
    g_task.add_argument("--task4_gemini_verifier_temperature", type=float, default=0.2,
                        help="Temperature used by Task4 Gemini verifier.")
    g_task.add_argument("--task4_gemini_verifier_thinking_level", type=str, default="low",
                        choices=["minimal", "low", "medium", "high"],
                        help="Gemini thinking level for Task4 verifier calls.")
    g_task.add_argument("--task4_gemini_verifier_thinking_budget", type=int, default=32,
                        help="Gemini thinking budget for Task4 verifier calls.")
    g_task.add_argument("--task4_gemini_verifier_media_resolution", type=str, default="high",
                        choices=["low", "medium", "high", "ultra_high"],
                        help="Gemini media resolution for Task4 verifier calls (ultra_high is per-part Gemini 3 only).")
    g_task.add_argument("--task4_gemini_verifier_structured_json", action="store_true", default=True,
                        help="Request structured JSON output for Task4 Gemini verifier calls.")
    g_task.add_argument("--task4_disable_gemini_verifier_structured_json", action="store_true", default=False,
                        help="Disable structured JSON mode for Task4 Gemini verifier.")
    g_task.add_argument("--task4_gemini_verifier_retry_on_partial_json", action="store_true", default=True,
                        help="Retry Task4 verifier Gemini calls with larger token budget when JSON is partial/invalid.")
    g_task.add_argument("--task4_disable_gemini_verifier_retry_on_partial_json", action="store_true", default=False,
                        help="Disable partial-JSON retry behavior for Task4 verifier calls.")
    g_task.add_argument("--task4_gemini_verifier_retry_token_multiplier", type=float, default=2.0,
                        help="Multiplier for retry maxOutputTokens on partial Task4 verifier JSON responses.")
    # Optional intrinsics load for traceability.
    g_task.add_argument("--load_intri", action="store_true", default=False,
                        help="DFS-search and load intri.yml for traceability (not required for current tasks).")

    # -------------------------------------------------------------------------
    # Data mixing / split selection
    # -------------------------------------------------------------------------
    g_mix = p.add_argument_group("Data mixing / split selection")
    # Interleave sequences across splits.
    g_mix.add_argument("--interleave_splits", action="store_true", default=True,
                       help="Interleave sequences across splits for scene diversity (default: on).")
    g_mix.add_argument("--no_interleave_splits", action="store_true", default=False,
                       help="Disable split interleaving and keep a single shuffled list.")
    # Filter to specific splits.
    g_mix.add_argument("--splits", type=str, nargs="*", default=[],
                       help="Optional list of splits to include (e.g., Commons Lab Kitchen).")
    # Balance samples across splits.
    g_mix.add_argument("--balance_splits", action="store_true", default=True,
                       help="Balance samples across splits for scene diversity (default: on).")
    g_mix.add_argument("--no_balance_splits", action="store_true", default=False,
                       help="Disable split balancing.")

    return p.parse_args()


ARGS = parse_args()
if ARGS.full_dataset:
    ARGS.max_frames = 0
    ARGS.max_sequences = 0
if ARGS.no_scan_dataset_stats:
    ARGS.scan_dataset_stats = False
if ARGS.allow_partial_tasks:
    ARGS.require_all_tasks = False
if ARGS.cot:
    ARGS.task1_semantic_arbiter = True
if ARGS.task1_disable_teacher_final:
    ARGS.task1_teacher_final = False
if ARGS.task1_disable_teacher_force_call:
    ARGS.task1_teacher_force_call = False
if ARGS.task1_disable_teacher_second_call:
    ARGS.task1_teacher_second_call_on_mismatch = False
if ARGS.task1_disable_teacher_structured_json:
    ARGS.task1_teacher_structured_json = False
if ARGS.task1_disable_teacher_retry_on_partial_json:
    ARGS.task1_teacher_retry_on_partial_json = False
if ARGS.task4_disable_gemini_verifier:
    ARGS.task4_gemini_verifier = False
if ARGS.task4_disable_gemini_reject_on_uncertain_conflict:
    ARGS.task4_gemini_verifier_reject_on_uncertain_conflict = False
if ARGS.task4_disable_gemini_verifier_structured_json:
    ARGS.task4_gemini_verifier_structured_json = False
if ARGS.task4_disable_gemini_verifier_retry_on_partial_json:
    ARGS.task4_gemini_verifier_retry_on_partial_json = False
if ARGS.task1_seg_preset == "clahe_bilateral":
    ARGS.task1_seg_bilateral = True
elif ARGS.task1_seg_preset == "clahe_bilateral_edge":
    ARGS.task1_seg_bilateral = True
    ARGS.task1_seg_lb_edge_boost = True


# =============================================================================
# Run dir + logging
# =============================================================================

def now_str():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _resolve_vlm_model_id(args):
    provider = str(getattr(args, "vlm_provider", "qwen")).strip().lower()
    override = str(getattr(args, "vlm_model", "")).strip()
    if provider == "qwen":
        return str(getattr(args, "qwen_model", "Qwen/Qwen2.5-VL-7B-Instruct")).strip()
    if override:
        return override
    if provider == "openai":
        return os.environ.get("OPENAI_MODEL", "gpt-4o")
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def _slugify_model_for_run_name(model_name, max_len=80):
    s = str(model_name or "").strip()
    if not s:
        return "unknown-model"
    # Keep it filesystem-safe and compact.
    s = s.replace("/", "-").replace("\\", "-").replace(":", "-")
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-_.")
    if not s:
        s = "unknown-model"
    if len(s) > int(max_len):
        s = s[: int(max_len)].rstrip("-_.")
    return s or "unknown-model"


_RESOLVED_VLM_MODEL_ID = _resolve_vlm_model_id(ARGS)
_RUN_MODEL_SUFFIX = _slugify_model_for_run_name(_RESOLVED_VLM_MODEL_ID)

RUN_NAME = ARGS.run_name.strip() or f"run_{now_str()}_v4_{_RUN_MODEL_SUFFIX}"
OUT_ROOT = Path(ARGS.out_root).resolve()
RUN_DIR = OUT_ROOT / "runs" / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

RAW_IMG_DIR = RUN_DIR / "images_raw"
RAW_IMG_DIR.mkdir(parents=True, exist_ok=True)

DEBUG_DIR = RUN_DIR / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = RUN_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

BENCH_JSON = RUN_DIR / "benchmark_gazevqa.json"
DEBUG_MANIFEST = DEBUG_DIR / "debug_manifest.jsonl"
RUN_LOG = LOG_DIR / "run.log"
CONFIG_JSON = RUN_DIR / "run_config.json"
VLM_USAGE_JSON = RUN_DIR / "vlm_usage_report.json"
GEMINI_TEACHER_VERIFIER_REPORT_JSON = RUN_DIR / "gemini_teacher_verifier_report.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(RUN_LOG, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("gazevqa")

logger.info("=== Gaze-VQA Builder v7.4 (Single-process) ===")
logger.info(f"Run dir: {RUN_DIR}")

with open(CONFIG_JSON, "w") as f:
    json.dump({
        "run_name": RUN_NAME,
        "out_root": str(OUT_ROOT),
        "run_dir": str(RUN_DIR),
        "args": vars(ARGS),
    }, f, indent=2)


# =============================================================================
# Global constants / config
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATASET_URL = ARGS.mvgt_url
LOCAL_ZIP_PATH = str(Path(ARGS.mvgt_zip).resolve()) if ARGS.mvgt_zip else ""

TARGET_TASK1, TARGET_TASK2, TARGET_TASK3, TARGET_TASK4 = ARGS.targets
TARGET_BUNDLE = min(TARGET_TASK1, TARGET_TASK2, TARGET_TASK3, TARGET_TASK4)
RESIZE_WH = tuple(ARGS.resize_wh)

GAZE_LINE_W = 5
GAZE_DOT_R = 7
GAZE_COLOR = (255, 0, 0)
# Ray marker tuning: place marker slightly before the gaze coordinate to reduce occlusion.
GAZE_ARROW_OFFSET_PX = 8
GAZE_ARROW_LEN = 12
GAZE_ARROW_HALF_W = 5

SAM2_REPO_DIR = str(Path(ARGS.sam2_repo).resolve())
SAM2_CHECKPOINT = str(Path(ARGS.sam2_ckpt).resolve())
SAM2_CFG_BASENAME = "sam2_hiera_l.yaml"

VLM_PROVIDER = str(ARGS.vlm_provider).strip().lower()
VLM_MODEL_OVERRIDE = str(ARGS.vlm_model).strip()
QWEN_MODEL_ID = ARGS.qwen_model
OPENAI_BASE_URL = str(ARGS.openai_base_url).strip()
GEMINI_API_BASE = str(ARGS.gemini_api_base).strip()
GEMINI_THINKING_BUDGET = int(ARGS.gemini_thinking_budget)
VLM_TIMEOUT_S = float(ARGS.vlm_timeout_s)
OPENAI_PRICE_INPUT_PER_1M = float(ARGS.openai_price_input_per_1m)
OPENAI_PRICE_CACHED_INPUT_PER_1M = float(ARGS.openai_price_cached_input_per_1m)
OPENAI_PRICE_OUTPUT_PER_1M = float(ARGS.openai_price_output_per_1m)
GEMINI_PRICE_INPUT_PER_1M = float(ARGS.gemini_price_input_per_1m)
GEMINI_PRICE_OUTPUT_PER_1M = float(ARGS.gemini_price_output_per_1m)

VLM_MODEL_ID = _RESOLVED_VLM_MODEL_ID

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 768 * 28 * 28

EMPTY_CACHE_EVERY = int(ARGS.empty_cache_every)
SAVE_DEBUG = bool(ARGS.save_debug)
DEBUG_EVERY_N_TASK1 = int(ARGS.debug_every_n_task1)
MAX_VIEWS = int(ARGS.max_views)
THREAD_IO = int(ARGS.thread_io)
REQUIRE_ALL_TASKS_PER_FRAME = bool(ARGS.require_all_tasks)
TASK1_CONF_THRESHOLD = float(ARGS.task1_conf_threshold)

# Task1 segmentation behavior
TASK1_USE_TIGHT_BOX = True # True
TASK1_POINT_BOX_SIZE = 220
TASK1_PAD_AROUND_MASK = 30
TASK1_PAD_AROUND_MASK_RATIO = 0.12
TASK1_PAD_AROUND_MASK_MAX = 50
TASK1_DILATE_MASK = True # False
TASK1_DILATE_ITER = 5 # 1
TASK1_MASK_MIN_AREA_RATIO = 0.000001
TASK1_MASK_MAX_AREA_RATIO = 0.65
TASK1_SMALL_OBJ_AREA_RATIO = 0.01
TASK1_SMALL_MASK_USE_CONTEXT = True
TASK1_SMALL_MASK_CONTEXT_AREA_RATIO = 0.05
TASK1_SMALL_MASK_CONTEXT_EXPAND_RATIO = 5
TASK1_GAZE_CONF_RADIUS = 4
TASK1_MIN_SOFT_CONF_AROUND_GAZE = 0.01
TASK1_SOFT_MASK_THRESHOLD = 0.01
TASK1_MASK_OVERLAY_ALPHA = 0.65
TASK1_SEG_USE_CLAHE = True
TASK1_SEG_CLAHE_CLIP = 2.0
TASK1_SEG_CLAHE_TILE = 8
TASK1_SEG_USE_BILATERAL = bool(ARGS.task1_seg_bilateral)
TASK1_SEG_BILATERAL_D = int(ARGS.task1_seg_bilateral_d)
TASK1_SEG_BILATERAL_SIGMA_COLOR = float(ARGS.task1_seg_bilateral_sigma_color)
TASK1_SEG_BILATERAL_SIGMA_SPACE = float(ARGS.task1_seg_bilateral_sigma_space)
TASK1_SEG_USE_LB_EDGE_BOOST = bool(ARGS.task1_seg_lb_edge_boost)
TASK1_SEG_EDGE_L_LOW = int(ARGS.task1_seg_edge_l_low)
TASK1_SEG_EDGE_L_HIGH = int(ARGS.task1_seg_edge_l_high)
TASK1_SEG_EDGE_B_LOW = int(ARGS.task1_seg_edge_b_low)
TASK1_SEG_EDGE_B_HIGH = int(ARGS.task1_seg_edge_b_high)
TASK1_SEG_EDGE_BOOST_ALPHA = float(ARGS.task1_seg_edge_boost_alpha)
TASK1_SEG_EDGE_DILATE_ITER = int(ARGS.task1_seg_edge_dilate_iter)
TASK1_LARGE_MASK_REFINE = bool(ARGS.task1_large_mask_refine)
TASK1_LARGE_MASK_REFINE_TRIGGER_RATIO = float(ARGS.task1_large_mask_refine_trigger_ratio)
TASK1_LARGE_MASK_REFINE_MAX_FRAC = float(ARGS.task1_large_mask_refine_max_frac)
TASK1_LARGE_MASK_REFINE_POINT_SCALE = float(ARGS.task1_large_mask_refine_point_scale)
TASK1_LARGE_MASK_REFINE_PAD_SCALE = float(ARGS.task1_large_mask_refine_pad_scale)
# Dynamic attempt gating:
# In loose-sweep->strict order, do not early-accept loose stages below this
# area ratio unless retries are exhausted.
TASK1_EARLY_ACCEPT_MIN_AREA_RATIO = 0.0314 #.01
# Require at least one cue agreement (dot/ray text vs mask label) before
# accepting a non-final attempt early.
TASK1_EARLY_ACCEPT_REQUIRE_CUE_AGREEMENT = True

# Mask-person overlap rejection: RETRY then fallback
TASK1_REJECT_IF_MASK_OVERLAPS_PERSON = True
TASK1_PERSON_OVERLAP_THRESHOLD = 0.5
# Anchor camera distance preference (normalized by person bbox diag).
# Mid-distance tends to avoid person overlap while staying close enough for accuracy.
TASK1_ANCHOR_DIST_TARGET = 0.6
TASK1_ANCHOR_DIST_SIGMA = 0.2
TASK1_ANCHOR_MEDIUM_BAND = 0.18
# Multiview voting tweak:
# If one view is extremely close (gaze point hugging person bbox), soften only that view.
TASK1_MV_CLOSEST_DIST_NORM_THRESH = 0.20
TASK1_MV_CLOSEST_WEIGHT_DOWNSCALE = 0.8

TASK2_AZIMUTH_PLANE = "xz"
TASK2_AXIS_DIAG = True
TASK2_AXIS_DIAG_MIN_CAMS = 4
TASK2_FRAME_DIST_MIN = 1e-6
TASK2_FRAME_DIST_MAX = 1e5

TASK3_USE_GT_VISIBILITY_IF_PRESENT = True
TASK3_OUTSIDE_SUMMARY = "The {person_desc} is looking at something outside these multi-view images."
TASK3_NUM_VIEWS_MIN = 2
TASK3_NUM_VIEWS_MAX = 6

TASK4_REQUIRE_VERIFIER_PASS = True
TASK4_OBJECT_PROPOSAL_MAX_TRIES = 5
TASK4_VERIFIER_MAX_TRIES = 4
TASK4_DISTRACTOR_MIN_AREA_RATIO = 0.0015
TASK4_DISTRACTOR_MIN_BBOX_PX = 25
TASK4_DISTRACTOR_VERIFY_LABEL = True
TASK4_DISTRACTOR_MAX_AREA_RATIO = 0.3
# Prefer distractor masks near this area ratio rather than always picking the largest mask.
TASK4_DISTRACTOR_TARGET_AREA_RATIO = 0.02
# Smoothness of area preference in log-area space (higher = flatter preference).
TASK4_DISTRACTOR_LOG_AREA_SIGMA = 0.9
# Small tiebreaker toward larger masks up to target-area scale.
TASK4_DISTRACTOR_SCORE_AREA_WEIGHT = 0.15
TASK4_DISTRACTOR_MIN_DIST_RATIO = 0.02818 * 3.14
TASK4_DISTRACTOR_MAX_DIST_RATIO = 0.95
TASK4_DISTRACTOR_EDGE_PAD_PX = 20
TASK4_BAD_DISTRACTOR_WORDS = {
    "grid", "pattern", "texture", "tiles", "tile", "carpet", "rug", "mat",
    "floor", "ceiling", "wall", "ground", "surface"
}
TASK4_DISTRACTOR_TANGIBLE_CHECK = True
TASK4_FORCE_VLM_REASONING = True
TASK4_PROXY_MULTIVIEW = bool(ARGS.task4_proxy_multiview)
TASK4_MV_MAX_CAMS = 3
TASK4_REQUIRE_PERSON_VISIBLE = True
TASK1_POSE_CHECK = True

REASONING_MODE = ARGS.reasoning_mode
DUMP_OVERLAP_DEBUG = bool(ARGS.dump_overlap_debug)
TASK4_USE_GAZE_TARGET = bool(ARGS.task4_use_gaze_target)
TASK1_REASONING_MODE = ARGS.task1_reasoning_mode
TASK1_SEMANTIC_ARBITER = bool(ARGS.task1_semantic_arbiter)
TASK1_SEMANTIC_ARBITER_MIN_CONF = "MEDIUM"
TASK1_HYBRID_GUARDRAIL = bool(ARGS.task1_hybrid_guardrail)
TASK1_GUARDRAIL_PROVIDER = str(ARGS.task1_guardrail_provider).strip().lower()
TASK1_GUARDRAIL_MODEL = str(ARGS.task1_guardrail_model).strip()
TASK1_GUARDRAIL_MIN_CONF = str(ARGS.task1_guardrail_min_conf).strip().upper()
TASK1_GUARDRAIL_MAX_NEW_TOKENS = int(ARGS.task1_guardrail_max_new_tokens)
TASK1_GUARDRAIL_SCENE_CHECK = not bool(ARGS.task1_guardrail_disable_scene_check)
TASK1_GUARDRAIL_QWEN_REFINE = not bool(ARGS.task1_guardrail_disable_qwen_refine)
TASK1_GUARDRAIL_GEMINI_THINKING_BUDGET = int(ARGS.task1_guardrail_gemini_thinking_budget)
TASK1_GUARDRAIL_GEMINI_TOP_P = float(ARGS.task1_guardrail_gemini_top_p)
TASK1_GUARDRAIL_GEMINI_TOP_K = int(ARGS.task1_guardrail_gemini_top_k)
TASK1_TEACHER_FINAL = bool(ARGS.task1_teacher_final)
TASK1_TEACHER_PROVIDER = str(ARGS.task1_teacher_provider).strip().lower()
TASK1_TEACHER_MODEL = str(ARGS.task1_teacher_model).strip()
TASK1_TEACHER_FORCE_CALL = bool(ARGS.task1_teacher_force_call)
TASK1_TEACHER_MAX_CALLS = max(1, int(ARGS.task1_teacher_max_calls))
TASK1_TEACHER_SECOND_CALL_ON_MISMATCH = bool(ARGS.task1_teacher_second_call_on_mismatch)
TASK1_TEACHER_MIN_CONF = str(ARGS.task1_teacher_min_conf).strip().upper()
TASK1_TEACHER_TEMPERATURE = float(ARGS.task1_teacher_temperature)
TASK1_TEACHER_MAX_NEW_TOKENS = int(ARGS.task1_teacher_max_new_tokens)
TASK1_TEACHER_GEMINI_THINKING_LEVEL = str(ARGS.task1_teacher_gemini_thinking_level).strip().lower()
TASK1_TEACHER_GEMINI_THINKING_BUDGET = int(ARGS.task1_teacher_gemini_thinking_budget)
TASK1_TEACHER_GEMINI_MEDIA_RESOLUTION = str(ARGS.task1_teacher_gemini_media_resolution).strip().lower()
TASK1_TEACHER_STRUCTURED_JSON = bool(ARGS.task1_teacher_structured_json)
TASK1_TEACHER_RETRY_ON_PARTIAL_JSON = bool(ARGS.task1_teacher_retry_on_partial_json)
TASK1_TEACHER_RETRY_TOKEN_MULTIPLIER = float(ARGS.task1_teacher_retry_token_multiplier)
TASK1_LEGACY_PIPELINE = bool(ARGS.task1_legacy_pipeline)
TASK4_GEMINI_VERIFIER = bool(ARGS.task4_gemini_verifier)
TASK4_GEMINI_VERIFIER_MODEL = str(ARGS.task4_gemini_verifier_model).strip()
TASK4_GEMINI_VERIFIER_MIN_FLIP_CONF = str(ARGS.task4_gemini_verifier_min_flip_conf).strip().upper()
TASK4_GEMINI_VERIFIER_REJECT_ON_UNCERTAIN_CONFLICT = bool(ARGS.task4_gemini_verifier_reject_on_uncertain_conflict)
TASK4_GEMINI_VERIFIER_MAX_NEW_TOKENS = int(ARGS.task4_gemini_verifier_max_new_tokens)
TASK4_GEMINI_VERIFIER_TEMPERATURE = float(ARGS.task4_gemini_verifier_temperature)
TASK4_GEMINI_VERIFIER_THINKING_LEVEL = str(ARGS.task4_gemini_verifier_thinking_level).strip().lower()
TASK4_GEMINI_VERIFIER_THINKING_BUDGET = int(ARGS.task4_gemini_verifier_thinking_budget)
TASK4_GEMINI_VERIFIER_MEDIA_RESOLUTION = str(ARGS.task4_gemini_verifier_media_resolution).strip().lower()
TASK4_GEMINI_VERIFIER_STRUCTURED_JSON = bool(ARGS.task4_gemini_verifier_structured_json)
TASK4_GEMINI_VERIFIER_RETRY_ON_PARTIAL_JSON = bool(ARGS.task4_gemini_verifier_retry_on_partial_json)
TASK4_GEMINI_VERIFIER_RETRY_TOKEN_MULTIPLIER = float(ARGS.task4_gemini_verifier_retry_token_multiplier)
BUNDLE_TARGET = TARGET_BUNDLE

CANON_CAMS = [f"Cam{i}" for i in range(1, 7)]

BAD_OBJECTS = {
    "floor", "wall", "ceiling", "room", "background", "outside", "nothing", "unknown",
    "scene", "image", "view", "camera", "person", "people", "man", "woman", "boy", "girl", "human", "humans",
}
BAD_GENERIC_WORDS = {"object", "thing", "item", "stuff", "something", "anything", "square", "circle", "shape"}
BAD_GENERIC_PHRASES = {
    "white object", "black object", "red object", "blue object",
    "small object", "large object"
}

# Global rejection stats (anti silent failure)
REJECT_STATS = {
    "t1_no_anchor": 0,
    "t1_no_coord": 0,
    "t1_no_images": 0,
    "t1_sam2_no_mask": 0,
    "t1_sam2_overlap_reject": 0,
    "t1_sam2_empty_crop": 0,
    "t1_sam2_total_fail": 0,
    "t1_phrase_missing": 0,
    "t1_canonical_fail": 0,
    "t1_all_anchor_candidates_failed": 0,
    "t1_pose_mismatch": 0,
    "t1_conf_below_threshold": 0,
    "t1_teacher_second_call": 0,
    "t1_teacher_fallback": 0,
    "t1_teacher_conflict": 0,
    "t1_teacher_parse_fail": 0,
    "t1_teacher_partial_retry": 0,

    "t2_no_tri": 0,
    "t2_no_cam_centers": 0,
    "t2_no_images": 0,
    "t2_bad_median_dist": 0,

    "t3_not_enough_views": 0,

    "t4_not_enough_views": 0,
    "t4_object_proposal_fail": 0,
    "t4_object_not_tangible": 0,
    "t4_object_not_in_view": 0,
    "t4_inconsistent_gt": 0,
    "t4_verifier_fail": 0,
    "t4_gemini_verifier_parse_fail": 0,
    "t4_gemini_verifier_presence_conflict": 0,
    "t4_gemini_verifier_person_not_present": 0,
    "t4_no_visibility": 0,
    "t4_gemini_verify_flip": 0,
    "t4_gemini_verify_reject": 0,
    "frame_missing_task1": 0,
    "frame_missing_task2": 0,
    "frame_missing_task3": 0,
    "frame_missing_task4": 0,
    "frame_incomplete_bundle": 0,

    "exceptions": 0,
}

# Frame accounting
FRAME_STATS = {
    "frames_seen": 0,
    "frames_with_min_views": 0,
    "dataset_total_frames": 0,
    "dataset_frames_with_gaze": 0,
    "dataset_total_gaze_points": 0,
    "dataset_total_sequences": 0,
}

DATASET_STATS = {}

TASK2_DIST_STATS = {
    "seen": 0,
    "accepted": 0,
    "rejected": 0,
    "med_dist_vals": [],
    "med_dist_reject_vals": [],
}

# Runtime context for per-frame VLM usage accounting.
CURRENT_FRAME_KEY = None
