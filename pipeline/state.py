import os
import json
import argparse
import time
import random
import logging
import sys
from pathlib import Path
import warnings
import numpy as np

warnings.filterwarnings("ignore")


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
    # Disable VLM entirely (debug only).
    g_models.add_argument("--skip_vlm", action="store_true",
                          help="Skip loading Qwen and any VLM calls (debug only; low quality / empty samples).")

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
    # Task4: query the gaze target itself (legacy).
    g_task.add_argument("--task4_use_gaze_target", action="store_true", default=False,
                        help="If set, Task4 queries the gaze target (legacy). Default: use distractor object.")
    # Task4: proxy-ray + multiview label synthesis.
    g_task.add_argument("--task4_proxy_multiview", action="store_true", default=False,
                        help="Enable Task4 proxy-ray + multiview label synthesis for queried object.")
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


# =============================================================================
# Run dir + logging
# =============================================================================

def now_str():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


RUN_NAME = ARGS.run_name.strip() or f"run_{now_str()}_v3"
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

GAZE_LINE_W = 2
GAZE_DOT_R = 3
GAZE_COLOR = (255, 0, 0)

SAM2_REPO_DIR = str(Path(ARGS.sam2_repo).resolve())
SAM2_CHECKPOINT = str(Path(ARGS.sam2_ckpt).resolve())
SAM2_CFG_BASENAME = "sam2_hiera_l.yaml"

QWEN_MODEL_ID = ARGS.qwen_model
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
TASK1_POINT_BOX_SIZE = 50
TASK1_PAD_AROUND_MASK = 20
TASK1_PAD_AROUND_MASK_RATIO = 0.12
TASK1_PAD_AROUND_MASK_MAX = 50
TASK1_DILATE_MASK = True # False
TASK1_DILATE_ITER = 2 # 1
TASK1_MASK_MIN_AREA_RATIO = 0.000001
TASK1_MASK_MAX_AREA_RATIO = 0.5
TASK1_SMALL_OBJ_AREA_RATIO = 0.05
TASK1_GAZE_CONF_RADIUS = 8
TASK1_MIN_SOFT_CONF_AROUND_GAZE = 0.027818
TASK1_SOFT_MASK_THRESHOLD = 0.027818
TASK1_MASK_OVERLAY_ALPHA = 0.55

# Mask-person overlap rejection: RETRY then fallback
TASK1_REJECT_IF_MASK_OVERLAPS_PERSON = True
TASK1_PERSON_OVERLAP_THRESHOLD = 0.5

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
TASK4_OBJECT_PROPOSAL_MAX_TRIES = 4
TASK4_VERIFIER_MAX_TRIES = 5
TASK4_DISTRACTOR_MIN_AREA_RATIO = 0.005
TASK4_DISTRACTOR_MIN_BBOX_PX = 1
TASK4_DISTRACTOR_VERIFY_LABEL = True
TASK4_DISTRACTOR_MAX_AREA_RATIO = 0.3
TASK4_DISTRACTOR_MIN_DIST_RATIO = 0.02818 * 3.14
TASK4_DISTRACTOR_MAX_DIST_RATIO = 0.95
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
    "t4_no_visibility": 0,
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
