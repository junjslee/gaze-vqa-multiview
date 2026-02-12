#!/bin/bash
#SBATCH --account=bfga-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --job-name=gazevqa-matrix20
#SBATCH --output=/work/nvme/bfga/jlee65/gaze_vqa/runs/slurm_%x_%j.out
#SBATCH --mail-user=jl193@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

ROOT="/work/nvme/bfga/jlee65/gaze_vqa"
cd "${ROOT}"

echo "[INFO] Host: $(hostname)"
echo "[INFO] Start: $(date)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Use containerized Python by default on Delta to ensure torch+cuda stack exists.
USE_APPTAINER="${USE_APPTAINER:-1}"
SIF="${SIF:-/work/nvme/bfga/jlee65/jun_fm.sif}"

run_python() {
  if [[ "${USE_APPTAINER}" == "1" ]]; then
    apptainer exec --nv "${SIF}" python3 "$@"
  else
    python3 "$@"
  fi
}

if [[ "${USE_APPTAINER}" == "1" ]]; then
  if [[ ! -f "${SIF}" ]]; then
    echo "[ERROR] Container not found: ${SIF}"
    exit 2
  fi
  echo "[INFO] Runtime: apptainer --nv ${SIF}"
else
  echo "[INFO] Runtime: host python3"
fi

run_python -c "import sys; print('[INFO] Python:', sys.executable); import torch; print('[INFO] torch:', torch.__version__)"

# Keep defaults deterministic and no partial acceptance.
# --allow_partial_tasks is intentionally NOT passed.
COMMON_ARGS=(
  --out_root "${ROOT}"
  --targets 20 20 20 20
  --max_frames 1200
  --no_scan_dataset_stats
)

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

run_case() {
  local name="$1"
  shift
  local run_name="${RUN_STAMP}_${name}"
  echo
  echo "=============================="
  echo "[RUN] ${run_name}"
  echo "=============================="
  run_python "${ROOT}/generate_benchmark_delta.py" \
    "${COMMON_ARGS[@]}" \
    --run_name "${run_name}" \
    "$@"
}

# 1) Qwen default baseline
run_case "qwen_default" \
  --vlm_provider qwen

# 2) Gemini-only baseline
run_case "gemini_only" \
  --vlm_provider gemini

# 3) Qwen primary + Gemini guardrail (no scene plausibility check)
run_case "qwen_hybrid_guardrail_no_scene" \
  --vlm_provider qwen \
  --task1_hybrid_guardrail \
  --task1_guardrail_provider gemini \
  --task1_guardrail_min_conf MEDIUM \
  --task1_guardrail_gemini_thinking_budget 48 \
  --task1_guardrail_gemini_top_p 0.85 \
  --task1_guardrail_gemini_top_k 32 \
  --task1_guardrail_disable_scene_check

# 4) Qwen primary + Gemini guardrail (scene plausibility check enabled)
run_case "qwen_hybrid_guardrail_scene" \
  --vlm_provider qwen \
  --task1_hybrid_guardrail \
  --task1_guardrail_provider gemini \
  --task1_guardrail_min_conf MEDIUM \
  --task1_guardrail_gemini_thinking_budget 48 \
  --task1_guardrail_gemini_top_p 0.85 \
  --task1_guardrail_gemini_top_k 32

echo
echo "[INFO] Finished: $(date)"
echo "[INFO] Runs written under: ${ROOT}/runs"
