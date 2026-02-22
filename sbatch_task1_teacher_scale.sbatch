#!/bin/bash
#SBATCH --account=bfga-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --constraint="scratch"
#SBATCH --gpu-bind=closest
#SBATCH --job-name=gazevqa-teacher-scale
#SBATCH --output=/work/nvme/bfga/jlee65/gaze_vqa/runs/slurm_%x_%j.out
#SBATCH --error=/work/nvme/bfga/jlee65/gaze_vqa/runs/slurm_%x_%j.err
#SBATCH --mail-user=jl193@illinois.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --no-requeue

set -euo pipefail

module reset
module list
nvidia-smi
echo "[INFO] Running on: $(hostname)"
echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"

ROOT="/work/nvme/bfga/jlee65/gaze_vqa"
cd "${ROOT}"

echo "[INFO] Host: $(hostname)"
echo "[INFO] Start: $(date)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

USE_APPTAINER="${USE_APPTAINER:-1}"
SIF="${SIF:-/work/nvme/bfga/jlee65/jun_fm.sif}"

run_python() {
  if [[ "${USE_APPTAINER}" == "1" ]]; then
    srun apptainer exec --nv "${SIF}" python3 "$@"
  else
    srun python3 "$@"
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

JOB_TMP_BASE="/tmp/${USER}/gazevqa_${SLURM_JOB_ID:-manual}"
mkdir -p "${JOB_TMP_BASE}/hf/hub" "${JOB_TMP_BASE}/hf/transformers" "${JOB_TMP_BASE}/xdg" "${JOB_TMP_BASE}/pip"
export HF_HOME="${JOB_TMP_BASE}/hf"
export HF_HUB_CACHE="${JOB_TMP_BASE}/hf/hub"
export TRANSFORMERS_CACHE="${JOB_TMP_BASE}/hf/transformers"
export XDG_CACHE_HOME="${JOB_TMP_BASE}/xdg"
export PIP_CACHE_DIR="${JOB_TMP_BASE}/pip"
echo "[INFO] HF_HOME=${HF_HOME}"
echo "[INFO] TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"
echo "[INFO] PIP_CACHE_DIR=${PIP_CACHE_DIR}"

run_python -c "import sys; print('[INFO] Python:', sys.executable); import torch; print('[INFO] torch:', torch.__version__)"

TARGET_TASK1="${TARGET_TASK1:-999999}"
TARGET_TASK2="${TARGET_TASK2:-999999}"
TARGET_TASK3="${TARGET_TASK3:-999999}"
TARGET_TASK4="${TARGET_TASK4:-999999}"
SPLITS="${SPLITS:-}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-}"
TASK1_TEACHER_MODEL="${TASK1_TEACHER_MODEL:-gemini-3-flash-preview}"
TASK4_VERIFIER_MODEL="${TASK4_VERIFIER_MODEL:-gemini-3-flash-preview}"
GEMINI_FALLBACK_MODELS="${GEMINI_FALLBACK_MODELS:-gemini-2.5-flash,gemini-2.0-flash,gemini-1.5-flash}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
CHECKPOINT_EVERY_N_ACCEPTS="${CHECKPOINT_EVERY_N_ACCEPTS:-1}"
MATERIALIZE_EVERY_N_CHECKPOINTS="${MATERIALIZE_EVERY_N_CHECKPOINTS:-10}"
VLM_TIMEOUT_S="${VLM_TIMEOUT_S:-150}"
export GEMINI_FALLBACK_MODELS

COMMON_ARGS=(
  --out_root "${ROOT}"
  --targets "${TARGET_TASK1}" "${TARGET_TASK2}" "${TARGET_TASK3}" "${TARGET_TASK4}"
  --save_debug
  --debug_every_n_task1 1
  --reasoning_mode gt
  --task1_reasoning_mode vlm
  --task1_conf_threshold 0.35
  --task1_seg_preset clahe_bilateral
  --full_dataset
  --no_scan_dataset_stats
  --checkpoint_every_n_accepts "${CHECKPOINT_EVERY_N_ACCEPTS}"
  --materialize_benchmark_every_n_checkpoints "${MATERIALIZE_EVERY_N_CHECKPOINTS}"
  --strict_gemini_success_required
)
if [[ -n "${RESUME_RUN_DIR}" ]]; then
  COMMON_ARGS+=(--resume_from_run_dir "${RESUME_RUN_DIR}")
fi
if [[ -n "${SPLITS}" ]]; then
  SPLITS_NORM="${SPLITS//,/ }"
  read -r -a SPLIT_ARR <<< "${SPLITS_NORM}"
  if [[ "${#SPLIT_ARR[@]}" -gt 0 ]]; then
    COMMON_ARGS+=(--splits "${SPLIT_ARR[@]}")
  fi
fi

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
SCRIPT_TAG="$(basename "${BASH_SOURCE[0]}" .sbatch)"
SUFFIX_CLEAN="$(echo "${RUN_NAME_SUFFIX}" | tr -cs 'A-Za-z0-9._-' '_' | sed 's/^_\\+//; s/_\\+$//')"
if [[ -n "${SUFFIX_CLEAN}" ]]; then
  RUN_NAME="${RUN_STAMP}_${SCRIPT_TAG}_${SUFFIX_CLEAN}"
else
  RUN_NAME="${RUN_STAMP}_${SCRIPT_TAG}"
fi

echo
echo "=============================="
echo "[RUN] ${RUN_NAME}"
echo "=============================="
echo "[INFO] Gemini fallback models: ${GEMINI_FALLBACK_MODELS}"
if [[ -n "${SPLITS}" ]]; then
  echo "[INFO] Split filter: ${SPLITS}"
fi
if [[ -n "${RESUME_RUN_DIR}" ]]; then
  echo "[INFO] Resuming from run dir: ${RESUME_RUN_DIR}"
fi

run_python "${ROOT}/generate_benchmark_delta.py" \
  "${COMMON_ARGS[@]}" \
  --run_name "${RUN_NAME}" \
  --vlm_provider qwen \
  --task1_teacher_final \
  --task1_teacher_provider gemini \
  --task1_teacher_model "${TASK1_TEACHER_MODEL}" \
  --vlm_timeout_s "${VLM_TIMEOUT_S}" \
  --task1_teacher_force_call \
  --task1_teacher_max_calls 2 \
  --task1_teacher_second_call_on_mismatch \
  --task1_teacher_min_conf MEDIUM \
  --task1_teacher_temperature 1.0 \
  --task1_teacher_max_new_tokens 512 \
  --task1_teacher_gemini_thinking_level medium \
  --task1_teacher_gemini_thinking_budget 64 \
  --task1_teacher_gemini_media_resolution high \
  --task1_teacher_structured_json \
  --task1_teacher_retry_on_partial_json \
  --task1_teacher_retry_token_multiplier 2.0 \
  --task4_gemini_verifier \
  --task4_gemini_verifier_model "${TASK4_VERIFIER_MODEL}" \
  --task4_gemini_verifier_min_flip_conf HIGH \
  --task4_gemini_verifier_reject_on_uncertain_conflict \
  --task4_gemini_verifier_max_new_tokens 192 \
  --task4_gemini_verifier_temperature 1.0 \
  --task4_gemini_verifier_thinking_level low \
  --task4_gemini_verifier_thinking_budget 32 \
  --task4_gemini_verifier_media_resolution high \
  --task4_gemini_verifier_structured_json \
  --task4_gemini_verifier_retry_on_partial_json \
  --task4_gemini_verifier_retry_token_multiplier 2.0

echo
echo "[INFO] Finished: $(date)"
echo "[INFO] Run written under: ${ROOT}/runs/${RUN_NAME}"
