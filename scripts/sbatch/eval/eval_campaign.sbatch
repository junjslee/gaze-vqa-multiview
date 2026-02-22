#!/bin/bash
#SBATCH --account=bfga-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240g
#SBATCH --constraint="scratch"
#SBATCH --gpu-bind=closest
#SBATCH --job-name=gazevqa-eval-campaign
#SBATCH --output=/work/nvme/bfga/jlee65/gaze_vqa/runs/slurm_%x_%j.out
#SBATCH --error=/work/nvme/bfga/jlee65/gaze_vqa/runs/slurm_%x_%j.err
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

USE_APPTAINER="${USE_APPTAINER:-1}"
SIF="${SIF:-/work/nvme/bfga/jlee65/jun_fm.sif}"
CAMPAIGN_NAME="${CAMPAIGN_NAME:-campaign_$(date +%Y%m%d_%H%M%S)}"
BENCHMARK_PATH="${BENCHMARK_PATH:-}"
RUN_TRACK_A="${RUN_TRACK_A:-1}"
RUN_TRACK_B="${RUN_TRACK_B:-1}"
RUN_JUDGE="${RUN_JUDGE:-1}"
RUN_AGGREGATE="${RUN_AGGREGATE:-1}"
JUDGE_MODEL="${JUDGE_MODEL:-gemini-3.1-pro-preview}"
REQUEST_INTERVAL="${REQUEST_INTERVAL:-0.0}"
GEMINI_REQUEST_INTERVAL="${GEMINI_REQUEST_INTERVAL:-30.0}"
JUDGE_REQUEST_INTERVAL="${JUDGE_REQUEST_INTERVAL:-0.0}"
RESET_PREDICTIONS="${RESET_PREDICTIONS:-0}"
RUN_GEMINI20FLASH="${RUN_GEMINI20FLASH:-1}"
SERVER_READY_RETRIES="${SERVER_READY_RETRIES:-900}"
SERVER_READY_SLEEP_S="${SERVER_READY_SLEEP_S:-2}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-${ROOT}/runs/server_logs}"
SERVER_LOG_TAIL_LINES="${SERVER_LOG_TAIL_LINES:-200}"

run_python() {
  if [[ "${USE_APPTAINER}" == "1" ]]; then
    srun apptainer exec --nv "${SIF}" python3 "$@"
  else
    srun python3 "$@"
  fi
}

if [[ "${USE_APPTAINER}" == "1" && ! -f "${SIF}" ]]; then
  echo "[ERROR] Container not found: ${SIF}"
  exit 2
fi

JOB_TMP_BASE="/tmp/${USER}/gazevqa_eval_campaign_${SLURM_JOB_ID:-manual}"
mkdir -p "${JOB_TMP_BASE}/hf/hub" "${JOB_TMP_BASE}/hf/transformers" "${JOB_TMP_BASE}/xdg" "${JOB_TMP_BASE}/pip"
export HF_HOME="${JOB_TMP_BASE}/hf"
export HF_HUB_CACHE="${JOB_TMP_BASE}/hf/hub"
export TRANSFORMERS_CACHE="${JOB_TMP_BASE}/hf/transformers"
export XDG_CACHE_HOME="${JOB_TMP_BASE}/xdg"
export PIP_CACHE_DIR="${JOB_TMP_BASE}/pip"

echo "[INFO] Campaign name: ${CAMPAIGN_NAME}"
if [[ -n "${BENCHMARK_PATH}" ]]; then
  run_python "${ROOT}/evaluate_benchmark_delta.py" freeze-gt \
    --campaign "${CAMPAIGN_NAME}" \
    --benchmark_path "${BENCHMARK_PATH}" \
    --container_sif "${SIF}"
else
  run_python "${ROOT}/evaluate_benchmark_delta.py" freeze-gt \
    --campaign "${CAMPAIGN_NAME}" \
    --container_sif "${SIF}"
fi

CAMPAIGN_PATH="${ROOT}/runs/eval_campaigns/${CAMPAIGN_NAME}"

TRACK_A_MODELS=(
  "gpt41"
  "gpt4o"
  "qwen3vl_alias"
  "internvl25"
  "deepseekvl2"
  "llava_video_7b"
  "llava_ov_7b"
)
GEMINI_MODELS=(
  "gemini25flash"
  "gemini30flash"
  "gemini25pro"
  "gemini30pro"
)
if [[ "${RUN_GEMINI20FLASH}" == "1" ]]; then
  GEMINI_MODELS=("gemini20flash" "${GEMINI_MODELS[@]}")
fi
TRACK_A_MODELS=("${GEMINI_MODELS[@]}" "${TRACK_A_MODELS[@]}")
TRACK_B_MODELS=(
  "llava_video_72b"
  "llava_ov_72b"
  "llava_next_8b"
  "llava_next_32b"
)

if [[ "${RUN_TRACK_A}" == "1" ]]; then
  for model_key in "${TRACK_A_MODELS[@]}"; do
    model_request_interval="${REQUEST_INTERVAL}"
    if [[ "${model_key}" == gemini* ]]; then
      model_request_interval="${GEMINI_REQUEST_INTERVAL}"
    fi
    echo "[INFO] Running model: ${model_key}"
    CAMPAIGN="${CAMPAIGN_PATH}" MODEL_KEY="${model_key}" USE_APPTAINER="${USE_APPTAINER}" SIF="${SIF}" REQUEST_INTERVAL="${model_request_interval}" RESET_PREDICTIONS="${RESET_PREDICTIONS}" \
      SERVER_READY_RETRIES="${SERVER_READY_RETRIES}" SERVER_READY_SLEEP_S="${SERVER_READY_SLEEP_S}" SERVER_LOG_DIR="${SERVER_LOG_DIR}" SERVER_LOG_TAIL_LINES="${SERVER_LOG_TAIL_LINES}" \
      srun bash "${ROOT}/scripts/sbatch/eval/eval_model.sbatch"
  done
fi

if [[ "${RUN_TRACK_B}" == "1" ]]; then
  for model_key in "${TRACK_B_MODELS[@]}"; do
    echo "[INFO] Running model: ${model_key}"
    CAMPAIGN="${CAMPAIGN_PATH}" MODEL_KEY="${model_key}" USE_APPTAINER="${USE_APPTAINER}" SIF="${SIF}" REQUEST_INTERVAL="${REQUEST_INTERVAL}" RESET_PREDICTIONS="${RESET_PREDICTIONS}" \
      SERVER_READY_RETRIES="${SERVER_READY_RETRIES}" SERVER_READY_SLEEP_S="${SERVER_READY_SLEEP_S}" SERVER_LOG_DIR="${SERVER_LOG_DIR}" SERVER_LOG_TAIL_LINES="${SERVER_LOG_TAIL_LINES}" \
      srun bash "${ROOT}/scripts/sbatch/eval/eval_model.sbatch"
  done
fi

if [[ "${RUN_JUDGE}" == "1" ]]; then
  run_python "${ROOT}/evaluate_benchmark_delta.py" judge \
    --campaign "${CAMPAIGN_PATH}" \
    --judge_model "${JUDGE_MODEL}" \
    --request_interval "${JUDGE_REQUEST_INTERVAL}"
fi

if [[ "${RUN_AGGREGATE}" == "1" ]]; then
  run_python "${ROOT}/evaluate_benchmark_delta.py" aggregate \
    --campaign "${CAMPAIGN_PATH}"
fi

echo "[INFO] Campaign complete: ${CAMPAIGN_PATH}"
