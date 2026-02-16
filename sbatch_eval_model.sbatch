#!/bin/bash
#SBATCH --account=bfga-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240g
#SBATCH --constraint="scratch"
#SBATCH --gpu-bind=closest
#SBATCH --job-name=gazevqa-eval-model
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

# Resource profile defaults to 72B-safe settings.
# For lighter models, override at submission time, e.g.:
# sbatch --gpus-per-node=1 --cpus-per-task=8 --mem=64g --export=ALL,CAMPAIGN=...,MODEL_KEY=... sbatch_eval_model.sbatch

CAMPAIGN="${CAMPAIGN:-}"
MODEL_KEY="${MODEL_KEY:-}"
SIF="${SIF:-/work/nvme/bfga/jlee65/jun_fm.sif}"
USE_APPTAINER="${USE_APPTAINER:-1}"

if [[ -z "${CAMPAIGN}" || -z "${MODEL_KEY}" ]]; then
  echo "[ERROR] CAMPAIGN and MODEL_KEY are required."
  echo "Example: sbatch --export=ALL,CAMPAIGN=campaign_foo,MODEL_KEY=llava_ov_7b sbatch_eval_model.sbatch"
  exit 2
fi

run_python() {
  if [[ "${USE_APPTAINER}" == "1" ]]; then
    apptainer exec --nv "${SIF}" python3 "$@"
  else
    python3 "$@"
  fi
}

run_python_srun() {
  local already_in_step=0
  if [[ -n "${SLURM_STEP_ID:-}" && "${SLURM_STEP_ID}" != "batch" ]]; then
    already_in_step=1
  fi

  if [[ "${USE_APPTAINER}" == "1" ]]; then
    if [[ "${already_in_step}" == "1" ]]; then
      apptainer exec --nv "${SIF}" python3 "$@"
    else
      srun apptainer exec --nv "${SIF}" python3 "$@"
    fi
  else
    if [[ "${already_in_step}" == "1" ]]; then
      python3 "$@"
    else
      srun python3 "$@"
    fi
  fi
}

if [[ "${USE_APPTAINER}" == "1" && ! -f "${SIF}" ]]; then
  echo "[ERROR] Container not found: ${SIF}"
  exit 2
fi

JOB_TMP_BASE="/tmp/${USER}/gazevqa_eval_${SLURM_JOB_ID:-manual}"
mkdir -p "${JOB_TMP_BASE}/hf/hub" "${JOB_TMP_BASE}/hf/transformers" "${JOB_TMP_BASE}/xdg" "${JOB_TMP_BASE}/pip"
export HF_HOME="${JOB_TMP_BASE}/hf"
export HF_HUB_CACHE="${JOB_TMP_BASE}/hf/hub"
export TRANSFORMERS_CACHE="${JOB_TMP_BASE}/hf/transformers"
export XDG_CACHE_HOME="${JOB_TMP_BASE}/xdg"
export PIP_CACHE_DIR="${JOB_TMP_BASE}/pip"

INSTALL_EXTRA_PACKAGES="${INSTALL_EXTRA_PACKAGES:-0}"
EXTRA_PACKAGES="${EXTRA_PACKAGES:-}"
if [[ "${INSTALL_EXTRA_PACKAGES}" == "1" ]]; then
  if [[ -z "${EXTRA_PACKAGES}" ]]; then
    echo "[WARN] INSTALL_EXTRA_PACKAGES=1 but EXTRA_PACKAGES is empty; skipping install."
  else
    read -r -a EXTRA_PKGS_ARR <<< "${EXTRA_PACKAGES}"
    echo "[INFO] Installing extra packages: ${EXTRA_PACKAGES}"
    run_python_srun -m pip install --user "${EXTRA_PKGS_ARR[@]}"
  fi
fi

EVAL_OUT="$(run_python - <<'PY'
import json,os
from pathlib import Path
root=Path('/work/nvme/bfga/jlee65/gaze_vqa')
campaign_env=os.environ.get('CAMPAIGN','').strip()
model_key=os.environ.get('MODEL_KEY','').strip()
campaign=(root/'runs'/'eval_campaigns'/str(Path(campaign_env))).resolve()
if not campaign.exists():
    p=Path(campaign_env)
    campaign=p.resolve() if p.exists() else campaign
meta=json.loads((campaign/'campaign_meta.json').read_text())
m=meta['model_map'][model_key]
print(m['engine'])
print(m.get('engine_fallback',''))
print(m['model_id'])
print(m.get('tensor_parallel_size',1))
PY
)"

ENGINE="$(echo "${EVAL_OUT}" | sed -n '1p')"
FALLBACK_ENGINE="$(echo "${EVAL_OUT}" | sed -n '2p')"
MODEL_ID="$(echo "${EVAL_OUT}" | sed -n '3p')"
TP_SIZE="$(echo "${EVAL_OUT}" | sed -n '4p')"

echo "[INFO] CAMPAIGN=${CAMPAIGN}"
echo "[INFO] MODEL_KEY=${MODEL_KEY}"
echo "[INFO] ENGINE=${ENGINE} FALLBACK=${FALLBACK_ENGINE}"
echo "[INFO] MODEL_ID=${MODEL_ID}"
echo "[INFO] TP_SIZE=${TP_SIZE}"

SGLANG_PORT="${SGLANG_PORT:-30000}"
VLLM_PORT="${VLLM_PORT:-8000}"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

wait_for_server() {
  local url="$1"
  local retries="${2:-120}"
  for ((i=1; i<=retries; i++)); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

if [[ "${ENGINE}" == "sglang" || ( "${ENGINE}" != "vllm" && "${FALLBACK_ENGINE}" == "sglang" ) ]]; then
  echo "[INFO] Launching SGLang server..."
  if [[ "${USE_APPTAINER}" == "1" ]]; then
    apptainer exec --nv "${SIF}" python3 -m sglang.launch_server \
      --model-path "${MODEL_ID}" \
      --host 0.0.0.0 \
      --port "${SGLANG_PORT}" \
      --tp "${TP_SIZE}" >/tmp/sglang_${SLURM_JOB_ID:-manual}.log 2>&1 &
  else
    python3 -m sglang.launch_server \
      --model-path "${MODEL_ID}" \
      --host 0.0.0.0 \
      --port "${SGLANG_PORT}" \
      --tp "${TP_SIZE}" >/tmp/sglang_${SLURM_JOB_ID:-manual}.log 2>&1 &
  fi
  SERVER_PID=$!
  wait_for_server "http://127.0.0.1:${SGLANG_PORT}/v1/models" 150 || {
    echo "[ERROR] SGLang failed to become ready."
    exit 3
  }
fi

if [[ "${ENGINE}" == "vllm" ]]; then
  echo "[INFO] Launching vLLM server..."
  if [[ "${USE_APPTAINER}" == "1" ]]; then
    apptainer exec --nv "${SIF}" python3 -m vllm.entrypoints.openai.api_server \
      --model "${MODEL_ID}" \
      --host 0.0.0.0 \
      --port "${VLLM_PORT}" \
      --tensor-parallel-size "${TP_SIZE}" >/tmp/vllm_${SLURM_JOB_ID:-manual}.log 2>&1 &
  else
    python3 -m vllm.entrypoints.openai.api_server \
      --model "${MODEL_ID}" \
      --host 0.0.0.0 \
      --port "${VLLM_PORT}" \
      --tensor-parallel-size "${TP_SIZE}" >/tmp/vllm_${SLURM_JOB_ID:-manual}.log 2>&1 &
  fi
  SERVER_PID=$!
  wait_for_server "http://127.0.0.1:${VLLM_PORT}/v1/models" 150 || {
    echo "[ERROR] vLLM failed to become ready."
    exit 4
  }
fi

run_python_srun "${ROOT}/evaluate_benchmark_delta.py" run-model \
  --campaign "${CAMPAIGN}" \
  --model_key "${MODEL_KEY}" \
  --sglang_base_url "http://127.0.0.1:${SGLANG_PORT}" \
  --vllm_base_url "http://127.0.0.1:${VLLM_PORT}" \
  --openai_api_key "${OPENAI_API_KEY:-}" \
  --gemini_api_key "${GEMINI_API_KEY:-}"

echo "[INFO] Done model run for ${MODEL_KEY}."
