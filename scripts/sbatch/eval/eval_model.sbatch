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
# sbatch --gpus-per-node=1 --cpus-per-task=8 --mem=64g --export=ALL,CAMPAIGN=...,MODEL_KEY=... scripts/sbatch/eval/eval_model.sbatch

CAMPAIGN="${CAMPAIGN:-}"
MODEL_KEY="${MODEL_KEY:-}"
SIF="${SIF:-/work/nvme/bfga/jlee65/jun_fm.sif}"
USE_APPTAINER="${USE_APPTAINER:-1}"
REQUEST_INTERVAL="${REQUEST_INTERVAL:-0.0}"
RESET_PREDICTIONS="${RESET_PREDICTIONS:-0}"
SERVER_READY_RETRIES="${SERVER_READY_RETRIES:-900}"
SERVER_READY_SLEEP_S="${SERVER_READY_SLEEP_S:-2}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-${ROOT}/runs/server_logs}"
SERVER_LOG_TAIL_LINES="${SERVER_LOG_TAIL_LINES:-200}"
LIBFIX_DIR="${LIBFIX_DIR:-/tmp/${USER}/gazevqa_libfix_${SLURM_JOB_ID:-manual}}"
VLLM_MM_HASHER_ALGORITHM="${VLLM_MM_HASHER_ALGORITHM:-sha256}"
VLLM_LIMIT_MM_PER_PROMPT_IMAGE="${VLLM_LIMIT_MM_PER_PROMPT_IMAGE:-}"
VLLM_LIMIT_MM_PER_PROMPT_IMAGE_AUTO="${VLLM_LIMIT_MM_PER_PROMPT_IMAGE_AUTO:-1}"
SGLANG_LIMIT_MM_DATA_PER_REQUEST="${SGLANG_LIMIT_MM_DATA_PER_REQUEST:-}"
FORCE_ENGINE="${FORCE_ENGINE:-}"
SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-triton}"
SGLANG_SAMPLING_BACKEND="${SGLANG_SAMPLING_BACKEND:-pytorch}"
SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-1}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.62}"
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-2048}"
SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-4096}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-32768}"
SGLANG_MAX_TOTAL_TOKENS="${SGLANG_MAX_TOTAL_TOKENS:-131072}"

if [[ -z "${CAMPAIGN}" || -z "${MODEL_KEY}" ]]; then
  echo "[ERROR] CAMPAIGN and MODEL_KEY are required."
  echo "Example: sbatch --export=ALL,CAMPAIGN=campaign_foo,MODEL_KEY=llava_ov_7b scripts/sbatch/eval/eval_model.sbatch"
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

# sglang's sgl_kernel requires libnuma.so.1 on A40 nodes. Make it available in-container.
if [[ "${USE_APPTAINER}" == "1" ]]; then
  mkdir -p "${LIBFIX_DIR}"
  if [[ ! -f "${LIBFIX_DIR}/libnuma.so.1.0.0" ]]; then
    if [[ -f "/usr/lib64/libnuma.so.1.0.0" ]]; then
      cp -f /usr/lib64/libnuma.so.1.0.0 "${LIBFIX_DIR}/libnuma.so.1.0.0"
    elif [[ -f "/lib64/libnuma.so.1.0.0" ]]; then
      cp -f /lib64/libnuma.so.1.0.0 "${LIBFIX_DIR}/libnuma.so.1.0.0"
    fi
  fi
  if [[ -f "${LIBFIX_DIR}/libnuma.so.1.0.0" ]]; then
    ln -sfn libnuma.so.1.0.0 "${LIBFIX_DIR}/libnuma.so.1"
    export APPTAINERENV_LD_LIBRARY_PATH="${LIBFIX_DIR}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs"
    export APPTAINERENV_VLLM_MM_HASHER_ALGORITHM="${VLLM_MM_HASHER_ALGORITHM}"
    export APPTAINERENV_PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    echo "[INFO] Enabled libnuma compatibility path for Apptainer: ${LIBFIX_DIR}"
  else
    echo "[WARN] Could not stage libnuma.so.1.0.0; SGLang may fail to start."
  fi
fi

# Keep non-container mode behavior consistent with container env.
export VLLM_MM_HASHER_ALGORITHM
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
if [[ "${USE_APPTAINER}" == "1" ]]; then
  export APPTAINERENV_VLLM_MM_HASHER_ALGORITHM="${VLLM_MM_HASHER_ALGORITHM}"
  export APPTAINERENV_PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}"
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

def resolve_manifest_path(meta_obj, campaign_dir):
    for key in ('active_manifest_path', 'gt_manifest_path'):
        raw = str(meta_obj.get(key) or '').strip()
        if not raw:
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = (campaign_dir / p).resolve()
        if p.exists():
            return p
    fallback = (campaign_dir / 'gt' / 'gt_manifest_v1.jsonl').resolve()
    return fallback

mm_limit = 1
limit_override = str(os.environ.get('VLLM_LIMIT_MM_PER_PROMPT_IMAGE', '')).strip()
if limit_override:
    try:
        mm_limit = max(1, int(limit_override))
    except Exception:
        mm_limit = 1
elif str(os.environ.get('VLLM_LIMIT_MM_PER_PROMPT_IMAGE_AUTO', '1')).strip() != '0':
    manifest = resolve_manifest_path(meta, campaign)
    if manifest.exists():
        try:
            with manifest.open('r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    mm_limit = max(mm_limit, len(row.get('image_paths') or []))
        except Exception:
            mm_limit = 1

print(m['engine'])
print(m.get('engine_fallback',''))
print(m['model_id'])
print(m.get('tensor_parallel_size',1))
print(mm_limit)
PY
)"

ENGINE="$(echo "${EVAL_OUT}" | sed -n '1p')"
FALLBACK_ENGINE="$(echo "${EVAL_OUT}" | sed -n '2p')"
MODEL_ID="$(echo "${EVAL_OUT}" | sed -n '3p')"
TP_SIZE="$(echo "${EVAL_OUT}" | sed -n '4p')"
VLLM_MM_IMAGE_LIMIT="$(echo "${EVAL_OUT}" | sed -n '5p')"
if ! [[ "${VLLM_MM_IMAGE_LIMIT}" =~ ^[0-9]+$ ]]; then
  VLLM_MM_IMAGE_LIMIT=1
fi
SGLANG_MM_LIMIT_JSON="${SGLANG_LIMIT_MM_DATA_PER_REQUEST}"
if [[ -z "${SGLANG_MM_LIMIT_JSON}" ]]; then
  SGLANG_MM_LIMIT_JSON="{\"image\": ${VLLM_MM_IMAGE_LIMIT}}"
fi

echo "[INFO] CAMPAIGN=${CAMPAIGN}"
echo "[INFO] MODEL_KEY=${MODEL_KEY}"
echo "[INFO] ENGINE=${ENGINE} FALLBACK=${FALLBACK_ENGINE}"
echo "[INFO] MODEL_ID=${MODEL_ID}"
echo "[INFO] TP_SIZE=${TP_SIZE}"
echo "[INFO] VLLM_MM_IMAGE_LIMIT=${VLLM_MM_IMAGE_LIMIT}"
echo "[INFO] SGLANG_MM_LIMIT_JSON=${SGLANG_MM_LIMIT_JSON}"

SGLANG_PORT="${SGLANG_PORT:-30000}"
VLLM_PORT="${VLLM_PORT:-8000}"
SERVER_PID=""
SERVER_KIND=""
SERVER_LOG=""
LAUNCH_ENGINE=""
mkdir -p "${SERVER_LOG_DIR}"

cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

print_server_log_tail() {
  local path="$1"
  if [[ -n "${path}" && -f "${path}" ]]; then
    echo "[INFO] Showing last ${SERVER_LOG_TAIL_LINES} lines from ${path}"
    tail -n "${SERVER_LOG_TAIL_LINES}" "${path}" || true
  else
    echo "[WARN] Server log not found: ${path}"
  fi
}

wait_for_server() {
  local url="$1"
  local retries="${2:-120}"
  local sleep_s="${3:-2}"
  for ((i=1; i<=retries; i++)); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      echo "[ERROR] Server process exited before readiness check passed."
      return 2
    fi
    sleep "${sleep_s}"
  done
  return 1
}

has_python_module() {
  local mod="$1"
  if run_python - <<PY >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("${mod}") else 1)
PY
  then
    return 0
  fi
  return 1
}

if [[ "${ENGINE}" == "sglang" || "${ENGINE}" == "vllm" ]]; then
  primary_mod="${ENGINE}"
  fallback_mod=""
  if [[ "${FALLBACK_ENGINE}" == "sglang" || "${FALLBACK_ENGINE}" == "vllm" ]]; then
    fallback_mod="${FALLBACK_ENGINE}"
  fi

  if [[ -n "${FORCE_ENGINE}" ]]; then
    if [[ "${FORCE_ENGINE}" != "sglang" && "${FORCE_ENGINE}" != "vllm" ]]; then
      echo "[ERROR] FORCE_ENGINE must be 'sglang' or 'vllm'; got '${FORCE_ENGINE}'."
      exit 7
    fi
    if ! has_python_module "${FORCE_ENGINE}"; then
      echo "[ERROR] FORCE_ENGINE='${FORCE_ENGINE}' requested, but module is unavailable."
      exit 7
    fi
    LAUNCH_ENGINE="${FORCE_ENGINE}"
    echo "[WARN] FORCE_ENGINE override active: ${LAUNCH_ENGINE} (model default=${ENGINE}, fallback=${FALLBACK_ENGINE:-none})"
  elif has_python_module "${primary_mod}"; then
    LAUNCH_ENGINE="${ENGINE}"
  elif [[ -n "${fallback_mod}" ]] && has_python_module "${fallback_mod}"; then
    LAUNCH_ENGINE="${FALLBACK_ENGINE}"
    echo "[WARN] Primary engine module '${primary_mod}' not found; using fallback '${LAUNCH_ENGINE}'."
  else
    echo "[ERROR] Neither primary engine '${primary_mod}' nor fallback '${fallback_mod:-none}' is available."
    echo "[ERROR] Install the missing package(s), then resubmit."
    exit 6
  fi
fi

if [[ "${LAUNCH_ENGINE}" == "sglang" ]]; then
  echo "[INFO] Launching SGLang server..."
  echo "[INFO] SGLang backend: attention=${SGLANG_ATTENTION_BACKEND} sampling=${SGLANG_SAMPLING_BACKEND} disable_cuda_graph=${SGLANG_DISABLE_CUDA_GRAPH}"
  echo "[INFO] SGLang memory: mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC} max_total_tokens=${SGLANG_MAX_TOTAL_TOKENS} chunked_prefill=${SGLANG_CHUNKED_PREFILL_SIZE} max_prefill=${SGLANG_MAX_PREFILL_TOKENS} context_length=${SGLANG_CONTEXT_LENGTH}"
  SERVER_KIND="sglang"
  SERVER_LOG="${SERVER_LOG_DIR}/${MODEL_KEY}_${SERVER_KIND}_${SLURM_JOB_ID:-manual}.log"

  SGLANG_ARGS=(
    --model-path "${MODEL_ID}"
    --trust-remote-code
    --attention-backend "${SGLANG_ATTENTION_BACKEND}"
    --sampling-backend "${SGLANG_SAMPLING_BACKEND}"
    --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
    --max-total-tokens "${SGLANG_MAX_TOTAL_TOKENS}"
    --chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}"
    --max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS}"
    --context-length "${SGLANG_CONTEXT_LENGTH}"
    --limit-mm-data-per-request "${SGLANG_MM_LIMIT_JSON}"
    --host 0.0.0.0
    --port "${SGLANG_PORT}"
    --tp "${TP_SIZE}"
  )
  if [[ "${SGLANG_DISABLE_CUDA_GRAPH}" == "1" ]]; then
    SGLANG_ARGS+=(--disable-cuda-graph)
  fi

  if [[ "${USE_APPTAINER}" == "1" ]]; then
    apptainer exec --nv "${SIF}" python3 -m sglang.launch_server \
      "${SGLANG_ARGS[@]}" >"${SERVER_LOG}" 2>&1 &
  else
    python3 -m sglang.launch_server \
      "${SGLANG_ARGS[@]}" >"${SERVER_LOG}" 2>&1 &
  fi
  SERVER_PID=$!
  wait_for_server "http://127.0.0.1:${SGLANG_PORT}/v1/models" "${SERVER_READY_RETRIES}" "${SERVER_READY_SLEEP_S}" || {
    echo "[ERROR] SGLang failed to become ready (retries=${SERVER_READY_RETRIES}, sleep_s=${SERVER_READY_SLEEP_S})."
    print_server_log_tail "${SERVER_LOG}"
    exit 3
  }
fi

if [[ "${LAUNCH_ENGINE}" == "vllm" ]]; then
  echo "[INFO] Launching vLLM server..."
  SERVER_KIND="vllm"
  SERVER_LOG="${SERVER_LOG_DIR}/${MODEL_KEY}_${SERVER_KIND}_${SLURM_JOB_ID:-manual}.log"
  VLLM_ARGS=(
    "${MODEL_ID}"
    --trust-remote-code
    --host 0.0.0.0
    --port "${VLLM_PORT}"
    --tensor-parallel-size "${TP_SIZE}"
    --limit-mm-per-prompt.image "${VLLM_MM_IMAGE_LIMIT}"
  )
  # vLLM docs require overriding the architecture name for DeepSeek-VL2.
  if [[ "${MODEL_ID}" == deepseek-ai/deepseek-vl2* ]]; then
    echo "[INFO] Applying vLLM DeepSeek-VL2 hf-overrides."
    VLLM_ARGS+=(--hf-overrides '{"architectures":["DeepseekVLV2ForCausalLM"]}')
  fi
  if [[ "${USE_APPTAINER}" == "1" ]]; then
    apptainer exec --nv "${SIF}" vllm serve \
      "${VLLM_ARGS[@]}" >"${SERVER_LOG}" 2>&1 &
  else
    vllm serve \
      "${VLLM_ARGS[@]}" >"${SERVER_LOG}" 2>&1 &
  fi
  SERVER_PID=$!
  wait_for_server "http://127.0.0.1:${VLLM_PORT}/v1/models" "${SERVER_READY_RETRIES}" "${SERVER_READY_SLEEP_S}" || {
    echo "[ERROR] vLLM failed to become ready (retries=${SERVER_READY_RETRIES}, sleep_s=${SERVER_READY_SLEEP_S})."
    print_server_log_tail "${SERVER_LOG}"
    exit 4
  }
fi

RUN_MODEL_ARGS=(
  "${ROOT}/evaluate_benchmark_delta.py" run-model
  --campaign "${CAMPAIGN}"
  --model_key "${MODEL_KEY}"
  --request_interval "${REQUEST_INTERVAL}"
  --sglang_base_url "http://127.0.0.1:${SGLANG_PORT}"
  --vllm_base_url "http://127.0.0.1:${VLLM_PORT}"
)
if [[ "${RESET_PREDICTIONS}" == "1" ]]; then
  RUN_MODEL_ARGS+=(--reset_predictions)
fi

run_python_srun "${RUN_MODEL_ARGS[@]}"

echo "[INFO] Done model run for ${MODEL_KEY}."
