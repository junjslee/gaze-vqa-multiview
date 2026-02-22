#!/bin/bash
#SBATCH --account=bfga-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --constraint="scratch"
#SBATCH --job-name=gazevqa-merge-campaign
#SBATCH --output=/work/nvme/bfga/jlee65/gaze_vqa/runs/slurm_%x_%j.out
#SBATCH --error=/work/nvme/bfga/jlee65/gaze_vqa/runs/slurm_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --no-requeue

set -euo pipefail

module reset
module list
echo "[INFO] Running on: $(hostname)"
echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"

ROOT="/work/nvme/bfga/jlee65/gaze_vqa"
cd "${ROOT}"

USE_APPTAINER="${USE_APPTAINER:-1}"
SIF="${SIF:-/work/nvme/bfga/jlee65/jun_fm.sif}"
PROPRIETARY_CAMPAIGN="${PROPRIETARY_CAMPAIGN:-}"
OSS_CAMPAIGN="${OSS_CAMPAIGN:-}"
GEMINI_CAMPAIGN="${GEMINI_CAMPAIGN:-}"
OUTPUT_CAMPAIGN="${OUTPUT_CAMPAIGN:-campaign_merged_$(date +%Y%m%d_%H%M%S)}"
STRICT_MANIFEST_MATCH="${STRICT_MANIFEST_MATCH:-1}"

if [[ -z "${PROPRIETARY_CAMPAIGN}" || -z "${OSS_CAMPAIGN}" ]]; then
  echo "[ERROR] PROPRIETARY_CAMPAIGN and OSS_CAMPAIGN are required."
  exit 2
fi

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

SOURCE_CAMPAIGNS=(
  "${PROPRIETARY_CAMPAIGN}"
  "${OSS_CAMPAIGN}"
)
if [[ -n "${GEMINI_CAMPAIGN}" ]]; then
  SOURCE_CAMPAIGNS+=("${GEMINI_CAMPAIGN}")
fi

MERGE_ARGS=(
  "${ROOT}/evaluate_benchmark_delta.py" merge-campaigns
  --source_campaigns
  "${SOURCE_CAMPAIGNS[@]}"
  --output_campaign "${OUTPUT_CAMPAIGN}"
)
if [[ "${STRICT_MANIFEST_MATCH}" == "0" ]]; then
  MERGE_ARGS+=(--no_strict_manifest_match)
fi

run_python "${MERGE_ARGS[@]}"

echo "[INFO] Merge complete: ${OUTPUT_CAMPAIGN}"
