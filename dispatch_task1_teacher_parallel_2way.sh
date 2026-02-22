#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-${SCRIPT_DIR}}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${ROOT}/sbatch_task1_teacher_scale.sbatch}"

SPLITS_CK="${SPLITS_CK:-Commons,Kitchen}"
SPLITS_LS="${SPLITS_LS:-Lab,Shop}"
SUFFIX_CK="${SUFFIX_CK:-ck}"
SUFFIX_LS="${SUFFIX_LS:-ls}"
RESUME_RUN_DIR_CK="${RESUME_RUN_DIR_CK:-}"
RESUME_RUN_DIR_LS="${RESUME_RUN_DIR_LS:-}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
  echo "[ERROR] Missing sbatch script: ${SBATCH_SCRIPT}" >&2
  exit 2
fi

submit_branch() {
  local branch_name="$1"
  local splits="$2"
  local suffix="$3"
  local resume_dir="$4"

  local cmd=(sbatch --export=ALL "${SBATCH_SCRIPT}")

  echo "[INFO] Submitting ${branch_name}: SPLITS=${splits} RUN_NAME_SUFFIX=${suffix}" >&2
  if [[ -n "${resume_dir}" ]]; then
    echo "[INFO] ${branch_name} resume dir: ${resume_dir}" >&2
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY-RUN] ' >&2
    printf 'SPLITS=%q RUN_NAME_SUFFIX=%q ' "${splits}" "${suffix}" >&2
    if [[ -n "${resume_dir}" ]]; then
      printf 'RESUME_RUN_DIR=%q ' "${resume_dir}" >&2
    fi
    printf '%q ' "${cmd[@]}" >&2
    printf '\n' >&2
    return 0
  fi

  local out
  if [[ -n "${resume_dir}" ]]; then
    out="$(SPLITS="${splits}" RUN_NAME_SUFFIX="${suffix}" RESUME_RUN_DIR="${resume_dir}" "${cmd[@]}")"
  else
    out="$(SPLITS="${splits}" RUN_NAME_SUFFIX="${suffix}" "${cmd[@]}")"
  fi
  echo "${out}" >&2
  local job_id
  job_id="$(awk '/Submitted batch job/{print $4}' <<< "${out}" | tail -n 1)"
  if [[ -z "${job_id}" ]]; then
    echo "[ERROR] Could not parse job ID for ${branch_name}" >&2
    exit 3
  fi

  echo "${job_id}"
}

echo "[INFO] Two-way parallel generation dispatcher"
echo "[INFO] sbatch script: ${SBATCH_SCRIPT}"
echo "[INFO] Inherited env: TASK1_TEACHER_MODEL, TASK4_VERIFIER_MODEL, GEMINI_FALLBACK_MODELS, TARGET_TASK1..4, VLM_TIMEOUT_S, etc."

if [[ "${DRY_RUN}" == "1" ]]; then
  submit_branch "branch_ck" "${SPLITS_CK}" "${SUFFIX_CK}" "${RESUME_RUN_DIR_CK}"
  submit_branch "branch_ls" "${SPLITS_LS}" "${SUFFIX_LS}" "${RESUME_RUN_DIR_LS}"
  echo "[INFO] Dry run complete."
  exit 0
fi

job_ck="$(submit_branch "branch_ck" "${SPLITS_CK}" "${SUFFIX_CK}" "${RESUME_RUN_DIR_CK}" | tail -n 1)"
job_ls="$(submit_branch "branch_ls" "${SPLITS_LS}" "${SUFFIX_LS}" "${RESUME_RUN_DIR_LS}" | tail -n 1)"

echo "[INFO] Submitted jobs:"
echo "  Commons+Kitchen: ${job_ck}"
echo "  Lab+Shop:        ${job_ls}"
echo
echo "[INFO] Monitor:"
echo "  squeue -u ${USER}"
echo "  tail -f ${ROOT}/runs/slurm_gazevqa-teacher-scale_${job_ck}.out"
echo "  tail -f ${ROOT}/runs/slurm_gazevqa-teacher-scale_${job_ls}.out"
