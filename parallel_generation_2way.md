# 2-Way Parallel Generation Runbook

## Goal
Run generation in two split-scoped jobs in parallel, without changing task labeling logic.

- Branch `ck`: `Commons,Kitchen`
- Branch `ls`: `Lab,Shop`

This changes scheduling only. Task prompts, gates, and acceptance logic stay unchanged.

## Why 2-way (not 4-way right now)
With one Gemini API key and observed transient timeout/503 behavior, 4-way is likely to increase API contention and reduce clean yield. Start with 2-way.

## Prerequisites
- `gaze_vqa/sbatch_task1_teacher_scale.sbatch`
- `gaze_vqa/dispatch_task1_teacher_parallel_2way.sh`
- `gaze_vqa/tools/merge_benchmark_runs.py`

## 1) Submit both branches
From repo root:

```bash
bash gaze_vqa/dispatch_task1_teacher_parallel_2way.sh
```

Optional env overrides (examples):

```bash
TASK1_TEACHER_MODEL=gemini-3-flash-preview \
TASK4_VERIFIER_MODEL=gemini-3-flash-preview \
GEMINI_FALLBACK_MODELS=gemini-2.5-flash,gemini-2.0-flash,gemini-1.5-flash \
TARGET_TASK1=999999 TARGET_TASK2=999999 TARGET_TASK3=999999 TARGET_TASK4=999999 \
VLM_TIMEOUT_S=150 \
bash gaze_vqa/dispatch_task1_teacher_parallel_2way.sh
```

The dispatcher uses `--export=ALL`, so these env vars are inherited by both jobs.

## 2) Resume a stopped branch (manual)
If one branch stops, resubmit only that branch using `RESUME_RUN_DIR`.

Example (`ck` only):

```bash
SPLITS_CK=Commons,Kitchen \
SUFFIX_CK=ck \
RESUME_RUN_DIR_CK=/work/nvme/bfga/jlee65/gaze_vqa/runs/<run_ck_dir> \
SPLITS_LS=Lab,Shop SUFFIX_LS=ls DRY_RUN=1 \
bash gaze_vqa/dispatch_task1_teacher_parallel_2way.sh
```

Then submit only the branch you need using the sbatch script directly if preferred:

```bash
sbatch --export=ALL,SPLITS=Commons,Kitchen,RUN_NAME_SUFFIX=ck,RESUME_RUN_DIR=/work/nvme/bfga/jlee65/gaze_vqa/runs/<run_ck_dir> \
  gaze_vqa/sbatch_task1_teacher_scale.sbatch
```

Rule: never run two jobs against the same run directory.

## 3) Merge both completed outputs into one benchmark
After both runs complete:

```bash
python3 gaze_vqa/tools/merge_benchmark_runs.py \
  --inputs \
    /work/nvme/bfga/jlee65/gaze_vqa/runs/<run_ck_dir> \
    /work/nvme/bfga/jlee65/gaze_vqa/runs/<run_ls_dir> \
  --output /work/nvme/bfga/jlee65/gaze_vqa/runs/<merge_name>/benchmark_gazevqa.json \
  --merge_name <merge_name>
```

Outputs:
- Merged benchmark JSON at `--output`
- Merge report JSON at `<output_stem>.merge_report.json`

Merge behavior:
- deterministic sample sort
- dedup by stable identity key
- keep-first on identity collision
- recompute `counts` from merged samples
- merge provenance stored in `meta.merge`

## 4) Continue normal evaluation flow
Use merged benchmark as source for campaign freeze/review/eval:

```bash
python3 gaze_vqa/evaluate_benchmark_delta.py freeze-gt \
  --campaign <campaign_name> \
  --benchmark_path /work/nvme/bfga/jlee65/gaze_vqa/runs/<merge_name>/benchmark_gazevqa.json
```

Then run:
- `build-review`
- `apply-review`
- `run-track`
- `judge`
- `aggregate`

## Monitoring quick commands
```bash
squeue -u "$USER"
tail -f /work/nvme/bfga/jlee65/gaze_vqa/runs/slurm_gazevqa-teacher-scale_<jobid>.out
```
