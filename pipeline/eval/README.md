# Gaze-VQA Eval Pipeline

This package integrates benchmark freezing, inference, judging, and aggregation.

## Entry point

Use:

```bash
python3 gaze_vqa/evaluate_benchmark_delta.py <subcommand> ...
```

Subcommands:

- `freeze-gt`
- `run-model`
- `run-track`
- `judge`
- `aggregate`

## Quick flow

```bash
python3 gaze_vqa/evaluate_benchmark_delta.py freeze-gt --campaign campaign_foo
python3 gaze_vqa/evaluate_benchmark_delta.py run-track --campaign campaign_foo --track A
python3 gaze_vqa/evaluate_benchmark_delta.py run-track --campaign campaign_foo --track B
python3 gaze_vqa/evaluate_benchmark_delta.py judge --campaign campaign_foo --judge_model gemini-2.5-flash
python3 gaze_vqa/evaluate_benchmark_delta.py aggregate --campaign campaign_foo
```

## Delta/Apptainer scripts

- `gaze_vqa/sbatch_eval_model.sbatch`: one model run.
- `gaze_vqa/sbatch_eval_campaign.sbatch`: sequential campaign run.

Both scripts default to:

- `USE_APPTAINER=1`
- `SIF=/work/nvme/bfga/jlee65/jun_fm.sif`

and write outputs under:

`gaze_vqa/runs/eval_campaigns/<campaign_name>/`

