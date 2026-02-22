# Gaze-VQA Eval Pipeline

This package integrates benchmark freezing, inference, judging, and aggregation.

## Entry point

Use:

```bash
python3 gaze_vqa/evaluate_benchmark_delta.py <subcommand> ...
```

Subcommands:

- `freeze-gt`
- `build-review`
- `apply-review`
- `review-status`
- `run-model`
- `run-track`
- `judge`
- `aggregate`

## Quick flow

```bash
python3 gaze_vqa/evaluate_benchmark_delta.py freeze-gt --campaign campaign_foo
python3 gaze_vqa/evaluate_benchmark_delta.py build-review --campaign campaign_foo
# Review in HTML, export decisions CSV/JSON, then apply:
python3 gaze_vqa/evaluate_benchmark_delta.py apply-review --campaign campaign_foo --decisions /path/to/review_decisions.csv
python3 gaze_vqa/evaluate_benchmark_delta.py run-track --campaign campaign_foo --track A
python3 gaze_vqa/evaluate_benchmark_delta.py run-track --campaign campaign_foo --track B
python3 gaze_vqa/evaluate_benchmark_delta.py judge --campaign campaign_foo --judge_model gemini-3.1-pro-preview
python3 gaze_vqa/evaluate_benchmark_delta.py aggregate --campaign campaign_foo
```

## One-Time Review Flow

1. `freeze-gt` creates campaign + frozen GT.
2. `build-review` creates:
   - `review/index.html` (image-centric reviewer)
   - `review/review_items.json`
   - `review/decisions_template.csv`
   - `review/thumbs/`
3. Approve/reject at frame bundle level in reviewer and export decisions.
4. `apply-review` writes `gt/gt_manifest_v1_reviewed.jsonl` and updates campaign active manifest.
5. All later `run-model` / `run-track` / `judge` use campaign active manifest automatically.

## Delta/Apptainer scripts

- `gaze_vqa/scripts/sbatch/eval/eval_model.sbatch`: one model run.
- `gaze_vqa/scripts/sbatch/eval/eval_campaign.sbatch`: sequential campaign run.

Both scripts default to:

- `USE_APPTAINER=1`
- `SIF=/work/nvme/bfga/jlee65/jun_fm.sif`

and write outputs under:

`gaze_vqa/runs/eval_campaigns/<campaign_name>/`
