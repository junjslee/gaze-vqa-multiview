# Gaze-VQA

Gaze-VQA repository for:
- data generation and benchmark construction (`pipeline/`)
- model evaluation and reporting (`pipeline/eval/`)
- Slurm-based campaign execution (`scripts/sbatch/`)

## Repo layout
- `pipeline/`: data generation pipeline (task orchestration, prompts, state, utilities)
- `pipeline/eval/`: evaluation pipeline (campaigns, inference runners, judging, aggregation)
- `scripts/sbatch/eval/`: evaluation sbatch entrypoints
- `scripts/sbatch/data_generation/`: data-generation sbatch entrypoints
- `runs/`: runtime artifacts (campaign outputs, logs, predictions)
- `tools/`: helper scripts

## Quick notes
- Data-generation documentation is in `pipeline/README.md`.
- Evaluation campaign outputs are currently written to `runs/eval_campaigns/`.
