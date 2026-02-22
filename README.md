# Gaze-VQA

Gaze-VQA repository for:
- data generation and benchmark construction (`pipeline/`)
- model evaluation and reporting (`pipeline/eval/`)
- Slurm-based campaign execution (`sbatch_*.sbatch`)

## Repo layout
- `pipeline/`: data generation pipeline (task orchestration, prompts, state, utilities)
- `pipeline/eval/`: evaluation pipeline (campaigns, inference runners, judging, aggregation)
- `runs/`: runtime artifacts (campaign outputs, logs, predictions)
- `tools/`: helper scripts

## Quick notes
- Data-generation documentation is in `pipeline/README.md`.
- Evaluation campaign outputs are currently written to `runs/eval_campaigns/`.
