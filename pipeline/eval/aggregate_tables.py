from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from .metrics_text import compute_text_metrics
from .schemas import iter_jsonl, read_json, utc_now_iso, write_json


TASK_ORDER = [
    "gaze_target_recognition",
    "relative_orientation_reasoning",
    "cross_view_visibility_estimation",
    "viewpoint_based_accessibility",
]


def _accuracy(correct: int, total: int) -> float:
    return round((correct / total), 4) if total > 0 else 0.0


def _load_manifest(campaign_dir: Path) -> Dict[str, Dict[str, Any]]:
    path = campaign_dir / "gt" / "gt_manifest_v1.jsonl"
    return {str(row["sample_uid"]): row for row in iter_jsonl(path)}


def _load_model_meta(campaign_dir: Path) -> Dict[str, Dict[str, Any]]:
    meta = read_json(campaign_dir / "campaign_meta.json")
    return dict(meta.get("model_map") or {})


def _judge_map(campaign_dir: Path, model_key: str) -> Dict[str, str]:
    path = campaign_dir / "judge" / f"{model_key}.jsonl"
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    for row in iter_jsonl(path):
        uid = str(row.get("sample_uid") or "")
        if uid:
            out[uid] = str(row.get("gemini_judge") or "").strip().lower()
    return out


def _prediction_rows(campaign_dir: Path, model_key: str) -> List[Dict[str, Any]]:
    path = campaign_dir / "predictions" / f"{model_key}.jsonl"
    if not path.exists():
        return []
    return list(iter_jsonl(path))


def _merged_rows(
    manifest_by_uid: Dict[str, Dict[str, Any]],
    preds: List[Dict[str, Any]],
    judges: Dict[str, str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in preds:
        uid = str(p.get("sample_uid") or "")
        m = manifest_by_uid.get(uid)
        if not m:
            continue
        rows.append(
            {
                "sample_uid": uid,
                "task_type": m.get("task_type"),
                "question": m.get("question"),
                "groundtruth_answer": m.get("groundtruth_answer"),
                "inference_answer": p.get("inference_answer", ""),
                "gemini_judge": judges.get(uid, ""),
                "error": p.get("error", ""),
                "scene": m.get("scene", "unknown"),
            }
        )
    return rows


def aggregate_campaign(campaign_dir: Path) -> Dict[str, Any]:
    manifest = _load_manifest(campaign_dir)
    model_meta = _load_model_meta(campaign_dir)

    table1_rows: List[Dict[str, Any]] = []
    table2_rows: List[Dict[str, Any]] = []
    leaderboard_rows: List[Dict[str, Any]] = []
    missing_models: List[str] = []

    for model_key, meta in model_meta.items():
        preds = _prediction_rows(campaign_dir, model_key)
        if not preds:
            missing_models.append(model_key)
            continue
        judges = _judge_map(campaign_dir, model_key)
        merged = _merged_rows(manifest_by_uid=manifest, preds=preds, judges=judges)

        per_task_counts = {t: {"correct": 0, "total": 0} for t in TASK_ORDER}
        total = 0
        correct = 0
        for row in merged:
            t = row["task_type"]
            if t not in per_task_counts:
                per_task_counts[t] = {"correct": 0, "total": 0}
            per_task_counts[t]["total"] += 1
            total += 1
            if row["gemini_judge"] == "correct":
                per_task_counts[t]["correct"] += 1
                correct += 1

        task_accs = {
            t: _accuracy(per_task_counts[t]["correct"], per_task_counts[t]["total"])
            for t in TASK_ORDER
        }
        avg_accuracy = round(sum(task_accs.values()) / len(TASK_ORDER), 4)
        overall_accuracy = _accuracy(correct, total)

        table1_rows.append(
            {
                "model_key": model_key,
                "model_label": meta.get("label", model_key),
                "group": meta.get("group", ""),
                "track": meta.get("track", ""),
                "overall_accuracy": overall_accuracy,
                "avg_task_accuracy": avg_accuracy,
                "gaze_target_recognition": task_accs["gaze_target_recognition"],
                "relative_orientation_reasoning": task_accs["relative_orientation_reasoning"],
                "cross_view_visibility_estimation": task_accs["cross_view_visibility_estimation"],
                "viewpoint_based_accessibility": task_accs["viewpoint_based_accessibility"],
                "correct": correct,
                "total": total,
            }
        )

        t2 = compute_text_metrics(merged)
        table2_rows.append(
            {
                "model_key": model_key,
                "model_label": meta.get("label", model_key),
                "group": meta.get("group", ""),
                "track": meta.get("track", ""),
                "sentence_sim": t2.get("sentence_sim"),
                "cider": t2.get("cider"),
                "bleu": t2.get("bleu"),
                "meteor": t2.get("meteor"),
                "rouge_l": t2.get("rouge"),
                "sample_count": t2.get("count", 0),
                "metric_errors": " | ".join(t2.get("errors", [])),
            }
        )

        scene_totals = defaultdict(lambda: {"correct": 0, "total": 0})
        for row in merged:
            s = row.get("scene", "unknown")
            scene_totals[s]["total"] += 1
            if row.get("gemini_judge") == "correct":
                scene_totals[s]["correct"] += 1

        leaderboard_rows.append(
            {
                "model_key": model_key,
                "model_label": meta.get("label", model_key),
                "model_id": meta.get("model_id", ""),
                "engine": meta.get("engine", ""),
                "group": meta.get("group", ""),
                "track": meta.get("track", ""),
                "overall_accuracy": overall_accuracy,
                "avg_task_accuracy": avg_accuracy,
                "scene_accuracy": {
                    s: _accuracy(v["correct"], v["total"])
                    for s, v in sorted(scene_totals.items())
                },
                "correct": correct,
                "total": total,
            }
        )

    # Rank by avg task accuracy, descending.
    table1_rows = sorted(table1_rows, key=lambda r: r["avg_task_accuracy"], reverse=True)
    for i, row in enumerate(table1_rows, start=1):
        row["rank"] = i
    rank_by_key = {r["model_key"]: r["rank"] for r in table1_rows}
    for row in table2_rows:
        row["rank"] = rank_by_key.get(row["model_key"])
    for row in leaderboard_rows:
        row["rank"] = rank_by_key.get(row["model_key"])

    out_dir = campaign_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    table1_csv = out_dir / "table1_accuracy.csv"
    table2_csv = out_dir / "table2_semantic.csv"
    leaderboard_csv = out_dir / "leaderboard_full.csv"
    summary_json = out_dir / "summary.json"

    def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            with path.open("w", encoding="utf-8") as f:
                f.write("")
            return
        cols = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in rows:
                w.writerow(row)

    write_csv(table1_csv, table1_rows)
    write_csv(table2_csv, table2_rows)

    # Flatten leaderboard scene accuracies for CSV.
    flat_leaderboard: List[Dict[str, Any]] = []
    for row in leaderboard_rows:
        flat = dict(row)
        scene_acc = flat.pop("scene_accuracy", {})
        for scene, acc in scene_acc.items():
            flat[f"scene_{scene}_accuracy"] = acc
        flat_leaderboard.append(flat)
    write_csv(leaderboard_csv, flat_leaderboard)

    summary = {
        "generated_at": utc_now_iso(),
        "campaign_dir": str(campaign_dir),
        "table1_rows": len(table1_rows),
        "table2_rows": len(table2_rows),
        "leaderboard_rows": len(leaderboard_rows),
        "missing_models": missing_models,
        "top_model": table1_rows[0]["model_key"] if table1_rows else None,
    }
    write_json(summary_json, summary)
    return {
        "table1_accuracy_csv": table1_csv,
        "table2_semantic_csv": table2_csv,
        "leaderboard_full_csv": leaderboard_csv,
        "summary_json": summary_json,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate campaign predictions into paper-style tables.")
    p.add_argument("--campaign_dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = aggregate_campaign(args.campaign_dir)
    print("[DONE] Aggregation complete.")
    for k, v in out.items():
        if k != "summary":
            print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()

