from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics_text import compute_text_metrics
from .schemas import iter_jsonl, read_json, utc_now_iso, write_json


TASK_ORDER = [
    "gaze_target_recognition",
    "relative_orientation_reasoning",
    "cross_view_visibility_estimation",
    "viewpoint_based_accessibility",
]
DEFAULT_NO_OVERLAP_EXCLUDED_MODEL_KEYS = ("gemini30flash",)


def _accuracy(correct: int, total: int) -> float:
    return round((correct / total), 4) if total > 0 else 0.0


def _load_manifest(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    path = manifest_path.resolve()
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


def _resolve_manifest_path(campaign_dir: Path, manifest_path_override: Optional[Path] = None) -> Path:
    if manifest_path_override is not None:
        p = manifest_path_override.resolve()
        if not p.exists():
            raise FileNotFoundError(f"Manifest override not found: {p}")
        return p
    fallback = campaign_dir / "gt" / "gt_manifest_v1.jsonl"
    return fallback.resolve()


def _rank_tables(
    table1_rows: List[Dict[str, Any]],
    table2_rows: List[Dict[str, Any]],
    leaderboard_rows: List[Dict[str, Any]],
) -> None:
    table1_rows.sort(key=lambda r: r["avg_task_accuracy"], reverse=True)
    for i, row in enumerate(table1_rows, start=1):
        row["rank"] = i
    rank_by_key = {r["model_key"]: r["rank"] for r in table1_rows}
    for row in table2_rows:
        row["rank"] = rank_by_key.get(row["model_key"])
    for row in leaderboard_rows:
        row["rank"] = rank_by_key.get(row["model_key"])


def _flatten_leaderboard_rows(leaderboard_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat_leaderboard: List[Dict[str, Any]] = []
    for row in leaderboard_rows:
        flat = dict(row)
        scene_acc = flat.pop("scene_accuracy", {})
        for scene, acc in scene_acc.items():
            flat[f"scene_{scene}_accuracy"] = acc
        flat_leaderboard.append(flat)
    return flat_leaderboard


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
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


def aggregate_campaign(
    campaign_dir: Path,
    *,
    model_sources: Optional[Dict[str, Path]] = None,
    model_meta_override: Optional[Dict[str, Dict[str, Any]]] = None,
    manifest_path_override: Optional[Path] = None,
    no_overlap_excluded_model_keys: Optional[List[str]] = None,
    summary_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    manifest_path = _resolve_manifest_path(campaign_dir, manifest_path_override=manifest_path_override)
    manifest = _load_manifest(manifest_path)
    model_meta = dict(model_meta_override or _load_model_meta(campaign_dir))
    model_sources = dict(model_sources or {})

    table1_rows: List[Dict[str, Any]] = []
    table2_rows: List[Dict[str, Any]] = []
    leaderboard_rows: List[Dict[str, Any]] = []
    missing_models: List[str] = []
    no_overlap_table1_rows: List[Dict[str, Any]] = []
    no_overlap_table2_rows: List[Dict[str, Any]] = []
    no_overlap_leaderboard_rows: List[Dict[str, Any]] = []
    no_overlap_missing_models: List[str] = []

    excluded_no_overlap_keys = {
        str(k).strip() for k in (no_overlap_excluded_model_keys or list(DEFAULT_NO_OVERLAP_EXCLUDED_MODEL_KEYS))
        if str(k).strip()
    }
    excluded_keys_present: List[str] = []

    for model_key, meta in model_meta.items():
        source_campaign = model_sources.get(model_key, campaign_dir).resolve()
        preds = _prediction_rows(source_campaign, model_key)
        include_no_overlap = model_key not in excluded_no_overlap_keys
        if model_key in excluded_no_overlap_keys:
            excluded_keys_present.append(model_key)
        if not preds:
            missing_models.append(model_key)
            if include_no_overlap:
                no_overlap_missing_models.append(model_key)
            continue
        judges = _judge_map(source_campaign, model_key)
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

        table1_row = {
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
        table1_rows.append(dict(table1_row))
        if include_no_overlap:
            no_overlap_table1_rows.append(dict(table1_row))

        t2 = compute_text_metrics(merged)
        table2_row = {
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
        table2_rows.append(dict(table2_row))
        if include_no_overlap:
            no_overlap_table2_rows.append(dict(table2_row))

        scene_totals = defaultdict(lambda: {"correct": 0, "total": 0})
        for row in merged:
            s = row.get("scene", "unknown")
            scene_totals[s]["total"] += 1
            if row.get("gemini_judge") == "correct":
                scene_totals[s]["correct"] += 1

        leaderboard_row = {
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
        leaderboard_rows.append(dict(leaderboard_row))
        if include_no_overlap:
            no_overlap_leaderboard_rows.append(dict(leaderboard_row))

    _rank_tables(table1_rows=table1_rows, table2_rows=table2_rows, leaderboard_rows=leaderboard_rows)
    _rank_tables(
        table1_rows=no_overlap_table1_rows,
        table2_rows=no_overlap_table2_rows,
        leaderboard_rows=no_overlap_leaderboard_rows,
    )

    out_dir = campaign_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    table1_csv = out_dir / "table1_accuracy.csv"
    table2_csv = out_dir / "table2_semantic.csv"
    leaderboard_csv = out_dir / "leaderboard_full.csv"
    no_overlap_table1_csv = out_dir / "table1_accuracy_no_overlap.csv"
    no_overlap_table2_csv = out_dir / "table2_semantic_no_overlap.csv"
    no_overlap_leaderboard_csv = out_dir / "leaderboard_full_no_overlap.csv"
    summary_json = out_dir / "summary.json"

    _write_csv(table1_csv, table1_rows)
    _write_csv(table2_csv, table2_rows)
    _write_csv(leaderboard_csv, _flatten_leaderboard_rows(leaderboard_rows))
    _write_csv(no_overlap_table1_csv, no_overlap_table1_rows)
    _write_csv(no_overlap_table2_csv, no_overlap_table2_rows)
    _write_csv(no_overlap_leaderboard_csv, _flatten_leaderboard_rows(no_overlap_leaderboard_rows))

    summary = {
        "generated_at": utc_now_iso(),
        "campaign_dir": str(campaign_dir),
        "manifest_path": str(manifest_path),
        "table1_rows": len(table1_rows),
        "table2_rows": len(table2_rows),
        "leaderboard_rows": len(leaderboard_rows),
        "missing_models": missing_models,
        "top_model": table1_rows[0]["model_key"] if table1_rows else None,
        "no_overlap_excluded_model_keys": sorted(excluded_no_overlap_keys),
        "no_overlap_excluded_model_keys_present": sorted(set(excluded_keys_present)),
        "no_overlap_table1_rows": len(no_overlap_table1_rows),
        "no_overlap_table2_rows": len(no_overlap_table2_rows),
        "no_overlap_leaderboard_rows": len(no_overlap_leaderboard_rows),
        "no_overlap_missing_models": no_overlap_missing_models,
        "no_overlap_top_model": no_overlap_table1_rows[0]["model_key"] if no_overlap_table1_rows else None,
    }
    if summary_extra:
        summary.update(summary_extra)
    write_json(summary_json, summary)
    return {
        "table1_accuracy_csv": table1_csv,
        "table2_semantic_csv": table2_csv,
        "leaderboard_full_csv": leaderboard_csv,
        "table1_accuracy_no_overlap_csv": no_overlap_table1_csv,
        "table2_semantic_no_overlap_csv": no_overlap_table2_csv,
        "leaderboard_full_no_overlap_csv": no_overlap_leaderboard_csv,
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
