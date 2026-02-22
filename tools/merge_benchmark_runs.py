#!/usr/bin/env python3
"""Merge split-scoped benchmark_gazevqa.json outputs into one canonical file.

This utility is orchestration-only: it does not alter task logic or labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_space(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def _normalize_lower(text: Any) -> str:
    return _normalize_space(text).lower()


def _resolve_benchmark_path(raw: str) -> Path:
    p = Path(raw).expanduser().resolve()
    if p.is_dir():
        candidate = p / "benchmark_gazevqa.json"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"No benchmark_gazevqa.json under run dir: {p}")
    if p.is_file():
        return p
    raise FileNotFoundError(f"Input path does not exist: {p}")


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Failed to parse JSON: {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object at {path}")
    samples = obj.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"Expected 'samples' list at {path}")
    return obj


def _sample_image_signature(sample: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    images = sample.get("input_images")
    out: List[Tuple[str, str]] = []
    if isinstance(images, list):
        for item in images:
            if not isinstance(item, dict):
                continue
            cam = _normalize_lower(item.get("cam"))
            img = str(item.get("image") or "").strip()
            stem = Path(img).name if img else ""
            out.append((cam, stem))
    return tuple(sorted(out))


def _stable_identity_tuple(sample: Dict[str, Any]) -> Tuple[Any, ...]:
    task_id = int(sample.get("task_id") or 0)
    task_type = _normalize_lower(sample.get("task_type"))
    scene = _normalize_lower(sample.get("scene"))
    timestamp = _normalize_space(sample.get("timestamp"))
    question = _normalize_lower(sample.get("question"))
    cams = tuple(sorted(_normalize_lower(c) for c in (sample.get("input_cams") or []) if str(c).strip()))
    imgs = _sample_image_signature(sample)
    return (task_id, task_type, scene, timestamp, question, cams, imgs)


def _stable_identity_key(sample: Dict[str, Any]) -> str:
    data = json.dumps(_stable_identity_tuple(sample), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha1(data.encode("utf-8")).hexdigest()


def _sample_sort_key(sample: Dict[str, Any]) -> Tuple[Any, ...]:
    task_id = int(sample.get("task_id") or 0)
    scene = _normalize_lower(sample.get("scene"))
    timestamp = _normalize_space(sample.get("timestamp"))
    question = _normalize_lower(sample.get("question"))
    return (scene, timestamp, task_id, question, _stable_identity_key(sample))


def _sample_fingerprint(sample: Dict[str, Any]) -> str:
    text = json.dumps(sample, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _to_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _merge_numeric_maps(items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for obj in items:
        if not isinstance(obj, dict):
            continue
        for k, v in obj.items():
            num = _to_number(v)
            if num is not None:
                prev = _to_number(merged.get(k))
                merged[k] = (prev or 0.0) + num
            elif k not in merged:
                merged[k] = v
    # Cast integral floats back to int for cleaner output.
    for k, v in list(merged.items()):
        if isinstance(v, float) and float(v).is_integer():
            merged[k] = int(v)
    return merged


def _compute_counts(samples: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"task1": 0, "task2": 0, "task3": 0, "task4": 0, "bundle": 0}
    bundles = set()
    for s in samples:
        task_id = int(s.get("task_id") or 0)
        if task_id == 1:
            counts["task1"] += 1
        elif task_id == 2:
            counts["task2"] += 1
        elif task_id == 3:
            counts["task3"] += 1
        elif task_id == 4:
            counts["task4"] += 1

        scene = _normalize_lower(s.get("scene"))
        timestamp = _normalize_space(s.get("timestamp"))
        if scene or timestamp:
            bundles.add((scene, timestamp))
    counts["bundle"] = len(bundles)
    return counts


def merge_benchmarks(
    inputs: List[Path],
    output_path: Path,
    merge_name: str,
    report_path: Path,
) -> Dict[str, Any]:
    resolved = [_resolve_benchmark_path(str(p)) for p in inputs]

    loaded: List[Tuple[Path, Dict[str, Any]]] = []
    for p in resolved:
        loaded.append((p, _load_json(p)))

    dedup: Dict[str, Dict[str, Any]] = {}
    first_seen: Dict[str, Dict[str, Any]] = {}
    exact_duplicates = 0
    conflict_duplicates = 0
    conflict_examples: List[Dict[str, Any]] = []

    source_counts: Dict[str, int] = {}
    source_runs: List[str] = []

    for src_path, bench in loaded:
        src_key = str(src_path)
        source_counts[src_key] = len(bench.get("samples") or [])
        source_runs.append(src_path.parent.name)

        for idx, sample in enumerate(bench.get("samples") or []):
            if not isinstance(sample, dict):
                continue
            key = _stable_identity_key(sample)
            fp = _sample_fingerprint(sample)
            if key not in dedup:
                dedup[key] = sample
                first_seen[key] = {
                    "source": src_key,
                    "index": idx,
                    "fingerprint": fp,
                }
                continue

            prior = first_seen[key]
            if fp == prior.get("fingerprint"):
                exact_duplicates += 1
            else:
                conflict_duplicates += 1
                if len(conflict_examples) < 50:
                    conflict_examples.append(
                        {
                            "identity_key": key,
                            "kept_source": prior.get("source"),
                            "kept_index": prior.get("index"),
                            "drop_source": src_key,
                            "drop_index": idx,
                        }
                    )

    merged_samples = sorted(dedup.values(), key=_sample_sort_key)

    reject_stats = _merge_numeric_maps((obj.get("reject_stats") or {}) for _, obj in loaded)
    frame_stats = _merge_numeric_maps((obj.get("frame_stats") or {}) for _, obj in loaded)
    counts = _compute_counts(merged_samples)

    base_meta = deepcopy(loaded[0][1].get("meta") or {}) if loaded else {}
    base_meta.update(
        {
            "run_name": merge_name,
            "merge": {
                "created_at": _utc_now_iso(),
                "merge_name": merge_name,
                "sources": [str(p) for p in resolved],
                "source_runs": source_runs,
                "source_sample_counts": source_counts,
                "dedup_policy": "stable_identity_keep_first",
                "exact_duplicates_dropped": exact_duplicates,
                "conflict_duplicates_dropped": conflict_duplicates,
                "conflict_examples": conflict_examples,
                "merged_sample_count": len(merged_samples),
            },
        }
    )

    merged_obj = {
        "meta": base_meta,
        "counts": counts,
        "reject_stats": reject_stats,
        "frame_stats": frame_stats,
        "samples": merged_samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged_obj, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "created_at": _utc_now_iso(),
        "merge_name": merge_name,
        "output_path": str(output_path),
        "output_sample_count": len(merged_samples),
        "counts": counts,
        "sources": [str(p) for p in resolved],
        "source_sample_counts": source_counts,
        "exact_duplicates_dropped": exact_duplicates,
        "conflict_duplicates_dropped": conflict_duplicates,
        "conflict_examples": conflict_examples,
        "reject_stats_merged": reject_stats,
        "frame_stats_merged": frame_stats,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge benchmark_gazevqa.json files from split runs")
    p.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Run dirs or benchmark_gazevqa.json paths (e.g., runs/<run_ck> runs/<run_ls>)",
    )
    p.add_argument("--output", type=str, required=True, help="Output merged benchmark_gazevqa.json path")
    p.add_argument(
        "--merge_name",
        type=str,
        default="",
        help="Optional name stored in meta.run_name for the merged artifact",
    )
    p.add_argument(
        "--report",
        type=str,
        default="",
        help="Optional report JSON path (default: <output>.merge_report.json)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    merge_name = (args.merge_name or output.parent.name or "merged_benchmark").strip()

    report = (
        Path(args.report).expanduser().resolve()
        if args.report
        else output.with_name(output.stem + ".merge_report.json")
    )

    result = merge_benchmarks(
        inputs=[Path(x) for x in args.inputs],
        output_path=output,
        merge_name=merge_name,
        report_path=report,
    )

    print("[OK] Merged benchmark written:", result["output_path"])
    print("[OK] Report written:", report)
    print("[OK] Samples:", result["output_sample_count"])
    print("[OK] Counts:", result["counts"])
    print(
        "[OK] Dedup drops: exact=%d conflict=%d"
        % (result["exact_duplicates_dropped"], result["conflict_duplicates_dropped"])
    )


if __name__ == "__main__":
    main()
