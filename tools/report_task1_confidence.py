#!/usr/bin/env python3
"""
Report Task1 confidence distribution from a benchmark JSON or run directory.

Usage:
  python3 gaze_vqa/tools/report_task1_confidence.py --run gaze_vqa/runs/run_YYYYMMDD_xxx_v3
  python3 gaze_vqa/tools/report_task1_confidence.py --file path/to/benchmark_gazevqa.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median


def _load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def _find_benchmark_file(run_dir: Path) -> Path | None:
    # Prefer common benchmark filenames.
    preferred = [
        "benchmark_gazevqa.json",
        "benchmark_delta.json",
        "benchmark_gazevqa_delta.json",
        "benchmark.json",
    ]
    for name in preferred:
        p = run_dir / name
        if p.exists():
            return p
    # Fallback: any json that looks like it contains samples.
    for p in sorted(run_dir.glob("*.json")):
        try:
            data = _load_json(p)
        except Exception:
            continue
        if isinstance(data, dict) and any(k in data for k in ("samples", "data")):
            return p
        if isinstance(data, list) and data and isinstance(data[0], dict) and ("task_id" in data[0] or "task" in data[0]):
            return p
    return None


def _extract_samples(data):
    if isinstance(data, dict):
        if "samples" in data and isinstance(data["samples"], list):
            return data["samples"]
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
    if isinstance(data, list):
        return data
    return []


def _percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    idx = int(round((p / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[max(0, min(len(sorted_vals) - 1, idx))]


def _histogram(values, bins=10):
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [(lo, hi, len(values))]
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        i = int((v - lo) / step)
        if i == bins:
            i -= 1
        counts[i] += 1
    out = []
    for i, c in enumerate(counts):
        b0 = lo + i * step
        b1 = lo + (i + 1) * step
        out.append((b0, b1, c))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None, help="Run directory (e.g., gaze_vqa/runs/run_..._v3)")
    ap.add_argument("--file", type=str, default=None, help="Benchmark JSON file")
    ap.add_argument("--bins", type=int, default=10, help="Histogram bins")
    args = ap.parse_args()

    src = None
    if args.file:
        src = Path(args.file)
        if not src.exists():
            raise SystemExit(f"File not found: {src}")
    elif args.run:
        run_dir = Path(args.run)
        if not run_dir.exists():
            raise SystemExit(f"Run dir not found: {run_dir}")
        src = _find_benchmark_file(run_dir)
        if src is None:
            raise SystemExit(f"No benchmark json found in {run_dir}")
    else:
        raise SystemExit("Provide --run or --file")

    data = _load_json(src)
    samples = _extract_samples(data)
    vals = []
    missing = 0
    for s in samples:
        if not isinstance(s, dict):
            continue
        task_id = s.get("task_id", s.get("task"))
        if task_id != 1:
            continue
        meta = s.get("meta") or {}
        conf = meta.get("confidence")
        if conf is None:
            missing += 1
            continue
        try:
            vals.append(float(conf))
        except Exception:
            missing += 1

    print(f"source: {src}")
    print(f"task1 samples: {len(vals)} (missing confidence: {missing})")
    if not vals:
        return

    vals_sorted = sorted(vals)
    print(f"min/median/mean/max: {min(vals):.3f} / {median(vals):.3f} / {mean(vals):.3f} / {max(vals):.3f}")
    for p in (10, 25, 50, 75, 90):
        v = _percentile(vals_sorted, p)
        print(f"p{p:02d}: {v:.3f}")

    print("\nHistogram:")
    for b0, b1, c in _histogram(vals, bins=max(1, int(args.bins))):
        print(f"{b0:.3f}–{b1:.3f}: {c}")


if __name__ == "__main__":
    main()
