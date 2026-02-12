#!/usr/bin/env python3
"""
Run comparable Gaze-VQA benchmark builds across VLM providers and summarize quality.

Example:
  python3 gaze_vqa/tools/compare_vlm_providers.py \
    --providers qwen,openai,gemini \
    --targets 10 10 10 10 \
    --allow_partial_tasks \
    --no_scan_dataset_stats
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


PREPS = {"in", "on", "at", "with", "of", "to", "from", "near", "by", "for"}
HUMAN = {"person", "man", "woman", "boy", "girl", "people"}
BAD_OBJECTS = {
    "floor", "wall", "ceiling", "room", "background", "outside", "nothing",
    "unknown", "scene", "image", "view", "camera", "person", "people",
    "man", "woman", "boy", "girl", "human", "humans",
}
BAD_GENERIC_WORDS = {"object", "thing", "item", "stuff", "something", "anything", "square", "circle", "shape"}
BAD_GENERIC_PHRASES = {"white object", "black object", "red object", "blue object", "small object", "large object"}


def parse_args():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Compare qwen/openai/gemini runs on shared benchmark settings.")
    p.add_argument("--providers", type=str, default="qwen,openai,gemini",
                   help="Comma-separated providers (subset of qwen,openai,gemini).")
    p.add_argument("--out_root", type=str, default=str(root), help="Same value as pipeline --out_root.")
    p.add_argument("--run_prefix", type=str, default="cmp10", help="Run name prefix for generated runs.")
    p.add_argument("--targets", type=int, nargs=4, default=[10, 10, 10, 10],
                   help="Targets for tasks 1..4 (default: 10 each).")
    p.add_argument("--max_frames", type=int, default=600, help="Pipeline --max_frames.")
    p.add_argument("--max_sequences", type=int, default=0, help="Pipeline --max_sequences.")
    g_partial = p.add_mutually_exclusive_group()
    g_partial.add_argument("--allow_partial_tasks", dest="allow_partial_tasks", action="store_true",
                           help="Use pipeline --allow_partial_tasks (default in this helper).")
    g_partial.add_argument("--no_allow_partial_tasks", dest="allow_partial_tasks", action="store_false",
                           help="Disable pipeline --allow_partial_tasks.")
    p.set_defaults(allow_partial_tasks=True)
    g_scan = p.add_mutually_exclusive_group()
    g_scan.add_argument("--no_scan_dataset_stats", dest="no_scan_dataset_stats", action="store_true",
                        help="Use pipeline --no_scan_dataset_stats (default in this helper).")
    g_scan.add_argument("--scan_dataset_stats", dest="no_scan_dataset_stats", action="store_false",
                        help="Disable --no_scan_dataset_stats and run full stats scan.")
    p.set_defaults(no_scan_dataset_stats=True)
    p.add_argument("--qwen_model", type=str, default="", help="Optional override for --qwen_model.")
    p.add_argument("--openai_model", type=str, default="", help="Optional override for OpenAI --vlm_model.")
    p.add_argument("--gemini_model", type=str, default="", help="Optional override for Gemini --vlm_model.")
    p.add_argument("--dry_run", action="store_true", default=False, help="Print commands without running them.")
    p.add_argument("--extra_args", nargs=argparse.REMAINDER, default=[],
                   help="Extra args forwarded to pipeline command after '--'.")
    args = p.parse_args()
    return args


def noun_phrase_ok(a: str):
    if not isinstance(a, str) or not a.strip():
        return False
    toks = a.strip().lower().split()
    if not toks:
        return False
    if toks[-1] in PREPS:
        return False
    if toks[0] in HUMAN:
        return False
    if len(toks) > 6:
        return False
    return True


def is_generic_label(label: str):
    if not isinstance(label, str) or not label.strip():
        return True
    low = label.strip().lower()
    toks = low.split()
    if low in BAD_OBJECTS or low in BAD_GENERIC_PHRASES:
        return True
    if toks and toks[0] in HUMAN:
        return True
    if any(t in BAD_GENERIC_WORDS for t in toks):
        return True
    return False


def _extract_task3_target(answer: str):
    m = re.match(r"\s*Gaze target:\s*([^.]+)\.", answer or "")
    return m.group(1).strip() if m else None


def _extract_task4_target(question: str):
    m = re.search(r"see the '([^']+)'", question or "")
    return m.group(1).strip() if m else None


def _extract_task4_target_from_answer(answer: str):
    m = re.search(r"In\s+Cam\w+,\s+the\s+([^\s].*?)\s+is\s+within", answer or "")
    if m:
        return m.group(1).strip()
    m = re.search(r"In\s+Cam\w+,\s+the\s+([^\s].*?)\s+is\s+outside", answer or "")
    return m.group(1).strip() if m else None


def sample_key(sample):
    scene = str(sample.get("scene", ""))
    ts = str(sample.get("timestamp", ""))
    tid = str(sample.get("task_id", ""))
    imgs = sample.get("input_images") or []
    names = []
    for ent in imgs:
        if isinstance(ent, dict) and ent.get("image"):
            names.append(Path(ent["image"]).name)
    names = sorted(set(names))
    return f"{tid}|{scene}|{ts}|{';'.join(names)}"


def summarize_benchmark(path: Path):
    data = json.loads(path.read_text())
    samples = data.get("samples", [])
    task_counts = Counter()
    missing_q = 0
    missing_a = 0
    empty_a = 0

    task1_labels = []
    task1_conf = []
    task1_generic = 0
    task1_noun_fail = 0

    t3_checked = 0
    t3_mismatch = 0
    t4_checked = 0
    t4_mismatch = 0

    for s in samples:
        tid = s.get("task_id")
        task_counts[int(tid)] += 1
        q = s.get("question")
        a = s.get("answer")
        if not q:
            missing_q += 1
        if a is None:
            missing_a += 1
        elif isinstance(a, str) and not a.strip():
            empty_a += 1

        if tid == 1:
            if isinstance(a, str):
                label = a.strip()
                task1_labels.append(label)
                if is_generic_label(label):
                    task1_generic += 1
                if not noun_phrase_ok(label):
                    task1_noun_fail += 1
            conf = (s.get("meta") or {}).get("confidence")
            if conf is not None:
                try:
                    task1_conf.append(float(conf))
                except Exception:
                    pass

        if tid == 3 and isinstance(q, str) and isinstance(a, str):
            q_target = re.search(r"gaze target is '([^']+)'", q.lower())
            q_t = q_target.group(1).strip() if q_target else None
            a_t = _extract_task3_target(a)
            if q_t and a_t:
                t3_checked += 1
                if q_t.lower() != a_t.lower():
                    t3_mismatch += 1

        if tid == 4 and isinstance(q, str) and isinstance(a, str):
            q_t = _extract_task4_target(q)
            a_t = _extract_task4_target_from_answer(a)
            if q_t and a_t:
                t4_checked += 1
                if q_t.lower() != a_t.lower():
                    t4_mismatch += 1

    t1_total = len(task1_labels)
    t1_unique = len(set([x.lower() for x in task1_labels]))
    t1_mean_conf = (sum(task1_conf) / len(task1_conf)) if task1_conf else None
    t1_generic_rate = (task1_generic / t1_total) if t1_total else None
    t1_noun_fail_rate = (task1_noun_fail / t1_total) if t1_total else None

    top_t1 = Counter([x.lower() for x in task1_labels]).most_common(10)

    return {
        "path": str(path),
        "meta": data.get("meta", {}),
        "counts": dict(task_counts),
        "total_samples": len(samples),
        "missing_question": missing_q,
        "missing_answer": missing_a,
        "empty_answer": empty_a,
        "task1": {
            "total": t1_total,
            "unique_labels": t1_unique,
            "mean_confidence": t1_mean_conf,
            "generic_label_rate": t1_generic_rate,
            "noun_phrase_fail_rate": t1_noun_fail_rate,
            "top10_labels": top_t1,
        },
        "task3": {
            "checked": t3_checked,
            "target_mismatch_rate": (t3_mismatch / t3_checked) if t3_checked else None,
            "target_mismatches": t3_mismatch,
        },
        "task4": {
            "checked": t4_checked,
            "target_mismatch_rate": (t4_mismatch / t4_checked) if t4_checked else None,
            "target_mismatches": t4_mismatch,
        }
    }


def run_provider(root: Path, out_root: Path, run_name: str, provider: str, args):
    cmd = [
        sys.executable,
        str(root / "generate_benchmark_delta.py"),
        "--out_root", str(out_root),
        "--run_name", run_name,
        "--targets",
        str(args.targets[0]), str(args.targets[1]), str(args.targets[2]), str(args.targets[3]),
        "--vlm_provider", provider,
        "--max_frames", str(args.max_frames),
    ]
    if args.max_sequences > 0:
        cmd += ["--max_sequences", str(args.max_sequences)]
    if args.allow_partial_tasks:
        cmd += ["--allow_partial_tasks"]
    if args.no_scan_dataset_stats:
        cmd += ["--no_scan_dataset_stats"]
    if provider == "qwen" and args.qwen_model:
        cmd += ["--qwen_model", args.qwen_model]
    if provider == "openai" and args.openai_model:
        cmd += ["--vlm_model", args.openai_model]
    if provider == "gemini" and args.gemini_model:
        cmd += ["--vlm_model", args.gemini_model]
    if args.extra_args:
        cmd += args.extra_args

    print(f"\n[{provider}] command:\n  {' '.join(cmd)}")
    if args.dry_run:
        return {"provider": provider, "run_name": run_name, "status": "dry_run", "command": cmd}

    proc = subprocess.run(cmd, cwd=str(root))
    status = "ok" if proc.returncode == 0 else f"exit_{proc.returncode}"
    out = {"provider": provider, "run_name": run_name, "status": status, "command": cmd}

    bench = out_root / "runs" / run_name / "benchmark_gazevqa.json"
    if proc.returncode == 0 and bench.exists():
        out["benchmark"] = str(bench)
        out["summary"] = summarize_benchmark(bench)
    else:
        out["benchmark"] = str(bench)
    return out


def _fmt_rate(v):
    if v is None:
        return "n/a"
    return f"{100.0 * v:.1f}%"


def print_summary(results):
    print("\n=== Provider Summary ===")
    for r in results:
        provider = r.get("provider")
        status = r.get("status")
        print(f"- {provider}: {status}")
        s = r.get("summary")
        if not s:
            continue
        t1 = s["task1"]
        print(
            f"  counts={s['counts']} | "
            f"task1 total={t1['total']} unique={t1['unique_labels']} "
            f"mean_conf={t1['mean_confidence'] if t1['mean_confidence'] is not None else 'n/a'} "
            f"generic={_fmt_rate(t1['generic_label_rate'])} "
            f"noun_fail={_fmt_rate(t1['noun_phrase_fail_rate'])}"
        )
        t3 = s["task3"]
        t4 = s["task4"]
        print(
            f"  task3 mismatch={t3['target_mismatches']}/{t3['checked']} ({_fmt_rate(t3['target_mismatch_rate'])}) | "
            f"task4 mismatch={t4['target_mismatches']}/{t4['checked']} ({_fmt_rate(t4['target_mismatch_rate'])})"
        )


def write_task1_label_matrix(report_dir: Path, report_prefix: str, results):
    provider_maps = {}
    all_keys = set()
    for r in results:
        p = r.get("provider")
        s = r.get("summary")
        if not p or not s:
            continue
        bench = Path(s["path"])
        data = json.loads(bench.read_text())
        labels = {}
        for sample in data.get("samples", []):
            if sample.get("task_id") != 1:
                continue
            key = sample_key(sample)
            labels[key] = sample.get("answer")
            all_keys.add(key)
        provider_maps[p] = labels

    if not provider_maps:
        return None

    providers = sorted(provider_maps.keys())
    csv_path = report_dir / f"{report_prefix}_task1_label_matrix.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_key"] + providers)
        for key in sorted(all_keys):
            row = [key] + [provider_maps[p].get(key, "") for p in providers]
            w.writerow(row)
    return csv_path


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    providers = [x.strip().lower() for x in args.providers.split(",") if x.strip()]
    allowed = {"qwen", "openai", "gemini"}
    bad = [x for x in providers if x not in allowed]
    if bad:
        raise SystemExit(f"Unsupported providers: {bad}. Allowed: {sorted(allowed)}")
    if not providers:
        raise SystemExit("No providers selected.")

    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    results = []
    for provider in providers:
        run_name = f"{args.run_prefix}_{provider}_{stamp}"
        results.append(run_provider(root, out_root, run_name, provider, args))

    print_summary(results)

    report_dir = out_root / "runs"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_prefix = f"{args.run_prefix}_provider_compare_{stamp}"
    report_path = report_dir / f"{report_prefix}.json"
    report_path.write_text(json.dumps({
        "timestamp": stamp,
        "providers": providers,
        "args": vars(args),
        "results": results,
    }, indent=2))
    print(f"\nWrote report: {report_path}")

    if not args.dry_run:
        matrix_path = write_task1_label_matrix(report_dir, report_prefix, results)
        if matrix_path is not None:
            print(f"Wrote Task1 label matrix: {matrix_path}")


if __name__ == "__main__":
    main()
