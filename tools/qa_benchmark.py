#!/usr/bin/env python3
import json
import re
import argparse
from collections import Counter, defaultdict
from pathlib import Path


PREPS = {"in", "on", "at", "with", "of", "to", "from", "near", "by", "for"}
HUMAN = {"person", "man", "woman", "boy", "girl", "people"}


def parse_args():
    p = argparse.ArgumentParser(description="QA checks for benchmark_gazevqa.json")
    p.add_argument("path", type=str, help="Path to benchmark_gazevqa.json")
    return p.parse_args()


def extract_task3_target(answer: str):
    # Expected: "Gaze target: <obj>. CamX: YES..."
    m = re.match(r"\\s*Gaze target:\\s*([^\\.]+)\\.", answer or "")
    return m.group(1).strip() if m else None


def extract_task4_target(question: str):
    # Expected: "In CamX, can the person wearing ... see the '<obj>' ...?"
    m = re.search(r"see the '([^']+)'", question or "")
    return m.group(1).strip() if m else None


def extract_task4_target_from_answer(answer: str):
    # Expected: "YES. In CamX, the <obj> is within ..."
    m = re.search(r"In\\s+Cam\\w+,\\s+the\\s+([^\\s].*?)\\s+is\\s+within", answer or "")
    if m:
        return m.group(1).strip()
    m = re.search(r"In\\s+Cam\\w+,\\s+the\\s+([^\\s].*?)\\s+is\\s+outside", answer or "")
    return m.group(1).strip() if m else None


def is_weird_person_question(q: str):
    if not q:
        return False
    ql = q.lower().strip()
    return ("person with wearing" in ql) or ql.endswith(" with") or ql.endswith(" wearing")


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


def main():
    args = parse_args()
    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

    data = json.loads(path.read_text())
    samples = data.get("samples", [])

    stats = {
        "total": len(samples),
        "task_counts": Counter(),
        "missing_question": 0,
        "missing_answer": 0,
        "empty_answer": 0,
        "short_answer": 0,
        "trailing_prep": 0,
        "human_label": 0,
        "weird_task1_question": 0,
        "noun_phrase_fail": 0,
        "task3_target_mismatch": 0,
        "task4_target_mismatch": 0,
    }

    label_counts = Counter()
    inconsistencies = defaultdict(list)

    for s in samples:
        tid = s.get("task_id")
        stats["task_counts"][tid] += 1

        q = s.get("question")
        a = s.get("answer")

        if not q:
            stats["missing_question"] += 1
        if a is None:
            stats["missing_answer"] += 1
        elif isinstance(a, str) and not a.strip():
            stats["empty_answer"] += 1
        elif isinstance(a, str) and len(a.strip()) <= 3:
            stats["short_answer"] += 1

        if isinstance(a, str):
            toks = a.strip().lower().split()
            if toks and toks[-1] in PREPS:
                stats["trailing_prep"] += 1
                inconsistencies["trailing_prep"].append((tid, a, q))
            if toks and toks[0] in HUMAN:
                stats["human_label"] += 1
                inconsistencies["human_label"].append((tid, a, q))
            if tid == 1:
                label_counts[a.strip().lower()] += 1
                if not noun_phrase_ok(a):
                    stats["noun_phrase_fail"] += 1
                    inconsistencies["noun_phrase_fail"].append((tid, a, q))

        if tid == 1 and is_weird_person_question(q):
            stats["weird_task1_question"] += 1
            inconsistencies["weird_question"].append((tid, a, q))

        if tid == 3:
            q_target = re.search(r"gaze target is '([^']+)'", (q or "").lower())
            q_t = q_target.group(1).strip() if q_target else None
            a_t = extract_task3_target(a) if isinstance(a, str) else None
            if q_t and a_t and q_t.lower() != a_t.lower():
                stats["task3_target_mismatch"] += 1
                inconsistencies["task3_target_mismatch"].append((tid, q_t, a_t, q))

        if tid == 4:
            q_t = extract_task4_target(q)
            a_t = extract_task4_target_from_answer(a) if isinstance(a, str) else None
            if q_t and a_t and q_t.lower() != a_t.lower():
                stats["task4_target_mismatch"] += 1
                inconsistencies["task4_target_mismatch"].append((tid, q_t, a_t, q))

    # Print summary
    print(f"Total samples: {stats['total']}")
    print("Task counts:", dict(stats["task_counts"]))
    print("Missing question:", stats["missing_question"])
    print("Missing answer:", stats["missing_answer"])
    print("Empty answer:", stats["empty_answer"])
    print("Short answer:", stats["short_answer"])
    print("Trailing preposition answers:", stats["trailing_prep"])
    print("Human-label answers:", stats["human_label"])
    print("Weird Task1 question:", stats["weird_task1_question"])
    print("Task1 noun-phrase fails:", stats["noun_phrase_fail"])
    print("Task3 target mismatches:", stats["task3_target_mismatch"])
    print("Task4 target mismatches:", stats["task4_target_mismatch"])

    if label_counts:
        print("\nTask1 label diversity:")
        print(" unique:", len(label_counts))
        print(" top-10:", label_counts.most_common(10))

    # Show a few examples of each inconsistency type
    for k, items in inconsistencies.items():
        if not items:
            continue
        print(f"\nExamples: {k}")
        for row in items[:5]:
            print(" ", row)


if __name__ == "__main__":
    main()
