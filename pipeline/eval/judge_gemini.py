from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from .schemas import append_jsonl, iter_jsonl, read_json, utc_now_iso, write_json


def _build_prompt(sample: Dict[str, Any]) -> str:
    task_type = sample.get("task_type") or "N/A"
    question = sample.get("question") or "N/A"
    gt = sample.get("groundtruth_answer") or ""
    pred = sample.get("inference_answer") or ""
    return (
        "You are a strict evaluator for semantic equivalence of short answers.\n"
        "Decide if the model answer means the same as the groundtruth, allowing paraphrase, synonyms, and minor wording differences.\n"
        "Mark as wrong if the model misses key objects/relations, contradicts, adds incorrect facts, answers 'I am not sure', or is off-topic.\n"
        "Respond with only one token: 'correct' or 'wrong'. No explanation.\n\n"
        f"Task type: {task_type}\n"
        f"Question: {question}\n"
        f"Groundtruth answer: {gt}\n"
        f"Model answer: {pred}\n"
        "Your verdict:"
    )


def _parse_verdict(text: str) -> str:
    normalized = (text or "").strip().lower()
    if "correct" in normalized and "wrong" not in normalized:
        return "correct"
    return "wrong"


def _judge_with_gemini(client, model: str, prompt: str, retry: int = 3, backoff: float = 2.0) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, int(retry) + 1):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return _parse_verdict(str(getattr(response, "text", "") or ""))
        except Exception as exc:
            last_exc = exc
            if attempt == int(retry):
                break
            time.sleep(float(backoff) * attempt)
    if last_exc:
        raise last_exc
    return "wrong"


def judge_json(
    input_path: Path,
    output_path: Path,
    model: str = "gemini-3.1-pro-preview",
    api_key: str = "",
    retry: int = 3,
    retry_backoff: float = 2.0,
    request_interval: float = 0.0,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    try:
        from google import genai
    except Exception as exc:
        raise RuntimeError("google-genai package is required for judge step.") from exc

    key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY for Gemini judge.")
    client = genai.Client(api_key=key)

    data = read_json(input_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON at {input_path}")

    existing_by_uid: Dict[str, Dict[str, Any]] = {}
    if skip_existing and output_path.exists():
        for row in iter_jsonl(output_path):
            uid = str(row.get("sample_uid") or "")
            if uid:
                existing_by_uid[uid] = row

    output_path.parent.mkdir(parents=True, exist_ok=True)
    judged = 0
    skipped = 0
    errors = 0

    for item in data:
        uid = str(item.get("sample_uid") or "")
        if not uid:
            continue
        if skip_existing and uid in existing_by_uid:
            skipped += 1
            continue
        pred = str(item.get("inference_answer") or "").strip()
        t0 = time.time()
        if not pred:
            verdict = "wrong"
            err = "missing_prediction"
        else:
            try:
                verdict = _judge_with_gemini(
                    client=client,
                    model=model,
                    prompt=_build_prompt(item),
                    retry=retry,
                    backoff=retry_backoff,
                )
                err = ""
            except Exception as exc:
                verdict = "wrong"
                err = f"{type(exc).__name__}: {exc}"
                errors += 1

        row = {
            "sample_uid": uid,
            "model_key": str(item.get("model_key") or ""),
            "gemini_judge": verdict,
            "judge_model": model,
            "judge_latency_s": round(time.time() - t0, 4),
            "error": err,
            "timestamp": utc_now_iso(),
        }
        append_jsonl(output_path, row)
        judged += 1
        if request_interval > 0:
            time.sleep(request_interval)

    meta = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "judge_model": model,
        "judged_new": judged,
        "skipped_existing": skipped,
        "errors": errors,
        "updated_at": utc_now_iso(),
    }
    write_json(output_path.with_suffix(".meta.json"), meta)
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gemini judge for model prediction JSON.")
    p.add_argument("--input_path", type=Path, required=True)
    p.add_argument("--output_path", type=Path, required=True)
    p.add_argument("--model", type=str, default="gemini-3.1-pro-preview")
    p.add_argument("--api_key", type=str, default="")
    p.add_argument("--retry", type=int, default=3)
    p.add_argument("--retry_backoff", type=float, default=2.0)
    p.add_argument("--request_interval", type=float, default=0.0)
    p.add_argument("--no_skip_existing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    meta = judge_json(
        input_path=args.input_path,
        output_path=args.output_path,
        model=args.model,
        api_key=args.api_key,
        retry=args.retry,
        retry_backoff=args.retry_backoff,
        request_interval=args.request_interval,
        skip_existing=not args.no_skip_existing,
    )
    print("[DONE] Gemini judge finished.")
    for k, v in meta.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
