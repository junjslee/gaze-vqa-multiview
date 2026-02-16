from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    return "application/octet-stream"


def _encode_image(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def infer_openai_multimodal(
    model: str,
    prompt: str,
    image_paths: List[str],
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_output_tokens: int = 256,
    retry: int = 3,
    retry_backoff: float = 2.0,
    timeout_s: float = 180.0,
    system_prompt: str = "",
) -> Tuple[str, Dict[str, int], str]:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is required for OpenAI API inference.") from exc

    api_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY for OpenAI API inference.")
    client = OpenAI(api_key=api_key, timeout=float(timeout_s))

    user_content = [{"type": "input_text", "text": prompt}]
    for p in image_paths:
        pp = Path(p)
        if not pp.exists():
            continue
        user_content.append(
            {
                "type": "input_image",
                "image_url": f"data:{_guess_mime(pp)};base64,{_encode_image(pp)}",
            }
        )

    last_err = ""
    for attempt in range(1, int(retry) + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system_prompt}] if system_prompt else [],
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                temperature=float(temperature),
                top_p=float(top_p),
                max_output_tokens=int(max_output_tokens),
            )
            text = str(response.output_text or "").strip()
            usage = response.usage or {}
            tokens = {
                "tokens_in": int(getattr(usage, "input_tokens", 0) or 0),
                "tokens_out": int(getattr(usage, "output_tokens", 0) or 0),
            }
            if text:
                return text, tokens, ""
            return "", tokens, "empty_response"
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt == int(retry):
                break
            time.sleep(float(retry_backoff) * attempt)
    return "", {"tokens_in": 0, "tokens_out": 0}, last_err

