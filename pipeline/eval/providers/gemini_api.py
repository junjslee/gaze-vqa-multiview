from __future__ import annotations

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


def infer_gemini_multimodal(
    model: str,
    prompt: str,
    image_paths: List[str],
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_output_tokens: int = 256,
    retry: int = 3,
    retry_backoff: float = 2.0,
    system_prompt: str = "",
) -> Tuple[str, Dict[str, int], str]:
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        raise RuntimeError("google-genai package is required for Gemini API inference.") from exc

    api_key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY for Gemini inference.")
    client = genai.Client(api_key=api_key)

    parts: List[types.Part] = [types.Part.from_text(text=prompt)]
    for p in image_paths:
        pp = Path(p)
        if not pp.exists():
            continue
        with pp.open("rb") as f:
            parts.append(types.Part.from_bytes(data=f.read(), mime_type=_guess_mime(pp)))

    cfg_kwargs = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_output_tokens": int(max_output_tokens),
        "system_instruction": system_prompt or None,
    }
    thinking_budget_raw = str(os.getenv("GEMINI_THINKING_BUDGET", "0")).strip()
    if thinking_budget_raw:
        try:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=int(thinking_budget_raw))
        except Exception:
            # Keep compatibility with older SDK/model combinations.
            pass
    try:
        cfg = types.GenerateContentConfig(**cfg_kwargs)
    except TypeError:
        cfg_kwargs.pop("thinking_config", None)
        cfg = types.GenerateContentConfig(**cfg_kwargs)

    last_err = ""
    for attempt in range(1, int(retry) + 1):
        try:
            response = client.models.generate_content(model=model, contents=parts, config=cfg)
            text = str(response.text or "").strip()
            usage = getattr(response, "usage_metadata", None)
            tokens = {
                "tokens_in": int(getattr(usage, "prompt_token_count", 0) or 0),
                "tokens_out": int(getattr(usage, "candidates_token_count", 0) or 0),
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
