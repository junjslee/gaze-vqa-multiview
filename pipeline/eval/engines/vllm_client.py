from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import requests


class VLLMClient:
    """Client for vLLM OpenAI-compatible endpoint."""

    def __init__(self, base_url: str, api_key: str = "", timeout_s: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.timeout_s = float(timeout_s)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def healthcheck(self) -> bool:
        for endpoint in ("/health", "/v1/models"):
            try:
                resp = requests.get(f"{self.base_url}{endpoint}", headers=self._headers(), timeout=10)
                if 200 <= resp.status_code < 300:
                    return True
            except Exception:
                continue
        return False

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 256,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }
        if extra:
            payload.update(extra)

        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()

        text = ""
        choices = data.get("choices") or []
        if choices:
            msg = (choices[0] or {}).get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                chunks = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        chunks.append(str(item["text"]))
                text = "\n".join(chunks).strip()

        usage = data.get("usage") or {}
        tokens = {
            "tokens_in": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            "tokens_out": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        }
        return text, tokens, data
