import base64
import gc
import io
import json
import math
import os
import re
import time
from pathlib import Path

import requests
from PIL import Image

from . import state as st

_qwen_model = None
_qwen_proc = None
_vlm_calls = 0
_vlm_usage_totals = None
_vlm_usage_per_frame = {}
_vlm_pricing = None
_vlm_usage_provider = None
_vlm_usage_model = None


def install_vlm_deps():
    from .io_utils import _run
    _run(["bash", "-lc", "python3 -m pip -q install transformers accelerate qwen-vl-utils"], check=True)


def load_qwen():
    global _qwen_model, _qwen_proc
    if _qwen_model is not None:
        return _qwen_model, _qwen_proc

    if st.ARGS.skip_vlm:
        raise RuntimeError("--skip_vlm enabled but Qwen load requested.")

    # Prefer /work cache on Delta to avoid HOME quota
    work_cache = "/work/nvme/bfga/jlee65/.cache/hf"
    if os.path.isdir("/work") and not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = work_cache
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(work_cache, "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(work_cache, "transformers"))
        os.environ.setdefault("XDG_CACHE_HOME", "/work/nvme/bfga/jlee65/.cache")

    install_vlm_deps()
    import torch
    from transformers import AutoProcessor
    from transformers import Qwen2_5_VLForConditionalGeneration

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    st.logger.info(f"Loading VLM: {st.QWEN_MODEL_ID}")
    try:
        _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            st.QWEN_MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
    except TypeError:
        # Older transformers without attn_implementation
        _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            st.QWEN_MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
    _qwen_proc = AutoProcessor.from_pretrained(
        st.QWEN_MODEL_ID,
        trust_remote_code=True,
        min_pixels=st.MIN_PIXELS,
        max_pixels=st.MAX_PIXELS
    )
    st.logger.info("✅ Qwen loaded.")
    return _qwen_model, _qwen_proc


def warmup_vlm():
    if st.ARGS.skip_vlm:
        return
    if st.VLM_PROVIDER == "qwen":
        load_qwen()
        return
    if st.VLM_PROVIDER == "openai":
        _require_openai_api_key()
    elif st.VLM_PROVIDER == "gemini":
        _require_gemini_api_key()
    else:
        raise ValueError(f"Unsupported VLM provider: {st.VLM_PROVIDER}")
    st.logger.info(f"Using VLM provider={st.VLM_PROVIDER} model={st.VLM_MODEL_ID}")


def _base_usage_bucket():
    return {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "billed_input_tokens": 0,
        "total_tokens": 0,
        "cost_usd_estimate": 0.0,
    }


def _normalize_model_name(model_name):
    return str(model_name or "").strip().lower()


def _openai_pricing_for_model(model_name):
    m = _normalize_model_name(model_name)
    # Model-specific defaults where known. Fallback uses state-configured rates.
    if m.startswith("gpt-4o-mini"):
        return {
            "input_per_1m": 0.15,
            "cached_input_per_1m": 0.075,
            "output_per_1m": 0.60,
            "source": "builtin:gpt-4o-mini",
        }
    return {
        "input_per_1m": float(st.OPENAI_PRICE_INPUT_PER_1M),
        "cached_input_per_1m": float(st.OPENAI_PRICE_CACHED_INPUT_PER_1M),
        "output_per_1m": float(st.OPENAI_PRICE_OUTPUT_PER_1M),
        "source": "state:OPENAI_*_PER_1M",
    }


def _gemini_pricing_for_model(model_name):
    _ = _normalize_model_name(model_name)
    return {
        "input_per_1m": float(st.GEMINI_PRICE_INPUT_PER_1M),
        "output_per_1m": float(st.GEMINI_PRICE_OUTPUT_PER_1M),
        "source": "state:GEMINI_*_PER_1M",
    }


def _ensure_usage_state(provider_name, model_name):
    global _vlm_usage_totals, _vlm_pricing, _vlm_usage_provider, _vlm_usage_model
    if _vlm_usage_totals is None:
        _vlm_usage_totals = _base_usage_bucket()
    if _vlm_usage_provider is None:
        _vlm_usage_provider = str(provider_name)
    if _vlm_usage_model is None:
        _vlm_usage_model = str(model_name)
    if _vlm_pricing is None:
        if provider_name == "openai":
            _vlm_pricing = _openai_pricing_for_model(model_name)
        elif provider_name == "gemini":
            _vlm_pricing = _gemini_pricing_for_model(model_name)
        else:
            _vlm_pricing = {"source": "none"}


def _record_usage(provider_name, model_name, input_tokens, output_tokens, cached_input_tokens=0, total_tokens=None):
    global _vlm_usage_per_frame, _vlm_pricing
    _ensure_usage_state(provider_name, model_name)

    inp = int(max(0, input_tokens or 0))
    out = int(max(0, output_tokens or 0))
    cached = int(max(0, cached_input_tokens or 0))
    billed_inp = int(max(0, inp - cached))
    total = int(total_tokens) if total_tokens is not None else int(inp + out)

    if provider_name == "openai":
        p = _openai_pricing_for_model(model_name)
        cost = (
            billed_inp * float(p["input_per_1m"]) +
            cached * float(p["cached_input_per_1m"]) +
            out * float(p["output_per_1m"])
        ) / 1_000_000.0
    elif provider_name == "gemini":
        p = _gemini_pricing_for_model(model_name)
        cost = (
            inp * float(p["input_per_1m"]) +
            out * float(p["output_per_1m"])
        ) / 1_000_000.0
    else:
        p = {"source": "none"}
        cost = 0.0

    _vlm_usage_totals["calls"] += 1
    _vlm_usage_totals["input_tokens"] += inp
    _vlm_usage_totals["output_tokens"] += out
    _vlm_usage_totals["cached_input_tokens"] += cached
    _vlm_usage_totals["billed_input_tokens"] += billed_inp
    _vlm_usage_totals["total_tokens"] += total
    _vlm_usage_totals["cost_usd_estimate"] += float(cost)

    frame_key = st.CURRENT_FRAME_KEY or "NO_FRAME_CONTEXT"
    fb = _vlm_usage_per_frame.get(frame_key)
    if fb is None:
        fb = _base_usage_bucket()
        _vlm_usage_per_frame[frame_key] = fb
    fb["calls"] += 1
    fb["input_tokens"] += inp
    fb["output_tokens"] += out
    fb["cached_input_tokens"] += cached
    fb["billed_input_tokens"] += billed_inp
    fb["total_tokens"] += total
    fb["cost_usd_estimate"] += float(cost)

    # Keep latest pricing snapshot used in this call.
    _vlm_pricing = p


def _extract_openai_usage(data):
    usage = data.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    # Compatibility with Responses-like usage fields.
    if prompt_tokens is None:
        prompt_tokens = usage.get("input_tokens")
    if completion_tokens is None:
        completion_tokens = usage.get("output_tokens")
    if total_tokens is None:
        total_tokens = usage.get("total_tokens")

    details = usage.get("prompt_tokens_details") or {}
    cached_tokens = details.get("cached_tokens")
    if cached_tokens is None:
        cached_tokens = usage.get("cached_input_tokens")

    try:
        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)
        total_tokens = int(total_tokens or (prompt_tokens + completion_tokens))
        cached_tokens = int(cached_tokens or 0)
    except Exception:
        return None
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "cached_input_tokens": cached_tokens,
        "total_tokens": total_tokens,
    }


def _extract_gemini_usage(data):
    usage = data.get("usageMetadata") or {}
    prompt_tokens = usage.get("promptTokenCount")
    completion_tokens = usage.get("candidatesTokenCount")
    total_tokens = usage.get("totalTokenCount")
    cached_tokens = usage.get("cachedContentTokenCount")
    try:
        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)
        total_tokens = int(total_tokens or (prompt_tokens + completion_tokens))
        cached_tokens = int(cached_tokens or 0)
        # Some Gemini responses omit candidatesTokenCount while still charging
        # non-prompt tokens (e.g., internal thinking). Use total-prompt fallback.
        if completion_tokens <= 0 and total_tokens > prompt_tokens:
            completion_tokens = int(total_tokens - prompt_tokens)
    except Exception:
        return None
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "cached_input_tokens": cached_tokens,
        "total_tokens": total_tokens,
    }


def get_vlm_usage_report():
    if _vlm_usage_totals is None:
        return None
    per_frame = []
    for fk in sorted(_vlm_usage_per_frame.keys()):
        rec = dict(_vlm_usage_per_frame[fk])
        rec["cost_usd_estimate"] = round(float(rec["cost_usd_estimate"]), 10)
        rec["frame_key"] = fk
        per_frame.append(rec)
    totals = dict(_vlm_usage_totals)
    totals["cost_usd_estimate"] = round(float(totals["cost_usd_estimate"]), 10)
    return {
        "provider": _vlm_usage_provider or st.VLM_PROVIDER,
        "model": _vlm_usage_model or st.VLM_MODEL_ID,
        "pricing_usd_per_1m": dict(_vlm_pricing or {}),
        "totals": totals,
        "per_frame": per_frame,
    }


def write_vlm_usage_report(path=None):
    report = get_vlm_usage_report()
    if report is None:
        return None
    p = Path(path or st.VLM_USAGE_JSON)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2))
    return str(p)


def _get_model_input_device(model):
    import torch
    dev = getattr(model, "device", None)
    if dev is not None and str(dev) != "meta":
        return dev
    for p in model.parameters():
        if hasattr(p, "device") and p.device is not None and p.device.type != "meta":
            return p.device
    return torch.device("cpu")


def _ensure_min_image_side(pil_img: Image.Image, min_side: int = 28) -> Image.Image:
    w, h = pil_img.size
    if w >= min_side and h >= min_side:
        return pil_img
    scale = max(min_side / max(1, w), min_side / max(1, h))
    new_w = max(min_side, int(math.ceil(w * scale)))
    new_h = max(min_side, int(math.ceil(h * scale)))
    return pil_img.resize((new_w, new_h), resample=Image.BICUBIC)


def _normalize_pil_images(images):
    if images is None:
        images_list = []
    elif isinstance(images, list):
        images_list = images
    else:
        images_list = [images]

    pil_images = []
    for im in images_list:
        if im is None:
            continue
        if isinstance(im, str):
            with Image.open(im) as src:
                pil = src.convert("RGB")
        elif isinstance(im, Image.Image):
            pil = im.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type in vlm_generate(): {type(im)}")
        pil_images.append(_ensure_min_image_side(pil, min_side=28))
    return pil_images


def _pil_to_jpeg_b64(pil_img: Image.Image):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _require_openai_api_key():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing. Put it in .env or your shell env.")
    return key


def _require_gemini_api_key():
    key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is missing. Put it in .env or your shell env.")
    return key


def _retry_max_attempts():
    try:
        v = int(os.environ.get("VLM_API_MAX_ATTEMPTS", "3"))
    except Exception:
        v = 3
    return max(1, min(v, 8))


def _retry_backoff_seconds():
    try:
        v = float(os.environ.get("VLM_API_RETRY_BACKOFF_S", "1.5"))
    except Exception:
        v = 1.5
    return max(0.0, min(v, 60.0))


def _is_transient_http_status(code):
    return int(code) in {408, 409, 429, 500, 502, 503, 504}


def _is_transient_api_error_message(msg):
    s = str(msg or "").upper()
    if "API REQUEST ERROR" in s:
        return True
    for c in ("408", "409", "429", "500", "502", "503", "504"):
        if f"API ERROR {c}" in s:
            return True
    return False


def _post_json(url, headers, payload, provider_name):
    max_attempts = _retry_max_attempts()
    backoff_s = _retry_backoff_seconds()

    for attempt_i in range(1, max_attempts + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=st.VLM_TIMEOUT_S)
        except requests.RequestException as e:
            if attempt_i < max_attempts:
                delay = backoff_s * (2 ** (attempt_i - 1))
                st.logger.warning(
                    f"{provider_name} API request error (attempt {attempt_i}/{max_attempts}): {e}. "
                    f"Retrying in {delay:.1f}s"
                )
                if delay > 0:
                    time.sleep(delay)
                continue
            raise RuntimeError(f"{provider_name} API request error: {e}") from e

        if resp.status_code >= 400:
            body = (resp.text or "").strip().replace("\n", " ")
            body = body[:1200]
            if _is_transient_http_status(resp.status_code) and attempt_i < max_attempts:
                delay = backoff_s * (2 ** (attempt_i - 1))
                st.logger.warning(
                    f"{provider_name} API transient error {resp.status_code} "
                    f"(attempt {attempt_i}/{max_attempts}). Retrying in {delay:.1f}s"
                )
                if delay > 0:
                    time.sleep(delay)
                continue
            raise RuntimeError(f"{provider_name} API error {resp.status_code}: {body}")

        try:
            return resp.json()
        except Exception as e:
            body = (resp.text or "").strip().replace("\n", " ")
            body = body[:1200]
            if attempt_i < max_attempts:
                delay = backoff_s * (2 ** (attempt_i - 1))
                st.logger.warning(
                    f"{provider_name} API non-JSON payload (attempt {attempt_i}/{max_attempts}). "
                    f"Retrying in {delay:.1f}s"
                )
                if delay > 0:
                    time.sleep(delay)
                continue
            raise RuntimeError(f"{provider_name} API returned non-JSON payload: {body}") from e

    # Unreachable by construction, but keep explicit for static safety.
    raise RuntimeError(f"{provider_name} API request failed after retries.")


def _qwen_generate(images, prompt, max_new_tokens):
    import torch

    model, proc = load_qwen()
    pil_images = _normalize_pil_images(images)

    if len(pil_images) == 0:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], padding=True, return_tensors="pt")
    else:
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": im} for im in pil_images] + [{"type": "text", "text": prompt}]
        }]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=pil_images, padding=True, return_tensors="pt")

    device = _get_model_input_device(model)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    pred = proc.batch_decode(out, skip_special_tokens=True)[0]

    # Qwen output cleanup
    if "assistant\n" in pred:
        pred = pred.split("assistant\n")[-1]
    pred = pred.strip()

    del inputs, out
    gc.collect()
    if torch.cuda.is_available() and (_vlm_calls % st.EMPTY_CACHE_EVERY == 0):
        torch.cuda.empty_cache()
    return pred


def _extract_openai_text(data):
    choices = data.get("choices") or []
    if choices:
        msg = (choices[0] or {}).get("message") or {}
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            texts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                txt = part.get("text")
                if txt:
                    texts.append(str(txt))
            if texts:
                return "\n".join(texts).strip()
        txt = (choices[0] or {}).get("text")
        if isinstance(txt, str) and txt.strip():
            return txt.strip()

    # Fallback if a Responses-style payload is returned by a compatible endpoint.
    output = data.get("output") or []
    texts = []
    for item in output:
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            txt = part.get("text")
            if txt:
                texts.append(str(txt))
    if texts:
        return "\n".join(texts).strip()
    return ""


def _openai_generate(images, prompt, max_new_tokens, model_id=None, generation_cfg=None):
    api_key = _require_openai_api_key()
    pil_images = _normalize_pil_images(images)
    cfg = generation_cfg or {}
    model_name = str(model_id or st.VLM_MODEL_ID).strip()

    content = [{"type": "text", "text": prompt}]
    for pil in pil_images:
        b64 = _pil_to_jpeg_b64(pil)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": int(max_new_tokens),
        "temperature": float(cfg.get("temperature", 0)),
    }
    if cfg.get("top_p") is not None:
        payload["top_p"] = float(cfg["top_p"])
    url = f"{st.OPENAI_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = _post_json(url, headers, payload, provider_name="OpenAI")
    usage = _extract_openai_usage(data)
    if usage is not None:
        _record_usage(
            "openai",
            model_name,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cached_input_tokens=usage["cached_input_tokens"],
            total_tokens=usage["total_tokens"],
        )
    text = _extract_openai_text(data)
    if text:
        return text
    raise RuntimeError(f"OpenAI returned no text output: {str(data)[:800]}")


def _extract_gemini_text(data):
    candidates = data.get("candidates") or []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        content = cand.get("content") or {}
        parts = content.get("parts") or []
        texts = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            txt = part.get("text")
            if txt:
                texts.append(str(txt))
        if texts:
            return "\n".join(texts).strip()
    return ""


def _extract_gemini_finish_reason(data):
    candidates = data.get("candidates") or []
    for cand in candidates:
        if isinstance(cand, dict):
            fr = cand.get("finishReason")
            if fr:
                return str(fr)
    return ""


def _gemini_generate(images, prompt, max_new_tokens, model_id=None, generation_cfg=None):
    api_key = _require_gemini_api_key()
    pil_images = _normalize_pil_images(images)
    cfg = generation_cfg or {}
    model_name = str(model_id or st.VLM_MODEL_ID).strip()

    parts = [{"text": prompt}]
    for pil in pil_images:
        b64 = _pil_to_jpeg_b64(pil)
        parts.append({
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": b64,
            }
        })

    def _build_payload(tok_budget):
        temperature = float(cfg.get("temperature", 0))
        top_p = cfg.get("top_p")
        top_k = cfg.get("top_k")
        candidate_count = cfg.get("candidate_count")
        thinking_budget = cfg.get("thinking_budget", st.GEMINI_THINKING_BUDGET)
        payload_local = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "maxOutputTokens": int(tok_budget),
                "temperature": temperature,
            },
        }
        if top_p is not None:
            payload_local["generationConfig"]["topP"] = float(top_p)
        if top_k is not None:
            payload_local["generationConfig"]["topK"] = int(top_k)
        if candidate_count is not None:
            payload_local["generationConfig"]["candidateCount"] = int(candidate_count)
        # Gemini 2.5 can spend budget on "thinking" and return empty parts when capped.
        # Defaulting this to 0 (configurable) improves short-label stability.
        payload_local["generationConfig"]["thinkingConfig"] = {
            "thinkingBudget": int(thinking_budget),
        }
        return payload_local

    url = f"{st.GEMINI_API_BASE.rstrip('/')}/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    token_budgets = [int(max_new_tokens)]
    if int(max_new_tokens) < 128:
        token_budgets.append(max(128, int(max_new_tokens) * 2))

    last_data = None
    for attempt_i, tok_budget in enumerate(token_budgets, start=1):
        payload = _build_payload(tok_budget)
        try:
            data = _post_json(url, headers, payload, provider_name="Gemini")
        except RuntimeError as e:
            if _is_transient_api_error_message(e):
                st.logger.warning(
                    "Gemini transient API failure after bounded retries; returning empty output for this call."
                )
                return ""
            raise
        last_data = data

        usage = _extract_gemini_usage(data)
        if usage is not None:
            _record_usage(
                "gemini",
                model_name,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                cached_input_tokens=usage["cached_input_tokens"],
                total_tokens=usage["total_tokens"],
            )

        text = _extract_gemini_text(data)
        if text:
            return text

        finish_reason = _extract_gemini_finish_reason(data).upper()
        if finish_reason == "MAX_TOKENS" and attempt_i < len(token_budgets):
            st.logger.warning(
                f"Gemini produced no text with finishReason=MAX_TOKENS; retrying "
                f"with maxOutputTokens={token_budgets[attempt_i]}"
            )
            continue
        break

    fb = (last_data or {}).get("promptFeedback")
    if fb:
        st.logger.warning(f"Gemini returned no text output. promptFeedback={fb}")
        return ""

    finish_reason = _extract_gemini_finish_reason(last_data or {}).upper()
    if finish_reason:
        st.logger.warning(
            f"Gemini returned no text output (finishReason={finish_reason}). "
            "Returning empty string."
        )
        return ""
    st.logger.warning(f"Gemini returned no text output: {str(last_data)[:800]}")
    return ""


def _default_model_for_provider(provider_name):
    p = str(provider_name or "").strip().lower()
    if p == "qwen":
        return str(st.QWEN_MODEL_ID)
    if p == "openai":
        return str(os.environ.get("OPENAI_MODEL", "gpt-4o"))
    if p == "gemini":
        return str(os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    return str(st.VLM_MODEL_ID)


def vlm_generate(images, prompt, max_new_tokens=128, provider=None, model_id=None, generation_cfg=None):
    global _vlm_calls
    _vlm_calls += 1

    if st.ARGS.skip_vlm:
        return "SKIP_VLM"

    provider_name = str(provider or st.VLM_PROVIDER).strip().lower()
    if provider_name not in {"qwen", "openai", "gemini"}:
        raise ValueError(f"Unsupported VLM provider: {provider_name}")
    if model_id is not None and str(model_id).strip():
        resolved_model = str(model_id).strip()
    elif provider_name == st.VLM_PROVIDER:
        resolved_model = str(st.VLM_MODEL_ID)
    else:
        resolved_model = _default_model_for_provider(provider_name)

    if st.ARGS.log_prompts:
        img_count = 0
        if images is None:
            img_count = 0
        elif isinstance(images, list):
            img_count = len([x for x in images if x is not None])
        else:
            img_count = 1
        st.logger.info(
            f"[VLM] provider={provider_name} model={resolved_model} "
            f"images={img_count} max_new_tokens={max_new_tokens}"
        )
        st.logger.info(f"[VLM] prompt:\n{prompt}")

    if provider_name == "qwen":
        if resolved_model and resolved_model != str(st.QWEN_MODEL_ID):
            st.logger.warning(
                f"[VLM] qwen override model '{resolved_model}' ignored; using loaded model '{st.QWEN_MODEL_ID}'."
            )
        pred = _qwen_generate(images, prompt, max_new_tokens=max_new_tokens)
    elif provider_name == "openai":
        pred = _openai_generate(
            images,
            prompt,
            max_new_tokens=max_new_tokens,
            model_id=resolved_model,
            generation_cfg=generation_cfg,
        )
    elif provider_name == "gemini":
        pred = _gemini_generate(
            images,
            prompt,
            max_new_tokens=max_new_tokens,
            model_id=resolved_model,
            generation_cfg=generation_cfg,
        )
    else:
        raise ValueError(f"Unsupported VLM provider: {provider_name}")

    if st.ARGS.log_prompts:
        st.logger.info(f"[VLM] output:\n{pred}")

    return pred


# =============================================================================
# Text utilities
# =============================================================================

def _first_two_sentences(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    parts = re.split(r'(?<=[.!?])\s+', text)
    parts = [p.strip() for p in parts if p.strip()]
    return " ".join(parts[:2]).strip()


def safe_reasoning(raw_reason: str, fallback: str) -> str:
    rr = _first_two_sentences(raw_reason)
    if rr:
        return rr
    return _first_two_sentences(fallback) or fallback


def clean_label(s, max_words=8):
    s = str(s).strip()
    s = s.split("\n")[-1].strip()
    s = re.sub(r"[\(\)\[\]\{\}]", " ", s)
    s = re.sub(r"[^A-Za-z0-9\s\-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return None
    toks = s.split(" ")
    s = " ".join(toks[:max_words]).strip()
    low = s.lower()
    if any(x in low for x in ("not enough information", "cannot determine", "cannot tell", "unclear", "not enough", "cannot")):
        return None
    if low in st.BAD_OBJECTS:
        return None
    if low in st.BAD_GENERIC_PHRASES:
        return None
    if any(w in st.BAD_GENERIC_WORDS for w in low.split()):
        return None
    return s


def strict_noun_phrase(s: str, max_words=4):
    if not s:
        return None
    s = clean_label(s, max_words=max_words)
    if not s:
        return None
    low = s.lower()

    bad_tokens = {
        "is","are","was","were","be","being","been",
    }
    toks = low.split()
    if any(t in bad_tokens for t in toks):
        if toks and toks[0] == "the" and len(toks) <= max_words:
            s2 = " ".join(s.split()[1:])
            s2 = clean_label(s2, max_words=max_words)
            if s2 and not any(t in bad_tokens for t in s2.lower().split()):
                return s2
        return None
    return s


def choose_by_letter(images, question, choice_map):
    from . import prompts
    prompt = prompts.prompt_choose_by_letter(question, choice_map)
    raw = vlm_generate(images, prompt, max_new_tokens=8).strip().upper()
    letter = raw[:1] if raw else list(choice_map.keys())[0]
    if letter not in choice_map:
        letter = list(choice_map.keys())[0]
    return choice_map[letter], raw
