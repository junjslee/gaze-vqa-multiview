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


def classify_gemini_error(err):
    s = str(err or "")
    su = s.upper()
    if (
        "RESOURCE_EXHAUSTED" in su
        or "INSUFFICIENT" in su
        or "QUOTA" in su
        or "BILLING" in su
        or "FREE TRIAL" in su
    ):
        return "quota"
    if "PERMISSION_DENIED" in su or "API ERROR 401" in su or "API ERROR 403" in su:
        return "auth"
    if "API ERROR 404" in su or "NOT_FOUND" in su:
        return "not_found"
    if "READ TIMED OUT" in su or _is_transient_api_error_message(s):
        return "transient"
    return "unknown"


def is_gemini_hard_error(err):
    return classify_gemini_error(err) in {"quota", "auth", "not_found"}


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


def _strip_json_fence(text):
    s = str(text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _gemini_json_parse_health(text):
    s = _strip_json_fence(text)
    if not s:
        return {"parse_ok": False, "partial": False, "status": "empty"}
    try:
        json.loads(s)
        return {"parse_ok": True, "partial": False, "status": "ok"}
    except Exception:
        pass
    opens = s.count("{")
    closes = s.count("}")
    partial = opens > closes or s.endswith(":") or s.endswith(",") or s.endswith('"')
    return {"parse_ok": False, "partial": bool(partial), "status": "invalid"}


def _is_gemini3_model(model_name):
    return _normalize_model_name(model_name).startswith("gemini-3")


def _extract_gemini_model_content(data):
    candidates = data.get("candidates") or []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        content = cand.get("content") or {}
        parts = content.get("parts") or []
        cleaned_parts = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            p = {}
            if "text" in part:
                p["text"] = str(part.get("text") or "")
            ts = part.get("thoughtSignature")
            if ts is None:
                ts = part.get("thought_signature")
            if ts:
                p["thoughtSignature"] = str(ts)
            # Function-call fields are preserved for forward compatibility.
            if isinstance(part.get("functionCall"), dict):
                p["functionCall"] = part["functionCall"]
            if isinstance(part.get("functionResponse"), dict):
                p["functionResponse"] = part["functionResponse"]
            if p:
                cleaned_parts.append(p)
        if cleaned_parts:
            role = str(content.get("role") or "model").strip().lower()
            if role not in {"user", "model"}:
                role = "model"
            return {"role": role, "parts": cleaned_parts}
    return {"role": "model", "parts": []}


def _extract_gemini_thought_signatures(model_content):
    out = []
    if not isinstance(model_content, dict):
        return out
    for part in model_content.get("parts") or []:
        if not isinstance(part, dict):
            continue
        ts = part.get("thoughtSignature")
        if ts:
            out.append(str(ts))
    return out


def _sanitize_gemini_history_contents(contents):
    if not isinstance(contents, list):
        return []
    allowed_part_keys = {
        "text",
        "inlineData",
        "fileData",
        "functionCall",
        "functionResponse",
        "executableCode",
        "codeExecutionResult",
        "thoughtSignature",
        "mediaResolution",
    }
    out = []
    for c in contents:
        if not isinstance(c, dict):
            continue
        role = str(c.get("role") or "user").strip().lower()
        if role not in {"user", "model"}:
            role = "user"
        parts_in = c.get("parts") or []
        parts_out = []
        if isinstance(parts_in, list):
            for p in parts_in:
                if not isinstance(p, dict):
                    continue
                q = {}
                for k in allowed_part_keys:
                    if k in p and p.get(k) is not None:
                        q[k] = p.get(k)
                ts = p.get("thought_signature")
                if ts and "thoughtSignature" not in q:
                    q["thoughtSignature"] = ts
                if q:
                    parts_out.append(q)
        if parts_out:
            out.append({"role": role, "parts": parts_out})
    return out


def _normalize_gemini_thinking_level(v):
    s = str(v or "high").strip().lower()
    if s not in {"minimal", "low", "medium", "high"}:
        s = "high"
    return s


def _normalize_gemini_media_resolution(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    if not s:
        return None
    aliases = {
        "low": "MEDIA_RESOLUTION_LOW",
        "medium": "MEDIA_RESOLUTION_MEDIUM",
        "high": "MEDIA_RESOLUTION_HIGH",
        "ultra_high": "MEDIA_RESOLUTION_ULTRA_HIGH",
        "ultra-high": "MEDIA_RESOLUTION_ULTRA_HIGH",
    }
    if s in aliases:
        return aliases[s]
    s2 = s.replace("-", "_").upper()
    if s2.startswith("MEDIA_RESOLUTION_"):
        return s2
    return f"MEDIA_RESOLUTION_{s2}"


def _gemini_api_versions_for_request(cfg, per_part_requested=False):
    req = str(cfg.get("api_version") or os.environ.get("GEMINI_API_VERSION", "v1beta")).strip().lower()
    if req not in {"v1beta", "v1alpha"}:
        req = "v1beta"
    allow_fallback = bool(cfg.get("allow_version_fallback", True))

    ordered = []
    if per_part_requested:
        ordered.extend(["v1alpha", req])
    else:
        ordered.append(req)
    if allow_fallback:
        ordered.extend(["v1beta", "v1alpha"])

    out = []
    seen = set()
    for v in ordered:
        if v in {"v1beta", "v1alpha"} and v not in seen:
            out.append(v)
            seen.add(v)
    return out or ["v1beta"]


def _gemini_generate(images, prompt, max_new_tokens, model_id=None, generation_cfg=None, return_meta=False):
    api_key = _require_gemini_api_key()
    pil_images = _normalize_pil_images(images)
    cfg = generation_cfg or {}
    model_name = str(model_id or st.VLM_MODEL_ID).strip()

    history_contents = _sanitize_gemini_history_contents(
        cfg.get("history_contents") or cfg.get("gemini_history_contents") or []
    )
    encoded_images = [_pil_to_jpeg_b64(pil) for pil in pil_images]
    media_resolution_enum = _normalize_gemini_media_resolution(cfg.get("media_resolution"))
    media_resolution_scope = str(cfg.get("media_resolution_scope", "auto")).strip().lower()
    if media_resolution_scope not in {"global", "per_part", "auto"}:
        media_resolution_scope = "global"

    def _build_user_parts(api_version, include_advanced, model_used):
        parts_local = [{"text": prompt}]
        per_part_enabled = (
            include_advanced
            and bool(media_resolution_enum)
            and _is_gemini3_model(model_used)
            and (
                media_resolution_scope == "per_part"
                or (media_resolution_scope == "auto" and media_resolution_enum == "MEDIA_RESOLUTION_ULTRA_HIGH")
            )
            and api_version == "v1alpha"
        )
        for b64 in encoded_images:
            part = {
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": b64,
                }
            }
            if per_part_enabled:
                part["mediaResolution"] = {"level": str(media_resolution_enum).lower()}
            parts_local.append(part)
        return parts_local, per_part_enabled

    def _build_payload(
        tok_budget,
        api_version,
        active_model,
        include_advanced=True,
        include_thinking=True,
        include_structured=False,
    ):
        temperature = float(cfg.get("temperature", 0))
        top_p = cfg.get("top_p")
        top_k = cfg.get("top_k")
        candidate_count = cfg.get("candidate_count")
        thinking_budget = int(cfg.get("thinking_budget", st.GEMINI_THINKING_BUDGET))
        thinking_level = _normalize_gemini_thinking_level(cfg.get("thinking_level", "high"))
        structured_json = bool(cfg.get("structured_json", False))
        json_schema = cfg.get("json_schema")
        user_parts, per_part_enabled = _build_user_parts(api_version, include_advanced, active_model)

        payload_local = {
            "contents": history_contents + [{"role": "user", "parts": user_parts}],
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
        if include_advanced and (not per_part_enabled) and media_resolution_enum:
            mr = str(media_resolution_enum)
            if mr == "MEDIA_RESOLUTION_ULTRA_HIGH" and api_version != "v1alpha":
                st.logger.warning(
                    "Gemini ultra_high media resolution requested without v1alpha per-part path; "
                    "downgrading to MEDIA_RESOLUTION_HIGH."
                )
                mr = "MEDIA_RESOLUTION_HIGH"
            payload_local["generationConfig"]["mediaResolution"] = mr
        if include_thinking:
            # Gemini 3 uses thinkingLevel; 2.5/legacy uses thinkingBudget.
            thinking_cfg = {}
            if _is_gemini3_model(active_model):
                thinking_cfg["thinkingLevel"] = thinking_level
            else:
                thinking_cfg["thinkingBudget"] = max(0, int(thinking_budget))
            payload_local["generationConfig"]["thinkingConfig"] = thinking_cfg
        if include_structured and structured_json:
            payload_local["generationConfig"]["responseMimeType"] = "application/json"
            if isinstance(json_schema, dict) and json_schema:
                payload_local["generationConfig"]["responseSchema"] = json_schema
        return payload_local, per_part_enabled

    def _is_schema_compat_error(err):
        txt = str(err or "").lower()
        return (
            "unknown name" in txt
            or "invalid json payload" in txt
            or "unsupported" in txt
            or "invalid argument" in txt
            or "unknown field" in txt
        )

    def _is_model_not_found_error(err):
        txt = str(err or "").lower()
        return (
            "gemini api error 404" in txt
            and (
                "is not found" in txt
                or "not found for api version" in txt
                or "not supported for generatecontent" in txt
            )
        )

    def _is_bad_request_error(err):
        txt = str(err or "").lower()
        return "gemini api error 400" in txt

    def _model_candidates(primary_model):
        defaults = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
        ]
        env_raw = str(os.environ.get("GEMINI_FALLBACK_MODELS", "")).strip()
        env_models = [m.strip() for m in re.split(r"[,\s]+", env_raw) if m.strip()] if env_raw else []
        ordered = [str(primary_model or "").strip()] + env_models + defaults
        out = []
        seen = set()
        for m in ordered:
            if not m:
                continue
            key = m.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(m)
        return out

    def _build_meta(
        text,
        active_model,
        api_version,
        mode_used,
        data_obj,
        finish_reason=None,
        error=None,
        token_budget_used=None,
        retry_count=0,
        schema_enabled=False,
        parse_health=None,
    ):
        model_content = _extract_gemini_model_content(data_obj or {})
        thought_sigs = _extract_gemini_thought_signatures(model_content)
        history_turn = []
        if prompt:
            history_turn.append({"role": "user", "parts": [{"text": prompt}]})
        if model_content.get("parts"):
            history_turn.append(model_content)
        return {
            "text": text or "",
            "provider": "gemini",
            "model_requested": model_name,
            "model_used": active_model or model_name,
            "api_version_used": api_version or "v1beta",
            "mode_used": mode_used or "full",
            "finish_reason": str(finish_reason or ""),
            "token_budget_used": int(token_budget_used or 0),
            "retry_count": int(max(0, retry_count or 0)),
            "schema_enabled": bool(schema_enabled),
            "parse_health": parse_health if isinstance(parse_health, dict) else None,
            "thought_signature_count": int(len(thought_sigs)),
            "thought_signatures": thought_sigs,
            "history_turn": history_turn,
            "error": str(error or ""),
        }

    headers = {"Content-Type": "application/json"}
    base_budget = max(1, int(max_new_tokens))
    structured_requested = bool(cfg.get("structured_json", False))
    retry_on_partial_json = bool(cfg.get("retry_on_partial_json", False))
    try:
        retry_token_multiplier = float(cfg.get("retry_token_multiplier", 2.0))
    except Exception:
        retry_token_multiplier = 2.0
    if retry_token_multiplier < 1.1:
        retry_token_multiplier = 1.1
    token_budgets = [base_budget]
    retry_budget = int(math.ceil(base_budget * retry_token_multiplier))
    if retry_budget > base_budget:
        token_budgets.append(retry_budget)
    if base_budget < 128:
        token_budgets.append(max(128, base_budget * 2))
    # De-duplicate while preserving order.
    token_budgets = list(dict.fromkeys(token_budgets))
    per_part_requested = (
        media_resolution_scope == "per_part"
        or (media_resolution_scope == "auto" and media_resolution_enum == "MEDIA_RESOLUTION_ULTRA_HIGH")
    )
    api_versions = _gemini_api_versions_for_request(cfg, per_part_requested=per_part_requested)

    payload_modes = []
    if structured_requested:
        payload_modes.extend([
            ("structured_full", True, True, True),
            ("structured_no_advanced", False, True, True),
            ("structured_minimal", False, False, True),
        ])
    payload_modes.extend([
        ("full", True, True, False),
        ("no_advanced", False, True, False),
        ("minimal", False, False, False),
    ])
    last_err = None
    for model_i, active_model in enumerate(_model_candidates(model_name), start=1):
        if model_i > 1:
            st.logger.warning(
                f"Gemini model fallback active: requested='{model_name}', trying='{active_model}'."
            )
        for api_version in api_versions:
            url = f"{st.GEMINI_API_BASE.rstrip('/')}/{api_version}/models/{active_model}:generateContent?key={api_key}"
            last_data = None
            try:
                for attempt_i, tok_budget in enumerate(token_budgets, start=1):
                    data = None
                    mode_used = None
                    mode_err = None
                    for mode_name, use_adv, use_thinking, use_structured in payload_modes:
                        payload, per_part_enabled = _build_payload(
                            tok_budget,
                            api_version=api_version,
                            active_model=active_model,
                            include_advanced=use_adv,
                            include_thinking=use_thinking,
                            include_structured=use_structured,
                        )
                        try:
                            data = _post_json(url, headers, payload, provider_name="Gemini")
                            mode_used = mode_name
                            if per_part_enabled:
                                st.logger.info("Gemini per-part media resolution path enabled.")
                            break
                        except RuntimeError as e:
                            mode_err = e
                            if _is_transient_api_error_message(e):
                                st.logger.warning(
                                    "Gemini transient API failure after bounded retries; returning empty output for this call."
                                )
                                if return_meta:
                                    return _build_meta(
                                        "",
                                        active_model=active_model,
                                        api_version=api_version,
                                        mode_used=mode_name,
                                        data_obj={},
                                        error=e,
                                    )
                                return ""
                            if _is_schema_compat_error(e):
                                st.logger.warning(
                                    f"Gemini config mode '{mode_name}' rejected by endpoint; trying compatibility fallback."
                                )
                                continue
                            raise
                    if data is None:
                        if mode_err is not None:
                            raise mode_err
                        raise RuntimeError("Gemini request failed before receiving response data.")

                    if mode_used and mode_used != "full":
                        st.logger.warning(f"Gemini request used compatibility mode: {mode_used}")
                    last_data = data

                    usage = _extract_gemini_usage(data)
                    if usage is not None:
                        _record_usage(
                            "gemini",
                            active_model,
                            input_tokens=usage["input_tokens"],
                            output_tokens=usage["output_tokens"],
                            cached_input_tokens=usage["cached_input_tokens"],
                            total_tokens=usage["total_tokens"],
                        )

                    finish_reason = _extract_gemini_finish_reason(data).upper()
                    text = _extract_gemini_text(data)
                    parse_health = None
                    if structured_requested:
                        parse_health = _gemini_json_parse_health(text)
                    if text:
                        should_retry_partial = (
                            structured_requested
                            and retry_on_partial_json
                            and attempt_i < len(token_budgets)
                            and (
                                (isinstance(parse_health, dict) and (not bool(parse_health.get("parse_ok"))))
                                or finish_reason == "MAX_TOKENS"
                            )
                        )
                        if should_retry_partial:
                            st.logger.warning(
                                "Gemini produced partial/invalid JSON; retrying "
                                f"with maxOutputTokens={token_budgets[attempt_i]}"
                            )
                            continue
                        if return_meta:
                            return _build_meta(
                                text,
                                active_model=active_model,
                                api_version=api_version,
                                mode_used=mode_used,
                                data_obj=data,
                                finish_reason=finish_reason,
                                token_budget_used=tok_budget,
                                retry_count=(attempt_i - 1),
                                schema_enabled=bool(mode_used and mode_used.startswith("structured_")),
                                parse_health=parse_health,
                            )
                        return text

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
                    if return_meta:
                        return _build_meta(
                            "",
                            active_model=active_model,
                            api_version=api_version,
                            mode_used=mode_used,
                            data_obj=last_data or {},
                            token_budget_used=tok_budget if "tok_budget" in locals() else 0,
                            retry_count=(attempt_i - 1) if "attempt_i" in locals() else 0,
                            schema_enabled=bool(mode_used and str(mode_used).startswith("structured_")),
                        )
                    return ""

                finish_reason = _extract_gemini_finish_reason(last_data or {}).upper()
                if finish_reason:
                    st.logger.warning(
                        f"Gemini returned no text output (finishReason={finish_reason}). "
                        "Returning empty string."
                    )
                    if return_meta:
                        return _build_meta(
                            "",
                            active_model=active_model,
                            api_version=api_version,
                            mode_used=mode_used,
                            data_obj=last_data or {},
                            finish_reason=finish_reason,
                            token_budget_used=tok_budget if "tok_budget" in locals() else 0,
                            retry_count=(attempt_i - 1) if "attempt_i" in locals() else 0,
                            schema_enabled=bool(mode_used and str(mode_used).startswith("structured_")),
                        )
                    return ""
                st.logger.warning(f"Gemini returned no text output: {str(last_data)[:800]}")
                if return_meta:
                    return _build_meta(
                        "",
                        active_model=active_model,
                        api_version=api_version,
                        mode_used=mode_used,
                        data_obj=last_data or {},
                        token_budget_used=tok_budget if "tok_budget" in locals() else 0,
                        retry_count=(attempt_i - 1) if "attempt_i" in locals() else 0,
                        schema_enabled=bool(mode_used and str(mode_used).startswith("structured_")),
                    )
                return ""
            except RuntimeError as e:
                last_err = e
                if _is_model_not_found_error(e):
                    st.logger.warning(
                        f"Gemini model '{active_model}' unavailable for API {api_version}; trying next candidate."
                    )
                    continue
                if _is_bad_request_error(e) and _is_gemini3_model(active_model):
                    st.logger.warning(
                        f"Gemini 3 request rejected on API {api_version}; attempting next API/model fallback."
                    )
                    continue
                raise

    if last_err is not None:
        raise last_err
    raise RuntimeError("Gemini model resolution failed without a usable candidate.")


def _default_model_for_provider(provider_name):
    p = str(provider_name or "").strip().lower()
    if p == "qwen":
        return str(st.QWEN_MODEL_ID)
    if p == "openai":
        return str(os.environ.get("OPENAI_MODEL", "gpt-4o"))
    if p == "gemini":
        return str(os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    return str(st.VLM_MODEL_ID)


def vlm_generate(
    images,
    prompt,
    max_new_tokens=128,
    provider=None,
    model_id=None,
    generation_cfg=None,
    return_meta=False,
):
    global _vlm_calls
    _vlm_calls += 1

    if st.ARGS.skip_vlm:
        if return_meta:
            return {
                "text": "SKIP_VLM",
                "provider": str(provider or st.VLM_PROVIDER),
                "model_requested": str(model_id or st.VLM_MODEL_ID),
                "model_used": str(model_id or st.VLM_MODEL_ID),
                "mode_used": "skip_vlm",
                "error": "",
            }
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
        meta = {
            "text": pred,
            "provider": "qwen",
            "model_requested": resolved_model,
            "model_used": str(st.QWEN_MODEL_ID),
            "mode_used": "default",
            "error": "",
        }
    elif provider_name == "openai":
        pred = _openai_generate(
            images,
            prompt,
            max_new_tokens=max_new_tokens,
            model_id=resolved_model,
            generation_cfg=generation_cfg,
        )
        meta = {
            "text": pred,
            "provider": "openai",
            "model_requested": resolved_model,
            "model_used": resolved_model,
            "mode_used": "default",
            "error": "",
        }
    elif provider_name == "gemini":
        gem_resp = _gemini_generate(
            images,
            prompt,
            max_new_tokens=max_new_tokens,
            model_id=resolved_model,
            generation_cfg=generation_cfg,
            return_meta=bool(return_meta),
        )
        if return_meta and isinstance(gem_resp, dict):
            meta = gem_resp
            pred = str(meta.get("text") or "")
        else:
            pred = str(gem_resp or "")
            meta = {
                "text": pred,
                "provider": "gemini",
                "model_requested": resolved_model,
                "model_used": resolved_model,
                "mode_used": "default",
                "error": "",
            }
    else:
        raise ValueError(f"Unsupported VLM provider: {provider_name}")

    if st.ARGS.log_prompts:
        st.logger.info(f"[VLM] output:\n{pred}")

    if return_meta:
        return meta
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
