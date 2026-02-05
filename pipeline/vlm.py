import gc
import math
import os
import re
from PIL import Image

from . import state as st

_qwen_model = None
_qwen_proc = None
_vlm_calls = 0


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


def vlm_generate(images, prompt, max_new_tokens=128):
    global _vlm_calls
    _vlm_calls += 1

    if st.ARGS.skip_vlm:
        return "SKIP_VLM"

    if st.ARGS.log_prompts:
        img_count = 0
        if images is None:
            img_count = 0
        elif isinstance(images, list):
            img_count = len([x for x in images if x is not None])
        else:
            img_count = 1
        st.logger.info(f"[VLM] images={img_count} max_new_tokens={max_new_tokens}")
        st.logger.info(f"[VLM] prompt:\n{prompt}")

    import torch
    model, proc = load_qwen()

    pil_images = []
    if images is None:
        images_list = []
    elif isinstance(images, list):
        images_list = images
    else:
        images_list = [images]

    for im in images_list:
        if im is None:
            continue
        if isinstance(im, str):
            pil = Image.open(im).convert("RGB")
        elif isinstance(im, Image.Image):
            pil = im.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type in vlm_generate(): {type(im)}")
        pil = _ensure_min_image_side(pil, min_side=28)
        pil_images.append(pil)

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

    if st.ARGS.log_prompts:
        st.logger.info(f"[VLM] output:\n{pred}")

    del inputs, out
    gc.collect()
    if torch.cuda.is_available() and (_vlm_calls % st.EMPTY_CACHE_EVERY == 0):
        torch.cuda.empty_cache()

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
