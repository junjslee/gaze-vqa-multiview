from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def _ensure_evaluator_imports() -> None:
    root = Path(__file__).resolve().parents[2]
    eval_dir = root / "draft_of_evaluation"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))


def _normalize_text(x: Any) -> str:
    return str(x or "").replace("assistant\n", "").strip()


def _safe_float(x: Any) -> float:
    try:
        val = float(x)
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return val
    except Exception:
        return 0.0


def compute_sentence_similarity(preds: List[str], gts: List[str]) -> Dict[str, Any]:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import pytorch_cos_sim
    except Exception as exc:
        return {"sentence_sim": None, "error": f"sentence-transformers unavailable: {exc}"}

    if not preds:
        return {"sentence_sim": 0.0, "error": ""}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    with torch.no_grad():
        ep = model.encode(preds, convert_to_tensor=True, show_progress_bar=False, batch_size=64)
        eg = model.encode(gts, convert_to_tensor=True, show_progress_bar=False, batch_size=64)
        sims = pytorch_cos_sim(ep, eg).diag().detach().cpu().numpy()
    return {"sentence_sim": _safe_float(np.mean(sims)), "error": ""}


def compute_ngram_metrics(preds: List[str], gts: List[str]) -> Dict[str, Any]:
    _ensure_evaluator_imports()
    metrics: Dict[str, Any] = {"cider": None, "bleu": None, "meteor": None, "rouge": None, "error": ""}
    if not preds:
        return {"cider": 0.0, "bleu": 0.0, "meteor": 0.0, "rouge": 0.0, "error": ""}

    try:
        from evaluator.ngram_metrics.bleu.bleu import Bleu
        from evaluator.ngram_metrics.cider.cider import Cider
        from evaluator.ngram_metrics.meteor.meteor import Meteor
        from evaluator.ngram_metrics.rouge.rouge import Rouge
    except Exception as exc:
        metrics["error"] = f"ngram metric imports unavailable: {exc}"
        return metrics

    gt_map = {i: [gts[i]] for i in range(len(gts))}
    pred_map = {i: [preds[i]] for i in range(len(preds))}
    errs: List[str] = []
    try:
        metrics["cider"] = _safe_float(Cider(n=4).compute_score(gt_map, pred_map)[0])
    except Exception as exc:
        errs.append(f"cider:{exc}")
    try:
        bleu_scores = Bleu(n=4).compute_score(gt_map, pred_map)[0]
        metrics["bleu"] = _safe_float(bleu_scores[-1] if isinstance(bleu_scores, (list, tuple)) else bleu_scores)
    except Exception as exc:
        errs.append(f"bleu:{exc}")
    try:
        metrics["meteor"] = _safe_float(Meteor().compute_score(gt_map, pred_map)[0])
    except Exception as exc:
        errs.append(f"meteor:{exc}")
    try:
        metrics["rouge"] = _safe_float(Rouge().compute_score(gt_map, pred_map)[0])
    except Exception as exc:
        errs.append(f"rouge:{exc}")
    metrics["error"] = " | ".join(errs)
    return metrics


def compute_text_metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    preds: List[str] = []
    gts: List[str] = []
    for row in rows:
        preds.append(_normalize_text(row.get("inference_answer")))
        gts.append(_normalize_text(row.get("groundtruth_answer")))

    sim = compute_sentence_similarity(preds=preds, gts=gts)
    ngram = compute_ngram_metrics(preds=preds, gts=gts)
    return {
        "sentence_sim": sim.get("sentence_sim"),
        "cider": ngram.get("cider"),
        "bleu": ngram.get("bleu"),
        "meteor": ngram.get("meteor"),
        "rouge": ngram.get("rouge"),
        "errors": [e for e in [sim.get("error"), ngram.get("error")] if e],
        "count": len(preds),
    }

