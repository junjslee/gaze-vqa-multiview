from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class EvalSample:
    sample_uid: str
    task_id: int
    task_type: str
    scene: str
    seq: str
    frame: str
    question: str
    groundtruth_answer: str
    image_paths: List[str]
    camera_ids: List[str]
    source_benchmark: str
    review_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InvalidSample:
    index: int
    reason: str
    raw_task_id: Any
    raw_task_type: Any
    raw_question: Any

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionRecord:
    sample_uid: str
    model_key: str
    model_id: str
    inference_answer: str
    error: str
    latency_s: float
    tokens_in: int
    tokens_out: int
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeRecord:
    sample_uid: str
    model_key: str
    gemini_judge: str
    judge_model: str
    judge_latency_s: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CampaignMeta:
    campaign_name: str
    created_at: str
    benchmark_path: str
    benchmark_sha256: str
    gt_manifest_path: str
    gt_manifest_sha256: str
    prompt_version: str
    model_map: Dict[str, Dict[str, Any]]
    container_sif: str
    slurm: Dict[str, Any]
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

