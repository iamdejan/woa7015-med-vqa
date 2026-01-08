from typing import Any, Dict, List, Tuple
from pathlib import Path
import re
import json
import random
import numpy as np
import os
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


import torch
import evaluate
import nltk


# =========================================================
# Text metrics: BLEU-4, ROUGE-L, METEOR-lite
# =========================================================
print("Loading metrics...")
# This will now crash with a helpful error if download fails
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")
print("Metrics loaded successfully.")

def corpus_text_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """
    Computes standard metrics using HuggingFace 'evaluate' library.
    """
    n = len(preds)
    if n == 0 or bleu_metric is None:
        return {"bleu": 0.0, "rougeL": 0.0, "meteor": 0.0, "n": n}

    # 1. BLEU
    # 'evaluate' (and sacrebleu) expects references to be a list of lists 
    # because one prediction can have multiple valid ground truths.
    # Format: [[ref1], [ref2], ...]
    bleu_refs = [[r] for r in refs]
    
    try:
        bleu_res = bleu_metric.compute(predictions=preds, references=bleu_refs)
        bleu_score = bleu_res["bleu"]
    except ZeroDivisionError:
        bleu_score = 0.0

    # 2. ROUGE-L
    # ROUGE expects simple list of strings for both
    rouge_res = rouge_metric.compute(predictions=preds, references=refs)
    rouge_l_score = rouge_res["rougeL"]

    # 3. METEOR
    # METEOR expects simple list of strings for both
    meteor_res = meteor_metric.compute(predictions=preds, references=refs)
    meteor_score = meteor_res["meteor"]

    return {
        "bleu": bleu_score,
        "rougeL": rouge_l_score,
        "meteor": meteor_score,
        "n": float(n)
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def format_hms(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def pick_question(ex: Dict[str, Any]) -> str:
    if "question" in ex and isinstance(ex["question"], str):
        return ex["question"]
    if "questions" in ex and ex["questions"]:
        return ex["questions"][0]
    return ""

def pick_answer(ex: Dict[str, Any]) -> str:
    if "answer" in ex and isinstance(ex["answer"], str):
        return ex["answer"]
    if "answers" in ex and ex["answers"]:
        return ex["answers"][0]
    return ""


def is_yesno(ans: str) -> bool:
    a = normalize_text(ans)
    return a in {"yes", "no"}


def resolve_image_path(root_dir: str | Path, split_name: str, ex: Dict[str, Any]) -> Path:
    root_dir = Path(root_dir)
    split_dir = root_dir / split_name

    raw = ex.get("image_path") or ex.get("image") or ex.get("img") or ex.get("path") or ""
    cand = str(raw).strip().replace("\\", "/").lstrip("./")
    if not cand:
        raise FileNotFoundError("No image path field found in meta entry.")

    candidates: List[Path] = []
    p0 = Path(cand)
    if p0.is_absolute():
        candidates.append(p0)
    else:
        candidates.append(Path(cand))
        root_norm = root_dir.as_posix().replace("\\", "/")
        split_norm = split_dir.as_posix().replace("\\", "/")

        if not cand.startswith(root_norm + "/"):
            candidates.append(root_dir / cand)
        if not cand.startswith(split_norm + "/"):
            candidates.append(split_dir / cand)
        if cand.startswith(split_name + "/") and not cand.startswith(root_norm + "/"):
            candidates.append(root_dir / cand)
        if "/" not in cand:
            candidates.append(split_dir / cand)

    for p in candidates:
        if p.exists():
            return p

    tried = "\n".join([f"  - {p.as_posix()}" for p in candidates])
    raise FileNotFoundError(
        f"Image file not found. raw='{raw}'\nTried:\n{tried}\n"
        f"Hint: check meta.jsonl path format (root='{root_dir.as_posix()}', split='{split_name}')."
    )


def print_accuracy_results(m: Dict[str, float]):
    line = "=" * 31
    closed_c = int(m.get("closed_correct_count", 0))
    closed_t = int(m.get("closed_total_count", 0))
    open_c   = int(m.get("open_correct_count", 0))
    open_t   = int(m.get("open_total_count", 0))
    all_c    = int(m.get("all_correct_count", 0))
    all_t    = int(m.get("all_total_count", 0))

    closed_acc = (closed_c / max(1, closed_t)) * 100.0
    open_acc   = (open_c / max(1, open_t)) * 100.0
    all_acc    = (all_c / max(1, all_t)) * 100.0

    print(line)
    print("ACCURACY RESULTS")
    print(line)
    print(f"CLOSED ACCURACY: {closed_acc:.2f}% ({closed_c}/{closed_t})")
    print(f"OPEN ACCURACY: {open_acc:.2f}% ({open_c}/{open_t})")
    print(f"ALL ACCURACY: {all_acc:.2f}% ({all_c}/{all_t})")
