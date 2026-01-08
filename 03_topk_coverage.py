import os
import re
import json
import csv
from collections import Counter
from typing import List, Dict, Any, Tuple

import utils

# =========================================================
# CONFIG (edit here)
# =========================================================
DATA_MODE = "jsonl"   # "jsonl" | "hf"

# --- jsonl mode paths ---
DATA_ROOT = "vqa-rad_extracted"
TRAIN_META = os.path.join(DATA_ROOT, "train", "meta.jsonl")
TEST_META  = os.path.join(DATA_ROOT, "test",  "meta.jsonl")

# --- hf mode ---
HF_DATASET_NAME = "vqa-rad"

TOP_K_LIST = [20, 50, 80, 100, 150, 200, 250, 300, 350, 400]
CSV_OUT = "topk_coverage_report.csv"

# =========================================================
# Answer normalization (keep consistent with your training code)
# =========================================================
_punc_re = re.compile(r"[^a-z0-9\s/]+", re.I)

_CANON = {
    "w contrast": "with contrast",
    "w/ contrast": "with contrast",
    "with iv contrast": "with contrast",
    "without contrast": "no contrast",
    "non contrast": "no contrast",
    "noncontrast": "no contrast",
    "x ray": "xray",
    "x-ray": "xray",
    "xray": "xray",
    "ct scan": "ct",
    "computed tomography": "ct",
    "mr": "mri",
    "mri scan": "mri",
    "yes.": "yes",
    "no.": "no",
    "y": "yes",
    "n": "no",
}

YESNO_SET = {"yes", "no"}

def normalize_answer(a: str) -> str:
    if a is None:
        return ""
    a = str(a).strip().lower()
    a = a.replace("&", " and ")
    a = _punc_re.sub(" ", a)
    a = " ".join(a.split())
    if a in _CANON:
        a = _CANON[a]
    a = a.replace("w/", "with ")
    a = a.replace(" w ", " with ")
    a = " ".join(a.split())
    if a in _CANON:
        a = _CANON[a]
    return " ".join(a.split())


# =========================================================
# Load data
# =========================================================
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"meta file not found: {path}")
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def load_data() -> Tuple[List[str], List[str]]:
    """Return (train_answers_norm, test_answers_norm)"""
    if DATA_MODE == "jsonl":
        train_items = read_jsonl(TRAIN_META)
        test_items  = read_jsonl(TEST_META)
        train_ans = [normalize_answer(x.get("answer", "")) for x in train_items]
        test_ans  = [normalize_answer(x.get("answer", "")) for x in test_items]
        return train_ans, test_ans

    if DATA_MODE == "hf":
        from datasets import load_dataset
        ds = load_dataset(HF_DATASET_NAME)
        train_ans = [normalize_answer(x.get("answer", "")) for x in ds["train"]]
        test_ans  = [normalize_answer(x.get("answer", "")) for x in ds["test"]]
        return train_ans, test_ans

    raise ValueError(f"Unknown DATA_MODE: {DATA_MODE}")

train_ans, test_ans = load_data()
print(f"Loaded train={len(train_ans)} test={len(test_ans)}")

# =========================================================
# Build ranking by TRAIN frequency
# =========================================================
train_cnt = Counter(train_ans)
ranked = [a for a, _ in train_cnt.most_common()]
unique_train = len(ranked)
print(f"Unique answers in train: {unique_train}")
print("Top-10 answers:", ranked[:10])

# =========================================================
# Coverage computation
# =========================================================
def coverage(ans_list: List[str], top_set: set) -> float:
    if not ans_list:
        return 0.0
    hit = sum(1 for a in ans_list if a in top_set)
    return hit / len(ans_list)

def yesno_open_stats(ans_list: List[str], top_set: set) -> Dict[str, float]:
    yn = [a for a in ans_list if utils.is_yesno(a)]
    op = [a for a in ans_list if not utils.is_yesno(a)]
    return {
        "yesno_ratio": len(yn) / max(1, len(ans_list)),
        "open_ratio":  len(op) / max(1, len(ans_list)),
        "yesno_cov":   coverage(yn, top_set) if yn else 0.0,
        "open_cov":    coverage(op, top_set) if op else 0.0,
    }

# =========================================================
# Produce rows
# =========================================================
rows = []
for k in TOP_K_LIST:
    k = int(k)
    k = max(1, min(k, unique_train))
    top_set = set(ranked[:k])

    tr_cov = coverage(train_ans, top_set)
    te_cov = coverage(test_ans, top_set)

    tr = yesno_open_stats(train_ans, top_set)
    te = yesno_open_stats(test_ans, top_set)

    rows.append({
        "K": k,
        "train_cov": tr_cov,
        "test_cov": te_cov,
        "train_yesno_ratio": tr["yesno_ratio"],
        "train_open_ratio": tr["open_ratio"],
        "train_yesno_cov": tr["yesno_cov"],
        "train_open_cov": tr["open_cov"],
        "test_yesno_ratio": te["yesno_ratio"],
        "test_open_ratio": te["open_ratio"],
        "test_yesno_cov": te["yesno_cov"],
        "test_open_cov": te["open_cov"],
    })

# =========================================================
# Write CSV
# =========================================================
fieldnames = [
    "K",
    "train_cov", "test_cov",
    "train_yesno_ratio", "train_open_ratio", "train_yesno_cov", "train_open_cov",
    "test_yesno_ratio", "test_open_ratio", "test_yesno_cov", "test_open_cov",
]

with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"\n✅ CSV saved to: {CSV_OUT}")

# =========================================================
# Print a compact preview
# =========================================================
print("\nPreview:")
print("K | train_cov | test_cov | test_open_cov")
print("-" * 44)
for r in rows:
    print(f"{r['K']:>3d} | {r['train_cov']:.4f}   | {r['test_cov']:.4f} | {r['test_open_cov']:.4f}")