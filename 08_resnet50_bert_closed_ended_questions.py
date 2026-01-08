from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import json
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from difflib import SequenceMatcher

from transformers import AutoTokenizer, AutoModel
from transformers.utils import logging as hf_logging


import utils


# Common Functions
from difflib import SequenceMatcher

IMAGE_SIZE=224


def split_train_val(train_all: List[Dict[str, Any]], seed: int, train_val_split: float):
    rng = np.random.RandomState(seed)
    idxs = np.arange(len(train_all))
    rng.shuffle(idxs)
    cut = int(len(train_all) * train_val_split)
    train_rows = [train_all[i] for i in idxs[:cut]]
    val_rows   = [train_all[i] for i in idxs[cut:]]
    return train_rows, val_rows


def build_open_vocab_for_testing(train_rows: List[Dict[str, Any]], top_k: int, add_unk: bool) -> Tuple[Dict[str, int], List[str], Dict[str, int]]:
    counts: Dict[str, int] = {}
    for ex in train_rows:
        a = utils.normalize_text(utils.pick_answer(ex))
        if not a or utils.is_yesno(a):
            continue
        counts[a] = counts.get(a, 0) + 1

    sorted_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    kept = [w for w, _ in sorted_items[:top_k]]

    ans2id = {w: i for i, w in enumerate(kept)}
    if add_unk:
        ans2id["<unk>"] = len(ans2id)

    id2ans = [None] * len(ans2id)
    for w, i in ans2id.items():
        id2ans[int(i)] = w

    return ans2id, id2ans, counts


def image_to_tensor_train_style(img: Image.Image) -> torch.Tensor:
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)


def fuzzy_score(a: str, b: str) -> float:
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    return SequenceMatcher(None, a, b).ratio()

def _safe_filename(s: str, max_len: int = 80) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    if not s:
        s = "sample"
    return s[:max_len]


def pick_device(pref: str = "cuda") -> torch.device:
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("\n", " ").replace("\t", " ")
    s = " ".join(s.split())
    return s


def fuzzy_sim(a: str, b: str) -> float:
    a = norm_text(a)
    b = norm_text(b)
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def safe_torch_load(path: str) -> Any:
    """
    Avoid FutureWarning spam by preferring weights_only=True.
    Falls back gracefully if not supported.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # older torch: no weights_only
        return torch.load(path, map_location="cpu")
    except Exception:
        # if weights_only=True fails due to checkpoint content, fallback
        return torch.load(path, map_location="cpu")

# ResNet50 + BERT (closed-ended questions)
import os
import json
import time
import random
import difflib
import warnings
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt

from transformers import BertTokenizerFast, BertModel
from transformers import logging as hf_logging


import utils


# --------------------------
# CONFIG (EDIT THESE)
# --------------------------
DATA_ROOT = "vqa-rad_extracted"
META_NAME = "meta.jsonl"

# ✅ 强制从 train 抽样（你要的）
SAMPLE_SPLIT = "train"

# checkpoint
CKPT_PATH = "Model_Results_TwoHead/_checkpoints/resnet50_bert_twohead_best_by_test.pt"

# vocab artifacts (will auto-create if missing)
VOCAB_DIR = "Model_Results_SingleHead/_artifacts"
ID2ANS_PATH = os.path.join(VOCAB_DIR, "id2ans.json")
ANS2ID_PATH = os.path.join(VOCAB_DIR, "ans2id.json")

# must match training (if you used UNK)
ADD_UNK = True
UNK_TOKEN = "unk"

# ✅ 只从“GT 在 vocab 内”的样本里抽
ONLY_IN_VOCAB = True
EXCLUDE_UNK = True  # 建议 True：抽样时排除 GT=unk

# random
RANDOM_SEED = None  # None = real random each run

# device
DEVICE_PREF = "cuda"  # "cuda" | "cpu"

# output figure
SAVE_FIG = True
OUT_FIG = f"vqa_debug_{int(time.time())}.png"

# bert
BERT_NAME = "bert-base-uncased"
MAX_Q_LEN = 64


# --------------------------
# Silence warnings (no "red spam")
# --------------------------
def silence_warnings():
    # kill most warnings that show as red blocks in notebooks
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # HuggingFace transformers verbosity
    hf_logging.set_verbosity_error()


# --------------------------
# Helpers
# --------------------------
def extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    # common patterns
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
        return ckpt_obj["state_dict"]
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
        return ckpt_obj["model_state_dict"]
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        return ckpt_obj["model"]
    # plain state dict
    if isinstance(ckpt_obj, dict) and all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
        return ckpt_obj
    raise RuntimeError("Unknown checkpoint format. Can't find a state_dict.")


def try_extract_vocab_from_ckpt(ckpt_obj: Any) -> Tuple[Optional[List[str]], Optional[Dict[str, int]]]:
    """
    If your training script saved vocab inside checkpoint, we reuse it (best).
    """
    if not isinstance(ckpt_obj, dict):
        return None, None

    id2ans_keys = ["id2ans", "idx2ans", "answer_vocab", "answers", "id_to_ans"]
    ans2id_keys = ["ans2id", "ans_to_id", "answer_to_id", "vocab_map"]

    id2ans = None
    ans2id = None

    for k in id2ans_keys:
        v = ckpt_obj.get(k, None)
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            id2ans = v
            break

    for k in ans2id_keys:
        v = ckpt_obj.get(k, None)
        if isinstance(v, dict) and all(isinstance(kk, str) and isinstance(vv, int) for kk, vv in v.items()):
            ans2id = v
            break

    return id2ans, ans2id


def infer_answer_head_from_ckpt(sd: Dict[str, torch.Tensor]) -> Tuple[str, int]:
    candidates = []
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor) or v.ndim != 2:
            continue

        kl = k.lower()
        out_dim = int(v.shape[0])

        # skip BERT word embeddings
        if "embeddings.word_embeddings" in kl:
            continue

        score = 0
        if "answer" in kl: score += 4
        if "classifier" in kl: score += 3
        if "fc2" in kl: score += 3
        if "head" in kl: score += 2
        if kl.endswith(".weight"): score += 1
        if out_dim > 5000: score -= 10

        if score > 0:
            candidates.append((score, out_dim, k))

    if not candidates:
        raise RuntimeError("Could not infer answer head from checkpoint.")

    candidates.sort(key=lambda x: (-x[0], x[1]))
    best = candidates[0]
    return best[2], best[1]


def build_vocab_from_train_meta(train_meta: List[Dict[str, Any]], top_k: int, add_unk: bool) -> Tuple[List[str], Dict[str, int]]:
    freq: Dict[str, int] = {}
    for ex in train_meta:
        ans = ex.get("answer_norm", ex.get("answer", ""))
        ans = norm_text(str(ans))
        if not ans:
            continue
        freq[ans] = freq.get(ans, 0) + 1

    sorted_ans = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    id2ans = [a for a, _ in sorted_ans[:top_k]]

    if add_unk and UNK_TOKEN not in id2ans:
        id2ans.append(UNK_TOKEN)

    ans2id = {a: i for i, a in enumerate(id2ans)}
    return id2ans, ans2id


def save_vocab(id2ans: List[str], ans2id: Dict[str, int], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "id2ans.json"), "w", encoding="utf-8") as f:
        json.dump(id2ans, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "ans2id.json"), "w", encoding="utf-8") as f:
        json.dump(ans2id, f, ensure_ascii=False, indent=2)


def load_vocab(id2ans_path: str, ans2id_path: str) -> Tuple[List[str], Dict[str, int]]:
    with open(id2ans_path, "r", encoding="utf-8") as f:
        id2ans = json.load(f)
    with open(ans2id_path, "r", encoding="utf-8") as f:
        ans2id = json.load(f)
    return id2ans, ans2id


def safe_torch_load(path: str) -> Any:
    """
    Avoid FutureWarning spam by preferring weights_only=True.
    Falls back gracefully if not supported.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # older torch: no weights_only
        return torch.load(path, map_location="cpu")
    except Exception:
        # if weights_only=True fails due to checkpoint content, fallback
        return torch.load(path, map_location="cpu")


# --------------------------
# Model
# --------------------------
class ResNetBertSingleHead(nn.Module):
    def __init__(self, num_answers: int, head_style: str = "answer_head"):
        super().__init__()
        cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(cnn.children())[:-1])  # (B,2048,1,1)
        self.bert = BertModel.from_pretrained(BERT_NAME)

        self.fuse = nn.Linear(2048 + self.bert.config.hidden_size, 1024)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.2)

        # keep multiple names to match common ckpt naming
        self.answer_head = nn.Linear(1024, num_answers)
        self.classifier = nn.Linear(1024, num_answers)
        self.fc2 = nn.Linear(1024, num_answers)

        self.head_style = head_style  # "answer_head" | "classifier" | "fc2"

    def forward(self, image_tensor, input_ids, attention_mask):
        x_img = self.cnn(image_tensor).flatten(1)
        x_txt = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        x = torch.cat([x_img, x_txt], dim=1)
        x = self.drop(self.act(self.fuse(x)))
        if self.head_style == "classifier":
            return self.classifier(x)
        if self.head_style == "fc2":
            return self.fc2(x)
        return self.answer_head(x)


# --------------------------
# Main
# --------------------------
def main():
    silence_warnings()

    # randomness
    if RANDOM_SEED is None:
        rng = random.SystemRandom()
    else:
        random.seed(RANDOM_SEED)
        rng = random

    device = pick_device(DEVICE_PREF)
    print("Using device:", device)

    # load meta
    train_meta_path = os.path.join(DATA_ROOT, "train", META_NAME)
    sample_meta_path = os.path.join(DATA_ROOT, SAMPLE_SPLIT, META_NAME)

    train_meta = utils.load_jsonl(train_meta_path)
    sample_meta = utils.load_jsonl(sample_meta_path)

    print(f"Loaded {len(train_meta)} train samples from {train_meta_path}")
    print(f"Loaded {len(sample_meta)} sample-candidate samples from {sample_meta_path}")
    print(f"[SAMPLE] Sampling split = {SAMPLE_SPLIT}")

    # load ckpt & infer answer head dim
    ckpt_obj = safe_torch_load(CKPT_PATH)
    sd = extract_state_dict(ckpt_obj)

    head_key, num_answers_ckpt = infer_answer_head_from_ckpt(sd)
    hk = head_key.lower()
    if "classifier" in hk:
        head_style = "classifier"
    elif "fc2" in hk:
        head_style = "fc2"
    else:
        head_style = "answer_head"

    print(f"[CKPT] inferred head: {head_key} | head_style={head_style} | num_answers={num_answers_ckpt}")

    # build/load vocab (best: from ckpt; else from artifacts; else build from train and save)
    id2ans, ans2id = try_extract_vocab_from_ckpt(ckpt_obj)
    if id2ans is not None and ans2id is not None and len(id2ans) == num_answers_ckpt:
        print(f"[VOCAB] using vocab found inside checkpoint (size={len(id2ans)})")
    else:
        if os.path.exists(ID2ANS_PATH) and os.path.exists(ANS2ID_PATH):
            id2ans, ans2id = load_vocab(ID2ANS_PATH, ANS2ID_PATH)
            print(f"[VOCAB] loaded from artifacts: id2ans={len(id2ans)} ans2id={len(ans2id)} ({VOCAB_DIR})")
        else:
            # auto-align TOP_K to ckpt
            if ADD_UNK:
                top_k = max(1, num_answers_ckpt - 1)
            else:
                top_k = num_answers_ckpt

            id2ans, ans2id = build_vocab_from_train_meta(train_meta, top_k, ADD_UNK)
            save_vocab(id2ans, ans2id, VOCAB_DIR)
            print(f"[VOCAB] auto-built & saved: id2ans={len(id2ans)} ans2id={len(ans2id)} ({VOCAB_DIR})")

    # sanity
    if len(id2ans) != num_answers_ckpt:
        print(f"⚠️ [VOCAB] id2ans size {len(id2ans)} != ckpt num_answers {num_answers_ckpt}.")
        print("    Mapping may be off if your training used different normalization/top-K.")
        # still continue

    print(f"[SAMPLE] ONLY_IN_VOCAB={ONLY_IN_VOCAB} | EXCLUDE_UNK={EXCLUDE_UNK}")

    # build sampling pool
    def get_gt_ans(ex: Dict[str, Any]) -> str:
        gt = ex.get("answer_norm", ex.get("answer", ""))
        return norm_text(str(gt))

    pool = sample_meta
    if ONLY_IN_VOCAB:
        in_vocab_pool = []
        for ex in sample_meta:
            gt = get_gt_ans(ex)
            if not gt:
                continue
            if EXCLUDE_UNK and gt == UNK_TOKEN:
                continue
            if gt in ans2id:
                in_vocab_pool.append(ex)
        print(f"[SAMPLE] pool: {len(sample_meta)} -> in_vocab_pool: {len(in_vocab_pool)}")
        if len(in_vocab_pool) > 0:
            pool = in_vocab_pool
        else:
            print("⚠️ [SAMPLE] No in-vocab samples found; falling back to full pool.")

    # build model
    model = ResNetBertSingleHead(num_answers=num_answers_ckpt, head_style=head_style).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded ckpt: {CKPT_PATH}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    model.eval()

    # sample one example
    ex = rng.choice(pool)

    question = str(ex.get("question", ""))
    gt_ans_raw = ex.get("answer_norm", ex.get("answer", ""))
    gt_ans_raw = str(gt_ans_raw)
    gt_ans_norm = norm_text(gt_ans_raw)

    image_path = utils.resolve_image_path(DATA_ROOT, SAMPLE_SPLIT, ex)

    tokenizer = BertTokenizerFast.from_pretrained(BERT_NAME)
    tok = tokenizer(
        question,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_Q_LEN
    )

    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)

    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    pil = Image.open(image_path).convert("RGB")
    img_t = tfm(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_t, input_ids, attention_mask)
        pred_id = int(torch.argmax(logits, dim=1).item())

    pred_ans = id2ans[pred_id] if pred_id < len(id2ans) else f"<id:{pred_id}>"

    exact = 1.0 if norm_text(pred_ans) == gt_ans_norm else 0.0
    fuzz = fuzzy_sim(pred_ans, gt_ans_raw)

    print("[OK] image_path:", image_path)
    print("[OK] split:", SAMPLE_SPLIT)
    print("[OK] Q:", question)
    print("[OK] pred_id:", pred_id)
    print("[OK] pred:", pred_ans)
    print("[OK] gt  :", gt_ans_raw)
    print("[OK] exact_match:", exact)
    print("[OK] fuzzy_sim  :", round(fuzz, 4))

    plt.figure(figsize=(7.5, 7.5))
    plt.imshow(pil)
    plt.axis("off")
    plt.title(
        f"Q: {question}\nPred: {pred_ans} | GT: {gt_ans_raw}\nExact: {exact:.1f} | Fuzzy: {fuzz:.4f}",
        fontsize=11
    )
    if SAVE_FIG:
        plt.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
        print("[OK] saved:", OUT_FIG)
    plt.show()


if __name__ == "__main__":
    main()