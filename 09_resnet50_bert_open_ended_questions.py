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
    return SequenceMatcher(None, a, b).ratio()


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

# ResNet50 + BERT (open-ended questions)
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


# =========================
# CONFIG (edit here)
# =========================
SEED = 39

DATA_ROOT = "vqa-rad_extracted"
TRAIN_META = os.path.join(DATA_ROOT, "train", "meta.jsonl")
TEST_META  = os.path.join(DATA_ROOT, "test",  "meta.jsonl")

TRAIN_VAL_SPLIT = 0.80

TOP_K_OPEN = 200
ADD_UNK_OPEN = True

TXT_BACKBONE = "bert-base-uncased"
MAX_TXT_LEN = 32
IMAGE_SIZE = 224

# which split to sample from: "train" | "val" | "test"
SAMPLE_SPLIT = "test"

# only sample open-ended whose answer is in open_vocab and not <unk>
ONLY_IN_VOCAB = True
EXCLUDE_UNK = True

# checkpoint
OUT_DIR = "Model_Results_TwoHead"
CKPT_PATH = os.path.join(OUT_DIR, "_checkpoints", "resnet50_bert_twohead_best.pt")

# ---- saving visualization ----
SAVE_FIG = True
SAVE_DIR = os.path.join(OUT_DIR, "_viz")
DPI = 200


# =========================
# Quiet warnings
# =========================
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()


# =========================
# Utils
# =========================
def show_vqa_result(
    img_pil: Image.Image,
    q: str,
    pred: str,
    gt: str,
    exact: float,
    fuzzy: float,
    save_dir: str,
    dpi: int = 200,
    show: bool = True,
) -> str:
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(7.5, 7.5), dpi=dpi)
    plt.imshow(img_pil)
    plt.axis("off")
    title = (
        f"Q: {q}\n"
        f"Pred: {pred} | GT: {gt}\n"
        f"Exact: {exact:.1f} | Fuzzy: {fuzzy:.4f}"
    )
    plt.title(title, fontsize=12, pad=12)
    plt.tight_layout()

    base = _safe_filename(q)
    fname = f"{base}__pred_{_safe_filename(pred)}__gt_{_safe_filename(gt)}.png"
    out_path = os.path.join(save_dir, fname)

    if os.path.exists(out_path):
        k = 2
        stem, ext = os.path.splitext(out_path)
        while os.path.exists(f"{stem}__{k}{ext}"):
            k += 1
        out_path = f"{stem}__{k}{ext}"

    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
    if show:
        plt.show()
    else:
        plt.close()

    return out_path


# =========================
# Model
# =========================
class ResNet50ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        import torchvision.models as models
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.proj = nn.Linear(2048, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x).flatten(1)
        return self.proj(feat)

class TwoHeadVQAModel(nn.Module):
    def __init__(self, open_vocab_size: int, fusion_dim: int = 512):
        super().__init__()
        self.img_enc = ResNet50ImageEncoder(out_dim=fusion_dim)
        self.txt_backbone = AutoModel.from_pretrained(TXT_BACKBONE)
        txt_dim = self.txt_backbone.config.hidden_size

        self.fuse = nn.Sequential(
            nn.Linear(fusion_dim + txt_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.yesno_head = nn.Linear(fusion_dim, 2)
        self.open_head  = nn.Linear(fusion_dim, open_vocab_size)

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.img_enc(image)
        txt_out = self.txt_backbone(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = txt_out.last_hidden_state[:, 0]
        z = torch.cat([img_feat, txt_feat], dim=1)
        z = self.fuse(z)
        return self.yesno_head(z), self.open_head(z)


# =========================
# One-shot pipeline
# =========================
@torch.no_grad()
def main():
    utils.set_seed(SEED)
    device = utils.get_device()
    print("Using device:", device)

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    train_all = utils.load_jsonl(TRAIN_META)
    test_rows = utils.load_jsonl(TEST_META)
    train_rows, val_rows = split_train_val(train_all, SEED, TRAIN_VAL_SPLIT)

    open_ans2id, open_id2ans, open_counts = build_open_vocab_for_testing(train_rows, TOP_K_OPEN, ADD_UNK_OPEN)
    unk_id = open_ans2id.get("<unk>", None)

    top10 = sorted(open_counts.items(), key=lambda kv: -kv[1])[:10]
    print(f"[OPEN VOCAB] size={len(open_id2ans)} | TOP_K_OPEN={TOP_K_OPEN} | ADD_UNK_OPEN={ADD_UNK_OPEN}")
    print("[OPEN TOP-10 freq]", top10)

    if SAMPLE_SPLIT == "train":
        rows, split_name = train_rows, "train"
    elif SAMPLE_SPLIT == "val":
        rows, split_name = val_rows, "train"
    else:
        rows, split_name = test_rows, "test"

    pool = []
    for ex in rows:
        gt = utils.normalize_text(utils.pick_answer(ex))
        if (not gt) or utils.is_yesno(gt):
            continue
        if ONLY_IN_VOCAB:
            if gt not in open_ans2id:
                continue
            if EXCLUDE_UNK and (unk_id is not None) and (open_ans2id[gt] == unk_id):
                continue
        pool.append(ex)

    if not pool:
        raise RuntimeError("No open-ended samples matched your filter. Try ONLY_IN_VOCAB=False or EXCLUDE_UNK=False.")

    ex = random.choice(pool)
    q = utils.pick_question(ex)
    gt = utils.normalize_text(utils.pick_answer(ex))

    img_path = utils.resolve_image_path(DATA_ROOT, split_name, ex)
    img = Image.open(img_path).convert("RGB")
    img_t = image_to_tensor_train_style(img).unsqueeze(0).to(device)

    tokenizer = AutoTokenizer.from_pretrained(TXT_BACKBONE)
    enc = tokenizer(q, padding="max_length", truncation=True, max_length=MAX_TXT_LEN, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    model = TwoHeadVQAModel(open_vocab_size=len(open_id2ans)).to(device)
    ckpt = safe_torch_load(CKPT_PATH)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    _, open_logits = model(img_t, input_ids, attn_mask)
    probs = torch.softmax(open_logits, dim=1)[0]
    topv, topi = torch.topk(probs, k=min(5, probs.numel()))
    top5 = [(int(i), open_id2ans[int(i)], float(v)) for v, i in zip(topv, topi)]

    pred_id = int(topi[0])
    pred = open_id2ans[pred_id]

    exact = 1.0 if pred.strip().lower() == gt.strip().lower() else 0.0
    fuzzy = fuzzy_score(pred, gt)

    print("\n[OPEN-ENDED RANDOM TEST]")
    print("split:", SAMPLE_SPLIT, "| image_path:", img_path.as_posix())
    print("Q:", q)
    print("GT:", gt)
    print("Pred:", pred, "| pred_id:", pred_id)
    print("Top-5:", top5)

    if SAVE_FIG:
        out_path = show_vqa_result(
            img, q, pred, gt, exact, fuzzy,
            save_dir=SAVE_DIR,
            dpi=DPI,
            show=True,   # 只想保存不展示 -> False
        )
        print("Saved figure:", out_path)

if __name__ == "__main__":
    main()