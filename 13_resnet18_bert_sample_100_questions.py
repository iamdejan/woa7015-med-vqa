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

# ResNet50 + BERT (sample 100 questions)
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

from transformers import AutoTokenizer, AutoModel
from transformers.utils import logging as hf_logging


import utils


# =========================
# CONFIG (edit here)
# =========================
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

# checkpoint (CHANGE THIS to your resnet18 checkpoint)
OUT_DIR = "Model_Results_TwoHead"
CKPT_PATH = os.path.join(OUT_DIR, "_checkpoints", "resnet18_bert_twohead_best_by_test.pt")

# seed sweep
SEED_START = 1
SEED_END = 100


# =========================
# Quiet warnings
# =========================
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()


# =========================
# Model
# =========================
class ResNet18ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        import torchvision.models as models
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # [B, 512, 1, 1]
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x).flatten(1)  # [B, 512]
        return self.proj(feat)              # [B, out_dim]

class TwoHeadVQAModel(nn.Module):
    def __init__(self, open_vocab_size: int, fusion_dim: int = 512):
        super().__init__()
        self.img_enc = ResNet18ImageEncoder(out_dim=fusion_dim)
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
# Seed sweep
# =========================
@torch.no_grad()
def main():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    if not os.path.exists(TRAIN_META):
        raise FileNotFoundError(f"Train meta not found: {TRAIN_META}")
    if not os.path.exists(TEST_META):
        raise FileNotFoundError(f"Test meta not found: {TEST_META}")

    device = utils.get_device()

    train_all = utils.load_jsonl(TRAIN_META)
    test_rows = utils.load_jsonl(TEST_META)

    tokenizer = AutoTokenizer.from_pretrained(TXT_BACKBONE)

    # head size must match training
    open_vocab_size_fixed = TOP_K_OPEN + (1 if ADD_UNK_OPEN else 0)

    model = TwoHeadVQAModel(open_vocab_size=open_vocab_size_fixed).to(device)
    ckpt = safe_torch_load(CKPT_PATH)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    correct_seeds: List[int] = []
    wrong_seeds: List[int] = []

    for seed in range(SEED_START, SEED_END + 1):
        utils.set_seed(seed)

        train_rows, val_rows = split_train_val(train_all, seed, TRAIN_VAL_SPLIT)

        open_ans2id, open_id2ans, _ = build_open_vocab_for_testing(train_rows, TOP_K_OPEN, ADD_UNK_OPEN)
        unk_id = open_ans2id.get("<unk>", None)

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
            wrong_seeds.append(seed)
            continue

        ex = random.choice(pool)
        q = utils.pick_question(ex)
        gt = utils.normalize_text(utils.pick_answer(ex))

        img_path = utils.resolve_image_path(DATA_ROOT, split_name, ex)
        img = Image.open(img_path).convert("RGB")
        img_t = image_to_tensor_train_style(img).unsqueeze(0).to(device)

        enc = tokenizer(
            q,
            padding="max_length",
            truncation=True,
            max_length=MAX_TXT_LEN,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        _, open_logits = model(img_t, input_ids, attn_mask)
        pred_id = int(open_logits.argmax(dim=1).item())

        if pred_id < 0 or pred_id >= len(open_id2ans) or open_id2ans[pred_id] is None:
            pred = "<unk>"
        else:
            pred = open_id2ans[pred_id]

        if pred.strip().lower() == gt.strip().lower():
            correct_seeds.append(seed)
        else:
            wrong_seeds.append(seed)

    print("CORRECT_SEEDS:", correct_seeds)
    print("WRONG_SEEDS:", wrong_seeds)


if __name__ == "__main__":
    main()