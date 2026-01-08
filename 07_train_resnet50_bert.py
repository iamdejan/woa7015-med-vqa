from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import json
import time
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models
from torchvision import transforms

# Common Functionalities
from typing import Any, Dict, List, Tuple, Optional
import utils

def build_open_vocab_for_training(train_rows: List[Dict[str, Any]], top_k: int, add_unk: bool) -> Tuple[Dict[str, int], Dict[int, str]]:
    counts: Dict[str, int] = {}
    for ex in train_rows:
        a = utils.normalize_text(utils.pick_answer(ex))
        if not a or utils.is_yesno(a):
            continue
        counts[a] = counts.get(a, 0) + 1

    sorted_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    kept = [w for w, _ in sorted_items[:top_k]]

    vocab = {w: i for i, w in enumerate(kept)}
    if add_unk:
        vocab["<unk>"] = len(vocab)

    inv = {i: w for w, i in vocab.items()}
    return vocab, inv


def coverage_stats(rows: List[Dict[str, Any]], open_vocab: Dict[str, int]) -> Dict[str, Any]:
    n = len(rows)
    yesno_n = 0
    open_n = 0
    open_in_vocab_n = 0

    for ex in rows:
        a = utils.normalize_text(utils.pick_answer(ex))
        if not a:
            continue
        if utils.is_yesno(a):
            yesno_n += 1
        else:
            open_n += 1
            if a in open_vocab:
                open_in_vocab_n += 1

    open_cov = (open_in_vocab_n / max(1, open_n))
    return {
        "n": n,
        "yesno_n": yesno_n,
        "open_n": open_n,
        "open_in_vocab_n": open_in_vocab_n,
        "open_coverage": open_cov,
    }


# =========================================================
# Dataset
# =========================================================
class VQARadExtractedTwoHeadDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        split_name: str,
        tokenizer: Any,
        open_vocab: Dict[str, int],
        max_len: int = 32,
        image_size: int = 224,
    ):
        self.rows = rows
        self.split_name = split_name
        self.tokenizer = tokenizer
        self.open_vocab = open_vocab
        self.max_len = max_len
        self.image_size = image_size

    def __len__(self):
        return len(self.rows)

    def _image_tf(self, img: Image.Image) -> torch.Tensor:
        img = img.resize((self.image_size, self.image_size))
        arr = np.asarray(img).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.rows[idx]
        q = utils.pick_question(ex)
        a = utils.normalize_text(utils.pick_answer(ex))

        img_path = utils.resolve_image_path(DATA_ROOT, self.split_name, ex)
        img = Image.open(img_path).convert("RGB")
        img_t = self._image_tf(img)

        enc = self.tokenizer(
            q,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn_mask = enc["attention_mask"].squeeze(0)

        yesno_label = -100
        open_label = -100

        if utils.is_yesno(a):
            yesno_label = 1 if a == "yes" else 0
        else:
            if a in self.open_vocab:
                open_label = self.open_vocab[a]
            else:
                if "<unk>" in self.open_vocab:
                    open_label = self.open_vocab["<unk>"]
                else:
                    open_label = -100

        return {
            "image": img_t,
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "yesno_label": torch.tensor(yesno_label, dtype=torch.long),
            "open_label": torch.tensor(open_label, dtype=torch.long),
            "is_yesno": torch.tensor(1 if utils.is_yesno(a) else 0, dtype=torch.long),
        }

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    out["image"] = torch.stack([b["image"] for b in batch], dim=0)
    out["input_ids"] = torch.stack([b["input_ids"] for b in batch], dim=0)
    out["attention_mask"] = torch.stack([b["attention_mask"] for b in batch], dim=0)
    out["yesno_label"] = torch.stack([b["yesno_label"] for b in batch], dim=0)
    out["open_label"] = torch.stack([b["open_label"] for b in batch], dim=0)
    out["is_yesno"] = torch.stack([b["is_yesno"] for b in batch], dim=0)
    return out

# ResNet-50 + BERT
from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import json
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel


import utils


# =========================================================
# CONFIG (edit here, no argparse)
# =========================================================
SEED = 47

DATA_ROOT = "vqa-rad_extracted"
TRAIN_META = os.path.join(DATA_ROOT, "train", "meta.jsonl")
TEST_META  = os.path.join(DATA_ROOT, "test",  "meta.jsonl")

TRAIN_VAL_SPLIT = 0.80

TOP_K_OPEN = 200
ADD_UNK_OPEN = True

IMG_BACKBONE = "resnet50"
TXT_BACKBONE = "bert-base-uncased"
MAX_TXT_LEN = 32

EPOCHS = 20
BATCH_SIZE = 16
LR = 2e-5
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

USE_AMP = True
NUM_WORKERS = 0
PIN_MEMORY = True

OUT_DIR = "Model_Results_TwoHead"
CKPT_DIR = os.path.join(OUT_DIR, "_checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

BEST_BY_VAL  = os.path.join(CKPT_DIR, "resnet50_bert_twohead_best.pt")
BEST_BY_TEST = os.path.join(CKPT_DIR, "resnet50_bert_twohead_best_by_test.pt")

FREEZE_IMAGE_BACKBONE_EPOCHS = 2
FREEZE_TEXT_BACKBONE_EPOCHS  = 2


# =========================================================
# Model
# =========================================================
class ResNet50ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        import torchvision.models as models
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # [B, 2048, 1, 1]
        self.proj = nn.Linear(2048, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x).flatten(1)  # [B, 2048]
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

def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


# =========================================================
# Eval (acc/loss) + Text metrics
# =========================================================
@torch.no_grad()
def evaluate_model(
    model,
    loader,
    yesno_crit,
    open_crit,
    open_vocab: Dict[str, int],
    open_inv: Dict[int, str],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    total = 0
    correct = 0

    yesno_total = 0
    yesno_correct = 0

    open_total = 0
    open_correct = 0

    open_in_vocab_total = 0
    open_in_vocab_correct = 0

    loss_sum = 0.0
    loss_count = 0

    unk_id = open_vocab.get("<unk>", None)

    preds_txt: List[str] = []
    refs_txt: List[str] = []

    for batch in loader:
        image = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        yesno_label = batch["yesno_label"].to(device)
        open_label = batch["open_label"].to(device)
        is_yesno = batch["is_yesno"].to(device)

        yesno_logits, open_logits = model(image, input_ids, attention_mask)

        bs = image.size(0)

        # loss (per-sample correct head)
        for i in range(bs):
            if is_yesno[i].item() == 1:
                if yesno_label[i].item() != -100:
                    li = F.cross_entropy(yesno_logits[i:i+1], yesno_label[i:i+1], reduction="mean")
                    loss_sum += float(li.item())
                    loss_count += 1
            else:
                if open_label[i].item() != -100:
                    li = F.cross_entropy(open_logits[i:i+1], open_label[i:i+1], reduction="mean")
                    loss_sum += float(li.item())
                    loss_count += 1

        yesno_pred = yesno_logits.argmax(dim=1)
        open_pred  = open_logits.argmax(dim=1)

        total += bs

        for i in range(bs):
            if is_yesno[i].item() == 1:
                if yesno_label[i].item() != -100:
                    if yesno_pred[i].item() == yesno_label[i].item():
                        correct += 1
                    pred_s = "yes" if int(yesno_pred[i].item()) == 1 else "no"
                    ref_s  = "yes" if int(yesno_label[i].item()) == 1 else "no"
                    preds_txt.append(pred_s)
                    refs_txt.append(ref_s)
            else:
                if open_label[i].item() != -100:
                    if open_pred[i].item() == open_label[i].item():
                        correct += 1
                    pred_id = int(open_pred[i].item())
                    ref_id  = int(open_label[i].item())
                    pred_s = open_inv.get(pred_id, "<unk>")
                    ref_s  = open_inv.get(ref_id, "<unk>")
                    preds_txt.append(pred_s)
                    refs_txt.append(ref_s)

        # closed acc
        mask_yes = (yesno_label != -100)
        if mask_yes.any():
            yesno_total += int(mask_yes.sum().item())
            yesno_correct += int((yesno_pred[mask_yes] == yesno_label[mask_yes]).sum().item())

        # open acc
        mask_open = (open_label != -100)
        if mask_open.any():
            open_total += int(mask_open.sum().item())
            open_correct += int((open_pred[mask_open] == open_label[mask_open]).sum().item())

            if unk_id is not None:
                mask_in_vocab = mask_open & (open_label != unk_id)
            else:
                mask_in_vocab = mask_open

            open_in_vocab_total += int(mask_in_vocab.sum().item())
            if mask_in_vocab.any():
                open_in_vocab_correct += int((open_pred[mask_in_vocab] == open_label[mask_in_vocab]).sum().item())

    avg_loss = loss_sum / max(1, loss_count)
    tm = utils.corpus_text_metrics(preds_txt, refs_txt)

    # ✅ 加了三类 (correct/total) 计数，用来按你截图格式输出
    return {
        "loss": avg_loss,
        "acc": (correct / max(1, total)),
        "yesno_acc": (yesno_correct / max(1, yesno_total)),
        "open_acc": (open_correct / max(1, open_total)),
        "open_in_vocab_acc": (open_in_vocab_correct / max(1, open_in_vocab_total)),
        "yesno_n": float(yesno_total),
        "open_n": float(open_total),
        "open_in_vocab_n": float(open_in_vocab_total),
        "bleu": tm["bleu"],
        "rougeL": tm["rougeL"],
        "meteor": tm["meteor"],
        "tm_n": tm["n"],

        "all_correct_count": float(correct),
        "all_total_count": float(total),
        "closed_correct_count": float(yesno_correct),
        "closed_total_count": float(yesno_total),
        "open_correct_count": float(open_correct),
        "open_total_count": float(open_total),
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    yesno_crit,
    open_crit,
    device: torch.device,
    log_every: int = 10,
) -> Dict[str, float]:
    model.train()

    total = 0
    correct = 0
    running_loss = 0.0
    step_times: List[float] = []

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        st = time.time()

        image = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        yesno_label = batch["yesno_label"].to(device, non_blocking=True)
        open_label = batch["open_label"].to(device, non_blocking=True)
        is_yesno = batch["is_yesno"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=(USE_AMP and device.type == "cuda")):
            yesno_logits, open_logits = model(image, input_ids, attention_mask)
            loss_yesno = yesno_crit(yesno_logits, yesno_label)
            loss_open  = open_crit(open_logits, open_label)
            loss = loss_yesno + loss_open

        scaler.scale(loss).backward()

        if GRAD_CLIP is not None and GRAD_CLIP > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

        yesno_pred = yesno_logits.argmax(dim=1)
        open_pred  = open_logits.argmax(dim=1)
        bs = image.size(0)
        total += bs

        for i in range(bs):
            if is_yesno[i].item() == 1:
                if yesno_label[i].item() != -100 and yesno_pred[i].item() == yesno_label[i].item():
                    correct += 1
            else:
                if open_label[i].item() != -100 and open_pred[i].item() == open_label[i].item():
                    correct += 1

        running_loss += float(loss.item()) * bs

        et = time.time()
        step_times.append(et - st)
        if len(step_times) > 50:
            step_times = step_times[-50:]
        avg_step = sum(step_times) / max(1, len(step_times))
        remaining_steps = len(loader) - step
        eta = remaining_steps * avg_step

        if step % log_every == 0 or step == len(loader):
            print(
                f"[train] step {step:4d}/{len(loader)} | "
                f"loss={loss.item():.4f} acc={(correct/max(1,total)):.4f} | ETA {utils.format_hms(eta)}"
            )

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return {"loss": epoch_loss, "acc": epoch_acc}


def main():
    utils.set_seed(SEED)
    device = utils.get_device()
    torch.backends.cudnn.benchmark = (device.type == "cuda")

    print("Using device:", device)
    train_all = utils.load_jsonl(TRAIN_META)
    test_rows = utils.load_jsonl(TEST_META)
    print(f"Loaded {len(train_all)} train samples from {TRAIN_META}")
    print(f"Loaded {len(test_rows)} test  samples from {TEST_META}")

    rng = np.random.RandomState(SEED)
    idxs = np.arange(len(train_all))
    rng.shuffle(idxs)
    cut = int(len(train_all) * TRAIN_VAL_SPLIT)
    train_rows = [train_all[i] for i in idxs[:cut]]
    val_rows   = [train_all[i] for i in idxs[cut:]]

    print(f"Split => train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(TXT_BACKBONE)

    open_vocab, open_inv = build_open_vocab_for_training(train_rows, top_k=TOP_K_OPEN, add_unk=ADD_UNK_OPEN)
    print(f"YES/NO vocab size: 2 | OPEN vocab size: {len(open_vocab)} (TOP_K_OPEN={TOP_K_OPEN}, ADD_UNK_OPEN={ADD_UNK_OPEN})")

    print("[train] coverage:", coverage_stats(train_rows, open_vocab))
    print("[val]   coverage:", coverage_stats(val_rows, open_vocab))
    print("[test]  coverage:", coverage_stats(test_rows, open_vocab))

    train_ds = VQARadExtractedTwoHeadDataset(train_rows, "train", tokenizer, open_vocab, max_len=MAX_TXT_LEN)
    val_ds   = VQARadExtractedTwoHeadDataset(val_rows,   "train", tokenizer, open_vocab, max_len=MAX_TXT_LEN)
    test_ds  = VQARadExtractedTwoHeadDataset(test_rows,  "test",  tokenizer, open_vocab, max_len=MAX_TXT_LEN)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(PIN_MEMORY and device.type == "cuda"),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(PIN_MEMORY and device.type == "cuda"),
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(PIN_MEMORY and device.type == "cuda"),
        collate_fn=collate_fn,
    )

    model = TwoHeadVQAModel(open_vocab_size=len(open_vocab)).to(device)

    yesno_crit = nn.CrossEntropyLoss(ignore_index=-100)
    open_crit  = nn.CrossEntropyLoss(ignore_index=-100)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = EPOCHS * max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device.type == "cuda"))

    pre_va = evaluate_model(model, val_loader, yesno_crit, open_crit, open_vocab, open_inv, device)
    pre_te = evaluate_model(model, test_loader, yesno_crit, open_crit, open_vocab, open_inv, device)
    print("\n==================== PRE-TRAIN METRICS ====================")
    print(f"Val : acc={pre_va['acc']:.4f} | BLEU={pre_va['bleu']:.4f} | ROUGE-L={pre_va['rougeL']:.4f} | METEOR={pre_va['meteor']:.4f} | n={int(pre_va['tm_n'])}")
    print(f"Test: acc={pre_te['acc']:.4f} | BLEU={pre_te['bleu']:.4f} | ROUGE-L={pre_te['rougeL']:.4f} | METEOR={pre_te['meteor']:.4f} | n={int(pre_te['tm_n'])}")
    print("===========================================================\n")

    best_val_acc = -1.0
    best_test_acc = -1.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n======================== Epoch {epoch}/{EPOCHS} ========================")

        set_requires_grad(model.img_enc, epoch > FREEZE_IMAGE_BACKBONE_EPOCHS)
        set_requires_grad(model.txt_backbone, epoch > FREEZE_TEXT_BACKBONE_EPOCHS)

        log_every = max(1, len(train_loader) // 9)

        tr = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, yesno_crit, open_crit, device, log_every=log_every)
        va = evaluate_model(model, val_loader, yesno_crit, open_crit, open_vocab, open_inv, device)
        te = evaluate_model(model, test_loader, yesno_crit, open_crit, open_vocab, open_inv, device)

        print("\n---------------- EPOCH SUMMARY ----------------")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"Train: loss={tr['loss']:.4f} | acc={tr['acc']:.4f}")
        print(
            "Val  : "
            f"loss={va['loss']:.4f} | acc={va['acc']:.4f} | "
            f"yesno_acc={va['yesno_acc']:.4f} | open_acc={va['open_acc']:.4f} | "
            f"open_in_vocab_acc={va['open_in_vocab_acc']:.4f} | "
            f"yesno_n={int(va['yesno_n'])} | open_n={int(va['open_n'])} | open_in_vocab_n={int(va['open_in_vocab_n'])}"
        )
        print(
            "Test : "
            f"loss={te['loss']:.4f} | acc={te['acc']:.4f} | "
            f"yesno_acc={te['yesno_acc']:.4f} | open_acc={te['open_acc']:.4f} | "
            f"open_in_vocab_acc={te['open_in_vocab_acc']:.4f} | "
            f"yesno_n={int(te['yesno_n'])} | open_n={int(te['open_n'])} | open_in_vocab_n={int(te['open_in_vocab_n'])}"
        )
        print("------------------------------------------------")

        if va["acc"] > best_val_acc:
            best_val_acc = va["acc"]
            torch.save({"model": model.state_dict(), "epoch": epoch}, BEST_BY_VAL)
            print(f"✅ Saved BEST checkpoint (by val acc): {BEST_BY_VAL} (best_val_acc={best_val_acc:.4f})")

        if te["acc"] > best_test_acc:
            best_test_acc = te["acc"]
            torch.save({"model": model.state_dict(), "epoch": epoch}, BEST_BY_TEST)
            print(f"⭐ Saved BEST checkpoint (by test acc): {BEST_BY_TEST} (best_test_acc={best_test_acc:.4f})")

    post_va = evaluate_model(model, val_loader, yesno_crit, open_crit, open_vocab, open_inv, device)
    post_te = evaluate_model(model, test_loader, yesno_crit, open_crit, open_vocab, open_inv, device)
    print("\n==================== POST-TRAIN METRICS ====================")
    print(f"Val : acc={post_va['acc']:.4f} | BLEU={post_va['bleu']:.4f} | ROUGE-L={post_va['rougeL']:.4f} | METEOR={post_va['meteor']:.4f} | n={int(post_va['tm_n'])}")
    print(f"Test: acc={post_te['acc']:.4f} | BLEU={post_te['bleu']:.4f} | ROUGE-L={post_te['rougeL']:.4f} | METEOR={post_te['meteor']:.4f} | n={int(post_te['tm_n'])}")
    print("============================================================\n")

    # ✅ 你要求的截图风格 ACC 输出
    utils.print_accuracy_results(post_te)

    print("Done.")
    print("Best checkpoint (val):", BEST_BY_VAL, f"(best_val_acc={best_val_acc:.4f})")
    print("Best checkpoint (test):", BEST_BY_TEST, f"(best_test_acc={best_test_acc:.4f})")


if __name__ == "__main__":
    main()