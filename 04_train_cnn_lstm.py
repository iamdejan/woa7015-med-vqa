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


import utils

# =========================================================
# CONFIG (edit here; no argparse)
# =========================================================
SEED = 47

DATA_ROOT = "vqa-rad_extracted"
TRAIN_META = os.path.join(DATA_ROOT, "train", "meta.jsonl")
TEST_META  = os.path.join(DATA_ROOT, "test",  "meta.jsonl")

TRAIN_VAL_SPLIT = 0.80

# vocab sizes (keep reasonable; you can lower TOP_K_ANS to make baseline weaker)
MAX_Q_LEN = 20
TOP_K_ANS = 200
ADD_UNK_ANS = True

# model
IMG_SIZE = 224
Q_EMBED_DIM = 200
LSTM_HIDDEN = 256
IMG_PROJ_DIM = 256
FUSION_HIDDEN = 256

# training (simple)
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4
NUM_WORKERS = 0
PIN_MEMORY = True

# output
OUT_DIR = "Model_Results_VGG19_LSTM_Simple"
CKPT_DIR = os.path.join(OUT_DIR, "_checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
BEST_BY_VAL  = os.path.join(CKPT_DIR, "vgg19_lstm_simple_best.pt")
BEST_BY_TEST = os.path.join(CKPT_DIR, "vgg19_lstm_simple_best_by_test.pt")


# =========================================================
# Simple vocab (questions + answers)
# =========================================================
REGEX = re.compile(r"(\W+)")

def tokenize_question(text: str) -> List[str]:
    text = (text or "").lower()
    return [t.strip() for t in REGEX.split(text) if t.strip()]

class Vocab:
    def __init__(self, tokens: List[str]):
        self.vocab = tokens
        self.vocab2idx = {w: i for i, w in enumerate(tokens)}
        self.vocab_size = len(tokens)

    def word2idx(self, w: str) -> int:
        if w in self.vocab2idx:
            return self.vocab2idx[w]
        return self.vocab2idx.get("<unk>", 0)

    def idx2word(self, idx: int) -> str:
        if 0 <= idx < len(self.vocab):
            return self.vocab[idx]
        return "<unk>"

def build_question_vocab(train_rows: List[Dict[str, Any]], test_rows: List[Dict[str, Any]]) -> Vocab:
    toks: List[str] = []
    for ex in train_rows:
        toks.extend(tokenize_question(utils.pick_question(ex)))
    for ex in test_rows:
        toks.extend(tokenize_question(utils.pick_question(ex)))
    uniq = sorted(set(toks))
    uniq = ["<pad>", "<unk>"] + uniq
    return Vocab(uniq)

def build_answer_vocab(train_rows: List[Dict[str, Any]], top_k: int, add_unk: bool) -> Vocab:
    counter = Counter()
    for ex in train_rows:
        a = utils.normalize_text(utils.pick_answer(ex))
        if not a:
            continue
        counter[a] += 1

    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab: List[str] = []
    if add_unk:
        vocab.append("<unk>")
    # ensure yes/no exist (helps closed acc)
    for tok in ["yes", "no"]:
        if tok in counter and tok not in vocab:
            vocab.append(tok)

    for w, _ in items:
        if w in vocab:
            continue
        vocab.append(w)
        if len(vocab) >= (top_k + (1 if add_unk else 0)):
            break
    return Vocab(vocab)

def question_to_tensor(q: str, q_vocab: Vocab, max_len: int) -> torch.Tensor:
    toks = tokenize_question(q)
    idxs = [q_vocab.word2idx(t) for t in toks][:max_len]
    pad = q_vocab.word2idx("<pad>")
    if len(idxs) < max_len:
        idxs += [pad] * (max_len - len(idxs))
    return torch.tensor(idxs, dtype=torch.long)


# =========================================================
# Dataset
# =========================================================
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

class VQARadExtracted(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], split_name: str, q_vocab: Vocab, a_vocab: Vocab, train_mode: bool):
        self.rows = rows
        self.split_name = split_name
        self.q_vocab = q_vocab
        self.a_vocab = a_vocab
        self.tf = train_tf if train_mode else eval_tf

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.rows[idx]
        q = utils.pick_question(ex)
        a_raw = utils.normalize_text(utils.pick_answer(ex))

        img_path = utils.resolve_image_path(DATA_ROOT, self.split_name, ex)
        img = Image.open(img_path).convert("RGB")
        img_t = self.tf(img)

        q_t = question_to_tensor(q, self.q_vocab, MAX_Q_LEN)

        # map answer to vocab (if not in vocab -> <unk>)
        if a_raw in self.a_vocab.vocab2idx:
            a_mapped = a_raw
            in_vocab = True
        else:
            a_mapped = "<unk>" if "<unk>" in self.a_vocab.vocab2idx else a_raw
            in_vocab = False

        a_idx = self.a_vocab.word2idx(a_mapped)

        # IMPORTANT: yes/no decision should be based on RAW answer, not mapped answer
        yesno_flag = 1 if utils.is_yesno(a_raw) else 0
        open_in_vocab = 1 if (yesno_flag == 0 and in_vocab) else 0

        return {
            "image": img_t,
            "question": q_t,
            "answer_idx": torch.tensor(a_idx, dtype=torch.long),
            "gt_text": a_mapped,
            "is_yesno": torch.tensor(yesno_flag, dtype=torch.long),
            "open_in_vocab": torch.tensor(open_in_vocab, dtype=torch.long),
        }

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "question": torch.stack([b["question"] for b in batch], dim=0),
        "answer_idx": torch.stack([b["answer_idx"] for b in batch], dim=0),
        "gt_text": [b["gt_text"] for b in batch],
        "is_yesno": torch.stack([b["is_yesno"] for b in batch], dim=0),
        "open_in_vocab": torch.stack([b["open_in_vocab"] for b in batch], dim=0),
    }


# =========================================================
# Model: VGG-19 + LSTM (single head)
# =========================================================
class VGG19Encoder(nn.Module):
    def __init__(self, proj_dim: int):
        super().__init__()
        try:
            weights = tv_models.VGG19_Weights.IMAGENET1K_V1
            m = tv_models.vgg19(weights=weights)
        except Exception:
            m = tv_models.vgg19(pretrained=True)

        self.features = m.features
        # keep it simple: freeze CNN feature extractor (fast baseline)
        for p in self.features.parameters():
            p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f = self.features(x)
            f = self.pool(f).flatten(1)
        return self.proj(f)

class QuestionLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=1, batch_first=True)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        x = self.emb(q)
        _, (h, _) = self.lstm(x)
        return h[-1]

class VGG19LSTMClassifier(nn.Module):
    def __init__(self, q_vocab_size: int, ans_vocab_size: int):
        super().__init__()
        self.img = VGG19Encoder(proj_dim=IMG_PROJ_DIM)
        self.txt = QuestionLSTM(q_vocab_size, Q_EMBED_DIM, LSTM_HIDDEN)
        self.fc1 = nn.Linear(IMG_PROJ_DIM + LSTM_HIDDEN, FUSION_HIDDEN)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(FUSION_HIDDEN, ans_vocab_size)

    def forward(self, image: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        vi = self.img(image)
        vt = self.txt(question)
        z = torch.cat([vi, vt], dim=1)
        z = self.act(self.fc1(z))
        return self.fc2(z)


# =========================================================
# Eval + Train
# =========================================================
@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, a_vocab: Vocab, device: torch.device, criterion: nn.Module) -> Dict[str, float]:
    model.eval()

    total = correct = 0
    yesno_total = yesno_correct = 0
    open_total = open_correct = 0
    open_in_vocab_total = open_in_vocab_correct = 0
    loss_sum = 0.0

    preds_txt: List[str] = []
    refs_txt: List[str] = []

    for batch in loader:
        image = batch["image"].to(device)
        question = batch["question"].to(device)
        y = batch["answer_idx"].to(device)

        logits = model(image, question)
        loss = criterion(logits, y)

        bs = image.size(0)
        loss_sum += float(loss.item()) * bs

        pred = logits.argmax(dim=1)
        total += bs
        correct += int((pred == y).sum().item())

        is_yesno_t = batch["is_yesno"].to(device)
        open_in_vocab_t = batch["open_in_vocab"].to(device)

        mask_yes = (is_yesno_t == 1)
        if mask_yes.any():
            yesno_total += int(mask_yes.sum().item())
            yesno_correct += int((pred[mask_yes] == y[mask_yes]).sum().item())

        mask_open = (is_yesno_t == 0)
        if mask_open.any():
            open_total += int(mask_open.sum().item())
            open_correct += int((pred[mask_open] == y[mask_open]).sum().item())

        mask_open_in_vocab = (mask_open & (open_in_vocab_t == 1))
        if mask_open_in_vocab.any():
            open_in_vocab_total += int(mask_open_in_vocab.sum().item())
            open_in_vocab_correct += int((pred[mask_open_in_vocab] == y[mask_open_in_vocab]).sum().item())

        for i in range(bs):
            preds_txt.append(a_vocab.idx2word(int(pred[i].item())))
            refs_txt.append(batch["gt_text"][i])

    tm = utils.corpus_text_metrics(preds_txt, refs_txt)
    return {
        "loss": loss_sum / max(1, total),
        "acc": correct / max(1, total),
        "yesno_acc": yesno_correct / max(1, yesno_total),
        "open_acc": open_correct / max(1, open_total),
        "open_in_vocab_acc": open_in_vocab_correct / max(1, open_in_vocab_total),
        "yesno_n": float(yesno_total),
        "open_n": float(open_total),
        "open_in_vocab_n": float(open_in_vocab_total),
        "bleu": tm["bleu"],
        "rougeL": tm["rougeL"],
        "meteor": tm["meteor"],
        "tm_n": tm["n"],

        # counts for exact "(correct/total)" output
        "all_correct_count": float(correct),
        "all_total_count": float(total),
        "closed_correct_count": float(yesno_correct),
        "closed_total_count": float(yesno_total),
        "open_correct_count": float(open_correct),
        "open_total_count": float(open_total),
    }

def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Dict[str, float]:
    model.train()

    total = correct = 0
    loss_sum = 0.0
    step_times: List[float] = []

    log_every = max(1, len(loader) // 9)

    for step, batch in enumerate(loader, start=1):
        utils.sync_device(device)
        t0 = time.time()

        image = batch["image"].to(device)
        question = batch["question"].to(device)
        y = batch["answer_idx"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(image, question)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        pred = logits.argmax(dim=1)
        bs = image.size(0)
        total += bs
        correct += int((pred == y).sum().item())
        loss_sum += float(loss.item()) * bs

        utils.sync_device(device)
        t1 = time.time()
        step_times.append(t1 - t0)
        if len(step_times) > 50:
            step_times = step_times[-50:]
        avg_step = sum(step_times) / len(step_times)
        eta = (len(loader) - step) * avg_step

        if step % log_every == 0 or step == len(loader):
            print(f"[train] step {step:4d}/{len(loader)} | loss={loss.item():.4f} acc={(correct/max(1,total)):.4f} | ETA {utils.format_hms(eta)}")

    return {"loss": loss_sum / max(1, total), "acc": correct / max(1, total)}


# =========================================================
# Main
# =========================================================
def main():
    utils.set_seed(SEED)
    device = utils.get_device()

    print("Using device:", device)
    train_all = utils.load_jsonl(TRAIN_META)
    test_rows = utils.load_jsonl(TEST_META)
    print(f"Loaded {len(train_all)} train samples from {TRAIN_META}")
    print(f"Loaded {len(test_rows)} test  samples from {TEST_META}")

    # split train->train/val
    idxs = np.arange(len(train_all))
    rng = np.random.RandomState(SEED)
    rng.shuffle(idxs)
    cut = int(len(train_all) * TRAIN_VAL_SPLIT)
    train_rows = [train_all[i] for i in idxs[:cut]]
    val_rows   = [train_all[i] for i in idxs[cut:]]
    print(f"Split => train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    q_vocab = build_question_vocab(train_rows, test_rows)
    a_vocab = build_answer_vocab(train_rows, TOP_K_ANS, ADD_UNK_ANS)
    print(f"Question vocab size: {q_vocab.vocab_size}")
    print(f"Answer vocab size after top-{TOP_K_ANS} + UNK: {a_vocab.vocab_size}")
    print("Kept answers example:", a_vocab.vocab[: min(10, len(a_vocab.vocab))], "...")

    train_ds = VQARadExtracted(train_rows, "train", q_vocab, a_vocab, train_mode=True)
    val_ds   = VQARadExtracted(val_rows,   "train", q_vocab, a_vocab, train_mode=False)
    test_ds  = VQARadExtracted(test_rows,  "test",  q_vocab, a_vocab, train_mode=False)

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

    model = VGG19LSTMClassifier(q_vocab.vocab_size, a_vocab.vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # pre-train metrics
    pre_va = evaluate_model(model, val_loader, a_vocab, device, criterion)
    pre_te = evaluate_model(model, test_loader, a_vocab, device, criterion)
    print("\n==================== PRE-TRAIN METRICS ====================")
    print(f"Val : acc={pre_va['acc']:.4f} | BLEU={pre_va['bleu']:.4f} | ROUGE-L={pre_va['rougeL']:.4f} | METEOR={pre_va['meteor']:.4f} | n={int(pre_va['tm_n'])}")
    print(f"Test: acc={pre_te['acc']:.4f} | BLEU={pre_te['bleu']:.4f} | ROUGE-L={pre_te['rougeL']:.4f} | METEOR={pre_te['meteor']:.4f} | n={int(pre_te['tm_n'])}")
    print("===========================================================\n")

    best_val_acc = -1.0
    best_test_acc = -1.0
    epoch_times: List[float] = []

    for epoch in range(1, EPOCHS + 1):
        ep_start = time.time()

        print(f"\n======================== Epoch {epoch}/{EPOCHS} ========================")
        tr = train_one_epoch(model, train_loader, device, optimizer, criterion)
        va = evaluate_model(model, val_loader, a_vocab, device, criterion)
        te = evaluate_model(model, test_loader, a_vocab, device, criterion)

        ep_end = time.time()
        epoch_times.append(ep_end - ep_start)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        eta_epochs = avg_epoch * (EPOCHS - epoch)

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
        print(f"Epoch ETA: {utils.format_hms(eta_epochs)}")
        print("------------------------------------------------")

        if va["acc"] > best_val_acc:
            best_val_acc = float(va["acc"])
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": best_val_acc}, BEST_BY_VAL)
            print(f"✅ Saved BEST checkpoint (by val acc): {BEST_BY_VAL} (best_val_acc={best_val_acc:.4f})")

        if te["acc"] > best_test_acc:
            best_test_acc = float(te["acc"])
            torch.save({"model": model.state_dict(), "epoch": epoch, "test_acc": best_test_acc}, BEST_BY_TEST)
            print(f"⭐ Saved BEST checkpoint (by test acc): {BEST_BY_TEST} (best_test_acc={best_test_acc:.4f})")

    # load best-by-val for final reporting
    if os.path.exists(BEST_BY_VAL):
        ckpt = torch.load(BEST_BY_VAL, map_location=device)
        model.load_state_dict(ckpt["model"])

    post_va = evaluate_model(model, val_loader, a_vocab, device, criterion)
    post_te = evaluate_model(model, test_loader, a_vocab, device, criterion)
    print("\n==================== POST-TRAIN METRICS ====================")
    print(f"Val : acc={post_va['acc']:.4f} | BLEU={post_va['bleu']:.4f} | ROUGE-L={post_va['rougeL']:.4f} | METEOR={post_va['meteor']:.4f} | n={int(post_va['tm_n'])}")
    print(f"Test: acc={post_te['acc']:.4f} | BLEU={post_te['bleu']:.4f} | ROUGE-L={post_te['rougeL']:.4f} | METEOR={post_te['meteor']:.4f} | n={int(post_te['tm_n'])}")
    print("============================================================\n")

    utils.print_accuracy_results(post_te)

    print("Done.")
    print("Best checkpoint (val):", BEST_BY_VAL, f"(best_val_acc={best_val_acc:.4f})")
    print("Best checkpoint (test):", BEST_BY_TEST, f"(best_test_acc={best_test_acc:.4f})")


if __name__ == "__main__":
    main()