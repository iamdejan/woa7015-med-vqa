from collections import Counter
from statistics import mean, median
from datasets import load_dataset
import numpy as np

# ===== 1. Load dataset =====
print("Loading dataset from HuggingFace...")
ds = load_dataset("flaviagiammarino/vqa-rad", cache_dir="./cache")

print("\n=== Dataset overview ===")
print(ds) # Displays information such as train/test.
print("Splits:", list(ds.keys()))
for split in ds.keys():
    print(f"{split}: {len(ds[split])} examples")

print("\nColumns in train split:", ds["train"].column_names)

# View sample to check the contents of the dataset.
print("\nExample sample from train:")
print(ds["train"][0])

# ===== 2. Answer distribution (train split) =====
train = ds["train"]

answers = [ex["answer"] for ex in train]
answer_counts = Counter(answers)

print("\n=== Answer statistics (train split) ===")
print("Number of QA pairs:", len(answers))
print("Number of unique answers (answer vocabulary size):", len(answer_counts))

print("\nTop 20 most frequent answers:")
for ans, cnt in answer_counts.most_common(20):
    print(f"{ans!r:>20} : {cnt}")

# yes / no vs other
def is_yes_no(ans: str) -> bool:
    a = ans.strip().lower()
    return a in {"yes", "no"}

yn_flags = [is_yes_no(a) for a in answers]
num_yes_no = sum(yn_flags)
num_other = len(answers) - num_yes_no

print("\nYes/No vs other answers (train split):")
print(f"Yes/No answers : {num_yes_no} ({num_yes_no / len(answers):.2%})")
print(f"Other answers  : {num_other} ({num_other / len(answers):.2%})")

# ===== 3. Question length statistics =====
questions = [ex["question"] for ex in train]

def question_length(q: str) -> int:
    # Simply using spaces to segment words is sufficient; you can modify it further if you develop a tokenizer later.
    return len(q.strip().split())

q_lengths = [question_length(q) for q in questions]

print("\n=== Question length statistics (train split) ===")
print("Min length :", min(q_lengths))
print("Max length :", max(q_lengths))
print("Mean       :", mean(q_lengths))
print("Median     :", median(q_lengths))

# Print the percentile
for q in [0.25, 0.5, 0.75, 0.9, 0.95]:
    val = float(np.quantile(q_lengths, q))
    print(f"{int(q*100):>2}th percentile: {val}")

# ===== 4. Show a few random examples =====
import random
print("\n=== Random examples from train split ===")
for i in range(5):
    idx = random.randint(0, len(train) - 1)
    ex = train[idx]
    print(f"[{idx}] Q:", ex["question"])
    print("     A:", ex["answer"])
    print("-" * 60)

print("\nEDA complete.")