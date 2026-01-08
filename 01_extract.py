import os
import json
from datasets import load_dataset


# =========================
# 0) USER CONFIG (edit me)
# =========================
EXPORT_DIR = "vqa-rad_extracted"
IMAGE_EXT = "png"

# 每个 split 导出多少条：None 表示全量导出
EXPORT_N_TRAIN = None   # e.g. 200 或 None
EXPORT_N_TEST  = None   # e.g. 200 或 None


# =========================
# 1) Load dataset
# =========================
ds = load_dataset("flaviagiammarino/vqa-rad", cache_dir="./cache")

# compatibility：DatasetDict or Dataset
if hasattr(ds, "keys") and callable(ds.keys):
    splits = list(ds.keys())
else:
    splits = ["data"]

print("✅ Loaded:", type(ds))
print("✅ Available splits:", splits)


# =========================
# 2) Export function
# =========================
def export_split(split_name: str, out_root: str, limit_n=None):
    if split_name == "data":
        d = ds
    else:
        if split_name not in ds:
            print(f"[SKIP] split not found: {split_name}")
            return
        d = ds[split_name]

    out_dir = os.path.join(out_root, split_name)
    os.makedirs(out_dir, exist_ok=True)

    meta_path = os.path.join(out_dir, "meta.jsonl")

    total = len(d) if (limit_n is None) else min(limit_n, len(d))
    print(f"➡️ Exporting split='{split_name}' total={total} -> {out_dir}")

    with open(meta_path, "w", encoding="utf-8") as f:
        for i in range(total):
            ex = d[i]

            img_path = None
            if "image" in ex and ex["image"] is not None:
                img = ex["image"]  # PIL image (usually)
                img_path = os.path.join(out_dir, f"{split_name}_{i:06d}.{IMAGE_EXT}")
                img.save(img_path)

            record = {
                "id": i,
                "split": split_name,
                "image_path": img_path,
                "question": ex.get("question"),
                "answer": ex.get("answer"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Done split='{split_name}'. meta file -> {meta_path}")


# =========================
# 3) Export train + test
# =========================
os.makedirs(EXPORT_DIR, exist_ok=True)

# Your data structure usually includes train/test.
if "train" in splits:
    export_split("train", EXPORT_DIR, EXPORT_N_TRAIN)
if "test" in splits:
    export_split("test", EXPORT_DIR, EXPORT_N_TEST)

# If your ds is not DatasetDict (no train/test), then export the data.
if splits == ["data"]:
    export_split("data", EXPORT_DIR, None)

print("\n🎉 All exports finished.")
print("Output root:", EXPORT_DIR)
print("Expect folders like:")
print("  -", os.path.join(EXPORT_DIR, "train"))
print("  -", os.path.join(EXPORT_DIR, "test"))