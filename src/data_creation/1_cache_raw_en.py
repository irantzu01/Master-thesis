"""Step 1 (English): cache the raw source text for the curriculum.

Interleaves English Wikipedia (20231101) and CulturaX 50/50 (seed 42) and writes
TOTAL_ROWS lines to <DATA_ROOT>/raw_english_master.txt.

CulturaX is gated: log in first with `huggingface-cli login` (or set HF_TOKEN).
"""
import os

from datasets import load_dataset, interleave_datasets

from data_paths import DATA_ROOT, TOTAL_ROWS

RAW_EN_FILE = os.path.join(DATA_ROOT, "raw_english_master.txt")
os.makedirs(DATA_ROOT, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN") or None  # None: use the cached login

print("[INFO] Connecting to English streaming assets...")
wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
cultura = load_dataset("uonlp/CulturaX", "en", split="train", streaming=True, token=HF_TOKEN)
master_stream = interleave_datasets([wiki, cultura], probabilities=[0.5, 0.5], seed=42)

count = 0

with open(RAW_EN_FILE, "w", encoding="utf-8") as f:
    for sample in master_stream:
        text = sample.get("text", "").replace("\n", " ").strip()
        if text:
            f.write(text + "\n")
            count += 1
            if count % 200000 == 0:
                print(f"    Cached {count} / {TOTAL_ROWS} rows...")
            if count >= TOTAL_ROWS:
                break

print(f"[SUCCESS] English raw cache ready: {RAW_EN_FILE}")
