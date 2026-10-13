import os
from datasets import load_dataset, interleave_datasets

OUT_DIR = "/user/i.elcoroalberdi/u25035/MASTER/thesis/data"
RAW_EN_FILE = os.path.join(OUT_DIR, "raw_english_master.txt")
os.makedirs(OUT_DIR, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN")

print("[INFO] Connecting to English streaming assets...")
wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
cultura = load_dataset("uonlp/CulturaX", "en", split="train", streaming=True, token=HF_TOKEN)
master_stream = interleave_datasets([wiki, cultura], probabilities=[0.5, 0.5], seed=42)

TOTAL_ROWS = 1374333
count = 0

with open(RAW_EN_FILE, 'w', encoding='utf-8') as f:
    for sample in master_stream:
        # Wikipedia English structures use 'text', Culturax uses 'text'
        text = sample.get('text', '').replace('\n', ' ').strip()
        if text:
            f.write(text + '\n')
            count += 1
            if count % 200000 == 0:
                print(f"    Cached {count} / {TOTAL_ROWS} rows...")
            if count >= TOTAL_ROWS:
                break

print(f"[SUCCESS] English raw cache ready: {RAW_EN_FILE}")