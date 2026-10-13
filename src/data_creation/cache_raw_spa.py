import os
from datasets import load_dataset, interleave_datasets

OUT_DIR = "/user/i.elcoroalberdi/u25035/MASTER/thesis/data"
RAW_ES_FILE = os.path.join(OUT_DIR, "raw_spanish_master.txt")
os.makedirs(OUT_DIR, exist_ok=True)

print("[INFO] Connecting to streaming data assets...")
wiki = load_dataset("wikimedia/wikipedia", "20231101.es", split="train", streaming=True)
cultura = load_dataset("uonlp/CulturaX", "es", split="train", streaming=True)
master_stream = interleave_datasets([wiki, cultura], probabilities=[0.5, 0.5], seed=42)

TOTAL_ROWS = 1374333
count = 0

print(f"[INFO] Caching {TOTAL_ROWS} raw rows to local disk...")
with open(RAW_ES_FILE, 'w', encoding='utf-8') as f:
    for sample in master_stream:
        text = sample.get('text', '').replace('\n', ' ').strip()
        if text:
            f.write(text + '\n')
            count += 1
            if count % 200000 == 0:
                print(f"    Cached {count} / {TOTAL_ROWS} rows...")
            if count >= TOTAL_ROWS:
                break

print(f"[SUCCESS] Monolingual raw cache created successfully at: {RAW_ES_FILE}")