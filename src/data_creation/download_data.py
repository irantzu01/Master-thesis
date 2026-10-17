"""Build the TokAlign stabilization corpus: 150,000 CulturaX samples, 80% Basque
(120,000) and 20% Spanish (30,000), shuffled with seed 0.

Output: <DATA_ROOT>/pretrain-corpus/pile00.json (one {"text": ...} per line).
Copy or symlink it to TokAlign/data/pretrain-corpus/pile00.json before running TokAlign.

CulturaX is gated: log in first with `huggingface-cli login` (or set HF_TOKEN).
"""
import json
import os
import random

from datasets import load_dataset

from data_paths import DATA_ROOT

HF_TOKEN = os.getenv("HF_TOKEN") or None  # None: use the cached login

print("[INFO] Initializing streaming data pipelines...")

# Stream both datasets to avoid blowing up cluster memory
basque_stream = load_dataset("uonlp/CulturaX", "eu", split="train", streaming=True, token=HF_TOKEN)
spanish_stream = load_dataset("uonlp/CulturaX", "es", split="train", streaming=True, token=HF_TOKEN)

output_dir = os.path.join(DATA_ROOT, "pretrain-corpus")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "pile00.json")

# --- 80/20 Basque/Spanish proportion ---
TOTAL_SAMPLES = 150000
BASQUE_RATIO = 0.8
basque_target = int(TOTAL_SAMPLES * BASQUE_RATIO)        # 120,000
spanish_target = TOTAL_SAMPLES - basque_target            # 30,000

SEED = 0
random.seed(SEED)

print(f"[INFO] Target counts -> Basque: {basque_target}, Spanish: {spanish_target}")
print(f"[INFO] Writing shuffled 80/20 corpus to: {output_file}")

basque_iter = iter(basque_stream)
spanish_iter = iter(spanish_stream)

rows = []

# Collect Basque samples
eu_count = 0
for sample in basque_iter:
    text = sample.get("text", "").strip()
    if not text:
        continue
    rows.append(text)
    eu_count += 1
    if eu_count >= basque_target:
        break

# Collect Spanish samples
es_count = 0
for sample in spanish_iter:
    text = sample.get("text", "").strip()
    if not text:
        continue
    rows.append(text)
    es_count += 1
    if es_count >= spanish_target:
        break

print(f"[INFO] Collected {eu_count} Basque rows and {es_count} Spanish rows.")

# Shuffle so that language is not blocked: natural mixed corpus, 80/20 overall.
random.shuffle(rows)

with open(output_file, "w", encoding="utf-8") as f:
    for text in rows:
        f.write(json.dumps({"text": text}) + "\n")

print(f"[SUCCESS] Extraction complete! Wrote {len(rows)} lines "
      f"({eu_count} Basque / {es_count} Spanish) to {output_file}")
