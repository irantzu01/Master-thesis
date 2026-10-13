import json
import random
from datasets import load_dataset

print("[INFO] Initializing streaming data pipelines...")

# Stream both datasets to avoid blowing up cluster memory
basque_stream = load_dataset("uonlp/CulturaX", "eu", split="train", streaming=True)
spanish_stream = load_dataset("uonlp/CulturaX", "es", split="train", streaming=True)

output_file = "/user/i.elcoroalberdi/u25035/MASTER/thesis/TokAlign/data/pretrain-corpus/pile00.json"

# --- 80/20 Basque/Spanish proportion ---
TOTAL_SAMPLES = 150000          # total rows in the final corpus (same scale as before)
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

# Shuffle so language isn't strictly alternating/blocked -- gives a natural mixed corpus
# while still respecting the 80/20 overall proportion.
random.shuffle(rows)

with open(output_file, "w", encoding="utf-8") as f:
    for text in rows:
        f.write(json.dumps({"text": text}) + "\n")

print(f"[SUCCESS] Extraction complete! Wrote {len(rows)} lines "
      f"({eu_count} Basque / {es_count} Spanish, ~{BASQUE_RATIO:.0%}/{1-BASQUE_RATIO:.0%}) "
      f"to {output_file}")