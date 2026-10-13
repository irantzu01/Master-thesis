import os
import sys
from datasets import load_dataset
import pandas 
from pandas import DataFrame

# --- Configuration Paths ---
OUTPUT_DIR = "/user/i.elcoroalberdi/u25035/MASTER/thesis/data/synthetic_spanish_adj_cs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PHASE_2_OUT = os.path.join(OUTPUT_DIR, "phase_2_sentence_level.txt")

# Parallel subtitle corpora paths (Adjust these to match your exact filenames on disk)
SUB_ES_PATH = "/user/i.elcoroalberdi/u25035/MASTER/thesis/data/es-eu.txt/OpenSubtitles.es-eu.es"
SUB_EU_PATH = "/user/i.elcoroalberdi/u25035/MASTER/thesis/data/es-eu.txt/OpenSubtitles.es-eu.eu"

TOTAL_BUDGET = 1374333
line_count = 0

def stream_offset_pairs(swap_order):
    """
    Yields combined text blocks from parallel subtitle files using offset pairing.
    swap_order=False: prev_es+curr_eu / prev_eu+curr_es  (original pass)
    swap_order=True:  curr_eu+prev_es / curr_es+prev_eu  (reordered pass, new strings)
    """
    with open(SUB_ES_PATH, 'r', encoding='utf-8') as f_es, \
         open(SUB_EU_PATH, 'r', encoding='utf-8') as f_eu:
        try:
            prev_es = next(f_es).strip().replace('\n', ' ')
            prev_eu = next(f_eu).strip().replace('\n', ' ')
        except StopIteration:
            return

        counter = 0
        for curr_es_raw, curr_eu_raw in zip(f_es, f_eu):
            curr_es = curr_es_raw.strip().replace('\n', ' ')
            curr_eu = curr_eu_raw.strip().replace('\n', ' ')
            if not curr_es or not curr_eu:
                continue

            if counter % 2 == 0:
                block = f"{curr_eu} {prev_es}" if swap_order else f"{prev_es} {curr_eu}"
            else:
                block = f"{curr_es} {prev_eu}" if swap_order else f"{prev_eu} {curr_es}"

            yield block
            counter += 1
            prev_es, prev_eu = curr_es, curr_eu

print(f"[INFO] Compiling Phase 2 Dataset using CSCL Guidelines (Target: {TOTAL_BUDGET} rows)...")

with open(PHASE_2_OUT, 'w', encoding='utf-8') as f_out:
    
    # --- STEP 1: LOAD NATIVE NATURAL CONVERSATIONS (EuskañolDS) ---
    with open("/user/i.elcoroalberdi/u25035/MASTER/thesis/data/EuskanolDS/silver.txt", 'r', encoding='utf-8') as f_in:
        print("[INFO] Streaming natural EuskañolDS conversational corpus...")
        for line in f_in:
            text = line.strip().replace('\n', ' ')
            if text:
                f_out.write(text + '\n')
                line_count += 1
                if line_count >= TOTAL_BUDGET:
                    break
        print(f"[SUCCESS] Wrote {line_count} rows from natural EuskañolDS dialogues.")


    # --- STEP 2: LOAD NATIVE NATURAL CONVERSATIONS (BasqueParl) ---
    print("[INFO] Streaming natural BasqueParl conversational corpus...")
    try:
        basqueparl_chunks = pandas.read_csv(
            "/user/i.elcoroalberdi/u25035/.cache/huggingface/hub/datasets--HiTZ--basqueparl/snapshots/d65bb6d4c8334b208057e7cc721e4834fbefb8d7/basqueparl-v1.1.tsv",
            sep='\t', usecols=['text'], chunksize=10000, iterator=True
        )
        for chunk in basqueparl_chunks:
            for text in chunk['text']:
                if pandas.isna(text):
                    continue
                text = str(text).replace('\n', ' ').strip()
                if text:
                    f_out.write(text + '\n')
                    line_count += 1
                    if line_count >= TOTAL_BUDGET:
                        break
            if line_count >= TOTAL_BUDGET:
                break
        print(f"[SUCCESS] Wrote {line_count} rows from natural parliamentary dialogues.")
    except Exception as e:
        print(f"[WARNING] BasqueParl stream failed/skipped: {e}")
    

    # --- STEP 3: CONTEXTUAL SUBTITLE OFFSET ENGINE (multi-pass) ---
    if os.path.exists(SUB_ES_PATH) and os.path.exists(SUB_EU_PATH):
        pass_configs = [False, True]  # add more variants here if still short after pass 2

        for pass_idx, swap in enumerate(pass_configs, start=1):
            if line_count >= TOTAL_BUDGET:
                break

            deficit = TOTAL_BUDGET - line_count
            print(f"[INFO] Pass {pass_idx} (swap_order={swap}) - Target Deficit: {deficit} rows.")
            sub_written = 0

            for block in stream_offset_pairs(swap):
                f_out.write(block + '\n')
                sub_written += 1
                line_count += 1

                if sub_written % 200000 == 0:
                    print(f"    Processed {sub_written} offset dialogue rows in pass {pass_idx}...")

                if line_count >= TOTAL_BUDGET:
                    print("[INFO] Phase 2 curriculum budget filled perfectly.")
                    break
            else:
                print(f"[WARNING] OpenSubtitles exhausted in pass {pass_idx} "
                      f"({sub_written} rows written) without filling budget — trying next pass.")

print(f"\n[SUCCESS] Phase 2 Curriculum Generation Complete!")
print(f" -> Unified Target Output: {PHASE_2_OUT}")
print(f" -> Total Lines Saved: {line_count} / {TOTAL_BUDGET}")