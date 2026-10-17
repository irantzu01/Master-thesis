"""Step 3 (Spanish): build the sentence-level code-switched data (phase 2).

Order of the TOTAL_ROWS lines:
  1. EuskañolDS silver corpus (real code-switched dialogue)
  2. BasqueParl (real code-switched parliamentary text)
  3. OpenSubtitles blocks: the previous line in one language joined with the current
     line in the other (alternating direction). A second pass with the two halves
     in reversed order fills any remaining budget.

Inputs (under <DATA_ROOT>):
  EuskanolDS/silver.txt                      (download from HiTZ/EuskanolDS)
  es-eu.txt/OpenSubtitles.es-eu.{es,eu}      (OPUS OpenSubtitles v2024, Moses format)
  BasqueParl is downloaded from the Hugging Face Hub (HiTZ/basqueparl, pinned revision).
Output: <DATA_ROOT>/synthetic_spanish_adj_cs/phase_2_sentence_level.txt
"""
import os
import sys

import pandas
from huggingface_hub import hf_hub_download

from data_paths import DATA_ROOT, TOTAL_ROWS

OUTPUT_DIR = os.path.join(DATA_ROOT, "synthetic_spanish_adj_cs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
PHASE_2_OUT = os.path.join(OUTPUT_DIR, "phase_2_sentence_level.txt")

SUB_ES_PATH = os.path.join(DATA_ROOT, "es-eu.txt", "OpenSubtitles.es-eu.es")
SUB_EU_PATH = os.path.join(DATA_ROOT, "es-eu.txt", "OpenSubtitles.es-eu.eu")
EUSKANOLDS_PATH = os.path.join(DATA_ROOT, "EuskanolDS", "silver.txt")

BASQUEPARL_REPO = "HiTZ/basqueparl"
BASQUEPARL_FILE = "basqueparl-v1.1.tsv"
BASQUEPARL_REVISION = "d65bb6d4c8334b208057e7cc721e4834fbefb8d7"  # snapshot used in the thesis

TOTAL_BUDGET = TOTAL_ROWS
line_count = 0

for required in (EUSKANOLDS_PATH, SUB_ES_PATH, SUB_EU_PATH):
    if not os.path.exists(required):
        print(f"[CRITICAL] Missing input file: {required}")
        sys.exit(1)


def stream_offset_pairs(swap_order):
    """
    Yields combined text blocks from parallel subtitle files using offset pairing.
    swap_order=False: prev_es+curr_eu / prev_eu+curr_es  (original pass)
    swap_order=True:  curr_eu+prev_es / curr_es+prev_eu  (reordered pass, new strings)
    """
    with open(SUB_ES_PATH, "r", encoding="utf-8") as f_es, \
         open(SUB_EU_PATH, "r", encoding="utf-8") as f_eu:
        try:
            prev_es = next(f_es).strip().replace("\n", " ")
            prev_eu = next(f_eu).strip().replace("\n", " ")
        except StopIteration:
            return

        counter = 0
        for curr_es_raw, curr_eu_raw in zip(f_es, f_eu):
            curr_es = curr_es_raw.strip().replace("\n", " ")
            curr_eu = curr_eu_raw.strip().replace("\n", " ")
            if not curr_es or not curr_eu:
                continue

            if counter % 2 == 0:
                block = f"{curr_eu} {prev_es}" if swap_order else f"{prev_es} {curr_eu}"
            else:
                block = f"{curr_es} {prev_eu}" if swap_order else f"{prev_eu} {curr_es}"

            yield block
            counter += 1
            prev_es, prev_eu = curr_es, curr_eu


print(f"[INFO] Compiling Phase 2 Dataset (Target: {TOTAL_BUDGET} rows)...")

with open(PHASE_2_OUT, "w", encoding="utf-8") as f_out:

    # --- STEP 1: EuskañolDS ---
    with open(EUSKANOLDS_PATH, "r", encoding="utf-8") as f_in:
        print("[INFO] Streaming EuskañolDS...")
        for line in f_in:
            text = line.strip().replace("\n", " ")
            if text:
                f_out.write(text + "\n")
                line_count += 1
                if line_count >= TOTAL_BUDGET:
                    break
        print(f"[SUCCESS] Wrote {line_count} rows from EuskañolDS.")

    # --- STEP 2: BasqueParl ---
    # Fails loudly if the download is not possible, so that the corpus is never
    # silently built without BasqueParl.
    print("[INFO] Streaming BasqueParl...")
    basqueparl_path = hf_hub_download(
        repo_id=BASQUEPARL_REPO, filename=BASQUEPARL_FILE,
        repo_type="dataset", revision=BASQUEPARL_REVISION,
    )
    basqueparl_chunks = pandas.read_csv(
        basqueparl_path, sep="\t", usecols=["text"], chunksize=10000, iterator=True
    )
    for chunk in basqueparl_chunks:
        for text in chunk["text"]:
            if pandas.isna(text):
                continue
            text = str(text).replace("\n", " ").strip()
            if text:
                f_out.write(text + "\n")
                line_count += 1
                if line_count >= TOTAL_BUDGET:
                    break
        if line_count >= TOTAL_BUDGET:
            break
    print(f"[SUCCESS] {line_count} rows written after BasqueParl.")

    # --- STEP 3: OpenSubtitles offset blocks (up to two passes) ---
    pass_configs = [False, True]

    for pass_idx, swap in enumerate(pass_configs, start=1):
        if line_count >= TOTAL_BUDGET:
            break

        deficit = TOTAL_BUDGET - line_count
        print(f"[INFO] Pass {pass_idx} (swap_order={swap}) - Target Deficit: {deficit} rows.")
        sub_written = 0

        for block in stream_offset_pairs(swap):
            f_out.write(block + "\n")
            sub_written += 1
            line_count += 1

            if sub_written % 200000 == 0:
                print(f"    Processed {sub_written} offset dialogue rows in pass {pass_idx}...")

            if line_count >= TOTAL_BUDGET:
                print("[INFO] Phase 2 curriculum budget filled.")
                break
        else:
            print(f"[WARNING] OpenSubtitles exhausted in pass {pass_idx} "
                  f"({sub_written} rows written) without filling budget - trying next pass.")

print("\n[SUCCESS] Phase 2 Curriculum Generation Complete!")
print(f" -> Output: {PHASE_2_OUT}")
print(f" -> Total Lines Saved: {line_count} / {TOTAL_BUDGET}")
