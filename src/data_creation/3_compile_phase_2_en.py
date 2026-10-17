"""Step 3 (English): build the sentence-level code-switched data (phase 2).

Synthetic only: OpenSubtitles blocks that join the previous line in one language
with the current line in the other (alternating direction). Pass 1 starts with
English -> Basque; if the budget is not filled, pass 2 repeats with the
language pairing inverted.

Inputs (under <DATA_ROOT>): en-eu.txt/OpenSubtitles.en-eu.{en,eu}
(OPUS OpenSubtitles v2024, Moses format).
Output: <DATA_ROOT>/synthetic_english_cs/phase_2_sentence_level.txt
(the folder that train_cscl_english.py reads).
"""
import os
import sys

from data_paths import DATA_ROOT, TOTAL_ROWS

OUTPUT_DIR = os.path.join(DATA_ROOT, "synthetic_english_cs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
PHASE_2_OUT = os.path.join(OUTPUT_DIR, "phase_2_sentence_level.txt")

SUB_EN_PATH = os.path.join(DATA_ROOT, "en-eu.txt", "OpenSubtitles.en-eu.en")
SUB_EU_PATH = os.path.join(DATA_ROOT, "en-eu.txt", "OpenSubtitles.en-eu.eu")

TOTAL_BUDGET = TOTAL_ROWS
line_count = 0

print(f"[INFO] Compiling English-Basque Phase 2 Dataset (Target: {TOTAL_BUDGET} rows)...")

if not os.path.exists(SUB_EN_PATH) or not os.path.exists(SUB_EU_PATH):
    print(f"[CRITICAL] Missing parallel subtitles files at {SUB_EN_PATH} or {SUB_EU_PATH}")
    sys.exit(1)

with open(PHASE_2_OUT, "w", encoding="utf-8") as f_out:
    sub_written = 0

    # --- PASS 1: English turn -> Basque turn ---
    print("[INFO] Executing Pass 1 (English leading conversational turns)...")
    with open(SUB_EN_PATH, "r", encoding="utf-8") as f_en, \
         open(SUB_EU_PATH, "r", encoding="utf-8") as f_eu:

        try:
            prev_en = next(f_en).strip().replace("\n", " ")
            prev_eu = next(f_eu).strip().replace("\n", " ")

            for curr_en_raw, curr_eu_raw in zip(f_en, f_eu):
                curr_en = curr_en_raw.strip().replace("\n", " ")
                curr_eu = curr_eu_raw.strip().replace("\n", " ")

                if not curr_en or not curr_eu:
                    continue

                # Offset pairing: adjacent turns, never a line with its own translation
                if sub_written % 2 == 0:
                    combined_block = f"{prev_en} {curr_eu}"
                else:
                    combined_block = f"{prev_eu} {curr_en}"

                f_out.write(combined_block + "\n")
                sub_written += 1
                line_count += 1

                prev_en = curr_en
                prev_eu = curr_eu

                if line_count % 200000 == 0:
                    print(f"    Processed {line_count} / {TOTAL_BUDGET} offset rows...")

                if line_count >= TOTAL_BUDGET:
                    break
        except StopIteration:
            print(f"[INFO] Pass 1 complete. Exhausted files at {line_count} rows.")

    # --- PASS 2: inverted language pairing, only if the budget is not yet filled ---
    deficit = TOTAL_BUDGET - line_count
    if deficit > 0:
        print(f"[INFO] Deficit detected: {deficit} rows. Launching Pass 2 (inverted language tracks)...")

        with open(SUB_EN_PATH, "r", encoding="utf-8") as f_en, \
             open(SUB_EU_PATH, "r", encoding="utf-8") as f_eu:

            try:
                prev_en = next(f_en).strip().replace("\n", " ")
                prev_eu = next(f_eu).strip().replace("\n", " ")

                for curr_en_raw, curr_eu_raw in zip(f_en, f_eu):
                    curr_en = curr_en_raw.strip().replace("\n", " ")
                    curr_eu = curr_eu_raw.strip().replace("\n", " ")

                    if not curr_en or not curr_eu:
                        continue

                    # Inverted pairing relative to pass 1: new sentence-boundary
                    # transitions from the same text.
                    if sub_written % 2 == 0:
                        combined_block = f"{prev_eu} {curr_en}"
                    else:
                        combined_block = f"{prev_en} {curr_eu}"

                    f_out.write(combined_block + "\n")
                    sub_written += 1
                    line_count += 1

                    prev_en = curr_en
                    prev_eu = curr_eu

                    if line_count % 200000 == 0:
                        print(f"    Processed {line_count} / {TOTAL_BUDGET} offset rows...")

                    if line_count >= TOTAL_BUDGET:
                        print("[INFO] Target curriculum budget reached during Pass 2.")
                        break
            except StopIteration:
                print("[CRITICAL] OpenSubtitles corpus exhausted in both passes.")

print("\n[SUCCESS] Phase 2 Curriculum Generation Complete!")
print(f" -> Output: {PHASE_2_OUT}")
print(f" -> Total Lines Saved: {line_count} / {TOTAL_BUDGET}")
