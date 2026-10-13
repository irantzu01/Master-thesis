import os
import sys

# --- Configuration Paths ---
OUTPUT_DIR = "/user/i.elcoroalberdi/u25035/MASTER/thesis/data/synthetic_english_adj_cs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PHASE_2_OUT = os.path.join(OUTPUT_DIR, "phase_2_sentence_level.txt")

# Parallel subtitle corpora paths (Adjust these to match your exact filenames on disk)
SUB_EN_PATH = "/user/i.elcoroalberdi/u25035/MASTER/thesis/data/en-eu.txt/OpenSubtitles.en-eu.en"
SUB_EU_PATH = "/user/i.elcoroalberdi/u25035/MASTER/thesis/data/en-eu.txt/OpenSubtitles.en-eu.eu"

TOTAL_BUDGET = 1374333
line_count = 0

print(f"[INFO] Compiling English-Basque Phase 2 Dataset (Target: {TOTAL_BUDGET} rows)...")

if not os.path.exists(SUB_EN_PATH) or not os.path.exists(SUB_EU_PATH):
    print(f"[CRITICAL] Missing parallel subtitles files at {SUB_EN_PATH} or {SUB_EU_PATH}")
    sys.exit(1)

with open(PHASE_2_OUT, 'w', encoding='utf-8') as f_out:
    sub_written = 0
    
    # --- ROUND 1: ENGLISH TURN -> BASQUE TURN ---
    print("[INFO] Executing Pass 1 (English leading conversational turns)...")
    with open(SUB_EN_PATH, 'r', encoding='utf-8') as f_en, \
         open(SUB_EU_PATH, 'r', encoding='utf-8') as f_eu:
        
        try:
            prev_en = next(f_en).strip().replace('\n', ' ')
            prev_eu = next(f_eu).strip().replace('\n', ' ')
            
            for curr_en_raw, curr_eu_raw in zip(f_en, f_eu):
                curr_en = curr_en_raw.strip().replace('\n', ' ')
                curr_eu = curr_eu_raw.strip().replace('\n', ' ')
                
                if not curr_en or not curr_eu:
                    continue
                    
                # Offset shifting creates clean continuous flow without literal translation repetition
                if sub_written % 2 == 0:
                    combined_block = f"{prev_en} {curr_eu}"
                else:
                    combined_block = f"{prev_eu} {curr_en}"
                    
                f_out.write(combined_block + '\n')
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

    # --- ROUND 2: INVERTED FALLBACK (BASQUE TURN -> ENGLISH TURN) ---
    # If we are still below the 1.65M budget, reopen files and run the cross-lingual inverse map
    deficit = TOTAL_BUDGET - line_count
    if deficit > 0:
        print(f"[INFO] Deficit detected: {deficit} rows. Launching Pass 2 (Inverted Language Tracks)...")
        
        with open(SUB_EN_PATH, 'r', encoding='utf-8') as f_en, \
             open(SUB_EU_PATH, 'r', encoding='utf-8') as f_eu:
            
            try:
                prev_en = next(f_en).strip().replace('\n', ' ')
                prev_eu = next(f_eu).strip().replace('\n', ' ')
                
                for curr_en_raw, curr_eu_raw in zip(f_en, f_eu):
                    curr_en = curr_en_raw.strip().replace('\n', ' ')
                    curr_eu = curr_eu_raw.strip().replace('\n', ' ')
                    
                    if not curr_en or not curr_eu:
                        continue
                    
                    # --- INVERTED SHIFT LOGIC ---
                    # Reverse the configuration mapping assignment patterns relative to Pass 1. 
                    # This yields brand-new sentence boundary transitions from the exact same tokens.
                    if sub_written % 2 == 0:
                        combined_block = f"{prev_eu} {curr_en}"
                    else:
                        combined_block = f"{prev_en} {curr_eu}"
                        
                    f_out.write(combined_block + '\n')
                    sub_written += 1
                    line_count += 1
                    
                    prev_en = curr_en
                    prev_eu = curr_eu
                    
                    if line_count % 200000 == 0:
                        print(f"    Processed {line_count} / {TOTAL_BUDGET} offset rows...")
                        
                    if line_count >= TOTAL_BUDGET:
                        print("[INFO] Target curriculum budget reached successfully during Pass 2.")
                        break
            except StopIteration:
                print("[CRITICAL] OpenSubtitles corpus completely exhausted in both structural directions.")

print(f"\n[SUCCESS] Phase 2 Curriculum Generation Complete!")
print(f" -> Unified Target Output: {PHASE_2_OUT}")
print(f" -> Total Lines Saved: {line_count} / {TOTAL_BUDGET}")