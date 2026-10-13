"""
Lightweight cleaning for the combined CCMatrix + ParaCrawl + Elhuyar
Spanish-Basque parallel corpus before GIZA alignment:
  1. Drop empty lines / misaligned pairs
  2. Deduplicate exact-duplicate sentence pairs (common across mined corpora)
  3. Filter extreme length-ratio pairs (common mined-corpus alignment error)
"""

import argparse

MAX_LENGTH_RATIO = 3.0  # discard pairs where one side is >3x longer than the other

def clean(es_path, eu_path, out_es_path, out_eu_path):
    seen_pairs = set()
    total, kept, dropped_empty, dropped_dup, dropped_ratio = 0, 0, 0, 0, 0

    with open(es_path, encoding="utf-8") as f_es, \
         open(eu_path, encoding="utf-8") as f_eu, \
         open(out_es_path, "w", encoding="utf-8") as out_es, \
         open(out_eu_path, "w", encoding="utf-8") as out_eu:

        for es_line, eu_line in zip(f_es, f_eu):
            total += 1
            es_line, eu_line = es_line.strip(), eu_line.strip()

            if not es_line or not eu_line:
                dropped_empty += 1
                continue

            pair_key = (es_line, eu_line)
            if pair_key in seen_pairs:
                dropped_dup += 1
                continue
            seen_pairs.add(pair_key)

            es_len, eu_len = len(es_line.split()), len(eu_line.split())
            if es_len == 0 or eu_len == 0:
                dropped_empty += 1
                continue
            ratio = max(es_len, eu_len) / min(es_len, eu_len)
            if ratio > MAX_LENGTH_RATIO:
                dropped_ratio += 1
                continue

            out_es.write(es_line + "\n")
            out_eu.write(eu_line + "\n")
            kept += 1

            if total % 500000 == 0:
                print(f"[INFO] Processed {total} lines, kept {kept} so far...")

    print(f"\n[SUCCESS] Cleaning complete.")
    print(f"  Total lines read:      {total}")
    print(f"  Kept:                  {kept}")
    print(f"  Dropped (empty):       {dropped_empty}")
    print(f"  Dropped (duplicate):   {dropped_dup}")
    print(f"  Dropped (length ratio):{dropped_ratio}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--es_in", required=True)
    parser.add_argument("--eu_in", required=True)
    parser.add_argument("--es_out", required=True)
    parser.add_argument("--eu_out", required=True)
    args = parser.parse_args()

    clean(args.es_in, args.eu_in, args.es_out, args.eu_out)