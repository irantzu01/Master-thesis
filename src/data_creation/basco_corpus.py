"""Extract the code-switched sentences from the BaSCo corpus.

Input:  <DATA_ROOT>/valid_utterances.json   (from https://github.com/Vicomtech/BaSCo-Corpus)
Output: <DATA_ROOT>/BaSCo_corpus.txt        (one code-switched sentence per line)
"""
import json
import os

from data_paths import DATA_ROOT

input_json_file = os.path.join(DATA_ROOT, "valid_utterances.json")
output_txt_file = os.path.join(DATA_ROOT, "BaSCo_corpus.txt")

print(f"[INFO] Parsing JSON file: {input_json_file}")

extracted_sentences = []

with open(input_json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

    for entry in data:
        # Check if the entry has a code-switching block
        if "code-switching" in entry:
            for cs_item in entry["code-switching"]:
                if "text" in cs_item:
                    sentence = cs_item["text"].strip()
                    if sentence:
                        extracted_sentences.append(sentence)

# Save them as a clean line-by-line text file
os.makedirs(os.path.dirname(output_txt_file), exist_ok=True)
with open(output_txt_file, "w", encoding="utf-8") as out:
    for sentence in extracted_sentences:
        out.write(sentence + "\n")

print(f"[SUCCESS] Extracted {len(extracted_sentences)} code-switched sentences!")
print(f"[INFO] Saved to: {output_txt_file}")
