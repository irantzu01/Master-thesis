"""Build the three curriculum dictionaries from an Apertium bilingual dictionary.

Each entry of the .dix file is written to one of three files, by part of speech
(first matching rule wins):
    dict_1: adjectives and adverbs   (tags adj, adv)
    dict_2: lexical verbs            (tag vblex)
    dict_3: common nouns             (tag n, not np)
Lines are "<source word>\t<Basque word>"; entries are lower-cased and stripped of
everything except letters and underscores.

Usage:
    python clean_local_apertium.py --lang es     # Spanish -> Basque
    python clean_local_apertium.py --lang en     # English -> Basque

The .dix files come from https://github.com/apertium/apertium-eu-es and
https://github.com/apertium/apertium-eu-en (clone them next to the data folder or
pass --dix). Output goes to <DATA_ROOT>/curriculum_dictionaries (es) or
<DATA_ROOT>/curriculum_dictionaries_en (en); English file names end in _en.
"""
import argparse
import io
import os
import sys
import xml.etree.ElementTree as ET

from data_paths import DATA_ROOT

_REPO_ROOT = os.path.dirname(DATA_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--lang", choices=["es", "en"], required=True,
                    help="source language of the dictionary")
parser.add_argument("--dix", default=None,
                    help="path to the Apertium .dix file "
                         "(default: <repo>/apertium-eu-<lang>/apertium-eu-<lang>.eu-<lang>.dix)")
parser.add_argument("--out_dir", default=None, help="output folder")
args = parser.parse_args()

lang = args.lang
suffix = "" if lang == "es" else "_en"
DIX_PATH = args.dix or os.path.join(
    _REPO_ROOT, f"apertium-eu-{lang}", f"apertium-eu-{lang}.eu-{lang}.dix")
OUTPUT_DIR = args.out_dir or os.path.join(
    DATA_ROOT, "curriculum_dictionaries" if lang == "es" else "curriculum_dictionaries_en")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(DIX_PATH):
    print(f"[ERROR] Could not find the file at {DIX_PATH}.")
    sys.exit(1)

print("Parsing local Apertium dictionary file into 3 distinct targets...")
tree = ET.parse(DIX_PATH)
root = tree.getroot()

dict_1_path = os.path.join(OUTPUT_DIR, f"dict_1_adj_adv{suffix}.txt")
dict_2_path = os.path.join(OUTPUT_DIR, f"dict_2_verbs{suffix}.txt")
dict_3_path = os.path.join(OUTPUT_DIR, f"dict_3_nouns{suffix}.txt")

counts = {"dict_1": 0, "dict_2": 0, "dict_3": 0}

with io.open(dict_1_path, "w", encoding="utf-8") as f1, \
     io.open(dict_2_path, "w", encoding="utf-8") as f2, \
     io.open(dict_3_path, "w", encoding="utf-8") as f3:

    for e in root.iter("e"):
        p = e.find("p")
        if p is not None:
            l_tag = p.find("l")  # Basque side
            r_tag = p.find("r")  # source-language side (Spanish or English)

            if l_tag is not None and r_tag is not None:
                eu_word = "".join(l_tag.itertext()).strip()
                src_word = "".join(r_tag.itertext()).strip()

                # Gather all part-of-speech tags for the current entry
                s_tags = [s.get("n") for s in l_tag.findall("s") + r_tag.findall("s") + e.findall("s")]

                # Strip formatting or punctuation codes
                src_clean = "".join(c for c in src_word if c.isalpha() or c == "_").lower()
                eu_clean = "".join(c for c in eu_word if c.isalpha() or c == "_").lower()

                if src_clean and eu_clean:
                    if "adj" in s_tags or "adv" in s_tags:
                        f1.write(f"{src_clean}\t{eu_clean}\n")
                        counts["dict_1"] += 1
                    elif "vblex" in s_tags:
                        f2.write(f"{src_clean}\t{eu_clean}\n")
                        counts["dict_2"] += 1
                    elif "n" in s_tags and "np" not in s_tags:
                        f3.write(f"{src_clean}\t{eu_clean}\n")
                        counts["dict_3"] += 1

print("\n[SUCCESS] Extraction finished successfully!")
print(f" -> Dict 1 (ADJ + ADV) saved to: {dict_1_path} ({counts['dict_1']} pairs)")
print(f" -> Dict 2 (VERB)      saved to: {dict_2_path} ({counts['dict_2']} pairs)")
print(f" -> Dict 3 (NOUN)      saved to: {dict_3_path} ({counts['dict_3']} pairs)")
