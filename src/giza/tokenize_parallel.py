"""Tokenise the cleaned Spanish-Basque parallel corpus for GIZA++.

Spanish is tokenised with the DeepESP/gpt2-spanish tokenizer, Basque with the
target tokenizer (basque_tokenizer_50257/). Pairs in which either side has more
than 101 tokens are dropped (GIZA++ maximum sentence length).

Input : <DATA_ROOT>/combined_clean.{es,eu}   (output of clean_parallel_corpus.py)
Output: <DATA_ROOT>/parallel.{es,eu}.tok     (one line per pair, tokens separated by spaces)

Run subsample_parallel.sh afterwards to draw the 4M pairs used for alignment.
"""
import argparse
import os
import sys

from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "data_creation"))
from data_paths import DATA_ROOT  # noqa: E402

MAX_TOKENS = 101

parser = argparse.ArgumentParser()
parser.add_argument("--es_in", default=os.path.join(DATA_ROOT, "combined_clean.es"))
parser.add_argument("--eu_in", default=os.path.join(DATA_ROOT, "combined_clean.eu"))
parser.add_argument("--es_out", default=os.path.join(DATA_ROOT, "parallel.es.tok"))
parser.add_argument("--eu_out", default=os.path.join(DATA_ROOT, "parallel.eu.tok"))
parser.add_argument("--basque_tokenizer", default=os.path.join(REPO_ROOT, "basque_tokenizer_50257"))
args = parser.parse_args()

tok_s = AutoTokenizer.from_pretrained("DeepESP/gpt2-spanish")
tok_t = AutoTokenizer.from_pretrained(args.basque_tokenizer)

kept = total = 0
with open(args.es_in, "r", encoding="utf-8") as f_es_in, \
     open(args.eu_in, "r", encoding="utf-8") as f_eu_in, \
     open(args.es_out, "w", encoding="utf-8") as f_es_out, \
     open(args.eu_out, "w", encoding="utf-8") as f_eu_out:

    for es_line, eu_line in zip(f_es_in, f_eu_in):
        total += 1
        es_line, eu_line = es_line.strip(), eu_line.strip()
        if not es_line or not eu_line:
            continue
        es_toks = tok_s.tokenize(es_line)
        eu_toks = tok_t.tokenize(eu_line)
        if len(es_toks) > MAX_TOKENS or len(eu_toks) > MAX_TOKENS:
            continue
        f_es_out.write(" ".join(es_toks) + "\n")
        f_eu_out.write(" ".join(eu_toks) + "\n")
        kept += 1

print(f"Kept {kept} of {total} pairs -> {args.es_out}, {args.eu_out}")
