from transformers import AutoTokenizer
import os


tok_s = AutoTokenizer.from_pretrained("DeepESP/gpt2-spanish")
tok_t = AutoTokenizer.from_pretrained("./basque_tokenizer_50257")  # your existing target tokenizer

es_path = "./data/combined_clean.es"
eu_path = "./data/combined_clean.eu"

with open(es_path, "r", encoding="utf-8") as f_es_in, \
     open(eu_path, "r", encoding="utf-8") as f_eu_in, \
     open("parallel.es.tok", "w", encoding="utf-8") as f_es_out, \
     open("parallel.eu.tok", "w", encoding="utf-8") as f_eu_out:

    for es_line, eu_line in zip(f_es_in, f_eu_in):
        es_line, eu_line = es_line.strip(), eu_line.strip()
        if not es_line or not eu_line:
            continue
        # GIZA++'s max sentence length is 101 tokens (paper's setting) -- skip longer pairs
        es_toks = tok_s.tokenize(es_line)
        eu_toks = tok_t.tokenize(eu_line)
        if len(es_toks) > 101 or len(eu_toks) > 101:
            continue
        f_es_out.write(" ".join(es_toks) + "\n")
        f_eu_out.write(" ".join(eu_toks) + "\n")