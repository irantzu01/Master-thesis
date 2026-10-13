import os
import sys
import re

def load_dictionary(path):
    d = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and '\t' in line:
                parts = line.split('\t')
                d[parts[0].lower()] = parts[1].lower()
    return d

def fast_swap_tokens(text, dictionary, forbidden_set):
    if not text:
        return ""
    tokens = re.findall(r'\w+|[^\w\s]|\s+', text)
    modified_tokens = []
    
    for token in tokens:
        if token.isalnum():
            token_lower = token.lower()
            if token_lower in dictionary and token_lower not in forbidden_set:
                basque_word = dictionary[token_lower]
                if token.isupper():
                    modified_tokens.append(basque_word.upper())
                elif token[0].isupper():
                    modified_tokens.append(basque_word.capitalize())
                else:
                    modified_tokens.append(basque_word)
            else:
                modified_tokens.append(token)
        else:
            modified_tokens.append(token)
            
    return "".join(modified_tokens)

if __name__ == "__main__":
    DICT_DIR = "/user/i.elcoroalberdi/u25035/MASTER/thesis/data/curriculum_dictionaries_en"
    OUTPUT_DIR = "/user/i.elcoroalberdi/u25035/MASTER/thesis/data/synthetic_english_cs"
    RAW_EN_FILE = "/user/i.elcoroalberdi/u25035/MASTER/thesis/data/raw_english_master.txt"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dict_adj_adv = load_dictionary(os.path.join(DICT_DIR, "dict_1_adj_adv_en.txt"))
    dict_verbs   = load_dictionary(os.path.join(DICT_DIR, "dict_2_verbs_en.txt"))
    dict_nouns   = load_dictionary(os.path.join(DICT_DIR, "dict_3_nouns_en.txt"))

    # English high-ambiguity functional particles to protect
    forbidden_words = {
        "to", "the", "a", "an", "and", "of", "in", "on", "at", "for", "from", "by",
        "is", "are", "was", "were", "be", "been", "it", "they", "he", "she", "this", "that",
        "not", "no", "more", "less", "with", "as", "but", "who", "which", "what"
    }

    print("[INFO] Loading raw English lines...")
    with open(RAW_EN_FILE, 'r', encoding='utf-8') as f:
        all_lines = [line.strip() for line in f if line.strip()]

        sub_phases = [
        {"name": "phase_1a_modifiers", "lines": all_lines[0:458111],       "dict": dict_adj_adv},
        {"name": "phase_1b_verbs",     "lines": all_lines[458111:916222], "dict": {**dict_adj_adv, **dict_verbs}},
        {"name": "phase_1c_nouns",     "lines": all_lines[916222:1374333], "dict": {**dict_adj_adv, **dict_verbs, **dict_nouns}}
    ]

    for p in sub_phases:
        out_file = os.path.join(OUTPUT_DIR, f"{p['name']}.txt")
        print(f"[PROCESSING] Compiling English-Basque: {p['name']}...")
        with open(out_file, 'w', encoding='utf-8') as f_out:
            for line in p['lines']:
                f_out.write(fast_swap_tokens(line, p['dict'], forbidden_words) + '\n')
        print(f"[SUCCESS] Created {out_file}")