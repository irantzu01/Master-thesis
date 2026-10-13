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
    # Blazing fast token splitter
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
    DICT_DIR = "/user/i.elcoroalberdi/u25035/MASTER/thesis/data/curriculum_dictionaries"
    OUTPUT_DIR = "/user/i.elcoroalberdi/u25035/MASTER/thesis/data/synthetic_spanish_adj_cs"
    RAW_ES_FILE = "/user/i.elcoroalberdi/u25035/MASTER/thesis/data/raw_spanish_master.txt"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(RAW_ES_FILE):
        print(f"[CRITICAL] Run Step 1 script first to create {RAW_ES_FILE}!")
        sys.exit(1)

    print("[INFO] Loading translation layers...")
    dict_adj_adv = load_dictionary(os.path.join(DICT_DIR, "dict_1_adj_adv.txt"))
    dict_verbs   = load_dictionary(os.path.join(DICT_DIR, "dict_2_verbs.txt"))
    dict_nouns   = load_dictionary(os.path.join(DICT_DIR, "dict_3_nouns.txt"))

    # High-ambiguity functional particles that should NEVER be swapped raw
    forbidden_words = {
        "fuera", "como", "no", "menos", "para", "por", "con", "que", "un", "una",
        "este", "esta", "es", "son", "fue", "era", "ser", "estar", "lo", "los", "su", "sus"
    }

    print("[INFO] Loading raw cache into memory array...")
    with open(RAW_ES_FILE, 'r', encoding='utf-8') as f:
        all_lines = [line.strip() for line in f if line.strip()]

    # Layout the 3 sub-phases using perfect sequential indices
    sub_phases = [
        {"name": "phase_1a_modifiers", "lines": all_lines[0:458111],       "dict": dict_adj_adv},
        {"name": "phase_1b_verbs",     "lines": all_lines[458111:916222], "dict": {**dict_adj_adv, **dict_verbs}},
        {"name": "phase_1c_nouns",     "lines": all_lines[916222:1374333], "dict": {**dict_adj_adv, **dict_verbs, **dict_nouns}}
    ]

    for p in sub_phases:
        out_file = os.path.join(OUTPUT_DIR, f"{p['name']}.txt")
        print(f"[PROCESSING] Generating: {p['name']}...")
        
        with open(out_file, 'w', encoding='utf-8') as f_out:
            for line in p['lines']:
                cs_text = fast_swap_tokens(line, p['dict'], forbidden_words)
                f_out.write(cs_text + '\n')
                
        print(f"[SUCCESS] Created {out_file}")

    print("\n[ALL TASKS COMPLETE] All three Phase 1 curriculum files are built!")