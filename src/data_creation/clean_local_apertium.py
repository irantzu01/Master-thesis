import xml.etree.ElementTree as ET
import io
import os

#DIX_PATH = "/user/i.elcoroalberdi/u25035/MASTER/thesis/apertium-eu-es/apertium-eu-es.eu-es.dix"
DIX_PATH = "/user/i.elcoroalberdi/u25035/MASTER/thesis/apertium-eu-en/apertium-eu-en.eu-en.dix"
#OUTPUT_DIR = "/user/i.elcoroalberdi/u25035/MASTER/thesis/data/curriculum_dictionaries"
OUTPUT_DIR = "/user/i.elcoroalberdi/u25035/MASTER/thesis/data/curriculum_dictionaries_en"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(DIX_PATH):
    print(f"[ERROR] Could not find the file at {DIX_PATH}.")
    exit(1)

print("Parsing local Apertium dictionary file into 3 distinct targets...")
tree = ET.parse(DIX_PATH)
root = tree.getroot()

# Define the paths for the 3 dictionaries
dict_1_path = os.path.join(OUTPUT_DIR, "dict_1_adj_adv_en.txt")
dict_2_path = os.path.join(OUTPUT_DIR, "dict_2_verbs_en.txt")
dict_3_path = os.path.join(OUTPUT_DIR, "dict_3_nouns_en.txt")

counts = {"dict_1": 0, "dict_2": 0, "dict_3": 0}

with io.open(dict_1_path, 'w', encoding='utf-8') as f1, \
     io.open(dict_2_path, 'w', encoding='utf-8') as f2, \
     io.open(dict_3_path, 'w', encoding='utf-8') as f3:
     
    for e in root.iter('e'):
        p = e.find('p')
        if p is not None:
            l_tag = p.find('l')  # Basque translation side
            r_tag = p.find('r')  # English source side
            
            if l_tag is not None and r_tag is not None:
                eu_word = "".join(l_tag.itertext()).strip()
                en_word = "".join(r_tag.itertext()).strip()
                
                # Gather all part-of-speech tags for the current entry
                s_tags = [s.get('n') for s in l_tag.findall('s') + r_tag.findall('s') + e.findall('s')]
                
                # Clean strings to strip formatting or punctuation codes
                en_clean = "".join(c for c in en_word if c.isalpha() or c == '_').lower()
                eu_clean = "".join(c for c in eu_word if c.isalpha() or c == '_').lower()
                
                if en_clean and eu_clean:
                    # 1. First dictionary: Adjectives and Adverbs
                    if 'adj' in s_tags or 'adv' in s_tags:
                        f1.write(f"{en_clean}\t{eu_clean}\n")
                        counts["dict_1"] += 1
                        
                    # 2. Second dictionary: Lexical Verbs
                    elif 'vblex' in s_tags:
                        f2.write(f"{en_clean}\t{eu_clean}\n")
                        counts["dict_2"] += 1
                        
                    # 3. Third dictionary: Standard Nouns (Excludes Proper Nouns)
                    elif 'n' in s_tags and 'np' not in s_tags:
                        f3.write(f"{en_clean}\t{eu_clean}\n")
                        counts["dict_3"] += 1

print(f"\n[SUCCESS] Extraction finished successfully!")
print(f" -> Dict 1 (ADJ + ADV) saved to: {dict_1_path} ({counts['dict_1']} pairs)")
print(f" -> Dict 2 (VERB)      saved to: {dict_2_path} ({counts['dict_2']} pairs)")
print(f" -> Dict 3 (NOUN)      saved to: {dict_3_path} ({counts['dict_3']} pairs)")