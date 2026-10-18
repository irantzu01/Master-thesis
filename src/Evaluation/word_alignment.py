import torch
import numpy as np
import os
import random
import xml.etree.ElementTree as ET
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from eval_paths import ckpt, is_local, APERTIUM_EU_ES_DIX, APERTIUM_EU_EN_DIX

# --- 1. Runtime Configurations ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Running word-level alignment diagnostics on: {device}")

N_BOOTSTRAP = 3
BOOTSTRAP_SEEDS = [42, 123, 2024]
N_WORD_PAIRS = 200  # matches the scale of the sentence-level FLORES analysis
SAMPLE_SEED = 42

APERTIUM_EU_ES_DIX = str(APERTIUM_EU_ES_DIX)
APERTIUM_EU_EN_DIX = str(APERTIUM_EU_EN_DIX)

MODELS_TO_TEST = {
    "Baseline":              {"path": ckpt("baseline"), "tokenizer_base": "DeepESP/gpt2-spanish"},
    "CSCL-es":               {"path": ckpt("cscl-es"),  "tokenizer_base": "DeepESP/gpt2-spanish"},
    "CSCL-en":               {"path": ckpt("cscl-en"),  "tokenizer_base": "openai-community/gpt2"},
    "GPT-2 Spanish":         {"path": "DeepESP/gpt2-spanish",   "tokenizer_base": "DeepESP/gpt2-spanish"},
    "GPT-2 English":         {"path": "openai-community/gpt2",  "tokenizer_base": "openai-community/gpt2"},
    "TokAlign-Untrained":    {"path": ckpt("tokalign"),         "tokenizer_base": ckpt("tokalign")},
    "Tokalign-CSCL":         {"path": ckpt("tokalign-cscl"),    "tokenizer_base": ckpt("tokalign-cscl")},
    "EnerGIZAr++-Untrained": {"path": ckpt("energizar"),        "tokenizer_base": ckpt("energizar")},
    "Energizar-CSCL":        {"path": ckpt("energizar-cscl"),   "tokenizer_base": ckpt("energizar-cscl")},
}


# --- 2. Apertium .dix Parser ---
def parse_apertium_dix(path):
    """
    Parses an Apertium bilingual .dix file and returns a list of
    (left_word, right_word) surface-form tuples from every <e><p><l>...
    </l><r>...</r></p></e> entry in the file. By Apertium's own naming
    convention (e.g. "eu-es.dix"), 'left' = first language code in the
    filename, 'right' = second -- VERIFY this against the printed sample
    below rather than assuming it holds for your specific files.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    pairs = []
    for e in root.iter("e"):
        p = e.find("p")
        if p is None:
            continue  # skip <i> (identity) entries and other non-pair entries
        l_elem = p.find("l")
        r_elem = p.find("r")
        if l_elem is None or r_elem is None:
            continue
        # itertext() joins the element's own text with any nested text,
        # which correctly skips over empty <s n="..."/> grammatical symbol
        # tags while still capturing multi-part surface forms.
        left_text = "".join(l_elem.itertext()).strip()
        right_text = "".join(r_elem.itertext()).strip()
        if left_text and right_text:
            pairs.append((left_text, right_text))

    return pairs


def clean_word_pairs(pairs, max_word_length=25):
    """
    Deduplicates pairs, drops entries where source == target (uninformative
    identity pairs -- proper nouns, numbers, etc. that would trivially
    inflate alignment scores), and drops unusually long entries (likely
    multi-word expressions rather than single lexical items).
    """
    seen = set()
    cleaned = []
    for src, tgt in pairs:
        if src == tgt:
            continue
        if len(src) > max_word_length or len(tgt) > max_word_length:
            continue
        key = (src, tgt)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append((src, tgt))
    return cleaned


# --- 3. Load and inspect the dictionaries ---
print("[INFO] Parsing Apertium Basque-Spanish dictionary...")
eu_es_pairs_raw = parse_apertium_dix(APERTIUM_EU_ES_DIX)
eu_es_pairs = clean_word_pairs(eu_es_pairs_raw)
print(f"[INFO] Parsed {len(eu_es_pairs_raw)} raw entries -> {len(eu_es_pairs)} after cleaning.")

print("\n[INFO] Parsing Apertium Basque-English dictionary...")
eu_en_pairs_raw = parse_apertium_dix(APERTIUM_EU_EN_DIX)
eu_en_pairs = clean_word_pairs(eu_en_pairs_raw)
print(f"[INFO] Parsed {len(eu_en_pairs_raw)} raw entries -> {len(eu_en_pairs)} after cleaning.")

print("\n[DIAGNOSTIC] First 20 Basque-Spanish pairs (verify: left=Basque, right=Spanish):")
for l, r in eu_es_pairs[:20]:
    print(f"    {l!r:>20} <-> {r!r}")

print("\n[DIAGNOSTIC] First 20 Basque-English pairs (verify: left=Basque, right=English):")
for l, r in eu_en_pairs[:20]:
    print(f"    {l!r:>20} <-> {r!r}")

print("\n" + "=" * 70)
print("STOP AND CHECK the diagnostic output above before proceeding.")
print("If 'left' and 'right' appear swapped, or the pairs look wrong,")
print("fix parse_apertium_dix() before trusting the results below.")
print("=" * 70)

# Sample a fixed-size, fixed-seed subset for consistency with the
# sentence-level analysis
rng = random.Random(SAMPLE_SEED)
eu_es_sample = rng.sample(eu_es_pairs, min(N_WORD_PAIRS, len(eu_es_pairs)))
rng = random.Random(SAMPLE_SEED)
eu_en_sample = rng.sample(eu_en_pairs, min(N_WORD_PAIRS, len(eu_en_pairs)))

eu_words_es_pair = [p[0] for p in eu_es_sample]
es_words = [p[1] for p in eu_es_sample]
eu_words_en_pair = [p[0] for p in eu_en_sample]
en_words = [p[1] for p in eu_en_sample]


# --- 4. Embedding extraction (single words, same mean-pooling as sentences) ---
def get_word_embeddings(model, tokenizer, words):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for w in words:
            inputs = tokenizer(w, return_tensors="pt", truncation=True, max_length=64).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            mean_pooled = torch.mean(last_hidden_state, dim=1).squeeze().float().cpu().numpy()
            embeddings.append(mean_pooled)
    return np.array(embeddings)


# --- 5. Bootstrap and random-pair-baseline helpers (same as sentence-level scripts) ---
def bootstrap_mean_std(values, n_bootstrap=3, seeds=None):
    if seeds is None:
        seeds = [42, 123, 2024][:n_bootstrap]
    values = np.array(values)
    n = len(values)
    per_seed_means = []
    for seed in seeds:
        rng_ = np.random.default_rng(seed)
        sample_indices = rng_.integers(0, n, size=n)
        per_seed_means.append(values[sample_indices].mean())
    per_seed_means = np.array(per_seed_means)
    return per_seed_means, per_seed_means.mean(), per_seed_means.std()


def random_derangement(n, seed):
    rng_ = np.random.default_rng(seed)
    while True:
        perm = rng_.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def compute_random_pair_baseline(source_vectors, target_vectors, similarity_fn, seeds):
    n = len(source_vectors)
    per_seed_means = []
    for seed in seeds:
        perm = random_derangement(n, seed)
        scores = [similarity_fn(source_vectors[i], target_vectors[perm[i]]) for i in range(n)]
        per_seed_means.append(np.mean(scores))
    per_seed_means = np.array(per_seed_means)
    return per_seed_means, per_seed_means.mean(), per_seed_means.std()


def continuous_jaccard_similarity(x, y):
    dot_product = np.dot(x, y)
    denom = np.dot(x, x) + np.dot(y, y) - dot_product
    return dot_product / denom if denom != 0 else 0.0


cosine_sim_fn = lambda x, y: cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0][0]


# --- 6. Main loop ---
print("\n=== STARTING WORD-LEVEL ALIGNMENT MATRIX ===")

for name, config in MODELS_TO_TEST.items():
    print(f"\n[ANALYZING] {name}...")

    is_hf_repo = name in ["GPT-2 Spanish", "GPT-2 English"]
    if not is_hf_repo and not os.path.isdir(config["path"]):
        print(f"[SKIP] Checkpoint directory not found: {config['path']}")
        continue

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_base"], local_files_only=is_local(config["tokenizer_base"]))
    model = AutoModelForCausalLM.from_pretrained(config["path"], local_files_only=is_local(config["path"])).to(device)

    if name in ["CSCL-en", "GPT-2 English"]:
        source_words, target_words, comparison_label = en_words, eu_words_en_pair, "English <-> Basque(word-level)"
    else:
        source_words, target_words, comparison_label = es_words, eu_words_es_pair, "Spanish <-> Basque(word-level)"

    source_vectors = get_word_embeddings(model, tokenizer, source_words)
    target_vectors = get_word_embeddings(model, tokenizer, target_words)

    cosine_scores = [cosine_sim_fn(source_vectors[i], target_vectors[i]) for i in range(len(source_vectors))]
    jaccard_scores = [continuous_jaccard_similarity(source_vectors[i], target_vectors[i]) for i in range(len(source_vectors))]

    cos_per_seed, cos_mean, cos_std = bootstrap_mean_std(cosine_scores, N_BOOTSTRAP, BOOTSTRAP_SEEDS)
    jac_per_seed, jac_mean, jac_std = bootstrap_mean_std(jaccard_scores, N_BOOTSTRAP, BOOTSTRAP_SEEDS)

    cos_rand_per_seed, cos_rand_mean, cos_rand_std = compute_random_pair_baseline(
        source_vectors, target_vectors, cosine_sim_fn, BOOTSTRAP_SEEDS)
    jac_rand_per_seed, jac_rand_mean, jac_rand_std = compute_random_pair_baseline(
        source_vectors, target_vectors, continuous_jaccard_similarity, BOOTSTRAP_SEEDS)

    print(f" -> Comparison Tracking         : {comparison_label}")
    print(f" -> Cosine  -- Matched          : mean={cos_mean:.4f}, std={cos_std:.4f}")
    print(f" -> Cosine  -- Random Baseline  : mean={cos_rand_mean:.4f}, std={cos_rand_std:.4f}")
    print(f" -> Cosine  -- Matched minus Random : {cos_mean - cos_rand_mean:+.4f}")
    print(f" -> Jaccard -- Matched          : mean={jac_mean:.4f}, std={jac_std:.4f}")
    print(f" -> Jaccard -- Random Baseline  : mean={jac_rand_mean:.4f}, std={jac_rand_std:.4f}")
    print(f" -> Jaccard -- Matched minus Random : {jac_mean - jac_rand_mean:+.4f}")

    del model
    torch.cuda.empty_cache()

print("\n[SUCCESS] Word-level alignment matrix complete.")