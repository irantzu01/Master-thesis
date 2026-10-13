import torch
import numpy as np
import os
import json
from transformers import AutoTokenizer, GPT2Tokenizer, AutoModelForCausalLM
from datasets import load_dataset

# --- 1. Runtime Configurations ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Running hidden state alignment diagnostics on: {device}")

N_BOOTSTRAP = 3
BOOTSTRAP_SEEDS = [42, 123, 2024]


MODELS_TO_TEST = {
    "Baseline": {
        "path": "/mnt/vast-react/projects/react_ag_beinborn_students/u25035/dumped/baseline_full_basque/checkpoint-400000",
        "tokenizer_base": "DeepESP/gpt2-spanish"
    },
    "CSCL-es": {
        "path": "/mnt/vast-react/projects/react_ag_beinborn_students/u25035/original_3_models/cscl_results/phase_3/final_milestone",
        "tokenizer_base": "DeepESP/gpt2-spanish"
    },
    "CSCL-en": {
        "path": "/mnt/vast-react/projects/react_ag_beinborn_students/u25035/original_3_models/cscl_results_english/phase_3/final_milestone",
        "tokenizer_base": "openai-community/gpt2"
    },
    "GPT-2 Spanish": {
        "path": "DeepESP/gpt2-spanish",
        "tokenizer_base": "DeepESP/gpt2-spanish"
    },
    "GPT-2 English": {
        "path": "openai-community/gpt2",
        "tokenizer_base": "openai-community/gpt2"
    },
    "TokAlign-Trained": {
        "path": "/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/dumped/gpt2-cscl-tokalign/phase_3/final_milestone",
        "tokenizer_base": "/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/dumped/gpt2-cscl-tokalign/phase_3/final_milestone"
    },
    "TokAlign-Untrained": {
        "path": "/mnt/vast-react/projects/react_ag_beinborn_students/u25035/TokAlign/log/gpt2/0_spanish_S2",
        "tokenizer_base": "/mnt/vast-react/projects/react_ag_beinborn_students/u25035/TokAlign/log/gpt2/0_spanish_S2"
    },
    "Tokalign-CSCL":{
        "path": "/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/dumped/tokalign-cscl-v4/phase_3/final_milestone",
        "tokenizer_base": "/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/dumped/tokalign-cscl-v4/phase_3/final_milestone"
    },
    "EnerGIZAr++-Untrained": {
        "path": "/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/energizar_initialized_model_v2",
        "tokenizer_base": "/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/energizar_initialized_model_v2"
    },
    "Energizar-CSCL":{
        "path": "/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/dumped/energizar-v2/phase_3/final_milestone",
        "tokenizer_base": "/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/dumped/energizar-v2/phase_3/final_milestone"
    }
}

# --- 2. Load Parallel Target Corpus Data ---
print("[INFO] Fetching parallel test vectors from local FLORES-200 files...")
dataset_path = "/user/i.elcoroalberdi/u25035/MASTER/thesis/data/flores200_dataset/dev/"

flores_eu = load_dataset("text", data_files=os.path.join(dataset_path, "eus_Latn.dev"))["train"].select(range(200))
flores_es = load_dataset("text", data_files=os.path.join(dataset_path, "spa_Latn.dev"))["train"].select(range(200))
flores_en = load_dataset("text", data_files=os.path.join(dataset_path, "eng_Latn.dev"))["train"].select(range(200))

eu_sentences = [sample["text"] for sample in flores_eu]
es_sentences = [sample["text"] for sample in flores_es]
en_sentences = [sample["text"] for sample in flores_en]

# --- 3. Feature Extraction Helper ---
def get_sentence_embeddings(model, tokenizer, sentences):
    """Extracts mean-pooled hidden states from the final layer of the model."""
    embeddings = []
    model.eval()
    with torch.no_grad():
        for sen in sentences:
            inputs = tokenizer(sen, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            mean_pooled = torch.mean(last_hidden_state, dim=1).squeeze().float().cpu().numpy()
            embeddings.append(mean_pooled)
    return np.array(embeddings)

# --- 4. Continuous Jaccard (Tanimoto) Helper ---
def continuous_jaccard_similarity(x, y):
    """Computes the continuous Jaccard (Tanimoto) index between two vectors."""
    dot_product = np.dot(x, y)
    norm_x_sq = np.dot(x, x)
    norm_y_sq = np.dot(y, y)
    denominator = norm_x_sq + norm_y_sq - dot_product
    if denominator == 0:
        return 0.0
    return dot_product / denominator

# --- 5. Bootstrap helper ---
def bootstrap_mean_std(values, n_bootstrap=3, seeds=None):
    """
    Bootstrap-resamples a fixed array of per-pair values (e.g. Jaccard
    scores) `n_bootstrap` times, WITH replacement, and returns the
    per-seed means plus their overall mean/std. Appropriate here because
    the underlying embeddings/scores are computed once and are fully
    deterministic -- there is no randomness in the embedding extraction
    itself, only in which subset of the fixed 200-pair set each "seed"
    happens to resample.
    """
    if seeds is None:
        seeds = [42, 123, 2024][:n_bootstrap]
    values = np.array(values)
    n = len(values)

    per_seed_means = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        sample_indices = rng.integers(0, n, size=n)
        resampled = values[sample_indices]
        per_seed_means.append(resampled.mean())

    per_seed_means = np.array(per_seed_means)
    return per_seed_means, per_seed_means.mean(), per_seed_means.std()

# --- 6. Main Diagnostic Evaluation Loop ---
print("\n=== STARTING REPRESENTATION SPACE ALIGNMENT MATRIX (JACCARD) ===")

for name, config in MODELS_TO_TEST.items():
    print(f"\n[ANALYZING] {name}...")

    is_hf_repo = name in ["GPT-2 Spanish", "GPT-2 English"]

    if not is_hf_repo and not os.path.isdir(config["path"]):
        print(f"[SKIP] Checkpoint directory not found at target: {config['path']}")
        continue

    # NOTE: this manual tokenizer.json parsing + GPT2Tokenizer fallback was
    # presumably added to work around a specific Rust-tokenizer error.
    # Worth testing whether plain AutoTokenizer.from_pretrained(...) (as
    # used in the cosine-similarity version of this script) now works fine
    # here too -- if so, this whole branch can be simplified away. Also
    # worth double-checking: get_vocab() returns the vocab parsed directly
    # from tokenizer.json, while __call__ delegates to a SEPARATE
    # GPT2Tokenizer loaded from tokenizer_base -- if these two ever
    # disagree, vocabulary IDs and actual tokenization could silently
    # mismatch.
    if not is_hf_repo and os.path.exists(os.path.join(config["path"], "tokenizer.json")):
        print(f" -> Bypassing Rust binary. Manually parsing raw vocabulary from tokenizer.json...")
        with open(os.path.join(config["path"], "tokenizer.json"), "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)

        if "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
            raw_vocab = tokenizer_data["model"]["vocab"]
        else:
            raw_vocab = tokenizer_data.get("vocab", {})

        class PurePythonTokenizer:
            def __init__(self, vocab_dict, base_path):
                self.vocab = vocab_dict
                self.fallback_tok = GPT2Tokenizer.from_pretrained(base_path, local_files_only=True)
            def __call__(self, text, **kwargs):
                return self.fallback_tok(text, **kwargs)
            def get_vocab(self):
                return self.vocab

        tokenizer = PurePythonTokenizer(raw_vocab, config["tokenizer_base"])
    else:
        print(f" -> Loading tokenizer from base cache fallback path...")
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_base"], local_files_only=True)

    model = AutoModelForCausalLM.from_pretrained(config["path"], local_files_only=True).to(device)

    # Expensive part -- runs the model ONCE over the fixed 200-sentence set.
    eu_vectors = get_sentence_embeddings(model, tokenizer, eu_sentences)

    if name in ["CSCL-en", "GPT-2 English"]:
        source_vectors = get_sentence_embeddings(model, tokenizer, en_sentences)
        comparison_label = "English <-> Basque"
    else:
        source_vectors = get_sentence_embeddings(model, tokenizer, es_sentences)
        comparison_label = "Spanish <-> Basque"

    jaccard_scores = []
    for i in range(len(eu_vectors)):
        score = continuous_jaccard_similarity(source_vectors[i], eu_vectors[i])
        jaccard_scores.append(score)

    # Variance Dispersion: spread of the 200 individual per-pair scores.
    # See note in the cosine version of this script -- this is what the
    # anisotropy argument in the Discussion actually depends on, distinct
    # from the bootstrap std below.
    variance_dispersion = float(np.std(jaccard_scores))

    # Bootstrap Std: spread of the MEAN across 3 resamples.
    per_seed_means, mean_jaccard, std_jaccard = bootstrap_mean_std(
        jaccard_scores, n_bootstrap=N_BOOTSTRAP, seeds=BOOTSTRAP_SEEDS
    )

    print(f" -> Comparison Tracking        : {comparison_label}")
    print(f" -> Per-seed means             : "
          f"{dict(zip(BOOTSTRAP_SEEDS, [round(m, 4) for m in per_seed_means]))}")
    print(f" -> Mean Jaccard Alignment     : {mean_jaccard:.4f}")
    print(f" -> Variance Dispersion        : {variance_dispersion:.4f}")
    print(f" -> Bootstrap Std (3 seeds)    : {std_jaccard:.4f}")

    del model
    torch.cuda.empty_cache()

print("\n[SUCCESS] Matrix processing complete. Jaccard results are ready for thesis inclusion!")