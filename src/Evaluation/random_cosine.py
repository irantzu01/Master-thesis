import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity

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

# --- 4. Bootstrap helper ---
def bootstrap_mean_std(values, n_bootstrap=3, seeds=None):
    """
    Bootstrap-resamples a fixed array of per-pair values (e.g. cosine
    similarities) `n_bootstrap` times, WITH replacement, and returns the
    per-seed means plus their overall mean/std. This is appropriate here
    because the underlying embeddings/similarities are computed once and
    are fully deterministic -- there is no actual randomness in the
    embedding extraction itself, only in which subset of the fixed
    200-pair set each "seed" happens to resample.
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


def random_derangement(n, seed):
    """
    Returns a random permutation of range(n) with NO fixed points
    (i.e., perm[i] != i for all i) -- used to generate genuinely
    mismatched sentence pairs for the random-pair baseline. Uses
    rejection sampling: for n~200 this converges in a handful of
    attempts on average.
    """
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def compute_random_pair_baseline(source_vectors, target_vectors, similarity_fn, seeds):
    """
    For each seed, generates a random derangement of target_vectors
    (i.e., pairs each source sentence with a DIFFERENT, non-matching
    target sentence) and computes the mean similarity under this
    mismatched pairing. Returns per-seed means plus overall mean/std,
    directly comparable to the matched-pair statistics -- this answers
    whether matched-pair alignment exceeds what chance pairing alone
    would produce.
    """
    n = len(source_vectors)
    per_seed_means = []
    for seed in seeds:
        perm = random_derangement(n, seed)
        mismatched_scores = [
            similarity_fn(source_vectors[i], target_vectors[perm[i]])
            for i in range(n)
        ]
        per_seed_means.append(np.mean(mismatched_scores))
    per_seed_means = np.array(per_seed_means)
    return per_seed_means, per_seed_means.mean(), per_seed_means.std()


print("\n=== STARTING REPRESENTATION SPACE ALIGNMENT MATRIX ===")

for name, config in MODELS_TO_TEST.items():
    print(f"\n[ANALYZING] {name}...")

    is_hf_repo = name in ["GPT-2 Spanish", "GPT-2 English"]

    if not is_hf_repo and not os.path.isdir(config["path"]):
        print(f"[SKIP] Checkpoint directory not found at target: {config['path']}")
        continue

    if not is_hf_repo and os.path.exists(os.path.join(config["path"], "tokenizer_config.json")):
        tokenizer_loader_path = config["path"]
        print(f" -> Loading tokenizer locally from checkpoint files...")
    else:
        tokenizer_loader_path = config["tokenizer_base"]
        print(f" -> Loading tokenizer from base cache fallback path...")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_loader_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(config["path"], local_files_only=True).to(device)

    # Expensive part -- runs the model ONCE over the fixed 200-sentence set.
    eu_vectors = get_sentence_embeddings(model, tokenizer, eu_sentences)

    if name in ["CSCL-en", "GPT-2 English"]:
        source_vectors = get_sentence_embeddings(model, tokenizer, en_sentences)
        comparison_label = "English <-> Basque"
    else:
        source_vectors = get_sentence_embeddings(model, tokenizer, es_sentences)
        comparison_label = "Spanish <-> Basque"

    similarities = []
    for i in range(len(eu_vectors)):
        sim = cosine_similarity(source_vectors[i].reshape(1, -1), eu_vectors[i].reshape(1, -1))[0][0]
        similarities.append(sim)

    # Variance Dispersion: spread of the 200 individual per-pair scores.
    # Tells you about the MODEL's behavior -- does it discriminate between
    # more and less related pairs, or assign uniform similarity to everything
    # (the anisotropy signature)? Computed once, no extra model calls needed.
    variance_dispersion = float(np.std(similarities))

    # Bootstrap Std: spread of the MEAN across 3 resamples of the fixed
    # 200-pair set. Tells you about the reliability of the reported mean,
    # not about the model itself. Complementary to, not a replacement for,
    # variance dispersion above.
    per_seed_means, mean_alignment, std_alignment = bootstrap_mean_std(
        similarities, n_bootstrap=N_BOOTSTRAP, seeds=BOOTSTRAP_SEEDS
    )

    print(f" -> Comparison Tracking      : {comparison_label}")
    print(f" -> Per-seed means           : "
          f"{dict(zip(BOOTSTRAP_SEEDS, [round(m, 4) for m in per_seed_means]))}")
    print(f" -> Mean Cosine Alignment    : {mean_alignment:.4f}")
    print(f" -> Variance Dispersion      : {variance_dispersion:.4f}")
    print(f" -> Bootstrap Std (3 seeds)  : {std_alignment:.4f}")

    # Random-pair baseline: mismatch each source sentence with a random,
    # DIFFERENT target sentence, and check whether the matched-pair
    # alignment above actually exceeds what chance pairing alone produces.
    cosine_sim_fn = lambda x, y: cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0][0]
    random_per_seed, random_mean, random_std = compute_random_pair_baseline(
        source_vectors, eu_vectors, cosine_sim_fn, BOOTSTRAP_SEEDS
    )
    print(f" -> Random-Pair Baseline     : mean={random_mean:.4f}, std={random_std:.4f} "
          f"(per-seed: {dict(zip(BOOTSTRAP_SEEDS, [round(m, 4) for m in random_per_seed]))})")
    print(f" -> Matched minus Random     : {mean_alignment - random_mean:+.4f}")

    del model
    torch.cuda.empty_cache()

print("\n[SUCCESS] Matrix processing complete. Results are ready for thesis inclusion!")