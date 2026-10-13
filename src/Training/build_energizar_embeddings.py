"""
Step 6: Parse the GIZA/mgiza translation probability table (.t3.final)
into the alignment matrix D^{t,s}, apply post-processing (delta
threshold, identity-forcing, fallback init), and compute the new
target embedding matrix Emb_t = D^{t,s} x Emb_s.

Usage:
    python build_energizar_embeddings.py
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==================== CONFIGURATION ====================
GIZA_DIR = "/mnt/vast-react/projects/react_ag_beinborn_students/u25035/giza/full-run-v2"
DIRECTION_DIR = f"{GIZA_DIR}/giza.eu.tok-es.tok"
T3_FINAL_PATH = f"{DIRECTION_DIR}/eu.tok-es.tok.t3.final"
SRC_VCB_PATH = f"{DIRECTION_DIR}/eu.tok-es.tok.trn.src.vcb"
TRG_VCB_PATH = f"{DIRECTION_DIR}/eu.tok-es.tok.trn.trg.vcb"

SOURCE_MODEL_PATH = "DeepESP/gpt2-spanish"
TARGET_TOKENIZER_PATH = "/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/basque_tokenizer_50257"

# NOTE: this is currently 0.01, but your Methodology text states delta=0.1
# following Singh et al. 2025. Reconcile these before finalizing -- either
# update the text to 0.01, or set this back to 0.1, whichever you actually
# intend to report.
DELTA = 0.01

OUTPUT_MODEL_DIR = "./energizar_initialized_model_v2"


# ==================== STEP 1: LOAD GIZA VOCAB FILES ====================
def load_vcb(path):
    """GIZA .vcb format: id  word  count -> returns {giza_id: subword_string}"""
    id_to_word = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                idx, word = parts[0], parts[1]
                id_to_word[int(idx)] = word
    return id_to_word


# ==================== STEP 2: PARSE .t3.final INTO A GIZA-ID-INDEXED DICT ====================
def parse_t3_final(path):
    """
    Returns a dict: {source_giza_id: [(target_giza_id, prob), ...]}
    Format of .t3.final lines: <source_id> <target_id> <probability>
    """
    table = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            src_id, tgt_id, prob = int(parts[0]), int(parts[1]), float(parts[2])
            table.setdefault(src_id, []).append((tgt_id, prob))
    return table


# ==================== STEP 3: REMAP GIZA IDS -> REAL TOKENIZER IDS ====================
def build_remapped_table(t3_table, src_vcb, trg_vcb, src_tokenizer, trg_tokenizer):
    """
    Converts the GIZA-ID-indexed table into a table indexed by the REAL
    source and target tokenizer vocab IDs. Subwords not found in the
    respective tokenizer vocab are dropped (rare, but can happen due to
    GIZA-side text normalization differences).
    """
    src_vocab = src_tokenizer.get_vocab()   # {token_string: real_id}
    trg_vocab = trg_tokenizer.get_vocab()

    remapped = {}  # {real_src_id: [(real_trg_id, prob), ...]}
    dropped = 0

    for giza_src_id, targets in t3_table.items():
        src_word = src_vcb.get(giza_src_id)
        if src_word is None or src_word not in src_vocab:
            dropped += len(targets)
            continue
        real_src_id = src_vocab[src_word]

        for giza_trg_id, prob in targets:
            trg_word = trg_vcb.get(giza_trg_id)
            if trg_word is None or trg_word not in trg_vocab:
                dropped += 1
                continue
            real_trg_id = trg_vocab[trg_word]
            remapped.setdefault(real_src_id, []).append((real_trg_id, prob))

    print(f"[INFO] Remapped table built. Dropped {dropped} entries "
          f"(subwords not found in respective tokenizer vocab).")
    return remapped


# ==================== STEP 4: BUILD D^{t,s} MATRIX WITH POST-PROCESSING ====================
def build_alignment_matrix(remapped_table, src_vocab_size, trg_vocab_size,
                            src_tokenizer, trg_tokenizer, delta=0.1):
    """
    Builds D^{t,s} of shape (trg_vocab_size, src_vocab_size), where
    D[t, s] = p(t | s), following Eq. 2 in the EnerGIZAr++ methodology.

    Post-processing:
      1. Zero out probabilities below delta.
      2. Track target tokens with no probability mass above 0.1 for
         fallback normal-distribution initialization.
      3. Force identity alignment (prob=1.0) for lexically identical
         source/target subwords.
    """
    D = np.zeros((trg_vocab_size, src_vocab_size), dtype=np.float32)

    for real_src_id, targets in remapped_table.items():
        for real_trg_id, prob in targets:
            if prob >= delta:
                D[real_trg_id, real_src_id] = prob

    # identify target tokens needing fallback init (no prob > 0.1 anywhere)
    row_max = D.max(axis=1)
    fallback_indices = np.where(row_max <= 0.1)[0]
    print(f"[INFO] {len(fallback_indices)} / {trg_vocab_size} target tokens "
          f"require fallback initialization (no aligned probability > 0.1).")

    # force identity alignment for lexically identical subwords
    src_vocab = src_tokenizer.get_vocab()
    trg_vocab = trg_tokenizer.get_vocab()
    identity_count = 0
    for token_str, trg_id in trg_vocab.items():
        if token_str in src_vocab:
            src_id = src_vocab[token_str]
            D[trg_id, :] = 0.0
            D[trg_id, src_id] = 1.0
            identity_count += 1
            # identity-forced tokens don't need fallback init
            fallback_indices = fallback_indices[fallback_indices != trg_id]

    print(f"[INFO] {identity_count} tokens force-aligned by lexical identity.")

    return D, fallback_indices


# ==================== STEP 5: COMPUTE NEW EMBEDDINGS ====================
def compute_target_embeddings(D, fallback_indices, source_embeddings, seed=42):
    """
    Emb_t = D_normalized @ Emb_s, with fallback rows drawn from
    N(mean(Emb_s), std(Emb_s)) instead of the matrix product.

    D is row-normalized before the matrix multiplication: without this,
    target subwords aligned to many source subwords that each individually
    clear the delta threshold accumulate row sums well above 1 (observed
    up to ~155 for common Basque auxiliary verb forms in practice),
    producing pathologically oversized embedding norms. Row normalization
    ensures each target embedding is a genuine weighted AVERAGE of aligned
    source embeddings, not an unnormalized weighted sum.
    """
    rng = np.random.default_rng(seed)

    row_sums = D.sum(axis=1, keepdims=True)
    safe_row_sums = np.where(row_sums == 0, 1.0, row_sums)  # avoid divide-by-zero for all-zero rows
    D_normalized = D / safe_row_sums

    print(f"[INFO] Row-normalized D. Pre-normalization row sums: "
          f"min={row_sums.min():.4f}, mean={row_sums.mean():.4f}, max={row_sums.max():.4f}")

    new_embeddings = D_normalized @ source_embeddings  # (trg_vocab_size, hidden_dim)

    mean_emb = source_embeddings.mean(axis=0)
    std_emb = source_embeddings.std(axis=0)

    for idx in fallback_indices:
        new_embeddings[idx] = rng.normal(mean_emb, std_emb)

    return new_embeddings


# ==================== MAIN ====================
if __name__ == "__main__":
    print("[INFO] Loading tokenizers and source model...")
    src_tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL_PATH)
    trg_tokenizer = AutoTokenizer.from_pretrained(TARGET_TOKENIZER_PATH)
    model = AutoModelForCausalLM.from_pretrained(SOURCE_MODEL_PATH)

    source_embeddings = model.get_input_embeddings().weight.detach().numpy()
    print(f"[INFO] Source embedding matrix shape: {source_embeddings.shape}")

    print("[INFO] Loading GIZA vocab files...")
    src_vcb = load_vcb(SRC_VCB_PATH)
    trg_vcb = load_vcb(TRG_VCB_PATH)

    print("[INFO] Parsing .t3.final...")
    t3_table = parse_t3_final(T3_FINAL_PATH)

    print("[INFO] Remapping GIZA IDs to real tokenizer IDs...")
    remapped = build_remapped_table(t3_table, src_vcb, trg_vcb, src_tokenizer, trg_tokenizer)

    print("[INFO] Building alignment matrix D^{t,s}...")
    D, fallback_indices = build_alignment_matrix(
        remapped,
        src_vocab_size=len(src_tokenizer),
        trg_vocab_size=len(trg_tokenizer),
        src_tokenizer=src_tokenizer,
        trg_tokenizer=trg_tokenizer,
        delta=DELTA,
    )

    print("[INFO] Computing new target embeddings...")
    new_embeddings = compute_target_embeddings(D, fallback_indices, source_embeddings)

    print("[INFO] Splicing new embeddings into model...")
    model.resize_token_embeddings(len(trg_tokenizer))
    model.get_input_embeddings().weight.data = torch.tensor(new_embeddings, dtype=model.dtype)
    # GPT-2 ties lm_head to wte by default, so this single assignment covers both.

    print(f"[INFO] Saving initialized model to {OUTPUT_MODEL_DIR}...")
    model.save_pretrained(OUTPUT_MODEL_DIR)
    trg_tokenizer.save_pretrained(OUTPUT_MODEL_DIR)

    print("[SUCCESS] EnerGIZAr++ embedding initialization complete.")