import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_parallel_sentences(flores_dir, lang_a_code, lang_b_code, n_sentences=None):
    """Loads FLORES-200 sentence-aligned parallel text for two languages."""
    path_a = os.path.join(flores_dir, f"{lang_a_code}.devtest")
    path_b = os.path.join(flores_dir, f"{lang_b_code}.devtest")

    with open(path_a, "r", encoding="utf-8") as f:
        sents_a = [line.strip() for line in f if line.strip()]
    with open(path_b, "r", encoding="utf-8") as f:
        sents_b = [line.strip() for line in f if line.strip()]

    assert len(sents_a) == len(sents_b), (
        f"Mismatched line counts: {lang_a_code}={len(sents_a)} vs {lang_b_code}={len(sents_b)}. "
        "FLORES devtest files should be line-aligned; check the raw files."
    )

    if n_sentences is not None:
        sents_a = sents_a[:n_sentences]
        sents_b = sents_b[:n_sentences]

    return sents_a, sents_b


def get_mean_pooled_hidden_states(sentences, tokenizer, model, layer_idx, batch_size=16):
    """
    Returns a (num_sentences, hidden_dim) array of mean-pooled hidden states
    at the given layer index. layer_idx refers to hidden_states tuple:
    index 0 = raw embedding layer, index 1..N = transformer block outputs.
    Pass layer_idx = model.config.n_layer for the final transformer block.
    """
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=512
            ).to(device)

            outputs = model(**inputs, output_hidden_states=True)
            # hidden_states: tuple of (n_layers + 1) tensors, shape (batch, seq_len, hidden_dim)
            # index 0 = embedding layer, index k = output of transformer block k
            layer_hidden = outputs.hidden_states[layer_idx]

            # attention-mask-aware mean pooling (exclude padding tokens)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            summed = (layer_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = (summed / counts).cpu().numpy()

            all_embeddings.append(mean_pooled)

    return np.concatenate(all_embeddings, axis=0)


def compute_cosine_similarities(emb_a, emb_b):
    """Computes pairwise cosine similarity between matched sentence vectors."""
    sims = []
    for i in range(len(emb_a)):
        sim = 1 - cosine(emb_a[i], emb_b[i])
        sims.append(sim)
    return np.array(sims)


def run_tsne_for_model(model_path, model_label, sents_lang_a, sents_lang_b,
                       lang_a_label, lang_b_label, output_dir,
                       layer_idx=None, seed=42, perplexity=50):
    print(f"\n[INFO] Processing model: {model_label} ({model_path})")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    num_layers = model.config.n_layer

    # Default to final transformer block output
    if layer_idx is None:
        layer_idx = num_layers
    print(f"[INFO] Model has {num_layers} transformer blocks; "
          f"extracting hidden_states[{layer_idx}] "
          f"({'final block' if layer_idx == num_layers else f'block {layer_idx}'}).")

    emb_a = get_mean_pooled_hidden_states(sents_lang_a, tokenizer, model, layer_idx)
    emb_b = get_mean_pooled_hidden_states(sents_lang_b, tokenizer, model, layer_idx)

    # --- Cosine similarity distribution ---
    sims = compute_cosine_similarities(emb_a, emb_b)
    print(f"[INFO] Cosine similarities — mean: {sims.mean():.4f}, "
          f"std: {sims.std():.4f}, min: {sims.min():.4f}, max: {sims.max():.4f}")

    # --- t-SNE ---
    combined = np.concatenate([emb_a, emb_b], axis=0)
    labels = [lang_a_label] * len(emb_a) + [lang_b_label] * len(emb_b)

    print(f"[INFO] Running t-SNE on {combined.shape[0]} points, "
          f"dim={combined.shape[1]}, perplexity={perplexity}...")
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca")
    projected = tsne.fit_transform(combined)

    # Plot t-SNE
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {lang_a_label: "#d62728", lang_b_label: "#1f77b4"}
    for lang in [lang_a_label, lang_b_label]:
        idxs = [i for i, l in enumerate(labels) if l == lang]
        ax.scatter(projected[idxs, 0], projected[idxs, 1],
                   label=lang, alpha=0.6, s=18, color=colors[lang])

    ax.set_title(f"t-SNE — {model_label}\n"
                 f"({lang_a_label} vs {lang_b_label}, "
                 f"hidden_states[{layer_idx}])")
    ax.legend()
    fig.tight_layout()

    tsne_path = os.path.join(output_dir, f"tsne_{model_label.replace(' ', '_')}.png")
    fig.savefig(tsne_path, dpi=200)
    plt.close(fig)
    print(f"[SUCCESS] Saved t-SNE plot to {tsne_path}")

    del model
    torch.cuda.empty_cache()

    return sims, tsne_path


def plot_cosine_distributions(all_sims, output_dir):
    """
    Plots overlapping cosine similarity distributions for all models.
    all_sims: dict of {model_label: np.array of cosine similarities}
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c",
        "#d62728", "#9467bd", "#8c564b", "#e377c2"
    ]
    for (label, sims), color in zip(all_sims.items(), colors):
        ax.hist(sims, bins=40, alpha=0.5, label=f"{label} (μ={sims.mean():.3f})",
                color=color, density=True)

    ax.set_xlabel("Cosine similarity (Spanish–Basque paired sentences)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of cross-lingual cosine alignment per model")
    ax.legend(fontsize=9)
    fig.tight_layout()

    dist_path = os.path.join(output_dir, "cosine_distributions.png")
    fig.savefig(dist_path, dpi=200)
    plt.close(fig)
    print(f"\n[SUCCESS] Saved cosine distribution plot to {dist_path}")
    return dist_path


def plot_cosine_violin(all_sims, output_dir):
    """
    Violin plot of cosine similarity distributions — cleaner than overlapping histograms
    when there are many models.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = list(all_sims.keys())
    data = [all_sims[l] for l in labels]

    parts = ax.violinplot(data, positions=range(len(labels)),
                          showmeans=True, showmedians=False)

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c",
        "#d62728", "#9467bd", "#8c564b", "#e377c2"
    ]
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Cross-lingual cosine alignment distributions\n"
                 "(Spanish–Basque FLORES-200 parallel pairs)")
    ax.set_ylim(-0.1, 1.05)
    fig.tight_layout()

    violin_path = os.path.join(output_dir, "cosine_violin.png")
    fig.savefig(violin_path, dpi=200)
    plt.close(fig)
    print(f"[SUCCESS] Saved violin plot to {violin_path}")
    return violin_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-lingual representation alignment: t-SNE + cosine distributions."
    )
    # Model paths — all optional except baseline so you can run subsets
    parser.add_argument("--baseline_path",          type=str, required=True)
    parser.add_argument("--cscl_es_path",           type=str, required=True)
    parser.add_argument("--cscl_en_path",           type=str, required=True)
    parser.add_argument("--tokalign_untrained_path",type=str, default=None)
    parser.add_argument("--tokalign_cscl_path",     type=str, default=None)
    parser.add_argument("--energizar_untrained_path", type=str, default=None)
    parser.add_argument("--gpt2_spanish_path",      type=str, default=None,
                        help="DeepESP/gpt2-spanish or local path, for baseline reference.")
    parser.add_argument("--gpt2_english_path",      type=str, default=None,
                        help="openai-community/gpt2 or local path, for baseline reference.")

    # Data
    parser.add_argument("--flores_dir", type=str, required=True,
                        help="Path to flores200_dataset/devtest directory")
    parser.add_argument("--n_sentences", type=int, default=500,
                        help="Number of parallel pairs to use (max ~1012)")

    # Analysis options
    parser.add_argument("--layer", type=str, default="final",
                        help="Which layer to extract: 'final', 'middle', or an integer index.")
    parser.add_argument("--perplexity", type=float, default=50,
                        help="t-SNE perplexity (try 30–50 for 500 points).")
    parser.add_argument("--skip_tsne", action="store_true",
                        help="Skip t-SNE (faster if you only want cosine distributions).")
    parser.add_argument("--output_dir", type=str, default="./alignment_plots")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load FLORES sentence pairs ---
    print("[INFO] Loading FLORES-200 parallel sentences...")
    spa_sents, eus_sents = load_parallel_sentences(
        args.flores_dir, "spa_Latn", "eus_Latn", args.n_sentences
    )
    eng_sents, eus_sents_en = load_parallel_sentences(
        args.flores_dir, "eng_Latn", "eus_Latn", args.n_sentences
    )

    # --- Build model list ---
    # Each entry: (path, label, lang_a_sents, lang_b_sents, lang_a_label, lang_b_label)
    models_to_run = []

    if args.gpt2_spanish_path:
        models_to_run.append((args.gpt2_spanish_path, "GPT-2 Spanish",
                               spa_sents, eus_sents, "Spanish", "Basque"))
    if args.gpt2_english_path:
        models_to_run.append((args.gpt2_english_path, "GPT-2 English",
                               eng_sents, eus_sents_en, "English", "Basque"))

    models_to_run += [
        (args.baseline_path,  "Baseline",  spa_sents, eus_sents, "Spanish", "Basque"),
        (args.cscl_es_path,   "CSCL-es",   spa_sents, eus_sents, "Spanish", "Basque"),
        (args.cscl_en_path,   "CSCL-en",   eng_sents, eus_sents_en, "English", "Basque"),
        (args.tokalign_untrained_path, "TokAlign-Untrained", spa_sents, eus_sents, "Spanish", "Basque"),
        (args.tokalign_cscl_path, "TokAlign-CSCL", spa_sents, eus_sents, "Spanish", "Basque"),
        (args.energizar_untrained_path, "Energizar-Untrained", spa_sents, eus_sents, "Spanish", "Basque")
    ]

    # --- Resolve layer index ---
    def resolve_layer(layer_str, num_layers):
        if layer_str == "final":
            return num_layers
        elif layer_str == "middle":
            return num_layers // 2
        else:
            return int(layer_str)

    # --- Run ---
    all_sims = {}

    for model_path, model_label, sents_a, sents_b, label_a, label_b in models_to_run:
        # Load model temporarily to get num_layers for layer resolution
        print(f"\n[INFO] Loading {model_label} to resolve layer index...")
        _tmp_model = AutoModelForCausalLM.from_pretrained(model_path)
        num_layers = _tmp_model.config.n_layer
        del _tmp_model
        torch.cuda.empty_cache()

        layer_idx = resolve_layer(args.layer, num_layers)

        sims, _ = run_tsne_for_model(
            model_path=model_path,
            model_label=model_label,
            sents_lang_a=sents_a,
            sents_lang_b=sents_b,
            lang_a_label=label_a,
            lang_b_label=label_b,
            output_dir=args.output_dir,
            layer_idx=layer_idx,
            seed=args.seed,
            perplexity=args.perplexity,
        ) if not args.skip_tsne else (None, None)

        # Recompute sims if tsne was skipped (model already freed above in run_tsne)
        if args.skip_tsne:
            print(f"\n[INFO] Computing cosine similarities only for {model_label}...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
            num_layers = model.config.n_layer
            layer_idx = resolve_layer(args.layer, num_layers)
            emb_a = get_mean_pooled_hidden_states(sents_a, tokenizer, model, layer_idx)
            emb_b = get_mean_pooled_hidden_states(sents_b, tokenizer, model, layer_idx)
            sims = compute_cosine_similarities(emb_a, emb_b)
            print(f"[INFO] {model_label} — mean cosine: {sims.mean():.4f}, "
                  f"std: {sims.std():.4f}")
            del model
            torch.cuda.empty_cache()

        all_sims[model_label] = sims

    # --- Summary table ---
    print("\n" + "=" * 60)
    print(f"{'Model':<25} {'Mean cosine':>12} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("=" * 60)
    for label, sims in all_sims.items():
        print(f"{label:<25} {sims.mean():>12.4f} {sims.std():>8.4f} "
              f"{sims.min():>8.4f} {sims.max():>8.4f}")
    print("=" * 60)

    # --- Distribution plots ---
    plot_cosine_distributions(all_sims, args.output_dir)
    plot_cosine_violin(all_sims, args.output_dir)

    print("\n[SUCCESS] All analyses complete.")