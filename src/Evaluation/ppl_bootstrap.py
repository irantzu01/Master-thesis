import torch
import math
import argparse
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_raw_flores_file(lang_code, is_debug):
    file_path = f"/user/i.elcoroalberdi/u25035/MASTER/thesis/data/flores200_dataset/devtest/{lang_code}.devtest"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing raw file: {file_path}. Did you extract the tar.gz?")

    print(f"[INFO] Loading raw text from {file_path}...")
    dataset = load_dataset("text", data_files={"test": file_path}, split="test", streaming=True)
    if is_debug:
        dataset = dataset.take(100)
    return dataset


def compute_per_sentence_stats(model_path, test_data):
    """
    Runs the model ONCE over every sentence and returns a list of
    (loss_sum, num_tokens) per sentence. This is the only step that
    needs the GPU -- everything downstream (bootstrap resampling) is
    just cheap arithmetic over these cached values.
    """
    print(f"[INFO] Loading model/weights from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()

    per_sentence_stats = []  # list of (loss_sum, num_tokens)

    with torch.no_grad():
        for entry in tqdm(test_data, desc="Computing per-sentence loss"):
            text = entry.get('text')
            if not text or len(text.strip()) == 0:
                continue

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            if inputs['input_ids'].size(1) <= 1:
                continue

            outputs = model(**inputs, labels=inputs['input_ids'])
            num_tokens = inputs['input_ids'].size(1)
            loss_sum = outputs.loss.item() * num_tokens  # un-average the loss

            per_sentence_stats.append((loss_sum, num_tokens))

    return per_sentence_stats


def bootstrap_perplexity(per_sentence_stats, n_bootstrap=3, seeds=None):
    """
    Bootstrap-resamples the cached per-sentence (loss_sum, num_tokens)
    pairs `n_bootstrap` times, computing overall PPL for each resample.
    Returns a list of (seed, avg_loss, ppl) and summary mean/std.
    """
    if seeds is None:
        seeds = [42, 123, 2024][:n_bootstrap]
    assert len(seeds) == n_bootstrap, "Number of seeds must match n_bootstrap"

    n_sentences = len(per_sentence_stats)
    loss_sums = np.array([s[0] for s in per_sentence_stats])
    token_counts = np.array([s[1] for s in per_sentence_stats])

    results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        sample_indices = rng.integers(0, n_sentences, size=n_sentences)  # sample WITH replacement

        resampled_loss = loss_sums[sample_indices].sum()
        resampled_tokens = token_counts[sample_indices].sum()

        avg_loss = resampled_loss / resampled_tokens
        ppl = math.exp(avg_loss)
        results.append((seed, avg_loss, ppl))

    ppls = np.array([r[2] for r in results])
    return results, ppls.mean(), ppls.std()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=['flores_es', 'flores_eu', 'flores_en'], required=True)
    parser.add_argument("--n_bootstrap", type=int, default=3,
                         help="Number of bootstrap resamples (i.e. 'seeds') for variance estimation")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.dataset == 'flores_es':
        test_stream = get_raw_flores_file("spa_Latn", args.debug)
    elif args.dataset == 'flores_eu':
        test_stream = get_raw_flores_file("eus_Latn", args.debug)
    elif args.dataset == 'flores_en':
        test_stream = get_raw_flores_file("eng_Latn", args.debug)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Single expensive pass over the model
    per_sentence_stats = compute_per_sentence_stats(args.model_path, test_stream)
    print(f"[INFO] Cached loss/token stats for {len(per_sentence_stats)} sentences.")

    # Cheap bootstrap resampling -- no GPU needed from here on
    results, mean_ppl, std_ppl = bootstrap_perplexity(per_sentence_stats, n_bootstrap=args.n_bootstrap)

    print(f"\nRESULTS for {args.dataset}:")
    for seed, avg_loss, ppl in results:
        print(f"  seed={seed}: avg_loss={avg_loss:.4f}, PPL={ppl:.4f}")
    print(f"\nMean PPL: {mean_ppl:.4f}  (+/- {std_ppl:.4f} std over {args.n_bootstrap} bootstrap resamples)")