#!/usr/bin/env python3
"""
Evaluate Baseline / CSCL-es / CSCL-en / TokAlign / etc. on all BasqueGLUE
tasks, across multiple seeds, since few-shot exemplar sampling
(build_fewshot_context) is genuinely stochastic -- unlike perplexity or
MultiBLiMP, a different seed here produces a different prompt, not just a
different resample of an already-computed result. Each seed therefore
requires a full re-run of the model over the dataset.

Usage (from the repository root; checkpoints are read from $CKPT_DIR, see eval_paths.py):
    python src/Evaluation/eval_basqueglue_multiseed.py --model baseline --shots 5
    python src/Evaluation/eval_basqueglue_multiseed.py --model cscl_es --task bec --seeds 42 123 2024
"""

import os
import json
import random
import html
import re
import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import f1_score
from tqdm import tqdm
from eval_paths import ckpt, RESULTS_DIR
from collections import Counter

# ==================== CONFIGURATION ====================
MODELS = {
    "baseline":            {"path": ckpt("baseline"),
                            "description": "Baseline (continual pretraining, Spanish->Basque)"},
    "cscl_es":             {"path": ckpt("cscl-es"),
                            "description": "CSCL-es (Spanish-base CSCL pipeline)"},
    "cscl_en":             {"path": ckpt("cscl-en"),
                            "description": "CSCL-en (English-base CSCL pipeline)"},
    "tokalign-untrained":  {"path": ckpt("tokalign"),
                            "description": "TokAlign (initialised and stabilised, no CSCL)"},
    "tokalign-cscl":       {"path": ckpt("tokalign-cscl"),
                            "description": "TokAlign-CSCL"},
    "energizar-untrained": {"path": ckpt("energizar"),
                            "description": "EnerGIZAr (initialised, no further training)"},
    "energizar-cscl":      {"path": ckpt("energizar-cscl"),
                            "description": "EnerGIZAr-CSCL"},
}

DEFAULT_SHOTS = 5
MAX_PROMPT_TOKENS = 900
DEFAULT_SEEDS = [42, 123, 2024]

# ==================== HELPER FUNCTIONS ====================
def general_detokenize(string):
    string = re.sub(r'\s+([.,;:!?)])', r'\1', string)
    string = re.sub(r'(\s+|^)\(\s+([^)]+)\s+\)', r'\1(\2)', string)
    string = re.sub(r'(\s+|^)\[\s+([^)]+)\s+\]', r'\1[\2]', string)
    string = re.sub(r'(\s+|^)"\s+([^"]+)\s+"', r'\1"\2"', string)
    string = re.sub(r"(\s+|^)'\s+([^']+)\s+'", r"\1'\2'", string)
    return string

def process_doc(string):
    string = html.unescape(string)
    string = general_detokenize(string)
    return string

def process_wic_docs(dataset):
    def _helper(doc):
        doc["sentence1"] = process_doc(doc["sentence1"])
        doc["sentence2"] = process_doc(doc["sentence2"])
        return doc
    return dataset.map(_helper)

def format_question_bec(item):
    return f"Testua: {item['text']}\nGaldera: Nolako jarrera agertzen du aurreko testuak?\nErantzuna:"

def format_question_bhtc(item):
    labels = ', '.join(CONFIGS['bhtc'][2])
    return f"Testua: {item['text']}\nGaldera: Zein da aurreko testuaren gaia? Aukeratu hauen artean: {labels}\nErantzuna:"

def format_question_coref(item):
    def _span_in_context(span_index, span_text):
        span_start = span_index
        span_end = span_start + len(span_text.split(" ")) - 1
        tokens[span_start] = f'*{tokens[span_start]}'
        tokens[span_end] = f'{tokens[span_end]}*'
    tokens = item["text"].split(" ")
    _span_in_context(item["span1_index"], item["span1_text"])
    _span_in_context(item["span2_index"] - 1, item["span2_text"])
    context = process_doc(" ".join(tokens))
    span_1 = process_doc(item["span1_text"])
    span_2 = process_doc(item["span2_text"])
    return f'Testua: {context}\nGaldera: Aurreko testuan, "*{span_1}*" eta "*{span_2}*" gauza bera dira?\nErantzuna:'

def format_question_qnli(item):
    return f"{item['question']}\n{item['sentence']}\nGaldera: aurreko galderari erantzuten al dio emandako testuak?\nErantzuna:"

def format_question_vaxx(item):
    return f"Testua: {item['text']}\nGaldera: Nolako jarrera agertzen du aurreko testuak txertoei buruz?\nErantzuna:"

def format_question_wic(item):
    return f"1. esaldia: {item['sentence1']}\n2. esaldia: {item['sentence2']}\nGaldera: Aurreko bi esaldietan, \"{item['word']}\" hitzak esanahi berdina du?\nErantzuna:"

def build_fewshot_example(item, format_question_func, labels):
    question = format_question_func(item)
    answer = labels[item['label']] if isinstance(item['label'], int) else item['label']
    return question + " " + answer

def build_fewshot_context(dataset_list, current_idx, format_question_func, labels, k=5):
    pool = [dataset_list[i] for i in range(len(dataset_list)) if i != current_idx]
    few_shot_examples = random.sample(pool, min(k, len(pool)))
    texts = [build_fewshot_example(ex, format_question_func, labels) for ex in few_shot_examples]
    return "\n\n".join(texts)

def micro_f1_score(items):
    golds, preds = list(zip(*items))
    return f1_score(golds, preds, average="micro")

def vaxx_f1_score(items):
    golds, preds = list(zip(*items))
    return f1_score(golds, preds, labels=[0, 2], average="macro")

def accuracy_score(items):
    arr = [int(g == p) for g, p in items]
    return sum(arr) / len(arr)

# ==================== TASK CONFIGURATIONS ====================
CONFIGS = {
    "bec": (format_question_bec, micro_f1_score, ['negatiboa', 'neutrala', 'positiboa']),
    "bhtc": (format_question_bhtc, micro_f1_score, [
        'Ekonomia', 'Euskal Herria', 'Euskara', 'Gizartea', 'Historia', 'Ingurumena',
        'Iritzia', 'Komunikazioa', 'Kultura', 'Nazioartea', 'Politika', 'Zientzia'
    ]),
    "coref": (format_question_coref, accuracy_score, ['ez', 'bai']),
    "qnli": (format_question_qnli, accuracy_score, ['bai', 'ez']),
    "vaxx": (format_question_vaxx, vaxx_f1_score, ['aurka', 'neutrala', 'alde']),
    "wic": (format_question_wic, accuracy_score, ['ez', 'bai']),
}

# ==================== SCORING ====================
def score_choice(model, prompt_ids, answer_ids, device):
    full_ids = prompt_ids + answer_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    answer_start = len(prompt_ids) - 1
    answer_log_probs = token_log_probs[:, answer_start:]

    num_answer_tokens = answer_log_probs.shape[1]
    return (answer_log_probs.sum() / num_answer_tokens).item()

def report_answer_token_lengths(tokenizer):
    print(f"\n{'='*60}\nAnswer label token lengths (sanity check)\n{'='*60}")
    for task, (_, _, labels) in CONFIGS.items():
        lengths = []
        for label in labels:
            ids = tokenizer.encode(" " + label, add_special_tokens=False)
            lengths.append((label, len(ids)))
        lengths_str = ", ".join(f"{l!r}={n}tok" for l, n in lengths)
        print(f"  {task:>8}: {lengths_str}")
    print(f"{'='*60}\n")

def truncate_prompt_ids(tokenizer, prompt_ids, max_len):
    if len(prompt_ids) > max_len:
        return prompt_ids[-max_len:]
    return prompt_ids

# ==================== EVALUATION (single seed) ====================
def evaluate_model_one_seed(model, tokenizer, dataset, task, device, seed, shots):
    """Runs one full evaluation pass over `dataset` for `task`, using `seed`
    to control few-shot exemplar sampling. Returns (score, results, diagnostics)."""
    random.seed(seed)

    format_question_func, eval_func, possible_answers = CONFIGS[task]

    y_gold_and_pred = []
    results = []
    dataset_list = list(dataset)
    n_truncated = 0

    for idx, item in enumerate(tqdm(dataset_list, desc=f"  {task} (seed={seed})")):
        fewshot_context = build_fewshot_context(
            dataset_list, idx, format_question_func, possible_answers, k=shots
        )
        query_text = format_question_func(item)
        prompt_text = fewshot_context + "\n\n" + query_text

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        if len(prompt_ids) > MAX_PROMPT_TOKENS:
            n_truncated += 1
            prompt_ids = truncate_prompt_ids(tokenizer, prompt_ids, MAX_PROMPT_TOKENS)

        scores = []
        for answer in possible_answers:
            answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
            scores.append(score_choice(model, prompt_ids, answer_ids, device))

        pred_idx = int(torch.tensor(scores).argmax().item())
        gold_idx = item["label"]

        y_gold_and_pred.append((gold_idx, pred_idx))
        results.append({
            "id": idx,
            "gold": gold_idx,
            "prediction": pred_idx,
            "correct": (pred_idx == gold_idx),
            "scores": scores,
        })

    if n_truncated > 0:
        print(f"  [WARNING] {n_truncated}/{len(dataset_list)} prompts exceeded "
              f"{MAX_PROMPT_TOKENS} tokens and were truncated.")

    pred_counts = Counter(p for _, p in y_gold_and_pred)
    gold_counts = Counter(g for g, _ in y_gold_and_pred)
    print(f"  [seed={seed}] Prediction distribution: {dict(pred_counts)}")
    print(f"  [seed={seed}] Gold label distribution: {dict(gold_counts)}")

    score = eval_func(y_gold_and_pred)
    return score, results

# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--task", type=str, default=None, choices=list(CONFIGS.keys()))
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS,
                         help="Number of few-shot examples (default: 5)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                         help="Seeds to evaluate with (default: 42 123 2024)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = MODELS[args.model]
    print(f"\n{'='*80}\nLOADING MODEL: {model_config['description']}\n{'='*80}")

    # Model and tokenizer are loaded ONCE -- only the few-shot exemplar
    # sampling changes per seed, not the model weights or dataset.
    tokenizer = AutoTokenizer.from_pretrained(model_config["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_config["path"]).to(device)
    model.eval()
    print("Model loaded")

    report_answer_token_lengths(tokenizer)

    tasks_to_eval = [args.task] if args.task else list(CONFIGS.keys())
    seeds = args.seeds

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_task_summaries = {}

    for task_name in tasks_to_eval:
        print(f"\n{'='*60}\nEvaluating task: {task_name} across {len(seeds)} seeds {seeds}\n{'='*60}")

        dataset = load_dataset("orai-nlp/basqueGLUE", name=task_name, split="test")
        if task_name == 'wic':
            dataset = process_wic_docs(dataset)
        print(f"Dataset size: {len(dataset)}")

        per_seed_scores = []
        for seed in seeds:
            score, results = evaluate_model_one_seed(model, tokenizer, dataset, task_name, device, seed, args.shots)
            per_seed_scores.append(score)

            results_file = RESULTS_DIR / f"basqueglue_{task_name}_{args.model}_{args.shots}shot_seed{seed}.jsonl"
            with open(results_file, "w") as f:
                for result in results:
                    f.write(json.dumps(result) + "\n")

        mean_score = float(np.mean(per_seed_scores))
        std_score = float(np.std(per_seed_scores))

        _, eval_func, _ = CONFIGS[task_name]
        metric_name = eval_func.__name__.replace('_', ' ').title()

        print(f"\n{task_name} - {metric_name} per seed: "
              f"{dict(zip(seeds, [round(s, 4) for s in per_seed_scores]))}")
        print(f"{task_name} - {metric_name}: mean={mean_score:.4f}, std={std_score:.4f}")

        all_task_summaries[task_name] = {
            "metric": metric_name,
            "per_seed_scores": dict(zip(seeds, per_seed_scores)),
            "mean": mean_score,
            "std": std_score,
        }

    print(f"\n{'='*80}\nRESULTS SUMMARY: {model_config['description']}\n{'='*80}")
    for task_name, summary in all_task_summaries.items():
        print(f"{task_name:>10}: mean={summary['mean']:.4f}  std={summary['std']:.4f}  ({summary['metric']})")

    summary_file = RESULTS_DIR / f"basqueglue_{args.model}_{args.shots}shot_multiseed_summary.json"
    summary_out = {
        "model": args.model,
        "description": model_config['description'],
        "shots": args.shots,
        "seeds": seeds,
        "results": all_task_summaries,
    }
    with open(summary_file, "w") as f:
        json.dump(summary_out, f, indent=2)

    print(f"\nSummary saved to: {summary_file}\n{'='*80}\n")

if __name__ == "__main__":
    main()
