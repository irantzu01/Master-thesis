#!/usr/bin/env python3
"""
Evaluate Baseline / CSCL-es / CSCL-en on all BasqueGLUE tasks.
Adapted from the Latxa authors' original evaluation script:
 - removed dynamic tokenization (ZETT hypernetwork) machinery, not needed
   for standard-tokenizer GPT-2-scale models
 - score_choices implemented locally (log-likelihood of answer tokens
   given the prompt), no external project dependency
 - load_metric (deprecated/removed in recent `datasets`) replaced with
   sklearn.metrics.f1_score

Usage:
    python eval_basqueglue.py --model baseline
    python eval_basqueglue.py --model cscl_es --task bec
    python eval_basqueglue.py --model cscl_en
"""

import os
import json
import random
import html
import re
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import f1_score
from tqdm import tqdm
from collections import Counter

# ==================== CONFIGURATION ====================
# TODO: update these three paths to your actual final checkpoints
MODELS = {
    "baseline": {
        "path": "/mnt/vast-react/projects/react_ag_beinborn_students/u25035/original_3_models/baseline_full_basque/checkpoint-400000",
        "description": "Baseline (continual pretraining, Spanish->Basque)",
    },
    "cscl_es": {
        "path": "/mnt/vast-react/projects/react_ag_beinborn_students/u25035/original_3_models/cscl_results/phase_3/final_milestone",
        "description": "CSCL-es (Spanish-base CSCL pipeline)",
    },
    "cscl_en": {
        "path": "/mnt/vast-react/projects/react_ag_beinborn_students/u25035/original_3_models/cscl_results_english/phase_3/final_milestone",
        "description": "CSCL-en (English-base CSCL pipeline)",
    },
    "tokalign-untrained": {
        "path": "/mnt/vast-react/projects/react_ag_beinborn_students/u25035/TokAlign/log/gpt2/0_spanish_S2",
        "description": "TokAlign-untrained (untrained TokAlign model)",
    },
    "tokalign-trained-cont": {
        "path": "/mnt/vast-react/projects/react_ag_beinborn_students/u25035/dumped/gpt2-cscl-tokalign/phase_3/final_milestone",
        "description": "TokAlign-trained (TokAlign model trained on CSCL-es)",
    },
    # "tokalign-restart": {
    #     "path": "/mnt/vast-react/projects/react_ag_beinborn_students/u25035/dumped/tokalign-cscl-restart/phase_3/final_milestone",
    #     "description": "TokAlign-restart (TokAlign model trained on CSCL-es with restart)",
    # },
    "energizar-untrained":{
        "path": "/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/energizar_initialized_model_v2",
        "description": "Energizar-untrained (untrained Energizar model)",
    }
}

SHOTS = 0  # number of few-shot examples to include in the prompt
MAX_PROMPT_TOKENS = 900  # matches training max_length; sequences longer than this are truncated from the LEFT
seed = 42
random.seed(seed)
torch.manual_seed(seed)

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
    """
    Returns the LENGTH-NORMALIZED log-probability of `answer_ids`
    conditioned on `prompt_ids` (mean log-prob per answer token, not the
    raw sum). Summing raw log-probs systematically penalizes answer
    choices that tokenize into more subword pieces, which biases scoring
    toward shorter labels regardless of actual fit -- this is the
    standard "acc_norm"-style fix used in lm-evaluation-harness.
    """
    full_ids = prompt_ids + answer_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits  # (1, seq_len, vocab)

    # next-token prediction: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # (1, seq_len-1)

    # only the answer tokens' log-probs count; answer starts at index
    # len(prompt_ids) in the (unshifted) sequence, i.e. len(prompt_ids)-1
    # in the shifted token_log_probs tensor.
    answer_start = len(prompt_ids) - 1
    answer_log_probs = token_log_probs[:, answer_start:]

    num_answer_tokens = answer_log_probs.shape[1]
    return (answer_log_probs.sum() / num_answer_tokens).item()

def report_answer_token_lengths(tokenizer):
    """Prints how many tokens each answer label takes for every task, so
    you can see at a glance whether any task's choices are asymmetric
    enough to worry about even after length normalization."""
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
    """Left-truncate (keep the END of the prompt, drop earliest few-shot
    examples first) if the prompt exceeds max_len tokens."""
    if len(prompt_ids) > max_len:
        return prompt_ids[-max_len:]
    return prompt_ids

# ==================== EVALUATION ====================
def evaluate_model(model, tokenizer, dataset, task, device):
    format_question_func, eval_func, possible_answers = CONFIGS[task]

    y_gold_and_pred = []
    results = []
    dataset_list = list(dataset)
    n_truncated = 0

    for idx, item in enumerate(tqdm(dataset_list, desc=f"  {task}")):
        fewshot_context = build_fewshot_context(
            dataset_list, idx, format_question_func, possible_answers, k=SHOTS
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
              f"{MAX_PROMPT_TOKENS} tokens and were truncated. Consider reducing SHOTS "
              f"if this fraction is large.")

    # Diagnostic: a collapsed prediction distribution (model always picking
    # the same class) can produce below-chance accuracy/F1 even with zero
    # "real" bug -- worth seeing at a glance alongside the score.
    pred_counts = Counter(p for _, p in y_gold_and_pred)
    gold_counts = Counter(g for g, _ in y_gold_and_pred)
    print(f"  Prediction distribution: {dict(pred_counts)}")
    print(f"  Gold label distribution: {dict(gold_counts)}")

    score = eval_func(y_gold_and_pred)
    return score, results

# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--task", type=str, default=None, choices=list(CONFIGS.keys()))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = MODELS[args.model]
    print(f"\n{'='*80}\nLOADING MODEL: {model_config['description']}\n{'='*80}")

    tokenizer = AutoTokenizer.from_pretrained(model_config["path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_config["path"]).to(device)
    model.eval()
    print("Model loaded")

    report_answer_token_lengths(tokenizer)

    tasks_to_eval = [args.task] if args.task else list(CONFIGS.keys())

    model_results = {}
    os.makedirs("results", exist_ok=True)

    for task_name in tasks_to_eval:
        print(f"\n{'='*60}\nEvaluating task: {task_name}\n{'='*60}")

        dataset = load_dataset("orai-nlp/basqueGLUE", name=task_name, split="test")
        if task_name == 'wic':
            dataset = process_wic_docs(dataset)

        print(f"Dataset size: {len(dataset)}")

        score, results = evaluate_model(model, tokenizer, dataset, task_name, device=device)
        model_results[task_name] = score

        results_file = f"results/basqueglue_{task_name}_{args.model}_{SHOTS}shot.jsonl"
        with open(results_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        _, eval_func, _ = CONFIGS[task_name]
        metric_name = eval_func.__name__.replace('_', ' ').title()
        print(f"\n{task_name} - {metric_name}: {score:.4f}")
        print(f"  Results saved to: {results_file}")

    print(f"\n{'='*80}\nRESULTS SUMMARY: {model_config['description']}\n{'='*80}")
    for task_name, score in model_results.items():
        _, eval_func, _ = CONFIGS[task_name]
        metric_name = eval_func.__name__.replace('_', ' ').title()
        print(f"{task_name:>10}: {score:.4f} ({metric_name})")

    summary_file = f"results/basqueglue_{args.model}_{SHOTS}shot_summary.json"
    summary = {
        "model": args.model,
        "description": model_config['description'],
        "shots": SHOTS,
        "results": model_results,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_file}\n{'='*80}\n")

if __name__ == "__main__":
    main()