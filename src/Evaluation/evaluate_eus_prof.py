import sys
import os
import json
import random
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from tqdm import tqdm


from evaluation_helper_functions import build_batch_tensors, score_choices
from eval_paths import ckpt, RESULTS_DIR, is_local

# ==================== CONFIGURATION ====================
# Checkpoints are the unpacked release folders (see eval_paths.py); GPT-2 models come from the Hugging Face hub.
def _local(name):
    return {"path": ckpt(name), "tokenizer_base": ckpt(name)}

MODELS = {
    "Baseline":              {"path": ckpt("baseline"), "tokenizer_base": "DeepESP/gpt2-spanish"},
    "CSCL-es":               {"path": ckpt("cscl-es"),  "tokenizer_base": "DeepESP/gpt2-spanish"},
    "CSCL-en":               {"path": ckpt("cscl-en"),  "tokenizer_base": "openai-community/gpt2"},
    "GPT-2 Spanish":         {"path": "DeepESP/gpt2-spanish",     "tokenizer_base": "DeepESP/gpt2-spanish"},
    "GPT-2 English":         {"path": "openai-community/gpt2",    "tokenizer_base": "openai-community/gpt2"},
    "TokAlign-Untrained":    _local("tokalign"),
    "Tokalign-CSCL":         _local("tokalign-cscl"),
    "EnerGIZAr++-Untrained": _local("energizar"),
    "Energizar-CSCL":        _local("energizar-cscl"),
}

seed = 42
random.seed(seed)

answer2letter = {0: "A", 1: "B", 2: "C", 3: "D"}
letters = ["A", "B", "C", "D"]


def format_question(item):
    """Format a question for the model."""
    question = item["question"]
    candidates = item["candidates"]
    
    formatted_question = (
        f"Galdera: {question}\n"
        f"A: {candidates[0]}\n"
        f"B: {candidates[1]}\n"
        f"C: {candidates[2]}\n"
        f"D: {candidates[3]}\n"
        f"Erantzuna:"
    )
    return formatted_question

def build_fewshot_example(item):
    """Build a complete example with answer."""
    return format_question(item) + " " + answer2letter[item["answer"]]

def build_fewshot_context(dataset, current_idx, k=5):
    """Build k-shot context excluding current item."""
    # Pool excludes current example
    pool = [dataset[i] for i in range(len(dataset)) if i != current_idx]
    few_shot_examples = random.sample(pool, min(k, len(pool)))
    texts = [build_fewshot_example(ex) for ex in few_shot_examples]
    return "\n\n".join(texts)


def evaluate_static_model(model, tokenizer, dataset, shots=5, device='cuda'):
    """Evaluate a model with static tokenization."""
    correct = 0
    total = 0
    results = []
    
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
    
    for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
        # Build prompt with few-shot examples
        fewshot_context = build_fewshot_context(dataset, idx, k=shots)
        query_text = format_question(item)
        prompt_text = fewshot_context + "\n\n" + query_text
        
        # Tokenize prompt
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        
        # Build sequences for each choice
        full_ids = []
        for i in range(4):
            choice_text = " " + letters[i]
            choice_ids = tokenizer.encode(choice_text, add_special_tokens=False)
            full_ids.append(prompt_ids + choice_ids)
        
        # Score choices
        input_ids, attention_mask = build_batch_tensors(full_ids, pad_id, device)
        scores = score_choices(model, input_ids, attention_mask)
        
        pred = torch.argmax(scores).item()
        is_correct = (pred == item["answer"])
        
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "id": idx,
            "gold": item["answer"],
            "prediction": pred,
            "correct": is_correct,
            "scores": scores.tolist()
        })
    
    accuracy = correct / total
    return accuracy, results


# ==================== MAIN EVALUATION ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                       choices=list(MODELS.keys()),
                       help="Model to evaluate")
    parser.add_argument("--shots", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    args = parser.parse_args()
    
    model_config = MODELS[args.model]
    
    print(f"\n{'='*60}")
    print(f"Model path: {model_config['path']}")
    print(f"Few-shot: {args.shots}")
    print(f"{'='*60}\n")
    
    # Load dataset
    print("Loading EusProficiency dataset...")
    dataset = load_dataset("HiTZ/EusProficiency", name="default", split="test")
    
    if args.limit:
        dataset = dataset.select(range(args.limit))
        print(f"Limited to {args.limit} examples")
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("\nLoading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_config["path"], local_files_only=is_local(model_config["path"]))
    tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_base"], local_files_only=is_local(model_config["tokenizer_base"]))
    model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    accuracy, results = evaluate_static_model(
        model, tokenizer, dataset, shots=args.shots, device=device
    )
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / f"eusproficiency_{args.model}_{args.shots}shot.jsonl"
    
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f} ({sum(r['correct'] for r in results)}/{len(results)})")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
