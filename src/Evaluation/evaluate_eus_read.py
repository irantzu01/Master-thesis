import sys
import os
import json
import random
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from tqdm import tqdm
from collections import defaultdict

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

# ==================== DATASET PROCESSING ====================
def group_by_context(dataset):
    """
    Group questions by their context/passage.
    Returns: dict mapping context -> list of questions
    """
    context_groups = defaultdict(list)
    
    for idx, item in enumerate(dataset):
        context = item["context"]
        context_groups[context].append({
            "original_idx": idx,
            "question": item["question"],
            "candidates": item["candidates"],
            "answer": item["answer"],
            "context": context
        })
    
    return context_groups

def format_question(doc) -> str:
    """
    Format a single question with its candidates.
    Does NOT include the passage/context.
    """
    candidates = doc["candidates"]
    # Filter out empty candidates
    candidates = [c for c in candidates if c and c.strip()]
    num_choices = len(candidates)
    
    if num_choices < 2:
        raise ValueError("Invalid number of candidates")
    
    choices = letters[:num_choices]
    formatted_choices = "\n".join(
        [f"{choice}: {candidates[i]}" for i, choice in enumerate(choices)]
    )
    return f"Galdera: {doc['question']}\n{formatted_choices}\nErantzuna:"

def format_passage_with_question(doc) -> str:
    """
    Format passage + question together.
    This is what we actually evaluate on.
    """
    candidates = doc["candidates"]
    candidates = [c for c in candidates if c and c.strip()]
    num_choices = len(candidates)
    
    if num_choices < 2:
        raise ValueError("Invalid number of candidates")
    
    choices = letters[:num_choices]
    formatted_choices = "\n".join(
        [f"{choice}: {candidates[i]}" for i, choice in enumerate(choices)]
    )
    return f"Pasartea: {doc['context']}\n\nGaldera: {doc['question']}\n{formatted_choices}\nErantzuna:"

def build_fewshot_example(doc):
    """Build a complete few-shot example: passage + question + answer."""
    return format_passage_with_question(doc) + " " + letters[doc["answer"]]

def build_fewshot_context(dataset_items, current_item_idx, k=5):
    """
    Build k-shot context by sampling k random (passage, question) pairs.
    This matches the original evaluation: each example is passage + question + answer.
    
    Args:
        dataset_items: List of all dataset items (each has context, question, candidates, answer)
        current_item_idx: Index of current item to exclude
        k: Number of few-shot examples
    """
    # Pool excludes current example
    pool = [dataset_items[i] for i in range(len(dataset_items)) if i != current_item_idx]
    
    if len(pool) < k:
        k = len(pool)
    
    # Sample k random examples (each is a passage + question pair)
    few_shot_examples = random.sample(pool, k)
    
    # Build few-shot text
    texts = [build_fewshot_example(ex) for ex in few_shot_examples]
    return "\n\n".join(texts)

# ==================== EVALUATION FUNCTIONS ====================
def evaluate_static_model(model, tokenizer, dataset, shots=5, device='cuda'):
    """Evaluate a model with static tokenization."""
    correct = 0
    total = 0
    results = []
    
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
    
    # Convert dataset to list for easier indexing
    dataset_items = list(dataset)
    
    for idx, item in enumerate(tqdm(dataset_items, desc="Evaluating")):
        # Filter candidates
        candidates = [c for c in item["candidates"] if c and c.strip()]
        if len(candidates) < 2:
            continue
        
        # Build few-shot context: k random (passage, question) pairs
        fewshot_context = build_fewshot_context(dataset_items, idx, k=shots)
        
        # Current question with its passage
        query_text = format_passage_with_question(item)
        prompt_text = fewshot_context + "\n\n" + query_text
        
        # Tokenize prompt with TRUNCATION
        #prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        max_prompt_length = 1020 
        prompt_ids = tokenizer.encode(
            prompt_text, 
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_length
        )
        
        # Build sequences for each choice
        full_ids = []
        num_choices = len(candidates)
        for i in range(num_choices):
            choice_text = " " + letters[i]
            choice_ids = tokenizer.encode(choice_text, add_special_tokens=False)
            full_ids.append(prompt_ids + choice_ids)
        
        
        # Score choices ONE AT A TIME to avoid OOM
        scores_list = []
        for seq_ids in full_ids:
            input_ids_tensor = torch.tensor([seq_ids], dtype=torch.long, device=device)
            attention_mask_tensor = torch.ones_like(input_ids_tensor)
            
            with torch.no_grad():
                score = score_choices(model, input_ids_tensor, attention_mask_tensor)
            scores_list.append(score.item())
            
            # Clear GPU cache after each choice
            torch.cuda.empty_cache()
        
        scores = torch.tensor(scores_list)
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
            "scores": scores.tolist(),
            "num_candidates": num_choices
        })
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, results


# ==================== MAIN EVALUATION ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                       choices=list(MODELS.keys()),
                       help="Model to evaluate")
    parser.add_argument("--shots", type=int, default=5, 
                       help="Number of few-shot examples (passages)")
    parser.add_argument("--limit", type=int, default=None, 
                       help="Limit number of questions")
    args = parser.parse_args()
    
    model_config = MODELS[args.model]
    
    print(f"\n{'='*60}")
    print(f"Model path: {model_config['path']}")
    print(f"Few-shot: {args.shots} passages")
    print(f"{'='*60}\n")
    
    # Load dataset
    print("Loading EusReading dataset...")
    dataset = load_dataset("HiTZ/EusReading", name="default", split="test")
    
    print(f"Total dataset size: {len(dataset)} questions")
    
    if args.limit:
        dataset = dataset.select(range(args.limit))
        print(f"Limited to {args.limit} questions")
    
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
    results_file = RESULTS_DIR / f"eusreading_{args.model}_{args.shots}shot.jsonl"
    
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
