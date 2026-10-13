import torch
import math
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, interleave_datasets
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_raw_flores_file(lang_code, is_debug):
    # This path assumes you ran the tar command in your flores folder
    file_path = f"/user/i.elcoroalberdi/u25035/MASTER/thesis/data/flores200_dataset/devtest/{lang_code}.devtest"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing raw file: {file_path}. Did you extract the tar.gz?")

    print(f"[INFO] Loading raw text from {file_path}...")
    dataset = load_dataset("text", data_files={"test": file_path}, split="test", streaming=True)
    if is_debug: dataset = dataset.take(100)
    return dataset

def calculate_perplexity(model_path, test_data):
    print(f"[INFO] Loading model/weights from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()

    total_loss, total_tokens = 0, 0
    
    with torch.no_grad():
        for entry in tqdm(test_data):
            text = entry.get('text')
            if not text or len(text.strip()) == 0: continue

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            if inputs['input_ids'].size(1) <= 1: continue
                
            outputs = model(**inputs, labels=inputs['input_ids'])
            num_tokens = inputs['input_ids'].size(1)
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=['flores_es', 'flores_eu', 'flores_en'], required=True)
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

    loss, ppl = calculate_perplexity(args.model_path, test_stream)
    
    print(f"\nRESULTS for {args.dataset}:")
    print(f"Perplexity: {ppl:.4f}")