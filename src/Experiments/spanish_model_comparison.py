import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math

# 1. Load the dataset (using a small slice for speed)
# We use the Spanish ('es') split of PAWS-X validation data
dataset = load_dataset("paws-x", "es", split="validation[:50]")
input_texts = dataset["sentence1"]

# 2. Load the Perplexity metric
perplexity = evaluate.load("perplexity", module_type="metric")

# 3. List the models you want to compare
models_to_test = [
    #"goldfish-models/spa_latn_1000mb",      # Your current model
    #"DeepESP/gpt2-spanish",                # Small general Spanish model
    "flax-community/gpt-2-spanish"          
]

print(f"{'Model Name':<40} | {'Mean Perplexity':<15}")
print("-" * 60)

for model_id in models_to_test:
    try:
        results = perplexity.compute(
            predictions=input_texts, 
            model_id=model_id, 
            add_start_token=False,
            device="cpu" # Change to "cuda" if you have a GPU
        )
        mean_ppl = results["mean_perplexity"]
        print(f"{model_id:<40} | {mean_ppl:.2f}")
    except Exception as e:
        print(f"Error testing {model_id}: {e}")


# Alternative manual perplexity calculation (if the metric fails)

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
models_to_test = [
    "DeepESP/gpt2-spanish",
    "flax-community/gpt-2-spanish"
]

# Load dataset and explicitly convert to a list of strings
dataset = load_dataset("paws-x", "es", split="validation[:30]")
texts = [str(x) for x in dataset["sentence1"]] # Explicit conversion to list of strings

def get_manual_ppl(model_id, input_texts):
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize as a proper list
    encodings = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model(encodings.input_ids, labels=encodings.input_ids)
        loss = outputs.loss.item()
    
    return math.exp(loss)

# 2. Run Comparison
print(f"\n{'Model Name':<40} | {'Perplexity':<10}")
print("-" * 55)

for m_id in models_to_test:
    try:
        ppl = get_manual_ppl(m_id, texts)
        print(f"{m_id:<40} | {ppl:.2f}")
    except Exception as e:
        print(f"{m_id:<40} | Error: {e}")



# RESULTS:
# Model Name                               | Mean Perplexity
# ------------------------------------------------------------
# goldfish-models/spa_latn_1000mb          | 361,680.11
# DeepESP/gpt2-spanish                     | 241.60
# flax-community/gpt-2-spanish             | 11,081.11 
