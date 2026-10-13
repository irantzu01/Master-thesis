import transformers
from transformers import pipeline
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer



### 1. Test some general text generation
# Suppress some of the logging
transformers.utils.logging.set_verbosity_error()

pipe = pipeline("text-generation", model="goldfish-models/spa_latn_1000mb")
# set max_new_tokens explicitly to avoid the warning
result = pipe("¿Qué clase de modelo eres?", max_new_tokens=50)
print("\n--- Model Output ---")
print(result[0]['generated_text'])




### 2. Calculate perplexity for different sentences
model_id = "goldfish-models/spa_latn_1000mb"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

def get_perplexity(text):
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        # Passing labels=input_ids tells the model to calculate the loss
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
        
    return math.exp(loss)

# 2. Test different sentences
sentences = [
    "Hola, ¿cómo estás?",
    "El plato está encima de la mesa.",
    "¿Sabes qué es lo que quiero?",
    "Cuéntame un chiste.",
    "El gato se subió al tejado para escapar del perro.",
]

print(f"{'Sentence':<40} | {'Perplexity':<10}")
print("-" * 55)
for s in sentences:
    ppl = get_perplexity(s)
    print(f"{s:<40} | {ppl:.2f}")