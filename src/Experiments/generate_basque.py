import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# 1. Setup
model_path = "/user/i.elcoroalberdi/u25035/MASTER/thesis/dumped/baseline_full_basque/checkpoint-280000"
device = 0 if torch.cuda.is_available() else -1

print(f"[INFO] Loading model from {model_path}...")

# 2. Load Model and Tokenizer
# We use the local path for both to ensure it uses your fine-tuned weights
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 3. Create Pipeline
generator = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    device=device,
    temperature=0.4
)

# 4. Basque Prompts to test
prompts = [
    "Euskara munduko hizkuntzarik zaharrenetarikoa da eta", # "Basque is one of the oldest languages in the world and..."
    "Gaur egun, teknologia berriek",                       # "Nowadays, new technologies..."
    "Donostia oso hiri polita da. Bertan",                        # "San Sebastian is a beautiful city where..."
    "Zientziaren ikuspuntutik,"                            # "From a scientific point of view..."
]

print("\n" + "="*50)
print("BASQUE TEXT GENERATION RESULTS")
print("="*50 + "\n")

for prompt in prompts:
    print(f"PROMPT: {prompt}")
    # We use sampling to get more creative/natural results
    output = generator(
        prompt, 
        max_length=60, 
        num_return_sequences=1, 
        do_sample=True, 
        top_k=50, 
        top_p=0.95,
        temperature=0.8,
        no_repeat_ngram_size=2
    )
    
    generated_text = output[0]['generated_text']
    print(f"GENERATED: {generated_text}")
    print("-" * 30)