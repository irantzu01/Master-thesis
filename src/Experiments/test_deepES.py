from transformers import pipeline

pipe = pipeline("text-generation", model="DeepESP/gpt2-spanish")

# Try 0.7 for a balance of logic and creativity
print(pipe("La inteligencia artificial en español es", max_new_tokens=40, temperature=0.7, do_sample=True)[0]['generated_text'])