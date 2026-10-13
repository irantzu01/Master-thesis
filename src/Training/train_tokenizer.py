import json
from datasets import load_dataset, interleave_datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2TokenizerFast


# 1. Setup the Latxa Streaming Corpus Interleave Pipeline
print("[STAGE 1] Linking to Latxa streams...")
configs = ['euscrawl-v1.1', 'wikipedia', 'culturax', 'colossal-oscar', 'hplt-v1']
streams = [
    load_dataset('HiTZ/latxa-corpus-v1.1', c, split="train", streaming=True) 
    for c in configs
]
dataset = interleave_datasets(streams).shuffle(seed=42, buffer_size=10000)


# 2. Extract Text Iterators for Tokenizer Training & Save a Copy for TokAlign
print("[STAGE 2] Sampling texts for training...")

max_tokenizer_samples = 200000  # Number of sentences to feed the BPE trainer
max_alignment_samples = 100000  # Number of lines to store for your TokAlign co-occurrence step

tokenizer_texts = []
alignment_records = []

for i, item in enumerate(dataset):
    text_content = item["text"]
    
    if i < max_tokenizer_samples:
        tokenizer_texts.append(text_content)
        
    if i < max_alignment_samples:
        # Wrap it securely into the exact JSON lines dictionary structure TokAlign needs
        alignment_records.append({"text": text_content})
        
    if i >= max(max_tokenizer_samples, max_alignment_samples):
        break

# Write out the alignment training corpus file right away
alignment_corpus_path = "basque_latxa_alignment.json"
with open(alignment_corpus_path, "w", encoding="utf-8") as f:
    for record in alignment_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
print(f" -> TokAlign text file saved locally to: '{alignment_corpus_path}'")


# 3. Train the perfectly bounded Basque BPE Tokenizer
print("[STAGE 3] Training custom Basque Tokenizer (Vocab locked to 50,257)...")
raw_tokenizer = Tokenizer(BPE(unk_token="<unk>"))
raw_tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=50257, 
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

# Train using our memory list iterator
raw_tokenizer.train_from_iterator(tokenizer_texts, trainer=trainer)


# 4. Wrap and Save as a Fast GPT-2 Compatible Tokenizer
final_tokenizer_dir = "./basque_tokenizer"
hf_tokenizer = GPT2TokenizerFast(tokenizer_object=raw_tokenizer)
hf_tokenizer.save_pretrained(final_tokenizer_dir)

print(f"\n[SUCCESS] Tokenizer trained and saved to: '{final_tokenizer_dir}'")
print(f"Vocabulary Size confirmation: {len(hf_tokenizer)}")