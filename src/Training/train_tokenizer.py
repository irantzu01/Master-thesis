"""Train the Basque BPE tokenizer (vocabulary locked to 50,257 tokens) on Latxa.

Also writes the first 100,000 texts of the same shuffled Latxa stream as a JSON-lines file
(basque_latxa_alignment.json), which is the Basque text used for the TokAlign co-occurrence step.

Usage (from the repository root; the Latxa corpus is gated on the Hugging Face hub, so log in first
with `huggingface-cli login` or export HF_TOKEN):
    python src/Training/train_tokenizer.py
    python src/Training/train_tokenizer.py --out_dir basque_tokenizer_50257 --alignment_out data/basque_latxa_alignment.json
"""
import argparse
import json
import os

from datasets import load_dataset, interleave_datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2TokenizerFast

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default="basque_tokenizer_50257", help="where the tokenizer is saved")
parser.add_argument("--alignment_out", default="basque_latxa_alignment.json",
                    help="JSON-lines file with the Basque texts used for the TokAlign step")
parser.add_argument("--tokenizer_samples", type=int, default=200000, help="texts used to train the BPE tokenizer")
parser.add_argument("--alignment_samples", type=int, default=100000, help="texts written for TokAlign")
args = parser.parse_args()

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

max_tokenizer_samples = args.tokenizer_samples  # Number of texts to feed the BPE trainer
max_alignment_samples = args.alignment_samples  # Number of texts to store for the TokAlign co-occurrence step

tokenizer_texts = []
alignment_records = []

for i, item in enumerate(dataset):
    text_content = item["text"]

    if i < max_tokenizer_samples:
        tokenizer_texts.append(text_content)

    if i < max_alignment_samples:
        # JSON-lines structure that TokAlign needs
        alignment_records.append({"text": text_content})

    if i >= max(max_tokenizer_samples, max_alignment_samples):
        break

# Write out the alignment training corpus file right away
alignment_corpus_path = args.alignment_out
os.makedirs(os.path.dirname(alignment_corpus_path) or ".", exist_ok=True)
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
final_tokenizer_dir = args.out_dir
hf_tokenizer = GPT2TokenizerFast(tokenizer_object=raw_tokenizer)
hf_tokenizer.save_pretrained(final_tokenizer_dir)

print(f"\n[SUCCESS] Tokenizer trained and saved to: '{final_tokenizer_dir}'")
print(f"Vocabulary Size confirmation: {len(hf_tokenizer)}")
