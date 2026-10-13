import argparse
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, concatenate_datasets


# 0. Handle Arguments (The bridge from Slurm)
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="./gpt2-basque-baseline", help="Where to save the model")
parser.add_argument("--debug", action="store_true", help="Run a quick smoke test")
parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint")
args = parser.parse_args()


# 1. Load Model & Tokenizer
model_name = "DeepESP/gpt2-spanish"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 


# 2. Load Local Latxa Dataset
import os
from datasets import load_dataset, concatenate_datasets

LATXA_DIR = "/user/i.elcoroalberdi/u25035/MASTER/thesis/data/latxa-corpus-v1.1"
FOLDERS = ['euscrawl-v1.1', 'wikipedia', 'CulturaX', 'colossal-oscar', 'hplt-v1']

def get_local_latxa_data(split_name, is_debug):
    # split_name is 'train' or 'valid' -- matches the actual filenames directly
    per_source_datasets = []
    for folder in FOLDERS:
        path = os.path.join(LATXA_DIR, folder, f"{split_name}.jsonl.gz")
        ds = load_dataset("json", data_files=path, split="train")
        per_source_datasets.append(ds)

    combined = concatenate_datasets(per_source_datasets)
    combined = combined.shuffle(seed=42)

    if is_debug:
        combined = combined.select(range(100))
    return combined

train_data = get_local_latxa_data('train', args.debug)
val_data = get_local_latxa_data('valid', args.debug)  # note: 'valid', not 'validation'


# 3. Tokenization wrapper
def tokenize_fn(examples):
    outputs = tokenizer(examples["text"], truncation=True, max_length=512)
    # If this prints 0 or only [50256], your data isn't reaching the model correctly
    #print(f"DEBUG: Tokenized length: {len(outputs['input_ids'][0])}")
    return outputs

tokenized_train = train_data.map(
    tokenize_fn, 
    batched=True, 
    remove_columns=["text"] # Add this line
)
tokenized_val = val_data.map(
    tokenize_fn, 
    batched=True, 
    remove_columns=["text"] # Add this line
)

# 4. Training Arguments
max_steps = 50 if args.debug else 400000
training_args = TrainingArguments(
    output_dir=args.output_dir,
    max_steps=max_steps,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4, 
    
    # Evaluation & Logging
    eval_strategy="steps" if not args.debug else "no",
    eval_steps=5000,
    logging_steps=10 if args.debug else 500,
    save_steps=20000,
    save_total_limit=5,
    
    # Optimization & Stability
    learning_rate=5e-5,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_steps=int(max_steps * 0.01),  # replaces warmup_ratio=0.01
    max_grad_norm=0.3,
    adam_epsilon=1e-4,
    
    # Precision - Use BF16 for A100 stability
    fp16=False, 
    bf16=True,
    
    # Efficiency
    dataloader_num_workers=4,
    remove_unused_columns=True,
)


# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print(f"[INFO] Starting training (Debug mode: {args.debug})")
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

log_history = trainer.state.log_history
log_path = os.path.join(args.output_dir, "log_history.json")
with open(log_path, "w") as f:
    json.dump(log_history, f, indent=2)
print(f"[INFO] Saved training log history to: {log_path}")