"""CSCL curriculum training on the Spanish-Basque curriculum, one phase per invocation.

Per-phase restart (as described in the thesis): every phase starts a fresh optimizer and its
own cosine schedule with a 5% warmup of that phase's step budget. Only the model weights are
carried over from the previous phase (its `final_milestone` folder).

    # CSCL-es (starts from GPT-2 Spanish, the default base model)
    python train_cscl_spanish.py --phase 1a --output_dir OUT
    python train_cscl_spanish.py --phase 1b --output_dir OUT --resume_from_checkpoint OUT/phase_1a/final_milestone
    ...
    # TokAlign-CSCL / EnerGIZAr-CSCL: same protocol, other starting model
    python train_cscl_spanish.py --phase 1a --output_dir OUT --base_model PATH_TO_INITIALISED_MODEL

Steps per phase: 1a 44,444 | 1b 44,444 | 1c 44,445 | 2 133,333 | 3 133,334 (400,000 in total).
Phases 1a-2 read the synthetic files built by src/data_creation (synthetic_spanish_adj_cs);
phase 3 reads the local Latxa corpus downloaded by src/data_creation/download_latxa.py.

Environment: THESIS_DATA (data folder, default <repo>/data).
`--resume_from_checkpoint` accepts either a previous phase's `final_milestone` folder (weights
only, fresh optimizer) or a mid-phase `checkpoint-N` folder of the CURRENT phase (full resume).
"""
import argparse
import json
import os
import sys

from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments,
)

# --- 0. Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="./gpt2-basque-cscl")
parser.add_argument("--phase", type=str, required=True, choices=["1a", "1b", "1c", "2", "3"])
parser.add_argument("--base_model", type=str, default="DeepESP/gpt2-spanish",
                    help="Model used to start phase 1a (hub name or local folder)")
parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                    help="Previous phase's final_milestone folder, or a checkpoint-N folder of this phase")
parser.add_argument("--debug", action="store_true", help="Quick 100-sample smoke test")
args = parser.parse_args()

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.environ.get("THESIS_DATA", os.path.join(_REPO_ROOT, "data"))

# --- 1. Model and tokenizer ---
if args.phase == "1a" and not args.resume_from_checkpoint:
    model_name = args.base_model
    print(f"[INFO] Initializing curriculum phase 1a from base model: {model_name}")
else:
    if not args.resume_from_checkpoint:
        sys.exit("[CRITICAL] Pass --resume_from_checkpoint with the previous phase's final_milestone folder.")
    model_name = args.resume_from_checkpoint
    print(f"[INFO] Initializing phase {args.phase} from: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# --- 2. Data ---
DATA_DIR = os.path.join(DATA_ROOT, "synthetic_spanish_adj_cs")
LATXA_DIR = os.path.join(DATA_ROOT, "latxa-corpus-v1.1")
FOLDERS = ['euscrawl-v1.1', 'wikipedia', 'CulturaX', 'colossal-oscar', 'hplt-v1']
PHASE_FILES = {
    "1a": "phase_1a_modifiers.txt",
    "1b": "phase_1b_verbs.txt",
    "1c": "phase_1c_nouns.txt",
    "2": "phase_2_sentence_level.txt",
}

def get_local_latxa_data(split_name, is_debug):
    per_source = []
    for folder in FOLDERS:
        path = os.path.join(LATXA_DIR, folder, f"{split_name}.jsonl.gz")
        per_source.append(load_dataset("json", data_files=path, split="train"))
    combined = concatenate_datasets(per_source).shuffle(seed=42)
    if is_debug:
        combined = combined.select(range(100))
    return combined

def load_curriculum_dataset(phase_name, is_debug):
    if phase_name == "3":
        return get_local_latxa_data("train", is_debug)
    dataset = load_dataset("text", data_files=os.path.join(DATA_DIR, PHASE_FILES[phase_name]), split="train")
    if is_debug:
        dataset = dataset.select(range(100))
    return dataset

print(f"[INFO] Loading training data for phase {args.phase}...")
train_dataset = load_curriculum_dataset(args.phase, args.debug)
print("[INFO] Loading static validation set (Basque Latxa)...")
val_dataset = get_local_latxa_data("valid", args.debug)

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized_val = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# --- 3. Step budget: three macro blocks of 1/3 of 400,000 steps (phase 1 split in three) ---
if args.debug:
    PHASE_STEPS, WARMUP_STEPS = 50, 10
else:
    PHASE_STEPS = {"1a": 44444, "1b": 44444, "1c": 44445, "2": 133333, "3": 133334}[args.phase]
    WARMUP_STEPS = int(PHASE_STEPS * 0.05)          # 5% of this phase, restarted every phase
print(f"[INFO] Phase {args.phase}: max steps {PHASE_STEPS}, warmup steps {WARMUP_STEPS}")

phase_output_dir = os.path.join(args.output_dir, f"phase_{args.phase}")

training_args = TrainingArguments(
    output_dir=phase_output_dir,
    max_steps=PHASE_STEPS,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,

    eval_strategy="steps" if not args.debug else "no",
    eval_steps=2000 if PHASE_STEPS < 50000 else 5000,
    logging_steps=10 if args.debug else 200,
    save_steps=10000 if PHASE_STEPS < 50000 else 20000,
    save_total_limit=2,

    learning_rate=5e-5,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,
    max_grad_norm=0.3,
    adam_epsilon=1e-4,

    fp16=False,
    bf16=True,
    dataloader_num_workers=4,
    remove_unused_columns=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print(f"[LAUNCH] Training phase {args.phase}...")
if args.resume_from_checkpoint and "final_milestone" in args.resume_from_checkpoint:
    # Weights come from the previous phase; optimizer and schedule start fresh.
    trainer.train(resume_from_checkpoint=None)
else:
    # Fresh phase 1a, or recovery from a mid-phase checkpoint of this phase.
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

with open(os.path.join(phase_output_dir, "log_history.json"), "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)

final_phase_save = os.path.join(phase_output_dir, "final_milestone")
trainer.save_model(final_phase_save)
tokenizer.save_pretrained(final_phase_save)
print(f"[SUCCESS] Completed phase {args.phase}. Saved to {final_phase_save}")
