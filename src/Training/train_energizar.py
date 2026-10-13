import argparse
import os
import sys
import torch
import json
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, TrainerCallback
)
from datasets import load_dataset, concatenate_datasets, Dataset


# --- 0. Handle Dynamic Slurm Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="./gpt2-basque-cscl-energizar")
parser.add_argument("--phase", type=str, required=True, choices=["1a", "1b", "1c", "2", "3"])
parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                     help="Path to the PREVIOUS phase's final checkpoint dir (must contain optimizer.pt/scheduler.pt)")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


# --- 1. Global schedule constants (MUST match Baseline, TokAlign-CSCL-es exactly) ---
TOTAL_STEPS = 50 if args.debug else 400000
WARMUP_STEPS = int(TOTAL_STEPS * 0.01)  # matches Yoo et al. 2025 appendix

PHASE_BOUNDARIES = {
    "1a": 44444,
    "1b": 88888,
    "1c": 133333,
    "2":  266666,
    "3":  400000,
}
if args.debug:
    debug_bounds = [10, 20, 30, 40, 50]
    PHASE_BOUNDARIES = dict(zip(["1a", "1b", "1c", "2", "3"], debug_bounds))

STOP_AT_STEP = PHASE_BOUNDARIES[args.phase]


# --- 2. Load Tokenizer & Model Context ---
# EnerGIZAr++ initialized checkpoint -- NOT TokAlign's. These are two separate
# experimental conditions and must not share a base model or output directory.
ENERGIZAR_CHECKPOINT = "/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/code/energizar_initialized_model"

if args.phase == "1a" and not args.resume_from_checkpoint:
    model_name = ENERGIZAR_CHECKPOINT
    print(f"[INFO] Initializing Curriculum Phase 1a from EnerGIZAr++-initialized checkpoint: {model_name}")
else:
    if not args.resume_from_checkpoint:
        print("[CRITICAL] You must pass --resume_from_checkpoint with the path to the previous phase's checkpoint!")
        sys.exit(1)
    model_name = args.resume_from_checkpoint
    print(f"[INFO] Initializing Phase {args.phase} from previous checkpoint: {model_name}")

# Tokenizer must match whatever checkpoint model_name points to -- the EnerGIZAr++
# checkpoint carries its own (Basque) tokenizer, saved alongside the model weights.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True)
tokenizer.pad_token = tokenizer.eos_token

# --- 3. Load Curriculum Datasets ---
DATA_DIR = "/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/data/synthetic_spanish_adj_cs"
LATXA_DIR = "/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/data/latxa-corpus-v1.1"
FOLDERS = ['euscrawl-v1.1', 'wikipedia', 'CulturaX', 'colossal-oscar', 'hplt-v1']

def get_local_latxa_data(split_name, is_debug):
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

def load_curriculum_dataset(phase_name, is_debug):
    if phase_name == "1a":
        dataset = load_dataset("text", data_files=os.path.join(DATA_DIR, "phase_1a_modifiers.txt"), split="train")
    elif phase_name == "1b":
        dataset = load_dataset("text", data_files=os.path.join(DATA_DIR, "phase_1b_verbs.txt"), split="train")
    elif phase_name == "1c":
        dataset = load_dataset("text", data_files=os.path.join(DATA_DIR, "phase_1c_nouns.txt"), split="train")
    elif phase_name == "2":
        dataset = load_dataset("text", data_files=os.path.join(DATA_DIR, "phase_2_sentence_level.txt"), split="train")
    elif phase_name == "3":
        dataset = get_local_latxa_data('train', is_debug=is_debug)

    if is_debug and phase_name != "3":
        dataset = dataset.select(range(100))
    return dataset

print(f"[INFO] Loading training vectors for Phase {args.phase}...")
train_dataset = load_curriculum_dataset(args.phase, args.debug)

print("[INFO] Loading static validation benchmark (Native Basque Latxa)...")
val_dataset = get_local_latxa_data('valid', args.debug)


# --- 4. Tokenization ---
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized_val = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])


# --- 5. Callback: stop THIS process once the global schedule reaches this phase's boundary ---
class StopAtGlobalStepCallback(TrainerCallback):
    def __init__(self, stop_step):
        self.stop_step = stop_step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_step:
            control.should_training_stop = True
        return control


# --- 6. Training Arguments: SAME max_steps/warmup_steps in every phase, Baseline, and TokAlign-CSCL-es ---
phase_output_dir = os.path.join(args.output_dir, f"phase_{args.phase}")

training_args = TrainingArguments(
    output_dir=phase_output_dir,
    max_steps=TOTAL_STEPS,              # GLOBAL horizon, identical across all conditions
    warmup_steps=WARMUP_STEPS,          # GLOBAL warmup, identical across all conditions
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,

    eval_strategy="steps" if not args.debug else "no",
    eval_steps=5000,
    logging_steps=10 if args.debug else 200,
    save_strategy="steps",
    save_steps=10000,
    save_total_limit=2,

    learning_rate=5e-5,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
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
    callbacks=[StopAtGlobalStepCallback(STOP_AT_STEP)],
)

print(f"[LAUNCH] Phase {args.phase}: training from global_step -> {STOP_AT_STEP} "
      f"(schedule horizon: {TOTAL_STEPS}, warmup_steps: {WARMUP_STEPS})")

# Resume from a REAL trainer checkpoint (has optimizer.pt/scheduler.pt/trainer_state.json)
# so global_step, Adam moments, and scheduler position all continue seamlessly.
# NOTE: no "final_milestone" special-casing here -- every phase after 1a must
# genuinely resume optimizer/scheduler state, not just reload weights.
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

# Save the log history to a JSON file for later analysis or plotting
log_history = trainer.state.log_history
log_path = os.path.join(phase_output_dir, "log_history.json")
with open(log_path, "w") as f:
    json.dump(log_history, f, indent=2)
print(f"[INFO] Saved training log history to: {log_path}")

# --- 7. Save a FULL checkpoint (not just weights) so the next phase can resume state ---
final_phase_save = os.path.join(phase_output_dir, "final_milestone")
os.makedirs(final_phase_save, exist_ok=True)

trainer.save_model(final_phase_save)
tokenizer.save_pretrained(final_phase_save)

torch.save(trainer.optimizer.state_dict(), os.path.join(final_phase_save, "optimizer.pt"))
torch.save(trainer.lr_scheduler.state_dict(), os.path.join(final_phase_save, "scheduler.pt"))
trainer.state.save_to_json(os.path.join(final_phase_save, "trainer_state.json"))

print(f"[SUCCESS] Completed Phase {args.phase}! Saved output directory: {final_phase_save}")