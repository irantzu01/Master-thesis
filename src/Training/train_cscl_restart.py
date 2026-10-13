import argparse
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, interleave_datasets, Dataset


# --- 0. Handle Dynamic Slurm Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="./gpt2-basque-cscl", help="Where to save the model checkpoints")
parser.add_argument("--phase", type=str, required=True, choices=["1a", "1b", "1c", "2", "3"], help="Active curriculum stage to run")
parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Explicit path to resume from inside active phase")
parser.add_argument("--debug", action="store_true", help="Run quick 100-sample smoke test")
args = parser.parse_args()


# --- 1. Load Tokenizer & Model Context ---
# If this is Phase 1a, start with the Spanish base. Otherwise, load from your previous phase folder!
if args.phase == "1a" and not args.resume_from_checkpoint:
    #model_name = "DeepESP/gpt2-spanish"
    model_name = "/mnt/vast-react/projects/react_ag_beinborn_students/u25035/TokAlign/log/gpt2/0_spanish_S2/"
    print(f"[INFO] Initializing Curriculum Phase 1a from base model: {model_name}")
else:
    # Safe fallback wrapper to ensure you build chronologically
    if args.resume_from_checkpoint:
        model_name = args.resume_from_checkpoint
    else:
        print(f"[CRITICAL] You must pass --resume_from_checkpoint with the path to the completed previous phase directory!")
        sys.exit(1)
    print(f"[INFO] Initializing Phase {args.phase} from previous milestone: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(
    "/mnt/vast-react/projects/react_ag_beinborn_students/u25035/TokAlign/log/gpt2/0_spanish_S2/"
)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 


# --- 2. Dynamic Dataset Loader Engine ---
DATA_DIR = "/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/data/synthetic_spanish_adj_cs"

def load_curriculum_dataset(phase_name, is_debug):
    """Loads the explicit text assets required for the given curriculum phase."""
    # Synthetic Phases (1a, 1b, 1c, 2) read from local text files we compiled
    if phase_name == "1a":
        dataset = load_dataset("text", data_files=os.path.join(DATA_DIR, "phase_1a_modifiers.txt"), split="train")
    elif phase_name == "1b":
        dataset = load_dataset("text", data_files=os.path.join(DATA_DIR, "phase_1b_verbs.txt"), split="train")
    elif phase_name == "1c":
        dataset = load_dataset("text", data_files=os.path.join(DATA_DIR, "phase_1c_nouns.txt"), split="train")
    elif phase_name == "2":
        dataset = load_dataset("text", data_files=os.path.join(DATA_DIR, "phase_2_sentence_level.txt"), split="train")
    
    # Phase 3: The Final Monolingual Target Target Language Stream (Native Basque)
    elif phase_name == "3":
        configs = ['euscrawl-v1.1', 'wikipedia', 'culturax', 'colossal-oscar', 'hplt-v1']
        streams = [
            load_dataset('HiTZ/latxa-corpus-v1.1', c, split="train", streaming=True) 
            for c in configs
        ]
        dataset = interleave_datasets(streams).shuffle(seed=42, buffer_size=10000)
    
    if is_debug:
        # Convert streaming or map iterable down for swift local compiler verification
        if phase_name == "3":
            dataset = dataset.take(100)
        else:
            dataset = dataset.select(range(100))
            
    return dataset

print(f"[INFO] Loading training vectors for Phase {args.phase}...")
train_dataset = load_curriculum_dataset(args.phase, args.debug)

# Always use your standard Latxa validation stream as the constant linguistic benchmark
print("[INFO] Loading static validation benchmark (Native Basque Latxa)...")
val_configs = ['euscrawl-v1.1', 'wikipedia', 'culturax', 'colossal-oscar', 'hplt-v1']
val_streams = [
    load_dataset('HiTZ/latxa-corpus-v1.1', c, split="validation", streaming=True) 
    for c in val_configs
]
val_dataset = interleave_datasets(val_streams)
if args.debug:
    val_dataset = val_dataset.take(100)

# --- 3. Tokenization Wrapper Pipeline ---
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized_val = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])


# --- 4. Dynamic Step Allocations (CSCL 3-Macro Block Rule) ---
if args.debug:
    PHASE_STEPS = 50
    WARMUP_STEPS = 10
else:
    # Phase 1 sub-phases get 1/9th of the total budget each (~44.4k steps)
    # Phases 2 and 3 get a full 1/3rd of the total budget each (~133.3k steps)
    phase_step_map = {
        "1a": 44444,
        "1b": 44444,
        "1c": 44445,
        "2":  133333,
        "3":  133334
    }
    PHASE_STEPS = phase_step_map[args.phase]
    
    # Maintain a standard 5% warmup ratio relative to each phase's length
    WARMUP_STEPS = int(PHASE_STEPS * 0.05)

print(f"[INFO] Phase {args.phase} Configuration -> Max Steps: {PHASE_STEPS}, Warmup Steps: {WARMUP_STEPS}")

phase_output_dir = os.path.join(args.output_dir, f"phase_{args.phase}")

training_args = TrainingArguments(
    output_dir=phase_output_dir,
    max_steps=PHASE_STEPS,                 # Dynamically scaled steps
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4, 
    
    # Evaluation & Logging
    eval_strategy="steps" if not args.debug else "no",
    eval_steps=2000 if PHASE_STEPS < 50000 else 5000, # More frequent eval for shorter sub-phases
    logging_steps=10 if args.debug else 200,
    save_steps=10000 if PHASE_STEPS < 50000 else 20000,
    save_total_limit=2,
    
    # Optimization Constants
    learning_rate=5e-5,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,             # Dynamically scaled warmup
    max_grad_norm=0.3,
    adam_epsilon=1e-4,
    
    # Precision & Performance
    fp16=False, 
    bf16=True, 
    dataloader_num_workers=4,
    remove_unused_columns=True,
)

# --- 5. Execution Orchestration Engine ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print(f"[LAUNCH] Executing Phase {args.phase} training script loop...")

# ONLY use resume_from_checkpoint if we are recovering a mid-phase crash inside the active phase folder
# Do NOT pass it if we are loading weights from a completed previous milestone phase
if args.resume_from_checkpoint and "final_milestone" in args.resume_from_checkpoint:
    print("[INFO] Loading weights from previous milestone. Starting training with fresh optimizer parameters.")
    trainer.train(resume_from_checkpoint=None)
else:
    # This allows normal recovery if a mid-phase checkpoint (like checkpoint-20000) is provided
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

# Save final clean milestone output directory for the active phase explicitly
final_phase_save = os.path.join(phase_output_dir, "final_milestone")
trainer.save_model(final_phase_save)
tokenizer.save_pretrained(final_phase_save)
print(f"[SUCCESS] Completed Phase {args.phase}! Saved output directory: {final_phase_save}")