#!/bin/bash
#SBATCH --job-name=CheckVocabs
#SBATCH --output=vocab_sizes.log
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=inet
#SBATCH --mem=64G                     # Over-provisioning RAM stops OOM kills instantly
#SBATCH --time=00:10:00
#SBATCH --partition=react
#SBATCH --mail-type=END,FAIL

set -e

# Load necessary modules (matching your environment)
module purge
module load gcc/13.2.0 python/3.11.9 cuda/12.6.2

# Activate your virtual environment
source /user/i.elcoroalberdi/u25035/MASTER/thesis/.venv/bin/activate

echo "=== RUNNING VOCABULARY DIAGNOSTIC ==="
python -u -c '
from transformers import AutoTokenizer

# 1. Load the online Spanish base tokenizer
try:
    source_tokenizer = AutoTokenizer.from_pretrained("DeepESP/gpt2-spanish")
    print(f"Source (Spanish GPT-2) Vocab Size (Vs): {len(source_tokenizer)}")
except Exception as e:
    print(f"Could not load Spanish tokenizer: {e}")

# 2. Load your custom local Basque tokenizer
target_path = "/user/i.elcoroalberdi/u25035/MASTER/thesis/basque_tokenizer_hf"
try:
    target_tokenizer = AutoTokenizer.from_pretrained(target_path)
    print(f"Target (Custom Basque) Vocab Size (Vt) : {len(target_tokenizer)}")
except Exception as e:
    print(f"Could not load Basque tokenizer: {e}")
'
echo "====================================="