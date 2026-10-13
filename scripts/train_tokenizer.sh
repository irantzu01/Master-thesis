#!/bin/bash
#SBATCH --job-name=Train_ES_EU
#SBATCH --output=logs/train_%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=inet
#SBATCH --time=24:00:00
#SBATCH --partition=react
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ========== ENVIRONMENT SETUP ==========
module purge
module load gcc/13.2.0
module load python/3.11.9

# 1. Define your base directory to avoid relative path confusion
BASE_DIR="/user/i.elcoroalberdi/u25035/MASTER/thesis"
cd "$BASE_DIR"

# 2. Activate environment
source "$BASE_DIR/.venv/bin/activate"

# Fixes
export HF_TOKEN="YOUR_HF_TOKEN"
export TORCH_LOAD_WEIGHTS_ONLY=FALSE


# 3. Run the tokenizer training script
python code/train_tokenizer.py