#!/bin/bash
#SBATCH --job-name=Train_ES_EU
#SBATCH --output=logs/train_%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=inet
#SBATCH --time=48:00:00
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

python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Torch Version:', torch.__version__)"

# 3. Double check python path (will show in your log)
which python

# ========== DIRECTORY SETUP ==========
OUTPUT_DIR="$BASE_DIR/dumped/baseline_full_basque"
mkdir -p "$OUTPUT_DIR"

# ========== RUN RESUMED CONTINUAL TRAINING ==========
echo "[INFO] Resuming Baseline Monolingual Training from Step 280,000..."

# Absolute path targeting the full state folder (Weights + Optimizer States + Trainer States)
# CHECKPOINT_PATH="$OUTPUT_DIR/checkpoint-280000"

python "$BASE_DIR/src/Training/continual_training_es.py" \
    --output_dir "$OUTPUT_DIR" \

echo "[INFO] Baseline complete. 400,000 total steps accounted for. Results saved in $OUTPUT_DIR"
