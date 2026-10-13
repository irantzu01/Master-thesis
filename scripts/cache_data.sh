#!/bin/bash
#SBATCH --job-name=create_data
#SBATCH --output=logs/create_data_%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=inet
#SBATCH --mem=64G                     # Over-provisioning RAM stops OOM kills instantly
#SBATCH --time=08:00:00
#SBATCH --partition=react
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ========== ENVIRONMENT SETUP ==========
module purge
module load gcc/13.2.0
module load python/3.11.9

BASE_DIR="/user/i.elcoroalberdi/u25035/MASTER/thesis"
cd "$BASE_DIR"

source "$BASE_DIR/.venv/bin/activate"

# PyTorch 2.6 Fix
export TORCH_LOAD_WEIGHTS_ONLY=FALSE
export HF_TOKEN="YOUR_HF_TOKEN"

# ========== RUN EVALUATION ==========
# echo "[INFO] Caching raw Spanish data..."
# python code/cache_raw_spa.py
# echo "[INFO] Data caching complete."

# echo "Starting phase 1 data generation..."
# python code/compile_phase_1_spa.py
# echo "Phase 1 data generation complete."

# echo "Starting phase 2 data generation..."
# python code/compile_phase_2_spa.py
# echo "Phase 2 data generation complete."

python code/giza/tokenize_parallel.py