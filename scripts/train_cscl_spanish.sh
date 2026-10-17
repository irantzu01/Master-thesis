#!/bin/bash
#SBATCH --job-name=train_cscl_es
#SBATCH --output=logs/train_cscl_es_%j.log
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

# ========== RUN CONTINUAL TRAINING ==========
echo "[INFO] Starting CSCL Spanish Training..."

# 4. Use the absolute path to the python script
python src/Training/train_cscl_spanish.py --phase 1a --output_dir ./dumped/cscl-spanish
python src/Training/train_cscl_spanish.py --phase 1b --output_dir ./dumped/cscl-spanish --resume_from_checkpoint ./dumped/cscl-spanish/phase_1a/final_milestone
python src/Training/train_cscl_spanish.py --phase 1c --output_dir ./dumped/cscl-spanish --resume_from_checkpoint ./dumped/cscl-spanish/phase_1b/final_milestone
python src/Training/train_cscl_spanish.py --phase 2  --output_dir ./dumped/cscl-spanish --resume_from_checkpoint ./dumped/cscl-spanish/phase_1c/final_milestone
python src/Training/train_cscl_spanish.py --phase 3  --output_dir ./dumped/cscl-spanish --resume_from_checkpoint ./dumped/cscl-spanish/phase_2/final_milestone
