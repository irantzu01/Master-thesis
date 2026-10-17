#!/bin/bash
#SBATCH --job-name=train_cscl_en
#SBATCH --output=logs/train_cscl_en_%j.log
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
OUTPUT_DIR="$BASE_DIR/dumped/cscl_english"
mkdir -p "$OUTPUT_DIR"

# ========== RUN CONTINUAL TRAINING ==========
echo "[INFO] Starting CSCL English Training..."

# 4. Use the absolute path to the python script
python src/Training/train_cscl_english.py --phase 1a --output_dir ./dumped/gpt2-cscl-english
python src/Training/train_cscl_english.py --phase 1b --output_dir /mnt/vast-react/projects/react_ag_beinborn_students/u25035/dumped/gpt2-cscl-english --resume_from_checkpoint /mnt/vast-react/projects/react_ag_beinborn_students/u25035/dumped/gpt2-cscl-english/phase_1a/final_milestone
python src/Training/train_cscl_english.py --phase 1c --output_dir /mnt/vast-react/projects/react_ag_beinborn_students/u25035/dumped/gpt2-cscl-english --resume_from_checkpoint /mnt/vast-react/projects/react_ag_beinborn_students/u25035/dumped/gpt2-cscl-english/phase_1b/final_milestone
python src/Training/train_cscl_english.py --phase 2  --output_dir /mnt/vast-react/projects/react_ag_beinborn_students/u25035/dumped/gpt2-cscl-english --resume_from_checkpoint /mnt/vast-react/projects/react_ag_beinborn_students/u25035/dumped/gpt2-cscl-english/phase_1c/final_milestone
python src/Training/train_cscl_english.py --phase 3  --output_dir /mnt/vast-react/projects/react_ag_beinborn_students/u25035/dumped/gpt2-cscl-english --resume_from_checkpoint /mnt/vast-react/projects/react_ag_beinborn_students/u25035/dumped/gpt2-cscl-english/phase_2/final_milestone
