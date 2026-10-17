#!/bin/bash
#SBATCH --job-name=energizar_cscl_v2
#SBATCH --output=logs/energizar_cscl_v2_%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=inet
#SBATCH --time=6:00:00
#SBATCH --partition=react
#SBATCH --mail-type=BEGIN,END,FAIL


set -euo pipefail

module purge
module load gcc/13.2.0
module load python/3.11.9

BASE="/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis"
cd "$BASE"

source "$BASE/.venv/bin/activate"

export HF_TOKEN="YOUR_HF_TOKEN"
export TORCH_LOAD_WEIGHTS_ONLY=FALSE

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# OUT="$BASE/dumped/energizar-v2"

# python code/train_energizar_v2.py --phase 1a --output_dir "$OUT"
# python code/train_energizar_v2.py --phase 1b --output_dir "$OUT" --resume_from_checkpoint "$OUT/phase_1a/final_milestone"
# python code/train_energizar_v2.py --phase 1c --output_dir "$OUT" --resume_from_checkpoint "$OUT/phase_1b/final_milestone"
# python code/train_energizar_v2.py --phase 2  --output_dir "$OUT" --resume_from_checkpoint "$OUT/phase_1c/final_milestone"
# python code/train_energizar_v2.py --phase 3  --output_dir "$OUT" --resume_from_checkpoint "$OUT/phase_2/final_milestone"

python code/train_energizar_v2.py --phase 3 \
    --output_dir /mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/dumped/energizar-v2 \
    --resume_from_checkpoint /mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/dumped/energizar-v2/phase_3/checkpoint-360000
