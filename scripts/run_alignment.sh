#!/bin/bash
#SBATCH --job-name=Align_Check
#SBATCH --output=logs/alignment_%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --time=00:20:00
#SBATCH --partition=react

set -euo pipefail

module purge
module load gcc/13.2.0 python/3.11.9 cuda/12.6.2

BASE_DIR="/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis"
cd "$BASE_DIR"
source "$BASE_DIR/.venv/bin/activate"
echo "Environment loaded."

export TORCH_LOAD_WEIGHTS_ONLY=FALSE

python code/Evaluation/alignment_multiseed.py
python code/Evaluation/jacard_multiseed.py