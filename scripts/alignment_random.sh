#!/bin/bash
#SBATCH --job-name=alignment_random
#SBATCH --output=logs/alignment_random_%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=inet
#SBATCH --time=02:00:00
#SBATCH --partition=react

# Cosine and Jaccard alignment with the random-pair baseline (matched, random, matched minus random). Prints results to the job log and saves the per-pair scores to results/raw_alignment_scores/*.npy (used for the plots).
# Usage (from the repository root, with logs/ existing):  sbatch scripts/alignment_random.sh
# The #SBATCH and module lines are specific to the author's cluster; adapt them to your system.
# FLORES-200 must be unpacked in data/flores200_dataset/ (or FLORES_DIR / FLORES_DEVTEST_DIR set); see "Evaluation" in the README.

set -euo pipefail
echo "Starting job ${SLURM_JOB_ID:-local} on node $(hostname)"

module purge
module load gcc/13.2.0
module load python/3.11.9

# Repository root = directory sbatch was called from (or the parent of scripts/).
BASE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BASE_DIR"
source "$BASE_DIR/.venv/bin/activate"
echo "Environment loaded."

# Unpacked release checkpoints; default <repo>/checkpoints, override with CKPT_DIR=/path sbatch ...
export CKPT_DIR="${CKPT_DIR:-$BASE_DIR/checkpoints}"
export TORCH_LOAD_WEIGHTS_ONLY=FALSE

python src/Evaluation/random_cosine.py
python src/Evaluation/random_jaccard.py

echo "Job finished."
