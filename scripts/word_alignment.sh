#!/bin/bash
#SBATCH --job-name=word_alignment
#SBATCH --output=logs/word_alignment_%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=inet
#SBATCH --time=02:00:00
#SBATCH --partition=react

# Word-level cosine/Jaccard alignment on 200 sampled Apertium eu-es / eu-en pairs, with random-pair baseline. Prints results to the job log. Needs the apertium-eu-es and apertium-eu-en clones in the repository root (or APERTIUM_EU_ES_DIX / APERTIUM_EU_EN_DIX); see the README.
# Usage (from the repository root, with logs/ existing):  sbatch scripts/word_alignment.sh
# The #SBATCH and module lines are specific to the author's cluster; adapt them to your system.

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

python src/Evaluation/word_alignment.py

echo "Job finished."
