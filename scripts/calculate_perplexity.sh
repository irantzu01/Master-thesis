#!/bin/bash
#SBATCH --job-name=calculate_perplexity
#SBATCH --output=logs/calculate_perplexity_%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=inet
#SBATCH --time=01:00:00
#SBATCH --partition=react

# Perplexity on FLORES-200 Basque (devtest), 3 bootstrap seeds, all models. Prints results to the job log.
# Usage (from the repository root, with logs/ existing):  sbatch scripts/calculate_perplexity.sh
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

run() {  # run <name> <model folder or hub id>
  echo "=== $1 ==="
  python src/Evaluation/ppl_bootstrap.py --model_path "$2" --dataset flores_eu
}
run "GPT-2 Spanish"  "DeepESP/gpt2-spanish"
run "GPT-2 English"  "openai-community/gpt2"
run "Baseline"       "$CKPT_DIR/baseline"
run "CSCL-es"        "$CKPT_DIR/cscl-es"
run "CSCL-en"        "$CKPT_DIR/cscl-en"
run "TokAlign"       "$CKPT_DIR/tokalign"
run "TokAlign-CSCL"  "$CKPT_DIR/tokalign-cscl"
run "EnerGIZAr"      "$CKPT_DIR/energizar"
run "EnerGIZAr-CSCL" "$CKPT_DIR/energizar-cscl"

echo "Job finished."
