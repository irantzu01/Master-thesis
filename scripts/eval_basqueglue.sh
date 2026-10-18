#!/bin/bash
#SBATCH --job-name=eval_basqueglue
#SBATCH --output=logs/eval_basqueglue_%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=inet
#SBATCH --time=4:00:00
#SBATCH --partition=react

# BasqueGLUE (BEC, BHTC, coref, QNLI, Vaxx, WiC), 5-shot, seeds 42/123/2024. Model paths: src/Evaluation/eval_basqueglue_multiseed.py (MODELS, via eval_paths.py).
# Usage (from the repository root, with logs/ existing):  sbatch scripts/eval_basqueglue.sh
# The #SBATCH and module lines are specific to the author's cluster; adapt them to your system.
# Datasets and GPT-2 reference models are downloaded from the Hugging Face hub (--constraint=inet = internet access).

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

# Unpacked release checkpoints (baseline, cscl-es, cscl-en, tokalign, tokalign-cscl, energizar, energizar-cscl).
# Default: <repo>/checkpoints; override with  CKPT_DIR=/path sbatch scripts/<name>.sh
export CKPT_DIR="${CKPT_DIR:-$BASE_DIR/checkpoints}"
# Results (.jsonl and summaries) go to src/Evaluation/results unless RESULTS_DIR is set.
export TORCH_LOAD_WEIGHTS_ONLY=FALSE


SHOTS="${SHOTS:-5}"
echo "Evaluating BasqueGLUE with $SHOTS-shot examples..."
for MODEL in baseline cscl_es cscl_en tokalign-untrained tokalign-cscl energizar-untrained energizar-cscl; do
  python src/Evaluation/eval_basqueglue_multiseed.py --model "$MODEL" --shots "$SHOTS"
done
echo "Job finished."
