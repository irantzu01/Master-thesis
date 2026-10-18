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

# CSCL-es: GPT-2 Spanish adapted to Basque with the Spanish-Basque curriculum
#
# Usage (from the repository root, with logs/ existing):
#     mkdir -p logs
#     sbatch scripts/train_cscl_spanish.sh
# Cluster-specific lines (#SBATCH, module load) must be adapted to your system.

set -euo pipefail

# ========== ENVIRONMENT SETUP ==========
module purge
module load gcc/13.2.0
module load python/3.11.9

# Repository root = the directory sbatch was called from (or the parent of scripts/).
BASE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$BASE_DIR"
source "$BASE_DIR/.venv/bin/activate"

# Hugging Face access (the Latxa and CulturaX data are gated): run `huggingface-cli login`
# once, or `export HF_TOKEN=...` in your shell before sbatch. Never write the token in this file.
export TORCH_LOAD_WEIGHTS_ONLY=FALSE

python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Torch Version:', torch.__version__)"
which python

# ========== DIRECTORY STRUCTURE ==========
# Training runs one curriculum phase at a time. Everything goes under one output folder:
#
#   $OUT/
#     phase_1a/   checkpoint-*/     intermediate checkpoints (the last two are kept)
#                 final_milestone/  weights + tokenizer at the end of the phase;
#                                   this is the starting point of the next phase
#     phase_1b/   ... starts from phase_1a/final_milestone
#     phase_1c/   ... starts from phase_1b/final_milestone
#     phase_2/    ... starts from phase_1c/final_milestone
#     phase_3/    ... starts from phase_2/final_milestone  (phase_3/final_milestone = final model)
#
# Each phase restarts the optimizer and the cosine schedule with a 5% warmup (per-phase restart).
# One job runs all five phases in order. To split the work over several jobs (for example
# because of the time limit), or to resume after a failure, choose the phases:
#     PHASES="1a 1b" sbatch scripts/train_cscl_spanish.sh
#     PHASES="1c 2 3" sbatch scripts/train_cscl_spanish.sh      # resumes from the existing phase_1b/final_milestone
OUT="${OUT:-$BASE_DIR/dumped/cscl_spanish}"
PHASES="${PHASES:-1a 1b 1c 2 3}"
mkdir -p "$OUT"

previous_phase() {
  case "$1" in
    1b) echo 1a ;; 1c) echo 1b ;; 2) echo 1c ;; 3) echo 2 ;; *) echo "" ;;
  esac
}

echo "[INFO] CSCL-es: GPT-2 Spanish adapted to Basque with the Spanish-Basque curriculum: phases $PHASES -> $OUT"
for PHASE in $PHASES; do
  PREV="$(previous_phase "$PHASE")"
  if [ -z "$PREV" ]; then
    RESUME_ARGS=()                                   # phase 1a starts from the base model
  else
    RESUME_ARGS=(--resume_from_checkpoint "$OUT/phase_$PREV/final_milestone")
  fi
  echo "[INFO] Phase $PHASE"
  python src/Training/train_cscl_spanish.py --phase "$PHASE" --output_dir "$OUT" ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
done
echo "[INFO] Done. Final model: $OUT/phase_3/final_milestone"
