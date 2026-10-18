#!/bin/bash
#SBATCH --job-name=multiblimp
#SBATCH --output=logs/multiblimp_%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=inet
#SBATCH --time=24:00:00
#SBATCH --partition=react

# MultiBLiMP (Basque) for all models. Needs the patched multiblimp clone in the repository root (patches/multiblimp.patch adds --local_model_path) and the Basque data in multiblimp/hf_cache/eus/.
# Usage (from the repository root, with logs/ existing):  sbatch scripts/eval_multiblimp.sh
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

cd "$BASE_DIR/multiblimp"
RESULTS=bgpt_multiblimp_results   # relative to multiblimp/

echo "GPT-2 Spanish"
python scripts/lm_eval/eval_model.py \
        --model_name DeepESP/gpt2-spanish \
        --data_dir hf_cache/eus/ --src_dir multiblimp \
        --results_dir "$RESULTS/GPT-2_SPA" --cache_dir hf_cache/

echo "GPT-2 English"
python scripts/lm_eval/eval_model.py \
        --model_name openai-community/gpt2 \
        --data_dir hf_cache/eus/ --src_dir multiblimp \
        --results_dir "$RESULTS/" --cache_dir hf_cache/

eval_local() {  # eval_local <label> <hub model name (architecture/tokenizer)> <checkpoint folder>
  echo "$1"
  python scripts/lm_eval/eval_model.py \
        --model_name "$2" --local_model_path "$3" \
        --data_dir hf_cache/eus/ --src_dir multiblimp \
        --results_dir "$RESULTS/trained" --cache_dir hf_cache/
}

echo "Evaluating trained models..."
eval_local "Baseline"           DeepESP/gpt2-spanish   "$CKPT_DIR/baseline"
eval_local "CSCL-es"            DeepESP/gpt2-spanish   "$CKPT_DIR/cscl-es"
eval_local "CSCL-en"            openai-community/gpt2  "$CKPT_DIR/cscl-en"
eval_local "Tokalign-Untrained" DeepESP/gpt2-spanish   "$CKPT_DIR/tokalign"
eval_local "Tokalign-CSCL"      DeepESP/gpt2-spanish   "$CKPT_DIR/tokalign-cscl"
eval_local "Energizar-Untrained" DeepESP/gpt2-spanish  "$CKPT_DIR/energizar"
eval_local "Energizar-CSCL"     DeepESP/gpt2-spanish   "$CKPT_DIR/energizar-cscl"

echo "Job finished."
