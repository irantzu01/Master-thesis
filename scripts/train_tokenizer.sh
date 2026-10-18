#!/bin/bash
#SBATCH --job-name=train_tokenizer
#SBATCH --output=logs/train_tokenizer_%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --constraint=inet
#SBATCH --time=02:00:00
#SBATCH --partition=react

# Basque BPE tokenizer (50,257 tokens) trained on 200,000 Latxa texts; also writes the 100,000 Basque texts
# used for the TokAlign step. Output: basque_tokenizer_50257/ and basque_latxa_alignment.json.
# The Latxa corpus is gated on the Hugging Face hub: run `huggingface-cli login` (or export HF_TOKEN=...)
# before submitting; no token is stored in this repository.
# Usage (from the repository root, with logs/ existing):  sbatch scripts/train_tokenizer.sh
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

python src/Training/train_tokenizer.py \
    --out_dir "${OUT:-basque_tokenizer_50257}" \
    --alignment_out "${ALIGNMENT_OUT:-basque_latxa_alignment.json}"

echo "Job finished."
