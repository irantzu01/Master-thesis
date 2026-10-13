#!/bin/bash
#SBATCH --job-name=Eval_PPL
#SBATCH --output=logs/eval_%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=inet
#SBATCH --time=00:40:00
#SBATCH --partition=react
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ========== ENVIRONMENT SETUP ==========
module purge
module load gcc/13.2.0
module load python/3.11.9

BASE_DIR="/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis"
cd "$BASE_DIR"
source "$BASE_DIR/.venv/bin/activate"
echo "Environment loaded."

# PyTorch 2.6 Fix
export TORCH_LOAD_WEIGHTS_ONLY=FALSE
export HF_TOKEN="YOUR_HF_TOKEN"

# ========== RUN EVALUATION ==========
echo "[INFO] Starting Perplexity Evaluation on Latxa Test Set..."

# echo "Baseline"
# python code/Evaluation/ppl_bootstrap.py \
#     --model_path "/mnt/vast-react/projects/react_ag_beinborn_students/u25035/original_3_models/baseline_full_basque/checkpoint-400000" \
#     --dataset "flores_eu"

# echo "CSCL-es Spanish source"
# python code/Evaluation/ppl_bootstrap.py \
#     --model_path "/mnt/vast-react/projects/react_ag_beinborn_students/u25035/original_3_models/cscl_results/phase_3/final_milestone" \
#     --dataset "flores_eu"

# echo "CSCL-en English source"
# python code/Evaluation/ppl_bootstrap.py \
#     --model_path "/mnt/vast-react/projects/react_ag_beinborn_students/u25035/original_3_models/cscl_results_english/phase_3/final_milestone" \
#     --dataset "flores_eu"

# echo "Tokalign Stabilized"
# python code/Evaluation/ppl_bootstrap.py \
#     --model_path "/mnt/vast-react/projects/react_ag_beinborn_students/u25035/TokAlign/log/gpt2/0_spanish_S2"\
#     --dataset "flores_eu"

# echo "Tokalign-CSCL (continuos version)"
# python code/Evaluation/ppl_bootstrap.py \
#     --model_path "/mnt/vast-react/projects/react_ag_beinborn_students/u25035/dumped/gpt2-cscl-tokalign/phase_3/final_milestone" \
#     --dataset "flores_eu"

# echo "Energizar Untrained"
# python code/Evaluation/ppl_bootstrap.py \
#     --model_path /mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/energizar_initialized_model_v2 \
#     --dataset flores_eu \

echo "Tokalign-CSCL"
python code/Evaluation/ppl_bootstrap.py \
    --model_path "/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/dumped/tokalign-cscl-v4/phase_3/final_milestone" \
    --dataset "flores_eu"

echo "Energizar-CSCL"
python code/Evaluation/ppl_bootstrap.py \
    --model_path "/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/dumped/energizar-v2/phase_3/final_milestone" \
    --dataset "flores_eu"
    
echo "[INFO] Evaluation complete."