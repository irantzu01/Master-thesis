#!/bin/bash
#SBATCH --job-name=multiblimp
#SBATCH --output=logs/multiblimp_%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=inet
#SBATCH --time=24:00:00
#SBATCH --partition=react

echo "Starting job $SLURM_JOB_ID on node $(hostname)"

# === 1. LOAD MODULES (MAY VARY BY CLUSTER) ===
module purge
module load gcc/13.2.0
module load python/3.11.9

# === 2. ACTIVATE YOUR VIRTUAL ENVIRONMENT ===
BASE_DIR="/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis"
cd "$BASE_DIR"
source "$BASE_DIR/.venv/bin/activate"
echo "Environment loaded."

# === 3. MOVE TO PROJECT DIRECTORY ===
cd "$BASE_DIR/multiblimp"

echo "Running python experiment..."

#=== 4. RUN YOUR PYTHON SCRIPT ===

echo "GPT-2 Spanish"
python scripts/lm_eval/eval_model.py \
        --model_name DeepESP/gpt2-spanish \
        --data_dir hf_cache/eus/ \
        --src_dir multiblimp \
        --results_dir bgpt_multiblimp_results/GPT-2_SPA \
        --cache_dir hf_cache/

echo "GPT-2 English"
python scripts/lm_eval/eval_model.py \
        --model_name openai-community/gpt2 \
        --data_dir hf_cache/eus/ \
        --src_dir multiblimp \
        --results_dir bgpt_multiblimp_results/ \
        --cache_dir hf_cache/

echo "Evaluating trained models..."
echo "Baseline"
python scripts/lm_eval/eval_model.py \
        --model_name DeepESP/gpt2-spanish \
        --local_model_path "/mnt/vast-react/projects/react_ag_beinborn_students/u25035/original_3_models/baseline_full_basque/checkpoint-400000" \
        --data_dir hf_cache/eus/ \
        --src_dir multiblimp \
        --results_dir bgpt_multiblimp_results/trained \
        --cache_dir hf_cache/

echo "CSCL-es"
python scripts/lm_eval/eval_model.py \
        --model_name DeepESP/gpt2-spanish \
        --local_model_path "/mnt/vast-react/projects/react_ag_beinborn_students/u25035/original_3_models/cscl_results/phase_3/final_milestone" \
        --data_dir hf_cache/eus/ \
        --src_dir multiblimp \
        --results_dir bgpt_multiblimp_results/trained \
        --cache_dir hf_cache/

echo "CSCL-en"
python scripts/lm_eval/eval_model.py \
        --model_name openai-community/gpt2 \
        --local_model_path "/mnt/vast-react/projects/react_ag_beinborn_students/u25035/original_3_models/cscl_results_english/phase_3/final_milestone" \
        --data_dir hf_cache/eus/ \
        --src_dir multiblimp \
        --results_dir bgpt_multiblimp_results/trained \
        --cache_dir hf_cache/

echo "Tokalign-Untrained"
python scripts/lm_eval/eval_model.py \
        --model_name DeepESP/gpt2-spanish \
        --local_model_path "/mnt/vast-react/projects/react_ag_beinborn_students/u25035/TokAlign/log/gpt2/0_spanish_S2"\
        --data_dir hf_cache/eus/ \
        --src_dir multiblimp \
        --results_dir bgpt_multiblimp_results/trained \
        --cache_dir hf_cache/

echo "Tokalign-CSCL"
python scripts/lm_eval/eval_model.py \
        --model_name DeepESP/gpt2-spanish \
        --local_model_path "/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/dumped/tokalign-cscl-v4/phase_3/final_milestone" \
        --data_dir hf_cache/eus/ \
        --src_dir multiblimp \
        --results_dir bgpt_multiblimp_results/trained \
        --cache_dir hf_cache/

echo "Energizar-Untrained"
python scripts/lm_eval/eval_model.py \
        --model_name DeepESP/gpt2-spanish \
        --local_model_path "/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/energizar_initialized_model_v2" \
        --data_dir hf_cache/eus/ \
        --src_dir multiblimp \
        --results_dir bgpt_multiblimp_results/trained \
        --cache_dir hf_cache/

echo "Energizar-CSCL"
python scripts/lm_eval/eval_model.py \
        --model_name DeepESP/gpt2-spanish \
        --local_model_path "/mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/dumped/energizar-v2/phase_3/final_milestone" \
        --data_dir hf_cache/eus/ \
        --src_dir multiblimp \
        --results_dir bgpt_multiblimp_results/trained \
        --cache_dir hf_cache/

echo "Job $SLURM_JOB_ID completed."


