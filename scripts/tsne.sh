#!/bin/bash
#SBATCH --job-name=eval_eus_read
#SBATCH --output=logs/eval_eus_read%j.log
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
BASE_DIR="/user/i.elcoroalberdi/u25035/MASTER/thesis"
cd "$BASE_DIR"
source "$BASE_DIR/.venv/bin/activate"
echo "Environment loaded."
export HF_TOKEN="YOUR_HF_TOKEN"


# === 3. MOVE TO PROJECT DIRECTORY ===
cd "$BASE_DIR/code/Evaluation"

# === 4. RUN YOUR PYTHON SCRIPT ===
# python tsne_crosslingual_alignment.py \
#     --baseline_path ./gpt2-baseline-es/final_milestone \
#     --cscl_es_path ./gpt2-basque-cscl/phase_3/final_milestone \
#     --cscl_en_path ./gpt2-basque-cscl-english/phase_3/final_milestone \
#     --flores_dir /user/i.elcoroalberdi/u25035/MASTER/thesis/data/flores200_dataset/devtest \
#     --output_dir ./tsne_plots \
#     --n_sentences 500


python tsne_crosslingual_alignment.py \
  --baseline_path /mnt/vast-react/projects/react_ag_beinborn_students/u25035/original_3_models/baseline_full_basque/checkpoint-400000 \
  --cscl_es_path /mnt/vast-react/projects/react_ag_beinborn_students/u25035/original_3_models/cscl_results/phase_3/final_milestone \
  --cscl_en_path /mnt/vast-react/projects/react_ag_beinborn_students/u25035/original_3_models/cscl_results_english/phase_3/final_milestone \
  --tokalign_untrained_path /mnt/vast-react/projects/react_ag_beinborn_students/u25035/TokAlign/log/gpt2/0_spanish_S2 \
  --tokalign_cscl_path /mnt/vast-react/projects/react_ag_beinborn_students/u25035/dumped/gpt2-cscl-tokalign/phase_3/final_milestone \
  --energizar_untrained_path /mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/code/energizar_initialized_model \
  --gpt2_spanish_path DeepESP/gpt2-spanish \
  --gpt2_english_path openai-community/gpt2 \
  --flores_dir /mnt/vast-react/home/i.elcoroalberdi/u25035/MASTER/thesis/data/flores200_dataset/devtest \
  --n_sentences 500 \
  --layer final \
  --output_dir ./alignment_plots


echo "Job finished."