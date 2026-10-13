#!/bin/bash
#SBATCH --job-name=eval_eus_read
#SBATCH --output=logs/eval_eus_read%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=inet
#SBATCH --time=10:00:00
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
cd "$BASE_DIR/code/Evaluation"

echo "Running python experiment..."

# === 4. RUN YOUR PYTHON SCRIPT ===
python evaluate_eus_read.py --model "Baseline" --shots 5
python evaluate_eus_read.py --model "CSCL-es" --shots 5
python evaluate_eus_read.py --model "CSCL-en" --shots 5
python evaluate_eus_read.py --model "GPT-2 Spanish" --shots 5
python evaluate_eus_read.py --model "GPT-2 English" --shots 5
python evaluate_eus_read.py --model "TokAlign-Untrained" --shots 5
python evaluate_eus_read.py --model "Tokalign-CSCL" --shots 5
python evaluate_eus_read.py --model "EnerGIZAr++-Untrained" --shots 5
python evaluate_eus_read.py --model "Energizar-CSCL" --shots 5

echo "Job finished."