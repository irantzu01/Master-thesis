#!/bin/bash
#SBATCH --job-name=eval_basqueglue
#SBATCH --output=logs/eval_basqueglue_%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=inet
#SBATCH --time=4:00:00
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
echo "Evaluating BasqueGLUE with 0-shot examples..."
python eval_basqueglue_multiseed.py --model baseline
python eval_basqueglue_multiseed.py --model cscl_es
python eval_basqueglue_multiseed.py --model cscl_en
python eval_basqueglue_multiseed.py --model tokalign-untrained
python eval_basqueglue_multiseed.py --model tokalign-cscl
python eval_basqueglue_multiseed.py --model energizar-untrained
python eval_basqueglue_multiseed.py --model energizar-cscl

echo "Job finished."