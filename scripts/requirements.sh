#!/bin/bash
#SBATCH --job-name=req
#SBATCH --output=logs/req_%j.log
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=inet
#SBATCH --time=00:15:00          # Changed to 15 min since downloading takes < 1 min!
#SBATCH --partition=react
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ========== ENVIRONMENT SETUP ==========
module purge
module load gcc/13.2.0
module load python/3.11.9
module load cuda/12.6.2      

BASE_DIR="/user/i.elcoroalberdi/u25035/MASTER/thesis"
cd "$BASE_DIR"

source "$BASE_DIR/.venv/bin/activate"

# ========== FIX: DOWNLOAD PRE-COMPILED BINARY WHEEL ==========
echo "[INFO] Upgrading core installation dependencies..."
pip install --upgrade pip setuptools wheel

echo "[INFO] Fetching pre-compiled binary wheel from GitHub releases..."
# Installs flash-attn 2.5.0 directly using pre-built wheels for Python 3.11 and CUDA 12
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.0/flash_attn-2.5.0+cu122torch2.1cxx11abiFalse-cp311-cp311-linux_x86_64.whl