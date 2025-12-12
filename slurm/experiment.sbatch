#!/bin/bash
#SBATCH --job-name=stag_test
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_%j.out    # STDOUT
#SBATCH --error=logs/test_%j.err     # STDERR

# === Setup ===
module purge
module load devel/cuda/11.8
source .venv/bin/activate
echo "Environment activated"

set -euo pipefail

mkdir -p logs

export HF_HOME="${HF_HOME:-$SCRATCH/hf-cache}"
mkdir -p "$HF_HOME"

python -m experiment.train_simple --moral_type utilitarian --opponent_type llm --num_episodes 150 --model_name Qwen/Qwen2.5-0.5B-Instruct --output_dir ./outputs --num_players 3 --llm_vs_llm

echo "Done."

# TODO: Get the action plot working
# TODO: Maybe standardized implementation of PPO