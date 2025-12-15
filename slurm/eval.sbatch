#!/bin/bash
#SBATCH --job-name=stag_test
#SBATCH --partition=gpu_a100_short
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

python -m eval.evaluate \
  --checkpoint_dir ./outputs/final_results/stag_hunt_n2_thr2_vs_copy_focal_utilitarian/final/player_0 \
  --tokenizer_dir  ./outputs/final_results/stag_hunt_n2_thr2_vs_copy_focal_utilitarian/final \
  --num_players 2  \
  --temperature 0.2 --top_p 0.9 \
  --eval_base \
  --normalize_regret

echo "Done."