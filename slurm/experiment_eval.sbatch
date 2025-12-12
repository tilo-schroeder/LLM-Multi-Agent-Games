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

python -m experiment.evaluate \
  --checkpoint_dir ./outputs/final_tests/stag_hunt_n3_thr2_llm_vs_llm_game/final/player_0 \
  --tokenizer_dir  ./outputs/final_tests/stag_hunt_n3_thr2_llm_vs_llm_game/final \
  --num_players 2  \
  --temperature 0.2 --top_p 0.9

echo "Done."