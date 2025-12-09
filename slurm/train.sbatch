#!/bin/bash
#SBATCH --job-name=pd_grpo
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/job_%A_%a.out    # STDOUT
#SBATCH --error=logs/job_%A_%a.err     # STDERR

# === Setup ===
module purge
module load devel/cuda/11.8
source .venv/bin/activate
echo "Environment activated"

set -euo pipefail

mkdir -p logs

export HF_HOME="${HF_HOME:-$SCRATCH/hf-cache}"
mkdir -p "$HF_HOME"

# === Experiment configuration (edit as needed) ===

# Which setup: "tft" (LLM vs Tit-for-Tat) or "two_llm" (two learning agents)
SETUP="tft"

# Models
MODEL_P1="Qwen/Qwen2.5-0.5B-Instruct"
MODEL_P2=""

# Moral reward type: "game", "deontological", "utilitarian", "game+deontological"
MORAL_TYPE="game+deontological"

# Sampling
TEMP=0.7
TOP_P=0.9

# Training hyperparameters
LR=1e-5
NUM_UPDATES=30
EPISODES_PER_BATCH=8
NUM_ROUNDS=5
MAX_GRAD_NORM=1.0

# Output directory (will also contain logs from your trainer)
OUTPUT_DIR="./runs/grpo_${SETUP}_${MORAL_TYPE}"

mkdir -p "$OUTPUT_DIR"

echo "Running training with:"
echo "  setup=${SETUP}"
echo "  model_p1=${MODEL_P1}"
echo "  model_p2=${MODEL_P2:-<same as p1>}"
echo "  moral_type=${MORAL_TYPE}"
echo "  temperature=${TEMP}, top_p=${TOP_P}"
echo "  lr=${LR}, num_updates=${NUM_UPDATES}, episodes_per_batch=${EPISODES_PER_BATCH}"
echo "  num_rounds=${NUM_ROUNDS}"
echo "  output_dir=${OUTPUT_DIR}"

# === Launch Python ===
python -m training.experiment \
  --setup "${SETUP}" \
  --model_p1 "${MODEL_P1}" \
  $( [[ -n "$MODEL_P2" ]] && echo --model_p2 "${MODEL_P2}" ) \
  --moral_type "${MORAL_TYPE}" \
  --temperature "${TEMP}" \
  --top_p "${TOP_P}" \
  --lr "${LR}" \
  --num_updates "${NUM_UPDATES}" \
  --episodes_per_batch "${EPISODES_PER_BATCH}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --num_rounds "${NUM_ROUNDS}" \
  --output_dir "${OUTPUT_DIR}"

echo "Done."