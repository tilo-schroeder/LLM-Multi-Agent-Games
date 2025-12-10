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

# === Experiment configuration ===

# Which setup:
#   "tft"        -> LLM vs fixed opponent (e.g. Tit-for-Tat, always defect, random)
#   "two_llm"    -> two separate learning LLMs (P1 and P2)
#   "shared_llm" -> two players controlled by a single shared policy
SETUP="tft"

# Fixed opponent strategy when SETUP="tft":
#   "tft", "always_cooperate", "always_defect", "random"
OPPONENT_TYPE="always_cooperate"

# Models (model 2 defaults to model 1 if empty; unused for shared_llm)
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

# Output dir
if [[ "$SETUP" == "tft" ]]; then
  OUTPUT_DIR="./runs/grpo_${SETUP}_${OPPONENT_TYPE}_${MORAL_TYPE}"
else
  OUTPUT_DIR="./runs/grpo_${SETUP}_${MORAL_TYPE}"
fi

mkdir -p "$OUTPUT_DIR"

echo "Running training with:"
echo "  setup=${SETUP}"
echo "  opponent_type=${OPPONENT_TYPE}"
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
  --opponent_type "${OPPONENT_TYPE}" \
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