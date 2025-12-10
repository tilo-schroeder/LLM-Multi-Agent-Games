#!/bin/bash
#SBATCH --job-name=dd_eval
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_%A.out      # STDOUT
#SBATCH --error=logs/eval_%A.err       # STDERR

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

SETUP="tft"

# For path matching
MORAL_TYPE="utilitarian"

# Directory where training saved its outputs
# TRAIN_OUTPUT_DIR="./runs/grpo_${SETUP}_${MORAL_TYPE}"
TRAIN_OUTPUT_DIR="./runs/tft_utilitarian_with_full_logging"

# Pick which trained agent to evaluate on Diner's Dilemma:
# - For SETUP="tft":      model is saved as agent_tft_p1
# - For SETUP="two_llm":  Player 1 is saved as agent1
if [[ "$SETUP" == "tft" ]]; then
  MODEL_SUBDIR="agent_tft_p1"
else
  MODEL_SUBDIR="agent1"
fi

MODEL_PATH="${TRAIN_OUTPUT_DIR}/${MODEL_SUBDIR}"
TOKENIZER_PATH="${TRAIN_OUTPUT_DIR}"   # tokenizer was saved here in training

# Diner's Dilemma environment parameters
DD_NUM_ROUNDS=5        # rounds per episode
DD_R_CC=3.0            # payoff for (COOP, COOP)
DD_R_DD=1.0            # payoff for (DEFECT, DEFECT)
DD_R_CD_COOP=0.0       # payoff for COOP player when the other DEFECTs
DD_R_CD_DEFECT=4.0     # payoff for DEFECT player when the other COOPs

# Evaluation meta-parameters
NUM_EPISODES=50
TEMP=0.7
TOP_P=0.9
XI=3.0                 # deontological penalty magnitude
ILLEGAL_PENALTY=-6.0   # penalty for illegal outputs

SEED=0

echo "Running Diner's Dilemma evaluation with:"
echo "  setup=${SETUP}"
echo "  train_output_dir=${TRAIN_OUTPUT_DIR}"
echo "  model_path=${MODEL_PATH}"
echo "  tokenizer_path=${TOKENIZER_PATH}"
echo "  num_episodes=${NUM_EPISODES}"
echo "  num_rounds=${DD_NUM_ROUNDS}"
echo "  payoffs: R_cc=${DD_R_CC}, R_dd=${DD_R_DD}, R_cd_coop=${DD_R_CD_COOP}, R_cd_defect=${DD_R_CD_DEFECT}"
echo "  sampling: temperature=${TEMP}, top_p=${TOP_P}"
echo "  xi=${XI}, illegal_penalty=${ILLEGAL_PENALTY}"
echo "  seed=${SEED}"

# === Launch Python ===
python -m eval.evaluate_dd \
  --model_path "${MODEL_PATH}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --num_episodes "${NUM_EPISODES}" \
  --num_rounds "${DD_NUM_ROUNDS}" \
  --R_cc "${DD_R_CC}" \
  --R_dd "${DD_R_DD}" \
  --R_cd_coop "${DD_R_CD_COOP}" \
  --R_cd_defect "${DD_R_CD_DEFECT}" \
  --temperature "${TEMP}" \
  --top_p "${TOP_P}" \
  --xi "${XI}" \
  --illegal_penalty "${ILLEGAL_PENALTY}" \
  --seed "${SEED}"

echo "Evaluation done."