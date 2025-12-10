#!/bin/bash
#SBATCH --job-name=multi_eval
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

# === Shared experiment configuration ===

SETUP="tft"

# For path matching / documentation
MORAL_TYPE="deontological"

# Directory where training saved its outputs
# TRAIN_OUTPUT_DIR="./runs/grpo_${SETUP}_${MORAL_TYPE}"
TRAIN_OUTPUT_DIR="./runs/tft_deontological_second_try"

# Pick which trained agent to evaluate:
# - For SETUP="tft":      model is saved as agent_tft_p1
# - For SETUP="two_llm":  Player 1 is saved as agent1
if [[ "$SETUP" == "tft" ]]; then
  MODEL_SUBDIR="agent_tft_p1"
elif [[ "$SETUP" == "shared_llm" ]]; then
  MODEL_SUBDIR="shared_policy"
else
  MODEL_SUBDIR="agent1"
fi

MODEL_PATH="${TRAIN_OUTPUT_DIR}/${MODEL_SUBDIR}"
TOKENIZER_PATH="${TRAIN_OUTPUT_DIR}"   # tokenizer was saved here in training

# Shared eval hyperparameters
NUM_EPISODES=50
TEMP=0.7
TOP_P=0.9
XI=3.0                 # deontological penalty magnitude
ILLEGAL_PENALTY=-6.0   # penalty for illegal outputs
SEED=0

echo "Using model:"
echo "  setup=${SETUP}"
echo "  moral_type=${MORAL_TYPE}"
echo "  train_output_dir=${TRAIN_OUTPUT_DIR}"
echo "  model_path=${MODEL_PATH}"
echo "  tokenizer_path=${TOKENIZER_PATH}"
echo "  num_episodes=${NUM_EPISODES}"
echo "  sampling: temperature=${TEMP}, top_p=${TOP_P}"
echo "  xi=${XI}, illegal_penalty=${ILLEGAL_PENALTY}"
echo "  seed=${SEED}"
echo

########################################
# 1) Diner's Dilemma evaluation
########################################

DD_NUM_ROUNDS=5        # rounds per episode
DD_R_CC=3.0            # payoff for (COOP, COOP)
DD_R_DD=1.0            # payoff for (DEFECT, DEFECT)
DD_R_CD_COOP=0.0       # payoff for COOP player when the other DEFECTs
DD_R_CD_DEFECT=4.0     # payoff for DEFECT player when the other COOPs

echo "=== Running Diner's Dilemma evaluation ==="
echo "  num_rounds=${DD_NUM_ROUNDS}"
echo "  payoffs: R_cc=${DD_R_CC}, R_dd=${DD_R_DD}, R_cd_coop=${DD_R_CD_COOP}, R_cd_defect=${DD_R_CD_DEFECT}"
echo

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

echo "=== Diner's Dilemma evaluation done ==="
echo

########################################
# 2) Public Goods evaluation
########################################

PG_NUM_ROUNDS=5
PG_ENDOWMENT=2.0
PG_CONTRIBUTION_COST=1.0
PG_MULTIPLIER=1.6

echo "=== Running Public Goods evaluation ==="
echo "  num_rounds=${PG_NUM_ROUNDS}"
echo "  endowment=${PG_ENDOWMENT}, contribution_cost=${PG_CONTRIBUTION_COST}, multiplier=${PG_MULTIPLIER}"
echo

python -m eval.evaluate_pg \
  --model_path "${MODEL_PATH}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --num_episodes "${NUM_EPISODES}" \
  --num_rounds "${PG_NUM_ROUNDS}" \
  --endowment "${PG_ENDOWMENT}" \
  --contribution_cost "${PG_CONTRIBUTION_COST}" \
  --multiplier "${PG_MULTIPLIER}" \
  --temperature "${TEMP}" \
  --top_p "${TOP_P}" \
  --xi "${XI}" \
  --illegal_penalty "${ILLEGAL_PENALTY}" \
  --seed "${SEED}"

echo "=== Public Goods evaluation done ==="
echo

########################################
# 3) Sealed-Bid Auction evaluation
########################################

AUC_NUM_ROUNDS=5
AUC_VALUE=10.0
AUC_BID_LOW=2.0
AUC_BID_HIGH=6.0

echo "=== Running Sealed-Bid Auction evaluation ==="
echo "  num_rounds=${AUC_NUM_ROUNDS}"
echo "  value=${AUC_VALUE}, bid_low=${AUC_BID_LOW}, bid_high=${AUC_BID_HIGH}"
echo

python -m eval.evaluate_sba \
  --model_path "${MODEL_PATH}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --num_episodes "${NUM_EPISODES}" \
  --num_rounds "${AUC_NUM_ROUNDS}" \
  --value "${AUC_VALUE}" \
  --bid_low "${AUC_BID_LOW}" \
  --bid_high "${AUC_BID_HIGH}" \
  --temperature "${TEMP}" \
  --top_p "${TOP_P}" \
  --xi "${XI}" \
  --illegal_penalty "${ILLEGAL_PENALTY}" \
  --seed "${SEED}"

echo "=== Sealed-Bid Auction evaluation done ==="
echo
echo "All evaluations finished."