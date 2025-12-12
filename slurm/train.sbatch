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

OPPONENT_TYPE="tft"              # one of: tft, always_cooperate, always_defect, random
MODEL_P1="Qwen/Qwen2.5-0.5B-Instruct"
# MODEL_P1="Qwen/Qwen3-4B-Instruct-2507"
MORAL_TYPE="utilitarian"         # game, deontological, utilitarian, game+deontological

# Sampling
TEMP=0.7
TOP_P=0.9

# Training hyperparameters (super small / fast)
LR=1e-6
NUM_UPDATES=20            # 1 PPO update (smoke test)
EPISODES_PER_BATCH=10     # 1 episode per update
PPO_EPOCHS=2             # 1 PPO epoch
MINI_BATCH_SIZE=5        # tiny batch
NUM_ROUNDS=5             # only 2 rounds per episode
MAX_GRAD_NORM=0.5

# PPO-specific
KL_COEF=0.1
GAMMA=1.0
LAM=0.95
CLIPRANGE=0.2
CLIPRANGE_VALUE=0.2
VF_COEF=0.5
ENTROPY_COEF=0.01

REWARD_SCALE=0.1     # Scale down rewards  
TARGET_KL=0.01       # Early stopping threshold
KL_COEF=0.2          # Higher KL penalty

# Output dir for this test run
OUTPUT_DIR="./runs/test_manual_ppo_${OPPONENT_TYPE}_${MORAL_TYPE}_05_v3"
mkdir -p "$OUTPUT_DIR"

echo "Running simple manual-PPO test run with:"
echo "  opponent_type=${OPPONENT_TYPE}"
echo "  model_p1=${MODEL_P1}"
echo "  moral_type=${MORAL_TYPE}"
echo "  temperature=${TEMP}, top_p=${TOP_P}"
echo "  lr=${LR}, num_updates=${NUM_UPDATES}, episodes_per_batch=${EPISODES_PER_BATCH}"
echo "  ppo_epochs=${PPO_EPOCHS}, mini_batch_size=${MINI_BATCH_SIZE}"
echo "  num_rounds=${NUM_ROUNDS}"
echo "  output_dir=${OUTPUT_DIR}"

# === Launch Python (manual-PPO Stag Hunt script) ===
python -m training.experiment \
  --opponent_type "${OPPONENT_TYPE}" \
  --model_p1 "${MODEL_P1}" \
  --moral_type "${MORAL_TYPE}" \
  --temperature "${TEMP}" \
  --top_p "${TOP_P}" \
  --lr "${LR}" \
  --num_updates "${NUM_UPDATES}" \
  --episodes_per_batch "${EPISODES_PER_BATCH}" \
  --ppo_epochs "${PPO_EPOCHS}" \
  --mini_batch_size "${MINI_BATCH_SIZE}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --kl_coef "${KL_COEF}" \
  --gamma "${GAMMA}" \
  --lam "${LAM}" \
  --cliprange "${CLIPRANGE}" \
  --cliprange_value "${CLIPRANGE_VALUE}" \
  --vf_coef "${VF_COEF}" \
  --entropy_coef "${ENTROPY_COEF}" \
  --num_rounds "${NUM_ROUNDS}" \
  --output_dir "${OUTPUT_DIR}" \
  --reward_scale "${REWARD_SCALE}" \
  --target_kl "${TARGET_KL}" \
  --normalize_rewards \
  --normalize_advantages \
  --use_adaptive_kl

echo "Done."