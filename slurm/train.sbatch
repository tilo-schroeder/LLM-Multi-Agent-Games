#!/bin/bash
#SBATCH --job-name=stag_ppo
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ===== Setup =====
module purge
module load devel/cuda/11.8
source .venv/bin/activate

mkdir -p logs

export HF_HOME="${HF_HOME:-$SCRATCH/hf-cache}"
mkdir -p "$HF_HOME"

# ===== Config =====
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"

# LoRA / quant
USE_LORA="${USE_LORA:-0}"          # 1=on, 0=off (driver default is on)
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
USE_4BIT="${USE_4BIT:-0}"          # 1=on, 0=off

# Game
NUM_PLAYERS="${NUM_PLAYERS:-2}"
THRESHOLD="${THRESHOLD:-2}"
STAG_SUCCESS_REWARD="${STAG_SUCCESS_REWARD:-4.0}"
STAG_FAIL_REWARD="${STAG_FAIL_REWARD:-0.0}"
HARE_REWARD="${HARE_REWARD:-2.0}"

# Morals
MORAL_TYPE="${MORAL_TYPE:-game}"
OPPONENT_MORAL_TYPE="${OPPONENT_MORAL_TYPE:-utilitarian}"
PLAYER_MORAL_TYPES="${PLAYER_MORAL_TYPES:-utilitarian,utilitarian}"   # e.g. "utilitarian,game,game" (length NUM_PLAYERS)

# Opponents / multi-agent
OPPONENT_TYPE="${OPPONENT_TYPE:-copy_focal}"   # always_stag|always_hare|random|copy_focal|llm
LLM_VS_LLM="${LLM_VS_LLM:-0}"                  # 1=on, 0=off (auto-enabled in code if OPPONENT_TYPE=llm)
SHARED_POLICY="${SHARED_POLICY:-0}"            # only meaningful with llm_vs_llm

# Training
NUM_EPISODES="${NUM_EPISODES:-200}"
BATCH_SIZE="${BATCH_SIZE:-5}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"

# Output
# OUTPUT_DIR="${OUTPUT_DIR:-./outputs/run_${SLURM_JOB_ID:-local}}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/final_results}"
SEED="${SEED:-22}"
LOG_EVERY="${LOG_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-100}"

# ===== Build args (only pass flags when enabled / non-empty) =====
ARGS=(
  --model_name "$MODEL_NAME"
  --num_players "$NUM_PLAYERS"
  --threshold "$THRESHOLD"
  --stag_success_reward "$STAG_SUCCESS_REWARD"
  --stag_fail_reward "$STAG_FAIL_REWARD"
  --hare_reward "$HARE_REWARD"
  --moral_type "$MORAL_TYPE"
  --opponent_moral_type "$OPPONENT_MORAL_TYPE"
  --opponent_type "$OPPONENT_TYPE"
  --num_episodes "$NUM_EPISODES"
  --batch_size "$BATCH_SIZE"
  --ppo_epochs "$PPO_EPOCHS"
  --learning_rate "$LEARNING_RATE"
  --output_dir "$OUTPUT_DIR"
  --seed "$SEED"
  --log_every "$LOG_EVERY"
  --save_every "$SAVE_EVERY"
  --lora_rank "$LORA_RANK"
  --lora_alpha "$LORA_ALPHA"
)

# LoRA enable/disable: your driver defaults use_lora=True.
if [[ "$USE_LORA" -eq 1 ]]; then
  ARGS+=(--use_lora)
fi

if [[ "$USE_4BIT" -eq 1 ]]; then
  ARGS+=(--use_4bit)
fi

if [[ "$LLM_VS_LLM" -eq 1 ]]; then
  ARGS+=(--llm_vs_llm)
fi

if [[ "$SHARED_POLICY" -eq 1 ]]; then
  ARGS+=(--shared_policy)
fi

if [[ -n "${PLAYER_MORAL_TYPES}" ]]; then
  ARGS+=(--player_moral_types "$PLAYER_MORAL_TYPES")
fi

# ===== Run =====
echo "Running: python -m training.train ${ARGS[*]}"
python -m training.train "${ARGS[@]}"

echo "Done."

# Qwen/Qwen3-4B-Instruct-2507
# Qwen/Qwen2.5-0.5B-Instruct
