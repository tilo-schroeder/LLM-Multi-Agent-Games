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

export HF_HOME="${HF_HOME:-$SCRATCH/hf-cache}"
mkdir -p "$HF_HOME"

# Run the pipeline
OUTDIR="${OUTDIR:-runs}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SEED="${SEED:-0}"

# Env params
ROUNDS="${ROUNDS:-20}"
T="${T:-5.0}"
R="${R:-3.0}"
P="${P:-1.0}"
S="${S:-0.0}"
NOISE="${NOISE:-0.0}"

# Train/eval sizes
EVAL_EP="${EVAL_EP:-50}"
TRAIN_EP="${TRAIN_EP:-200}"

# GRPO/FT params
LR="${LR:-5e-6}"
BATCH="${BATCH:-64}"
MINIBATCH="${MINIBATCH:-16}"
EPOCHS="${EPOCHS:-1}"
NUMGEN="${NUMGEN:-8}"
MAXTOK="${MAXTOK:-2}"
TEMP="${TEMP:-0.7}"
SOCIAL="${SOCIAL:-true}"

SOC_FLAG="--social_reward"
if [ "$SOCIAL" != "true" ]; then
  SOC_FLAG="--no_social_reward"
fi

python -m scripts.run_pipeline \
  --model "$MODEL" --seed "$SEED" \
  --rounds "$ROUNDS" --T "$T" --R "$R" --P "$P" --S "$S" --noise "$NOISE" \
  --eval_episodes "$EVAL_EP" --train_episodes "$TRAIN_EP" \
  --lr "$LR" --batch "$BATCH" --minibatch "$MINIBATCH" --epochs "$EPOCHS" \
  --num_generations "$NUMGEN" --max_new_tokens "$MAXTOK" \
  $SOC_FLAG \
  --outdir "$OUTDIR"

echo "Done."