import os
import json
import csv
import argparse
from typing import List
from dataclasses import dataclass

import numpy as np
import torch
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

from envs.stag_hunt import StagHuntEnv, StagHuntConfig
from utils.logging_utils import get_logger

from .data_structures import DecisionSample
from .prompting import extract_action_from_completion
from .rollouts_tft import collect_batch_moral_vs_tft
from .rollouts_two_llm import collect_batch_moral_two_llm_agents
from .rollouts_shared_policy import collect_batch_moral_shared_policy
from .logprob import compute_logprob_for_sample

import matplotlib
matplotlib.use("Agg")  # safe for SLURM/non-GUI environments
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Self-play GRPO-style finetuning on repeated Stag Hunt."
    )

    # High-level setup
    parser.add_argument(
        "--setup",
        type=str,
        choices=["tft", "two_llm", "shared_llm"],
        default="tft",
        help=(
            'Training setup: '
            '"tft" (LLM vs fixed opponent such as Tit-for-Tat), '
            '"two_llm" (two separate learning LLMs), or '
            '"shared_llm" (two players share a single policy).'
        ),
    )

    parser.add_argument(
        "--opponent_type",
        type=str,
        choices=["tft", "always_cooperate", "always_defect", "random"],
        default="tft",
        help=(
            'Fixed opponent strategy when setup="tft": '
            '"tft", "always_cooperate", "always_defect", or "random".'
        ),
    )

    # Models
    parser.add_argument(
        "--model_p1",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model name for Player 1 (HF repo id or local path).",
    )
    parser.add_argument(
        "--model_p2",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model name for Player 2 (if None, use model_p1).",
    )

    # Temperature / sampling
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter.",
    )

    # Reward / moral type
    parser.add_argument(
        "--moral_type",
        type=str,
        choices=["game", "deontological", "utilitarian", "game+deontological"],
        default="utilitarian",
        help="Type of intrinsic/moral reward.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--num_updates",
        type=int,
        default=30,
        help="Number of gradient updates.",
    )
    parser.add_argument(
        "--episodes_per_batch",
        type=int,
        default=8,
        help="Number of episodes collected per update.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm.",
    )

    # Environment config
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=5,
        help="Number of repeated rounds in the Stag Hunt game.",
    )

    # Output / logging
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./runs/tft_utilitarian_with_full_logging",
        help="Directory to save models, logs, and plots.",
    )

    return parser.parse_args()


@dataclass
class DecisionSample:
    episode_id: int
    player_id: int
    round_idx: int
    prompt: str
    completion: str
    # reward will be filled after decision; here it's intrinsic/moral reward
    reward: float = 0.0
    action: str = ""             # "STAG" or "HARE" (parsed action)
    opp_prev_action: str = ""    # opponent's previous move, "" if none
    is_legal: bool = True


def train_grpo_stag_hunt_local(args):
    """
    Main training loop.

    Use the `setup` flag below to choose between:
      - "tft":     Player 1 (LLM) vs fixed Tit-for-Tat opponent
      - "two_llm": Player 1 (LLM) vs Player 2 (LLM), both learning
      - "shared_llm": Single shared policy controlling both players
    """
    setup = args.setup
    opponent_type = args.opponent_type  # (used when setup == "tft")

    base_model_name_p1 = args.model_p1
    base_model_name_p2 = args.model_p2 or args.model_p1  # default: same as P1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = args.output_dir
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(output_dir, exist_ok=True)

    logger = get_logger(log_dir)

    stag_cfg = StagHuntConfig(
        num_players=2,
        num_rounds=args.num_rounds,
        R_stag_stag=4.0,
        R_hare_hare=1.0,
        R_stag_hare=0.0,
        R_hare_stag=3.0,
    )

    moral_type = args.moral_type

    num_updates = args.num_updates
    episodes_per_batch = args.episodes_per_batch
    lr = args.lr
    max_grad_norm = args.max_grad_norm

    logger.info(f"Starting GRPO Stag Hunt training, setup={setup}")
    logger.info(f"Base model P1: {base_model_name_p1}")
    logger.info(f"Base model P2: {base_model_name_p2}")
    logger.info(f"StagHuntConfig: {stag_cfg}")
    logger.info(
        f"moral_type={moral_type}, num_updates={num_updates}, "
        f"episodes_per_batch={episodes_per_batch}, lr={lr}, "
        f"temperature={args.temperature}, top_p={args.top_p}"
    )

    # ---- Load tokenizer (shared) ----
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_p1, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === containers for logging stats ===
    stats = {
        "update": [],
        "mean_reward": [],
        "std_reward": [],
        "mean_reward_p1": [],
        "mean_reward_p2": [],
        "avg_loss_p1": [],
        "avg_loss_p2": [],
        "p1_stag_rate": [],
        "p2_stag_rate": [],
        "global_stag_rate": [],
    }

    action_categories = [
        "STAG|STAG",    # P? STAG, opp previously STAG
        "STAG|HARE",    # P? STAG, opp previously HARE
        "HARE|STAG",    # P? HARE, opp previously STAG
        "HARE|HARE",    # P? HARE, opp previously HARE
        "illegal|STAG", # illegal completion, opp previously STAG
        "illegal|HARE", # illegal completion, opp previously HARE
    ]
    action_stats = {"episode": []}
    for cat in action_categories:
        action_stats[cat] = []

    def stag_rate(decisions: List[DecisionSample]) -> float:
        if not decisions:
            return 0.0
        stags = sum(
            1 for s in decisions
            if extract_action_from_completion(s.completion)[0] == StagHuntEnv.ACTION_STAG
        )
        return stags / len(decisions)

    # =========================
    # Case 1: LLM vs fixed opponent
    # =========================
    if setup == "tft":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_p1,
            trust_remote_code=True,
        ).to(device)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.train()

        optimizer = AdamW(model.parameters(), lr=lr)
        total_steps = num_updates
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        for update in range(1, num_updates + 1):
            logger.info(
                f"=== Running update {update}/{num_updates} "
                f"(LLM vs fixed opponent='{opponent_type}') ==="
            )

            # 1) Collect on-policy data
            samples, opp_stag_rate, opp_rewards = collect_batch_moral_vs_tft(
                model=model,
                tokenizer=tokenizer,
                config=stag_cfg,
                device=device,
                num_episodes=episodes_per_batch,
                moral_type=moral_type,
                temperature=args.temperature,
                top_p=args.top_p,
                logger=logger,
                opponent_type=opponent_type,
            )

            # ----- separate P1 / opponent and global stats -----
            rewards_p1 = torch.tensor([s.reward for s in samples], dtype=torch.float32)
            mean_r_p1 = rewards_p1.mean()
            std_r_p1 = rewards_p1.std(unbiased=False).clamp(min=1e-6)

            # Opponent reward stats for this batch
            if len(opp_rewards) > 0:
                rewards_p2 = torch.tensor(opp_rewards, dtype=torch.float32)
                mean_r_p2 = rewards_p2.mean()
                all_rewards = torch.cat([rewards_p1, rewards_p2], dim=0)
            else:
                rewards_p2 = torch.tensor([0.0], dtype=torch.float32)
                mean_r_p2 = rewards_p2.mean()
                all_rewards = rewards_p1

            mean_r_global = all_rewards.mean()
            std_r_global = all_rewards.std(unbiased=False).clamp(min=1e-6)

            advantages = (rewards_p1 - mean_r_p1) / std_r_p1

            # 2) Policy gradient update (P1 only)
            model.train()
            optimizer.zero_grad()
            total_loss = 0.0

            for s, adv in zip(samples, advantages):
                adv_i = adv.to(device)
                logprob_i = compute_logprob_for_sample(
                    model=model,
                    tokenizer=tokenizer,
                    sample=s,
                    device=device,
                    max_length=512,
                )
                loss_i = -adv_i * logprob_i
                loss_i.backward()
                total_loss += loss_i.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            # --- action stats per episode (Player 1 only) ---
            for ep_local in range(episodes_per_batch):
                global_ep_idx = (update - 1) * episodes_per_batch + ep_local + 1
                counts = {cat: 0 for cat in action_categories}

                for s in samples:
                    if s.episode_id != ep_local:
                        continue
                    if s.opp_prev_action not in (
                        StagHuntEnv.ACTION_STAG,
                        StagHuntEnv.ACTION_HARE,
                    ):
                        continue
                    opp = s.opp_prev_action
                    if not s.is_legal:
                        key = f"illegal|{opp}"
                    else:
                        act = s.action or extract_action_from_completion(s.completion)[0]
                        key = f"{act}|{opp}"
                    if key in counts:
                        counts[key] += 1

                action_stats["episode"].append(global_ep_idx)
                for cat in action_categories:
                    action_stats[cat].append(counts[cat])

            # === cooperation statistics ===
            p1_stag = stag_rate(samples)
            p2_stag = opp_stag_rate  # fixed opponent STAG rate
            global_stag = 0.5 * (p1_stag + p2_stag)

            avg_loss_p1 = total_loss / max(len(samples), 1)
            avg_loss_p2 = 0.0

            stats["update"].append(update)
            stats["mean_reward"].append(mean_r_global.item())
            stats["std_reward"].append(std_r_global.item())
            stats["mean_reward_p1"].append(mean_r_p1.item())
            stats["mean_reward_p2"].append(mean_r_p2.item())
            stats["avg_loss_p1"].append(avg_loss_p1)
            stats["avg_loss_p2"].append(avg_loss_p2)
            stats["p1_stag_rate"].append(p1_stag)
            stats["p2_stag_rate"].append(p2_stag)
            stats["global_stag_rate"].append(global_stag)

            logger.info(
                f"[Update {update}/{num_updates}] "
                f"Loss P1: {avg_loss_p1:.4f} | "
                f"Mean moral reward (global): {mean_r_global.item():.3f} "
                f"(P1={mean_r_p1.item():.3f}, Opp={mean_r_p2.item():.3f}) | "
                f"Std reward (global): {std_r_global.item():.3f} | "
                f"P1 STAG rate: {p1_stag:.3f} | "
                f"Opponent STAG rate: {p2_stag:.3f} | "
                f"Global STAG rate: {global_stag:.3f} | "
                f"Num samples (P1): {len(samples)}"
            )

        # Save model/tokenizer at end of fixed-opponent case
        model.save_pretrained(os.path.join(output_dir, "agent_fixed_opp_p1"))
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved fine-tuned Player 1 model (vs fixed opponent) to {output_dir}")

    # =========================
    # Case 2: Two learning LLM agents
    # =========================
    elif setup == "two_llm":
        model_p1 = AutoModelForCausalLM.from_pretrained(
            base_model_name_p1,
            trust_remote_code=True,
        ).to(device)

        model_p2 = AutoModelForCausalLM.from_pretrained(
            base_model_name_p2,
            trust_remote_code=True,
        ).to(device)

        model_p1.config.pad_token_id = tokenizer.pad_token_id
        model_p2.config.pad_token_id = tokenizer.pad_token_id

        model_p1.train()
        model_p2.train()

        optimizer_p1 = AdamW(model_p1.parameters(), lr=lr)
        optimizer_p2 = AdamW(model_p2.parameters(), lr=lr)

        total_steps = num_updates
        scheduler_p1 = get_linear_schedule_with_warmup(
            optimizer_p1,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )
        scheduler_p2 = get_linear_schedule_with_warmup(
            optimizer_p2,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        for update in range(1, num_updates + 1):
            logger.info(f"=== Running update {update}/{num_updates} (two LLM agents) ===")

            # 1) Collect data under current policies (on-policy, no grad)
            samples = collect_batch_moral_two_llm_agents(
                model_p1=model_p1,
                model_p2=model_p2,
                tokenizer=tokenizer,
                config=stag_cfg,
                device=device,
                num_episodes=episodes_per_batch,
                moral_type=moral_type,
                temperature=args.temperature,
                top_p=args.top_p,
                logger=logger,
            )

            # Separate samples by player
            samples_p1 = [s for s in samples if s.player_id == 1]
            samples_p2 = [s for s in samples if s.player_id == 2]

            if not samples_p1 or not samples_p2:
                logger.warning("One of the players has no samples in this batch. Skipping update.")
                continue

            rewards_p1 = torch.tensor([s.reward for s in samples_p1], dtype=torch.float32)
            rewards_p2 = torch.tensor([s.reward for s in samples_p2], dtype=torch.float32)

            # Global reward stats for logging
            all_rewards = torch.cat([rewards_p1, rewards_p2], dim=0)
            mean_r_global = all_rewards.mean()
            std_r_global = all_rewards.std(unbiased=False).clamp(min=1e-6)

            # 2) Compute GRPO-style advantages per agent
            mean_r_p1 = rewards_p1.mean()
            std_r_p1 = rewards_p1.std(unbiased=False).clamp(min=1e-6)
            advantages_p1 = (rewards_p1 - mean_r_p1) / std_r_p1

            mean_r_p2 = rewards_p2.mean()
            std_r_p2 = rewards_p2.std(unbiased=False).clamp(min=1e-6)
            advantages_p2 = (rewards_p2 - mean_r_p2) / std_r_p2

            # 3) Policy gradient update for Player 1
            model_p1.train()
            optimizer_p1.zero_grad()
            total_loss_p1 = 0.0

            for s, adv in zip(samples_p1, advantages_p1):
                adv_i = adv.to(device)  # scalar
                logprob_i = compute_logprob_for_sample(
                    model=model_p1,
                    tokenizer=tokenizer,
                    sample=s,
                    device=device,
                    max_length=512,
                )
                loss_i = -adv_i * logprob_i
                loss_i.backward()
                total_loss_p1 += loss_i.item()

            torch.nn.utils.clip_grad_norm_(model_p1.parameters(), max_grad_norm)
            optimizer_p1.step()
            scheduler_p1.step()

            # 4) Policy gradient update for Player 2
            model_p2.train()
            optimizer_p2.zero_grad()
            total_loss_p2 = 0.0

            for s, adv in zip(samples_p2, advantages_p2):
                adv_i = adv.to(device)  # scalar
                logprob_i = compute_logprob_for_sample(
                    model=model_p2,
                    tokenizer=tokenizer,
                    sample=s,
                    device=device,
                    max_length=512,
                )
                loss_i = -adv_i * logprob_i
                loss_i.backward()
                total_loss_p2 += loss_i.item()

            torch.nn.utils.clip_grad_norm_(model_p2.parameters(), max_grad_norm)
            optimizer_p2.step()
            scheduler_p2.step()

            # --- action stats per episode ---
            for ep_local in range(episodes_per_batch):
                global_ep_idx = (update - 1) * episodes_per_batch + ep_local + 1

                counts = {cat: 0 for cat in action_categories}

                for s in samples:
                    if s.episode_id != ep_local:
                        continue

                    # only if we know the opponent's previous move
                    if s.opp_prev_action not in (
                        StagHuntEnv.ACTION_STAG,
                        StagHuntEnv.ACTION_HARE,
                    ):
                        continue

                    opp = s.opp_prev_action  # "STAG" or "HARE"
                    if not s.is_legal:
                        key = f"illegal|{opp}"
                    else:
                        act = s.action or extract_action_from_completion(s.completion)[0]
                        key = f"{act}|{opp}"

                    if key in counts:
                        counts[key] += 1

                action_stats["episode"].append(global_ep_idx)
                for cat in action_categories:
                    action_stats[cat].append(counts[cat])

            # === cooperation statistics ===
            p1_stag = stag_rate(samples_p1)
            p2_stag = stag_rate(samples_p2)
            global_stag = 0.5 * (p1_stag + p2_stag)

            avg_loss_p1 = total_loss_p1 / max(len(samples_p1), 1)
            avg_loss_p2 = total_loss_p2 / max(len(samples_p2), 1)

            stats["update"].append(update)
            stats["mean_reward"].append(mean_r_global.item())
            stats["std_reward"].append(std_r_global.item())
            stats["mean_reward_p1"].append(mean_r_p1.item())
            stats["mean_reward_p2"].append(mean_r_p2.item())
            stats["avg_loss_p1"].append(avg_loss_p1)
            stats["avg_loss_p2"].append(avg_loss_p2)
            stats["p1_stag_rate"].append(p1_stag)
            stats["p2_stag_rate"].append(p2_stag)
            stats["global_stag_rate"].append(global_stag)

            logger.info(
                f"[Update {update}/{num_updates}] "
                f"Loss P1: {avg_loss_p1:.4f} | Loss P2: {avg_loss_p2:.4f} | "
                f"Mean moral reward (global): {mean_r_global.item():.3f} "
                f"(P1={mean_r_p1.item():.3f}, P2={mean_r_p2.item():.3f}) | "
                f"Std reward (global): {std_r_global.item():.3f} | "
                f"P1 STAG rate: {p1_stag:.3f} | "
                f"P2 STAG rate: {p2_stag:.3f} | "
                f"Global STAG rate: {global_stag:.3f} | "
                f"Num samples: {len(samples)}"
            )

        # Save fine-tuned models for two-agent case
        agent1_dir = os.path.join(output_dir, "agent1")
        agent2_dir = os.path.join(output_dir, "agent2")
        os.makedirs(agent1_dir, exist_ok=True)
        os.makedirs(agent2_dir, exist_ok=True)

        model_p1.save_pretrained(agent1_dir)
        model_p2.save_pretrained(agent2_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved fine-tuned Player 1 model to {agent1_dir}")
        logger.info(f"Saved fine-tuned Player 2 model to {agent2_dir}")
        logger.info(f"Saved tokenizer to {output_dir}")

    # =========================
    # Case 3: Two LLM agents with SHARED policy
    # =========================
    elif setup == "shared_llm":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_p1,
            trust_remote_code=True,
        ).to(device)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.train()

        optimizer = AdamW(model.parameters(), lr=lr)
        total_steps = num_updates
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        for update in range(1, num_updates + 1):
            logger.info(
                f"=== Running update {update}/{num_updates} "
                f"(two LLM roles, shared policy) ==="
            )

            # 1) Collect data under current shared policy
            samples = collect_batch_moral_shared_policy(
                model=model,
                tokenizer=tokenizer,
                config=stag_cfg,
                device=device,
                num_episodes=episodes_per_batch,
                moral_type=moral_type,
                temperature=args.temperature,
                top_p=args.top_p,
                logger=logger,
            )

            samples_p1 = [s for s in samples if s.player_id == 1]
            samples_p2 = [s for s in samples if s.player_id == 2]

            if not samples_p1 or not samples_p2:
                logger.warning(
                    "One of the players has no samples in this batch (shared policy). "
                    "Skipping update."
                )
                continue

            rewards_p1 = torch.tensor([s.reward for s in samples_p1], dtype=torch.float32)
            rewards_p2 = torch.tensor([s.reward for s in samples_p2], dtype=torch.float32)

            all_samples = samples_p1 + samples_p2
            rewards_all = torch.tensor(
                [s.reward for s in all_samples],
                dtype=torch.float32,
            )

            mean_r_global = rewards_all.mean()
            std_r_global = rewards_all.std(unbiased=False).clamp(min=1e-6)

            mean_r_p1 = rewards_p1.mean()
            mean_r_p2 = rewards_p2.mean()

            advantages_all = (rewards_all - mean_r_global) / std_r_global

            # 2) Policy gradient update (shared policy)
            model.train()
            optimizer.zero_grad()
            total_loss = 0.0

            for s, adv in zip(all_samples, advantages_all):
                adv_i = adv.to(device)
                logprob_i = compute_logprob_for_sample(
                    model=model,
                    tokenizer=tokenizer,
                    sample=s,
                    device=device,
                    max_length=512,
                )
                loss_i = -adv_i * logprob_i
                loss_i.backward()
                total_loss += loss_i.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            # --- action stats per episode (both players) ---
            for ep_local in range(episodes_per_batch):
                global_ep_idx = (update - 1) * episodes_per_batch + ep_local + 1

                counts = {cat: 0 for cat in action_categories}
                for s in samples:
                    if s.episode_id != ep_local:
                        continue

                    if s.opp_prev_action not in (
                        StagHuntEnv.ACTION_STAG,
                        StagHuntEnv.ACTION_HARE,
                    ):
                        continue

                    opp = s.opp_prev_action
                    if not s.is_legal:
                        key = f"illegal|{opp}"
                    else:
                        act = s.action or extract_action_from_completion(s.completion)[0]
                        key = f"{act}|{opp}"
                    if key in counts:
                        counts[key] += 1

                action_stats["episode"].append(global_ep_idx)
                for cat in action_categories:
                    action_stats[cat].append(counts[cat])

            # === cooperation statistics ===
            p1_stag = stag_rate(samples_p1)
            p2_stag = stag_rate(samples_p2)
            global_stag = 0.5 * (p1_stag + p2_stag)

            avg_loss = total_loss / max(len(all_samples), 1)
            avg_loss_p1 = avg_loss
            avg_loss_p2 = avg_loss  # same model

            stats["update"].append(update)
            stats["mean_reward"].append(mean_r_global.item())
            stats["std_reward"].append(std_r_global.item())
            stats["mean_reward_p1"].append(mean_r_p1.item())
            stats["mean_reward_p2"].append(mean_r_p2.item())
            stats["avg_loss_p1"].append(avg_loss_p1)
            stats["avg_loss_p2"].append(avg_loss_p2)
            stats["p1_stag_rate"].append(p1_stag)
            stats["p2_stag_rate"].append(p2_stag)
            stats["global_stag_rate"].append(global_stag)

            logger.info(
                f"[Update {update}/{num_updates}] "
                f"Shared policy loss: {avg_loss:.4f} | "
                f"Mean moral reward (global): {mean_r_global.item():.3f} "
                f"(P1={mean_r_p1.item():.3f}, P2={mean_r_p2.item():.3f}) | "
                f"Std reward (global): {std_r_global.item():.3f} | "
                f"P1 STAG rate: {p1_stag:.3f} | "
                f"P2 STAG rate: {p2_stag:.3f} | "
                f"Global STAG rate: {global_stag:.3f} | "
                f"Num samples (both players): {len(all_samples)}"
            )

        # Save shared-policy model
        shared_dir = os.path.join(output_dir, "shared_policy")
        os.makedirs(shared_dir, exist_ok=True)
        model.save_pretrained(shared_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved fine-tuned shared policy model to {shared_dir}")

    else:
        raise ValueError(f"Unknown setup: {setup}")

    # === save stats to JSON & CSV ===
    stats_path_json = os.path.join(log_dir, "training_stats.json")
    with open(stats_path_json, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved training stats (JSON) to {stats_path_json}")

    stats_path_csv = os.path.join(log_dir, "training_stats.csv")
    with open(stats_path_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "update",
                "mean_reward",
                "std_reward",
                "mean_reward_p1",
                "mean_reward_p2",
                "avg_loss_p1",
                "avg_loss_p2",
                "p1_stag_rate",
                "p2_stag_rate",
                "global_stag_rate",
            ]
        )
        for i in range(len(stats["update"])):
            writer.writerow([
                stats["update"][i],
                stats["mean_reward"][i],
                stats["std_reward"][i],
                stats["mean_reward_p1"][i],
                stats["mean_reward_p2"][i],
                stats["avg_loss_p1"][i],
                stats["avg_loss_p2"][i],
                stats["p1_stag_rate"][i],
                stats["p2_stag_rate"][i],
                stats["global_stag_rate"][i],
            ])
    logger.info(f"Saved training stats (CSV) to {stats_path_csv}")

    # === learning curve plotting ===
    # 1) Global mean moral reward
    plt.figure()
    plt.plot(stats["update"], stats["mean_reward"], marker="o", label="Global mean")
    plt.plot(stats["update"], stats["mean_reward_p1"], marker="x", linestyle="--", label="P1 mean")
    plt.plot(stats["update"], stats["mean_reward_p2"], marker="x", linestyle="--", label="P2 mean")
    plt.xlabel("Update")
    plt.ylabel("Mean intrinsic (moral) reward")
    plt.title(f"Learning Curve: Mean Moral Reward vs. Update ({setup})")
    plt.legend()
    plt.grid(True)
    curve_path = os.path.join(log_dir, "learning_curve_reward.png")
    plt.savefig(curve_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved learning curve plot to {curve_path}")

    # 2) Cooperation rate
    plt.figure()
    plt.plot(stats["update"], stats["p1_stag_rate"], marker="o", label="P1 STAG rate")
    plt.plot(stats["update"], stats["p2_stag_rate"], marker="s", label="P2 STAG rate")
    plt.plot(stats["update"], stats["global_stag_rate"], marker="^", label="Global STAG rate")
    plt.xlabel("Update")
    plt.ylabel("STAG frequency")
    plt.title(f"Cooperation (STAG) Rate vs. Update ({setup})")
    plt.legend()
    plt.grid(True)
    coop_curve_path = os.path.join(log_dir, "learning_curve_cooperation.png")
    plt.savefig(coop_curve_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved cooperation curve plot to {coop_curve_path}")

    # 3) Figure-3-style action distribution plot
    episodes_for_plot = action_stats["episode"]
    if len(episodes_for_plot) > 0:
        cat_keys = action_categories  # preserve order
        counts_array = np.vstack([action_stats[k] for k in cat_keys])  # [6, num_eps]

        plt.figure(figsize=(12, 4))
        bottoms = np.zeros(len(episodes_for_plot))

        for i, cat in enumerate(cat_keys):
            vals = counts_array[i]
            plt.bar(
                episodes_for_plot,
                vals,
                bottom=bottoms,
                label=cat,
                width=1.0,
            )
            bottoms += vals

        plt.xlabel("Episode")
        plt.ylabel("Count of actions per episode")
        plt.title(
            f"Actions conditioned on opponent's previous move "
            f"({moral_type}, {setup})"
        )
        plt.legend(loc="upper right", fontsize=7)
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        fig3_path = os.path.join(log_dir, "figure3_style_actions.png")
        plt.savefig(fig3_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved Figure-3-style action distribution plot to {fig3_path}")

if __name__ == "__main__":
    args = parse_args()
    train_grpo_stag_hunt_local(args)