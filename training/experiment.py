"""
Self-play GRPO-style finetuning on a 2-player repeated Stag Hunt game,
where each player is a *local decision* LLM policy.

This updated version supports TWO learning agents (Player 1 and Player 2),
each with its own model and optimizer, trained via self-play with
intrinsic / moral rewards.

We also keep the one-learning-agent-vs-Tit-for-Tat rollout utilities,
and we can switch between:

- setup="tft":      Player 1 (LLM) vs fixed Tit-for-Tat opponent
- setup="two_llm":  Player 1 (LLM) vs Player 2 (LLM), both learning

Both use local decision prompts with per-round intrinsic rewards:
    - "game":               own material payoff
    - "deontological":      penalty for defecting vs previous cooperator
    - "utilitarian":        own + opponent payoff
    - "game+deontological": game payoff minus norm-violation term
Illegal outputs (not of the form "ACTION: STAG/HARE") are penalized.
"""

import os
import json
import csv
import logging
import argparse
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import torch
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

from envs.stag_hunt import StagHuntEnv, StagHuntConfig
from utils.logging_utils import get_logger

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
        choices=["tft", "two_llm"],
        default="tft",
        help='Training setup: "tft" (LLM vs Tit-for-Tat) or "two_llm" (two learning LLMs).',
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


# =======================================
# Fixed opponent: Tit-for-Tat (kept)
# =======================================

class TitForTatOpponent:
    """
    Classic Tit-for-Tat opponent:

    - Round 1: play STAG (cooperate).
    - Later rounds: copy Player 1's last action.
    """

    def reset(self):
        pass

    def act(self, env_obs: Dict[str, Any]) -> str:
        history = env_obs["history"]
        if not history:
            # First move: cooperate
            return StagHuntEnv.ACTION_STAG
        last_a1, _ = history[-1]
        # Copy last action of Player 1
        return last_a1


# =======================================
# Moral reward functions for Stag Hunt
# =======================================

def compute_moral_reward_stag_hunt(
    moral_type: str,
    agent_action: str,
    opp_prev_action: str,  # opponent's last move (None on first round)
    r_agent: float,
    r_opp: float,
    is_legal: bool,
    xi: float = 3.0,
    illegal_penalty: float = -6.0,
) -> float:
    """
    Intrinsic moral rewards for a Stag Hunt matrix game.

    moral_type:
      - "game"               : agent's own payoff (material reward)
      - "deontological"      : -xi if defect vs previous cooperator, else 0
      - "utilitarian"        : r_agent + r_opp
      - "game+deontological" : game payoff minus xi on norm violation

    Illegal actions (non-parsable completions) receive illegal_penalty.
    """
    if not is_legal:
        return float(illegal_penalty)

    COOP = StagHuntEnv.ACTION_STAG
    DEFECT = StagHuntEnv.ACTION_HARE

    if moral_type == "game":
        return float(r_agent)

    if moral_type == "deontological":
        if opp_prev_action is not None and opp_prev_action == COOP and agent_action == DEFECT:
            return -float(xi)
        else:
            return 0.0

    if moral_type == "utilitarian":
        return float(r_agent + r_opp)

    if moral_type == "game+deontological":
        base = float(r_agent)
        if opp_prev_action is not None and opp_prev_action == COOP and agent_action == DEFECT:
            base -= float(xi)
        return base

    raise ValueError(f"Unknown moral_type: {moral_type}")


# =======================================
# Local decision prompting & parsing
# =======================================

def format_history_for_player(history: List[Tuple[str, str]]) -> str:
    """
    Format the state using only the last joint action.
    """
    if not history:
        return "You have not played with this opponent before.\n"

    last_a1, last_a2 = history[-1]
    return f"Last time, Player 1 chose {last_a1}, and Player 2 chose {last_a2}.\n"


def build_decision_prompt(
    player_id: int,
    env_obs: Dict[str, Any],
    config: StagHuntConfig,
    tokenizer,
) -> str:
    """
    Build a chat-formatted prompt for a single player decision,
    using the model's chat template.
    """
    assert player_id in (1, 2)
    round_idx = env_obs["round"] + 1  # 1-based
    history = env_obs["history"]
    history_text = format_history_for_player(history)

    system_msg = (
        f"You are playing as Player {player_id} "
        f"in a repeated 2-player Stag Hunt game."
    )

    user_msg = f"""
Game rules:
- There are exactly 2 players and {config.num_rounds} rounds.
- In each round, each player chooses one of two actions: STAG or HARE.
- Payoffs for each player in a round:
    - If both players choose STAG: each gets {config.R_stag_stag}.
    - If both players choose HARE: each gets {config.R_hare_hare}.
    - If one chooses STAG and the other chooses HARE:
        - The STAG hunter gets {config.R_stag_hare}.
        - The HARE hunter gets {config.R_hare_stag}.

Current state:
- This is round {round_idx} of {config.num_rounds}.
- Past rounds and actions:
{history_text}
- You do not know the other player's action in this round yet.

Your task:
- Decide whether to hunt STAG or HARE this round.
- You MUST reply with exactly ONE line, with no explanation, in this exact format:
  ACTION: STAG
  or
  ACTION: HARE

Reply now with exactly one line.
""".strip()

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def extract_action_from_completion(text: str) -> Tuple[str, bool]:
    """
    Given a completion that ends with something like "ACTION: STAG",
    parse and return (action, is_legal) where action is "STAG" or "HARE".

    If parsing fails, default to HARE and mark as illegal.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in reversed(lines):  # search from the end
        if line.upper().startswith("ACTION:"):
            tail = line.split(":", 1)[1].strip().upper()
            if "STAG" in tail:
                return StagHuntEnv.ACTION_STAG, True
            if "HARE" in tail:
                return StagHuntEnv.ACTION_HARE, True
    # Fallback: default to HARE (safe action) if parsing fails
    return StagHuntEnv.ACTION_HARE, False


# ==================================
# 4. Episode rollout & data logging
# ==================================

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


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 48,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a single completion for a local decision prompt."""
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if full_text.startswith(prompt):
        completion = full_text[len(prompt):]
    else:
        completion = full_text
    return completion


# ===== Moral agent vs Tit-for-Tat opponent =====

def rollout_episode_vs_tft(
    model,
    tokenizer,
    config: StagHuntConfig,
    device: torch.device,
    episode_id: int,
    moral_type: str,
    opponent: TitForTatOpponent,
    temperature: float = 0.7,
    top_p: float = 0.9,
    xi: float = 3.0,
    illegal_penalty: float = -6.0,
) -> Tuple[List[DecisionSample], float, List[float], int, int]:
    """
    Returns:
        samples: DecisionSample list for Player 1
        total_moral_p1: sum of P1 moral rewards
        tft_rewards: list of TFT (P2) moral rewards for each env step
        p2_stag_count: how many times TFT played STAG
        p2_total_actions: how many times TFT acted (env steps)
    """

    env = StagHuntEnv(config)
    obs = env.reset(random_initial_state=True)
    opponent.reset()

    samples: List[DecisionSample] = []
    step_rewards_p1: List[float] = []
    step_rewards_p2: List[float] = []

    p2_stag_count = 0
    p2_total_actions = 0

    for t in range(config.num_rounds):
        round_idx = t + 1

        history = obs["history"]

        # From P1's perspective:
        opp_prev_action = None
        prev_a1, prev_a2 = (None, None)
        if history:
            prev_a1, prev_a2 = history[-1]
            opp_prev_action = prev_a2  # P2's last action

        prompt = build_decision_prompt(1, obs, config, tokenizer)
        completion = generate_completion(
            model, tokenizer, prompt, device,
            temperature=temperature, top_p=top_p,
        )
        action_p1, is_legal = extract_action_from_completion(completion)

        sample = DecisionSample(
            episode_id=episode_id,
            player_id=1,
            round_idx=round_idx,
            prompt=prompt,
            completion=completion,
            reward=0.0,
            action=action_p1,
            opp_prev_action=opp_prev_action or "",
            is_legal=is_legal,
        )

        if not is_legal:
            # Illegal action → no env step; penalty for P1 only
            moral_r1 = compute_moral_reward_stag_hunt(
                moral_type=moral_type,
                agent_action=StagHuntEnv.ACTION_HARE,
                opp_prev_action=opp_prev_action,
                r_agent=0.0,
                r_opp=0.0,
                is_legal=False,
                xi=xi,
                illegal_penalty=illegal_penalty,
            )
            step_rewards_p1.append(moral_r1)
            samples.append(sample)
            # TFT does not act; no P2 reward for this "round"
            continue

        # legal → TFT acts, env steps
        action_p2 = opponent.act(obs)

        p2_total_actions += 1
        if action_p2 == StagHuntEnv.ACTION_STAG:
            p2_stag_count += 1

        obs, (r1, r2), done, info = env.step(action_p1, action_p2)

        # P1 moral reward
        moral_r1 = compute_moral_reward_stag_hunt(
            moral_type=moral_type,
            agent_action=action_p1,
            opp_prev_action=opp_prev_action,
            r_agent=r1,
            r_opp=r2,
            is_legal=True,
            xi=xi,
            illegal_penalty=illegal_penalty,
        )

        # TFT moral reward (treat TFT as agent, P1 as opponent)
        moral_r2 = compute_moral_reward_stag_hunt(
            moral_type=moral_type,
            agent_action=action_p2,
            opp_prev_action=prev_a1,  # previous P1 move
            r_agent=r2,
            r_opp=r1,
            is_legal=True,
            xi=xi,
            illegal_penalty=illegal_penalty,
        )

        step_rewards_p1.append(moral_r1)
        step_rewards_p2.append(moral_r2)
        samples.append(sample)

    total_moral_p1 = float(sum(step_rewards_p1))

    for s, r in zip(samples, step_rewards_p1):
        s.reward = r

    # Note: step_rewards_p2 can be shorter than samples (no entries for illegal P1 moves)
    return samples, total_moral_p1, step_rewards_p2, p2_stag_count, p2_total_actions


def collect_batch_moral_vs_tft(
    model,
    tokenizer,
    config: StagHuntConfig,
    device: torch.device,
    num_episodes: int,
    moral_type: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    logger: logging.Logger = None,
) -> Tuple[List[DecisionSample], float, List[float]]:
    """
    Collect num_episodes of (LLM vs Tit-for-Tat).

    Returns:
        all_samples: DecisionSample list for P1
        tft_stag_rate: overall STAG rate of TFT in this batch
        tft_rewards: list of TFT moral rewards for all env steps in batch
    """
    if logger is None:
        logger = get_logger()

    all_samples: List[DecisionSample] = []
    opponent = TitForTatOpponent()

    total_p2_stags = 0
    total_p2_actions = 0
    tft_rewards_batch: List[float] = []

    for ep_id in range(num_episodes):
        logger.info(f"  Rolling out moral episode {ep_id} (type={moral_type}) vs TFT...")
        samples, total_moral_p1, tft_rewards_ep, p2_stags, p2_actions = rollout_episode_vs_tft(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            episode_id=ep_id,
            moral_type=moral_type,
            opponent=opponent,
            temperature=temperature,
            top_p=top_p,
        )
        all_samples.extend(samples)
        tft_rewards_batch.extend(tft_rewards_ep)

        total_p2_stags += p2_stags
        total_p2_actions += p2_actions

        if ep_id < 2:
            logger.info(f"    Episode {ep_id} total moral return (P1)={total_moral_p1:.3f}")
            rounds = [
                (s.round_idx, s.player_id,
                 extract_action_from_completion(s.completion)[0])
                for s in samples
            ]
            logger.info("    Parsed actions (round, player, action):")
            for triple in rounds:
                logger.info(f"      {triple}")

            tft_ep_rate = (p2_stags / p2_actions) if p2_actions > 0 else 0.0
            tft_ep_mean = (sum(tft_rewards_ep) / len(tft_rewards_ep)) if tft_rewards_ep else 0.0
            logger.info(f"    TFT STAG rate (episode {ep_id})={tft_ep_rate:.3f}")
            logger.info(f"    TFT mean moral reward (episode {ep_id})={tft_ep_mean:.3f}")

    tft_stag_rate = (total_p2_stags / total_p2_actions) if total_p2_actions > 0 else 0.0
    return all_samples, tft_stag_rate, tft_rewards_batch


# ===== Moral self-play with TWO separate learning agents =====

def rollout_episode_two_llm_agents(
    model_p1,
    model_p2,
    tokenizer,
    config: StagHuntConfig,
    device: torch.device,
    episode_id: int,
    moral_type: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    xi: float = 3.0,
    illegal_penalty: float = -6.0,
) -> Tuple[List[DecisionSample], float, float]:
    """
    Self-play rollout: two distinct LLM policies (model_p1, model_p2)
    play Player 1 and Player 2 respectively. Both receive moral,
    per-decision rewards.
    """

    env = StagHuntEnv(config)
    obs = env.reset(random_initial_state=True)

    samples: List[DecisionSample] = []
    rewards_p1: List[float] = []
    rewards_p2: List[float] = []

    for t in range(config.num_rounds):
        round_idx = t + 1
        history = obs["history"]

        prev_a1, prev_a2 = (None, None)
        if history:
            prev_a1, prev_a2 = history[-1]

        # ---- Player 1 decision (model_p1) ----
        prompt1 = build_decision_prompt(1, obs, config, tokenizer)
        completion1 = generate_completion(
            model_p1, tokenizer, prompt1, device,
            temperature=temperature, top_p=top_p,
        )
        a1, legal1 = extract_action_from_completion(completion1)
        sample1 = DecisionSample(
            episode_id=episode_id,
            player_id=1,
            round_idx=round_idx,
            prompt=prompt1,
            completion=completion1,
            reward=0.0,
            action=a1,
            opp_prev_action=prev_a2 or "",
            is_legal=legal1,
        )

        # ---- Player 2 decision (model_p2) ----
        prompt2 = build_decision_prompt(2, obs, config, tokenizer)
        completion2 = generate_completion(
            model_p2, tokenizer, prompt2, device,
            temperature=temperature, top_p=top_p,
        )
        a2, legal2 = extract_action_from_completion(completion2)
        sample2 = DecisionSample(
            episode_id=episode_id,
            player_id=2,
            round_idx=round_idx,
            prompt=prompt2,
            completion=completion2,
            reward=0.0,
            action=a2,
            opp_prev_action=prev_a1 or "",
            is_legal=legal2,
        )

        # If either player is illegal, do not advance env; just give penalties
        if not legal1 or not legal2:
            r1 = r2 = 0.0

            moral_r1 = compute_moral_reward_stag_hunt(
                moral_type=moral_type,
                agent_action=a1 if legal1 else StagHuntEnv.ACTION_HARE,
                opp_prev_action=prev_a2,
                r_agent=r1,
                r_opp=r2,
                is_legal=legal1,
                xi=xi,
                illegal_penalty=illegal_penalty,
            )
            moral_r2 = compute_moral_reward_stag_hunt(
                moral_type=moral_type,
                agent_action=a2 if legal2 else StagHuntEnv.ACTION_HARE,
                opp_prev_action=prev_a1,
                r_agent=r2,
                r_opp=r1,
                is_legal=legal2,
                xi=xi,
                illegal_penalty=illegal_penalty,
            )

            sample1.reward = moral_r1
            sample2.reward = moral_r2

            rewards_p1.append(moral_r1)
            rewards_p2.append(moral_r2)
            samples.extend([sample1, sample2])
            # State unchanged
            continue

        # ---- Both legal → env step ----
        obs, (r1, r2), done, info = env.step(a1, a2)

        moral_r1 = compute_moral_reward_stag_hunt(
            moral_type=moral_type,
            agent_action=a1,
            opp_prev_action=prev_a2,
            r_agent=r1,
            r_opp=r2,
            is_legal=True,
            xi=xi,
            illegal_penalty=illegal_penalty,
        )
        moral_r2 = compute_moral_reward_stag_hunt(
            moral_type=moral_type,
            agent_action=a2,
            opp_prev_action=prev_a1,
            r_agent=r2,
            r_opp=r1,
            is_legal=True,
            xi=xi,
            illegal_penalty=illegal_penalty,
        )

        sample1.reward = moral_r1
        sample2.reward = moral_r2

        rewards_p1.append(moral_r1)
        rewards_p2.append(moral_r2)
        samples.extend([sample1, sample2])

    total_moral_p1 = float(sum(rewards_p1))
    total_moral_p2 = float(sum(rewards_p2))

    return samples, total_moral_p1, total_moral_p2


def collect_batch_moral_two_llm_agents(
    model_p1,
    model_p2,
    tokenizer,
    config: StagHuntConfig,
    device: torch.device,
    num_episodes: int,
    moral_type: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    logger: logging.Logger = None,
) -> List[DecisionSample]:
    """
    Collect num_episodes of self-play (LLM vs LLM, two distinct models),
    with intrinsic moral rewards attached per decision for both players.
    """
    if logger is None:
        logger = get_logger()

    all_samples: List[DecisionSample] = []

    for ep_id in range(num_episodes):
        logger.info(f"  Rolling out LLM-vs-LLM moral episode {ep_id} (type={moral_type})...")
        samples, total_moral_p1, total_moral_p2 = rollout_episode_two_llm_agents(
            model_p1=model_p1,
            model_p2=model_p2,
            tokenizer=tokenizer,
            config=config,
            device=device,
            episode_id=ep_id,
            moral_type=moral_type,
            temperature=temperature,
            top_p=top_p,
        )
        all_samples.extend(samples)

        if ep_id < 2:
            logger.info(
                f"    Episode {ep_id} total moral returns: "
                f"P1={total_moral_p1:.3f}, P2={total_moral_p2:.3f}"
            )
            rounds = [
                (s.round_idx, s.player_id,
                 extract_action_from_completion(s.completion)[0])
                for s in samples
            ]
            logger.info("    Parsed actions (round, player, action):")
            for triple in rounds:
                logger.info(f"      {triple}")

    return all_samples


# ===========================
# 5. GRPO-style RL training
# ===========================

def compute_logprob_for_sample(
    model,
    tokenizer,
    sample: DecisionSample,
    device: torch.device,
    max_length: int = 512,
) -> torch.Tensor:
    """
    Compute mean log p(completion | prompt) for a single DecisionSample.

    Returns:
        logprob: scalar tensor (requires_grad=True)
    """
    full_text = sample.prompt + sample.completion

    enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Figure out where the completion starts
    prompt_enc = tokenizer(
        sample.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)
    prompt_len = prompt_enc["input_ids"].shape[1]

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits  # [1, seq_len, vocab]

    # Standard next-token shift
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_attn = attention_mask[:, 1:].contiguous()

    # Mask: only completion tokens (everything after prompt)
    completion_mask = torch.zeros_like(shift_attn)
    completion_mask[:, prompt_len - 1:] = 1  # from last prompt token onward

    log_probs_all = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs_all.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)  # [1, seq_len]

    mask = shift_attn * completion_mask
    token_log_probs = token_log_probs * mask

    num_tokens = mask.sum().clamp(min=1)
    logprob = token_log_probs.sum() / num_tokens  # mean logprob over completion

    return logprob  # scalar with grad


def train_grpo_stag_hunt_local(args):
    """
    Main training loop.

    Use the `setup` flag below to choose between:
      - "tft":     Player 1 (LLM) vs fixed Tit-for-Tat opponent
      - "two_llm": Player 1 (LLM) vs Player 2 (LLM), both learning
    """
    # ---- High-level experiment setup from CLI ----
    setup = args.setup  # "tft" or "two_llm"

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

    # Moral reward type (from CLI)
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
    # Case 1: LLM vs Tit-for-Tat
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
            logger.info(f"=== Running update {update}/{num_updates} (LLM vs TFT) ===")

            # 1) Collect on-policy data
            samples, tft_stag_rate, tft_rewards = collect_batch_moral_vs_tft(
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

            # ----- NEW: separate P1 / TFT and global stats -----
            rewards_p1 = torch.tensor([s.reward for s in samples], dtype=torch.float32)
            mean_r_p1 = rewards_p1.mean()
            std_r_p1 = rewards_p1.std(unbiased=False).clamp(min=1e-6)

            # TFT (P2) reward stats for this batch
            if len(tft_rewards) > 0:
                rewards_p2 = torch.tensor(tft_rewards, dtype=torch.float32)
                mean_r_p2 = rewards_p2.mean()
                # Global stats: average over both players
                all_rewards = torch.cat([rewards_p1, rewards_p2], dim=0)
            else:
                # Should basically never happen, but be robust
                rewards_p2 = torch.tensor([0.0], dtype=torch.float32)
                mean_r_p2 = rewards_p2.mean()
                all_rewards = rewards_p1

            mean_r_global = all_rewards.mean()
            std_r_global = all_rewards.std(unbiased=False).clamp(min=1e-6)
            # ---------------------------------------------------

            # GRPO-style advantages for P1 are still computed
            # using P1's own baseline
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
            p2_stag = tft_stag_rate  # TFT opponent STAG rate
            global_stag = 0.5 * (p1_stag + p2_stag)

            avg_loss_p1 = total_loss / max(len(samples), 1)
            avg_loss_p2 = 0.0

            # ----- NEW: log TFT reward and global reward into stats -----
            stats["update"].append(update)
            stats["mean_reward"].append(mean_r_global.item())        # global mean
            stats["std_reward"].append(std_r_global.item())          # global std
            stats["mean_reward_p1"].append(mean_r_p1.item())         # P1 mean
            stats["mean_reward_p2"].append(mean_r_p2.item())         # TFT mean
            stats["avg_loss_p1"].append(avg_loss_p1)
            stats["avg_loss_p2"].append(avg_loss_p2)
            stats["p1_stag_rate"].append(p1_stag)
            stats["p2_stag_rate"].append(p2_stag)
            stats["global_stag_rate"].append(global_stag)
            # ------------------------------------------------------------

            logger.info(
                f"[Update {update}/{num_updates}] "
                f"Loss P1: {avg_loss_p1:.4f} | "
                f"Mean moral reward (global): {mean_r_global.item():.3f} "
                f"(P1={mean_r_p1.item():.3f}, TFT={mean_r_p2.item():.3f}) | "
                f"Std reward (global): {std_r_global.item():.3f} | "
                f"P1 STAG rate: {p1_stag:.3f} | "
                f"TFT STAG rate: {p2_stag:.3f} | "
                f"Global STAG rate: {global_stag:.3f} | "
                f"Num samples (P1): {len(samples)}"
            )

        # Save model/tokenizer at end of TFT case
        model.save_pretrained(os.path.join(output_dir, "agent_tft_p1"))
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved fine-tuned Player 1 model (vs TFT) to {output_dir}")

    # =========================
    # Case 2: Two learning LLM agents
    # =========================
    else:
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

            # 2) Compute GRPO-style advantages *per agent*
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

            # --- action stats per episode (for Figure-3-style plot) ---
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

    # === save stats to JSON & CSV (common to both setups) ===
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

    # === learning curve plotting (uses whatever stats are filled) ===
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


# ================
# 6. Entry point
# ================

if __name__ == "__main__":
    args = parse_args()
    train_grpo_stag_hunt_local(args)