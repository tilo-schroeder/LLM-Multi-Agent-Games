"""
Self-play GRPO-style finetuning on a 2-player repeated Stag Hunt game,
where the LLM is a local decision policy.

This modified version trains ONE learning agent (Player 1) against a
fixed Tit-for-Tat opponent (Player 2) with *moral / intrinsic* rewards,
inspired by moral-alignment setups for matrix games.

Key changes vs original:

- Player 2 is a fixed Tit-for-Tat policy.
- Player 1 receives *per-decision* intrinsic rewards:
    - "game":        own material payoff
    - "deontological": penalty for defecting vs a previous cooperator
    - "utilitarian": own + opponent payoff
    - "game+deontological": game payoff minus norm-violation term
- Illegal outputs (not of the form "ACTION: STAG/HARE") are penalized.
- GRPO update stays the same, but now uses these intrinsic rewards.
"""

# TODO: Replicate plots from Huggingface site

import os
import json
import csv
import logging
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


# =======================================
# Fixed opponent: Tit-for-Tat
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
    Format the state using only the last joint action,
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


# ===== Original self-play rollout (unused right now)====

def rollout_episode(
    model,
    tokenizer,
    config: StagHuntConfig,
    device: torch.device,
    episode_id: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Tuple[List[DecisionSample], float, float]:
    """
    Original self-play rollout for 2 learning agents.

    NOTE: Not used in the final training loop, which uses a single
    learning agent vs a fixed Tit-for-Tat opponent with moral rewards.
    """
    env = StagHuntEnv(config)
    env.reset()

    samples: List[DecisionSample] = []

    done = False
    while not done:
        obs = env._get_obs()
        round_idx = obs["round"] + 1

        actions = {}

        for player_id in [1, 2]:
            prompt = build_decision_prompt(player_id, obs, config, tokenizer)
            completion = generate_completion(
                model, tokenizer, prompt, device,
                temperature=temperature, top_p=top_p,
            )
            action, _ = extract_action_from_completion(completion)
            actions[player_id] = action

            samples.append(
                DecisionSample(
                    episode_id=episode_id,
                    player_id=player_id,
                    round_idx=round_idx,
                    prompt=prompt,
                    completion=completion,
                    reward=0.0,  # could be set to material payoff if desired
                )
            )

        obs, (r1, r2), done, info = env.step(actions[1], actions[2])

    total_r1, total_r2 = env.compute_episode_returns()
    return samples, total_r1, total_r2


def collect_batch(
    model,
    tokenizer,
    config: StagHuntConfig,
    device: torch.device,
    num_episodes: int,
    reward_mode: str = "average",
    temperature: float = 0.7,
    top_p: float = 0.9,
    logger: logging.Logger = None,
) -> List[DecisionSample]:
    """
    Original self-play batch collection.

    NOTE: Not used in the final training loop.
    """
    if logger is None:
        logger = get_logger()

    all_samples: List[DecisionSample] = []
    episode_rewards = {}

    for ep_id in range(num_episodes):
        logger.info(f"  Rolling out episode {ep_id}...")
        samples, r1, r2 = rollout_episode(
            model, tokenizer, config, device,
            episode_id=ep_id,
            temperature=temperature,
            top_p=top_p,
        )
        all_samples.extend(samples)

        if ep_id < 2:
            logger.info(f"    Episode {ep_id} total_r1={r1}, total_r2={r2}")
            rounds = []
            for s in samples:
                if s.episode_id == ep_id:
                    a, _ = extract_action_from_completion(s.completion)
                    rounds.append((s.round_idx, s.player_id, a))
            logger.info("    Parsed actions (round, player, action):")
            for triple in rounds:
                logger.info(f"      {triple}")

        if reward_mode == "selfish_p1":
            R = float(r1)
        elif reward_mode == "selfish_p2":
            R = float(r2)
        elif reward_mode == "cooperative":
            R = float(r1 + r2)
        elif reward_mode == "average":
            R = 0.5 * float(r1 + r2)
        else:
            raise ValueError(f"Unknown reward_mode: {reward_mode}")

        episode_rewards[ep_id] = R

    for s in all_samples:
        s.reward = episode_rewards[s.episode_id]

    return all_samples


# ===== New rollout: moral agent vs Tit-for-Tat opponent =====

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
) -> Tuple[List[DecisionSample], float]:

    env = StagHuntEnv(config)
    obs = env.reset(random_initial_state=True)
    opponent.reset()

    samples: List[DecisionSample] = []
    step_rewards: List[float] = []

    for t in range(config.num_rounds):
        round_idx = t + 1

        # --- NEW: get opponent's previous action from current obs ---
        history = obs["history"]
        opp_prev_action = None
        if history:
            _, last_a2 = history[-1]
            opp_prev_action = last_a2

        prompt = build_decision_prompt(1, obs, config, tokenizer)
        completion = generate_completion(
            model, tokenizer, prompt, device,
            temperature=temperature, top_p=top_p,
        )
        action_p1, is_legal = extract_action_from_completion(completion)

        # --- NEW: store action, opp_prev_action, legality in sample ---
        sample = DecisionSample(
            episode_id=episode_id,
            player_id=1,
            round_idx=round_idx,
            prompt=prompt,
            completion=completion,
            reward=0.0,      # will fill later
            action=action_p1,
            opp_prev_action=opp_prev_action or "",
            is_legal=is_legal,
        )

        if not is_legal:
            # Illegal action → no env step; apply penalty reward only
            moral_r = compute_moral_reward_stag_hunt(
                moral_type=moral_type,
                agent_action=StagHuntEnv.ACTION_HARE,
                opp_prev_action=opp_prev_action,
                r_agent=0.0,
                r_opp=0.0,
                is_legal=False,
                xi=xi,
                illegal_penalty=illegal_penalty,
            )
            step_rewards.append(moral_r)
            samples.append(sample)
            # state does NOT change
            continue

        # legal action → opponent acts, env steps
        action_p2 = opponent.act(obs)
        obs, (r1, r2), done, info = env.step(action_p1, action_p2)

        moral_r = compute_moral_reward_stag_hunt(
            moral_type=moral_type,
            agent_action=action_p1,
            opp_prev_action=opp_prev_action,
            r_agent=r1,
            r_opp=r2,
            is_legal=True,
            xi=xi,
            illegal_penalty=illegal_penalty,
        )

        step_rewards.append(moral_r)
        samples.append(sample)

    total_moral_return = float(sum(step_rewards))

    # for s in samples:
    #     s.reward = total_moral_return
    for s, r in zip(samples, step_rewards):
        s.reward = r                # purely local reward

    return samples, total_moral_return


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
) -> List[DecisionSample]:
    """
    Collect num_episodes of (LLM vs Tit-for-Tat), with intrinsic moral rewards
    already attached per decision.
    """
    if logger is None:
        logger = get_logger()

    all_samples: List[DecisionSample] = []
    opponent = TitForTatOpponent()

    for ep_id in range(num_episodes):
        logger.info(f"  Rolling out moral episode {ep_id} (type={moral_type})...")
        samples, total_moral = rollout_episode_vs_tft(
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

        if ep_id < 2:
            logger.info(f"    Episode {ep_id} total moral return={total_moral:.3f}")
            rounds = [(s.round_idx, s.player_id,
                       extract_action_from_completion(s.completion)[0])
                      for s in samples]
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


def train_grpo_stag_hunt_local():
    # ---- Config ----
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    # model_name = "Qwen/Qwen3-4B-Instruct-2507"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = "./runs/setup_test_utilitarian_long"
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(output_dir, exist_ok=True)

    logger = get_logger(log_dir)

    stag_cfg = StagHuntConfig(
        num_players=2,
        num_rounds=5,
        R_stag_stag=4.0,
        R_hare_hare=1.0,
        R_stag_hare=0.0,
        R_hare_stag=3.0,
    )

    # Moral reward type for Player 1:
    #   "game", "deontological", "utilitarian", "game+deontological"
    moral_type = "utilitarian"

    num_updates = 30      # gradient updates
    episodes_per_batch = 8
    lr = 1e-5
    max_grad_norm = 1.0

    logger.info("Starting GRPO Stag Hunt training (moral agent vs Tit-for-Tat)")
    logger.info(f"Model: {model_name}")
    logger.info(f"StagHuntConfig: {stag_cfg}")
    logger.info(
        f"moral_type={moral_type}, num_updates={num_updates}, "
        f"episodes_per_batch={episodes_per_batch}, lr={lr}"
    )

    # ---- Load model & tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)

    total_steps = num_updates
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # === containers for logging stats ===
    stats = {
        "update": [],
        "mean_reward": [],
        "std_reward": [],
        "avg_loss": [],
        "p1_stag_rate": [],
        "p2_stag_rate": [],
        "global_stag_rate": [],
    }

    action_categories = [
        "STAG|STAG",    # P1 STAG, opp previously STAG
        "STAG|HARE",    # P1 STAG, opp previously HARE
        "HARE|STAG",    # P1 HARE, opp previously STAG
        "HARE|HARE",    # P1 HARE, opp previously HARE
        "illegal|STAG", # illegal completion, opp previously STAG
        "illegal|HARE", # illegal completion, opp previously HARE
    ]
    action_stats = {"episode": []}
    for cat in action_categories:
        action_stats[cat] = []

    for update in range(1, num_updates + 1):
        logger.info(f"=== Running update {update}/{num_updates} ===")

        # Example of switching moral objective halfway through:
        # if update == num_updates // 2 + 1:
        #     moral_type = "utilitarian"
        #     logger.info(f"*** Switching moral_type to {moral_type} at update {update} ***")

        # 1) Collect data under current policy (on-policy, no grad)
        samples = collect_batch_moral_vs_tft(
            model=model,
            tokenizer=tokenizer,
            config=stag_cfg,
            device=device,
            num_episodes=episodes_per_batch,
            moral_type=moral_type,
            temperature=1.7,
            top_p=0.9,
            logger=logger,
        )

        # Rewards as CPU tensor (intrinsic moral rewards)
        rewards = torch.tensor([s.reward for s in samples], dtype=torch.float32)

        # 2) Compute GRPO-style advantages on CPU
        mean_r = rewards.mean()
        std_r = rewards.std(unbiased=False).clamp(min=1e-6)
        advantages = (rewards - mean_r) / std_r  # shape [N], on CPU

        # 3) Policy gradient update:
        model.train()
        optimizer.zero_grad()

        total_loss = 0.0
        for i, sample in enumerate(samples):
            adv_i = advantages[i].to(device)  # scalar

            logprob_i = compute_logprob_for_sample(
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                device=device,
                max_length=512,
            )

            # Loss_i = - A_i * log pi
            loss_i = -adv_i * logprob_i
            loss_i.backward()  # backward for this sample only

            total_loss += loss_i.item()

        # clip gradients across all params
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()

        for ep_local in range(episodes_per_batch):
            global_ep_idx = (update - 1) * episodes_per_batch + ep_local + 1

            counts = {cat: 0 for cat in action_categories}

            for s in samples:
                if s.episode_id != ep_local:
                    continue

                # we only care if we know the opponent's previous move
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

        # === compute cooperation statistics ===
        total_samples = len(samples)
        avg_reward = rewards.mean().item()
        std_reward = rewards.std(unbiased=False).item()
        avg_loss = total_loss / max(total_samples, 1)

        def stag_rate(decisions: List[DecisionSample]) -> float:
            if not decisions:
                return 0.0
            stags = sum(
                1 for s in decisions
                if extract_action_from_completion(s.completion)[0] == StagHuntEnv.ACTION_STAG
            )
            return stags / len(decisions)

        # Only Player 1 is a learning agent here
        p1_decisions = samples
        p1_stag = stag_rate(p1_decisions)
        global_stag = p1_stag

        stats["update"].append(update)
        stats["mean_reward"].append(avg_reward)
        stats["std_reward"].append(std_reward)
        stats["avg_loss"].append(avg_loss)
        stats["p1_stag_rate"].append(p1_stag)
        stats["p2_stag_rate"].append(0.0)  # fixed opponent, not tracked here
        stats["global_stag_rate"].append(global_stag)

        logger.info(
            f"[Update {update}/{num_updates}] "
            f"Loss: {avg_loss:.4f} | "
            f"Mean moral reward: {avg_reward:.3f} | "
            f"Std reward: {std_reward:.3f} | "
            f"P1 STAG rate: {p1_stag:.3f} | "
            f"Global STAG rate: {global_stag:.3f} | "
            f"Num samples: {total_samples}"
        )

    # === save stats to JSON & CSV ===
    stats_path_json = os.path.join(log_dir, "training_stats.json")
    with open(stats_path_json, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved training stats (JSON) to {stats_path_json}")

    stats_path_csv = os.path.join(log_dir, "training_stats.csv")
    with open(stats_path_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["update", "mean_reward", "std_reward",
             "avg_loss", "p1_stag_rate", "p2_stag_rate", "global_stag_rate"]
        )
        for i in range(len(stats["update"])):
            writer.writerow([
                stats["update"][i],
                stats["mean_reward"][i],
                stats["std_reward"][i],
                stats["avg_loss"][i],
                stats["p1_stag_rate"][i],
                stats["p2_stag_rate"][i],
                stats["global_stag_rate"][i],
            ])
    logger.info(f"Saved training stats (CSV) to {stats_path_csv}")

    # === learning curve plotting ===
    plt.figure()
    plt.plot(stats["update"], stats["mean_reward"], marker="o")
    plt.xlabel("Update")
    plt.ylabel("Mean intrinsic (moral) reward")
    plt.title("Learning Curve: Mean Moral Reward vs. Update")
    plt.grid(True)
    curve_path = os.path.join(log_dir, "learning_curve_reward.png")
    plt.savefig(curve_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved learning curve plot to {curve_path}")

    # plot cooperation rate as well
    plt.figure()
    plt.plot(stats["update"], stats["p1_stag_rate"], marker="o", label="P1 STAG rate")
    plt.xlabel("Update")
    plt.ylabel("STAG frequency (Player 1)")
    plt.title("Cooperation (STAG) Rate vs. Update (vs Tit-for-Tat)")
    plt.legend()
    plt.grid(True)
    coop_curve_path = os.path.join(log_dir, "learning_curve_cooperation.png")
    plt.savefig(coop_curve_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved cooperation curve plot to {coop_curve_path}")


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
        plt.title(f"P1 actions vs TFT conditioned on opponent's previous move ({moral_type})")
        plt.legend(loc="upper right", fontsize=7)
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        fig3_path = os.path.join(log_dir, "figure3_style_actions_vs_opp_prev.png")
        plt.savefig(fig3_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved Figure-3-style action distribution plot to {fig3_path}")


    # Save fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Saved fine-tuned model to {output_dir}")


# ================
# 6. Entry point
# ================

if __name__ == "__main__":
    train_grpo_stag_hunt_local()