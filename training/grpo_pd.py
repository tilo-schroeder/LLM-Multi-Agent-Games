from __future__ import annotations

import os
import json
import csv
import time
import uuid
import random
import re
from collections import Counter
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional

import torch
from torch import nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import LoraConfig
from trl import PPOConfig, PPOTrainer

from envs.repeated_pd import RepeatedPD, Config as EnvConfig, ACTIONS
from policy.utils import format_pd_prompt
from policy.llm_policy import LLMPolicy
from config import TrainConfig


# ---------------------------------------------------------------------
# Simple logging helper
# ---------------------------------------------------------------------


def _log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[ppo_train_selfplay {ts}] {msg}", flush=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def make_uid() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------
# Step-wise rollout + evaluation (used for baseline / post-training)
# ---------------------------------------------------------------------


def rollout_episode(env: RepeatedPD, policy: LLMPolicy) -> Dict:
    """
    Roll out a single repeated PD episode with the given LLM policy, using
    step-wise prompts (each round one query per agent).
    """
    obs = env.reset()
    done = False
    total = {"agent_0": 0.0, "agent_1": 0.0}
    actions: List[Tuple[str, str]] = []

    while not done:
        p0 = format_pd_prompt(policy.tokenizer, obs["agent_0"])
        p1 = format_pd_prompt(policy.tokenizer, obs["agent_1"])
        _, acts, _, _ = policy.act([p0, p1])
        a0, a1 = acts
        obs, (r0, r1), done = env.step(a0, a1)
        total["agent_0"] += r0
        total["agent_1"] += r1
        actions.append((a0, a1))

    return {"return": total, "actions": actions}


def evaluate(
    env_cfg: EnvConfig,
    policy: LLMPolicy,
    episodes: int = 50,
    seed: int = 0,
    log_dir: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate a policy in self-play on the repeated PD environment.
    Returns average payoffs and cooperation rate.
    """
    import numpy as np

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    avg_pay, coop, total_moves = [], 0, 0
    ep_rows: List[Tuple[int, float, float, float]] = []

    for e in range(episodes):
        e_cfg = EnvConfig(**{**env_cfg.__dict__, "seed": seed + e})
        env = RepeatedPD(e_cfg)
        traj = rollout_episode(env, policy)
        r0, r1 = traj["return"]["agent_0"], traj["return"]["agent_1"]
        avg_pay.append((r0, r1))
        c = sum(int(a0 == "C") + int(a1 == "C") for a0, a1 in traj["actions"])
        moves = 2 * len(traj["actions"])
        coop += c
        total_moves += moves
        ep_rows.append((e, r0, r1, c / max(1, moves)))

    avg0 = float(np.mean([x for x, _ in avg_pay])) if avg_pay else 0.0
    avg1 = float(np.mean([y for _, y in avg_pay])) if avg_pay else 0.0
    coop_rate = coop / max(1, total_moves)

    metrics = {
        "avg_payoff_agent0": avg0,
        "avg_payoff_agent1": avg1,
        "cooperation_rate": coop_rate,
    }

    if log_dir:
        with open(os.path.join(log_dir, "episodes.csv"), "w", newline="") as f:
            w = csv.writer(f, delimiter=",")
            w.writerow(["episode", "ret_agent0", "ret_agent1", "coop_rate"])
            for r in ep_rows:
                w.writerow(r)
        with open(os.path.join(log_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


# ---------------------------------------------------------------------
# Episode-level prompts and parsing (full-game generation)
# ---------------------------------------------------------------------


def format_pd_episode_prompt(
    tokenizer, env_cfg: EnvConfig, uid: Optional[str] = None
) -> str:
    """
    Build a chat-style prompt asking the model to generate the *entire episode*
    of a repeated PD between agent_0 and agent_1.
    """
    system_msg = (
        "You are simulating a repeated Prisoner's Dilemma between two agents, "
        "agent_0 and agent_1. "
        f"The game lasts exactly {env_cfg.rounds} rounds. "
        "For each round i, you must output a line in the format:\n"
        "Round i: A0 A1\n"
        "where A0 is agent_0's action and A1 is agent_1's action, "
        "each either 'C' (cooperate) or 'D' (defect).\n"
        "You may optionally include brief commentary after the actions on the same line, "
        "but the first two action tokens after 'Round i:' must be A0 and A1."
    )

    user_content = (
        "Generate a complete sequence of actions for the entire game.\n"
        "Remember: exactly one line per round, starting from Round 1 up to Round "
        f"{env_cfg.rounds}. Use only 'C' or 'D' for the actions.\n"
        "For example, a line should look like:\n"
        "Round i: A0 A1\n"
        "where A0, A1 are placeholders that you must replace with 'C' or 'D'.\n"
        "Do not skip rounds."
    )

    if uid is not None:
        user_content += f"\n\n<ID><UID:{uid}></ID>"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


_ROUND_LINE_RE = re.compile(
    r"round\s*(\d+)\s*:\s*([cCdD])\s+([cCdD])",
    re.IGNORECASE,
)
_PAIR_RE = re.compile(r"\b([cCdD])\s+([cCdD])\b")


def parse_episode_actions(text: str, rounds: int) -> List[Tuple[str, str]]:
    """
    Parse an episode completion into a list of (a0, a1) actions for each round.
    """
    actions: List[Tuple[str, str]] = []

    for line in text.splitlines():
        m = _ROUND_LINE_RE.search(line)
        if m:
            a0, a1 = m.group(2).upper(), m.group(3).upper()
            actions.append((a0, a1))

    if len(actions) < rounds:
        for m in _PAIR_RE.finditer(text):
            a0, a1 = m.group(1).upper(), m.group(2).upper()
            actions.append((a0, a1))
            if len(actions) >= rounds:
                break

    if not actions:
        return []

    actions = actions[:rounds]
    if len(actions) < rounds:
        actions.extend([("D", "D")] * (rounds - len(actions)))

    return actions


def pd_episode_reward_func(
    completions: List[str],
    *,
    env_cfg: EnvConfig,
    invalid_penalty: float = -0.5,
) -> List[float]:
    """
    Episode-level reward: each completion is treated as a *full game transcript*.
    """
    T, R, P, S = env_cfg.T, env_cfg.R, env_cfg.P, env_cfg.S
    denom = max(T - S, 1e-6)
    rewards: List[float] = []

    for i, comp in enumerate(completions):
        actions = parse_episode_actions(comp or "", env_cfg.rounds)
        if not actions:
            r_invalid = max(0.0, min(1.0, invalid_penalty))
            rewards.append(r_invalid)
            continue

        e_cfg = EnvConfig(**{**env_cfg.__dict__, "seed": env_cfg.seed + i})
        env = RepeatedPD(e_cfg)
        env.reset()

        total0 = 0.0
        total1 = 0.0

        for a0, a1 in actions:
            if a0 not in ACTIONS:
                a0 = "D"
            if a1 not in ACTIONS:
                a1 = "D"
            _, (r0, r1), done = env.step(a0, a1)
            total0 += r0
            total1 += r1
            if done:
                break

        steps = env.t
        if steps == 0:
            r_invalid = max(0.0, min(1.0, invalid_penalty))
            rewards.append(r_invalid)
            continue

        avg_self_both = (total0 + total1) / (2.0 * steps)
        welfare = (avg_self_both - S) / denom
        val = float(max(0.0, min(1.0, welfare)))
        rewards.append(val)

    return rewards


def collect_episode_prompts(
    env_cfg: EnvConfig,
    tokenizer,
    episodes: int,
    seed: int = 0,
) -> Dataset:
    """
    Build a Dataset where each row is a single prompt asking the model to
    generate an entire PD episode (all rounds).
    """
    rng = random.Random(seed)
    _ = rng

    prompts: List[str] = []
    for _e in range(episodes):
        uid = make_uid()
        prompt = format_pd_episode_prompt(tokenizer, env_cfg, uid=uid)
        prompts.append(prompt)

    ds = Dataset.from_dict({"prompt": prompts})
    return ds


# ---------------------------------------------------------------------
# Reward model wrapper for PPO (episode-level PD reward)
# ---------------------------------------------------------------------


class PDRewardModel(nn.Module):
    """
    Wrapper turning the hand-crafted PD episode reward into a
    transformers-compatible "reward model" for PPOTrainer.
    """

    def __init__(self, tokenizer, env_cfg: EnvConfig, invalid_penalty: float = -0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.env_cfg = env_cfg
        self.invalid_penalty = invalid_penalty

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        device = input_ids.device
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        rewards = pd_episode_reward_func(
            texts,
            env_cfg=self.env_cfg,
            invalid_penalty=self.invalid_penalty,
        )

        scores = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1)
        return SequenceClassifierOutput(logits=scores)


# ---------------------------------------------------------------------
# PPO training with experimental PPOTrainer (with extra logging)
# ---------------------------------------------------------------------


def ppo_train_selfplay(
    env_cfg: EnvConfig,
    train_cfg: TrainConfig,
    episodes: int = 200,
    save_dir: str = "./ppo_adapter",
    log_every_steps: int = 10,
):
    """
    PPO training where each PPO "episode" is a *full repeated PD game*
    generated in one completion, scored by a social-welfare reward.
    """
    _log("Entered ppo_train_selfplay")
    _log(f"Training episodes: {episodes}")
    _log(f"TrainConfig: {train_cfg}")
    _log(f"EnvConfig: {env_cfg}")

    torch.manual_seed(train_cfg.seed)
    random.seed(train_cfg.seed)

    os.makedirs(save_dir, exist_ok=True)
    hist_path = os.path.join(save_dir, "action_hist.csv")
    sample_path = os.path.join(save_dir, "train_samples.jsonl")

    if not os.path.exists(hist_path):
        with open(hist_path, "w") as f:
            f.write(
                "step,timestamp,total_sequences,C,D,unknown,mean_reward,std_reward\n"
            )
        _log(f"Created history CSV at {hist_path}")
    else:
        _log(f"History CSV already exists at {hist_path}")

    # Tokenizer
    t0 = time.time()
    _log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        train_cfg.model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    _log(f"Tokenizer loaded in {time.time() - t0:.2f}s")

    # Dataset of episode prompts
    _log("Building episode-level dataset...")
    t0 = time.time()
    train_ds = collect_episode_prompts(
        env_cfg,
        tokenizer=tokenizer,
        episodes=episodes,
        seed=train_cfg.seed,
    )
    _log(f"Dataset built in {time.time() - t0:.2f}s, size={len(train_ds)}")

    # Policy model (actor)
    _log("Loading policy model (CausalLM)...")
    t0 = time.time()
    policy_model = AutoModelForCausalLM.from_pretrained(
        train_cfg.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    _log(f"Policy model loaded in {time.time() - t0:.2f}s")

    # Value model (critic)
    _log("Loading value model (SequenceClassification)...")
    t0 = time.time()
    value_model = AutoModelForSequenceClassification.from_pretrained(
        train_cfg.model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    _log(f"Value model loaded in {time.time() - t0:.2f}s")

    # Reward model: wraps the hand-crafted PD episode reward
    _log("Instantiating PDRewardModel wrapper...")
    reward_model = PDRewardModel(tokenizer, env_cfg)
    reward_model.to(policy_model.device)
    _log("Reward model ready")

    # LoRA config
    _log("Creating LoRA config...")
    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    _log("LoRA config created")

    # PPO config – extra logging
    response_len = min(train_cfg.max_new_tokens, 64)
    _log("Creating PPOConfig...")
    ppo_args = PPOConfig(
        output_dir=save_dir,
        seed=train_cfg.seed,
        learning_rate=train_cfg.learning_rate,
        per_device_train_batch_size=train_cfg.mini_batch_size,
        gradient_accumulation_steps=max(
            1, train_cfg.batch_size // train_cfg.mini_batch_size
        ),
        num_ppo_epochs=train_cfg.grpo_epochs,
        num_mini_batches=1,
        total_episodes=episodes,
        response_length=response_len,
        temperature=train_cfg.temperature,
        logging_steps=log_every_steps,
        remove_unused_columns=False,
        sft_model_path=train_cfg.model_name,
        reward_model_path=train_cfg.model_name,
        num_sample_generations=1,
    )
    _log(f"PPOConfig created: {ppo_args}")

    # PPO trainer
    _log("Initializing PPOTrainer...")
    t0 = time.time()
    trainer = PPOTrainer(
        args=ppo_args,
        processing_class=tokenizer,
        model=policy_model,
        ref_model=None,
        reward_model=reward_model,
        train_dataset=train_ds,
        value_model=value_model,
        peft_config=peft_cfg,
    )
    _log(f"PPOTrainer initialized in {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # Custom logging: hook into reward model to record action stats
    # ------------------------------------------------------------------
    _log("Wrapping reward model forward for extra logging...")

    reward_call_counter = {"n": 0}

    def wrapped_reward_forward(input_ids=None, attention_mask=None, **kwargs):
        reward_call_counter["n"] += 1
        call_id = reward_call_counter["n"]

        _log(
            f"RewardModel forward call #{call_id} "
            f"(batch_size={input_ids.size(0)}, seq_len={input_ids.size(1)})"
        )

        with torch.no_grad():
            texts = tokenizer.batch_decode(
                input_ids, skip_special_tokens=True
            )

        # Log first 1–2 sequences of the first calls
        if call_id <= 3:
            for i, ttxt in enumerate(texts[:2]):
                _log(
                    f"[reward_call {call_id}] sample {i} decoded prefix: "
                    f"{ttxt[:200].replace(chr(10), ' ')}"
                )

        rewards = pd_episode_reward_func(
            texts,
            env_cfg=env_cfg,
            invalid_penalty=-0.5,
        )

        action_counts = Counter()
        for txt in texts:
            seq = parse_episode_actions(txt or "", env_cfg.rounds)
            if not seq:
                action_counts["unknown"] += 1
                continue
            for a0, a1 in seq:
                if a0 in ("C", "D"):
                    action_counts[a0] += 1
                else:
                    action_counts["unknown"] += 1
                if a1 in ("C", "D"):
                    action_counts[a1] += 1
                else:
                    action_counts["unknown"] += 1

        n = max(1, len(rewards))
        mean_r = float(sum(rewards) / n)
        var_r = float(sum((r - mean_r) ** 2 for r in rewards) / n)
        std_r = var_r**0.5

        step = trainer.state.global_step or 0
        ts = int(time.time())

        _log(
            f"Reward call #{call_id} -> global_step={step}, "
            f"mean_reward={mean_r:.3f}, std_reward={std_r:.3f}, "
            f"C={action_counts.get('C', 0)}, D={action_counts.get('D', 0)}, "
            f"unknown={action_counts.get('unknown', 0)}"
        )

        with open(hist_path, "a") as f:
            f.write(
                f"{step},{ts},{len(rewards)},"
                f"{action_counts.get('C', 0)},{action_counts.get('D', 0)},{action_counts.get('unknown', 0)},"
                f"{mean_r:.6f},{std_r:.6f}\n"
            )

        with open(sample_path, "a") as f:
            for txt, r in zip(texts, rewards):
                if random.random() < 0.05:
                    row = {
                        "step": step,
                        "completion": txt,
                        "reward": r,
                    }
                    f.write(json.dumps(row) + "\n")

        device = input_ids.device
        scores = torch.tensor(
            rewards, dtype=torch.float32, device=device
        ).unsqueeze(-1)
        return SequenceClassifierOutput(logits=scores)

    reward_model.forward = wrapped_reward_forward  # type: ignore[assignment]
    _log("Reward model wrapped. Starting PPO training...")

    # Train
    t0 = time.time()
    trainer.train()
    _log(f"PPO training finished in {time.time() - t0:.2f}s")

    # Save adapter / tokenizer in save_dir so LLMPolicy(adapter_dir=...) can load it
    _log("Saving trained policy + tokenizer...")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    _log("Models saved")

    # Also save configs + log history
    _log("Saving configs and log history...")
    with open(os.path.join(save_dir, "train_config.json"), "w") as f:
        json.dump(asdict(train_cfg), f, indent=2)
    with open(os.path.join(save_dir, "env_config.json"), "w") as f:
        json.dump(asdict(env_cfg), f, indent=2)
    with open(os.path.join(save_dir, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    csv_path = os.path.join(save_dir, "train_history.csv")
    keys = sorted({k for d in trainer.state.log_history for k in d.keys()})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in trainer.state.log_history:
            w.writerow(row)
    _log(f"Train history CSV written to {csv_path}")

    learner = LLMPolicy(train_cfg, adapter_dir=save_dir)
    _log("ppo_train_selfplay complete – returning learner + log history")
    return learner, trainer.state.log_history