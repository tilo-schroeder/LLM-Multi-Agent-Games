from __future__ import annotations

import os
import time
import json
import random
import re
from dataclasses import asdict
from typing import List, Optional, Dict, Tuple

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from envs.repeated_pd import RepeatedPD, Config as EnvConfig, ACTIONS
from policy.llm_policy import LLMPolicy
from policy.utils import format_pd_prompt
from config import TrainConfig

import csv
import time
from collections import Counter


# -------------------------------------------------------------------
# Prompt + parsing for "single-agent full episode" completions
# -------------------------------------------------------------------

_SINGLE_LINE_RE = re.compile(
    r"round\s*(\d+)\s*:\s*([cCdD])",
    re.IGNORECASE,
)
_SINGLE_LETTER_RE = re.compile(r"\b([cCdD])\b")


def format_single_agent_episode_prompt(
    tokenizer,
    env_cfg: EnvConfig,
    role: str,
    uid: Optional[str] = None,
) -> str:
    """
    Prompt the model as a *single* agent (agent_0 or agent_1) to generate
    its own entire action sequence for the episode.
    """
    assert role in ("agent_0", "agent_1")
    other = "agent_1" if role == "agent_0" else "agent_0"

    system_msg = (
        f"You are {role} in a repeated Prisoner's Dilemma against {other}. "
        f"The game lasts exactly {env_cfg.rounds} rounds. "
        "For each round i, you must output YOUR action only, in the format:\n"
        "Round i: A\n"
        "where A is either 'C' (cooperate) or 'D' (defect).\n"
        "You may optionally include brief commentary after the action on the same line, "
        "but the first token after 'Round i:' must be A."
    )

    user_content = (
        "Generate a complete sequence of YOUR actions for the entire game.\n"
        "Remember: exactly one line per round, starting from Round 1 up to Round "
        f"{env_cfg.rounds}. Use only 'C' or 'D' for the actions.\n"
        "For example:\n"
        "Round 1: C\n"
        "Round 2: D\n"
        "...\n"
        "Do not skip rounds."
    )

    if uid is not None:
        user_content += f"\n\n<ID><UID:{uid}></ID>"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_content},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def parse_single_agent_actions(text: str, rounds: int) -> List[str]:
    """
    Parse 'Round i: A' style single-agent transcripts into a list of actions.
    Falls back to scanning for bare C/D letters if needed.
    """
    acts: List[str] = []

    for line in text.splitlines():
        m = _SINGLE_LINE_RE.search(line)
        if m:
            acts.append(m.group(2).upper())

    if len(acts) < rounds:
        for m in _SINGLE_LETTER_RE.finditer(text):
            acts.append(m.group(1).upper())
            if len(acts) >= rounds:
                break

    if not acts:
        return []

    acts = acts[:rounds]
    if len(acts) < rounds:
        acts.extend(["D"] * (rounds - len(acts)))
    return acts


def make_uid() -> str:
    import uuid
    return uuid.uuid4().hex[:12]


def collect_single_agent_episode_prompts(
    env_cfg: EnvConfig,
    tokenizer,
    episodes: int,
    role: str,
    seed: int = 0,
) -> Dataset:
    """
    Dataset where each row is a prompt asking one *specific* agent (role)
    to generate its full action sequence.
    """
    rng = random.Random(seed)
    prompts: List[str] = []

    for _ in range(episodes):
        uid = make_uid()
        prompt = format_single_agent_episode_prompt(tokenizer, env_cfg, role, uid=uid)
        prompts.append(prompt)

    ds = Dataset.from_dict({"prompt": prompts})
    return ds


# -------------------------------------------------------------------
# Reward for one agent vs a fixed opponent policy
# -------------------------------------------------------------------

def single_agent_episode_reward_func(
    completions: List[str],
    *,
    env_cfg: EnvConfig,
    role: str,
    opponent_policy: LLMPolicy,
    social_reward: bool,
    coop_bonus: float,
    invalid_penalty: float = -0.5,
) -> List[float]:
    """
    Reward for one agent (role='agent_0' or 'agent_1') playing a full episode
    against a fixed opponent_policy.

    Each completion is parsed into that agent's actions over rounds.
    The opponent's actions are generated on-the-fly from its own LLMPolicy
    using per-round observations from the environment.

    Reward is:
      - social welfare if social_reward=True
      - selfish payoff for this agent otherwise
      plus optional coop_bonus * cooperation_rate shaping.
    """
    assert role in ("agent_0", "agent_1")
    other = "agent_1" if role == "agent_0" else "agent_0"

    T, R, P, S = env_cfg.T, env_cfg.R, env_cfg.P, env_cfg.S
    denom = max(T - S, 1e-6)

    rewards: List[float] = []

    for i, comp in enumerate(completions):
        my_actions = parse_single_agent_actions(comp or "", env_cfg.rounds)
        if not my_actions:
            # could not parse anything reasonably episode-like
            r_invalid = max(0.0, min(1.0, invalid_penalty))
            rewards.append(r_invalid)
            continue

        e_cfg = EnvConfig(**{**env_cfg.__dict__, "seed": env_cfg.seed + i})
        env = RepeatedPD(e_cfg)
        obs = env.reset()

        total0 = 0.0
        total1 = 0.0
        coops = 0

        for t in range(env.cfg.rounds):
            my_a = my_actions[t] if t < len(my_actions) else "D"
            if my_a not in ACTIONS:
                my_a = "D"

            # opponent action from its own LLMPolicy, given its observation
            if role == "agent_0":
                opp_obs_text = obs["agent_1"]
                opp_prompt = format_pd_prompt(opponent_policy.tokenizer, opp_obs_text)
                _, opp_act_list, _, _ = opponent_policy.act([opp_prompt])
                opp_a = opp_act_list[0]
                if opp_a not in ACTIONS:
                    opp_a = "D"
                a0, a1 = my_a, opp_a
            else:
                opp_obs_text = obs["agent_0"]
                opp_prompt = format_pd_prompt(opponent_policy.tokenizer, opp_obs_text)
                _, opp_act_list, _, _ = opponent_policy.act([opp_prompt])
                opp_a = opp_act_list[0]
                if opp_a not in ACTIONS:
                    opp_a = "D"
                a0, a1 = opp_a, my_a

            obs, (r0, r1), done = env.step(a0, a1)
            total0 += r0
            total1 += r1

            if a0 == "C":
                coops += 1
            if a1 == "C":
                coops += 1

            if done:
                break

        steps = env.t
        if steps == 0:
            r_invalid = max(0.0, min(1.0, invalid_penalty))
            rewards.append(r_invalid)
            continue

        # base reward
        if social_reward:
            base = (total0 + total1) / (2.0 * steps)
        else:
            base = (total0 / steps) if role == "agent_0" else (total1 / steps)

        base_norm = (base - S) / denom
        base_norm = float(max(0.0, min(1.0, base_norm)))

        # shaping for cooperation
        if coop_bonus > 0.0:
            coop_rate = coops / (2.0 * steps)
            shaped = base_norm + coop_bonus * coop_rate
        else:
            shaped = base_norm

        shaped = float(max(0.0, min(1.0, shaped)))
        rewards.append(shaped)

    return rewards


# -------------------------------------------------------------------
# GRPO training for a single agent vs fixed opponent
# -------------------------------------------------------------------

def grpo_train_single_agent(
    env_cfg: EnvConfig,
    train_cfg: TrainConfig,
    role: str,
    opponent_policy: LLMPolicy,
    episodes_per_iter: int = 200,
    outer_iters: int = 1,
    save_dir: str = "./grpo_agent",
    log_every_steps: int = 10,
):
    """
    Train a *single* agent (agent_0 or agent_1) with GRPO against a fixed opponent_policy.

    Returns:
      - LLMPolicy that uses the final adapter in save_dir
      - trainer log history
    """
    assert role in ("agent_0", "agent_1")

    os.makedirs(save_dir, exist_ok=True)
    iters_dir = os.path.join(save_dir, "iters")
    os.makedirs(iters_dir, exist_ok=True)

    torch.manual_seed(train_cfg.seed)

        # Logging paths
    sample_path = os.path.join(save_dir, "train_samples.jsonl")
    hist_path   = os.path.join(save_dir, "reward_hist.csv")
    if not os.path.exists(hist_path):
        with open(hist_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "outer_iter",
                "step",
                "timestamp",
                "role",
                "num_completions",
                "C",
                "D",
                "unknown",
                "mean_reward",
                "std_reward",
            ])

    tokenizer = AutoTokenizer.from_pretrained(
        train_cfg.model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    global_logs: List[Dict] = []
    wall = 0.0
    total_steps = 0

    for it in range(outer_iters):
        call_idx = {"i": 0}  # count reward_fn calls within this iter
        
        # 1) dataset for this agent/role
        train_ds = collect_single_agent_episode_prompts(
            env_cfg,
            tokenizer=tokenizer,
            episodes=episodes_per_iter,
            role=role,
            seed=train_cfg.seed + it,
        )

        # 2) reward function closure
        def reward_fn(completions, **kwargs):
            # Try to recover prompts from kwargs (trl passes them under different names)
            prompts = None
            for k in ("prompts", "queries", "input_texts"):
                if k in kwargs and kwargs[k] is not None:
                    prompts = kwargs[k]
                    break

            # Compute rewards
            rewards = single_agent_episode_reward_func(
                completions,
                env_cfg=env_cfg,
                role=role,
                opponent_policy=opponent_policy,
                social_reward=train_cfg.social_reward,
                coop_bonus=getattr(train_cfg, "coop_bonus", 0.0),
                invalid_penalty=-0.5,
            )

            # ---------- Logging ----------
            call_idx["i"] += 1
            step = call_idx["i"]
            ts = int(time.time())

            # Count actions for quick sanity check
            action_counts = Counter()
            for comp in completions:
                acts = parse_single_agent_actions(comp or "", env_cfg.rounds)
                if not acts:
                    action_counts["unknown"] += 1
                    continue
                for a in acts:
                    if a in ("C", "D"):
                        action_counts[a] += 1
                    else:
                        action_counts["unknown"] += 1

            # Reward stats
            import numpy as np
            n = max(1, len(rewards))
            mean_r = float(sum(rewards) / n)
            var_r = float(sum((r - mean_r) ** 2 for r in rewards) / n)
            std_r = var_r ** 0.5

            # Append a line to CSV
            with open(hist_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    it,
                    step,
                    ts,
                    role,
                    len(completions),
                    action_counts.get("C", 0),
                    action_counts.get("D", 0),
                    action_counts.get("unknown", 0),
                    f"{mean_r:.6f}",
                    f"{std_r:.6f}",
                ])

            # Sample a subset of (prompt, completion, reward) to JSONL
            with open(sample_path, "a") as f:
                for i, (c, r) in enumerate(zip(completions, rewards)):
                    # e.g., log ~10% of samples
                    if random.random() < 0.10:
                        row = {
                            "outer_iter": it,
                            "step": step,
                            "role": role,
                            "completion": c,
                            "reward": r,
                        }
                        if prompts is not None and i < len(prompts):
                            row["prompt"] = prompts[i]
                        f.write(json.dumps(row) + "\n")

            # --------------------------------
            return rewards

        # 3) GRPO config
        args = GRPOConfig(
            output_dir=os.path.join(iters_dir, f"iter_{it:02d}"),
            seed=train_cfg.seed,
            learning_rate=train_cfg.learning_rate,
            per_device_train_batch_size=train_cfg.mini_batch_size,
            gradient_accumulation_steps=max(
                1, train_cfg.batch_size // train_cfg.mini_batch_size
            ),
            num_train_epochs=train_cfg.grpo_epochs,
            max_prompt_length=512,
            max_completion_length=train_cfg.max_new_tokens,
            num_generations=max(2, train_cfg.num_generations),
            temperature=train_cfg.temperature,
            top_p=1.0,
            remove_unused_columns=False,
            logging_steps=log_every_steps,
            save_steps=0,
            model_init_kwargs={
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16
                if torch.cuda.is_available()
                else torch.float32,
            },
        )

        trainer = GRPOTrainer(
            model=train_cfg.model_name,
            reward_funcs=reward_fn,
            train_dataset=train_ds,
            processing_class=tokenizer,
            args=args,
            peft_config=peft_cfg,
        )

        t0 = time.time()
        trainer.train()
        wall += time.time() - t0
        total_steps += trainer.state.global_step or 0

        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        for row in trainer.state.log_history:
            row = dict(row)
            row["outer_iter"] = it
            global_logs.append(row)

    # final adapter = last iter's folder
    final_adapter_dir = args.output_dir
    learner = LLMPolicy(train_cfg, adapter_dir=final_adapter_dir)

    # save compact copy in save_dir for easy loading
    learner.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    with open(os.path.join(save_dir, "train_config.json"), "w") as f:
        json.dump(asdict(train_cfg), f, indent=2)
    with open(os.path.join(save_dir, "env_config.json"), "w") as f:
        json.dump(asdict(env_cfg), f, indent=2)
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump({"wall_time_sec": wall, "steps": total_steps}, f, indent=2)

    return learner, global_logs


# -------------------------------------------------------------------
# Evaluation helpers for two distinct policies
# -------------------------------------------------------------------

def rollout_episode_two_policies(
    env_cfg: EnvConfig,
    policy0: LLMPolicy,
    policy1: LLMPolicy,
    seed: int,
) -> Dict:
    env = RepeatedPD(EnvConfig(**{**env_cfg.__dict__, "seed": seed}))
    obs = env.reset()
    done = False
    total = {"agent_0": 0.0, "agent_1": 0.0}
    actions: List[Tuple[str, str]] = []

    while not done:
        p0 = format_pd_prompt(policy0.tokenizer, obs["agent_0"])
        p1 = format_pd_prompt(policy1.tokenizer, obs["agent_1"])

        _, acts0, _, _ = policy0.act([p0])
        _, acts1, _, _ = policy1.act([p1])
        a0, a1 = acts0[0], acts1[0]

        obs, (r0, r1), done = env.step(a0, a1)
        total["agent_0"] += r0
        total["agent_1"] += r1
        actions.append((a0, a1))

    return {"return": total, "actions": actions}


def evaluate_two_agents(
    env_cfg: EnvConfig,
    policy0: LLMPolicy,
    policy1: LLMPolicy,
    episodes: int = 50,
    seed: int = 0,
    log_dir: Optional[str] = None,
) -> Dict[str, float]:
    import numpy as np
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    avg_pay, coop, total_moves = [], 0, 0
    ep_rows: List[Tuple[int, float, float, float]] = []

    for e in range(episodes):
        traj = rollout_episode_two_policies(
            env_cfg, policy0, policy1, seed=seed + e
        )
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
        import csv
        with open(os.path.join(log_dir, "episodes.csv"), "w", newline="") as f:
            w = csv.writer(f, delimiter=",")
            w.writerow(["episode", "ret_agent0", "ret_agent1", "coop_rate"])
            for r in ep_rows:
                w.writerow(r)
        with open(os.path.join(log_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics