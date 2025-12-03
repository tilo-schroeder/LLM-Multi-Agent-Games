from __future__ import annotations

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
import re, uuid, os, json, csv, random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import Counter

from envs.repeated_pd import RepeatedPD, Config as EnvConfig, ACTIONS
from policy.utils import to_prompt, _parse_action_strict, format_pd_prompt
from policy.llm_policy import LLMPolicy
from config import TrainConfig

def make_uid() -> str:
    return uuid.uuid4().hex[:12]

def rollout_episode(env: RepeatedPD, policy: LLMPolicy) -> Dict:
    obs = env.reset()
    done = False
    total = {"agent_0": 0.0, "agent_1": 0.0}
    actions: List[Tuple[str, str]] = []

    while not done:
        # p0 = to_prompt(obs["agent_0"])
        # p1 = to_prompt(obs["agent_1"])
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
    import numpy as np
    os.makedirs(log_dir, exist_ok=True) if log_dir else None

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

    metrics = {"avg_payoff_agent0": avg0, "avg_payoff_agent1": avg1, "cooperation_rate": coop_rate}

    if log_dir:
        # episode-level CSV
        with open(os.path.join(log_dir, "episodes.csv"), "w", newline="") as f:
            w = csv.writer(f, delimiter=",")
            w.writerow(["episode", "ret_agent0", "ret_agent1", "coop_rate"])
            for r in ep_rows:
                w.writerow(r)
        with open(os.path.join(log_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics

# ----------------------------
# Episode-level helpers
# ----------------------------

# For episode-level prompts we embed a UID the same way, so _UID_RE still works.

def format_pd_episode_prompt(tokenizer, env_cfg: EnvConfig, uid: Optional[str] = None) -> str:
    """
    Build a chat-style prompt asking the model to generate the *entire episode*
    of a repeated PD between agent_0 and agent_1.

    Expected format (softly enforced):
      Round 1: C D
      Round 2: C C
      ...
    where the two action letters are in {C,D}, first for agent_0, second for agent_1.
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
        "For example:\n"
        "Round 1: C D\n"
        "Round 2: C C\n"
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

# Parse lines like "Round 1: C D" or any "C D" pairs as fallback
_ROUND_LINE_RE = re.compile(
    r"round\s*(\d+)\s*:\s*([cCdD])\s+([cCdD])",
    re.IGNORECASE,
)
_PAIR_RE = re.compile(r"\b([cCdD])\s+([cCdD])\b")

def parse_episode_actions(text: str, rounds: int) -> List[Tuple[str, str]]:
    """
    Parse an episode completion into a list of (a0, a1) actions for each round.

    Strategy:
      1) Look for lines "Round i: C D" and collect them in order of appearance.
      2) If we still have fewer than rounds, fall back to scanning for generic pairs "C D".
      3) Trim to rounds; if still fewer, pad with ('D','D') so every episode
         has the same length from the reward's perspective.
    """
    actions: List[Tuple[str, str]] = []

    # 1) Structured lines
    for line in text.splitlines():
        m = _ROUND_LINE_RE.search(line)
        if m:
            a0, a1 = m.group(2).upper(), m.group(3).upper()
            actions.append((a0, a1))

    # 2) Fallback: any "C D" pair in the text
    if len(actions) < rounds:
        for m in _PAIR_RE.finditer(text):
            a0, a1 = m.group(1).upper(), m.group(2).upper()
            actions.append((a0, a1))
            if len(actions) >= rounds:
                break

    if not actions:
        return []

    # 3) Trim/pad to fixed horizon
    actions = actions[:rounds]
    if len(actions) < rounds:
        actions.extend([("D", "D")] * (rounds - len(actions)))

    return actions

def pd_episode_reward_func(
    completions: List[str],
    *,
    prompts: Optional[List[str]],
    env_cfg: EnvConfig,
    social_reward: bool = True,
    coop_bonus: float = 0.0,
    invalid_penalty: float = -0.5,
) -> List[float]:
    """
    Episode-level reward: each completion is treated as a *full game transcript*.

    We parse a sequence of (a0,a1) actions for all rounds, roll it through
    RepeatedPD offline, and assign a scalar reward based on social welfare.

    Reward is normalized to [0,1] using (avg_welfare - S) / (T - S).
    """
    T, R, P, S = env_cfg.T, env_cfg.R, env_cfg.P, env_cfg.S
    denom = max(T - S, 1e-6)
    rewards: List[float] = []

    for i, comp in enumerate(completions):
        actions = parse_episode_actions(comp or "", env_cfg.rounds)
        if not actions:
            # could not parse anything reasonably episode-like
            r_invalid = max(0.0, min(1.0, invalid_penalty))
            rewards.append(r_invalid)
            continue

        # deterministically roll out the episode under these actions
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

        # --- base reward: social vs selfish ---
        if social_reward:
            # average payoff per agent per step
            base = (total0 + total1) / (2.0 * steps)
        else:
            # selfish: only agent_0 payoff per step
            base = total0 / steps

        base_norm = (base - S) / denom
        base_norm = float(max(0.0, min(1.0, base_norm)))

        # --- shaping term: cooperation rate over the episode ---
        if coop_bonus > 0.0:
            coops = 0
            for a0, a1 in actions:
                if a0 == "C":
                    coops += 1
                if a1 == "C":
                    coops += 1
            coop_rate = coops / (2.0 * steps)
            shaped = base_norm + coop_bonus * coop_rate
        else:
            shaped = base_norm

        shaped = float(max(0.0, min(1.0, shaped)))
        rewards.append(shaped)

    return rewards

def collect_episode_prompts(
    env_cfg: EnvConfig,
    tokenizer,
    episodes: int,
    seed: int = 0,
):
    """
    Build a Dataset where each row is a single prompt asking the model to
    generate an entire PD episode (all rounds).

    This matches the PLAYPEN-style "full game per completion" setup.
    """
    rng = random.Random(seed)
    prompts: List[str] = []

    for e in range(episodes):
        uid = make_uid()
        prompt = format_pd_episode_prompt(tokenizer, env_cfg, uid=uid)
        prompts.append(prompt)

    ds = Dataset.from_dict({"prompt": prompts})
    return ds

# ----------------------------
# GRPO training
# ----------------------------
def grpo_train_selfplay(
    env_cfg: EnvConfig,
    train_cfg: TrainConfig,
    outer_iters: int = 1,
    episodes_per_iter: int = 200,
    save_dir: str = "./grpo_adapter",
    log_every_steps: int = 10,
):
    import os, time, json, csv, random
    from dataclasses import asdict

    torch.manual_seed(train_cfg.seed)

    # Root/train folder + per-iter folder
    root_dir = save_dir
    iters_dir = os.path.join(root_dir, "iters")
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(iters_dir, exist_ok=True)

    # Tokenizer for prompts / GRPO processing
    tokenizer = AutoTokenizer.from_pretrained(
        train_cfg.model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # LoRA config
    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    sample_path = os.path.join(root_dir, "train_samples.jsonl")
    hist_path   = os.path.join(root_dir, "action_hist.csv")
    if not os.path.exists(hist_path):
        with open(hist_path, "w") as f:
            f.write("iter,step,timestamp,total,C,D,unknown,mean_reward,std_reward\n")

    global_step_logs = []
    total_steps = 0
    wall = 0.0

    for it in range(outer_iters):
        # 1) Dataset: each prompt asks for a full episode
        train_ds = collect_episode_prompts(
            env_cfg,
            tokenizer=tokenizer,
            episodes=episodes_per_iter,
            seed=train_cfg.seed + it,
        )

        # 2) Reward function: episode-level
        call_idx = {"i": 0}

        def reward_fn(completions, **kwargs):
            prompts = None
            for k in ("prompts", "queries", "input_texts"):
                if k in kwargs and kwargs[k] is not None:
                    prompts = kwargs[k]
                    break

            rewards = pd_episode_reward_func(
                completions,
                prompts=prompts,
                env_cfg=env_cfg,
                social_reward=train_cfg.social_reward,
                coop_bonus=getattr(train_cfg, "coop_bonus", 0.0),
                invalid_penalty=-0.5,
            )

            # For logging: count total Cs / Ds over all parsed actions
            action_counts = Counter()
            for comp in completions:
                seq = parse_episode_actions(comp or "", env_cfg.rounds)
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

            call_idx["i"] += 1
            step = call_idx["i"]
            ts = int(time.time())

            # Debug: show per-query reward std over groups
            import numpy as np
            G = max(1, train_cfg.num_generations)
            for q in range(len(completions) // G):
                group = completions[q * G : (q + 1) * G]
                g_rewards = rewards[q * G : (q + 1) * G]
                print(f"query {q}: episode_rewards={g_rewards}")
                print(f"  std_reward={np.std(g_rewards):.4f}")

            n = max(1, len(rewards))
            mean_r = float(sum(rewards) / n)
            var_r = float(sum((r - mean_r) ** 2 for r in rewards) / n)
            std_r = var_r ** 0.5

            with open(hist_path, "a") as f:
                f.write(
                    f"{it},{step},{ts},{len(completions)},"
                    f"{action_counts.get('C', 0)},{action_counts.get('D', 0)},{action_counts.get('unknown', 0)},"
                    f"{mean_r:.6f},{std_r:.6f}\n"
                )

            # ~5% sampling to jsonl for inspection
            with open(sample_path, "a") as f:
                for i, (c, r) in enumerate(zip(completions, rewards)):
                    if random.random() < 0.05:
                        row = {
                            "outer_iter": it,
                            "step": step,
                            "completion": c,
                            "reward": r,
                        }
                        if prompts is not None and i < len(prompts):
                            row["prompt"] = prompts[i]
                        f.write(json.dumps(row) + "\n")

            return rewards

        # 3) GRPO config: completions are full episodes, so allow length
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

        # Save per-iter adapter + tokenizer inside /iters/iter_XX
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Accumulate logs with an 'outer_iter' tag
        for row in trainer.state.log_history:
            row = dict(row)
            row["outer_iter"] = it
            global_step_logs.append(row)
        total_steps += trainer.state.global_step or 0

    # 4) Save the latest adapter + tokenizer directly into root_dir
    # (useful for loading as LLMPolicy(adapter_dir=save_dir))
    final_adapter_dir = args.output_dir  # last iter's folder
    # copy the adapter weights to root_dir
    # (simplest: load via LLMPolicy from final_adapter_dir; we also save a copy)
    learner = LLMPolicy(train_cfg, adapter_dir=final_adapter_dir)
    learner.model.save_pretrained(root_dir)
    tokenizer.save_pretrained(root_dir)

    # Persist trainer-style logs/configs in root_dir
    with open(os.path.join(root_dir, "train_config.json"), "w") as f:
        json.dump(asdict(train_cfg), f, indent=2)
    with open(os.path.join(root_dir, "env_config.json"), "w") as f:
        json.dump(asdict(env_cfg), f, indent=2)
    with open(os.path.join(root_dir, "log_history.json"), "w") as f:
        json.dump(global_step_logs, f, indent=2)
    with open(os.path.join(root_dir, "meta.json"), "w") as f:
        json.dump({"wall_time_sec": wall, "steps": total_steps}, f, indent=2)

    csv_path = os.path.join(root_dir, "train_history.csv")
    keys = sorted({k for d in global_step_logs for k in d.keys()})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in global_step_logs:
            w.writerow(row)

    return learner, global_step_logs