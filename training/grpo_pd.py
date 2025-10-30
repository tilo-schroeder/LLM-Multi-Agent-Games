from __future__ import annotations
import os, json, csv, time, random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import GRPOConfig, GRPOTrainer

from envs.repeated_pd import RepeatedPD, Config as EnvConfig, ACTIONS
from policy.utils import to_prompt, parse_action


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    learning_rate: float = 5e-6
    batch_size: int = 64                # global = per_device * grad_accum * n_gpus
    mini_batch_size: int = 16           # per_device_train_batch_size
    ppo_epochs: int = 1                 # mapped to num_train_epochs in GRPO
    seed: int = 0
    max_new_tokens: int = 2
    do_sample: bool = True
    temperature: float = 0.7
    social_reward: bool = True          # if True: encourage 'C', else encourage 'D'
    num_generations: int = 8            # GRPO group size per prompt


def build_pd_prompt_dataset(env_cfg: EnvConfig, episodes: int, seed: int) -> Dataset:
    rng = random.Random(seed)
    prompts: List[str] = []

    for e in range(episodes):
        # diversify with varying seeds
        e_cfg = EnvConfig(**{**env_cfg.__dict__, "seed": env_cfg.seed + e})
        env = RepeatedPD(e_cfg)
        obs = env.reset()
        for _ in range(env.cfg.rounds):
            prompts.append(to_prompt(obs["agent_0"]))
            prompts.append(to_prompt(obs["agent_1"]))
            a0 = rng.choice(ACTIONS)
            a1 = rng.choice(ACTIONS)
            obs, _, done = env.step(a0, a1)
            if done:
                break

    return Dataset.from_dict({"prompt": prompts})


def pd_reward_func(
    completions: List[str],
    *,
    social_reward: bool,
    **kwargs,
) -> List[float]:
    # Simple verifiable signal:
    # - social_reward=True: reward 1.0 for 'C'
    # - social_reward=False: reward 1.0 for 'D'
    out: List[float] = []
    for c in completions:
        a = parse_action(c)
        out.append(1.0 if ((social_reward and a == "C") or ((not social_reward) and a == "D")) else 0.0)
    return out


class LLMPolicy:
    def __init__(self, cfg: TrainConfig, adapter_dir: Optional[str] = None):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_dir or cfg.model_name, trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        base = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        if adapter_dir is not None:
            self.model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
        else:
            self.model = base
        self.model.eval()

    @torch.inference_mode()
    def act(self, prompts: List[str]) -> Tuple[List[str], List[str], None, None]:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        texts, actions = [], []
        for i, g in enumerate(gen):
            full = self.tokenizer.decode(g, skip_special_tokens=True)
            pref = prompts[i]
            completion = full[len(pref):] if full.startswith(pref) else full
            texts.append(completion)
            actions.append(parse_action(completion))
        return texts, actions, None, None


def rollout_episode(env: RepeatedPD, policy: LLMPolicy) -> Dict:
    obs = env.reset()
    done = False
    total = {"agent_0": 0.0, "agent_1": 0.0}
    actions: List[Tuple[str, str]] = []

    while not done:
        p0 = to_prompt(obs["agent_0"])
        p1 = to_prompt(obs["agent_1"])
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
# GRPO training
# ----------------------------
def grpo_train(
    env_cfg: EnvConfig,
    train_cfg: TrainConfig,
    total_episodes: int = 200,
    save_dir: str = "./ppo_adapter",
    log_every_steps: int = 10,
):
    torch.manual_seed(train_cfg.seed)
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(train_cfg.model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Build prompts
    train_ds = build_pd_prompt_dataset(env_cfg, episodes=total_episodes, seed=train_cfg.seed)

    grad_accum = max(1, train_cfg.batch_size // train_cfg.mini_batch_size)

    args = GRPOConfig(
        output_dir=save_dir,
        seed=train_cfg.seed,
        learning_rate=train_cfg.learning_rate,
        per_device_train_batch_size=train_cfg.mini_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=train_cfg.ppo_epochs,
        max_prompt_length=256,
        max_completion_length=train_cfg.max_new_tokens,
        num_generations=train_cfg.num_generations,
        temperature=train_cfg.temperature,
        remove_unused_columns=False,
        logging_steps=log_every_steps,
        save_steps=0,
        model_init_kwargs={
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        },
    )

    peft_cfg = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM")

    def reward_fn(completions, **kwargs):
        return pd_reward_func(completions, social_reward=train_cfg.social_reward, **kwargs)

    trainer = GRPOTrainer(
        model=train_cfg.model_name,
        reward_funcs=reward_fn,
        train_dataset=train_ds,
        processing_class=tokenizer,
        args=args,
        peft_config=peft_cfg,
    )

    start = time.time()
    trainer.train()
    wall = time.time() - start

    # Save adapters + tokenizer
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Persist trainer log history
    with open(os.path.join(save_dir, "train_config.json"), "w") as f:
        json.dump(asdict(train_cfg), f, indent=2)
    with open(os.path.join(save_dir, "env_config.json"), "w") as f:
        json.dump(asdict(env_cfg), f, indent=2)
    with open(os.path.join(save_dir, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump({"wall_time_sec": wall, "steps": trainer.state.global_step}, f, indent=2)

    csv_path = os.path.join(save_dir, "train_history.csv")
    keys = sorted({k for d in trainer.state.log_history for k in d.keys()})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in trainer.state.log_history:
            w.writerow(row)

    policy = LLMPolicy(train_cfg, adapter_dir=save_dir)
    logs = [{"step": r.get("step", i), **{k: v for k, v in r.items() if k != "step"}} for i, r in enumerate(trainer.state.log_history)]
    return policy, logs
