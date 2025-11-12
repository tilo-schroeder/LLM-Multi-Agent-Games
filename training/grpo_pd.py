from __future__ import annotations

import torch
from datasets import Dataset
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import GRPOConfig, GRPOTrainer
import re, uuid, os, json, csv, time, random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

from envs.repeated_pd import RepeatedPD, Config as EnvConfig, ACTIONS
from policy.utils import to_prompt, parse_action, _parse_action_strict 


_EXACT_ONE_LETTER = re.compile(r"^\s*[CD]\s*$", flags=re.IGNORECASE)
_UID_RE = re.compile(r"<UID:(\w+)>")

def _is_exact_single_letter(s: str) -> bool:
    return bool(_EXACT_ONE_LETTER.match(s or ""))


@dataclass
class StepMeta:
    opp_action: str           # 'C' or 'D' actually played this step
    recip_bonus_if_C: float   # shaping: bonus if we choose 'C' at this step
    recip_bonus_if_D: float   # shaping: bonus if we choose 'D' at this step

def make_uid() -> str:
    return uuid.uuid4().hex[:12]

def add_uid_to_prompt(prompt: str, uid: str) -> str:
    return f"{prompt}\n<ID><UID:{uid}></ID>"

def collect_selfplay_prompts(
    env_cfg: EnvConfig,
    learner: "LLMPolicy",
    opponent: "LLMPolicy",
    episodes: int,
    seed: int = 0,
    recip_weight: float = 0.05,   # small next-step reciprocity credit
):
    rng = random.Random(seed)
    data_prompts, metas = [], {}

    for e in range(episodes):
        env = RepeatedPD(EnvConfig(**{**env_cfg.__dict__, "seed": env_cfg.seed + e}))
        obs = env.reset()
        done = False
        # we need last step opp action at t and opp response at t+1 for shaping
        last_pair = None  # (uid0, uid1, opp_actions_at_t)
        while not done:
            p0 = to_prompt(obs["agent_0"]) + "\nIgnore any <ID> tags; they are bookkeeping."
            p1 = to_prompt(obs["agent_1"]) + "\nIgnore any <ID> tags; they are bookkeeping."
            # opponent acts using its own policy on *its* view
            _, opp_pair, _, _ = opponent.act([p0, p1])
            a0_opp, a1_opp = opp_pair  # opponent's actions against each role

            # record two learner decision points, one per role
            uid0, uid1 = make_uid(), make_uid()
            lp0 = add_uid_to_prompt(p0, uid0)
            lp1 = add_uid_to_prompt(p1, uid1)

            # for shaping, we set recip bonuses to zero now; we’ll fill *previous* step’s bonus using what happens next
            metas[uid0] = StepMeta(opp_action=a1_opp, recip_bonus_if_C=0.0, recip_bonus_if_D=0.0)
            metas[uid1] = StepMeta(opp_action=a0_opp, recip_bonus_if_C=0.0, recip_bonus_if_D=0.0)

            data_prompts.extend([lp0, lp1])

            # advance env using *learner-as-both* to collect a realistic next state for shaping signal
            # we use a single greedy sample to step the env (cheap and consistent)
            _, learner_pair, _, _ = learner.act([p0, p1])
            a0_learn, a1_learn = learner_pair
            obs, (_, _), done = env.step(a0_learn, a1_learn)

            # after stepping, we can reward reciprocity from the opponent's *response at t+1*:
            if last_pair is not None:
                prev_uid0, prev_uid1, (prev_opp0, prev_opp1) = last_pair
                # if the opponent *now* cooperated, give a small bonus to having chosen C at previous step
                metas[prev_uid0].recip_bonus_if_C += recip_weight * (1.0 if a1_opp == "C" else 0.0)
                metas[prev_uid1].recip_bonus_if_C += recip_weight * (1.0 if a0_opp == "C" else 0.0)

            last_pair = (uid0, uid1, (a0_opp, a1_opp))

    # build HF dataset of prompts; reward_fn will look up UID → meta
    ds = Dataset.from_dict({"prompt": data_prompts})
    return ds, metas


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    learning_rate: float = 5e-6
    batch_size: int = 64                # global = per_device * grad_accum * n_gpus
    mini_batch_size: int = 16           # per_device_train_batch_size
    grpo_epochs: int = 1                 # mapped to num_train_epochs in GRPO
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
    prompts: Optional[List[str]] = None,
    env_cfg: EnvConfig,
    social_reward: bool,
    invalid_penalty: float = -0.5,
    long_penalty: float = -0.05,
    uid_meta: Optional[Dict[str, StepMeta]] = None,
    welfare_lambda: float = 0.5,   # blend of social vs selfish
) -> List[float]:
    T, R, P, S = env_cfg.T, env_cfg.R, env_cfg.P, env_cfg.S
    denom = max(T - S, 1e-6)
    out = []

    for i, comp in enumerate(completions):
        a = _parse_action_strict(comp)
        if a not in ("C","D"):
            out.append(invalid_penalty); continue

        # exact opponent action if collected
        opp_a = None
        recip_bonus_C = recip_bonus_D = 0.0
        if prompts is not None and uid_meta is not None and i < len(prompts):
            m = _UID_RE.search(prompts[i])
            if m:
                meta = uid_meta.get(m.group(1))
                if meta:
                    opp_a = meta.opp_action
                    recip_bonus_C = meta.recip_bonus_if_C
                    recip_bonus_D = meta.recip_bonus_if_D

        if opp_a is None:
            # fall back to mild prior if no meta (should be rare)
            base = 0.5 + (0.05 if (social_reward and a == "C") else 0.0)
            if not _is_exact_single_letter(comp): base += long_penalty
            out.append(max(0.0, min(1.0, base))); continue

        # compute payoffs from actual opponent action
        if opp_a == "C":
            self_payoff = R if a == "C" else T
            opp_payoff  = R if a == "C" else S
        else:
            self_payoff = S if a == "C" else P
            opp_payoff  = T if a == "C" else P

        welfare = ((self_payoff + opp_payoff) / 2.0 - S) / denom
        selfish = (self_payoff - S) / denom
        val = welfare_lambda * welfare + (1 - welfare_lambda) * selfish

        # tiny reciprocity credit (from next-step observation)
        if a == "C": val += recip_bonus_C
        else:        val += recip_bonus_D

        if not _is_exact_single_letter(comp): val += long_penalty
        out.append(float(max(0.0, min(1.0, val))))
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
        attn = inputs["attention_mask"]
        for i in range(gen.size(0)):
            prompt_len = int(attn[i].sum().item())
            completion_ids = gen[i, prompt_len:]
            completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
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
    from collections import Counter

    torch.manual_seed(train_cfg.seed)

    # Root/train folder (old structure) + internal per-iter folder
    root_dir = save_dir
    iters_dir = os.path.join(root_dir, "iters")
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(iters_dir, exist_ok=True)

    # learner starts from base; opponent is lagged (frozen copy)
    learner = LLMPolicy(train_cfg, adapter_dir=None)
    opponent = LLMPolicy(train_cfg, adapter_dir=None)

    tokenizer = AutoTokenizer.from_pretrained(train_cfg.model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    peft_cfg = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM")

    sample_path = os.path.join(root_dir, "train_samples.jsonl")
    hist_path   = os.path.join(root_dir, "action_hist.csv")
    if not os.path.exists(hist_path):
        with open(hist_path, "w") as f:
            f.write("iter,step,timestamp,total,C,D,unknown,mean_reward,std_reward\n")

    global_step_logs = []
    total_steps = 0
    wall = 0.0

    for it in range(outer_iters):
        # 1) collect on-policy dataset vs lagged opponent
        train_ds, uid_meta = collect_selfplay_prompts(
            env_cfg, learner=learner, opponent=opponent, episodes=episodes_per_iter, seed=train_cfg.seed + it
        )

        # 2) wire reward_fn that closes over uid_meta and also writes history
        call_idx = {"i": 0}
        def reward_fn(completions, **kwargs):
            prompts = None
            for k in ("prompts","queries","input_texts"):
                if k in kwargs and kwargs[k] is not None:
                    prompts = kwargs[k]; break

            rewards = pd_reward_func(
                completions,
                prompts=prompts,
                env_cfg=env_cfg,
                social_reward=train_cfg.social_reward,
                invalid_penalty=-0.5,
                long_penalty=-0.05,
                uid_meta=uid_meta,
                welfare_lambda=0.8,
            )

            # Old-style per-call histogram
            parsed = [_parse_action_strict(c) for c in completions]
            cnt = Counter(a if a in ("C","D") else "unknown" for a in parsed)
            call_idx["i"] += 1
            step = call_idx["i"]
            ts = int(time.time())

            n = max(1, len(rewards))
            mean_r = float(sum(rewards) / n)
            var_r = float(sum((r - mean_r) ** 2 for r in rewards) / n)
            std_r = var_r ** 0.5

            with open(hist_path, "a") as f:
                f.write(
                    f"{it},{step},{ts},{len(completions)},"
                    f"{cnt.get('C',0)},{cnt.get('D',0)},{cnt.get('unknown',0)},"
                    f"{mean_r:.6f},{std_r:.6f}\n"
                )

            # ~5% sampling to jsonl
            with open(sample_path, "a") as f:
                for i, (c, r, a) in enumerate(zip(completions, rewards, parsed)):
                    if random.random() < 0.05:
                        row = {"outer_iter": it, "step": step, "completion": c, "parsed": a, "reward": r}
                        if prompts is not None and i < len(prompts):
                            row["prompt"] = prompts[i]
                        f.write(json.dumps(row) + "\n")

            return rewards

        args = GRPOConfig(
            output_dir=os.path.join(iters_dir, f"iter_{it:02d}"),
            seed=train_cfg.seed,
            learning_rate=train_cfg.learning_rate,
            per_device_train_batch_size=train_cfg.mini_batch_size,
            gradient_accumulation_steps=max(1, train_cfg.batch_size // train_cfg.mini_batch_size),
            num_train_epochs=train_cfg.grpo_epochs,
            max_prompt_length=256,
            max_completion_length=train_cfg.max_new_tokens,
            num_generations=max(8, train_cfg.num_generations),
            temperature=train_cfg.temperature,
            top_p=0.9,
            remove_unused_columns=False,
            logging_steps=log_every_steps,
            save_steps=0,
            model_init_kwargs={
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
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

        # Accumulate logs with an 'outer_iter' tag (helps CSV/JSON downstream)
        for row in trainer.state.log_history:
            row = dict(row)  # shallow copy
            row["outer_iter"] = it
            global_step_logs.append(row)
        total_steps += trainer.state.global_step or 0

        # Refresh learner/opponent to latest adapter
        learner = LLMPolicy(train_cfg, adapter_dir=args.output_dir)
        opponent = LLMPolicy(train_cfg, adapter_dir=args.output_dir)

    # Save the latest adapter + tokenizer directly into root_dir
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