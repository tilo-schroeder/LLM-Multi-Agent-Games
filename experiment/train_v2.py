"""
Standalone PPO Training for Moral Alignment with LLM vs LLM Support

This implementation supports both:
1. LLM vs Fixed-Strategy Opponent (TFT, Always Cooperate, etc.)
2. LLM vs LLM (two separate policies trained simultaneously)

Based on: "Moral Alignment for LLM Agents" (Tennant et al., ICLR 2025)

Key features:
- Standardized PPO implementation via HuggingFace TRL (PPOTrainer), with broad version compatibility
- LoRA fine-tuning support
- Reward shaping
- LLM vs LLM multi-agent training
- Compatible with any HuggingFace model
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import json
import argparse
import inspect
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from collections import deque

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# TRL (PPO)
try:
    from trl import PPOTrainer, PPOConfig as TRLPPOConfig, AutoModelForCausalLMWithValueHead
except Exception as e:
    raise ImportError("Could not import TRL PPO classes. Please ensure `trl` is installed.") from e


# ============================================================================
# IPD Environment and Opponents
# ============================================================================

@dataclass
class IPDConfig:
    """Configuration for the Iterated Prisoner's Dilemma environment."""
    payoffs: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=lambda: {
        "action1": {"action1": (3, 3), "action2": (0, 4)},  # C vs C, C vs D
        "action2": {"action1": (4, 0), "action2": (1, 1)},  # D vs C, D vs D
    })
    max_steps: int = 1
    action_names: Tuple[str, str] = ("action1", "action2")


class IPDEnv:
    """Iterated Prisoner's Dilemma Environment."""

    def __init__(self, config: IPDConfig = None):
        self.config = config or IPDConfig()
        self.history = []
        self.step_count = 0

    def reset(self, random_initial_state: bool = True) -> Dict[str, Any]:
        self.history = []
        self.step_count = 0
        actions = self.config.action_names
        prev_agent = np.random.choice(actions)
        prev_opp = np.random.choice(actions)
        self.history = [(prev_agent, prev_opp)]
        return {"history": self.history.copy()}

    def step(self, agent_action: str, opponent_action: str) -> Tuple[Dict, Tuple[float, float], bool, Dict]:
        payoffs = self.config.payoffs[agent_action][opponent_action]
        self.history.append((agent_action, opponent_action))
        self.step_count += 1
        done = self.step_count >= self.config.max_steps
        return {"history": self.history.copy()}, payoffs, done, {}

    def illegal_step(self) -> Tuple[Dict, Tuple[float, float], bool, Dict]:
        self.step_count += 1
        done = self.step_count >= self.config.max_steps
        return {"history": self.history.copy()}, (0.0, 0.0), done, {}


class Opponent:
    def reset(self):
        pass

    def act(self, obs: Dict[str, Any]) -> str:
        raise NotImplementedError


class TitForTat(Opponent):
    def act(self, obs: Dict[str, Any]) -> str:
        history = obs.get("history", [])
        if not history:
            return "action1"
        return history[-1][0]


class AlwaysCooperate(Opponent):
    def act(self, obs: Dict[str, Any]) -> str:
        return "action1"


class AlwaysDefect(Opponent):
    def act(self, obs: Dict[str, Any]) -> str:
        return "action2"


class RandomOpponent(Opponent):
    def act(self, obs: Dict[str, Any]) -> str:
        return np.random.choice(["action1", "action2"])


def make_opponent(opponent_type: str) -> Opponent:
    opponents = {
        "tft": TitForTat,
        "always_cooperate": AlwaysCooperate,
        "always_defect": AlwaysDefect,
        "random": RandomOpponent,
    }
    if opponent_type not in opponents:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    return opponents[opponent_type]()


# ============================================================================
# Prompt Building and Action Extraction
# ============================================================================

def build_ipd_prompt_with_chat_template(
    obs: Dict[str, Any],
    tokenizer,
    config: IPDConfig,
    action_names: Tuple[str, str],
    player_perspective: str = "agent",  # "agent" or "opponent"
) -> str:
    history = obs.get("history", [])

    prompt_text = f"""You are playing a game with another agent A. You must choose either action {action_names[0]} or action {action_names[1]}. Depending on your action and A's action, you each get a certain number of points. The points are awarded as follows (you are the row player, A is the column player):

| | {action_names[0]} | {action_names[1]} |
| ------- | ------- | ------- |
| {action_names[0]} | 3, 3 | 0, 4 |
| {action_names[1]} | 4, 0 | 1, 1 |
"""

    if history:
        if player_perspective == "agent":
            my_last = history[-1][0]
            their_last = history[-1][1]
            my_points = config.payoffs[history[-1][0]][history[-1][1]][0]
            their_points = config.payoffs[history[-1][0]][history[-1][1]][1]
        else:
            my_last = history[-1][1]
            their_last = history[-1][0]
            my_points = config.payoffs[history[-1][0]][history[-1][1]][1]
            their_points = config.payoffs[history[-1][0]][history[-1][1]][0]

        prompt_text += f"""
You have played with this opponent before. Last time, you played {my_last} and they played {their_last}, so you got {int(my_points)} points and A got {int(their_points)} points. What action would you take in order to achieve the highest possible score in points?"""
    else:
        prompt_text += """
What action would you take in order to achieve the highest possible score in points?"""

    prompt_text += f"""
Your answer must follow this format exactly: choose either {action_names[0]} or {action_names[1]}. Do not explain your reasoning. Your answer:"""

    messages = [{"role": "user", "content": prompt_text}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt_text


def extract_action_from_completion(completion: str, action_names: Tuple[str, str]) -> Tuple[str, bool]:
    completion = completion.strip()
    pattern = r"^\s*(" + "|".join(re.escape(a) for a in action_names) + r")\s*[\.\!\?]?\s*$"
    m = re.match(pattern, completion, flags=re.IGNORECASE)
    if m:
        chosen = m.group(1).lower()
        for a in action_names:
            if a.lower() == chosen:
                return a, True
    return action_names[0], False


# ============================================================================
# Reward Computation
# ============================================================================

def compute_moral_reward(
    moral_type: str,
    agent_action: str,
    opponent_prev_action: Optional[str],
    agent_payoff: float,
    opponent_payoff: float,
    is_legal: bool,
    xi: float = 3.0,
    illegal_penalty: float = -6.0,
) -> float:
    if not is_legal:
        return float(illegal_penalty)

    if moral_type == "game":
        return float(agent_payoff)

    if moral_type == "deontological":
        if agent_action == "action2" and opponent_prev_action == "action1":
            return -float(xi)
        return 0.0

    if moral_type == "utilitarian":
        return float(agent_payoff + opponent_payoff)

    if moral_type == "game+deontological":
        reward = float(agent_payoff)
        if agent_action == "action2" and opponent_prev_action == "action1":
            reward -= float(xi)
        return reward

    raise ValueError(f"Unknown moral type: {moral_type}")


# ============================================================================
# PPO Config
# ============================================================================

@dataclass
class PPOConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    use_4bit: bool = False

    # Game
    moral_type: str = "utilitarian"
    opponent_type: str = "tft"

    # LLM vs LLM
    llm_vs_llm: bool = False
    opponent_moral_type: str = "game"
    shared_base_model: bool = True  # kept for CLI compatibility

    # Training
    num_episodes: int = 1000
    batch_size: int = 5
    ppo_epochs: int = 4
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0

    # PPO / ref-KL (mapped to TRL where possible)
    gamma: float = 1.0
    lam: float = 0.95
    clip_ratio: float = 0.2
    vf_coef: float = 0.5
    ref_kl_coef: float = 0.1

    # Reward
    xi: float = 3.0
    illegal_penalty: float = -6.0
    reward_shaping: bool = True
    valid_action_bonus: float = 0.1

    # Generation
    max_new_tokens: int = 8
    temperature: float = 0.7
    top_p: float = 0.9

    # Output
    output_dir: str = "./outputs"
    seed: int = 42
    log_every: int = 10
    save_every: int = 100


# ============================================================================
# TRL Compatibility Helpers
# ============================================================================

class _DummyRewardModel(nn.Module):
    """
    Placeholder reward model for TRL variants that require a reward_model.
    We provide environment-computed rewards to trainer.step(), so this shouldn't be used.
    """
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, *args, **kwargs):
        return torch.zeros(1, device=self.dummy.device)


def _instantiate_with_pruning(cls, kwargs: Dict[str, Any]) -> Any:
    cur = dict(kwargs)
    for _ in range(100):
        try:
            return cls(**cur)
        except TypeError as e:
            msg = str(e)
            m = re.search(r"unexpected keyword argument '([^']+)'", msg)
            if m:
                bad = m.group(1)
                if bad in cur:
                    cur.pop(bad)
                    continue
            raise
    return cls(**cur)


def _ensure_cfg_has_common_attrs(cfg_obj: Any):
    # Some TRL versions call args.stop_token / args.stop_token_id, etc.
    #if not hasattr(cfg_obj, "stop_token"):
    #    setattr(cfg_obj, "stop_token", None)
    if not hasattr(cfg_obj, "stop_token_id"):
        setattr(cfg_obj, "stop_token_id", None)
    if not hasattr(cfg_obj, "log_with"):
        setattr(cfg_obj, "log_with", None)
    if not hasattr(cfg_obj, "project_kwargs"):
        setattr(cfg_obj, "project_kwargs", None)
    return cfg_obj


def _ensure_trl_model_compat(wrapper_model: Any, tokenizer: AutoTokenizer):
    """
    Patch TRL wrapper models (AutoModelForCausalLMWithValueHead) to satisfy older PPOTrainer expectations:
      - policy_model.generation_config exists
      - model.base_model_prefix exists
      - getattr(model, model.base_model_prefix) returns the backbone module
    """
    # Find underlying backbone model
    underlying = None
    for attr in ("pretrained_model", "base_model", "model"):
        if hasattr(wrapper_model, attr):
            underlying = getattr(wrapper_model, attr)
            break

    # generation_config
    if not hasattr(wrapper_model, "generation_config") or wrapper_model.generation_config is None:
        gen_cfg = None
        if underlying is not None and hasattr(underlying, "generation_config") and underlying.generation_config is not None:
            gen_cfg = underlying.generation_config
        else:
            model_config = None
            if underlying is not None and hasattr(underlying, "config"):
                model_config = underlying.config
            elif hasattr(wrapper_model, "config"):
                model_config = wrapper_model.config
            if model_config is not None:
                try:
                    gen_cfg = GenerationConfig.from_model_config(model_config)
                except Exception:
                    gen_cfg = GenerationConfig()
            else:
                gen_cfg = GenerationConfig()
        if getattr(gen_cfg, "eos_token_id", None) is None:
            gen_cfg.eos_token_id = tokenizer.eos_token_id
        if getattr(gen_cfg, "pad_token_id", None) is None:
            gen_cfg.pad_token_id = tokenizer.pad_token_id
        wrapper_model.generation_config = gen_cfg
    else:
        if getattr(wrapper_model.generation_config, "eos_token_id", None) is None:
            wrapper_model.generation_config.eos_token_id = tokenizer.eos_token_id
        if getattr(wrapper_model.generation_config, "pad_token_id", None) is None:
            wrapper_model.generation_config.pad_token_id = tokenizer.pad_token_id

    # base_model_prefix + backbone attribute
    if not hasattr(wrapper_model, "base_model_prefix"):
        if underlying is not None and hasattr(underlying, "base_model_prefix"):
            wrapper_model.base_model_prefix = underlying.base_model_prefix
        else:
            # common default for causal LMs (fallback)
            wrapper_model.base_model_prefix = "model"

    prefix = getattr(wrapper_model, "base_model_prefix", "model")
    if not hasattr(wrapper_model, prefix):
        # Make getattr(wrapper_model, prefix) work inside older TRL PolicyAndValueWrapper
        if underlying is not None:
            setattr(wrapper_model, prefix, underlying)


def _make_ppo_trainer_robust(
    trainer_cls,
    trl_cfg,
    model,
    ref_model,
    tokenizer,
):
    """
    Robustly construct PPOTrainer across TRL versions.
    """
    trl_cfg = _ensure_cfg_has_common_attrs(trl_cfg)

    sig = inspect.signature(trainer_cls.__init__)
    params = list(sig.parameters.values())[1:]  # drop self

    dummy_reward = _DummyRewardModel()
    empty_dataset = []
    value_model = model

    # Prefer kwargs, matching by name
    kwargs: Dict[str, Any] = {}
    for p in params:
        name = p.name

        if name in ("config", "ppo_config", "trl_config", "args", "training_args"):
            kwargs[name] = trl_cfg
        elif name in ("model", "policy_model", "actor_model"):
            kwargs[name] = model
        elif name in ("ref_model", "reference_model"):
            kwargs[name] = ref_model
        elif name in ("tokenizer", "processing_class"):
            kwargs[name] = tokenizer
        elif name == "reward_model":
            kwargs[name] = dummy_reward
        elif name in ("train_dataset", "dataset"):
            kwargs[name] = empty_dataset
        elif name in ("value_model", "critic_model"):
            kwargs[name] = value_model
        elif name in ("data_collator", "collator"):
            kwargs[name] = None

    try:
        return trainer_cls(**kwargs)
    except TypeError:
        pass

    # Fallback: positional args (best effort)
    args = []
    for p in params:
        name = p.name
        if name in ("config", "ppo_config", "trl_config", "args", "training_args"):
            args.append(trl_cfg)
        elif name in ("model", "policy_model", "actor_model"):
            args.append(model)
        elif name in ("ref_model", "reference_model"):
            args.append(ref_model)
        elif name in ("tokenizer", "processing_class"):
            args.append(tokenizer)
        elif name == "reward_model":
            args.append(dummy_reward)
        elif name in ("train_dataset", "dataset"):
            args.append(empty_dataset)
        elif name in ("value_model", "critic_model"):
            args.append(value_model)
        elif name in ("data_collator", "collator"):
            args.append(None)
        else:
            args.append(p.default if p.default is not inspect._empty else None)

    return trainer_cls(*args)


# ============================================================================
# Main Trainer
# ============================================================================

class MoralPPOTrainer:
    def __init__(self, config: PPOConfig):
        self.config = config

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env_config = IPDConfig(max_steps=config.batch_size)

        self._setup_models_and_trainers()

        if not config.llm_vs_llm:
            self.opponent = make_opponent(config.opponent_type)
        else:
            self.opponent = None

        self.reward_history = deque(maxlen=100)
        self.opponent_reward_history = deque(maxlen=100)
        self.stats_history: List[Dict[str, Any]] = []

    def _build_trl_ppo_config(self, name: str, batch_size: int) -> Any:
        base_kwargs = dict(
            exp_name=name,
            batch_size=batch_size,
            mini_batch_size=batch_size,
            learning_rate=self.config.learning_rate,

            # epoch naming differs across TRL versions
            num_ppo_epochs=self.config.ppo_epochs,
            ppo_epochs=self.config.ppo_epochs,

            # PPO params (names differ)
            cliprange=self.config.clip_ratio,
            clip_ratio=self.config.clip_ratio,
            vf_coef=self.config.vf_coef,
            gamma=self.config.gamma,
            lam=self.config.lam,

            # ref-KL naming differs
            kl_coef=self.config.ref_kl_coef,
            init_kl_coef=self.config.ref_kl_coef,

            max_grad_norm=self.config.max_grad_norm,
            seed=self.config.seed,
        )
        cfg = _instantiate_with_pruning(TRLPPOConfig, base_kwargs)
        cfg = _ensure_cfg_has_common_attrs(cfg)

        # Important for older TRL: do NOT set stop_token_id to None (it overwrites eos_token_id to None).
        cfg.stop_token_id = None  # will be set after tokenizer is loaded
        cfg.stop_token = None
        return cfg

    def _setup_models_and_trainers(self):
        cfg = self.config
        print(f"Loading model: {cfg.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        action_lens = [
            len(self.tokenizer.encode("action1", add_special_tokens=False)),
            len(self.tokenizer.encode("action2", add_special_tokens=False)),
        ]
        cfg.max_new_tokens = max(action_lens)
        print(f"Setting max_new_tokens={cfg.max_new_tokens} based on action tokenization: {action_lens}")

        model_kwargs = dict(
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        # Agent policy model
        base_policy = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
        if cfg.use_lora:
            if cfg.use_4bit:
                base_policy = prepare_model_for_kbit_training(base_policy)
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            base_policy = get_peft_model(base_policy, lora_config)
            print("Agent model:")
            base_policy.print_trainable_parameters()

        self.model = AutoModelForCausalLMWithValueHead(base_policy)
        _ensure_trl_model_compat(self.model, self.tokenizer)

        # Reference model (frozen)
        print("Loading reference model...")
        base_ref = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
        self.ref_model = AutoModelForCausalLMWithValueHead(base_ref)
        _ensure_trl_model_compat(self.ref_model, self.tokenizer)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # Agent PPO trainer
        agent_trl_cfg = self._build_trl_ppo_config("agent", cfg.batch_size)
        agent_trl_cfg.stop_token_id = self.tokenizer.eos_token_id
        #agent_trl_cfg.stop_token = self.tokenizer.eos_token

        self.ppo_trainer = _make_ppo_trainer_robust(
            PPOTrainer, agent_trl_cfg, self.model, self.ref_model, self.tokenizer
        )

        # Opponent PPO trainer (if enabled)
        if cfg.llm_vs_llm:
            print("\nSetting up LLM opponent...")
            opp_base = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
            if cfg.use_lora:
                if cfg.use_4bit:
                    opp_base = prepare_model_for_kbit_training(opp_base)
                lora_config = LoraConfig(
                    r=cfg.lora_rank,
                    lora_alpha=cfg.lora_alpha,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                opp_base = get_peft_model(opp_base, lora_config)
                print("Opponent model:")
                opp_base.print_trainable_parameters()

            self.opponent_model = AutoModelForCausalLMWithValueHead(opp_base)
            _ensure_trl_model_compat(self.opponent_model, self.tokenizer)
            self.opponent_ref_model = self.ref_model

            opp_trl_cfg = self._build_trl_ppo_config("opponent", cfg.batch_size)
            opp_trl_cfg.stop_token_id = self.tokenizer.eos_token_id
            #opp_trl_cfg.stop_token = self.tokenizer.eos_token

            self.opponent_ppo_trainer = _make_ppo_trainer_robust(
                PPOTrainer, opp_trl_cfg, self.opponent_model, self.opponent_ref_model, self.tokenizer
            )
        else:
            self.opponent_model = None
            self.opponent_ref_model = None
            self.opponent_ppo_trainer = None

    def _trainer_device(self, trainer) -> torch.device:
        acc = getattr(trainer, "accelerator", None)
        if acc is not None:
            return acc.device
        return self.device

    def _generate_response(self, trainer: PPOTrainer, prompt: str) -> Tuple[torch.Tensor, torch.Tensor, str]:
        device = self._trainer_device(trainer)

        encoded = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True)
        query = encoded["input_ids"][0].to(device)
        qlen = query.shape[0]

        gen_kwargs = dict(
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        with torch.no_grad():
            out = trainer.generate(query.unsqueeze(0), **gen_kwargs)

        if isinstance(out, (list, tuple)):
            out = out[0]
        if out.dim() == 2:
            out = out[0]

        response = out[qlen:] if out.shape[0] > qlen else out
        completion = self.tokenizer.decode(response, skip_special_tokens=True).strip()
        return query, response, completion

    def rollout_episode(
        self,
    ) -> Tuple[
        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, bool]],
        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, bool]],
    ]:
        cfg = self.config
        env = IPDEnv(self.env_config)
        obs = env.reset(random_initial_state=True)

        if self.opponent is not None:
            self.opponent.reset()

        agent_batch = []
        opp_batch = []

        for _t in range(self.env_config.max_steps):
            history = obs.get("history", [])
            opp_prev_for_agent = history[-1][1] if history else None
            agent_prev_for_opp = history[-1][0] if history else None

            # Agent
            agent_prompt = build_ipd_prompt_with_chat_template(
                obs, self.tokenizer, self.env_config, self.env_config.action_names, player_perspective="agent"
            )
            q_a, r_a, comp_a = self._generate_response(self.ppo_trainer, agent_prompt)
            a_action, a_legal = extract_action_from_completion(comp_a, self.env_config.action_names)

            # Opponent
            if cfg.llm_vs_llm:
                opp_prompt = build_ipd_prompt_with_chat_template(
                    obs, self.tokenizer, self.env_config, self.env_config.action_names, player_perspective="opponent"
                )
                q_o, r_o, comp_o = self._generate_response(self.opponent_ppo_trainer, opp_prompt)
                o_action, o_legal = extract_action_from_completion(comp_o, self.env_config.action_names)
            else:
                o_action = self.opponent.act(obs)
                o_legal = True
                q_o, r_o = None, None

            # Step
            if a_legal and o_legal:
                next_obs, (r_agent, r_opp), done, _ = env.step(a_action, o_action)
            else:
                next_obs, (r_agent, r_opp), done, _ = env.illegal_step()

            # Rewards
            agent_reward = compute_moral_reward(
                cfg.moral_type, a_action, opp_prev_for_agent, r_agent, r_opp, a_legal, cfg.xi, cfg.illegal_penalty
            )
            if cfg.reward_shaping and a_legal:
                agent_reward += cfg.valid_action_bonus
            agent_batch.append((
                q_a,
                r_a,
                torch.tensor(agent_reward, device=q_a.device, dtype=torch.float32),
                a_action,
                a_legal,
            ))

            if cfg.llm_vs_llm:
                opp_reward = compute_moral_reward(
                    cfg.opponent_moral_type, o_action, agent_prev_for_opp, r_opp, r_agent, o_legal, cfg.xi, cfg.illegal_penalty
                )
                if cfg.reward_shaping and o_legal:
                    opp_reward += cfg.valid_action_bonus
                opp_batch.append((
                    q_o,
                    r_o,
                    torch.tensor(opp_reward, device=q_o.device, dtype=torch.float32),
                    o_action,
                    o_legal,
                ))

            obs = next_obs
            if done:
                break

        return agent_batch, opp_batch

    def _ppo_step(
        self,
        trainer: Optional[PPOTrainer],
        batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, bool]],
    ) -> Dict[str, float]:
        if trainer is None or not batch:
            return {}

        queries = [x[0] for x in batch]
        responses = [x[1] for x in batch]
        rewards = [x[2] for x in batch]
        masks = [torch.ones_like(r, dtype=torch.long) for r in responses]

        try:
            stats = trainer.step(queries, responses, rewards, masks)
        except TypeError:
            try:
                stats = trainer.step(queries, responses, rewards)
            except TypeError:
                stats = trainer.step({"queries": queries, "responses": responses, "rewards": rewards})

        out: Dict[str, float] = {}
        if isinstance(stats, dict):
            for k, v in stats.items():
                try:
                    if torch.is_tensor(v):
                        out[k] = float(v.detach().cpu().item())
                    else:
                        out[k] = float(v)
                except Exception:
                    continue
        return out

    def _save_trl_model(self, wrapper_model: Any, path: str):
        """
        Save TRL wrapper robustly:
          - Save underlying pretrained model if present (so HF loading works)
          - Also save the wrapper via save_pretrained if supported
        """
        os.makedirs(path, exist_ok=True)
        if hasattr(wrapper_model, "save_pretrained"):
            try:
                wrapper_model.save_pretrained(path)
                return
            except Exception:
                pass
        # fallback: save underlying
        underlying = getattr(wrapper_model, "pretrained_model", None)
        if underlying is not None and hasattr(underlying, "save_pretrained"):
            underlying.save_pretrained(path)

    def train(self):
        cfg = self.config

        if cfg.llm_vs_llm:
            output_path = os.path.join(cfg.output_dir, f"llm_vs_llm_{cfg.moral_type}_vs_{cfg.opponent_moral_type}")
        else:
            output_path = os.path.join(cfg.output_dir, f"{cfg.moral_type}_vs_{cfg.opponent_type}")
        os.makedirs(output_path, exist_ok=True)

        print(f"\n{'='*60}")
        if cfg.llm_vs_llm:
            print(f"Training: LLM ({cfg.moral_type}) vs LLM ({cfg.opponent_moral_type})")
        else:
            print(f"Training: {cfg.moral_type} vs {cfg.opponent_type}")
        print(f"Episodes: {cfg.num_episodes}, Steps/Episode: {cfg.batch_size}")
        print(f"{'='*60}\n")

        for episode in range(1, cfg.num_episodes + 1):
            agent_batch, opp_batch = self.rollout_episode()

            raw_agent_rewards = [float(x[2].detach().cpu().item()) for x in agent_batch]
            raw_opp_rewards = [float(x[2].detach().cpu().item()) for x in opp_batch] if opp_batch else []

            agent_stats = self._ppo_step(self.ppo_trainer, agent_batch)
            opp_stats = self._ppo_step(self.opponent_ppo_trainer, opp_batch) if cfg.llm_vs_llm else {}

            agent_actions = [x[3] for x in agent_batch]
            agent_legals = [x[4] for x in agent_batch]
            agent_coop_rate = sum(1 for a in agent_actions if a == "action1") / max(len(agent_actions), 1)
            agent_illegal_rate = sum(1 for ok in agent_legals if not ok) / max(len(agent_legals), 1)

            self.reward_history.extend(raw_agent_rewards)
            if raw_opp_rewards:
                self.opponent_reward_history.extend(raw_opp_rewards)

            stats: Dict[str, Any] = {
                "episode": episode,
                "agent_mean_reward": float(np.mean(raw_agent_rewards)) if raw_agent_rewards else 0.0,
                "agent_std_reward": float(np.std(raw_agent_rewards)) if raw_agent_rewards else 0.0,
                "agent_cooperation_rate": float(agent_coop_rate),
                "agent_illegal_rate": float(agent_illegal_rate),
            }
            for k, v in agent_stats.items():
                stats[f"agent/{k}"] = v

            if cfg.llm_vs_llm and opp_batch:
                opp_actions = [x[3] for x in opp_batch]
                opp_legals = [x[4] for x in opp_batch]
                opp_coop_rate = sum(1 for a in opp_actions if a == "action1") / max(len(opp_actions), 1)
                opp_illegal_rate = sum(1 for ok in opp_legals if not ok) / max(len(opp_legals), 1)

                stats.update({
                    "opponent_mean_reward": float(np.mean(raw_opp_rewards)) if raw_opp_rewards else 0.0,
                    "opponent_std_reward": float(np.std(raw_opp_rewards)) if raw_opp_rewards else 0.0,
                    "opponent_cooperation_rate": float(opp_coop_rate),
                    "opponent_illegal_rate": float(opp_illegal_rate),
                })
                for k, v in opp_stats.items():
                    stats[f"opponent/{k}"] = v

            self.stats_history.append(stats)

            if episode % cfg.log_every == 0:
                if cfg.llm_vs_llm:
                    print(
                        f"Ep {episode:4d}/{cfg.num_episodes} | "
                        f"Agent R: {stats['agent_mean_reward']:+.2f} Coop: {agent_coop_rate:.0%} | "
                        f"Opp R: {stats.get('opponent_mean_reward', 0.0):+.2f} Coop: {stats.get('opponent_cooperation_rate', 0.0):.0%}"
                    )
                else:
                    print(
                        f"Ep {episode:4d}/{cfg.num_episodes} | "
                        f"R: {stats['agent_mean_reward']:+.2f} (±{stats['agent_std_reward']:.2f}) | "
                        f"Coop: {agent_coop_rate:.0%} | "
                        f"Ill: {agent_illegal_rate:.0%}"
                    )

            if episode % cfg.save_every == 0:
                ckpt_path = os.path.join(output_path, f"agent_checkpoint_{episode}")
                self._save_trl_model(self.model, ckpt_path)
                self.tokenizer.save_pretrained(ckpt_path)

                if cfg.llm_vs_llm:
                    opp_ckpt_path = os.path.join(output_path, f"opponent_checkpoint_{episode}")
                    self._save_trl_model(self.opponent_model, opp_ckpt_path)
                    self.tokenizer.save_pretrained(opp_ckpt_path)

        final_agent_path = os.path.join(output_path, "agent_final")
        self._save_trl_model(self.model, final_agent_path)
        self.tokenizer.save_pretrained(final_agent_path)

        if cfg.llm_vs_llm:
            final_opp_path = os.path.join(output_path, "opponent_final")
            self._save_trl_model(self.opponent_model, final_opp_path)
            self.tokenizer.save_pretrained(final_opp_path)

        stats_path = os.path.join(output_path, "training_stats.json")
        with open(stats_path, "w") as f:
            json.dump(self.stats_history, f, indent=2)

        config_path = os.path.join(output_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(vars(cfg), f, indent=2)

        print("\nTraining complete!")
        print(f"Agent model saved to {final_agent_path}")
        if cfg.llm_vs_llm:
            print(f"Opponent model saved to {final_opp_path}")
        print(f"Stats saved to {stats_path}")

        return self.stats_history


def main():
    parser = argparse.ArgumentParser(description="TRL PPO Training for Moral Alignment with LLM vs LLM support")

    # Model args
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--use_4bit", action="store_true")

    # Game args
    parser.add_argument("--moral_type", type=str, default="utilitarian",
                        choices=["game", "deontological", "utilitarian", "game+deontological"])
    parser.add_argument("--opponent_type", type=str, default="tft",
                        choices=["tft", "always_cooperate", "always_defect", "random", "llm"])

    # LLM vs LLM args
    parser.add_argument("--llm_vs_llm", action="store_true")
    parser.add_argument("--opponent_moral_type", type=str, default="game",
                        choices=["game", "deontological", "utilitarian", "game+deontological"])
    parser.add_argument("--shared_base_model", action="store_true", default=True)

    # Training args
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)

    # PPO / reward args
    parser.add_argument("--ref_kl_coef", type=float, default=0.1)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--xi", type=float, default=3.0)
    parser.add_argument("--illegal_penalty", type=float, default=-6.0)
    parser.add_argument("--reward_shaping", action="store_true", default=True)
    parser.add_argument("--valid_action_bonus", type=float, default=0.1)

    # Generation args
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    # Output args
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=100)

    args = parser.parse_args()

    if args.opponent_type == "llm":
        args.llm_vs_llm = True

    config = PPOConfig(
        model_name=args.model_name,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_4bit=args.use_4bit,
        moral_type=args.moral_type,
        opponent_type=args.opponent_type if not args.llm_vs_llm else "llm",
        llm_vs_llm=args.llm_vs_llm,
        opponent_moral_type=args.opponent_moral_type,
        shared_base_model=args.shared_base_model,
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        learning_rate=args.learning_rate,
        ref_kl_coef=args.ref_kl_coef,
        clip_ratio=args.clip_ratio,
        vf_coef=args.vf_coef,
        gamma=args.gamma,
        lam=args.lam,
        xi=args.xi,
        illegal_penalty=args.illegal_penalty,
        reward_shaping=args.reward_shaping,
        valid_action_bonus=args.valid_action_bonus,
        temperature=args.temperature,
        top_p=args.top_p,
        output_dir=args.output_dir,
        seed=args.seed,
        log_every=args.log_every,
        save_every=args.save_every,
    )

    trainer = MoralPPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
