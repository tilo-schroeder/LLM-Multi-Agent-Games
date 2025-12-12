"""
Standalone PPO Training for Moral Alignment with Multi-Player Stag Hunt (2+ players)

This implementation supports both:
1) LLM vs Fixed-Strategy Opponents (for players 1..N-1)
2) LLM vs LLM (all N players are separate policies trained simultaneously)

Game: N-player Stag Hunt (true joint-action payoffs; NO round-robin)

Key features:
- Clean PPO implementation with KL penalty
- LoRA fine-tuning support
- Reward scaling and normalization
- Multi-agent (N-player) training
- Compatible with any HuggingFace causal LM

Notes:
- By default, only Player 0 is an LLM policy unless --llm_vs_llm is set.
- If you train many LLM players, memory use scales with #players (one model per player).
"""

import re
import os
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ============================================================================
# N-Player Stag Hunt Environment (true joint action payoffs)
# ============================================================================

@dataclass
class StagHuntConfig:
    """Configuration for the N-player Stag Hunt environment."""
    num_players: int = 2
    max_steps: int = 1

    action_names: Tuple[str, str] = ("stag", "hare")

    # Threshold Stag Hunt:
    # - If #stag >= threshold: stag players get stag_success_reward
    # - Else: stag players get stag_fail_reward
    # - Hare players always get hare_reward
    threshold: int = 2
    stag_success_reward: float = 4.0
    stag_fail_reward: float = 0.0
    hare_reward: float = 2.0


class StagHuntEnv:
    """
    True N-player Stag Hunt. Payoffs computed from the joint action profile.
    history is List[List[str]] of joint action vectors (length num_players).
    """

    def __init__(self, config: StagHuntConfig = None):
        self.config = config or StagHuntConfig()
        self.history: List[List[str]] = []
        self.step_count = 0

    def reset(self, random_initial_state: bool = True) -> Dict[str, Any]:
        self.history = []
        self.step_count = 0

        # Keep the "random previous move" initialization pattern from the IPD script
        if random_initial_state:
            acts = list(self.config.action_names)
            prev = [str(np.random.choice(acts)) for _ in range(self.config.num_players)]
            self.history.append(prev)

        return {"history": [h.copy() for h in self.history]}

    def step(self, actions: List[str]) -> Tuple[Dict[str, Any], List[float], bool, Dict]:
        cfg = self.config
        assert len(actions) == cfg.num_players

        k = sum(1 for a in actions if a == cfg.action_names[0])  # #stag
        success = (k >= cfg.threshold)

        payoffs = []
        for a in actions:
            if a == cfg.action_names[1]:  # hare
                payoffs.append(cfg.hare_reward)
            else:  # stag
                payoffs.append(cfg.stag_success_reward if success else cfg.stag_fail_reward)

        self.history.append(list(actions))
        self.step_count += 1
        done = self.step_count >= cfg.max_steps
        return {"history": [h.copy() for h in self.history]}, payoffs, done, {}

    def illegal_step(self) -> Tuple[Dict[str, Any], List[float], bool, Dict]:
        self.step_count += 1
        done = self.step_count >= self.config.max_steps
        return {"history": [h.copy() for h in self.history]}, [0.0] * self.config.num_players, done, {}


# ============================================================================
# Fixed Opponents (for non-LLM players)
# ============================================================================

class Opponent:
    def reset(self):
        pass

    def act(self, obs: Dict[str, Any], player_id: int, focal_id: int = 0) -> str:
        raise NotImplementedError


class AlwaysStag(Opponent):
    def act(self, obs: Dict[str, Any], player_id: int, focal_id: int = 0) -> str:
        return "stag"


class AlwaysHare(Opponent):
    def act(self, obs: Dict[str, Any], player_id: int, focal_id: int = 0) -> str:
        return "hare"


class RandomOpponent(Opponent):
    def act(self, obs: Dict[str, Any], player_id: int, focal_id: int = 0) -> str:
        return np.random.choice(["stag", "hare"])


class CopyFocalLast(Opponent):
    """
    Copies the focal player's last action (default focal=Player 0).
    This is a rough "TFT-like" behavior in a simultaneous-action setting.
    """

    def act(self, obs: Dict[str, Any], player_id: int, focal_id: int = 0) -> str:
        history = obs.get("history", [])
        if not history:
            return "hare"
        last = history[-1]
        return last[focal_id]


def make_opponent(opponent_type: str) -> Opponent:
    opponents = {
        "always_stag": AlwaysStag,
        "always_hare": AlwaysHare,
        "random": RandomOpponent,
        "copy_focal": CopyFocalLast,
    }
    if opponent_type not in opponents:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    return opponents[opponent_type]()


# ============================================================================
# Prompt Building and Action Extraction
# ============================================================================

def _player_label(i: int) -> str:
    return chr(ord("A") + i) if i < 26 else f"P{i}"


def build_stag_hunt_prompt_with_chat_template(
    obs: Dict[str, Any],
    tokenizer,
    cfg: StagHuntConfig,
    player_id: int,
) -> str:
    me = _player_label(player_id)
    history = obs.get("history", [])

    stag, hare = cfg.action_names
    prompt = f"""You are playing a game with {cfg.num_players-1} other agents. You are Player {me}.
Each round, every player chooses one action: {stag} or {hare}.

Payoffs:
- If at least {cfg.threshold} players choose {stag}, then each player who chose {stag} gets {cfg.stag_success_reward} points.
- Otherwise, each player who chose {stag} gets {cfg.stag_fail_reward} points.
- Any player who chose {hare} always gets {cfg.hare_reward} points.

"""

    if history:
        last = history[-1]
        prompt += "Last round actions:\n"
        for i, a in enumerate(last):
            prompt += f"- Player {_player_label(i)} played {a}\n"

    prompt += f"""
What action would you take in order to achieve the highest possible score in points?

Your answer must follow this format exactly: choose either {stag} or {hare}. Do not explain your reasoning. Your answer:"""

    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt


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
# Moral Reward Computation (multi-player)
# ============================================================================

def compute_moral_reward(
    moral_type: str,
    agent_action: str,
    opponent_prev_actions: Optional[List[str]],
    agent_payoff: float,
    all_payoffs: List[float],
    is_legal: bool,
    xi: float = 3.0,
    illegal_penalty: float = -6.0,
) -> float:
    if not is_legal:
        return illegal_penalty

    if moral_type == "game":
        return float(agent_payoff)

    elif moral_type == "deontological":
        # Example norm: penalize choosing hare when others previously chose stag (scaled by count).
        # You can swap this for whatever norm you want.
        if opponent_prev_actions and agent_action == "hare":
            stag_count = sum(1 for a in opponent_prev_actions if a == "stag")
            if stag_count > 0:
                return -xi * float(stag_count)
        return 0.0

    elif moral_type == "utilitarian":
        return float(sum(all_payoffs))

    elif moral_type == "game+deontological":
        reward = float(agent_payoff)
        if opponent_prev_actions and agent_action == "hare":
            stag_count = sum(1 for a in opponent_prev_actions if a == "stag")
            if stag_count > 0:
                reward -= xi * float(stag_count)
        return reward

    else:
        raise ValueError(f"Unknown moral type: {moral_type}")


# ============================================================================
# PPO Components
# ============================================================================

@dataclass
class PPOConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    use_4bit: bool = False

    # Game (Stag Hunt)
    num_players: int = 2
    threshold: int = 2
    stag_success_reward: float = 4.0
    stag_fail_reward: float = 0.0
    hare_reward: float = 2.0

    # Moral types
    moral_type: str = "utilitarian"         # default for player 0
    opponent_moral_type: str = "game"       # default for others if llm_vs_llm
    player_moral_types: List[str] = field(default_factory=list)

    # Opponent settings (when not llm_vs_llm)
    opponent_type: str = "copy_focal"       # used for players 1..N-1 in fixed mode

    # Multi-agent training mode
    llm_vs_llm: bool = False

    # Training
    num_episodes: int = 1000
    batch_size: int = 5           # used as env.max_steps
    ppo_epochs: int = 4
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0

    # PPO hyperparameters
    gamma: float = 1.0
    lam: float = 0.95
    clip_ratio: float = 0.2
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.1
    target_kl: Optional[float] = 0.05

    # KL to reference model (RLHF-style shaping)
    ref_kl_coef: float = 0.1

    # Reward shaping / normalization
    xi: float = 3.0
    illegal_penalty: float = -6.0
    reward_scale: float = 1.0
    normalize_rewards: bool = False
    normalize_advantages: bool = True
    reward_shaping: bool = True
    valid_action_bonus: float = 0.1

    # Generation
    max_new_tokens: int = 8  # will be overwritten based on tokenization of actions
    temperature: float = 0.7
    top_p: float = 0.9

    # Output
    output_dir: str = "./outputs"
    seed: int = 42
    log_every: int = 10
    save_every: int = 100


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_size, hidden_size // 2)
        self.linear2 = nn.Linear(hidden_size // 2, 1)

        nn.init.normal_(self.linear1.weight, std=0.01)
        nn.init.zeros_(self.linear1.bias)
        nn.init.normal_(self.linear2.weight, std=0.01)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 3:
            x = hidden_states[:, -1, :]
        elif hidden_states.dim() == 2:
            x = hidden_states
        else:
            raise ValueError(f"hidden_states must be 2D or 3D, got {hidden_states.shape}")

        if x.dtype != self.linear1.weight.dtype:
            x = x.to(self.linear1.weight.dtype)

        x = self.dropout(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x.squeeze(-1)


class PolicyModelWithValueHead(nn.Module):
    def __init__(self, base_model, hidden_size: int, device=None):
        super().__init__()
        self.base_model = base_model
        self.value_head = ValueHead(hidden_size)
        self.config = base_model.config
        self._device = device

        if device is not None:
            self.value_head = self.value_head.to(device)

        try:
            model_dtype = next(base_model.parameters()).dtype
            self.value_head = self.value_head.to(model_dtype)
        except StopIteration:
            pass

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        hidden_states = outputs.hidden_states[-1]

        if self.value_head.linear1.weight.device != hidden_states.device:
            self.value_head = self.value_head.to(hidden_states.device)
        if self.value_head.linear1.weight.dtype != hidden_states.dtype:
            self.value_head = self.value_head.to(hidden_states.dtype)

        value = self.value_head(hidden_states)
        return outputs, value

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.base_model.save_pretrained(path)
        torch.save(
            {k: v.cpu() for k, v in self.value_head.state_dict().items()},
            os.path.join(path, "value_head.pt")
        )

    @classmethod
    def from_pretrained(cls, path: str, device=None, **kwargs):
        base_model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        hidden_size = base_model.config.hidden_size
        model = cls(base_model, hidden_size, device=device)
        vh_path = os.path.join(path, "value_head.pt")
        if os.path.exists(vh_path):
            state = torch.load(vh_path, map_location="cpu")
            model.value_head.load_state_dict(state)
            if device is not None:
                model.value_head = model.value_head.to(device)
        return model


def compute_log_probs(
    model: PolicyModelWithValueHead,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_model = model.base_model
    outputs = base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    logits = outputs.logits
    hidden_states = outputs.hidden_states[-1]

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    seq_len = shift_labels.shape[1]
    mask = torch.zeros_like(token_log_probs)
    if prompt_length - 1 < seq_len:
        mask[:, prompt_length - 1:] = 1.0
    if attention_mask is not None:
        mask = mask * attention_mask[:, 1:]

    total_log_prob = (token_log_probs * mask).sum(dim=-1)

    value_token_idx = prompt_length - 1
    value_hidden = hidden_states[:, value_token_idx, :]
    value = model.value_head(value_hidden)

    gen_logits = shift_logits[:, prompt_length - 1:, :]
    return total_log_prob, value, gen_logits


@dataclass
class Experience:
    prompt: str
    completion: str
    input_ids: torch.Tensor
    prompt_length: int
    reward: float
    value: float
    log_prob: float
    ref_log_prob: float
    action: str
    opponent_prev_actions: Optional[List[str]]
    is_legal: bool
    player_id: int = 0
    advantage: float = 0.0
    returns: float = 0.0


# ============================================================================
# Trainer
# ============================================================================

def _infer_max_new_tokens_for_actions(tokenizer, actions: Tuple[str, str]) -> int:
    lens = [len(tokenizer.encode(a, add_special_tokens=False)) for a in actions]
    return max(lens)


def _model_device(m: nn.Module) -> torch.device:
    return next(m.parameters()).device


class MoralPPOTrainer:
    def __init__(self, config: PPOConfig):
        self.config = config

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.stats_history: List[Dict[str, Any]] = []
        self.reward_history = deque(maxlen=1000)

        self._setup_models_and_tokenizer()
        self._setup_game()
        self._setup_players()

    def _setup_models_and_tokenizer(self):
        cfg = self.config
        print(f"Loading tokenizer: {cfg.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Reference model (frozen)
        print("Loading reference model...")
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
        ref_base = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
        hidden_size = ref_base.config.hidden_size
        self.ref_model = PolicyModelWithValueHead(ref_base, hidden_size, device=_model_device(ref_base))
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def _setup_game(self):
        cfg = self.config
        # Stag Hunt config
        self.env_config = StagHuntConfig(
            num_players=cfg.num_players,
            max_steps=cfg.batch_size,
            threshold=cfg.threshold,
            stag_success_reward=cfg.stag_success_reward,
            stag_fail_reward=cfg.stag_fail_reward,
            hare_reward=cfg.hare_reward,
        )
        # Set generation tokens based on action tokenization
        cfg.max_new_tokens = _infer_max_new_tokens_for_actions(self.tokenizer, self.env_config.action_names)
        print(f"Setting max_new_tokens={cfg.max_new_tokens} based on action tokenization for {self.env_config.action_names}")

    def _setup_players(self):
        cfg = self.config

        # Moral types per player
        if not cfg.player_moral_types:
            if cfg.llm_vs_llm:
                cfg.player_moral_types = [cfg.moral_type] + [cfg.opponent_moral_type] * (cfg.num_players - 1)
            else:
                cfg.player_moral_types = [cfg.moral_type] + ["game"] * (cfg.num_players - 1)
        if len(cfg.player_moral_types) != cfg.num_players:
            raise ValueError(f"player_moral_types must have length num_players={cfg.num_players}")

        # Which players are LLM-controlled?
        self.llm_player_ids = list(range(cfg.num_players)) if cfg.llm_vs_llm else [0]

        # Build player models + optimizers (one model per LLM player)
        self.player_models: Dict[int, PolicyModelWithValueHead] = {}
        self.player_optimizers: Dict[int, AdamW] = {}

        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }

        for pid in self.llm_player_ids:
            print(f"\nLoading LLM policy for Player {pid}...")
            base = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)

            if cfg.use_lora:
                if cfg.use_4bit:
                    base = prepare_model_for_kbit_training(base)
                lora_config = LoraConfig(
                    r=cfg.lora_rank,
                    lora_alpha=cfg.lora_alpha,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                base = get_peft_model(base, lora_config)
                print(f"Player {pid} model trainables:")
                base.print_trainable_parameters()

            hidden_size = base.config.hidden_size
            model = PolicyModelWithValueHead(base, hidden_size, device=_model_device(base))
            self.player_models[pid] = model
            self.player_optimizers[pid] = AdamW(model.parameters(), lr=cfg.learning_rate, eps=1e-5)

        # Fixed opponents for non-LLM players (when not llm_vs_llm)
        self.fixed_opponents: Dict[int, Opponent] = {}
        if not cfg.llm_vs_llm:
            for pid in range(1, cfg.num_players):
                self.fixed_opponents[pid] = make_opponent(cfg.opponent_type)

    def rollout_episode(self) -> Dict[int, List[Experience]]:
        cfg = self.config
        env = StagHuntEnv(self.env_config)
        obs = env.reset(random_initial_state=True)

        for opp in self.fixed_opponents.values():
            opp.reset()

        trajectories: Dict[int, List[Experience]] = {pid: [] for pid in self.llm_player_ids}
        stag, hare = self.env_config.action_names

        for _t in range(self.env_config.max_steps):
            history = obs.get("history", [])
            last_actions = history[-1] if history else None  # joint action vector

            actions: List[str] = [hare] * cfg.num_players
            legal: List[bool] = [True] * cfg.num_players
            step_exps: Dict[int, Experience] = {}

            # 1) Choose actions
            for pid in range(cfg.num_players):
                if pid in self.llm_player_ids:
                    model = self.player_models[pid]
                    model.eval()
                    dev = _model_device(model)

                    opp_prev_actions = None
                    if last_actions is not None:
                        opp_prev_actions = [last_actions[j] for j in range(cfg.num_players) if j != pid]

                    prompt = build_stag_hunt_prompt_with_chat_template(
                        obs, self.tokenizer, self.env_config, player_id=pid
                    )

                    encoded = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(dev)
                    prompt_length = encoded["input_ids"].shape[1]

                    with torch.no_grad():
                        output_ids = model.generate(
                            input_ids=encoded["input_ids"],
                            attention_mask=encoded["attention_mask"],
                            max_new_tokens=cfg.max_new_tokens,
                            do_sample=True,
                            temperature=cfg.temperature,
                            top_p=cfg.top_p,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )

                    full_ids = output_ids
                    attn = torch.ones_like(full_ids)

                    with torch.no_grad():
                        log_prob, value, _ = compute_log_probs(model, full_ids, attn, prompt_length)

                        ref_dev = _model_device(self.ref_model)
                        ref_ids = full_ids.to(ref_dev)
                        ref_attn = torch.ones_like(ref_ids)
                        ref_log_prob, _, _ = compute_log_probs(self.ref_model, ref_ids, ref_attn, prompt_length)

                    completion = self.tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=True).strip()
                    act, is_legal = extract_action_from_completion(completion, self.env_config.action_names)

                    actions[pid] = act
                    legal[pid] = is_legal

                    step_exps[pid] = Experience(
                        prompt=prompt,
                        completion=completion,
                        input_ids=full_ids.detach().cpu(),
                        prompt_length=prompt_length,
                        reward=0.0,
                        value=float(value.item()),
                        log_prob=float(log_prob.item()),
                        ref_log_prob=float(ref_log_prob.item()),
                        action=act,
                        opponent_prev_actions=opp_prev_actions,
                        is_legal=is_legal,
                        player_id=pid,
                    )
                else:
                    # Fixed opponent
                    opp = self.fixed_opponents[pid]
                    actions[pid] = opp.act(obs, player_id=pid, focal_id=0)
                    legal[pid] = True

            # 2) Environment step
            if all(legal):
                next_obs, payoffs, done, _ = env.step(actions)
            else:
                next_obs, payoffs, done, _ = env.illegal_step()

            # 3) Rewards for LLM players
            for pid, exp in step_exps.items():
                moral = cfg.player_moral_types[pid]
                r = compute_moral_reward(
                    moral_type=moral,
                    agent_action=exp.action,
                    opponent_prev_actions=exp.opponent_prev_actions,
                    agent_payoff=payoffs[pid],
                    all_payoffs=payoffs,
                    is_legal=exp.is_legal,
                    xi=cfg.xi,
                    illegal_penalty=cfg.illegal_penalty,
                )
                if cfg.reward_shaping and exp.is_legal:
                    r += cfg.valid_action_bonus
                exp.reward = float(r)
                trajectories[pid].append(exp)

            obs = next_obs
            if done:
                break

        return trajectories

    def collect_batch(self) -> Dict[int, List[List[Experience]]]:
        trajs = self.rollout_episode()
        return {pid: [trajs[pid]] for pid in trajs.keys()}

    def compute_advantages(self, trajectories: List[List[Experience]]) -> List[Experience]:
        cfg = self.config
        gamma, lam = cfg.gamma, cfg.lam

        all_exps: List[Experience] = [e for traj in trajectories for e in traj]
        if not all_exps:
            return []

        task_rewards = np.array([e.reward for e in all_exps], dtype=np.float32)

        if cfg.ref_kl_coef != 0.0:
            kl_terms = np.array([e.log_prob - e.ref_log_prob for e in all_exps], dtype=np.float32)
            rewards = task_rewards - cfg.ref_kl_coef * kl_terms
        else:
            rewards = task_rewards.copy()

        rewards = rewards * cfg.reward_scale

        if cfg.normalize_rewards and len(rewards) > 1:
            std = rewards.std()
            rewards = (rewards - rewards.mean()) / (std + 1e-8) if std > 1e-6 else (rewards - rewards.mean())

        for e, r in zip(all_exps, rewards):
            e.reward = float(r)

        advantages: List[float] = []
        returns: List[float] = []

        for traj in trajectories:
            T = len(traj)
            if T == 0:
                continue

            next_value = 0.0
            next_adv = 0.0
            traj_adv = [0.0] * T
            traj_ret = [0.0] * T

            for t in reversed(range(T)):
                e = traj[t]
                delta = e.reward + gamma * next_value - e.value
                adv = delta + gamma * lam * next_adv
                ret = adv + e.value

                traj_adv[t] = adv
                traj_ret[t] = ret

                next_value = e.value
                next_adv = adv

            advantages.extend(traj_adv)
            returns.extend(traj_ret)

        advantages = np.array(advantages, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)

        if cfg.normalize_advantages and len(advantages) > 1:
            std = advantages.std()
            advantages = (advantages - advantages.mean()) / (std + 1e-8) if std > 1e-6 else (advantages - advantages.mean())

        for e, a, R in zip(all_exps, advantages, returns):
            e.advantage = float(a)
            e.returns = float(R)

        return all_exps

    def ppo_update(
        self,
        experiences: List[Experience],
        model: PolicyModelWithValueHead,
        optimizer: AdamW
    ) -> Dict[str, float]:
        cfg = self.config
        model.train()
        device = _model_device(model)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        num_updates = 0

        if not experiences:
            return {"policy_loss": 0.0, "value_loss": 0.0, "kl": 0.0, "entropy": 0.0}

        for _epoch in range(cfg.ppo_epochs):
            indices = np.random.permutation(len(experiences))

            for idx in indices:
                exp = experiences[idx]
                if abs(exp.advantage) < 1e-8:
                    continue

                input_ids = exp.input_ids.to(device)
                attention_mask = torch.ones_like(input_ids)

                log_prob, value, gen_logits = compute_log_probs(
                    model, input_ids, attention_mask, exp.prompt_length
                )

                gen_probs = F.softmax(gen_logits, dim=-1)
                gen_log_probs = F.log_softmax(gen_logits, dim=-1)
                entropy = -(gen_probs * gen_log_probs).sum(dim=-1).mean()

                old_log_prob = torch.tensor(exp.log_prob, device=device, dtype=log_prob.dtype)
                old_value = torch.tensor(exp.value, device=device, dtype=value.dtype)
                advantage = torch.tensor(exp.advantage, device=device, dtype=log_prob.dtype)
                returns = torch.tensor(exp.returns, device=device, dtype=value.dtype)

                log_ratio = log_prob - old_log_prob
                ratio = torch.exp(log_ratio)

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * advantage
                policy_loss = -torch.min(surr1, surr2)

                value_clipped = old_value + torch.clamp(
                    value - old_value, -cfg.clip_ratio, cfg.clip_ratio
                )
                value_loss1 = (value - returns) ** 2
                value_loss2 = (value_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(value_loss1, value_loss2)

                approx_kl = 0.5 * (old_log_prob - log_prob) ** 2

                loss = (
                    policy_loss
                    + cfg.vf_coef * value_loss
                    + cfg.kl_coef * approx_kl
                    - cfg.entropy_coef * entropy
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_kl += float(approx_kl.item())
                total_entropy += float(entropy.item())
                num_updates += 1

            if num_updates > 0 and cfg.target_kl:
                avg_kl = total_kl / num_updates
                if avg_kl > cfg.target_kl * 1.5:
                    break

        n = max(num_updates, 1)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "kl": total_kl / n,
            "entropy": total_entropy / n,
        }

    def train(self):
        cfg = self.config

        # Output path
        tag = f"stag_hunt_n{cfg.num_players}_thr{cfg.threshold}"
        mode = "llm_vs_llm" if cfg.llm_vs_llm else f"vs_{cfg.opponent_type}"
        output_path = os.path.join(cfg.output_dir, f"{tag}_{mode}_{cfg.moral_type}")
        os.makedirs(output_path, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Training Stag Hunt | N={cfg.num_players} threshold={cfg.threshold} steps/ep={cfg.batch_size}")
        print(f"Mode: {'LLM vs LLM (all players)' if cfg.llm_vs_llm else 'LLM vs Fixed (players 1..N-1)'}")
        print(f"LLM players: {self.llm_player_ids}")
        print(f"Player moral types: {cfg.player_moral_types}")
        print(f"Episodes: {cfg.num_episodes}")
        print(f"{'='*70}\n")

        for episode in range(1, cfg.num_episodes + 1):
            player_trajectories = self.collect_batch()
            stats: Dict[str, Any] = {"episode": episode}

            # Update each LLM player
            for pid in self.llm_player_ids:
                trajs = player_trajectories[pid]
                flat = [e for tr in trajs for e in tr]
                raw_rewards = [e.reward for e in flat]

                exps = self.compute_advantages(trajs)
                upd = self.ppo_update(exps, self.player_models[pid], self.player_optimizers[pid])

                coop = sum(1 for e in exps if e.action == "stag") / max(len(exps), 1)
                ill = sum(1 for e in exps if not e.is_legal) / max(len(exps), 1)

                stats.update({
                    f"p{pid}_mean_reward": float(np.mean(raw_rewards)) if raw_rewards else 0.0,
                    f"p{pid}_std_reward": float(np.std(raw_rewards)) if raw_rewards else 0.0,
                    f"p{pid}_stag_rate": coop,
                    f"p{pid}_illegal_rate": ill,
                    f"p{pid}_policy_loss": upd["policy_loss"],
                    f"p{pid}_value_loss": upd["value_loss"],
                    f"p{pid}_kl": upd["kl"],
                    f"p{pid}_entropy": upd["entropy"],
                })

            self.stats_history.append(stats)

            if episode % cfg.log_every == 0:
                parts = [f"Ep {episode:4d}/{cfg.num_episodes}"]
                for pid in self.llm_player_ids:
                    parts.append(f"P{pid} R:{stats[f'p{pid}_mean_reward']:+.2f} Stag:{stats[f'p{pid}_stag_rate']:.0%}")
                parts.append(f"KL(P0):{stats.get('p0_kl', 0.0):.4f}")
                print(" | ".join(parts))

            if episode % cfg.save_every == 0:
                ckpt_dir = os.path.join(output_path, f"checkpoint_{episode}")
                os.makedirs(ckpt_dir, exist_ok=True)
                self.tokenizer.save_pretrained(ckpt_dir)
                for pid in self.llm_player_ids:
                    self.player_models[pid].save_pretrained(os.path.join(ckpt_dir, f"player_{pid}"))

        # Final save
        final_dir = os.path.join(output_path, "final")
        os.makedirs(final_dir, exist_ok=True)
        self.tokenizer.save_pretrained(final_dir)
        for pid in self.llm_player_ids:
            self.player_models[pid].save_pretrained(os.path.join(final_dir, f"player_{pid}"))

        stats_path = os.path.join(output_path, "training_stats.json")
        with open(stats_path, "w") as f:
            json.dump(self.stats_history, f, indent=2)

        config_path = os.path.join(output_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(vars(cfg), f, indent=2)

        print("\nTraining complete!")
        print(f"Saved to: {output_path}")
        print(f"Stats: {stats_path}")
        return self.stats_history


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PPO Training for Moral Alignment (N-player Stag Hunt)")

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--use_4bit", action="store_true")

    # Game
    parser.add_argument("--num_players", type=int, default=2)
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--stag_success_reward", type=float, default=4.0)
    parser.add_argument("--stag_fail_reward", type=float, default=0.0)
    parser.add_argument("--hare_reward", type=float, default=2.0)

    # Moral types
    parser.add_argument("--moral_type", type=str, default="utilitarian",
                        choices=["game", "deontological", "utilitarian", "game+deontological"])
    parser.add_argument("--opponent_moral_type", type=str, default="game",
                        choices=["game", "deontological", "utilitarian", "game+deontological"])
    parser.add_argument("--player_moral_types", type=str, default="",
                        help="Comma-separated moral types per player (length num_players). "
                             "Example: utilitarian,game,game")

    # Opponents (fixed-mode)
    parser.add_argument("--opponent_type", type=str, default="copy_focal",
                        choices=["always_stag", "always_hare", "random", "copy_focal", "llm"],
                        help="In fixed-mode, used for players 1..N-1. If set to 'llm', enables llm_vs_llm.")

    # Multi-agent mode
    parser.add_argument("--llm_vs_llm", action="store_true",
                        help="If set, all players are LLM policies trained simultaneously.")

    # Training
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=100)

    args = parser.parse_args()

    # If opponent_type is "llm", enable llm_vs_llm
    if args.opponent_type == "llm":
        args.llm_vs_llm = True

    # Parse player moral types
    player_morals: List[str] = []
    if args.player_moral_types.strip():
        player_morals = [s.strip() for s in args.player_moral_types.split(",") if s.strip()]

    cfg = PPOConfig(
        model_name=args.model_name,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_4bit=args.use_4bit,

        num_players=args.num_players,
        threshold=args.threshold,
        stag_success_reward=args.stag_success_reward,
        stag_fail_reward=args.stag_fail_reward,
        hare_reward=args.hare_reward,

        moral_type=args.moral_type,
        opponent_moral_type=args.opponent_moral_type,
        player_moral_types=player_morals,

        opponent_type=args.opponent_type if not args.llm_vs_llm else "llm",
        llm_vs_llm=args.llm_vs_llm,

        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        learning_rate=args.learning_rate,

        output_dir=args.output_dir,
        seed=args.seed,
        log_every=args.log_every,
        save_every=args.save_every,
    )

    # Helpful default: if user didn't set threshold, make it "all-to-stag" for N>2?
    # (commented out to avoid surprising behavior)
    # if cfg.threshold == 2 and cfg.num_players > 2:
    #     cfg.threshold = cfg.num_players

    trainer = MoralPPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()