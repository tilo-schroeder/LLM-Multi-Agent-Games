"""Self-play PPO finetuning on a 2-player repeated Stag Hunt game,
where Player 1 is an LLM local-decision policy trained against a fixed
(opponent_type) strategy such as Tit-for-Tat, Always Cooperate, etc.

This script implements PPO (Proximal Policy Optimization) for LLM training
with custom environment rewards.

FIXES from v1:
- Added proper value function initialization and normalization
- Added reward normalization/scaling
- Added early stopping on KL divergence
- Added proper entropy calculation
- Fixed advantage computation to handle per-token vs per-sequence
- Added gradient accumulation for stability
- Added repetition penalty detection
"""

import os
import json
import csv
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

from envs.stag_hunt import StagHuntEnv, StagHuntConfig
from utils.logging_utils import get_logger

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="PPO finetuning on repeated Stag Hunt (LLM vs fixed opponent)."
    )
    parser.add_argument("--opponent_type", type=str, choices=["tft", "always_cooperate", "always_defect", "random"], default="tft")
    parser.add_argument("--model_p1", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--moral_type", type=str, choices=["game", "deontological", "utilitarian", "game+deontological"], default="utilitarian")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate (reduced for stability)")
    parser.add_argument("--num_updates", type=int, default=30)
    parser.add_argument("--episodes_per_batch", type=int, default=8)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Gradient clipping (reduced)")
    parser.add_argument("--kl_coef", type=float, default=0.2, help="KL penalty coefficient (increased)")
    parser.add_argument("--target_kl", type=float, default=0.01, help="Target KL for early stopping")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--cliprange_value", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="./runs/ppo_tft_utilitarian")
    parser.add_argument("--reward_scale", type=float, default=0.1, help="Scale rewards to prevent large gradients")
    parser.add_argument("--normalize_rewards", action="store_true", default=True, help="Normalize rewards per batch")
    parser.add_argument("--normalize_advantages", action="store_true", default=True, help="Normalize advantages")
    parser.add_argument("--use_adaptive_kl", action="store_true", default=True, help="Use adaptive KL penalty")
    return parser.parse_args()


class ValueHead(nn.Module):
    """Value head with proper initialization for stable training."""
    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) # You forgot to use the dropout arg
        self.dense = nn.Linear(hidden_size, hidden_size // 2)
        self.dense2 = nn.Linear(hidden_size // 2, 1)
        
        # Initialize to output small values
        nn.init.normal_(self.dense.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.dense.bias)
        nn.init.normal_(self.dense2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.dense2.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states shape: [batch, seq_len, hidden]
        # Use the last token's embedding
        x = hidden_states[:, -1, :] 
        x = self.dropout(x)
        x = F.tanh(self.dense(x))
        return self.dense2(x).squeeze(-1)


class CausalLMWithValueHead(nn.Module):
    def __init__(self, base_model: AutoModelForCausalLM):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.value_head = ValueHead(base_model.config.hidden_size)
        
        # Freeze embedding layers for stability (optional but helps)
        # for param in self.base_model.model.embed_tokens.parameters():
        #     param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True, 
            **kwargs
        )
        hidden_states = outputs.hidden_states[-1]
        value = self.value_head(hidden_states)
        return outputs, value

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.base_model.save_pretrained(path)
        torch.save(self.value_head.state_dict(), os.path.join(path, "value_head.pt"))

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        base_model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        model = cls(base_model)
        value_head_path = os.path.join(path, "value_head.pt")
        if os.path.exists(value_head_path):
            model.value_head.load_state_dict(torch.load(value_head_path, weights_only=True))
        return model


# =======================================
# Reward normalization helper
# =======================================

class RunningMeanStd:
    """Running mean and std for reward normalization."""
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# =======================================
# Fixed opponents
# =======================================

class TitForTatOpponent:
    def reset(self): pass
    def act(self, env_obs: Dict[str, Any]) -> str:
        history = env_obs["history"]
        if not history:
            return StagHuntEnv.ACTION_STAG
        return history[-1][0]


class AlwaysCooperateOpponent:
    def reset(self): pass
    def act(self, env_obs: Dict[str, Any]) -> str:
        return StagHuntEnv.ACTION_STAG


class AlwaysDefectOpponent:
    def reset(self): pass
    def act(self, env_obs: Dict[str, Any]) -> str:
        return StagHuntEnv.ACTION_HARE


class RandomOpponent:
    def reset(self): pass
    def act(self, env_obs: Dict[str, Any]) -> str:
        return np.random.choice([StagHuntEnv.ACTION_STAG, StagHuntEnv.ACTION_HARE])


def make_fixed_opponent(opponent_type: str):
    if opponent_type == "tft": return TitForTatOpponent()
    if opponent_type == "always_cooperate": return AlwaysCooperateOpponent()
    if opponent_type == "always_defect": return AlwaysDefectOpponent()
    if opponent_type == "random": return RandomOpponent()
    raise ValueError(f"Unknown opponent_type: {opponent_type}")


def compute_moral_reward_stag_hunt(moral_type, agent_action, opp_prev_action, r_agent, r_opp, is_legal, xi=3.0, illegal_penalty=-6.0):
    if not is_legal:
        return float(illegal_penalty)
    COOP, DEFECT = StagHuntEnv.ACTION_STAG, StagHuntEnv.ACTION_HARE
    if moral_type == "game":
        return float(r_agent)
    if moral_type == "deontological":
        return -float(xi) if (opp_prev_action == COOP and agent_action == DEFECT) else 0.0
    if moral_type == "utilitarian":
        return float(r_agent + r_opp)
    if moral_type == "game+deontological":
        base = float(r_agent)
        if opp_prev_action == COOP and agent_action == DEFECT:
            base -= float(xi)
        return base
    raise ValueError(f"Unknown moral_type: {moral_type}")


def format_history_for_player(history):
    if not history:
        return "You have not played with this opponent before.\n"
    last_a1, last_a2 = history[-1]
    return f"Last time, Player 1 chose {last_a1}, and Player 2 chose {last_a2}.\n"


def build_decision_prompt(player_id, env_obs, config, tokenizer):
    round_idx = env_obs["round"] + 1
    history_text = format_history_for_player(env_obs["history"])
    system_msg = f"You are playing as Player {player_id} in a repeated 2-player Stag Hunt game."
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
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_action_from_completion(text):
    """Extract action with better error handling."""
    text = text.strip()
    
    # Check for repetition (sign of degenerate output)
    if len(text) > 50:
        # Check if there's heavy repetition
        words = text.split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # Too much repetition
                return StagHuntEnv.ACTION_HARE, False
    
    if "\n" in text:
        # Try to parse just the first line
        first_line = text.split("\n")[0].strip()
        if first_line.upper().startswith("ACTION:"):
            tail = first_line.split(":", 1)[1].strip().upper()
            if tail == "STAG": return StagHuntEnv.ACTION_STAG, True
            if tail == "HARE": return StagHuntEnv.ACTION_HARE, True
        return StagHuntEnv.ACTION_HARE, False
    
    if not text.upper().startswith("ACTION:"):
        return StagHuntEnv.ACTION_HARE, False
    
    tail = text.split(":", 1)[1].strip().upper()
    if tail == "STAG": return StagHuntEnv.ACTION_STAG, True
    if tail == "HARE": return StagHuntEnv.ACTION_HARE, True
    return StagHuntEnv.ACTION_HARE, False


@dataclass
class PPOStepData:
    episode_id: int
    round_idx: int
    prompt: str
    completion: str
    input_ids: torch.Tensor
    prompt_length: int
    reward: float
    value: float
    log_prob: float
    ref_log_prob: float  # Added: log prob under reference policy
    action: str
    opp_prev_action: str
    is_legal: bool
    advantage: float = 0.0
    return_value: float = 0.0


def log_completions(step_data_list, log_file, meta):
    if not step_data_list: return
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        for s in step_data_list:
            row = {"setup": meta.get("setup"), "update": meta.get("update"), "moral_type": meta.get("moral_type"),
                   "opponent_type": meta.get("opponent_type"), "episode_id": s.episode_id, "round_idx": s.round_idx,
                   "completion": s.completion, "action": s.action, "opp_prev_action": s.opp_prev_action,
                   "reward": s.reward, "value": s.value, "log_prob": s.log_prob, "is_legal": s.is_legal}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compute_log_probs_and_entropy(model, input_ids, attention_mask, prompt_length, device):
    """
    Compute log probabilities, value, AND entropy for a sequence.
    
    Returns:
        log_prob: SUM of log probabilities of completion tokens (not mean!)
        value: value estimate
        entropy: entropy of the policy over completion tokens
    """
    outputs, value = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    
    # Compute log probs and entropy
    log_probs_all = F.log_softmax(shift_logits, dim=-1)
    probs_all = F.softmax(shift_logits, dim=-1)
    
    # Token-level log probs
    token_log_probs = log_probs_all.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Token-level entropy: -sum(p * log(p))
    token_entropy = -(probs_all * log_probs_all).sum(dim=-1)
    
    # Mask: only completion tokens
    seq_len = shift_labels.shape[1]
    completion_mask = torch.zeros(1, seq_len, device=device)
    if prompt_length - 1 < seq_len:
        completion_mask[:, prompt_length - 1:] = 1.0
    
    mask = attention_mask[:, 1:] * completion_mask
    
    # SUM log prob over completion (for proper PPO ratio calculation)
    masked_log_probs = token_log_probs * mask
    sum_log_prob = masked_log_probs.sum()
    
    # MEAN entropy over completion tokens
    masked_entropy = token_entropy * mask
    num_tokens = mask.sum().clamp(min=1)
    mean_entropy = masked_entropy.sum() / num_tokens
    
    return sum_log_prob, value, mean_entropy


def compute_gae(rewards, values, gamma, lam):
    """Compute GAE with proper handling."""
    n = len(rewards)
    if n == 0:
        return [], []
    
    advantages = []
    gae = 0.0
    values_with_terminal = values + [0.0]
    
    for t in reversed(range(n)):
        delta = rewards[t] + gamma * values_with_terminal[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def rollout_episode_for_ppo(
    model, ref_model, tokenizer, config, device, episode_id, moral_type, opponent,
    temperature=0.7, top_p=0.9, max_new_tokens=32, xi=3.0, illegal_penalty=-6.0
):
    """Roll out one episode with both policy and reference log probs."""
    env = StagHuntEnv(config)
    obs = env.reset(random_initial_state=True)
    opponent.reset()
    step_data_list, step_rewards_p1, step_rewards_p2 = [], [], []
    p2_stag_count, p2_total_actions = 0, 0
    model.eval()
    ref_model.eval()

    for t in range(config.num_rounds):
        round_idx = t + 1
        history = obs["history"]
        opp_prev_action, prev_a1 = (history[-1][1], history[-1][0]) if history else (None, None)
        
        prompt = build_decision_prompt(1, obs, config, tokenizer)
        prompt_encoding = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = prompt_encoding["input_ids"].shape[1]

        with torch.no_grad():
            # Generate with repetition penalty to avoid degenerate outputs
            output_ids = model.generate(
                input_ids=prompt_encoding["input_ids"],
                attention_mask=prompt_encoding["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Add repetition penalty
            )
        
        full_input_ids = output_ids
        completion_text = tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=True).strip()
        attention_mask = torch.ones_like(full_input_ids)
        
        with torch.no_grad():
            log_prob, value, entropy = compute_log_probs_and_entropy(
                model, full_input_ids, attention_mask, prompt_length, device
            )
            ref_log_prob, _, _ = compute_log_probs_and_entropy(
                ref_model, full_input_ids, attention_mask, prompt_length, device
            )
        
        action_p1, is_legal = extract_action_from_completion(completion_text)
        
        step_data = PPOStepData(
            episode_id=episode_id, round_idx=round_idx, prompt=prompt, completion=completion_text,
            input_ids=full_input_ids.cpu(), prompt_length=prompt_length, reward=0.0, 
            value=value.item(), log_prob=log_prob.item(), ref_log_prob=ref_log_prob.item(),
            action=action_p1, opp_prev_action=opp_prev_action or "", is_legal=is_legal
        )

        if not is_legal:
            moral_r1 = compute_moral_reward_stag_hunt(
                moral_type, StagHuntEnv.ACTION_HARE, opp_prev_action, 0.0, 0.0, False, xi, illegal_penalty
            )
            step_rewards_p1.append(moral_r1)
            step_data.reward = moral_r1
            step_data_list.append(step_data)
            continue

        action_p2 = opponent.act(obs)
        p2_total_actions += 1
        if action_p2 == StagHuntEnv.ACTION_STAG: 
            p2_stag_count += 1
        obs, (r1, r2), done, info = env.step(action_p1, action_p2)

        moral_r1 = compute_moral_reward_stag_hunt(moral_type, action_p1, opp_prev_action, r1, r2, True, xi, illegal_penalty)
        moral_r2 = compute_moral_reward_stag_hunt(moral_type, action_p2, prev_a1, r2, r1, True, xi, illegal_penalty)
        step_rewards_p1.append(moral_r1)
        step_rewards_p2.append(moral_r2)
        step_data.reward = moral_r1
        step_data_list.append(step_data)

    return step_data_list, float(sum(step_rewards_p1)), step_rewards_p2, p2_stag_count, p2_total_actions


def collect_batch_for_ppo(
    model, ref_model, tokenizer, config, device, num_episodes, moral_type,
    temperature=0.7, top_p=0.9, logger=None, opponent_type="tft"
):
    if logger is None: 
        logger = get_logger()
    all_step_data, opp_rewards_batch = [], []
    opponent = make_fixed_opponent(opponent_type)
    total_p2_stags, total_p2_actions = 0, 0

    for ep_id in range(num_episodes):
        logger.info(f"  Rolling out episode {ep_id} (type={moral_type}) vs '{opponent_type}'...")
        step_data_list, total_moral_p1, opp_rewards_ep, p2_stags, p2_actions = rollout_episode_for_ppo(
            model, ref_model, tokenizer, config, device, ep_id, moral_type, opponent, temperature, top_p
        )
        all_step_data.extend(step_data_list)
        opp_rewards_batch.extend(opp_rewards_ep)
        total_p2_stags += p2_stags
        total_p2_actions += p2_actions
        
        if ep_id < 2:
            logger.info(f"    Episode {ep_id} total moral return (P1)={total_moral_p1:.3f}")
            # Log illegal rate
            illegal_count = sum(1 for s in step_data_list if not s.is_legal)
            logger.info(f"    Illegal actions: {illegal_count}/{len(step_data_list)}")

    opp_stag_rate = (total_p2_stags / total_p2_actions) if total_p2_actions > 0 else 0.0
    return all_step_data, opp_stag_rate, opp_rewards_batch


def ppo_train_step(
    model, ref_model, optimizer, step_data_list, device, 
    ppo_epochs, mini_batch_size, cliprange, cliprange_value, 
    vf_coef, entropy_coef, kl_coef, gamma, lam, max_grad_norm,
    target_kl, reward_scale, normalize_rewards, normalize_advantages,
    use_adaptive_kl, logger
):
    """PPO training step with improved stability."""
    
    if len(step_data_list) == 0:
        return {"policy_loss": 0, "value_loss": 0, "kl": 0, "entropy": 0, "approx_kl": 0}
    
    # Scale rewards
    for s in step_data_list:
        s.reward = s.reward * reward_scale
    
    # Optionally normalize rewards across batch
    if normalize_rewards:
        rewards = np.array([s.reward for s in step_data_list])
        reward_mean = rewards.mean()
        reward_std = rewards.std() + 1e-8
        for s in step_data_list:
            s.reward = (s.reward - reward_mean) / reward_std
    
    # Compute GAE per episode
    episodes = {}
    for s in step_data_list:
        if s.episode_id not in episodes: 
            episodes[s.episode_id] = []
        episodes[s.episode_id].append(s)

    all_advantages, all_returns = [], []
    for ep_id in sorted(episodes.keys()):
        ep_data = episodes[ep_id]
        rewards = [s.reward for s in ep_data]
        values = [s.value for s in ep_data]
        advantages, returns = compute_gae(rewards, values, gamma, lam)
        all_advantages.extend(advantages)
        all_returns.extend(returns)

    # Normalize advantages
    if normalize_advantages and len(all_advantages) > 1:
        advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32)
        adv_mean = advantages_tensor.mean()
        adv_std = advantages_tensor.std().clamp(min=1e-8)
        normalized_advantages = ((advantages_tensor - adv_mean) / adv_std).tolist()
    else:
        normalized_advantages = all_advantages
    
    for i, s in enumerate(step_data_list):
        s.advantage = normalized_advantages[i]
        s.return_value = all_returns[i]

    # Training loop with early stopping
    model.train()
    total_policy_loss, total_value_loss, total_kl, total_entropy = 0.0, 0.0, 0.0, 0.0
    total_approx_kl = 0.0
    num_batches = 0
    early_stop = False

    for epoch in range(ppo_epochs):
        if early_stop:
            break
            
        indices = list(range(len(step_data_list)))
        np.random.shuffle(indices)
        
        for batch_start in range(0, len(indices), mini_batch_size):
            batch_indices = indices[batch_start:batch_start + mini_batch_size]
            batch_data = [step_data_list[i] for i in batch_indices]
            
            batch_policy_loss, batch_value_loss, batch_kl, batch_entropy = 0.0, 0.0, 0.0, 0.0
            batch_approx_kl = 0.0
            optimizer.zero_grad()

            for s in batch_data:
                input_ids = s.input_ids.to(device)
                attention_mask = torch.ones_like(input_ids)
                
                # Current policy
                curr_log_prob, curr_value, curr_entropy = compute_log_probs_and_entropy(
                    model, input_ids, attention_mask, s.prompt_length, device
                )
                
                # Compute ratio using OLD log prob (from rollout)
                old_log_prob = torch.tensor(s.log_prob, device=device)
                ref_log_prob = torch.tensor(s.ref_log_prob, device=device)
                advantage = torch.tensor(s.advantage, device=device, dtype=torch.float32)
                
                # Policy ratio
                log_ratio = curr_log_prob - old_log_prob
                ratio = torch.exp(log_ratio)
                
                # Clipped surrogate objective
                clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
                policy_loss = -torch.min(advantage * ratio, advantage * clipped_ratio)
                
                # Value loss with clipping
                return_value = torch.tensor(s.return_value, device=device, dtype=torch.float32)
                old_value = torch.tensor(s.value, device=device, dtype=torch.float32)
                
                value_pred_clipped = old_value + torch.clamp(
                    curr_value - old_value, -cliprange_value, cliprange_value
                )
                value_loss1 = (curr_value - return_value) ** 2
                value_loss2 = (value_pred_clipped - return_value) ** 2
                value_loss = 0.5 * torch.max(value_loss1, value_loss2)
                
                # KL divergence from reference (for penalty)
                kl_div = ref_log_prob - curr_log_prob  # KL(ref || curr) approximation
                
                # Approximate KL for early stopping (KL from old policy)
                approx_kl = ((ratio - 1) - log_ratio).mean()
                
                # Total loss
                loss = policy_loss + vf_coef * value_loss + kl_coef * kl_div - entropy_coef * curr_entropy
                loss.backward()

                batch_policy_loss += policy_loss.item()
                batch_value_loss += value_loss.item()
                batch_kl += kl_div.item()
                batch_entropy += curr_entropy.item()
                batch_approx_kl += approx_kl.item()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            n = len(batch_data)
            total_policy_loss += batch_policy_loss / n
            total_value_loss += batch_value_loss / n
            total_kl += batch_kl / n
            total_entropy += batch_entropy / n
            total_approx_kl += batch_approx_kl / n
            num_batches += 1
            
            # Early stopping on KL divergence
            avg_approx_kl = total_approx_kl / num_batches
            if target_kl is not None and avg_approx_kl > 1.5 * target_kl:
                logger.info(f"Early stopping at epoch {epoch} due to KL divergence: {avg_approx_kl:.4f}")
                early_stop = True
                break

    n_batches = max(num_batches, 1)
    return {
        "policy_loss": total_policy_loss / n_batches,
        "value_loss": total_value_loss / n_batches,
        "kl": total_kl / n_batches,
        "entropy": total_entropy / n_batches,
        "approx_kl": total_approx_kl / n_batches,
    }


def train_ppo_stag_hunt(args):
    setup, opponent_type = "fixed_opponent", args.opponent_type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir, log_dir = args.output_dir, os.path.join(args.output_dir, "logs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(log_dir)

    completion_log_path = os.path.join(log_dir, "completions.jsonl")
    if os.path.exists(completion_log_path): 
        os.remove(completion_log_path)

    stag_cfg = StagHuntConfig(
        num_players=2, num_rounds=args.num_rounds, 
        R_stag_stag=4.0, R_hare_hare=1.0, R_stag_hare=0.0, R_hare_stag=3.0
    )
    
    logger.info(f"Starting PPO Stag Hunt training (LLM vs fixed opponent='{opponent_type}')")
    logger.info(f"Base model P1: {args.model_p1}, StagHuntConfig: {stag_cfg}")
    logger.info(f"moral_type={args.moral_type}, num_updates={args.num_updates}, episodes_per_batch={args.episodes_per_batch}")
    logger.info(f"lr={args.lr}, reward_scale={args.reward_scale}, target_kl={args.target_kl}")
    logger.info(f"PPO params: ppo_epochs={args.ppo_epochs}, mini_batch_size={args.mini_batch_size}, "
                f"cliprange={args.cliprange}, kl_coef={args.kl_coef}, vf_coef={args.vf_coef}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_p1, trust_remote_code=True)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load models
    logger.info("Loading policy model...")
    base_model = AutoModelForCausalLM.from_pretrained(args.model_p1, trust_remote_code=True)
    model = CausalLMWithValueHead(base_model).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    logger.info("Loading reference model...")
    ref_base_model = AutoModelForCausalLM.from_pretrained(args.model_p1, trust_remote_code=True)
    ref_model = CausalLMWithValueHead(ref_base_model).to(device)
    ref_model.eval()
    for param in ref_model.parameters(): 
        param.requires_grad = False

    # Optimizer with lower learning rate
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=max(1, int(0.1 * args.num_updates)), 
        num_training_steps=args.num_updates
    )

    # Stats
    stats = {
        "update": [], "mean_reward": [], "std_reward": [], "mean_reward_p1": [], "mean_reward_p2": [],
        "policy_loss": [], "value_loss": [], "kl": [], "entropy": [], "approx_kl": [],
        "p1_stag_rate": [], "p2_stag_rate": [], "global_stag_rate": [], "illegal_rate": []
    }
    action_categories = ["STAG|STAG", "STAG|HARE", "HARE|STAG", "HARE|HARE", "illegal|STAG", "illegal|HARE"]
    action_stats = {"episode": []}
    for cat in action_categories: 
        action_stats[cat] = []

    # Adaptive KL coefficient
    current_kl_coef = args.kl_coef

    def stag_rate(step_data_list):
        if not step_data_list: return 0.0
        return sum(1 for s in step_data_list if s.action == StagHuntEnv.ACTION_STAG and s.is_legal) / len(step_data_list)

    def illegal_rate(step_data_list):
        if not step_data_list: return 0.0
        return sum(1 for s in step_data_list if not s.is_legal) / len(step_data_list)

    for update in range(1, args.num_updates + 1):
        logger.info(f"=== Running update {update}/{args.num_updates} (LLM vs '{opponent_type}') ===")
        
        # Collect data
        step_data_list, opp_stag_rate, opp_rewards = collect_batch_for_ppo(
            model, ref_model, tokenizer, stag_cfg, device, 
            args.episodes_per_batch, args.moral_type, 
            args.temperature, args.top_p, logger, opponent_type
        )
        
        log_completions(step_data_list, completion_log_path, {
            "setup": setup, "update": update, "moral_type": args.moral_type, "opponent_type": opponent_type
        })

        # PPO update
        train_stats = ppo_train_step(
            model, ref_model, optimizer, step_data_list, device,
            args.ppo_epochs, args.mini_batch_size, args.cliprange, args.cliprange_value,
            args.vf_coef, args.entropy_coef, current_kl_coef, args.gamma, args.lam,
            args.max_grad_norm, args.target_kl, args.reward_scale,
            args.normalize_rewards, args.normalize_advantages, args.use_adaptive_kl, logger
        )
        scheduler.step()

        # Adaptive KL coefficient
        if args.use_adaptive_kl and args.target_kl is not None:
            if train_stats["approx_kl"] > args.target_kl * 2:
                current_kl_coef *= 1.5
                logger.info(f"Increased KL coef to {current_kl_coef:.4f}")
            elif train_stats["approx_kl"] < args.target_kl * 0.5:
                current_kl_coef *= 0.5
                current_kl_coef = max(current_kl_coef, 0.01)
                logger.info(f"Decreased KL coef to {current_kl_coef:.4f}")

        # Stats (using unscaled rewards for logging)
        raw_rewards_p1 = [s.reward / args.reward_scale for s in step_data_list] if args.reward_scale != 0 else [s.reward for s in step_data_list]
        rewards_p1 = torch.tensor(raw_rewards_p1, dtype=torch.float32)
        mean_r_p1 = rewards_p1.mean()
        
        rewards_p2 = torch.tensor(opp_rewards, dtype=torch.float32) if opp_rewards else torch.tensor([0.0])
        mean_r_p2 = rewards_p2.mean()
        all_rewards = torch.cat([rewards_p1, rewards_p2], dim=0) if opp_rewards else rewards_p1
        mean_r_global = all_rewards.mean()
        std_r_global = all_rewards.std(unbiased=False).clamp(min=1e-6)

        # Action stats per episode
        for ep_local in range(args.episodes_per_batch):
            global_ep_idx = (update - 1) * args.episodes_per_batch + ep_local + 1
            counts = {cat: 0 for cat in action_categories}
            for s in step_data_list:
                if s.episode_id != ep_local or s.opp_prev_action not in (StagHuntEnv.ACTION_STAG, StagHuntEnv.ACTION_HARE): 
                    continue
                key = f"{'illegal' if not s.is_legal else s.action}|{s.opp_prev_action}"
                if key in counts: 
                    counts[key] += 1
            action_stats["episode"].append(global_ep_idx)
            for cat in action_categories: 
                action_stats[cat].append(counts[cat])

        p1_stag = stag_rate(step_data_list)
        p2_stag = opp_stag_rate
        global_stag = 0.5 * (p1_stag + p2_stag)
        ill_rate = illegal_rate(step_data_list)

        stats["update"].append(update)
        stats["mean_reward"].append(mean_r_global.item())
        stats["std_reward"].append(std_r_global.item())
        stats["mean_reward_p1"].append(mean_r_p1.item())
        stats["mean_reward_p2"].append(mean_r_p2.item())
        stats["policy_loss"].append(train_stats["policy_loss"])
        stats["value_loss"].append(train_stats["value_loss"])
        stats["kl"].append(train_stats["kl"])
        stats["entropy"].append(train_stats["entropy"])
        stats["approx_kl"].append(train_stats["approx_kl"])
        stats["p1_stag_rate"].append(p1_stag)
        stats["p2_stag_rate"].append(p2_stag)
        stats["global_stag_rate"].append(global_stag)
        stats["illegal_rate"].append(ill_rate)

        logger.info(
            f"[Update {update}/{args.num_updates}] "
            f"Policy Loss: {train_stats['policy_loss']:.4f} | "
            f"Value Loss: {train_stats['value_loss']:.4f} | "
            f"Approx KL: {train_stats['approx_kl']:.4f} | "
            f"Entropy: {train_stats['entropy']:.4f} | "
            f"Mean reward: {mean_r_global.item():.3f} (P1={mean_r_p1.item():.3f}) | "
            f"P1 STAG: {p1_stag:.3f} | "
            f"Illegal rate: {ill_rate:.3f}"
        )

    # Save model
    model.save_pretrained(os.path.join(output_dir, "agent_fixed_opp_p1"))
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Saved fine-tuned model to {output_dir}")

    # Save stats
    with open(os.path.join(log_dir, "training_stats.json"), "w") as f: 
        json.dump(stats, f, indent=2)
    
    with open(os.path.join(log_dir, "training_stats.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        headers = list(stats.keys())
        writer.writerow(headers)
        for i in range(len(stats["update"])):
            writer.writerow([stats[h][i] for h in headers])

    # Plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Reward
    axes[0, 0].plot(stats["update"], stats["mean_reward"], marker="o", label="Global mean")
    axes[0, 0].plot(stats["update"], stats["mean_reward_p1"], marker="x", linestyle="--", label="P1 mean")
    axes[0, 0].set_xlabel("Update"); axes[0, 0].set_ylabel("Mean reward")
    axes[0, 0].set_title("Reward Learning Curve"); axes[0, 0].legend(); axes[0, 0].grid(True)
    
    # Cooperation
    axes[0, 1].plot(stats["update"], stats["p1_stag_rate"], marker="o", label="P1 STAG")
    axes[0, 1].plot(stats["update"], stats["p2_stag_rate"], marker="s", label="Opp STAG")
    axes[0, 1].plot(stats["update"], stats["illegal_rate"], marker="^", label="Illegal rate", color="red")
    axes[0, 1].set_xlabel("Update"); axes[0, 1].set_ylabel("Rate")
    axes[0, 1].set_title("Cooperation & Illegal Rates"); axes[0, 1].legend(); axes[0, 1].grid(True)
    
    # Losses
    axes[0, 2].plot(stats["update"], stats["policy_loss"], marker="o", label="Policy")
    axes[0, 2].plot(stats["update"], stats["value_loss"], marker="s", label="Value")
    axes[0, 2].set_xlabel("Update"); axes[0, 2].set_ylabel("Loss")
    axes[0, 2].set_title("PPO Losses"); axes[0, 2].legend(); axes[0, 2].grid(True)
    
    # KL
    axes[1, 0].plot(stats["update"], stats["approx_kl"], marker="o", label="Approx KL")
    axes[1, 0].axhline(y=args.target_kl, color='r', linestyle='--', label=f"Target KL={args.target_kl}")
    axes[1, 0].set_xlabel("Update"); axes[1, 0].set_ylabel("KL Divergence")
    axes[1, 0].set_title("KL Divergence"); axes[1, 0].legend(); axes[1, 0].grid(True)
    
    # Entropy
    axes[1, 1].plot(stats["update"], stats["entropy"], marker="o", color="orange")
    axes[1, 1].set_xlabel("Update"); axes[1, 1].set_ylabel("Entropy")
    axes[1, 1].set_title("Policy Entropy"); axes[1, 1].grid(True)
    
    # Action distribution
    if action_stats["episode"]:
        counts_array = np.vstack([action_stats[k] for k in action_categories])
        bottoms = np.zeros(len(action_stats["episode"]))
        for i, cat in enumerate(action_categories):
            axes[1, 2].bar(action_stats["episode"], counts_array[i], bottom=bottoms, label=cat, width=1.0)
            bottoms += counts_array[i]
        axes[1, 2].set_xlabel("Episode"); axes[1, 2].set_ylabel("Actions")
        axes[1, 2].set_title("Action Distribution"); axes[1, 2].legend(fontsize=6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "training_summary.png"), bbox_inches="tight", dpi=150)
    plt.close()

    logger.info("Training complete!")


if __name__ == "__main__":
    args = parse_args()
    train_ppo_stag_hunt(args)