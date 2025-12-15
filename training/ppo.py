from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


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
    moral_type: str = "utilitarian"
    opponent_moral_type: str = "game"
    player_moral_types: List[str] = field(default_factory=list)

    # Opponent settings (when not llm_vs_llm)
    opponent_type: str = "copy_focal"

    # Multi-agent training mode
    llm_vs_llm: bool = False
    shared_policy: bool = False

    # Training
    num_episodes: int = 1000
    batch_size: int = 5
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
    grad_accum_steps: int = 4

    # Generation
    max_new_tokens: int = 8
    temperature: float = 0.5
    top_p: float = 0.9

    # Output
    output_dir: str = "./outputs"
    seed: int = 42
    log_every: int = 10
    save_every: int = 100

    debug_log_path: str = "./debug_generations.jsonl"
    debug_log_topk: int = 10
    debug_log_only_illegal: bool = False


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
    gen_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      total_log_prob: sum of logprobs over generated tokens only (length = gen_len)
      value: value estimate (we keep value from prompt end token)
      gen_logits: logits over generated positions only (for entropy)
    """
    base_model = model.base_model
    outputs = base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    logits = outputs.logits
    hidden_states = outputs.hidden_states[-1]

    # shift for next-token likelihood
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    seq_len = shift_labels.shape[1]

    # Score ONLY the generated segment
    start = prompt_length - 1
    end = start + gen_len  # exclusive (in shifted space)

    mask = torch.zeros_like(token_log_probs)
    if start < seq_len:
        mask[:, start:min(end, seq_len)] = 1.0

    if attention_mask is not None:
        mask = mask * attention_mask[:, 1:]  # align with shift_labels

    total_log_prob = (token_log_probs * mask).sum(dim=-1)

    # Value from the last prompt token hidden state
    value_token_idx = prompt_length - 1
    value_hidden = hidden_states[:, value_token_idx, :]
    value = model.value_head(value_hidden)

    # logits only on generated positions
    gen_logits = shift_logits[:, start:min(end, seq_len), :]
    return total_log_prob, value, gen_logits


def compute_action_entropy(
    model: PolicyModelWithValueHead,
    prompt_ids: torch.Tensor,          # [1, prompt_len]
    action_token_ids: List[List[int]], # e.g. [[...],[...]]
) -> torch.Tensor:
    """
    Returns scalar entropy over the 2 (or K) actions by scoring each action *sequence*:
      logp(action) = sum_t log p(token_t | prompt, previous tokens)
    then p(action) = softmax(logp(action)), entropy = -sum p log p.
    """
    device = prompt_ids.device
    prompt_len = prompt_ids.shape[1]

    logps = []
    for a_ids in action_token_ids:
        a = torch.tensor(a_ids, device=device, dtype=prompt_ids.dtype).unsqueeze(0)  # [1, a_len]
        seq = torch.cat([prompt_ids, a], dim=1)                                      # [1, prompt+a_len]
        attn = torch.ones_like(seq)

        # Scores ONLY the appended action tokens (gen_len = len(a_ids))
        lp, _, _ = compute_log_probs(
            model=model,
            input_ids=seq,
            attention_mask=attn,
            prompt_length=prompt_len,
            gen_len=len(a_ids),
        )
        logps.append(lp)  # each is [1]

    logps = torch.stack(logps, dim=-1).squeeze(0)         # [K]
    probs = torch.softmax(logps, dim=-1)                  # [K]
    entropy = -(probs * torch.log(probs + 1e-12)).sum()   # scalar
    return entropy

@dataclass
class Experience:
    prompt: str
    completion: str
    input_ids: torch.Tensor
    prompt_length: int
    gen_len: int

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