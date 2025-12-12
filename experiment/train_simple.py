# """
# Standalone PPO Training for Moral Alignment

# This implementation doesn't rely on TRL's PPOTrainer (which has unstable API),
# instead using a clean, self-contained PPO implementation that follows
# the paper's methodology.

# Based on: "Moral Alignment for LLM Agents" (Tennant et al., ICLR 2025)

# Key features:
# - Clean PPO implementation with KL penalty
# - LoRA fine-tuning support
# - Reward scaling and normalization
# - Compatible with any HuggingFace model
# """

# import os
# import sys
# import json
# import argparse
# from typing import List, Dict, Any, Tuple, Optional
# from dataclasses import dataclass
# import numpy as np
# from collections import deque

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim import AdamW
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     get_linear_schedule_with_warmup,
# )
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# # Local imports
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# from .ipd import IPDEnv, IPDConfig, make_opponent
# from .prompts import build_ipd_prompt_with_chat_template, extract_action_from_completion
# from .rewards import compute_moral_reward


# @dataclass
# class PPOConfig:
#     """PPO training configuration."""
#     # Model
#     model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
#     use_lora: bool = True
#     lora_rank: int = 64
#     lora_alpha: int = 128
#     use_4bit: bool = False
    
#     # Game
#     moral_type: str = "utilitarian"
#     opponent_type: str = "tft"
    
#     # Training
#     num_episodes: int = 1000
#     batch_size: int = 5
#     ppo_epochs: int = 4
#     learning_rate: float = 1e-5
#     max_grad_norm: float = 1.0
    
#     # PPO hyperparameters
#     gamma: float = 1.0
#     lam: float = 0.95
#     clip_ratio: float = 0.2
#     vf_coef: float = 0.5
#     entropy_coef: float = 0.01  # Entropy bonus to prevent collapse
#     kl_coef: float = 0.1
#     target_kl: Optional[float] = 0.05
    
#     # Reward
#     xi: float = 3.0
#     illegal_penalty: float = -6.0
#     reward_scale: float = 1.0
#     normalize_rewards: bool = False
#     normalize_advantages: bool = True
#     # Reward shaping: add small positive reward for valid actions
#     reward_shaping: bool = True
#     valid_action_bonus: float = 0.1  # Small bonus for producing valid output
    
#     # Generation
#     max_new_tokens: int = 8
#     temperature: float = 0.7
#     top_p: float = 0.9
    
#     # Output
#     output_dir: str = "./outputs"
#     seed: int = 42
#     log_every: int = 10
#     save_every: int = 100


# class ValueHead(nn.Module):
#     """Value head for PPO."""
    
#     def __init__(self, hidden_size: int, dropout: float = 0.1):
#         super().__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.linear1 = nn.Linear(hidden_size, hidden_size // 2)
#         self.linear2 = nn.Linear(hidden_size // 2, 1)
        
#         # Initialize with small values
#         nn.init.normal_(self.linear1.weight, std=0.01)
#         nn.init.zeros_(self.linear1.bias)
#         nn.init.normal_(self.linear2.weight, std=0.01)
#         nn.init.zeros_(self.linear2.bias)
    
#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             hidden_states: [batch, seq_len, hidden_size]
#         Returns:
#             values: [batch]
#         """
#         # Use last token's hidden state
#         x = hidden_states[:, -1, :]
        
#         # Convert to same dtype as weights if needed
#         if x.dtype != self.linear1.weight.dtype:
#             x = x.to(self.linear1.weight.dtype)
        
#         x = self.dropout(x)
#         x = F.relu(self.linear1(x))
#         x = self.linear2(x)
#         return x.squeeze(-1)


# class PolicyModelWithValueHead(nn.Module):
#     """Wrapper that adds a value head to a causal LM."""
    
#     def __init__(self, base_model, hidden_size: int, device=None):
#         super().__init__()
#         self.base_model = base_model
#         self.value_head = ValueHead(hidden_size)
#         self.config = base_model.config
#         self._device = device
        
#         # Move value head to same device and dtype as model if specified
#         if device is not None:
#             self.value_head = self.value_head.to(device)
        
#         # Match dtype of base model
#         try:
#             model_dtype = next(base_model.parameters()).dtype
#             self.value_head = self.value_head.to(model_dtype)
#         except StopIteration:
#             pass
    
#     def forward(self, input_ids, attention_mask=None, **kwargs):
#         outputs = self.base_model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             output_hidden_states=True,
#             **kwargs
#         )
#         hidden_states = outputs.hidden_states[-1]
        
#         # Ensure value head is on same device as hidden states
#         target_device = hidden_states.device
#         target_dtype = hidden_states.dtype
        
#         if self.value_head.linear1.weight.device != target_device:
#             self.value_head = self.value_head.to(target_device)
        
#         if self.value_head.linear1.weight.dtype != target_dtype:
#             self.value_head = self.value_head.to(target_dtype)
        
#         value = self.value_head(hidden_states)
#         return outputs, value
    
#     def generate(self, *args, **kwargs):
#         return self.base_model.generate(*args, **kwargs)
    
#     def to(self, device):
#         """Override to method to also move value head."""
#         self.base_model = self.base_model.to(device)
#         self.value_head = self.value_head.to(device)
#         self._device = device
#         return self
    
#     def save_pretrained(self, path: str):
#         os.makedirs(path, exist_ok=True)
#         self.base_model.save_pretrained(path)
#         # Save value head on CPU
#         torch.save(
#             {k: v.cpu() for k, v in self.value_head.state_dict().items()}, 
#             os.path.join(path, "value_head.pt")
#         )
    
#     @classmethod
#     def from_pretrained(cls, path: str, device=None, **kwargs):
#         base_model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
#         hidden_size = base_model.config.hidden_size
#         model = cls(base_model, hidden_size, device=device)
        
#         value_head_path = os.path.join(path, "value_head.pt")
#         if os.path.exists(value_head_path):
#             state_dict = torch.load(value_head_path, map_location="cpu")
#             model.value_head.load_state_dict(state_dict)
#             if device is not None:
#                 model.value_head = model.value_head.to(device)
#         return model


# def compute_log_probs(
#     model,
#     input_ids: torch.Tensor,
#     attention_mask: torch.Tensor,
#     prompt_length: int,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Compute log probabilities and value for a sequence.
    
#     Returns:
#         log_prob: Sum of log probs for generated tokens
#         value: Value estimate
#     """
#     outputs, value = model(input_ids=input_ids, attention_mask=attention_mask)
#     logits = outputs.logits
    
#     # Shift for next token prediction
#     shift_logits = logits[:, :-1, :]
#     shift_labels = input_ids[:, 1:]
    
#     # Log probs
#     log_probs = F.log_softmax(shift_logits, dim=-1)
#     token_log_probs = log_probs.gather(
#         dim=-1, 
#         index=shift_labels.unsqueeze(-1)
#     ).squeeze(-1)
    
#     # Mask for generated tokens only (after prompt)
#     seq_len = shift_labels.shape[1]
#     mask = torch.zeros_like(token_log_probs)
#     if prompt_length - 1 < seq_len:
#         mask[:, prompt_length - 1:] = 1.0
    
#     # Apply attention mask
#     if attention_mask is not None:
#         mask = mask * attention_mask[:, 1:]
    
#     # Sum log prob over generated tokens
#     masked_log_probs = token_log_probs * mask
#     total_log_prob = masked_log_probs.sum(dim=-1)
    
#     return total_log_prob, value


# def compute_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#     """Compute entropy of the policy."""
#     probs = F.softmax(logits, dim=-1)
#     log_probs = F.log_softmax(logits, dim=-1)
#     entropy = -(probs * log_probs).sum(dim=-1)
    
#     # Masked mean
#     masked_entropy = (entropy * mask).sum() / mask.sum().clamp(min=1)
#     return masked_entropy


# @dataclass
# class Experience:
#     """Single experience from rollout."""
#     prompt: str
#     completion: str
#     input_ids: torch.Tensor
#     prompt_length: int
#     reward: float
#     value: float
#     log_prob: float
#     ref_log_prob: float
#     action: str
#     opponent_prev_action: Optional[str]
#     is_legal: bool
#     advantage: float = 0.0
#     returns: float = 0.0


# class MoralPPOTrainer:
#     """
#     PPO trainer for moral alignment of LLM agents.
#     """
    
#     def __init__(self, config: PPOConfig):
#         self.config = config
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         # Set seeds
#         torch.manual_seed(config.seed)
#         np.random.seed(config.seed)
        
#         # Load models
#         self._setup_models()
        
#         # Optimizer
#         self.optimizer = AdamW(
#             self.model.parameters(),
#             lr=config.learning_rate,
#             eps=1e-5,
#         )
        
#         # Environment
#         self.env_config = IPDConfig()
#         self.opponent = make_opponent(config.opponent_type)
        
#         # Stats
#         self.reward_history = deque(maxlen=100)
#         self.stats_history = []
    
#     def _setup_models(self):
#         """Initialize policy and reference models."""
#         config = self.config
        
#         print(f"Loading model: {config.model_name}")
        
#         # Tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             config.model_name,
#             trust_remote_code=True,
#             padding_side="left",
#         )
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
        
#         # Quantization
#         bnb_config = None
#         # if config.use_4bit:
#         #     bnb_config = BitsAndBytesConfig(
#         #         load_in_4bit=True,
#         #         bnb_4bit_compute_dtype=torch.float16,
#         #         bnb_4bit_quant_type="nf4",
#         #     )
        
#         # Model loading kwargs
#         model_kwargs = {
#             "trust_remote_code": True,
#         }
        
#         if bnb_config is not None:
#             model_kwargs["quantization_config"] = bnb_config
#             model_kwargs["device_map"] = "auto"
#         else:
#             # Load to specific device without quantization
#             model_kwargs["device_map"] = "auto"
#             model_kwargs["torch_dtype"] = torch.float16
        
#         # Load base model
#         base_model = AutoModelForCausalLM.from_pretrained(
#             config.model_name,
#             **model_kwargs,
#         )
        
#         # Apply LoRA
#         if config.use_lora:
#             if config.use_4bit:
#                 base_model = prepare_model_for_kbit_training(base_model)
            
#             lora_config = LoraConfig(
#                 r=config.lora_rank,
#                 lora_alpha=config.lora_alpha,
#                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#                 lora_dropout=0.05,
#                 bias="none",
#                 task_type="CAUSAL_LM",
#             )
#             base_model = get_peft_model(base_model, lora_config)
#             base_model.print_trainable_parameters()
        
#         # Get the device the model is on
#         model_device = next(base_model.parameters()).device
        
#         # Wrap with value head
#         hidden_size = base_model.config.hidden_size
#         self.model = PolicyModelWithValueHead(base_model, hidden_size, device=model_device)
        
#         # Reference model (frozen)
#         print("Loading reference model...")
#         ref_base = AutoModelForCausalLM.from_pretrained(
#             config.model_name,
#             **model_kwargs,
#         )
#         ref_device = next(ref_base.parameters()).device
#         self.ref_model = PolicyModelWithValueHead(ref_base, hidden_size, device=ref_device)
#         self.ref_model.eval()
#         for param in self.ref_model.parameters():
#             param.requires_grad = False
    
#     def rollout_episode(self) -> List[Experience]:
#         """
#         Run one episode and collect experiences.
#         """
#         config = self.config
#         env = IPDEnv(self.env_config)
#         obs = env.reset(random_initial_state=True)
#         self.opponent.reset()
        
#         experiences = []
#         self.model.eval()
        
#         # Single step game (as in paper - each episode is one interaction)
#         history = obs.get("history", [])
#         opp_prev = history[-1][1] if history else None
        
#         # Build prompt
#         prompt = build_ipd_prompt_with_chat_template(
#             obs, self.tokenizer, self.env_config, ("action1", "action2")
#         )
        
#         # Tokenize
#         encoded = self.tokenizer(
#             prompt,
#             return_tensors="pt",
#             padding=False,
#             truncation=True,
#         ).to(self.device)
#         prompt_length = encoded["input_ids"].shape[1]
        
#         # Generate
#         with torch.no_grad():
#             output_ids = self.model.generate(
#                 input_ids=encoded["input_ids"],
#                 attention_mask=encoded["attention_mask"],
#                 max_new_tokens=config.max_new_tokens,
#                 do_sample=True,
#                 temperature=config.temperature,
#                 top_p=config.top_p,
#                 pad_token_id=self.tokenizer.pad_token_id,
#             )
        
#         # Full sequence
#         full_ids = output_ids
#         attention_mask = torch.ones_like(full_ids)
        
#         # Compute log probs and value
#         with torch.no_grad():
#             log_prob, value = compute_log_probs(
#                 self.model, full_ids, attention_mask, prompt_length
#             )
#             ref_log_prob, _ = compute_log_probs(
#                 self.ref_model, full_ids, attention_mask, prompt_length
#             )
        
#         # Decode completion
#         completion = self.tokenizer.decode(
#             output_ids[0, prompt_length:],
#             skip_special_tokens=True
#         ).strip()
        
#         # Parse action
#         action, is_legal = extract_action_from_completion(
#             completion, ("action1", "action2")
#         )
        
#         # Get opponent action and compute rewards
#         opp_action = self.opponent.act(obs)
        
#         if is_legal:
#             _, (r_agent, r_opp), _, _ = env.step(action, opp_action)
#         else:
#             r_agent, r_opp = 0.0, 0.0
        
#         # Compute moral reward
#         reward = compute_moral_reward(
#             config.moral_type,
#             action,
#             opp_prev,
#             r_agent,
#             r_opp,
#             is_legal,
#             config.xi,
#             config.illegal_penalty,
#         )
        
#         # Apply reward shaping: small bonus for valid actions
#         # This helps with sparse reward signals (especially deontological)
#         if config.reward_shaping and is_legal:
#             reward += config.valid_action_bonus
        
#         exp = Experience(
#             prompt=prompt,
#             completion=completion,
#             input_ids=full_ids.cpu(),
#             prompt_length=prompt_length,
#             reward=reward,
#             value=value.item(),
#             log_prob=log_prob.item(),
#             ref_log_prob=ref_log_prob.item(),
#             action=action,
#             opponent_prev_action=opp_prev,
#             is_legal=is_legal,
#         )
#         experiences.append(exp)
        
#         return experiences
    
#     def collect_batch(self) -> List[Experience]:
#         """Collect a batch of experiences."""
#         all_experiences = []
        
#         for _ in range(self.config.batch_size):
#             exps = self.rollout_episode()
#             all_experiences.extend(exps)
        
#         return all_experiences
    
#     def compute_advantages(self, experiences: List[Experience]) -> List[Experience]:
#         """Compute advantages using GAE."""
#         config = self.config
        
#         # Get raw rewards and values
#         raw_rewards = np.array([e.reward for e in experiences])
#         values = np.array([e.value for e in experiences])
        
#         # Scale rewards (but don't normalize to 0 mean yet)
#         rewards = raw_rewards * config.reward_scale
        
#         # Only normalize if there's meaningful variance
#         if config.normalize_rewards and len(rewards) > 1:
#             reward_std = rewards.std()
#             if reward_std > 1e-6:  # Only normalize if there's variance
#                 rewards = (rewards - rewards.mean()) / (reward_std + 1e-8)
#             # If all rewards are the same, just center them
#             else:
#                 rewards = rewards - rewards.mean()
        
#         # Simple advantage: reward - value (since single-step episodes)
#         advantages = rewards - values
#         returns = rewards
        
#         # Normalize advantages only if there's meaningful variance
#         if config.normalize_advantages and len(advantages) > 1:
#             adv_std = advantages.std()
#             if adv_std > 1e-6:
#                 advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
        
#         # Update experiences
#         for i, exp in enumerate(experiences):
#             exp.advantage = float(advantages[i])
#             exp.returns = float(returns[i])
#             exp.reward = float(rewards[i])  # Store scaled reward
        
#         return experiences
    
#     def ppo_update(self, experiences: List[Experience]) -> Dict[str, float]:
#         """Perform PPO update."""
#         config = self.config
#         self.model.train()
        
#         total_policy_loss = 0.0
#         total_value_loss = 0.0
#         total_entropy = 0.0
#         total_kl = 0.0
#         num_updates = 0
        
#         for epoch in range(config.ppo_epochs):
#             # Shuffle experiences
#             indices = np.random.permutation(len(experiences))
            
#             for idx in indices:
#                 exp = experiences[idx]
                
#                 # Skip if advantage is 0 (no learning signal)
#                 if abs(exp.advantage) < 1e-8:
#                     continue
                
#                 # Move to device
#                 input_ids = exp.input_ids.to(self.device)
#                 attention_mask = torch.ones_like(input_ids)
                
#                 # Forward pass - get full outputs for entropy calculation
#                 outputs, value = self.model(input_ids=input_ids, attention_mask=attention_mask)
#                 logits = outputs.logits
                
#                 # Compute log prob for the generated tokens
#                 log_prob, _ = compute_log_probs(
#                     self.model, input_ids, attention_mask, exp.prompt_length
#                 )
                
#                 # Compute entropy over generated tokens for regularization
#                 # Use logits from position prompt_length-1 onward
#                 gen_logits = logits[:, exp.prompt_length-1:-1, :]  # [1, gen_len, vocab]
#                 gen_probs = F.softmax(gen_logits, dim=-1)
#                 gen_log_probs = F.log_softmax(gen_logits, dim=-1)
#                 entropy = -(gen_probs * gen_log_probs).sum(dim=-1).mean()  # Mean entropy per token
                
#                 # Old values (from rollout)
#                 old_log_prob = torch.tensor(exp.log_prob, device=self.device, dtype=log_prob.dtype)
#                 old_value = torch.tensor(exp.value, device=self.device, dtype=value.dtype)
#                 advantage = torch.tensor(exp.advantage, device=self.device, dtype=log_prob.dtype)
#                 returns = torch.tensor(exp.returns, device=self.device, dtype=value.dtype)
#                 ref_log_prob = torch.tensor(exp.ref_log_prob, device=self.device, dtype=log_prob.dtype)
                
#                 # Policy loss with clipping
#                 log_ratio = log_prob - old_log_prob
#                 ratio = torch.exp(log_ratio)
                
#                 # Clipped surrogate objective
#                 surr1 = ratio * advantage
#                 surr2 = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * advantage
#                 policy_loss = -torch.min(surr1, surr2)
                
#                 # Value loss with clipping
#                 value_clipped = old_value + torch.clamp(
#                     value - old_value, -config.clip_ratio, config.clip_ratio
#                 )
#                 value_loss1 = (value - returns) ** 2
#                 value_loss2 = (value_clipped - returns) ** 2
#                 value_loss = 0.5 * torch.max(value_loss1, value_loss2)
                
#                 # KL divergence approximation (always positive)
#                 approx_kl = 0.5 * (old_log_prob - log_prob) ** 2
                
#                 # Total loss with entropy bonus (negative because we want to maximize entropy)
#                 loss = (policy_loss 
#                         + config.vf_coef * value_loss 
#                         + config.kl_coef * approx_kl
#                         - config.entropy_coef * entropy)  # Subtract to maximize entropy
                
#                 # Backward
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
#                 self.optimizer.step()
                
#                 # Stats
#                 total_policy_loss += policy_loss.item()
#                 total_value_loss += value_loss.item()
#                 total_kl += approx_kl.item()
#                 total_entropy += entropy.item()
#                 num_updates += 1
            
#             # Early stopping on KL
#             if num_updates > 0:
#                 avg_kl = total_kl / num_updates
#                 if config.target_kl and avg_kl > config.target_kl * 1.5:
#                     print(f"  Early stopping at epoch {epoch+1} due to KL: {avg_kl:.4f}")
#                     break
        
#         n = max(num_updates, 1)
#         return {
#             "policy_loss": total_policy_loss / n,
#             "value_loss": total_value_loss / n,
#             "kl": total_kl / n,
#             "entropy": total_entropy / n,
#         }
    
#     def train(self):
#         """Main training loop."""
#         config = self.config
        
#         output_path = os.path.join(
#             config.output_dir,
#             f"{config.moral_type}_vs_{config.opponent_type}"
#         )
#         os.makedirs(output_path, exist_ok=True)
        
#         print(f"\n{'='*60}")
#         print(f"Training: {config.moral_type} vs {config.opponent_type}")
#         print(f"Episodes: {config.num_episodes}, Batch size: {config.batch_size}")
#         print(f"{'='*60}\n")
        
#         for episode in range(1, config.num_episodes + 1):
#             # Collect experiences
#             experiences = self.collect_batch()
            
#             # Store raw rewards before any processing
#             raw_rewards = [e.reward for e in experiences]
            
#             # Compute advantages
#             experiences = self.compute_advantages(experiences)
            
#             # PPO update
#             train_stats = self.ppo_update(experiences)
            
#             # Calculate episode stats using RAW rewards (before scaling)
#             coop_rate = sum(1 for e in experiences if e.action == "action1") / len(experiences)
#             illegal_rate = sum(1 for e in experiences if not e.is_legal) / len(experiences)
            
#             self.reward_history.extend(raw_rewards)
            
#             stats = {
#                 "episode": episode,
#                 "mean_reward": float(np.mean(raw_rewards)),  # Raw reward for interpretability
#                 "std_reward": float(np.std(raw_rewards)),
#                 "min_reward": float(np.min(raw_rewards)),
#                 "max_reward": float(np.max(raw_rewards)),
#                 "cooperation_rate": coop_rate,
#                 "illegal_rate": illegal_rate,
#                 **train_stats,
#             }
#             self.stats_history.append(stats)
            
#             # Logging
#             if episode % config.log_every == 0:
#                 entropy_str = f"Ent: {train_stats.get('entropy', 0):.2f} | " if 'entropy' in train_stats else ""
#                 print(
#                     f"Ep {episode:4d}/{config.num_episodes} | "
#                     f"R: {stats['mean_reward']:+.2f} (±{stats['std_reward']:.2f}) | "
#                     f"Coop: {coop_rate:.0%} | "
#                     f"Ill: {illegal_rate:.0%} | "
#                     f"{entropy_str}"
#                     f"KL: {train_stats['kl']:.4f} | "
#                     f"PL: {train_stats['policy_loss']:.4f}"
#                 )
            
#             # Save checkpoint
#             if episode % config.save_every == 0:
#                 ckpt_path = os.path.join(output_path, f"checkpoint_{episode}")
#                 self.model.save_pretrained(ckpt_path)
#                 self.tokenizer.save_pretrained(ckpt_path)
        
#         # Save final model
#         final_path = os.path.join(output_path, "final_model")
#         self.model.save_pretrained(final_path)
#         self.tokenizer.save_pretrained(final_path)
        
#         # Save stats
#         stats_path = os.path.join(output_path, "training_stats.json")
#         with open(stats_path, "w") as f:
#             json.dump(self.stats_history, f, indent=2)
        
#         # Save config
#         config_path = os.path.join(output_path, "config.json")
#         with open(config_path, "w") as f:
#             json.dump(vars(config), f, indent=2)
        
#         print(f"\nTraining complete!")
#         print(f"Model saved to {final_path}")
#         print(f"Stats saved to {stats_path}")
        
#         return self.stats_history


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
#     parser.add_argument("--moral_type", type=str, default="utilitarian",
#                        choices=["game", "deontological", "utilitarian", "game+deontological"])
#     parser.add_argument("--opponent_type", type=str, default="tft",
#                        choices=["tft", "always_cooperate", "always_defect", "random"])
#     parser.add_argument("--num_episodes", type=int, default=1000)
#     parser.add_argument("--batch_size", type=int, default=5)
#     parser.add_argument("--ppo_epochs", type=int, default=4)
#     parser.add_argument("--learning_rate", type=float, default=1e-5)
#     parser.add_argument("--lora_rank", type=int, default=64)
#     parser.add_argument("--output_dir", type=str, default="./outputs")
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--use_4bit", action="store_true")
#     parser.add_argument("--log_every", type=int, default=10)
#     parser.add_argument("--save_every", type=int, default=100)
    
#     args = parser.parse_args()
    
#     config = PPOConfig(
#         model_name=args.model_name,
#         moral_type=args.moral_type,
#         opponent_type=args.opponent_type,
#         num_episodes=args.num_episodes,
#         batch_size=args.batch_size,
#         ppo_epochs=args.ppo_epochs,
#         learning_rate=args.learning_rate,
#         lora_rank=args.lora_rank,
#         output_dir=args.output_dir,
#         seed=args.seed,
#         use_4bit=args.use_4bit,
#         log_every=args.log_every,
#         save_every=args.save_every,
#     )
    
#     trainer = MoralPPOTrainer(config)
#     trainer.train()


# if __name__ == "__main__":
#     main()


"""
Standalone PPO Training for Moral Alignment with LLM vs LLM Support

This implementation supports both:
1. LLM vs Fixed-Strategy Opponent (TFT, Always Cooperate, etc.)
2. LLM vs LLM (two separate policies trained simultaneously)

Based on: "Moral Alignment for LLM Agents" (Tennant et al., ICLR 2025)

Key features:
- Clean PPO implementation with KL penalty
- LoRA fine-tuning support
- Reward scaling and normalization
- LLM vs LLM multi-agent training
- Compatible with any HuggingFace model
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from collections import deque
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ============================================================================
# IPD Environment and Opponents
# ============================================================================

@dataclass
class IPDConfig:
    """Configuration for the Iterated Prisoner's Dilemma environment."""
    # Payoff matrix: (row_player_payoff, col_player_payoff)
    # Format: payoffs[row_action][col_action]
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
        """Reset the environment."""
        self.history = []
        self.step_count = 0
        
        if random_initial_state and np.random.random() < 0.5:
            # Generate a random initial history
            actions = self.config.action_names
            prev_agent = np.random.choice(actions)
            prev_opp = np.random.choice(actions)
            self.history = [(prev_agent, prev_opp)]
        
        return {"history": self.history.copy()}
    
    def step(self, agent_action: str, opponent_action: str) -> Tuple[Dict, Tuple[float, float], bool, Dict]:
        """Take a step in the environment."""
        payoffs = self.config.payoffs[agent_action][opponent_action]
        self.history.append((agent_action, opponent_action))
        self.step_count += 1
        
        done = self.step_count >= self.config.max_steps
        
        return (
            {"history": self.history.copy()},
            payoffs,
            done,
            {}
        )


class Opponent:
    """Base class for opponents."""
    
    def reset(self):
        pass
    
    def act(self, obs: Dict[str, Any]) -> str:
        raise NotImplementedError


class TitForTat(Opponent):
    """Tit-for-Tat opponent: cooperates first, then copies opponent's last move."""
    
    def act(self, obs: Dict[str, Any]) -> str:
        history = obs.get("history", [])
        if not history:
            return "action1"  # Cooperate first
        # Copy opponent's (the agent's) last action
        return history[-1][0]


class AlwaysCooperate(Opponent):
    """Always cooperates."""
    
    def act(self, obs: Dict[str, Any]) -> str:
        return "action1"


class AlwaysDefect(Opponent):
    """Always defects."""
    
    def act(self, obs: Dict[str, Any]) -> str:
        return "action2"


class RandomOpponent(Opponent):
    """Plays randomly."""
    
    def act(self, obs: Dict[str, Any]) -> str:
        return np.random.choice(["action1", "action2"])


def make_opponent(opponent_type: str) -> Opponent:
    """Factory function to create opponents."""
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
    player_perspective: str = "agent"  # "agent" or "opponent" for LLM vs LLM
) -> str:
    """Build the IPD prompt using the model's chat template."""
    
    history = obs.get("history", [])
    
    # Build the game description
    prompt_text = f"""You are playing a game with another agent A. You must choose either action {action_names[0]} or action {action_names[1]}. Depending on your action and A's action, you each get a certain number of points. The points are awarded as follows (you are the row player, A is the column player):

| | {action_names[0]} | {action_names[1]} |
| ------- | ------- | ------- |
| {action_names[0]} | 3, 3 | 0, 4 |
| {action_names[1]} | 4, 0 | 1, 1 |
"""
    
    # Add history if available
    if history:
        if player_perspective == "agent":
            my_last = history[-1][0]
            their_last = history[-1][1]
        else:  # opponent perspective (for LLM vs LLM)
            my_last = history[-1][1]
            their_last = history[-1][0]
        
        my_points = config.payoffs[history[-1][0]][history[-1][1]][0 if player_perspective == "agent" else 1]
        their_points = config.payoffs[history[-1][0]][history[-1][1]][1 if player_perspective == "agent" else 0]
        
        prompt_text += f"""
You have played with this opponent before. Last time, you played {my_last} and they played {their_last}, so you got {int(my_points)} points and A got {int(their_points)} points. What action would you take in order to achieve the highest possible score in points?"""
    else:
        prompt_text += """
What action would you take in order to achieve the highest possible score in points?"""
    
    prompt_text += f"""
Your answer must follow this format exactly: choose either {action_names[0]} or {action_names[1]}. Do not explain your reasoning. Your answer:"""
    
    # Apply chat template
    messages = [{"role": "user", "content": prompt_text}]
    
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        # Fallback if chat template not available
        formatted = prompt_text
    
    return formatted


def extract_action_from_completion(
    completion: str,
    action_names: Tuple[str, str]
) -> Tuple[str, bool]:
    """Extract the action from the model's completion."""
    completion_lower = completion.lower().strip()
    
    # Check for exact matches first
    for action in action_names:
        if action.lower() in completion_lower:
            return action, True
    
    # Check if completion starts with action
    for action in action_names:
        if completion_lower.startswith(action.lower()):
            return action, True
    
    # Default to first action but mark as illegal
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
    """
    Compute moral reward based on the moral framework.
    
    Args:
        moral_type: One of "game", "deontological", "utilitarian", "game+deontological"
        agent_action: The action taken by the agent
        opponent_prev_action: The opponent's previous action (for deontological)
        agent_payoff: The agent's payoff from the game
        opponent_payoff: The opponent's payoff from the game
        is_legal: Whether the action was legal
        xi: Penalty for violating deontological norm
        illegal_penalty: Penalty for illegal moves
    """
    if not is_legal:
        return illegal_penalty
    
    if moral_type == "game":
        return agent_payoff
    
    elif moral_type == "deontological":
        # Penalty for defecting against a cooperator
        if agent_action == "action2" and opponent_prev_action == "action1":
            return -xi
        return 0.0
    
    elif moral_type == "utilitarian":
        # Collective welfare
        return agent_payoff + opponent_payoff
    
    elif moral_type == "game+deontological":
        reward = agent_payoff
        if agent_action == "action2" and opponent_prev_action == "action1":
            reward -= xi
        return reward
    
    else:
        raise ValueError(f"Unknown moral type: {moral_type}")


# ============================================================================
# PPO Components
# ============================================================================

@dataclass
class PPOConfig:
    """PPO training configuration."""
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    use_4bit: bool = False
    
    # Game
    moral_type: str = "utilitarian"
    opponent_type: str = "tft"
    
    # LLM vs LLM settings
    llm_vs_llm: bool = False
    opponent_moral_type: str = "game"  # Moral type for opponent LLM
    shared_base_model: bool = True  # Whether to share the base model (only LoRA differs)
    
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
    
    # Reward
    xi: float = 3.0
    illegal_penalty: float = -6.0
    reward_scale: float = 1.0
    normalize_rewards: bool = False
    normalize_advantages: bool = True
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


class ValueHead(nn.Module):
    """Value head for PPO."""
    
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
        x = hidden_states[:, -1, :]
        if x.dtype != self.linear1.weight.dtype:
            x = x.to(self.linear1.weight.dtype)
        x = self.dropout(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x.squeeze(-1)


class PolicyModelWithValueHead(nn.Module):
    """Wrapper that adds a value head to a causal LM."""
    
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
        
        target_device = hidden_states.device
        target_dtype = hidden_states.dtype
        
        if self.value_head.linear1.weight.device != target_device:
            self.value_head = self.value_head.to(target_device)
        
        if self.value_head.linear1.weight.dtype != target_dtype:
            self.value_head = self.value_head.to(target_dtype)
        
        value = self.value_head(hidden_states)
        return outputs, value
    
    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)
    
    def to(self, device):
        self.base_model = self.base_model.to(device)
        self.value_head = self.value_head.to(device)
        self._device = device
        return self
    
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
        
        value_head_path = os.path.join(path, "value_head.pt")
        if os.path.exists(value_head_path):
            state_dict = torch.load(value_head_path, map_location="cpu")
            model.value_head.load_state_dict(state_dict)
            if device is not None:
                model.value_head = model.value_head.to(device)
        return model


def compute_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute log probabilities and value for a sequence."""
    outputs, value = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, 
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    seq_len = shift_labels.shape[1]
    mask = torch.zeros_like(token_log_probs)
    if prompt_length - 1 < seq_len:
        mask[:, prompt_length - 1:] = 1.0
    
    if attention_mask is not None:
        mask = mask * attention_mask[:, 1:]
    
    masked_log_probs = token_log_probs * mask
    total_log_prob = masked_log_probs.sum(dim=-1)
    
    return total_log_prob, value


@dataclass
class Experience:
    """Single experience from rollout."""
    prompt: str
    completion: str
    input_ids: torch.Tensor
    prompt_length: int
    reward: float
    value: float
    log_prob: float
    ref_log_prob: float
    action: str
    opponent_prev_action: Optional[str]
    is_legal: bool
    player_id: int = 0  # 0 for agent, 1 for opponent (in LLM vs LLM)
    advantage: float = 0.0
    returns: float = 0.0


# ============================================================================
# LLM Opponent for LLM vs LLM Training
# ============================================================================

class LLMOpponent:
    """
    An LLM-based opponent for LLM vs LLM training.
    This wraps a PolicyModelWithValueHead and provides the same interface as fixed opponents.
    """
    
    def __init__(
        self,
        model: PolicyModelWithValueHead,
        ref_model: PolicyModelWithValueHead,
        tokenizer,
        config: PPOConfig,
        env_config: IPDConfig,
        moral_type: str = "game",
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.env_config = env_config
        self.moral_type = moral_type
        self.device = next(model.parameters()).device
        
        # Store experiences for training
        self.experiences: List[Experience] = []
        self.last_experience: Optional[Experience] = None
    
    def reset(self):
        """Reset the opponent's state."""
        self.experiences = []
        self.last_experience = None
    
    def act(self, obs: Dict[str, Any]) -> Tuple[str, Experience]:
        """
        Generate an action and return both the action and the experience.
        The experience is stored for later training.
        """
        self.model.eval()
        
        history = obs.get("history", [])
        opp_prev = history[-1][0] if history else None  # Agent's last action from opponent's perspective
        
        # Build prompt from opponent's perspective
        prompt = build_ipd_prompt_with_chat_template(
            obs, self.tokenizer, self.env_config,
            self.config.action_names if hasattr(self.config, 'action_names') else ("action1", "action2"),
            player_perspective="opponent"
        )
        
        # Tokenize
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
        ).to(self.device)
        prompt_length = encoded["input_ids"].shape[1]
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        full_ids = output_ids
        attention_mask = torch.ones_like(full_ids)
        
        # Compute log probs and value
        with torch.no_grad():
            log_prob, value = compute_log_probs(
                self.model, full_ids, attention_mask, prompt_length
            )
            ref_log_prob, _ = compute_log_probs(
                self.ref_model, full_ids, attention_mask, prompt_length
            )
        
        # Decode completion
        completion = self.tokenizer.decode(
            output_ids[0, prompt_length:],
            skip_special_tokens=True
        ).strip()
        
        # Parse action
        action, is_legal = extract_action_from_completion(
            completion, ("action1", "action2")
        )
        
        # Create experience (reward will be filled in later)
        exp = Experience(
            prompt=prompt,
            completion=completion,
            input_ids=full_ids.cpu(),
            prompt_length=prompt_length,
            reward=0.0,  # Will be computed after both players act
            value=value.item(),
            log_prob=log_prob.item(),
            ref_log_prob=ref_log_prob.item(),
            action=action,
            opponent_prev_action=opp_prev,
            is_legal=is_legal,
            player_id=1,  # Mark as opponent
        )
        
        self.last_experience = exp
        return action, exp
    
    def finalize_experience(
        self,
        agent_action: str,
        agent_payoff: float,
        opponent_payoff: float,
    ):
        """
        Finalize the last experience by computing the reward.
        Called after both players have acted.
        """
        if self.last_experience is None:
            return
        
        exp = self.last_experience
        
        # Compute reward from opponent's perspective
        # Note: from opponent's perspective, their action is exp.action
        # and the "opponent" (agent) took agent_action
        reward = compute_moral_reward(
            self.moral_type,
            exp.action,  # Opponent's action
            exp.opponent_prev_action,  # Agent's previous action
            opponent_payoff,  # Opponent's payoff
            agent_payoff,  # Agent's payoff (other player for opponent)
            exp.is_legal,
            self.config.xi,
            self.config.illegal_penalty,
        )
        
        if self.config.reward_shaping and exp.is_legal:
            reward += self.config.valid_action_bonus
        
        exp.reward = reward
        self.experiences.append(exp)
        self.last_experience = None
    
    def get_experiences(self) -> List[Experience]:
        """Get all collected experiences."""
        return self.experiences


# ============================================================================
# Main Trainer
# ============================================================================

class MoralPPOTrainer:
    """
    PPO trainer for moral alignment of LLM agents.
    Supports both LLM vs Fixed-Strategy and LLM vs LLM training.
    """
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Load models
        self._setup_models()
        
        # Optimizer for agent
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
        )
        
        # Environment
        self.env_config = IPDConfig()
        
        # Setup opponent (fixed or LLM)
        if config.llm_vs_llm:
            self._setup_llm_opponent()
        else:
            self.opponent = make_opponent(config.opponent_type)
            self.llm_opponent = None
        
        # Stats
        self.reward_history = deque(maxlen=100)
        self.opponent_reward_history = deque(maxlen=100)
        self.stats_history = []
    
    def _setup_models(self):
        """Initialize policy and reference models for the agent."""
        config = self.config
        
        print(f"Loading model: {config.model_name}")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model loading kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
        
        # Load base model for agent
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs,
        )
        
        # Apply LoRA
        if config.use_lora:
            if config.use_4bit:
                base_model = prepare_model_for_kbit_training(base_model)
            
            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            base_model = get_peft_model(base_model, lora_config)
            print("Agent model:")
            base_model.print_trainable_parameters()
        
        model_device = next(base_model.parameters()).device
        hidden_size = base_model.config.hidden_size
        self.model = PolicyModelWithValueHead(base_model, hidden_size, device=model_device)
        
        # Reference model (frozen)
        print("Loading reference model...")
        ref_base = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs,
        )
        ref_device = next(ref_base.parameters()).device
        self.ref_model = PolicyModelWithValueHead(ref_base, hidden_size, device=ref_device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def _setup_llm_opponent(self):
        """Setup the LLM opponent for LLM vs LLM training."""
        config = self.config
        print("\nSetting up LLM opponent...")
        
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
        
        if config.shared_base_model:
            # Share the base model but use separate LoRA adapters
            print("Using shared base model with separate LoRA adapters")
            
            # Load a fresh base model for opponent
            opp_base = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                **model_kwargs,
            )
            
            if config.use_lora:
                if config.use_4bit:
                    opp_base = prepare_model_for_kbit_training(opp_base)
                
                # Use a different LoRA config for opponent (can be same parameters)
                lora_config = LoraConfig(
                    r=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                opp_base = get_peft_model(opp_base, lora_config)
                print("Opponent model:")
                opp_base.print_trainable_parameters()
        else:
            # Completely separate models
            print("Using separate base models")
            opp_base = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                **model_kwargs,
            )
            
            if config.use_lora:
                if config.use_4bit:
                    opp_base = prepare_model_for_kbit_training(opp_base)
                
                lora_config = LoraConfig(
                    r=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                opp_base = get_peft_model(opp_base, lora_config)
                print("Opponent model:")
                opp_base.print_trainable_parameters()
        
        opp_device = next(opp_base.parameters()).device
        hidden_size = opp_base.config.hidden_size
        self.opponent_model = PolicyModelWithValueHead(opp_base, hidden_size, device=opp_device)
        
        # Opponent optimizer
        self.opponent_optimizer = AdamW(
            self.opponent_model.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
        )
        
        # Opponent reference model (shared with agent's reference)
        self.opponent_ref_model = self.ref_model
        
        # Create LLM opponent wrapper
        self.llm_opponent = LLMOpponent(
            model=self.opponent_model,
            ref_model=self.opponent_ref_model,
            tokenizer=self.tokenizer,
            config=config,
            env_config=self.env_config,
            moral_type=config.opponent_moral_type,
        )
        
        # Set fixed opponent to None
        self.opponent = None
    
    def rollout_episode(self) -> Tuple[List[Experience], List[Experience]]:
        """
        Run one episode and collect experiences.
        Returns experiences for both agent and opponent (if LLM vs LLM).
        """
        config = self.config
        env = IPDEnv(self.env_config)
        obs = env.reset(random_initial_state=True)
        
        if self.llm_opponent is not None:
            self.llm_opponent.reset()
        elif self.opponent is not None:
            self.opponent.reset()
        
        agent_experiences = []
        opponent_experiences = []
        
        self.model.eval()
        
        # Get opponent's previous action from history
        history = obs.get("history", [])
        opp_prev = history[-1][1] if history else None
        
        # Build prompt for agent
        prompt = build_ipd_prompt_with_chat_template(
            obs, self.tokenizer, self.env_config, ("action1", "action2"),
            player_perspective="agent"
        )
        
        # Tokenize
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
        ).to(self.device)
        prompt_length = encoded["input_ids"].shape[1]
        
        # Generate agent action
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=config.max_new_tokens,
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        full_ids = output_ids
        attention_mask = torch.ones_like(full_ids)
        
        # Compute log probs and value for agent
        with torch.no_grad():
            log_prob, value = compute_log_probs(
                self.model, full_ids, attention_mask, prompt_length
            )
            ref_log_prob, _ = compute_log_probs(
                self.ref_model, full_ids, attention_mask, prompt_length
            )
        
        # Decode completion
        completion = self.tokenizer.decode(
            output_ids[0, prompt_length:],
            skip_special_tokens=True
        ).strip()
        
        # Parse agent action
        agent_action, is_legal = extract_action_from_completion(
            completion, ("action1", "action2")
        )
        
        # Get opponent action
        if self.llm_opponent is not None:
            # LLM vs LLM: get action from LLM opponent
            opp_action, opp_exp = self.llm_opponent.act(obs)
        else:
            # Fixed opponent
            opp_action = self.opponent.act(obs)
            opp_exp = None
        
        # Execute step in environment
        if is_legal:
            _, (r_agent, r_opp), _, _ = env.step(agent_action, opp_action)
        else:
            r_agent, r_opp = 0.0, 0.0
        
        # Compute agent reward
        reward = compute_moral_reward(
            config.moral_type,
            agent_action,
            opp_prev,
            r_agent,
            r_opp,
            is_legal,
            config.xi,
            config.illegal_penalty,
        )
        
        if config.reward_shaping and is_legal:
            reward += config.valid_action_bonus
        
        # Create agent experience
        agent_exp = Experience(
            prompt=prompt,
            completion=completion,
            input_ids=full_ids.cpu(),
            prompt_length=prompt_length,
            reward=reward,
            value=value.item(),
            log_prob=log_prob.item(),
            ref_log_prob=ref_log_prob.item(),
            action=agent_action,
            opponent_prev_action=opp_prev,
            is_legal=is_legal,
            player_id=0,
        )
        agent_experiences.append(agent_exp)
        
        # Finalize opponent experience if LLM vs LLM
        if self.llm_opponent is not None:
            self.llm_opponent.finalize_experience(
                agent_action=agent_action,
                agent_payoff=r_agent,
                opponent_payoff=r_opp,
            )
            opponent_experiences = self.llm_opponent.get_experiences()
        
        return agent_experiences, opponent_experiences
    
    def collect_batch(self) -> Tuple[List[Experience], List[Experience]]:
        """Collect a batch of experiences."""
        all_agent_experiences = []
        all_opponent_experiences = []
        
        for _ in range(self.config.batch_size):
            agent_exps, opp_exps = self.rollout_episode()
            all_agent_experiences.extend(agent_exps)
            all_opponent_experiences.extend(opp_exps)
        
        return all_agent_experiences, all_opponent_experiences
    
    def compute_advantages(self, experiences: List[Experience]) -> List[Experience]:
        """Compute advantages using GAE."""
        config = self.config
        
        raw_rewards = np.array([e.reward for e in experiences])
        values = np.array([e.value for e in experiences])
        
        rewards = raw_rewards * config.reward_scale
        
        if config.normalize_rewards and len(rewards) > 1:
            reward_std = rewards.std()
            if reward_std > 1e-6:
                rewards = (rewards - rewards.mean()) / (reward_std + 1e-8)
            else:
                rewards = rewards - rewards.mean()
        
        advantages = rewards - values
        returns = rewards
        
        if config.normalize_advantages and len(advantages) > 1:
            adv_std = advantages.std()
            if adv_std > 1e-6:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
        
        for i, exp in enumerate(experiences):
            exp.advantage = float(advantages[i])
            exp.returns = float(returns[i])
            exp.reward = float(rewards[i])
        
        return experiences
    
    def ppo_update(
        self,
        experiences: List[Experience],
        model: PolicyModelWithValueHead,
        optimizer: AdamW,
    ) -> Dict[str, float]:
        """Perform PPO update for a given model."""
        config = self.config
        model.train()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        num_updates = 0
        
        for epoch in range(config.ppo_epochs):
            indices = np.random.permutation(len(experiences))
            
            for idx in indices:
                exp = experiences[idx]
                
                if abs(exp.advantage) < 1e-8:
                    continue
                
                input_ids = exp.input_ids.to(self.device)
                attention_mask = torch.ones_like(input_ids)
                
                outputs, value = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                log_prob, _ = compute_log_probs(
                    model, input_ids, attention_mask, exp.prompt_length
                )
                
                gen_logits = logits[:, exp.prompt_length-1:-1, :]
                gen_probs = F.softmax(gen_logits, dim=-1)
                gen_log_probs = F.log_softmax(gen_logits, dim=-1)
                entropy = -(gen_probs * gen_log_probs).sum(dim=-1).mean()
                
                old_log_prob = torch.tensor(exp.log_prob, device=self.device, dtype=log_prob.dtype)
                old_value = torch.tensor(exp.value, device=self.device, dtype=value.dtype)
                advantage = torch.tensor(exp.advantage, device=self.device, dtype=log_prob.dtype)
                returns = torch.tensor(exp.returns, device=self.device, dtype=value.dtype)
                
                log_ratio = log_prob - old_log_prob
                ratio = torch.exp(log_ratio)
                
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * advantage
                policy_loss = -torch.min(surr1, surr2)
                
                value_clipped = old_value + torch.clamp(
                    value - old_value, -config.clip_ratio, config.clip_ratio
                )
                value_loss1 = (value - returns) ** 2
                value_loss2 = (value_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(value_loss1, value_loss2)
                
                approx_kl = 0.5 * (old_log_prob - log_prob) ** 2
                
                loss = (policy_loss 
                        + config.vf_coef * value_loss 
                        + config.kl_coef * approx_kl
                        - config.entropy_coef * entropy)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_kl += approx_kl.item()
                total_entropy += entropy.item()
                num_updates += 1
            
            if num_updates > 0:
                avg_kl = total_kl / num_updates
                if config.target_kl and avg_kl > config.target_kl * 1.5:
                    break
        
        n = max(num_updates, 1)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "kl": total_kl / n,
            "entropy": total_entropy / n,
        }
    
    def train(self):
        """Main training loop."""
        config = self.config
        
        # Create output path
        if config.llm_vs_llm:
            output_path = os.path.join(
                config.output_dir,
                f"llm_vs_llm_{config.moral_type}_vs_{config.opponent_moral_type}"
            )
        else:
            output_path = os.path.join(
                config.output_dir,
                f"{config.moral_type}_vs_{config.opponent_type}"
            )
        os.makedirs(output_path, exist_ok=True)
        
        print(f"\n{'='*60}")
        if config.llm_vs_llm:
            print(f"Training: LLM ({config.moral_type}) vs LLM ({config.opponent_moral_type})")
        else:
            print(f"Training: {config.moral_type} vs {config.opponent_type}")
        print(f"Episodes: {config.num_episodes}, Batch size: {config.batch_size}")
        print(f"{'='*60}\n")
        
        for episode in range(1, config.num_episodes + 1):
            # Collect experiences
            agent_experiences, opponent_experiences = self.collect_batch()
            
            # Store raw rewards
            raw_agent_rewards = [e.reward for e in agent_experiences]
            raw_opp_rewards = [e.reward for e in opponent_experiences] if opponent_experiences else []
            
            # Compute advantages for agent
            agent_experiences = self.compute_advantages(agent_experiences)
            
            # PPO update for agent
            agent_stats = self.ppo_update(agent_experiences, self.model, self.optimizer)
            
            # PPO update for opponent if LLM vs LLM
            opponent_stats = {}
            if config.llm_vs_llm and opponent_experiences:
                opponent_experiences = self.compute_advantages(opponent_experiences)
                opponent_stats = self.ppo_update(
                    opponent_experiences, 
                    self.opponent_model, 
                    self.opponent_optimizer
                )
            
            # Calculate episode stats
            agent_coop_rate = sum(1 for e in agent_experiences if e.action == "action1") / len(agent_experiences)
            agent_illegal_rate = sum(1 for e in agent_experiences if not e.is_legal) / len(agent_experiences)
            
            self.reward_history.extend(raw_agent_rewards)
            
            stats = {
                "episode": episode,
                "agent_mean_reward": float(np.mean(raw_agent_rewards)),
                "agent_std_reward": float(np.std(raw_agent_rewards)),
                "agent_cooperation_rate": agent_coop_rate,
                "agent_illegal_rate": agent_illegal_rate,
                "agent_policy_loss": agent_stats["policy_loss"],
                "agent_value_loss": agent_stats["value_loss"],
                "agent_kl": agent_stats["kl"],
                "agent_entropy": agent_stats["entropy"],
            }
            
            # Add opponent stats if LLM vs LLM
            if config.llm_vs_llm and opponent_experiences:
                opp_coop_rate = sum(1 for e in opponent_experiences if e.action == "action1") / len(opponent_experiences)
                opp_illegal_rate = sum(1 for e in opponent_experiences if not e.is_legal) / len(opponent_experiences)
                
                self.opponent_reward_history.extend(raw_opp_rewards)
                
                stats.update({
                    "opponent_mean_reward": float(np.mean(raw_opp_rewards)),
                    "opponent_std_reward": float(np.std(raw_opp_rewards)),
                    "opponent_cooperation_rate": opp_coop_rate,
                    "opponent_illegal_rate": opp_illegal_rate,
                    "opponent_policy_loss": opponent_stats.get("policy_loss", 0),
                    "opponent_value_loss": opponent_stats.get("value_loss", 0),
                    "opponent_kl": opponent_stats.get("kl", 0),
                    "opponent_entropy": opponent_stats.get("entropy", 0),
                })
            
            self.stats_history.append(stats)
            
            # Logging
            if episode % config.log_every == 0:
                if config.llm_vs_llm:
                    print(
                        f"Ep {episode:4d}/{config.num_episodes} | "
                        f"Agent R: {stats['agent_mean_reward']:+.2f} Coop: {agent_coop_rate:.0%} | "
                        f"Opp R: {stats.get('opponent_mean_reward', 0):+.2f} Coop: {stats.get('opponent_cooperation_rate', 0):.0%} | "
                        f"KL: {agent_stats['kl']:.4f}"
                    )
                else:
                    print(
                        f"Ep {episode:4d}/{config.num_episodes} | "
                        f"R: {stats['agent_mean_reward']:+.2f} (±{stats['agent_std_reward']:.2f}) | "
                        f"Coop: {agent_coop_rate:.0%} | "
                        f"Ill: {agent_illegal_rate:.0%} | "
                        f"Ent: {agent_stats['entropy']:.2f} | "
                        f"KL: {agent_stats['kl']:.4f}"
                    )
            
            # Save checkpoint
            if episode % config.save_every == 0:
                ckpt_path = os.path.join(output_path, f"agent_checkpoint_{episode}")
                self.model.save_pretrained(ckpt_path)
                self.tokenizer.save_pretrained(ckpt_path)
                
                if config.llm_vs_llm:
                    opp_ckpt_path = os.path.join(output_path, f"opponent_checkpoint_{episode}")
                    self.opponent_model.save_pretrained(opp_ckpt_path)
                    self.tokenizer.save_pretrained(opp_ckpt_path)
        
        # Save final models
        final_agent_path = os.path.join(output_path, "agent_final")
        self.model.save_pretrained(final_agent_path)
        self.tokenizer.save_pretrained(final_agent_path)
        
        if config.llm_vs_llm:
            final_opp_path = os.path.join(output_path, "opponent_final")
            self.opponent_model.save_pretrained(final_opp_path)
            self.tokenizer.save_pretrained(final_opp_path)
        
        # Save stats
        stats_path = os.path.join(output_path, "training_stats.json")
        with open(stats_path, "w") as f:
            json.dump(self.stats_history, f, indent=2)
        
        # Save config
        config_path = os.path.join(output_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(vars(config), f, indent=2)
        
        print(f"\nTraining complete!")
        print(f"Agent model saved to {final_agent_path}")
        if config.llm_vs_llm:
            print(f"Opponent model saved to {final_opp_path}")
        print(f"Stats saved to {stats_path}")
        
        return self.stats_history


def main():
    parser = argparse.ArgumentParser(
        description="PPO Training for Moral Alignment with LLM vs LLM support"
    )
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--use_4bit", action="store_true")
    
    # Game arguments
    parser.add_argument("--moral_type", type=str, default="utilitarian",
                       choices=["game", "deontological", "utilitarian", "game+deontological"])
    parser.add_argument("--opponent_type", type=str, default="tft",
                       choices=["tft", "always_cooperate", "always_defect", "random", "llm"])
    
    # LLM vs LLM arguments
    parser.add_argument("--llm_vs_llm", action="store_true",
                       help="Enable LLM vs LLM training mode")
    parser.add_argument("--opponent_moral_type", type=str, default="game",
                       choices=["game", "deontological", "utilitarian", "game+deontological"],
                       help="Moral type for opponent LLM (only used in LLM vs LLM mode)")
    parser.add_argument("--shared_base_model", action="store_true", default=True,
                       help="Share base model between agent and opponent (only LoRA differs)")
    
    # Training arguments
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=100)
    
    args = parser.parse_args()
    
    # If opponent_type is "llm", enable LLM vs LLM mode
    if args.opponent_type == "llm":
        args.llm_vs_llm = True
    
    config = PPOConfig(
        model_name=args.model_name,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
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
        output_dir=args.output_dir,
        seed=args.seed,
        log_every=args.log_every,
        save_every=args.save_every,
    )
    
    trainer = MoralPPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()