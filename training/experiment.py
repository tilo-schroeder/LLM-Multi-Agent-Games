"""
Self-play GRPO-style finetuning on a 2-player repeated Stag Hunt game,
where the LLM is a local decision policy:

- Each decision: separate prompt per player ("You are Player 1...").
- Players do NOT see the other player's action in the same round.
- We collect (prompt, completion, episode reward) for each decision.
- Then apply a GRPO-style policy gradient:
    A_i = (r_i - mean(r)) / (std(r) + eps)
    loss = - E[ A_i * log pi(action | prompt) ]
"""

import torch
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

# =========================
# 1. Stag Hunt Environment
# =========================

@dataclass
class StagHuntConfig:
    num_players: int = 2
    num_rounds: int = 5
    R_stag_stag: float = 4.0
    R_hare_hare: float = 2.0
    R_stag_hare: float = 0.0
    R_hare_stag: float = 3.0


class StagHuntEnv:
    """Simple 2-player repeated Stag Hunt."""

    ACTION_STAG = "STAG"
    ACTION_HARE = "HARE"

    def __init__(self, config: StagHuntConfig):
        assert config.num_players == 2, "This env is implemented for 2 players."
        self.config = config
        self.reset()

    def reset(self):
        self.round = 0
        # history: list of tuples (a1, a2) for past rounds
        self.history: List[Tuple[str, str]] = []
        return self._get_obs()

    def _get_obs(self) -> Dict[str, Any]:
        return {
            "round": self.round,
            "history": list(self.history),
        }

    def step(self, action_p1: str, action_p2: str):
        assert action_p1 in (self.ACTION_STAG, self.ACTION_HARE)
        assert action_p2 in (self.ACTION_STAG, self.ACTION_HARE)

        self.history.append((action_p1, action_p2))
        self.round += 1

        c = self.config
        if action_p1 == self.ACTION_STAG and action_p2 == self.ACTION_STAG:
            r1 = c.R_stag_stag
            r2 = c.R_stag_stag
        elif action_p1 == self.ACTION_HARE and action_p2 == self.ACTION_HARE:
            r1 = c.R_hare_hare
            r2 = c.R_hare_hare
        elif action_p1 == self.ACTION_STAG and action_p2 == self.ACTION_HARE:
            r1 = c.R_stag_hare
            r2 = c.R_hare_stag
        else:  # action_p1 == HARE, action_p2 == STAG
            r1 = c.R_hare_stag
            r2 = c.R_stag_hare

        done = self.round >= c.num_rounds
        obs = self._get_obs()
        info = {}
        return obs, (r1, r2), done, info

    def compute_episode_returns(self) -> Tuple[float, float]:
        """Recompute total returns for each player from history."""
        c = self.config
        total_r1 = 0.0
        total_r2 = 0.0
        for a1, a2 in self.history:
            if a1 == self.ACTION_STAG and a2 == self.ACTION_STAG:
                r1 = c.R_stag_stag
                r2 = c.R_stag_stag
            elif a1 == self.ACTION_HARE and a2 == self.ACTION_HARE:
                r1 = c.R_hare_hare
                r2 = c.R_hare_hare
            elif a1 == self.ACTION_STAG and a2 == self.ACTION_HARE:
                r1 = c.R_stag_hare
                r2 = c.R_hare_stag
            else:  # HARE, STAG
                r1 = c.R_hare_stag
                r2 = c.R_stag_hare
            total_r1 += r1
            total_r2 += r2
        return total_r1, total_r2


# =======================================
# 2. Local decision prompting & parsing
# =======================================

def format_history_for_player(history: List[Tuple[str, str]]) -> str:
    """Format past rounds in a natural language way."""
    if not history:
        return "No previous rounds have been played.\n"
    lines = []
    for t, (a1, a2) in enumerate(history, start=1):
        lines.append(f"Round {t}: Player 1 chose {a1}, Player 2 chose {a2}.")
    return "\n".join(lines) + "\n"


def build_decision_prompt(
    player_id: int,
    env_obs: Dict[str, Any],
    config: StagHuntConfig,
    tokenizer,
) -> str:
    """
    Build a chat-formatted prompt for a single player decision,
    using the model's chat template.
    """
    assert player_id in (1, 2)
    round_idx = env_obs["round"] + 1  # 1-based
    history = env_obs["history"]
    history_text = format_history_for_player(history)

    system_msg = (
        f"You are playing as Player {player_id} "
        f"in a repeated 2-player Stag Hunt game."
    )

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

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def extract_action_from_completion(text: str) -> str:
    """
    Given a completion that ends with something like "ACTION: STAG",
    parse and return "STAG" or "HARE".
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in reversed(lines):  # search from the end
        if line.upper().startswith("ACTION:"):
            tail = line.split(":", 1)[1].strip().upper()
            if "STAG" in tail:
                return StagHuntEnv.ACTION_STAG
            if "HARE" in tail:
                return StagHuntEnv.ACTION_HARE
    # Fallback: default to HARE (safe action) if parsing fails
    return StagHuntEnv.ACTION_HARE


# ==================================
# 3. Episode rollout & data logging
# ==================================

@dataclass
class DecisionSample:
    episode_id: int
    player_id: int
    round_idx: int
    prompt: str
    completion: str
    # reward will be filled *after* episode ends:
    reward: float = 0.0


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 48,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a single completion for a local decision prompt."""
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if full_text.startswith(prompt):
        completion = full_text[len(prompt):]
    else:
        completion = full_text
    return completion


def rollout_episode(
    model,
    tokenizer,
    config: StagHuntConfig,
    device: torch.device,
    episode_id: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Tuple[List[DecisionSample], float, float]:
    """
    Run one episode of self-play:

    - For each round:
        - For each player (1, 2):
            - Build local prompt based on history so far.
            - Query the model once to get a completion.
            - Parse action from completion.
        - After both actions are chosen, step the environment.

    Returns:
        - list of DecisionSample (one per decision)
        - total reward of player 1
        - total reward of player 2
    """
    env = StagHuntEnv(config)
    env.reset()

    samples: List[DecisionSample] = []

    done = False
    while not done:
        obs = env._get_obs()
        round_idx = obs["round"] + 1

        # Collect actions for both players *without* revealing same-round actions.
        actions = {}

        for player_id in [1, 2]:
            prompt = build_decision_prompt(player_id, obs, config, tokenizer)
            completion = generate_completion(
                model, tokenizer, prompt, device,
                temperature=temperature, top_p=top_p,
            )
            action = extract_action_from_completion(completion)
            actions[player_id] = action

            samples.append(
                DecisionSample(
                    episode_id=episode_id,
                    player_id=player_id,
                    round_idx=round_idx,
                    prompt=prompt,
                    completion=completion,
                    reward=0.0,  # filled later
                )
            )

        # Now step the environment with both actions
        obs, (r1, r2), done, info = env.step(actions[1], actions[2])

    total_r1, total_r2 = env.compute_episode_returns()
    return samples, total_r1, total_r2


def collect_batch(
    model,
    tokenizer,
    config: StagHuntConfig,
    device: torch.device,
    num_episodes: int,
    reward_mode: str = "average",
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[DecisionSample]:
    """
    Collect a batch of episodes and attach a scalar reward to each decision.

    reward_mode:
      - "selfish_p1": each decision gets reward = total return of player 1
      - "selfish_p2": each decision gets reward = total return of player 2
      - "cooperative": reward = r1 + r2
      - "average": reward = 0.5 * (r1 + r2)
    """
    all_samples: List[DecisionSample] = []
    episode_rewards = {}

    for ep_id in range(num_episodes):
        print(f"  Rolling out episode {ep_id}...")
        samples, r1, r2 = rollout_episode(
            model, tokenizer, config, device,
            episode_id=ep_id,
            temperature=temperature,
            top_p=top_p,
        )
        all_samples.extend(samples)

        if ep_id < 2:
            print(f"    Episode {ep_id} total_r1={r1}, total_r2={r2}")
            rounds = []
            env = StagHuntEnv(config)
            env.reset()
            for s in samples:
                if s.episode_id == ep_id:
                    # re-parse actions from completion (this is crude but fine for debug)
                    a = extract_action_from_completion(s.completion)
                    print(f"Unparsed action: {s.completion}")
                    print("---"*30)
                    rounds.append((s.round_idx, s.player_id, a))
            print("    Parsed actions (round, player, action):")
            for triple in rounds:
                print("    ", triple)

        if reward_mode == "selfish_p1":
            R = float(r1)
        elif reward_mode == "selfish_p2":
            R = float(r2)
        elif reward_mode == "cooperative":
            R = float(r1 + r2)
        elif reward_mode == "average":
            R = 0.5 * float(r1 + r2)
        else:
            raise ValueError(f"Unknown reward_mode: {reward_mode}")

        episode_rewards[ep_id] = R

    # Fill in reward for each decision from its episode
    for s in all_samples:
        s.reward = episode_rewards[s.episode_id]

    return all_samples


# ===========================
# 4. GRPO-style RL training
# ===========================


def compute_logprob_for_sample(
    model,
    tokenizer,
    sample: DecisionSample,
    device: torch.device,
    max_length: int = 512,
) -> torch.Tensor:
    """
    Compute mean log p(completion | prompt) for a single DecisionSample.

    Returns:
        logprob: scalar tensor (requires_grad=True)
    """
    full_text = sample.prompt + sample.completion

    enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Figure out where the completion starts
    prompt_enc = tokenizer(
        sample.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)
    prompt_len = prompt_enc["input_ids"].shape[1]

    # Forward pass WITH grad
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits  # [1, seq_len, vocab]

    # Standard next-token shift
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_attn = attention_mask[:, 1:].contiguous()

    # Mask: only completion tokens (everything after prompt)
    completion_mask = torch.zeros_like(shift_attn)
    completion_mask[:, prompt_len - 1 :] = 1  # from last prompt token onward

    log_probs_all = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs_all.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)  # [1, seq_len]

    mask = shift_attn * completion_mask
    token_log_probs = token_log_probs * mask

    num_tokens = mask.sum().clamp(min=1)
    logprob = token_log_probs.sum() / num_tokens  # mean logprob over completion

    return logprob  # scalar with grad


def train_grpo_stag_hunt_local():
    # ---- Config ----
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stag_cfg = StagHuntConfig(
        num_players=2,
        num_rounds=5,
        R_stag_stag=4.0,
        R_hare_hare=2.0,
        R_stag_hare=0.0,
        R_hare_stag=3.0,
    )

    reward_mode = "average"  # "selfish_p1", "selfish_p2", "cooperative", "average"

    num_updates = 20      # gradient updates
    episodes_per_batch = 8
    lr = 1e-5
    max_grad_norm = 1.0

    # ---- Load model & tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)

    # Optional: a scheduler
    total_steps = num_updates
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for update in range(1, num_updates + 1):
        print(f"Running update {update}")
        # 1) Collect data under current policy (on-policy, no grad)
        samples = collect_batch(
            model=model,
            tokenizer=tokenizer,
            config=stag_cfg,
            device=device,
            num_episodes=episodes_per_batch,
            reward_mode=reward_mode,
            temperature=0.7,
            top_p=0.9,
        )

        # Rewards as CPU tensor
        rewards = torch.tensor([s.reward for s in samples], dtype=torch.float32)

        # 2) Compute GRPO-style advantages on CPU
        mean_r = rewards.mean()
        std_r = rewards.std(unbiased=False).clamp(min=1e-6)
        advantages = (rewards - mean_r) / std_r  # shape [N], on CPU

        # 3) Policy gradient update:
        #    loop over samples, do forward+backward one by one
        model.train()
        optimizer.zero_grad()

        total_loss = 0.0
        for i, sample in enumerate(samples):
            adv_i = advantages[i].to(device)  # scalar

            logprob_i = compute_logprob_for_sample(
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                device=device,
                max_length=512,
            )

            # Loss_i = - A_i * log pi
            loss_i = -adv_i * logprob_i
            loss_i.backward()  # backward for this sample only

            total_loss += loss_i.item()

        # Optional: clip gradients across all params
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()

        if update % 10 == 0:
            avg_reward = rewards.mean().item()
            std_reward = rewards.std().item()
            avg_loss = total_loss / len(samples)
            print(
                f"[Update {update}/{num_updates}] "
                f"Loss: {avg_loss:.4f} | "
                f"Mean reward: {avg_reward:.3f} | "
                f"Std reward: {std_reward:.3f} | "
                f"Num samples: {len(samples)}"
            )

    # Save fine-tuned model
    output_dir = "./stag_hunt_local_agent_grpo"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved fine-tuned model to {output_dir}")


# ================
# 5. Entry point
# ================

if __name__ == "__main__":
    train_grpo_stag_hunt_local()