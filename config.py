from dataclasses import dataclass

@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    learning_rate: float = 5e-6

    # PPO batch configuration:
    # batch_size     = number of (query, response, reward) samples per PPO update
    # mini_batch_size = per-device mini-batch for PPO inner epochs
    batch_size: int = 64
    mini_batch_size: int = 16

    # Number of PPO epochs per update
    grpo_epochs: int = 1  # now used as ppo_epochs

    seed: int = 0

    # How many new tokens we allow for an "action" completion
    # (1–2 is typical; keep small so PPO is focused on the choice token)
    max_new_tokens: int = 2

    do_sample: bool = True
    temperature: float = 0.7

    # If True, use a "social" reward (approx. utilitarian):
    #   reward = (r_self + r_other) / 2
    # If False, use selfish (game) reward:
    #   reward = r_self
    social_reward: bool = True

    # Negative reward added if the model outputs something
    # that does not parse as a legal action
    illegal_penalty: float = -1.0

    # (Not used anymore, but kept for CLI compatibility)
    num_generations: int = 8