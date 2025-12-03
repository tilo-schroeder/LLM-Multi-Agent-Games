from dataclasses import dataclass

@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    learning_rate: float = 5e-6
    batch_size: int = 64                # global = per_device * grad_accum * n_gpus
    mini_batch_size: int = 16           # per_device_train_batch_size
    grpo_epochs: int = 1                # mapped to num_train_epochs in GRPO
    seed: int = 0
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.7
    social_reward: bool = True          # if True: social welfare, else selfish
    num_generations: int = 8            # GRPO group size per prompt

    # NEW: reward shaping – bias towards cooperative transcripts
    # 0.0 = off, >0 encourages more Cs in the episode.
    coop_bonus: float = 0.2