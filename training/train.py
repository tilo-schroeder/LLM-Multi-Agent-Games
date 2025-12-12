from __future__ import annotations

import argparse
from typing import List

from .ppo import PPOConfig
from .trainer import MoralPPOTrainer


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
    parser.add_argument(
        "--moral_type",
        type=str,
        default="utilitarian",
        choices=["game", "deontological", "utilitarian", "game+deontological"],
    )
    parser.add_argument(
        "--opponent_moral_type",
        type=str,
        default="game",
        choices=["game", "deontological", "utilitarian", "game+deontological"],
    )
    parser.add_argument(
        "--player_moral_types",
        type=str,
        default="",
        help="Comma-separated moral types per player (length num_players). Example: utilitarian,game,game",
    )

    # Opponents (fixed-mode)
    parser.add_argument(
        "--opponent_type",
        type=str,
        default="copy_focal",
        choices=["always_stag", "always_hare", "random", "copy_focal", "llm"],
        help="In fixed-mode, used for players 1..N-1. If set to 'llm', enables llm_vs_llm.",
    )

    # Multi-agent mode
    parser.add_argument("--llm_vs_llm", action="store_true", help="If set, all players are LLM policies trained simultaneously.")

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

    trainer = MoralPPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()