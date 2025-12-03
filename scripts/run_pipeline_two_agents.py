from __future__ import annotations
import os
import json
import argparse
import datetime

from envs.repeated_pd import Config as EnvConfig
from config import TrainConfig
from policy.llm_policy import LLMPolicy
from training.two_agent_grpo import (
    grpo_train_single_agent,
    evaluate_two_agents,
)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    ap = argparse.ArgumentParser()

    # Model/train
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--minibatch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--social_reward", action="store_true", default=True)
    ap.add_argument("--no_social_reward", dest="social_reward", action="store_false")
    ap.add_argument(
        "--coop_bonus",
        type=float,
        default=0.2,
        help="Shaping weight for cooperation rate in episode reward (0.0 = off).",
    )

    # Env
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--T", type=float, default=5.0)
    ap.add_argument("--R", type=float, default=3.0)
    ap.add_argument("--P", type=float, default=1.0)
    ap.add_argument("--S", type=float, default=0.0)
    ap.add_argument("--noise", type=float, default=0.0)

    # Eval/train sizes
    ap.add_argument("--eval_episodes", type=int, default=50)
    ap.add_argument("--train_episodes", type=int, default=200)

    # Alternating BR iters
    ap.add_argument("--alt_iters", type=int, default=2)

    # IO
    ap.add_argument("--outdir", type=str, default="runs_two_agents")
    ap.add_argument("--run_name", type=str, default=None)

    args = ap.parse_args()

    # Directories
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{ts}_{args.model.replace('/', '_')}_two_agents"
    run_dir = os.path.join(args.outdir, run_name)
    base_dir = os.path.join(run_dir, "baseline_eval")
    post_dir = os.path.join(run_dir, "post_eval")
    agent0_dir = os.path.join(run_dir, "agent0")
    agent1_dir = os.path.join(run_dir, "agent1")
    os.makedirs(run_dir, exist_ok=True)

    # Configs
    env_cfg = EnvConfig(
        rounds=args.rounds,
        T=args.T,
        R=args.R,
        P=args.P,
        S=args.S,
        action_error=args.noise,
        seed=args.seed,
    )

    # same hyperparams for both agents; you can diverge later if you want
    base_cfg0 = TrainConfig(
        model_name=args.model,
        learning_rate=args.lr,
        batch_size=args.batch,
        mini_batch_size=args.minibatch,
        grpo_epochs=args.epochs,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        social_reward=args.social_reward,
        num_generations=args.num_generations,
        coop_bonus=args.coop_bonus,
    )
    base_cfg1 = TrainConfig(**base_cfg0.__dict__)

    # 1) Baseline eval: both agents = same base model (no adapters)
    base_policy0 = LLMPolicy(base_cfg0)
    base_policy1 = LLMPolicy(base_cfg1)
    base_metrics = evaluate_two_agents(
        env_cfg,
        base_policy0,
        base_policy1,
        episodes=args.eval_episodes,
        seed=args.seed,
        log_dir=base_dir,
    )

    # 2) Alternating best-response GRPO for agent_0 and agent_1
    current_policy0 = base_policy0
    current_policy1 = base_policy1

    for k in range(args.alt_iters):
        print(f"=== Alternating iteration {k} / {args.alt_iters} ===")

        # Train agent_0 vs fixed agent_1
        print("Training agent_0...")
        learner0, _ = grpo_train_single_agent(
            env_cfg,
            base_cfg0,
            role="agent_0",
            opponent_policy=current_policy1,
            episodes_per_iter=args.train_episodes,
            outer_iters=1,
            save_dir=agent0_dir,
        )
        current_policy0 = learner0

        # Train agent_1 vs fixed agent_0
        print("Training agent_1...")
        learner1, _ = grpo_train_single_agent(
            env_cfg,
            base_cfg1,
            role="agent_1",
            opponent_policy=current_policy0,
            episodes_per_iter=args.train_episodes,
            outer_iters=1,
            save_dir=agent1_dir,
        )
        current_policy1 = learner1

    # 3) Post-training evaluation with two separate adapters
    tuned_policy0 = LLMPolicy(base_cfg0, adapter_dir=agent0_dir)
    tuned_policy1 = LLMPolicy(base_cfg1, adapter_dir=agent1_dir)
    post_metrics = evaluate_two_agents(
        env_cfg,
        tuned_policy0,
        tuned_policy1,
        episodes=args.eval_episodes,
        seed=args.seed,
        log_dir=post_dir,
    )

    # Save configs + comparison
    save_json(os.path.join(run_dir, "env_config.json"), env_cfg.__dict__)
    save_json(os.path.join(run_dir, "train_config_agent0.json"), base_cfg0.__dict__)
    save_json(os.path.join(run_dir, "train_config_agent1.json"), base_cfg1.__dict__)
    save_json(os.path.join(run_dir, "baseline_metrics.json"), base_metrics)
    save_json(os.path.join(run_dir, "post_metrics.json"), post_metrics)

    diff = {
        "avg_payoff_agent0_gain": post_metrics["avg_payoff_agent0"] - base_metrics["avg_payoff_agent0"],
        "avg_payoff_agent1_gain": post_metrics["avg_payoff_agent1"] - base_metrics["avg_payoff_agent1"],
        "coop_rate_gain": post_metrics["cooperation_rate"] - base_metrics["cooperation_rate"],
    }
    save_json(os.path.join(run_dir, "comparison.json"), diff)

    print("=== Baseline metrics ===")
    print(json.dumps(base_metrics, indent=2))
    print("=== Post-training metrics ===")
    print(json.dumps(post_metrics, indent=2))
    print("=== Gains ===")
    print(json.dumps(diff, indent=2))
    print(f"\nRun directory: {run_dir}")


if __name__ == "__main__":
    main()