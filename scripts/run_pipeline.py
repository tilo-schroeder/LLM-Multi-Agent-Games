from __future__ import annotations
import os, json, argparse, datetime
from training.grpo_pd import TrainConfig, LLMPolicy, evaluate, grpo_train
from envs.repeated_pd import Config as EnvConfig


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
    ap.add_argument("--max_new_tokens", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--social_reward", action="store_true", default=True)
    ap.add_argument("--no_social_reward", dest="social_reward", action="store_false")

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

    # IO
    ap.add_argument("--outdir", type=str, default="runs")
    ap.add_argument("--run_name", type=str, default=None)

    args = ap.parse_args()

    # Directories
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{ts}_{args.model.replace('/', '_')}"
    run_dir = os.path.join(args.outdir, run_name)
    base_dir = os.path.join(run_dir, "baseline_eval")
    train_dir = os.path.join(run_dir, "train")
    post_dir = os.path.join(run_dir, "post_eval")
    os.makedirs(run_dir, exist_ok=True)

    # Configs
    env_cfg = EnvConfig(
        rounds=args.rounds, T=args.T, R=args.R, P=args.P, S=args.S,
        action_error=args.noise, seed=args.seed
    )
    tcfg = TrainConfig(
        model_name=args.model,
        learning_rate=args.lr,
        batch_size=args.batch,
        mini_batch_size=args.minibatch,
        ppo_epochs=args.epochs,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        social_reward=args.social_reward,
        num_generations=args.num_generations,
    )

    # 1) Baseline evaluation
    base_policy = LLMPolicy(tcfg)
    base_metrics = evaluate(env_cfg, base_policy, episodes=args.eval_episodes, seed=args.seed, log_dir=base_dir)

    # 2) Training (GRPO)
    policy, _ = grpo_train(env_cfg, tcfg, total_episodes=args.train_episodes, save_dir=train_dir)

    # 3) Post-training evaluation
    tuned_policy = LLMPolicy(tcfg, adapter_dir=train_dir)
    post_metrics = evaluate(env_cfg, tuned_policy, episodes=args.eval_episodes, seed=args.seed, log_dir=post_dir)

    # Manifest + comparison
    save_json(os.path.join(run_dir, "env_config.json"), env_cfg.__dict__)
    save_json(os.path.join(run_dir, "train_config.json"), tcfg.__dict__)
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