import os, json, argparse
from training.grpo_pd import TrainConfig, LLMPolicy, evaluate
from envs.repeated_pd import Config as EnvConfig

ap = argparse.ArgumentParser()
ap.add_argument("--model", type=str, required=True)
ap.add_argument("--adapter_dir", type=str, default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rounds", type=int, default=20)
ap.add_argument("--T", type=float, default=5.0)
ap.add_argument("--R", type=float, default=3.0)
ap.add_argument("--P", type=float, default=1.0)
ap.add_argument("--S", type=float, default=0.0)
ap.add_argument("--noise", type=float, default=0.0)
ap.add_argument("--episodes", type=int, default=50)
ap.add_argument("--outdir", type=str, required=True)
args = ap.parse_args()

env_cfg = EnvConfig(rounds=args.rounds, T=args.T, R=args.R, P=args.P, S=args.S, action_error=args.noise, seed=args.seed)
tcfg = TrainConfig(model_name=args.model, do_sample=False, temperature=0.0)
policy = LLMPolicy(tcfg, adapter_dir=args.adapter_dir)
metrics = evaluate(env_cfg, policy, episodes=args.episodes, seed=args.seed, log_dir=args.outdir)

with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(metrics)