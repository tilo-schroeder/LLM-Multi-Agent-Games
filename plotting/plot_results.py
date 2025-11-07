from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _safe_read_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def plot_episodes(run_dir: str, phase: str) -> str | None:
    """
    phase in {"baseline_eval","post_eval"}
    Produces two-axis line plot: returns (two lines) and coop rate.
    """
    ep_path = os.path.join(run_dir, phase, "episodes.csv")
    df = _safe_read_csv(ep_path)
    if df.empty:
        return None

    df = df.rename(
        columns={
            "ret_agent0": "ret0",
            "ret_agent1": "ret1",
            "coop_rate": "coop",
        }
    )
    outdir = _ensure_dir(os.path.join(run_dir, "plots"))
    out_png = os.path.join(outdir, f"episodes_{'baseline' if phase=='baseline_eval' else 'post'}.png")

    # primary y: returns; secondary y: coop rate
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(df["episode"], df["ret0"], label="agent0 return")
    ax1.plot(df["episode"], df["ret1"], label="agent1 return")
    ax1.set_xlabel("episode")
    ax1.set_ylabel("return")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(df["episode"], df["coop"], linestyle="--", label="coop rate")
    ax2.set_ylabel("cooperation rate")
    ax2.set_ylim(0, 1)

    # one combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_metrics_comparison(run_dir: str) -> str | None:
    base = _safe_read_json(os.path.join(run_dir, "baseline_metrics.json"))
    post = _safe_read_json(os.path.join(run_dir, "post_metrics.json"))
    if not base or not post:
        # fallback to nested files
        base = _safe_read_json(os.path.join(run_dir, "baseline_eval", "metrics.json"))
        post = _safe_read_json(os.path.join(run_dir, "post_eval", "metrics.json"))
    if not base or not post:
        return None

    metrics = [
        ("avg_payoff_agent0", "Agent 0 payoff"),
        ("avg_payoff_agent1", "Agent 1 payoff"),
        ("cooperation_rate", "Cooperation rate"),
    ]
    rows = []
    for key, label in metrics:
        rows.append({"metric": label, "phase": "baseline", "value": base.get(key, 0.0)})
        rows.append({"metric": label, "phase": "post", "value": post.get(key, 0.0)})
    df = pd.DataFrame(rows)

    outdir = _ensure_dir(os.path.join(run_dir, "plots"))
    out_png = os.path.join(outdir, "metrics_comparison.png")

    # grouped bar chart
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    order = [m[1] for m in metrics]
    df["metric"] = pd.Categorical(df["metric"], categories=order, ordered=True)
    df_p = df.pivot(index="metric", columns="phase", values="value").loc[order]

    x = range(len(df_p))
    width = 0.4
    ax.bar([i - width / 2 for i in x], df_p["baseline"], width, label="baseline")
    ax.bar([i + width / 2 for i in x], df_p["post"], width, label="post")
    ax.set_xticks(list(x))
    ax.set_xticklabels(order, rotation=0)
    ax.set_ylabel("value")
    ax.set_title("Baseline vs Post-training")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")

    # show cooperation as 0..1
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_training_curves(run_dir: str) -> List[str]:
    """
    Auto-detect numeric columns in train/train_history.csv and plot them individually.
    Also maps common TRL keys to friendlier names when present.
    """
    csv_path = os.path.join(run_dir, "train", "train_history.csv")
    df = _safe_read_csv(csv_path)
    outputs: List[str] = []
    if df.empty:
        return outputs

    # clean column names
    df.columns = [c.strip() for c in df.columns]
    # step/x axis
    step_col = "step" if "step" in df.columns else ("global_step" if "global_step" in df.columns else None)
    if step_col is None:
        # fabricate a step index
        df["step"] = range(len(df))
        step_col = "step"

    # prefer these keys if available (common in TRL/GRPO logs)
    preferred = {
        "train/loss": "train_loss",
        "loss": "train_loss",
        "kl": "kl",
        "policy/kl": "kl",
        "rewards/mean": "reward_mean",
        "reward/mean": "reward_mean",
        "advantage/mean": "adv_mean",
        "advantages/mean": "adv_mean",
        "entropy": "entropy",
        "lr": "learning_rate",
        "learning_rate": "learning_rate",
    }

    # Build a map of found metrics -> pretty name
    found: Dict[str, str] = {}
    for k, pretty in preferred.items():
        if k in df.columns:
            found[k] = pretty

    # Add any other numeric columns (excluding obvious bookkeeping)
    ignore = {step_col, "epoch", "eval_loss", "eval_runtime", "eval_samples_per_second"}
    for col in df.columns:
        if col in ignore:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            # avoid duplicates if already mapped
            if col not in found:
                found[col] = col.replace("/", "_")

    outdir = _ensure_dir(os.path.join(run_dir, "plots"))
    for raw_key, pretty in found.items():
        try:
            s = df[[step_col, raw_key]].dropna()
            if s.empty:
                continue
            fig, ax = plt.subplots(figsize=(7.5, 4))
            ax.plot(s[step_col], s[raw_key])
            ax.set_xlabel(step_col)
            ax.set_ylabel(pretty)
            ax.set_title(f"Training curve: {pretty}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out_png = os.path.join(outdir, f"training_curve_{pretty}.png")
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
            outputs.append(out_png)
        except Exception:
            # Be robust to weird columns
            continue

    return outputs


def write_report(run_dir: str, imgs: Dict[str, str | None], extra_imgs: List[str]) -> str:
    outdir = _ensure_dir(os.path.join(run_dir, "plots"))
    rpt = os.path.join(outdir, "report.md")

    base = _safe_read_json(os.path.join(run_dir, "baseline_metrics.json")) or \
           _safe_read_json(os.path.join(run_dir, "baseline_eval", "metrics.json"))
    post = _safe_read_json(os.path.join(run_dir, "post_metrics.json")) or \
           _safe_read_json(os.path.join(run_dir, "post_eval", "metrics.json"))
    cmpf = _safe_read_json(os.path.join(run_dir, "comparison.json"))

    def fmt(m):
        if not m:
            return "`(missing)`"
        return (
            f"- Agent0 payoff: **{m.get('avg_payoff_agent0', 0):.3f}**\n"
            f"- Agent1 payoff: **{m.get('avg_payoff_agent1', 0):.3f}**\n"
            f"- Coop rate: **{m.get('cooperation_rate', 0)*100:.1f}%**\n"
        )

    with open(rpt, "w") as f:
        f.write("# PD GRPO – Run report\n\n")
        f.write("## Baseline metrics\n")
        f.write(fmt(base) + "\n")
        if imgs.get("baseline"):
            f.write(f"![Baseline episodes]({os.path.relpath(imgs['baseline'], outdir)})\n\n")

        f.write("## Post-training metrics\n")
        f.write(fmt(post) + "\n")
        if imgs.get("post"):
            f.write(f"![Post episodes]({os.path.relpath(imgs['post'], outdir)})\n\n")

        if imgs.get("cmp"):
            f.write("## Baseline vs Post\n")
            f.write(f"![Comparison]({os.path.relpath(imgs['cmp'], outdir)})\n\n")

        if cmpf:
            f.write("### Gains\n")
            f.write(
                f"- Δ Agent0 payoff: **{cmpf.get('avg_payoff_agent0_gain', 0):+.3f}**\n"
                f"- Δ Agent1 payoff: **{cmpf.get('avg_payoff_agent1_gain', 0):+.3f}**\n"
                f"- Δ Coop rate: **{cmpf.get('coop_rate_gain', 0)*100:+.1f}%**\n\n"
            )

        if extra_imgs:
            f.write("## Training curves\n")
            for p in extra_imgs:
                f.write(f"![{os.path.basename(p)}]({os.path.relpath(p, outdir)})\n\n")

    return rpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to a single run directory (created by run_pipeline.py)")
    args = ap.parse_args()

    # Episode plots
    base_img = plot_episodes(args.run_dir, "baseline_eval")
    post_img = plot_episodes(args.run_dir, "post_eval")
    cmp_img = plot_metrics_comparison(args.run_dir)

    # Training curves
    curves = plot_training_curves(args.run_dir)

    rpt = write_report(args.run_dir, {"baseline": base_img, "post": post_img, "cmp": cmp_img}, curves)

    print("Saved plots to:", os.path.join(args.run_dir, "plots"))
    print("Report:", rpt)


if __name__ == "__main__":
    main()