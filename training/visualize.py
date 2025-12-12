#!/usr/bin/env python3
"""
Visualize multi-player Stag Hunt PPO training stats.

Works with the NEW training_stats.json produced by the updated trainer, where
per-player keys look like:

  p0_mean_reward, p0_std_reward, p0_stag_rate, p0_kl, p0_policy_loss, p0_value_loss, p0_entropy, ...
  p1_mean_reward, ...

This script auto-detects which players exist in the stats and plots:
- Mean reward (per player)
- Stag rate (per player)
- KL (per player)
- Policy + value loss (per player)
- Entropy (per player)

Usage:
  python viz_stag_hunt.py --stats_file /path/to/training_stats.json --output_dir ./plots --window 25
"""

import os
import re
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def moving_average(x: List[float], window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x, kernel, mode="valid")


def detect_players(stats: List[Dict]) -> List[int]:
    if not stats:
        return []
    players = set()
    pat = re.compile(r"^p(\d+)_")
    for k in stats[0].keys():
        m = pat.match(k)
        if m:
            players.add(int(m.group(1)))
    return sorted(players)


def series(stats: List[Dict], key: str) -> Optional[List[float]]:
    if not stats or key not in stats[0]:
        return None
    return [float(s.get(key, np.nan)) for s in stats]


def detect_conditioned_categories(stats: List[Dict]) -> Dict[int, List[Tuple[str, str]]]:
    """
    Finds keys of the form: p{pid}_cat_{act}_prev_{prev}
    Returns {pid: [(act, prev), ...]}.
    """
    if not stats:
        return {}
    cats = {}
    pat = re.compile(r"^p(\d+)_cat_(.+?)_prev_(.+?)$")
    for k in stats[0].keys():
        m = pat.match(k)
        if m:
            pid = int(m.group(1))
            act = m.group(2)
            prev = m.group(3)
            cats.setdefault(pid, set()).add((act, prev))
    return {pid: sorted(list(s)) for pid, s in cats.items()}


def collect_conditioned_series(stats: List[Dict], pid: int) -> Dict[Tuple[str, str], List[float]]:
    """
    Returns {(act, prev): [values per episode]}
    """
    out = {}
    pat = re.compile(rf"^p{pid}_cat_(.+?)_prev_(.+?)$")
    # Build list from first row to keep stable ordering
    keys = []
    for k in stats[0].keys():
        m = pat.match(k)
        if m:
            keys.append((m.group(1), m.group(2), k))  # (act, prev, full_key)

    for act, prev, full_key in keys:
        out[(act, prev)] = [float(s.get(full_key, 0.0)) for s in stats]
    return out


def plot_conditioned_stack(
    episodes: List[int],
    cat_series: Dict[Tuple[str, str], List[float]],
    window: int,
    title: str,
    out_path: str,
    dpi: int,
):
    """
    Stacked area plot of episode fractions for each (act, prev) category.
    """
    if not cat_series:
        return

    # Optional smoothing
    def smooth(y: List[float]) -> Tuple[List[int], np.ndarray]:
        y_ma = moving_average(y, window)
        if len(y_ma) == len(y):
            return episodes, y_ma
        else:
            return episodes[window - 1 :], y_ma

    # Heuristic ordering to look like the paper: group by prev then action; keep illegal last
    acts = sorted({a for (a, _p) in cat_series.keys()})
    # Put illegal last if present
    if "illegal" in acts:
        acts = [a for a in acts if a != "illegal"] + ["illegal"]
    prevs = sorted({p for (_a, p) in cat_series.keys()})

    ordered = []
    for p in prevs:
        for a in acts:
            if (a, p) in cat_series:
                ordered.append((a, p))

    # Build stacked arrays (after smoothing)
    ep_plot = None
    Ys = []
    labels = []
    for (a, p) in ordered:
        ep_s, y_s = smooth(cat_series[(a, p)])
        if ep_plot is None:
            ep_plot = ep_s
        Ys.append(y_s)
        labels.append(f"{a} | prev={p}")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.stackplot(ep_plot, Ys, labels=labels, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Fraction of steps in episode")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncols=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_multi(
    ax,
    episodes: List[int],
    per_player: Dict[int, List[float]],
    window: int,
    title: str,
    ylabel: str,
):
    # raw (faint) + smoothed (bold)
    for pid, y in per_player.items():
        ax.plot(episodes, y, alpha=0.2, linewidth=0.75)

        y_ma = moving_average(y, window)
        if len(y_ma) == len(y):
            ep_ma = episodes
        else:
            ep_ma = episodes[window - 1 :]
        ax.plot(ep_ma, y_ma, linewidth=2.0, label=f"P{pid}")

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats_file", type=str, required=True, help="Path to training_stats.json")
    ap.add_argument("--output_dir", type=str, default="./plots", help="Where to save plots")
    ap.add_argument("--window", type=int, default=25, help="Moving average window (episodes)")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.stats_file, "r") as f:
        stats: List[Dict] = json.load(f)

    if not stats:
        raise SystemExit("stats_file is empty")

    players = detect_players(stats)
    if not players:
        raise SystemExit("No per-player keys found (expected keys like p0_mean_reward, p0_stag_rate, ...).")

    episodes = [int(s["episode"]) for s in stats]

    # Collect series per player (only if present)
    def collect(metric_suffix: str) -> Dict[int, List[float]]:
        out = {}
        for pid in players:
            k = f"p{pid}_{metric_suffix}"
            vals = series(stats, k)
            if vals is not None:
                out[pid] = vals
        return out

    reward = collect("mean_reward")
    stag = collect("stag_rate")
    kl = collect("kl")
    ploss = collect("policy_loss")
    vloss = collect("value_loss")
    ent = collect("entropy")

    # Dashboard figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    if reward:
        plot_multi(axes[0, 0], episodes, reward, args.window, "Mean reward", "Reward")
    else:
        axes[0, 0].text(0.5, 0.5, "No p*_mean_reward", ha="center", va="center", transform=axes[0, 0].transAxes)

    if stag:
        plot_multi(axes[0, 1], episodes, stag, args.window, "Stag rate", "Rate")
        axes[0, 1].set_ylim(0, 1)
    else:
        axes[0, 1].text(0.5, 0.5, "No p*_stag_rate", ha="center", va="center", transform=axes[0, 1].transAxes)

    if kl:
        plot_multi(axes[0, 2], episodes, kl, args.window, "KL", "KL")
    else:
        axes[0, 2].text(0.5, 0.5, "No p*_kl", ha="center", va="center", transform=axes[0, 2].transAxes)

    if ploss:
        plot_multi(axes[1, 0], episodes, ploss, args.window, "Policy loss", "Loss")
    else:
        axes[1, 0].text(0.5, 0.5, "No p*_policy_loss", ha="center", va="center", transform=axes[1, 0].transAxes)

    if vloss:
        plot_multi(axes[1, 1], episodes, vloss, args.window, "Value loss", "Loss")
    else:
        axes[1, 1].text(0.5, 0.5, "No p*_value_loss", ha="center", va="center", transform=axes[1, 1].transAxes)

    if ent:
        plot_multi(axes[1, 2], episodes, ent, args.window, "Entropy", "Entropy")
    else:
        axes[1, 2].text(0.5, 0.5, "No p*_entropy", ha="center", va="center", transform=axes[1, 2].transAxes)

    fig.suptitle("Multi-player Stag Hunt PPO training", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    out_dash = os.path.join(args.output_dir, "training_dashboard.png")
    fig.savefig(out_dash, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dash}")

    # Also save a couple of single-focus plots that are handy in papers
    def save_single(name: str, per_player: Dict[int, List[float]], title: str, ylabel: str, ylim: Optional[Tuple[float, float]] = None):
        if not per_player:
            return
        fig, ax = plt.subplots(figsize=(9, 4.5))
        plot_multi(ax, episodes, per_player, args.window, title, ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        fig.tight_layout()
        path = os.path.join(args.output_dir, name)
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    save_single("reward.png", reward, "Mean reward", "Reward")
    save_single("stag_rate.png", stag, "Stag rate", "Rate", ylim=(0, 1))

    cond = detect_conditioned_categories(stats)
    for pid in players:
        if pid not in cond:
            continue
        cat_series = collect_conditioned_series(stats, pid)
        out_path = os.path.join(args.output_dir, f"action_conditioned_p{pid}.png")
        plot_conditioned_stack(
            episodes=episodes,
            cat_series=cat_series,
            window=args.window,
            title=f"Action distribution conditioned on opponent previous move (P{pid})",
            out_path=out_path,
            dpi=args.dpi,
        )

if __name__ == "__main__":
    main()