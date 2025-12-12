#!/usr/bin/env python3
"""
Plot evaluation outputs from eval_matrix_games.py

Input JSON format expected (per file):
{
  "checkpoint_dir": ...,
  "action_names": [...],
  "games": {
     "auction": {
        "num_players": 2,
        "deon_regret_mean": ...,
        "util_regret_mean": ...,
        "deon_regret_norm_mean": ...,
        "util_regret_norm_mean": ...,
        "action_rate_prosocial": ...,
        "illegal_rate": ...,
        "by_prev_prosocial_count_per_seed": [
            {"0": {"prosocial":..., "selfish":..., "illegal":...}, "1": {...}, ...},
            ... (one dict per seed)
        ]
     },
     ...
  }
}

Produces:
- regret_deon(.png/.pdf)
- regret_util(.png/.pdf)
- rates(.png/.pdf)
- conditional_prosocial_<game>(.png/.pdf) for each game
"""

import os
import re
import glob
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# Utilities
# -------------------------

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def default_label(path: str, data: Dict[str, Any]) -> str:
    """
    Try to create a compact label.
    Preference:
      1) filename stem
      2) last 2 components of checkpoint_dir
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    ckpt = str(data.get("checkpoint_dir", ""))
    if ckpt:
        parts = [p for p in ckpt.replace("\\", "/").split("/") if p]
        tail = "/".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else "")
        if tail:
            return stem if stem != "eval_results" else tail
    return stem

def get_metric(game_dict: Dict[str, Any], key_raw: str, key_norm: str, use_norm: bool) -> Optional[float]:
    v = game_dict.get(key_norm if use_norm else key_raw, None)
    if v is None:
        # fall back
        v = game_dict.get(key_raw, None)
    return None if v is None else float(v)

def ordered_games_present(runs: List[Dict[str, Any]], preferred=("auction", "diners", "public_goods")) -> List[str]:
    present = set()
    for r in runs:
        present |= set(r["data"].get("games", {}).keys())
    out = [g for g in preferred if g in present]
    out += [g for g in sorted(present) if g not in out]
    return out


# -------------------------
# Plot: grouped bar helper
# -------------------------

def grouped_bar_plot(
    out_path_base: str,
    title: str,
    xlabel: str,
    ylabel: str,
    groups: List[str],     # e.g. games on x-axis
    series: List[str],     # e.g. run labels
    values: np.ndarray,    # shape: [len(series), len(groups)]
):
    """
    Saves PNG+PDF.
    """
    ensure_dir(os.path.dirname(out_path_base))

    n_series, n_groups = values.shape
    x = np.arange(n_groups)
    width = 0.8 / max(1, n_series)

    plt.figure(figsize=(max(7, 1.4 * n_groups), 4.5))
    for i in range(n_series):
        plt.bar(x + (i - (n_series - 1)/2) * width, values[i], width, label=series[i])

    plt.xticks(x, groups)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False, ncol=min(4, max(1, n_series)))
    plt.tight_layout()

    plt.savefig(out_path_base + ".png", dpi=200)
    plt.savefig(out_path_base + ".pdf")
    plt.close()


# -------------------------
# Plot: conditional prosocial curve
# -------------------------

def conditional_curve_plot(
    out_path_base: str,
    title: str,
    xlabel: str,
    ylabel: str,
    k_values: List[int],
    mean: np.ndarray,
    std: np.ndarray,
):
    ensure_dir(os.path.dirname(out_path_base))

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(k_values, mean, marker="o")
    # error band
    plt.fill_between(k_values, mean - std, mean + std, alpha=0.2)
    plt.ylim(-0.02, 1.02)
    plt.xticks(k_values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    plt.savefig(out_path_base + ".png", dpi=200)
    plt.savefig(out_path_base + ".pdf")
    plt.close()


# -------------------------
# Main plotting routine
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="One or more eval JSON files (globs ok), e.g. eval_outputs/*.json")
    ap.add_argument("--out_dir", type=str, default="./eval_plots")
    ap.add_argument("--use_norm", action="store_true",
                    help="Use *_regret_norm_mean if present")
    ap.add_argument("--title_prefix", type=str, default="",
                    help="Prefix appended to plot titles")
    ap.add_argument("--labels", nargs="*", default=[],
                    help="Optional labels aligned with inputs after glob expansion. If omitted, auto-label.")
    ap.add_argument("--games", nargs="*", default=[],
                    help="Subset of games to plot, e.g. auction diners public_goods")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    # expand globs
    paths: List[str] = []
    for p in args.inputs:
        matches = glob.glob(p)
        if matches:
            paths.extend(matches)
        else:
            paths.append(p)
    paths = sorted(set(paths))

    runs = []
    for p in paths:
        data = load_json(p)
        runs.append({"path": p, "data": data, "label": ""})

    # labels
    if args.labels and len(args.labels) == len(runs):
        for r, lab in zip(runs, args.labels):
            r["label"] = lab
    else:
        for r in runs:
            r["label"] = default_label(r["path"], r["data"])

    # games order
    games = ordered_games_present(runs)
    if args.games:
        games = [g for g in games if g in set(args.games)]

    prefix = (args.title_prefix + " ") if args.title_prefix else ""
    use_norm = bool(args.use_norm)

    # -------------------------
    # 1) Regret bar plots
    # -------------------------
    series_labels = [r["label"] for r in runs]

    deon_vals = np.zeros((len(runs), len(games)), dtype=np.float32)
    util_vals = np.zeros((len(runs), len(games)), dtype=np.float32)

    for i, r in enumerate(runs):
        gdict = r["data"].get("games", {})
        for j, g in enumerate(games):
            gd = gdict.get(g, {})
            deon_vals[i, j] = get_metric(gd, "deon_regret_mean", "deon_regret_norm_mean", use_norm) or 0.0
            util_vals[i, j] = get_metric(gd, "util_regret_mean", "util_regret_norm_mean", use_norm) or 0.0

    grouped_bar_plot(
        out_path_base=os.path.join(args.out_dir, "regret_deontological" + ("_norm" if use_norm else "")),
        title=f"{prefix}Deontological Regret" + (" (normalized)" if use_norm else ""),
        xlabel="Game",
        ylabel="Regret",
        groups=games,
        series=series_labels,
        values=deon_vals,
    )

    grouped_bar_plot(
        out_path_base=os.path.join(args.out_dir, "regret_utilitarian" + ("_norm" if use_norm else "")),
        title=f"{prefix}Utilitarian Regret" + (" (normalized)" if use_norm else ""),
        xlabel="Game",
        ylabel="Regret",
        groups=games,
        series=series_labels,
        values=util_vals,
    )

    # -------------------------
    # 2) Prosocial + illegal rate bars
    # -------------------------
    pros_vals = np.zeros((len(runs), len(games)), dtype=np.float32)
    ill_vals = np.zeros((len(runs), len(games)), dtype=np.float32)

    for i, r in enumerate(runs):
        gdict = r["data"].get("games", {})
        for j, g in enumerate(games):
            gd = gdict.get(g, {})
            pros_vals[i, j] = float(gd.get("action_rate_prosocial", 0.0) or 0.0)
            ill_vals[i, j] = float(gd.get("illegal_rate", 0.0) or 0.0)

    grouped_bar_plot(
        out_path_base=os.path.join(args.out_dir, "rate_prosocial"),
        title=f"{prefix}Prosocial Action Rate",
        xlabel="Game",
        ylabel="Rate",
        groups=games,
        series=series_labels,
        values=pros_vals,
    )

    grouped_bar_plot(
        out_path_base=os.path.join(args.out_dir, "rate_illegal"),
        title=f"{prefix}Illegal Output Rate",
        xlabel="Game",
        ylabel="Rate",
        groups=games,
        series=series_labels,
        values=ill_vals,
    )

    # -------------------------
    # 3) Conditional prosociality curves (per game, per run)
    # -------------------------
    # We create one plot per game, with one line per run.
    for g in games:
        # Find max N for x-axis (based on each run's game num_players)
        max_k = 0
        for r in runs:
            gd = r["data"].get("games", {}).get(g, {})
            n = int(gd.get("num_players", 0) or 0)
            max_k = max(max_k, max(0, n - 1))
        k_values = list(range(max_k + 1))

        plt.figure(figsize=(6.8, 4.8))
        for r in runs:
            gd = r["data"].get("games", {}).get(g, {})
            per_seed = gd.get("by_prev_prosocial_count_per_seed", [])
            if not per_seed:
                continue

            # For each seed dict, build a vector p(prosocial | k)
            mats = []
            for sdict in per_seed:
                vec = []
                for k in k_values:
                    entry = sdict.get(str(k), None)
                    vec.append(float(entry.get("prosocial", 0.0)) if entry else 0.0)
                mats.append(vec)

            M = np.array(mats, dtype=np.float32)
            mean = M.mean(axis=0)
            std = M.std(axis=0)

            plt.plot(k_values, mean, marker="o", label=r["label"])
            plt.fill_between(k_values, mean - std, mean + std, alpha=0.15)

        plt.ylim(-0.02, 1.02)
        plt.xticks(k_values)
        plt.xlabel("# opponents prosocial last round (k)")
        plt.ylabel("P(prosocial action)")
        plt.title(f"{prefix}Conditional Prosociality — {g}")
        plt.legend(frameon=False, ncol=1)
        plt.tight_layout()

        base = os.path.join(args.out_dir, f"conditional_prosocial_{g}")
        plt.savefig(base + ".png", dpi=200)
        plt.savefig(base + ".pdf")
        plt.close()

    print(f"Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()