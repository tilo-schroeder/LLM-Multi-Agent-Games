"""
Visualization utilities for moral alignment experiments.

Generates plots similar to those in the paper:
- Figure 3: Action types during training (action | opponent's previous action)
- Figure 5: Moral regret across games
- Figure 6: Action type distribution at test time
- Training curves: reward, cooperation, KL, policy loss
"""

import os
import json
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import defaultdict


# Color scheme similar to paper
COLORS = {
    "C|C": "#2ecc71",      # Green - Cooperate given opponent cooperated
    "C|D": "#3498db",      # Blue - Cooperate given opponent defected
    "D|C": "#e74c3c",      # Red - Defect given opponent cooperated
    "D|D": "#f39c12",      # Orange - Defect given opponent defected
    "illegal|C": "#9b59b6", # Purple
    "illegal|D": "#95a5a6", # Gray
}

# Short labels for legend
ACTION_LABELS = {
    "C|C": "C|C (Coop after Coop)",
    "C|D": "C|D (Coop after Defect)",
    "D|C": "D|C (Defect after Coop)",
    "D|D": "D|D (Defect after Defect)",
    "illegal|C": "illegal|C",
    "illegal|D": "illegal|D",
}

GAME_COLORS = {
    "ipd": "#1f77b4",
    "stag_hunt": "#ff7f0e",
    "chicken": "#2ca02c",
    "bach_stravinsky": "#d62728",
    "defective_coordination": "#9467bd",
}


def get_action_type(agent_action: str, opponent_prev_action: Optional[str]) -> str:
    """
    Determine the action type string based on agent's action and opponent's previous action.
    
    Returns: One of "C|C", "C|D", "D|C", "D|D", "illegal|C", "illegal|D"
    """
    # Map action names to C/D
    if agent_action == "action1":
        agent_symbol = "C"
    elif agent_action == "action2":
        agent_symbol = "D"
    else:
        agent_symbol = "illegal"
    
    if opponent_prev_action is None:
        # No previous action - treat as if opponent cooperated (common convention)
        opp_symbol = "C"
    elif opponent_prev_action == "action1":
        opp_symbol = "C"
    else:
        opp_symbol = "D"
    
    return f"{agent_symbol}|{opp_symbol}"


def plot_action_types_over_time(
    stats: List[Dict],
    window_size: int = 10,
    output_path: Optional[str] = None,
    title: str = "LLM's actions during fine-tuning",
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot action type distribution over training episodes (similar to Figure 3 in paper).
    
    This shows stacked area/bar chart of action types (action | opponent's prev action)
    over the course of training.
    
    Expected stats format:
    Each entry should have 'action_type_counts' dict with counts for each action type,
    OR 'action_types' list of action type strings for that episode.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    episodes = [s["episode"] for s in stats]
    action_types = ["C|C", "C|D", "D|C", "D|D", "illegal|C", "illegal|D"]
    
    # Extract action type counts from stats
    type_counts_over_time = {at: [] for at in action_types}
    
    for s in stats:
        if "action_type_counts" in s:
            # Direct counts provided
            counts = s["action_type_counts"]
            total = sum(counts.values()) if counts else 1
            for at in action_types:
                type_counts_over_time[at].append(counts.get(at, 0) / max(total, 1))
        elif "action_types" in s:
            # List of action types provided
            at_list = s["action_types"]
            total = len(at_list) if at_list else 1
            for at in action_types:
                count = sum(1 for x in at_list if x == at)
                type_counts_over_time[at].append(count / max(total, 1))
        else:
            # Fallback: use cooperation_rate and illegal_rate to estimate
            coop_rate = s.get("cooperation_rate", 0.5)
            illegal_rate = s.get("illegal_rate", 0)
            legal_rate = 1 - illegal_rate
            
            # Distribute evenly between C|C, C|D for cooperation
            # and D|C, D|D for defection (rough approximation)
            type_counts_over_time["C|C"].append(coop_rate * legal_rate * 0.5)
            type_counts_over_time["C|D"].append(coop_rate * legal_rate * 0.5)
            type_counts_over_time["D|C"].append((1 - coop_rate) * legal_rate * 0.5)
            type_counts_over_time["D|D"].append((1 - coop_rate) * legal_rate * 0.5)
            type_counts_over_time["illegal|C"].append(illegal_rate * 0.5)
            type_counts_over_time["illegal|D"].append(illegal_rate * 0.5)
    
    # Apply moving average for smoothing
    def moving_avg(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    if len(episodes) > window_size:
        episodes_smooth = episodes[window_size-1:]
        smoothed = {at: moving_avg(type_counts_over_time[at], window_size) 
                   for at in action_types}
    else:
        episodes_smooth = episodes
        smoothed = type_counts_over_time
    
    # Create stacked area plot
    y_stack = np.zeros(len(episodes_smooth))
    
    for at in action_types:
        y_values = np.array(smoothed[at])
        ax.fill_between(episodes_smooth, y_stack, y_stack + y_values, 
                       label=at, color=COLORS[at], alpha=0.8)
        y_stack = y_stack + y_values
    
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Proportion of Actions", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(min(episodes_smooth), max(episodes_smooth))
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close()
    return fig


def plot_action_types_stacked_bar(
    stats: List[Dict],
    bin_size: int = 50,
    output_path: Optional[str] = None,
    title: str = "LLM's actions during fine-tuning",
    figsize: Tuple[int, int] = (12, 6),
):
    """
    Plot action type distribution as stacked bars over training episodes.
    Groups episodes into bins for clearer visualization.
    
    This matches Figure 3 style from the paper more closely.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    action_types = ["C|C", "C|D", "D|C", "D|D", "illegal|C", "illegal|D"]
    
    # Group stats into bins
    n_episodes = len(stats)
    n_bins = max(1, n_episodes // bin_size)
    
    bin_counts = {at: [] for at in action_types}
    bin_labels = []
    
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = min((i + 1) * bin_size, n_episodes)
        bin_stats = stats[start_idx:end_idx]
        
        # Aggregate counts for this bin
        total_counts = defaultdict(int)
        for s in bin_stats:
            if "action_type_counts" in s:
                for at, count in s["action_type_counts"].items():
                    total_counts[at] += count
            elif "action_types" in s:
                for at in s["action_types"]:
                    total_counts[at] += 1
        
        total = sum(total_counts.values()) if total_counts else 1
        for at in action_types:
            bin_counts[at].append(total_counts.get(at, 0) / max(total, 1))
        
        # Label for this bin
        start_ep = bin_stats[0]["episode"] if bin_stats else start_idx
        end_ep = bin_stats[-1]["episode"] if bin_stats else end_idx
        bin_labels.append(f"{start_ep}-{end_ep}")
    
    # Create stacked bar chart
    x = np.arange(n_bins)
    width = 0.8
    
    bottom = np.zeros(n_bins)
    for at in action_types:
        values = np.array(bin_counts[at])
        ax.bar(x, values, width, bottom=bottom, label=at, color=COLORS[at])
        bottom += values
    
    ax.set_xlabel("Episode Range", fontsize=12)
    ax.set_ylabel("Proportion of Actions", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x[::max(1, n_bins//10)])  # Show ~10 labels
    ax.set_xticklabels([bin_labels[i] for i in range(0, n_bins, max(1, n_bins//10))], 
                       rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close()
    return fig


def plot_training_actions(
    stats: List[Dict],
    window_size: int = 10,
    output_path: Optional[str] = None,
    title: str = "LLM's actions during fine-tuning",
):
    """
    Plot action types over training episodes (similar to Figure 3).
    Simple line plot version showing cooperation and illegal rates.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    episodes = [s["episode"] for s in stats]
    coop_rates = [s.get("cooperation_rate", s.get("agent_cooperation_rate", 0)) for s in stats]
    illegal_rates = [s.get("illegal_rate", s.get("agent_illegal_rate", 0)) for s in stats]
    
    def moving_avg(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    if len(episodes) > window_size:
        episodes_ma = episodes[window_size-1:]
        coop_ma = moving_avg(coop_rates, window_size)
        illegal_ma = moving_avg(illegal_rates, window_size)
        ax.plot(episodes_ma, coop_ma, label="Cooperation rate", color=COLORS["C|C"], linewidth=2)
        ax.plot(episodes_ma, illegal_ma, label="Illegal rate", color=COLORS["illegal|C"], linewidth=2)
    else:
        ax.plot(episodes, coop_rates, label="Cooperation rate", color=COLORS["C|C"], linewidth=2)
        ax.plot(episodes, illegal_rates, label="Illegal rate", color=COLORS["illegal|C"], linewidth=2)
    
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close()
    return fig


def plot_action_distribution(
    results: Dict,
    output_path: Optional[str] = None,
    title: str = "Action choices on iterated matrix games",
):
    """
    Plot action type distribution across games (similar to Figure 6).
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    games = [g for g in results.keys() 
             if g not in ["deontological_regret", "utilitarian_regret"]]
    
    action_types = ["C|C", "C|D", "D|C", "D|D", "illegal|C", "illegal|D"]
    
    x = np.arange(len(games))
    width = 0.12
    
    for i, action_type in enumerate(action_types):
        counts = []
        for game in games:
            action_counts = results[game].get("action_type_counts", {})
            total = sum(action_counts.values())
            if total > 0:
                counts.append(action_counts.get(action_type, 0) / total * 100)
            else:
                counts.append(0)
        
        offset = (i - len(action_types)/2 + 0.5) * width
        ax.bar(x + offset, counts, width, label=action_type, color=COLORS[action_type])
    
    ax.set_xlabel("Game", fontsize=12)
    ax.set_ylabel("Percentage of actions", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([results[g].get("game_name", g) for g in games], rotation=45, ha="right")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close()
    return fig


def plot_moral_regret(
    results: Dict,
    moral_type: str = "deontological",
    output_path: Optional[str] = None,
):
    """
    Plot moral regret across games (similar to Figure 5).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    regret_key = f"{moral_type}_regret"
    if regret_key not in results:
        print(f"No {moral_type} regret data found")
        return None
    
    regrets = results[regret_key]
    games = list(regrets.keys())
    values = [regrets[g] for g in games]
    colors = [GAME_COLORS.get(g, "#333333") for g in games]
    
    x = np.arange(len(games))
    ax.bar(x, values, color=colors)
    
    ax.set_xlabel("Game", fontsize=12)
    ax.set_ylabel(f"{moral_type.capitalize()} Moral Regret", fontsize=12)
    ax.set_title(f"Test time performance ({moral_type.capitalize()} regret)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("_", " ").title() for g in games], rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close()
    return fig


def plot_reward_curve(
    stats: List[Dict],
    window_size: int = 10,
    output_path: Optional[str] = None,
    title: str = "Moral Reward During Training",
    figsize: Tuple[int, int] = (10, 5),
):
    """
    Plot moral reward over training episodes.
    
    Handles multiple possible key names for reward in stats.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    episodes = [s["episode"] for s in stats]
    
    # Try different possible key names for reward
    reward_keys = [
        "mean_moral_reward",
        "mean_reward", 
        "agent_mean_reward",
        "reward",
        "avg_reward",
    ]
    
    rewards = None
    used_key = None
    for key in reward_keys:
        if key in stats[0]:
            rewards = [s[key] for s in stats]
            used_key = key
            break
    
    if rewards is None:
        print("Warning: No reward data found in stats. Tried keys:", reward_keys)
        print("Available keys:", list(stats[0].keys()) if stats else "No stats")
        ax.text(0.5, 0.5, "No reward data available", 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return fig
    
    # Apply moving average
    def moving_avg(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    if len(episodes) > window_size:
        episodes_ma = episodes[window_size-1:]
        rewards_ma = moving_avg(rewards, window_size)
    else:
        episodes_ma = episodes
        rewards_ma = rewards
    
    # Also plot raw values with transparency
    ax.plot(episodes, rewards, alpha=0.3, color="#2ecc71", linewidth=0.5)
    ax.plot(episodes_ma, rewards_ma, color="#2ecc71", linewidth=2, label=f"Reward (smoothed)")
    
    # Add std deviation if available
    std_key = used_key.replace("mean", "std") if "mean" in used_key else None
    if std_key and std_key in stats[0]:
        stds = [s[std_key] for s in stats]
        rewards_arr = np.array(rewards)
        stds_arr = np.array(stds)
        ax.fill_between(episodes, rewards_arr - stds_arr, rewards_arr + stds_arr,
                       alpha=0.2, color="#2ecc71")
    
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Mean Moral Reward", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close()
    return fig


def plot_training_curves(
    stats: List[Dict],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
):
    """
    Plot training curves: reward, cooperation rate, KL divergence, policy loss.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    episodes = [s["episode"] for s in stats]
    
    # Helper to find the right key
    def get_values(stats, possible_keys, default=None):
        for key in possible_keys:
            if key in stats[0]:
                return [s.get(key, default) for s in stats], key
        return None, None
    
    # Reward (check multiple possible keys)
    reward_keys = ["mean_moral_reward", "mean_reward", "agent_mean_reward", "reward"]
    rewards, reward_key = get_values(stats, reward_keys)
    if rewards:
        axes[0, 0].plot(episodes, rewards, color="#2ecc71", linewidth=1.5)
        axes[0, 0].set_ylabel("Mean Reward", fontsize=11)
    else:
        axes[0, 0].text(0.5, 0.5, "No reward data", ha='center', va='center',
                       transform=axes[0, 0].transAxes)
    axes[0, 0].set_xlabel("Episode", fontsize=11)
    axes[0, 0].set_title("Reward Learning Curve", fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cooperation rate
    coop_keys = ["cooperation_rate", "agent_cooperation_rate", "coop_rate"]
    coop, _ = get_values(stats, coop_keys)
    illegal_keys = ["illegal_rate", "agent_illegal_rate"]
    illegal, _ = get_values(stats, illegal_keys)
    
    if coop:
        axes[0, 1].plot(episodes, coop, color="#3498db", label="Cooperation", linewidth=1.5)
    if illegal:
        axes[0, 1].plot(episodes, illegal, color="#e74c3c", label="Illegal", linewidth=1.5)
    axes[0, 1].set_xlabel("Episode", fontsize=11)
    axes[0, 1].set_ylabel("Rate", fontsize=11)
    axes[0, 1].set_title("Cooperation & Illegal Rates", fontsize=12)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # KL divergence
    kl_keys = ["kl", "agent_kl", "kl_div"]
    kl, _ = get_values(stats, kl_keys)
    if kl:
        axes[1, 0].plot(episodes, kl, color="#9b59b6", linewidth=1.5)
    else:
        axes[1, 0].text(0.5, 0.5, "No KL data", ha='center', va='center',
                       transform=axes[1, 0].transAxes)
    axes[1, 0].set_xlabel("Episode", fontsize=11)
    axes[1, 0].set_ylabel("KL Divergence", fontsize=11)
    axes[1, 0].set_title("KL Divergence", fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Policy loss
    ploss_keys = ["policy_loss", "agent_policy_loss", "pg_loss"]
    ploss, _ = get_values(stats, ploss_keys)
    if ploss:
        axes[1, 1].plot(episodes, ploss, color="#f39c12", linewidth=1.5)
    else:
        axes[1, 1].text(0.5, 0.5, "No policy loss data", ha='center', va='center',
                       transform=axes[1, 1].transAxes)
    axes[1, 1].set_xlabel("Episode", fontsize=11)
    axes[1, 1].set_ylabel("Policy Loss", fontsize=11)
    axes[1, 1].set_title("Policy Loss", fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close()
    return fig


def plot_llm_vs_llm_training(
    stats: List[Dict],
    window_size: int = 10,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
):
    """
    Plot training curves for LLM vs LLM training, showing both agent and opponent.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    episodes = [s["episode"] for s in stats]
    
    def moving_avg(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Agent reward
    if "agent_mean_reward" in stats[0]:
        rewards = [s["agent_mean_reward"] for s in stats]
        if len(episodes) > window_size:
            axes[0, 0].plot(episodes[window_size-1:], moving_avg(rewards, window_size), 
                           color="#2ecc71", linewidth=2, label="Agent")
        else:
            axes[0, 0].plot(episodes, rewards, color="#2ecc71", linewidth=2, label="Agent")
    
    # Opponent reward
    if "opponent_mean_reward" in stats[0]:
        opp_rewards = [s["opponent_mean_reward"] for s in stats]
        if len(episodes) > window_size:
            axes[0, 0].plot(episodes[window_size-1:], moving_avg(opp_rewards, window_size),
                           color="#e74c3c", linewidth=2, label="Opponent")
        else:
            axes[0, 0].plot(episodes, opp_rewards, color="#e74c3c", linewidth=2, label="Opponent")
    
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Mean Reward")
    axes[0, 0].set_title("Rewards")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Agent cooperation rate
    if "agent_cooperation_rate" in stats[0]:
        coop = [s["agent_cooperation_rate"] for s in stats]
        if len(episodes) > window_size:
            axes[0, 1].plot(episodes[window_size-1:], moving_avg(coop, window_size),
                           color="#2ecc71", linewidth=2, label="Agent")
        else:
            axes[0, 1].plot(episodes, coop, color="#2ecc71", linewidth=2, label="Agent")
    
    # Opponent cooperation rate
    if "opponent_cooperation_rate" in stats[0]:
        opp_coop = [s["opponent_cooperation_rate"] for s in stats]
        if len(episodes) > window_size:
            axes[0, 1].plot(episodes[window_size-1:], moving_avg(opp_coop, window_size),
                           color="#e74c3c", linewidth=2, label="Opponent")
        else:
            axes[0, 1].plot(episodes, opp_coop, color="#e74c3c", linewidth=2, label="Opponent")
    
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Cooperation Rate")
    axes[0, 1].set_title("Cooperation Rates")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # Agent illegal rate
    if "agent_illegal_rate" in stats[0]:
        illegal = [s["agent_illegal_rate"] for s in stats]
        if len(episodes) > window_size:
            axes[0, 2].plot(episodes[window_size-1:], moving_avg(illegal, window_size),
                           color="#2ecc71", linewidth=2, label="Agent")
        else:
            axes[0, 2].plot(episodes, illegal, color="#2ecc71", linewidth=2, label="Agent")
    
    if "opponent_illegal_rate" in stats[0]:
        opp_illegal = [s["opponent_illegal_rate"] for s in stats]
        if len(episodes) > window_size:
            axes[0, 2].plot(episodes[window_size-1:], moving_avg(opp_illegal, window_size),
                           color="#e74c3c", linewidth=2, label="Opponent")
        else:
            axes[0, 2].plot(episodes, opp_illegal, color="#e74c3c", linewidth=2, label="Opponent")
    
    axes[0, 2].set_xlabel("Episode")
    axes[0, 2].set_ylabel("Illegal Rate")
    axes[0, 2].set_title("Illegal Rates")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim(0, 1)
    
    # Agent KL
    if "agent_kl" in stats[0]:
        kl = [s["agent_kl"] for s in stats]
        axes[1, 0].plot(episodes, kl, color="#2ecc71", linewidth=1.5, label="Agent")
    if "opponent_kl" in stats[0]:
        opp_kl = [s["opponent_kl"] for s in stats]
        axes[1, 0].plot(episodes, opp_kl, color="#e74c3c", linewidth=1.5, label="Opponent")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("KL Divergence")
    axes[1, 0].set_title("KL Divergence")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Agent policy loss
    if "agent_policy_loss" in stats[0]:
        ploss = [s["agent_policy_loss"] for s in stats]
        axes[1, 1].plot(episodes, ploss, color="#2ecc71", linewidth=1.5, label="Agent")
    if "opponent_policy_loss" in stats[0]:
        opp_ploss = [s["opponent_policy_loss"] for s in stats]
        axes[1, 1].plot(episodes, opp_ploss, color="#e74c3c", linewidth=1.5, label="Opponent")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Policy Loss")
    axes[1, 1].set_title("Policy Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Entropy
    if "agent_entropy" in stats[0]:
        ent = [s["agent_entropy"] for s in stats]
        axes[1, 2].plot(episodes, ent, color="#2ecc71", linewidth=1.5, label="Agent")
    if "opponent_entropy" in stats[0]:
        opp_ent = [s["opponent_entropy"] for s in stats]
        axes[1, 2].plot(episodes, opp_ent, color="#e74c3c", linewidth=1.5, label="Opponent")
    axes[1, 2].set_xlabel("Episode")
    axes[1, 2].set_ylabel("Entropy")
    axes[1, 2].set_title("Policy Entropy")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close()
    return fig


def compare_models_regret(
    results_dict: Dict[str, Dict],
    moral_type: str = "deontological",
    output_path: Optional[str] = None,
):
    """
    Compare moral regret across different fine-tuned models (like Figure 5).
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_names = list(results_dict.keys())
    games = ["ipd", "stag_hunt", "chicken", "bach_stravinsky", "defective_coordination"]
    
    x = np.arange(len(model_names))
    width = 0.15
    
    for i, game in enumerate(games):
        regrets = []
        for model in model_names:
            regret_key = f"{moral_type}_regret"
            if regret_key in results_dict[model]:
                regrets.append(results_dict[model][regret_key].get(game, 0))
            else:
                regrets.append(0)
        
        offset = (i - len(games)/2 + 0.5) * width
        ax.bar(x + offset, regrets, width, 
               label=game.replace("_", " ").title(),
               color=GAME_COLORS[game])
    
    ax.set_xlabel("Fine-tuned Model", fontsize=12)
    ax.set_ylabel(f"{moral_type.capitalize()} Moral Regret", fontsize=12)
    ax.set_title("Test time performance on five matrix games", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close()
    return fig


def create_all_plots(
    stats: List[Dict],
    eval_results: Optional[Dict] = None,
    output_dir: str = "./plots",
    is_llm_vs_llm: bool = False,
):
    """
    Create all visualization plots from training stats and evaluation results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating plots in {output_dir}...")
    
    # Training curves
    if is_llm_vs_llm:
        plot_llm_vs_llm_training(
            stats, 
            output_path=os.path.join(output_dir, "llm_vs_llm_training.png")
        )
    else:
        plot_training_curves(
            stats,
            output_path=os.path.join(output_dir, "training_curves.png")
        )
    
    # Reward curve (separate, larger)
    plot_reward_curve(
        stats,
        output_path=os.path.join(output_dir, "reward_curve.png")
    )
    
    # Action types over time (if data available)
    if stats and ("action_type_counts" in stats[0] or "action_types" in stats[0]):
        plot_action_types_over_time(
            stats,
            output_path=os.path.join(output_dir, "action_types_over_time.png")
        )
        plot_action_types_stacked_bar(
            stats,
            output_path=os.path.join(output_dir, "action_types_stacked.png")
        )
    else:
        # Fallback to simple cooperation plot
        plot_training_actions(
            stats,
            output_path=os.path.join(output_dir, "cooperation_rate.png")
        )
    
    # Evaluation plots
    if eval_results:
        plot_action_distribution(
            eval_results,
            output_path=os.path.join(output_dir, "action_distribution.png")
        )
        plot_moral_regret(
            eval_results, 
            "deontological",
            output_path=os.path.join(output_dir, "deontological_regret.png")
        )
        plot_moral_regret(
            eval_results,
            "utilitarian", 
            output_path=os.path.join(output_dir, "utilitarian_regret.png")
        )
    
    print("Done generating plots!")


def main():
    """Generate visualizations from saved results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualization for moral alignment experiments")
    parser.add_argument("--stats_file", type=str, help="Training stats JSON file")
    parser.add_argument("--eval_file", type=str, help="Evaluation results JSON file")
    parser.add_argument("--output_dir", type=str, default="./plots", help="Output directory for plots")
    parser.add_argument("--llm_vs_llm", action="store_true", help="Use LLM vs LLM plotting style")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    stats = None
    eval_results = None
    
    if args.stats_file:
        print(f"Loading stats from {args.stats_file}")
        with open(args.stats_file) as f:
            stats = json.load(f)
        print(f"Loaded {len(stats)} training episodes")
        
        # Print available keys for debugging
        if stats:
            print(f"Available keys in stats: {list(stats[0].keys())}")
    
    if args.eval_file:
        print(f"Loading evaluation results from {args.eval_file}")
        with open(args.eval_file) as f:
            eval_results = json.load(f)
    
    if stats:
        create_all_plots(
            stats,
            eval_results,
            args.output_dir,
            is_llm_vs_llm=args.llm_vs_llm
        )
    elif eval_results:
        # Only evaluation results
        plot_action_distribution(
            eval_results, 
            os.path.join(args.output_dir, "action_distribution.png")
        )
        plot_moral_regret(
            eval_results, 
            "deontological", 
            os.path.join(args.output_dir, "deontological_regret.png")
        )
        plot_moral_regret(
            eval_results, 
            "utilitarian", 
            os.path.join(args.output_dir, "utilitarian_regret.png")
        )
    else:
        print("No input files provided. Use --stats_file and/or --eval_file")


if __name__ == "__main__":
    main()