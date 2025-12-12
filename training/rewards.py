from __future__ import annotations

from typing import List, Optional


def compute_moral_reward(
    moral_type: str,
    agent_action: str,
    opponent_prev_actions: Optional[List[str]],
    agent_payoff: float,
    all_payoffs: List[float],
    is_legal: bool,
    xi: float = 3.0,
    illegal_penalty: float = -6.0,
) -> float:
    if not is_legal:
        return illegal_penalty

    if moral_type == "game":
        return float(agent_payoff)

    elif moral_type == "deontological":
        # Example norm: penalize choosing hare when others previously chose stag (scaled by count).
        if opponent_prev_actions and agent_action == "hare":
            stag_count = sum(1 for a in opponent_prev_actions if a == "stag")
            if stag_count > 0:
                return -xi * float(stag_count)
        return 0.0

    elif moral_type == "utilitarian":
        return float(sum(all_payoffs))

    elif moral_type == "game+deontological":
        reward = float(agent_payoff)
        if opponent_prev_actions and agent_action == "hare":
            stag_count = sum(1 for a in opponent_prev_actions if a == "stag")
            if stag_count > 0:
                reward -= xi * float(stag_count)
        return reward

    else:
        raise ValueError(f"Unknown moral type: {moral_type}")