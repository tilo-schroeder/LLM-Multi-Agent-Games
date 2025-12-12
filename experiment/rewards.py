"""
Moral Reward Functions for LLM Agent Fine-tuning

Based on Table 1 in the paper "Moral Alignment for LLM Agents" (ICLR 2025)

Reward types:
1. Game reward (selfish): R_t_M = R_t_Mgame
2. Deontological reward: -xi if defecting against a cooperator, 0 otherwise  
3. Utilitarian reward: R_t_Mgame + R_t_Ogame (collective payoff)
4. Game+Deontological: Game reward with deontological penalty
"""

from typing import Optional
from .ipd import IPDEnv


def compute_moral_reward(
    moral_type: str,
    agent_action: str,
    opponent_prev_action: Optional[str],
    agent_game_reward: float,
    opponent_game_reward: float,
    is_legal: bool,
    xi: float = 3.0,
    illegal_penalty: float = -6.0,
) -> float:
    """
    Compute moral reward based on the specified moral framework.
    
    Args:
        moral_type: One of 'game', 'deontological', 'utilitarian', 'game+deontological'
        agent_action: The agent's action (action1=C or action2=D)
        opponent_prev_action: Opponent's previous action (for deontological check)
        agent_game_reward: Agent's game payoff
        opponent_game_reward: Opponent's game payoff
        is_legal: Whether the agent produced a valid action token
        xi: Deontological penalty magnitude (default 3.0 as in paper)
        illegal_penalty: Penalty for illegal actions (default -6.0 as in paper)
    
    Returns:
        Moral reward value
    """
    # Illegal action penalty (from paper: R_illegal = -6)
    if not is_legal:
        return float(illegal_penalty)
    
    C = IPDEnv.C  # action1 = Cooperate
    D = IPDEnv.D  # action2 = Defect
    
    if moral_type == "game":
        # Selfish/rational agent: just game payoff
        return float(agent_game_reward)
    
    elif moral_type == "deontological":
        # Deontological norm: "do not defect against an opponent who previously cooperated"
        # R = -xi if (agent defects AND opponent previously cooperated), else 0
        if opponent_prev_action == C and agent_action == D:
            return float(-xi)
        return 0.0
    
    elif moral_type == "utilitarian":
        # Utilitarian: maximize collective welfare
        # R = R_agent + R_opponent
        return float(agent_game_reward + opponent_game_reward)
    
    elif moral_type == "game+deontological":
        # Multi-objective: game payoff with deontological penalty
        # R = R_game - xi (if violating norm), else R_game
        base_reward = float(agent_game_reward)
        if opponent_prev_action == C and agent_action == D:
            base_reward -= float(xi)
        return base_reward
    
    else:
        raise ValueError(f"Unknown moral_type: {moral_type}. "
                        f"Choose from: game, deontological, utilitarian, game+deontological")


def get_max_moral_reward(moral_type: str, game_config=None) -> float:
    """
    Get the maximum possible moral reward for normalization.
    
    Useful for computing moral regret as done in the paper.
    """
    if game_config is None:
        from envs.ipd import IPDConfig
        game_config = IPDConfig()
    
    if moral_type == "game":
        return game_config.R_DC  # Max selfish reward: defect while opponent cooperates
    elif moral_type == "deontological":
        return 0.0  # Max is 0 (no violation)
    elif moral_type == "utilitarian":
        return 2 * game_config.R_CC  # Max collective: both cooperate
    elif moral_type == "game+deontological":
        return game_config.R_DC  # Max without penalty
    else:
        raise ValueError(f"Unknown moral_type: {moral_type}")


def get_min_moral_reward(moral_type: str, xi: float = 3.0, game_config=None) -> float:
    """Get the minimum possible moral reward."""
    if game_config is None:
        from envs.ipd import IPDConfig
        game_config = IPDConfig()
    
    if moral_type == "game":
        return game_config.R_CD  # Min: cooperate while opponent defects
    elif moral_type == "deontological":
        return -xi  # Violation penalty
    elif moral_type == "utilitarian":
        return 2 * game_config.R_DD  # Both defect
    elif moral_type == "game+deontological":
        return game_config.R_CD - xi  # Worst case
    else:
        raise ValueError(f"Unknown moral_type: {moral_type}")


def compute_moral_regret(
    moral_type: str,
    achieved_reward: float,
    game_config=None,
    xi: float = 3.0,
) -> float:
    """
    Compute moral regret as defined in the paper.
    
    Regret = max_possible_reward - achieved_reward
    
    This is used in the paper's evaluation (Figure 5).
    """
    max_reward = get_max_moral_reward(moral_type, game_config)
    return max_reward - achieved_reward