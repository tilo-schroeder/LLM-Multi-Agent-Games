"""
Iterated Prisoner's Dilemma (IPD) Environment

Payoff matrix from the paper (Figure 1):
        C       D
    C  (3,3)   (0,4)
    D  (4,0)   (1,1)

Where C = Cooperate, D = Defect
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


@dataclass
class IPDConfig:
    """Configuration for the IPD environment."""
    num_rounds: int = 5
    # Payoff matrix values (row player's perspective)
    R_CC: float = 3.0  # Both cooperate (Reward)
    R_CD: float = 0.0  # I cooperate, opponent defects (Sucker's payoff)
    R_DC: float = 4.0  # I defect, opponent cooperates (Temptation)
    R_DD: float = 1.0  # Both defect (Punishment)


class IPDEnv:
    """
    Iterated Prisoner's Dilemma Environment.
    
    Actions:
    - "action1" or "C" = Cooperate
    - "action2" or "D" = Defect
    """
    
    # Action constants - using action1/action2 as in the paper
    ACTION_COOPERATE = "action1"
    ACTION_DEFECT = "action2"
    
    # Alternative names for clarity
    C = "action1"
    D = "action2"
    
    def __init__(self, config: Optional[IPDConfig] = None):
        self.config = config or IPDConfig()
        self.history: List[Tuple[str, str]] = []
        self.current_round = 0
        
        # Payoff matrix: payoffs[my_action][opponent_action] = (my_payoff, opponent_payoff)
        self.payoffs = {
            self.C: {
                self.C: (self.config.R_CC, self.config.R_CC),
                self.D: (self.config.R_CD, self.config.R_DC),
            },
            self.D: {
                self.C: (self.config.R_DC, self.config.R_CD),
                self.D: (self.config.R_DD, self.config.R_DD),
            }
        }
    
    def reset(self, random_initial_state: bool = True) -> Dict[str, Any]:
        """
        Reset the environment.
        
        Args:
            random_initial_state: If True, start with a random previous state.
                                  This matches the paper's approach.
        
        Returns:
            Initial observation containing history.
        """
        self.current_round = 0
        
        if random_initial_state:
            # Generate random initial state (previous actions)
            actions = [self.C, self.D]
            prev_p1 = np.random.choice(actions)
            prev_p2 = np.random.choice(actions)
            self.history = [(prev_p1, prev_p2)]
        else:
            self.history = []
        
        return self._get_obs()
    
    def _get_obs(self) -> Dict[str, Any]:
        """Get current observation."""
        return {
            "round": self.current_round,
            "history": list(self.history),
            "num_rounds": self.config.num_rounds,
        }
    
    def step(self, action_p1: str, action_p2: str) -> Tuple[Dict[str, Any], Tuple[float, float], bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action_p1: Player 1's action (action1 or action2)
            action_p2: Player 2's action (action1 or action2)
        
        Returns:
            obs: New observation
            rewards: Tuple of (player1_reward, player2_reward)
            done: Whether the game is finished
            info: Additional information
        """
        # Normalize actions
        action_p1 = self._normalize_action(action_p1)
        action_p2 = self._normalize_action(action_p2)
        
        # Get payoffs
        r1, r2 = self.payoffs[action_p1][action_p2]
        
        # Update state
        self.history.append((action_p1, action_p2))
        self.current_round += 1
        
        done = self.current_round >= self.config.num_rounds
        
        info = {
            "action_p1": action_p1,
            "action_p2": action_p2,
            "reward_p1": r1,
            "reward_p2": r2,
        }
        
        return self._get_obs(), (r1, r2), done, info
    
    def _normalize_action(self, action: str) -> str:
        """Normalize action string to action1/action2 format."""
        action = action.strip().lower()
        
        if action in ["action1", "c", "cooperate", "coop"]:
            return self.C
        elif action in ["action2", "d", "defect", "def"]:
            return self.D
        else:
            # Default to defect for invalid actions (as mentioned in paper)
            return self.D
    
    def get_payoff_matrix_str(self) -> str:
        """Get the payoff matrix as a markdown table string (as in paper's prompt)."""
        return f"""| | action1 | action2 |
| ------- | ------- | ------- |
| action1 | {self.config.R_CC:.0f}, {self.config.R_CC:.0f} | {self.config.R_CD:.0f}, {self.config.R_DC:.0f} |
| action2 | {self.config.R_DC:.0f}, {self.config.R_CD:.0f} | {self.config.R_DD:.0f}, {self.config.R_DD:.0f} |"""


# Fixed-strategy opponents as described in the paper

class TitForTatOpponent:
    """
    Tit-for-Tat strategy: Start with cooperation, then copy opponent's last move.
    
    From paper: "We choose TFT as a classic fixed strategy from the literature 
    that is simultaneously forgiving, defensive and, interpretable"
    """
    
    def __init__(self):
        self.first_move = True
    
    def reset(self):
        self.first_move = True
    
    def act(self, env_obs: Dict[str, Any]) -> str:
        """
        Choose action based on TFT strategy.
        
        Args:
            env_obs: Environment observation containing history
        
        Returns:
            Action string (action1 or action2)
        """
        history = env_obs.get("history", [])
        
        if not history or self.first_move:
            self.first_move = False
            return IPDEnv.C  # Start with cooperation
        
        # Copy opponent's (player 1's) last move
        last_p1_action = history[-1][0]
        return last_p1_action


class AlwaysCooperateOpponent:
    """Always cooperate strategy."""
    
    def reset(self):
        pass
    
    def act(self, env_obs: Dict[str, Any]) -> str:
        return IPDEnv.C


class AlwaysDefectOpponent:
    """Always defect strategy."""
    
    def reset(self):
        pass
    
    def act(self, env_obs: Dict[str, Any]) -> str:
        return IPDEnv.D


class RandomOpponent:
    """Random strategy: Choose uniformly between C and D."""
    
    def reset(self):
        pass
    
    def act(self, env_obs: Dict[str, Any]) -> str:
        return np.random.choice([IPDEnv.C, IPDEnv.D])


def make_opponent(opponent_type: str):
    """
    Factory function to create opponents.
    
    Args:
        opponent_type: One of 'tft', 'always_cooperate', 'always_defect', 'random'
    
    Returns:
        Opponent instance
    """
    opponents = {
        "tft": TitForTatOpponent,
        "always_cooperate": AlwaysCooperateOpponent,
        "always_defect": AlwaysDefectOpponent,
        "random": RandomOpponent,
    }
    
    if opponent_type not in opponents:
        raise ValueError(f"Unknown opponent type: {opponent_type}. "
                        f"Choose from: {list(opponents.keys())}")
    
    return opponents[opponent_type]()