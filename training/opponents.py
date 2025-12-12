import numpy as np
from typing import Dict, Any

from envs.stag_hunt import StagHuntEnv


class TitForTatOpponent:
    """
    Classic Tit-for-Tat opponent:

    - Round 1: play STAG (cooperate).
    - Later rounds: copy Player 1's last action.
    """

    def reset(self):
        pass

    def act(self, env_obs: Dict[str, Any]) -> str:
        history = env_obs["history"]
        if not history:
            # First move: cooperate
            return StagHuntEnv.ACTION_STAG
        last_a1, _ = history[-1]
        # Copy last action of Player 1
        return last_a1


class AlwaysCooperateOpponent:
    """Always play STAG (cooperate)."""

    def reset(self):
        pass

    def act(self, env_obs: Dict[str, Any]) -> str:
        return StagHuntEnv.ACTION_STAG


class AlwaysDefectOpponent:
    """Always play HARE (defect)."""

    def reset(self):
        pass

    def act(self, env_obs: Dict[str, Any]) -> str:
        return StagHuntEnv.ACTION_HARE


class RandomOpponent:
    """Play STAG/HARE uniformly at random."""

    def reset(self):
        pass

    def act(self, env_obs: Dict[str, Any]) -> str:
        return np.random.choice(
            [StagHuntEnv.ACTION_STAG, StagHuntEnv.ACTION_HARE]
        )


def make_fixed_opponent(opponent_type: str):
    """
    Factory for fixed opponents.

    opponent_type in {"tft", "always_cooperate", "always_defect", "random"}.
    """
    if opponent_type == "tft":
        return TitForTatOpponent()
    if opponent_type == "always_cooperate":
        return AlwaysCooperateOpponent()
    if opponent_type == "always_defect":
        return AlwaysDefectOpponent()
    if opponent_type == "random":
        return RandomOpponent()
    raise ValueError(f"Unknown opponent_type: {opponent_type}")