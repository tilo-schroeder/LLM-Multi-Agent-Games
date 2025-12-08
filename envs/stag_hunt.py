import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

# =========================
# 1. Stag Hunt Environment
# =========================

@dataclass
class StagHuntConfig:
    num_players: int = 2
    num_rounds: int = 5
    R_stag_stag: float = 4.0
    R_hare_hare: float = 2.0
    R_stag_hare: float = 0.0
    R_hare_stag: float = 3.0


class StagHuntEnv:
    """Simple 2-player repeated Stag Hunt."""

    ACTION_STAG = "STAG"
    ACTION_HARE = "HARE"

    def __init__(self, config: StagHuntConfig):
        assert config.num_players == 2, "This env is implemented for 2 players."
        self.config = config
        self.initial_state: Tuple[str, str] | None = None
        self.reset()

    def reset(self, random_initial_state: bool = False):
        self.round = 0
        # history: list of tuples (a1, a2) for *real* past rounds
        self.history: List[Tuple[str, str]] = []
        self.initial_state = None

        if random_initial_state:
            # Random “previous joint action” for the state, as in the paper
            a1 = random.choice([self.ACTION_STAG, self.ACTION_HARE])
            a2 = random.choice([self.ACTION_STAG, self.ACTION_HARE])
            self.initial_state = (a1, a2)

        return self._get_obs()

    def _get_obs(self) -> Dict[str, Any]:
        # Observation history = optional initial_state + real rounds
        full_history: List[Tuple[str, str]] = []
        if self.initial_state is not None:
            full_history.append(self.initial_state)
        full_history.extend(self.history)

        return {
            "round": self.round,          # number of *real* rounds played so far
            "history": full_history,      # what the agent sees as “previous moves”
        }

    def step(self, action_p1: str, action_p2: str):
        assert action_p1 in (self.ACTION_STAG, self.ACTION_HARE)
        assert action_p2 in (self.ACTION_STAG, self.ACTION_HARE)

        # Only *legal* moves call step; illegal moves should never get here
        self.history.append((action_p1, action_p2))
        self.round += 1

        c = self.config
        if action_p1 == self.ACTION_STAG and action_p2 == self.ACTION_STAG:
            r1 = c.R_stag_stag
            r2 = c.R_stag_stag
        elif action_p1 == self.ACTION_HARE and action_p2 == self.ACTION_HARE:
            r1 = c.R_hare_hare
            r2 = c.R_hare_hare
        elif action_p1 == self.ACTION_STAG and action_p2 == self.ACTION_HARE:
            r1 = c.R_stag_hare
            r2 = c.R_hare_stag
        else:  # action_p1 == HARE, action_p2 == STAG
            r1 = c.R_hare_stag
            r2 = c.R_stag_hare

        done = self.round >= c.num_rounds
        obs = self._get_obs()
        info = {}
        return obs, (r1, r2), done, info

    def compute_episode_returns(self) -> Tuple[float, float]:
        """Recompute total returns for each player from history."""
        c = self.config
        total_r1 = 0.0
        total_r2 = 0.0
        for a1, a2 in self.history:
            if a1 == self.ACTION_STAG and a2 == self.ACTION_STAG:
                r1 = c.R_stag_stag
                r2 = c.R_stag_stag
            elif a1 == self.ACTION_HARE and a2 == self.ACTION_HARE:
                r1 = c.R_hare_hare
                r2 = c.R_hare_hare
            elif a1 == self.ACTION_STAG and a2 == self.ACTION_HARE:
                r1 = c.R_stag_hare
                r2 = c.R_hare_stag
            else:  # HARE, STAG
                r1 = c.R_hare_stag
                r2 = c.R_stag_hare
            total_r1 += r1
            total_r2 += r2
        return total_r1, total_r2
