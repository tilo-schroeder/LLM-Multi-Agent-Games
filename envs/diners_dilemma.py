import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


@dataclass
class DinersDilemmaConfig:
    """
    Symmetric 2-player Diner's Dilemma payoff matrix.

    We model it as a 2x2 matrix with actions:
        - COOPERATE  (order the cheap meal)
        - DEFECT     (order the expensive meal)

    Payoffs are for (Player 1, Player 2):

                    P2: COOP         P2: DEFECT
        P1: COOP   (R_cc, R_cc)     (R_cd_coop,  R_cd_defect)
        P1: DEFECT (R_cd_defect, R_cd_coop)  (R_dd, R_dd)

    Typical Diner’s Dilemma structure:
        - R_cd_defect > R_cc > R_dd > R_cd_coop
      So individually rational is DEFECT, collectively rational is COOP.
    """
    num_players: int = 2
    num_rounds: int = 5

    # Default example payoffs – feel free to change these to your preferred matrix.
    R_cc: float = 3.0          # both order cheap
    R_dd: float = 1.0          # both order expensive
    R_cd_coop: float = 0.0     # you cooperate, they defect
    R_cd_defect: float = 4.0   # you defect, they cooperate


class DinersDilemmaEnv:
    """Simple 2-player repeated Diner's Dilemma."""

    ACTION_COOP = "COOPERATE"
    ACTION_DEFECT = "DEFECT"

    def __init__(self, config: DinersDilemmaConfig):
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
            a1 = random.choice([self.ACTION_COOP, self.ACTION_DEFECT])
            a2 = random.choice([self.ACTION_COOP, self.ACTION_DEFECT])
            self.initial_state = (a1, a2)

        return self._get_obs()

    def _get_obs(self) -> Dict[str, Any]:
        # Observation history = optional initial_state + real rounds
        full_history: List[Tuple[str, str]] = []
        if self.initial_state is not None:
            full_history.append(self.initial_state)
        full_history.extend(self.history)

        return {
            "round": self.round,        # number of *real* rounds played so far
            "history": full_history,    # what the agent sees as “previous moves”
        }

    def step(self, action_p1: str, action_p2: str):
        assert action_p1 in (self.ACTION_COOP, self.ACTION_DEFECT)
        assert action_p2 in (self.ACTION_COOP, self.ACTION_DEFECT)

        self.history.append((action_p1, action_p2))
        self.round += 1

        c = self.config

        if action_p1 == self.ACTION_COOP and action_p2 == self.ACTION_COOP:
            r1 = c.R_cc
            r2 = c.R_cc
        elif action_p1 == self.ACTION_DEFECT and action_p2 == self.ACTION_DEFECT:
            r1 = c.R_dd
            r2 = c.R_dd
        elif action_p1 == self.ACTION_COOP and action_p2 == self.ACTION_DEFECT:
            r1 = c.R_cd_coop
            r2 = c.R_cd_defect
        else:  # P1 DEFECT, P2 COOP
            r1 = c.R_cd_defect
            r2 = c.R_cd_coop

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
            if a1 == self.ACTION_COOP and a2 == self.ACTION_COOP:
                r1 = c.R_cc
                r2 = c.R_cc
            elif a1 == self.ACTION_DEFECT and a2 == self.ACTION_DEFECT:
                r1 = c.R_dd
                r2 = c.R_dd
            elif a1 == self.ACTION_COOP and a2 == self.ACTION_DEFECT:
                r1 = c.R_cd_coop
                r2 = c.R_cd_defect
            else:
                r1 = c.R_cd_defect
                r2 = c.R_cd_coop
            total_r1 += r1
            total_r2 += r2

        return total_r1, total_r2