import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


@dataclass
class PublicGoodsConfig:
    """
    Two-player linear public goods game.

    Each round:
      - Each player has an endowment E.
      - Action CONTRIBUTE: pay cost C into the public pot.
      - Action WITHHOLD: pay 0 into the pot.
      - Total contributions are multiplied by M and split equally.

    Payoff for player i:
      payoff_i = E - (C if contribute else 0) + M * (total_contribution / N)

    We choose parameters such that:
      - Withholding is individually optimal.
      - Mutual contribution is socially optimal.
    """
    num_players: int = 2
    num_rounds: int = 5

    endowment: float = 2.0
    contribution_cost: float = 1.0
    multiplier: float = 1.6  # < num_players ⇒ social dilemma structure


class PublicGoodsEnv:
    """Simple 2-player repeated Public Goods game."""

    ACTION_CONTRIBUTE = "CONTRIBUTE"
    ACTION_WITHHOLD = "WITHHOLD"

    def __init__(self, config: PublicGoodsConfig):
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
            a1 = random.choice(
                [self.ACTION_CONTRIBUTE, self.ACTION_WITHHOLD]
            )
            a2 = random.choice(
                [self.ACTION_CONTRIBUTE, self.ACTION_WITHHOLD]
            )
            self.initial_state = (a1, a2)

        return self._get_obs()

    def _get_obs(self) -> Dict[str, Any]:
        full_history: List[Tuple[str, str]] = []
        if self.initial_state is not None:
            full_history.append(self.initial_state)
        full_history.extend(self.history)

        return {
            "round": self.round,
            "history": full_history,
        }

    def _round_payoffs(
        self,
        a1: str,
        a2: str,
    ) -> Tuple[float, float]:
        c = self.config

        contrib1 = c.contribution_cost if a1 == self.ACTION_CONTRIBUTE else 0.0
        contrib2 = c.contribution_cost if a2 == self.ACTION_CONTRIBUTE else 0.0
        total_contrib = contrib1 + contrib2

        public_return = c.multiplier * total_contrib / c.num_players

        r1 = c.endowment - contrib1 + public_return
        r2 = c.endowment - contrib2 + public_return
        return r1, r2

    def step(self, action_p1: str, action_p2: str):
        assert action_p1 in (self.ACTION_CONTRIBUTE, self.ACTION_WITHHOLD)
        assert action_p2 in (self.ACTION_CONTRIBUTE, self.ACTION_WITHHOLD)

        self.history.append((action_p1, action_p2))
        self.round += 1

        r1, r2 = self._round_payoffs(action_p1, action_p2)

        done = self.round >= self.config.num_rounds
        obs = self._get_obs()
        info = {}
        return obs, (r1, r2), done, info

    def compute_episode_returns(self) -> Tuple[float, float]:
        """Recompute total returns for each player from history."""
        total_r1 = 0.0
        total_r2 = 0.0
        for a1, a2 in self.history:
            r1, r2 = self._round_payoffs(a1, a2)
            total_r1 += r1
            total_r2 += r2
        return total_r1, total_r2