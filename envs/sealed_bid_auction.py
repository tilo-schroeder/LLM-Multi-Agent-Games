import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


@dataclass
class SealedBidAuctionConfig:
    """
    Two-player first-price sealed-bid auction with symmetric value.

    - Each player values the item at V (known to both).
    - Each chooses a bid: BID_LOW or BID_HIGH.
    - Highest bid wins; the winner pays their bid and gets value V.
    - Ties are split: each wins with probability 0.5, so we use expected payoff.

    Example with value=10, bid_low=2, bid_high=6:

                    P2: BID_LOW       P2: BID_HIGH
        P1: BID_LOW   (4, 4)           (0, 4)
        P1: BID_HIGH  (4, 0)           (2, 2)
    """
    num_players: int = 2
    num_rounds: int = 5

    value: float = 10.0
    bid_low: float = 2.0
    bid_high: float = 6.0


class SealedBidAuctionEnv:
    """Simple 2-player repeated sealed-bid auction."""

    ACTION_BID_LOW = "BID_LOW"
    ACTION_BID_HIGH = "BID_HIGH"

    def __init__(self, config: SealedBidAuctionConfig):
        assert config.num_players == 2, "This env is implemented for 2 players."
        self.config = config
        self.initial_state: Tuple[str, str] | None = None
        self.reset()

    def reset(self, random_initial_state: bool = False):
        self.round = 0
        self.history: List[Tuple[str, str]] = []
        self.initial_state = None

        if random_initial_state:
            a1 = random.choice([self.ACTION_BID_LOW, self.ACTION_BID_HIGH])
            a2 = random.choice([self.ACTION_BID_LOW, self.ACTION_BID_HIGH])
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
        v = c.value
        low = c.bid_low
        high = c.bid_high

        # map actions to bids
        b1 = low if a1 == self.ACTION_BID_LOW else high
        b2 = low if a2 == self.ACTION_BID_LOW else high

        # tie
        if b1 == b2:
            # each has 0.5 chance of winning
            payoff = 0.5 * (v - b1)
            return payoff, payoff

        # player 1 bids higher
        if b1 > b2:
            return v - b1, 0.0

        # player 2 bids higher
        return 0.0, v - b2

    def step(self, action_p1: str, action_p2: str):
        assert action_p1 in (self.ACTION_BID_LOW, self.ACTION_BID_HIGH)
        assert action_p2 in (self.ACTION_BID_LOW, self.ACTION_BID_HIGH)

        self.history.append((action_p1, action_p2))
        self.round += 1

        r1, r2 = self._round_payoffs(action_p1, action_p2)

        done = self.round >= self.config.num_rounds
        obs = self._get_obs()
        info = {}
        return obs, (r1, r2), done, info

    def compute_episode_returns(self) -> Tuple[float, float]:
        total_r1 = 0.0
        total_r2 = 0.0
        for a1, a2 in self.history:
            r1, r2 = self._round_payoffs(a1, a2)
            total_r1 += r1
            total_r2 += r2
        return total_r1, total_r2