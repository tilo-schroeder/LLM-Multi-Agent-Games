from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import numpy as np


# ============================================================================
# N-Player Stag Hunt Environment (true joint action payoffs)
# ============================================================================

@dataclass
class StagHuntConfig:
    """Configuration for the N-player Stag Hunt environment."""
    num_players: int = 2
    max_steps: int = 1

    # Environment actions (internal semantics)
    action_names: Tuple[str, str] = ("stag", "hare")

    # LLM-visible "legal" action tokens (stabilizes parsing & PPO)
    # action_tokens: Tuple[str, str] = ("action1", "action2")
    action_tokens: Tuple[str, str] = ("stag", "hare")

    # Threshold Stag Hunt:
    threshold: int = 2
    stag_success_reward: float = 4.0
    stag_fail_reward: float = 0.0
    hare_reward: float = 2.0

    def token_to_action(self, tok: str) -> str:
        """Map LLM token -> env action."""
        if tok == self.action_tokens[0]:
            return self.action_names[0]  # action1 -> stag
        if tok == self.action_tokens[1]:
            return self.action_names[1]  # action2 -> hare
        return self.action_names[1]


class StagHuntEnv:
    """
    True N-player Stag Hunt. Payoffs computed from the joint action profile.
    history is List[List[str]] of joint action vectors (length num_players).
    """

    def __init__(self, config: Optional[StagHuntConfig] = None):
        self.config = config or StagHuntConfig()
        self.history: List[List[str]] = []
        self.step_count = 0

    def reset(self, random_initial_state: bool = True) -> Dict[str, Any]:
        self.history = []
        self.step_count = 0

        # Random previous move initialization
        if random_initial_state:
            acts = list(self.config.action_names)
            prev = [str(np.random.choice(acts)) for _ in range(self.config.num_players)]
            self.history.append(prev)

        return {"history": [h.copy() for h in self.history]}

    def step(self, actions: List[str]) -> Tuple[Dict[str, Any], List[float], bool, Dict]:
        cfg = self.config
        assert len(actions) == cfg.num_players

        k = sum(1 for a in actions if a == cfg.action_names[0])  # #stag
        success = (k >= cfg.threshold)

        payoffs: List[float] = []
        for a in actions:
            if a == cfg.action_names[1]:  # hare
                payoffs.append(cfg.hare_reward)
            else:  # stag
                payoffs.append(cfg.stag_success_reward if success else cfg.stag_fail_reward)

        self.history.append(list(actions))
        self.step_count += 1
        done = self.step_count >= cfg.max_steps
        return {"history": [h.copy() for h in self.history]}, payoffs, done, {}

    def step_with_legality(
        self,
        actions: List[str],
        legal: List[bool],
    ) -> Tuple[Dict[str, Any], List[float], bool, Dict]:
        """
        If a player is illegal, do NOT let their action update the public state.
        We freeze their action to the previous state's action for state update/payoff calc.

        (The illegal player still receives illegal penalty in reward function.)
        """
        cfg = self.config
        assert len(actions) == cfg.num_players
        assert len(legal) == cfg.num_players

        last = self.history[-1] if self.history else [cfg.action_names[1]] * cfg.num_players
        effective = [actions[i] if legal[i] else last[i] for i in range(cfg.num_players)]
        return self.step(effective)


# ============================================================================
# Fixed Opponents (for non-LLM players)
# ============================================================================

class Opponent:
    def reset(self) -> None:
        pass

    def act(self, obs: Dict[str, Any], player_id: int, focal_id: int = 0) -> str:
        raise NotImplementedError


class AlwaysStag(Opponent):
    def act(self, obs: Dict[str, Any], player_id: int, focal_id: int = 0) -> str:
        return "stag"


class AlwaysHare(Opponent):
    def act(self, obs: Dict[str, Any], player_id: int, focal_id: int = 0) -> str:
        return "hare"


class RandomOpponent(Opponent):
    def act(self, obs: Dict[str, Any], player_id: int, focal_id: int = 0) -> str:
        return str(np.random.choice(["stag", "hare"]))


class CopyFocalLast(Opponent):
    """
    Copies the focal player's last action (default focal=Player 0).
    This is a rough "TFT-like" behavior in a simultaneous-action setting.
    """

    def act(self, obs: Dict[str, Any], player_id: int, focal_id: int = 0) -> str:
        history = obs.get("history", [])
        if not history:
            return "hare"
        last = history[-1]
        return last[focal_id]


class CopyMajorityLast(Opponent):
    """
    Multiplayer TFT: copies the MAJORITY action taken by OTHER players last round.
    Tie-break defaults to 'hare' (safer / risk-dominant), configurable.
    """

    def __init__(self, tie_break: str = "hare"):
        assert tie_break in ("stag", "hare", "random", "focal")
        self.tie_break = tie_break

    def act(self, obs: Dict[str, Any], player_id: int, focal_id: int = 0) -> str:
        history = obs.get("history", [])
        if not history:
            return "hare" if self.tie_break != "stag" else "stag"

        last = history[-1]
        if not last or player_id >= len(last):
            return "hare" if self.tie_break != "stag" else "stag"

        # Majority among OTHER players (exclude self)
        others = [last[j] for j in range(len(last)) if j != player_id]
        if not others:
            return "hare" if self.tie_break != "stag" else "stag"

        stag_count = sum(1 for a in others if a == "stag")
        hare_count = len(others) - stag_count

        if stag_count > hare_count:
            return "stag"
        if hare_count > stag_count:
            return "hare"

        # tie
        if self.tie_break == "stag":
            return "stag"
        if self.tie_break == "hare":
            return "hare"
        if self.tie_break == "focal":
            return last[focal_id] if focal_id < len(last) else "hare"
        # random
        return str(np.random.choice(["stag", "hare"]))


def make_opponent(opponent_type: str) -> Opponent:
    opponents = {
        "always_stag": AlwaysStag,
        "always_hare": AlwaysHare,
        "random": RandomOpponent,
        "copy_focal": CopyFocalLast,
        "copy_majority": CopyMajorityLast
    }
    if opponent_type not in opponents:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    return opponents[opponent_type]()