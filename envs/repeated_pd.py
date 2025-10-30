from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

ACTIONS = ["C", "D"]  # Cooperate / Defect

@dataclass
class Config:
    rounds: int = 20
    # Payoffs with T>R>P>S
    T: float = 5.0
    R: float = 3.0
    P: float = 1.0
    S: float = 0.0
    # Probability an intended action flips (trembling hand)
    action_error: float = 0.0
    seed: int = 0

class RepeatedPD:
    """
    Repeated Prisoner's Dilemma (two players). Each step both choose C or D.
    Returns per-agent rewards and observations summarizing last actions & scores.
    """

    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.t = 0
        self.last_actions: Dict[str, Optional[str]] = {"agent_0": None, "agent_1": None}
        self.cum: Dict[str, float] = {"agent_0": 0.0, "agent_1": 0.0}

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.last_actions = {"agent_0": None, "agent_1": None}
        self.cum = {"agent_0": 0.0, "agent_1": 0.0}
        return self._obs()

    def step(self, a0: str, a1: str):
        assert a0 in ACTIONS and a1 in ACTIONS
        self.t += 1
        a0_eff = self._noisy(a0)
        a1_eff = self._noisy(a1)
        r0, r1 = self._payoff(a0_eff, a1_eff)
        self.cum["agent_0"] += r0
        self.cum["agent_1"] += r1
        self.last_actions = {"agent_0": a0_eff, "agent_1": a1_eff}
        done = self.t >= self.cfg.rounds
        return self._obs(), (r0, r1), done

    def _noisy(self, a: str) -> str:
        if self.cfg.action_error > 0 and self.rng.random() < self.cfg.action_error:
            return "D" if a == "C" else "C"
        return a

    def _payoff(self, a0: str, a1: str) -> Tuple[float, float]:
        if a0 == "C" and a1 == "C":
            return self.cfg.R, self.cfg.R
        if a0 == "C" and a1 == "D":
            return self.cfg.S, self.cfg.T
        if a0 == "D" and a1 == "C":
            return self.cfg.T, self.cfg.S
        # D, D
        return self.cfg.P, self.cfg.P

    def _obs(self):
        steps_left = self.cfg.rounds - self.t
        def fmt(agent, other):
            la = self.last_actions[agent]
            lb = self.last_actions[other]
            return (
                f"You are {agent} in a repeated Prisoner's Dilemma."
                f"Round {self.t+1} of {self.cfg.rounds}. Steps left: {steps_left}."
                f"Last actions — you: {la}, opponent: {lb}."
                f"Cumulative payoff — you: {self.cum[agent]:.1f}, opponent: {self.cum[other]:.1f}."
                f"Respond with exactly one of {{C,D}} only (C=cooperate, D=defect)."
            )
        return {"agent_0": fmt("agent_0", "agent_1"), "agent_1": fmt("agent_1", "agent_0")}
