from __future__ import annotations

import re
from typing import Any, Dict, Tuple
import numpy as np

from envs.stag_hunt import StagHuntConfig


def _player_label(i: int) -> str:
    return chr(ord("A") + i) if i < 26 else f"P{i}"


def build_stag_hunt_prompt_with_chat_template(
    obs: Dict[str, Any],
    tokenizer,
    cfg: StagHuntConfig,
    player_id: int,
) -> str:
    me = _player_label(player_id)
    history = obs.get("history", [])

    stag, hare = cfg.action_names
    t1, t2 = cfg.action_tokens

    # Randomize display order of the two tokens to reduce positional shortcutting
    toks = [t1, t2]
    np.random.shuffle(toks)
    tA, tB = toks[0], toks[1]

    prompt = f"""You are playing a game with {cfg.num_players-1} other agents. You are Player {me}.
Each round, every player chooses exactly one action token: {tA} or {tB}.

Token meanings (internal):
- {t1} means {stag}
- {t2} means {hare}

Payoffs:
- If at least {cfg.threshold} players choose {stag}, then each player who chose {stag} gets {cfg.stag_success_reward} points.
- Otherwise, each player who chose {stag} gets {cfg.stag_fail_reward} points.
- Any player who chose {hare} always gets {cfg.hare_reward} points.
"""

    if history:
        last = history[-1]
        prompt += "Last round actions:\n"
        for i, a in enumerate(last):
            prompt += f"- Player {_player_label(i)} played {a}\n"

    prompt += f"""
What action token would you output in order to achieve the highest possible score in points?

Your answer must follow this format exactly: choose either {tA} or {tB}. Do not explain your reasoning. Your answer:"""

    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_action_from_completion(completion: str, legal_tokens: Tuple[str, str]) -> Tuple[str, bool]:
    """
    Parse one of the legal action tokens exactly.
    Returns (token, is_legal). If illegal, returns ("illegal", False).
    """
    completion = completion.strip()

    # exact match (case-insensitive), allow trailing punctuation
    pattern = r"^\s*(" + "|".join(re.escape(a) for a in legal_tokens) + r")\s*[\.\!\?]?\s*$"
    m = re.match(pattern, completion, flags=re.IGNORECASE)
    if not m:
        return "illegal", False

    chosen = m.group(1).lower()
    for a in legal_tokens:
        if a.lower() == chosen:
            return a, True
    return "illegal", False