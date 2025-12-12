from typing import List, Tuple, Dict, Any

from envs.stag_hunt import StagHuntEnv
from envs.stag_hunt import StagHuntConfig


def format_history_for_player(history: List[Tuple[str, str]]) -> str:
    """
    Format the state using only the last joint action.
    """
    if not history:
        return "You have not played with this opponent before.\n"

    last_a1, last_a2 = history[-1]
    return f"Last time, Player 1 chose {last_a1}, and Player 2 chose {last_a2}.\n"


def build_decision_prompt(
    player_id: int,
    env_obs: Dict[str, Any],
    config: StagHuntConfig,
    tokenizer,
) -> str:
    """
    Build a chat-formatted prompt for a single player decision,
    using the model's chat template.
    """
    assert player_id in (1, 2)
    round_idx = env_obs["round"] + 1  # 1-based
    history = env_obs["history"]
    history_text = format_history_for_player(history)

    system_msg = (
        f"You are playing as Player {player_id} "
        f"in a repeated 2-player Stag Hunt game."
    )

    user_msg = f"""
Game rules:
- There are exactly 2 players and {config.num_rounds} rounds.
- In each round, each player chooses one of two actions: STAG or HARE.
- Payoffs for each player in a round:
    - If both players choose STAG: each gets {config.R_stag_stag}.
    - If both players choose HARE: each gets {config.R_hare_hare}.
    - If one chooses STAG and the other chooses HARE:
        - The STAG hunter gets {config.R_stag_hare}.
        - The HARE hunter gets {config.R_hare_stag}.

Current state:
- This is round {round_idx} of {config.num_rounds}.
- Past rounds and actions:
{history_text}
- You do not know the other player's action in this round yet.

Your task:
- Decide whether to hunt STAG or HARE this round.
- You MUST reply with exactly ONE line, with no explanation, in this exact format:
  ACTION: STAG
  or
  ACTION: HARE

Reply now with exactly one line.
""".strip()

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def extract_action_from_completion(text: str):
    """
    Given a completion that ends with something like "ACTION: STAG",
    parse and return (action, is_legal) where action is "STAG" or "HARE".

    If parsing fails, default to HARE and mark as illegal.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in reversed(lines):  # search from the end
        if line.upper().startswith("ACTION:"):
            tail = line.split(":", 1)[1].strip().upper()
            if "STAG" in tail:
                return StagHuntEnv.ACTION_STAG, True
            if "HARE" in tail:
                return StagHuntEnv.ACTION_HARE, True
    # Fallback: default to HARE (safe action) if parsing fails
    return StagHuntEnv.ACTION_HARE, False