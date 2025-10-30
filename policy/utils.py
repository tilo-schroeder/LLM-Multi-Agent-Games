import re

ALLOWED = ["C", "D"]
ACTION_HELP = "Respond with exactly one of {C,D} (C = cooperate, D = defect). Output only the single letter."

SYSTEM_PREFIX = (
    "You are a strategic agent playing a repeated Prisoner's Dilemma."
    + ACTION_HELP
)

def to_prompt(obs_text: str) -> str:
    return f"<SYSTEM>{SYSTEM_PREFIX}</SYSTEM><OBS>{obs_text}</OBS><Action>"

def parse_action(text: str) -> str:
    m = re.search(r"[CD]", text.upper())
    return m.group(0) if m else "D"  # default to defect if unclear
