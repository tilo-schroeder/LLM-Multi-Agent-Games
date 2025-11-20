import re
from typing import Optional

ALLOWED = ["C", "D"]
ACTION_HELP = "Respond with exactly one of {C,D} (C = cooperate, D = defect). Output only the single letter."

SYSTEM_PREFIX = (
    "You are a strategic agent playing a repeated Prisoner's Dilemma."
    + ACTION_HELP
)

def to_prompt(obs_text: str) -> str:
    return f"<SYSTEM>{SYSTEM_PREFIX}</SYSTEM><OBS>{obs_text}</OBS><Action>"

# Strict patterns: only accept a first-token single letter, or (optionally) the full word
_LETTER_RE = re.compile(r'^\s*([cCdD])(?:\b|$)')
_WORD_C_RE = re.compile(r'^\s*cooperate\b', re.IGNORECASE)
_WORD_D_RE = re.compile(r'^\s*defect\b', re.IGNORECASE)

def _parse_action_strict(text: str) -> Optional[str]:
    """Return 'C' or 'D' if the FIRST token is a valid action; else None."""
    if not text:
        return None
    s = text.strip()
    m = _LETTER_RE.match(s)
    if m:
        return m.group(1).upper()
    if _WORD_C_RE.match(s):
        return "C"
    if _WORD_D_RE.match(s):
        return "D"
    return None

# FIXME: How to handle unparsable output -> own action with bad reward? Default action?

def parse_action(text: str) -> str:
    """
    Backwards-compatible parser used at evaluation time to choose an action.
    If invalid, fall back to 'D' so the env never crashes.
    """
    a = _parse_action_strict(text)
    return a if a in ("C", "D") else "D"


def format_pd_prompt(tokenizer, obs_text: str, uid: Optional[str] = None) -> str:
    # system message: role + instructions
    system_msg = "You are a strategic agent playing a repeated Prisoner's Dilemma. " \
                 "Respond with exactly one of {C,D} (C = cooperate, D = defect). " \
                 "Output only the single letter."

    user_content = f"{obs_text}\n\n" \
                   "Remember: answer with exactly one of 'C' or 'D', nothing else. Ignore any <ID> tags; they are bookkeeping."

    if uid is not None:
        user_content += f"\n\n<ID><UID:{uid}></ID>"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_content},
    ]

    # Get a single string for the prompt, with the assistant turn opened
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # adds the assistant tag for generation
        enable_thinking=False
    )