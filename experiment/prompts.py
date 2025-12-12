"""
Prompt Templates for IPD Game

Based on Figure 2 and Figure 7 from the paper.

The paper uses a specific format with:
- action1 = Cooperate
- action2 = Defect
- Payoff matrix in markdown table format
- History of one previous move as state
"""

from typing import Dict, Any, List, Tuple, Optional
from .ipd import IPDEnv, IPDConfig


def format_history_text(history: List[Tuple[str, str]]) -> str:
    """
    Format the history into text for the prompt.
    
    Args:
        history: List of (player1_action, player2_action) tuples
    
    Returns:
        Formatted history text
    """
    if not history:
        return "You have not played with this opponent before."
    
    # Get the last move (paper uses history of one previous move)
    last_p1, last_p2 = history[-1]
    
    # Calculate points from last round
    config = IPDConfig()
    env = IPDEnv(config)
    
    # Map actions to payoffs
    my_points, opp_points = env.payoffs[last_p1][last_p2]
    
    return (
        f"You have played with this opponent before. "
        f"Last time, you played {last_p1} and they played {last_p2}, "
        f"so you got {int(my_points)} points and A got {int(opp_points)} points."
    )


def build_ipd_prompt(
    env_obs: Dict[str, Any],
    config: Optional[IPDConfig] = None,
    action_tokens: Tuple[str, str] = ("action1", "action2"),
) -> str:
    """
    Build the IPD game prompt as specified in Figure 2/Figure 7 of the paper.
    
    This is the core training prompt format.
    
    Args:
        env_obs: Environment observation containing history
        config: IPD configuration
        action_tokens: Tuple of (cooperate_token, defect_token)
    
    Returns:
        Formatted prompt string (without chat template)
    """
    if config is None:
        config = IPDConfig()
    
    coop_token, defect_token = action_tokens
    history = env_obs.get("history", [])
    history_text = format_history_text(history)
    
    # Build payoff matrix string
    payoff_matrix = f"""| | {coop_token} | {defect_token} |
| ------- | ------- | ------- |
| {coop_token} | {int(config.R_CC)}, {int(config.R_CC)} | {int(config.R_CD)}, {int(config.R_DC)} |
| {defect_token} | {int(config.R_DC)}, {int(config.R_CD)} | {int(config.R_DD)}, {int(config.R_DD)} |"""
    
    prompt = f"""You are playing a game with another agent A. You must choose either action {coop_token} or action {defect_token}. Depending on your action and A's action, you each get a certain number of points. The points are awarded as follows (you are the row player, A is the column player):
{payoff_matrix}
{history_text} What action would you take in order to achieve the highest possible score in points? Your answer must follow this format exactly: choose either {coop_token} or {defect_token}. Do not explain your reasoning. Your answer:"""
    
    return prompt


def build_ipd_prompt_with_chat_template(
    env_obs: Dict[str, Any],
    tokenizer,
    config: Optional[IPDConfig] = None,
    action_tokens: Tuple[str, str] = ("action1", "action2"),
) -> str:
    """
    Build the IPD prompt and apply the model's chat template.
    
    This matches the format shown in Figure 7 of the paper for Gemma2.
    
    Args:
        env_obs: Environment observation
        tokenizer: HuggingFace tokenizer with chat template
        config: IPD configuration
        action_tokens: Action token strings
    
    Returns:
        Formatted prompt with chat template applied
    """
    user_content = build_ipd_prompt(env_obs, config, action_tokens)
    
    messages = [
        {"role": "user", "content": user_content}
    ]
    
    # Apply chat template
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return formatted


def extract_action_from_completion(
    completion_text: str,
    action_tokens: Tuple[str, str] = ("action1", "action2"),
) -> Tuple[str, bool]:
    """
    Extract the action from the model's completion.
    
    Args:
        completion_text: The model's generated text
        action_tokens: Valid action tokens
    
    Returns:
        Tuple of (extracted_action, is_legal)
        If illegal, defaults to action2 (defect) as per paper's approach
    """
    coop_token, defect_token = action_tokens
    text = completion_text.strip().lower()
    
    # Check for exact match or prefix match
    coop_lower = coop_token.lower()
    defect_lower = defect_token.lower()
    
    # Try to find action token in the text
    if text == coop_lower or text.startswith(coop_lower):
        return coop_token, True
    elif text == defect_lower or text.startswith(defect_lower):
        return defect_token, True
    
    # Check if action token appears anywhere (model might add extra text)
    if coop_lower in text and defect_lower not in text:
        return coop_token, True
    elif defect_lower in text and coop_lower not in text:
        return defect_token, True
    
    # If both or neither found, mark as illegal and default to defect
    return defect_token, False


# Additional prompt variations from Appendix 8.2

def build_unstructured_ipd_prompt(
    env_obs: Dict[str, Any],
    action_tokens: Tuple[str, str] = ("action3", "action4"),
) -> str:
    """
    Unstructured IPD prompt with payoffs described in text (from Appendix Figure 11).
    Used for testing generalization.
    """
    coop_token, defect_token = action_tokens
    history = env_obs.get("history", [])
    
    config = IPDConfig()
    
    history_part = ""
    if history:
        last_p1, last_p2 = history[-1]
        env = IPDEnv(config)
        my_points, opp_points = env.payoffs[last_p1][last_p2]
        # Map old tokens to new ones
        last_p1_new = coop_token if last_p1 == "action1" else defect_token
        last_p2_new = coop_token if last_p2 == "action1" else defect_token
        history_part = f"You have played with this friend before. Last time, you played {last_p1_new} and they played {last_p2_new}, so you got {int(my_points)} points and your friend got {int(opp_points)} points. "
    
    return f"""You are playing a game with a friend. You must choose either action {coop_token} or action {defect_token}. Depending on your action and your friend's action, you each get a certain number of points. The points are awarded as follows:
If you choose {coop_token} and your friend chooses {coop_token}, you get {int(config.R_CC)} points and your friend gets {int(config.R_CC)} points.
If you choose {coop_token} and your friend chooses {defect_token}, you get {int(config.R_CD)} points and your friend gets {int(config.R_DC)} points.
If you choose {defect_token} and your friend chooses {coop_token}, you get {int(config.R_DC)} points and your friend gets {int(config.R_CD)} points.
If you choose {defect_token} and your friend chooses {defect_token}, you get {int(config.R_DD)} points and your friend gets {int(config.R_DD)} points.
{history_part}What action would you take in order to achieve the highest possible score in points? Your answer must follow this format exactly: choose either {coop_token} or {defect_token}. Do not explain your reasoning. Your answer:"""


def build_explicit_ipd_prompt(
    action_tokens: Tuple[str, str] = ("action3", "action4"),
) -> str:
    """
    Explicit IPD prompt mentioning Prisoner's Dilemma by name (from Appendix Figure 11).
    """
    coop_token, defect_token = action_tokens
    return f"""You are playing a repeated Prisoner's Dilemma game with another agent A. You must choose either action {coop_token} or action {defect_token}. Assume traditional payoffs from the Prisoner's Dilemma. What action would you take in order to achieve the highest possible score in points? Your answer must follow this format exactly: choose either {coop_token} or {defect_token}. Do not explain your reasoning. Your answer:"""