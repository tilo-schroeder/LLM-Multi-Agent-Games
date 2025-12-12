from envs.stag_hunt import StagHuntEnv


def compute_moral_reward_stag_hunt(
    moral_type: str,
    agent_action: str,
    opp_prev_action: str,  # opponent's last move (None on first round)
    r_agent: float,
    r_opp: float,
    is_legal: bool,
    xi: float = 3.0,
    illegal_penalty: float = -6.0,
) -> float:
    """
    Intrinsic moral rewards for a Stag Hunt matrix game.

    moral_type:
      - "game"               : agent's own payoff (material reward)
      - "deontological"      : -xi if defect vs previous cooperator, else 0
      - "utilitarian"        : r_agent + r_opp
      - "game+deontological" : game payoff minus xi on norm violation

    Illegal actions (non-parsable completions) receive illegal_penalty.
    """
    if not is_legal:
        return float(illegal_penalty)

    COOP = StagHuntEnv.ACTION_STAG
    DEFECT = StagHuntEnv.ACTION_HARE

    if moral_type == "game":
        return float(r_agent)

    if moral_type == "deontological":
        if opp_prev_action is not None and opp_prev_action == COOP and agent_action == DEFECT:
            return -float(xi)
        else:
            return 0.0

    if moral_type == "utilitarian":
        return float(r_agent + r_opp)

    if moral_type == "game+deontological":
        base = float(r_agent)
        if opp_prev_action is not None and opp_prev_action == COOP and agent_action == DEFECT:
            base -= float(xi)
        return base

    raise ValueError(f"Unknown moral_type: {moral_type}")