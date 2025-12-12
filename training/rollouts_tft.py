from typing import List, Tuple, Any, Dict

import logging
import torch
import numpy as np

from envs.stag_hunt import StagHuntEnv, StagHuntConfig
from stag_hunt_grpo.data_structures import DecisionSample
from stag_hunt_grpo.moral_rewards import compute_moral_reward_stag_hunt
from stag_hunt_grpo.prompting import build_decision_prompt, extract_action_from_completion
from stag_hunt_grpo.sampling import generate_completion
from stag_hunt_grpo.opponents import make_fixed_opponent, TitForTatOpponent
from utils.logging_utils import get_logger


def rollout_episode_vs_tft(
    model,
    tokenizer,
    config: StagHuntConfig,
    device: torch.device,
    episode_id: int,
    moral_type: str,
    opponent: TitForTatOpponent,
    temperature: float = 0.7,
    top_p: float = 0.9,
    xi: float = 3.0,
    illegal_penalty: float = -6.0,
) -> Tuple[List[DecisionSample], float, List[float], int, int]:
    """
    Returns:
        samples: DecisionSample list for Player 1
        total_moral_p1: sum of P1 moral rewards
        tft_rewards: list of TFT (P2) moral rewards for each env step
        p2_stag_count: how many times TFT played STAG
        p2_total_actions: how many times TFT acted (env steps)
    """

    env = StagHuntEnv(config)
    obs = env.reset(random_initial_state=True)
    opponent.reset()

    samples: List[DecisionSample] = []
    step_rewards_p1: List[float] = []
    step_rewards_p2: List[float] = []

    p2_stag_count = 0
    p2_total_actions = 0

    for t in range(config.num_rounds):
        round_idx = t + 1

        history = obs["history"]

        # From P1's perspective:
        opp_prev_action = None
        prev_a1, prev_a2 = (None, None)
        if history:
            prev_a1, prev_a2 = history[-1]
            opp_prev_action = prev_a2  # P2's last action

        prompt = build_decision_prompt(1, obs, config, tokenizer)
        completion = generate_completion(
            model, tokenizer, prompt, device,
            temperature=temperature, top_p=top_p,
        )
        action_p1, is_legal = extract_action_from_completion(completion)

        sample = DecisionSample(
            episode_id=episode_id,
            player_id=1,
            round_idx=round_idx,
            prompt=prompt,
            completion=completion,
            reward=0.0,
            action=action_p1,
            opp_prev_action=opp_prev_action or "",
            is_legal=is_legal,
        )

        if not is_legal:
            # Illegal action → no env step; penalty for P1 only
            moral_r1 = compute_moral_reward_stag_hunt(
                moral_type=moral_type,
                agent_action=StagHuntEnv.ACTION_HARE,
                opp_prev_action=opp_prev_action,
                r_agent=0.0,
                r_opp=0.0,
                is_legal=False,
                xi=xi,
                illegal_penalty=illegal_penalty,
            )
            step_rewards_p1.append(moral_r1)
            samples.append(sample)
            # TFT does not act; no P2 reward for this "round"
            continue

        # legal → TFT acts, env steps
        action_p2 = opponent.act(obs)

        p2_total_actions += 1
        if action_p2 == StagHuntEnv.ACTION_STAG:
            p2_stag_count += 1

        obs, (r1, r2), done, info = env.step(action_p1, action_p2)

        # P1 moral reward
        moral_r1 = compute_moral_reward_stag_hunt(
            moral_type=moral_type,
            agent_action=action_p1,
            opp_prev_action=opp_prev_action,
            r_agent=r1,
            r_opp=r2,
            is_legal=True,
            xi=xi,
            illegal_penalty=illegal_penalty,
        )

        # TFT moral reward (treat TFT as agent, P1 as opponent)
        moral_r2 = compute_moral_reward_stag_hunt(
            moral_type=moral_type,
            agent_action=action_p2,
            opp_prev_action=prev_a1,  # previous P1 move
            r_agent=r2,
            r_opp=r1,
            is_legal=True,
            xi=xi,
            illegal_penalty=illegal_penalty,
        )

        step_rewards_p1.append(moral_r1)
        step_rewards_p2.append(moral_r2)
        samples.append(sample)

    total_moral_p1 = float(sum(step_rewards_p1))

    for s, r in zip(samples, step_rewards_p1):
        s.reward = r

    # Note: step_rewards_p2 can be shorter than samples (no entries for illegal P1 moves)
    return samples, total_moral_p1, step_rewards_p2, p2_stag_count, p2_total_actions


def collect_batch_moral_vs_tft(
    model,
    tokenizer,
    config: StagHuntConfig,
    device: torch.device,
    num_episodes: int,
    moral_type: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    logger: logging.Logger = None,
    opponent_type: str = "tft",
) -> Tuple[List[DecisionSample], float, List[float]]:
    """
    Collect num_episodes of (LLM vs fixed opponent).

    Returns:
        all_samples: DecisionSample list for P1
        opp_stag_rate: overall STAG rate of the opponent in this batch
        opp_rewards: list of opponent moral rewards for all env steps in batch
    """
    if logger is None:
        logger = get_logger()

    all_samples: List[DecisionSample] = []
    opponent = make_fixed_opponent(opponent_type)

    total_p2_stags = 0
    total_p2_actions = 0
    opp_rewards_batch: List[float] = []

    for ep_id in range(num_episodes):
        logger.info(
            f"  Rolling out moral episode {ep_id} "
            f"(type={moral_type}) vs fixed opponent '{opponent_type}'..."
        )
        samples, total_moral_p1, opp_rewards_ep, p2_stags, p2_actions = rollout_episode_vs_tft(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            episode_id=ep_id,
            moral_type=moral_type,
            opponent=opponent,
            temperature=temperature,
            top_p=top_p,
        )
        all_samples.extend(samples)
        opp_rewards_batch.extend(opp_rewards_ep)

        total_p2_stags += p2_stags
        total_p2_actions += p2_actions

        if ep_id < 2:
            logger.info(f"    Episode {ep_id} total moral return (P1)={total_moral_p1:.3f}")
            rounds = [
                (s.round_idx, s.player_id,
                 extract_action_from_completion(s.completion)[0])
                for s in samples
            ]
            logger.info("    Parsed actions (round, player, action):")
            for triple in rounds:
                logger.info(f"      {triple}")

            opp_ep_rate = (p2_stags / p2_actions) if p2_actions > 0 else 0.0
            opp_ep_mean = (sum(opp_rewards_ep) / len(opp_rewards_ep)) if opp_rewards_ep else 0.0
            logger.info(f"    Opponent STAG rate (episode {ep_id})={opp_ep_rate:.3f}")
            logger.info(f"    Opponent mean moral reward (episode {ep_id})={opp_ep_mean:.3f}")

    opp_stag_rate = (total_p2_stags / total_p2_actions) if total_p2_actions > 0 else 0.0
    return all_samples, opp_stag_rate, opp_rewards_batch