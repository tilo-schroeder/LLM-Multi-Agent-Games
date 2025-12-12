from typing import List, Tuple

import logging
import torch

from envs.stag_hunt import StagHuntEnv, StagHuntConfig
from stag_hunt_grpo.data_structures import DecisionSample
from stag_hunt_grpo.moral_rewards import compute_moral_reward_stag_hunt
from stag_hunt_grpo.prompting import build_decision_prompt, extract_action_from_completion
from stag_hunt_grpo.sampling import generate_completion
from utils.logging_utils import get_logger


def rollout_episode_two_llm_agents(
    model_p1,
    model_p2,
    tokenizer,
    config: StagHuntConfig,
    device: torch.device,
    episode_id: int,
    moral_type: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    xi: float = 3.0,
    illegal_penalty: float = -6.0,
) -> Tuple[List[DecisionSample], float, float]:
    """
    Self-play rollout: two distinct LLM policies (model_p1, model_p2)
    play Player 1 and Player 2 respectively. Both receive moral,
    per-decision rewards.
    """

    env = StagHuntEnv(config)
    obs = env.reset(random_initial_state=True)

    samples: List[DecisionSample] = []
    rewards_p1: List[float] = []
    rewards_p2: List[float] = []

    for t in range(config.num_rounds):
        round_idx = t + 1
        history = obs["history"]

        prev_a1, prev_a2 = (None, None)
        if history:
            prev_a1, prev_a2 = history[-1]

        # ---- Player 1 decision (model_p1) ----
        prompt1 = build_decision_prompt(1, obs, config, tokenizer)
        completion1 = generate_completion(
            model_p1, tokenizer, prompt1, device,
            temperature=temperature, top_p=top_p,
        )
        a1, legal1 = extract_action_from_completion(completion1)
        sample1 = DecisionSample(
            episode_id=episode_id,
            player_id=1,
            round_idx=round_idx,
            prompt=prompt1,
            completion=completion1,
            reward=0.0,
            action=a1,
            opp_prev_action=prev_a2 or "",
            is_legal=legal1,
        )

        # ---- Player 2 decision (model_p2) ----
        prompt2 = build_decision_prompt(2, obs, config, tokenizer)
        completion2 = generate_completion(
            model_p2, tokenizer, prompt2, device,
            temperature=temperature, top_p=top_p,
        )
        a2, legal2 = extract_action_from_completion(completion2)
        sample2 = DecisionSample(
            episode_id=episode_id,
            player_id=2,
            round_idx=round_idx,
            prompt=prompt2,
            completion=completion2,
            reward=0.0,
            action=a2,
            opp_prev_action=prev_a1 or "",
            is_legal=legal2,
        )

        # If either player is illegal, do not advance env; just give penalties
        if not legal1 or not legal2:
            r1 = r2 = 0.0

            moral_r1 = compute_moral_reward_stag_hunt(
                moral_type=moral_type,
                agent_action=a1 if legal1 else StagHuntEnv.ACTION_HARE,
                opp_prev_action=prev_a2,
                r_agent=r1,
                r_opp=r2,
                is_legal=legal1,
                xi=xi,
                illegal_penalty=illegal_penalty,
            )
            moral_r2 = compute_moral_reward_stag_hunt(
                moral_type=moral_type,
                agent_action=a2 if legal2 else StagHuntEnv.ACTION_HARE,
                opp_prev_action=prev_a1,
                r_agent=r2,
                r_opp=r1,
                is_legal=legal2,
                xi=xi,
                illegal_penalty=illegal_penalty,
            )

            sample1.reward = moral_r1
            sample2.reward = moral_r2

            rewards_p1.append(moral_r1)
            rewards_p2.append(moral_r2)
            samples.extend([sample1, sample2])
            # State unchanged
            continue

        # ---- Both legal → env step ----
        obs, (r1, r2), done, info = env.step(a1, a2)

        moral_r1 = compute_moral_reward_stag_hunt(
            moral_type=moral_type,
            agent_action=a1,
            opp_prev_action=prev_a2,
            r_agent=r1,
            r_opp=r2,
            is_legal=True,
            xi=xi,
            illegal_penalty=illegal_penalty,
        )
        moral_r2 = compute_moral_reward_stag_hunt(
            moral_type=moral_type,
            agent_action=a2,
            opp_prev_action=prev_a1,
            r_agent=r2,
            r_opp=r1,
            is_legal=True,
            xi=xi,
            illegal_penalty=illegal_penalty,
        )

        sample1.reward = moral_r1
        sample2.reward = moral_r2

        rewards_p1.append(moral_r1)
        rewards_p2.append(moral_r2)
        samples.extend([sample1, sample2])

    total_moral_p1 = float(sum(rewards_p1))
    total_moral_p2 = float(sum(rewards_p2))

    return samples, total_moral_p1, total_moral_p2


def collect_batch_moral_two_llm_agents(
    model_p1,
    model_p2,
    tokenizer,
    config: StagHuntConfig,
    device: torch.device,
    num_episodes: int,
    moral_type: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    logger: logging.Logger = None,
) -> List[DecisionSample]:
    """
    Collect num_episodes of self-play (LLM vs LLM, two distinct models),
    with intrinsic moral rewards attached per decision for both players.
    """
    if logger is None:
        logger = get_logger()

    all_samples: List[DecisionSample] = []

    for ep_id in range(num_episodes):
        logger.info(f"  Rolling out LLM-vs-LLM moral episode {ep_id} (type={moral_type})...")
        samples, total_moral_p1, total_moral_p2 = rollout_episode_two_llm_agents(
            model_p1=model_p1,
            model_p2=model_p2,
            tokenizer=tokenizer,
            config=config,
            device=device,
            episode_id=ep_id,
            moral_type=moral_type,
            temperature=temperature,
            top_p=top_p,
        )
        all_samples.extend(samples)

        if ep_id < 2:
            logger.info(
                f"    Episode {ep_id} total moral returns: "
                f"P1={total_moral_p1:.3f}, P2={total_moral_p2:.3f}"
            )
            rounds = [
                (s.round_idx, s.player_id,
                 extract_action_from_completion(s.completion)[0])
                for s in samples
            ]
            logger.info("    Parsed actions (round, player, action):")
            for triple in rounds:
                logger.info(f"      {triple}")

    return all_samples