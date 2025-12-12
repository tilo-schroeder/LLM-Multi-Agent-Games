from __future__ import annotations

import json
import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .stag_hunt import StagHuntConfig, StagHuntEnv, Opponent, make_opponent
from .prompting import build_stag_hunt_prompt_with_chat_template, extract_action_from_completion
from .rewards import compute_moral_reward
from .ppo import PPOConfig, PolicyModelWithValueHead, Experience, compute_log_probs


def _infer_max_new_tokens_for_actions(tokenizer, actions: Tuple[str, str]) -> int:
    lens = [len(tokenizer.encode(a, add_special_tokens=False)) for a in actions]
    return max(lens)


def _model_device(m: nn.Module) -> torch.device:
    return next(m.parameters()).device


class MoralPPOTrainer:
    def __init__(self, config: PPOConfig):
        self.config = config

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.stats_history: List[Dict[str, Any]] = []
        self.reward_history = deque(maxlen=1000)

        # Adaptive KL controller state
        self.ref_kl_coef = float(config.ref_kl_coef)

        self._setup_models_and_tokenizer()
        self._setup_game()
        self._setup_players()

    def _setup_models_and_tokenizer(self):
        cfg = self.config
        print(f"Loading tokenizer: {cfg.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Reference model (frozen)
        print("Loading reference model...")
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
        ref_base = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
        hidden_size = ref_base.config.hidden_size
        self.ref_model = PolicyModelWithValueHead(ref_base, hidden_size, device=_model_device(ref_base))
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def _setup_game(self):
        cfg = self.config
        self.env_config = StagHuntConfig(
            num_players=cfg.num_players,
            max_steps=cfg.batch_size,
            threshold=cfg.threshold,
            stag_success_reward=cfg.stag_success_reward,
            stag_fail_reward=cfg.stag_fail_reward,
            hare_reward=cfg.hare_reward,
        )

        # IMPORTANT: set generation length based on legal action TOKENS
        cfg.max_new_tokens = _infer_max_new_tokens_for_actions(self.tokenizer, self.env_config.action_tokens)
        print(f"Setting max_new_tokens={cfg.max_new_tokens} based on action tokenization for {self.env_config.action_tokens}")

    def _setup_players(self):
        cfg = self.config

        # Moral types per player
        if not cfg.player_moral_types:
            if cfg.llm_vs_llm:
                cfg.player_moral_types = [cfg.moral_type] + [cfg.opponent_moral_type] * (cfg.num_players - 1)
            else:
                cfg.player_moral_types = [cfg.moral_type] + ["game"] * (cfg.num_players - 1)
        if len(cfg.player_moral_types) != cfg.num_players:
            raise ValueError(f"player_moral_types must have length num_players={cfg.num_players}")

        # Which players are LLM-controlled?
        self.llm_player_ids = list(range(cfg.num_players)) if cfg.llm_vs_llm else [0]

        # Build player models + optimizers (one model per LLM player)
        self.player_models: Dict[int, PolicyModelWithValueHead] = {}
        self.player_optimizers: Dict[int, AdamW] = {}

        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }

        for pid in self.llm_player_ids:
            print(f"\nLoading LLM policy for Player {pid}...")
            base = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)

            if cfg.use_lora:
                if cfg.use_4bit:
                    base = prepare_model_for_kbit_training(base)
                lora_config = LoraConfig(
                    r=cfg.lora_rank,
                    lora_alpha=cfg.lora_alpha,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                base = get_peft_model(base, lora_config)
                print(f"Player {pid} model trainables:")
                base.print_trainable_parameters()

            hidden_size = base.config.hidden_size
            model = PolicyModelWithValueHead(base, hidden_size, device=_model_device(base))
            self.player_models[pid] = model
            self.player_optimizers[pid] = AdamW(model.parameters(), lr=cfg.learning_rate, eps=1e-5)

        # Fixed opponents for non-LLM players (when not llm_vs_llm)
        self.fixed_opponents: Dict[int, Opponent] = {}
        if not cfg.llm_vs_llm:
            for pid in range(1, cfg.num_players):
                self.fixed_opponents[pid] = make_opponent(cfg.opponent_type)

    def _episode_action_conditioned_stats(self, exps: List[Experience], pid: int) -> Dict[str, float]:
        """
        Returns per-episode fractions of (agent_action, opponent_prev_state) pairs.

        For N=2, opponent_prev_state is the single opponent's previous action: "stag" or "hare".
        For N>2, opponent_prev_state is bucketed as: "stagcount{k}" where k is #opponents who played "stag" last round.
        """
        cfg = self.config
        stag, hare = self.env_config.action_names

        total = len(exps)
        if total == 0:
            return {}

        if cfg.num_players == 2:
            prev_states = [stag, hare]
        else:
            prev_states = [f"stagcount{k}" for k in range(cfg.num_players)]

        act_states = [stag, hare, "illegal"]
        counts = {(a, p): 0 for a in act_states for p in prev_states}

        for e in exps:
            a = e.action if e.is_legal else "illegal"

            if not e.opponent_prev_actions:
                prev = hare if cfg.num_players == 2 else "stagcount0"
            else:
                if cfg.num_players == 2:
                    prev = e.opponent_prev_actions[0]
                else:
                    k = sum(1 for x in e.opponent_prev_actions if x == stag)
                    prev = f"stagcount{k}"

            if (a, prev) in counts:
                counts[(a, prev)] += 1

        out: Dict[str, float] = {}
        for prev in prev_states:
            for a in act_states:
                key = f"p{pid}_cat_{a}_prev_{prev}"
                out[key] = counts[(a, prev)] / float(total)

        return out

    def _debug_log(self, payload: Dict[str, Any]):
        path = self.config.debug_log_path
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def rollout_episode(self) -> Dict[int, List[Experience]]:
        cfg = self.config
        env = StagHuntEnv(self.env_config)
        obs = env.reset(random_initial_state=True)

        for opp in self.fixed_opponents.values():
            opp.reset()

        trajectories: Dict[int, List[Experience]] = {pid: [] for pid in self.llm_player_ids}
        stag, hare = self.env_config.action_names

        for _t in range(self.env_config.max_steps):
            history = obs.get("history", [])
            last_actions = history[-1] if history else None  # joint action vector

            actions: List[str] = [hare] * cfg.num_players
            legal: List[bool] = [True] * cfg.num_players
            step_exps: Dict[int, Experience] = {}

            # 1) Choose actions
            for pid in range(cfg.num_players):
                if pid in self.llm_player_ids:
                    model = self.player_models[pid]
                    model.eval()
                    dev = _model_device(model)

                    opp_prev_actions = None
                    if last_actions is not None:
                        opp_prev_actions = [last_actions[j] for j in range(cfg.num_players) if j != pid]

                    prompt = build_stag_hunt_prompt_with_chat_template(
                        obs, self.tokenizer, self.env_config, player_id=pid
                    )

                    encoded = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(dev)
                    prompt_length = encoded["input_ids"].shape[1]

                    with torch.no_grad():
                        gen_out = model.generate(
                            input_ids=encoded["input_ids"],
                            attention_mask=encoded["attention_mask"],
                            max_new_tokens=cfg.max_new_tokens,
                            do_sample=True,
                            temperature=cfg.temperature,
                            top_p=cfg.top_p,
                            pad_token_id=self.tokenizer.pad_token_id,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )

                    full_ids = gen_out.sequences  # [1, prompt+gen]
                    gen_len = int(full_ids.shape[1] - prompt_length)
                    attn = torch.ones_like(full_ids)

                    with torch.no_grad():
                        log_prob, value, _ = compute_log_probs(model, full_ids, attn, prompt_length, gen_len)

                        ref_dev = _model_device(self.ref_model)
                        ref_ids = full_ids.to(ref_dev)
                        ref_attn = torch.ones_like(ref_ids)
                        ref_log_prob, _, _ = compute_log_probs(self.ref_model, ref_ids, ref_attn, prompt_length, gen_len)

                    completion = self.tokenizer.decode(full_ids[0, prompt_length:], skip_special_tokens=True).strip()
                    tok, is_legal = extract_action_from_completion(completion, self.env_config.action_tokens)

                    if is_legal:
                        act = self.env_config.token_to_action(tok)
                    else:
                        act = hare  # env.step_with_legality will freeze state anyway

                    actions[pid] = act
                    legal[pid] = is_legal

                    # Debug logging
                    gen_ids = full_ids[0, prompt_length:].tolist()
                    gen_tokens = self.tokenizer.convert_ids_to_tokens(gen_ids)

                    topk = []
                    if getattr(gen_out, "scores", None) and len(gen_out.scores) > 0:
                        first_logits = gen_out.scores[0][0]
                        probs = torch.softmax(first_logits, dim=-1)
                        k = min(cfg.debug_log_topk, probs.numel())
                        vals, idxs = torch.topk(probs, k)
                        topk = [
                            {
                                "token_id": int(i.item()),
                                "token": self.tokenizer.convert_ids_to_tokens(int(i.item())),
                                "prob": float(v.item()),
                            }
                            for v, i in zip(vals, idxs)
                        ]

                    if (not cfg.debug_log_only_illegal) or (not is_legal):
                        self._debug_log({
                            "t": int(_t),
                            "player_id": int(pid),
                            "max_new_tokens": int(cfg.max_new_tokens),
                            "prompt_tail": prompt[-500:],
                            "completion_raw": completion,
                            "parsed_token": tok,
                            "parsed_action": act,
                            "is_legal": bool(is_legal),
                            "gen_token_ids": gen_ids,
                            "gen_tokens": gen_tokens,
                            "topk_first_token": topk,
                        })

                    step_exps[pid] = Experience(
                        prompt=prompt,
                        completion=completion,
                        input_ids=full_ids.detach().cpu(),
                        prompt_length=prompt_length,
                        gen_len=gen_len,
                        reward=0.0,
                        value=float(value.item()),
                        log_prob=float(log_prob.item()),
                        ref_log_prob=float(ref_log_prob.item()),
                        action=act,
                        opponent_prev_actions=opp_prev_actions,
                        is_legal=is_legal,
                        player_id=pid,
                    )
                else:
                    opp = self.fixed_opponents[pid]
                    actions[pid] = opp.act(obs, player_id=pid, focal_id=0)
                    legal[pid] = True

            # 2) Environment step (legality-aware)
            next_obs, payoffs, done, _ = env.step_with_legality(actions, legal)

            # 3) Rewards for LLM players
            for pid, exp in step_exps.items():
                moral = cfg.player_moral_types[pid]
                r = compute_moral_reward(
                    moral_type=moral,
                    agent_action=exp.action,
                    opponent_prev_actions=exp.opponent_prev_actions,
                    agent_payoff=payoffs[pid],
                    all_payoffs=payoffs,
                    is_legal=exp.is_legal,
                    xi=cfg.xi,
                    illegal_penalty=cfg.illegal_penalty,
                )
                if cfg.reward_shaping and exp.is_legal:
                    r += cfg.valid_action_bonus
                exp.reward = float(r)
                trajectories[pid].append(exp)

            obs = next_obs
            if done:
                break

        return trajectories

    def collect_batch(self) -> Dict[int, List[List[Experience]]]:
        trajs = self.rollout_episode()
        return {pid: [trajs[pid]] for pid in trajs.keys()}

    def compute_advantages(self, trajectories: List[List[Experience]]) -> List[Experience]:
        cfg = self.config
        gamma, lam = cfg.gamma, cfg.lam

        all_exps: List[Experience] = [e for traj in trajectories for e in traj]
        if not all_exps:
            return []

        task_rewards = np.array([e.reward for e in all_exps], dtype=np.float32)

        # KL-to-reference shaping (adaptive coef)
        if self.ref_kl_coef != 0.0:
            kl_terms = np.array([e.log_prob - e.ref_log_prob for e in all_exps], dtype=np.float32)
            rewards = task_rewards - float(self.ref_kl_coef) * kl_terms
        else:
            rewards = task_rewards.copy()

        rewards = rewards * cfg.reward_scale

        # Reward normalization
        if cfg.normalize_rewards and len(rewards) > 1:
            std = rewards.std()
            rewards = (rewards - rewards.mean()) / (std + 1e-8) if std > 1e-6 else (rewards - rewards.mean())

        for e, r in zip(all_exps, rewards):
            e.reward = float(r)

        advantages: List[float] = []
        returns: List[float] = []

        for traj in trajectories:
            T = len(traj)
            if T == 0:
                continue

            next_value = 0.0
            next_adv = 0.0
            traj_adv = [0.0] * T
            traj_ret = [0.0] * T

            for t in reversed(range(T)):
                e = traj[t]
                delta = e.reward + gamma * next_value - e.value
                adv = delta + gamma * lam * next_adv
                ret = adv + e.value

                traj_adv[t] = adv
                traj_ret[t] = ret

                next_value = e.value
                next_adv = adv

            advantages.extend(traj_adv)
            returns.extend(traj_ret)

        advantages_arr = np.array(advantages, dtype=np.float32)
        returns_arr = np.array(returns, dtype=np.float32)

        if cfg.normalize_advantages and len(advantages_arr) > 1:
            std = advantages_arr.std()
            advantages_arr = (advantages_arr - advantages_arr.mean()) / (std + 1e-8) if std > 1e-6 else (advantages_arr - advantages_arr.mean())

        for e, a, R in zip(all_exps, advantages_arr, returns_arr):
            e.advantage = float(a)
            e.returns = float(R)

        return all_exps

    def ppo_update(
        self,
        experiences: List[Experience],
        model: PolicyModelWithValueHead,
        optimizer: AdamW,
    ) -> Dict[str, float]:
        cfg = self.config
        model.train()
        device = _model_device(model)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        num_updates = 0

        if not experiences:
            return {"policy_loss": 0.0, "value_loss": 0.0, "kl": 0.0, "entropy": 0.0}

        for _epoch in range(cfg.ppo_epochs):
            indices = np.random.permutation(len(experiences))
            optimizer.zero_grad(set_to_none=True)
            accum = 0

            for idx in indices:
                exp = experiences[idx]
                if abs(exp.advantage) < 1e-8:
                    continue

                input_ids = exp.input_ids.to(device)
                attention_mask = torch.ones_like(input_ids)

                log_prob, value, gen_logits = compute_log_probs(
                    model, input_ids, attention_mask, exp.prompt_length, exp.gen_len
                )

                # Entropy on generated segment only
                gen_probs = F.softmax(gen_logits, dim=-1)
                gen_log_probs = F.log_softmax(gen_logits, dim=-1)
                entropy = -(gen_probs * gen_log_probs).sum(dim=-1).mean()

                old_log_prob = torch.tensor(exp.log_prob, device=device, dtype=log_prob.dtype)
                old_value = torch.tensor(exp.value, device=device, dtype=value.dtype)
                advantage = torch.tensor(exp.advantage, device=device, dtype=log_prob.dtype)
                returns = torch.tensor(exp.returns, device=device, dtype=value.dtype)

                log_ratio = log_prob - old_log_prob
                ratio = torch.exp(log_ratio)

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * advantage
                policy_loss = -torch.min(surr1, surr2)

                # Value clipping
                value_clipped = old_value + torch.clamp(value - old_value, -cfg.clip_ratio, cfg.clip_ratio)
                value_loss1 = (value - returns) ** 2
                value_loss2 = (value_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(value_loss1, value_loss2)

                approx_kl = (old_log_prob - log_prob).mean()

                loss = policy_loss + cfg.vf_coef * value_loss - cfg.entropy_coef * entropy
                (loss / cfg.grad_accum_steps).backward()
                accum += 1

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                total_approx_kl += float(approx_kl.item())
                num_updates += 1

                if accum % cfg.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            if accum % cfg.grad_accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if num_updates > 0 and cfg.target_kl:
                avg_kl = total_approx_kl / num_updates
                if abs(avg_kl) > cfg.target_kl * 1.5:
                    break

        n = max(num_updates, 1)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "kl": total_approx_kl / n,
            "entropy": total_entropy / n,
        }

    def train(self):
        cfg = self.config

        tag = f"stag_hunt_n{cfg.num_players}_thr{cfg.threshold}"
        mode = "llm_vs_llm" if cfg.llm_vs_llm else f"vs_{cfg.opponent_type}"
        output_path = os.path.join(cfg.output_dir, f"{tag}_{mode}_{cfg.moral_type}")
        os.makedirs(output_path, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Training Stag Hunt | N={cfg.num_players} threshold={cfg.threshold} steps/ep={cfg.batch_size}")
        print(f"Mode: {'LLM vs LLM (all players)' if cfg.llm_vs_llm else 'LLM vs Fixed (players 1..N-1)'}")
        print(f"LLM players: {self.llm_player_ids}")
        print(f"Player moral types: {cfg.player_moral_types}")
        print(f"Episodes: {cfg.num_episodes}")
        print(f"{'='*70}\n")

        for episode in range(1, cfg.num_episodes + 1):
            player_trajectories = self.collect_batch()
            stats: Dict[str, Any] = {"episode": episode}

            for pid in self.llm_player_ids:
                trajs = player_trajectories[pid]
                flat = [e for tr in trajs for e in tr]
                raw_rewards = [e.reward for e in flat]

                exps = self.compute_advantages(trajs)
                upd = self.ppo_update(exps, self.player_models[pid], self.player_optimizers[pid])

                if exps and cfg.target_kl:
                    kls = [((e.log_prob - e.ref_log_prob) / max(e.gen_len, 1)) for e in exps]
                    avg_kl = float(np.mean(kls))

                    if avg_kl > cfg.target_kl * 1.5:
                        self.ref_kl_coef *= 1.5
                    elif avg_kl < cfg.target_kl / 1.5:
                        self.ref_kl_coef /= 1.5

                    self.ref_kl_coef = float(np.clip(self.ref_kl_coef, 1e-4, 10.0))

                    stats["kl_per_token"] = avg_kl
                    stats["ref_kl_coef"] = self.ref_kl_coef

                legal_exps = [e for e in exps if e.is_legal]
                coop = sum(1 for e in legal_exps if e.action == "stag") / max(len(legal_exps), 1)
                ill = 1.0 - (len(legal_exps) / max(len(exps), 1))

                stats.update(self._episode_action_conditioned_stats(exps, pid))
                stats.update({
                    f"p{pid}_mean_reward": float(np.mean(raw_rewards)) if raw_rewards else 0.0,
                    f"p{pid}_std_reward": float(np.std(raw_rewards)) if raw_rewards else 0.0,
                    f"p{pid}_stag_rate": coop,
                    f"p{pid}_illegal_rate": ill,
                    f"p{pid}_policy_loss": upd["policy_loss"],
                    f"p{pid}_value_loss": upd["value_loss"],
                    f"p{pid}_kl": upd["kl"],
                    f"p{pid}_entropy": upd["entropy"],
                })

            self.stats_history.append(stats)

            if episode % cfg.log_every == 0:
                parts = [f"Ep {episode:4d}/{cfg.num_episodes}"]
                for pid in self.llm_player_ids:
                    parts.append(f"P{pid} R:{stats[f'p{pid}_mean_reward']:+.2f} Stag:{stats[f'p{pid}_stag_rate']:.0%}")
                parts.append(f"KL(P0):{stats.get('p0_kl', 0.0):.4f}")
                print(" | ".join(parts))

            if episode % cfg.save_every == 0:
                ckpt_dir = os.path.join(output_path, f"checkpoint_{episode}")
                os.makedirs(ckpt_dir, exist_ok=True)
                self.tokenizer.save_pretrained(ckpt_dir)
                for pid in self.llm_player_ids:
                    self.player_models[pid].save_pretrained(os.path.join(ckpt_dir, f"player_{pid}"))

        final_dir = os.path.join(output_path, "final")
        os.makedirs(final_dir, exist_ok=True)
        self.tokenizer.save_pretrained(final_dir)
        for pid in self.llm_player_ids:
            self.player_models[pid].save_pretrained(os.path.join(final_dir, f"player_{pid}"))

        stats_path = os.path.join(output_path, "training_stats.json")
        with open(stats_path, "w") as f:
            json.dump(self.stats_history, f, indent=2)

        config_path = os.path.join(output_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(vars(cfg), f, indent=2)

        print("\nTraining complete!")
        print(f"Saved to: {output_path}")
        print(f"Stats: {stats_path}")
        return self.stats_history