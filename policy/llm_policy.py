from __future__ import annotations

from typing import List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import TrainConfig
from policy.utils import parse_action


class LLMPolicy:
    """
    Simple policy wrapper around a causal LM.

    If adapter_dir is provided, we load the model and tokenizer from there
    (this is where PPOTrainer.save_pretrained() writes the fine-tuned model).
    Otherwise we load the base pretrained model from cfg.model_name.
    """

    def __init__(self, cfg: TrainConfig, adapter_dir: Optional[str] = None):
        self.cfg = cfg
        model_path = adapter_dir or cfg.model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding is fine here since we slice completions using
        # input length, which is the same for all batch elements.
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    @torch.inference_mode()
    def act(self, prompts: List[str]) -> Tuple[List[str], List[str], None, None]:
        """
        Generate actions for a list of prompts, returning:
          - raw completions (strings)
          - parsed actions ('C' or 'D', default 'D' on parse failure)
        """
        device = self.model.device
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        texts, actions = [], []
        input_len = inputs["input_ids"].shape[1]

        for i in range(gen.size(0)):
            completion_ids = gen[i, input_len:]
            completion = self.tokenizer.decode(
                completion_ids, skip_special_tokens=True
            )
            texts.append(completion)
            actions.append(parse_action(completion))

        return texts, actions, None, None