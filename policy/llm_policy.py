from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import List, Tuple, Optional

from config import TrainConfig
from policy.utils import parse_action, _parse_action_strict

class LLMPolicy:
    def __init__(self, cfg: TrainConfig, adapter_dir: Optional[str] = None):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_dir or cfg.model_name, trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        base = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        if adapter_dir is not None:
            self.model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
        else:
            self.model = base
        self.model.eval()

    @torch.inference_mode()
    def act(self, prompts: List[str]) -> Tuple[List[str], List[str], None, None]:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        texts, actions = [], []
        attn = inputs["attention_mask"]
        for i in range(gen.size(0)):
            prompt_len = int(attn[i].sum().item())
            completion_ids = gen[i, prompt_len:]
            completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            texts.append(completion)
            
            # TODO: Check if the parse_action really has to be liberal or if strict parsing also works

            actions.append(parse_action(completion))
        return texts, actions, None, None
    
