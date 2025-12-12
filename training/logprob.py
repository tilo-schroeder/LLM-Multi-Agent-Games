import torch

from stag_hunt_grpo.data_structures import DecisionSample


def compute_logprob_for_sample(
    model,
    tokenizer,
    sample: DecisionSample,
    device: torch.device,
    max_length: int = 512,
) -> torch.Tensor:
    """
    Compute mean log p(completion | prompt) for a single DecisionSample.

    Returns:
        logprob: scalar tensor (requires_grad=True)
    """
    full_text = sample.prompt + sample.completion

    enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Figure out where the completion starts
    prompt_enc = tokenizer(
        sample.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)
    prompt_len = prompt_enc["input_ids"].shape[1]

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits  # [1, seq_len, vocab]

    # Standard next-token shift
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_attn = attention_mask[:, 1:].contiguous()

    # Mask: only completion tokens (everything after prompt)
    completion_mask = torch.zeros_like(shift_attn)
    completion_mask[:, prompt_len - 1:] = 1  # from last prompt token onward

    log_probs_all = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs_all.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)  # [1, seq_len]

    mask = shift_attn * completion_mask
    token_log_probs = token_log_probs * mask

    num_tokens = mask.sum().clamp(min=1)
    logprob = token_log_probs.sum() / num_tokens  # mean logprob over completion

    return logprob  # scalar with grad