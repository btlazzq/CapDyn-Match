"""Frozen reference-encoder text and last-non-padding hidden states.

Every historical candidate of a question is encoded by the same frozen model.
The encoded string is the chat-formatted query plus the complete candidate
response. The representation is the hidden state at the configured layer and
the last non-padding token. HF hidden_states[0] is the embedding layer, so the
transformer block `layer_index` is hidden_states[layer_index + 1].
"""

from __future__ import annotations

from typing import Optional

import numpy as np

MATH_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
EVALPLUS_INSTRUCTION = (
    "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
)
EVALPLUS_RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"
)
EVALPLUS_MAGIC_SPLITTER = "-[[]]-this-is-really-our-highest-priority-[[]]-"


def last_nonpad_index(attention_mask):
    import torch

    attn = attention_mask if isinstance(attention_mask, torch.Tensor) else torch.as_tensor(attention_mask)
    flipped = attn.flip(dims=[1])
    last_from_end = flipped.argmax(dim=1)
    return attn.shape[1] - 1 - last_from_end


def hidden_at_last_nonpad(hidden_states, attention_mask, layer_index: int):
    import torch

    hs = hidden_states[layer_index + 1]
    idx = last_nonpad_index(attention_mask)
    batch = torch.arange(hs.shape[0], device=hs.device)
    return hs[batch, idx]


def _apply_chat(tokenizer, messages, add_generation_prompt: bool = True, enable_thinking=None) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt, add_special_tokens=True)
    if enable_thinking is not None:
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
        except TypeError:
            pass
    return tokenizer.apply_chat_template(messages, **kwargs)


def build_prompt(tokenizer, question: str, chat_mode: str = "math_system", extra: Optional[dict] = None, system_prompt: str = MATH_SYSTEM_PROMPT) -> str:
    """Chat-formatted prompt with the generation prefix, and no response yet."""
    extra = extra or {}
    mode = chat_mode or "math_system"
    if mode == "math_system":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        return _apply_chat(tokenizer, messages, add_generation_prompt=True)
    if mode == "user_only":
        user = extra.get("encoder_user") or question
        return _apply_chat(tokenizer, [{"role": "user", "content": user}], add_generation_prompt=True)
    if mode == "evalplus_prefill":
        task_prompt = extra.get("task_prompt") or question
        user = f"{EVALPLUS_INSTRUCTION}\n```\n{(task_prompt or '').strip()}\n```\n"
        response = f"{EVALPLUS_RESPONSE_PREFIX}\n```python\n{EVALPLUS_MAGIC_SPLITTER}\n```\n"
        rendered = _apply_chat(
            tokenizer,
            [{"role": "user", "content": user}, {"role": "assistant", "content": response}],
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return rendered.split(EVALPLUS_MAGIC_SPLITTER)[0]
    if mode == "lcb_codeqwen":
        prefix = extra.get("encoder_prefix")
        if not prefix:
            raise ValueError("lcb_codeqwen requires encoder_prefix on the record")
        return prefix
    raise ValueError(f"unknown chat_mode {mode}")


def build_encoder_text(
    tokenizer,
    question: str,
    response: str,
    chat_mode: str = "math_system",
    extra: Optional[dict] = None,
    system_prompt: str = MATH_SYSTEM_PROMPT,
) -> str:
    prompt = build_prompt(tokenizer, question, chat_mode=chat_mode, extra=extra, system_prompt=system_prompt)
    return prompt + (response or "")


def load_encoder(model_path: str, dtype_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tokenizer, model


def encode_texts(model, tokenizer, texts: list[str], device: str, max_length: int, layer_index: int) -> np.ndarray:
    import torch

    try:
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True, use_cache=False)
            h = hidden_at_last_nonpad(out.hidden_states, enc["attention_mask"], layer_index)
            arr = h.float().cpu().numpy()
        del out, enc
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(texts) == 1:
            raise
        mid = max(1, len(texts) // 2)
        left = encode_texts(model, tokenizer, texts[:mid], device, max_length, layer_index)
        right = encode_texts(model, tokenizer, texts[mid:], device, max_length, layer_index)
        return np.concatenate([left, right], axis=0)
    if not np.isfinite(arr).all():
        raise RuntimeError("non-finite hidden state")
    return arr
