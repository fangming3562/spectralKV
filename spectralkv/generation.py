from __future__ import annotations

from contextlib import nullcontext
from collections.abc import Sequence

import torch

from .attention_patch import apply_strict_attention_patch
from .cache_utils import assert_rectangular_cache


@torch.no_grad()
def greedy_generate_original_positions(
    *,
    model,
    tokenizer,
    past_key_values,
    next_logits: torch.Tensor,
    prompt_len: int,
    max_new_tokens: int,
    allow_nonrectangular_cache: bool = True,
    generated_prefix: list[int] | None = None,
    start_step: int = 0,
) -> tuple[list[int], str]:
    generated: list[int] = list(generated_prefix or [])
    past = past_key_values
    eos_token_id = tokenizer.eos_token_id
    patch_context = apply_strict_attention_patch(model, record_query_content=False) if bool(allow_nonrectangular_cache) else nullcontext()
    with torch.inference_mode(), patch_context:
        for step in range(int(start_step), int(max_new_tokens)):
            next_token = int(torch.argmax(next_logits, dim=-1).item())
            if eos_token_id is not None and next_token == int(eos_token_id):
                break
            generated.append(next_token)
            token_tensor = torch.tensor([[next_token]], device=next_logits.device, dtype=torch.long)
            attention_mask = None
            if not bool(allow_nonrectangular_cache):
                compressed_len = assert_rectangular_cache(past)
                attention_mask = torch.ones((1, int(compressed_len) + 1), device=next_logits.device, dtype=torch.long)
            original_pos = int(prompt_len) + int(step)
            out = model(
                input_ids=token_tensor,
                past_key_values=past,
                use_cache=True,
                cache_position=torch.tensor([original_pos], device=next_logits.device, dtype=torch.long),
                position_ids=torch.tensor([[original_pos]], device=next_logits.device, dtype=torch.long),
                attention_mask=attention_mask,
            )
            past = out.past_key_values
            next_logits = out.logits[:, -1, :]
    return generated, tokenizer.decode(generated, skip_special_tokens=True)


@torch.no_grad()
def greedy_generate_original_positions_batched(
    *,
    model,
    tokenizer,
    past_key_values,
    next_logits: torch.Tensor,
    prompt_lens: Sequence[int],
    max_new_tokens: int | Sequence[int],
) -> tuple[list[list[int]], list[str]]:
    batch_size = int(next_logits.shape[0])
    if batch_size != int(len(prompt_lens)):
        raise ValueError(f"Batch size mismatch: logits={batch_size} prompt_lens={len(prompt_lens)}.")
    if isinstance(max_new_tokens, int):
        max_steps_by_row = [int(max_new_tokens)] * batch_size
    else:
        max_steps_by_row = [int(value) for value in max_new_tokens]
        if batch_size != int(len(max_steps_by_row)):
            raise ValueError(
                f"Batch size mismatch: logits={batch_size} max_new_tokens={len(max_steps_by_row)}."
            )

    generated: list[list[int]] = [[] for _ in range(batch_size)]
    past = past_key_values
    eos_token_id = tokenizer.eos_token_id
    dummy_token_id = tokenizer.pad_token_id
    if dummy_token_id is None:
        dummy_token_id = eos_token_id if eos_token_id is not None else 0

    max_steps = max(max_steps_by_row, default=0)
    finished = torch.tensor(
        [int(limit) <= 0 for limit in max_steps_by_row],
        device=next_logits.device,
        dtype=torch.bool,
    )
    prompt_lens_tensor = torch.tensor(
        [int(length) for length in prompt_lens],
        device=next_logits.device,
        dtype=torch.long,
    )
    patch_context = apply_strict_attention_patch(model, record_query_content=False)
    with torch.inference_mode(), patch_context:
        for step in range(int(max_steps)):
            next_tokens = torch.argmax(next_logits, dim=-1)
            was_finished = finished.clone()

            for row_idx in range(batch_size):
                if bool(was_finished[int(row_idx)].item()) or int(step) >= int(max_steps_by_row[int(row_idx)]):
                    continue
                token_id = int(next_tokens[int(row_idx)].item())
                if eos_token_id is not None and token_id == int(eos_token_id):
                    continue
                generated[int(row_idx)].append(token_id)

            for row_idx in range(batch_size):
                if bool(was_finished[int(row_idx)].item()):
                    continue
                reached_limit = int(step) + 1 >= int(max_steps_by_row[int(row_idx)])
                hit_eos = eos_token_id is not None and int(next_tokens[int(row_idx)].item()) == int(eos_token_id)
                if bool(reached_limit or hit_eos):
                    finished[int(row_idx)] = True
            if bool(finished.all().item()):
                break

            model_input_ids = next_tokens.clone()
            model_input_ids[finished] = int(dummy_token_id)
            position_ids = (prompt_lens_tensor + int(step)).view(batch_size, 1)
            out = model(
                input_ids=model_input_ids.unsqueeze(1),
                past_key_values=past,
                use_cache=True,
                position_ids=position_ids,
                attention_mask=None,
            )
            past = out.past_key_values
            next_logits = out.logits[:, -1, :]
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return generated, [str(text) for text in texts]
