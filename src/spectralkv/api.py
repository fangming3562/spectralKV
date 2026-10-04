"""Public single-prefill, multiple-budget greedy generation API."""

import time

import torch
from transformers import DynamicCache

from .config import Config
from .runtime import DynamicRuntime, FixedRuntime


def cache_metadata(runtime, percentage):
    layers = runtime.packs[percentage]
    size = sum(
        getattr(p, field).numel() * getattr(p, field).element_size()
        for p in layers.values()
        for field in ("k", "v", "b", "meta")
    )
    heads = runtime.model.config.num_key_value_heads
    cap = (runtime.n * percentage // 100) * len(layers) * heads * runtime.bpt
    if size > cap:
        raise RuntimeError("Packed cache exceeded the byte budget.")
    return dict(
        kv_bytes=size,
        cap_bytes=cap,
        actual_ratio=size / (runtime.n * len(layers) * heads * runtime.bpt),
        counts=[p.meta[:, 1].cpu().tolist() for p in layers.values()],
    )


@torch.inference_mode()
def generate(model, tokenizer, input_ids, *, config=Config(), max_new_tokens=128):
    """Reuse one full prefill for all percentages, then decode each independently.

    The first generated token is shared from the full prefill. Subsequent KV
    rows are exact and excluded from the compressed *prompt* byte budget.
    The tokenizer's prompt/chat formatting is owned by the caller.
    """
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer.")
    if model.device.type != "cuda":
        raise ValueError("The validated runtime requires a CUDA model.")
    runtime_cls = DynamicRuntime if config.layer_budget == "dynamic" else FixedRuntime
    torch.cuda.synchronize(model.device)
    start = time.perf_counter()
    with runtime_cls(model, input_ids, tokenizer, config) as runtime:
        out = runtime.prefill()
        initial = out.logits[:, -1].argmax(-1, keepdim=True)
        del out  # Dynamic full prompt cache is unnecessary after construction.
        torch.cuda.synchronize(model.device)
        ready_seconds = time.perf_counter() - start
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        results = {}
        for pct in config.percentages:
            metadata = cache_metadata(runtime, pct)
            runtime.fast_prefix = runtime.packs[pct]
            runtime.mode = "compressed"
            runtime.decode_calls = 0
            suffix = DynamicCache(config=model.config)
            token = initial
            tokens = [int(initial.item())]
            for t in range(max_new_tokens - 1):
                if tokens[-1] in eos:
                    break
                position = torch.tensor([runtime.n + t], device=model.device)
                out = model(
                    token,
                    past_key_values=suffix,
                    position_ids=position[None],
                    cache_position=position,
                    use_cache=True,
                    logits_to_keep=1,
                )
                token = out.logits[:, -1].argmax(-1, keepdim=True)
                tokens.append(int(token.item()))
                del out
            results[str(pct)] = dict(
                token_ids=tokens,
                prediction=tokenizer.decode(tokens, skip_special_tokens=True),
                generated=len(tokens),
                layer_budget=config.layer_budget,
                decode_calls=runtime.decode_calls,
                selection_forwards=1,
                **metadata,
            )
        return dict(
            prefill_and_compression_seconds=ready_seconds,
            prompt_tokens=runtime.n,
            percentages=list(config.percentages),
            results=results,
        )
