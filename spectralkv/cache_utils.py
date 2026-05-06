from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import DynamicCache


def extract_cache_layers(past_key_values) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(past_key_values, "layers"):
        return [(layer.keys, layer.values) for layer in past_key_values.layers]
    if isinstance(past_key_values, DynamicCache):
        if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
            return list(zip(past_key_values.key_cache, past_key_values.value_cache))
        return [(layer[0], layer[1]) for layer in past_key_values]
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))
    if isinstance(past_key_values, tuple):
        return [(layer[0], layer[1]) for layer in past_key_values]
    raise TypeError(f"Unsupported cache type: {type(past_key_values)}")


def num_cache_layers(past_key_values) -> int:
    if hasattr(past_key_values, "layers"):
        return int(len(past_key_values.layers))
    if isinstance(past_key_values, DynamicCache):
        if hasattr(past_key_values, "key_cache"):
            return int(len(past_key_values.key_cache))
        return int(len(past_key_values))
    if hasattr(past_key_values, "key_cache"):
        return int(len(past_key_values.key_cache))
    if isinstance(past_key_values, tuple):
        return int(len(past_key_values))
    raise TypeError(f"Unsupported cache type: {type(past_key_values)}")


def get_layer_cache(past_key_values, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one cache layer without materializing references to every layer."""

    idx = int(layer_idx)
    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[idx]
        return layer.keys, layer.values
    if isinstance(past_key_values, DynamicCache):
        if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
            return past_key_values.key_cache[idx], past_key_values.value_cache[idx]
        layer = past_key_values[idx]
        return layer[0], layer[1]
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return past_key_values.key_cache[idx], past_key_values.value_cache[idx]
    if isinstance(past_key_values, tuple):
        layer = past_key_values[idx]
        return layer[0], layer[1]
    raise TypeError(f"Unsupported cache type: {type(past_key_values)}")


def cache_layer_length(past_key_values, layer_idx: int = 0) -> int:
    return int(get_layer_cache(past_key_values, int(layer_idx))[0].shape[-2])


def set_layer_cache(past_key_values, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> None:
    """Mutate one layer of a HuggingFace DynamicCache-like object."""

    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[int(layer_idx)]
        if not hasattr(layer, "keys") or not hasattr(layer, "values"):
            raise TypeError(f"Cache layer {layer_idx} does not expose keys/values fields.")
        layer.keys = keys
        layer.values = values
        if hasattr(layer, "is_initialized"):
            layer.is_initialized = True
        if hasattr(layer, "dtype"):
            layer.dtype = keys.dtype
        if hasattr(layer, "device"):
            layer.device = keys.device
        if hasattr(layer, "cumulative_length"):
            layer.cumulative_length = int(keys.shape[-2])
        return
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        past_key_values.key_cache[int(layer_idx)] = keys
        past_key_values.value_cache[int(layer_idx)] = values
        return
    raise TypeError(f"Unsupported mutable cache type: {type(past_key_values)}")


def set_layer_log_bias(past_key_values, layer_idx: int, log_bias: torch.Tensor | None) -> None:
    """Attach per-retained-token additive attention log-bias to a cache layer."""

    if log_bias is None:
        return
    if not hasattr(past_key_values, "strictmerge_log_biases"):
        past_key_values.strictmerge_log_biases = {}
    past_key_values.strictmerge_log_biases[int(layer_idx)] = log_bias.detach()


def build_dynamic_cache(layer_pairs: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> DynamicCache:
    cache = DynamicCache()
    for layer_idx, (keys, values) in enumerate(layer_pairs):
        cache.update(keys, values, int(layer_idx))
    return cache


def build_dynamic_cache_no_copy(layer_pairs: Sequence[tuple[torch.Tensor, torch.Tensor]], *, config=None) -> DynamicCache:
    """Build a DynamicCache by directly binding layer tensors when possible."""

    try:
        cache = DynamicCache(config=config) if config is not None else DynamicCache()
        if not hasattr(cache, "layers") or len(getattr(cache, "layers", [])) < len(layer_pairs):
            raise TypeError("DynamicCache was not pre-initialized with enough layers.")
        for layer_idx, (keys, values) in enumerate(layer_pairs):
            set_layer_cache(cache, int(layer_idx), keys, values)
        return cache
    except Exception:
        return build_dynamic_cache(layer_pairs)


def build_padded_batch_cache(past_key_values_list: Sequence) -> DynamicCache:
    """Pad per-sample compressed caches into one batched DynamicCache.

    Each input cache is expected to have batch size 1. Layers may have different
    retained lengths both within and across samples. The returned cache stores a
    per-layer boolean mask in ``strictmerge_padding_masks`` so the attention
    patch can hide padded KV slots during batched decode.
    """

    if not past_key_values_list:
        raise ValueError("past_key_values_list must not be empty.")
    layer_lists = [extract_cache_layers(cache) for cache in past_key_values_list]
    num_layers = len(layer_lists[0])
    if any(len(layers) != num_layers for layers in layer_lists):
        raise ValueError("All caches must have the same number of layers.")

    padding_masks: dict[int, torch.Tensor] = {}
    padded_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    padded_log_biases: dict[int, torch.Tensor] = {}
    batch_size = len(layer_lists)
    for layer_idx in range(num_layers):
        keys_by_sample = [layers[layer_idx][0] for layers in layer_lists]
        values_by_sample = [layers[layer_idx][1] for layers in layer_lists]
        if any(int(keys.shape[0]) != 1 or int(values.shape[0]) != 1 for keys, values in zip(keys_by_sample, values_by_sample)):
            raise NotImplementedError("build_padded_batch_cache expects batch_size=1 input caches.")
        max_len = max(int(keys.shape[-2]) for keys in keys_by_sample)
        padded_keys: list[torch.Tensor] = []
        padded_values: list[torch.Tensor] = []
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=keys_by_sample[0].device)
        layer_has_log_bias = any(
            isinstance(getattr(cache, "strictmerge_log_biases", None), dict)
            and isinstance(getattr(cache, "strictmerge_log_biases", {}).get(int(layer_idx)), torch.Tensor)
            for cache in past_key_values_list
        )
        reference_bias_heads = None
        if layer_has_log_bias:
            for cache in past_key_values_list:
                biases = getattr(cache, "strictmerge_log_biases", None)
                bias = biases.get(int(layer_idx)) if isinstance(biases, dict) else None
                if isinstance(bias, torch.Tensor):
                    reference_bias_heads = int(bias.shape[1])
                    break
        padded_biases: list[torch.Tensor] = []
        for sample_idx, (keys, values) in enumerate(zip(keys_by_sample, values_by_sample)):
            length = int(keys.shape[-2])
            pad_len = int(max_len) - int(length)
            if pad_len > 0:
                key_pad = torch.zeros((*keys.shape[:-2], pad_len, keys.shape[-1]), dtype=keys.dtype, device=keys.device)
                value_pad = torch.zeros((*values.shape[:-2], pad_len, values.shape[-1]), dtype=values.dtype, device=values.device)
                keys = torch.cat([keys, key_pad], dim=-2)
                values = torch.cat([values, value_pad], dim=-2)
            padded_keys.append(keys)
            padded_values.append(values)
            mask[int(sample_idx), :length] = True
            if layer_has_log_bias:
                biases = getattr(past_key_values_list[int(sample_idx)], "strictmerge_log_biases", None)
                bias = biases.get(int(layer_idx)) if isinstance(biases, dict) else None
                if isinstance(bias, torch.Tensor):
                    bias = bias.to(device=keys.device, dtype=torch.float32)
                    if int(bias.dim()) != 3 or int(bias.shape[0]) != 1:
                        raise RuntimeError(f"StrictMerge log bias must have shape [1,H,K], got {tuple(bias.shape)}.")
                    if reference_bias_heads is not None and int(bias.shape[1]) != int(reference_bias_heads):
                        raise RuntimeError(
                            f"StrictMerge log bias head mismatch at layer {layer_idx}: "
                            f"bias={tuple(bias.shape)} expected_heads={reference_bias_heads}."
                        )
                    if int(bias.shape[-1]) != length:
                        raise RuntimeError(
                            f"StrictMerge log bias length mismatch at layer {layer_idx}: "
                            f"bias={tuple(bias.shape)} keys={tuple(keys.shape)}."
                        )
                else:
                    bias_heads = int(reference_bias_heads) if reference_bias_heads is not None else int(keys.shape[1])
                    bias = torch.zeros((1, bias_heads, length), dtype=torch.float32, device=keys.device)
                if pad_len > 0:
                    bias_pad = torch.zeros((1, int(bias.shape[1]), pad_len), dtype=bias.dtype, device=bias.device)
                    bias = torch.cat([bias, bias_pad], dim=-1)
                padded_biases.append(bias.contiguous())
        padded_pairs.append(
            (
                torch.cat(padded_keys, dim=0).contiguous(),
                torch.cat(padded_values, dim=0).contiguous(),
            )
        )
        padding_masks[int(layer_idx)] = mask
        if layer_has_log_bias:
            padded_log_biases[int(layer_idx)] = torch.cat(padded_biases, dim=0).contiguous()
    cache = build_dynamic_cache(padded_pairs)
    cache.strictmerge_padding_masks = padding_masks
    if padded_log_biases:
        cache.strictmerge_log_biases = padded_log_biases
    return cache


def clone_cache(past_key_values, *, detach: bool = True) -> DynamicCache:
    """Clone a DynamicCache-like object into an independent DynamicCache."""

    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for keys, values in extract_cache_layers(past_key_values):
        if detach:
            keys = keys.detach()
            values = values.detach()
        pairs.append((keys.clone(), values.clone()))
    return build_dynamic_cache(pairs)


def shallow_dynamic_cache(past_key_values, *, config=None) -> DynamicCache:
    """Build an independent DynamicCache object sharing existing KV tensors.

    This is useful for one-token branch rollouts and budget probes: appending to
    the returned cache rebinds that cache layer's tensors, while the original
    prompt cache object remains untouched.
    """

    cache = DynamicCache(config=config)
    for layer_idx, (keys, values) in enumerate(extract_cache_layers(past_key_values)):
        set_layer_cache(cache, int(layer_idx), keys, values)
    return cache


def assert_rectangular_cache(past_key_values) -> int:
    lengths = [int(keys.shape[-2]) for keys, _ in extract_cache_layers(past_key_values)]
    if not lengths:
        raise ValueError("Cache has no layers.")
    unique = sorted(set(lengths))
    if len(unique) != 1:
        raise ValueError(f"Expected rectangular cache lengths, got {unique}")
    return int(unique[0])
