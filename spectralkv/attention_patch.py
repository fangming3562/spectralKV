from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MethodType
from typing import Any, Generator

import torch
from .cache_utils import set_layer_cache, set_layer_log_bias
try:
    from transformers.models.llama.modeling_llama import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )
except ImportError:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    ALL_ATTENTION_FUNCTIONS = None

    def eager_attention_forward(
        module,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        dropout: float = 0.0,
        scaling: float | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_states = repeat_kv(key_states, int(module.num_key_value_groups))
        value_states = repeat_kv(value_states, int(module.num_key_value_groups))
        scale = float(scaling) if scaling is not None else float(module.scaling)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        if float(dropout) > 0.0:
            attn_weights = torch.dropout(attn_weights, p=float(dropout), train=bool(module.training))
        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
        return attn_output, attn_weights


def attention_interface(config):
    if ALL_ATTENTION_FUNCTIONS is None:
        return eager_attention_forward
    get_interface = getattr(ALL_ATTENTION_FUNCTIONS, "get_interface", None)
    if callable(get_interface):
        return get_interface(config._attn_implementation, eager_attention_forward)
    if config._attn_implementation != "eager":
        return ALL_ATTENTION_FUNCTIONS[config._attn_implementation]
    return eager_attention_forward


try:
    from flash_attn import flash_attn_varlen_func
except Exception:
    flash_attn_varlen_func = None


def _cached_num_heads(past_key_values, layer_idx: int) -> int | None:
    try:
        if hasattr(past_key_values, "layers"):
            layer = past_key_values.layers[int(layer_idx)]
            keys = getattr(layer, "keys", None)
            if keys is not None and int(keys.numel()) > 0:
                return int(keys.shape[1])
        if hasattr(past_key_values, "key_cache"):
            keys = past_key_values.key_cache[int(layer_idx)]
            if keys is not None and int(keys.numel()) > 0:
                return int(keys.shape[1])
    except Exception:
        return None
    return None


def _repeat_kv_for_query_heads(states: torch.Tensor, groups: int) -> torch.Tensor:
    groups = max(int(groups), 1)
    if groups == 1:
        return states
    bsz, hkv, seq_len, head_dim = states.shape
    return (
        states[:, :, None, :, :]
        .expand(int(bsz), int(hkv), int(groups), int(seq_len), int(head_dim))
        .reshape(int(bsz), int(hkv) * int(groups), int(seq_len), int(head_dim))
        .contiguous()
    )


def _apply_optional_head_norm(module: Any, states: torch.Tensor, attr_name: str) -> torch.Tensor:
    norm = getattr(module, attr_name, None)
    if norm is None:
        return states
    return norm(states)


def _attention_forward_no_repeat(
    *,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * float(scaling)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _ragged_flash_decode_forward(
    *,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    padding_mask: torch.Tensor,
    scaling: float,
) -> torch.Tensor | None:
    if flash_attn_varlen_func is None:
        return None
    if int(query_states.shape[-2]) != 1:
        return None
    if query_states.dtype not in {torch.float16, torch.bfloat16}:
        return None
    batch_size = int(query_states.shape[0])
    key_len = int(key_states.shape[-2])
    valid_counts = padding_mask.to(device=query_states.device, dtype=torch.bool).sum(dim=-1).to(torch.int32)
    if int(valid_counts.min().item()) <= 0:
        return None
    max_seqlen_k = int(valid_counts.max().item())
    q = query_states.transpose(1, 2).contiguous()
    k = key_states.transpose(1, 2).contiguous()
    v = value_states.transpose(1, 2).contiguous()
    flat_mask = padding_mask.reshape(batch_size * key_len).to(device=query_states.device, dtype=torch.bool)
    valid_idx = torch.nonzero(flat_mask, as_tuple=False).flatten()
    k_unpad = k.reshape(batch_size * key_len, int(k.shape[2]), int(k.shape[3])).index_select(0, valid_idx)
    v_unpad = v.reshape(batch_size * key_len, int(v.shape[2]), int(v.shape[3])).index_select(0, valid_idx)
    q_unpad = q.reshape(batch_size, int(q.shape[1]), int(q.shape[2]), int(q.shape[3])).reshape(
        batch_size * int(q.shape[1]), int(q.shape[2]), int(q.shape[3])
    )
    cu_q = torch.arange(
        0,
        (batch_size + 1) * int(q.shape[1]),
        int(q.shape[1]),
        device=query_states.device,
        dtype=torch.int32,
    )
    cu_k = torch.empty((batch_size + 1,), device=query_states.device, dtype=torch.int32)
    cu_k[0] = 0
    cu_k[1:] = torch.cumsum(valid_counts, dim=0)
    out = flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_q,
        cu_k,
        max_seqlen_q=int(q.shape[1]),
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=float(scaling),
        causal=False,
    )
    return out.reshape(batch_size, int(q.shape[1]), int(q.shape[2]), int(q.shape[3])).contiguous()


def _strictmerge_layer_log_bias(past_key_values, layer_idx: int) -> torch.Tensor | None:
    biases = getattr(past_key_values, "strictmerge_log_biases", None)
    if not isinstance(biases, dict):
        return None
    bias = biases.get(int(layer_idx))
    if bias is None:
        return None
    if not isinstance(bias, torch.Tensor):
        return None
    return bias


def _strictmerge_layer_padding_mask(
    past_key_values,
    layer_idx: int,
    *,
    key_len: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    masks = getattr(past_key_values, "strictmerge_padding_masks", None)
    if not isinstance(masks, dict):
        return None
    mask = masks.get(int(layer_idx))
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        return None
    mask = mask.to(device=device, dtype=torch.bool)
    if int(mask.dim()) != 2:
        raise RuntimeError(f"StrictMerge padding mask must have shape [B,K], got {tuple(mask.shape)}.")
    if int(mask.shape[0]) != int(batch_size):
        raise RuntimeError(
            f"StrictMerge padding mask batch mismatch at layer {layer_idx}: "
            f"mask={tuple(mask.shape)} query_batch={batch_size}."
        )
    mask_len = int(mask.shape[-1])
    if mask_len < int(key_len):
        pad = torch.ones(
            (int(batch_size), int(key_len) - int(mask_len)),
            dtype=torch.bool,
            device=device,
        )
        mask = torch.cat([mask, pad], dim=-1).contiguous()
        masks[int(layer_idx)] = mask
    elif mask_len > int(key_len):
        mask = mask[:, : int(key_len)].contiguous()
        masks[int(layer_idx)] = mask
    return mask


@dataclass
class StrictQueryContentState:
    record_query_content: bool = False
    record_full_query_content: bool = False
    record_tail_window: int = 16
    record_decode_queries: bool = False
    layer_query_contents: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    layer_content_positions: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    layer_batch_query_contents: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    layer_batch_content_positions: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    layer_decode_queries: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    layer_decode_positions: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    prefill_cache_compressor: Any = None
    integrated_prefill_debug: dict[int, dict[str, Any]] = field(default_factory=dict)

    def record_content(self, layer_idx: int, query_content_states: torch.Tensor, positions: torch.Tensor) -> None:
        if not bool(self.record_query_content):
            return
        batch_size = int(query_content_states.shape[0])
        q_len = int(query_content_states.shape[2])
        positions = positions.to(device=query_content_states.device, dtype=torch.long)
        if int(positions.dim()) == 1:
            if int(positions.numel()) != q_len:
                raise ValueError("Recorded content positions must match q_len.")
            batch_positions = positions.unsqueeze(0).expand(batch_size, q_len)
        elif int(positions.dim()) == 2:
            if int(positions.shape[0]) != batch_size or int(positions.shape[1]) != q_len:
                raise ValueError(
                    f"Recorded batched content positions must have shape [B,q_len], "
                    f"got {tuple(positions.shape)} for content {tuple(query_content_states.shape)}."
                )
            batch_positions = positions
        else:
            raise ValueError(f"Recorded content positions must be 1D or 2D, got {tuple(positions.shape)}.")
        if bool(self.record_full_query_content):
            keep_slice = slice(None)
        else:
            width = min(max(int(self.record_tail_window), 1), q_len)
            keep_slice = slice(-width, None)
        kept_content = query_content_states[:, :, keep_slice, :].detach().contiguous()
        kept_positions = batch_positions[:, keep_slice].detach().contiguous()
        if batch_size == 1:
            self.layer_query_contents.setdefault(int(layer_idx), []).append(kept_content[0].contiguous())
            self.layer_content_positions.setdefault(int(layer_idx), []).append(kept_positions[0].contiguous())
            return
        self.layer_batch_query_contents.setdefault(int(layer_idx), []).append(kept_content)
        self.layer_batch_content_positions.setdefault(int(layer_idx), []).append(kept_positions)

    def sample_state(self, sample_idx: int, prompt_len: int) -> "StrictQueryContentState":
        out = StrictQueryContentState(
            record_query_content=bool(self.record_query_content),
            record_full_query_content=bool(self.record_full_query_content),
            record_tail_window=int(self.record_tail_window),
            record_decode_queries=bool(self.record_decode_queries),
            prefill_cache_compressor=self.prefill_cache_compressor,
        )
        if not self.layer_batch_query_contents:
            if int(sample_idx) != 0:
                raise IndexError("Only sample 0 is available in an unbatched StrictQueryContentState.")
            out.layer_query_contents = {
                int(layer_idx): [item.detach().contiguous() for item in items]
                for layer_idx, items in self.layer_query_contents.items()
            }
            out.layer_content_positions = {
                int(layer_idx): [item.detach().contiguous() for item in items]
                for layer_idx, items in self.layer_content_positions.items()
            }
            return out

        for layer_idx, items in self.layer_batch_query_contents.items():
            pos_items = self.layer_batch_content_positions.get(int(layer_idx), [])
            if len(items) != len(pos_items):
                raise RuntimeError(f"Recorded batched content/position count mismatch for layer {layer_idx}.")
            for content, positions in zip(items, pos_items):
                if int(sample_idx) < 0 or int(sample_idx) >= int(content.shape[0]):
                    raise IndexError(f"sample_idx={sample_idx} is out of range for content {tuple(content.shape)}.")
                width = min(max(int(prompt_len), 0), int(content.shape[2]))
                if width <= 0:
                    continue
                out.layer_query_contents.setdefault(int(layer_idx), []).append(
                    content[int(sample_idx), :, -width:, :].detach().contiguous()
                )
                out.layer_content_positions.setdefault(int(layer_idx), []).append(
                    positions[int(sample_idx), -width:].detach().contiguous()
                )
        return out

    def concat_content(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        items = self.layer_query_contents.get(int(layer_idx), [])
        if not items:
            return None
        pos_items = self.layer_content_positions.get(int(layer_idx), [])
        if len(items) != len(pos_items):
            raise RuntimeError(f"Recorded content/position count mismatch for layer {layer_idx}.")
        return torch.cat(items, dim=1).contiguous(), torch.cat(pos_items, dim=0).contiguous()

    def record_decode(self, layer_idx: int, query_states: torch.Tensor, positions: torch.Tensor) -> None:
        if not bool(self.record_decode_queries):
            return
        positions = positions.to(device=query_states.device, dtype=torch.long).flatten()
        if int(positions.numel()) != int(query_states.shape[2]):
            raise ValueError("Recorded decode query positions must match q_len.")
        self.layer_decode_queries.setdefault(int(layer_idx), []).append(query_states[0].detach().clone())
        self.layer_decode_positions.setdefault(int(layer_idx), []).append(positions.detach().clone())

    def concat_decode(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        items = self.layer_decode_queries.get(int(layer_idx), [])
        if not items:
            return None
        pos_items = self.layer_decode_positions.get(int(layer_idx), [])
        if len(items) != len(pos_items):
            raise RuntimeError(f"Recorded decode query/position count mismatch for layer {layer_idx}.")
        return torch.cat(items, dim=1).contiguous(), torch.cat(pos_items, dim=0).contiguous()


def make_strict_llama_attention_forward(state: StrictQueryContentState):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        query_states = _apply_optional_head_norm(self, query_states, "q_norm").transpose(1, 2)
        key_states = _apply_optional_head_norm(self, key_states, "k_norm").transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_content_states = query_states

        if position_embeddings is None:
            raise ValueError("StrictMerge requires position_embeddings from model forward.")
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        q_len = int(hidden_states.shape[1])
        content_positions = None
        needs_prefill_content = q_len > 1 and (
            bool(state.record_query_content) or callable(state.prefill_cache_compressor)
        )
        if needs_prefill_content:
            position_ids = kwargs.get("position_ids")
            if isinstance(position_ids, torch.Tensor) and int(position_ids.dim()) == 2 and int(position_ids.shape[-1]) >= q_len:
                content_positions = position_ids[:, -q_len:].to(device=query_states.device, dtype=torch.long)
            elif cache_position is not None and int(cache_position.numel()) >= q_len:
                content_positions = cache_position[-q_len:]
            else:
                content_positions = torch.arange(q_len, device=query_states.device, dtype=torch.long)
        if q_len > 1 and bool(state.record_query_content):
            if content_positions is None:
                raise RuntimeError("Internal error: missing prefill content positions.")
            state.record_content(int(self.layer_idx), query_content_states, content_positions)
        elif q_len == 1 and bool(state.record_decode_queries):
            if cache_position is not None and int(cache_position.numel()) >= 1:
                decode_positions = cache_position[-1:].to(device=query_states.device, dtype=torch.long)
            else:
                decode_positions = torch.zeros((1,), device=query_states.device, dtype=torch.long)
            state.record_decode(int(self.layer_idx), query_states, decode_positions)

        expanded_query_head_cache = False
        integrated_prefill_compress = (
            past_key_values is not None
            and q_len > 1
            and callable(state.prefill_cache_compressor)
        )
        if past_key_values is not None:
            if not bool(integrated_prefill_compress):
                cached_heads = _cached_num_heads(past_key_values, int(self.layer_idx))
                if (
                    cached_heads is not None
                    and int(cached_heads) == int(query_states.shape[1])
                    and int(key_states.shape[1]) != int(query_states.shape[1])
                ):
                    key_states = _repeat_kv_for_query_heads(key_states, int(self.num_key_value_groups))
                    value_states = _repeat_kv_for_query_heads(value_states, int(self.num_key_value_groups))
                    expanded_query_head_cache = True
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
                expanded_query_head_cache = expanded_query_head_cache or int(key_states.shape[1]) == int(query_states.shape[1])
            else:
                deferred_prefill_cache_update = None
                if content_positions is None:
                    raise RuntimeError("Internal error: missing integrated prefill positions.")
                compress_query_content_states = query_content_states
                compress_content_positions = content_positions
                if not bool(state.record_full_query_content):
                    tail_width = min(
                        max(int(state.record_tail_window), 1),
                        int(query_content_states.shape[2]),
                    )
                    compress_query_content_states = query_content_states[:, :, -tail_width:, :].detach().contiguous()
                    if isinstance(content_positions, torch.Tensor) and int(content_positions.dim()) == 2:
                        compress_content_positions = content_positions[:, -tail_width:].detach().contiguous()
                    else:
                        compress_content_positions = content_positions[-tail_width:].detach().contiguous()
                    query_content_states = compress_query_content_states
                compressed_keys, compressed_values, log_bias, debug_item = state.prefill_cache_compressor(
                    layer_idx=int(self.layer_idx),
                    prompt_keys=key_states,
                    prompt_values=value_states,
                    query_content_states=compress_query_content_states,
                    content_positions=compress_content_positions,
                    module=self,
                )
                if isinstance(debug_item, dict) and callable(debug_item.get("_defer_cache_update")):
                    deferred_prefill_cache_update = debug_item.pop("_defer_cache_update")
                    if isinstance(debug_item, dict):
                        state.integrated_prefill_debug[int(self.layer_idx)] = debug_item
                else:
                    set_layer_cache(past_key_values, int(self.layer_idx), compressed_keys, compressed_values)
                    if isinstance(log_bias, torch.Tensor):
                        set_layer_log_bias(past_key_values, int(self.layer_idx), log_bias)
                    if isinstance(debug_item, dict):
                        state.integrated_prefill_debug[int(self.layer_idx)] = debug_item

        log_bias = None
        if past_key_values is not None and not bool(integrated_prefill_compress):
            log_bias = _strictmerge_layer_log_bias(past_key_values, int(self.layer_idx))
            if log_bias is not None:
                log_bias = log_bias.to(device=query_states.device, dtype=torch.float32)
                if int(log_bias.dim()) != 3:
                    raise RuntimeError(f"StrictMerge log bias must have shape [B,H,K], got {tuple(log_bias.shape)}.")
                key_len = int(key_states.shape[-2])
                bias_len = int(log_bias.shape[-1])
                if bias_len < key_len:
                    pad = torch.zeros(
                        (int(log_bias.shape[0]), int(log_bias.shape[1]), key_len - bias_len),
                        dtype=log_bias.dtype,
                        device=log_bias.device,
                    )
                    log_bias = torch.cat([log_bias, pad], dim=-1)
                elif bias_len > key_len:
                    log_bias = log_bias[..., -key_len:]
                if int(log_bias.shape[1]) != int(query_states.shape[1]):
                    log_bias = _repeat_kv_for_query_heads(log_bias.unsqueeze(-1), int(self.num_key_value_groups)).squeeze(-1)
                if int(key_states.shape[1]) != int(query_states.shape[1]):
                    key_states = _repeat_kv_for_query_heads(key_states, int(self.num_key_value_groups))
                    value_states = _repeat_kv_for_query_heads(value_states, int(self.num_key_value_groups))
                expanded_query_head_cache = True

        layer_attention_mask = attention_mask
        if layer_attention_mask is not None:
            key_len = int(key_states.shape[-2])
            mask_len = int(layer_attention_mask.shape[-1])
            if mask_len != key_len:
                if int(query_states.shape[-2]) == 1:
                    # Greedy decode with batch=1 and no padding: the current
                    # token can attend all retained keys in this layer.
                    layer_attention_mask = None
                elif mask_len > key_len:
                    layer_attention_mask = layer_attention_mask[..., -key_len:]
                else:
                    raise RuntimeError(
                        f"Attention mask length {mask_len} is shorter than layer {self.layer_idx} KV length {key_len}."
                    )
        padding_mask = None
        if past_key_values is not None and not bool(integrated_prefill_compress):
            padding_mask = _strictmerge_layer_padding_mask(
                past_key_values,
                int(self.layer_idx),
                key_len=int(key_states.shape[-2]),
                batch_size=int(query_states.shape[0]),
                device=query_states.device,
            )
        if padding_mask is not None and log_bias is None and layer_attention_mask is None:
            flash_output = _ragged_flash_decode_forward(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                padding_mask=padding_mask,
                scaling=float(self.scaling),
            )
            if flash_output is not None:
                attn_output = flash_output.reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, None
        if padding_mask is not None:
            if int(key_states.shape[1]) != int(query_states.shape[1]):
                key_states = _repeat_kv_for_query_heads(key_states, int(self.num_key_value_groups))
                value_states = _repeat_kv_for_query_heads(value_states, int(self.num_key_value_groups))
            expanded_query_head_cache = True
        if log_bias is not None:
            bias_mask = log_bias[:, :, None, :].to(device=query_states.device, dtype=query_states.dtype)
            layer_attention_mask = bias_mask if layer_attention_mask is None else layer_attention_mask + bias_mask
        if padding_mask is not None:
            pad_bias = torch.zeros(
                (int(padding_mask.shape[0]), 1, 1, int(padding_mask.shape[1])),
                dtype=query_states.dtype,
                device=query_states.device,
            )
            pad_bias = pad_bias.masked_fill(~padding_mask[:, None, None, :], torch.finfo(query_states.dtype).min)
            layer_attention_mask = pad_bias if layer_attention_mask is None else layer_attention_mask + pad_bias

        if bool(expanded_query_head_cache):
            attn_output, attn_weights = _attention_forward_no_repeat(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=layer_attention_mask,
                scaling=float(self.scaling),
            )
        else:
            attn_output, attn_weights = attention_interface(self.config)(
                self,
                query_states,
                key_states,
                value_states,
                layer_attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
        del query_states, key_states, value_states
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if past_key_values is not None and bool(integrated_prefill_compress) and deferred_prefill_cache_update is not None:
            deferred_prefill_cache_update(past_key_values)
        return attn_output, attn_weights

    return forward


@contextmanager
def apply_strict_attention_patch(
    model,
    *,
    record_query_content: bool = False,
    record_full_query_content: bool = False,
    record_tail_window: int = 16,
    record_decode_queries: bool = False,
    prefill_cache_compressor: Any = None,
) -> Generator[StrictQueryContentState, None, None]:
    state = StrictQueryContentState(
        record_query_content=bool(record_query_content),
        record_full_query_content=bool(record_full_query_content),
        record_tail_window=max(int(record_tail_window), 1),
        record_decode_queries=bool(record_decode_queries),
        prefill_cache_compressor=prefill_cache_compressor,
    )
    originals: list[tuple[Any, Any]] = []
    try:
        for layer in model.model.layers:
            attn = layer.self_attn
            originals.append((attn, attn.forward))
            attn.forward = MethodType(make_strict_llama_attention_forward(state), attn)
        yield state
    finally:
        for attn, original in originals:
            attn.forward = original
