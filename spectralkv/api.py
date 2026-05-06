from __future__ import annotations

import json
import importlib
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from .attention_patch import apply_strict_attention_patch
from .attention_patch import StrictQueryContentState
from .budget import uniform_layer_budget
from .cache_utils import extract_cache_layers, set_layer_cache, set_layer_log_bias
from .compress import (
    _attention_entropy_layer_score,
    _attention_entropy_layer_budgets,
    _layer_budget_floor_cap,
    _needs_full_query_content,
    _profile_enabled,
    _prompt_qobs_for_layer,
    build_dynamic_cache_no_copy,
    compress_single_prompt_layer_strictmerge,
    compress_prompt_cache_strictmerge,
    compress_prompt_cache_strictmerge_sample_batched,
    compress_prompt_layer_strictmerge_batched_static,
    prefill_with_strictmerge,
    prefill_with_strictmerge_batched_samples,
    summarize_layer_kept,
)
from .config import StrictMergeConfig
from .masks import build_force_keep_mask


@dataclass(frozen=True)
class SpectralKVConfig:
    """Public SpectralKV configuration.

    The defaults match the current best_v1_2 mainline:
    attention-entropy layer budget, q-covariance key metric, OProj diagonal
    value metric, spectral CSD KV local selection, chunk size 16, and tail
    observation window 8.
    """

    keep_ratio: float = 0.0625
    fixed_budget: int = 0
    observation_window: int = 8
    force_sink: int = 4
    force_recent: int = 8
    local_chunk_size: int = 16
    layer_budget_rho: float = 1.0
    layer_budget_max_keep_ratio: float = 0.5
    metric_mode: str = "oproj_diag"
    key_metric_mode: str = "qcov_diag"

    def to_strict_config(self) -> StrictMergeConfig:
        cfg = StrictMergeConfig(
            keep_ratio=float(self.keep_ratio),
            fixed_budget=int(self.fixed_budget),
            observation_window=int(self.observation_window),
            force_sink=int(self.force_sink),
            force_recent=int(self.force_recent),
            selector_mode="local_jaoc",
            selection_granularity="kv_head",
            cache_head_mode="kv",
            metric_mode=str(self.metric_mode),
            key_metric_mode=str(self.key_metric_mode),
            local_score_mode="spectral_csd_kv",
            local_chunk_size=int(self.local_chunk_size),
            layer_budget_rho=float(self.layer_budget_rho),
            layer_budget_max_keep_ratio=float(self.layer_budget_max_keep_ratio),
            observation_mode="da_tail",
            risk_mode="mean",
        )
        cfg.validate()
        return cfg


def build_default_config(**overrides: Any) -> SpectralKVConfig:
    return SpectralKVConfig(**overrides)


def load_static_layer_schedule(
    source: Sequence[float] | str | Path | dict[str, Any],
    *,
    key: str = "static_schedule",
) -> list[float]:
    """Load a static layer schedule from a JSON file, payload dict, or raw sequence."""

    if isinstance(source, (str, Path)):
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    elif isinstance(source, dict):
        payload = source
    else:
        return [float(value) for value in source]

    if key not in payload:
        raise KeyError(f"Missing {key!r} in schedule payload.")
    schedule = payload[key]
    if isinstance(schedule, (str, bytes)) or not isinstance(schedule, Sequence):
        raise TypeError(f"{key!r} must be a sequence of floats.")
    return [float(value) for value in schedule]


def prefill_compress(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    config: SpectralKVConfig | StrictMergeConfig | None = None,
) -> tuple[Any, dict[str, Any]]:
    strict_config = _as_strict_config(config)
    return prefill_with_strictmerge(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        config=strict_config,
        budget_mode="attention_entropy",
    )


def prefill_compress_batched(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    prompt_lens: Sequence[int],
    config: SpectralKVConfig | StrictMergeConfig | None = None,
) -> tuple[list[Any], list[dict[str, Any]], torch.Tensor]:
    strict_config = _as_strict_config(config)
    return prefill_with_strictmerge_batched_samples(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lens=[int(value) for value in prompt_lens],
        config=strict_config,
        budget_mode="attention_entropy",
    )


def prefill_compress_static_schedule_batched(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    prompt_lens: Sequence[int],
    schedule: Sequence[float] | str | Path | dict[str, Any],
    config: SpectralKVConfig | StrictMergeConfig | None = None,
) -> tuple[list[Any], list[dict[str, Any]], torch.Tensor]:
    """Batched static-schedule compression using a precomputed layer allocation."""

    strict_config = _as_strict_config(config)
    if int(input_ids.shape[0]) != int(len(prompt_lens)):
        raise ValueError("prompt_lens length must match batch size.")
    layer_schedule = load_static_layer_schedule(schedule)
    num_layers = int(getattr(model.config, "num_hidden_layers", len(layer_schedule)))
    if int(num_layers) != int(len(layer_schedule)):
        raise ValueError(
            f"schedule has {len(layer_schedule)} layers but model expects {num_layers} hidden layers."
        )
    position_ids = None
    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0).to(device=input_ids.device, dtype=torch.long)
    with apply_strict_attention_patch(
        model,
        record_query_content=True,
        record_full_query_content=_needs_full_query_content(strict_config),
        record_tail_window=int(strict_config.observation_window),
    ) as query_state:
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                logits_to_keep=1,
            )
        except TypeError:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )
        prefill_layers = extract_cache_layers(outputs.past_key_values)
        compressed_caches: list[Any] = []
        sample_query_states = []
        fixed_budgets_by_sample: list[list[int]] = []
        for sample_idx, prompt_len in enumerate(prompt_lens):
            prompt_len_i = int(prompt_len)
            if prompt_len_i <= 0:
                raise ValueError(f"prompt_len must be positive, got {prompt_len_i} for sample {sample_idx}.")
            sample_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
            for keys, values in prefill_layers:
                sample_keys = keys[int(sample_idx) : int(sample_idx) + 1, :, -prompt_len_i:, :]
                sample_values = values[int(sample_idx) : int(sample_idx) + 1, :, -prompt_len_i:, :]
                sample_layers.append((sample_keys, sample_values))
            sample_cache = build_dynamic_cache_no_copy(sample_layers, config=model.config)
            compressed_caches.append(sample_cache)
            sample_query_states.append(query_state.sample_state(int(sample_idx), int(prompt_len_i)))
            fixed_budgets_by_sample.append(
                compute_static_layer_budgets(
                    schedule=layer_schedule,
                    seq_len=int(prompt_len_i),
                    num_layers=int(num_layers),
                    config=strict_config,
                )
            )
        profile = {} if _profile_enabled() else None
        debug_infos = compress_prompt_cache_strictmerge_sample_batched(
            model=model,
            prompt_past_key_values_list=compressed_caches,
            query_states=sample_query_states,
            config=strict_config,
            budget_mode="fixed",
            fixed_layer_budgets_by_sample=fixed_budgets_by_sample,
            profile=profile,
        )
        next_logits = outputs.logits[:, -1, :].detach().contiguous()
    return compressed_caches, debug_infos, next_logits


def prefill_compress_static_schedule_layerwise(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    schedule: Sequence[float] | str | Path | dict[str, Any],
    config: SpectralKVConfig | StrictMergeConfig | None = None,
) -> tuple[CausalLMOutputWithPast, dict[str, Any]]:
    """Static-schedule prefill with layer-wise compression and early full-KV release.

    This deployment-oriented path requires a precomputed layer schedule. It runs
    the decoder stack manually: after each layer produces its full prompt K/V,
    that layer is compressed immediately and the full layer cache is replaced by
    the compressed one before forwarding the next layer.
    """

    strict_config = _as_strict_config(config)
    if int(input_ids.shape[0]) != 1:
        raise NotImplementedError("layer-wise static SpectralKV currently supports batch_size=1.")
    if attention_mask is not None and int(attention_mask.shape[0]) != 1:
        raise ValueError("attention_mask batch size must match input_ids.")

    base_model = getattr(model, "model", None)
    if base_model is None or not hasattr(base_model, "layers"):
        raise TypeError("layer-wise static SpectralKV expects a Llama/Qwen-style CausalLM with model.layers.")
    layer_schedule = load_static_layer_schedule(schedule)
    num_layers = int(getattr(model.config, "num_hidden_layers", len(layer_schedule)))
    if int(num_layers) != int(len(layer_schedule)):
        raise ValueError(
            f"schedule has {len(layer_schedule)} layers but model expects {num_layers} hidden layers."
        )

    prompt_len = int(input_ids.shape[-1])
    layer_budgets = compute_static_layer_budgets(
        schedule=layer_schedule,
        seq_len=int(prompt_len),
        num_layers=int(num_layers),
        config=strict_config,
    )
    profile = {} if _profile_enabled() else None
    device = input_ids.device
    cache_position = torch.arange(int(prompt_len), device=device, dtype=torch.long)
    position_ids = cache_position.unsqueeze(0)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)

    with torch.inference_mode(), apply_strict_attention_patch(
        model,
        record_query_content=True,
        record_full_query_content=_needs_full_query_content(strict_config),
        record_tail_window=int(strict_config.observation_window),
    ) as query_state:
        hidden_states = base_model.embed_tokens(input_ids)
        prompt_cache = DynamicCache(config=model.config)
        causal_masks = _build_causal_mask_mapping(
            base_model=base_model,
            input_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=prompt_cache,
            position_ids=position_ids,
        )
        position_embeddings = base_model.rotary_emb(hidden_states, position_ids)
        debug_layers: list[dict[str, Any]] = []
        for layer_idx, decoder_layer in enumerate(base_model.layers[:num_layers]):
            attention_type = str(getattr(decoder_layer, "attention_type", "full_attention"))
            layer_mask = causal_masks.get(attention_type, causal_masks.get("full_attention"))
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=prompt_cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
            prompt_keys, prompt_values = extract_cache_layers(prompt_cache)[int(layer_idx)]
            compressed_keys, compressed_values, log_bias, debug_item = compress_single_prompt_layer_strictmerge(
                model=model,
                layer_idx=int(layer_idx),
                prompt_keys=prompt_keys,
                prompt_values=prompt_values,
                query_state=query_state,
                config=strict_config,
                budget=int(layer_budgets[int(layer_idx)]),
                profile=profile,
            )
            set_layer_cache(prompt_cache, int(layer_idx), compressed_keys, compressed_values)
            if isinstance(log_bias, torch.Tensor):
                set_layer_log_bias(prompt_cache, int(layer_idx), log_bias)
            debug_layers.append(debug_item)
            _drop_recorded_layer_queries(query_state, int(layer_idx))
            del prompt_keys, prompt_values, compressed_keys, compressed_values, log_bias, layer_outputs
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.empty_cache()

        hidden_states = base_model.norm(hidden_states)
        logits = model.lm_head(hidden_states[:, -1:, :]).contiguous()

    debug: dict[str, Any] = {
        "patch": "spectralkv_layerwise",
        "budget_mode": "static_schedule_layerwise",
        "static_schedule": [float(value) for value in layer_schedule],
        "fixed_layer_budgets": [int(value) for value in layer_budgets],
        "config": strict_config.__dict__,
        "layers": list(debug_layers),
        "total_select_sec": float(sum(float(layer.get("select_sec", 0.0)) for layer in debug_layers)),
        "early_free_full_layer_kv": True,
    }
    if profile is not None:
        debug["profile"] = dict(sorted(profile.items()))
    return CausalLMOutputWithPast(logits=logits, past_key_values=prompt_cache), debug


def prefill_compress_static_schedule_layerwise_async(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    schedule: Sequence[float] | str | Path | dict[str, Any],
    config: SpectralKVConfig | StrictMergeConfig | None = None,
    max_pending_layers: int = 2,
) -> tuple[CausalLMOutputWithPast, dict[str, Any]]:
    """Static layer-wise prefill with asynchronous per-layer compression.

    The main thread forwards decoder layers on the default stream. After each
    layer materializes its prompt K/V, the tensors and recorded observation
    queries are handed to a worker thread bound to a separate CUDA stream. The
    worker waits on a per-layer event, compresses that layer with the static
    budget, and stores the compressed K/V. Before returning, all compression
    work is synchronized and assembled into a DynamicCache for append-only
    decoding.

    This path is intended for deployment profiling. It currently supports
    batch size 1, matching the memory-optimized layer-wise path.
    """

    if not torch.cuda.is_available() or input_ids.device.type != "cuda":
        return prefill_compress_static_schedule_layerwise(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            schedule=schedule,
            config=config,
        )

    strict_config = _as_strict_config(config)
    if int(input_ids.shape[0]) != 1:
        raise NotImplementedError("async layer-wise static SpectralKV currently supports batch_size=1.")
    if attention_mask is not None and int(attention_mask.shape[0]) != 1:
        raise ValueError("attention_mask batch size must match input_ids.")

    base_model = getattr(model, "model", None)
    if base_model is None or not hasattr(base_model, "layers"):
        raise TypeError("async layer-wise static SpectralKV expects a Llama/Qwen-style CausalLM with model.layers.")
    layer_schedule = load_static_layer_schedule(schedule)
    num_layers = int(getattr(model.config, "num_hidden_layers", len(layer_schedule)))
    if int(num_layers) != int(len(layer_schedule)):
        raise ValueError(
            f"schedule has {len(layer_schedule)} layers but model expects {num_layers} hidden layers."
        )

    prompt_len = int(input_ids.shape[-1])
    layer_budgets = compute_static_layer_budgets(
        schedule=layer_schedule,
        seq_len=int(prompt_len),
        num_layers=int(num_layers),
        config=strict_config,
    )
    device = input_ids.device
    cache_position = torch.arange(int(prompt_len), device=device, dtype=torch.long)
    position_ids = cache_position.unsqueeze(0)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)

    max_inflight = max(int(max_pending_layers), 1)
    task_queue: queue.Queue[Any] = queue.Queue()
    result_queue: queue.Queue[Any] = queue.Queue()
    stop_token = object()
    worker_stream = torch.cuda.Stream(device=device)
    main_stream = torch.cuda.current_stream(device=device)
    worker_error: list[BaseException] = []
    inflight_slots = threading.BoundedSemaphore(value=max_inflight)

    def _worker() -> None:
        try:
            while True:
                task = task_queue.get()
                if task is stop_token:
                    task_queue.task_done()
                    break
                try:
                    (
                        layer_idx,
                        prompt_keys,
                        prompt_values,
                        query_state_layer,
                        budget,
                        ready_event,
                    ) = task
                    with torch.cuda.stream(worker_stream):
                        worker_stream.wait_event(ready_event)
                        prompt_keys.record_stream(worker_stream)
                        prompt_values.record_stream(worker_stream)
                        t0 = time.perf_counter()
                        compressed_keys, compressed_values, log_bias, debug_item = compress_single_prompt_layer_strictmerge(
                            model=model,
                            layer_idx=int(layer_idx),
                            prompt_keys=prompt_keys,
                            prompt_values=prompt_values,
                            query_state=query_state_layer,
                            config=strict_config,
                            budget=int(budget),
                            profile=None,
                        )
                        debug_item["async_worker_select_wall_sec"] = float(time.perf_counter() - t0)
                        compressed_keys = compressed_keys.detach()
                        compressed_values = compressed_values.detach()
                        if isinstance(log_bias, torch.Tensor):
                            log_bias = log_bias.detach()
                        done_event = torch.cuda.Event()
                        done_event.record(worker_stream)
                        done_event.synchronize()
                        compressed_keys.record_stream(worker_stream)
                        compressed_values.record_stream(worker_stream)
                        if isinstance(log_bias, torch.Tensor):
                            log_bias.record_stream(worker_stream)
                    result_queue.put((int(layer_idx), compressed_keys, compressed_values, log_bias, debug_item, done_event))
                finally:
                    del task
                    try:
                        del prompt_keys, prompt_values, query_state_layer, ready_event
                    except UnboundLocalError:
                        pass
                    inflight_slots.release()
                    task_queue.task_done()
        except BaseException as exc:  # noqa: BLE001
            worker_error.append(exc)
            result_queue.put(exc)
            try:
                task_queue.task_done()
            except Exception:
                pass

    worker = threading.Thread(target=_worker, name="spectralkv-async-compress", daemon=True)
    debug_layers: list[dict[str, Any] | None] = [None] * int(num_layers)
    compressed_layers: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None] = [None] * int(num_layers)

    with torch.inference_mode(), apply_strict_attention_patch(
        model,
        record_query_content=True,
        record_full_query_content=_needs_full_query_content(strict_config),
        record_tail_window=int(strict_config.observation_window),
    ) as query_state:
        worker.start()
        hidden_states = base_model.embed_tokens(input_ids)
        prompt_cache = DynamicCache(config=model.config)
        causal_masks = _build_causal_mask_mapping(
            base_model=base_model,
            input_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=prompt_cache,
            position_ids=position_ids,
        )
        position_embeddings = base_model.rotary_emb(hidden_states, position_ids)
        forward_start = time.perf_counter()
        submitted = 0
        for layer_idx, decoder_layer in enumerate(base_model.layers[:num_layers]):
            if worker_error:
                raise worker_error[0]
            attention_type = str(getattr(decoder_layer, "attention_type", "full_attention"))
            layer_mask = causal_masks.get(attention_type, causal_masks.get("full_attention"))
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=prompt_cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
            prompt_keys, prompt_values = extract_cache_layers(prompt_cache)[int(layer_idx)]
            layer_state = query_state.sample_state(0, int(prompt_len))
            for attr_name in ("layer_query_contents", "layer_content_positions"):
                mapping = getattr(layer_state, attr_name, None)
                if isinstance(mapping, dict):
                    for key in list(mapping.keys()):
                        if int(key) != int(layer_idx):
                            mapping.pop(key, None)
            ready_event = torch.cuda.Event()
            ready_event.record(main_stream)
            slot_acquired = False
            try:
                # Backpressure is applied after the current layer has already
                # forwarded. This allows layer i compression to overlap with
                # layer i+1 forward; if compression falls behind, only then do
                # we block before submitting another full layer to the worker.
                inflight_slots.acquire()
                slot_acquired = True
                task_queue.put(
                    (
                        int(layer_idx),
                        prompt_keys.detach(),
                        prompt_values.detach(),
                        layer_state,
                        int(layer_budgets[int(layer_idx)]),
                        ready_event,
                    )
                )
                slot_acquired = False
            except BaseException:
                if bool(slot_acquired):
                    inflight_slots.release()
                raise
            submitted += 1
            empty_shape = (*prompt_keys.shape[:-2], 0, prompt_keys.shape[-1])
            empty_keys = prompt_keys.new_empty(empty_shape)
            empty_values = prompt_values.new_empty(empty_shape)
            set_layer_cache(prompt_cache, int(layer_idx), empty_keys, empty_values)
            _drop_recorded_layer_queries(query_state, int(layer_idx))
            del prompt_keys, prompt_values, empty_keys, empty_values, layer_outputs

        hidden_states = base_model.norm(hidden_states)
        logits = model.lm_head(hidden_states[:, -1:, :]).contiguous()
        forward_sec = float(time.perf_counter() - forward_start)

        task_queue.put(stop_token)
        task_queue.join()
        worker.join()
        if worker_error:
            raise worker_error[0]

        collected = 0
        while collected < submitted:
            item = result_queue.get()
            if isinstance(item, BaseException):
                raise item
            layer_idx, compressed_keys, compressed_values, log_bias, debug_item, done_event = item
            done_event.synchronize()
            compressed_layers[int(layer_idx)] = (compressed_keys, compressed_values, log_bias)
            debug_layers[int(layer_idx)] = debug_item
            collected += 1

    out_cache = DynamicCache(config=model.config)
    for layer_idx, item in enumerate(compressed_layers):
        if item is None:
            raise RuntimeError(f"Missing compressed layer {layer_idx}.")
        compressed_keys, compressed_values, log_bias = item
        set_layer_cache(out_cache, int(layer_idx), compressed_keys, compressed_values)
        if isinstance(log_bias, torch.Tensor):
            set_layer_log_bias(out_cache, int(layer_idx), log_bias)

    layer_debug = [item for item in debug_layers if item is not None]
    debug: dict[str, Any] = {
        "patch": "spectralkv_layerwise_async",
        "budget_mode": "static_schedule_layerwise_async",
        "static_schedule": [float(value) for value in layer_schedule],
        "fixed_layer_budgets": [int(value) for value in layer_budgets],
        "config": strict_config.__dict__,
        "layers": layer_debug,
        "total_select_sec": float(sum(float(layer.get("select_sec", 0.0)) for layer in layer_debug)),
        "forward_sec": float(forward_sec),
        "async_worker_stream": True,
        "early_free_full_layer_kv": True,
        "max_pending_layers": int(max_inflight),
        "bounded_inflight_full_layers": True,
        "submit_backpressure_after_forward": True,
    }
    return CausalLMOutputWithPast(logits=logits, past_key_values=out_cache), debug


def prefill_compress_static_schedule_integrated(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    schedule: Sequence[float] | str | Path | dict[str, Any],
    config: SpectralKVConfig | StrictMergeConfig | None = None,
) -> tuple[CausalLMOutputWithPast, dict[str, Any]]:
    """Forward-integrated static compression.

    During prompt prefill, each attention layer still computes its output from
    the full prompt K/V, but the K/V written to ``past_key_values`` is already
    compressed with the static layer budget. This mirrors PyramidInfer-style
    model-forward integration and avoids storing a full prompt cache for all
    layers before compression.
    """

    strict_config = _as_strict_config(config)
    batch_size = int(input_ids.shape[0])
    prompt_len = int(input_ids.shape[-1])
    layer_schedule = load_static_layer_schedule(schedule)
    num_layers = int(getattr(model.config, "num_hidden_layers", len(layer_schedule)))
    if int(num_layers) != int(len(layer_schedule)):
        raise ValueError(
            f"schedule has {len(layer_schedule)} layers but model expects {num_layers} hidden layers."
        )
    layer_budgets = compute_static_layer_budgets(
        schedule=layer_schedule,
        seq_len=int(prompt_len),
        num_layers=int(num_layers),
        config=strict_config,
    )
    profile = {} if _profile_enabled() else None

    def _compress_from_attention(
        *,
        layer_idx: int,
        prompt_keys: torch.Tensor,
        prompt_values: torch.Tensor,
        query_content_states: torch.Tensor,
        content_positions: torch.Tensor,
        module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        del module
        if int(query_content_states.shape[0]) > 1:
            if bool(_needs_full_query_content(strict_config)):
                raise NotImplementedError("Batched integrated static SpectralKV does not support full-query-content selectors.")
            tail_width = min(
                max(int(strict_config.observation_window), 1),
                int(query_content_states.shape[2]),
            )
            batched_query_content = query_content_states[:, :, -tail_width:, :].detach().contiguous()
            pos = content_positions
            if int(pos.dim()) == 2:
                batched_positions = pos[:, -tail_width:].detach().contiguous()
            else:
                batched_positions = pos[-tail_width:].detach().contiguous()
            return compress_prompt_layer_strictmerge_batched_static(
                model=model,
                layer_idx=int(layer_idx),
                prompt_keys=prompt_keys,
                prompt_values=prompt_values,
                query_content_states=batched_query_content,
                content_positions=batched_positions,
                config=strict_config,
                budgets=[int(layer_budgets[int(layer_idx)]) for _ in range(int(query_content_states.shape[0]))],
                profile=profile,
            )
        local_state = StrictQueryContentState(
            record_query_content=True,
            record_full_query_content=_needs_full_query_content(strict_config),
            record_tail_window=int(strict_config.observation_window),
        )
        pos = content_positions
        if int(pos.dim()) == 2:
            pos = pos[0]
        if not bool(_needs_full_query_content(strict_config)):
            tail_width = min(
                max(int(strict_config.observation_window), 1),
                int(query_content_states.shape[2]),
                int(pos.numel()),
            )
            query_content = query_content_states[0, :, -tail_width:, :]
            pos = pos[-tail_width:]
        else:
            query_content = query_content_states[0]
        local_state.layer_query_contents[int(layer_idx)] = [query_content.detach().contiguous()]
        local_state.layer_content_positions[int(layer_idx)] = [pos.detach().contiguous()]
        return compress_single_prompt_layer_strictmerge(
            model=model,
            layer_idx=int(layer_idx),
            prompt_keys=prompt_keys,
            prompt_values=prompt_values,
            query_state=local_state,
            config=strict_config,
            budget=int(layer_budgets[int(layer_idx)]),
            profile=profile,
        )

    with apply_strict_attention_patch(
        model,
        record_query_content=False,
        record_full_query_content=_needs_full_query_content(strict_config),
        record_tail_window=int(strict_config.observation_window),
        prefill_cache_compressor=_compress_from_attention,
    ) as patch_state:
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                logits_to_keep=1,
            )
        except TypeError:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

    debug_layers = [patch_state.integrated_prefill_debug[idx] for idx in sorted(patch_state.integrated_prefill_debug)]
    debug: dict[str, Any] = {
        "patch": "spectralkv_integrated_forward",
        "budget_mode": "static_schedule_integrated",
        "static_schedule": [float(value) for value in layer_schedule],
        "fixed_layer_budgets": [int(value) for value in layer_budgets],
        "config": strict_config.__dict__,
        "layers": list(debug_layers),
        "total_select_sec": float(sum(float(layer.get("select_sec", 0.0)) for layer in debug_layers)),
        "forward_integrated_cache_write": True,
        "batch_size": int(batch_size),
        "batched_integrated_static": bool(batch_size > 1),
    }
    if profile is not None:
        debug["profile"] = dict(sorted(profile.items()))
    return outputs, debug


def prefill_compress_static_schedule_integrated_async(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    schedule: Sequence[float] | str | Path | dict[str, Any],
    config: SpectralKVConfig | StrictMergeConfig | None = None,
    max_pending_layers: int = 2,
) -> tuple[CausalLMOutputWithPast, dict[str, Any]]:
    """Forward-integrated static compression with asynchronous cache writes.

    Unlike the manual layer-wise async path, this keeps the model's normal
    forward path. Each attention layer computes its prefill output from the full
    prompt K/V, but submits the full K/V to a worker CUDA stream and writes a
    temporary empty cache entry immediately. After prefill, compressed K/V
    entries are assembled into the returned cache.
    """

    if not torch.cuda.is_available() or input_ids.device.type != "cuda":
        return prefill_compress_static_schedule_integrated(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            schedule=schedule,
            config=config,
        )

    strict_config = _as_strict_config(config)
    if int(input_ids.shape[0]) != 1:
        raise NotImplementedError("async integrated static SpectralKV currently supports batch_size=1.")
    prompt_len = int(input_ids.shape[-1])
    layer_schedule = load_static_layer_schedule(schedule)
    num_layers = int(getattr(model.config, "num_hidden_layers", len(layer_schedule)))
    if int(num_layers) != int(len(layer_schedule)):
        raise ValueError(
            f"schedule has {len(layer_schedule)} layers but model expects {num_layers} hidden layers."
        )
    layer_budgets = compute_static_layer_budgets(
        schedule=layer_schedule,
        seq_len=int(prompt_len),
        num_layers=int(num_layers),
        config=strict_config,
    )

    device = input_ids.device
    max_inflight = max(int(max_pending_layers), 1)
    task_queue: queue.Queue[Any] = queue.Queue()
    result_queue: queue.Queue[Any] = queue.Queue()
    stop_token = object()
    worker_stream = torch.cuda.Stream(device=device)
    main_stream = torch.cuda.current_stream(device=device)
    inflight_slots = threading.BoundedSemaphore(value=max_inflight)
    worker_error: list[BaseException] = []
    submitted_layers: list[int] = []
    submit_wait_sec = 0.0
    submit_put_sec = 0.0
    submit_count = 0

    def _worker() -> None:
        try:
            while True:
                task = task_queue.get()
                if task is stop_token:
                    task_queue.task_done()
                    break
                try:
                    layer_idx, prompt_keys, prompt_values, layer_state, budget, ready_event = task
                    with torch.cuda.stream(worker_stream):
                        worker_stream.wait_event(ready_event)
                        prompt_keys.record_stream(worker_stream)
                        prompt_values.record_stream(worker_stream)
                        for items in getattr(layer_state, "layer_query_contents", {}).values():
                            for tensor in items:
                                if isinstance(tensor, torch.Tensor):
                                    tensor.record_stream(worker_stream)
                        for items in getattr(layer_state, "layer_content_positions", {}).values():
                            for tensor in items:
                                if isinstance(tensor, torch.Tensor):
                                    tensor.record_stream(worker_stream)
                        worker_t0 = time.perf_counter()
                        compressed_keys, compressed_values, log_bias, debug_item = compress_single_prompt_layer_strictmerge(
                            model=model,
                            layer_idx=int(layer_idx),
                            prompt_keys=prompt_keys,
                            prompt_values=prompt_values,
                            query_state=layer_state,
                            config=strict_config,
                            budget=int(budget),
                            profile=None,
                        )
                        compressed_keys = compressed_keys.detach()
                        compressed_values = compressed_values.detach()
                        if isinstance(log_bias, torch.Tensor):
                            log_bias = log_bias.detach()
                        done_event = torch.cuda.Event()
                        done_event.record(worker_stream)
                        done_event.synchronize()
                        debug_item["async_worker_wall_sec"] = float(time.perf_counter() - worker_t0)
                    result_queue.put((int(layer_idx), compressed_keys, compressed_values, log_bias, debug_item))
                finally:
                    del task
                    try:
                        del prompt_keys, prompt_values, layer_state, ready_event
                    except UnboundLocalError:
                        pass
                    inflight_slots.release()
                    task_queue.task_done()
        except BaseException as exc:  # noqa: BLE001
            worker_error.append(exc)
            result_queue.put(exc)
            try:
                task_queue.task_done()
            except Exception:
                pass

    def _compress_from_attention(
        *,
        layer_idx: int,
        prompt_keys: torch.Tensor,
        prompt_values: torch.Tensor,
        query_content_states: torch.Tensor,
        content_positions: torch.Tensor,
        module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        del module
        layer_idx = int(layer_idx)
        pos = content_positions
        if int(pos.dim()) == 2:
            pos = pos[0]
        if not bool(_needs_full_query_content(strict_config)):
            tail_width = min(
                max(int(strict_config.observation_window), 1),
                int(query_content_states.shape[2]),
                int(pos.numel()),
            )
            query_content = query_content_states[0, :, -tail_width:, :].detach().contiguous()
            pos = pos[-tail_width:].detach().contiguous()
        else:
            query_content = query_content_states[0].detach().contiguous()
            pos = pos.detach().contiguous()
        layer_state = StrictQueryContentState(
            record_query_content=True,
            record_full_query_content=_needs_full_query_content(strict_config),
            record_tail_window=int(strict_config.observation_window),
        )
        layer_state.layer_query_contents[layer_idx] = [query_content]
        layer_state.layer_content_positions[layer_idx] = [pos]

        empty_shape = (*prompt_keys.shape[:-2], 0, prompt_keys.shape[-1])
        empty_keys = prompt_keys.new_empty(empty_shape)
        empty_values = prompt_values.new_empty(empty_shape)

        prompt_keys_detached = prompt_keys.detach()
        prompt_values_detached = prompt_values.detach()

        def _defer_cache_update(past_key_values) -> None:
            nonlocal submit_wait_sec, submit_put_sec, submit_count
            ready_event = torch.cuda.Event()
            ready_event.record(main_stream)
            submit_t0 = time.perf_counter()
            inflight_slots.acquire()
            submit_wait_sec += float(time.perf_counter() - submit_t0)
            put_t0 = time.perf_counter()
            task_queue.put(
                (
                    int(layer_idx),
                    prompt_keys_detached,
                    prompt_values_detached,
                    layer_state,
                    int(layer_budgets[layer_idx]),
                    ready_event,
                )
            )
            submit_put_sec += float(time.perf_counter() - put_t0)
            submit_count += 1
            submitted_layers.append(int(layer_idx))
            set_layer_cache(past_key_values, int(layer_idx), empty_keys, empty_values)

        return (
            empty_keys,
            empty_values,
            None,
            {
                "layer_idx": int(layer_idx),
                "seq_len": int(prompt_len),
                "budget": int(layer_budgets[layer_idx]),
                "queued_async": True,
                "_defer_cache_update": _defer_cache_update,
                "select_sec": 0.0,
            },
        )

    worker = threading.Thread(target=_worker, name="spectralkv-integrated-async-compress", daemon=True)
    worker.start()
    forward_start = time.perf_counter()
    try:
        with apply_strict_attention_patch(
            model,
            record_query_content=False,
            record_full_query_content=_needs_full_query_content(strict_config),
            record_tail_window=int(strict_config.observation_window),
            prefill_cache_compressor=_compress_from_attention,
        ):
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    logits_to_keep=1,
                )
            except TypeError:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
        model_forward_sec = float(time.perf_counter() - forward_start)
    finally:
        tail_wait_start = time.perf_counter()
        task_queue.put(stop_token)
        task_queue.join()
        worker.join()
    tail_wait_sec = float(time.perf_counter() - tail_wait_start)
    forward_sec = float(time.perf_counter() - forward_start)
    if worker_error:
        raise worker_error[0]

    compressed_layers: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None] = [None] * int(num_layers)
    debug_layers: list[dict[str, Any] | None] = [None] * int(num_layers)
    collected = 0
    expected = len(submitted_layers)
    while collected < expected:
        item = result_queue.get()
        if isinstance(item, BaseException):
            raise item
        layer_idx, compressed_keys, compressed_values, log_bias, debug_item = item
        compressed_layers[int(layer_idx)] = (compressed_keys, compressed_values, log_bias)
        debug_layers[int(layer_idx)] = debug_item
        collected += 1

    out_cache = DynamicCache(config=model.config)
    for layer_idx, item in enumerate(compressed_layers):
        if item is None:
            raise RuntimeError(f"Missing compressed layer {layer_idx}.")
        compressed_keys, compressed_values, log_bias = item
        set_layer_cache(out_cache, int(layer_idx), compressed_keys, compressed_values)
        if isinstance(log_bias, torch.Tensor):
            set_layer_log_bias(out_cache, int(layer_idx), log_bias)

    layer_debug = [item for item in debug_layers if item is not None]
    debug: dict[str, Any] = {
        "patch": "spectralkv_integrated_forward_async",
        "budget_mode": "static_schedule_integrated_async",
        "static_schedule": [float(value) for value in layer_schedule],
        "fixed_layer_budgets": [int(value) for value in layer_budgets],
        "config": strict_config.__dict__,
        "layers": layer_debug,
        "total_select_sec": float(sum(float(layer.get("select_sec", 0.0)) for layer in layer_debug)),
        "forward_sec": float(forward_sec),
        "model_forward_sec": float(model_forward_sec),
        "async_tail_wait_sec": float(tail_wait_sec),
        "async_submit_wait_sec": float(submit_wait_sec),
        "async_submit_put_sec": float(submit_put_sec),
        "async_submit_count": int(submit_count),
        "async_worker_stream": True,
        "forward_integrated_cache_write": True,
        "temporary_empty_cache_write": True,
        "max_pending_layers": int(max_inflight),
    }
    return CausalLMOutputWithPast(
        logits=outputs.logits,
        past_key_values=out_cache,
        hidden_states=getattr(outputs, "hidden_states", None),
        attentions=getattr(outputs, "attentions", None),
    ), debug


def estimate_layer_budget_schedule(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    config: SpectralKVConfig | StrictMergeConfig | None = None,
) -> dict[str, Any]:
    """Run prefill and compute dynamic layer budgets without decoding.

    This is intended for calibration. It records the same tail observation
    queries as SpectralKV, computes the attention-entropy budget, and returns
    per-layer budgets and ratios. It does not perform token selection,
    compression, or generation. The implementation is layer-wise: after each
    layer score is computed, that layer's full prompt KV and recorded queries
    are released, so the retained state is only one scalar score per layer.
    """

    strict_config = _as_strict_config(config)
    if int(input_ids.shape[0]) != 1:
        raise ValueError("estimate_layer_budget_schedule expects batch_size=1.")
    profile = {} if _profile_enabled() else None
    base_model = getattr(model, "model", None)
    if base_model is None or not hasattr(base_model, "layers"):
        raise TypeError("estimate_layer_budget_schedule expects a Llama/Qwen-style CausalLM with model.layers.")
    num_layers = int(getattr(model.config, "num_hidden_layers", len(base_model.layers)))
    prompt_len = int(input_ids.shape[-1])
    device = input_ids.device
    cache_position = torch.arange(int(prompt_len), device=device, dtype=torch.long)
    position_ids = cache_position.unsqueeze(0)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)

    with torch.inference_mode(), apply_strict_attention_patch(
        model,
        record_query_content=True,
        record_full_query_content=False,
        record_tail_window=int(strict_config.observation_window),
    ) as query_state:
        hidden_states = base_model.embed_tokens(input_ids)
        prompt_cache = DynamicCache(config=model.config)
        causal_masks = _build_causal_mask_mapping(
            base_model=base_model,
            input_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=prompt_cache,
            position_ids=position_ids,
        )
        position_embeddings = base_model.rotary_emb(hidden_states, position_ids)
        seq_lens: list[int] = []
        force_counts: list[int] = []
        layer_scores: list[float] = []
        for layer_idx, decoder_layer in enumerate(base_model.layers[:num_layers]):
            attention_type = str(getattr(decoder_layer, "attention_type", "full_attention"))
            layer_mask = causal_masks.get(attention_type, causal_masks.get("full_attention"))
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=prompt_cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
            prompt_keys, prompt_values = extract_cache_layers(prompt_cache)[int(layer_idx)]
            seq_len = int(prompt_keys.shape[-2])
            force_mask = build_force_keep_mask(
                seq_len=int(seq_len),
                force_sink=int(strict_config.force_sink),
                force_recent=int(strict_config.force_recent),
                force_prefix=int(strict_config.force_prefix),
                device=prompt_keys.device,
            )
            force_count = int(force_mask.sum().item())
            seq_lens.append(int(seq_len))
            force_counts.append(int(force_count))
            recent = min(max(int(strict_config.force_recent), 0), seq_len)
            old_len = int(seq_len) - int(recent)
            if old_len <= 0:
                layer_scores.append(0.0)
            else:
                q_obs, _obs_positions = _prompt_qobs_for_layer(
                    model=model,
                    query_state=query_state,
                    layer_idx=int(layer_idx),
                    prompt_keys=prompt_keys,
                    prompt_len=int(seq_len),
                    config=strict_config,
                )
                hkv = int(prompt_keys.shape[1])
                groups = int(model.config.num_attention_heads // max(hkv, 1))
                entropy_score = _attention_entropy_layer_score(
                    q_obs=q_obs,
                    keys=prompt_keys,
                    old_len=int(old_len),
                    num_key_value_groups=int(groups),
                    force_mask=force_mask,
                )
                layer_scores.append(float(entropy_score))
                del q_obs
            empty_shape = (*prompt_keys.shape[:-2], 0, prompt_keys.shape[-1])
            empty_keys = prompt_keys.new_empty(empty_shape)
            empty_values = prompt_values.new_empty(empty_shape)
            set_layer_cache(prompt_cache, int(layer_idx), empty_keys, empty_values)
            _drop_recorded_layer_queries(query_state, int(layer_idx))
            del prompt_keys, prompt_values, force_mask, layer_outputs, empty_keys, empty_values
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.empty_cache()
        del hidden_states, position_embeddings, prompt_cache, query_state
        budgets = _attention_entropy_layer_budgets(
            seq_lens=seq_lens,
            force_counts=force_counts,
            layer_scores=layer_scores,
            config=strict_config,
        )
    total_budget = max(int(sum(budgets)), 1)
    return {
        "budget_mode": "attention_entropy",
        "config": strict_config.__dict__,
        "seq_lens": [int(value) for value in seq_lens],
        "force_counts": [int(value) for value in force_counts],
        "layer_scores": [float(value) for value in layer_scores],
        "layer_budgets": [int(value) for value in budgets],
        "layer_budget_ratios": [float(value) / float(total_budget) for value in budgets],
        "layer_keep_ratios": [
            float(budget) / max(float(seq_len), 1.0)
            for budget, seq_len in zip(budgets, seq_lens)
        ],
        "total_budget": int(total_budget),
        "profile": dict(sorted(profile.items())) if profile is not None else {},
    }


def compute_static_layer_budgets(
    *,
    schedule: Sequence[float],
    seq_len: int,
    num_layers: int,
    config: SpectralKVConfig | StrictMergeConfig | None = None,
) -> list[int]:
    """Convert a learned layer-ratio schedule into integer layer budgets."""

    strict_config = _as_strict_config(config)
    if int(num_layers) <= 0:
        raise ValueError("num_layers must be positive.")
    if len(schedule) != int(num_layers):
        raise ValueError(f"schedule has {len(schedule)} entries but num_layers={num_layers}.")
    weights = torch.tensor([max(float(value), 0.0) for value in schedule], dtype=torch.float64)
    if float(weights.sum().item()) <= 0.0:
        weights = torch.ones((int(num_layers),), dtype=torch.float64)
    weights = weights / weights.sum()

    floors: list[int] = []
    ceils: list[int] = []
    target_total = 0
    force_count = int(strict_config.force_sink) + int(strict_config.force_recent) + int(strict_config.force_prefix)
    force_count = min(max(int(force_count), 1), int(seq_len))
    for _layer_idx in range(int(num_layers)):
        floor, ceil = _layer_budget_floor_cap(seq_len=int(seq_len), force_count=int(force_count), config=strict_config)
        floors.append(int(floor))
        ceils.append(int(ceil))
        target_total += uniform_layer_budget(
            seq_len=int(seq_len),
            keep_ratio=float(strict_config.keep_ratio),
            force_keep_count=int(force_count),
        )
    if int(strict_config.fixed_budget) > 0:
        target_total = int(strict_config.fixed_budget) * int(num_layers)
    target_total = max(int(target_total), int(sum(floors)))

    raw = weights * float(target_total)
    budgets = [min(max(int(torch.floor(raw[idx]).item()), floors[idx]), ceils[idx]) for idx in range(int(num_layers))]
    remaining = int(target_total) - int(sum(budgets))
    remainders = [(float(raw[idx].item()) - float(torch.floor(raw[idx]).item()), idx) for idx in range(int(num_layers))]
    if remaining > 0:
        for _frac, idx in sorted(remainders, reverse=True):
            if remaining <= 0:
                break
            if budgets[idx] < ceils[idx]:
                budgets[idx] += 1
                remaining -= 1
    elif remaining < 0:
        for _frac, idx in sorted(remainders):
            if remaining >= 0:
                break
            if budgets[idx] > floors[idx]:
                budgets[idx] -= 1
                remaining += 1
    return [int(value) for value in budgets]


def _build_causal_mask_mapping(
    *,
    base_model,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor,
    past_key_values,
    position_ids: torch.Tensor,
) -> dict[str, torch.Tensor | None]:
    if isinstance(attention_mask, dict):
        return attention_mask
    module = importlib.import_module(str(base_model.__class__.__module__))
    create_causal_mask = getattr(module, "create_causal_mask", None)
    if not callable(create_causal_mask):
        return {"full_attention": None}
    mask_kwargs = {
        "config": base_model.config,
        "input_embeds": input_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    masks = {"full_attention": create_causal_mask(**mask_kwargs)}
    create_sliding = getattr(module, "create_sliding_window_causal_mask", None)
    if bool(getattr(base_model, "has_sliding_layers", False)) and callable(create_sliding):
        masks["sliding_attention"] = create_sliding(**mask_kwargs)
    return masks


def _drop_recorded_layer_queries(query_state, layer_idx: int) -> None:
    for attr_name in (
        "layer_query_contents",
        "layer_content_positions",
        "layer_batch_query_contents",
        "layer_batch_content_positions",
    ):
        mapping = getattr(query_state, attr_name, None)
        if isinstance(mapping, dict):
            mapping.pop(int(layer_idx), None)


def prefill_compress_static_schedule(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    schedule: Sequence[float] | str | Path | dict[str, Any],
    config: SpectralKVConfig | StrictMergeConfig | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Compress with a learned static layer schedule instead of dynamic budgets."""

    strict_config = _as_strict_config(config)
    if int(input_ids.shape[0]) != 1:
        raise ValueError("prefill_compress_static_schedule currently expects batch_size=1.")
    prompt_len = int(input_ids.shape[-1])
    layer_schedule = load_static_layer_schedule(schedule)
    num_layers = int(getattr(model.config, "num_hidden_layers", len(layer_schedule)))
    if int(num_layers) != int(len(layer_schedule)):
        raise ValueError(
            f"schedule has {len(layer_schedule)} layers but model expects {num_layers} hidden layers."
        )
    layer_budgets = compute_static_layer_budgets(
        schedule=layer_schedule,
        seq_len=int(prompt_len),
        num_layers=int(num_layers),
        config=strict_config,
    )
    profile = {} if _profile_enabled() else None
    with apply_strict_attention_patch(
        model,
        record_query_content=True,
        record_full_query_content=_needs_full_query_content(strict_config),
        record_tail_window=int(strict_config.observation_window),
    ) as query_state:
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                logits_to_keep=1,
            )
        except TypeError:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        debug_layers = compress_prompt_cache_strictmerge(
            model=model,
            prompt_past_key_values=outputs.past_key_values,
            query_state=query_state,
            config=strict_config,
            budget_mode="fixed",
            fixed_layer_budgets=layer_budgets,
            profile=profile,
        )
    debug: dict[str, Any] = {
        "patch": "spectralkv",
        "budget_mode": "static_schedule",
        "static_schedule": [float(value) for value in layer_schedule],
        "fixed_layer_budgets": [int(value) for value in layer_budgets],
        "config": strict_config.__dict__,
        "layers": list(debug_layers),
        "total_select_sec": float(sum(float(layer.get("select_sec", 0.0)) for layer in debug_layers)),
    }
    if profile is not None:
        debug["profile"] = dict(sorted(profile.items()))
    return outputs, debug


def summarize_debug_layers(debug: dict[str, Any], prompt_len: int) -> dict[str, Any]:
    return summarize_layer_kept(debug, int(prompt_len))


def _as_strict_config(config: SpectralKVConfig | StrictMergeConfig | None) -> StrictMergeConfig:
    if config is None:
        return SpectralKVConfig().to_strict_config()
    if isinstance(config, SpectralKVConfig):
        return config.to_strict_config()
    config.validate()
    return config
