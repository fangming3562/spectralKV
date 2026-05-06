"""Clean public API for SpectralKV.

This package exposes only the SpectralKV mainline: attention-entropy layer
allocation, metric-aligned spectral CSD KV coreset selection, and append-only
decoding helpers. Experimental baselines and historical runner interfaces live
outside this package.
"""

from .api import (
    SpectralKVConfig,
    build_default_config,
    compute_static_layer_budgets,
    estimate_layer_budget_schedule,
    load_static_layer_schedule,
    prefill_compress,
    prefill_compress_batched,
    prefill_compress_static_schedule,
    prefill_compress_static_schedule_batched,
    prefill_compress_static_schedule_integrated,
    prefill_compress_static_schedule_integrated_async,
    prefill_compress_static_schedule_layerwise,
    prefill_compress_static_schedule_layerwise_async,
    summarize_debug_layers,
)
from .generation import greedy_generate_original_positions, greedy_generate_original_positions_batched

__all__ = [
    "SpectralKVConfig",
    "build_default_config",
    "compute_static_layer_budgets",
    "estimate_layer_budget_schedule",
    "load_static_layer_schedule",
    "greedy_generate_original_positions",
    "greedy_generate_original_positions_batched",
    "prefill_compress",
    "prefill_compress_batched",
    "prefill_compress_static_schedule",
    "prefill_compress_static_schedule_batched",
    "prefill_compress_static_schedule_integrated",
    "prefill_compress_static_schedule_integrated_async",
    "prefill_compress_static_schedule_layerwise",
    "prefill_compress_static_schedule_layerwise_async",
    "summarize_debug_layers",
]
