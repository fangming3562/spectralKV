from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrictMergeConfig:
    """SpectralKV mainline configuration.

    The package keeps a few historical dataclass fields for compatibility with
    old JSON/config construction code, but validation only accepts the current
    paper path: DA-tail observation, KV-head local spectral CSD selection, and
    mean-risk residual pivoting. Static schedules are handled as fixed per-layer
    budgets by the public API, not as a separate selector mode.
    """

    keep_ratio: float = 0.125
    fixed_budget: int = 0
    observation_window: int = 16
    force_sink: int = 4
    force_recent: int = 16
    force_prefix: int = 0
    batch_drop: int = 256
    score_block_size: int = 512
    candidate_pool_factor: float = 0.0
    candidate_pool_min: int = 0
    pair_pool_size: int = 0
    precompute_x_sq: bool = True
    precompute_x_sq_max_elements: int = 64_000_000
    denom_eps: float = 1e-4
    solver: str = "auto"
    selection_granularity: str = "kv_head"
    observation_mode: str = "da_tail"
    atom_size: int = 1
    selector_mode: str = "local_jaoc"
    cache_head_mode: str = "kv"
    metric_mode: str = "oproj_diag"
    key_metric_mode: str = "qcov_diag"
    local_chunk_size: int = 32
    local_score_mode: str = "spectral_csd_kv"
    dynamic_csd_block_size: int = 32
    coreset_key_weight: float = 1.0
    coreset_value_weight: float = 1.0
    coreset_tau: float = 1.0
    coreset_knn: int = 8
    operator_real_probe_limit: int = 128
    operator_key_probe_limit: int = 32
    operator_probe_source: str = "local_plus_tail"
    operator_mass_uniform_mix: float = 0.1
    operator_min_chunk_keep: int = 0
    snap_kernel_size: int = 7
    snap_pooling: str = "maxpool"
    layer_budget_rho: float = 1.0
    layer_budget_min_keep_ratio: float = 0.0
    layer_budget_max_keep_ratio: float = 0.5
    risk_mode: str = "mean"

    def validate(self) -> None:
        if not (0.0 < float(self.keep_ratio) <= 1.0):
            raise ValueError("keep_ratio must be in (0, 1].")
        if int(self.fixed_budget) < 0:
            raise ValueError("fixed_budget must be non-negative.")
        if int(self.observation_window) < 1:
            raise ValueError("observation_window must be positive.")
        if int(self.force_sink) < 0 or int(self.force_recent) < 0 or int(self.force_prefix) < 0:
            raise ValueError("force counts must be non-negative.")
        if int(self.batch_drop) < 1:
            raise ValueError("batch_drop must be positive.")
        if int(self.score_block_size) < 1:
            raise ValueError("score_block_size must be positive.")
        if float(self.candidate_pool_factor) < 0.0:
            raise ValueError("candidate_pool_factor must be non-negative.")
        if int(self.candidate_pool_min) < 0:
            raise ValueError("candidate_pool_min must be non-negative.")
        if int(self.pair_pool_size) < 0:
            raise ValueError("pair_pool_size must be non-negative.")
        if int(self.precompute_x_sq_max_elements) < 0:
            raise ValueError("precompute_x_sq_max_elements must be non-negative.")
        if float(self.denom_eps) <= 0.0:
            raise ValueError("denom_eps must be positive.")
        if str(self.solver) not in {"auto", "drop", "keep"}:
            raise ValueError("solver must be one of: auto, drop, keep.")
        if str(self.selection_granularity) != "kv_head":
            raise ValueError("SpectralKV supports only kv_head selection_granularity.")
        if str(self.observation_mode) != "da_tail":
            raise ValueError("SpectralKV supports only da_tail observation_mode.")
        if int(self.atom_size) != 1:
            raise ValueError("SpectralKV supports only atom_size=1.")
        if str(self.selector_mode) != "local_jaoc":
            raise ValueError("SpectralKV supports only local_jaoc selector_mode.")
        if str(self.cache_head_mode) != "kv":
            raise ValueError("SpectralKV supports only kv cache_head_mode.")
        if str(self.metric_mode) not in {"value_l2", "oproj_diag"}:
            raise ValueError("metric_mode must be one of: value_l2, oproj_diag.")
        if str(self.key_metric_mode) not in {"raw", "qcov_diag"}:
            raise ValueError("key_metric_mode must be one of: raw, qcov_diag.")
        if int(self.local_chunk_size) < 2:
            raise ValueError("local_chunk_size must be at least 2.")
        if str(self.local_score_mode) != "spectral_csd_kv":
            raise ValueError("SpectralKV supports only spectral_csd_kv local_score_mode.")
        if int(self.dynamic_csd_block_size) < 1:
            raise ValueError("dynamic_csd_block_size must be positive.")
        if float(self.coreset_key_weight) < 0.0 or float(self.coreset_value_weight) < 0.0:
            raise ValueError("coreset key/value weights must be non-negative.")
        if float(self.coreset_key_weight) == 0.0 and float(self.coreset_value_weight) == 0.0:
            raise ValueError("at least one coreset feature weight must be positive.")
        if float(self.coreset_tau) <= 0.0:
            raise ValueError("coreset_tau must be positive.")
        if int(self.coreset_knn) < 0:
            raise ValueError("coreset_knn must be non-negative.")
        if int(self.operator_real_probe_limit) < 0 or int(self.operator_key_probe_limit) < 0:
            raise ValueError("operator probe limits must be non-negative; 0 means use all available probes.")
        if str(self.operator_probe_source) not in {"local_plus_tail", "local_only", "tail_only"}:
            raise ValueError("operator_probe_source must be one of: local_plus_tail, local_only, tail_only.")
        if not (0.0 <= float(self.operator_mass_uniform_mix) <= 1.0):
            raise ValueError("operator_mass_uniform_mix must be in [0, 1].")
        if int(self.operator_min_chunk_keep) < 0:
            raise ValueError("operator_min_chunk_keep must be non-negative.")
        if int(self.snap_kernel_size) < 1:
            raise ValueError("snap_kernel_size must be positive.")
        if str(self.snap_pooling) not in {"avgpool", "maxpool"}:
            raise ValueError("snap_pooling must be one of: avgpool, maxpool.")
        if float(self.layer_budget_rho) < 0.0:
            raise ValueError("layer_budget_rho must be non-negative.")
        if not (0.0 <= float(self.layer_budget_min_keep_ratio) <= float(self.layer_budget_max_keep_ratio) <= 1.0):
            raise ValueError("layer budget min/max keep ratios must satisfy 0 <= min <= max <= 1.")
        if str(self.risk_mode) != "mean":
            raise ValueError("SpectralKV supports only mean risk_mode.")
