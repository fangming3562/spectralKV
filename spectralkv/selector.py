from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .masks import build_observation_causal_mask


@dataclass
class JAOCSelection:
    keep_indices: torch.Tensor
    dropped_indices: torch.Tensor
    final_loss: float
    target_keep: int
    forced_count: int
    rounds: int
    loss_trace: list[float] = field(default_factory=list)
    budget_trace: list[int] = field(default_factory=list)
    min_denominator: float = 1.0
    batch_drop: int = 1
    score_block_size: int = 512
    solver: str = "drop"


def repeat_kv_for_scoring(cache: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    """Repeat [Hkv, T, D] cache to [Hq, T, D] for GQA scoring."""

    if cache.ndim != 3:
        raise ValueError(f"cache must be [heads, seq, dim], got {tuple(cache.shape)}")
    groups = max(int(num_key_value_groups), 1)
    if groups == 1:
        return cache
    return cache[:, None, :, :].expand(-1, groups, -1, -1).reshape(
        int(cache.shape[0]) * groups,
        int(cache.shape[1]),
        int(cache.shape[2]),
    )


def _validate_inputs(
    *,
    q_obs: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    force_keep_mask: torch.Tensor,
    num_key_value_groups: int,
) -> tuple[int, int, int, int]:
    if q_obs.ndim != 3:
        raise ValueError(f"q_obs must be [Hq, R, D], got {tuple(q_obs.shape)}")
    if k_cache.ndim != 3 or v_cache.ndim != 3:
        raise ValueError("k_cache and v_cache must be [Hkv, T, D].")
    if tuple(k_cache.shape) != tuple(v_cache.shape):
        raise ValueError(f"k/v shape mismatch: k={tuple(k_cache.shape)} v={tuple(v_cache.shape)}")
    hq, obs_count, head_dim = (int(q_obs.shape[0]), int(q_obs.shape[1]), int(q_obs.shape[2]))
    hkv, seq_len, kv_dim = (int(k_cache.shape[0]), int(k_cache.shape[1]), int(k_cache.shape[2]))
    if kv_dim != head_dim:
        raise ValueError(f"head dim mismatch: q={head_dim} kv={kv_dim}")
    if hkv * max(int(num_key_value_groups), 1) != hq:
        raise ValueError(
            f"GQA mismatch: Hkv={hkv}, groups={int(num_key_value_groups)}, Hq={hq}"
        )
    if force_keep_mask.shape != (seq_len,):
        raise ValueError(f"force_keep_mask must be [{seq_len}], got {tuple(force_keep_mask.shape)}")
    if obs_count < 1 or seq_len < 1:
        raise ValueError("obs_count and seq_len must be positive.")
    return hq, obs_count, seq_len, head_dim


def compute_observation_attention(
    *,
    q_obs: torch.Tensor,
    k_cache: torch.Tensor,
    num_key_value_groups: int,
    obs_causal_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute observation attention alpha [Hq, R, T] from RoPE-applied q/k."""

    groups = max(int(num_key_value_groups), 1)
    if groups == 1:
        logits = torch.einsum("hrd,htd->hrt", q_obs.float(), k_cache.float()) / math.sqrt(
            float(q_obs.shape[-1])
        )
    else:
        hkv = int(k_cache.shape[0])
        scale = 1.0 / math.sqrt(float(q_obs.shape[-1]))
        logits = torch.einsum(
            "hgrd,htd->hgrt",
            q_obs.float().reshape(hkv, groups, int(q_obs.shape[1]), int(q_obs.shape[2])),
            k_cache.float(),
        ).reshape(int(q_obs.shape[0]), int(q_obs.shape[1]), int(k_cache.shape[1])) * scale
    if obs_causal_mask is not None:
        if obs_causal_mask.shape != (int(q_obs.shape[1]), int(k_cache.shape[1])):
            raise ValueError(
                f"obs_causal_mask must be [R,T], got {tuple(obs_causal_mask.shape)}"
            )
        logits = logits.masked_fill(~obs_causal_mask.bool().view(1, int(q_obs.shape[1]), int(k_cache.shape[1])), float("-inf"))
    return torch.softmax(logits, dim=-1, dtype=torch.float32)


def _full_outputs(alpha: torch.Tensor, v_rep: torch.Tensor) -> torch.Tensor:
    return torch.einsum("hrt,htd->hrd", alpha.float(), v_rep.float())


def _full_outputs_grouped(
    *,
    alpha: torch.Tensor,
    v_cache: torch.Tensor,
    num_key_value_groups: int,
) -> torch.Tensor:
    """Full observation outputs without materializing repeated GQA values."""

    groups = max(int(num_key_value_groups), 1)
    if groups == 1:
        return torch.einsum("hrt,htd->hrd", alpha.float(), v_cache.float())
    hkv = int(v_cache.shape[0])
    return torch.einsum(
        "hgrt,htd->hgrd",
        alpha.float().reshape(hkv, groups, int(alpha.shape[1]), int(alpha.shape[2])),
        v_cache.float(),
    ).reshape(int(alpha.shape[0]), int(alpha.shape[1]), int(v_cache.shape[-1]))


def _precompute_x_sq_grouped(
    *,
    outputs: torch.Tensor,
    v_cache: torch.Tensor,
    num_key_value_groups: int,
) -> torch.Tensor:
    """Precompute ||v_t - o_r||^2 for every query head/query/token."""

    groups = max(int(num_key_value_groups), 1)
    if groups == 1:
        value_sq = v_cache.float().square().sum(dim=-1)
        output_sq = outputs.square().sum(dim=-1)
        output_value = torch.einsum("hrd,htd->hrt", outputs, v_cache.float())
        return (value_sq[:, None, :] + output_sq[:, :, None] - 2.0 * output_value).clamp_min_(0.0)

    hkv = int(v_cache.shape[0])
    obs_count = int(outputs.shape[1])
    head_dim = int(v_cache.shape[-1])
    outputs_g = outputs.reshape(hkv, groups, obs_count, head_dim)
    value_sq = v_cache.float().square().sum(dim=-1)
    output_sq = outputs_g.square().sum(dim=-1)
    output_value = torch.einsum("hgrd,htd->hgrt", outputs_g, v_cache.float())
    return (
        value_sq[:, None, None, :]
        + output_sq[:, :, :, None]
        - 2.0 * output_value
    ).clamp_min_(0.0).reshape(int(outputs.shape[0]), obs_count, int(v_cache.shape[1]))


def _prepare_query_weights(
    *,
    query_weights: torch.Tensor | None,
    num_heads: int,
    obs_count: int,
    device: torch.device,
) -> torch.Tensor | None:
    if query_weights is None:
        return None
    weights = query_weights.to(device=device, dtype=torch.float32)
    if weights.ndim == 1:
        if int(weights.numel()) != int(obs_count):
            raise ValueError(f"query_weights must have length {obs_count}, got {int(weights.numel())}")
        weights = weights.view(1, int(obs_count)).expand(int(num_heads), int(obs_count))
    elif weights.ndim == 2:
        if tuple(weights.shape) != (int(num_heads), int(obs_count)):
            raise ValueError(
                f"query_weights must be [{num_heads}, {obs_count}], got {tuple(weights.shape)}"
            )
    else:
        raise ValueError(f"query_weights must be [R] or [H,R], got {tuple(weights.shape)}")
    weights = torch.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
    mean = weights.mean().clamp_min(1e-8)
    return weights / mean


def _prepare_metric_diag(
    *,
    metric_diag: torch.Tensor | None,
    num_heads: int,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor | None:
    if metric_diag is None:
        return None
    metric = metric_diag.to(device=device, dtype=torch.float32)
    expected = (int(num_heads), int(head_dim))
    if tuple(metric.shape) != expected:
        raise ValueError(f"metric_diag must be {expected}, got {tuple(metric.shape)}")
    metric = torch.nan_to_num(metric, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
    mean = metric.mean()
    if not bool(torch.isfinite(mean).item()) or float(mean.item()) <= 1e-8:
        return torch.ones(expected, dtype=torch.float32, device=device)
    return metric / mean.clamp_min(1e-8)


def _aggregate_group_loss_terms(
    loss_terms: torch.Tensor,
    *,
    risk_mode: str,
    cvar_beta: float,
    logsumexp_tau: float,
) -> torch.Tensor:
    """Aggregate per-head/query losses.

    loss_terms is either [H,R] for the current set or [Hkv,G,R,B] for candidate
    scores. The candidate dimension, when present, is kept in the output.
    """

    mode = str(risk_mode)
    if mode not in {"mean", "max", "cvar", "logsumexp"}:
        raise ValueError("risk_mode must be one of: mean, max, cvar, logsumexp.")

    if loss_terms.ndim == 2:
        flat = loss_terms.reshape(-1)
        candidate_dim = False
    elif loss_terms.ndim == 4:
        flat = loss_terms.reshape(-1, int(loss_terms.shape[-1]))
        candidate_dim = True
    else:
        raise ValueError(f"loss_terms must be [H,R] or [Hkv,G,R,B], got {tuple(loss_terms.shape)}")

    if mode == "mean":
        return flat.mean(dim=0) if candidate_dim else flat.mean()

    if mode == "max":
        return flat.max(dim=0).values if candidate_dim else flat.max()

    group_count = int(flat.shape[0])
    if mode == "cvar":
        tail_count = max(1, int(math.ceil((1.0 - float(cvar_beta)) * float(group_count))))
        return torch.topk(flat, k=tail_count, dim=0, largest=True).values.mean(dim=0)

    tau = float(logsumexp_tau)
    value = torch.logsumexp(tau * flat, dim=0) / tau - math.log(max(group_count, 1)) / tau
    return value


def _weighted_loss_mean(
    *,
    residual: torch.Tensor,
    denominator: torch.Tensor,
    query_weights: torch.Tensor | None,
    metric_diag: torch.Tensor | None = None,
    risk_mode: str = "mean",
    cvar_beta: float = 0.9,
    logsumexp_tau: float = 10.0,
) -> torch.Tensor:
    if metric_diag is None:
        numerator = residual.square().sum(dim=-1)
    else:
        numerator = (residual.square() * metric_diag[:, None, :]).sum(dim=-1)
    loss = numerator / denominator.clamp_min(1e-30).square() / float(residual.shape[-1])
    if query_weights is not None:
        loss = loss * query_weights
    return _aggregate_group_loss_terms(
        loss,
        risk_mode=str(risk_mode),
        cvar_beta=float(cvar_beta),
        logsumexp_tau=float(logsumexp_tau),
    )


def jaoc_formula_loss_from_drop(
    *,
    q_obs: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    drop_indices: torch.Tensor,
    num_key_value_groups: int,
    obs_causal_mask: torch.Tensor | None = None,
    denom_eps: float = 1e-4,
) -> torch.Tensor:
    """Exact JAOC formula loss for a dropped token set."""

    force_mask = torch.zeros((int(k_cache.shape[1]),), dtype=torch.bool, device=k_cache.device)
    _validate_inputs(
        q_obs=q_obs,
        k_cache=k_cache,
        v_cache=v_cache,
        force_keep_mask=force_mask,
        num_key_value_groups=int(num_key_value_groups),
    )
    drop_indices = drop_indices.to(device=k_cache.device, dtype=torch.long).flatten().unique(sorted=True)
    alpha = compute_observation_attention(
        q_obs=q_obs,
        k_cache=k_cache,
        num_key_value_groups=int(num_key_value_groups),
        obs_causal_mask=obs_causal_mask,
    )
    v_rep = repeat_kv_for_scoring(v_cache, num_key_value_groups).float()
    output = _full_outputs(alpha, v_rep)
    if int(drop_indices.numel()) == 0:
        return torch.zeros((), dtype=torch.float32, device=k_cache.device)
    alpha_drop = alpha.index_select(2, drop_indices)
    v_drop = v_rep.index_select(1, drop_indices)
    mass = alpha_drop.sum(dim=-1)
    residual = torch.einsum("hrk,hkd->hrd", alpha_drop, v_drop) - mass.unsqueeze(-1) * output
    return (residual / (1.0 - mass).clamp_min(float(denom_eps)).unsqueeze(-1)).square().mean()


def jaoc_direct_loss_from_keep(
    *,
    q_obs: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    keep_indices: torch.Tensor,
    num_key_value_groups: int,
    obs_causal_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Direct full-vs-compressed attention output loss for a kept token set."""

    force_mask = torch.zeros((int(k_cache.shape[1]),), dtype=torch.bool, device=k_cache.device)
    _validate_inputs(
        q_obs=q_obs,
        k_cache=k_cache,
        v_cache=v_cache,
        force_keep_mask=force_mask,
        num_key_value_groups=int(num_key_value_groups),
    )
    keep_indices = keep_indices.to(device=k_cache.device, dtype=torch.long).flatten().unique(sorted=True)
    alpha_full = compute_observation_attention(
        q_obs=q_obs,
        k_cache=k_cache,
        num_key_value_groups=int(num_key_value_groups),
        obs_causal_mask=obs_causal_mask,
    )
    v_rep = repeat_kv_for_scoring(v_cache, num_key_value_groups).float()
    full_output = _full_outputs(alpha_full, v_rep)
    k_keep = k_cache.index_select(1, keep_indices)
    v_keep = v_cache.index_select(1, keep_indices)
    keep_mask = None
    if obs_causal_mask is not None:
        keep_mask = obs_causal_mask.index_select(1, keep_indices)
    alpha_keep = compute_observation_attention(
        q_obs=q_obs,
        k_cache=k_keep,
        num_key_value_groups=int(num_key_value_groups),
        obs_causal_mask=keep_mask,
    )
    compressed_output = _full_outputs(alpha_keep, repeat_kv_for_scoring(v_keep, num_key_value_groups).float())
    return (full_output - compressed_output).square().mean()


def _score_candidates_expanded(
    *,
    alpha: torch.Tensor,
    values: torch.Tensor,
    outputs: torch.Tensor,
    candidate_idx: torch.Tensor,
    residual: torch.Tensor,
    mass: torch.Tensor,
    current_loss: torch.Tensor,
    score_block_size: int,
    denom_eps: float,
) -> torch.Tensor:
    """Score candidate one-token deletions without materializing [T,R,D] phi."""

    scores = torch.empty((int(candidate_idx.numel()),), dtype=torch.float32, device=values.device)
    block_size = max(int(score_block_size), 1)
    den = (1.0 - mass).clamp_min(float(denom_eps))
    residual_sq = residual.square().sum(dim=-1)
    residual_output = (residual * outputs).sum(dim=-1)
    output_sq = outputs.square().sum(dim=-1)
    for start in range(0, int(candidate_idx.numel()), block_size):
        block = slice(start, min(start + block_size, int(candidate_idx.numel())))
        idx = candidate_idx[block]
        local_alpha = alpha.index_select(2, idx)
        local_values = values.index_select(1, idx)
        value_sq = local_values.square().sum(dim=-1)
        output_value = torch.einsum("hrd,hbd->hrb", outputs, local_values)
        x_sq = (value_sq[:, None, :] + output_sq[:, :, None] - 2.0 * output_value).clamp_min(0.0)
        residual_value = torch.einsum("hrd,hbd->hrb", residual, local_values)
        inner = residual_value - residual_output[:, :, None]
        den_new = den[:, :, None] - local_alpha
        numerator_new = (
            residual_sq[:, :, None]
            + 2.0 * local_alpha * inner
            + local_alpha.square() * x_sq
        )
        loss_new = numerator_new / den_new.clamp_min(float(denom_eps)).square()
        cand_loss = loss_new.mean(dim=(0, 1)) / float(values.shape[-1])
        delta = cand_loss - current_loss
        invalid = (den_new <= float(denom_eps)).any(dim=(0, 1))
        scores[block] = delta.masked_fill(invalid, float("inf"))
    return scores


def _score_candidates_grouped(
    *,
    alpha: torch.Tensor,
    v_cache: torch.Tensor,
    outputs: torch.Tensor,
    candidate_idx: torch.Tensor,
    residual: torch.Tensor,
    mass: torch.Tensor,
    current_loss: torch.Tensor,
    num_key_value_groups: int,
    x_sq_cache: torch.Tensor | None,
    query_weights: torch.Tensor | None,
    metric_diag: torch.Tensor | None = None,
    risk_mode: str = "mean",
    cvar_beta: float = 0.9,
    logsumexp_tau: float = 10.0,
    score_block_size: int,
    denom_eps: float,
) -> torch.Tensor:
    """Score one-token deletions without repeated GQA values."""

    groups = max(int(num_key_value_groups), 1)
    scores = torch.empty((int(candidate_idx.numel()),), dtype=torch.float32, device=v_cache.device)
    block_size = max(int(score_block_size), 1)
    den = (1.0 - mass).clamp_min(float(denom_eps))
    if metric_diag is None:
        residual_sq = residual.square().sum(dim=-1)
        residual_output = (residual * outputs).sum(dim=-1)
        output_sq = outputs.square().sum(dim=-1)
    else:
        residual_sq = (residual.square() * metric_diag[:, None, :]).sum(dim=-1)
        residual_output = (residual * outputs * metric_diag[:, None, :]).sum(dim=-1)
        output_sq = (outputs.square() * metric_diag[:, None, :]).sum(dim=-1)
    hkv = int(v_cache.shape[0])
    obs_count = int(alpha.shape[1])
    head_dim = int(v_cache.shape[-1])
    alpha_g = alpha.reshape(hkv, groups, obs_count, int(alpha.shape[2]))
    outputs_g = outputs.reshape(hkv, groups, obs_count, head_dim)
    residual_g = residual.reshape(hkv, groups, obs_count, head_dim)
    den_g = den.reshape(hkv, groups, obs_count)
    residual_sq_g = residual_sq.reshape(hkv, groups, obs_count)
    residual_output_g = residual_output.reshape(hkv, groups, obs_count)
    output_sq_g = output_sq.reshape(hkv, groups, obs_count)
    metric_g = None
    if metric_diag is not None:
        metric_g = metric_diag.reshape(hkv, groups, head_dim)
    query_weights_g = None
    if query_weights is not None:
        query_weights_g = query_weights.reshape(hkv, groups, obs_count)
    x_sq_cache_g = None
    if x_sq_cache is not None:
        if metric_diag is not None:
            raise ValueError("x_sq_cache is only valid for the value_l2 metric.")
        if tuple(x_sq_cache.shape) != (int(alpha.shape[0]), int(alpha.shape[1]), int(alpha.shape[2])):
            raise ValueError(f"x_sq_cache has wrong shape: {tuple(x_sq_cache.shape)}")
        x_sq_cache_g = x_sq_cache.reshape(hkv, groups, obs_count, int(alpha.shape[2]))

    for start_idx in range(0, int(candidate_idx.numel()), block_size):
        block = slice(start_idx, min(start_idx + block_size, int(candidate_idx.numel())))
        idx = candidate_idx[block]
        local_alpha = alpha_g.index_select(3, idx)
        local_values = v_cache.index_select(1, idx).float()
        if x_sq_cache_g is None:
            if metric_g is None:
                value_sq = local_values.square().sum(dim=-1)
                output_value = torch.einsum("hgrd,hbd->hgrb", outputs_g, local_values)
                x_sq = (
                    value_sq[:, None, None, :]
                    + output_sq_g[:, :, :, None]
                    - 2.0 * output_value
                ).clamp_min(0.0)
            else:
                value_sq = (local_values[:, None, :, :].square() * metric_g[:, :, None, :]).sum(dim=-1)
                output_value = (
                    outputs_g[:, :, :, None, :]
                    * local_values[:, None, None, :, :]
                    * metric_g[:, :, None, None, :]
                ).sum(dim=-1)
                x_sq = (
                    value_sq[:, :, None, :]
                    + output_sq_g[:, :, :, None]
                    - 2.0 * output_value
                ).clamp_min(0.0)
        else:
            x_sq = x_sq_cache_g.index_select(3, idx)
        if metric_g is None:
            residual_value = torch.einsum("hgrd,hbd->hgrb", residual_g, local_values)
        else:
            residual_value = (
                residual_g[:, :, :, None, :]
                * local_values[:, None, None, :, :]
                * metric_g[:, :, None, None, :]
            ).sum(dim=-1)
        inner = residual_value - residual_output_g[:, :, :, None]
        den_new = den_g[:, :, :, None] - local_alpha
        numerator_new = (
            residual_sq_g[:, :, :, None]
            + 2.0 * local_alpha * inner
            + local_alpha.square() * x_sq
        )
        loss_terms = numerator_new / den_new.clamp_min(float(denom_eps)).square() / float(head_dim)
        if query_weights_g is not None:
            loss_terms = loss_terms * query_weights_g[:, :, :, None]
        invalid = (den_new <= float(denom_eps)).any(dim=(0, 1, 2))

        cand_loss = _aggregate_group_loss_terms(
            loss_terms,
            risk_mode=str(risk_mode),
            cvar_beta=float(cvar_beta),
            logsumexp_tau=float(logsumexp_tau),
        )
        delta = cand_loss - current_loss
        scores[block] = delta.masked_fill(invalid, float("inf"))
    return scores


def _score_candidates_keep_grouped(
    *,
    alpha: torch.Tensor,
    v_cache: torch.Tensor,
    outputs: torch.Tensor,
    candidate_idx: torch.Tensor,
    residual: torch.Tensor,
    mass: torch.Tensor,
    current_loss: torch.Tensor,
    num_key_value_groups: int,
    x_sq_cache: torch.Tensor | None,
    query_weights: torch.Tensor | None,
    metric_diag: torch.Tensor | None = None,
    risk_mode: str = "mean",
    cvar_beta: float = 0.9,
    logsumexp_tau: float = 10.0,
    score_block_size: int,
    denom_eps: float,
) -> torch.Tensor:
    """Score one-token additions for the keep-side JAOC objective.

    This is the complement form of the drop objective. For a kept set S, the
    exact compressed-output error is ||U(S) / M(S)||^2, where U(S) and M(S)
    are cumulative centered attention contribution and attention mass.
    """

    groups = max(int(num_key_value_groups), 1)
    scores = torch.empty((int(candidate_idx.numel()),), dtype=torch.float32, device=v_cache.device)
    block_size = max(int(score_block_size), 1)
    den = mass.clamp_min(float(denom_eps))
    if metric_diag is None:
        residual_sq = residual.square().sum(dim=-1)
        residual_output = (residual * outputs).sum(dim=-1)
        output_sq = outputs.square().sum(dim=-1)
    else:
        residual_sq = (residual.square() * metric_diag[:, None, :]).sum(dim=-1)
        residual_output = (residual * outputs * metric_diag[:, None, :]).sum(dim=-1)
        output_sq = (outputs.square() * metric_diag[:, None, :]).sum(dim=-1)
    hkv = int(v_cache.shape[0])
    obs_count = int(alpha.shape[1])
    head_dim = int(v_cache.shape[-1])
    alpha_g = alpha.reshape(hkv, groups, obs_count, int(alpha.shape[2]))
    outputs_g = outputs.reshape(hkv, groups, obs_count, head_dim)
    residual_g = residual.reshape(hkv, groups, obs_count, head_dim)
    den_g = den.reshape(hkv, groups, obs_count)
    residual_sq_g = residual_sq.reshape(hkv, groups, obs_count)
    residual_output_g = residual_output.reshape(hkv, groups, obs_count)
    output_sq_g = output_sq.reshape(hkv, groups, obs_count)
    metric_g = None
    if metric_diag is not None:
        metric_g = metric_diag.reshape(hkv, groups, head_dim)
    query_weights_g = None
    if query_weights is not None:
        query_weights_g = query_weights.reshape(hkv, groups, obs_count)
    x_sq_cache_g = None
    if x_sq_cache is not None:
        if metric_diag is not None:
            raise ValueError("x_sq_cache is only valid for the value_l2 metric.")
        if tuple(x_sq_cache.shape) != (int(alpha.shape[0]), int(alpha.shape[1]), int(alpha.shape[2])):
            raise ValueError(f"x_sq_cache has wrong shape: {tuple(x_sq_cache.shape)}")
        x_sq_cache_g = x_sq_cache.reshape(hkv, groups, obs_count, int(alpha.shape[2]))

    for start_idx in range(0, int(candidate_idx.numel()), block_size):
        block = slice(start_idx, min(start_idx + block_size, int(candidate_idx.numel())))
        idx = candidate_idx[block]
        local_alpha = alpha_g.index_select(3, idx)
        local_values = v_cache.index_select(1, idx).float()
        if x_sq_cache_g is None:
            if metric_g is None:
                value_sq = local_values.square().sum(dim=-1)
                output_value = torch.einsum("hgrd,hbd->hgrb", outputs_g, local_values)
                x_sq = (
                    value_sq[:, None, None, :]
                    + output_sq_g[:, :, :, None]
                    - 2.0 * output_value
                ).clamp_min(0.0)
            else:
                value_sq = (local_values[:, None, :, :].square() * metric_g[:, :, None, :]).sum(dim=-1)
                output_value = (
                    outputs_g[:, :, :, None, :]
                    * local_values[:, None, None, :, :]
                    * metric_g[:, :, None, None, :]
                ).sum(dim=-1)
                x_sq = (
                    value_sq[:, :, None, :]
                    + output_sq_g[:, :, :, None]
                    - 2.0 * output_value
                ).clamp_min(0.0)
        else:
            x_sq = x_sq_cache_g.index_select(3, idx)
        if metric_g is None:
            residual_value = torch.einsum("hgrd,hbd->hgrb", residual_g, local_values)
        else:
            residual_value = (
                residual_g[:, :, :, None, :]
                * local_values[:, None, None, :, :]
                * metric_g[:, :, None, None, :]
            ).sum(dim=-1)
        inner = residual_value - residual_output_g[:, :, :, None]
        den_new = den_g[:, :, :, None] + local_alpha
        numerator_new = (
            residual_sq_g[:, :, :, None]
            + 2.0 * local_alpha * inner
            + local_alpha.square() * x_sq
        )
        loss_terms = numerator_new / den_new.clamp_min(float(denom_eps)).square() / float(head_dim)
        if query_weights_g is not None:
            loss_terms = loss_terms * query_weights_g[:, :, :, None]

        cand_loss = _aggregate_group_loss_terms(
            loss_terms,
            risk_mode=str(risk_mode),
            cvar_beta=float(cvar_beta),
            logsumexp_tau=float(logsumexp_tau),
        )
        scores[block] = cand_loss - current_loss
    return scores


def _select_safe_batch(
    *,
    candidate_idx: torch.Tensor,
    scores: torch.Tensor,
    alpha: torch.Tensor,
    mass: torch.Tensor,
    max_count: int,
    denom_eps: float,
) -> torch.Tensor:
    """Pick a mini-batch whose cumulative removed mass remains feasible.

    _score_candidates_expanded only checks whether each candidate is safe when
    deleted alone from the current state. Mini-batch greedy also needs the
    accepted candidates to be safe jointly, because their attention masses add.
    """

    max_count = max(int(max_count), 0)
    if max_count == 0 or int(candidate_idx.numel()) == 0:
        return candidate_idx[:0]

    num_candidates = int(candidate_idx.numel())
    top_n = min(num_candidates, max(max_count * 2, 16))
    scores_cpu = scores.detach().cpu()
    candidate_cpu = candidate_idx.detach().cpu()
    finite_pos = torch.nonzero(torch.isfinite(scores_cpu), as_tuple=False).flatten()
    if int(finite_pos.numel()) == 0:
        return candidate_idx[:0]
    ordered_all = finite_pos.index_select(0, torch.argsort(scores_cpu.index_select(0, finite_pos)))
    accepted_ids: list[int] = []

    while True:
        ordered_pos = ordered_all[:top_n]
        top_token_ids_cpu = candidate_cpu.index_select(0, ordered_pos)
        top_token_ids = top_token_ids_cpu.to(device=candidate_idx.device, dtype=torch.long)
        alpha_top_cpu = alpha.index_select(2, top_token_ids).detach().cpu()
        trial_mass = mass.detach().cpu()
        accepted_ids = []
        for local_pos, token_idx_cpu in enumerate(top_token_ids_cpu.tolist()):
            token_mass = alpha_top_cpu[:, :, int(local_pos)]
            if bool(((1.0 - trial_mass - token_mass) <= float(denom_eps)).any().item()):
                continue
            accepted_ids.append(int(token_idx_cpu))
            trial_mass = trial_mass + token_mass
            if len(accepted_ids) >= max_count:
                break

        if len(accepted_ids) >= max_count or top_n >= int(ordered_all.numel()):
            break
        top_n = min(int(ordered_all.numel()), max(top_n * 2, top_n + 1))

    if not accepted_ids:
        return candidate_idx[:0]
    return torch.tensor(accepted_ids, device=candidate_idx.device, dtype=torch.long)


def _select_lowest_score_batch(
    *,
    candidate_idx: torch.Tensor,
    scores: torch.Tensor,
    max_count: int,
) -> torch.Tensor:
    """Pick the lowest finite-score candidates without a feasibility pass."""

    max_count = max(int(max_count), 0)
    if max_count == 0 or int(candidate_idx.numel()) == 0:
        return candidate_idx[:0]
    finite_pos = torch.nonzero(torch.isfinite(scores), as_tuple=False).flatten()
    if int(finite_pos.numel()) == 0:
        return candidate_idx[:0]
    k = min(max_count, int(finite_pos.numel()))
    finite_scores = scores.index_select(0, finite_pos)
    selected_pos = finite_pos.index_select(0, torch.topk(finite_scores, k=k, largest=False).indices)
    return candidate_idx.index_select(0, selected_pos)


def _build_nonforced_atoms(
    *,
    seq_len: int,
    force_keep_mask: torch.Tensor,
    atom_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build disjoint contiguous atoms over non-forced tokens.

    Forced tokens remain individual hard constraints outside the candidate atom
    set. Each candidate atom is a contiguous span inside a non-forced run.
    """

    size = max(int(atom_size), 1)
    device = force_keep_mask.device
    rows: list[list[int]] = []
    lengths: list[int] = []
    start = 0
    seq_len = int(seq_len)
    while start < seq_len:
        while start < seq_len and bool(force_keep_mask[start].item()):
            start += 1
        if start >= seq_len:
            break
        end = start
        while end < seq_len and not bool(force_keep_mask[end].item()):
            end += 1
        pos = start
        while pos < end:
            span = list(range(pos, min(pos + size, end)))
            rows.append(span + [span[-1]] * (size - len(span)))
            lengths.append(len(span))
            pos += size
        start = end
    if not rows:
        empty_idx = torch.empty((0, size), dtype=torch.long, device=device)
        empty_mask = torch.empty((0, size), dtype=torch.bool, device=device)
        empty_len = torch.empty((0,), dtype=torch.long, device=device)
        return empty_idx, empty_mask, empty_len
    atom_indices = torch.tensor(rows, dtype=torch.long, device=device)
    atom_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    arange = torch.arange(size, device=device).view(1, size)
    atom_mask = arange < atom_lengths.view(-1, 1)
    return atom_indices, atom_mask, atom_lengths


def _apply_atom_selected(
    *,
    alpha: torch.Tensor,
    values: torch.Tensor,
    outputs: torch.Tensor,
    residual: torch.Tensor,
    mass: torch.Tensor,
    atom_indices: torch.Tensor,
    atom_mask: torch.Tensor,
    selected_atoms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(selected_atoms.numel()) == 0:
        return mass, residual
    idx = atom_indices.index_select(0, selected_atoms)
    mask = atom_mask.index_select(0, selected_atoms).to(dtype=torch.float32)
    local_alpha = alpha[:, :, idx] * mask.view(1, 1, int(selected_atoms.numel()), -1)
    local_values = values[:, idx, :]
    mass_add = local_alpha.sum(dim=(2, 3))
    value_sum = torch.einsum("hral,hald->hrd", local_alpha, local_values)
    residual_add = value_sum - mass_add.unsqueeze(-1) * outputs
    return mass + mass_add, residual + residual_add


def _score_atom_candidates(
    *,
    alpha: torch.Tensor,
    values: torch.Tensor,
    outputs: torch.Tensor,
    atom_indices: torch.Tensor,
    atom_mask: torch.Tensor,
    candidate_atoms: torch.Tensor,
    residual: torch.Tensor,
    mass: torch.Tensor,
    current_loss: torch.Tensor,
    mode: str,
    score_block_size: int,
    denom_eps: float,
) -> torch.Tensor:
    scores = torch.empty((int(candidate_atoms.numel()),), dtype=torch.float32, device=values.device)
    block_size = max(int(score_block_size), 1)
    residual_sq = residual.square().sum(dim=-1)
    for start in range(0, int(candidate_atoms.numel()), block_size):
        block = slice(start, min(start + block_size, int(candidate_atoms.numel())))
        atom_ids = candidate_atoms[block]
        idx = atom_indices.index_select(0, atom_ids)
        mask = atom_mask.index_select(0, atom_ids).to(dtype=torch.float32)
        local_alpha = alpha[:, :, idx] * mask.view(1, 1, int(atom_ids.numel()), -1)
        local_values = values[:, idx, :]
        mass_atom = local_alpha.sum(dim=-1)
        value_sum = torch.einsum("hral,hald->hrad", local_alpha, local_values)
        residual_atom = value_sum - mass_atom.unsqueeze(-1) * outputs[:, :, None, :]
        numerator = (residual[:, :, None, :] + residual_atom).square().sum(dim=-1)
        if str(mode) == "keep":
            den_new = mass[:, :, None] + mass_atom
            loss_new = numerator / den_new.clamp_min(float(denom_eps)).square()
            cand_loss = loss_new.mean(dim=(0, 1)) / float(values.shape[-1])
            scores[block] = cand_loss - current_loss
        elif str(mode) == "drop":
            den_new = (1.0 - mass[:, :, None]) - mass_atom
            loss_new = numerator / den_new.clamp_min(float(denom_eps)).square()
            cand_loss = loss_new.mean(dim=(0, 1)) / float(values.shape[-1])
            invalid = (den_new <= float(denom_eps)).any(dim=(0, 1))
            scores[block] = (cand_loss - current_loss).masked_fill(invalid, float("inf"))
        else:
            raise ValueError("mode must be keep or drop.")
    return scores


def _select_atom_batch(
    *,
    candidate_atoms: torch.Tensor,
    scores: torch.Tensor,
    atom_lengths: torch.Tensor,
    max_tokens: int,
) -> torch.Tensor:
    if int(max_tokens) <= 0 or int(candidate_atoms.numel()) == 0:
        return candidate_atoms[:0]
    finite_pos = torch.nonzero(torch.isfinite(scores), as_tuple=False).flatten()
    if int(finite_pos.numel()) == 0:
        return candidate_atoms[:0]
    ordered = finite_pos.index_select(0, torch.argsort(scores.index_select(0, finite_pos)))
    selected: list[int] = []
    used = 0
    for pos in ordered.tolist():
        atom_id = int(candidate_atoms[int(pos)].item())
        cost = int(atom_lengths[atom_id].item())
        if used + cost > int(max_tokens):
            continue
        selected.append(atom_id)
        used += cost
        if used >= int(max_tokens):
            break
    if not selected:
        return candidate_atoms[:0]
    return torch.tensor(selected, dtype=torch.long, device=candidate_atoms.device)


@torch.no_grad()
def jaoc_select_layer_span(
    *,
    q_obs: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    budget: int,
    force_keep_mask: torch.Tensor,
    num_key_value_groups: int,
    atom_size: int,
    obs_positions: torch.Tensor | None = None,
    obs_causal_mask: torch.Tensor | None = None,
    use_causal_obs_mask: bool = True,
    batch_drop: int = 256,
    score_block_size: int = 128,
    denom_eps: float = 1e-4,
    solver: str = "auto",
) -> JAOCSelection:
    """JAOC with fixed contiguous span atoms over non-forced tokens."""

    if int(atom_size) <= 1:
        return jaoc_select_layer_fast(
            q_obs=q_obs,
            k_cache=k_cache,
            v_cache=v_cache,
            budget=int(budget),
            force_keep_mask=force_keep_mask,
            num_key_value_groups=int(num_key_value_groups),
            obs_positions=obs_positions,
            obs_causal_mask=obs_causal_mask,
            use_causal_obs_mask=bool(use_causal_obs_mask),
            batch_drop=int(batch_drop),
            score_block_size=int(score_block_size),
            denom_eps=float(denom_eps),
            solver=str(solver),
        )

    solver_mode = str(solver).lower()
    if solver_mode not in {"auto", "drop", "keep"}:
        raise ValueError("solver must be one of: auto, drop, keep.")
    _, _, seq_len, _ = _validate_inputs(
        q_obs=q_obs,
        k_cache=k_cache,
        v_cache=v_cache,
        force_keep_mask=force_keep_mask,
        num_key_value_groups=int(num_key_value_groups),
    )
    if bool(use_causal_obs_mask) and obs_causal_mask is None:
        if obs_positions is None:
            obs_positions = torch.arange(seq_len - int(q_obs.shape[1]), seq_len, device=q_obs.device, dtype=torch.long)
        obs_causal_mask = build_observation_causal_mask(obs_positions=obs_positions, seq_len=seq_len)

    force_keep_mask = force_keep_mask.to(device=k_cache.device, dtype=torch.bool).clone()
    target_keep = max(int(budget), int(force_keep_mask.sum().item()), 1)
    target_keep = min(target_keep, seq_len)
    if target_keep >= seq_len:
        keep = torch.arange(seq_len, device=k_cache.device, dtype=torch.long)
        return JAOCSelection(
            keep_indices=keep,
            dropped_indices=keep[:0],
            final_loss=0.0,
            target_keep=int(target_keep),
            forced_count=int(force_keep_mask.sum().item()),
            rounds=0,
            loss_trace=[0.0],
            min_denominator=1.0,
            batch_drop=int(batch_drop),
            score_block_size=int(score_block_size),
            solver="span_full",
        )

    alpha = compute_observation_attention(
        q_obs=q_obs,
        k_cache=k_cache,
        num_key_value_groups=int(num_key_value_groups),
        obs_causal_mask=obs_causal_mask,
    )
    values = repeat_kv_for_scoring(v_cache, int(num_key_value_groups)).float()
    outputs = _full_outputs(alpha, values)
    residual = torch.zeros_like(outputs, dtype=torch.float32, device=v_cache.device)
    mass = torch.zeros((int(alpha.shape[0]), int(alpha.shape[1])), dtype=torch.float32, device=v_cache.device)
    atom_indices, atom_mask, atom_lengths = _build_nonforced_atoms(
        seq_len=int(seq_len),
        force_keep_mask=force_keep_mask,
        atom_size=int(atom_size),
    )
    atom_count = int(atom_lengths.numel())
    if atom_count == 0:
        keep = torch.nonzero(force_keep_mask, as_tuple=False).flatten().sort().values
        return JAOCSelection(
            keep_indices=keep,
            dropped_indices=torch.nonzero(~force_keep_mask, as_tuple=False).flatten().sort().values,
            final_loss=0.0,
            target_keep=int(target_keep),
            forced_count=int(force_keep_mask.sum().item()),
            rounds=0,
            min_denominator=1.0,
            batch_drop=int(batch_drop),
            score_block_size=int(score_block_size),
            solver="span_empty",
        )

    forced_count = int(force_keep_mask.sum().item())
    target_drop_tokens = max(seq_len - target_keep, 0)
    target_add_tokens = max(target_keep - forced_count, 0)
    chosen_solver = solver_mode
    if chosen_solver == "auto":
        chosen_solver = "keep" if int(target_add_tokens) <= int(target_drop_tokens) else "drop"

    selected_atoms_mask = torch.zeros((atom_count,), dtype=torch.bool, device=v_cache.device)
    rounds = 0

    if chosen_solver == "keep":
        kept_mask = force_keep_mask.clone()
        forced_idx = torch.nonzero(force_keep_mask, as_tuple=False).flatten()
        if int(forced_idx.numel()) > 0:
            force_atom_indices = forced_idx.view(-1, 1)
            force_atom_mask = torch.ones_like(force_atom_indices, dtype=torch.bool)
            mass, residual = _apply_atom_selected(
                alpha=alpha,
                values=values,
                outputs=outputs,
                residual=residual,
                mass=mass,
                atom_indices=force_atom_indices,
                atom_mask=force_atom_mask,
                selected_atoms=torch.arange(int(forced_idx.numel()), device=v_cache.device, dtype=torch.long),
            )
        keep_count = int(forced_count)
        while keep_count < target_keep:
            candidate_atoms = torch.nonzero(~selected_atoms_mask, as_tuple=False).flatten()
            if int(candidate_atoms.numel()) == 0:
                break
            den = mass.clamp_min(float(denom_eps))
            current_loss = (residual / den.unsqueeze(-1)).square().mean()
            remaining = int(target_keep) - int(keep_count)
            scores = _score_atom_candidates(
                alpha=alpha,
                values=values,
                outputs=outputs,
                atom_indices=atom_indices,
                atom_mask=atom_mask,
                candidate_atoms=candidate_atoms,
                residual=residual,
                mass=mass,
                current_loss=current_loss,
                mode="keep",
                score_block_size=int(score_block_size),
                denom_eps=float(denom_eps),
            )
            selected = _select_atom_batch(
                candidate_atoms=candidate_atoms,
                scores=scores,
                atom_lengths=atom_lengths,
                max_tokens=min(int(batch_drop) * int(atom_size), int(remaining)),
            )
            if int(selected.numel()) == 0:
                break
            selected_atoms_mask.index_fill_(0, selected, True)
            mass, residual = _apply_atom_selected(
                alpha=alpha,
                values=values,
                outputs=outputs,
                residual=residual,
                mass=mass,
                atom_indices=atom_indices,
                atom_mask=atom_mask,
                selected_atoms=selected,
            )
            flat_idx = atom_indices.index_select(0, selected).flatten()
            flat_mask = atom_mask.index_select(0, selected).flatten()
            add_idx = flat_idx[flat_mask]
            kept_mask.index_fill_(0, add_idx, True)
            keep_count = int(kept_mask.sum().item())
            rounds += 1
        keep_indices = torch.nonzero(kept_mask, as_tuple=False).flatten().sort().values
        final_den = mass.clamp_min(float(denom_eps))
        final_loss = (residual / final_den.unsqueeze(-1)).square().mean()
        return JAOCSelection(
            keep_indices=keep_indices,
            dropped_indices=torch.nonzero(~kept_mask, as_tuple=False).flatten().sort().values,
            final_loss=float(final_loss.item()),
            target_keep=int(target_keep),
            forced_count=int(forced_count),
            rounds=int(rounds),
            min_denominator=float(mass.min().item()),
            batch_drop=int(batch_drop),
            score_block_size=int(score_block_size),
            solver=f"span{int(atom_size)}_keep",
        )

    dropped_atoms_mask = torch.zeros((atom_count,), dtype=torch.bool, device=v_cache.device)
    dropped_mask = torch.zeros((seq_len,), dtype=torch.bool, device=v_cache.device)
    dropped_tokens = 0
    while dropped_tokens < target_drop_tokens:
        candidate_atoms = torch.nonzero(~dropped_atoms_mask, as_tuple=False).flatten()
        if int(candidate_atoms.numel()) == 0:
            break
        den = (1.0 - mass).clamp_min(float(denom_eps))
        current_loss = (residual / den.unsqueeze(-1)).square().mean()
        remaining = int(target_drop_tokens) - int(dropped_tokens)
        scores = _score_atom_candidates(
            alpha=alpha,
            values=values,
            outputs=outputs,
            atom_indices=atom_indices,
            atom_mask=atom_mask,
            candidate_atoms=candidate_atoms,
            residual=residual,
            mass=mass,
            current_loss=current_loss,
            mode="drop",
            score_block_size=int(score_block_size),
            denom_eps=float(denom_eps),
        )
        selected = _select_atom_batch(
            candidate_atoms=candidate_atoms,
            scores=scores,
            atom_lengths=atom_lengths,
            max_tokens=min(int(batch_drop) * int(atom_size), int(remaining)),
        )
        if int(selected.numel()) == 0:
            break
        dropped_atoms_mask.index_fill_(0, selected, True)
        mass, residual = _apply_atom_selected(
            alpha=alpha,
            values=values,
            outputs=outputs,
            residual=residual,
            mass=mass,
            atom_indices=atom_indices,
            atom_mask=atom_mask,
            selected_atoms=selected,
        )
        flat_idx = atom_indices.index_select(0, selected).flatten()
        flat_mask = atom_mask.index_select(0, selected).flatten()
        drop_idx = flat_idx[flat_mask]
        dropped_mask.index_fill_(0, drop_idx, True)
        dropped_tokens = int(dropped_mask.sum().item())
        rounds += 1
    keep_indices = torch.nonzero(~dropped_mask, as_tuple=False).flatten().sort().values
    final_den = (1.0 - mass).clamp_min(float(denom_eps))
    final_loss = (residual / final_den.unsqueeze(-1)).square().mean()
    return JAOCSelection(
        keep_indices=keep_indices,
        dropped_indices=torch.nonzero(dropped_mask, as_tuple=False).flatten().sort().values,
        final_loss=float(final_loss.item()),
        target_keep=int(target_keep),
        forced_count=int(forced_count),
        rounds=int(rounds),
        min_denominator=float((1.0 - mass).min().item()),
        batch_drop=int(batch_drop),
        score_block_size=int(score_block_size),
        solver=f"span{int(atom_size)}_drop",
    )


def _candidate_pool_size(
    *,
    candidate_count: int,
    batch_count: int,
    candidate_pool_factor: float,
    candidate_pool_min: int,
) -> int:
    if float(candidate_pool_factor) <= 0.0:
        return int(candidate_count)
    pool = int(math.ceil(max(int(batch_count), 1) * float(candidate_pool_factor)))
    pool = max(pool, int(candidate_pool_min), int(batch_count), 1)
    return min(int(pool), int(candidate_count))


def _candidate_pool_by_attention_proxy(
    *,
    candidate_idx: torch.Tensor,
    attention_proxy: torch.Tensor,
    pool_size: int,
    largest: bool = False,
) -> torch.Tensor:
    """Pick low-attention candidates for exact JAOC rescoring.

    The proxy is only a GPU prefilter. Final ranking inside the pool still uses
    exact JAOC marginal damage and the cumulative denominator feasibility check.
    """

    pool_size = min(max(int(pool_size), 1), int(candidate_idx.numel()))
    if pool_size >= int(candidate_idx.numel()):
        return candidate_idx
    proxy = attention_proxy.index_select(0, candidate_idx)
    pos = torch.topk(proxy, k=pool_size, largest=bool(largest)).indices
    return candidate_idx.index_select(0, pos)


def _pair_pool_from_single_scores(
    *,
    candidate_idx: torch.Tensor,
    single_scores: torch.Tensor,
    pair_pool_size: int,
) -> torch.Tensor:
    """Build a pair pool from both easy and hard single-token candidates."""

    pool_size = min(max(int(pair_pool_size), 0), int(candidate_idx.numel()))
    if pool_size < 2:
        return candidate_idx[:0]
    finite_pos = torch.nonzero(torch.isfinite(single_scores), as_tuple=False).flatten()
    if int(finite_pos.numel()) < 2:
        return candidate_idx[:0]
    if int(finite_pos.numel()) <= pool_size:
        return candidate_idx.index_select(0, finite_pos)
    ordered = finite_pos.index_select(0, torch.argsort(single_scores.index_select(0, finite_pos)))
    low_count = max(pool_size // 2, 1)
    high_count = max(pool_size - low_count, 1)
    pool_pos = torch.cat([ordered[:low_count], ordered[-high_count:]], dim=0).unique(sorted=False)
    return candidate_idx.index_select(0, pool_pos)


def _score_pairs_materialized(
    *,
    alpha: torch.Tensor,
    v_cache: torch.Tensor,
    outputs: torch.Tensor,
    residual: torch.Tensor,
    mass: torch.Tensor,
    pair_idx: torch.Tensor,
    current_loss: torch.Tensor,
    num_key_value_groups: int,
    x_sq_cache: torch.Tensor | None,
    denom_eps: float,
) -> torch.Tensor:
    """Exact joint marginal scores for all unordered pairs in a small pool."""

    pool_size = int(pair_idx.numel())
    if pool_size < 2:
        return torch.empty((pool_size, pool_size), dtype=torch.float32, device=alpha.device).fill_(float("inf"))

    groups = max(int(num_key_value_groups), 1)
    values = repeat_kv_for_scoring(v_cache.index_select(1, pair_idx), groups).float()
    local_alpha = alpha.index_select(2, pair_idx)
    residual_sq = residual.square().sum(dim=-1)
    residual_output = (residual * outputs).sum(dim=-1)
    residual_value = torch.einsum("hrd,hpd->hrp", residual, values)
    residual_phi = local_alpha * (residual_value - residual_output[:, :, None])

    if x_sq_cache is None:
        value_sq = values.square().sum(dim=-1)
        output_sq = outputs.square().sum(dim=-1)
        output_value = torch.einsum("hrd,hpd->hrp", outputs, values)
        x_sq = (value_sq[:, None, :] + output_sq[:, :, None] - 2.0 * output_value).clamp_min(0.0)
    else:
        x_sq = x_sq_cache.index_select(2, pair_idx)
    phi_sq = local_alpha.square() * x_sq

    value_dot = torch.einsum("hpd,hqd->hpq", values, values)
    output_value = torch.einsum("hrd,hpd->hrp", outputs, values)
    output_sq = outputs.square().sum(dim=-1)
    x_dot = (
        value_dot[:, None, :, :]
        - output_value[:, :, :, None]
        - output_value[:, :, None, :]
        + output_sq[:, :, None, None]
    )
    phi_dot = local_alpha[:, :, :, None] * local_alpha[:, :, None, :] * x_dot

    numerator = (
        residual_sq[:, :, None, None]
        + 2.0 * (residual_phi[:, :, :, None] + residual_phi[:, :, None, :])
        + phi_sq[:, :, :, None]
        + phi_sq[:, :, None, :]
        + 2.0 * phi_dot
    )
    den_new = (
        (1.0 - mass)[:, :, None, None]
        - local_alpha[:, :, :, None]
        - local_alpha[:, :, None, :]
    )
    loss = numerator / den_new.clamp_min(float(denom_eps)).square()
    pair_scores = loss.sum(dim=(0, 1)) / float(alpha.shape[0] * alpha.shape[1] * v_cache.shape[-1])
    pair_scores = pair_scores - current_loss
    pair_scores = pair_scores / 2.0
    invalid = (den_new <= float(denom_eps)).any(dim=(0, 1))
    pair_scores = pair_scores.masked_fill(invalid, float("inf"))
    same_or_lower = torch.ones((pool_size, pool_size), dtype=torch.bool, device=alpha.device).tril()
    return pair_scores.masked_fill(same_or_lower, float("inf"))


def _select_safe_pair_batch(
    *,
    pair_idx: torch.Tensor,
    pair_scores: torch.Tensor,
    alpha: torch.Tensor,
    mass: torch.Tensor,
    max_tokens: int,
    denom_eps: float,
) -> torch.Tensor:
    """Select non-overlapping safe pairs from pair score matrix."""

    max_pairs = max(int(max_tokens) // 2, 0)
    if max_pairs == 0 or int(pair_idx.numel()) < 2:
        return pair_idx[:0]
    pool_size = int(pair_idx.numel())
    flat_scores_cpu = pair_scores.flatten().detach().cpu()
    finite_pos = torch.nonzero(torch.isfinite(flat_scores_cpu), as_tuple=False).flatten()
    if int(finite_pos.numel()) == 0:
        return pair_idx[:0]
    ordered_pos = finite_pos.index_select(0, torch.argsort(flat_scores_cpu.index_select(0, finite_pos)))
    pair_idx_cpu = pair_idx.detach().cpu()
    alpha_pool_cpu = alpha.index_select(2, pair_idx).detach().cpu()
    trial_mass = mass.detach().cpu()
    used = torch.zeros((pool_size,), dtype=torch.bool)
    accepted_ids: list[int] = []
    for flat_pos_tensor in ordered_pos:
        flat_pos = int(flat_pos_tensor.item())
        left = flat_pos // pool_size
        right = flat_pos % pool_size
        if left >= right or bool(used[left].item()) or bool(used[right].item()):
            continue
        pair_mass = alpha_pool_cpu[:, :, left] + alpha_pool_cpu[:, :, right]
        if bool(((1.0 - trial_mass - pair_mass) <= float(denom_eps)).any().item()):
            continue
        accepted_ids.extend([int(pair_idx_cpu[left].item()), int(pair_idx_cpu[right].item())])
        used[left] = True
        used[right] = True
        trial_mass = trial_mass + pair_mass
        if len(accepted_ids) >= max_pairs * 2:
            break

    if not accepted_ids:
        return pair_idx[:0]
    return torch.tensor(accepted_ids, device=pair_idx.device, dtype=torch.long)


def _apply_selected_grouped(
    *,
    alpha: torch.Tensor,
    v_cache: torch.Tensor,
    outputs: torch.Tensor,
    residual: torch.Tensor,
    mass: torch.Tensor,
    selected: torch.Tensor,
    num_key_value_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update mass/residual for selected deletions without repeated GQA values."""

    groups = max(int(num_key_value_groups), 1)
    alpha_sel = alpha.index_select(2, selected)
    mass_add = alpha_sel.sum(dim=-1)
    hkv = int(v_cache.shape[0])
    obs_count = int(alpha.shape[1])
    head_dim = int(v_cache.shape[-1])
    residual_g = residual.reshape(hkv, groups, obs_count, head_dim)
    outputs_g = outputs.reshape(hkv, groups, obs_count, head_dim)
    alpha_sel_g = alpha_sel.reshape(hkv, groups, obs_count, int(selected.numel()))
    mass_add_g = mass_add.reshape(hkv, groups, obs_count)
    value_sel = v_cache.index_select(1, selected).float()
    residual_add = torch.einsum("hgrk,hkd->hgrd", alpha_sel_g, value_sel) - mass_add_g.unsqueeze(-1) * outputs_g
    new_residual = (residual_g + residual_add).reshape_as(residual)
    return mass + mass_add, new_residual


def _prepare_projection_directions(
    *,
    projection_directions: torch.Tensor,
    num_heads: int,
    obs_count: int,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    directions = projection_directions.to(device=device, dtype=torch.float32)
    if directions.ndim == 3:
        directions = directions.unsqueeze(0)
    if directions.ndim != 4:
        raise ValueError(
            "projection_directions must be [M,H,R,D] or [H,R,D], "
            f"got {tuple(directions.shape)}"
        )
    expected = (int(num_heads), int(obs_count), int(head_dim))
    if tuple(directions.shape[1:]) != expected:
        raise ValueError(
            f"projection_directions must be [M,{expected[0]},{expected[1]},{expected[2]}], "
            f"got {tuple(directions.shape)}"
        )
    return torch.nan_to_num(directions, nan=0.0, posinf=0.0, neginf=0.0)


def _projected_loss_mean(
    *,
    projected_residual: torch.Tensor,
    denominator: torch.Tensor,
    query_weights: torch.Tensor | None,
) -> torch.Tensor:
    loss = (projected_residual / denominator.unsqueeze(0)).square()
    if query_weights is not None:
        loss = loss * query_weights.unsqueeze(0)
    return loss.sum() / float(projected_residual.numel())


def _score_projected_candidates_grouped(
    *,
    alpha: torch.Tensor,
    v_cache: torch.Tensor,
    outputs: torch.Tensor,
    projection_directions: torch.Tensor,
    candidate_idx: torch.Tensor,
    projected_residual: torch.Tensor,
    mass: torch.Tensor,
    current_loss: torch.Tensor,
    num_key_value_groups: int,
    query_weights: torch.Tensor | None,
    score_block_size: int,
    denom_eps: float,
    side: str,
) -> torch.Tensor:
    """Score one-token additions/deletions for projected JAOC."""

    if str(side) not in {"keep", "drop"}:
        raise ValueError("side must be keep or drop.")
    groups = max(int(num_key_value_groups), 1)
    scores = torch.empty((int(candidate_idx.numel()),), dtype=torch.float32, device=v_cache.device)
    block_size = max(int(score_block_size), 1)
    hq = int(alpha.shape[0])
    obs_count = int(alpha.shape[1])
    seq_len = int(alpha.shape[2])
    hkv = int(v_cache.shape[0])
    head_dim = int(v_cache.shape[-1])
    num_proj = int(projection_directions.shape[0])
    normalizer = float(num_proj * hq * obs_count)

    alpha_g = alpha.reshape(hkv, groups, obs_count, seq_len)
    outputs_g = outputs.reshape(hkv, groups, obs_count, head_dim)
    directions_g = projection_directions.reshape(num_proj, hkv, groups, obs_count, head_dim)
    residual_g = projected_residual.reshape(num_proj, hkv, groups, obs_count)
    mass_g = mass.reshape(hkv, groups, obs_count)
    query_weights_g = None
    if query_weights is not None:
        query_weights_g = query_weights.reshape(hkv, groups, obs_count)
    direction_output = (directions_g * outputs_g.unsqueeze(0)).sum(dim=-1)
    residual_sq = residual_g.square()
    if str(side) == "keep":
        den = mass_g.clamp_min(float(denom_eps))
    else:
        den = (1.0 - mass_g).clamp_min(float(denom_eps))

    for start_idx in range(0, int(candidate_idx.numel()), block_size):
        block = slice(start_idx, min(start_idx + block_size, int(candidate_idx.numel())))
        idx = candidate_idx[block]
        local_alpha = alpha_g.index_select(3, idx)
        local_values = v_cache.index_select(1, idx).float()
        direction_value = torch.einsum("mhgrd,hbd->mhgrb", directions_g, local_values)
        centered_projection = direction_value - direction_output[:, :, :, :, None]
        contribution = local_alpha.unsqueeze(0) * centered_projection
        if str(side) == "keep":
            den_new = den[:, :, :, None] + local_alpha
        else:
            den_new = den[:, :, :, None] - local_alpha
        numerator_new = (
            residual_sq[:, :, :, :, None]
            + 2.0 * residual_g[:, :, :, :, None] * contribution
            + contribution.square()
        )
        loss_terms = numerator_new / den_new.clamp_min(float(denom_eps)).square().unsqueeze(0)
        if query_weights_g is not None:
            loss_terms = loss_terms * query_weights_g[None, :, :, :, None]
        cand_loss = loss_terms.sum(dim=(0, 1, 2, 3)) / normalizer
        delta = cand_loss - current_loss
        if str(side) == "drop":
            invalid = (den_new <= float(denom_eps)).any(dim=(0, 1, 2))
            delta = delta.masked_fill(invalid, float("inf"))
        scores[block] = delta
    return scores


def _apply_projected_selected_grouped(
    *,
    alpha: torch.Tensor,
    v_cache: torch.Tensor,
    outputs: torch.Tensor,
    projection_directions: torch.Tensor,
    projected_residual: torch.Tensor,
    mass: torch.Tensor,
    selected: torch.Tensor,
    num_key_value_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = max(int(num_key_value_groups), 1)
    alpha_sel = alpha.index_select(2, selected)
    mass_add = alpha_sel.sum(dim=-1)
    hkv = int(v_cache.shape[0])
    obs_count = int(alpha.shape[1])
    head_dim = int(v_cache.shape[-1])
    num_proj = int(projection_directions.shape[0])
    alpha_sel_g = alpha_sel.reshape(hkv, groups, obs_count, int(selected.numel()))
    mass_add_g = mass_add.reshape(hkv, groups, obs_count)
    outputs_g = outputs.reshape(hkv, groups, obs_count, head_dim)
    directions_g = projection_directions.reshape(num_proj, hkv, groups, obs_count, head_dim)
    residual_g = projected_residual.reshape(num_proj, hkv, groups, obs_count)
    value_sel = v_cache.index_select(1, selected).float()
    direction_value = torch.einsum("mhgrd,hkd->mhgrk", directions_g, value_sel)
    direction_output = (directions_g * outputs_g.unsqueeze(0)).sum(dim=-1)
    centered_projection = direction_value - direction_output[:, :, :, :, None]
    residual_add = (alpha_sel_g.unsqueeze(0) * centered_projection).sum(dim=-1)
    new_residual = (residual_g + residual_add).reshape_as(projected_residual)
    return mass + mass_add, new_residual


def jaoc_select_layer_projected_fast(
    *,
    q_obs: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    projection_directions: torch.Tensor,
    budget: int,
    force_keep_mask: torch.Tensor,
    num_key_value_groups: int,
    obs_positions: torch.Tensor | None = None,
    obs_causal_mask: torch.Tensor | None = None,
    use_causal_obs_mask: bool = True,
    batch_drop: int = 256,
    score_block_size: int = 512,
    candidate_pool_factor: float = 0.0,
    candidate_pool_min: int = 0,
    denom_eps: float = 1e-4,
    return_loss_trace: bool = True,
    solver: str = "auto",
    query_weights: torch.Tensor | None = None,
) -> JAOCSelection:
    """Projected Fisher/LogitLens JAOC selector for one layer.

    projection_directions is [M,Hq,R,D] and defines the metric directions
    <g_m,h,r, delta_o_h,r>. This keeps the same joint set objective as JAOC
    but scores projected residuals instead of full value-space L2 residuals.
    """

    solver_mode = str(solver).lower()
    if solver_mode not in {"auto", "drop", "keep"}:
        raise ValueError("solver must be one of: auto, drop, keep.")
    hq, obs_count, seq_len, head_dim = _validate_inputs(
        q_obs=q_obs,
        k_cache=k_cache,
        v_cache=v_cache,
        force_keep_mask=force_keep_mask,
        num_key_value_groups=int(num_key_value_groups),
    )
    if bool(use_causal_obs_mask) and obs_causal_mask is None:
        if obs_positions is None:
            obs_positions = torch.arange(seq_len - int(q_obs.shape[1]), seq_len, device=q_obs.device, dtype=torch.long)
        obs_causal_mask = build_observation_causal_mask(obs_positions=obs_positions, seq_len=seq_len)

    force_keep_mask = force_keep_mask.to(device=k_cache.device, dtype=torch.bool).clone()
    if bool(use_causal_obs_mask):
        if obs_positions is None:
            raise ValueError("obs_positions are required when use_causal_obs_mask=True.")
        obs_idx = obs_positions.to(device=k_cache.device, dtype=torch.long).flatten()
        obs_idx = obs_idx[(obs_idx >= 0) & (obs_idx < seq_len)]
        if int(obs_idx.numel()) > 0:
            force_keep_mask.index_fill_(0, obs_idx, True)

    target_keep = max(int(budget), int(force_keep_mask.sum().item()), 1)
    target_keep = min(target_keep, seq_len)
    if target_keep >= seq_len:
        keep = torch.arange(seq_len, device=k_cache.device, dtype=torch.long)
        return JAOCSelection(
            keep_indices=keep,
            dropped_indices=keep[:0],
            final_loss=0.0,
            target_keep=int(target_keep),
            forced_count=int(force_keep_mask.sum().item()),
            rounds=0,
            loss_trace=[0.0],
            budget_trace=[int(seq_len)],
            min_denominator=1.0,
            batch_drop=int(batch_drop),
            score_block_size=int(score_block_size),
            solver="projected_full",
        )

    alpha = compute_observation_attention(
        q_obs=q_obs,
        k_cache=k_cache,
        num_key_value_groups=int(num_key_value_groups),
        obs_causal_mask=obs_causal_mask,
    )
    query_weights = _prepare_query_weights(
        query_weights=query_weights,
        num_heads=int(alpha.shape[0]),
        obs_count=int(alpha.shape[1]),
        device=alpha.device,
    )
    projection_directions = _prepare_projection_directions(
        projection_directions=projection_directions,
        num_heads=int(alpha.shape[0]),
        obs_count=int(alpha.shape[1]),
        head_dim=int(v_cache.shape[-1]),
        device=v_cache.device,
    )
    outputs = _full_outputs_grouped(
        alpha=alpha,
        v_cache=v_cache,
        num_key_value_groups=int(num_key_value_groups),
    )
    num_proj = int(projection_directions.shape[0])
    projected_residual = torch.zeros(
        (num_proj, int(alpha.shape[0]), int(alpha.shape[1])),
        dtype=torch.float32,
        device=v_cache.device,
    )
    mass = torch.zeros((int(alpha.shape[0]), int(alpha.shape[1])), dtype=torch.float32, device=v_cache.device)
    dropped = torch.zeros((seq_len,), dtype=torch.bool, device=v_cache.device)
    target_drop = min(seq_len - target_keep, int((~force_keep_mask).sum().item()))
    keep_count = seq_len
    rounds = 0
    loss_trace: list[float] = [0.0]
    budget_trace: list[int] = [int(seq_len)] if bool(return_loss_trace) else []
    proxy_alpha = alpha if query_weights is None else alpha * query_weights[:, :, None]
    attention_proxy = proxy_alpha.amax(dim=(0, 1))

    forced_count = int(force_keep_mask.sum().item())
    target_add = max(int(target_keep) - int(forced_count), 0)
    chosen_solver = solver_mode
    if chosen_solver == "auto":
        chosen_solver = "keep" if int(target_add) <= int(target_drop) else "drop"

    if chosen_solver == "keep":
        kept = force_keep_mask.clone()
        keep_count = int(forced_count)
        if int(forced_count) > 0:
            forced_idx = torch.nonzero(kept, as_tuple=False).flatten()
            mass, projected_residual = _apply_projected_selected_grouped(
                alpha=alpha,
                v_cache=v_cache,
                outputs=outputs,
                projection_directions=projection_directions,
                projected_residual=projected_residual,
                mass=mass,
                selected=forced_idx,
                num_key_value_groups=int(num_key_value_groups),
            )
        loss_trace = [
            float(
                _projected_loss_mean(
                    projected_residual=projected_residual,
                    denominator=mass.clamp_min(float(denom_eps)),
                    query_weights=query_weights,
                ).item()
            )
        ] if bool(return_loss_trace) else []
        while keep_count < target_keep:
            candidate_idx = torch.nonzero(~kept, as_tuple=False).flatten()
            if int(candidate_idx.numel()) == 0:
                break
            den = mass.clamp_min(float(denom_eps))
            current_loss = _projected_loss_mean(
                projected_residual=projected_residual,
                denominator=den,
                query_weights=query_weights,
            )
            remaining = int(target_keep) - int(keep_count)
            k = min(max(int(batch_drop), 1), int(remaining), int(candidate_idx.numel()))
            pool_size = _candidate_pool_size(
                candidate_count=int(candidate_idx.numel()),
                batch_count=int(k),
                candidate_pool_factor=float(candidate_pool_factor),
                candidate_pool_min=int(candidate_pool_min),
            )
            score_idx = _candidate_pool_by_attention_proxy(
                candidate_idx=candidate_idx,
                attention_proxy=attention_proxy,
                pool_size=int(pool_size),
                largest=True,
            )
            scores = _score_projected_candidates_grouped(
                alpha=alpha,
                v_cache=v_cache,
                outputs=outputs,
                projection_directions=projection_directions,
                candidate_idx=score_idx,
                projected_residual=projected_residual,
                mass=mass,
                current_loss=current_loss,
                num_key_value_groups=int(num_key_value_groups),
                query_weights=query_weights,
                score_block_size=int(score_block_size),
                denom_eps=float(denom_eps),
                side="keep",
            )
            selected = _select_lowest_score_batch(candidate_idx=score_idx, scores=scores, max_count=k)
            if int(selected.numel()) == 0:
                break
            kept.index_fill_(0, selected, True)
            mass, projected_residual = _apply_projected_selected_grouped(
                alpha=alpha,
                v_cache=v_cache,
                outputs=outputs,
                projection_directions=projection_directions,
                projected_residual=projected_residual,
                mass=mass,
                selected=selected,
                num_key_value_groups=int(num_key_value_groups),
            )
            keep_count += int(selected.numel())
            rounds += 1
            if bool(return_loss_trace):
                loss_trace.append(
                    float(
                        _projected_loss_mean(
                            projected_residual=projected_residual,
                            denominator=mass.clamp_min(float(denom_eps)),
                            query_weights=query_weights,
                        ).item()
                    )
                )

        keep_indices = torch.nonzero(kept, as_tuple=False).flatten().sort().values
        dropped_indices = torch.nonzero(~kept, as_tuple=False).flatten().sort().values
        final_den = mass.clamp_min(float(denom_eps))
        final_loss = _projected_loss_mean(
            projected_residual=projected_residual,
            denominator=final_den,
            query_weights=query_weights,
        )
        return JAOCSelection(
            keep_indices=keep_indices,
            dropped_indices=dropped_indices,
            final_loss=float(final_loss.item()),
            target_keep=int(target_keep),
            forced_count=int(forced_count),
            rounds=int(rounds),
            loss_trace=loss_trace,
            min_denominator=float(mass.min().item()),
            batch_drop=int(batch_drop),
            score_block_size=int(score_block_size),
            solver="projected_keep",
        )

    while int(dropped.sum().item()) < target_drop and keep_count > target_keep:
        candidate_idx = torch.nonzero((~dropped) & (~force_keep_mask), as_tuple=False).flatten()
        if int(candidate_idx.numel()) == 0:
            break
        den = (1.0 - mass).clamp_min(float(denom_eps))
        current_loss = _projected_loss_mean(
            projected_residual=projected_residual,
            denominator=den,
            query_weights=query_weights,
        )
        remaining = min(target_drop - int(dropped.sum().item()), keep_count - target_keep)
        k = min(max(int(batch_drop), 1), int(remaining), int(candidate_idx.numel()))
        pool_size = _candidate_pool_size(
            candidate_count=int(candidate_idx.numel()),
            batch_count=int(k),
            candidate_pool_factor=float(candidate_pool_factor),
            candidate_pool_min=int(candidate_pool_min),
        )
        while True:
            score_idx = _candidate_pool_by_attention_proxy(
                candidate_idx=candidate_idx,
                attention_proxy=attention_proxy,
                pool_size=int(pool_size),
                largest=False,
            )
            scores = _score_projected_candidates_grouped(
                alpha=alpha,
                v_cache=v_cache,
                outputs=outputs,
                projection_directions=projection_directions,
                candidate_idx=score_idx,
                projected_residual=projected_residual,
                mass=mass,
                current_loss=current_loss,
                num_key_value_groups=int(num_key_value_groups),
                query_weights=query_weights,
                score_block_size=int(score_block_size),
                denom_eps=float(denom_eps),
                side="drop",
            )
            selected = _select_safe_batch(
                candidate_idx=score_idx,
                scores=scores,
                alpha=alpha,
                mass=mass,
                max_count=k,
                denom_eps=float(denom_eps),
            )
            if int(selected.numel()) > 0 or int(score_idx.numel()) >= int(candidate_idx.numel()):
                break
            pool_size = min(int(candidate_idx.numel()), max(int(pool_size) * 2, int(pool_size) + 1))
        if int(selected.numel()) == 0:
            break
        dropped.index_fill_(0, selected, True)
        mass, projected_residual = _apply_projected_selected_grouped(
            alpha=alpha,
            v_cache=v_cache,
            outputs=outputs,
            projection_directions=projection_directions,
            projected_residual=projected_residual,
            mass=mass,
            selected=selected,
            num_key_value_groups=int(num_key_value_groups),
        )
        keep_count -= int(selected.numel())
        rounds += 1
        if bool(return_loss_trace):
            loss_trace.append(
                float(
                    _projected_loss_mean(
                        projected_residual=projected_residual,
                        denominator=(1.0 - mass).clamp_min(float(denom_eps)),
                        query_weights=query_weights,
                    ).item()
                )
            )

    keep_indices = torch.nonzero(~dropped, as_tuple=False).flatten().sort().values
    dropped_indices = torch.nonzero(dropped, as_tuple=False).flatten().sort().values
    final_den = (1.0 - mass).clamp_min(float(denom_eps))
    final_loss = _projected_loss_mean(
        projected_residual=projected_residual,
        denominator=final_den,
        query_weights=query_weights,
    )
    return JAOCSelection(
        keep_indices=keep_indices,
        dropped_indices=dropped_indices,
        final_loss=float(final_loss.item()),
        target_keep=int(target_keep),
        forced_count=int(force_keep_mask.sum().item()),
        rounds=int(rounds),
        loss_trace=loss_trace,
        min_denominator=float((1.0 - mass).min().item()),
        batch_drop=int(batch_drop),
        score_block_size=int(score_block_size),
        solver="projected_drop",
    )


def jaoc_select_layer_fast(
    *,
    q_obs: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    budget: int,
    force_keep_mask: torch.Tensor,
    num_key_value_groups: int,
    obs_positions: torch.Tensor | None = None,
    obs_causal_mask: torch.Tensor | None = None,
    use_causal_obs_mask: bool = True,
    batch_drop: int = 256,
    score_block_size: int = 512,
    candidate_pool_factor: float = 0.0,
    candidate_pool_min: int = 0,
    pair_pool_size: int = 0,
    precompute_x_sq: bool = True,
    precompute_x_sq_max_elements: int = 64_000_000,
    denom_eps: float = 1e-4,
    return_loss_trace: bool = True,
    solver: str = "auto",
    query_weights: torch.Tensor | None = None,
    metric_diag: torch.Tensor | None = None,
    risk_mode: str = "mean",
    cvar_beta: float = 0.9,
    logsumexp_tau: float = 10.0,
    alpha_override: torch.Tensor | None = None,
) -> JAOCSelection:
    """Joint attention-output token eviction for one layer.

    Inputs must be batch-free tensors from one layer:
    q_obs [Hq, R, D] and k_cache/v_cache [Hkv, T, D]. q/k must already have
    RoPE applied. The returned keep_indices are sorted token positions shared
    by all heads in this layer.
    """

    solver_mode = str(solver).lower()
    if solver_mode not in {"auto", "drop", "keep"}:
        raise ValueError("solver must be one of: auto, drop, keep.")

    _, _, seq_len, _ = _validate_inputs(
        q_obs=q_obs,
        k_cache=k_cache,
        v_cache=v_cache,
        force_keep_mask=force_keep_mask,
        num_key_value_groups=int(num_key_value_groups),
    )
    if bool(use_causal_obs_mask) and obs_causal_mask is None:
        if obs_positions is None:
            obs_positions = torch.arange(seq_len - int(q_obs.shape[1]), seq_len, device=q_obs.device, dtype=torch.long)
        obs_causal_mask = build_observation_causal_mask(obs_positions=obs_positions, seq_len=seq_len)

    force_keep_mask = force_keep_mask.to(device=k_cache.device, dtype=torch.bool).clone()
    if bool(use_causal_obs_mask):
        if obs_positions is None:
            raise ValueError("obs_positions are required when use_causal_obs_mask=True.")
        obs_idx = obs_positions.to(device=k_cache.device, dtype=torch.long).flatten()
        obs_idx = obs_idx[(obs_idx >= 0) & (obs_idx < seq_len)]
        if int(obs_idx.numel()) > 0:
            force_keep_mask.index_fill_(0, obs_idx, True)

    target_keep = max(int(budget), int(force_keep_mask.sum().item()), 1)
    target_keep = min(target_keep, seq_len)
    if target_keep >= seq_len:
        keep = torch.arange(seq_len, device=k_cache.device, dtype=torch.long)
        return JAOCSelection(
            keep_indices=keep,
            dropped_indices=keep[:0],
            final_loss=0.0,
            target_keep=int(target_keep),
            forced_count=int(force_keep_mask.sum().item()),
            rounds=0,
            loss_trace=[0.0],
            min_denominator=1.0,
            batch_drop=int(batch_drop),
            score_block_size=int(score_block_size),
            solver="full",
        )

    if alpha_override is None:
        alpha = compute_observation_attention(
            q_obs=q_obs,
            k_cache=k_cache,
            num_key_value_groups=int(num_key_value_groups),
            obs_causal_mask=obs_causal_mask,
        )
    else:
        alpha = alpha_override.to(device=k_cache.device, dtype=torch.float32).contiguous()
        expected_alpha_shape = (int(q_obs.shape[0]), int(q_obs.shape[1]), int(k_cache.shape[1]))
        if tuple(alpha.shape) != expected_alpha_shape:
            raise ValueError(f"alpha_override must be {expected_alpha_shape}, got {tuple(alpha.shape)}")
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    query_weights = _prepare_query_weights(
        query_weights=query_weights,
        num_heads=int(alpha.shape[0]),
        obs_count=int(alpha.shape[1]),
        device=alpha.device,
    )
    metric_diag = _prepare_metric_diag(
        metric_diag=metric_diag,
        num_heads=int(alpha.shape[0]),
        head_dim=int(v_cache.shape[-1]),
        device=v_cache.device,
    )
    outputs = _full_outputs_grouped(
        alpha=alpha,
        v_cache=v_cache,
        num_key_value_groups=int(num_key_value_groups),
    )
    residual = torch.zeros_like(outputs, dtype=torch.float32, device=v_cache.device)
    mass = torch.zeros((int(alpha.shape[0]), int(alpha.shape[1])), dtype=torch.float32, device=v_cache.device)
    dropped = torch.zeros((seq_len,), dtype=torch.bool, device=v_cache.device)
    target_drop = min(seq_len - target_keep, int((~force_keep_mask).sum().item()))
    loss_trace: list[float] = [0.0]
    budget_trace: list[int] = [int(seq_len)] if bool(return_loss_trace) else []
    keep_count = seq_len
    rounds = 0
    proxy_alpha = alpha if query_weights is None else alpha * query_weights[:, :, None]
    attention_proxy = proxy_alpha.amax(dim=(0, 1))
    x_sq_cache = None
    if metric_diag is None and bool(precompute_x_sq):
        cache_elements = int(alpha.shape[0]) * int(alpha.shape[1]) * int(alpha.shape[2])
        if int(precompute_x_sq_max_elements) == 0 or cache_elements <= int(precompute_x_sq_max_elements):
            x_sq_cache = _precompute_x_sq_grouped(
                outputs=outputs,
                v_cache=v_cache,
                num_key_value_groups=int(num_key_value_groups),
            )

    forced_count = int(force_keep_mask.sum().item())
    target_add = max(int(target_keep) - int(forced_count), 0)
    chosen_solver = solver_mode
    if chosen_solver == "auto":
        chosen_solver = "keep" if int(target_add) <= int(target_drop) else "drop"

    if chosen_solver == "keep":
        kept = force_keep_mask.clone()
        keep_count = int(forced_count)
        if int(forced_count) > 0:
            forced_idx = torch.nonzero(kept, as_tuple=False).flatten()
            mass, residual = _apply_selected_grouped(
                alpha=alpha,
                v_cache=v_cache,
                outputs=outputs,
                residual=residual,
                mass=mass,
                selected=forced_idx,
                num_key_value_groups=int(num_key_value_groups),
            )
        initial_den = mass.clamp_min(float(denom_eps))
        loss_trace = [
            float(
                _weighted_loss_mean(
                    residual=residual,
                    denominator=initial_den,
                    query_weights=query_weights,
                    metric_diag=metric_diag,
                    risk_mode=str(risk_mode),
                    cvar_beta=float(cvar_beta),
                    logsumexp_tau=float(logsumexp_tau),
                ).item()
            )
        ] if bool(return_loss_trace) else []
        budget_trace = [int(keep_count)] if bool(return_loss_trace) else []
        rounds = 0

        while keep_count < target_keep:
            candidate_idx = torch.nonzero(~kept, as_tuple=False).flatten()
            if int(candidate_idx.numel()) == 0:
                break
            den = mass.clamp_min(float(denom_eps))
            current_loss = _weighted_loss_mean(
                residual=residual,
                denominator=den,
                query_weights=query_weights,
                metric_diag=metric_diag,
                risk_mode=str(risk_mode),
                cvar_beta=float(cvar_beta),
                logsumexp_tau=float(logsumexp_tau),
            )
            remaining = int(target_keep) - int(keep_count)
            k = min(max(int(batch_drop), 1), int(remaining), int(candidate_idx.numel()))
            pool_size = _candidate_pool_size(
                candidate_count=int(candidate_idx.numel()),
                batch_count=int(k),
                candidate_pool_factor=float(candidate_pool_factor),
                candidate_pool_min=int(candidate_pool_min),
            )
            score_idx = _candidate_pool_by_attention_proxy(
                candidate_idx=candidate_idx,
                attention_proxy=attention_proxy,
                pool_size=int(pool_size),
                largest=True,
            )
            scores = _score_candidates_keep_grouped(
                alpha=alpha,
                v_cache=v_cache,
                outputs=outputs,
                candidate_idx=score_idx,
                residual=residual,
                mass=mass,
                current_loss=current_loss,
                num_key_value_groups=int(num_key_value_groups),
                x_sq_cache=x_sq_cache,
                query_weights=query_weights,
                metric_diag=metric_diag,
                risk_mode=str(risk_mode),
                cvar_beta=float(cvar_beta),
                logsumexp_tau=float(logsumexp_tau),
                score_block_size=int(score_block_size),
                denom_eps=float(denom_eps),
            )
            selected = _select_lowest_score_batch(
                candidate_idx=score_idx,
                scores=scores,
                max_count=k,
            )
            if int(selected.numel()) == 0:
                break
            kept.index_fill_(0, selected, True)
            mass, residual = _apply_selected_grouped(
                alpha=alpha,
                v_cache=v_cache,
                outputs=outputs,
                residual=residual,
                mass=mass,
                selected=selected,
                num_key_value_groups=int(num_key_value_groups),
            )
            keep_count += int(selected.numel())
            rounds += 1
            if bool(return_loss_trace):
                den_after = mass.clamp_min(float(denom_eps))
                loss_trace.append(
                    float(
                        _weighted_loss_mean(
                            residual=residual,
                            denominator=den_after,
                            query_weights=query_weights,
                            metric_diag=metric_diag,
                            risk_mode=str(risk_mode),
                            cvar_beta=float(cvar_beta),
                            logsumexp_tau=float(logsumexp_tau),
                        ).item()
                    )
                )
                budget_trace.append(int(keep_count))

        keep_indices = torch.nonzero(kept, as_tuple=False).flatten().sort().values
        dropped_indices = torch.nonzero(~kept, as_tuple=False).flatten().sort().values
        final_den = mass.clamp_min(float(denom_eps))
        final_loss = _weighted_loss_mean(
            residual=residual,
            denominator=final_den,
            query_weights=query_weights,
            metric_diag=metric_diag,
            risk_mode=str(risk_mode),
            cvar_beta=float(cvar_beta),
            logsumexp_tau=float(logsumexp_tau),
        )
        return JAOCSelection(
            keep_indices=keep_indices,
            dropped_indices=dropped_indices,
            final_loss=float(final_loss.item()),
            target_keep=int(target_keep),
            forced_count=int(forced_count),
            rounds=int(rounds),
            loss_trace=loss_trace,
            budget_trace=budget_trace,
            min_denominator=float(mass.min().item()),
            batch_drop=int(batch_drop),
            score_block_size=int(score_block_size),
            solver="keep",
        )

    while int(dropped.sum().item()) < target_drop and keep_count > target_keep:
        candidate_idx = torch.nonzero((~dropped) & (~force_keep_mask), as_tuple=False).flatten()
        if int(candidate_idx.numel()) == 0:
            break
        den = (1.0 - mass).clamp_min(float(denom_eps))
        current_loss = _weighted_loss_mean(
            residual=residual,
            denominator=den,
            query_weights=query_weights,
            metric_diag=metric_diag,
            risk_mode=str(risk_mode),
            cvar_beta=float(cvar_beta),
            logsumexp_tau=float(logsumexp_tau),
        )
        remaining = min(target_drop - int(dropped.sum().item()), keep_count - target_keep)
        k = min(max(int(batch_drop), 1), int(remaining), int(candidate_idx.numel()))
        score_idx = candidate_idx
        selected = candidate_idx[:0]
        pool_size = _candidate_pool_size(
            candidate_count=int(candidate_idx.numel()),
            batch_count=int(k),
            candidate_pool_factor=float(candidate_pool_factor),
            candidate_pool_min=int(candidate_pool_min),
        )
        while True:
            score_idx = _candidate_pool_by_attention_proxy(
                candidate_idx=candidate_idx,
                attention_proxy=attention_proxy,
                pool_size=int(pool_size),
                largest=False,
            )
            scores = _score_candidates_grouped(
                alpha=alpha,
                v_cache=v_cache,
                outputs=outputs,
                candidate_idx=score_idx,
                residual=residual,
                mass=mass,
                current_loss=current_loss,
                num_key_value_groups=int(num_key_value_groups),
                x_sq_cache=x_sq_cache,
                query_weights=query_weights,
                metric_diag=metric_diag,
                risk_mode=str(risk_mode),
                cvar_beta=float(cvar_beta),
                logsumexp_tau=float(logsumexp_tau),
                score_block_size=int(score_block_size),
                denom_eps=float(denom_eps),
            )
            selected = _select_safe_batch(
                candidate_idx=score_idx,
                scores=scores,
                alpha=alpha,
                mass=mass,
                max_count=k,
                denom_eps=float(denom_eps),
            )
            if int(selected.numel()) > 0 or int(score_idx.numel()) >= int(candidate_idx.numel()):
                break
            pool_size = min(int(candidate_idx.numel()), max(int(pool_size) * 2, int(pool_size) + 1))
        pair_pool_allowed = (
            metric_diag is None
            and str(risk_mode) == "mean"
            and query_weights is None
        )
        if bool(pair_pool_allowed) and int(pair_pool_size) >= 2 and int(k) >= 2:
            pair_idx = _pair_pool_from_single_scores(
                candidate_idx=score_idx,
                single_scores=scores,
                pair_pool_size=int(pair_pool_size),
            )
            if int(pair_idx.numel()) >= 2:
                pair_scores = _score_pairs_materialized(
                    alpha=alpha,
                    v_cache=v_cache,
                    outputs=outputs,
                    residual=residual,
                    mass=mass,
                    pair_idx=pair_idx,
                    current_loss=current_loss,
                    num_key_value_groups=int(num_key_value_groups),
                    x_sq_cache=x_sq_cache,
                    denom_eps=float(denom_eps),
                )
                pair_selected = _select_safe_pair_batch(
                    pair_idx=pair_idx,
                    pair_scores=pair_scores,
                    alpha=alpha,
                    mass=mass,
                    max_tokens=k,
                    denom_eps=float(denom_eps),
                )
                if int(pair_selected.numel()) > 0:
                    selected = pair_selected
        if int(selected.numel()) == 0:
            break
        dropped.index_fill_(0, selected, True)
        mass, residual = _apply_selected_grouped(
            alpha=alpha,
            v_cache=v_cache,
            outputs=outputs,
            residual=residual,
            mass=mass,
            selected=selected,
            num_key_value_groups=int(num_key_value_groups),
        )
        keep_count -= int(selected.numel())
        rounds += 1
        if bool(return_loss_trace):
            den_after = (1.0 - mass).clamp_min(float(denom_eps))
            loss_trace.append(
                float(
                    _weighted_loss_mean(
                        residual=residual,
                        denominator=den_after,
                        query_weights=query_weights,
                        metric_diag=metric_diag,
                        risk_mode=str(risk_mode),
                        cvar_beta=float(cvar_beta),
                        logsumexp_tau=float(logsumexp_tau),
                    ).item()
                )
            )
            budget_trace.append(int(keep_count))

    keep_indices = torch.nonzero(~dropped, as_tuple=False).flatten().sort().values
    dropped_indices = torch.nonzero(dropped, as_tuple=False).flatten().sort().values
    final_den = (1.0 - mass).clamp_min(float(denom_eps))
    final_loss = _weighted_loss_mean(
        residual=residual,
        denominator=final_den,
        query_weights=query_weights,
        metric_diag=metric_diag,
        risk_mode=str(risk_mode),
        cvar_beta=float(cvar_beta),
        logsumexp_tau=float(logsumexp_tau),
    )
    return JAOCSelection(
        keep_indices=keep_indices,
        dropped_indices=dropped_indices,
        final_loss=float(final_loss.item()),
        target_keep=int(target_keep),
        forced_count=int(force_keep_mask.sum().item()),
        rounds=int(rounds),
        loss_trace=loss_trace,
        budget_trace=budget_trace,
        min_denominator=float((1.0 - mass).min().item()),
        batch_drop=int(batch_drop),
        score_block_size=int(score_block_size),
        solver="drop",
    )


def jaoc_select_layer_oneshot(
    *,
    q_obs: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    budget: int,
    force_keep_mask: torch.Tensor,
    num_key_value_groups: int,
    obs_positions: torch.Tensor | None = None,
    obs_causal_mask: torch.Tensor | None = None,
    use_causal_obs_mask: bool = True,
    score_block_size: int = 512,
    denom_eps: float = 1e-4,
) -> JAOCSelection:
    """Debug selector: score L({t}) once and drop the lowest-score tokens."""

    return jaoc_select_layer_fast(
        q_obs=q_obs,
        k_cache=k_cache,
        v_cache=v_cache,
        budget=budget,
        force_keep_mask=force_keep_mask,
        num_key_value_groups=num_key_value_groups,
        obs_positions=obs_positions,
        obs_causal_mask=obs_causal_mask,
        use_causal_obs_mask=use_causal_obs_mask,
        batch_drop=max(int(k_cache.shape[1]) - int(budget), 1),
        score_block_size=score_block_size,
        pair_pool_size=0,
        precompute_x_sq=True,
        denom_eps=denom_eps,
        return_loss_trace=True,
        solver="drop",
    )
