from __future__ import annotations

from dataclasses import replace
from typing import Any
import math
import os
import time

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from .attention_patch import StrictQueryContentState, apply_strict_attention_patch
from .budget import uniform_layer_budget
from .cache_utils import (
    build_dynamic_cache,
    build_dynamic_cache_no_copy,
    extract_cache_layers,
    get_layer_cache,
    num_cache_layers,
    set_layer_cache,
    set_layer_log_bias,
    shallow_dynamic_cache,
)
from .config import StrictMergeConfig
from .masks import build_force_keep_mask
from .selector import jaoc_select_layer_fast, jaoc_select_layer_span


def _profile_enabled() -> bool:
    value = os.environ.get("BEST_V1_PROFILE", "")
    return str(value).lower() not in {"", "0", "false", "no", "off"}


def _profile_start(device: torch.device | None, profile: dict[str, float] | None) -> float:
    if profile is None:
        return 0.0
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _profile_add(
    profile: dict[str, float] | None,
    key: str,
    start: float,
    device: torch.device | None,
) -> None:
    if profile is None:
        return
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    profile[key] = float(profile.get(key, 0.0)) + float(time.perf_counter() - start)


def _active_local_columns(mask: torch.Tensor) -> list[int]:
    if int(mask.numel()) == 0:
        return []
    return [int(item) for item in torch.nonzero(mask.any(dim=0), as_tuple=False).flatten().tolist()]


def _project_selected_chunk_columns(
    residual: torch.Tensor,
    local_selected: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Project chunk residuals against already-selected local columns.

    ``residual`` has shape ``[..., chunks, width, features]`` and
    ``local_selected`` has shape ``[..., chunks, width]``.  The previous
    implementation scanned every local position and updated every chunk even
    when forced tokens only occupied the first few rows.  This keeps the same
    Gram-Schmidt order, but only touches rows that actually contain a forced
    token at that local position.
    """

    if int(residual.numel()) == 0 or int(local_selected.numel()) == 0:
        return residual
    active_mask = local_selected.to(dtype=torch.bool) & valid_mask.to(
        device=local_selected.device,
        dtype=torch.bool,
    ).expand_as(local_selected)
    if int(active_mask.numel()) == 0:
        return residual
    reduce_dims = tuple(range(int(active_mask.dim()) - 1))
    forced_cols_t = torch.nonzero(active_mask.any(dim=reduce_dims), as_tuple=False).flatten()
    if int(forced_cols_t.numel()) == 0:
        return residual

    width = int(residual.shape[-2])
    feature_dim = int(residual.shape[-1])
    flat_residual = residual.reshape(-1, width, feature_dim)
    flat_active = active_mask.reshape(-1, width)
    for pos in [int(item) for item in forced_cols_t.detach().cpu().tolist()]:
        active_rows = torch.nonzero(flat_active[:, int(pos)], as_tuple=False).flatten()
        if int(active_rows.numel()) == 0:
            continue
        rows = flat_residual.index_select(0, active_rows)
        vec = rows[:, int(pos), :]
        basis = vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        coeff = torch.bmm(rows, basis.unsqueeze(-1)).squeeze(-1)
        flat_residual[active_rows] = rows - coeff.unsqueeze(-1) * basis.unsqueeze(1)
    return flat_residual.reshape_as(residual)


def _project_selected_chunk_gram(
    gram: torch.Tensor,
    local_selected: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Deflate selected local columns directly in chunk Gram space."""

    if int(gram.numel()) == 0 or int(local_selected.numel()) == 0:
        return gram
    active_mask = local_selected.to(dtype=torch.bool) & valid_mask.to(
        device=local_selected.device,
        dtype=torch.bool,
    ).expand_as(local_selected)
    if int(active_mask.numel()) == 0:
        return gram
    reduce_dims = tuple(range(int(active_mask.dim()) - 1))
    forced_cols_t = torch.nonzero(active_mask.any(dim=reduce_dims), as_tuple=False).flatten()
    if int(forced_cols_t.numel()) == 0:
        return gram

    width = int(gram.shape[-1])
    flat_gram = gram.reshape(-1, width, width)
    flat_active = active_mask.reshape(-1, width)
    for pos in [int(item) for item in forced_cols_t.detach().cpu().tolist()]:
        active_rows = torch.nonzero(flat_active[:, int(pos)], as_tuple=False).flatten()
        if int(active_rows.numel()) == 0:
            continue
        rows = flat_gram.index_select(0, active_rows)
        pivot_col = rows[:, :, int(pos)]
        denom = pivot_col[:, int(pos)].clamp_min(1e-12)
        rows = rows - pivot_col.unsqueeze(2) * pivot_col.unsqueeze(1) / denom.view(-1, 1, 1)
        rows = 0.5 * (rows + rows.transpose(1, 2))
        flat_gram.index_copy_(0, active_rows, rows)
    return flat_gram.reshape_as(gram)


def _project_pivot_chunk_rows(
    residual: torch.Tensor,
    local_idx: torch.Tensor,
    chosen: torch.Tensor,
) -> torch.Tensor:
    """Project only rows that selected a new CPQR pivot in this round."""

    if int(residual.numel()) == 0 or int(chosen.numel()) == 0:
        return residual
    width = int(residual.shape[-2])
    feature_dim = int(residual.shape[-1])
    flat_residual = residual.reshape(-1, width, feature_dim)
    flat_idx = local_idx.reshape(-1).to(device=residual.device, dtype=torch.long)
    flat_chosen = chosen.reshape(-1).to(device=residual.device, dtype=torch.bool)
    active_rows = torch.nonzero(flat_chosen, as_tuple=False).flatten()
    if int(active_rows.numel()) == 0:
        return residual

    # Dense projection is faster when nearly every row is active; sparse wins
    # once quotas thin out after the first pivot.
    if int(active_rows.numel()) * 4 >= int(flat_residual.shape[0]) * 3:
        vec = flat_residual.gather(
            1,
            flat_idx.clamp(0, max(width - 1, 0)).view(-1, 1, 1).expand(-1, 1, feature_dim),
        ).squeeze(1)
        basis = vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        coeff = torch.bmm(flat_residual, basis.unsqueeze(-1)).squeeze(-1)
        updated = flat_residual - coeff.unsqueeze(-1) * basis.unsqueeze(1)
        flat_residual = torch.where(flat_chosen.view(-1, 1, 1), updated, flat_residual)
        return flat_residual.reshape_as(residual)

    rows = flat_residual.index_select(0, active_rows)
    pivots = flat_idx.index_select(0, active_rows).clamp(0, max(width - 1, 0))
    vec = rows.gather(1, pivots.view(-1, 1, 1).expand(-1, 1, feature_dim)).squeeze(1)
    basis = vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    coeff = torch.bmm(rows, basis.unsqueeze(-1)).squeeze(-1)
    flat_residual[active_rows] = rows - coeff.unsqueeze(-1) * basis.unsqueeze(1)
    return flat_residual.reshape_as(residual)


def _project_pivot_flat_chunk_rows(
    residual: torch.Tensor,
    flat_rows: torch.Tensor,
    local_idx: torch.Tensor,
) -> torch.Tensor:
    """Project selected flattened chunk rows without building dense masks."""

    if int(residual.numel()) == 0 or int(flat_rows.numel()) == 0:
        return residual
    width = int(residual.shape[-2])
    feature_dim = int(residual.shape[-1])
    flat_residual = residual.reshape(-1, width, feature_dim)
    flat_rows = flat_rows.to(device=residual.device, dtype=torch.long).flatten()
    local_idx = local_idx.to(device=residual.device, dtype=torch.long).flatten()
    if int(flat_rows.numel()) != int(local_idx.numel()):
        raise ValueError("flat_rows and local_idx must have the same length.")
    rows = flat_residual.index_select(0, flat_rows)
    pivots = local_idx.clamp(0, max(width - 1, 0))
    vec = rows.gather(1, pivots.view(-1, 1, 1).expand(-1, 1, feature_dim)).squeeze(1)
    basis = vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    coeff = torch.bmm(rows, basis.unsqueeze(-1)).squeeze(-1)
    flat_residual[flat_rows] = rows - coeff.unsqueeze(-1) * basis.unsqueeze(1)
    return flat_residual.reshape_as(residual)


def _repeat_kv_heads_for_storage(states: torch.Tensor, groups: int) -> torch.Tensor:
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


def _stats(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    vals = torch.tensor(values, dtype=torch.float64)
    return (
        float(vals.mean().item()),
        float(vals.min().item()),
        float(vals.max().item()),
    )


def _oproj_metric_diag_for_layer(
    *,
    model,
    layer_idx: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Diagonal of (W_O^h)^T W_O^h for each query head.

    This approximates residual-stream perturbation ||W_O^h e_h||^2 while
    keeping the selector's per-head/group implementation unchanged.
    """

    attn = model.model.layers[int(layer_idx)].self_attn
    weight = attn.o_proj.weight.detach().to(device=device, dtype=torch.float32)
    metric_rows: list[torch.Tensor] = []
    for head_idx in range(int(num_heads)):
        start = int(head_idx) * int(head_dim)
        end = start + int(head_dim)
        block = weight[:, start:end]
        metric_rows.append(block.square().sum(dim=0))
    metric = torch.stack(metric_rows, dim=0)
    return metric / metric.mean().clamp_min(1e-8)


def _get_oproj_metric_diag_cached(
    *,
    model,
    layer_idx: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    cache = getattr(model, "_best_v1_oproj_diag_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(model, "_best_v1_oproj_diag_cache", cache)
    key = (int(layer_idx), str(device), int(num_heads), int(head_dim))
    metric = cache.get(key)
    if metric is None:
        metric = _oproj_metric_diag_for_layer(
            model=model,
            layer_idx=int(layer_idx),
            num_heads=int(num_heads),
            head_dim=int(head_dim),
            device=device,
        )
        cache[key] = metric
    return metric


@torch.no_grad()
def _attention_rank_scores(
    *,
    q_obs: torch.Tensor,
    keys: torch.Tensor,
    force_recent: int,
    num_key_value_groups: int,
    config: StrictMergeConfig,
) -> tuple[torch.Tensor, int, int, str]:
    """Return Snap/CAKE-style old-token scores shaped [Hkv, old_len]."""

    if int(keys.shape[0]) != 1:
        raise NotImplementedError("StrictMerge supports batch_size=1.")
    hkv = int(keys.shape[1])
    seq_len = int(keys.shape[2])
    head_dim = int(keys.shape[3])
    recent = min(max(int(force_recent), 0), seq_len)
    old_len = seq_len - recent
    if old_len <= 0:
        empty = torch.empty((hkv, 0), dtype=torch.float32, device=keys.device)
        return empty, int(old_len), int(recent), "rank_empty"

    groups = max(int(num_key_value_groups), 1)
    scale = 1.0 / (float(head_dim) ** 0.5)
    logits = torch.einsum(
        "hgrd,htd->hgrt",
        q_obs.float().reshape(hkv, groups, int(q_obs.shape[1]), head_dim),
        keys[0].float(),
    ) * scale
    # Match Snap/Cake prompt-tail scoring: observation queries at original tail
    # positions cannot attend to later tokens within the recent window.
    obs_count = int(q_obs.shape[1])
    if recent > 0 and obs_count > 0:
        tail_mask = torch.full((obs_count, recent), torch.finfo(logits.dtype).min, device=logits.device)
        row = torch.arange(obs_count, device=logits.device).view(obs_count, 1)
        col = torch.arange(recent, device=logits.device).view(1, recent)
        tail_mask = tail_mask.masked_fill(col <= row + (recent - obs_count), 0)
        logits[..., -obs_count:, -recent:] += tail_mask.view(1, 1, obs_count, recent)
    attn = torch.softmax(logits, dim=-1, dtype=torch.float32)

    mode = str(config.selector_mode)
    if mode == "jaoc_anchor":
        mode = str(config.anchor_score_mode)
    if mode == "cake":
        old_attn = attn[..., :old_len]
        attn_mean = old_attn.mean(dim=2)
        attn_var = old_attn.var(dim=2, unbiased=bool(int(q_obs.shape[1]) > 1))
        scores = (attn_mean + float(config.cake_gamma) * attn_var).mean(dim=1)
        kernel = int(config.cake_kernel_size)
        if kernel > 1:
            scores = F.avg_pool1d(scores.unsqueeze(0), kernel_size=kernel, padding=kernel // 2, stride=1).squeeze(0)
            scores = scores[..., :old_len]
        return scores, int(old_len), int(recent), f"cake_gamma{float(config.cake_gamma):g}_avgpool_k{kernel}"

    scores = attn[..., :old_len].mean(dim=2).mean(dim=1)  # [Hkv, old_len]
    kernel = int(config.snap_kernel_size)
    if kernel > 1:
        if str(config.snap_pooling) == "maxpool":
            scores = F.max_pool1d(scores.unsqueeze(0), kernel_size=kernel, padding=kernel // 2, stride=1).squeeze(0)
        else:
            scores = F.avg_pool1d(scores.unsqueeze(0), kernel_size=kernel, padding=kernel // 2, stride=1).squeeze(0)
        scores = scores[..., :old_len]
    return scores, int(old_len), int(recent), f"snap_{config.snap_pooling}_k{kernel}"


@torch.no_grad()
def _local_chunk_jaoc_scores(
    *,
    q_obs: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    force_recent: int,
    num_key_value_groups: int,
    config: StrictMergeConfig,
) -> tuple[torch.Tensor, int, int, str]:
    """Local single-token counterfactual scores over old tokens.

    For each KV head and old-token chunk, compute local attention beta over the
    chunk, local output O_chunk=sum beta_i V_i, and score each token by
    beta_i/(1-beta_i) * ||V_i - O_chunk||. Scores are aggregated over the GQA
    query-head group and observation queries.
    """

    if int(keys.shape[0]) != 1 or int(values.shape[0]) != 1:
        raise NotImplementedError("StrictMerge supports batch_size=1.")
    hkv = int(keys.shape[1])
    seq_len = int(keys.shape[2])
    head_dim = int(keys.shape[3])
    recent = min(max(int(force_recent), 0), seq_len)
    old_len = seq_len - recent
    if old_len <= 0:
        empty = torch.empty((hkv, 0), dtype=torch.float32, device=keys.device)
        return empty, int(old_len), int(recent), "local_jaoc_empty"

    groups = max(int(num_key_value_groups), 1)
    obs_count = int(q_obs.shape[1])
    q_grouped = q_obs.float().reshape(hkv, groups, obs_count, head_dim)
    chunk_size = max(int(config.local_chunk_size), 2)
    scale = 1.0 / (float(head_dim) ** 0.5)
    scores = torch.empty((hkv, old_len), dtype=torch.float32, device=keys.device)

    for start in range(0, old_len, chunk_size):
        end = min(start + chunk_size, old_len)
        k_chunk = keys[0, :, start:end, :].float()
        v_chunk = values[0, :, start:end, :].float()
        logits = torch.einsum("hgrd,hld->hgrl", q_grouped, k_chunk) * scale
        beta = torch.softmax(logits, dim=-1, dtype=torch.float32)
        local_output = torch.einsum("hgrl,hld->hgrd", beta, v_chunk)
        diff = v_chunk[:, None, None, :, :] - local_output[:, :, :, None, :]
        diff_norm = diff.square().sum(dim=-1).clamp_min(0.0).sqrt()
        if str(config.local_score_mode) in {"attn_cos", "anchor_then_diverse"}:
            # Prototype-anchor variant: keep high-attention tokens whose value
            # direction aligns with the chunk-local attention output.
            dot = torch.einsum("hld,hgrd->hgrl", v_chunk, local_output)
            v_norm = v_chunk.square().sum(dim=-1).sqrt()
            output_norm = local_output.square().sum(dim=-1).sqrt()
            cos = dot / (v_norm[:, None, None, :] * output_norm[:, :, :, None]).clamp_min(1e-6)
            prototype_terms = beta * cos.clamp_min(0.0)
            if str(config.local_score_mode) == "attn_cos":
                token_terms = prototype_terms
            else:
                token_terms = beta / (1.0 - beta).clamp_min(float(config.denom_eps)) * diff_norm
        elif str(config.local_score_mode) == "attn_close":
            # Representative-anchor variant: prefer high-attention tokens whose
            # value is close to the chunk-local attention output. Normalize the
            # distance inside each chunk/query/head to avoid layer scale effects.
            norm = diff_norm.mean(dim=-1, keepdim=True).clamp_min(1e-6)
            closeness = 1.0 / (1.0 + diff_norm / norm)
            token_terms = beta * closeness
        else:
            token_terms = beta / (1.0 - beta).clamp_min(float(config.denom_eps)) * diff_norm
        if str(config.risk_mode) == "max":
            chunk_scores = token_terms.amax(dim=(1, 2))
        elif str(config.risk_mode) == "cvar":
            flat = token_terms.reshape(hkv, groups * obs_count, end - start)
            tail_count = max(1, int((1.0 - float(config.cvar_beta)) * float(groups * obs_count) + 0.999999))
            chunk_scores = torch.topk(flat, k=tail_count, dim=1, largest=True).values.mean(dim=1)
        elif str(config.risk_mode) == "logsumexp":
            flat = token_terms.reshape(hkv, groups * obs_count, end - start)
            tau = float(config.logsumexp_tau)
            chunk_scores = torch.logsumexp(tau * flat, dim=1) / tau
        else:
            chunk_scores = token_terms.mean(dim=(1, 2))
        if str(config.local_score_mode) == "anchor_then_diverse":
            # Ensure every chunk that receives any quota first contains one
            # high-attention prototype close to the local anchor. The remaining
            # quota is then filled by the deletion-loss/diversity score above.
            if str(config.risk_mode) == "max":
                proto_scores = prototype_terms.amax(dim=(1, 2))
            elif str(config.risk_mode) == "cvar":
                flat = prototype_terms.reshape(hkv, groups * obs_count, end - start)
                tail_count = max(1, int((1.0 - float(config.cvar_beta)) * float(groups * obs_count) + 0.999999))
                proto_scores = torch.topk(flat, k=tail_count, dim=1, largest=True).values.mean(dim=1)
            elif str(config.risk_mode) == "logsumexp":
                flat = prototype_terms.reshape(hkv, groups * obs_count, end - start)
                tau = float(config.logsumexp_tau)
                proto_scores = torch.logsumexp(tau * flat, dim=1) / tau
            else:
                proto_scores = prototype_terms.mean(dim=(1, 2))
            proto_idx = torch.argmax(proto_scores, dim=-1)
            for head in range(hkv):
                local_max = chunk_scores[head].amax()
                chunk_scores[head, int(proto_idx[head].item())] = local_max + local_max.abs() + 1.0
        scores[:, start:end] = chunk_scores

    return scores, int(old_len), int(recent), f"local_jaoc_{config.local_score_mode}_chunk{chunk_size}_{config.risk_mode}"


def _chunkwise_quota_plan(
    *,
    selected: torch.Tensor,
    old_len: int,
    target: int,
    chunk_size: int,
) -> tuple[list[int], list[int], list[int]]:
    seq_len = int(selected.numel())
    old_len = max(min(int(old_len), seq_len), 0)
    target = min(max(int(target), int(selected.sum().item())), seq_len)
    recent_forced = int(selected[old_len:].sum().item())
    target_old = min(max(target - recent_forced, int(selected[:old_len].sum().item())), old_len)
    chunk_size = max(int(chunk_size), 2)
    starts = list(range(0, old_len, chunk_size))
    lengths = [min(chunk_size, old_len - s) for s in starts]
    if not starts:
        return [], [], []

    ideal = [float(length) * float(target_old) / float(max(old_len, 1)) for length in lengths]
    quotas = [int(x) for x in ideal]
    remainder = int(target_old) - int(sum(quotas))
    order = sorted(range(len(starts)), key=lambda i: ideal[i] - quotas[i], reverse=True)
    for i in order[: max(remainder, 0)]:
        quotas[i] += 1

    forced_counts = [int(selected[s : s + l].sum().item()) for s, l in zip(starts, lengths)]
    quotas = [min(max(q, f), l) for q, f, l in zip(quotas, forced_counts, lengths)]
    while sum(quotas) > int(target_old):
        candidates = [i for i, q in enumerate(quotas) if q > forced_counts[i]]
        if not candidates:
            break
        i = min(candidates, key=lambda j: ideal[j] - quotas[j])
        quotas[i] -= 1
    while sum(quotas) < int(target_old):
        candidates = [i for i, q in enumerate(quotas) if q < lengths[i]]
        if not candidates:
            break
        i = max(candidates, key=lambda j: ideal[j] - quotas[j])
        quotas[i] += 1
    return starts, lengths, quotas


def _coreset_features(
    *,
    k_chunk: torch.Tensor,
    v_chunk: torch.Tensor,
    config: StrictMergeConfig,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    key_weight = float(config.coreset_key_weight)
    value_weight = float(config.coreset_value_weight)
    if key_weight > 0.0:
        parts.append(F.normalize(k_chunk.float(), p=2, dim=-1, eps=1e-6) * key_weight)
    if value_weight > 0.0:
        parts.append(F.normalize(v_chunk.float(), p=2, dim=-1, eps=1e-6) * value_weight)
    if not parts:
        raise ValueError("At least one coreset feature weight must be positive.")
    return torch.cat(parts, dim=-1)


def _coreset_similarity(
    *,
    features: torch.Tensor,
    config: StrictMergeConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = int(features.shape[0])
    if length <= 0:
        raise ValueError("features must be non-empty.")
    if length == 1:
        one = torch.ones((1,), dtype=torch.float32, device=features.device)
        return one.view(1, 1), one, one

    x = features.float()
    x_sq = x.square().sum(dim=-1)
    dist_sq = (x_sq[:, None] + x_sq[None, :] - 2.0 * (x @ x.T)).clamp_min(0.0)
    eye = torch.eye(length, dtype=torch.bool, device=features.device)
    off_diag = dist_sq.masked_select(~eye)
    positive = off_diag.masked_select(off_diag > 1e-12)
    if int(positive.numel()) > 0:
        base_scale_sq = positive.median()
    elif int(off_diag.numel()) > 0:
        base_scale_sq = off_diag.mean().clamp_min(1e-6)
    else:
        base_scale_sq = torch.ones((), dtype=torch.float32, device=features.device)
    if not bool(torch.isfinite(base_scale_sq).item()) or float(base_scale_sq.item()) <= 1e-12:
        base_scale_sq = torch.ones((), dtype=torch.float32, device=features.device)
    scale_sq = base_scale_sq * float(config.coreset_tau) * float(config.coreset_tau)
    scale_sq = scale_sq.clamp_min(1e-6)

    sim = torch.exp(-dist_sq / scale_sq)
    knn = int(config.coreset_knn)
    if 0 < knn < length:
        top_k = min(max(int(knn), 1), length)
        keep = torch.zeros_like(sim, dtype=torch.bool)
        keep.scatter_(1, torch.topk(sim, k=top_k, dim=1, largest=True).indices, True)
        sim = sim.masked_fill(~keep, 0.0)

    center = x.mean(dim=0)
    center_dist_sq = (x - center).square().sum(dim=-1).clamp_min(0.0)
    center_score = torch.exp(-center_dist_sq / scale_sq)
    density = sim.sum(dim=0)
    return sim, density, center_score


def _is_anchor_facility_mode(config: StrictMergeConfig) -> bool:
    return str(config.local_score_mode) in {"anchor_then_cover", "anchor_then_facility", "anchor_then_weighted_facility"}


def _is_weighted_anchor_facility_mode(config: StrictMergeConfig) -> bool:
    return str(config.local_score_mode) == "anchor_then_weighted_facility"


def _is_attention_operator_coreset_mode(config: StrictMergeConfig) -> bool:
    return str(config.local_score_mode) in {"attention_operator_coreset", "mass_weighted_attention_operator_coreset"}


def _is_mass_weighted_attention_operator_mode(config: StrictMergeConfig) -> bool:
    return str(config.local_score_mode) == "mass_weighted_attention_operator_coreset"


def _is_grop_kv_mode(config: StrictMergeConfig) -> bool:
    return str(config.local_score_mode) == "grop_kv"


def _is_csd_kv_mode(config: StrictMergeConfig) -> bool:
    return str(config.local_score_mode) == "csd_kv"


def _is_spectral_csd_kv_mode(config: StrictMergeConfig) -> bool:
    return str(config.local_score_mode) == "spectral_csd_kv"


def _is_dynamic_csd_kv_mode(config: StrictMergeConfig) -> bool:
    return str(config.local_score_mode) == "dynamic_csd_kv"


def _is_spectral_csd_kv_bias_mode(config: StrictMergeConfig) -> bool:
    return str(config.local_score_mode) == "spectral_csd_kv_bias"


def _is_chunked_csd_family_mode(config: StrictMergeConfig) -> bool:
    return _is_csd_kv_mode(config) or _is_spectral_csd_kv_mode(config) or _is_dynamic_csd_kv_mode(config) or _is_spectral_csd_kv_bias_mode(config)


def _uses_operator_probe_groups(config: StrictMergeConfig) -> bool:
    return _is_attention_operator_coreset_mode(config) or _is_grop_kv_mode(config)


def _needs_full_query_content(config: StrictMergeConfig) -> bool:
    return (
        str(config.selector_mode) == "local_jaoc"
        and _uses_operator_probe_groups(config)
        and str(config.operator_probe_source) in {"local_plus_tail", "local_only"}
    )


def _qcov_key_metrics_from_qobs(
    *,
    q_obs: torch.Tensor,
    hkv: int,
    groups: int,
    head_dim: int,
) -> torch.Tensor | None:
    if int(q_obs.numel()) <= 0:
        return None
    expected_hq = int(hkv) * int(groups)
    if int(q_obs.shape[0]) != int(expected_hq):
        raise ValueError(f"q_obs head mismatch: expected {expected_hq}, got {int(q_obs.shape[0])}")
    obs_count = int(q_obs.shape[1])
    if int(obs_count) <= 1:
        return None
    q_grouped = q_obs.float().reshape(int(hkv), int(groups), int(obs_count), int(head_dim))
    metric = q_grouped.var(dim=2, unbiased=False).mean(dim=1)
    metric = torch.nan_to_num(metric, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
    mean = metric.mean(dim=-1, keepdim=True).clamp_min(1e-6)
    return (metric / mean).contiguous()


def _qcov_key_metrics_from_qobs_batched(
    *,
    q_obs: torch.Tensor,
    hkv: int,
    groups: int,
    head_dim: int,
) -> torch.Tensor | None:
    if int(q_obs.numel()) <= 0:
        return None
    batch_size = int(q_obs.shape[0])
    expected_hq = int(hkv) * int(groups)
    if int(q_obs.shape[1]) != int(expected_hq):
        raise ValueError(f"q_obs head mismatch: expected {expected_hq}, got {int(q_obs.shape[1])}")
    obs_count = int(q_obs.shape[2])
    if int(obs_count) <= 1:
        return None
    q_grouped = q_obs.float().reshape(batch_size, int(hkv), int(groups), int(obs_count), int(head_dim))
    metric = q_grouped.var(dim=3, unbiased=False).mean(dim=2)
    metric = torch.nan_to_num(metric, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
    mean = metric.mean(dim=-1, keepdim=True).clamp_min(1e-6)
    return (metric / mean).contiguous()


def _key_metric_label(config: StrictMergeConfig, key_metrics: torch.Tensor | None) -> str:
    if str(getattr(config, "key_metric_mode", "raw")) == "qcov_diag" and key_metrics is not None:
        return "qcovdiag"
    return "rawk"


def _probe_limit_indices(total: int, limit: int, device: torch.device) -> torch.Tensor:
    total = int(total)
    limit = int(limit)
    if total <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    if limit <= 0 or limit >= total:
        return torch.arange(total, dtype=torch.long, device=device)
    # Deterministic stratified subsampling. It preserves nested reproducibility
    # while avoiding random per-run probe noise.
    return torch.div(torch.arange(limit, dtype=torch.long, device=device) * total, limit, rounding_mode="floor")


def _chunkwise_quota_plan_from_weights(
    *,
    selected: torch.Tensor,
    old_len: int,
    target: int,
    chunk_size: int,
    chunk_weights: torch.Tensor,
    min_chunk_keep: int = 0,
) -> tuple[list[int], list[int], list[int]]:
    seq_len = int(selected.numel())
    old_len = max(min(int(old_len), seq_len), 0)
    target = min(max(int(target), int(selected.sum().item())), seq_len)
    recent_forced = int(selected[old_len:].sum().item())
    target_old = min(max(target - recent_forced, int(selected[:old_len].sum().item())), old_len)
    chunk_size = max(int(chunk_size), 2)
    starts = list(range(0, old_len, chunk_size))
    lengths = [min(chunk_size, old_len - s) for s in starts]
    if not starts:
        return [], [], []

    forced_counts = [int(selected[s : s + l].sum().item()) for s, l in zip(starts, lengths)]
    min_keep = max(int(min_chunk_keep), 0)
    base_counts = [min(max(int(force), min_keep), int(length)) for force, length in zip(forced_counts, lengths)]
    if sum(base_counts) > int(target_old):
        # Budget is too small to cover every chunk. Keep the strongest chunks
        # by mass while preserving all already-forced tokens.
        weights_tmp = torch.nan_to_num(chunk_weights.float().flatten(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        if int(weights_tmp.numel()) != len(starts) or float(weights_tmp.sum().item()) <= 0.0:
            weights_tmp = torch.tensor(lengths, dtype=torch.float32, device=selected.device)
        quotas = list(forced_counts)
        remaining_base = max(int(target_old) - int(sum(quotas)), 0)
        order = torch.argsort(weights_tmp, descending=True).tolist()
        for idx in order:
            if remaining_base <= 0:
                break
            room = max(min_keep, 1) - quotas[int(idx)]
            room = min(max(int(room), 0), int(lengths[int(idx)]) - int(quotas[int(idx)]))
            take = min(int(room), int(remaining_base))
            quotas[int(idx)] += int(take)
            remaining_base -= int(take)
        return starts, lengths, quotas

    quotas = list(base_counts)
    remaining = max(int(target_old) - int(sum(quotas)), 0)
    if remaining <= 0:
        return starts, lengths, quotas

    weights = torch.nan_to_num(chunk_weights.float().flatten(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if int(weights.numel()) != len(starts):
        raise ValueError(f"chunk_weights must have {len(starts)} entries, got {int(weights.numel())}")
    if float(weights.sum().item()) <= 0.0:
        weights = torch.tensor(lengths, dtype=torch.float32, device=selected.device)
    weights = weights / weights.sum().clamp_min(1e-12)
    raw_extra = weights * float(remaining)
    extras = torch.floor(raw_extra).to(torch.long).tolist()
    extras = [min(int(extra), max(int(length) - int(force), 0)) for extra, length, force in zip(extras, lengths, forced_counts)]
    quotas = [int(force) + int(extra) for force, extra in zip(forced_counts, extras)]
    remaining = max(int(target_old) - int(sum(quotas)), 0)
    fractional_order = torch.argsort(raw_extra - torch.floor(raw_extra), descending=True).tolist()
    while remaining > 0:
        changed = False
        for idx in fractional_order:
            if remaining <= 0:
                break
            if quotas[int(idx)] < lengths[int(idx)]:
                quotas[int(idx)] += 1
                remaining -= 1
                changed = True
        if not changed:
            break
    return starts, lengths, quotas


def _tail_chunk_mass_weights(
    *,
    q_obs: torch.Tensor,
    head_keys: torch.Tensor,
    starts: list[int],
    lengths: list[int],
    uniform_mix: float,
) -> torch.Tensor:
    device = head_keys.device
    old_len = int(head_keys.shape[0])
    if old_len <= 0 or not starts:
        return torch.empty((0,), dtype=torch.float32, device=device)
    q = q_obs.float().reshape(-1, int(head_keys.shape[-1]))
    if int(q.numel()) == 0:
        return torch.tensor(lengths, dtype=torch.float32, device=device)
    logits = (q @ head_keys.float().T) / (float(head_keys.shape[-1]) ** 0.5)
    alpha = torch.softmax(logits, dim=-1, dtype=torch.float32)
    masses = []
    for start, length in zip(starts, lengths):
        end = int(start) + int(length)
        masses.append(alpha[:, start:end].sum(dim=-1).mean())
    mass = torch.stack(masses, dim=0).clamp_min(0.0)
    uniform = torch.tensor(lengths, dtype=torch.float32, device=device)
    uniform = uniform / uniform.sum().clamp_min(1e-12)
    mass = mass / mass.sum().clamp_min(1e-12)
    mix = min(max(float(uniform_mix), 0.0), 1.0)
    return (1.0 - mix) * mass + mix * uniform


@torch.no_grad()
def _chunkwise_keep_attention_operator_coreset(
    *,
    keys: torch.Tensor,
    values: torch.Tensor,
    q_obs: torch.Tensor,
    q_all: torch.Tensor | None,
    kv_head: int,
    old_len: int,
    target: int,
    force_mask: torch.Tensor,
    chunk_size: int,
    num_key_value_groups: int,
    config: StrictMergeConfig,
    value_metric: torch.Tensor | None = None,
) -> torch.Tensor:
    """Greedy coreset for approximating each chunk-local attention operator.

    Each chunk is optimized independently under a fixed quota. For every
    candidate, the objective compares the full chunk readout against the
    subset-renormalized readout over real query probes plus key-direction
    probes. The implementation keeps the greedy loop only over quota steps and
    evaluates all chunks/candidates in batched tensor algebra.
    """

    selected = force_mask.clone()
    seq_len = int(selected.numel())
    old_len = max(min(int(old_len), seq_len), 0)
    target = min(max(int(target), int(selected.sum().item())), seq_len)
    if old_len <= 0 or int(selected.sum().item()) >= target:
        return torch.nonzero(selected, as_tuple=False).flatten().sort().values

    device = keys.device
    head_dim = int(keys.shape[-1])
    groups = max(int(num_key_value_groups), 1)
    q_start = int(kv_head) * groups
    q_end = q_start + groups
    head_keys = keys[0, int(kv_head), :old_len, :].float()
    head_values = values[0, int(kv_head), :old_len, :].float()

    if _is_mass_weighted_attention_operator_mode(config):
        base_starts = list(range(0, int(old_len), max(int(chunk_size), 2)))
        base_lengths = [min(max(int(chunk_size), 2), int(old_len) - s) for s in base_starts]
        chunk_weights = _tail_chunk_mass_weights(
            q_obs=q_obs[q_start:q_end],
            head_keys=head_keys,
            starts=base_starts,
            lengths=base_lengths,
            uniform_mix=float(config.operator_mass_uniform_mix),
        )
        starts, lengths, quotas = _chunkwise_quota_plan_from_weights(
            selected=selected,
            old_len=int(old_len),
            target=int(target),
            chunk_size=int(chunk_size),
            chunk_weights=chunk_weights,
            min_chunk_keep=int(config.operator_min_chunk_keep),
        )
    else:
        starts, lengths, quotas = _chunkwise_quota_plan(
            selected=selected,
            old_len=int(old_len),
            target=int(target),
            chunk_size=int(chunk_size),
        )
    if not starts:
        return torch.nonzero(selected, as_tuple=False).flatten().sort().values

    num_chunks = len(starts)
    width = max(lengths)

    starts_t = torch.tensor(starts, dtype=torch.long, device=device).view(num_chunks, 1)
    lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
    quotas_t = torch.tensor(quotas, dtype=torch.long, device=device)
    local_pos = torch.arange(width, dtype=torch.long, device=device).view(1, width)
    valid = local_pos < lengths_t.view(num_chunks, 1)
    chunk_idx = (starts_t + local_pos).clamp_max(max(int(old_len) - 1, 0))
    flat_idx = chunk_idx.reshape(-1)

    k_chunks = head_keys.index_select(0, flat_idx).reshape(num_chunks, width, head_dim)
    u_chunks = head_values.index_select(0, flat_idx).reshape(num_chunks, width, head_dim)
    if value_metric is not None:
        metric = value_metric.to(device=device, dtype=torch.float32).flatten()
        if int(metric.numel()) != int(head_dim):
            raise ValueError(f"value_metric must have length {head_dim}, got {int(metric.numel())}")
        metric = torch.nan_to_num(metric, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
        u_chunks = u_chunks * metric.sqrt().view(1, 1, head_dim)
    u_chunks = u_chunks.masked_fill(~valid.unsqueeze(-1), 0.0)
    local_selected = selected.index_select(0, flat_idx).reshape(num_chunks, width) & valid

    q_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    probe_valid_parts: list[torch.Tensor] = []

    probe_source = str(config.operator_probe_source)
    include_local = probe_source in {"local_plus_tail", "local_only"}
    include_tail = probe_source in {"local_plus_tail", "tail_only"}

    if q_all is not None and include_local:
        q_all_head = q_all[q_start:q_end, :old_len, :].float()
        q_local = q_all_head.index_select(1, flat_idx).permute(1, 0, 2).reshape(num_chunks, width, groups, head_dim)
        q_local = q_local.permute(0, 2, 1, 3).reshape(num_chunks, groups * width, head_dim)
        local_probe_valid = valid[:, None, :].expand(num_chunks, groups, width).reshape(num_chunks, groups * width)
        offsets = torch.arange(width, dtype=torch.long, device=device).repeat(groups)
        local_causal = local_pos <= offsets.view(1, groups * width, 1)
        local_mask = valid[:, None, :] & local_causal & local_probe_valid[:, :, None]
        q_parts.append(q_local)
        mask_parts.append(local_mask)
        probe_valid_parts.append(local_probe_valid)

    q_tail = q_obs[q_start:q_end].float().reshape(groups * int(q_obs.shape[1]), head_dim)
    if include_tail and int(q_tail.numel()) > 0:
        q_tail = q_tail.unsqueeze(0).expand(num_chunks, -1, -1).contiguous()
        tail_valid = torch.ones((num_chunks, int(q_tail.shape[1])), dtype=torch.bool, device=device)
        q_parts.append(q_tail)
        mask_parts.append(valid[:, None, :].expand(num_chunks, int(q_tail.shape[1]), width))
        probe_valid_parts.append(tail_valid)

    if q_parts:
        real_q = torch.cat(q_parts, dim=1)
        real_mask = torch.cat(mask_parts, dim=1)
        real_probe_valid = torch.cat(probe_valid_parts, dim=1)
        real_idx = _probe_limit_indices(int(real_q.shape[1]), int(config.operator_real_probe_limit), device=device)
        real_q = real_q.index_select(1, real_idx)
        real_mask = real_mask.index_select(1, real_idx)
        real_probe_valid = real_probe_valid.index_select(1, real_idx)
    else:
        real_q = torch.empty((num_chunks, 0, head_dim), dtype=torch.float32, device=device)
        real_mask = torch.empty((num_chunks, 0, width), dtype=torch.bool, device=device)
        real_probe_valid = torch.empty((num_chunks, 0), dtype=torch.bool, device=device)

    q_norm_source = q_obs[q_start:q_end].float().reshape(-1, head_dim)
    if int(q_norm_source.numel()) > 0:
        gamma = q_norm_source.norm(dim=-1).median().clamp_min(1e-6)
    else:
        gamma = head_keys.norm(dim=-1).median().clamp_min(1e-6)
    key_idx = _probe_limit_indices(width, int(config.operator_key_probe_limit), device=device)
    key_idx = key_idx.index_select(0, torch.nonzero(key_idx < width, as_tuple=False).flatten())
    if int(key_idx.numel()) > 0:
        key_q = F.normalize(k_chunks.index_select(1, key_idx), p=2, dim=-1, eps=1e-6) * gamma
        key_probe_valid = valid.index_select(1, key_idx)
        key_mask = valid[:, None, :].expand(num_chunks, int(key_idx.numel()), width) & key_probe_valid[:, :, None]
    else:
        key_q = torch.empty((num_chunks, 0, head_dim), dtype=torch.float32, device=device)
        key_mask = torch.empty((num_chunks, 0, width), dtype=torch.bool, device=device)
        key_probe_valid = torch.empty((num_chunks, 0), dtype=torch.bool, device=device)

    probes = torch.cat([real_q, key_q], dim=1)
    probe_mask = torch.cat([real_mask, key_mask], dim=1)
    probe_valid = torch.cat([real_probe_valid, key_probe_valid], dim=1)
    if int(probes.shape[1]) <= 0:
        return _chunkwise_keep_anchor_then_cover(
            keys=keys,
            values=values,
            kv_head=int(kv_head),
            old_len=int(old_len),
            target=int(target),
            force_mask=force_mask,
            chunk_size=int(chunk_size),
            config=config,
            value_metric=value_metric,
        )

    scale = 1.0 / (float(head_dim) ** 0.5)
    logits = torch.bmm(probes, k_chunks.transpose(1, 2)) * scale
    logits = logits.masked_fill(~probe_mask, -1.0e30)
    row_max = logits.amax(dim=-1, keepdim=True)
    exp_logits = torch.exp(logits - row_max).masked_fill(~probe_mask, 0.0)
    full_z = exp_logits.sum(dim=-1).clamp_min(1e-12)
    target_out = torch.bmm(exp_logits, u_chunks) / full_z.unsqueeze(-1)
    target_out = target_out.masked_fill(~probe_valid.unsqueeze(-1), 0.0)

    valid_f = valid.float()
    denom = lengths_t.float().clamp_min(1.0).view(num_chunks, 1, 1)
    mean_u = (u_chunks * valid_f.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom
    var_u = ((u_chunks - mean_u).square().sum(dim=-1) * valid_f).sum(dim=1) / lengths_t.float().clamp_min(1.0)
    var_u = var_u.clamp_min(float(config.denom_eps))

    selected_f = local_selected.float()
    z_sel = (exp_logits * selected_f[:, None, :]).sum(dim=-1)
    n_sel = torch.bmm(exp_logits * selected_f[:, None, :], u_chunks)

    target_norm2 = target_out.square().sum(dim=-1)
    target_dot_u = torch.bmm(target_out, u_chunks.transpose(1, 2))
    u_norm2 = u_chunks.square().sum(dim=-1)

    need = (quotas_t - local_selected.sum(dim=1)).clamp_min(0)
    max_need = int(need.max().item()) if int(need.numel()) > 0 else 0
    for _ in range(max_need):
        active = need > 0
        candidate_mask = active[:, None] & valid & (~local_selected)
        if not bool(candidate_mask.any().item()):
            break

        target_dot_nsel = (target_out * n_sel).sum(dim=-1)
        nsel_norm2 = n_sel.square().sum(dim=-1)
        nsel_dot_u = torch.bmm(n_sel, u_chunks.transpose(1, 2))

        z = (z_sel[:, :, None] + exp_logits).clamp_min(1e-12)
        target_dot_n = target_dot_nsel[:, :, None] + exp_logits * target_dot_u
        n_norm2 = (
            nsel_norm2[:, :, None]
            + 2.0 * exp_logits * nsel_dot_u
            + exp_logits.square() * u_norm2[:, None, :]
        )
        per_probe_loss = target_norm2[:, :, None] - 2.0 * target_dot_n / z + n_norm2 / z.square()
        per_probe_loss = per_probe_loss.clamp_min(0.0) * probe_valid[:, :, None].float()
        losses = per_probe_loss.sum(dim=1) / var_u.view(num_chunks, 1)
        losses = losses.masked_fill(~candidate_mask, float("inf"))

        local_idx = torch.argmin(losses, dim=1)
        chosen_loss = losses.gather(1, local_idx.view(num_chunks, 1)).squeeze(1)
        chosen = active & torch.isfinite(chosen_loss)
        if not bool(chosen.any().item()):
            break
        local_selected[chosen, local_idx[chosen]] = True
        need = need - chosen.long()

        chosen_e = exp_logits.gather(2, local_idx.view(num_chunks, 1, 1).expand(num_chunks, int(probes.shape[1]), 1)).squeeze(2)
        chosen_u = u_chunks.gather(1, local_idx.view(num_chunks, 1, 1).expand(num_chunks, 1, head_dim)).squeeze(1)
        z_sel = torch.where(chosen.view(num_chunks, 1), z_sel + chosen_e, z_sel)
        n_add = chosen_e.unsqueeze(-1) * chosen_u[:, None, :]
        n_sel = torch.where(chosen.view(num_chunks, 1, 1), n_sel + n_add, n_sel)

    old_selected = chunk_idx[local_selected & valid]
    if int(old_selected.numel()) > 0:
        selected.index_fill_(0, old_selected.to(device=selected.device, dtype=torch.long), True)

    current = int(selected.sum().item())
    if current < target:
        need_fill = int(target) - current
        candidates = torch.nonzero(~selected[:old_len], as_tuple=False).flatten()
        take = min(int(need_fill), int(candidates.numel()))
        if take > 0:
            selected.index_fill_(0, candidates[:take].to(device=selected.device, dtype=torch.long), True)
    elif current > target:
        excess = current - int(target)
        removable = selected[:old_len] & (~force_mask[:old_len])
        pos = torch.nonzero(removable, as_tuple=False).flatten()
        if int(pos.numel()) > 0:
            selected.index_fill_(0, pos[:excess].to(device=selected.device, dtype=torch.long), False)

    return torch.nonzero(selected, as_tuple=False).flatten().sort().values


@torch.no_grad()
def _chunkwise_keep_csd_kv(
    *,
    keys: torch.Tensor,
    values: torch.Tensor,
    q_obs: torch.Tensor,
    kv_head: int,
    old_len: int,
    target: int,
    force_mask: torch.Tensor,
    chunk_size: int,
    num_key_value_groups: int,
    config: StrictMergeConfig,
    value_metric: torch.Tensor | None = None,
) -> torch.Tensor:
    """Chunked Spine-Detail KV selection for one KV head.

    CSD-KV uses observation attention as an importance prior, allocates chunk
    budgets by attention mass plus one-anchor residual complexity, then selects
    a spine medoid followed by attention-weighted residual detail tokens.
    """

    selected = force_mask.clone()
    seq_len = int(selected.numel())
    old_len = max(min(int(old_len), seq_len), 0)
    target = min(max(int(target), int(selected.sum().item())), seq_len)
    if old_len <= 0 or int(selected.sum().item()) >= target:
        return torch.nonzero(selected, as_tuple=False).flatten().sort().values

    device = keys.device
    head_dim = int(keys.shape[-1])
    groups = max(int(num_key_value_groups), 1)
    q_start = int(kv_head) * groups
    q_end = q_start + groups
    head_keys_full = keys[0, int(kv_head), :, :].float()
    head_keys = head_keys_full[:old_len]
    head_values = values[0, int(kv_head), :old_len, :].float()

    q = q_obs[q_start:q_end].float().reshape(-1, head_dim)
    if int(q.numel()) > 0:
        logits = (q @ head_keys_full.T) / (float(head_dim) ** 0.5)
        attn = torch.softmax(logits, dim=-1, dtype=torch.float32)[:, :old_len]
        p_mean = attn.mean(dim=0)
        p_max = attn.amax(dim=0)
        importance = 0.5 * p_mean + 0.5 * p_max
    else:
        importance = torch.ones((old_len,), dtype=torch.float32, device=device)
    importance = torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    kernel = int(config.snap_kernel_size)
    if kernel > 1 and old_len > 0:
        pooled = importance.view(1, 1, old_len)
        if str(config.snap_pooling) == "maxpool":
            importance = F.max_pool1d(pooled, kernel_size=kernel, padding=kernel // 2, stride=1).view(-1)[:old_len]
        else:
            importance = F.avg_pool1d(pooled, kernel_size=kernel, padding=kernel // 2, stride=1).view(-1)[:old_len]
    if float(importance.sum().item()) <= 0.0:
        importance = torch.ones_like(importance)

    starts = list(range(0, int(old_len), max(int(chunk_size), 2)))
    lengths = [min(max(int(chunk_size), 2), int(old_len) - s) for s in starts]
    if not starts:
        return torch.nonzero(selected, as_tuple=False).flatten().sort().values
    num_chunks = len(starts)
    width = max(lengths)
    starts_t = torch.tensor(starts, dtype=torch.long, device=device).view(num_chunks, 1)
    lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
    local_pos = torch.arange(width, dtype=torch.long, device=device).view(1, width)
    valid = local_pos < lengths_t.view(num_chunks, 1)
    chunk_idx = (starts_t + local_pos).clamp_max(max(int(old_len) - 1, 0))
    flat_idx = chunk_idx.reshape(-1)

    k_chunks = head_keys.index_select(0, flat_idx).reshape(num_chunks, width, head_dim)
    v_chunks = head_values.index_select(0, flat_idx).reshape(num_chunks, width, head_dim)
    p_chunks = importance.index_select(0, flat_idx).reshape(num_chunks, width).masked_fill(~valid, 0.0)

    v_features = v_chunks
    if value_metric is not None:
        metric = value_metric.to(device=device, dtype=torch.float32).flatten()
        if int(metric.numel()) != int(head_dim):
            raise ValueError(f"value_metric must have length {head_dim}, got {int(metric.numel())}")
        metric = torch.nan_to_num(metric, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
        v_features = v_features * metric.sqrt().view(1, 1, head_dim)

    key_weight = float(config.coreset_key_weight)
    value_weight = float(config.coreset_value_weight)
    parts: list[torch.Tensor] = []
    if key_weight > 0.0:
        parts.append(F.normalize(k_chunks, p=2, dim=-1, eps=1e-6) * key_weight)
    if value_weight > 0.0:
        parts.append(F.normalize(v_features, p=2, dim=-1, eps=1e-6) * value_weight)
    if not parts:
        raise ValueError("At least one coreset feature weight must be positive.")
    features = F.normalize(torch.cat(parts, dim=-1), p=2, dim=-1, eps=1e-6)
    features = features.masked_fill(~valid.unsqueeze(-1), 0.0)
    k_features = F.normalize(k_chunks, p=2, dim=-1, eps=1e-6).masked_fill(~valid.unsqueeze(-1), 0.0)

    sim = torch.matmul(features, features.transpose(1, 2))
    sim = ((sim + 1.0) * 0.5).clamp(0.0, 1.0)
    sim_k = torch.matmul(k_features, k_features.transpose(1, 2))
    sim_k = ((sim_k + 1.0) * 0.5).clamp(0.0, 1.0)
    valid_pair = valid[:, :, None] & valid[:, None, :]
    sim = sim.masked_fill(~valid_pair, 0.0)
    sim_k = sim_k.masked_fill(~valid_pair, 0.0)

    chunk_mass = p_chunks.sum(dim=1).clamp_min(0.0)
    uniform_w = valid.float() / lengths_t.float().clamp_min(1.0).view(num_chunks, 1)
    weights_i = torch.where(
        chunk_mass.view(num_chunks, 1) > 1e-12,
        p_chunks / chunk_mass.view(num_chunks, 1).clamp_min(1e-12),
        uniform_w,
    ).masked_fill(~valid, 0.0)

    p_mean_chunk = (p_chunks.sum(dim=1, keepdim=True) / lengths_t.float().clamp_min(1.0).view(num_chunks, 1)).clamp_min(1e-12)
    p_rel = (p_chunks / p_mean_chunk).clamp(0.0, 4.0).masked_fill(~valid, 0.0)
    beta = float(config.anchor_ratio)
    anchor_scores = (weights_i[:, :, None] * sim_k).sum(dim=1) + beta * p_rel
    anchor_scores = anchor_scores.masked_fill(~valid, float("-inf"))
    anchor_idx = torch.argmax(anchor_scores, dim=1)
    anchor_sim = sim.gather(2, anchor_idx.view(num_chunks, 1, 1).expand(num_chunks, width, 1)).squeeze(2)
    residual = ((1.0 - anchor_sim).clamp_min(0.0) * weights_i).sum(dim=1)

    rho = 0.75
    gamma = max(float(config.coreset_tau), 0.0)
    chunk_weights = (chunk_mass + 1e-12).pow(rho) * (1.0 + gamma * residual.clamp(0.0, 1.0))
    mix = min(max(float(config.operator_mass_uniform_mix), 0.0), 1.0)
    if mix > 0.0:
        uniform = torch.tensor(lengths, dtype=torch.float32, device=device)
        chunk_weights = (1.0 - mix) * chunk_weights + mix * uniform / uniform.sum().clamp_min(1e-12) * chunk_weights.sum().clamp_min(1e-12)

    starts, lengths, quotas = _chunkwise_quota_plan_from_weights(
        selected=selected,
        old_len=int(old_len),
        target=int(target),
        chunk_size=int(chunk_size),
        chunk_weights=chunk_weights,
        min_chunk_keep=int(config.operator_min_chunk_keep),
    )
    quotas_t = torch.tensor(quotas, dtype=torch.long, device=device)
    local_selected = selected.index_select(0, flat_idx).reshape(num_chunks, width) & valid
    need = (quotas_t - local_selected.sum(dim=1)).clamp_min(0)

    # First token per active chunk: attention-weighted key medoid spine.
    active = need > 0
    if bool(active.any().item()):
        spine_scores = anchor_scores.masked_fill(local_selected | (~valid) | (~active[:, None]), float("-inf"))
        spine_idx = torch.argmax(spine_scores, dim=1)
        spine_score = spine_scores.gather(1, spine_idx.view(num_chunks, 1)).squeeze(1)
        chosen = active & torch.isfinite(spine_score)
        local_selected[chosen, spine_idx[chosen]] = True
        need = need - chosen.long()

    covered = sim.masked_fill(~local_selected[:, None, :], 0.0).amax(dim=2)
    max_need = int(need.max().item()) if int(need.numel()) > 0 else 0
    lambda_res = 1.0
    for _ in range(max_need):
        active = need > 0
        candidate_mask = active[:, None] & valid & (~local_selected)
        if not bool(candidate_mask.any().item()):
            break
        novelty = (1.0 - covered).clamp(0.0, 1.0)
        scores = p_rel * (1.0 + lambda_res * novelty)
        # Tie-break toward candidates that cover more important local mass.
        gains = ((sim - covered[:, :, None]).clamp_min(0.0) * weights_i[:, :, None]).sum(dim=1)
        scores = scores + 1e-3 * gains + 1e-6 * anchor_scores.clamp_min(0.0)
        scores = scores.masked_fill(~candidate_mask, float("-inf"))
        local_idx = torch.argmax(scores, dim=1)
        chosen_score = scores.gather(1, local_idx.view(num_chunks, 1)).squeeze(1)
        chosen = active & torch.isfinite(chosen_score)
        if not bool(chosen.any().item()):
            break
        local_selected[chosen, local_idx[chosen]] = True
        need = need - chosen.long()
        chosen_sim = sim.gather(2, local_idx.view(num_chunks, 1, 1).expand(num_chunks, width, 1)).squeeze(2)
        covered = torch.where(chosen.view(num_chunks, 1), torch.maximum(covered, chosen_sim), covered)

    old_selected = chunk_idx[local_selected & valid]
    if int(old_selected.numel()) > 0:
        selected.index_fill_(0, old_selected.to(device=selected.device, dtype=torch.long), True)

    current = int(selected.sum().item())
    if current < target:
        need_fill = int(target) - current
        candidates = torch.nonzero(~selected[:old_len], as_tuple=False).flatten()
        if int(candidates.numel()) > 0:
            cand_scores = importance.index_select(0, candidates)
            take = min(int(need_fill), int(candidates.numel()))
            fill = candidates.index_select(0, torch.topk(cand_scores, k=take, largest=True).indices)
            selected.index_fill_(0, fill.to(device=selected.device, dtype=torch.long), True)
    elif current > target:
        excess = current - int(target)
        removable = selected[:old_len] & (~force_mask[:old_len])
        pos = torch.nonzero(removable, as_tuple=False).flatten()
        if int(pos.numel()) > 0:
            drop = pos.index_select(0, torch.argsort(importance.index_select(0, pos))[:excess])
            selected.index_fill_(0, drop.to(device=selected.device, dtype=torch.long), False)

    return torch.nonzero(selected, as_tuple=False).flatten().sort().values


@torch.no_grad()
def _spectral_csd_importance_scores(
    *,
    q_obs: torch.Tensor,
    keys: torch.Tensor,
    old_len: int,
    num_key_value_groups: int,
    config: StrictMergeConfig,
) -> torch.Tensor:
    if int(keys.shape[0]) != 1:
        raise NotImplementedError("StrictMerge supports batch_size=1.")
    hkv = int(keys.shape[1])
    seq_len = int(keys.shape[2])
    head_dim = int(keys.shape[3])
    old_len = max(min(int(old_len), int(seq_len)), 0)
    if old_len <= 0:
        return torch.empty((hkv, 0), dtype=torch.float32, device=keys.device)

    groups = max(int(num_key_value_groups), 1)
    if int(q_obs.numel()) > 0:
        expected_hq = hkv * groups
        if int(q_obs.shape[0]) != int(expected_hq):
            raise ValueError(f"q_obs head mismatch: expected {expected_hq}, got {int(q_obs.shape[0])}")
        obs_count = int(q_obs.shape[1])
        q_grouped = q_obs.float().reshape(hkv, groups, obs_count, head_dim)
        logits = torch.einsum("hgrd,htd->hgrt", q_grouped, keys[0].float())
        logits = logits / (float(head_dim) ** 0.5)
        attn = torch.softmax(logits, dim=-1, dtype=torch.float32)[..., :old_len]
        p_mean = attn.mean(dim=(1, 2))
        p_max = attn.amax(dim=(1, 2))
        importance = 0.5 * p_mean + 0.5 * p_max
    else:
        importance = torch.ones((hkv, old_len), dtype=torch.float32, device=keys.device)

    importance = torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    kernel = int(config.snap_kernel_size)
    if kernel > 1 and old_len > 0:
        pooled = importance.unsqueeze(1)
        if str(config.snap_pooling) == "maxpool":
            importance = F.max_pool1d(pooled, kernel_size=kernel, padding=kernel // 2, stride=1).squeeze(1)
        else:
            importance = F.avg_pool1d(pooled, kernel_size=kernel, padding=kernel // 2, stride=1).squeeze(1)
        importance = importance[:, :old_len]
    zero_heads = importance.sum(dim=1, keepdim=True) <= 0.0
    return torch.where(zero_heads, torch.ones_like(importance), importance)


@torch.no_grad()
def _spectral_csd_importance_scores_batched(
    *,
    q_obs: torch.Tensor,
    keys: torch.Tensor,
    old_lens: list[int],
    seq_lens: list[int],
    num_key_value_groups: int,
    config: StrictMergeConfig,
) -> torch.Tensor:
    if int(keys.dim()) != 4:
        raise ValueError(f"keys must have shape [B,H,K,D], got {tuple(keys.shape)}")
    batch_size = int(keys.shape[0])
    hkv = int(keys.shape[1])
    max_seq = int(keys.shape[2])
    head_dim = int(keys.shape[3])
    max_old = max((int(v) for v in old_lens), default=0)
    if max_old <= 0:
        return torch.empty((batch_size, hkv, 0), dtype=torch.float32, device=keys.device)

    groups = max(int(num_key_value_groups), 1)
    expected_hq = hkv * groups
    if int(q_obs.shape[0]) != batch_size or int(q_obs.shape[1]) != expected_hq:
        raise ValueError(f"q_obs must have shape [B,{expected_hq},R,D], got {tuple(q_obs.shape)}")
    obs_count = int(q_obs.shape[2])
    old_lens_t = torch.tensor([int(v) for v in old_lens], dtype=torch.long, device=keys.device)
    seq_lens_t = torch.tensor([int(v) for v in seq_lens], dtype=torch.long, device=keys.device)
    pos = torch.arange(max_seq, dtype=torch.long, device=keys.device).view(1, max_seq)
    valid_seq = pos < seq_lens_t.view(batch_size, 1)
    old_pos = torch.arange(max_old, dtype=torch.long, device=keys.device).view(1, max_old)
    valid_old = old_pos < old_lens_t.view(batch_size, 1)

    if int(q_obs.numel()) > 0:
        q_grouped = q_obs.float().reshape(batch_size, hkv, groups, obs_count, head_dim)
        logits = torch.einsum("bhgrd,bhtd->bhgrt", q_grouped, keys.float())
        logits = logits / (float(head_dim) ** 0.5)
        logits = logits.masked_fill(~valid_seq.view(batch_size, 1, 1, 1, max_seq), float("-inf"))
        attn = torch.softmax(logits, dim=-1, dtype=torch.float32)[..., :max_old]
        attn = attn.masked_fill(~valid_old.view(batch_size, 1, 1, 1, max_old), 0.0)
        p_mean = attn.mean(dim=(2, 3))
        p_max = attn.amax(dim=(2, 3))
        importance = 0.5 * p_mean + 0.5 * p_max
    else:
        importance = valid_old[:, None, :].expand(batch_size, hkv, max_old).to(dtype=torch.float32)

    importance = torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    kernel = int(config.snap_kernel_size)
    if kernel > 1 and max_old > 0:
        pooled = importance.reshape(batch_size * hkv, 1, max_old)
        if str(config.snap_pooling) == "maxpool":
            pooled = F.max_pool1d(pooled, kernel_size=kernel, padding=kernel // 2, stride=1)
        else:
            pooled = F.avg_pool1d(pooled, kernel_size=kernel, padding=kernel // 2, stride=1)
        importance = pooled[:, :, :max_old].reshape(batch_size, hkv, max_old)
    importance = importance.masked_fill(~valid_old[:, None, :], 0.0)
    zero_heads = importance.sum(dim=2, keepdim=True) <= 0.0
    fallback = valid_old[:, None, :].expand(batch_size, hkv, max_old).to(dtype=importance.dtype)
    return torch.where(zero_heads, fallback, importance)


@torch.no_grad()
def _chunkwise_keep_spectral_csd_kv(
    *,
    keys: torch.Tensor,
    values: torch.Tensor,
    q_obs: torch.Tensor,
    kv_head: int,
    old_len: int,
    target: int,
    force_mask: torch.Tensor,
    chunk_size: int,
    num_key_value_groups: int,
    config: StrictMergeConfig,
    value_metric: torch.Tensor | None = None,
    key_metric: torch.Tensor | None = None,
    importance: torch.Tensor | None = None,
    profile: dict[str, float] | None = None,
) -> torch.Tensor:
    """Spectral-CSD-KV selection for one KV head.

    The selector uses pooled observation attention as p_i, forms chunk-local
    weighted features z_i = sqrt(p_i) * [scaled K_i, scaled V_i], allocates
    chunk budgets by spectral water-filling over Z Z^T, then performs batched
    greedy CPQR pivots with any forced tokens already in the span.
    """

    selected = force_mask.clone()
    seq_len = int(selected.numel())
    old_len = max(min(int(old_len), seq_len), 0)
    target = min(max(int(target), int(selected.sum().item())), seq_len)
    if old_len <= 0 or int(selected.sum().item()) >= target:
        return torch.nonzero(selected, as_tuple=False).flatten().sort().values

    device = keys.device
    head_dim = int(keys.shape[-1])
    stage_t = _profile_start(device, profile)
    head_cache_keys_rope_full = keys[0, int(kv_head), :, :].float()
    head_keys = head_cache_keys_rope_full[:old_len]
    head_values = values[0, int(kv_head), :old_len, :].float()

    if importance is None:
        all_importance = _spectral_csd_importance_scores(
            q_obs=q_obs,
            keys=keys,
            old_len=int(old_len),
            num_key_value_groups=int(num_key_value_groups),
            config=config,
        )
        importance = all_importance[int(kv_head)]
    else:
        importance = importance.to(device=device, dtype=torch.float32).flatten()
        if int(importance.numel()) != int(old_len):
            raise ValueError(f"importance must have length {old_len}, got {int(importance.numel())}")
    importance = torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if float(importance.sum().item()) <= 0.0:
        importance = torch.ones_like(importance)
    _profile_add(profile, "chunk_importance_sec", stage_t, device)

    stage_t = _profile_start(device, profile)
    chunk_size = max(int(chunk_size), 2)
    starts = list(range(0, int(old_len), chunk_size))
    lengths = [min(chunk_size, int(old_len) - s) for s in starts]
    if not starts:
        return torch.nonzero(selected, as_tuple=False).flatten().sort().values

    num_chunks = len(starts)
    width = max(lengths)
    starts_t = torch.tensor(starts, dtype=torch.long, device=device).view(num_chunks, 1)
    lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
    local_pos = torch.arange(width, dtype=torch.long, device=device).view(1, width)
    valid = local_pos < lengths_t.view(num_chunks, 1)
    chunk_idx = (starts_t + local_pos).clamp_max(max(int(old_len) - 1, 0))
    flat_idx = chunk_idx.reshape(-1)

    k_chunks = head_keys.index_select(0, flat_idx).reshape(num_chunks, width, head_dim)
    v_chunks = head_values.index_select(0, flat_idx).reshape(num_chunks, width, head_dim)
    p_chunks = importance.index_select(0, flat_idx).reshape(num_chunks, width).masked_fill(~valid, 0.0)
    local_selected = selected.index_select(0, flat_idx).reshape(num_chunks, width) & valid
    _profile_add(profile, "chunk_gather_sec", stage_t, device)

    stage_t = _profile_start(device, profile)
    k_features = k_chunks
    if key_metric is not None:
        metric = key_metric.to(device=device, dtype=torch.float32).flatten()
        if int(metric.numel()) != int(head_dim):
            raise ValueError(f"key_metric must have length {head_dim}, got {int(metric.numel())}")
        metric = torch.nan_to_num(metric, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
        k_features = k_features * metric.sqrt().view(1, 1, head_dim)
    v_features = v_chunks
    if value_metric is not None:
        metric = value_metric.to(device=device, dtype=torch.float32).flatten()
        if int(metric.numel()) != int(head_dim):
            raise ValueError(f"value_metric must have length {head_dim}, got {int(metric.numel())}")
        metric = torch.nan_to_num(metric, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
        v_features = v_features * metric.sqrt().view(1, 1, head_dim)

    parts: list[torch.Tensor] = []
    key_weight = float(config.coreset_key_weight)
    value_weight = float(config.coreset_value_weight)
    if key_weight > 0.0:
        key_energy = k_features.square().sum(dim=-1).masked_fill(~valid, 0.0)
        key_scale = (key_energy.sum(dim=1) / lengths_t.float().clamp_min(1.0)).sqrt().clamp_min(1e-6)
        parts.append((k_features / key_scale.view(num_chunks, 1, 1)) * key_weight)
    if value_weight > 0.0:
        value_energy = v_features.square().sum(dim=-1).masked_fill(~valid, 0.0)
        value_scale = (value_energy.sum(dim=1) / lengths_t.float().clamp_min(1.0)).sqrt().clamp_min(1e-6)
        parts.append((v_features / value_scale.view(num_chunks, 1, 1)) * value_weight)
    if not parts:
        raise ValueError("At least one coreset feature weight must be positive.")

    features = torch.cat(parts, dim=-1).masked_fill(~valid.unsqueeze(-1), 0.0)
    z = features * torch.sqrt(p_chunks.clamp_min(0.0) + 1e-12).unsqueeze(-1)
    z = z.masked_fill(~valid.unsqueeze(-1), 0.0)
    _profile_add(profile, "chunk_feature_sec", stage_t, device)

    # Use the same forced-token projection for both quota allocation and
    # CPQR pivots: forced rows already explain their span, so water-filling
    # should see only residual spectral energy.
    stage_t = _profile_start(device, profile)
    residual_after_forced = _project_selected_chunk_columns(z, local_selected, valid)
    _profile_add(profile, "chunk_forced_project_sec", stage_t, device)

    recent_forced = int(selected[old_len:].sum().item())
    forced_counts = local_selected.sum(dim=1).to(torch.long)
    target_old = min(max(int(target) - recent_forced, int(selected[:old_len].sum().item())), int(old_len))
    min_keep = max(int(config.operator_min_chunk_keep), 0)
    if min_keep > 0:
        min_counts = torch.minimum(lengths_t, torch.full_like(lengths_t, min_keep))
        base_counts = torch.maximum(forced_counts, min_counts)
        if int(base_counts.sum().item()) <= int(target_old):
            quotas_t = base_counts.clone()
        else:
            quotas_t = forced_counts.clone()
    else:
        quotas_t = forced_counts.clone()

    extra_total = min(
        max(int(target_old) - int(quotas_t.sum().item()), 0),
        int((lengths_t - quotas_t).clamp_min(0).sum().item()),
    )
    if extra_total > 0:
        stage_t = _profile_start(device, profile)
        gram = torch.matmul(residual_after_forced, residual_after_forced.transpose(1, 2))
        eigvals = _eigvalsh_symmetric_chunked(
            gram,
            max_batch=int(os.environ.get("BEST_V1_SELECT_EIG_BATCH", "512")),
        ).clamp_min(0.0)
        eig_desc = torch.flip(eigvals, dims=[1])
        capacity = (lengths_t - quotas_t).clamp_min(0)
        rank_idx = torch.arange(width, dtype=torch.long, device=device).view(1, width)
        eig_scores = eig_desc.masked_fill(rank_idx >= capacity.view(num_chunks, 1), float("-inf"))
        flat_scores = eig_scores.reshape(-1)
        top = torch.topk(flat_scores, k=int(extra_total), largest=True).indices
        extra_counts = torch.bincount(torch.div(top, width, rounding_mode="floor"), minlength=num_chunks).to(torch.long)
        quotas_t = quotas_t + extra_counts
        _profile_add(profile, "chunk_eig_quota_sec", stage_t, device)

    need = (quotas_t - local_selected.sum(dim=1)).clamp_min(0)
    residual = residual_after_forced
    selected_gain = torch.full((num_chunks, width), float("-inf"), dtype=torch.float32, device=device)

    stage_t = _profile_start(device, profile)
    max_need = int(need.max().item()) if int(need.numel()) > 0 else 0
    for _ in range(max_need):
        active = need > 0
        candidate_mask = active[:, None] & valid & (~local_selected)
        scores = residual.square().sum(dim=-1).masked_fill(~candidate_mask, float("-inf"))
        local_idx = torch.argmax(scores, dim=1)
        chosen_score = scores.gather(1, local_idx.view(num_chunks, 1)).squeeze(1)
        chosen = active & torch.isfinite(chosen_score)
        local_selected[chosen, local_idx[chosen]] = True
        selected_gain[chosen, local_idx[chosen]] = chosen_score[chosen]
        next_need = need - chosen.long()
        residual = _project_pivot_chunk_rows(residual, local_idx, chosen & (next_need > 0))
        need = next_need
    _profile_add(profile, "chunk_cpqr_sec", stage_t, device)

    stage_t = _profile_start(device, profile)
    old_selected = chunk_idx[local_selected & valid]
    if int(old_selected.numel()) > 0:
        selected.index_fill_(0, old_selected.to(device=selected.device, dtype=torch.long), True)

    current = int(selected.sum().item())
    if current == int(target):
        out = torch.nonzero(selected, as_tuple=False).flatten().sort().values
        _profile_add(profile, "chunk_finalize_sec", stage_t, device)
        return out
    gain_by_old = torch.full((int(old_len),), float("-inf"), dtype=torch.float32, device=device)
    if int(old_len) > 0:
        residual_scores = residual.square().sum(dim=-1).masked_fill(~valid, float("-inf"))
        gain_by_old.index_copy_(0, chunk_idx[valid].flatten(), residual_scores[valid].flatten())
    selected_gain_by_old = torch.full((int(old_len),), float("-inf"), dtype=torch.float32, device=device)
    selected_gain_by_old.index_copy_(0, chunk_idx[valid].flatten(), selected_gain[valid].flatten())
    if current < target:
        need_fill = int(target) - current
        candidates = torch.nonzero(~selected[:old_len], as_tuple=False).flatten()
        if int(candidates.numel()) > 0:
            cand_scores = gain_by_old.index_select(0, candidates)
            if not bool(torch.isfinite(cand_scores).any().item()):
                cand_scores = importance.index_select(0, candidates)
            take = min(int(need_fill), int(candidates.numel()))
            fill = candidates.index_select(0, torch.topk(cand_scores, k=take, largest=True).indices)
            selected.index_fill_(0, fill.to(device=selected.device, dtype=torch.long), True)
    elif current > target:
        excess = current - int(target)
        removable = selected[:old_len] & (~force_mask[:old_len])
        pos = torch.nonzero(removable, as_tuple=False).flatten()
        if int(pos.numel()) > 0:
            gains = selected_gain_by_old.index_select(0, pos)
            if not bool(torch.isfinite(gains).any().item()):
                gains = importance.index_select(0, pos)
            drop = pos.index_select(0, torch.argsort(gains)[:excess])
            selected.index_fill_(0, drop.to(device=selected.device, dtype=torch.long), False)

    out = torch.nonzero(selected, as_tuple=False).flatten().sort().values
    _profile_add(profile, "chunk_finalize_sec", stage_t, device)
    return out


@torch.no_grad()
def _chunkwise_keep_spectral_csd_kv_all_heads(
    *,
    keys: torch.Tensor,
    values: torch.Tensor,
    old_len: int,
    target: int,
    force_mask: torch.Tensor,
    chunk_size: int,
    config: StrictMergeConfig,
    value_metrics: torch.Tensor | None = None,
    key_metrics: torch.Tensor | None = None,
    importance: torch.Tensor | None = None,
    profile: dict[str, float] | None = None,
) -> torch.Tensor:
    """Vectorized Spectral-CSD-KV selection over all KV heads.

    This is the same per-head selector as ``_chunkwise_keep_spectral_csd_kv``,
    but batched over the KV-head axis. It keeps the external cache contract
    unchanged: every KV head returns the same number of retained token indices.
    """

    if int(keys.shape[0]) != 1 or int(values.shape[0]) != 1:
        raise NotImplementedError("StrictMerge supports batch_size=1.")
    device = keys.device
    hkv = int(keys.shape[1])
    seq_len = int(keys.shape[2])
    head_dim = int(keys.shape[3])
    selected = force_mask.to(device=device, dtype=torch.bool).view(1, seq_len).expand(hkv, seq_len).clone()
    old_len = max(min(int(old_len), seq_len), 0)
    target = min(max(int(target), int(force_mask.sum().item())), seq_len)
    if old_len <= 0 or int(force_mask.sum().item()) >= int(target):
        keep = torch.nonzero(force_mask.to(device=device, dtype=torch.bool), as_tuple=False).flatten().sort().values
        return keep.view(1, -1).expand(hkv, -1).contiguous()

    if importance is None:
        raise ValueError("importance must be precomputed for all-head spectral selection.")
    importance = importance.to(device=device, dtype=torch.float32).contiguous()
    expected_importance = (hkv, int(old_len))
    if tuple(importance.shape) != expected_importance:
        raise ValueError(f"importance must have shape {expected_importance}, got {tuple(importance.shape)}")
    importance = torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    zero_heads = importance.sum(dim=1, keepdim=True) <= 0.0
    importance = torch.where(zero_heads, torch.ones_like(importance), importance)

    chunk_size = max(int(chunk_size), 2)
    starts = list(range(0, int(old_len), chunk_size))
    block_chunks = int(os.environ.get("BEST_V1_SELECT_CHUNK_BLOCK", "0"))
    if block_chunks > 0 and len(starts) > int(block_chunks):
        return _chunkwise_keep_spectral_csd_kv_all_heads_blocked(
            keys=keys,
            values=values,
            old_len=int(old_len),
            target=int(target),
            force_mask=force_mask,
            chunk_size=int(chunk_size),
            chunk_block_size=int(block_chunks),
            config=config,
            value_metrics=value_metrics,
            key_metrics=key_metrics,
            importance=importance,
            profile=profile,
        )

    stage_t = _profile_start(device, profile)
    head_keys = keys[0, :, :old_len, :].float()
    head_values = values[0, :, :old_len, :].float()
    lengths = [min(chunk_size, int(old_len) - s) for s in starts]
    if not starts:
        keep = torch.nonzero(force_mask.to(device=device, dtype=torch.bool), as_tuple=False).flatten().sort().values
        return keep.view(1, -1).expand(hkv, -1).contiguous()

    num_chunks = len(starts)
    width = max(lengths)
    starts_t = torch.tensor(starts, dtype=torch.long, device=device).view(num_chunks, 1)
    lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
    local_pos = torch.arange(width, dtype=torch.long, device=device).view(1, width)
    valid = local_pos < lengths_t.view(num_chunks, 1)
    valid_h = valid.view(1, num_chunks, width)
    chunk_idx = (starts_t + local_pos).clamp_max(max(int(old_len) - 1, 0))
    flat_idx = chunk_idx.reshape(-1)

    k_chunks = head_keys.index_select(1, flat_idx).reshape(hkv, num_chunks, width, head_dim)
    v_chunks = head_values.index_select(1, flat_idx).reshape(hkv, num_chunks, width, head_dim)
    p_chunks = importance.index_select(1, flat_idx).reshape(hkv, num_chunks, width).masked_fill(~valid_h, 0.0)
    local_selected = selected[:, :old_len].index_select(1, flat_idx).reshape(hkv, num_chunks, width) & valid_h
    _profile_add(profile, "chunk_all_heads_gather_sec", stage_t, device)

    stage_t = _profile_start(device, profile)
    k_features = k_chunks
    if key_metrics is not None:
        metrics = key_metrics.to(device=device, dtype=torch.float32)
        if tuple(metrics.shape) != (hkv, head_dim):
            raise ValueError(f"key_metrics must have shape {(hkv, head_dim)}, got {tuple(metrics.shape)}")
        metrics = torch.nan_to_num(metrics, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
        k_features = k_features * metrics.sqrt().view(hkv, 1, 1, head_dim)
    v_features = v_chunks
    if value_metrics is not None:
        metrics = value_metrics.to(device=device, dtype=torch.float32)
        if tuple(metrics.shape) != (hkv, head_dim):
            raise ValueError(f"value_metrics must have shape {(hkv, head_dim)}, got {tuple(metrics.shape)}")
        metrics = torch.nan_to_num(metrics, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
        v_features = v_features * metrics.sqrt().view(hkv, 1, 1, head_dim)

    key_weight = float(config.coreset_key_weight)
    value_weight = float(config.coreset_value_weight)
    if key_weight <= 0.0 and value_weight <= 0.0:
        raise ValueError("At least one coreset feature weight must be positive.")
    gram_residual = torch.zeros((hkv, num_chunks, width, width), dtype=torch.float32, device=device)
    if key_weight > 0.0:
        key_energy = k_features.square().sum(dim=-1).masked_fill(~valid_h, 0.0)
        key_scale = (key_energy.sum(dim=2) / lengths_t.view(1, num_chunks).float().clamp_min(1.0)).sqrt().clamp_min(1e-6)
        k_normed = (k_features / key_scale.view(hkv, num_chunks, 1, 1)).masked_fill(~valid_h.unsqueeze(-1), 0.0)
        gram_residual = gram_residual + torch.matmul(k_normed, k_normed.transpose(2, 3)) * key_weight * key_weight
    if value_weight > 0.0:
        value_energy = v_features.square().sum(dim=-1).masked_fill(~valid_h, 0.0)
        value_scale = (value_energy.sum(dim=2) / lengths_t.view(1, num_chunks).float().clamp_min(1.0)).sqrt().clamp_min(1e-6)
        v_normed = (v_features / value_scale.view(hkv, num_chunks, 1, 1)).masked_fill(~valid_h.unsqueeze(-1), 0.0)
        gram_residual = gram_residual + torch.matmul(v_normed, v_normed.transpose(2, 3)) * value_weight * value_weight
    p_sqrt = torch.sqrt(p_chunks.clamp_min(0.0) + 1e-12).masked_fill(~valid_h, 0.0)
    gram_residual = gram_residual * p_sqrt.unsqueeze(-1) * p_sqrt.unsqueeze(-2)
    gram_residual = 0.5 * (gram_residual + gram_residual.transpose(2, 3))
    _profile_add(profile, "chunk_all_heads_feature_sec", stage_t, device)

    stage_t = _profile_start(device, profile)
    gram_residual = _project_selected_chunk_gram(gram_residual, local_selected, valid_h)
    _profile_add(profile, "chunk_all_heads_forced_project_sec", stage_t, device)

    recent_forced = int(force_mask[int(old_len) :].sum().item())
    forced_counts = local_selected.sum(dim=2).to(torch.long)
    target_old = min(max(int(target) - int(recent_forced), int(force_mask[:old_len].sum().item())), int(old_len))
    min_keep = max(int(config.operator_min_chunk_keep), 0)
    if min_keep > 0:
        min_counts = torch.minimum(lengths_t.view(1, num_chunks), torch.full_like(forced_counts, min_keep))
        base_counts = torch.maximum(forced_counts, min_counts)
        if int(base_counts[0].sum().item()) <= int(target_old):
            quotas_t = base_counts.clone()
        else:
            quotas_t = forced_counts.clone()
    else:
        quotas_t = forced_counts.clone()

    extra_total = min(
        max(int(target_old) - int(quotas_t[0].sum().item()), 0),
        int((lengths_t - quotas_t[0]).clamp_min(0).sum().item()),
    )
    if extra_total > 0:
        stage_t = _profile_start(device, profile)
        eigvals = _eigvalsh_symmetric_chunked(
            gram_residual,
            max_batch=int(os.environ.get("BEST_V1_SELECT_EIG_BATCH", "512")),
        ).clamp_min(0.0)
        eig_desc = torch.flip(eigvals, dims=[2])
        capacity = (lengths_t.view(1, num_chunks) - quotas_t).clamp_min(0)
        rank_idx = torch.arange(width, dtype=torch.long, device=device).view(1, 1, width)
        eig_scores = eig_desc.masked_fill(rank_idx >= capacity.unsqueeze(-1), float("-inf"))
        top = torch.topk(eig_scores.reshape(hkv, -1), k=int(extra_total), dim=1, largest=True).indices
        chunk_ids = torch.div(top, width, rounding_mode="floor")
        extra_counts = torch.zeros((hkv, num_chunks), dtype=torch.long, device=device)
        extra_counts.scatter_add_(1, chunk_ids, torch.ones_like(chunk_ids, dtype=torch.long))
        quotas_t = quotas_t + extra_counts
        _profile_add(profile, "chunk_all_heads_eig_quota_sec", stage_t, device)

    need = (quotas_t - local_selected.sum(dim=2)).clamp_min(0)
    selected_gain = torch.full((hkv, num_chunks, width), float("-inf"), dtype=torch.float32, device=device)

    stage_t = _profile_start(device, profile)
    max_need = int(need.max().item()) if int(need.numel()) > 0 else 0
    valid_all = valid_h.expand(hkv, num_chunks, width)
    if max_need == 1:
        diag_scores = gram_residual.diagonal(dim1=-2, dim2=-1).clamp_min(0.0)
        candidate_mask = (need > 0).unsqueeze(-1) & valid_all & (~local_selected)
        scores = diag_scores.masked_fill(~candidate_mask, float("-inf"))
        local_idx = torch.argmax(scores, dim=2)
        chosen_score = scores.gather(2, local_idx.unsqueeze(-1)).squeeze(-1)
        chosen = (need > 0) & torch.isfinite(chosen_score)
        if bool(chosen.any().item()):
            head_idx, chunk_ids = torch.nonzero(chosen, as_tuple=True)
            chosen_local = local_idx[head_idx, chunk_ids]
            local_selected[head_idx, chunk_ids, chosen_local] = True
            selected_gain[head_idx, chunk_ids, chosen_local] = chosen_score[head_idx, chunk_ids]
            need[head_idx, chunk_ids] = 0
    elif max_need > 0:
        flat_gram = gram_residual.reshape(hkv * num_chunks, width, width)
        flat_selected = local_selected.reshape(hkv * num_chunks, width)
        flat_valid = valid_all.reshape(hkv * num_chunks, width)
        flat_need = need.reshape(hkv * num_chunks)
        flat_gain = selected_gain.reshape(hkv * num_chunks, width)
        for _ in range(max_need):
            active_rows = torch.nonzero(flat_need > 0, as_tuple=False).flatten()
            if int(active_rows.numel()) == 0:
                break
            row_gram = flat_gram.index_select(0, active_rows)
            row_selected = flat_selected.index_select(0, active_rows)
            row_valid = flat_valid.index_select(0, active_rows)
            scores = row_gram.diagonal(dim1=-2, dim2=-1).clamp_min(0.0).masked_fill(
                ~(row_valid & (~row_selected)),
                float("-inf"),
            )
            local_idx = torch.argmax(scores, dim=1)
            chosen_score = scores.gather(1, local_idx.view(-1, 1)).squeeze(1)
            chosen = torch.isfinite(chosen_score)
            if not bool(chosen.any().item()):
                break
            chosen_rows = torch.nonzero(chosen, as_tuple=False).flatten()
            active_rows = active_rows.index_select(0, chosen_rows)
            local_idx = local_idx.index_select(0, chosen_rows)
            chosen_score = chosen_score.index_select(0, chosen_rows)
            flat_selected[active_rows, local_idx] = True
            flat_gain[active_rows, local_idx] = chosen_score
            flat_need[active_rows] = flat_need[active_rows] - 1
            still_active = flat_need[active_rows] > 0
            if bool(still_active.any().item()):
                still_rows = torch.nonzero(still_active, as_tuple=False).flatten()
                update_rows = active_rows.index_select(0, still_rows)
                update_pivots = local_idx.index_select(0, still_rows)
                update_gram = flat_gram.index_select(0, update_rows)
                pivot_col = update_gram.gather(
                    2,
                    update_pivots.view(-1, 1, 1).expand(-1, width, 1),
                ).squeeze(2)
                denom = pivot_col.gather(1, update_pivots.view(-1, 1)).clamp_min(1e-12)
                update_gram = update_gram - pivot_col.unsqueeze(2) * pivot_col.unsqueeze(1) / denom.view(-1, 1, 1)
                update_gram = 0.5 * (update_gram + update_gram.transpose(1, 2))
                flat_gram.index_copy_(0, update_rows, update_gram)
        local_selected = flat_selected.view(hkv, num_chunks, width)
        need = flat_need.view(hkv, num_chunks)
        selected_gain = flat_gain.view(hkv, num_chunks, width)
    _profile_add(profile, "chunk_all_heads_cpqr_sec", stage_t, device)

    stage_t = _profile_start(device, profile)
    old_flags = local_selected & valid_h
    old_positions = chunk_idx.view(1, num_chunks, width).expand(hkv, num_chunks, width)
    old_selected_i8 = torch.zeros((hkv, int(old_len)), dtype=torch.int8, device=device)
    old_selected_i8.scatter_reduce_(
        1,
        old_positions.reshape(hkv, num_chunks * width),
        old_flags.to(dtype=torch.int8).reshape(hkv, num_chunks * width),
        reduce="amax",
        include_self=True,
    )
    selected[:, :old_len] = selected[:, :old_len] | old_selected_i8.to(dtype=torch.bool)

    counts = selected.sum(dim=1)
    if bool((counts == int(target)).all().item()):
        positions = torch.arange(seq_len, dtype=torch.long, device=device).view(1, seq_len).expand(hkv, seq_len)
        out = positions[selected].view(hkv, int(target)).contiguous()
        _profile_add(profile, "chunk_all_heads_finalize_sec", stage_t, device)
        return out

    if gram_residual is None:
        gram_residual = torch.zeros((hkv, num_chunks, width, width), dtype=torch.float32, device=device)
    residual_scores = gram_residual.diagonal(dim1=-2, dim2=-1).clamp_min(0.0).masked_fill(~valid_h, float("-inf"))
    gain_by_old = torch.full((hkv, int(old_len)), float("-inf"), dtype=torch.float32, device=device)
    selected_gain_by_old = torch.full((hkv, int(old_len)), float("-inf"), dtype=torch.float32, device=device)
    valid_all = valid_h.expand(hkv, -1, -1)
    h_ids, c_ids, w_ids = torch.nonzero(valid_all, as_tuple=True)
    old_pos = chunk_idx[c_ids, w_ids]
    gain_by_old[h_ids, old_pos] = residual_scores[h_ids, c_ids, w_ids]
    selected_gain_by_old[h_ids, old_pos] = selected_gain[h_ids, c_ids, w_ids]

    outputs: list[torch.Tensor] = []
    for head in range(hkv):
        current = int(selected[head].sum().item())
        if current < target:
            need_fill = int(target) - current
            candidates = torch.nonzero(~selected[head, :old_len], as_tuple=False).flatten()
            if int(candidates.numel()) > 0:
                cand_scores = gain_by_old[head].index_select(0, candidates)
                if not bool(torch.isfinite(cand_scores).any().item()):
                    cand_scores = importance[head].index_select(0, candidates)
                take = min(int(need_fill), int(candidates.numel()))
                fill = candidates.index_select(0, torch.topk(cand_scores, k=take, largest=True).indices)
                selected[head].index_fill_(0, fill.to(device=device, dtype=torch.long), True)
        elif current > target:
            excess = current - int(target)
            removable = selected[head, :old_len] & (~force_mask[:old_len].to(device=device, dtype=torch.bool))
            pos = torch.nonzero(removable, as_tuple=False).flatten()
            if int(pos.numel()) > 0:
                gains = selected_gain_by_old[head].index_select(0, pos)
                if not bool(torch.isfinite(gains).any().item()):
                    gains = importance[head].index_select(0, pos)
                drop = pos.index_select(0, torch.argsort(gains)[:excess])
                selected[head].index_fill_(0, drop.to(device=device, dtype=torch.long), False)
        outputs.append(torch.nonzero(selected[head], as_tuple=False).flatten().sort().values)

    keep_len = int(outputs[0].numel()) if outputs else 0
    if any(int(item.numel()) != keep_len for item in outputs):
        raise RuntimeError("batched spectral selector produced ragged head lengths.")
    _profile_add(profile, "chunk_all_heads_finalize_sec", stage_t, device)
    return torch.stack(outputs, dim=0).contiguous()


@torch.no_grad()
def _chunkwise_keep_spectral_csd_kv_all_heads_blocked(
    *,
    keys: torch.Tensor,
    values: torch.Tensor,
    old_len: int,
    target: int,
    force_mask: torch.Tensor,
    chunk_size: int,
    chunk_block_size: int,
    config: StrictMergeConfig,
    value_metrics: torch.Tensor | None = None,
    key_metrics: torch.Tensor | None = None,
    importance: torch.Tensor | None = None,
    profile: dict[str, float] | None = None,
) -> torch.Tensor:
    """Memory-bounded all-head Spectral-CSD-KV selector.

    This keeps the same external selection contract as the vectorized all-head
    path, but streams over contiguous chunk blocks so long prompts do not
    materialize K/V/features/residual for every chunk at once.
    """

    if int(keys.shape[0]) != 1 or int(values.shape[0]) != 1:
        raise NotImplementedError("StrictMerge supports batch_size=1.")
    if importance is None:
        raise ValueError("importance must be precomputed for blocked all-head spectral selection.")

    device = keys.device
    hkv = int(keys.shape[1])
    seq_len = int(keys.shape[2])
    head_dim = int(keys.shape[3])
    old_len = max(min(int(old_len), seq_len), 0)
    target = min(max(int(target), int(force_mask.sum().item())), seq_len)
    force_bool = force_mask.to(device=device, dtype=torch.bool)
    selected = force_bool.view(1, seq_len).expand(hkv, seq_len).clone()
    if old_len <= 0 or int(force_bool.sum().item()) >= int(target):
        keep = torch.nonzero(force_bool, as_tuple=False).flatten().sort().values
        return keep.view(1, -1).expand(hkv, -1).contiguous()

    importance = importance.to(device=device, dtype=torch.float32).contiguous()
    expected_importance = (hkv, int(old_len))
    if tuple(importance.shape) != expected_importance:
        raise ValueError(f"importance must have shape {expected_importance}, got {tuple(importance.shape)}")
    importance = torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    zero_heads = importance.sum(dim=1, keepdim=True) <= 0.0
    importance = torch.where(zero_heads, torch.ones_like(importance), importance)

    chunk_size = max(int(chunk_size), 2)
    starts = list(range(0, int(old_len), chunk_size))
    if not starts:
        keep = torch.nonzero(force_bool, as_tuple=False).flatten().sort().values
        return keep.view(1, -1).expand(hkv, -1).contiguous()
    num_chunks = len(starts)
    width = min(max(int(chunk_size), 1), int(old_len))
    chunk_block_size = max(int(chunk_block_size), 1)
    block_ranges = [(start, min(start + chunk_block_size, num_chunks)) for start in range(0, num_chunks, chunk_block_size)]

    metric_key_sqrt = None
    if key_metrics is not None:
        metrics = key_metrics.to(device=device, dtype=torch.float32)
        if tuple(metrics.shape) != (hkv, head_dim):
            raise ValueError(f"key_metrics must have shape {(hkv, head_dim)}, got {tuple(metrics.shape)}")
        metric_key_sqrt = torch.nan_to_num(metrics, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0).sqrt()
    metric_value_sqrt = None
    if value_metrics is not None:
        metrics = value_metrics.to(device=device, dtype=torch.float32)
        if tuple(metrics.shape) != (hkv, head_dim):
            raise ValueError(f"value_metrics must have shape {(hkv, head_dim)}, got {tuple(metrics.shape)}")
        metric_value_sqrt = torch.nan_to_num(metrics, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0).sqrt()

    key_weight = float(config.coreset_key_weight)
    value_weight = float(config.coreset_value_weight)
    if key_weight <= 0.0 and value_weight <= 0.0:
        raise ValueError("At least one coreset feature weight must be positive.")

    head_keys = keys[0, :, :old_len, :]
    head_values = values[0, :, :old_len, :]
    local_pos_full = torch.arange(width, dtype=torch.long, device=device).view(1, width)
    def build_block(
        block_start: int,
        block_end: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        block_starts = torch.arange(
            int(block_start) * int(chunk_size),
            int(block_end) * int(chunk_size),
            int(chunk_size),
            dtype=torch.long,
            device=device,
        )
        block_lengths = (int(old_len) - block_starts).clamp(min=0, max=int(chunk_size)).to(dtype=torch.long)
        block_width = int(block_lengths.max().item()) if int(block_lengths.numel()) > 0 else 0
        local_pos = local_pos_full[:, :block_width]
        valid = local_pos < block_lengths.view(-1, 1)
        valid_h = valid.view(1, int(block_end) - int(block_start), block_width)
        chunk_idx = (block_starts.view(-1, 1) + local_pos).clamp_max(max(int(old_len) - 1, 0))
        flat_idx = chunk_idx.reshape(-1)
        k_chunks = head_keys.index_select(1, flat_idx).reshape(hkv, int(block_end) - int(block_start), block_width, head_dim).float()
        v_chunks = head_values.index_select(1, flat_idx).reshape(hkv, int(block_end) - int(block_start), block_width, head_dim).float()
        p_chunks = importance.index_select(1, flat_idx).reshape(hkv, int(block_end) - int(block_start), block_width).masked_fill(~valid_h, 0.0)
        local_selected = selected[:, :old_len].index_select(1, flat_idx).reshape(hkv, int(block_end) - int(block_start), block_width) & valid_h
        return k_chunks, v_chunks, p_chunks, local_selected, valid_h, chunk_idx

    def build_residual(
        k_chunks: torch.Tensor,
        v_chunks: torch.Tensor,
        p_chunks: torch.Tensor,
        local_selected: torch.Tensor,
        valid_h: torch.Tensor,
        block_lengths: torch.Tensor,
    ) -> torch.Tensor:
        k_features = k_chunks
        if metric_key_sqrt is not None:
            k_features = k_features * metric_key_sqrt.view(hkv, 1, 1, head_dim)
        v_features = v_chunks
        if metric_value_sqrt is not None:
            v_features = v_features * metric_value_sqrt.view(hkv, 1, 1, head_dim)

        parts: list[torch.Tensor] = []
        block_chunks = int(k_chunks.shape[1])
        if key_weight > 0.0:
            key_energy = k_features.square().sum(dim=-1).masked_fill(~valid_h, 0.0)
            key_scale = (
                key_energy.sum(dim=2) / block_lengths.view(1, block_chunks).float().clamp_min(1.0)
            ).sqrt().clamp_min(1e-6)
            parts.append((k_features / key_scale.view(hkv, block_chunks, 1, 1)) * key_weight)
        if value_weight > 0.0:
            value_energy = v_features.square().sum(dim=-1).masked_fill(~valid_h, 0.0)
            value_scale = (
                value_energy.sum(dim=2) / block_lengths.view(1, block_chunks).float().clamp_min(1.0)
            ).sqrt().clamp_min(1e-6)
            parts.append((v_features / value_scale.view(hkv, block_chunks, 1, 1)) * value_weight)
        features = torch.cat(parts, dim=-1).masked_fill(~valid_h.unsqueeze(-1), 0.0)
        z = features * torch.sqrt(p_chunks.clamp_min(0.0) + 1e-12).unsqueeze(-1)
        z = z.masked_fill(~valid_h.unsqueeze(-1), 0.0)
        return _project_selected_chunk_columns(z, local_selected, valid_h)

    stage_t = _profile_start(device, profile)
    forced_counts = torch.empty((hkv, num_chunks), dtype=torch.long, device=device)
    lengths_t = torch.empty((num_chunks,), dtype=torch.long, device=device)
    eig_scores_all = torch.full((hkv, num_chunks, width), float("-inf"), dtype=torch.float32, device=device)
    for block_start, block_end in block_ranges:
        block_starts = torch.arange(
            int(block_start) * int(chunk_size),
            int(block_end) * int(chunk_size),
            int(chunk_size),
            dtype=torch.long,
            device=device,
        )
        block_lengths = (int(old_len) - block_starts).clamp(min=0, max=int(chunk_size)).to(dtype=torch.long)
        lengths_t[int(block_start) : int(block_end)] = block_lengths
        k_chunks, v_chunks, p_chunks, local_selected, valid_h, _chunk_idx = build_block(block_start, block_end)
        forced_counts[:, int(block_start) : int(block_end)] = local_selected.sum(dim=2).to(torch.long)
        residual = build_residual(k_chunks, v_chunks, p_chunks, local_selected, valid_h, block_lengths)
        candidate_mask = valid_h & (~local_selected)
        residual = residual.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
        gram = torch.matmul(residual, residual.transpose(2, 3))
        eigvals = _eigvalsh_symmetric_chunked(
            gram,
            max_batch=int(os.environ.get("BEST_V1_SELECT_EIG_BATCH", "512")),
        ).clamp_min(0.0)
        eig_desc = torch.flip(eigvals, dims=[2])
        block_width = int(eig_desc.shape[-1])
        eig_scores_all[:, int(block_start) : int(block_end), :block_width] = eig_desc
    _profile_add(profile, "chunk_all_heads_blocked_eig_stream_sec", stage_t, device)

    recent_forced = int(force_bool[int(old_len) :].sum().item())
    target_old = min(max(int(target) - int(recent_forced), int(force_bool[:old_len].sum().item())), int(old_len))
    min_keep = max(int(config.operator_min_chunk_keep), 0)
    if min_keep > 0:
        min_counts = torch.minimum(lengths_t.view(1, num_chunks), torch.full_like(forced_counts, min_keep))
        base_counts = torch.maximum(forced_counts, min_counts)
        if int(base_counts[0].sum().item()) <= int(target_old):
            quotas_t = base_counts.clone()
        else:
            quotas_t = forced_counts.clone()
    else:
        quotas_t = forced_counts.clone()

    extra_total = min(
        max(int(target_old) - int(quotas_t[0].sum().item()), 0),
        int((lengths_t - quotas_t[0]).clamp_min(0).sum().item()),
    )
    stage_t = _profile_start(device, profile)
    if extra_total > 0:
        capacity = (lengths_t.view(1, num_chunks) - quotas_t).clamp_min(0)
        rank_idx = torch.arange(width, dtype=torch.long, device=device).view(1, 1, width)
        eig_scores = eig_scores_all.masked_fill(rank_idx >= capacity.unsqueeze(-1), float("-inf"))
        top = torch.topk(eig_scores.reshape(hkv, -1), k=int(extra_total), dim=1, largest=True).indices
        chunk_ids = torch.div(top, width, rounding_mode="floor")
        extra_counts = torch.zeros((hkv, num_chunks), dtype=torch.long, device=device)
        extra_counts.scatter_add_(1, chunk_ids, torch.ones_like(chunk_ids, dtype=torch.long))
        quotas_t = quotas_t + extra_counts
    del eig_scores_all
    _profile_add(profile, "chunk_all_heads_eig_quota_sec", stage_t, device)

    stage_t = _profile_start(device, profile)
    selected_gain_by_old = torch.full((hkv, int(old_len)), float("-inf"), dtype=torch.float32, device=device)
    gain_by_old = torch.full((hkv, int(old_len)), float("-inf"), dtype=torch.float32, device=device)
    for block_start, block_end in block_ranges:
        block_starts = torch.arange(
            int(block_start) * int(chunk_size),
            int(block_end) * int(chunk_size),
            int(chunk_size),
            dtype=torch.long,
            device=device,
        )
        block_lengths = (int(old_len) - block_starts).clamp(min=0, max=int(chunk_size)).to(dtype=torch.long)
        block_chunks = int(block_end) - int(block_start)
        k_chunks, v_chunks, p_chunks, local_selected, valid_h, chunk_idx = build_block(block_start, block_end)
        residual = build_residual(k_chunks, v_chunks, p_chunks, local_selected, valid_h, block_lengths)
        need = (quotas_t[:, int(block_start) : int(block_end)] - local_selected.sum(dim=2)).clamp_min(0)
        selected_gain = torch.full((hkv, block_chunks, int(chunk_idx.shape[1])), float("-inf"), dtype=torch.float32, device=device)
        max_need = int(need.max().item()) if int(need.numel()) > 0 else 0
        valid_all = valid_h.expand(hkv, block_chunks, int(chunk_idx.shape[1]))
        residual_feature_dim = int(residual.shape[-1])
        for _ in range(max_need):
            active_rows = torch.nonzero(need.reshape(-1) > 0, as_tuple=False).flatten()
            if int(active_rows.numel()) == 0:
                break
            row_residual = residual.reshape(hkv * block_chunks, int(chunk_idx.shape[1]), residual_feature_dim).index_select(0, active_rows)
            row_selected = local_selected.reshape(hkv * block_chunks, int(chunk_idx.shape[1])).index_select(0, active_rows)
            row_valid = valid_all.reshape(hkv * block_chunks, int(chunk_idx.shape[1])).index_select(0, active_rows)
            scores = row_residual.square().sum(dim=-1).masked_fill(~(row_valid & (~row_selected)), float("-inf"))
            local_idx = torch.argmax(scores, dim=1)
            chosen_score = scores.gather(1, local_idx.view(-1, 1)).squeeze(1)
            chosen = torch.isfinite(chosen_score)
            if not bool(chosen.any().item()):
                break
            chosen_rows = torch.nonzero(chosen, as_tuple=False).flatten()
            active_rows = active_rows.index_select(0, chosen_rows)
            local_idx = local_idx.index_select(0, chosen_rows)
            chosen_score = chosen_score.index_select(0, chosen_rows)
            head_idx = torch.div(active_rows, block_chunks, rounding_mode="floor")
            local_chunk_ids = active_rows - head_idx * int(block_chunks)
            local_selected[head_idx, local_chunk_ids, local_idx] = True
            selected_gain[head_idx, local_chunk_ids, local_idx] = chosen_score
            need[head_idx, local_chunk_ids] = need[head_idx, local_chunk_ids] - 1
            still_active = need[head_idx, local_chunk_ids] > 0
            if bool(still_active.any().item()):
                still_rows = torch.nonzero(still_active, as_tuple=False).flatten()
                residual = _project_pivot_flat_chunk_rows(
                    residual,
                    active_rows.index_select(0, still_rows),
                    local_idx.index_select(0, still_rows),
                )

        old_flags = local_selected & valid_h
        old_positions = chunk_idx.view(1, block_chunks, int(chunk_idx.shape[1])).expand(hkv, block_chunks, int(chunk_idx.shape[1]))
        old_selected_i8 = torch.zeros((hkv, int(old_len)), dtype=torch.int8, device=device)
        old_selected_i8.scatter_reduce_(
            1,
            old_positions.reshape(hkv, block_chunks * int(chunk_idx.shape[1])),
            old_flags.to(dtype=torch.int8).reshape(hkv, block_chunks * int(chunk_idx.shape[1])),
            reduce="amax",
            include_self=True,
        )
        selected[:, :old_len] = selected[:, :old_len] | old_selected_i8.to(dtype=torch.bool)
        residual_scores = residual.square().sum(dim=-1).masked_fill(~valid_h, float("-inf"))
        valid_all = valid_h.expand(hkv, -1, -1)
        h_ids, c_ids, w_ids = torch.nonzero(valid_all, as_tuple=True)
        old_pos = chunk_idx[c_ids, w_ids]
        gain_by_old[h_ids, old_pos] = residual_scores[h_ids, c_ids, w_ids]
        selected_gain_by_old[h_ids, old_pos] = selected_gain[h_ids, c_ids, w_ids]
    _profile_add(profile, "chunk_all_heads_cpqr_sec", stage_t, device)

    stage_t = _profile_start(device, profile)
    outputs: list[torch.Tensor] = []
    for head in range(hkv):
        current = int(selected[head].sum().item())
        if current < target:
            need_fill = int(target) - current
            candidates = torch.nonzero(~selected[head, :old_len], as_tuple=False).flatten()
            if int(candidates.numel()) > 0:
                cand_scores = gain_by_old[head].index_select(0, candidates)
                if not bool(torch.isfinite(cand_scores).any().item()):
                    cand_scores = importance[head].index_select(0, candidates)
                take = min(int(need_fill), int(candidates.numel()))
                fill = candidates.index_select(0, torch.topk(cand_scores, k=take, largest=True).indices)
                selected[head].index_fill_(0, fill.to(device=device, dtype=torch.long), True)
        elif current > target:
            excess = current - int(target)
            removable = selected[head, :old_len] & (~force_bool[:old_len])
            pos = torch.nonzero(removable, as_tuple=False).flatten()
            if int(pos.numel()) > 0:
                gains = selected_gain_by_old[head].index_select(0, pos)
                if not bool(torch.isfinite(gains).any().item()):
                    gains = importance[head].index_select(0, pos)
                drop = pos.index_select(0, torch.argsort(gains)[:excess])
                selected[head].index_fill_(0, drop.to(device=device, dtype=torch.long), False)
        outputs.append(torch.nonzero(selected[head], as_tuple=False).flatten().sort().values)

    keep_len = int(outputs[0].numel()) if outputs else 0
    if any(int(item.numel()) != keep_len for item in outputs):
        raise RuntimeError("blocked spectral selector produced ragged head lengths.")
    _profile_add(profile, "chunk_all_heads_finalize_sec", stage_t, device)
    return torch.stack(outputs, dim=0).contiguous()


@torch.no_grad()
def _spectral_csd_bias_and_merged_values(
    *,
    keys: torch.Tensor,
    values: torch.Tensor,
    q_obs: torch.Tensor,
    kv_head: int,
    keep_idx: torch.Tensor,
    old_len: int,
    chunk_size: int,
    num_key_value_groups: int,
    config: StrictMergeConfig,
    value_metric: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return value-aggregated retained values and additive log attention bias."""

    device = keys.device
    head_dim = int(keys.shape[-1])
    keep_idx = keep_idx.to(device=device, dtype=torch.long)
    kept_len = int(keep_idx.numel())
    raw_values = values[0, int(kv_head)].float()
    merged_values = raw_values.index_select(0, keep_idx).clone()
    log_bias = torch.zeros((kept_len,), dtype=torch.float32, device=device)
    old_len = max(min(int(old_len), int(raw_values.shape[0])), 0)
    if old_len <= 0 or kept_len <= 0:
        return merged_values.to(dtype=values.dtype), log_bias

    groups = max(int(num_key_value_groups), 1)
    q_start = int(kv_head) * groups
    q_end = q_start + groups
    head_keys_full = keys[0, int(kv_head), :, :].float()
    head_keys = head_keys_full[:old_len]
    head_values = raw_values[:old_len]

    q = q_obs[q_start:q_end].float().reshape(-1, head_dim)
    if int(q.numel()) > 0:
        logits = (q @ head_keys_full.T) / (float(head_dim) ** 0.5)
        attn = torch.softmax(logits, dim=-1, dtype=torch.float32)[:, :old_len]
        importance = 0.5 * attn.mean(dim=0) + 0.5 * attn.amax(dim=0)
    else:
        importance = torch.ones((old_len,), dtype=torch.float32, device=device)
    importance = torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    kernel = int(config.snap_kernel_size)
    if kernel > 1:
        pooled = importance.view(1, 1, old_len)
        if str(config.snap_pooling) == "maxpool":
            importance = F.max_pool1d(pooled, kernel_size=kernel, padding=kernel // 2, stride=1).view(-1)[:old_len]
        else:
            importance = F.avg_pool1d(pooled, kernel_size=kernel, padding=kernel // 2, stride=1).view(-1)[:old_len]
    if float(importance.sum().item()) <= 0.0:
        importance = torch.ones_like(importance)

    keep_pos = torch.full((int(raw_values.shape[0]),), -1, dtype=torch.long, device=device)
    keep_pos.index_copy_(0, keep_idx, torch.arange(kept_len, dtype=torch.long, device=device))
    old_keep_pos = keep_pos[:old_len]
    if not bool((old_keep_pos >= 0).any().item()):
        return merged_values.to(dtype=values.dtype), log_bias

    chunk_size = max(int(chunk_size), 2)
    starts = list(range(0, int(old_len), chunk_size))
    lengths = [min(chunk_size, int(old_len) - s) for s in starts]
    num_chunks = len(starts)
    width = max(lengths)
    starts_t = torch.tensor(starts, dtype=torch.long, device=device).view(num_chunks, 1)
    lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
    local_pos = torch.arange(width, dtype=torch.long, device=device).view(1, width)
    valid = local_pos < lengths_t.view(num_chunks, 1)
    chunk_idx = (starts_t + local_pos).clamp_max(max(int(old_len) - 1, 0))
    flat_idx = chunk_idx.reshape(-1)

    k_chunks = head_keys.index_select(0, flat_idx).reshape(num_chunks, width, head_dim)
    v_features = head_values.index_select(0, flat_idx).reshape(num_chunks, width, head_dim)
    if value_metric is not None:
        metric = value_metric.to(device=device, dtype=torch.float32).flatten()
        if int(metric.numel()) != int(head_dim):
            raise ValueError(f"value_metric must have length {head_dim}, got {int(metric.numel())}")
        metric = torch.nan_to_num(metric, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
        v_features = v_features * metric.sqrt().view(1, 1, head_dim)

    parts: list[torch.Tensor] = []
    if float(config.coreset_key_weight) > 0.0:
        key_energy = k_chunks.square().sum(dim=-1).masked_fill(~valid, 0.0)
        key_scale = (key_energy.sum(dim=1) / lengths_t.float().clamp_min(1.0)).sqrt().clamp_min(1e-6)
        parts.append((k_chunks / key_scale.view(num_chunks, 1, 1)) * float(config.coreset_key_weight))
    if float(config.coreset_value_weight) > 0.0:
        value_energy = v_features.square().sum(dim=-1).masked_fill(~valid, 0.0)
        value_scale = (value_energy.sum(dim=1) / lengths_t.float().clamp_min(1.0)).sqrt().clamp_min(1e-6)
        parts.append((v_features / value_scale.view(num_chunks, 1, 1)) * float(config.coreset_value_weight))
    if not parts:
        return merged_values.to(dtype=values.dtype), log_bias
    features = F.normalize(torch.cat(parts, dim=-1).masked_fill(~valid.unsqueeze(-1), 0.0), p=2, dim=-1, eps=1e-6)

    local_keep_pos = keep_pos.index_select(0, flat_idx).reshape(num_chunks, width)
    retained = (local_keep_pos >= 0) & valid
    has_retained = retained.any(dim=1)
    sim = torch.matmul(features, features.transpose(1, 2))
    scores = sim.masked_fill(~retained[:, None, :], float("-inf"))
    assign_local = torch.argmax(scores, dim=2)
    assign_local = torch.where(retained, local_pos.expand(num_chunks, width), assign_local)
    valid_assign = valid & has_retained.view(num_chunks, 1)
    assigned_pos = local_keep_pos.gather(1, assign_local).masked_fill(~valid_assign, -1)

    flat_assigned = assigned_pos.reshape(-1)
    assign_mask = flat_assigned >= 0
    if not bool(assign_mask.any().item()):
        return merged_values.to(dtype=values.dtype), log_bias
    assigned_comp = flat_assigned[assign_mask]
    assigned_old = flat_idx[assign_mask]

    p = importance.index_select(0, assigned_old).clamp_min(0.0)
    value_w = p + 1e-12
    mass = torch.zeros((kept_len,), dtype=torch.float32, device=device)
    value_mass = torch.zeros((kept_len,), dtype=torch.float32, device=device)
    value_sum = torch.zeros((kept_len, head_dim), dtype=torch.float32, device=device)
    mass.scatter_add_(0, assigned_comp, p)
    value_mass.scatter_add_(0, assigned_comp, value_w)
    value_sum.scatter_add_(0, assigned_comp.view(-1, 1).expand(-1, head_dim), head_values.index_select(0, assigned_old) * value_w.view(-1, 1))

    old_retained = keep_idx < old_len
    old_keep = keep_idx[old_retained]
    old_pos = torch.nonzero(old_retained, as_tuple=False).flatten()
    own_mass = torch.zeros((kept_len,), dtype=torch.float32, device=device)
    if int(old_keep.numel()) > 0:
        own_mass.index_copy_(0, old_pos, importance.index_select(0, old_keep).clamp_min(0.0))

    has_value = value_mass > 0.0
    merged_values[has_value] = value_sum[has_value] / value_mass[has_value].view(-1, 1).clamp_min(1e-12)
    c = (mass + 1e-12) / own_mass.clamp_min(1e-12)
    c = c.clamp_min(1.0).clamp_max(8.0)
    log_bias = torch.log(c).masked_fill(~old_retained, 0.0)
    return merged_values.to(dtype=values.dtype), log_bias


@torch.no_grad()
def _global_keep_grop_kv(
    *,
    keys: torch.Tensor,
    values: torch.Tensor,
    q_obs: torch.Tensor,
    q_all: torch.Tensor | None,
    kv_head: int,
    old_len: int,
    target: int,
    force_mask: torch.Tensor,
    num_key_value_groups: int,
    config: StrictMergeConfig,
    value_metric: torch.Tensor | None = None,
) -> torch.Tensor:
    """Global Renormalized Operator Pruning for one KV head.

    GROP starts from the full cache and deletes old, non-forced tokens until the
    target budget is reached. The score for a deletion is the full-renormalized
    attention readout error under local, tail, and key-direction probe groups;
    groups are normalized independently and combined by minimax.
    """

    seq_len = int(force_mask.numel())
    old_len = max(min(int(old_len), seq_len), 0)
    target = min(max(int(target), int(force_mask.sum().item()), 1), seq_len)
    selected = torch.ones((seq_len,), dtype=torch.bool, device=keys.device)
    if seq_len <= 0 or int(target) >= seq_len:
        return torch.arange(seq_len, dtype=torch.long, device=keys.device)

    device = keys.device
    head_dim = int(keys.shape[-1])
    groups = max(int(num_key_value_groups), 1)
    q_start = int(kv_head) * groups
    q_end = q_start + groups
    head_keys = keys[0, int(kv_head), :, :].float()
    head_values = values[0, int(kv_head), :, :].float()
    if value_metric is not None:
        metric = value_metric.to(device=device, dtype=torch.float32).flatten()
        if int(metric.numel()) != int(head_dim):
            raise ValueError(f"value_metric must have length {head_dim}, got {int(metric.numel())}")
        metric = torch.nan_to_num(metric, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
        head_values = head_values * metric.sqrt().view(1, head_dim)

    scale = 1.0 / (float(head_dim) ** 0.5)
    token_pos = torch.arange(seq_len, dtype=torch.long, device=device)
    eps = float(config.denom_eps)
    probe_groups: list[dict[str, Any]] = []

    def add_probe_group(name: str, probes: torch.Tensor, mask: torch.Tensor) -> None:
        if int(probes.numel()) == 0:
            return
        probes_f = probes.to(device=device, dtype=torch.float32).reshape(-1, head_dim)
        mask_b = mask.to(device=device, dtype=torch.bool).reshape(int(probes_f.shape[0]), seq_len)
        if int(probes_f.shape[0]) <= 0:
            return
        logits = (probes_f @ head_keys.T) * scale
        logits = logits.masked_fill(~mask_b, -1.0e30)
        row_max = logits.amax(dim=-1, keepdim=True)
        beta = torch.exp(logits - row_max).masked_fill(~mask_b, 0.0)
        z = beta.sum(dim=-1)
        valid = z > 1e-12
        if not bool(valid.any().item()):
            return
        beta = beta.index_select(0, torch.nonzero(valid, as_tuple=False).flatten()).contiguous()
        z = z.index_select(0, torch.nonzero(valid, as_tuple=False).flatten()).contiguous()
        target_out = (beta @ head_values) / z.unsqueeze(-1)
        target_norm = target_out.square().sum().clamp_min(eps)
        probe_groups.append(
            {
                "name": str(name),
                "beta": beta,
                "target": target_out.contiguous(),
                "target_norm2": target_out.square().sum(dim=-1).contiguous(),
                "norm": target_norm,
                "a": z.clone().contiguous(),
                "b": (beta @ head_values).contiguous(),
            }
        )

    probe_source = str(config.operator_probe_source)
    include_local = probe_source in {"local_plus_tail", "local_only"}
    include_tail = probe_source in {"local_plus_tail", "tail_only"}
    real_limit = int(config.operator_real_probe_limit)

    if include_local and q_all is not None:
        q_all_head = q_all[q_start:q_end, :seq_len, :].float()
        local_total = int(groups) * int(seq_len)
        local_flat = _probe_limit_indices(local_total, real_limit, device=device)
        if int(local_flat.numel()) > 0:
            local_group = torch.div(local_flat, seq_len, rounding_mode="floor")
            local_pos = local_flat - local_group * int(seq_len)
            local_q = q_all_head[local_group, local_pos, :]
            local_mask = token_pos.view(1, seq_len) <= local_pos.view(-1, 1)
            add_probe_group("local", local_q, local_mask)

    if include_tail:
        q_tail = q_obs[q_start:q_end].float().reshape(-1, head_dim)
        tail_idx = _probe_limit_indices(int(q_tail.shape[0]), real_limit, device=device)
        if int(tail_idx.numel()) > 0:
            q_tail = q_tail.index_select(0, tail_idx)
            tail_mask = torch.ones((int(q_tail.shape[0]), seq_len), dtype=torch.bool, device=device)
            add_probe_group("tail", q_tail, tail_mask)

    key_limit = int(config.operator_key_probe_limit)
    if old_len > 0 and key_limit >= 0:
        key_idx = _probe_limit_indices(int(old_len), key_limit, device=device)
        if int(key_idx.numel()) > 0:
            q_norm_source = q_obs[q_start:q_end].float().reshape(-1, head_dim)
            if int(q_norm_source.numel()) > 0:
                gamma = q_norm_source.norm(dim=-1).median().clamp_min(1e-6)
            else:
                gamma = head_keys[:old_len].norm(dim=-1).median().clamp_min(1e-6)
            q_key = F.normalize(head_keys.index_select(0, key_idx), p=2, dim=-1, eps=1e-6) * gamma
            key_mask = torch.ones((int(q_key.shape[0]), seq_len), dtype=torch.bool, device=device)
            add_probe_group("key", q_key, key_mask)

    if not probe_groups:
        return _chunkwise_keep_anchor_then_cover(
            keys=keys,
            values=values,
            kv_head=int(kv_head),
            old_len=int(old_len),
            target=int(target),
            force_mask=force_mask,
            chunk_size=int(config.local_chunk_size),
            config=config,
            value_metric=value_metric,
        )

    block_size = max(int(config.score_block_size), 1)
    max_batch = max(int(config.batch_drop), 1)
    protected = force_mask.to(device=device, dtype=torch.bool)
    min_chunk_keep = max(int(config.operator_min_chunk_keep), 0)
    chunk_ids = torch.full((seq_len,), -1, dtype=torch.long, device=device)
    chunk_counts: torch.Tensor | None = None
    if min_chunk_keep > 0 and old_len > 0:
        chunk_size = max(int(config.local_chunk_size), 2)
        num_chunks = (int(old_len) + chunk_size - 1) // chunk_size
        old_chunk_ids = torch.div(torch.arange(old_len, dtype=torch.long, device=device), chunk_size, rounding_mode="floor")
        chunk_ids[:old_len] = old_chunk_ids
        lengths = torch.bincount(old_chunk_ids, minlength=num_chunks).to(dtype=torch.long)
        min_old = int(torch.minimum(lengths, torch.full_like(lengths, min_chunk_keep)).sum().item())
        recent_forced = int(protected[old_len:].sum().item())
        target = min(max(int(target), min_old + recent_forced), seq_len)
        chunk_counts = lengths.clone()

    while int(selected.sum().item()) > int(target):
        removable = selected & (~protected)
        if chunk_counts is not None:
            old_removable = removable[:old_len]
            can_remove_old = chunk_counts.index_select(0, chunk_ids[:old_len]).gt(int(min_chunk_keep))
            removable[:old_len] = old_removable & can_remove_old
        candidate_pos = torch.nonzero(removable, as_tuple=False).flatten()
        remaining_delete = int(selected.sum().item()) - int(target)
        if int(candidate_pos.numel()) <= 0 or remaining_delete <= 0:
            break
        if int(candidate_pos.numel()) <= remaining_delete:
            selected.index_fill_(0, candidate_pos.to(dtype=torch.long), False)
            break

        all_scores = torch.empty((int(candidate_pos.numel()),), dtype=torch.float32, device=device)
        for start in range(0, int(candidate_pos.numel()), block_size):
            end = min(start + block_size, int(candidate_pos.numel()))
            block = candidate_pos[start:end]
            u_block = head_values.index_select(0, block)
            u_norm2 = u_block.square().sum(dim=-1)
            block_score = torch.full((int(block.numel()),), float("-inf"), dtype=torch.float32, device=device)

            for group in probe_groups:
                beta = group["beta"]
                beta_block = beta.index_select(1, block)
                a = group["a"]
                b = group["b"]
                target_out = group["target"]
                target_norm2 = group["target_norm2"]

                z = a[:, None] - beta_block
                invalid = z <= 1e-12
                z = z.clamp_min(1e-12)
                target_dot_b = (target_out * b).sum(dim=-1)
                b_norm2 = b.square().sum(dim=-1)
                target_dot_u = target_out @ u_block.T
                b_dot_u = b @ u_block.T
                target_dot_n = target_dot_b[:, None] - beta_block * target_dot_u
                n_norm2 = b_norm2[:, None] - 2.0 * beta_block * b_dot_u + beta_block.square() * u_norm2[None, :]
                loss = target_norm2[:, None] - 2.0 * target_dot_n / z + n_norm2 / z.square()
                loss = loss.clamp_min(0.0).masked_fill(invalid, float("inf"))
                group_score = loss.sum(dim=0) / group["norm"]
                block_score = torch.maximum(block_score, group_score)

            all_scores[start:end] = block_score

        finite = torch.isfinite(all_scores)
        if not bool(finite.any().item()):
            fallback = candidate_pos[:remaining_delete]
            selected.index_fill_(0, fallback.to(dtype=torch.long), False)
            break
        if not bool(finite.all().item()):
            all_scores = all_scores.masked_fill(~finite, float("inf"))
        batch_cap = min(max_batch, max(1, (int(candidate_pos.numel()) + 19) // 20))
        drop_count = min(int(remaining_delete), int(batch_cap))
        drop_local = torch.topk(all_scores, k=drop_count, largest=False).indices
        drop_idx = candidate_pos.index_select(0, drop_local)
        selected.index_fill_(0, drop_idx.to(dtype=torch.long), False)
        if chunk_counts is not None:
            old_drop = drop_idx[drop_idx < old_len]
            if int(old_drop.numel()) > 0:
                chunk_counts = chunk_counts - torch.bincount(
                    chunk_ids.index_select(0, old_drop),
                    minlength=int(chunk_counts.numel()),
                ).to(dtype=torch.long)

        u_drop = head_values.index_select(0, drop_idx)
        for group in probe_groups:
            beta_drop = group["beta"].index_select(1, drop_idx)
            group["a"] = (group["a"] - beta_drop.sum(dim=1)).clamp_min(1e-12)
            group["b"] = group["b"] - beta_drop @ u_drop

    current = int(selected.sum().item())
    if current > int(target):
        excess = current - int(target)
        removable = selected & (~protected)
        if chunk_counts is not None:
            old_removable = removable[:old_len]
            can_remove_old = chunk_counts.index_select(0, chunk_ids[:old_len]).gt(int(min_chunk_keep))
            removable[:old_len] = old_removable & can_remove_old
        pos = torch.nonzero(removable, as_tuple=False).flatten()
        if int(pos.numel()) > 0:
            selected.index_fill_(0, pos[:excess].to(dtype=torch.long), False)
    elif current < int(target):
        need = int(target) - current
        candidates = torch.nonzero(~selected, as_tuple=False).flatten()
        take = min(int(need), int(candidates.numel()))
        if take > 0:
            selected.index_fill_(0, candidates[:take].to(dtype=torch.long), True)

    return torch.nonzero(selected, as_tuple=False).flatten().sort().values


def _chunkwise_keep_anchor_then_cover(
    *,
    keys: torch.Tensor,
    values: torch.Tensor,
    kv_head: int,
    old_len: int,
    target: int,
    force_mask: torch.Tensor,
    chunk_size: int,
    config: StrictMergeConfig,
    value_metric: torch.Tensor | None = None,
    importance_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select each old-token chunk by gist anchor followed by facility gain."""

    selected = force_mask.clone()
    seq_len = int(selected.numel())
    old_len = max(min(int(old_len), seq_len), 0)
    target = min(max(int(target), int(selected.sum().item())), seq_len)
    if old_len <= 0 or int(selected.sum().item()) >= target:
        return torch.nonzero(selected, as_tuple=False).flatten().sort().values

    starts, lengths, quotas = _chunkwise_quota_plan(
        selected=selected,
        old_len=int(old_len),
        target=int(target),
        chunk_size=int(chunk_size),
    )
    if not starts:
        return torch.nonzero(selected, as_tuple=False).flatten().sort().values

    num_chunks = len(starts)
    width = max(lengths)
    device = keys.device
    starts_t = torch.tensor(starts, dtype=torch.long, device=device).view(num_chunks, 1)
    lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
    quotas_t = torch.tensor(quotas, dtype=torch.long, device=device)
    local_pos = torch.arange(width, dtype=torch.long, device=device).view(1, width)
    valid = local_pos < lengths_t.view(num_chunks, 1)
    chunk_idx = (starts_t + local_pos).clamp_max(max(int(old_len) - 1, 0))

    flat_idx = chunk_idx.reshape(-1)
    head_keys = keys[0, int(kv_head), :old_len, :].float()
    head_values = values[0, int(kv_head), :old_len, :].float()
    k_chunks = head_keys.index_select(0, flat_idx).reshape(num_chunks, width, -1)
    v_chunks = head_values.index_select(0, flat_idx).reshape(num_chunks, width, -1)
    local_selected = selected.index_select(0, flat_idx).reshape(num_chunks, width) & valid
    if importance_scores is None:
        importance = torch.ones((num_chunks, width), dtype=torch.float32, device=device)
    else:
        head_importance = importance_scores.to(device=device, dtype=torch.float32).flatten()
        if int(head_importance.numel()) < int(old_len):
            raise ValueError(f"importance_scores must cover old_len={int(old_len)}, got {int(head_importance.numel())}")
        importance = head_importance[:old_len].index_select(0, flat_idx).reshape(num_chunks, width)
        importance = torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        importance = importance.masked_fill(~valid, 0.0)
        # Local scores differ substantially across layers/heads. Use relative
        # chunk weights and clip them so intrinsic importance guides facility
        # without destroying coverage.
        imp_mean = importance.sum(dim=1, keepdim=True) / lengths_t.float().clamp_min(1.0).view(num_chunks, 1)
        importance = (importance / imp_mean.clamp_min(1e-12)).clamp(0.25, 4.0)
    importance = importance.masked_fill(~valid, 0.0)

    parts: list[torch.Tensor] = []
    key_weight = float(config.coreset_key_weight)
    value_weight = float(config.coreset_value_weight)
    if key_weight > 0.0:
        parts.append(F.normalize(k_chunks, p=2, dim=-1, eps=1e-6) * key_weight)
    if value_weight > 0.0:
        v_features = v_chunks
        if value_metric is not None:
            metric = value_metric.to(device=device, dtype=torch.float32).flatten()
            if int(metric.numel()) != int(v_chunks.shape[-1]):
                raise ValueError(
                    f"value_metric must have length {int(v_chunks.shape[-1])}, got {int(metric.numel())}"
                )
            metric = torch.nan_to_num(metric, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
            v_features = v_features * metric.sqrt().view(1, 1, -1)
        parts.append(F.normalize(v_features, p=2, dim=-1, eps=1e-6) * value_weight)
    if not parts:
        raise ValueError("At least one coreset feature weight must be positive.")
    features = torch.cat(parts, dim=-1)
    features = features.masked_fill(~valid.unsqueeze(-1), 0.0)

    x_sq = features.square().sum(dim=-1)
    dist_sq = (x_sq[:, :, None] + x_sq[:, None, :] - 2.0 * torch.matmul(features, features.transpose(1, 2))).clamp_min(0.0)
    valid_pair = valid[:, :, None] & valid[:, None, :]
    eye = torch.eye(width, dtype=torch.bool, device=device).view(1, width, width)
    offdiag = valid_pair & (~eye)
    scale_sq = (dist_sq * offdiag.float()).sum(dim=(1, 2)) / offdiag.sum(dim=(1, 2)).clamp_min(1).float()
    scale_sq = scale_sq * float(config.coreset_tau) * float(config.coreset_tau)
    scale_sq = torch.where(scale_sq > 1e-6, scale_sq, torch.ones_like(scale_sq))
    sim = torch.exp(-dist_sq / scale_sq.view(num_chunks, 1, 1).clamp_min(1e-6))
    sim = sim.masked_fill(~valid_pair, 0.0)

    knn = int(config.coreset_knn)
    if 0 < knn < width:
        top_k = min(max(knn, 1), width)
        sparse = torch.zeros_like(sim, dtype=torch.bool)
        sparse.scatter_(2, torch.topk(sim, k=top_k, dim=2, largest=True).indices, True)
        sim = sim.masked_fill(~(sparse & valid_pair), 0.0)

    row_importance = importance
    candidate_importance = importance
    density = (sim * row_importance[:, :, None]).sum(dim=1)
    denom = lengths_t.float().clamp_min(1.0).view(num_chunks, 1)
    center = features.sum(dim=1) / denom
    center_dist_sq = (features - center[:, None, :]).square().sum(dim=-1).clamp_min(0.0)
    center_score = torch.exp(-center_dist_sq / scale_sq.view(num_chunks, 1).clamp_min(1e-6)).masked_fill(~valid, 0.0)

    covered = sim.masked_fill(~local_selected[:, None, :], 0.0).amax(dim=2)
    need = (quotas_t - local_selected.sum(dim=1)).clamp_min(0)
    max_need = int(need.max().item()) if int(need.numel()) > 0 else 0
    for step in range(max_need):
        active = need > 0
        candidate_mask = active[:, None] & valid & (~local_selected)
        if not bool(candidate_mask.any().item()):
            break
        if step == 0:
            scores = density * center_score * candidate_importance
        else:
            gains = ((sim - covered[:, :, None]).clamp_min(0.0) * row_importance[:, :, None]).sum(dim=1)
            scores = gains * candidate_importance + 1e-6 * density + 1e-7 * center_score
        scores = scores.masked_fill(~candidate_mask, float("-inf"))
        local_idx = torch.argmax(scores, dim=1)
        chosen_score = scores.gather(1, local_idx.view(num_chunks, 1)).squeeze(1)
        chosen = active & torch.isfinite(chosen_score)
        if not bool(chosen.any().item()):
            break
        local_selected[chosen, local_idx[chosen]] = True
        need = need - chosen.long()
        chosen_sim = sim.gather(2, local_idx.view(num_chunks, 1, 1).expand(num_chunks, width, 1)).squeeze(2)
        covered = torch.where(chosen.view(num_chunks, 1), torch.maximum(covered, chosen_sim), covered)

    old_selected = chunk_idx[local_selected & valid]
    if int(old_selected.numel()) > 0:
        selected.index_fill_(0, old_selected.to(device=selected.device, dtype=torch.long), True)

    current = int(selected.sum().item())
    if current < target:
        need = int(target) - current
        candidates = torch.nonzero(~selected[:old_len], as_tuple=False).flatten()
        take = min(int(need), int(candidates.numel()))
        if take > 0:
            selected.index_fill_(0, candidates[:take].to(device=selected.device, dtype=torch.long), True)
    elif current > target:
        excess = current - int(target)
        removable = selected[:old_len] & (~force_mask[:old_len])
        pos = torch.nonzero(removable, as_tuple=False).flatten()
        if int(pos.numel()) > 0:
            selected.index_fill_(0, pos[:excess].to(device=selected.device, dtype=torch.long), False)

    return torch.nonzero(selected, as_tuple=False).flatten().sort().values


def _chunkwise_keep_from_scores(
    *,
    scores: torch.Tensor,
    kv_head: int,
    old_len: int,
    target: int,
    force_mask: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Select old tokens with a fixed non-overlapping chunk quota."""

    selected = force_mask.clone()
    seq_len = int(selected.numel())
    old_len = max(min(int(old_len), seq_len), 0)
    target = min(max(int(target), int(selected.sum().item())), seq_len)
    if old_len <= 0 or int(selected.sum().item()) >= target:
        return torch.nonzero(selected, as_tuple=False).flatten().sort().values

    recent_forced = int(selected[old_len:].sum().item())
    target_old = min(max(target - recent_forced, int(selected[:old_len].sum().item())), old_len)
    chunk_size = max(int(chunk_size), 2)
    starts = list(range(0, old_len, chunk_size))
    lengths = [min(chunk_size, old_len - s) for s in starts]
    if not starts:
        return torch.nonzero(selected, as_tuple=False).flatten().sort().values

    ideal = [float(length) * float(target_old) / float(max(old_len, 1)) for length in lengths]
    quotas = [int(x) for x in ideal]
    remainder = int(target_old) - int(sum(quotas))
    order = sorted(range(len(starts)), key=lambda i: ideal[i] - quotas[i], reverse=True)
    for i in order[: max(remainder, 0)]:
        quotas[i] += 1

    forced_counts = [int(selected[s : s + l].sum().item()) for s, l in zip(starts, lengths)]
    quotas = [min(max(q, f), l) for q, f, l in zip(quotas, forced_counts, lengths)]
    while sum(quotas) > int(target_old):
        candidates = [i for i, q in enumerate(quotas) if q > forced_counts[i]]
        if not candidates:
            break
        i = min(candidates, key=lambda j: ideal[j] - quotas[j])
        quotas[i] -= 1
    while sum(quotas) < int(target_old):
        candidates = [i for i, q in enumerate(quotas) if q < lengths[i]]
        if not candidates:
            break
        i = max(candidates, key=lambda j: ideal[j] - quotas[j])
        quotas[i] += 1

    head_scores = scores[int(kv_head)]
    for start, length, quota in zip(starts, lengths, quotas):
        end = int(start) + int(length)
        have = int(selected[start:end].sum().item())
        need = int(quota) - int(have)
        if need <= 0:
            continue
        local_scores = head_scores[start:end].clone()
        local_scores = local_scores.masked_fill(selected[start:end], float("-inf"))
        finite = torch.nonzero(torch.isfinite(local_scores), as_tuple=False).flatten()
        take = min(int(need), int(finite.numel()))
        if take <= 0:
            continue
        local_idx = torch.topk(local_scores, k=take, largest=True).indices + int(start)
        selected.index_fill_(0, local_idx.to(device=selected.device, dtype=torch.long), True)

    # Fill or trim residual rounding/pathological gaps without breaking forced tokens.
    current = int(selected.sum().item())
    if current < target:
        need = target - current
        all_scores = head_scores.clone()
        all_scores = all_scores.masked_fill(selected[:old_len], float("-inf"))
        finite = torch.nonzero(torch.isfinite(all_scores), as_tuple=False).flatten()
        take = min(int(need), int(finite.numel()))
        if take > 0:
            idx = torch.topk(all_scores, k=take, largest=True).indices
            selected.index_fill_(0, idx.to(device=selected.device, dtype=torch.long), True)
    elif current > target:
        excess = current - target
        removable = selected[:old_len] & (~force_mask[:old_len])
        pos = torch.nonzero(removable, as_tuple=False).flatten()
        if int(pos.numel()) > 0:
            drop = pos.index_select(0, torch.argsort(head_scores.index_select(0, pos))[:excess])
            selected.index_fill_(0, drop.to(device=selected.device, dtype=torch.long), False)

    return torch.nonzero(selected, as_tuple=False).flatten().sort().values


@torch.no_grad()
def _local_chunk_jaoc_select_and_compress_layer(
    *,
    q_obs: torch.Tensor,
    q_all: torch.Tensor | None = None,
    keys: torch.Tensor,
    values: torch.Tensor,
    budget: int,
    force_mask: torch.Tensor,
    force_recent: int,
    num_key_value_groups: int,
    config: StrictMergeConfig,
    metric_diag: torch.Tensor | None = None,
    spectral_importance: torch.Tensor | None = None,
    profile: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if int(keys.shape[0]) != 1 or int(values.shape[0]) != 1:
        raise NotImplementedError("StrictMerge supports batch_size=1.")
    seq_len = int(keys.shape[2])
    scores: torch.Tensor | None
    importance_scores: torch.Tensor | None = None
    if _uses_operator_probe_groups(config) or _is_chunked_csd_family_mode(config):
        recent = min(max(int(force_recent), 0), seq_len)
        old_len = seq_len - recent
        scores = None
        metric_label = "oprojdiag" if metric_diag is not None else "rawv"
        key_metric_label = str(getattr(config, "key_metric_mode", "raw"))
        if _is_csd_kv_mode(config):
            solver_name = (
                f"local_jaoc_csd_kv_chunk{int(config.local_chunk_size)}"
                f"_kw{float(config.coreset_key_weight):g}"
                f"_vw{float(config.coreset_value_weight):g}"
                f"_gamma{float(config.coreset_tau):g}"
                f"_massmix{float(config.operator_mass_uniform_mix):g}"
                f"_minchunk{int(config.operator_min_chunk_keep)}"
                f"_{str(config.snap_pooling)}k{int(config.snap_kernel_size)}"
                f"_{key_metric_label}"
                f"_{metric_label}"
            )
        elif _is_spectral_csd_kv_mode(config):
            solver_name = (
                f"local_jaoc_spectral_csd_kv_chunk{int(config.local_chunk_size)}"
                f"_kw{float(config.coreset_key_weight):g}"
                f"_vw{float(config.coreset_value_weight):g}"
                f"_waterfill"
                f"_sqrtp"
                f"_minchunk{int(config.operator_min_chunk_keep)}"
                f"_{str(config.snap_pooling)}k{int(config.snap_kernel_size)}"
                f"_{key_metric_label}"
                f"_{metric_label}"
            )
        elif _is_dynamic_csd_kv_mode(config):
            solver_name = (
                f"local_jaoc_dynamic_csd_kv_chunk{int(config.local_chunk_size)}"
                f"_kw{float(config.coreset_key_weight):g}"
                f"_vw{float(config.coreset_value_weight):g}"
                f"_globalresidual"
                f"_block{int(config.dynamic_csd_block_size)}"
                f"_sqrtp"
                f"_{str(config.snap_pooling)}k{int(config.snap_kernel_size)}"
                f"_{key_metric_label}"
                f"_{metric_label}"
            )
        elif _is_spectral_csd_kv_bias_mode(config):
            solver_name = (
                f"local_jaoc_spectral_csd_kv_bias_chunk{int(config.local_chunk_size)}"
                f"_kw{float(config.coreset_key_weight):g}"
                f"_vw{float(config.coreset_value_weight):g}"
                f"_waterfill"
                f"_sqrtp"
                f"_logc8_vmerge"
                f"_minchunk{int(config.operator_min_chunk_keep)}"
                f"_{str(config.snap_pooling)}k{int(config.snap_kernel_size)}"
                f"_{key_metric_label}"
                f"_{metric_label}"
            )
        elif _is_grop_kv_mode(config):
            solver_name = (
                f"local_jaoc_grop_kv_global"
                f"_real{int(config.operator_real_probe_limit)}"
                f"_key{int(config.operator_key_probe_limit)}"
                f"_{str(config.operator_probe_source)}_{metric_label}"
                f"_batch{int(config.batch_drop)}"
                f"_minchunk{int(config.operator_min_chunk_keep)}"
            )
        else:
            solver_name = (
                f"local_jaoc_{str(config.local_score_mode)}_chunk{int(config.local_chunk_size)}"
                f"_real{int(config.operator_real_probe_limit)}"
                f"_key{int(config.operator_key_probe_limit)}"
                f"_{str(config.operator_probe_source)}_{metric_label}"
                f"_massmix{float(config.operator_mass_uniform_mix):g}"
                f"_minchunk{int(config.operator_min_chunk_keep)}"
            )
    elif _is_anchor_facility_mode(config):
        recent = min(max(int(force_recent), 0), seq_len)
        old_len = seq_len - recent
        scores = None
        if _is_weighted_anchor_facility_mode(config):
            importance_scores, _, _, _ = _local_chunk_jaoc_scores(
                q_obs=q_obs,
                keys=keys,
                values=values,
                force_recent=int(force_recent),
                num_key_value_groups=int(num_key_value_groups),
                config=replace(config, local_score_mode="attn_close"),
            )
        metric_label = "oprojdiag" if metric_diag is not None else "rawv"
        importance_label = "_iw_attnclose" if _is_weighted_anchor_facility_mode(config) else ""
        solver_name = (
            f"local_jaoc_anchor_then_facility_chunk{int(config.local_chunk_size)}"
            f"_knn{int(config.coreset_knn)}"
            f"_tau{float(config.coreset_tau):g}"
            f"_kw{float(config.coreset_key_weight):g}"
            f"_vw{float(config.coreset_value_weight):g}"
            f"{importance_label}"
            f"_{metric_label}"
        )
    else:
        scores, old_len, recent, solver_name = _local_chunk_jaoc_scores(
            q_obs=q_obs,
            keys=keys,
            values=values,
            force_recent=int(force_recent),
            num_key_value_groups=int(num_key_value_groups),
            config=config,
        )
    if old_len <= 0 or int(budget) >= seq_len:
        return keys.contiguous(), values.contiguous(), {
            "selection_granularity": "kv_head",
            "selector_mode": "local_jaoc",
            "cache_head_mode": str(config.cache_head_mode),
            "kept": int(seq_len),
            "old_budget": int(old_len),
            "recent": int(recent),
            "rounds": 0,
            "solver": f"{solver_name}_full",
            "final_loss": 0.0,
            "min_denominator": 1.0,
        }

    hkv = int(keys.shape[1])
    base_force = force_mask.to(device=keys.device, dtype=torch.bool).clone()
    target = min(max(int(budget), int(base_force.sum().item()), 1), seq_len)
    compressed_keys: list[torch.Tensor] = []
    compressed_values: list[torch.Tensor] = []
    compressed_log_biases: list[torch.Tensor] = []
    kept_count: int | None = None
    groups = max(int(num_key_value_groups), 1)
    if spectral_importance is not None:
        spectral_importance = spectral_importance.to(device=keys.device, dtype=torch.float32)
        expected_shape = (int(hkv), int(old_len))
        if tuple(spectral_importance.shape) != expected_shape:
            raise ValueError(
                f"spectral_importance must have shape {expected_shape}, got {tuple(spectral_importance.shape)}"
            )
    elif _is_spectral_csd_kv_mode(config) or _is_dynamic_csd_kv_mode(config) or _is_spectral_csd_kv_bias_mode(config):
        spectral_importance = _spectral_csd_importance_scores(
            q_obs=q_obs,
            keys=keys,
            old_len=int(old_len),
            num_key_value_groups=int(num_key_value_groups),
            config=config,
        )
    key_metrics = None
    if (
        str(getattr(config, "key_metric_mode", "raw")) == "qcov_diag"
        and (_is_spectral_csd_kv_mode(config) or _is_dynamic_csd_kv_mode(config) or _is_spectral_csd_kv_bias_mode(config))
    ):
        key_metrics = _qcov_key_metrics_from_qobs(
            q_obs=q_obs,
            hkv=int(hkv),
            groups=int(groups),
            head_dim=int(keys.shape[-1]),
        )
    if _is_spectral_csd_kv_mode(config) or _is_dynamic_csd_kv_mode(config):
        value_metrics = None
        if metric_diag is not None:
            per_head_metrics: list[torch.Tensor] = []
            for head in range(hkv):
                q_start = int(head) * groups
                q_end = q_start + groups
                per_head_metrics.append(metric_diag[q_start:q_end].float().mean(dim=0))
            value_metrics = torch.stack(per_head_metrics, dim=0).contiguous()
        if _is_dynamic_csd_kv_mode(config):
            keep_by_head = _chunkwise_keep_spectral_csd_kv_batched_samples(
                keys=keys,
                values=values,
                old_lens=[int(old_len)],
                seq_lens=[int(seq_len)],
                targets=[int(target)],
                force_masks=base_force.view(1, int(seq_len)),
                chunk_size=int(config.local_chunk_size),
                config=config,
                value_metrics=value_metrics,
                key_metrics=None if key_metrics is None else key_metrics.unsqueeze(0),
                importance=spectral_importance.unsqueeze(0) if spectral_importance is not None else None,
                profile=profile,
            )[0]
        else:
            keep_by_head = _chunkwise_keep_spectral_csd_kv_all_heads(
                keys=keys,
                values=values,
                old_len=int(old_len),
                target=int(target),
                force_mask=base_force,
                chunk_size=int(config.local_chunk_size),
                config=config,
                value_metrics=value_metrics,
                key_metrics=key_metrics,
                importance=spectral_importance,
                profile=profile,
            )
        kept_count = int(keep_by_head.shape[1])
        gather_idx = keep_by_head.view(1, hkv, kept_count, 1).expand(1, hkv, kept_count, int(keys.shape[-1]))
        return (
            keys.gather(2, gather_idx).contiguous(),
            values.gather(2, gather_idx).contiguous(),
            {
                "selection_granularity": "kv_head",
                "selector_mode": "local_jaoc",
                "cache_head_mode": str(config.cache_head_mode),
                "kept": int(kept_count),
                "old_budget": int(max(int(target) - int(base_force[old_len:].sum().item()), 0)),
                "recent": int(recent),
                "rounds": 0,
                "solver": str(solver_name),
                "final_loss": 0.0,
                "min_denominator": 1.0,
                "coreset_key_weight": float(config.coreset_key_weight),
                "coreset_value_weight": float(config.coreset_value_weight),
                "coreset_tau": float(config.coreset_tau),
                "coreset_knn": int(config.coreset_knn),
                "coreset_key_feature": _key_metric_label(config, key_metrics),
                "coreset_value_feature": "oproj_diag" if metric_diag is not None else "raw_value",
                "operator_real_probe_limit": int(config.operator_real_probe_limit),
                "operator_key_probe_limit": int(config.operator_key_probe_limit),
                "operator_probe_source": str(config.operator_probe_source),
                "operator_mass_uniform_mix": float(config.operator_mass_uniform_mix),
                "operator_min_chunk_keep": int(config.operator_min_chunk_keep),
                "batched_kv_heads": True,
            },
        )
    for head in range(hkv):
        if _is_anchor_facility_mode(config) or _uses_operator_probe_groups(config) or _is_chunked_csd_family_mode(config):
            value_metric = None
            if metric_diag is not None:
                q_start = int(head) * groups
                q_end = q_start + groups
                value_metric = metric_diag[q_start:q_end].float().mean(dim=0)
            if _is_csd_kv_mode(config):
                keep_idx = _chunkwise_keep_csd_kv(
                    keys=keys,
                    values=values,
                    q_obs=q_obs,
                    kv_head=int(head),
                    old_len=int(old_len),
                    target=int(target),
                    force_mask=base_force,
                    chunk_size=int(config.local_chunk_size),
                    num_key_value_groups=int(num_key_value_groups),
                    config=config,
                    value_metric=value_metric,
                )
            elif _is_spectral_csd_kv_mode(config) or _is_spectral_csd_kv_bias_mode(config):
                keep_idx = _chunkwise_keep_spectral_csd_kv(
                    keys=keys,
                    values=values,
                    q_obs=q_obs,
                    kv_head=int(head),
                    old_len=int(old_len),
                    target=int(target),
                    force_mask=base_force,
                    chunk_size=int(config.local_chunk_size),
                    num_key_value_groups=int(num_key_value_groups),
                    config=config,
                    value_metric=value_metric,
                    key_metric=None if key_metrics is None else key_metrics[int(head)],
                    importance=None if spectral_importance is None else spectral_importance[int(head)],
                    profile=profile,
                )
            elif _is_grop_kv_mode(config):
                keep_idx = _global_keep_grop_kv(
                    keys=keys,
                    values=values,
                    q_obs=q_obs,
                    q_all=q_all,
                    kv_head=int(head),
                    old_len=int(old_len),
                    target=int(target),
                    force_mask=base_force,
                    num_key_value_groups=int(num_key_value_groups),
                    config=config,
                    value_metric=value_metric,
                )
            elif _is_attention_operator_coreset_mode(config):
                keep_idx = _chunkwise_keep_attention_operator_coreset(
                    keys=keys,
                    values=values,
                    q_obs=q_obs,
                    q_all=q_all,
                    kv_head=int(head),
                    old_len=int(old_len),
                    target=int(target),
                    force_mask=base_force,
                    chunk_size=int(config.local_chunk_size),
                    num_key_value_groups=int(num_key_value_groups),
                    config=config,
                    value_metric=value_metric,
                )
            else:
                keep_idx = _chunkwise_keep_anchor_then_cover(
                    keys=keys,
                    values=values,
                    kv_head=int(head),
                    old_len=int(old_len),
                    target=int(target),
                    force_mask=base_force,
                    chunk_size=int(config.local_chunk_size),
                    config=config,
                    value_metric=value_metric,
                    importance_scores=None if importance_scores is None else importance_scores[int(head)],
                )
        else:
            if scores is None:
                raise RuntimeError("local_jaoc scores were not computed.")
            keep_idx = _chunkwise_keep_from_scores(
                scores=scores,
                kv_head=int(head),
                old_len=int(old_len),
                target=int(target),
                force_mask=base_force,
                chunk_size=int(config.local_chunk_size),
            )
        if kept_count is None:
            kept_count = int(keep_idx.numel())
        elif int(keep_idx.numel()) != int(kept_count):
            raise RuntimeError("local_jaoc produced ragged head lengths.")
        compressed_keys.append(keys[0, head].index_select(0, keep_idx).contiguous())
        if _is_spectral_csd_kv_bias_mode(config):
            merged_values, log_bias = _spectral_csd_bias_and_merged_values(
                keys=keys,
                values=values,
                q_obs=q_obs,
                kv_head=int(head),
                keep_idx=keep_idx,
                old_len=int(old_len),
                chunk_size=int(config.local_chunk_size),
                num_key_value_groups=int(num_key_value_groups),
                config=config,
                value_metric=value_metric,
            )
            compressed_values.append(merged_values.contiguous())
            compressed_log_biases.append(log_bias.contiguous())
        else:
            compressed_values.append(values[0, head].index_select(0, keep_idx).contiguous())

    private_debug: dict[str, Any] = {}
    if compressed_log_biases:
        private_debug["_strictmerge_log_bias"] = torch.stack(compressed_log_biases, dim=0).unsqueeze(0).contiguous()
        private_debug["log_bias_mode"] = "logc8_value_merge"
        bias_tensor = private_debug["_strictmerge_log_bias"]
        private_debug["log_bias_mean"] = float(bias_tensor.float().mean().item())
        private_debug["log_bias_max"] = float(bias_tensor.float().max().item())

    return (
        torch.stack(compressed_keys, dim=0).unsqueeze(0).contiguous(),
        torch.stack(compressed_values, dim=0).unsqueeze(0).contiguous(),
        {
            "selection_granularity": "kv_head",
            "selector_mode": "local_jaoc",
            "cache_head_mode": str(config.cache_head_mode),
            "kept": int(kept_count or 0),
            "old_budget": int(max(int(target) - int(base_force[old_len:].sum().item()), 0)),
            "recent": int(recent),
            "rounds": 0,
            "solver": str(solver_name),
            "final_loss": 0.0,
            "min_denominator": 1.0,
            "coreset_key_weight": float(config.coreset_key_weight),
            "coreset_value_weight": float(config.coreset_value_weight),
            "coreset_tau": float(config.coreset_tau),
            "coreset_knn": int(config.coreset_knn),
            "coreset_key_feature": _key_metric_label(config, key_metrics),
            "coreset_value_feature": "oproj_diag" if metric_diag is not None else "raw_value",
            "operator_real_probe_limit": int(config.operator_real_probe_limit),
            "operator_key_probe_limit": int(config.operator_key_probe_limit),
            "operator_probe_source": str(config.operator_probe_source),
            "operator_mass_uniform_mix": float(config.operator_mass_uniform_mix),
            "operator_min_chunk_keep": int(config.operator_min_chunk_keep),
            **private_debug,
        },
    )


@torch.no_grad()
def _attention_rank_select_and_compress_layer(
    *,
    q_obs: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    budget: int,
    force_recent: int,
    num_key_value_groups: int,
    config: StrictMergeConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Snap/CAKE-style attention ranker inside the StrictMerge runner.

    This is a diagnostic selector: score old tokens by the recent observation
    attention pattern, pool the scores, select per KV head, and concatenate the
    recent window like SnapKV/CAKE-style implementations.
    """

    if int(keys.shape[0]) != 1 or int(values.shape[0]) != 1:
        raise NotImplementedError("StrictMerge supports batch_size=1.")
    seq_len = int(keys.shape[2])
    head_dim = int(keys.shape[3])
    scores, old_len, recent, solver_name = _attention_rank_scores(
        q_obs=q_obs,
        keys=keys,
        force_recent=int(force_recent),
        num_key_value_groups=int(num_key_value_groups),
        config=config,
    )
    mode = str(config.selector_mode)
    if old_len <= 0 or int(budget) >= seq_len:
        return keys.contiguous(), values.contiguous(), {
            "selection_granularity": "kv_head",
            "selector_mode": mode,
            "cache_head_mode": str(config.cache_head_mode),
            "kept": int(seq_len),
            "rounds": 0,
            "solver": f"{solver_name}_full",
            "final_loss": 0.0,
            "min_denominator": 1.0,
        }

    old_budget = max(min(int(budget) - int(recent), int(old_len)), 0)
    if old_budget <= 0:
        return keys[:, :, -recent:, :].contiguous(), values[:, :, -recent:, :].contiguous(), {
            "selection_granularity": "kv_head",
            "selector_mode": mode,
            "cache_head_mode": str(config.cache_head_mode),
            "kept": int(recent),
            "rounds": 0,
            "solver": f"{solver_name}_recent_only",
            "final_loss": 0.0,
            "min_denominator": 1.0,
        }

    hkv = int(keys.shape[1])
    idx = torch.topk(scores, k=int(old_budget), dim=-1, largest=True).indices.sort(dim=-1).values
    gather_idx = idx.view(1, hkv, old_budget, 1).expand(1, hkv, old_budget, head_dim)
    old_keys = keys[:, :, :old_len, :].gather(2, gather_idx)
    old_values = values[:, :, :old_len, :].gather(2, gather_idx)
    recent_keys = keys[:, :, old_len:, :]
    recent_values = values[:, :, old_len:, :]
    compressed_keys = torch.cat([old_keys, recent_keys], dim=2).contiguous()
    compressed_values = torch.cat([old_values, recent_values], dim=2).contiguous()
    return compressed_keys, compressed_values, {
        "selection_granularity": "kv_head",
        "selector_mode": mode,
        "cache_head_mode": str(config.cache_head_mode),
        "kept": int(compressed_keys.shape[2]),
        "old_budget": int(old_budget),
        "recent": int(recent),
        "rounds": 0,
        "solver": str(solver_name),
        "final_loss": 0.0,
        "min_denominator": 1.0,
    }


@torch.no_grad()
def _add_attention_anchors_to_force_mask(
    *,
    q_obs: torch.Tensor,
    keys: torch.Tensor,
    budget: int,
    force_mask: torch.Tensor,
    force_recent: int,
    num_key_value_groups: int,
    config: StrictMergeConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Add layer-shared high-attention anchors before running JAOC.

    The anchor score is Snap/CAKE-style attention rank, aggregated by max over
    KV heads. JAOC still selects the remaining budget under its centered
    attention-output objective.
    """

    if float(config.anchor_ratio) <= 0.0:
        return force_mask, {"anchor_count": 0, "anchor_solver": "none"}
    scores, old_len, recent, solver_name = _attention_rank_scores(
        q_obs=q_obs,
        keys=keys,
        force_recent=int(force_recent),
        num_key_value_groups=int(num_key_value_groups),
        config=replace(config, selector_mode=str(config.anchor_score_mode)),
    )
    seq_len = int(force_mask.numel())
    if old_len <= 0:
        return force_mask, {"anchor_count": 0, "anchor_solver": solver_name}

    force_count = int(force_mask.sum().item())
    budget_room = max(int(budget) - int(force_count), 0)
    anchor_count = min(int(round(float(config.anchor_ratio) * float(budget))), int(budget_room), int(old_len))
    if anchor_count <= 0:
        return force_mask, {
            "anchor_count": 0,
            "anchor_solver": solver_name,
            "anchor_ratio": float(config.anchor_ratio),
        }

    invalid_old = force_mask[:old_len]
    selected = torch.zeros((int(old_len),), dtype=torch.bool, device=force_mask.device)
    if str(config.anchor_selection) == "max_over_heads":
        shared_scores = scores.max(dim=0).values
        shared_scores = shared_scores.masked_fill(invalid_old, float("-inf"))
        finite_count = int(torch.isfinite(shared_scores).sum().item())
        anchor_count = min(int(anchor_count), int(finite_count))
        if anchor_count <= 0:
            return force_mask, {
                "anchor_count": 0,
                "anchor_solver": solver_name,
                "anchor_ratio": float(config.anchor_ratio),
                "anchor_selection": str(config.anchor_selection),
            }
        anchor_idx = torch.topk(shared_scores, k=int(anchor_count), largest=True).indices
    else:
        head_scores = scores.masked_fill(invalid_old.view(1, old_len), float("-inf"))
        order = torch.argsort(head_scores, dim=-1, descending=True)
        anchors: list[torch.Tensor] = []
        for rank in range(int(old_len)):
            for head in range(int(order.shape[0])):
                idx = order[head, rank]
                if not torch.isfinite(head_scores[head, idx]):
                    continue
                if bool(selected[idx].item()):
                    continue
                selected[idx] = True
                anchors.append(idx)
                if len(anchors) >= int(anchor_count):
                    break
            if len(anchors) >= int(anchor_count):
                break
        if not anchors:
            return force_mask, {
                "anchor_count": 0,
                "anchor_solver": solver_name,
                "anchor_ratio": float(config.anchor_ratio),
                "anchor_selection": str(config.anchor_selection),
            }
        anchor_idx = torch.stack(anchors, dim=0)
        anchor_count = int(anchor_idx.numel())
    out = force_mask.clone()
    out.index_fill_(0, anchor_idx.to(device=out.device, dtype=torch.long), True)
    return out, {
        "anchor_count": int(anchor_count),
        "anchor_solver": solver_name,
        "anchor_ratio": float(config.anchor_ratio),
        "anchor_score_mode": str(config.anchor_score_mode),
        "anchor_old_len": int(old_len),
        "anchor_recent": int(recent),
        "anchor_shared": str(config.anchor_selection),
        "anchor_force_before": int(force_count),
        "anchor_force_after": int(out.sum().item()),
        "anchor_seq_len": int(seq_len),
    }


@torch.no_grad()
def _select_and_compress_layer(
    *,
    q_obs: torch.Tensor,
    q_all: torch.Tensor | None = None,
    keys: torch.Tensor,
    values: torch.Tensor,
    budget: int,
    force_mask: torch.Tensor,
    num_key_value_groups: int,
    obs_positions: torch.Tensor,
    config: StrictMergeConfig,
    metric_diag: torch.Tensor | None = None,
    spectral_importance: torch.Tensor | None = None,
    profile: dict[str, float] | None = None,
    use_causal_obs_mask: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Run StrictMerge selection for one layer and return compressed K/V.

    ``layer`` mode returns one shared token set for all KV heads. ``kv_head``
    mode returns one token set per KV head, with the same target length for all
    heads, so the stored cache remains a dense [B, Hkv, K, D] tensor.
    """

    if int(keys.shape[0]) != 1 or int(values.shape[0]) != 1:
        raise NotImplementedError("StrictMerge supports batch_size=1.")
    if str(config.cache_head_mode) == "query" and int(num_key_value_groups) > 1:
        keys = _repeat_kv_heads_for_storage(keys, int(num_key_value_groups))
        values = _repeat_kv_heads_for_storage(values, int(num_key_value_groups))
        num_key_value_groups = 1

    if str(config.selector_mode) == "local_jaoc":
        return _local_chunk_jaoc_select_and_compress_layer(
            q_obs=q_obs,
            q_all=q_all,
            keys=keys,
            values=values,
            budget=int(budget),
            force_mask=force_mask,
            force_recent=int(config.force_recent),
            num_key_value_groups=int(num_key_value_groups),
            config=config,
            metric_diag=metric_diag,
            spectral_importance=spectral_importance,
            profile=profile,
        )

    if str(config.selector_mode) in {"snap", "cake"}:
        return _attention_rank_select_and_compress_layer(
            q_obs=q_obs,
            keys=keys,
            values=values,
            budget=int(budget),
            force_recent=int(config.force_recent),
            num_key_value_groups=int(num_key_value_groups),
            config=config,
        )

    anchor_debug: dict[str, Any] = {}
    if str(config.selector_mode) == "jaoc_anchor":
        force_mask, anchor_debug = _add_attention_anchors_to_force_mask(
            q_obs=q_obs,
            keys=keys,
            budget=int(budget),
            force_mask=force_mask,
            force_recent=int(config.force_recent),
            num_key_value_groups=int(num_key_value_groups),
            config=config,
        )
    mode = str(config.selection_granularity)
    if mode == "layer":
        if int(config.atom_size) > 1:
            if metric_diag is not None or str(config.risk_mode) != "mean":
                raise NotImplementedError("span atoms currently support only value_l2 + mean risk.")
            selection = jaoc_select_layer_span(
                q_obs=q_obs,
                k_cache=keys[0].contiguous(),
                v_cache=values[0].contiguous(),
                budget=int(budget),
                force_keep_mask=force_mask,
                num_key_value_groups=int(num_key_value_groups),
                atom_size=int(config.atom_size),
                obs_positions=obs_positions,
                use_causal_obs_mask=bool(use_causal_obs_mask),
                batch_drop=int(config.batch_drop),
                score_block_size=int(config.score_block_size),
                denom_eps=float(config.denom_eps),
                solver=str(config.solver),
            )
        else:
            selection = jaoc_select_layer_fast(
                q_obs=q_obs,
                k_cache=keys[0].contiguous(),
                v_cache=values[0].contiguous(),
                budget=int(budget),
                force_keep_mask=force_mask,
                num_key_value_groups=int(num_key_value_groups),
                obs_positions=obs_positions,
                use_causal_obs_mask=bool(use_causal_obs_mask),
                batch_drop=int(config.batch_drop),
                score_block_size=int(config.score_block_size),
                candidate_pool_factor=float(config.candidate_pool_factor),
                candidate_pool_min=int(config.candidate_pool_min),
                pair_pool_size=int(config.pair_pool_size),
                precompute_x_sq=bool(config.precompute_x_sq),
                precompute_x_sq_max_elements=int(config.precompute_x_sq_max_elements),
                denom_eps=float(config.denom_eps),
                return_loss_trace=False,
                solver=str(config.solver),
                risk_mode=str(config.risk_mode),
                cvar_beta=float(config.cvar_beta),
                logsumexp_tau=float(config.logsumexp_tau),
                metric_diag=metric_diag,
            )
        keep_idx = selection.keep_indices.to(device=keys.device, dtype=torch.long).sort().values
        return (
            keys.index_select(2, keep_idx).contiguous(),
            values.index_select(2, keep_idx).contiguous(),
            {
                "selection_granularity": "layer",
                "selector_mode": str(config.selector_mode),
                "cache_head_mode": str(config.cache_head_mode),
                "kept": int(keep_idx.numel()),
                "rounds": int(selection.rounds),
                "solver": str(selection.solver),
                "final_loss": float(selection.final_loss),
                "min_denominator": float(selection.min_denominator),
                **anchor_debug,
            },
        )

    if mode != "kv_head":
        raise ValueError("selection_granularity must be one of: layer, kv_head.")

    hkv = int(keys.shape[1])
    groups = int(num_key_value_groups)
    expected_hq = hkv * max(groups, 1)
    if int(q_obs.shape[0]) != expected_hq:
        raise ValueError(f"q_obs heads {int(q_obs.shape[0])} do not match Hkv*groups {expected_hq}.")

    compressed_keys: list[torch.Tensor] = []
    compressed_values: list[torch.Tensor] = []
    losses: list[float] = []
    min_denominators: list[float] = []
    rounds: list[float] = []
    solvers: list[str] = []
    kept_count: int | None = None

    for kv_head in range(hkv):
        q_start = int(kv_head) * groups
        q_end = q_start + groups
        metric_head = None
        if metric_diag is not None:
            metric_head = metric_diag[q_start:q_end].contiguous()
        if int(config.atom_size) > 1:
            selection = jaoc_select_layer_span(
                q_obs=q_obs[q_start:q_end].contiguous(),
                k_cache=keys[0, kv_head : kv_head + 1].contiguous(),
                v_cache=values[0, kv_head : kv_head + 1].contiguous(),
                budget=int(budget),
                force_keep_mask=force_mask,
                num_key_value_groups=int(groups),
                atom_size=int(config.atom_size),
                obs_positions=obs_positions,
                use_causal_obs_mask=bool(use_causal_obs_mask),
                batch_drop=int(config.batch_drop),
                score_block_size=int(config.score_block_size),
                denom_eps=float(config.denom_eps),
                solver=str(config.solver),
            )
        else:
            selection = jaoc_select_layer_fast(
                q_obs=q_obs[q_start:q_end].contiguous(),
                k_cache=keys[0, kv_head : kv_head + 1].contiguous(),
                v_cache=values[0, kv_head : kv_head + 1].contiguous(),
                budget=int(budget),
                force_keep_mask=force_mask,
                num_key_value_groups=int(groups),
                obs_positions=obs_positions,
                use_causal_obs_mask=bool(use_causal_obs_mask),
                batch_drop=int(config.batch_drop),
                score_block_size=int(config.score_block_size),
                candidate_pool_factor=float(config.candidate_pool_factor),
                candidate_pool_min=int(config.candidate_pool_min),
                pair_pool_size=int(config.pair_pool_size),
                precompute_x_sq=bool(config.precompute_x_sq),
                precompute_x_sq_max_elements=int(config.precompute_x_sq_max_elements),
                denom_eps=float(config.denom_eps),
                return_loss_trace=False,
                solver=str(config.solver),
                risk_mode=str(config.risk_mode),
                cvar_beta=float(config.cvar_beta),
                logsumexp_tau=float(config.logsumexp_tau),
                metric_diag=metric_head,
            )
        keep_idx = selection.keep_indices.to(device=keys.device, dtype=torch.long).sort().values
        if kept_count is None:
            kept_count = int(keep_idx.numel())
        elif int(keep_idx.numel()) != int(kept_count):
            raise RuntimeError(
                "Per-KV-head selection produced ragged head lengths: "
                f"head0={kept_count}, head{kv_head}={int(keep_idx.numel())}."
            )
        compressed_keys.append(keys[0, kv_head].index_select(0, keep_idx).contiguous())
        compressed_values.append(values[0, kv_head].index_select(0, keep_idx).contiguous())
        losses.append(float(selection.final_loss))
        min_denominators.append(float(selection.min_denominator))
        rounds.append(float(selection.rounds))
        solvers.append(str(selection.solver))

    mean_loss, min_loss, max_loss = _stats(losses)
    mean_den, min_den, max_den = _stats(min_denominators)
    mean_rounds, min_rounds, max_rounds = _stats(rounds)
    return (
        torch.stack(compressed_keys, dim=0).unsqueeze(0).contiguous(),
        torch.stack(compressed_values, dim=0).unsqueeze(0).contiguous(),
        {
            "selection_granularity": "kv_head",
            "selector_mode": str(config.selector_mode),
            "cache_head_mode": str(config.cache_head_mode),
            "kept": int(kept_count or 0),
            "rounds": int(round(mean_rounds)),
            "solver": ",".join(sorted(set(solvers))),
            "final_loss": float(mean_loss),
            "min_denominator": float(min_den),
            "head_final_loss_mean": float(mean_loss),
            "head_final_loss_min": float(min_loss),
            "head_final_loss_max": float(max_loss),
            "head_min_denominator_mean": float(mean_den),
            "head_min_denominator_min": float(min_den),
            "head_min_denominator_max": float(max_den),
            "head_rounds_mean": float(mean_rounds),
            "head_rounds_min": float(min_rounds),
            "head_rounds_max": float(max_rounds),
            **anchor_debug,
        },
    )


def rope_query_content_at_future_position(
    *,
    model,
    query_content: torch.Tensor,
    future_position: int,
) -> torch.Tensor:
    if query_content.ndim != 3:
        raise ValueError(f"query_content must be [H,R,D], got {tuple(query_content.shape)}")
    device = query_content.device
    obs_count = int(query_content.shape[1])
    position_ids = torch.full((1, obs_count), int(future_position), device=device, dtype=torch.long)
    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is None:
        raise AttributeError("Expected model.model.rotary_emb for future RoPE.")
    hidden_size = int(getattr(model.config, "hidden_size", int(query_content.shape[0]) * int(query_content.shape[2])))
    dummy = torch.empty((1, obs_count, hidden_size), device=device, dtype=query_content.dtype)
    cos, sin = rotary(dummy, position_ids)
    q = query_content.unsqueeze(0)
    q_rope, _ = apply_rotary_pos_emb(q, q, cos, sin)
    return q_rope[0].contiguous()


def rope_query_content_batch_at_future_position(
    *,
    model,
    query_content: torch.Tensor,
    future_position: int,
) -> torch.Tensor:
    if query_content.ndim != 4:
        raise ValueError(f"query_content must be [B,H,R,D], got {tuple(query_content.shape)}")
    device = query_content.device
    batch_size = int(query_content.shape[0])
    obs_count = int(query_content.shape[2])
    position_ids = torch.full((batch_size, obs_count), int(future_position), device=device, dtype=torch.long)
    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is None:
        raise AttributeError("Expected model.model.rotary_emb for future RoPE.")
    hidden_size = int(getattr(model.config, "hidden_size", int(query_content.shape[1]) * int(query_content.shape[3])))
    dummy = torch.empty((batch_size, obs_count, hidden_size), device=device, dtype=query_content.dtype)
    cos, sin = rotary(dummy, position_ids)
    q_rope, _ = apply_rotary_pos_emb(query_content, query_content, cos, sin)
    return q_rope.contiguous()


def rope_query_content_at_original_positions(
    *,
    model,
    query_content: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    if query_content.ndim != 3:
        raise ValueError(f"query_content must be [H,R,D], got {tuple(query_content.shape)}")
    positions = positions.to(device=query_content.device, dtype=torch.long).flatten()
    if int(positions.numel()) != int(query_content.shape[1]):
        raise ValueError("positions must match query_content observation count.")
    obs_count = int(query_content.shape[1])
    position_ids = positions.view(1, obs_count)
    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is None:
        raise AttributeError("Expected model.model.rotary_emb for RoPE.")
    hidden_size = int(getattr(model.config, "hidden_size", int(query_content.shape[0]) * int(query_content.shape[2])))
    dummy = torch.empty((1, obs_count, hidden_size), device=query_content.device, dtype=query_content.dtype)
    cos, sin = rotary(dummy, position_ids)
    q = query_content.unsqueeze(0)
    q_rope, _ = apply_rotary_pos_emb(q, q, cos, sin)
    return q_rope[0].contiguous()


def _layer_budget_floor_cap(
    *,
    seq_len: int,
    force_count: int,
    config: StrictMergeConfig,
) -> tuple[int, int]:
    seq_len = int(seq_len)
    floor = max(
        int(force_count),
        int(round(seq_len * float(config.layer_budget_min_keep_ratio))),
        1,
    )
    floor = min(int(floor), seq_len)
    ceil = max(int(floor), int(round(seq_len * float(config.layer_budget_max_keep_ratio))))
    ceil = min(int(ceil), seq_len)
    return int(floor), int(ceil)


def _capped_largest_remainder_budget(
    *,
    floors: list[int],
    ceils: list[int],
    target_total: int,
    weights: torch.Tensor,
) -> list[int]:
    budgets = list(map(int, floors))
    capacity = [max(int(ceil) - int(floor), 0) for floor, ceil in zip(floors, ceils)]
    remaining = max(int(target_total) - int(sum(budgets)), 0)
    if remaining <= 0 or int(sum(capacity)) <= 0:
        return budgets

    weights = weights.to(dtype=torch.float64)
    cap_t = torch.tensor(capacity, dtype=torch.float64)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if float(weights.sum().item()) <= 0.0:
        weights = cap_t.clamp_min(1.0)
    weights = weights * (cap_t > 0).to(dtype=torch.float64)
    weights = weights / weights.sum().clamp_min(1e-12)
    raw_extra = weights * float(remaining)
    extras = torch.floor(raw_extra).to(torch.long).tolist()
    extras = [min(int(extra), int(cap)) for extra, cap in zip(extras, capacity)]
    budgets = [int(floor) + int(extra) for floor, extra in zip(floors, extras)]
    remaining = max(int(target_total) - int(sum(budgets)), 0)
    fractional_order = torch.argsort(raw_extra - torch.floor(raw_extra), descending=True).tolist()
    while remaining > 0:
        changed = False
        for idx in fractional_order:
            if remaining <= 0:
                break
            if budgets[int(idx)] < int(ceils[int(idx)]):
                budgets[int(idx)] += 1
                remaining -= 1
                changed = True
        if not changed:
            break
    return budgets


def _prompt_qobs_for_layer(
    *,
    model,
    query_state: StrictQueryContentState,
    layer_idx: int,
    prompt_keys: torch.Tensor,
    prompt_len: int,
    config: StrictMergeConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    content_and_pos = query_state.concat_content(int(layer_idx))
    if content_and_pos is None:
        raise RuntimeError(f"No prompt query content recorded for layer {layer_idx}.")
    query_content, content_positions = content_and_pos
    query_content = query_content.to(device=prompt_keys.device, dtype=prompt_keys.dtype).contiguous()
    content_positions = content_positions.to(device=prompt_keys.device, dtype=torch.long).contiguous()
    recorded_len = int(query_content.shape[1])
    if int(content_positions.numel()) != int(recorded_len):
        raise RuntimeError(
            f"Recorded query content/position length mismatch at layer {layer_idx}: "
            f"content={tuple(query_content.shape)} positions={tuple(content_positions.shape)}"
        )
    window = min(int(config.observation_window), int(prompt_len), int(recorded_len))
    selected_content_pos = torch.arange(
        int(recorded_len) - int(window),
        int(recorded_len),
        device=prompt_keys.device,
        dtype=torch.long,
    )
    obs_positions = content_positions.index_select(0, selected_content_pos).contiguous()
    q_content_obs = query_content.index_select(1, selected_content_pos).contiguous()
    if str(config.observation_mode) == "snap_tail" or str(config.selector_mode) in {"snap", "cake", "jaoc_anchor"}:
        q_obs = rope_query_content_at_original_positions(
            model=model,
            query_content=q_content_obs,
            positions=obs_positions,
        ).to(device=prompt_keys.device, dtype=prompt_keys.dtype).contiguous()
    else:
        q_obs = rope_query_content_at_future_position(
            model=model,
            query_content=q_content_obs,
            future_position=int(prompt_len),
        ).to(device=prompt_keys.device, dtype=prompt_keys.dtype).contiguous()
    return q_obs, obs_positions


@torch.no_grad()
def _spectral_csd_head_marginal_lambdas(
    *,
    keys: torch.Tensor,
    values: torch.Tensor,
    importance: torch.Tensor,
    kv_head: int,
    old_len: int,
    force_mask: torch.Tensor,
    chunk_size: int,
    config: StrictMergeConfig,
    value_metric: torch.Tensor | None,
    key_metric: torch.Tensor | None = None,
    profile: dict[str, float] | None = None,
) -> torch.Tensor:
    if int(keys.shape[0]) != 1 or int(values.shape[0]) != 1:
        raise NotImplementedError("StrictMerge supports batch_size=1.")
    device = keys.device
    seq_len = int(force_mask.numel())
    old_len = max(min(int(old_len), seq_len), 0)
    if old_len <= 0:
        return torch.empty((0,), dtype=torch.float32, device=device)
    head_dim = int(keys.shape[-1])
    selected = force_mask.to(device=device, dtype=torch.bool).clone()
    stage_t = _profile_start(device, profile)
    head_keys = keys[0, int(kv_head), :old_len, :].float()
    head_values = values[0, int(kv_head), :old_len, :].float()
    importance = importance.to(device=device, dtype=torch.float32).flatten()
    if int(importance.numel()) != int(old_len):
        raise ValueError(f"importance must have length {old_len}, got {int(importance.numel())}")
    importance = torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if float(importance.sum().item()) <= 0.0:
        importance = torch.ones_like(importance)
    _profile_add(profile, "budget_head_setup_sec", stage_t, device)

    stage_t = _profile_start(device, profile)
    chunk_size = max(int(chunk_size), 2)
    starts = list(range(0, int(old_len), chunk_size))
    lengths = [min(chunk_size, int(old_len) - s) for s in starts]
    if not starts:
        return torch.empty((0,), dtype=torch.float32, device=device)

    num_chunks = len(starts)
    width = max(lengths)
    starts_t = torch.tensor(starts, dtype=torch.long, device=device).view(num_chunks, 1)
    lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
    local_pos = torch.arange(width, dtype=torch.long, device=device).view(1, width)
    valid = local_pos < lengths_t.view(num_chunks, 1)
    chunk_idx = (starts_t + local_pos).clamp_max(max(int(old_len) - 1, 0))
    flat_idx = chunk_idx.reshape(-1)

    k_chunks = head_keys.index_select(0, flat_idx).reshape(num_chunks, width, head_dim)
    v_chunks = head_values.index_select(0, flat_idx).reshape(num_chunks, width, head_dim)
    p_chunks = importance.index_select(0, flat_idx).reshape(num_chunks, width).masked_fill(~valid, 0.0)
    local_selected = selected.index_select(0, flat_idx).reshape(num_chunks, width) & valid
    _profile_add(profile, "budget_head_gather_sec", stage_t, device)

    stage_t = _profile_start(device, profile)
    k_features = k_chunks
    if key_metric is not None:
        metric = key_metric.to(device=device, dtype=torch.float32).flatten()
        if int(metric.numel()) != int(head_dim):
            raise ValueError(f"key_metric must have length {head_dim}, got {int(metric.numel())}")
        metric = torch.nan_to_num(metric, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
        k_features = k_features * metric.sqrt().view(1, 1, head_dim)
    v_features = v_chunks
    if value_metric is not None:
        metric = value_metric.to(device=device, dtype=torch.float32).flatten()
        if int(metric.numel()) != int(head_dim):
            raise ValueError(f"value_metric must have length {head_dim}, got {int(metric.numel())}")
        metric = torch.nan_to_num(metric, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
        v_features = v_features * metric.sqrt().view(1, 1, head_dim)

    parts: list[torch.Tensor] = []
    if float(config.coreset_key_weight) > 0.0:
        key_energy = k_features.square().sum(dim=-1).masked_fill(~valid, 0.0)
        key_scale = (key_energy.sum(dim=1) / lengths_t.float().clamp_min(1.0)).sqrt().clamp_min(1e-6)
        parts.append((k_features / key_scale.view(num_chunks, 1, 1)) * float(config.coreset_key_weight))
    if float(config.coreset_value_weight) > 0.0:
        value_energy = v_features.square().sum(dim=-1).masked_fill(~valid, 0.0)
        value_scale = (value_energy.sum(dim=1) / lengths_t.float().clamp_min(1.0)).sqrt().clamp_min(1e-6)
        parts.append((v_features / value_scale.view(num_chunks, 1, 1)) * float(config.coreset_value_weight))
    if not parts:
        return torch.empty((0,), dtype=torch.float32, device=device)

    features = torch.cat(parts, dim=-1).masked_fill(~valid.unsqueeze(-1), 0.0)
    z = features * torch.sqrt(p_chunks.clamp_min(0.0) + 1e-12).unsqueeze(-1)
    z = z.masked_fill(~valid.unsqueeze(-1), 0.0)
    _profile_add(profile, "budget_head_feature_sec", stage_t, device)

    stage_t = _profile_start(device, profile)
    residual = _project_selected_chunk_columns(z, local_selected, valid)
    _profile_add(profile, "budget_head_forced_project_sec", stage_t, device)

    stage_t = _profile_start(device, profile)
    candidate_mask = valid & (~local_selected)
    z_budget = residual.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
    gram = torch.matmul(z_budget, z_budget.transpose(1, 2))
    eigvals = _eigvalsh_symmetric_chunked(
        gram,
        max_batch=int(os.environ.get("BEST_V1_SELECT_EIG_BATCH", "512")),
    ).clamp_min(0.0)
    eig_desc = torch.flip(eigvals, dims=[1])
    capacity = (lengths_t - local_selected.sum(dim=1).to(torch.long)).clamp_min(0)
    rank_idx = torch.arange(width, dtype=torch.long, device=device).view(1, width)
    eig_scores = eig_desc.masked_fill(rank_idx >= capacity.view(num_chunks, 1), float("-inf"))
    vals = eig_scores[torch.isfinite(eig_scores)].flatten()
    if int(vals.numel()) == 0:
        return torch.empty((0,), dtype=torch.float32, device=device)
    out = torch.sort(vals, descending=True).values.contiguous()
    _profile_add(profile, "budget_head_eig_sec", stage_t, device)
    return out


@torch.no_grad()
def _spectral_csd_all_head_marginal_lambdas(
    *,
    keys: torch.Tensor,
    values: torch.Tensor,
    importance: torch.Tensor,
    old_len: int,
    force_mask: torch.Tensor,
    chunk_size: int,
    config: StrictMergeConfig,
    value_metrics: torch.Tensor | None,
    key_metrics: torch.Tensor | None = None,
    profile: dict[str, float] | None = None,
) -> list[torch.Tensor]:
    if int(keys.shape[0]) != 1 or int(values.shape[0]) != 1:
        raise NotImplementedError("StrictMerge supports batch_size=1.")
    device = keys.device
    hkv = int(keys.shape[1])
    seq_len = int(force_mask.numel())
    old_len = max(min(int(old_len), seq_len), 0)
    if old_len <= 0:
        return [torch.empty((0,), dtype=torch.float32, device=device) for _ in range(hkv)]
    head_dim = int(keys.shape[-1])
    selected = force_mask.to(device=device, dtype=torch.bool).view(1, seq_len).expand(hkv, seq_len).clone()

    stage_t = _profile_start(device, profile)
    head_keys = keys[0, :, :old_len, :].float()
    head_values = values[0, :, :old_len, :].float()
    importance = importance.to(device=device, dtype=torch.float32).contiguous()
    expected_importance = (hkv, int(old_len))
    if tuple(importance.shape) != expected_importance:
        raise ValueError(f"importance must have shape {expected_importance}, got {tuple(importance.shape)}")
    importance = torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    zero_heads = importance.sum(dim=1, keepdim=True) <= 0.0
    importance = torch.where(zero_heads, torch.ones_like(importance), importance)
    _profile_add(profile, "budget_all_heads_setup_sec", stage_t, device)

    stage_t = _profile_start(device, profile)
    chunk_size = max(int(chunk_size), 2)
    starts = list(range(0, int(old_len), chunk_size))
    lengths = [min(chunk_size, int(old_len) - s) for s in starts]
    if not starts:
        return [torch.empty((0,), dtype=torch.float32, device=device) for _ in range(hkv)]

    num_chunks = len(starts)
    width = max(lengths)
    starts_t = torch.tensor(starts, dtype=torch.long, device=device).view(num_chunks, 1)
    lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
    local_pos = torch.arange(width, dtype=torch.long, device=device).view(1, width)
    valid = local_pos < lengths_t.view(num_chunks, 1)
    valid_h = valid.view(1, num_chunks, width)
    chunk_idx = (starts_t + local_pos).clamp_max(max(int(old_len) - 1, 0))
    flat_idx = chunk_idx.reshape(-1)

    k_chunks = head_keys.index_select(1, flat_idx).reshape(hkv, num_chunks, width, head_dim)
    v_chunks = head_values.index_select(1, flat_idx).reshape(hkv, num_chunks, width, head_dim)
    p_chunks = importance.index_select(1, flat_idx).reshape(hkv, num_chunks, width).masked_fill(~valid_h, 0.0)
    local_selected = selected[:, :old_len].index_select(1, flat_idx).reshape(hkv, num_chunks, width) & valid_h
    _profile_add(profile, "budget_all_heads_gather_sec", stage_t, device)

    stage_t = _profile_start(device, profile)
    k_features = k_chunks
    if key_metrics is not None:
        metrics = key_metrics.to(device=device, dtype=torch.float32)
        if tuple(metrics.shape) != (hkv, head_dim):
            raise ValueError(f"key_metrics must have shape {(hkv, head_dim)}, got {tuple(metrics.shape)}")
        metrics = torch.nan_to_num(metrics, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
        k_features = k_features * metrics.sqrt().view(hkv, 1, 1, head_dim)
    v_features = v_chunks
    if value_metrics is not None:
        metrics = value_metrics.to(device=device, dtype=torch.float32)
        if tuple(metrics.shape) != (hkv, head_dim):
            raise ValueError(f"value_metrics must have shape {(hkv, head_dim)}, got {tuple(metrics.shape)}")
        metrics = torch.nan_to_num(metrics, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
        v_features = v_features * metrics.sqrt().view(hkv, 1, 1, head_dim)

    parts: list[torch.Tensor] = []
    if float(config.coreset_key_weight) > 0.0:
        key_energy = k_features.square().sum(dim=-1).masked_fill(~valid_h, 0.0)
        key_scale = (key_energy.sum(dim=2) / lengths_t.view(1, num_chunks).float().clamp_min(1.0)).sqrt().clamp_min(1e-6)
        parts.append((k_features / key_scale.view(hkv, num_chunks, 1, 1)) * float(config.coreset_key_weight))
    if float(config.coreset_value_weight) > 0.0:
        value_energy = v_features.square().sum(dim=-1).masked_fill(~valid_h, 0.0)
        value_scale = (value_energy.sum(dim=2) / lengths_t.view(1, num_chunks).float().clamp_min(1.0)).sqrt().clamp_min(1e-6)
        parts.append((v_features / value_scale.view(hkv, num_chunks, 1, 1)) * float(config.coreset_value_weight))
    if not parts:
        return [torch.empty((0,), dtype=torch.float32, device=device) for _ in range(hkv)]

    features = torch.cat(parts, dim=-1).masked_fill(~valid_h.unsqueeze(-1), 0.0)
    z = features * torch.sqrt(p_chunks.clamp_min(0.0) + 1e-12).unsqueeze(-1)
    z = z.masked_fill(~valid_h.unsqueeze(-1), 0.0)
    _profile_add(profile, "budget_all_heads_feature_sec", stage_t, device)

    stage_t = _profile_start(device, profile)
    residual = _project_selected_chunk_columns(z, local_selected, valid_h)
    _profile_add(profile, "budget_all_heads_forced_project_sec", stage_t, device)

    stage_t = _profile_start(device, profile)
    candidate_mask = valid_h & (~local_selected)
    z_budget = residual.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
    gram = torch.matmul(z_budget, z_budget.transpose(2, 3))
    eigvals = torch.linalg.eigvalsh(gram.float()).clamp_min(0.0)
    eig_desc = torch.flip(eigvals, dims=[2])
    capacity = (lengths_t.view(1, num_chunks) - local_selected.sum(dim=2).to(torch.long)).clamp_min(0)
    rank_idx = torch.arange(width, dtype=torch.long, device=device).view(1, 1, width)
    eig_scores = eig_desc.masked_fill(rank_idx >= capacity.unsqueeze(-1), float("-inf"))
    out: list[torch.Tensor] = []
    for head in range(hkv):
        vals = eig_scores[int(head)][torch.isfinite(eig_scores[int(head)])].flatten()
        if int(vals.numel()) == 0:
            out.append(torch.empty((0,), dtype=torch.float32, device=device))
        else:
            out.append(torch.sort(vals, descending=True).values.contiguous())
    _profile_add(profile, "budget_all_heads_eig_sec", stage_t, device)
    return out


def _batched_spectral_supported(config: StrictMergeConfig, budget_mode: str) -> bool:
    return (
        str(config.selector_mode) == "local_jaoc"
        and str(config.selection_granularity) == "kv_head"
        and str(config.cache_head_mode) == "kv"
        and _is_spectral_csd_kv_mode(config)
        and not _needs_full_query_content(config)
        and str(budget_mode) in {"attention_entropy", "fixed"}
    )


def _eigvalsh_symmetric_chunked(gram: torch.Tensor, *, max_batch: int | None = None) -> torch.Tensor:
    """Batched symmetric eigvalsh with a bounded cuSOLVER batch size.

    cuSOLVER's batched eigensolver can reject very large flattened batch sizes
    even for tiny matrices. We keep the sample/head/chunk vectorization intact,
    but call eigvalsh in chunks over the flattened leading dimensions.
    """

    if int(gram.dim()) < 2 or int(gram.shape[-1]) != int(gram.shape[-2]):
        raise ValueError(f"gram must end with a square matrix, got {tuple(gram.shape)}")
    gram = torch.nan_to_num(gram.float(), nan=0.0, posinf=0.0, neginf=0.0)
    gram = 0.5 * (gram + gram.transpose(-1, -2))
    leading = tuple(gram.shape[:-2])
    width = int(gram.shape[-1])
    flat = gram.reshape(-1, width, width).contiguous()
    if max_batch is None:
        max_batch = int(os.environ.get("BEST_V1_EIG_BATCH", "4096"))
    max_batch = max(int(max_batch), 1)

    def stable_eigvalsh(part: torch.Tensor) -> torch.Tensor:
        try:
            return torch.linalg.eigvalsh(part).clamp_min(0.0)
        except Exception:
            eye = torch.eye(width, device=part.device, dtype=part.dtype).view(1, width, width)
            scale = part.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1).clamp_min(1.0).view(-1, 1, 1)
            for eps in (1e-6, 1e-5, 1e-4):
                try:
                    return torch.linalg.eigvalsh(part + eye * scale * float(eps)).clamp_min(0.0)
                except Exception:
                    pass
            cpu_part = part.detach().to(device="cpu", dtype=torch.float64)
            cpu_eye = torch.eye(width, dtype=torch.float64).view(1, width, width)
            cpu_scale = cpu_part.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1).clamp_min(1.0).view(-1, 1, 1)
            for eps in (1e-6, 1e-5, 1e-4):
                try:
                    values = torch.linalg.eigvalsh(cpu_part + cpu_eye * cpu_scale * float(eps)).clamp_min(0.0)
                    return values.to(device=part.device, dtype=part.dtype)
                except Exception:
                    pass
            values = torch.sort(torch.linalg.svdvals(cpu_part), dim=-1).values.clamp_min(0.0)
            return values.to(device=part.device, dtype=part.dtype)

    pieces: list[torch.Tensor] = []
    for start in range(0, int(flat.shape[0]), max_batch):
        part = flat[start : start + max_batch]
        pieces.append(stable_eigvalsh(part))
    return torch.cat(pieces, dim=0).reshape(*leading, width).contiguous()


@torch.no_grad()
def _spectral_csd_all_head_marginal_lambdas_batched(
    *,
    keys: torch.Tensor,
    values: torch.Tensor,
    importance: torch.Tensor,
    old_lens: list[int],
    seq_lens: list[int],
    force_masks: torch.Tensor,
    chunk_size: int,
    config: StrictMergeConfig,
    value_metrics: torch.Tensor | None,
    key_metrics: torch.Tensor | None = None,
    profile: dict[str, float] | None = None,
    return_eig_desc: bool = False,
    approximate_diagonal: bool = False,
    curve_caps: list[int] | None = None,
) -> list[list[torch.Tensor]] | tuple[list[list[torch.Tensor]], torch.Tensor]:
    if int(keys.dim()) != 4 or int(values.dim()) != 4:
        raise ValueError("keys/values must have shape [B,H,K,D].")
    device = keys.device
    batch_size = int(keys.shape[0])
    hkv = int(keys.shape[1])
    max_old = max((int(v) for v in old_lens), default=0)
    if max_old <= 0:
        empty = [[torch.empty((0,), dtype=torch.float32, device=device) for _ in range(hkv)] for _ in range(batch_size)]
        if bool(return_eig_desc):
            return empty, torch.empty((batch_size, hkv, 0, 0), dtype=torch.float32, device=device)
        return empty
    head_dim = int(keys.shape[-1])
    old_lens_t = torch.tensor([int(v) for v in old_lens], dtype=torch.long, device=device)
    old_pos = torch.arange(max_old, dtype=torch.long, device=device).view(1, max_old)
    old_valid = old_pos < old_lens_t.view(batch_size, 1)

    stage_t = _profile_start(device, profile)
    head_keys = keys[:, :, :max_old, :].float()
    head_values = values[:, :, :max_old, :].float()
    importance = importance[:, :, :max_old].to(device=device, dtype=torch.float32).contiguous()
    importance = torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    importance = importance.masked_fill(~old_valid[:, None, :], 0.0)
    zero_heads = importance.sum(dim=2, keepdim=True) <= 0.0
    importance = torch.where(zero_heads, old_valid[:, None, :].to(dtype=importance.dtype), importance)
    selected = force_masks[:, None, :].to(device=device, dtype=torch.bool).expand(batch_size, hkv, -1).clone()
    _profile_add(profile, "budget_batched_setup_sec", stage_t, device)

    stage_t = _profile_start(device, profile)
    chunk_size = max(int(chunk_size), 2)
    starts = list(range(0, int(max_old), chunk_size))
    if not starts:
        empty = [[torch.empty((0,), dtype=torch.float32, device=device) for _ in range(hkv)] for _ in range(batch_size)]
        if bool(return_eig_desc):
            return empty, torch.empty((batch_size, hkv, 0, 0), dtype=torch.float32, device=device)
        return empty
    num_chunks = len(starts)
    width = max(min(chunk_size, int(max_old) - s) for s in starts)
    starts_t = torch.tensor(starts, dtype=torch.long, device=device).view(1, num_chunks, 1)
    local_pos = torch.arange(width, dtype=torch.long, device=device).view(1, 1, width)
    lengths_bc = (old_lens_t.view(batch_size, 1) - torch.tensor(starts, dtype=torch.long, device=device).view(1, num_chunks)).clamp(0, width)
    valid = local_pos < lengths_bc.view(batch_size, num_chunks, 1)
    valid_h = valid[:, None, :, :]
    chunk_idx = (starts_t + local_pos).clamp_max(max(int(max_old) - 1, 0)).view(num_chunks, width)
    flat_idx = chunk_idx.reshape(-1)

    k_chunks = head_keys.index_select(2, flat_idx).reshape(batch_size, hkv, num_chunks, width, head_dim)
    v_chunks = head_values.index_select(2, flat_idx).reshape(batch_size, hkv, num_chunks, width, head_dim)
    p_chunks = importance.index_select(2, flat_idx).reshape(batch_size, hkv, num_chunks, width).masked_fill(~valid_h, 0.0)
    local_selected = selected[:, :, :max_old].index_select(2, flat_idx).reshape(batch_size, hkv, num_chunks, width) & valid_h
    _profile_add(profile, "budget_batched_gather_sec", stage_t, device)

    stage_t = _profile_start(device, profile)
    k_features = k_chunks
    if key_metrics is not None:
        metrics = key_metrics.to(device=device, dtype=torch.float32)
        if tuple(metrics.shape) == (hkv, head_dim):
            metrics = metrics.view(1, hkv, head_dim).expand(batch_size, hkv, head_dim)
        if tuple(metrics.shape) != (batch_size, hkv, head_dim):
            raise ValueError(f"key_metrics must have shape {(batch_size, hkv, head_dim)} or {(hkv, head_dim)}, got {tuple(metrics.shape)}")
        metrics = torch.nan_to_num(metrics, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
        k_features = k_features * metrics.sqrt().view(batch_size, hkv, 1, 1, head_dim)
    v_features = v_chunks
    if value_metrics is not None:
        metrics = value_metrics.to(device=device, dtype=torch.float32)
        if tuple(metrics.shape) != (hkv, head_dim):
            raise ValueError(f"value_metrics must have shape {(hkv, head_dim)}, got {tuple(metrics.shape)}")
        metrics = torch.nan_to_num(metrics, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
        v_features = v_features * metrics.sqrt().view(1, hkv, 1, 1, head_dim)

    parts: list[torch.Tensor] = []
    denom = lengths_bc[:, None, :].float().clamp_min(1.0)
    if float(config.coreset_key_weight) > 0.0:
        key_energy = k_features.square().sum(dim=-1).masked_fill(~valid_h, 0.0)
        key_scale = (key_energy.sum(dim=3) / denom).sqrt().clamp_min(1e-6)
        parts.append((k_features / key_scale.view(batch_size, hkv, num_chunks, 1, 1)) * float(config.coreset_key_weight))
    if float(config.coreset_value_weight) > 0.0:
        value_energy = v_features.square().sum(dim=-1).masked_fill(~valid_h, 0.0)
        value_scale = (value_energy.sum(dim=3) / denom).sqrt().clamp_min(1e-6)
        parts.append((v_features / value_scale.view(batch_size, hkv, num_chunks, 1, 1)) * float(config.coreset_value_weight))
    if not parts:
        return [[torch.empty((0,), dtype=torch.float32, device=device) for _ in range(hkv)] for _ in range(batch_size)]
    features = torch.cat(parts, dim=-1).masked_fill(~valid_h.unsqueeze(-1), 0.0)
    z = features * torch.sqrt(p_chunks.clamp_min(0.0) + 1e-12).unsqueeze(-1)
    z = z.masked_fill(~valid_h.unsqueeze(-1), 0.0)
    _profile_add(profile, "budget_batched_feature_sec", stage_t, device)

    stage_t = _profile_start(device, profile)
    residual = _project_selected_chunk_columns(z, local_selected, valid_h)
    _profile_add(profile, "budget_batched_forced_project_sec", stage_t, device)

    stage_t = _profile_start(device, profile)
    candidate_mask = valid_h & (~local_selected)
    z_budget = residual.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
    capacity = (lengths_bc[:, None, :] - local_selected.sum(dim=3).to(torch.long)).clamp_min(0)
    rank_idx = torch.arange(width, dtype=torch.long, device=device).view(1, 1, 1, width)
    if bool(approximate_diagonal):
        eig_scores = z_budget.square().sum(dim=-1).masked_fill(~candidate_mask, float("-inf"))
        eig_desc = torch.sort(eig_scores, dim=3, descending=True).values if bool(return_eig_desc) else eig_scores
        profile_key = "budget_batched_diag_curve_sec"
    else:
        gram = torch.matmul(z_budget, z_budget.transpose(3, 4))
        eigvals = _eigvalsh_symmetric_chunked(gram)
        eig_desc = torch.flip(eigvals, dims=[3])
        eig_scores = eig_desc.masked_fill(rank_idx >= capacity.unsqueeze(-1), float("-inf"))
        profile_key = "budget_batched_eig_sec"
    out: list[list[torch.Tensor]] = []
    if bool(approximate_diagonal) and curve_caps is not None:
        caps = [max(int(cap), 0) for cap in curve_caps]
        max_curve = min(max(caps, default=0), int(eig_scores.shape[2]) * int(eig_scores.shape[3]))
        if max_curve <= 0:
            out = [[torch.empty((0,), dtype=torch.float32, device=device) for _ in range(hkv)] for _ in range(batch_size)]
        else:
            flat_scores = eig_scores.reshape(batch_size, hkv, -1)
            top_vals = torch.topk(flat_scores, k=int(max_curve), dim=2, largest=True, sorted=True).values
            total_scores = torch.nan_to_num(
                flat_scores.masked_fill(~torch.isfinite(flat_scores), 0.0),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_min(0.0).sum(dim=2)
            valid_counts = torch.isfinite(flat_scores).sum(dim=2).detach().cpu().tolist()
            for sample_idx in range(batch_size):
                sample_out = []
                cap = min(int(caps[int(sample_idx)]), int(max_curve))
                for head in range(hkv):
                    take = min(int(cap), int(valid_counts[int(sample_idx)][int(head)]))
                    vals = top_vals[int(sample_idx), int(head), :take].contiguous() if take > 0 else torch.empty((0,), dtype=torch.float32, device=device)
                    valid_total = int(valid_counts[int(sample_idx)][int(head)])
                    if valid_total > take:
                        tail = (total_scores[int(sample_idx), int(head)] - vals.float().sum()).clamp_min(0.0).view(1)
                        vals = torch.cat([vals, tail.to(device=device, dtype=vals.dtype)], dim=0)
                    sample_out.append(vals)
                out.append(sample_out)
    else:
        for sample_idx in range(batch_size):
            sample_out = []
            for head in range(hkv):
                vals = eig_scores[int(sample_idx), int(head)][torch.isfinite(eig_scores[int(sample_idx), int(head)])].flatten()
                sample_out.append(torch.sort(vals, descending=True).values.contiguous() if int(vals.numel()) > 0 else torch.empty((0,), dtype=torch.float32, device=device))
            out.append(sample_out)
    _profile_add(profile, profile_key, stage_t, device)
    if bool(return_eig_desc):
        return out, eig_desc.detach().contiguous()
    return out


@torch.no_grad()
def _chunkwise_keep_spectral_csd_kv_batched_samples(
    *,
    keys: torch.Tensor,
    values: torch.Tensor,
    old_lens: list[int],
    seq_lens: list[int],
    targets: list[int],
    force_masks: torch.Tensor,
    chunk_size: int,
    config: StrictMergeConfig,
    value_metrics: torch.Tensor | None = None,
    key_metrics: torch.Tensor | None = None,
    importance: torch.Tensor | None = None,
    precomputed_eig_desc: torch.Tensor | None = None,
    approximate_quota_curve: bool = False,
    profile: dict[str, float] | None = None,
) -> list[torch.Tensor]:
    if int(keys.dim()) != 4 or int(values.dim()) != 4:
        raise ValueError("keys/values must have shape [B,H,K,D].")
    if importance is None:
        raise ValueError("importance must be precomputed for batched spectral selection.")
    device = keys.device
    batch_size = int(keys.shape[0])
    hkv = int(keys.shape[1])
    max_seq = int(keys.shape[2])
    head_dim = int(keys.shape[3])
    max_old = max((int(v) for v in old_lens), default=0)
    old_lens_t = torch.tensor([int(v) for v in old_lens], dtype=torch.long, device=device)
    seq_lens_t = torch.tensor([int(v) for v in seq_lens], dtype=torch.long, device=device)
    targets_t = torch.tensor([int(v) for v in targets], dtype=torch.long, device=device)
    selected = force_masks[:, None, :].to(device=device, dtype=torch.bool).expand(batch_size, hkv, max_seq).clone()
    if max_old <= 0:
        return [
            torch.nonzero(force_masks[int(i), : int(seq_lens[i])], as_tuple=False).flatten().sort().values.view(1, -1).expand(hkv, -1).contiguous()
            for i in range(batch_size)
        ]

    stage_t = _profile_start(device, profile)
    old_pos = torch.arange(max_old, dtype=torch.long, device=device).view(1, max_old)
    old_valid = old_pos < old_lens_t.view(batch_size, 1)
    head_keys = keys[:, :, :max_old, :].float()
    head_values = values[:, :, :max_old, :].float()
    importance = importance[:, :, :max_old].to(device=device, dtype=torch.float32).contiguous()
    importance = torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    importance = importance.masked_fill(~old_valid[:, None, :], 0.0)
    zero_heads = importance.sum(dim=2, keepdim=True) <= 0.0
    importance = torch.where(zero_heads, old_valid[:, None, :].to(dtype=importance.dtype), importance)
    _profile_add(profile, "chunk_batched_setup_sec", stage_t, device)

    stage_t = _profile_start(device, profile)
    chunk_size = max(int(chunk_size), 2)
    starts = list(range(0, int(max_old), chunk_size))
    if not starts:
        return [
            torch.nonzero(force_masks[int(i), : int(seq_lens[i])], as_tuple=False).flatten().sort().values.view(1, -1).expand(hkv, -1).contiguous()
            for i in range(batch_size)
        ]
    num_chunks = len(starts)
    width = max(min(chunk_size, int(max_old) - s) for s in starts)
    starts_t = torch.tensor(starts, dtype=torch.long, device=device).view(1, num_chunks, 1)
    local_pos = torch.arange(width, dtype=torch.long, device=device).view(1, 1, width)
    lengths_bc = (old_lens_t.view(batch_size, 1) - torch.tensor(starts, dtype=torch.long, device=device).view(1, num_chunks)).clamp(0, width)
    valid = local_pos < lengths_bc.view(batch_size, num_chunks, 1)
    valid_h = valid[:, None, :, :]
    chunk_idx = (starts_t + local_pos).clamp_max(max(int(max_old) - 1, 0)).view(num_chunks, width)
    flat_idx = chunk_idx.reshape(-1)

    k_chunks = head_keys.index_select(2, flat_idx).reshape(batch_size, hkv, num_chunks, width, head_dim)
    v_chunks = head_values.index_select(2, flat_idx).reshape(batch_size, hkv, num_chunks, width, head_dim)
    p_chunks = importance.index_select(2, flat_idx).reshape(batch_size, hkv, num_chunks, width).masked_fill(~valid_h, 0.0)
    local_selected = selected[:, :, :max_old].index_select(2, flat_idx).reshape(batch_size, hkv, num_chunks, width) & valid_h
    _profile_add(profile, "chunk_batched_gather_sec", stage_t, device)

    stage_t = _profile_start(device, profile)
    k_features = k_chunks
    if key_metrics is not None:
        metrics = key_metrics.to(device=device, dtype=torch.float32)
        if tuple(metrics.shape) == (hkv, head_dim):
            metrics = metrics.view(1, hkv, head_dim).expand(batch_size, hkv, head_dim)
        if tuple(metrics.shape) != (batch_size, hkv, head_dim):
            raise ValueError(f"key_metrics must have shape {(batch_size, hkv, head_dim)} or {(hkv, head_dim)}, got {tuple(metrics.shape)}")
        metrics = torch.nan_to_num(metrics, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
        k_features = k_features * metrics.sqrt().view(batch_size, hkv, 1, 1, head_dim)
    v_features = v_chunks
    if value_metrics is not None:
        metrics = value_metrics.to(device=device, dtype=torch.float32)
        if tuple(metrics.shape) != (hkv, head_dim):
            raise ValueError(f"value_metrics must have shape {(hkv, head_dim)}, got {tuple(metrics.shape)}")
        metrics = torch.nan_to_num(metrics, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
        v_features = v_features * metrics.sqrt().view(1, hkv, 1, 1, head_dim)

    parts: list[torch.Tensor] = []
    denom = lengths_bc[:, None, :].float().clamp_min(1.0)
    if float(config.coreset_key_weight) > 0.0:
        key_energy = k_features.square().sum(dim=-1).masked_fill(~valid_h, 0.0)
        key_scale = (key_energy.sum(dim=3) / denom).sqrt().clamp_min(1e-6)
        parts.append((k_features / key_scale.view(batch_size, hkv, num_chunks, 1, 1)) * float(config.coreset_key_weight))
    if float(config.coreset_value_weight) > 0.0:
        value_energy = v_features.square().sum(dim=-1).masked_fill(~valid_h, 0.0)
        value_scale = (value_energy.sum(dim=3) / denom).sqrt().clamp_min(1e-6)
        parts.append((v_features / value_scale.view(batch_size, hkv, num_chunks, 1, 1)) * float(config.coreset_value_weight))
    if not parts:
        raise ValueError("At least one coreset feature weight must be positive.")
    features = torch.cat(parts, dim=-1).masked_fill(~valid_h.unsqueeze(-1), 0.0)
    z = features * torch.sqrt(p_chunks.clamp_min(0.0) + 1e-12).unsqueeze(-1)
    z = z.masked_fill(~valid_h.unsqueeze(-1), 0.0)
    _profile_add(profile, "chunk_batched_feature_sec", stage_t, device)

    stage_t = _profile_start(device, profile)
    residual_after_forced = _project_selected_chunk_columns(z, local_selected, valid_h)
    _profile_add(profile, "chunk_batched_forced_project_sec", stage_t, device)

    force_pos = torch.arange(max_seq, dtype=torch.long, device=device).view(1, max_seq)
    recent_forced = (force_masks & (force_pos >= old_lens_t.view(batch_size, 1)) & (force_pos < seq_lens_t.view(batch_size, 1))).sum(dim=1).to(torch.long)
    old_forced = (force_masks[:, :max_old] & old_valid).sum(dim=1).to(torch.long)
    target_old = torch.minimum(
        torch.maximum(targets_t - recent_forced, old_forced),
        old_lens_t,
    )
    residual = residual_after_forced
    selected_gain = torch.full((batch_size, hkv, num_chunks, width), float("-inf"), dtype=torch.float32, device=device)
    if _is_dynamic_csd_kv_mode(config):
        old_selected_count = local_selected.sum(dim=(2, 3)).to(torch.long)
        capacity = ((valid_h & (~local_selected)).sum(dim=(2, 3))).to(torch.long)
        need = torch.minimum((target_old.view(batch_size, 1) - old_selected_count).clamp_min(0), capacity)

        stage_t = _profile_start(device, profile)
        candidate_mask = valid_h & (~local_selected)
        token_scores = residual.square().sum(dim=-1).masked_fill(~candidate_mask, float("-inf"))
        chunk_best_score, chunk_best_local = token_scores.max(dim=3)
        max_need = int(need.max().item()) if int(need.numel()) > 0 else 0
        block_size = max(int(os.environ.get("BEST_V1_DYNAMIC_CSD_BLOCK", int(config.dynamic_csd_block_size))), 1)
        block_size = min(int(block_size), max(int(num_chunks), 1))
        max_rounds = int(math.ceil(float(max_need) / float(block_size))) if max_need > 0 else 0
        picked_flat = torch.zeros((batch_size, hkv, num_chunks * width), dtype=torch.bool, device=device)
        local_idx_per_chunk = torch.zeros((batch_size, hkv, num_chunks), dtype=torch.long, device=device)
        chosen_chunk = torch.zeros((batch_size, hkv, num_chunks), dtype=torch.bool, device=device)
        flat_selected_gain = selected_gain.view(batch_size, hkv, num_chunks * width)
        valid_all = valid_h.expand(batch_size, hkv, num_chunks, width)
        rank_idx = torch.arange(block_size, dtype=torch.long, device=device).view(1, 1, block_size)
        chunk_gather_shape = (batch_size, hkv, block_size, width)
        residual_gather_shape = (batch_size, hkv, block_size, width, head_dim)
        for _ in range(max_rounds):
            active = need > 0
            chunk_scores = chunk_best_score.masked_fill(~active.unsqueeze(-1), float("-inf"))
            chosen_score, chunk_idx_sel = torch.topk(chunk_scores, k=int(block_size), dim=2, largest=True)
            chosen = active.unsqueeze(-1) & torch.isfinite(chosen_score) & (rank_idx < need.unsqueeze(-1))
            local_idx_sel = chunk_best_local.gather(2, chunk_idx_sel)
            flat_choice = (chunk_idx_sel * int(width) + local_idx_sel).clamp(0, max(num_chunks * width - 1, 0))

            picked_flat.zero_()
            picked_flat.scatter_(2, flat_choice, chosen)
            local_selected = local_selected | picked_flat.view(batch_size, hkv, num_chunks, width)
            previous_gain = flat_selected_gain.gather(2, flat_choice)
            flat_selected_gain.scatter_(2, flat_choice, torch.where(chosen, chosen_score, previous_gain))
            need = need - chosen.sum(dim=2).to(dtype=torch.long)

            local_idx_per_chunk.zero_()
            local_idx_per_chunk.scatter_(2, chunk_idx_sel, local_idx_sel)
            chosen_chunk.zero_()
            chosen_chunk.scatter_(2, chunk_idx_sel, chosen)
            residual = _project_pivot_chunk_rows(residual, local_idx_per_chunk, chosen_chunk)

            chunk_gather = chunk_idx_sel.view(batch_size, hkv, block_size, 1).expand(chunk_gather_shape)
            residual_gather = chunk_idx_sel.view(batch_size, hkv, block_size, 1, 1).expand(residual_gather_shape)
            chunk_residual = residual.gather(2, residual_gather)
            chunk_valid = valid_all.gather(2, chunk_gather)
            chunk_selected = local_selected.gather(2, chunk_gather)
            new_scores = chunk_residual.square().sum(dim=-1).masked_fill(~(chunk_valid & (~chunk_selected)), float("-inf"))
            new_best_score, new_best_local = new_scores.max(dim=3)
            old_best_score = chunk_best_score.gather(2, chunk_idx_sel)
            old_best_local = chunk_best_local.gather(2, chunk_idx_sel)
            chunk_best_score.scatter_(2, chunk_idx_sel, torch.where(chosen, new_best_score, old_best_score))
            chunk_best_local.scatter_(2, chunk_idx_sel, torch.where(chosen, new_best_local, old_best_local))
        _profile_add(profile, "chunk_batched_dynamic_cpqr_sec", stage_t, device)
    else:
        forced_counts = local_selected.sum(dim=3).to(torch.long)
        min_keep = max(int(config.operator_min_chunk_keep), 0)
        if min_keep > 0:
            min_counts = torch.minimum(lengths_bc, torch.full_like(lengths_bc, min_keep))
            base_counts = torch.maximum(forced_counts, min_counts[:, None, :])
            use_base = base_counts[:, 0, :].sum(dim=1) <= target_old
            quotas_t = torch.where(use_base.view(batch_size, 1, 1), base_counts, forced_counts)
        else:
            quotas_t = forced_counts.clone()

        extra_total = torch.minimum(
            (target_old - quotas_t[:, 0, :].sum(dim=1)).clamp_min(0),
            (lengths_bc - quotas_t[:, 0, :]).clamp_min(0).sum(dim=1),
        )
        if bool((extra_total > 0).any().item()):
            stage_t = _profile_start(device, profile)
            if precomputed_eig_desc is not None:
                eig_desc = precomputed_eig_desc.to(device=device, dtype=torch.float32).contiguous()
                expected = (batch_size, hkv, num_chunks, width)
                if tuple(eig_desc.shape) != expected:
                    raise ValueError(f"precomputed_eig_desc must have shape {expected}, got {tuple(eig_desc.shape)}")
                profile_key = "chunk_batched_eig_quota_reuse_sec"
            else:
                candidate_mask = valid_h & (~local_selected)
                z_budget = residual_after_forced.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
                if bool(approximate_quota_curve):
                    eig_desc = torch.sort(z_budget.square().sum(dim=-1), dim=3, descending=True).values
                    profile_key = "chunk_batched_diag_quota_sec"
                else:
                    gram = torch.matmul(z_budget, z_budget.transpose(3, 4))
                    eigvals = _eigvalsh_symmetric_chunked(gram)
                    eig_desc = torch.flip(eigvals, dims=[3])
                    profile_key = "chunk_batched_eig_quota_sec"
            capacity = (lengths_bc[:, None, :] - quotas_t).clamp_min(0)
            rank_idx = torch.arange(width, dtype=torch.long, device=device).view(1, 1, 1, width)
            eig_scores = eig_desc.masked_fill(rank_idx >= capacity.unsqueeze(-1), float("-inf"))
            extra_counts = torch.zeros((batch_size, hkv, num_chunks), dtype=torch.long, device=device)
            for sample_idx in range(batch_size):
                extra = int(extra_total[int(sample_idx)].item())
                if extra <= 0:
                    continue
                top = torch.topk(eig_scores[int(sample_idx)].reshape(hkv, -1), k=int(extra), dim=1, largest=True).indices
                chunk_ids = torch.div(top, width, rounding_mode="floor")
                extra_counts[int(sample_idx)].scatter_add_(1, chunk_ids, torch.ones_like(chunk_ids, dtype=torch.long))
            quotas_t = quotas_t + extra_counts
            _profile_add(profile, profile_key, stage_t, device)

        need = (quotas_t - local_selected.sum(dim=3)).clamp_min(0)
        stage_t = _profile_start(device, profile)
        max_need = int(need.max().item()) if int(need.numel()) > 0 else 0
        for _ in range(max_need):
            active = need > 0
            candidate_mask = active.unsqueeze(-1) & valid_h & (~local_selected)
            scores = residual.square().sum(dim=-1).masked_fill(~candidate_mask, float("-inf"))
            local_idx = torch.argmax(scores, dim=3)
            chosen_score = scores.gather(3, local_idx.unsqueeze(-1)).squeeze(-1)
            chosen = active & torch.isfinite(chosen_score)
            picked = torch.zeros_like(local_selected)
            picked.scatter_(3, local_idx.unsqueeze(-1), True)
            picked = picked & chosen.unsqueeze(-1)
            local_selected = local_selected | picked
            selected_gain = torch.where(picked, chosen_score.unsqueeze(-1), selected_gain)
            next_need = need - chosen.long()
            residual = _project_pivot_chunk_rows(residual, local_idx, chosen & (next_need > 0))
            need = next_need
        _profile_add(profile, "chunk_batched_cpqr_sec", stage_t, device)

    stage_t = _profile_start(device, profile)
    old_flags = local_selected & valid_h
    old_positions = chunk_idx.view(1, 1, num_chunks, width).expand(batch_size, hkv, num_chunks, width)
    old_selected_i8 = torch.zeros((batch_size, hkv, max_old), dtype=torch.int8, device=device)
    old_selected_i8.scatter_reduce_(
        2,
        old_positions.reshape(batch_size, hkv, num_chunks * width),
        old_flags.to(dtype=torch.int8).reshape(batch_size, hkv, num_chunks * width),
        reduce="amax",
        include_self=True,
    )
    selected[:, :, :max_old] = selected[:, :, :max_old] | old_selected_i8.to(dtype=torch.bool)

    target_values: list[int] = []
    for sample_idx in range(batch_size):
        seq_len = int(seq_lens[int(sample_idx)])
        target_values.append(
            min(max(int(targets[int(sample_idx)]), int(force_masks[int(sample_idx), :seq_len].sum().item())), seq_len)
        )
    target_values_t = torch.tensor(target_values, dtype=torch.long, device=device)
    pos_seq = torch.arange(max_seq, dtype=torch.long, device=device).view(1, 1, max_seq)
    valid_seq = pos_seq < seq_lens_t.view(batch_size, 1, 1)
    counts = (selected[:, :, :max_seq] & valid_seq).sum(dim=2)
    if bool((counts == target_values_t.view(batch_size, 1)).all().item()):
        pos_score = torch.where(
            selected[:, :, :max_seq] & valid_seq,
            -pos_seq.expand(batch_size, hkv, max_seq).to(dtype=torch.float32),
            torch.full((batch_size, hkv, max_seq), float("-inf"), dtype=torch.float32, device=device),
        )
        outputs = [
            torch.topk(pos_score[int(sample_idx)], k=int(target_values[int(sample_idx)]), dim=1, largest=True)
            .indices.sort(dim=1)
            .values.contiguous()
            for sample_idx in range(batch_size)
        ]
        _profile_add(profile, "chunk_batched_finalize_sec", stage_t, device)
        return outputs

    residual_scores = residual.square().sum(dim=-1).masked_fill(~valid_h, float("-inf"))
    gain_by_old = torch.full((batch_size, hkv, max_old), float("-inf"), dtype=torch.float32, device=device)
    selected_gain_by_old = torch.full((batch_size, hkv, max_old), float("-inf"), dtype=torch.float32, device=device)
    flat_positions = old_positions.reshape(batch_size, hkv, num_chunks * width)
    valid_all = valid_h.expand(batch_size, hkv, num_chunks, width)
    flat_residual_scores = residual_scores.masked_fill(~valid_all, float("-inf")).reshape(batch_size, hkv, num_chunks * width)
    flat_selected_gain = selected_gain.masked_fill(~valid_all, float("-inf")).reshape(batch_size, hkv, num_chunks * width)
    gain_by_old.scatter_reduce_(2, flat_positions, flat_residual_scores, reduce="amax", include_self=True)
    selected_gain_by_old.scatter_reduce_(2, flat_positions, flat_selected_gain, reduce="amax", include_self=True)

    counts = (selected[:, :, :max_seq] & valid_seq).sum(dim=2)
    if not bool((counts == target_values_t.view(batch_size, 1)).all().item()):
        for sample_idx in range(batch_size):
            seq_len = int(seq_lens[int(sample_idx)])
            old_len = int(old_lens[int(sample_idx)])
            target = int(target_values[int(sample_idx)])
            for head in range(hkv):
                current = int(selected[int(sample_idx), int(head), :seq_len].sum().item())
                if current < target:
                    need_fill = int(target) - current
                    candidates = torch.nonzero(~selected[int(sample_idx), int(head), :old_len], as_tuple=False).flatten()
                    if int(candidates.numel()) > 0:
                        cand_scores = gain_by_old[int(sample_idx), int(head)].index_select(0, candidates)
                        if not bool(torch.isfinite(cand_scores).any().item()):
                            cand_scores = importance[int(sample_idx), int(head)].index_select(0, candidates)
                        take = min(int(need_fill), int(candidates.numel()))
                        fill = candidates.index_select(0, torch.topk(cand_scores, k=take, largest=True).indices)
                        selected[int(sample_idx), int(head)].index_fill_(0, fill.to(device=device, dtype=torch.long), True)
                elif current > target:
                    excess = current - int(target)
                    removable = selected[int(sample_idx), int(head), :old_len] & (~force_masks[int(sample_idx), :old_len].to(device=device, dtype=torch.bool))
                    pos = torch.nonzero(removable, as_tuple=False).flatten()
                    if int(pos.numel()) > 0:
                        gains = selected_gain_by_old[int(sample_idx), int(head)].index_select(0, pos)
                        if not bool(torch.isfinite(gains).any().item()):
                            gains = importance[int(sample_idx), int(head)].index_select(0, pos)
                        drop = pos.index_select(0, torch.argsort(gains)[:excess])
                        selected[int(sample_idx), int(head)].index_fill_(0, drop.to(device=device, dtype=torch.long), False)

    pos_score = torch.where(
        selected[:, :, :max_seq] & valid_seq,
        -pos_seq.expand(batch_size, hkv, max_seq).to(dtype=torch.float32),
        torch.full((batch_size, hkv, max_seq), float("-inf"), dtype=torch.float32, device=device),
    )
    outputs = [
        torch.topk(pos_score[int(sample_idx)], k=int(target_values[int(sample_idx)]), dim=1, largest=True)
        .indices.sort(dim=1)
        .values.contiguous()
        for sample_idx in range(batch_size)
    ]
    _profile_add(profile, "chunk_batched_finalize_sec", stage_t, device)
    return outputs


@torch.no_grad()
def _attention_entropy_layer_score(
    *,
    q_obs: torch.Tensor,
    keys: torch.Tensor,
    old_len: int,
    num_key_value_groups: int,
    force_mask: torch.Tensor | None = None,
) -> float:
    if int(keys.shape[0]) != 1:
        raise NotImplementedError("StrictMerge supports batch_size=1.")
    hkv = int(keys.shape[1])
    seq_len = int(keys.shape[2])
    head_dim = int(keys.shape[3])
    old_len = max(min(int(old_len), int(seq_len)), 0)
    if old_len <= 0:
        return 0.0

    device = keys.device
    compressible = torch.ones((old_len,), dtype=torch.bool, device=device)
    if force_mask is not None:
        compressible = ~force_mask[:old_len].to(device=device, dtype=torch.bool)
    candidate_count = int(compressible.sum().item())
    if candidate_count <= 1:
        return 0.0
    if int(q_obs.numel()) <= 0:
        return 1.0

    groups = max(int(num_key_value_groups), 1)
    expected_hq = hkv * groups
    if int(q_obs.shape[0]) != int(expected_hq):
        raise ValueError(f"q_obs head mismatch: expected {expected_hq}, got {int(q_obs.shape[0])}")

    obs_count = int(q_obs.shape[1])
    q_grouped = q_obs.float().reshape(hkv, groups, obs_count, head_dim)
    logits = torch.einsum("hgrd,htd->hgrt", q_grouped, keys[0].float())
    logits = logits / (float(head_dim) ** 0.5)
    attn = torch.softmax(logits, dim=-1, dtype=torch.float32)[..., :old_len]
    attn = attn * compressible.view(1, 1, 1, old_len).to(dtype=attn.dtype)
    mass = attn.sum(dim=-1)
    p = attn / mass.clamp_min(1e-12).unsqueeze(-1)
    entropy = -(p * p.clamp_min(1e-12).log()).sum(dim=-1)
    entropy = entropy / max(float(math.log(candidate_count)), 1e-12)
    concentration = (1.0 - entropy).clamp_min(0.0)
    weighted_concentration = concentration * mass.clamp_min(0.0)
    score_t = 0.5 * weighted_concentration.mean() + 0.5 * weighted_concentration.amax()
    score = float(torch.nan_to_num(score_t, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0).item())
    return score


@torch.no_grad()
def _attention_entropy_layer_budget_inputs(
    *,
    model,
    layers: list[tuple[torch.Tensor, torch.Tensor]],
    query_state: StrictQueryContentState,
    config: StrictMergeConfig,
    profile: dict[str, float] | None = None,
) -> tuple[list[int], list[int], list[float], dict[int, dict[str, Any]]]:
    seq_lens: list[int] = []
    force_counts: list[int] = []
    layer_scores: list[float] = []
    layer_precompute: dict[int, dict[str, Any]] = {}

    for layer_idx, (prompt_keys, prompt_values) in enumerate(layers):
        del prompt_values
        seq_len = int(prompt_keys.shape[-2])
        force_mask = build_force_keep_mask(
            seq_len=int(seq_len),
            force_sink=int(config.force_sink),
            force_recent=int(config.force_recent),
            force_prefix=int(config.force_prefix),
            device=prompt_keys.device,
        )
        force_count = int(force_mask.sum().item())
        seq_lens.append(int(seq_len))
        force_counts.append(int(force_count))
        recent = min(max(int(config.force_recent), 0), seq_len)
        old_len = seq_len - recent
        if old_len <= 0:
            layer_scores.append(0.0)
            continue

        stage_t = _profile_start(prompt_keys.device, profile)
        q_obs, obs_positions = _prompt_qobs_for_layer(
            model=model,
            query_state=query_state,
            layer_idx=int(layer_idx),
            prompt_keys=prompt_keys,
            prompt_len=int(seq_len),
            config=config,
        )
        _profile_add(profile, "budget_qobs_sec", stage_t, prompt_keys.device)
        use_causal_obs_mask = False
        hkv = int(prompt_keys.shape[1])
        groups = int(model.config.num_attention_heads // max(hkv, 1))
        stage_t = _profile_start(prompt_keys.device, profile)
        layer_score = _attention_entropy_layer_score(
            q_obs=q_obs,
            keys=prompt_keys,
            old_len=int(old_len),
            num_key_value_groups=int(groups),
            force_mask=force_mask,
        )
        _profile_add(profile, "budget_attention_entropy_sec", stage_t, prompt_keys.device)
        stage_t = _profile_start(prompt_keys.device, profile)
        importance = _spectral_csd_importance_scores(
            q_obs=q_obs,
            keys=prompt_keys,
            old_len=int(old_len),
            num_key_value_groups=int(groups),
            config=config,
        )
        _profile_add(profile, "budget_importance_sec", stage_t, prompt_keys.device)
        layer_precompute[int(layer_idx)] = {
            "q_obs": q_obs.detach().contiguous(),
            "obs_positions": obs_positions.detach().contiguous(),
            "use_causal_obs_mask": bool(use_causal_obs_mask),
            "importance": importance.detach().contiguous(),
            "attention_entropy_score": float(layer_score),
        }
        layer_scores.append(float(layer_score))
    return seq_lens, force_counts, layer_scores, layer_precompute


def _attention_entropy_layer_budgets(
    *,
    seq_lens: list[int],
    force_counts: list[int],
    layer_scores: list[float],
    config: StrictMergeConfig,
) -> list[int]:
    floors: list[int] = []
    ceils: list[int] = []
    target_total = 0
    for seq_len, force_count in zip(seq_lens, force_counts):
        floor, ceil = _layer_budget_floor_cap(seq_len=int(seq_len), force_count=int(force_count), config=config)
        floors.append(floor)
        ceils.append(ceil)
        if int(config.fixed_budget) > 0:
            target_total += min(max(int(config.fixed_budget), int(force_count), 1), int(seq_len))
        else:
            target_total += uniform_layer_budget(
                seq_len=int(seq_len),
                keep_ratio=float(config.keep_ratio),
                force_keep_count=int(force_count),
            )
    target_total = max(int(target_total), int(sum(floors)))
    rho = max(float(config.layer_budget_rho), 0.0)
    weights = torch.tensor(layer_scores, dtype=torch.float64).clamp_min(1e-12)
    weights = weights.pow(rho)
    return _capped_largest_remainder_budget(
        floors=floors,
        ceils=ceils,
        target_total=int(target_total),
        weights=weights,
    )


@torch.no_grad()
def compress_prompt_cache_strictmerge(
    *,
    model,
    prompt_past_key_values,
    query_state: StrictQueryContentState,
    config: StrictMergeConfig,
    budget_mode: str = "attention_entropy",
    fixed_layer_budgets: list[int] | None = None,
    profile: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Compress prompt cache using strict DA-tail K observation queries."""

    config.validate()
    if str(budget_mode) not in {"attention_entropy", "fixed"}:
        raise ValueError("SpectralKV budget_mode must be attention_entropy or fixed.")

    layers = extract_cache_layers(prompt_past_key_values)
    num_layers = int(len(layers))
    spectral_precompute: dict[int, dict[str, Any]] = {}
    if str(budget_mode) == "attention_entropy":
        seq_lens, force_counts, layer_scores, spectral_precompute = _attention_entropy_layer_budget_inputs(
            model=model,
            layers=layers,
            query_state=query_state,
            config=config,
            profile=profile,
        )
        fixed_layer_budgets = _attention_entropy_layer_budgets(
            seq_lens=seq_lens,
            force_counts=force_counts,
            layer_scores=layer_scores,
            config=config,
        )
        del layer_scores
    if str(budget_mode) == "fixed" and fixed_layer_budgets is None:
        raise ValueError("fixed budget mode requires fixed_layer_budgets.")
    del layers
    debug: list[dict[str, Any]] = []
    for layer_idx in range(num_layers):
        prompt_keys, prompt_values = get_layer_cache(prompt_past_key_values, int(layer_idx))
        if int(prompt_keys.shape[0]) != 1:
            raise NotImplementedError("StrictMerge supports batch_size=1.")
        prompt_len = int(prompt_keys.shape[-2])
        if prompt_len <= 1:
            del prompt_keys, prompt_values
            continue

        needs_full_query_content = _needs_full_query_content(config)
        q_all = None
        spectral_importance: torch.Tensor | None = None
        cached_obs = None if bool(needs_full_query_content) else spectral_precompute.get(int(layer_idx))
        if cached_obs is not None:
            q_obs = cached_obs["q_obs"].to(device=prompt_keys.device, dtype=prompt_keys.dtype).contiguous()
            obs_positions = cached_obs["obs_positions"].to(device=prompt_keys.device, dtype=torch.long).contiguous()
            use_causal_obs_mask = bool(cached_obs.get("use_causal_obs_mask", False))
            maybe_importance = cached_obs.get("importance")
            if isinstance(maybe_importance, torch.Tensor):
                spectral_importance = maybe_importance.to(device=prompt_keys.device, dtype=torch.float32).contiguous()
        else:
            content_and_pos = query_state.concat_content(int(layer_idx))
            if content_and_pos is None:
                raise RuntimeError(f"No prompt query content recorded for layer {layer_idx}.")
            query_content, content_positions = content_and_pos
            query_content = query_content.to(device=prompt_keys.device, dtype=prompt_keys.dtype).contiguous()
            content_positions = content_positions.to(device=prompt_keys.device, dtype=torch.long).contiguous()
            recorded_len = int(query_content.shape[1])
            if int(content_positions.numel()) != int(recorded_len):
                raise RuntimeError(
                    f"Recorded query content/position length mismatch at layer {layer_idx}: "
                    f"content={tuple(query_content.shape)} positions={tuple(content_positions.shape)}"
                )
            if bool(needs_full_query_content) and int(recorded_len) != int(prompt_len):
                raise RuntimeError(
                    f"This selector needs full prompt query content at layer {layer_idx}: "
                    f"content={tuple(query_content.shape)} prompt_len={prompt_len}"
                )

            window = min(int(config.observation_window), int(prompt_len), int(recorded_len))
            selected_content_pos = torch.arange(
                int(recorded_len) - int(window),
                int(recorded_len),
                device=prompt_keys.device,
                dtype=torch.long,
            )
            obs_positions = content_positions.index_select(0, selected_content_pos).contiguous()
            q_content_obs = query_content.index_select(1, selected_content_pos).contiguous()
            if str(config.observation_mode) == "snap_tail" or str(config.selector_mode) in {"snap", "cake", "jaoc_anchor"}:
                q_obs = rope_query_content_at_original_positions(
                    model=model,
                    query_content=q_content_obs,
                    positions=obs_positions,
                ).to(device=prompt_keys.device, dtype=prompt_keys.dtype).contiguous()
                use_causal_obs_mask = True
            else:
                q_obs = rope_query_content_at_future_position(
                    model=model,
                    query_content=q_content_obs,
                    future_position=int(prompt_len),
                ).to(device=prompt_keys.device, dtype=prompt_keys.dtype).contiguous()
                use_causal_obs_mask = False

            if bool(needs_full_query_content):
                q_all = rope_query_content_at_original_positions(
                    model=model,
                    query_content=query_content,
                    positions=content_positions,
                ).to(device=prompt_keys.device, dtype=prompt_keys.dtype).contiguous()

        force_mask = build_force_keep_mask(
            seq_len=int(prompt_len),
            force_sink=int(config.force_sink),
            force_recent=int(config.force_recent),
            force_prefix=int(config.force_prefix),
            device=prompt_keys.device,
        )
        if fixed_layer_budgets is not None:
            budget = max(int(fixed_layer_budgets[int(layer_idx)]), int(force_mask.sum().item()), 1)
        elif int(config.fixed_budget) > 0:
            budget = max(int(config.fixed_budget), int(force_mask.sum().item()), 1)
        else:
            budget = uniform_layer_budget(
                seq_len=int(prompt_len),
                keep_ratio=float(config.keep_ratio),
                force_keep_count=int(force_mask.sum().item()),
            )
        if int(budget) >= int(prompt_len):
            del prompt_keys, prompt_values, q_obs, obs_positions, force_mask, q_all, spectral_importance
            continue

        num_kv_heads = int(prompt_keys.shape[1])
        num_groups = int(model.config.num_attention_heads // max(num_kv_heads, 1))
        select_start = time.perf_counter()
        metric_diag = None
        if str(config.metric_mode) == "oproj_diag":
            metric_diag = _get_oproj_metric_diag_cached(
                model=model,
                layer_idx=int(layer_idx),
                num_heads=int(model.config.num_attention_heads),
                head_dim=int(prompt_keys.shape[-1]),
                device=prompt_keys.device,
            )
        compressed_keys, compressed_values, selection_debug = _select_and_compress_layer(
            q_obs=q_obs,
            q_all=q_all,
            keys=prompt_keys,
            values=prompt_values,
            budget=int(budget),
            force_mask=force_mask,
            num_key_value_groups=int(num_groups),
            obs_positions=obs_positions,
            config=config,
            metric_diag=metric_diag,
            spectral_importance=spectral_importance,
            profile=profile,
            use_causal_obs_mask=bool(use_causal_obs_mask),
        )
        log_bias = selection_debug.pop("_strictmerge_log_bias", None)
        select_sec = float(time.perf_counter() - select_start)
        set_layer_cache(prompt_past_key_values, int(layer_idx), compressed_keys, compressed_values)
        if isinstance(log_bias, torch.Tensor):
            set_layer_log_bias(prompt_past_key_values, int(layer_idx), log_bias)
        debug_item = {
                "layer_idx": int(layer_idx),
                "seq_len": int(prompt_len),
                "obs_count": int(q_obs.shape[1]),
                "budget": int(budget),
                "kept": int(selection_debug["kept"]),
                "keep_ratio": float(int(selection_debug["kept"]) / max(int(prompt_len), 1)),
                "forced": int(force_mask.sum().item()),
                "observation_source": str(config.observation_mode),
                "use_causal_obs_mask": bool(use_causal_obs_mask),
                "metric_mode": str(config.metric_mode),
                "key_metric_mode": str(getattr(config, "key_metric_mode", "raw")),
                "risk_mode": str(config.risk_mode),
                "atom_size": int(config.atom_size),
                "selector_mode": str(config.selector_mode),
                "cache_head_mode": str(config.cache_head_mode),
                "spectral_precompute_reused": bool(cached_obs is not None),
                "select_sec": float(select_sec),
        }
        debug_item.update(selection_debug)
        debug.append(debug_item)
        for mapping_name in ("layer_query_contents", "layer_content_positions"):
            mapping = getattr(query_state, mapping_name, None)
            if isinstance(mapping, dict):
                mapping.pop(int(layer_idx), None)
        spectral_precompute.pop(int(layer_idx), None)
        del (
            prompt_keys,
            prompt_values,
            compressed_keys,
            compressed_values,
            log_bias,
            q_obs,
            obs_positions,
            force_mask,
            q_all,
            spectral_importance,
            selection_debug,
            debug_item,
        )
        if torch.cuda.is_available() and prompt_past_key_values is not None:
            torch.cuda.empty_cache()
    return debug


@torch.no_grad()
def compress_single_prompt_layer_strictmerge(
    *,
    model,
    layer_idx: int,
    prompt_keys: torch.Tensor,
    prompt_values: torch.Tensor,
    query_state: StrictQueryContentState,
    config: StrictMergeConfig,
    budget: int,
    profile: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    """Compress one prompt-cache layer with a precomputed static budget."""

    config.validate()
    if int(prompt_keys.shape[0]) != 1:
        raise NotImplementedError("Layer-wise SpectralKV currently supports batch_size=1.")
    prompt_len = int(prompt_keys.shape[-2])
    if prompt_len <= 0:
        raise ValueError("prompt cache length must be positive.")

    content_and_pos = query_state.concat_content(int(layer_idx))
    if content_and_pos is None:
        raise RuntimeError(f"No prompt query content recorded for layer {layer_idx}.")
    query_content, content_positions = content_and_pos
    query_content = query_content.to(device=prompt_keys.device, dtype=prompt_keys.dtype).contiguous()
    content_positions = content_positions.to(device=prompt_keys.device, dtype=torch.long).contiguous()
    recorded_len = int(query_content.shape[1])
    if int(content_positions.numel()) != int(recorded_len):
        raise RuntimeError(
            f"Recorded query content/position length mismatch at layer {layer_idx}: "
            f"content={tuple(query_content.shape)} positions={tuple(content_positions.shape)}"
        )
    if bool(_needs_full_query_content(config)) and int(recorded_len) != int(prompt_len):
        raise RuntimeError(
            f"This selector needs full prompt query content at layer {layer_idx}: "
            f"content={tuple(query_content.shape)} prompt_len={prompt_len}"
        )

    window = min(int(config.observation_window), int(prompt_len), int(recorded_len))
    selected_content_pos = torch.arange(
        int(recorded_len) - int(window),
        int(recorded_len),
        device=prompt_keys.device,
        dtype=torch.long,
    )
    obs_positions = content_positions.index_select(0, selected_content_pos).contiguous()
    q_content_obs = query_content.index_select(1, selected_content_pos).contiguous()
    if str(config.observation_mode) == "snap_tail" or str(config.selector_mode) in {"snap", "cake", "jaoc_anchor"}:
        q_obs = rope_query_content_at_original_positions(
            model=model,
            query_content=q_content_obs,
            positions=obs_positions,
        ).to(device=prompt_keys.device, dtype=prompt_keys.dtype).contiguous()
        use_causal_obs_mask = True
    else:
        q_obs = rope_query_content_at_future_position(
            model=model,
            query_content=q_content_obs,
            future_position=int(prompt_len),
        ).to(device=prompt_keys.device, dtype=prompt_keys.dtype).contiguous()
        use_causal_obs_mask = False

    q_all = None
    if bool(_needs_full_query_content(config)):
        q_all = rope_query_content_at_original_positions(
            model=model,
            query_content=query_content,
            positions=content_positions,
        ).to(device=prompt_keys.device, dtype=prompt_keys.dtype).contiguous()

    force_mask = build_force_keep_mask(
        seq_len=int(prompt_len),
        force_sink=int(config.force_sink),
        force_recent=int(config.force_recent),
        force_prefix=int(config.force_prefix),
        device=prompt_keys.device,
    )
    budget = max(int(budget), int(force_mask.sum().item()), 1)
    if int(budget) >= int(prompt_len):
        debug_item = {
            "layer_idx": int(layer_idx),
            "seq_len": int(prompt_len),
            "obs_count": int(q_obs.shape[1]),
            "budget": int(budget),
            "kept": int(prompt_len),
            "keep_ratio": 1.0,
            "forced": int(force_mask.sum().item()),
            "observation_source": str(config.observation_mode),
            "use_causal_obs_mask": bool(use_causal_obs_mask),
            "metric_mode": str(config.metric_mode),
            "key_metric_mode": str(getattr(config, "key_metric_mode", "raw")),
            "risk_mode": str(config.risk_mode),
            "selector_mode": str(config.selector_mode),
            "cache_head_mode": str(config.cache_head_mode),
            "solver": "full",
            "select_sec": 0.0,
        }
        return prompt_keys.contiguous(), prompt_values.contiguous(), None, debug_item

    num_kv_heads = int(prompt_keys.shape[1])
    num_groups = int(model.config.num_attention_heads // max(num_kv_heads, 1))
    select_start = time.perf_counter()
    metric_diag = None
    if str(config.metric_mode) == "oproj_diag":
        metric_diag = _get_oproj_metric_diag_cached(
            model=model,
            layer_idx=int(layer_idx),
            num_heads=int(model.config.num_attention_heads),
            head_dim=int(prompt_keys.shape[-1]),
            device=prompt_keys.device,
        )
    compressed_keys, compressed_values, selection_debug = _select_and_compress_layer(
        q_obs=q_obs,
        q_all=q_all,
        keys=prompt_keys,
        values=prompt_values,
        budget=int(budget),
        force_mask=force_mask,
        num_key_value_groups=int(num_groups),
        obs_positions=obs_positions,
        config=config,
        metric_diag=metric_diag,
        spectral_importance=None,
        profile=profile,
        use_causal_obs_mask=bool(use_causal_obs_mask),
    )
    log_bias = selection_debug.pop("_strictmerge_log_bias", None)
    select_sec = float(time.perf_counter() - select_start)
    debug_item = {
        "layer_idx": int(layer_idx),
        "seq_len": int(prompt_len),
        "obs_count": int(q_obs.shape[1]),
        "budget": int(budget),
        "kept": int(selection_debug["kept"]),
        "keep_ratio": float(int(selection_debug["kept"]) / max(int(prompt_len), 1)),
        "forced": int(force_mask.sum().item()),
        "observation_source": str(config.observation_mode),
        "use_causal_obs_mask": bool(use_causal_obs_mask),
        "metric_mode": str(config.metric_mode),
        "key_metric_mode": str(getattr(config, "key_metric_mode", "raw")),
        "risk_mode": str(config.risk_mode),
        "atom_size": int(config.atom_size),
        "selector_mode": str(config.selector_mode),
        "cache_head_mode": str(config.cache_head_mode),
        "spectral_precompute_reused": False,
        "select_sec": float(select_sec),
    }
    debug_item.update(selection_debug)
    return compressed_keys, compressed_values, log_bias, debug_item


@torch.no_grad()
def compress_prompt_layer_strictmerge_batched_static(
    *,
    model,
    layer_idx: int,
    prompt_keys: torch.Tensor,
    prompt_values: torch.Tensor,
    query_content_states: torch.Tensor,
    content_positions: torch.Tensor,
    config: StrictMergeConfig,
    budgets: list[int],
    profile: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    """Compress one prefill layer for a rectangular batch using static budgets.

    This is the forward-integrated batched counterpart of
    ``compress_single_prompt_layer_strictmerge``. It is intentionally scoped to
    the SpectralKV main path so deployment profiling cannot silently fall back
    to full-prefill-then-compress behavior.
    """

    config.validate()
    if not _batched_spectral_supported(config, "fixed"):
        raise NotImplementedError(
            "Batched integrated static SpectralKV currently supports the main "
            "spectral/dynamic CSD KV selector only."
        )
    if _needs_full_query_content(config):
        raise NotImplementedError("Batched integrated static SpectralKV does not support full-query-content selectors.")
    if int(prompt_keys.dim()) != 4 or int(prompt_values.dim()) != 4:
        raise ValueError("prompt_keys/prompt_values must have shape [B,H,K,D].")
    if tuple(prompt_keys.shape) != tuple(prompt_values.shape):
        raise ValueError("prompt_keys and prompt_values must have identical shapes.")
    batch_size = int(prompt_keys.shape[0])
    hkv = int(prompt_keys.shape[1])
    prompt_len = int(prompt_keys.shape[-2])
    head_dim = int(prompt_keys.shape[-1])
    if len(budgets) != batch_size:
        raise ValueError(f"budgets length {len(budgets)} does not match batch size {batch_size}.")
    if int(query_content_states.dim()) != 4:
        raise ValueError(f"query_content_states must be [B,Hq,R,D], got {tuple(query_content_states.shape)}")
    if int(query_content_states.shape[0]) != batch_size:
        raise ValueError("query_content_states batch mismatch.")
    if int(query_content_states.shape[-1]) != head_dim:
        raise ValueError("query_content_states head_dim mismatch.")
    obs_count = int(query_content_states.shape[2])
    if obs_count <= 0:
        raise ValueError("query_content_states must contain at least one observation token.")

    pos = content_positions.to(device=prompt_keys.device, dtype=torch.long)
    if int(pos.dim()) == 1:
        if int(pos.numel()) != obs_count:
            raise ValueError("1D content_positions must match observation count.")
        pos = pos.unsqueeze(0).expand(batch_size, obs_count)
    elif int(pos.dim()) == 2:
        if int(pos.shape[0]) == 1 and int(batch_size) > 1 and int(pos.shape[1]) == obs_count:
            pos = pos.expand(batch_size, obs_count)
        if int(pos.shape[0]) != batch_size or int(pos.shape[1]) != obs_count:
            raise ValueError(
                f"content_positions must have shape [{batch_size},{obs_count}], got {tuple(pos.shape)}."
            )
    else:
        raise ValueError(f"content_positions must be 1D or 2D, got {tuple(pos.shape)}.")

    use_causal_obs_mask = str(config.observation_mode) == "snap_tail" or str(config.selector_mode) in {"snap", "cake", "jaoc_anchor"}
    if str(config.observation_mode) == "snap_tail" or str(config.selector_mode) in {"snap", "cake", "jaoc_anchor"}:
        raise NotImplementedError("Batched integrated static path is implemented for default da_tail observation.")
    q_obs = rope_query_content_batch_at_future_position(
        model=model,
        query_content=query_content_states.to(device=prompt_keys.device, dtype=prompt_keys.dtype).contiguous(),
        future_position=int(prompt_len),
    )

    force_mask = build_force_keep_mask(
        seq_len=int(prompt_len),
        force_sink=int(config.force_sink),
        force_recent=int(config.force_recent),
        force_prefix=int(config.force_prefix),
        device=prompt_keys.device,
    )
    force_count = int(force_mask.sum().item())
    targets = [min(max(int(budget), int(force_count), 1), int(prompt_len)) for budget in budgets]
    if all(int(target) >= int(prompt_len) for target in targets):
        debug_item = {
            "layer_idx": int(layer_idx),
            "seq_len": int(prompt_len),
            "obs_count": int(obs_count),
            "budget": int(max(targets) if targets else prompt_len),
            "kept": int(prompt_len),
            "keep_ratio": 1.0,
            "forced": int(force_count),
            "observation_source": str(config.observation_mode),
            "use_causal_obs_mask": bool(use_causal_obs_mask),
            "metric_mode": str(config.metric_mode),
            "key_metric_mode": str(getattr(config, "key_metric_mode", "raw")),
            "risk_mode": str(config.risk_mode),
            "selector_mode": str(config.selector_mode),
            "cache_head_mode": str(config.cache_head_mode),
            "solver": "full",
            "select_sec": 0.0,
            "batch_size": int(batch_size),
            "batched_integrated_static": True,
        }
        return prompt_keys.contiguous(), prompt_values.contiguous(), None, debug_item

    groups = int(model.config.num_attention_heads // max(hkv, 1))
    recent = min(max(int(config.force_recent), 0), prompt_len)
    old_len = int(prompt_len) - int(recent)
    seq_lens = [int(prompt_len)] * int(batch_size)
    old_lens = [int(old_len)] * int(batch_size)
    force_batch = force_mask.view(1, prompt_len).expand(batch_size, prompt_len).contiguous()

    select_start = time.perf_counter()
    importance_batch = _spectral_csd_importance_scores_batched(
        q_obs=q_obs,
        keys=prompt_keys,
        old_lens=old_lens,
        seq_lens=seq_lens,
        num_key_value_groups=int(groups),
        config=config,
    )
    metric_diag = None
    if str(config.metric_mode) == "oproj_diag":
        metric_diag = _get_oproj_metric_diag_cached(
            model=model,
            layer_idx=int(layer_idx),
            num_heads=int(model.config.num_attention_heads),
            head_dim=int(head_dim),
            device=prompt_keys.device,
        )
    value_metrics = None
    if metric_diag is not None:
        per_head_metrics: list[torch.Tensor] = []
        for head in range(hkv):
            q_start = int(head) * int(groups)
            q_end = q_start + int(groups)
            per_head_metrics.append(metric_diag[q_start:q_end].float().mean(dim=0))
        value_metrics = torch.stack(per_head_metrics, dim=0).contiguous()
    key_metrics_batch = None
    if str(getattr(config, "key_metric_mode", "raw")) == "qcov_diag":
        key_metrics_batch = _qcov_key_metrics_from_qobs_batched(
            q_obs=q_obs,
            hkv=int(hkv),
            groups=int(groups),
            head_dim=int(head_dim),
        )

    keep_by_sample = _chunkwise_keep_spectral_csd_kv_batched_samples(
        keys=prompt_keys,
        values=prompt_values,
        old_lens=old_lens,
        seq_lens=seq_lens,
        targets=targets,
        force_masks=force_batch,
        chunk_size=int(config.local_chunk_size),
        config=config,
        value_metrics=value_metrics,
        key_metrics=key_metrics_batch,
        importance=importance_batch,
        profile=profile,
    )
    select_sec = float(time.perf_counter() - select_start)

    kept_counts = [int(item.shape[1]) for item in keep_by_sample]
    if len(set(kept_counts)) != 1:
        raise RuntimeError(f"Batched integrated static requires rectangular retained lengths, got {kept_counts}.")
    kept_count = int(kept_counts[0])
    keep_idx = torch.stack([item.to(device=prompt_keys.device, dtype=torch.long) for item in keep_by_sample], dim=0)
    gather_idx = keep_idx.view(batch_size, hkv, kept_count, 1).expand(batch_size, hkv, kept_count, head_dim)
    compressed_keys = prompt_keys.gather(2, gather_idx).contiguous()
    compressed_values = prompt_values.gather(2, gather_idx).contiguous()
    solver = (
        f"local_jaoc_{'dynamic_csd_kv' if _is_dynamic_csd_kv_mode(config) else 'spectral_csd_kv'}_chunk{int(config.local_chunk_size)}"
        f"_kw{float(config.coreset_key_weight):g}"
        f"_vw{float(config.coreset_value_weight):g}"
        f"_{'globalresidual' if _is_dynamic_csd_kv_mode(config) else 'waterfill'}_sqrtp_minchunk{int(config.operator_min_chunk_keep)}"
        f"{'_block' + str(int(config.dynamic_csd_block_size)) if _is_dynamic_csd_kv_mode(config) else ''}"
        f"_{str(config.snap_pooling)}k{int(config.snap_kernel_size)}"
        f"_{_key_metric_label(config, key_metrics_batch)}"
        f"_{'oprojdiag' if metric_diag is not None else 'rawv'}"
    )
    debug_item = {
        "layer_idx": int(layer_idx),
        "seq_len": int(prompt_len),
        "obs_count": int(obs_count),
        "budget": int(targets[0]) if targets else 0,
        "kept": int(kept_count),
        "keep_ratio": float(int(kept_count) / max(int(prompt_len), 1)),
        "forced": int(force_count),
        "observation_source": str(config.observation_mode),
        "use_causal_obs_mask": bool(use_causal_obs_mask),
        "metric_mode": str(config.metric_mode),
        "key_metric_mode": str(getattr(config, "key_metric_mode", "raw")),
        "risk_mode": str(config.risk_mode),
        "atom_size": int(config.atom_size),
        "selector_mode": str(config.selector_mode),
        "cache_head_mode": str(config.cache_head_mode),
        "selection_granularity": "kv_head",
        "solver": solver,
        "final_loss": 0.0,
        "min_denominator": 1.0,
        "coreset_key_weight": float(config.coreset_key_weight),
        "coreset_value_weight": float(config.coreset_value_weight),
        "coreset_tau": float(config.coreset_tau),
        "coreset_knn": int(config.coreset_knn),
        "coreset_key_feature": _key_metric_label(config, key_metrics_batch),
        "coreset_value_feature": "oproj_diag" if metric_diag is not None else "raw_value",
        "operator_real_probe_limit": int(config.operator_real_probe_limit),
        "operator_key_probe_limit": int(config.operator_key_probe_limit),
        "operator_probe_source": str(config.operator_probe_source),
        "operator_mass_uniform_mix": float(config.operator_mass_uniform_mix),
        "operator_min_chunk_keep": int(config.operator_min_chunk_keep),
        "old_budget": int(max(int(targets[0]) - int(force_batch[0, old_len:].sum().item()), 0)) if targets else 0,
        "recent": int(recent),
        "rounds": 0,
        "batched_kv_heads": True,
        "batched_samples": True,
        "batched_integrated_static": True,
        "batch_size": int(batch_size),
        "select_sec": float(select_sec),
        "per_sample_select_sec": float(select_sec / max(int(batch_size), 1)),
    }
    return compressed_keys, compressed_values, None, debug_item


@torch.no_grad()
def compress_prompt_cache_strictmerge_sample_batched(
    *,
    model,
    prompt_past_key_values_list: list[Any],
    query_states: list[StrictQueryContentState],
    config: StrictMergeConfig,
    budget_mode: str = "attention_entropy",
    fixed_layer_budgets_by_sample: list[list[int]] | None = None,
    profile: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    if not prompt_past_key_values_list:
        return []
    config.validate()
    if str(budget_mode) not in {"attention_entropy", "fixed"}:
        raise ValueError("SpectralKV batched budget_mode must be attention_entropy or fixed.")
    if not _batched_spectral_supported(config, str(budget_mode)):
        raise ValueError("SpectralKV batched compression supports only the current spectral CSD KV mainline.")

    batch_size = len(prompt_past_key_values_list)
    if batch_size != len(query_states):
        raise ValueError("prompt_past_key_values_list and query_states must have the same length.")
    if fixed_layer_budgets_by_sample is not None and len(fixed_layer_budgets_by_sample) != batch_size:
        raise ValueError("fixed_layer_budgets_by_sample must have the same batch size as prompt_past_key_values_list.")
    layer_lists = [extract_cache_layers(cache) for cache in prompt_past_key_values_list]
    num_layers = len(layer_lists[0])
    if any(len(layers) != num_layers for layers in layer_lists):
        raise ValueError("All sample caches must have the same number of layers.")
    if fixed_layer_budgets_by_sample is not None:
        for sample_idx, budgets in enumerate(fixed_layer_budgets_by_sample):
            if len(budgets) != num_layers:
                raise ValueError(
                    f"fixed_layer_budgets_by_sample[{sample_idx}] has {len(budgets)} layers but expected {num_layers}."
                )

    layer_precompute: list[dict[int, dict[str, Any]]] = [dict() for _ in range(batch_size)]
    seq_lens_by_sample: list[list[int]] = [[] for _ in range(batch_size)]
    force_counts_by_sample: list[list[int]] = [[] for _ in range(batch_size)]
    entropy_scores_by_sample: list[list[float]] = [[] for _ in range(batch_size)]

    for layer_idx in range(num_layers):
        prompt_keys0 = layer_lists[0][int(layer_idx)][0]
        hkv = int(prompt_keys0.shape[1])
        groups = int(model.config.num_attention_heads // max(hkv, 1))
        metric_diag = None
        if str(config.metric_mode) == "oproj_diag":
            metric_diag = _get_oproj_metric_diag_cached(
                model=model,
                layer_idx=int(layer_idx),
                num_heads=int(model.config.num_attention_heads),
                head_dim=int(prompt_keys0.shape[-1]),
                device=prompt_keys0.device,
            )
        value_metrics = None
        if metric_diag is not None:
            per_head_metrics: list[torch.Tensor] = []
            for head in range(hkv):
                q_start = int(head) * int(groups)
                q_end = q_start + int(groups)
                per_head_metrics.append(metric_diag[q_start:q_end].float().mean(dim=0))
            value_metrics = torch.stack(per_head_metrics, dim=0).contiguous()
        key_metrics_for_budget: torch.Tensor | None = None

        seq_lens: list[int] = []
        old_lens: list[int] = []
        keys_for_batch: list[torch.Tensor] = []
        values_for_batch: list[torch.Tensor] = []
        force_for_batch: list[torch.Tensor] = []
        qobs_for_batch: list[torch.Tensor] = []
        active_sample_indices: list[int] = []
        for sample_idx in range(batch_size):
            prompt_keys, prompt_values = layer_lists[int(sample_idx)][int(layer_idx)]
            if int(prompt_keys.shape[0]) != 1:
                raise NotImplementedError("StrictMerge batched compression expects per-sample batch_size=1 caches.")
            seq_len = int(prompt_keys.shape[-2])
            seq_lens_by_sample[int(sample_idx)].append(int(seq_len))
            force_mask = build_force_keep_mask(
                seq_len=int(seq_len),
                force_sink=int(config.force_sink),
                force_recent=int(config.force_recent),
                force_prefix=int(config.force_prefix),
                device=prompt_keys.device,
            )
            force_count = int(force_mask.sum().item())
            force_counts_by_sample[int(sample_idx)].append(int(force_count))
            recent = min(max(int(config.force_recent), 0), seq_len)
            old_len = seq_len - recent
            if old_len <= 0:
                entropy_scores_by_sample[int(sample_idx)].append(0.0)
                continue
            stage_t = _profile_start(prompt_keys.device, profile)
            q_obs, obs_positions = _prompt_qobs_for_layer(
                model=model,
                query_state=query_states[int(sample_idx)],
                layer_idx=int(layer_idx),
                prompt_keys=prompt_keys,
                prompt_len=int(seq_len),
                config=config,
            )
            _profile_add(profile, "budget_batched_qobs_sec", stage_t, prompt_keys.device)
            use_causal_obs_mask = False
            entropy_score = 0.0
            if str(budget_mode) == "attention_entropy":
                stage_t = _profile_start(prompt_keys.device, profile)
                entropy_score = _attention_entropy_layer_score(
                    q_obs=q_obs,
                    keys=prompt_keys,
                    old_len=int(old_len),
                    num_key_value_groups=int(groups),
                    force_mask=force_mask,
                )
                _profile_add(profile, "budget_batched_attention_entropy_sec", stage_t, prompt_keys.device)
            layer_precompute[int(sample_idx)][int(layer_idx)] = {
                "q_obs": q_obs.detach().contiguous(),
                "obs_positions": obs_positions.detach().contiguous(),
                "use_causal_obs_mask": bool(use_causal_obs_mask),
                "attention_entropy_score": float(entropy_score),
            }
            entropy_scores_by_sample[int(sample_idx)].append(float(entropy_score))
            seq_lens.append(int(seq_len))
            old_lens.append(int(old_len))
            keys_for_batch.append(prompt_keys[0])
            values_for_batch.append(prompt_values[0])
            force_for_batch.append(force_mask)
            qobs_for_batch.append(q_obs)
            active_sample_indices.append(int(sample_idx))

        if keys_for_batch:
            max_seq = max(int(v) for v in seq_lens)
            max_old = max(int(v) for v in old_lens)
            head_dim = int(keys_for_batch[0].shape[-1])
            keys_batch = torch.zeros((len(keys_for_batch), hkv, max_seq, head_dim), dtype=keys_for_batch[0].dtype, device=keys_for_batch[0].device)
            values_batch = torch.zeros_like(keys_batch)
            force_batch = torch.zeros((len(keys_for_batch), max_seq), dtype=torch.bool, device=keys_for_batch[0].device)
            for local_idx, (keys_i, values_i, force_i) in enumerate(zip(keys_for_batch, values_for_batch, force_for_batch)):
                seq_len = int(seq_lens[int(local_idx)])
                keys_batch[int(local_idx), :, :seq_len, :] = keys_i[:, :seq_len, :]
                values_batch[int(local_idx), :, :seq_len, :] = values_i[:, :seq_len, :]
                force_batch[int(local_idx), :seq_len] = force_i[:seq_len]
            stage_t = _profile_start(keys_batch.device, profile)
            same_qobs_shape = len({tuple(item.shape) for item in qobs_for_batch}) == 1
            if same_qobs_shape:
                qobs_batch = torch.stack(qobs_for_batch, dim=0).to(device=keys_batch.device, dtype=keys_batch.dtype)
                importance_batch = _spectral_csd_importance_scores_batched(
                    q_obs=qobs_batch,
                    keys=keys_batch,
                    old_lens=old_lens,
                    seq_lens=seq_lens,
                    num_key_value_groups=int(groups),
                    config=config,
                )
                if str(getattr(config, "key_metric_mode", "raw")) == "qcov_diag":
                    key_metrics_for_budget = _qcov_key_metrics_from_qobs_batched(
                        q_obs=qobs_batch,
                        hkv=int(hkv),
                        groups=int(groups),
                        head_dim=int(head_dim),
                    )
            else:
                importance_batch = torch.zeros((len(keys_for_batch), hkv, max_old), dtype=torch.float32, device=keys_batch.device)
                key_metric_items: list[torch.Tensor | None] = []
                for local_idx, (q_obs_i, keys_i) in enumerate(zip(qobs_for_batch, keys_for_batch)):
                    old_len = int(old_lens[int(local_idx)])
                    importance_i = _spectral_csd_importance_scores(
                        q_obs=q_obs_i,
                        keys=keys_i.unsqueeze(0),
                        old_len=int(old_len),
                        num_key_value_groups=int(groups),
                        config=config,
                    )
                    importance_batch[int(local_idx), :, :old_len] = importance_i[:, :old_len]
                    if str(getattr(config, "key_metric_mode", "raw")) == "qcov_diag":
                        key_metric_items.append(
                            _qcov_key_metrics_from_qobs(
                                q_obs=q_obs_i,
                                hkv=int(hkv),
                                groups=int(groups),
                                head_dim=int(head_dim),
                            )
                        )
                if key_metric_items:
                    if all(item is not None for item in key_metric_items):
                        key_metrics_for_budget = torch.stack([item for item in key_metric_items if item is not None], dim=0).contiguous()
            _profile_add(profile, "budget_batched_importance_sec", stage_t, keys_batch.device)
            for local_idx, sample_idx in enumerate(active_sample_indices):
                old_len = int(old_lens[int(local_idx)])
                layer_precompute[int(sample_idx)][int(layer_idx)]["importance"] = (
                    importance_batch[int(local_idx), :, :old_len].detach().contiguous()
                )
    fixed_budgets_by_sample: list[list[int] | None] = [None for _ in range(batch_size)]
    if fixed_layer_budgets_by_sample is not None:
        fixed_budgets_by_sample = [list(map(int, budgets)) for budgets in fixed_layer_budgets_by_sample]
    elif str(budget_mode) == "attention_entropy":
        fixed_budgets_by_sample = [
            _attention_entropy_layer_budgets(
                seq_lens=seq_lens_by_sample[int(sample_idx)],
                force_counts=force_counts_by_sample[int(sample_idx)],
                layer_scores=entropy_scores_by_sample[int(sample_idx)],
                config=config,
            )
            for sample_idx in range(batch_size)
        ]
    elif str(budget_mode) == "fixed":
        raise ValueError("fixed budget mode requires fixed_layer_budgets_by_sample.")

    debug_layers_by_sample: list[list[dict[str, Any]]] = [[] for _ in range(batch_size)]
    for layer_idx in range(num_layers):
        prompt_keys0 = layer_lists[0][int(layer_idx)][0]
        hkv = int(prompt_keys0.shape[1])
        groups = int(model.config.num_attention_heads // max(hkv, 1))
        metric_diag = None
        if str(config.metric_mode) == "oproj_diag":
            metric_diag = _get_oproj_metric_diag_cached(
                model=model,
                layer_idx=int(layer_idx),
                num_heads=int(model.config.num_attention_heads),
                head_dim=int(prompt_keys0.shape[-1]),
                device=prompt_keys0.device,
            )
        value_metrics = None
        if metric_diag is not None:
            per_head_metrics = []
            for head in range(hkv):
                q_start = int(head) * int(groups)
                q_end = q_start + int(groups)
                per_head_metrics.append(metric_diag[q_start:q_end].float().mean(dim=0))
            value_metrics = torch.stack(per_head_metrics, dim=0).contiguous()

        active_samples: list[int] = []
        seq_lens: list[int] = []
        old_lens: list[int] = []
        targets: list[int] = []
        keys_for_batch: list[torch.Tensor] = []
        values_for_batch: list[torch.Tensor] = []
        force_for_batch: list[torch.Tensor] = []
        importance_for_batch: list[torch.Tensor] = []
        qobs_for_select_batch: list[torch.Tensor] = []
        select_start = time.perf_counter()
        for sample_idx in range(batch_size):
            prompt_keys, prompt_values = layer_lists[int(sample_idx)][int(layer_idx)]
            prompt_len = int(prompt_keys.shape[-2])
            if prompt_len <= 1:
                continue
            force_mask = build_force_keep_mask(
                seq_len=int(prompt_len),
                force_sink=int(config.force_sink),
                force_recent=int(config.force_recent),
                force_prefix=int(config.force_prefix),
                device=prompt_keys.device,
            )
            if fixed_budgets_by_sample[int(sample_idx)] is not None:
                budget = max(int(fixed_budgets_by_sample[int(sample_idx)][int(layer_idx)]), int(force_mask.sum().item()), 1)
            elif int(config.fixed_budget) > 0:
                budget = max(int(config.fixed_budget), int(force_mask.sum().item()), 1)
            else:
                budget = uniform_layer_budget(
                    seq_len=int(prompt_len),
                    keep_ratio=float(config.keep_ratio),
                    force_keep_count=int(force_mask.sum().item()),
                )
            if int(budget) >= int(prompt_len):
                continue
            cached_obs = layer_precompute[int(sample_idx)].get(int(layer_idx))
            if cached_obs is None:
                q_obs, obs_positions = _prompt_qobs_for_layer(
                    model=model,
                    query_state=query_states[int(sample_idx)],
                    layer_idx=int(layer_idx),
                    prompt_keys=prompt_keys,
                    prompt_len=int(prompt_len),
                    config=config,
                )
                use_causal_obs_mask = str(config.observation_mode) == "snap_tail" or str(config.selector_mode) in {"snap", "cake", "jaoc_anchor"}
                recent = min(max(int(config.force_recent), 0), prompt_len)
                old_len = prompt_len - recent
                importance = _spectral_csd_importance_scores(
                    q_obs=q_obs,
                    keys=prompt_keys,
                    old_len=int(old_len),
                    num_key_value_groups=int(groups),
                    config=config,
                )
                cached_obs = {
                    "q_obs": q_obs.detach().contiguous(),
                    "obs_positions": obs_positions.detach().contiguous(),
                    "use_causal_obs_mask": bool(use_causal_obs_mask),
                    "importance": importance.detach().contiguous(),
                }
                layer_precompute[int(sample_idx)][int(layer_idx)] = cached_obs
            recent = min(max(int(config.force_recent), 0), prompt_len)
            old_len = prompt_len - recent
            active_samples.append(int(sample_idx))
            seq_lens.append(int(prompt_len))
            old_lens.append(int(old_len))
            targets.append(int(budget))
            keys_for_batch.append(prompt_keys[0])
            values_for_batch.append(prompt_values[0])
            force_for_batch.append(force_mask)
            importance_for_batch.append(cached_obs["importance"].to(device=prompt_keys.device, dtype=torch.float32))
            qobs_for_select_batch.append(cached_obs["q_obs"].to(device=prompt_keys.device, dtype=prompt_keys.dtype))

        if not active_samples:
            continue

        max_seq = max(int(v) for v in seq_lens)
        max_old = max(int(v) for v in old_lens)
        head_dim = int(keys_for_batch[0].shape[-1])
        keys_batch = torch.zeros((len(active_samples), hkv, max_seq, head_dim), dtype=keys_for_batch[0].dtype, device=keys_for_batch[0].device)
        values_batch = torch.zeros_like(keys_batch)
        force_batch = torch.zeros((len(active_samples), max_seq), dtype=torch.bool, device=keys_for_batch[0].device)
        importance_batch = torch.zeros((len(active_samples), hkv, max_old), dtype=torch.float32, device=keys_for_batch[0].device)
        key_metrics_batch = None
        if str(getattr(config, "key_metric_mode", "raw")) == "qcov_diag":
            same_qobs_shape = len({tuple(item.shape) for item in qobs_for_select_batch}) == 1
            if same_qobs_shape:
                qobs_batch = torch.stack(qobs_for_select_batch, dim=0).to(device=keys_for_batch[0].device, dtype=keys_for_batch[0].dtype)
                key_metrics_batch = _qcov_key_metrics_from_qobs_batched(
                    q_obs=qobs_batch,
                    hkv=int(hkv),
                    groups=int(groups),
                    head_dim=int(head_dim),
                )
            else:
                key_metric_items = [
                    _qcov_key_metrics_from_qobs(
                        q_obs=q_obs_i,
                        hkv=int(hkv),
                        groups=int(groups),
                        head_dim=int(head_dim),
                    )
                    for q_obs_i in qobs_for_select_batch
                ]
                if all(item is not None for item in key_metric_items):
                    key_metrics_batch = torch.stack([item for item in key_metric_items if item is not None], dim=0).contiguous()
        for local_idx, (keys_i, values_i, force_i, importance_i) in enumerate(zip(keys_for_batch, values_for_batch, force_for_batch, importance_for_batch)):
            seq_len = int(seq_lens[int(local_idx)])
            old_len = int(old_lens[int(local_idx)])
            keys_batch[int(local_idx), :, :seq_len, :] = keys_i[:, :seq_len, :]
            values_batch[int(local_idx), :, :seq_len, :] = values_i[:, :seq_len, :]
            force_batch[int(local_idx), :seq_len] = force_i[:seq_len]
            importance_batch[int(local_idx), :, :old_len] = importance_i[:, :old_len]

        keep_by_sample = _chunkwise_keep_spectral_csd_kv_batched_samples(
            keys=keys_batch,
            values=values_batch,
            old_lens=old_lens,
            seq_lens=seq_lens,
            targets=targets,
            force_masks=force_batch,
            chunk_size=int(config.local_chunk_size),
            config=config,
            value_metrics=value_metrics,
            key_metrics=key_metrics_batch,
            importance=importance_batch,
            profile=profile,
        )
        select_sec = float(time.perf_counter() - select_start)
        for local_idx, sample_idx in enumerate(active_samples):
            prompt_keys, prompt_values = layer_lists[int(sample_idx)][int(layer_idx)]
            keep_by_head = keep_by_sample[int(local_idx)].to(device=prompt_keys.device, dtype=torch.long)
            kept_count = int(keep_by_head.shape[1])
            gather_idx = keep_by_head.view(1, hkv, kept_count, 1).expand(1, hkv, kept_count, int(prompt_keys.shape[-1]))
            compressed_keys = prompt_keys.gather(2, gather_idx).contiguous()
            compressed_values = prompt_values.gather(2, gather_idx).contiguous()
            set_layer_cache(prompt_past_key_values_list[int(sample_idx)], int(layer_idx), compressed_keys, compressed_values)
            cached_obs = layer_precompute[int(sample_idx)][int(layer_idx)]
            prompt_len = int(seq_lens[int(local_idx)])
            force_count = int(force_for_batch[int(local_idx)].sum().item())
            selection_debug = {
                "selection_granularity": "kv_head",
                "selector_mode": "local_jaoc",
                "cache_head_mode": str(config.cache_head_mode),
                "kept": int(kept_count),
                "old_budget": int(
                    max(
                        int(targets[int(local_idx)])
                        - int(force_for_batch[int(local_idx)][int(old_lens[int(local_idx)]) :].sum().item()),
                        0,
                    )
                ),
                "recent": int(min(max(int(config.force_recent), 0), prompt_len)),
                "rounds": 0,
                "solver": (
                    f"local_jaoc_{'dynamic_csd_kv' if _is_dynamic_csd_kv_mode(config) else 'spectral_csd_kv'}_chunk{int(config.local_chunk_size)}"
                    f"_kw{float(config.coreset_key_weight):g}"
                    f"_vw{float(config.coreset_value_weight):g}"
                    f"_{'globalresidual' if _is_dynamic_csd_kv_mode(config) else 'waterfill'}_sqrtp_minchunk{int(config.operator_min_chunk_keep)}"
                    f"{'_block' + str(int(config.dynamic_csd_block_size)) if _is_dynamic_csd_kv_mode(config) else ''}"
                    f"_{str(config.snap_pooling)}k{int(config.snap_kernel_size)}"
                    f"_{_key_metric_label(config, key_metrics_batch)}"
                    f"_{'oprojdiag' if metric_diag is not None else 'rawv'}"
                ),
                "final_loss": 0.0,
                "min_denominator": 1.0,
                "coreset_key_weight": float(config.coreset_key_weight),
                "coreset_value_weight": float(config.coreset_value_weight),
                "coreset_tau": float(config.coreset_tau),
                "coreset_knn": int(config.coreset_knn),
                "coreset_key_feature": _key_metric_label(config, key_metrics_batch),
                "coreset_value_feature": "oproj_diag" if metric_diag is not None else "raw_value",
                "operator_real_probe_limit": int(config.operator_real_probe_limit),
                "operator_key_probe_limit": int(config.operator_key_probe_limit),
                "operator_probe_source": str(config.operator_probe_source),
                "operator_mass_uniform_mix": float(config.operator_mass_uniform_mix),
                "operator_min_chunk_keep": int(config.operator_min_chunk_keep),
                "batched_kv_heads": True,
                "batched_samples": True,
            }
            debug_item = {
                "layer_idx": int(layer_idx),
                "seq_len": int(prompt_len),
                "obs_count": int(cached_obs["q_obs"].shape[1]),
                "budget": int(targets[int(local_idx)]),
                "kept": int(kept_count),
                "keep_ratio": float(int(kept_count) / max(int(prompt_len), 1)),
                "forced": int(force_count),
                "observation_source": str(config.observation_mode),
                "use_causal_obs_mask": bool(cached_obs.get("use_causal_obs_mask", False)),
                "metric_mode": str(config.metric_mode),
                "key_metric_mode": str(getattr(config, "key_metric_mode", "raw")),
                "risk_mode": str(config.risk_mode),
                "atom_size": int(config.atom_size),
                "selector_mode": str(config.selector_mode),
                "cache_head_mode": str(config.cache_head_mode),
                "spectral_precompute_reused": True,
                "select_sec": float(select_sec / max(len(active_samples), 1)),
                "batch_select_sec": float(select_sec),
            }
            debug_item.update(selection_debug)
            debug_layers_by_sample[int(sample_idx)].append(debug_item)

    debug_infos: list[dict[str, Any]] = []
    for sample_idx, layers_debug in enumerate(debug_layers_by_sample):
        debug: dict[str, Any] = {
            "patch": "strictmerge",
            "budget_mode": str(budget_mode),
            "config": config.__dict__,
            "layers": list(layers_debug),
            "total_select_sec": float(sum(float(layer.get("select_sec", 0.0)) for layer in layers_debug)),
            "batched_prefill": True,
            "batched_compression": True,
        }
        if profile is not None:
            debug["profile"] = dict(sorted(profile.items()))
        debug_infos.append(debug)
    return debug_infos


@torch.no_grad()
def compress_cache_with_qobs_strictmerge(
    *,
    model,
    past_key_values,
    layer_qobs: dict[int, tuple[torch.Tensor, torch.Tensor]],
    prompt_len: int,
    config: StrictMergeConfig,
    budget_mode: str = "fixed",
    fixed_layer_budgets: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Compress a prompt+generated cache using already observed decode queries.

    This is the NDQ path: the first generated token is part of the natural
    decode trajectory, not a branch. Its K/V is forced to remain, while prompt
    tokens compete under the same JAOC value-space objective.
    """

    config.validate()
    if str(budget_mode) != "fixed":
        raise ValueError("SpectralKV NDQ compression supports only fixed budget_mode.")
    if fixed_layer_budgets is None:
        raise ValueError("fixed budget mode requires fixed_layer_budgets.")

    layers = extract_cache_layers(past_key_values)
    debug: list[dict[str, Any]] = []
    for layer_idx, (keys, values) in enumerate(layers):
        if int(keys.shape[0]) != 1:
            raise NotImplementedError("StrictMerge NDQ supports batch_size=1.")
        score_len = int(keys.shape[-2])
        prompt_len_i = int(prompt_len)
        generated_context = max(int(score_len) - int(prompt_len_i), 0)
        if prompt_len_i <= 1 or score_len < prompt_len_i:
            continue
        q_item = layer_qobs.get(int(layer_idx))
        if q_item is None:
            raise RuntimeError(f"No NDQ decode query recorded for layer {layer_idx}.")
        q_obs, obs_positions = q_item
        q_obs = q_obs.to(device=keys.device, dtype=keys.dtype).contiguous()
        obs_positions = obs_positions.to(device=keys.device, dtype=torch.long).contiguous()

        prompt_force = build_force_keep_mask(
            seq_len=int(prompt_len_i),
            force_sink=int(config.force_sink),
            force_recent=int(config.force_recent),
            force_prefix=int(config.force_prefix),
            device=keys.device,
        )
        generated_force = torch.ones((generated_context,), dtype=torch.bool, device=keys.device)
        force_mask = torch.cat([prompt_force, generated_force], dim=0)

        if fixed_layer_budgets is not None:
            prompt_budget = max(int(fixed_layer_budgets[int(layer_idx)]), int(prompt_force.sum().item()), 1)
        elif int(config.fixed_budget) > 0:
            prompt_budget = max(int(config.fixed_budget), int(prompt_force.sum().item()), 1)
        else:
            prompt_budget = uniform_layer_budget(
                seq_len=int(prompt_len_i),
                keep_ratio=float(config.keep_ratio),
                force_keep_count=int(prompt_force.sum().item()),
            )
        score_budget = int(prompt_budget) + int(generated_context)
        if int(prompt_budget) >= int(prompt_len_i):
            continue

        num_kv_heads = int(keys.shape[1])
        num_groups = int(model.config.num_attention_heads // max(num_kv_heads, 1))
        select_start = time.perf_counter()
        metric_diag = None
        if str(config.metric_mode) == "oproj_diag":
            metric_diag = _get_oproj_metric_diag_cached(
                model=model,
                layer_idx=int(layer_idx),
                num_heads=int(model.config.num_attention_heads),
                head_dim=int(keys.shape[-1]),
                device=keys.device,
            )
        compressed_keys, compressed_values, selection_debug = _select_and_compress_layer(
            q_obs=q_obs,
            q_all=None,
            keys=keys,
            values=values,
            budget=int(score_budget),
            force_mask=force_mask,
            num_key_value_groups=int(num_groups),
            obs_positions=obs_positions,
            config=config,
            metric_diag=metric_diag,
        )
        log_bias = selection_debug.pop("_strictmerge_log_bias", None)
        select_sec = float(time.perf_counter() - select_start)
        set_layer_cache(past_key_values, int(layer_idx), compressed_keys, compressed_values)
        if isinstance(log_bias, torch.Tensor):
            set_layer_log_bias(past_key_values, int(layer_idx), log_bias)
        kept_prompt = int(selection_debug["kept"]) - int(generated_context)
        debug_item = {
                "layer_idx": int(layer_idx),
                "seq_len": int(prompt_len_i),
                "score_seq_len": int(score_len),
                "generated_context": int(generated_context),
                "obs_count": int(q_obs.shape[1]),
                "budget": int(prompt_budget),
                "score_budget": int(score_budget),
                "kept": int(kept_prompt),
                "score_kept": int(selection_debug["kept"]),
                "keep_ratio": float(int(kept_prompt) / max(int(prompt_len_i), 1)),
                "forced": int(prompt_force.sum().item()),
                "score_forced": int(force_mask.sum().item()),
                "observation_source": "ndq1",
                "use_causal_obs_mask": False,
                "metric_mode": str(config.metric_mode),
                "key_metric_mode": str(getattr(config, "key_metric_mode", "raw")),
                "risk_mode": "mean",
                "atom_size": int(config.atom_size),
                "select_sec": float(select_sec),
        }
        debug_item.update(selection_debug)
        debug_item["kept"] = int(kept_prompt)
        debug_item["score_kept"] = int(selection_debug["kept"])
        debug_item["keep_ratio"] = float(int(kept_prompt) / max(int(prompt_len_i), 1))
        debug.append(debug_item)
    return debug


def summarize_layer_kept(debug_info: dict[str, Any], prompt_len: int) -> dict[str, float | int]:
    layers = list(debug_info.get("layers", []))
    kept_values = [int(layer.get("kept", 0)) for layer in layers if "kept" in layer]
    if not kept_values:
        kept = int(prompt_len)
        return {
            "kept_prompt_tokens": kept,
            "actual_keep_ratio": float(kept / max(int(prompt_len), 1)),
            "layer0_kept_prompt_tokens": kept,
            "min_layer_kept_prompt_tokens": kept,
            "max_layer_kept_prompt_tokens": kept,
            "mean_layer_kept_prompt_tokens": float(kept),
        }
    layer0 = int(kept_values[0])
    mean_kept = float(sum(kept_values) / max(len(kept_values), 1))
    return {
        "kept_prompt_tokens": int(round(mean_kept)),
        "actual_keep_ratio": float(mean_kept / max(int(prompt_len), 1)),
        "layer0_kept_prompt_tokens": layer0,
        "min_layer_kept_prompt_tokens": int(min(kept_values)),
        "max_layer_kept_prompt_tokens": int(max(kept_values)),
        "mean_layer_kept_prompt_tokens": float(mean_kept),
    }


@torch.no_grad()
def prefill_with_strictmerge(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    config: StrictMergeConfig,
    budget_mode: str = "attention_entropy",
) -> tuple[Any, dict[str, Any]]:
    profile = {} if _profile_enabled() else None
    with apply_strict_attention_patch(
        model,
        record_query_content=True,
        record_full_query_content=_needs_full_query_content(config),
        record_tail_window=int(config.observation_window),
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
            config=config,
            budget_mode=str(budget_mode),
            profile=profile,
        )
    debug: dict[str, Any] = {
        "patch": "strictmerge",
        "budget_mode": str(budget_mode),
        "config": config.__dict__,
        "layers": list(debug_layers),
        "total_select_sec": float(sum(float(layer.get("select_sec", 0.0)) for layer in debug_layers)),
    }
    if profile is not None:
        debug["profile"] = dict(sorted(profile.items()))
    return outputs, debug


@torch.no_grad()
def prefill_with_strictmerge_batched_samples(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    prompt_lens: list[int],
    config: StrictMergeConfig,
    budget_mode: str = "attention_entropy",
) -> tuple[list[Any], list[dict[str, Any]], torch.Tensor]:
    """Batched model prefill, followed by per-sample StrictMerge compression.

    The model forward is executed once for the whole padded batch. Each sample's
    prompt cache and recorded query content are then sliced back to its true
    prompt length before reusing the existing single-sample compressor.
    """

    batch_size = int(input_ids.shape[0])
    if batch_size != int(len(prompt_lens)):
        raise ValueError(f"prompt_lens length mismatch: batch={batch_size} prompt_lens={len(prompt_lens)}.")
    if attention_mask is not None and int(attention_mask.shape[0]) != batch_size:
        raise ValueError("attention_mask batch size must match input_ids.")

    position_ids = None
    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0).to(device=input_ids.device, dtype=torch.long)

    with apply_strict_attention_patch(
        model,
        record_query_content=True,
        record_full_query_content=_needs_full_query_content(config),
        record_tail_window=int(config.observation_window),
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
        sample_query_states: list[StrictQueryContentState] = []
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
            sample_query_state = query_state.sample_state(int(sample_idx), int(prompt_len_i))
            compressed_caches.append(sample_cache)
            sample_query_states.append(sample_query_state)

        profile = {} if _profile_enabled() else None
        debug_infos = compress_prompt_cache_strictmerge_sample_batched(
            model=model,
            prompt_past_key_values_list=compressed_caches,
            query_states=sample_query_states,
            config=config,
            budget_mode=str(budget_mode),
            profile=profile,
        )

        next_logits = outputs.logits[:, -1, :].detach().contiguous()
    return compressed_caches, debug_infos, next_logits
