from __future__ import annotations

from collections.abc import Iterable

import torch


def build_observation_positions(seq_len: int, observation_window: int, *, device: torch.device) -> torch.Tensor:
    window = min(max(int(observation_window), 1), int(seq_len))
    return torch.arange(int(seq_len) - window, int(seq_len), device=device, dtype=torch.long)


def build_observation_causal_mask(
    *,
    obs_positions: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Return [R, T] mask where query position p may attend to key <= p."""

    if obs_positions.ndim != 1:
        raise ValueError(f"obs_positions must be 1D, got {tuple(obs_positions.shape)}")
    key_positions = torch.arange(int(seq_len), device=obs_positions.device, dtype=torch.long)
    return key_positions.view(1, -1) <= obs_positions.long().view(-1, 1)


def build_force_keep_mask(
    *,
    seq_len: int,
    force_sink: int = 0,
    force_recent: int = 0,
    force_prefix: int = 0,
    extra_keep_indices: Iterable[int] | torch.Tensor | None = None,
    device: torch.device,
) -> torch.Tensor:
    """Build token-level hard keep constraints.

    These are stability constraints, not residual rescue. They are intended for
    sink tokens, recent prompt tokens, and optional format/prefix tokens.
    """

    seq_len = int(seq_len)
    mask = torch.zeros((seq_len,), dtype=torch.bool, device=device)
    prefix = min(max(int(force_prefix), 0), seq_len)
    sink = min(max(int(force_sink), 0), seq_len)
    recent = min(max(int(force_recent), 0), seq_len)
    if prefix > 0:
        mask[:prefix] = True
    if sink > 0:
        mask[:sink] = True
    if recent > 0:
        mask[seq_len - recent :] = True
    if extra_keep_indices is not None:
        if isinstance(extra_keep_indices, torch.Tensor):
            idx = extra_keep_indices.to(device=device, dtype=torch.long).flatten()
        else:
            clean = [int(x) for x in extra_keep_indices if 0 <= int(x) < seq_len]
            idx = torch.tensor(clean, device=device, dtype=torch.long)
        if int(idx.numel()) > 0:
            idx = idx[(idx >= 0) & (idx < seq_len)]
            if int(idx.numel()) > 0:
                mask.index_fill_(0, idx, True)
    return mask
