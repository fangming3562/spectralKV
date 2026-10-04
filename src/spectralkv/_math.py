"""Budget apportionment, positional queries, and word membership."""

import math

import numpy as np
import torch

WINDOW = 32


def apportion(weights, B, N, window=32):
    w = np.asarray(weights, dtype=np.float64)
    L = len(w)
    assert L and np.isfinite(w).all() and (w >= 0).all() and window <= B <= N
    remaining = L * (B - window)
    cap = N - window
    allocation = np.zeros(L)
    active = np.ones(L, dtype=bool)
    while remaining > 0 and active.any():
        ww = w[active]
        ww = ww if ww.sum() > 0 else np.ones_like(ww)
        proposal = remaining * ww / ww.sum()
        ids = np.flatnonzero(active)
        sat = proposal >= cap
        if not sat.any():
            allocation[ids] = proposal
            break
        allocation[ids[sat]] = cap
        remaining -= cap * int(sat.sum())
        active[ids[sat]] = False
    floor = np.floor(allocation + 1e-10).astype(int)
    left = L * (B - window) - int(floor.sum())
    order = np.argsort(-(allocation - floor), kind="stable")
    for j in order:
        if left == 0:
            break
        if floor[j] < cap:
            floor[j] += 1
            left -= 1
    assert left == 0
    out = (floor + window).tolist()
    assert sum(out) == L * B and min(out) >= window and max(out) <= N
    return out


def piece_units(pieces, special=()):
    starts = [0]
    for i in range(1, len(pieces)):
        current, previous = pieces[i], pieces[i - 1]
        if (
            current.startswith(("Ġ", "Ċ", "▁"))
            or previous.endswith(("Ġ", "Ċ", "▁"))
            or current in special
            or previous in special
        ):
            starts.append(i)
    return list(zip(starts, starts[1:] + [len(pieces)]))


def rotate(model, raw, n, shift=0):
    d = raw.shape[-1]
    t = raw.shape[-2]
    pos = torch.arange(n - t + shift, n + shift, device=raw.device)[None]
    cos, sin = model.model.rotary_emb(raw, pos)
    return raw * cos[:, None] + torch.cat((-raw[..., d // 2 :], raw[..., : d // 2]), -1) * sin[:, None]


def word_spread(scores, membership, gamma=0.95):
    """Max-product readout restricted to a whitespace-delimited token unit.
    Batched binary lifting computes exact segmented max scans without a loop
    over all document words and without an arbitrary context-wide radius.
    """
    if not 0 < gamma <= 1:
        raise ValueError(gamma)
    out = scores.clone()
    step = 1
    n = scores.shape[-1]
    while step < n:
        same = membership[step:] == membership[:-step]
        previous = out[..., :-step] * gamma**step
        update = torch.maximum(out[..., step:], previous)
        out = torch.cat((out[..., :step], torch.where(same, update, out[..., step:])), dim=-1)
        step *= 2
    return out


def _observations(model, raw, cl, n):
    H = cl.keys.shape[1]
    D = cl.keys.shape[-1]
    G = model.config.num_attention_heads // H
    q = rotate(model, raw[:, :, -WINDOW:], n)[0].reshape(H, G, WINDOW, D).float()
    logits = q @ cl.keys[0, :, None].float().transpose(-1, -2) / math.sqrt(D)
    mask = torch.arange(n, device=q.device)[None, :] > torch.arange(n - WINDOW, n, device=q.device)[:, None]
    a = logits.masked_fill(mask[None, None], -torch.inf).softmax(-1)
    return q, a


def _attention(model, queries, keys, n):
    d = queries.shape[-1]
    window = queries.shape[-2]
    pos = torch.arange(n - window, n, device=queries.device)[None]
    cos, sin = model.model.rotary_emb(queries, pos)
    rotated = torch.cat((-queries[..., d // 2 :], queries[..., : d // 2]), -1)
    q = queries * cos[:, None] + rotated * sin[:, None]
    k = keys.repeat_interleave(q.shape[1] // keys.shape[1], dim=1)
    logits = q @ k.transpose(-1, -2) / d**0.5
    mask = torch.ones_like(logits, dtype=torch.bool).triu(diagonal=n - window + 1)
    return logits.masked_fill(mask, -torch.inf).softmax(-1, dtype=torch.float32).to(queries.dtype)


def query_moments(model, raw, n, heads):
    rope_type = getattr(model.model.rotary_emb, "rope_type", "default")
    if "dynamic" in rope_type:
        raise ValueError("Batched RoPE translations require fixed frequencies")
    groups = raw.shape[1] // heads
    window = raw.shape[-2]
    d = raw.shape[-1]
    shifts = torch.arange(0, 129, 8, device=raw.device)
    positions = (torch.arange(n - window, n, device=raw.device)[None] + shifts[:, None]).reshape(1, -1)
    cos, sin = model.model.rotary_emb(raw, positions)
    cos = cos.reshape(1, 1, len(shifts), window, d)
    sin = sin.reshape(1, 1, len(shifts), window, d)
    half = torch.cat((-raw[..., d // 2 :], raw[..., : d // 2]), -1)
    rotated = raw[:, :, None] * cos + half[:, :, None] * sin
    bank = (
        rotated[0]
        .reshape(heads, groups, len(shifts), window, d)
        .permute(0, 2, 1, 3, 4)
        .reshape(heads, -1, d)
        .float()
    )
    mean = bank.mean(1)
    center = bank - mean[:, None]
    covariance = center.transpose(-1, -2) @ center / center.shape[1]
    return mean, covariance
