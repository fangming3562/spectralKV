"""Output-projected L2 headmatch and within-word max-product propagation."""

import math

import torch
import triton as tr
import triton.language as tl


@tr.jit(do_not_specialize=["N"])
def _norm(
    V,
    M,
    O,
    N,
    D: tl.constexpr,
    G: tl.constexpr,
    VH: tl.constexpr,
    VT: tl.constexpr,
    VD: tl.constexpr,
    B: tl.constexpr,
):
    qh = tl.program_id(0)
    row = tl.program_id(1) * B + tl.arange(0, B)
    d = tl.arange(0, D)
    x = tl.load(V + (qh // G) * VH + row[:, None] * VT + d[None, :] * VD, row[:, None] < N, 0).to(tl.float32)
    m = tl.load(M + qh * D * D + d[:, None] * D + d[None, :])
    y = tl.dot(x, m, input_precision="tf32x3")
    norm = tl.sqrt(tl.maximum(tl.sum(x * y, 1), 0.0))
    tl.store(O + qh * N + row, norm, row < N)


def head_norms(values, metrics):
    h, n, d = values.shape
    g = metrics.shape[1]
    assert metrics.shape == (h, g, d, d) and metrics.dtype == torch.float32
    metrics = metrics.contiguous()
    out = torch.empty((h, g, n), device=values.device, dtype=torch.float32)
    _norm[(h * g, tr.cdiv(n, 32))](values, metrics, out, n, d, g, *values.stride(), 32, num_warps=4)
    return out


def importance(values, metrics, attention):
    norms = head_norms(values, metrics)
    denominator = norms.mean(1).sum(0, keepdim=True).clamp_min(1e-20)
    return (attention.amax(2) * norms).amax(1) / denominator


def attach_words(membership, words):
    buckets = {}
    for start, end in words:
        width = tr.next_power_of_2(end - start)
        buckets.setdefault(width, []).append((start, end - start))
    result = []
    for width, rows in sorted(buckets.items()):
        host = torch.tensor(rows, dtype=torch.int32, pin_memory=True)
        result.append((width, host.to(membership.device, non_blocking=True)))
    membership._spread_buckets = result


@tr.jit
def _word(
    S,
    UNITS,
    O,
    N: tl.constexpr,
    U: tl.constexpr,
    WIDTH: tl.constexpr,
    ROWS: tl.constexpr,
    LOG: tl.constexpr,
    DECAYS: tl.constexpr,
):
    head = tl.program_id(1)
    u = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    j = tl.arange(0, WIDTH)
    start = tl.load(UNITS + 2 * u, u < U, 0)
    size = tl.load(UNITS + 2 * u + 1, u < U, 0)
    valid = (u[:, None] < U) & (j[None, :] < size[:, None])
    acc = tl.load(S + head * N + start[:, None] + j[None, :], valid, 0)
    for level in tl.static_range(LOG):
        step = 1 << level
        idx = tl.broadcast_to(tl.maximum(j - step, 0)[None, :], (ROWS, WIDTH))
        before = tl.gather(acc, idx, axis=1)
        propagated = (before.to(tl.float32) * DECAYS[level]).to(acc.dtype)
        acc = tl.where(j[None, :] >= step, tl.maximum(acc, propagated), acc)
    tl.store(O + head * N + start[:, None] + j[None, :], acc, valid)


def word_spread(scores, membership, gamma=0.95):
    assert scores.ndim == 2 and scores.is_contiguous()
    # Very long whitespace-free units otherwise trigger large, slow first-use
    # compilations. The original segmented scan has identical arithmetic.
    if any(width > 256 for width, _ in membership._spread_buckets):
        from .._math import word_spread as reference

        return reference(scores, membership, gamma)
    out = torch.empty_like(scores)
    for width, units in membership._spread_buckets:
        rows = min(32, max(1, 1024 // width))
        _word[(tr.cdiv(len(units), rows), scores.shape[0])](
            scores,
            units,
            out,
            scores.shape[1],
            len(units),
            width,
            rows,
            int(math.log2(width)),
            tuple(gamma ** (1 << i) for i in range(int(math.log2(width)))),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
