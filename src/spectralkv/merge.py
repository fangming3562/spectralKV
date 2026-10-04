"""Deterministic representative accumulation and observed-query head gates."""

import torch
import triton as tr
import triton.language as tl


@torch.inference_mode()
def merge_candidates(cache_values, attention, metrics, prepared, selections):
    """Check each head after BF16 rounding; rejected heads retain original values."""
    h, n, d = cache_values.shape[1:]
    old = n - 32
    device = cache_values.device
    v, y, neighbors, recovery, ratios = prepared
    # Preserve the validated contiguous layout and deterministic source order.
    values = torch.stack([v[hi] for hi in range(h)])
    output = torch.stack([y[hi] for hi in range(h)])
    sources = values[:, :old].reshape(-1, d)
    nb = torch.stack([neighbors[hi] for hi in range(h)])
    kap = torch.stack([recovery[hi] for hi in range(h)])
    ratios = torch.stack([ratios[hi] for hi in range(h)])
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    result = {}
    try:
        for pct, keep in selections.items():
            counts = [len(x) for x in keep]
            total = sum(counts)
            owner, original = owners_values(values, keep)
            target = torch.empty(h * old, device=device, dtype=torch.long)
            coef = values.new_empty(h * old)
            recover = torch.empty_like(coef)
            active = torch.empty(h * old, device=device, dtype=torch.bool)
            _route[(tr.cdiv(h * old, 128),)](
                nb,
                kap,
                ratios,
                owner,
                target,
                coef,
                recover,
                active,
                old,
                total,
                h * old,
                8,
                128,
                num_warps=4,
            )
            numerator = torch.cat((original, values.new_zeros(h * old, d)))
            mass = torch.cat((values.new_ones(total), values.new_zeros(h * old)))
            mass.index_add_(0, target, coef)
            numerator.index_add_(0, target, coef[:, None] * sources)
            merged = numerator[:total] / mass[:total, None]
            good, _ = head_gates(
                attention, metrics, output, keep, original, merged, mass[:total], cache_values.dtype
            )
            good = good & (active.reshape(h, old).sum(-1) > 0)
            rowgood = good.repeat_interleave(device_counts(counts, device), output_size=total)
            final = torch.where(rowgood[:, None], merged, original).to(cache_values.dtype)
            weights = torch.where(rowgood, mass[:total], 1.0)
            result[pct] = (final, weights)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)
    return result


def device_counts(counts, dev):
    return torch.tensor(counts, dtype=torch.long, pin_memory=True).to(dev, non_blocking=True)


@tr.jit
def _owners_values(
    KEEP, META, VALUES, OWNER, ORIGINAL, N: tl.constexpr, OLD: tl.constexpr, D: tl.constexpr, B: tl.constexpr
):
    h = tl.program_id(1)
    row = tl.program_id(0) * B + tl.arange(0, B)
    offset = tl.load(META + 2 * h)
    count = tl.load(META + 2 * h + 1)
    ix = tl.load(KEEP + offset + row, row < count, 0)
    tl.store(OWNER + h * OLD + ix, offset + row, row < count - 32)
    dd = tl.arange(0, D)
    value = tl.load(VALUES + (h * N + ix[:, None]) * D + dd[None, :], row[:, None] < count, 0)
    tl.store(ORIGINAL + (offset + row[:, None]) * D + dd[None, :], value, row[:, None] < count)


def owners_values(values, keep):
    h, n, d = values.shape
    counts = [len(ix) for ix in keep]
    offset = 0
    meta = []
    for count in counts:
        meta.append([offset, count])
        offset += count
    meta = torch.tensor(meta, dtype=torch.int64, pin_memory=True).to(values.device, non_blocking=True)
    original = torch.empty((offset, d), dtype=values.dtype, device=values.device)
    owner = torch.full((h, n - 32), -1, dtype=torch.long, device=values.device)
    _owners_values[(tr.cdiv(max(counts), 16), h)](
        torch.cat(keep), meta, values, owner, original, n, n - 32, d, 16, num_warps=4
    )
    return owner, original


def head_gates(a, metrics, outputs, keep, original, merged, mass, dtype):
    """Original normalized odd-query test batched over ragged heads.

    Padding carries zero attention and does not consume persistent cache bytes.
    Matrix products/reductions can round differently from per-head GEMMs.
    """
    h, g, _, _ = a.shape
    counts = [len(x) for x in keep]
    total = sum(counts)
    dev = a.device
    sizes = device_counts(counts, dev)
    offset = sizes.cumsum(0) - sizes
    position = torch.arange(max(counts), device=dev)
    valid = position[None, :] < sizes[:, None]
    flat = (offset[:, None] + position[None, :]).clamp_max(total - 1)
    indices = torch.cat(keep)[flat]
    selected = a[:, :, 1::2].gather(-1, indices[:, None, None, :].expand(h, g, 16, -1))
    selected = selected * valid[:, None, None, :]
    weights = torch.stack((selected, selected * mass[flat][:, None, None, :]))
    values = torch.stack((original[flat], merged.to(dtype).float()[flat]))
    prediction = torch.matmul(weights, values[:, :, None]) / weights.sum(-1, keepdim=True).clamp_min(1e-30)
    error = prediction - outputs[None, :, :, 1::2]
    squared = torch.einsum("bhgtd,hgde,bhgte->bhgt", error, metrics, error).clamp_min(0)
    norms = squared.sqrt().mean((2, 3))
    return norms[1] < norms[0], norms


@tr.jit(do_not_specialize=["N", "TOTAL", "ROWS"])
def _route(
    NB, KAP, R, OWNER, TARGET, COEF, RECOVER, ACTIVE, N, TOTAL, ROWS, M: tl.constexpr, B: tl.constexpr
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    s = tl.arange(0, M)
    h = i // N
    valid = i < ROWS
    nb = tl.load(NB + i[:, None] * M + s[None, :], valid[:, None], 0)
    target = tl.load(OWNER + h[:, None] * N + nb, valid[:, None], -1)
    score = tl.load(KAP + i[:, None] * M + s[None, :], valid[:, None], 0)
    score = tl.where(target >= 0, score, 0.0)
    best = tl.max(score, 1)
    edge = tl.min(tl.where(score == best[:, None], s[None, :], M), 1)
    owner = tl.load(OWNER + i, valid, -1)
    active = valid & (owner < 0) & (best > 0)
    chosen = tl.sum(tl.where(s[None, :] == edge[:, None], target, 0), 1)
    coef = tl.load(R + i * M + edge, active, 0.0)
    # Distinct zero-weight sinks avoid serializing every rejected member into
    # one huge deterministic-reduction segment.
    tl.store(TARGET + i, tl.where(active, chosen, TOTAL + i), valid)
    tl.store(COEF + i, coef, valid)
    tl.store(RECOVER + i, tl.where(active, best, 0.0), valid)
    tl.store(ACTIVE + i, active, valid)
