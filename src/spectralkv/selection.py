"""Exact sparse facility greedy with optional independent-head prefix evaluation."""

import heapq
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from . import native

POOL = ThreadPoolExecutor(max_workers=8)


def prepare(nbr, kappa, weight, *, return_lower=False, check=False):
    n, m = nbr.shape
    nb = np.ascontiguousarray(nbr, dtype=np.int64)
    # Production graph and headmatch outputs are FP32; refuse lossy hidden input conversion.
    if kappa.dtype != np.float32 or weight.dtype != np.float32:
        raise TypeError("Coverage and importance must be float32")
    k = np.ascontiguousarray(kappa)
    wf = np.ascontiguousarray(weight)
    if check:
        assert k.shape == (n, m) and wf.shape == (n,)
        assert (
            np.isfinite(k).all()
            and np.isfinite(wf).all()
            and (k >= 0).all()
            and (k <= 1).all()
            and (wf >= 0).all()
        )
        assert not m or (nb.min() >= 0 and nb.max() < n and not np.any(nb == np.arange(n)[:, None]))
    w = np.empty(n, dtype=np.float64)
    offsets = np.empty(n + 1, dtype=np.int64)
    incoming = np.empty(n * m, dtype=np.int64)
    strength = np.empty(n * m, dtype=np.float64)
    initial = np.empty(n, dtype=np.float64)
    lower = np.empty(n, dtype=np.float64) if return_lower else None
    arrays = (nb, k, wf, w, offsets, incoming, strength, initial)
    size = native.sparse().prepare(
        n, m, *[a.ctypes.data for a in arrays], lower.ctypes.data if lower is not None else None
    )
    result = (w, offsets, incoming[:size], strength[:size], initial)
    return (*result, lower) if return_lower else result


def solve(nbr, kappa, weight, steps, *, batch=1024, prune=True, array_marginals=False):
    """Exact picks. Deployment array mode omits the unused final objective
    diagnostic and Python-list conversion; it never changes gain evaluation.
    """
    n, m = nbr.shape
    if not 0 <= steps <= n or batch < 1:
        raise ValueError("Invalid cardinality/batch")
    started = time.perf_counter()
    prepared_arrays = prepare(nbr, kappa, weight, return_lower=prune)
    w, offsets, incoming, strength, initial = prepared_arrays[:5]
    prepared = time.perf_counter()
    eligible = np.ones(n, dtype=np.uint8)
    threshold = 0.0
    if prune and steps:
        lower = prepared_arrays[5]
        threshold = float(np.partition(lower, n - steps)[n - steps])
        # Conservatively widen bounds for summation/rounding. Strict comparison
        # keeps ties; no floating-point "approximately equal" pruning.
        eps = np.finfo(np.float64).eps
        margin = 8 * eps * (np.diff(offsets) + 2) * np.maximum(initial, threshold) + np.finfo(np.float64).tiny
        eligible[:] = initial + margin >= threshold
    selected = np.empty(steps, dtype=np.int64)
    marginals = np.empty(steps, dtype=np.float64)
    covered = np.zeros(n, dtype=np.float64)
    stats = np.zeros(4, dtype=np.int64)
    pruned = time.perf_counter()
    lib = native.greedy()
    arrays = (w, offsets, incoming, strength, initial, eligible, selected, marginals, covered, stats)
    lib.solve(n, steps, batch, *[a.ctypes.data for a in arrays], native.dot_pointer())
    ended = time.perf_counter()
    return selected, dict(
        evaluations=int(stats[0]),
        objective=None if array_marginals else float(np.dot(w, covered)),
        marginals=marginals if array_marginals else marginals.tolist(),
        candidates=int(eligible.sum()),
        pruned_candidates=int(n - eligible.sum()),
        threshold=threshold,
        rounds=int(stats[1]),
        mean_prefix=steps / max(int(stats[1]), 1),
        max_prefix=int(stats[2]),
        conflicts=int(stats[3]),
        prepare_seconds=prepared - started,
        prune_seconds=pruned - prepared,
        solve_seconds=ended - pruned,
        batch=batch,
        prune=prune,
    )


def merge(curves, steps, old):
    gains = np.concatenate([np.asarray(c[1]["marginals"]) for c in curves])
    ids = np.concatenate([c[0] + h * old for h, c in enumerate(curves)])
    heads = np.concatenate([np.full(len(c[0]), h, dtype=np.int64) for h, c in enumerate(curves)])
    rank = np.concatenate([np.arange(len(c[0])) for c in curves])
    order = np.lexsort((ids, -gains))[:steps]
    valid = True
    for h in range(len(curves)):
        selected = rank[order[heads[order] == h]]
        if not np.array_equal(selected, np.arange(len(selected))):
            valid = False
            break
    if valid:
        return ids[order], gains[order], np.bincount(heads[order], minlength=len(curves))
    # Preserve dependencies even in a rare nonmonotone floating-point tie.
    heap = [(-c[1]["marginals"][0], int(c[0][0]) + h * old, h, 0) for h, c in enumerate(curves) if len(c[0])]
    heapq.heapify(heap)
    picked = []
    pg = []
    counts = np.zeros(len(curves), dtype=np.int64)
    for _ in range(steps):
        neg, j, h, t = heapq.heappop(heap)
        picked.append(j)
        pg.append(-neg)
        counts[h] += 1
        if t + 1 < len(curves[h][0]):
            heapq.heappush(
                heap, (-curves[h][1]["marginals"][t + 1], int(curves[h][0][t + 1]) + h * old, h, t + 1)
            )
    return np.asarray(picked), np.asarray(pg), counts


def parallel_facility(nbr, kappa, weight, steps, heads, *, array_marginals=False):
    # Initialise ctypes signatures and the BLAS callback on the caller thread.
    # Concurrent first use otherwise exposes a partially initialised global CDLL.
    native.greedy()
    native.sparse()
    old = len(weight) // heads
    assert len(weight) == old * heads
    if not steps:
        return torch.empty(0, dtype=torch.long), dict(marginals=[], head_parallel=True)
    nb = nbr.numpy()
    kap = kappa.numpy()
    w = weight.numpy()

    def calculate(h, size):
        sl = slice(h * old, (h + 1) * old)
        return solve(nb[sl] - h * old, kap[sl], w[sl], size, array_marginals=array_marginals)

    initial = min(old, max(1, 2 * math.ceil(steps / heads)))
    curves = list(POOL.map(lambda h: calculate(h, initial), range(heads)))
    extensions = 0
    while True:
        picked, gains, counts = merge(curves, steps, old)
        cutoff = gains[-1]
        extend = [
            h
            for h, c in enumerate(curves)
            if counts[h] == len(c[0]) and len(c[0]) < old and c[1]["marginals"][-1] >= cutoff
        ]
        if not extend:
            break
        futures = {h: POOL.submit(calculate, h, min(old, 2 * len(curves[h][0]))) for h in extend}
        for h, f in futures.items():
            curves[h] = f.result()
            extensions += 1
    return torch.from_numpy(picked.copy()), dict(
        marginals=gains if array_marginals else gains.tolist(),
        head_parallel=True,
        curve_rows=sum(len(c[0]) for c in curves),
        head_curve_extensions=extensions,
    )


def compiled_facility(nbr, kappa, weight, steps):
    selected, info = solve(nbr.numpy(), kappa.numpy(), weight.numpy(), steps, batch=64, array_marginals=True)
    return torch.from_numpy(selected), info
