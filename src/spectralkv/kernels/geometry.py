"""Window distances and empirical LS recovery, preserving validated arithmetic."""

import triton
import triton as tr
import triton.language as tl


@tr.jit(do_not_specialize=["N"])
def _distance(
    X,
    S,
    I,
    O,
    N,
    D: tl.constexpr,
    NB: tl.constexpr,
    RADIUS: tl.constexpr,
    BLOCK: tl.constexpr,
    COLS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    tile = tl.program_id(0) // (BLOCK // BM)
    original = tl.load(I + tile)
    head = original // NB
    start = (original % NB) * BLOCK
    row = start + (tl.program_id(0) % (BLOCK // BM)) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    key = start - RADIUS + col
    depth = tl.arange(0, BD)
    dot = tl.full((BM, BN), 0, tl.float32)
    for begin in range(0, D, BD):
        kk = begin + depth
        x = tl.load(
            X + (head * N + row[:, None]) * D + kk[None, :], (row[:, None] < N) & (kk[None, :] < D), 0
        )
        z = tl.load(
            X + (head * N + key[None, :]) * D + kk[:, None],
            (key[None, :] >= 0) & (key[None, :] < N) & (kk[:, None] < D) & (col[None, :] < COLS),
            0,
        )
        dot = tl.dot(2 * x, z, dot, input_precision="tf32x3")
    si = tl.load(S + head * N + row, row < N, 0)
    sj = tl.load(S + head * N + key, (key >= 0) & (key < N) & (col < COLS), 0)
    delta = row[:, None] - key[None, :]
    valid = (
        (row[:, None] < N)
        & (key[None, :] >= 0)
        & (key[None, :] < N)
        & (delta != 0)
        & (tl.abs(delta) <= RADIUS)
    )
    result = tl.where(valid, (si[:, None] + sj[None, :]) - dot, float("inf"))
    outrow = tl.program_id(0) * BM + tl.arange(0, BM)
    tl.store(O + outrow[:, None] * COLS + col[None, :], result, col[None, :] < COLS)


@tr.jit(do_not_specialize=["E"])
def _content(V, F, S, NBR, C, E, D: tl.constexpr, M: tl.constexpr, B: tl.constexpr):
    edge = tl.program_id(0) * B + tl.arange(0, B)
    dd = tl.arange(0, D)
    source = tl.load(S + edge // M, edge < E, 0)
    dest = tl.load(NBR + edge, edge < E, 0)
    x = tl.load(V + source[:, None] * D + dd[None, :], edge[:, None] < E, 0)
    z = tl.load(V + dest[:, None] * D + dd[None, :], edge[:, None] < E, 0)
    far = tl.load(F + source, edge < E, 1.0)
    fraction = tl.maximum(0.0, 1.0 - tl.sqrt(tl.sum((x - z) * (x - z), 1)) / far)
    tl.store(C + edge, fraction, edge < E)


@triton.jit(do_not_specialize=["N", "E"])
def _fit(
    L,
    A,
    C,
    NEIGHBOUR,
    CONTENT,
    SOURCE,
    KAPPA,
    RATIO,
    N,
    E,
    M: tl.constexpr,
    T: tl.constexpr,
    B: tl.constexpr,
    OBS: tl.constexpr,
    STRIDE: tl.constexpr,
    UNIFORM: tl.constexpr,
    INDEXED: tl.constexpr,
    HEADS: tl.constexpr = False,
    LEAST_SQUARES: tl.constexpr = False,
    RECOVERY: tl.constexpr = True,
):
    edge = tl.program_id(0) * B + tl.arange(0, B)
    sample = tl.arange(0, T)
    valid = edge < E
    source = edge // M
    if INDEXED:
        source = tl.load(SOURCE + source, valid, 0)
    target = tl.load(NEIGHBOUR + edge, valid, 0)
    if HEADS:
        # Source/target IDs are h*N+i. Flatten the sample axis after heads.
        offset = (source // N) * (T * 2 * STRIDE - 1) * N
        source += offset
        target += offset
    content = tl.load(CONTENT + edge, valid, 0.0)
    active = valid & (content > 0)
    # Original even/odd split; optional deterministic sub-sampling of pairs.
    pair_count: tl.constexpr = OBS // (2 * STRIDE)
    query = (sample // pair_count) * OBS + (sample % pair_count) * 2 * STRIDE
    ai = tl.load(A + query[None, :] * N + source[:, None], active[:, None], 0.0)
    aj = tl.load(A + query[None, :] * N + target[:, None], active[:, None], 0.0)
    bi = (
        tl.full((B, T), 1.0, tl.float32)
        if UNIFORM
        else tl.load(C + query[None, :] * N + source[:, None], active[:, None], 0.0)
    )
    if LEAST_SQUARES:
        ww = bi * bi
        fit = tl.sum(ww * ai * aj, 1) / tl.maximum(tl.sum(ww * aj * aj, 1), 1e-30)
        cap = tl.sum(ai, 1) / tl.maximum(tl.sum(aj, 1), 1e-30)
        ratio = tl.where(active, tl.minimum(fit, cap), 0.0)
    else:
        li = tl.load(L + query[None, :] * N + source[:, None], active[:, None], 0.0)
        lj = tl.load(L + query[None, :] * N + target[:, None], active[:, None], 0.0)
        x = li - lj
        weight = aj * bi
        # Stable lexicographic sort: FP32 value, then original sample index.
        bits = x.to(tl.uint32, bitcast=True)
        ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
        sortkey = (ordered.to(tl.uint64) << 16) | sample[None, :].to(tl.uint64)
        sortkey = tl.sort(sortkey, descending=False)
        order = (sortkey & 65535).to(tl.int32)
        ws = tl.gather(weight, order, 1)
        cumulative = tl.cumsum(ws, 1)
        total = tl.sum(tl.where(sample[None, :] == T - 1, cumulative, 0.0), 1)
        median = tl.minimum(tl.sum((cumulative < total[:, None] * 0.5).to(tl.int32), 1), T - 1)
        index = tl.sum(tl.where(sample[None, :] == median[:, None], order, 0), 1)
        xr = tl.sum(tl.where(sample[None, :] == index[:, None], x, 0.0), 1)
        cap = tl.log(tl.maximum(tl.sum(ai, 1), 1e-30)) - tl.log(tl.maximum(tl.sum(aj, 1), 1e-30))
        ratio = tl.where((total > 0) & active, tl.exp(tl.minimum(xr, cap)), 0.0)
    if RECOVERY:
        odd = query + 1
        av = tl.load(A + odd[None, :] * N + source[:, None], active[:, None], 0.0)
        bv = tl.load(A + odd[None, :] * N + target[:, None], active[:, None], 0.0)
        cv = (
            tl.full((B, T), 1.0, tl.float32)
            if UNIFORM
            else tl.load(C + odd[None, :] * N + source[:, None], active[:, None], 0.0)
        )
        loss = tl.sum(cv * tl.abs(ratio[:, None] * bv - av), 1)
        deletion = tl.maximum(tl.sum(cv * av, 1), 1e-30)
        recovery = tl.minimum(tl.maximum(1 - loss / deletion, 0.0), 1.0)
        tl.store(KAPPA + edge, tl.minimum(recovery, content), valid)
    tl.store(RATIO + edge, ratio, valid)
