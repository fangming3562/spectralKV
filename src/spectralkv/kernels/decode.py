"""Ragged weighted-prefix attention with an exact generated-token suffix."""

import torch
import triton as tr
import triton.language as tl


@tr.jit(do_not_specialize=["SPLITS"])
def _reduce(P, O, D: tl.constexpr, SPLITS, BS: tl.constexpr):
    h = tl.program_id(0)
    s = tl.arange(0, BS)
    d = tl.arange(0, D)
    src = P + (h * SPLITS + s) * (D + 2)
    m = tl.load(src + D, s < SPLITS, -float("inf"))
    den = tl.load(src + D + 1, s < SPLITS, 0)
    maximum = tl.max(m, 0)
    w = tl.exp(m - maximum)
    nums = tl.load(src[:, None] + d[None, :], s[:, None] < SPLITS, 0)
    out = tl.sum(w[:, None] * nums, 0) / tl.sum(w * den, 0)
    tl.store(O + h * D + d, out)


@tr.jit(do_not_specialize=["T", "SH", "ST", "SD", "VH", "VT", "VD", "SPLITS"])
def _partial(
    Q,
    K,
    V,
    B,
    M,
    SK,
    SV,
    P,
    SCALE: tl.constexpr,
    HG: tl.constexpr,
    D: tl.constexpr,
    T,
    QH: tl.constexpr,
    SH,
    ST,
    SD,
    VH,
    VT,
    VD,
    SPLITS,
    BN: tl.constexpr,
):
    qh = tl.program_id(0)
    split = tl.program_id(1)
    kh = qh // HG
    off = tl.load(M + kh * 4)
    n = tl.load(M + kh * 4 + 1)
    bo = tl.load(M + kh * 4 + 3)
    i = split * BN + tl.arange(0, BN)
    d = tl.arange(0, D)
    q = tl.load(Q + qh * QH + d).to(tl.float32)
    prefix = i < n
    suffix = (i >= n) & (i < n + T)
    kk = tl.load(K + (off + i[:, None]) * D + d[None, :], prefix[:, None], 0).to(tl.float32)
    ks = tl.load(SK + kh * SH + (i[:, None] - n) * ST + d[None, :] * SD, suffix[:, None], 0).to(tl.float32)
    z = tl.sum((kk + ks) * q[None, :], 1) * SCALE
    bias = tl.load(B + bo + i, prefix, 0)
    z = tl.where(i < n + T, z + bias, -float("inf"))
    mx = tl.max(z, 0)
    prob = tl.exp(z - tl.where(mx == -float("inf"), 0.0, mx))
    den = tl.sum(prob, 0)
    vv = tl.load(V + (off + i[:, None]) * D + d[None, :], prefix[:, None], 0).to(tl.float32)
    vs = tl.load(SV + kh * VH + (i[:, None] - n) * VT + d[None, :] * VD, suffix[:, None], 0).to(tl.float32)
    num = tl.sum(prob[:, None] * (vv + vs), 0)
    dst = P + (qh * SPLITS + split) * (D + 2)
    tl.store(dst + d, num)
    tl.store(dst + D, mx)
    tl.store(dst + D + 1, den)


class WeightedPrefix:
    def __init__(self, keys, values, log_weights, groups):
        self.hg = groups
        self.d = keys[0].shape[-1]
        self.k = torch.cat(keys)
        self.v = torch.cat(values)
        self.b = torch.cat(log_weights).float()
        self.max_length = max(map(len, keys))
        meta = []
        offset = 0
        for k in keys:
            meta.append([offset, len(k), len(k), offset])
            offset += len(k)
        self.meta = torch.tensor(meta, device=self.k.device, dtype=torch.int32)
        self.metadata_bytes = self.meta.numel() * 4

    def attend(self, q, k, v, scaling):
        assert q.shape[0] == 1 and q.shape[2] == 1
        t = k.shape[2]
        splits = tr.cdiv(self.max_length + t, 128)
        h = q.shape[1]
        d = self.d
        partial = torch.empty((h, splits, d + 2), device=q.device, dtype=torch.float32)
        out = torch.empty((1, h, 1, d), device=q.device, dtype=q.dtype)
        _partial[(h, splits)](
            q,
            self.k,
            self.v,
            self.b,
            self.meta,
            k,
            v,
            partial,
            SCALE=scaling,
            HG=self.hg,
            D=d,
            T=t,
            QH=q.stride(1),
            SH=k.stride(1),
            ST=k.stride(2),
            SD=k.stride(3),
            VH=v.stride(1),
            VT=v.stride(2),
            VD=v.stride(3),
            SPLITS=splits,
            BN=128,
            num_warps=4,
        )
        _reduce[(h,)](
            partial,
            out,
            D=d,
            SPLITS=splits,
            BS=tr.next_power_of_2(splits),
            num_warps=4 if splits <= 128 else 16,
        )
        return out.transpose(1, 2).contiguous()


@tr.jit
def _gather(
    K,
    I,
    O,
    N: tl.constexpr,
    D: tl.constexpr,
    ROWS,
    SH: tl.constexpr,
    ST: tl.constexpr,
    SD: tl.constexpr,
    B: tl.constexpr,
):
    x = tl.program_id(0) * B + tl.arange(0, B)
    row = x // D
    d = x % D
    ix = tl.load(I + row, row < ROWS, 0)
    value = tl.load(K + (ix // N) * SH + (ix % N) * ST + d * SD, row < ROWS, 0)
    tl.store(O + x, value, row < ROWS)


def gather_keys(keys, indices):
    out = torch.empty((len(indices), keys.shape[-1]), device=keys.device, dtype=keys.dtype)
    _gather[(tr.cdiv(out.numel(), 256),)](
        keys, indices, out, keys.shape[1], keys.shape[2], len(indices), *keys.stride(), 256
    )
    return out
