"""Budget-independent per-layer redundancy graph construction."""

import torch
import triton as tr

from .kernels.geometry import _content, _distance, _fit


@torch.inference_mode()
def build(keys, values, q, a, query_factors, output_factors, active):
    """Returns original values, observed outputs, and the H x (N-32) x 8 graph."""
    h, n, d = values.shape[1:]
    old = n - 32
    m = 8
    k = keys[0].float()
    v = values[0].float()
    y = torch.matmul(a, v[:, None])
    transforms = torch.stack(output_factors)
    factors = torch.stack(query_factors)
    projected = torch.bmm(v, transforms)
    center = torch.bmm(y.mean((1, 2))[:, None], transforms)
    dev = (projected - center).square().sum(-1)
    scatter = dev.mean(-1)
    vf = projected[:, :old].contiguous()
    far = dev[:, :old].sqrt().clamp_min(1e-12).contiguous()
    normalized = torch.where(
        scatter[:, None, None] > 0, vf / torch.where(scatter > 0, scatter.sqrt(), 1.0)[:, None, None], 0.0
    )
    features = torch.cat((torch.bmm(k[:, :old], factors), normalized), -1).contiguous()
    scale = features.abs().amax((1, 2), keepdim=True).clamp_min(1e-20)
    source, near = neighbours((features / scale).to(torch.bfloat16), active, 16, 2048, 32)
    a = a.contiguous()
    samples = q.shape[1] * 16
    nbr = near[:, :m].contiguous()
    content = v.new_empty(nbr.shape)
    if len(source):
        _content[(tr.cdiv(nbr.numel(), 16),)](
            vf, far, source, nbr, content, nbr.numel(), d, m, 16, num_warps=4, enable_fp_fusion=False
        )
        src = source // old * n + source % old
        dest = (nbr // old * n + nbr % old).contiguous()
        kap = torch.empty_like(content)
        ratio = torch.empty_like(content)
        _fit[(tr.cdiv(nbr.numel(), 8),)](
            a,
            a,
            a,
            dest,
            content,
            src,
            kap,
            ratio,
            n,
            nbr.numel(),
            m,
            samples,
            8,
            32,
            1,
            True,
            True,
            True,
            True,
            num_warps=4,
            enable_fp_fusion=False,
        )
    full = (
        (torch.arange(old, device=v.device)[:, None] + torch.arange(1, m + 1, device=v.device)[None]) % old
    ).repeat(h, 1)
    ks = v.new_zeros((h * old, m))
    rs = torch.zeros_like(ks)
    if len(source):
        full.index_copy_(0, source, nbr % old)
        ks.index_copy_(0, source, kap)
        rs.index_copy_(0, source, ratio)
    return v, y, full.reshape(h, old, m), ks.reshape(h, old, m), rs.reshape(h, old, m)


def neighbours(features, blocks, m=16, radius=2048, block=256):
    """Filter the small top-k indices instead of copying the huge distance matrix."""
    h, n, d = features.shape
    nb = tr.cdiv(n, block)
    dev = features.device
    assert all(hasattr(x, "_deploy_host") for x in blocks)
    host_tiles = torch.cat([x._deploy_host + hi * nb for hi, x in enumerate(blocks)])
    tiles = host_tiles.pin_memory().to(dev, non_blocking=True)
    if not len(tiles):
        return tiles, tiles.new_empty((0, m))
    radius = min(radius, n)
    cols = block + 2 * radius
    norm = features.float().square().sum(-1)
    distance = torch.empty((len(tiles) * block, cols), device=dev)
    _distance[(len(tiles) * block // 32, tr.cdiv(cols, 64))](
        features,
        norm,
        tiles,
        distance,
        n,
        d,
        nb,
        radius,
        block,
        cols,
        32,
        64,
        32,
        num_warps=4,
        enable_fp_fusion=False,
    )
    near = distance.topk(m, largest=False).indices
    pos_cpu = (host_tiles[:, None] % nb * block + torch.arange(block)[None]).flatten()
    head_cpu = (host_tiles // nb).repeat_interleave(block)
    valid = pos_cpu < n

    def send(x):
        return x.pin_memory().to(dev, non_blocking=True)

    pos = send(pos_cpu[valid])
    head = send(head_cpu[valid])
    if not bool(valid.all()):
        near = near.index_select(0, send(valid.nonzero().flatten()))
    near = near + (pos // block * block - radius)[:, None]
    return head * n + pos, head[:, None] * n + near
