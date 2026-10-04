"""Pinned asynchronous copies with explicit host consumption barriers."""

import numpy as np
import torch
import triton


def to_host(tensor):
    out = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True)
    out.copy_(tensor, non_blocking=True)
    done = torch.cuda.Event()
    done.record()
    done.synchronize()
    return out


def to_host_many(tensors):
    outputs = [torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True) for t in tensors]
    for out, tensor in zip(outputs, tensors):
        out.copy_(tensor, non_blocking=True)
    done = torch.cuda.Event()
    done.record()
    done.synchronize()
    return outputs


def factors_and_blocks(covariance, dimension, weight, epsilon, block=32):
    """Schedule independent tiny host decisions before one event wait."""
    matrices = covariance / dimension
    factors, info = torch.linalg.cholesky_ex(matrices, check_errors=False)
    h, n = weight.shape
    nb = triton.cdiv(n, block)
    masses = torch.nn.functional.pad(weight.double(), (0, nb * block - n)).reshape(h, nb, block).sum(-1)
    info_host, mass_host = to_host_many((info, masses))
    failures = info_host.tolist()
    result = []
    for hi, failed in enumerate(failures):
        if failed:
            ev, u = torch.linalg.eigh(matrices[hi])
            result.append(u * ev.clamp_min(0).sqrt())
        else:
            result.append(factors[hi])
    flat = mass_host.numpy().flatten()
    order = np.argsort(-flat, kind="stable")
    total = float(flat.sum())
    count = (
        int(np.searchsorted(np.cumsum(flat[order]), (1 - epsilon) * total, side="left")) + 1
        if total > 0
        else 0
    )
    kept = order[: min(count, len(order))]
    heads = []
    for hi in range(h):
        host = torch.from_numpy(np.sort(kept[kept // nb == hi] % nb)).pin_memory()
        device = host.to(weight.device, non_blocking=True)
        device._deploy_host = host
        heads.append(device)
    return result, failures, heads, dict(kept_blocks=len(kept), total_blocks=h * nb)
