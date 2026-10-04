"""Model-weight-only projected metrics, cached on pinned host memory."""

import torch


@torch.inference_mode()
def output_geometry(model, layer, heads, groups, dimension):
    projection = model.model.layers[layer].self_attn.o_proj.weight
    signature = (
        id(projection),
        projection._version,
        projection.device,
        projection.dtype,
        heads,
        groups,
        dimension,
    )
    bank = getattr(model, "_spectralkv_output_geometry", None)
    if bank is None:
        bank = {}
        model._spectralkv_output_geometry = bank
    cached = bank.get(layer)
    if cached is None or cached[0] != signature:
        weight = projection.float().T.reshape(heads, groups, dimension, -1)
        metrics = weight @ weight.transpose(-1, -2)
        # Match the existing per-head eigensolve, including its sign convention.
        factors = []
        for metric in metrics:
            eigenvalues, vectors = torch.linalg.eigh(metric.mean(0))
            factors.append(vectors * eigenvalues.clamp_min(0).sqrt())
        # Persistent storage belongs on the host: do not add 80--90 MiB to
        # the model's decoding footprint just to save construction work.
        bank[layer] = (
            signature,
            metrics.cpu().pin_memory(),
            [factor.cpu().pin_memory() for factor in factors],
        )
    _, metrics, factors = bank[layer]
    return (
        metrics.to(projection.device, non_blocking=True),
        [factor.to(projection.device, non_blocking=True) for factor in factors],
    )


def storage_bytes(model):
    return sum(
        metrics.numel() * metrics.element_size() + sum(f.numel() * f.element_size() for f in factors)
        for _, metrics, factors in getattr(model, "_spectralkv_output_geometry", {}).values()
    )
