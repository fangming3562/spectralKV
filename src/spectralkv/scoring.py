"""Observed-query importance and dynamic layer statistics."""

from torch.nn import functional as F

from ._math import _attention
from .kernels.scoring import importance


def score_layer(model, raw, keys, values, n, attention, metrics, *, need_variance):
    heads = keys.shape[1]
    groups = model.config.num_attention_heads // heads
    old = n - 32
    native = _attention(model, raw, keys, n)[0]
    z = native[..., :old].float()
    smooth = F.avg_pool1d(z, 5, 1, 2)
    smooth = smooth / smooth.sum(-1, keepdim=True).clamp_min(1e-20) * z.sum(-1, keepdim=True)
    score = importance(values[0, :, :old], metrics, smooth.reshape(heads, groups, 32, old))
    variance = attention[..., :old].var(-2, unbiased=False).sum(-1).mean() if need_variance else None
    return score, variance
