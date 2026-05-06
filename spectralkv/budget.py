from __future__ import annotations


def uniform_layer_budget(
    *,
    seq_len: int,
    keep_ratio: float,
    force_keep_count: int = 0,
    min_budget: int = 1,
) -> int:
    target = int(round(int(seq_len) * float(keep_ratio)))
    return max(int(target), int(force_keep_count), int(min_budget))
