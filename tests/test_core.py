from types import SimpleNamespace

import numpy as np
import pytest
import torch

from spectralkv import Config
from spectralkv._math import apportion
from spectralkv.selection import parallel_facility, solve


@pytest.mark.parametrize(
    "kwargs",
    [
        {"percentages": ()},
        {"percentages": (5, 5)},
        {"percentages": (True,)},
        {"percentages": (0,)},
        {"layer_budget": "unknown"},
        {"profile": "qwen3_4b"},
        {"layer_budget": "fixed", "layer_weights": (float("nan"),)},
    ],
)
def test_config_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_fixed_profile_architecture_and_weights():
    cfg = SimpleNamespace(
        model_type="qwen3",
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=2560,
    )
    weights = Config(layer_budget="fixed").fixed_weights(cfg)
    assert len(weights) == 36 and all(w >= 0 for w in weights)
    cfg.num_hidden_layers = 2
    with pytest.raises(ValueError, match="architecture"):
        Config(layer_budget="fixed").fixed_weights(cfg)
    assert Config(layer_budget="fixed", layer_weights=(1.0, 2.0)).fixed_weights(cfg) == [1.0, 2.0]


@pytest.mark.parametrize("weights", [[0, 0, 0], [1, 1, 1], [0, 0, 100], [1, 1000, 2]])
@pytest.mark.parametrize("budget", [32, 33, 64, 128])
def test_apportion_conserves_rows_and_protected_window(weights, budget):
    counts = apportion(weights, budget, 128)
    assert sum(counts) == len(weights) * budget
    assert min(counts) >= 32 and max(counts) <= 128


def graph(seed, heads=1, old=23):
    rng = np.random.default_rng(seed)
    nbr = np.stack(
        [
            rng.choice(np.delete(np.arange(old), i), 4, replace=False) + h * old
            for h in range(heads)
            for i in range(old)
        ]
    )
    kap = rng.uniform(0, 1, nbr.shape).astype(np.float32)
    weight = rng.uniform(0.01, 2, heads * old).astype(np.float32)
    return nbr, kap, weight


def reference(nbr, kap, weight, steps):
    n = len(weight)
    matrix = np.eye(n, dtype=np.float64)
    for i in range(n):
        matrix[i, nbr[i]] = kap[i]
    coverage = np.zeros(n)
    selected = []
    for _ in range(steps):
        gains = (weight[:, None].astype(np.float64) * np.maximum(matrix - coverage[:, None], 0)).sum(0)
        gains[selected] = -np.inf
        j = int(gains.argmax())
        selected.append(j)
        coverage = np.maximum(coverage, matrix[:, j])
    return np.array(selected), float(weight @ coverage)


@pytest.mark.parametrize("seed", range(8))
def test_native_greedy_and_head_prefixes_match_reference(seed):
    nb, kap, w = graph(seed, heads=3)
    expected, value = reference(nb, kap, w, 31)
    selected, info = solve(nb, kap, w, 31)
    assert np.array_equal(selected, expected)
    assert info["objective"] == pytest.approx(value, rel=1e-12)
    parallel, _ = parallel_facility(torch.from_numpy(nb), torch.from_numpy(kap), torch.from_numpy(w), 31, 3)
    assert np.array_equal(parallel.numpy(), expected)


def test_native_zero_budget_and_equal_gain_ties():
    nb, kap, w = graph(12)
    kap.fill(0)
    w.fill(1)
    assert solve(nb, kap, w, 0)[0].size == 0
    selected, _ = solve(nb, kap, w, len(w))
    assert np.array_equal(selected, np.arange(len(w)))
