import pytest
import torch

from spectralkv._math import word_spread as reference_spread
from spectralkv.kernels.decode import WeightedPrefix
from spectralkv.kernels.scoring import attach_words, word_spread

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.mark.parametrize("groups", [1, 4])
@pytest.mark.parametrize("suffix_length", [0, 3, 137])
def test_weighted_ragged_decode_against_dense_attention(groups, suffix_length):
    torch.manual_seed(9)
    d = 64
    counts = [32, 145, 263]
    keys = [torch.randn(n, d, device="cuda", dtype=torch.bfloat16) for n in counts]
    vals = [torch.randn_like(k) for k in keys]
    biases = [torch.randn(n, device="cuda") for n in counts]
    prefix = WeightedPrefix(keys, vals, biases, groups)
    q = torch.randn(1, 3 * groups, 1, d, device="cuda", dtype=torch.bfloat16)
    sk = torch.randn(1, 3, suffix_length, d, device="cuda", dtype=torch.bfloat16)
    sv = torch.randn_like(sk)
    result = prefix.attend(q, sk, sv, d**-0.5)[0, 0]
    refs = []
    for hi in range(3 * groups):
        h = hi // groups
        k = torch.cat((keys[h], sk[0, h])).float()
        v = torch.cat((vals[h], sv[0, h])).float()
        b = torch.cat((biases[h], torch.zeros(suffix_length, device="cuda")))
        refs.append(((q[0, hi, 0].float() @ k.T / d**0.5 + b).softmax(-1) @ v))
    torch.testing.assert_close(result.float(), torch.stack(refs), atol=0.004, rtol=0.01)


def test_word_scan_short_and_long_units():
    words = [(0, 2), (2, 19), (19, 620), (620, 623)]
    membership = torch.repeat_interleave(
        torch.arange(len(words), device="cuda"), torch.tensor([b - a for a, b in words], device="cuda")
    )
    attach_words(membership, words)
    torch.manual_seed(7)
    scores = torch.rand(3, 623, device="cuda")
    torch.testing.assert_close(
        word_spread(scores, membership, 0.95),
        reference_spread(scores, membership, 0.95),
        atol=2e-7,
        rtol=2e-6,
    )
