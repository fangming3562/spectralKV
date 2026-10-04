import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from spectralkv import Config, generate
from spectralkv.runtime import DynamicRuntime, FixedRuntime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


class Tokenizer:
    all_special_tokens = []

    def convert_ids_to_tokens(self, ids):
        return ["▁test"] * len(ids)

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


@pytest.fixture
def tiny_model():
    torch.manual_seed(81)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        eos_token_id=None,
    )
    config._attn_implementation = "sdpa"
    return LlamaForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()


@pytest.mark.parametrize("policy", ["dynamic", "fixed"])
def test_shared_prefill_budget_and_cleanup(tiny_model, policy):
    ids = torch.arange(1024, device="cuda")[None] % 64
    config = Config(layer_budget=policy, layer_weights=(1.0, 2.0) if policy == "fixed" else None)
    prior = ALL_ATTENTION_FUNCTIONS["sdpa"]
    hooks = [len(layer.self_attn.q_proj._forward_hooks) for layer in tiny_model.model.layers]
    with torch.inference_mode():
        first = int(tiny_model(ids, logits_to_keep=1).logits[:, -1].argmax())
    result = generate(tiny_model, Tokenizer(), ids, config=config, max_new_tokens=4)
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is prior
    assert hooks == [len(layer.self_attn.q_proj._forward_hooks) for layer in tiny_model.model.layers]
    for row in result["results"].values():
        assert row["token_ids"][0] == first
        assert row["generated"] == 4 and row["selection_forwards"] == 1
        assert row["decode_calls"] == 6
        assert row["kv_bytes"] <= row["cap_bytes"]
        assert all(n >= 32 for layer in row["counts"] for n in layer)


@pytest.mark.parametrize("runtime_cls", [FixedRuntime, DynamicRuntime])
def test_worker_failure_restores_dispatch_and_releases_runtime(tiny_model, runtime_cls, monkeypatch):
    ids = torch.arange(1024, device="cuda")[None] % 64
    config = (
        Config(layer_budget="fixed", layer_weights=(1.0, 1.0)) if runtime_cls is FixedRuntime else Config()
    )
    prior = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def fail(*args, **kwargs):
        raise RuntimeError("injected worker failure")

    with monkeypatch.context() as patch:
        patch.setattr(FixedRuntime, "prepare_layer", fail)
        with pytest.raises(RuntimeError, match="injected worker failure"):
            with runtime_cls(tiny_model, ids, Tokenizer(), config) as runtime:
                runtime.prefill()
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is prior
    assert not any(layer.self_attn.q_proj._forward_hooks for layer in tiny_model.model.layers)
    with runtime_cls(tiny_model, ids, Tokenizer(), config):
        with pytest.raises(RuntimeError, match="Only one active"):
            runtime_cls(tiny_model, ids, Tokenizer(), config)
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is prior


@pytest.mark.model
@pytest.mark.skipif(
    not os.environ.get("SPECTRALKV_TEST_MODEL"), reason="Set SPECTRALKV_TEST_MODEL for integration"
)
def test_local_real_model():
    path = os.environ["SPECTRALKV_TEST_MODEL"]
    model = (
        AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
        )
        .cuda()
        .eval()
    )
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    ids = tok(
        ("A red key opens the north gate. A blue key opens the south gate.\n" * 150)
        + "Which key opens the north gate?",
        return_tensors="pt",
    ).input_ids.cuda()
    out = generate(model, tok, ids, config=Config(layer_budget="fixed"), max_new_tokens=8)
    assert set(out["results"]) == {"5", "10"}
    assert all(row["kv_bytes"] <= row["cap_bytes"] for row in out["results"].values())
