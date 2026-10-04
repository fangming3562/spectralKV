"""Layer pipelines for one unpadded BF16 prompt on one CUDA device."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import torch
from transformers import DynamicCache
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from . import geometry, native, transfers
from ._math import _observations, apportion, piece_units, query_moments
from .config import Config
from .kernels.decode import WeightedPrefix, gather_keys
from .kernels.scoring import attach_words, word_spread
from .merge import merge_candidates
from .operators import output_geometry
from .scoring import score_layer
from .selection import compiled_facility, parallel_facility

_active_runtime = threading.Lock()


class _TransientCache(DynamicCache):
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return key_states, value_states


class FixedRuntime:
    """Full attention per layer; previously completed layers retain compact KV only.

    Pending worker jobs may own multiple full layers. One active runtime per
    process is supported. Always close, preferably using the context manager.
    """

    def __init__(self, model, input_ids, tokenizer, config=Config(layer_budget="fixed")):
        if model.config.model_type not in ("llama", "qwen3"):
            raise ValueError("Supported architectures: Llama and Qwen3.")
        if model.config._attn_implementation != "sdpa" or model.training:
            raise ValueError('Load an eval model with attn_implementation="sdpa".')
        if model.device.type != "cuda" or model.dtype != torch.bfloat16:
            raise ValueError("The validated runtime requires CUDA and BF16 model weights.")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be one unpadded int64 prompt of shape [1, N].")
        if input_ids.device != model.device:
            raise ValueError("input_ids and model must be on the same CUDA device.")
        if any(p.device != model.device for p in model.parameters()):
            raise ValueError("Model sharding/offloading is not supported by this runtime.")
        if any(layer.self_attn.head_dim not in (64, 128, 256) for layer in model.model.layers):
            raise ValueError("Supported head dimensions: 64, 128, 256.")
        if "dynamic" in getattr(model.model.rotary_emb, "rope_type", "default"):
            raise ValueError("Dynamic RoPE frequency updates are not supported.")
        self.model = model
        self.prompt_ids = input_ids
        self.config = config
        self.n = input_ids.shape[1]
        self.percentages = config.percentages
        d = model.model.layers[0].self_attn.head_dim
        self.bpt = 4 * d
        self.budgets = {p: ((self.n * p // 100) * self.bpt - 16) // (self.bpt + 4) for p in self.percentages}
        if min(self.budgets.values()) < 32:
            raise ValueError("The byte budget cannot fit 32 protected rows per head.")
        weights = (
            config.fixed_weights(model.config)
            if config.layer_budget == "fixed"
            else [1.0] * len(model.model.layers)
        )
        self.rows = {p: apportion(weights, b, self.n) for p, b in self.budgets.items()}
        native.greedy()
        native.sparse()
        if not _active_runtime.acquire(blocking=False):
            raise RuntimeError("Only one active SpectralKV runtime is supported per process.")
        self.closed = False
        self.handles = []
        self.previous_attention = None
        self.pool = None
        self.geometry_pool = None
        try:
            self.pool = ThreadPoolExecutor(max_workers=1)
            self.geometry_pool = ThreadPoolExecutor(max_workers=1)
            self.stream = torch.cuda.Stream(device=model.device, priority=-1)
            self.geometry_stream = torch.cuda.Stream(device=model.device, priority=-1)
            self.futures = {}
            self.packs = {p: {} for p in self.percentages}
            self.raw = {}
            self.mode = "capture"
            self.fast_prefix = {}
            self.decode_calls = 0
            old = self.n - 32
            words = piece_units(
                tokenizer.convert_ids_to_tokens(input_ids[0, :old].tolist()),
                set(tokenizer.all_special_tokens),
            )
            self.membership = torch.repeat_interleave(
                torch.arange(len(words), device=model.device),
                torch.tensor([b - a for a, b in words], device=model.device),
                output_size=old,
            )
            attach_words(self.membership, words)
            self.stream.wait_stream(torch.cuda.current_stream(model.device))
            self.membership.record_stream(self.stream)
            self.modules = {id(layer.self_attn) for layer in model.model.layers}
            self.previous_attention = ALL_ATTENTION_FUNCTIONS["sdpa"]
            self.dispatch = self.attention
            ALL_ATTENTION_FUNCTIONS["sdpa"] = self.dispatch
            for li, layer in enumerate(model.model.layers):

                def capture(module, args, output, idx=li):
                    if self.mode != "capture":
                        return
                    att = self.model.model.layers[idx].self_attn
                    raw = output[:, -32:].view(1, -1, model.config.num_attention_heads, att.head_dim)
                    if hasattr(att, "q_norm"):
                        raw = att.q_norm(raw)
                    self.raw[idx] = raw.transpose(1, 2).detach().clone()

                self.handles.append(layer.self_attn.q_proj.register_forward_hook(capture))
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()

    def _attend(self, module, q, k, v, mask, **kwargs):
        if id(module) not in self.modules:
            return self.previous_attention(module, q, k, v, mask, **kwargs)
        if self.mode == "compressed":
            self.decode_calls += 1
            return self.fast_prefix[module.layer_idx].attend(q, k, v, module.scaling), None
        return sdpa_attention_forward(module, q, k, v, mask, **kwargs)

    def attention(self, module, q, k, v, mask, **kwargs):
        result = self._attend(module, q, k, v, mask, **kwargs)
        if id(module) in self.modules and self.mode == "capture":
            li = module.layer_idx
            if li in self.futures:
                raise RuntimeError("One unchunked prefill per runtime is required.")
            ready = torch.cuda.Event()
            ready.record()
            raw = self.raw.pop(li)
            job = self.geometry_pool.submit(self.prepare_layer, li, k, v, raw, ready)
            self.futures[li] = self.pool.submit(self.finish_prepared, li, job)
        return result

    @torch.inference_mode()
    def prepare_layer(self, li, keys, values, raw, event, precomputed=None):
        torch.cuda.set_device(keys.device)
        stream = self.geometry_stream
        stream.wait_event(event)
        with torch.cuda.stream(stream):
            for x in (keys, values, raw):
                x.record_stream(stream)
            h = keys.shape[1]
            d = keys.shape[-1]
            g = self.model.config.num_attention_heads // h
            if precomputed is None:
                metrics, vf = output_geometry(self.model, li, h, g, d)
                q, a = _observations(self.model, raw, SimpleNamespace(keys=keys), self.n)
                score, _ = score_layer(self.model, raw, keys, values, self.n, a, metrics, need_variance=False)
            else:
                q, a, score, (metrics, vf) = precomputed
            for x in (metrics, *vf):
                x.record_stream(stream)
            weight = word_spread(score, self.membership, 0.95)
            covariance = query_moments(self.model, raw, self.n, h)[1]
            factors, failures, active, _ = transfers.factors_and_blocks(covariance, d, weight, 0.01, block=32)
            prepared = geometry.build(keys, values, q, a, factors, vf, active)
            old = self.n - 32
            nb, kap, weights = transfers.to_host_many(
                (
                    torch.cat([prepared[2][hi] + hi * old for hi in range(h)]),
                    torch.cat([prepared[3][hi] for hi in range(h)]),
                    weight.flatten(),
                )
            )
            ready = torch.cuda.Event()
            ready.record(stream)
            return keys, values, a, weight, metrics, prepared, nb, kap, weights, ready

    def solve_layer(self, li, nb, kap, weights, steps, heads):
        if self.model.config.model_type == "qwen3":
            return parallel_facility(nb, kap, weights, steps, heads, array_marginals=True)
        return compiled_facility(nb, kap, weights, steps)

    def finish_prepared(self, li, future):
        return self.finish_layer(li, future.result())

    @torch.inference_mode()
    def finish_layer(self, li, state):
        keys, values, a, weight, metrics, prepared, nb, kap, weights, ready = state
        torch.cuda.set_device(keys.device)
        stream = self.stream
        stream.wait_event(ready)
        with torch.cuda.stream(stream):
            n = self.n
            old = n - 32
            h = keys.shape[1]
            d = keys.shape[-1]
            g = self.model.config.num_attention_heads // h
            for x in (keys, values, a, weight, metrics, *prepared):
                x.record_stream(stream)
            steps = max(h * (self.rows[p][li] - 32) for p in self.percentages)
            ix, _ = self.solve_layer(li, nb, kap, weights, steps, h)
            arrays = []
            layout = []
            for p in self.percentages:
                prefix = ix.numpy()[: h * (self.rows[p][li] - 32)]
                owner = prefix // old
                position = prefix % old
                for hi in range(h):
                    rows = np.concatenate((np.sort(position[owner == hi]), np.arange(old, n, dtype=np.int64)))
                    arrays.append(rows)
                    layout.append((p, hi, len(rows)))
            index = torch.from_numpy(np.concatenate(arrays)).pin_memory().to(keys.device, non_blocking=True)
            choices = {}
            offset = 0
            for p, hi, count in layout:
                if hi == 0:
                    choices[p] = []
                choices[p].append(index[offset : offset + count])
                offset += count
            merged = merge_candidates(values, a, metrics, prepared, choices)
            for p in self.percentages:
                keep = choices[p]
                counts = [len(x) for x in keep]
                obj = WeightedPrefix.__new__(WeightedPrefix)
                obj.hg = g
                obj.d = d
                obj.k = gather_keys(keys[0], torch.cat([x + hi * n for hi, x in enumerate(keep)]))
                obj.v = merged[p][0]
                obj.b = merged[p][1].log().float()
                offsets = np.cumsum([0] + counts[:-1]).tolist()
                obj.meta = torch.tensor(
                    [[o, c, c, o] for o, c in zip(offsets, counts)], dtype=torch.int32, pin_memory=True
                ).to(keys.device, non_blocking=True)
                obj.metadata_bytes = h * 16
                obj.max_length = max(counts)
                self.packs[p][li] = obj
            done = torch.cuda.Event()
            done.record(stream)
            return done

    def _finish(self):
        for future in self.futures.values():
            torch.cuda.current_stream(self.model.device).wait_event(future.result())
        self.futures.clear()
        for layers in self.packs.values():
            for pack in layers.values():
                for attr in ("k", "v", "b", "meta"):
                    getattr(pack, attr).record_stream(torch.cuda.current_stream(self.model.device))
        self.mode = "ready"

    @torch.inference_mode()
    def prefill(self):
        if self.mode != "capture":
            raise RuntimeError("This runtime has already prefetched a prompt.")
        out = self.model(
            self.prompt_ids,
            past_key_values=_TransientCache(config=self.model.config),
            use_cache=True,
            logits_to_keep=1,
        )
        self._finish()
        return out

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.geometry_pool:
                self.geometry_pool.shutdown(wait=True)
            if self.pool:
                self.pool.shutdown(wait=True)
            for name in ("geometry_stream", "stream"):
                stream = getattr(self, name, None)
                if stream is not None:
                    stream.synchronize()
        finally:
            for hook in self.handles:
                hook.remove()
            if self.previous_attention is not None:
                ALL_ATTENTION_FUNCTIONS["sdpa"] = self.previous_attention
            for name in ("futures", "raw", "packs", "fast_prefix"):
                if hasattr(self, name):
                    getattr(self, name).clear()
            _active_runtime.release()


class DynamicRuntime(FixedRuntime):
    """Overlap scores/geometry/greedy prefixes; allocate after all layer variances."""

    def __init__(self, model, input_ids, tokenizer, config=Config()):
        super().__init__(model, input_ids, tokenizer, config)
        try:
            self.rows = {}
            self.budget_ready = threading.Event()
            self.score_pool = ThreadPoolExecutor(max_workers=1)
            self.score_stream = torch.cuda.Stream(device=model.device, priority=-1)
            self.spec_pool = ThreadPoolExecutor(max_workers=1)
            self.score_jobs = {}
            self.spec_sequences = {}
            self.spec_extensions = 0
        except BaseException:
            self.close()
            raise

    def attention(self, module, q, k, v, mask, **kwargs):
        result = self._attend(module, q, k, v, mask, **kwargs)
        if id(module) in self.modules and self.mode == "capture":
            li = module.layer_idx
            if li in self.futures:
                raise RuntimeError("One unchunked prefill per runtime is required.")
            ready = torch.cuda.Event()
            ready.record()
            raw = self.raw.pop(li)
            self.score_jobs[li] = self.score_pool.submit(self.score_layer, li, k, v, raw, ready)
            graph = self.geometry_pool.submit(self.prepare_signaled, li, k, v, raw, self.score_jobs[li])
            selection = self.spec_pool.submit(self.prepare_selection, li, graph)
            self.futures[li] = self.pool.submit(self.finish_prepared, li, selection)
        return result

    @torch.inference_mode()
    def score_layer(self, li, keys, values, raw, ready):
        torch.cuda.set_device(keys.device)
        self.score_stream.wait_event(ready)
        with torch.cuda.stream(self.score_stream):
            for x in (keys, values, raw):
                x.record_stream(self.score_stream)
            q, a = _observations(self.model, raw, SimpleNamespace(keys=keys), self.n)
            h = keys.shape[1]
            g = self.model.config.num_attention_heads // h
            factors = output_geometry(self.model, li, h, g, keys.shape[-1])
            score, var = score_layer(self.model, raw, keys, values, self.n, a, factors[0], need_variance=True)
            done = torch.cuda.Event()
            done.record()
            return q, a, score, var, done, factors

    def prepare_signaled(self, li, keys, values, raw, future):
        q, a, score, _, ready, factors = future.result()
        with torch.cuda.stream(self.geometry_stream):
            for x in (q, a, score):
                x.record_stream(self.geometry_stream)
        return super().prepare_layer(li, keys, values, raw, ready, (q, a, score, factors))

    def prepare_selection(self, li, future):
        state = future.result()
        keys = state[0]
        h = keys.shape[1]
        steps = min(h * (self.n - 32), 2 * h * (max(self.budgets.values()) - 32))
        self.spec_sequences[li] = super().solve_layer(li, state[6], state[7], state[8], steps, h)
        return state

    def solve_layer(self, li, nb, kap, weights, steps, heads):
        if li in self.spec_sequences:
            ix, info = self.spec_sequences.pop(li)
            if len(ix) >= steps:
                return ix[:steps], dict(info, marginals=info["marginals"][:steps])
            self.spec_extensions += 1
        return super().solve_layer(li, nb, kap, weights, steps, heads)

    def finish_prepared(self, li, future):
        self.budget_ready.wait()
        if not self.rows:
            raise RuntimeError("Global allocation did not finish.")
        return super().finish_prepared(li, future)

    @torch.inference_mode()
    def prefill(self):
        if self.mode != "capture":
            raise RuntimeError("This runtime has already prefetched a prompt.")
        try:
            out = self.model(self.prompt_ids, use_cache=True, logits_to_keep=1)
            variances = []
            for li in range(len(self.model.model.layers)):
                _, _, _, var, event, _ = self.score_jobs[li].result()
                self.stream.wait_event(event)
                var.record_stream(self.stream)
                variances.append(var)
            with torch.cuda.stream(self.stream):
                self.variance_values = transfers.to_host(torch.stack(variances)).tolist()
            self.rows = {p: apportion(self.variance_values, b, self.n) for p, b in self.budgets.items()}
            self.budget_ready.set()
            self._finish()
            self.score_jobs.clear()
            return out
        except BaseException:
            self.budget_ready.set()
            raise

    def close(self):
        if getattr(self, "closed", False):
            return
        if hasattr(self, "budget_ready"):
            self.budget_ready.set()
        if hasattr(self, "score_pool"):
            self.score_pool.shutdown(wait=True)
        if hasattr(self, "spec_pool"):
            self.spec_pool.shutdown(wait=True)
        super().close()
        if hasattr(self, "score_stream"):
            self.score_stream.synchronize()
        if hasattr(self, "score_jobs"):
            self.score_jobs.clear()
        if hasattr(self, "spec_sequences"):
            self.spec_sequences.clear()
