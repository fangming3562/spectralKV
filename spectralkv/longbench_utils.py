#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, rotate_half

DEFAULT_MODEL_PATH = "/home/ubuntu/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3___1-8B-Instruct"
DEFAULT_DATASET_PATH = str(Path(__file__).resolve().parents[1] / "data" / "longbench_mix_6x10.jsonl")

NO_CHAT_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}

DATASET2MAXLEN = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
}

DATASET2PROMPT = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": 'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: ',
    "passage_retrieval_zh": '以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是"段落1"，"段落2"等格式\n\n答案是：',
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}

MODEL2MAXLEN = {
    "llama3.1": 131_072,
    "llama-3.1": 131_072,
    "llama2": 3950,
    "llama-2": 3950,
    "llama3": 7950,
    "llama-3": 7950,
    "qwen3": 40_900,
    "qwen-3": 40_900,
    "mistral": 31500,
}


@dataclass
class LayerProbeState:
    values: torch.Tensor
    attn_mean_global: torch.Tensor
    token_key_norm_avg: torch.Tensor | None = None
    token_matrix_normalized: torch.Tensor | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_model_max_prompt_tokens(model_path: str) -> int:
    lowered = str(model_path).lower()
    for key, value in MODEL2MAXLEN.items():
        if key in lowered:
            return int(value)
    return 0


def build_chat_prompt(tokenizer, prompt: str, model_path: str) -> str:
    lowered = str(model_path).lower()
    if "qwen3" in lowered or "qwen-3" in lowered:
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if "qwen" in lowered:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if "llama3" in lowered or "llama-3" in lowered:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if "llama2" in lowered or "llama-2" in lowered:
        return f"[INST] {prompt} [/INST]"
    if "mistral" in lowered:
        return f"<s>[INST] {prompt} [/INST]"
    return prompt


def build_prompt(tokenizer, row: Dict[str, Any], model_path: str, kvfactory_prompt_truncation: bool) -> str:
    dataset = str(row["dataset"])
    prompt = DATASET2PROMPT[dataset].format(**row)
    if kvfactory_prompt_truncation:
        model_max_len = resolve_model_max_prompt_tokens(model_path)
        if model_max_len > 0:
            tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if int(tokenized.shape[0]) > model_max_len:
                half = int(model_max_len / 2)
                prompt = tokenizer.decode(tokenized[:half], skip_special_tokens=True) + tokenizer.decode(
                    tokenized[-half:],
                    skip_special_tokens=True,
                )
    if dataset not in NO_CHAT_DATASETS:
        prompt = build_chat_prompt(tokenizer, prompt, model_path)
    return prompt


def extract_cache_layers(past_key_values) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(past_key_values, "layers"):
        return [(layer.keys, layer.values) for layer in past_key_values.layers]
    if isinstance(past_key_values, DynamicCache):
        if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
            return list(zip(past_key_values.key_cache, past_key_values.value_cache))
        return [(layer[0], layer[1]) for layer in past_key_values]
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))
    if isinstance(past_key_values, tuple):
        return [(layer[0], layer[1]) for layer in past_key_values]
    raise TypeError(f"Unsupported cache type: {type(past_key_values)}")


def extract_sample_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    batch_idx: int,
    start_idx: int,
) -> Tuple[torch.Tensor, ...]:
    return tuple(state[batch_idx : batch_idx + 1, start_idx:].contiguous() for state in hidden_states)


def extract_sample_past_key_values(
    past_key_values,
    batch_idx: int,
    start_idx: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    sample_layers: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for keys, values in extract_cache_layers(past_key_values):
        sample_layers.append(
            (
                keys[batch_idx : batch_idx + 1, :, start_idx:, :].contiguous(),
                values[batch_idx : batch_idx + 1, :, start_idx:, :].contiguous(),
            )
        )
    return tuple(sample_layers)


@torch.no_grad()
def build_probe_query_grouped_from_hidden(
    *,
    model,
    hidden_in: torch.Tensor,
    layer_idx: int,
    probe_positions: Sequence[int],
) -> torch.Tensor:
    attn_module = model.model.layers[int(layer_idx)].self_attn
    proj_dtype = attn_module.q_proj.weight.dtype
    proj_device = attn_module.q_proj.weight.device
    q = attn_module.q_proj(hidden_in[0].to(device=proj_device, dtype=proj_dtype))
    num_heads = int(model.config.num_attention_heads)
    head_dim = int(attn_module.head_dim)
    q = q.view(len(probe_positions), num_heads, head_dim)
    q_norm = getattr(attn_module, "q_norm", None)
    if q_norm is not None:
        q = q_norm(q)
    q = q.permute(1, 0, 2).unsqueeze(0)
    position_ids = torch.tensor([list(probe_positions)], device=q.device, dtype=torch.long)
    cos, sin = model.model.rotary_emb(q, position_ids)
    cos = cos.to(device=q.device, dtype=q.dtype)
    sin = sin.to(device=q.device, dtype=q.dtype)
    q, _ = apply_rotary_pos_emb(q, q, cos, sin)
    q = q[0].permute(1, 0, 2).contiguous()
    num_kv_heads = int(model.config.num_key_value_heads)
    num_kv_groups = int(num_heads // max(num_kv_heads, 1))
    return q.view(len(probe_positions), num_kv_heads, num_kv_groups, head_dim)


def gather_layer_input_hidden(
    hidden_source: Sequence[torch.Tensor] | Mapping[int, torch.Tensor],
    *,
    layer_idx: int,
    positions: Sequence[int],
) -> torch.Tensor:
    if isinstance(hidden_source, Mapping):
        layer_hidden = hidden_source[int(layer_idx)]
    else:
        layer_hidden = hidden_source[int(layer_idx)]
    return layer_hidden[0][list(positions)]


@torch.no_grad()
def build_probe_query_grouped(
    *,
    model,
    hidden_source: Sequence[torch.Tensor] | Mapping[int, torch.Tensor],
    layer_idx: int,
    probe_positions: Sequence[int],
) -> torch.Tensor:
    hidden_in = gather_layer_input_hidden(
        hidden_source,
        layer_idx=int(layer_idx),
        positions=probe_positions,
    )
    return build_probe_query_grouped_from_hidden(
        model=model,
        hidden_in=hidden_in,
        layer_idx=int(layer_idx),
        probe_positions=probe_positions,
    )


@torch.no_grad()
def build_layer_probe_states(
    *,
    model,
    hidden_source: Sequence[torch.Tensor] | Mapping[int, torch.Tensor],
    past_key_values,
    attention_layer_indices: Sequence[int],
    probe_positions: Sequence[int],
    prefix_len: int,
) -> Dict[int, LayerProbeState]:
    layer_pairs = extract_cache_layers(past_key_values)
    states: Dict[int, LayerProbeState] = {}
    for layer_idx in attention_layer_indices:
        keys, values = layer_pairs[int(layer_idx)]
        keys = keys[0, :, :prefix_len, :].contiguous()
        values = values[0, :, :prefix_len, :].contiguous()
        q_grouped = build_probe_query_grouped(
            model=model,
            hidden_source=hidden_source,
            layer_idx=int(layer_idx),
            probe_positions=probe_positions,
        )
        logits_all = torch.einsum("mhgd,hsd->mhgs", q_grouped, keys) / math.sqrt(keys.shape[-1])
        attn_all = torch.softmax(logits_all, dim=-1, dtype=torch.float32)
        attn_mean_global = attn_all.mean(dim=(0, 2))
        token_key_norm_avg = keys.float().norm(dim=-1).mean(dim=0)
        states[int(layer_idx)] = LayerProbeState(
            values=values,
            attn_mean_global=attn_mean_global,
            token_key_norm_avg=token_key_norm_avg,
        )
    return states


@torch.no_grad()
def build_layer_probe_state_from_batch(
    *,
    model,
    query_hidden: torch.Tensor,
    past_key_values,
    layer_idx: int,
    probe_positions: Sequence[int],
    prefix_len: int,
    batch_idx: int,
    start_idx: int,
    need_token_key_norm: bool = True,
    need_token_matrix_normalized: bool = True,
) -> LayerProbeState:
    layer_pairs = extract_cache_layers(past_key_values)
    keys, values = layer_pairs[int(layer_idx)]
    keys = keys[int(batch_idx), :, int(start_idx) : int(start_idx) + int(prefix_len), :].contiguous()
    values = values[int(batch_idx), :, int(start_idx) : int(start_idx) + int(prefix_len), :].contiguous()
    q_grouped = build_probe_query_grouped_from_hidden(
        model=model,
        hidden_in=query_hidden,
        layer_idx=int(layer_idx),
        probe_positions=probe_positions,
    )
    logits_all = torch.einsum("mhgd,hsd->mhgs", q_grouped, keys) / math.sqrt(keys.shape[-1])
    attn_all = torch.softmax(logits_all, dim=-1, dtype=torch.float32)
    attn_mean_global = attn_all.mean(dim=(0, 2))
    token_key_norm_avg = None
    if bool(need_token_key_norm):
        token_key_norm_avg = keys.float().norm(dim=-1).mean(dim=0)
    token_matrix_normalized = None
    if bool(need_token_matrix_normalized):
        token_matrix_normalized = F.normalize(
            values.permute(1, 0, 2).contiguous().reshape(int(prefix_len), -1).float(),
            dim=-1,
        )
    return LayerProbeState(
        values=values,
        attn_mean_global=attn_mean_global,
        token_key_norm_avg=token_key_norm_avg,
        token_matrix_normalized=token_matrix_normalized,
    )


def build_dynamic_cache(layer_pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> DynamicCache:
    cache = DynamicCache()
    for layer_idx, (keys, values) in enumerate(layer_pairs):
        cache.update(keys, values, layer_idx)
    return cache


def greedy_generate(
    *,
    model,
    tokenizer,
    past_key_values: DynamicCache,
    next_logits: torch.Tensor,
    prompt_len: int,
    max_new_tokens: int,
) -> Tuple[List[int], str]:
    generated: List[int] = []
    past = past_key_values
    eos_token_id = tokenizer.eos_token_id
    with torch.inference_mode():
        for step in range(int(max_new_tokens)):
            next_token = int(torch.argmax(next_logits, dim=-1).item())
            if eos_token_id is not None and next_token == int(eos_token_id):
                break
            generated.append(next_token)
            input_ids = torch.tensor([[next_token]], device=next_logits.device, dtype=torch.long)
            current_len = int(past.get_seq_length())
            out = model(
                input_ids=input_ids,
                past_key_values=past,
                use_cache=True,
                cache_position=torch.tensor([current_len], device=next_logits.device, dtype=torch.long),
                position_ids=torch.tensor([[prompt_len + step]], device=next_logits.device, dtype=torch.long),
                attention_mask=torch.ones((1, current_len + 1), device=next_logits.device, dtype=torch.long),
            )
            past = out.past_key_values
            next_logits = out.logits[:, -1, :]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return generated, text


def greedy_generate_batched(
    *,
    model,
    tokenizer,
    past_key_values: DynamicCache,
    next_logits: torch.Tensor,
    prompt_lens: Sequence[int],
    past_lens: Sequence[int],
    max_new_tokens: int,
) -> Tuple[List[List[int]], List[str]]:
    batch_size = int(next_logits.shape[0])
    if batch_size != int(len(prompt_lens)) or batch_size != int(len(past_lens)):
        raise ValueError(
            f"Batch size mismatch: logits={batch_size}, prompt_lens={len(prompt_lens)}, past_lens={len(past_lens)}"
        )

    generated: List[List[int]] = [[] for _ in range(batch_size)]
    past = past_key_values
    eos_token_id = tokenizer.eos_token_id
    dummy_token_id = tokenizer.pad_token_id
    if dummy_token_id is None:
        dummy_token_id = eos_token_id if eos_token_id is not None else 0

    max_past_len = int(max((int(length) for length in past_lens), default=0))
    if int(past.get_seq_length()) != int(max_past_len):
        raise ValueError(
            f"Batched cache length mismatch: cache={past.get_seq_length()} expected={max_past_len}"
        )

    attention_mask = torch.zeros(
        (batch_size, max_past_len),
        device=next_logits.device,
        dtype=torch.long,
    )
    for row_idx, past_len in enumerate(past_lens):
        if int(past_len) > 0:
            attention_mask[int(row_idx), : int(past_len)] = 1

    finished = torch.zeros(batch_size, device=next_logits.device, dtype=torch.bool)
    with torch.inference_mode():
        for step in range(int(max_new_tokens)):
            next_tokens = torch.argmax(next_logits, dim=-1)
            was_finished = finished.clone()

            for row_idx in range(batch_size):
                if bool(was_finished[int(row_idx)].item()):
                    continue
                token_id = int(next_tokens[int(row_idx)].item())
                if eos_token_id is not None and token_id == int(eos_token_id):
                    continue
                generated[int(row_idx)].append(token_id)

            if eos_token_id is not None:
                finished = finished | next_tokens.eq(int(eos_token_id))
            if bool(finished.all().item()):
                break

            model_input_ids = next_tokens.clone()
            model_input_ids[finished] = int(dummy_token_id)
            current_len = int(past.get_seq_length())
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((batch_size, 1), device=next_logits.device, dtype=torch.long),
                ],
                dim=1,
            )
            position_ids = torch.tensor(
                [[int(prompt_lens[row_idx]) + int(step)] for row_idx in range(batch_size)],
                device=next_logits.device,
                dtype=torch.long,
            )
            out = model(
                input_ids=model_input_ids.unsqueeze(1),
                past_key_values=past,
                use_cache=True,
                cache_position=torch.tensor([current_len], device=next_logits.device, dtype=torch.long),
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            past = out.past_key_values
            next_logits = out.logits[:, -1, :]

    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return generated, [str(text) for text in texts]
