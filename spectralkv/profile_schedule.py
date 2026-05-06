from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .api import SpectralKVConfig, estimate_layer_budget_schedule
from .longbench_utils import build_prompt, load_jsonl, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate a static SpectralKV layer-budget schedule from a calibration JSONL. "
            "This runs full prefill and dynamic budget estimation only; it does not decode."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--prompt-field", type=str, default="")
    parser.add_argument("--text-field", type=str, default="input")
    parser.add_argument("--use-longbench-template", action="store_true")
    parser.add_argument("--prompt-truncation", action="store_true")
    parser.add_argument("--max-prompt-tokens", type=int, default=0)
    parser.add_argument("--keep-ratio", type=float, default=0.0625)
    parser.add_argument("--fixed-budget", type=int, default=0)
    parser.add_argument("--observation-window", type=int, default=8)
    parser.add_argument("--force-sink", type=int, default=4)
    parser.add_argument("--force-recent", type=int, default=8)
    parser.add_argument("--local-chunk-size", type=int, default=16)
    parser.add_argument("--layer-budget-rho", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    rows = load_jsonl(str(args.dataset_path))
    rows = rows[max(int(args.start_index), 0) :]
    if int(args.max_samples) > 0:
        rows = rows[: int(args.max_samples)]
    if not rows:
        raise ValueError("No calibration rows selected.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation=str(args.attn_implementation),
        use_cache=True,
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    input_device = model.get_input_embeddings().weight.device

    config = SpectralKVConfig(
        keep_ratio=float(args.keep_ratio),
        fixed_budget=int(args.fixed_budget),
        observation_window=int(args.observation_window),
        force_sink=int(args.force_sink),
        force_recent=int(args.force_recent),
        local_chunk_size=int(args.local_chunk_size),
        layer_budget_rho=float(args.layer_budget_rho),
    )

    sample_summaries: list[dict[str, Any]] = []
    ratio_accum: torch.Tensor | None = None
    keep_accum: torch.Tensor | None = None
    score_accum: torch.Tensor | None = None
    for sample_idx, row in enumerate(rows, start=1):
        prompt = build_calibration_prompt(
            tokenizer=tokenizer,
            row=row,
            model_path=str(args.model_path),
            args=args,
        )
        encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
        input_ids = encoded["input_ids"].to(input_device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(input_device)
        info = estimate_layer_budget_schedule(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            config=config,
        )
        ratios = torch.tensor(info["layer_budget_ratios"], dtype=torch.float64)
        keep_ratios = torch.tensor(info["layer_keep_ratios"], dtype=torch.float64)
        scores = torch.tensor(info["layer_scores"], dtype=torch.float64)
        ratio_accum = ratios if ratio_accum is None else ratio_accum + ratios
        keep_accum = keep_ratios if keep_accum is None else keep_accum + keep_ratios
        score_accum = scores if score_accum is None else score_accum + scores
        sample_summaries.append(
            {
                "sample_idx": int(sample_idx),
                "row_id": row.get("_id", row.get("id", sample_idx)),
                "dataset": row.get("dataset"),
                "prompt_tokens": int(input_ids.shape[-1]),
                "total_budget": int(info["total_budget"]),
                "layer_budgets": info["layer_budgets"],
                "layer_budget_ratios": info["layer_budget_ratios"],
                "layer_keep_ratios": info["layer_keep_ratios"],
                "layer_scores": info["layer_scores"],
            }
        )
        print(
            "[schedule] sample={sample} prompt={prompt} total_budget={budget}".format(
                sample=int(sample_idx),
                prompt=int(input_ids.shape[-1]),
                budget=int(info["total_budget"]),
            ),
            flush=True,
        )
        del input_ids, attention_mask, encoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    count = len(sample_summaries)
    mean_ratios = (ratio_accum / max(count, 1)).tolist() if ratio_accum is not None else []
    # Re-normalize the average because samples can have different force floors.
    denom = max(float(sum(float(value) for value in mean_ratios)), 1e-12)
    static_schedule = [float(value) / denom for value in mean_ratios]
    output = {
        "model_path": str(args.model_path),
        "dataset_path": str(args.dataset_path),
        "num_samples": int(count),
        "config": config.__dict__,
        "static_schedule": static_schedule,
        "mean_layer_budget_ratios": mean_ratios,
        "mean_layer_keep_ratios": (keep_accum / max(count, 1)).tolist() if keep_accum is not None else [],
        "mean_layer_scores": (score_accum / max(count, 1)).tolist() if score_accum is not None else [],
        "samples": sample_summaries,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[schedule] wrote {output_path}", flush=True)


def build_calibration_prompt(*, tokenizer, row: dict[str, Any], model_path: str, args: argparse.Namespace) -> str:
    if str(args.prompt_field):
        prompt = str(row[str(args.prompt_field)])
    elif bool(args.use_longbench_template) or ("dataset" in row and "context" in row):
        prompt = build_prompt(
            tokenizer,
            row,
            model_path=str(model_path),
            kvfactory_prompt_truncation=bool(args.prompt_truncation),
        )
    else:
        prompt = str(row[str(args.text_field)])
    if int(args.max_prompt_tokens) > 0:
        tokenized = tokenizer(prompt, return_attention_mask=False, truncation=False)["input_ids"]
        if len(tokenized) > int(args.max_prompt_tokens):
            prompt = tokenizer.decode(tokenized[-int(args.max_prompt_tokens) :], skip_special_tokens=False)
    return prompt


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
