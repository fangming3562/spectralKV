"""Run a local or Hugging Face model with one shared prefill."""

import argparse
import json
from pathlib import Path

from .config import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt-file", type=Path)
    source.add_argument("--input-ids", type=Path, help="JSON token list, or an object with input_ids")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--layer-budget", choices=["dynamic", "fixed"], default="dynamic")
    parser.add_argument("--profile", help="llama31_8b, qwen3_4b, or a profile JSON path")
    parser.add_argument("--percentages", default="5,10")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    config = Config(
        percentages=tuple(map(int, args.percentages.split(","))),
        layer_budget=args.layer_budget,
        profile=args.profile,
    )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .api import generate

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=args.local_files_only,
        )
        .to(args.device)
        .eval()
    )
    if args.input_ids:
        ids = json.loads(args.input_ids.read_text())
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        input_ids = torch.tensor([ids], device=args.device, dtype=torch.long)
    else:
        prompt = args.prompt_file.read_text()
        if args.chat_template:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
            )
            input_ids = torch.tensor([ids], device=args.device, dtype=torch.long)
        else:
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(args.device)
    result = generate(model, tokenizer, input_ids, config=config, max_new_tokens=args.max_new_tokens)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
