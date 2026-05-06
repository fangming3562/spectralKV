# Data

This directory contains actual JSONL evaluation data used by the SpectralKV
experiments. It does not contain model weights.

## LongBench

`longbench/` contains the 1500-example LongBench mixture split by task, with
100 examples per task:

- `narrativeqa.jsonl`
- `qasper.jsonl`
- `multifieldqa_en.jsonl`
- `hotpotqa.jsonl`
- `2wikimqa.jsonl`
- `musique.jsonl`
- `gov_report.jsonl`
- `qmsum.jsonl`
- `multi_news.jsonl`
- `trec.jsonl`
- `triviaqa.jsonl`
- `samsum.jsonl`
- `passage_retrieval_en.jsonl`
- `lcc.jsonl`
- `repobench-p.jsonl`

Each row is one LongBench prompt example.

## PG-19

`pg-19/` contains tokenized continuation-evaluation subsets. Each row stores a
tokenized prompt and a 512-token continuation target.

- `pg19_llama31_77_p16k_t512_tokens.jsonl`: Llama-3.1 tokenizer, 16K prompt.
- `pg19_llama31_80_p32k_t512_tokens.jsonl`: Llama-3.1 tokenizer, 32K prompt.
- `pg19_qwen3_77_p16k_t512_tokens.jsonl`: Qwen3 tokenizer, 16K prompt.
- `pg19_qwen3_80_p32k_t512_tokens.jsonl`: Qwen3 tokenizer, 32K prompt.
