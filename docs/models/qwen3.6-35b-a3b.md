# Qwen3.6-35B-A3B on Tachyon-v100

Run [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (MoE: 256 experts, top-8 + a shared expert, about 3B active parameters, same Gated DeltaNet / attention blocks as Qwen3.8, MTP head)
on a single NVIDIA V100 32 GB with [Tachyon-v100](https://github.com/blueqat/tachyon-v100).
This page contains **no model weights**; it tells you which upstream files to use and what to expect.

## Weights (download from upstream, unmodified)

| File | Repository | Size | Use |
|---|---|---|---|
| `Qwen3.6-35B-A3B-Q4_K_M.gguf` | [ggml-org/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF) | 20.5 GB | main model (`--gguf`) |
| `mtp-Qwen3.6-35B-A3B-Q8_0.gguf` | same | 2.0 GB | MTP draft head with MoE (`--mtp-gguf`) |
| `mmproj-F16.gguf` | [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | 0.90 GB | vision tower (`--mmproj`; Tachyon reads F16 only. Alternatively convert ggml-org's `mmproj-Qwen3.6-35B-A3B-Q8_0.gguf` with `tools/mmproj_to_f16.py`) |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | small | `--hf <dir>` |
| `draft_vocab_q38.txt` | this repository (`data/`) | 0.4 MB | same tokenizer as Qwen3.8; reduced vocabulary for the MTP lm_head |

Pin the upstream revision and record the sha256 of each file when you publish numbers.

```sh
huggingface-cli download ggml-org/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-Q4_K_M.gguf mtp-Qwen3.6-35B-A3B-Q8_0.gguf --local-dir weights/q36
huggingface-cli download unsloth/Qwen3.6-35B-A3B-GGUF mmproj-F16.gguf --local-dir weights/q36
huggingface-cli download Qwen/Qwen3.6-35B-A3B tokenizer.json tokenizer_config.json chat_template.jinja --local-dir weights/q36/hf
CUDA_VISIBLE_DEVICES=0 target/release/tachyon serve --gguf weights/q36/Qwen3.6-35B-A3B-Q4_K_M.gguf --hf weights/q36/hf \
  --mtp-gguf weights/q36/mtp-Qwen3.6-35B-A3B-Q8_0.gguf --draft-vocab data/draft_vocab_q38.txt \
  --mmproj weights/q36/mmproj-F16.gguf --keys keys.json --port 8080 --ctx 98304 --max-seqs 4 --max-running 4 --thinking-default on
```

## Benchmarks (preliminary)

One V100-SXM3-32GB, same GGUF for both engines. llama-server = llama.cpp with `-c 131072 -np 4 -ub 2048 -b 2048`, KV q8_0.
Internal measurements, partly on a shared GPU; to be re-measured on the current upstream files before quoting.

| | Tachyon-v100 | llama-server |
|---|---|---|
| Decode, 1 user, temp 0, no draft | 120 tok/s | 99 tok/s |
| Decode, 1 user, temp 0.7 (GPU sampling) | 117 tok/s | 98.5 tok/s |
| TTFT, 1K / 8K prompt | 0.29 s / 2.02 s | 0.80 s / 3.05 s |
| MTP k=6, temp 0: JP prose / JP chat / code | 171 / 153 / 268 tok/s | not yet measured under the same conditions |
| TTFT, 17K-token prompt | 3.71 s (4,630 tok/s prefill) | — |
| Next turn of the same 17K conversation (RAM prefix tier) | 0.27 s | — |
| 4 / 8 concurrent streams, temp 0.7 (total) | 166.7 / 167 tok/s | 159 (np 4) / 180 (np 8) tok/s |
| JSON mode (`response_format`) | 187.5 tok/s | 59.3 tok/s (grammar) |

At 8 streams llama-server is faster: batching tokens that picked the same expert is not implemented yet.

## Accuracy

Against a CPU float32 reference (6 prompts, 384 teacher-forced positions):

| | Top-1 agreement | Mean KL (top-20) | Max KL |
|---|---|---|---|
| Tachyon-v100 | 98.2% | 0.0006 | 0.044 |
| llama-server | 97.6% | 0.0024 | 0.132 |

With MTP, temperature-0 output is token-identical to decoding without a draft.

## License

- Tachyon-v100 and the small files here (`draft_vocab_q38.txt`, examples): Apache-2.0, Copyright 2026 blueqat Inc.
- Qwen3.6-35B-A3B: Apache-2.0, Copyright 2026 Alibaba Cloud (see the [LICENSE](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE) of the upstream repository). The GGUF conversions are by ggml-org and unsloth under the same license.
- This is not an official Qwen release.
