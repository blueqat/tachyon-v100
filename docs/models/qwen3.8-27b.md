# Qwen3.8-27B on Tachyon-v100

Run [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (dense, 64 layers, Gated DeltaNet linear attention 3:1 with full attention, MTP head)
on a single NVIDIA V100 32 GB with [Tachyon-v100](https://github.com/blueqat/tachyon-v100).
This page contains **no model weights**; it tells you which upstream files to use and what to expect.

## Weights (download from upstream, unmodified)

| File | Repository | Size | Use |
|---|---|---|---|
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | 17.7 GB | main model (`--gguf`) |
| `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` | same | 1.4 GB | MTP draft head (`--mtp-gguf`) |
| `mmproj-F16.gguf` | same | 0.93 GB | vision tower (`--mmproj`) |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | small | `--hf <dir>` |
| `draft_vocab_q38.txt` | this repository (`data/`) | 0.4 MB | reduced vocabulary for the MTP lm_head (`--draft-vocab`) |

Pin the upstream revision and record the sha256 of each file when you publish numbers.

```sh
huggingface-cli download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf MTP/mtp-Qwen3.8-27B-Q4_0.gguf mmproj-F16.gguf --local-dir weights/q38-27b
huggingface-cli download Qwen/Qwen3.8-27B tokenizer.json tokenizer_config.json chat_template.jinja --local-dir weights/q38-27b/hf
CUDA_VISIBLE_DEVICES=0 target/release/tachyon serve --gguf weights/q38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf --hf weights/q38-27b/hf \
  --mtp-gguf weights/q38-27b/MTP/mtp-Qwen3.8-27B-Q4_0.gguf --draft-vocab data/draft_vocab_q38.txt \
  --mmproj weights/q38-27b/mmproj-F16.gguf --keys keys.json --port 8080 --ctx 65536 --max-seqs 4 --max-running 4
```

## Benchmarks (preliminary)

One V100-SXM3-32GB, same GGUF for both engines, llama-server = llama.cpp (unsloth build b12209). Internal measurements, partly on a shared GPU; to be re-measured on the current upstream files before quoting.

| | Tachyon-v100 | llama-server |
|---|---|---|
| Decode, 1 user, greedy, no draft | 40 tok/s | 35–37 tok/s |
| Decode with MTP (k=3, 12 prompts aggregate) | 75 tok/s | 51–53 tok/s |
| TTFT, 550-token prompt | 0.70 s | 1.0 s |
| TTFT, 8K-token prompt | 7.9 s | 11.4 s |
| HTTP, temp 0, MTP: Japanese prose / code / tool call | 59 / 105–122 / 102–111 tok/s | — |
| HTTP, temp 0.7, GPU sampling, MTP k=6: JP prose / JP chat / code | 61 / 62 / 102 tok/s | — |
| 4 concurrent short requests (total) | 107 tok/s | — |

## Accuracy

- Top-1 agreement with llama-server on 742 teacher-forced positions: 740 (both differences are near-ties)
- Against a CPU float32 reference (`tools/ref_qwen35.py`): 192/192
- With MTP, temperature-0 output is token-identical to decoding without a draft

## License

- Tachyon-v100 and the small files here (`draft_vocab_q38.txt`, examples): Apache-2.0, Copyright 2026 blueqat Inc.
- Qwen3.8-27B: Apache-2.0, released by the Qwen team (see the [LICENSE](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE) of the upstream repository). The GGUF conversions are by unsloth under the same license.
- This is not an official Qwen release.
