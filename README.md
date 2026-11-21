English | [日本語](README.ja.md)

# Tachyon-v100

A from-scratch LLM inference engine in Rust with hand-written CUDA kernels for NVIDIA V100 (sm_70).

## Status

Research preview. Current release: **Tachyon V100 Loop** (adds looped-transformer models, see [Loop models](#loop-models-tachyon-v100-loop)).

### Changes in Tachyon V100 Loop

- Loop models: SenseNova-Loop-9B (`sensenova_loop`) and Inkling-2-Mini (`inkling2_loop`); `--repeat-layers` (research option, off by default)
- `--kv8`: int8 KV cache (32-element scales; the MTP head stays fp16)
- YaRN RoPE scaling (`--rope-scaling yarn --rope-factor F --rope-orig N`, or from the GGUF `rope.scaling.*` keys)
- Prefix-cache owners per end user behind a trusted proxy (`--prefix-owner-accounts`, header `x-tachyon-prefix-owner`; off by default).
  Only list accounts whose key is held by your own proxy, and make that proxy set (overwrite) the header itself: a client that can send the header for such an account chooses whose cache it shares
- Seeded sampling noise shared by MTP draft and target (`TACHYON_HASH_RNG`, off by default), RAM prefix tier fixes
- Qwen3.8-27B and Qwen3.6-35B-A3B: on our 12 golden prompts (greedy, 64 tokens, plus teacher-forced top-20 logprobs on the prefill path) the outputs are bit-identical to the engine before the loop work

## Why V100

V100 is dropped by vLLM >= 0.20, and FlashAttention 2/3 do not support it. V100 has no bf16, no int8/fp8 tensor cores, and no `cp.async`. Tachyon-v100 provides its own fp16 kernels (Volta `mma`), its own FA1-style tiled attention, and quantized GEMV that reads GGUF blocks directly.

## Supported models

Per-model pages with download commands, benchmarks and license notes: [docs/models/qwen3.8-27b.md](docs/models/qwen3.8-27b.md), [docs/models/qwen3.6-35b-a3b.md](docs/models/qwen3.6-35b-a3b.md),
[docs/models/sensenova-loop-9b.md](docs/models/sensenova-loop-9b.md), [docs/models/inkling-2-mini.md](docs/models/inkling-2-mini.md) (loop models, see below).

| Model | GGUF files | Notes |
|---|---|---|
| Qwen3.8-27B | `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL (17.5 GB) + MTP Q4_0 + mmproj-F16 | Dense, Gated DeltaNet linear attention 3:1 with full attention, 64 layers, MTP head |
| Qwen3.6-35B-A3B | `ggml-org/Qwen3.6-35B-A3B-GGUF` Q4_K_M + `mtp-Qwen3.6-35B-A3B-Q8_0` + mmproj (F16; Q8_0 mmproj converted with `tools/mmproj_to_f16.py`) | MoE 256 experts top-8 + shared expert, same GDN/attention blocks, MTP head |

Reads unmodified GGUF from the llama.cpp ecosystem.

## Loop models (Tachyon V100 Loop)

Looped transformers run part of their layer stack several times per token with shared weights. Tachyon keeps the weights once in VRAM
and gives every execution its own KV cache and recurrent state, so prefill, CUDA Graph decode, MTP, prefix caching and batching work unchanged.
Architecture notes: [docs/LOOP.md](docs/LOOP.md).

| Model | Weights (download from upstream; not redistributed here) | Loop | Page |
|---|---|---|---|
| SenseNova-Loop-9B | [sensenova/SenseNova-Loop-9B-GGUF](https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF) Q4_K_M or Q8_0 (MTP head inside) + mmproj-F16; tokenizer from [sensenova/SenseNova-Loop-9B-AWQ](https://huggingface.co/sensenova/SenseNova-Loop-9B-AWQ) | layers 4..27 x2 = 56 executions, two-stream mHC (Sinkhorn mixing per token) | [docs/models/sensenova-loop-9b.md](docs/models/sensenova-loop-9b.md) |
| Inkling-2-Mini | [ggml-org/Inkling-2-Mini-GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) IQ4_NL (ggml-org conversion of the official weights, arch `inkling2_loop`); tokenizer from [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) | MoE 128 experts top-8; layers 6..17 x3 = 48 executions, restart passes with depth-attention carry; no RoPE (learned position bias) | [docs/models/inkling-2-mini.md](docs/models/inkling-2-mini.md) |

Inkling-2-Mini is also subject to the [Thinking Machines Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/)
(by using the model you agree to it; if you deploy it to others you are responsible for their compliance, must tell users about known
limitations and that they are talking to an AI where required, and must not use it for automated decisions affecting people's rights,
such as financial, legal, employment, healthcare, housing, insurance or social-services decisions). See the model page.

```sh
# SenseNova-Loop-9B (Apache-2.0)
huggingface-cli download sensenova/SenseNova-Loop-9B-GGUF SenseNova-Loop-9B-Q4_K_M.gguf mmproj-SenseNova-Loop-9B-F16.gguf --local-dir weights/loop9b
huggingface-cli download sensenova/SenseNova-Loop-9B-AWQ tokenizer.json tokenizer_config.json chat_template.jinja --local-dir weights/loop9b/hf
CUDA_VISIBLE_DEVICES=0 target/release/tachyon serve --gguf weights/loop9b/SenseNova-Loop-9B-Q4_K_M.gguf --hf weights/loop9b/hf \
  --mmproj weights/loop9b/mmproj-SenseNova-Loop-9B-F16.gguf --draft-vocab data/draft_vocab_q38.txt \
  --model-name sensenova-loop-9b --keys keys.json --host 127.0.0.1 --port 8080

# Inkling-2-Mini (Apache-2.0 + Model AUP)
huggingface-cli download ggml-org/Inkling-2-Mini-GGUF Inkling-2-Mini-IQ4_NL.gguf --local-dir weights/inkling2mini
huggingface-cli download thinkingmachines/Inkling-2-Mini tokenizer.json tokenizer_config.json chat_template.jinja --local-dir weights/inkling2mini/hf
CUDA_VISIBLE_DEVICES=0 target/release/tachyon serve --gguf weights/inkling2mini/Inkling-2-Mini-IQ4_NL.gguf --hf weights/inkling2mini/hf \
  --model-name inkling-2-mini --kv8 --keys keys.json --host 127.0.0.1 --port 8080 --ctx 32768 --max-seqs 8 --max-running 8
```



Verified numbers (one Tesla V100-SXM3-32GB, CUDA 12.5, temperature 0; preliminary internal measurements, details and conditions on the model pages):

| | SenseNova-Loop-9B Q4_K_M | Inkling-2-Mini IQ4_NL |
|---|---|---|
| Top-1 agreement vs the official code on CPU f32 (5 prompts) | 107/107 with Q8_0 (101/107 with Q4_K_M) | 104/107 (97.2%) |
| Decode, 1 user, no draft | 55 tok/s | 62-64 tok/s |
| Decode with MTP | 98 tok/s (k=3) | — (draft head not ported) |
| TTFT, 8K-token prompt | 4.1 s | 6.7 s |
| Concurrent streams (total) | 197 / 296 / 391 tok/s at 4 / 8 / 16 | 154 / 196 / 238 / 270 tok/s at 4 / 8 / 16 / 32 |

Limitations: Inkling-2-Mini vision and its block-draft (MTP) head are not ported yet. `--repeat-layers START:END:N` (repeating layers of a model
that was not trained to loop) is included for research only: on Qwen3.8-27B, in all 16 settings we tried (2-4 passes over 4-32 layers), next-token NLL got worse and reasoning accuracy dropped (docs/LOOP.md, section 4).

## Features

- OpenAI-compatible `/v1/chat/completions`: streaming, usage, `tools`/`tool_choice`, `response_format` json_object/json_schema subset (via token masking), `chat_template_kwargs` (`enable_thinking`/`reasoning_effort`), thinking budget, `stop`, `seed`, `top_k`/`min_p`, presence/frequency/repetition penalty, image input as data URI via mmproj
- `/v1/models`, `/health`
- API keys stored as sha256, with per-account rate/concurrency limits
- Prefix cache isolated per account
- Paged KV cache with RAM tier
- Continuous batching (up to 4-8 sequences)
- CUDA Graph decode
- MTP speculative decoding (greedy output identical to no-draft)
- GPU sampling

Not supported: `logprobs`, `n>1`, `/props`. Single GPU only (TP2 in progress).

## Benchmarks

Single V100-SXM3-32GB, same GGUF weights, `llama-server` = llama.cpp unsloth build b12209, single user unless noted. **Numbers are preliminary**: internal measurements on shared GPUs, to be re-measured before release.

### Qwen3.8-27B UD-Q4_K_XL

| Metric | Tachyon | llama-server |
|---|---|---|
| Decode, 1 user, greedy, no draft | 40 tok/s | 35-37 tok/s |
| With MTP | 75 tok/s (k=3, aggregate of 12 golden prompts) | 51-53 tok/s |
| TTFT, 550-token prompt | 0.70 s | 1.0 s |
| TTFT, 8K prompt | 7.9 s | 11.4 s |
| 4 concurrent short requests | 107 tok/s total | — |

HTTP, temp 0, 400 tokens, MTP: Japanese prose 59, code 105-122, tool call 102-111 tok/s. Temp 0.7 with GPU sampling (MTP k=6): JP prose 61, JP chat 62, code 102 tok/s.

### Qwen3.6-35B-A3B Q4_K_M

llama-server with the same flags as our deployment: `-c 131072 -np 4 -ub/-b 2048`, KV q8_0.

| Metric | Tachyon | llama-server |
|---|---|---|
| Decode, 1 user, temp 0, no draft | 120 tok/s | 99 tok/s |
| Decode, 1 user, temp 0.7, GPU sampling | 117 tok/s | 98.5 tok/s |
| TTFT, 1K prompt | 0.29 s | 0.80 s |
| TTFT, 8K prompt | 2.02 s | 3.05 s |
| 4 streams, temp 0.7 | 166.7 tok/s total | 159 tok/s total (np4) |
| 8 streams, temp 0.7 | 167 tok/s total | 180 tok/s total (np8) |
| JSON-mode (`response_format`) | 187.5 tok/s | 59.3 tok/s (grammar) |

MTP k=6, temp 0: JP prose 171, JP chat 153, code 268 tok/s (llama-server + MTP has not been measured under the same conditions yet). 17K-token prompt TTFT 3.71 s (4,630 tok/s prefill); with the RAM prefix tier, next turn of the same conversation 0.27 s. llama-server wins at 8 streams; MoE batching of same experts not done yet.

## Accuracy

- **27B**: top-1 agreement with llama-server 740/742 positions (the 2 differences are near-ties); 192/192 vs CPU f32 reference.
- **3.6**: vs CPU f32 reference top-1 98.2%, mean KL 0.0006 (llama-server: 97.6%, 0.0024).

Verified with per-kernel checks vs numpy (`tools/kcheck.py` etc.), per-layer vs a CPU f32 reference (`tools/ref_qwen35.py`), and golden compare vs `llama-server`.

## Build

Requirements:

- Linux x86_64
- CUDA 12.x (tested 12.5; CUDA 13 removed Volta support), with a driver supporting it
- Rust stable (tested 1.99)

`nvcc` is called from `build.rs` with `-arch=sm_70`; `CUDA_HOME` defaults to `/usr/local/cuda`. GPU is chosen with `CUDA_VISIBLE_DEVICES`.

```sh
cargo build --release
# -> target/release/tachyon
```

Commands:

```
tachyon generate | layers | bench [--lens 547,8192 --decode 64] | spec --k 3 --mtp-gguf
tachyon serve --gguf ... --hf <tokenizer dir> --keys keys.json --port 8080 [--mtp-gguf] [--mmproj] [--max-seqs 4] [--ctx 65536]
```

## Quick start

Weights are not part of this repository. Download them from Hugging Face (Apache-2.0):

```sh
# Qwen3.8-27B
huggingface-cli download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf MTP/mtp-Qwen3.8-27B-Q4_0.gguf mmproj-F16.gguf --local-dir weights/q38-27b
huggingface-cli download Qwen/Qwen3.8-27B tokenizer.json tokenizer_config.json chat_template.jinja --local-dir weights/q38-27b/hf
```

Create `keys.json` (only the sha256 of each key is stored; plain keys are for tests only):

```sh
KEY=$(openssl rand -hex 24)            # keep this value; send it as the Bearer token
printf '{"accounts": {"local": {"keys_sha256": ["%s"], "rpm": 600, "max_concurrent": 4}}}\n' \
  "$(printf %s "$KEY" | sha256sum | cut -d' ' -f1)" > keys.json
```

Serve:

```sh
CUDA_VISIBLE_DEVICES=0 target/release/tachyon serve \
  --gguf weights/q38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --hf weights/q38-27b/hf \
  --mtp-gguf weights/q38-27b/MTP/mtp-Qwen3.8-27B-Q4_0.gguf --draft-vocab data/draft_vocab_q38.txt \
  --mmproj weights/q38-27b/mmproj-F16.gguf \
  --keys keys.json --host 127.0.0.1 --port 8080 \
  --ctx 65536 --max-seqs 4 --max-running 4
```

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

For Qwen3.6-35B-A3B use `ggml-org/Qwen3.6-35B-A3B-GGUF` (`Qwen3.6-35B-A3B-Q4_K_M.gguf`, `mtp-Qwen3.6-35B-A3B-Q8_0.gguf`),
the tokenizer files from `Qwen/Qwen3.6-35B-A3B`, and an F16 mmproj (`unsloth/Qwen3.6-35B-A3B-GGUF` `mmproj-F16.gguf`, or convert the Q8_0 one with `tools/mmproj_to_f16.py`).
The MoE model uses the same `data/draft_vocab_q38.txt` (same tokenizer).

## Minimal reproducible benchmark

1. Download the GGUF with `huggingface-cli`.
2. `cargo build --release`
3. `CUDA_VISIBLE_DEVICES=0 target/release/tachyon bench --gguf weights/q38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf --lens 547,8192 --decode 64` (engine only: prefill tok/s per length, single-user decode tok/s)
4. Compare with mainline llama.cpp on the same GPU: `llama-bench -m <same gguf> -p 547,8192 -n 64 -ngl 99 -fa 1`.
5. End to end over HTTP (same prompts and TTFT definition for both servers): start `tachyon serve --no-auth --host 127.0.0.1` and `llama-server` on two ports, then `python3 tools/q36_bench.py <port> <label> 3 0.7` for each (400 tokens, 3 runs; see `docs/BENCHMARKS.md`).

## Repository layout

- `src/`: Rust (GGUF loader, model, KV manager, spec/MTP, tokenizer, chat template, vision, server)
- `kernels/`: CUDA
- `tools/`: reference implementations, kernel checks, conversion, benchmarks
- `tests/`: HTTP API tests with a fake backend
- `data/`: draft vocabulary for MTP lm_head
- `docs/`

## License

Apache-2.0, Copyright 2026 blueqat Inc. Third-party licenses are listed in `THIRD_PARTY_LICENSES.md`. Model weights are **not** included and follow their own licenses.

## Acknowledgements

- llama.cpp/ggml (quant formats and reference)
- Qwen team
- unsloth and ggml-org GGUF conversions
- Tencent DSpine (design reference)
- SenseTime / OpenSenseNova (SenseNova-Loop-9B and its reference code)
- Thinking Machines Lab (Inkling-2-Mini), the transformers and llama.cpp contributors who published its reference implementations, and ggml-org (GGUF conversion)
- YASAKANI (ideas for seeded sampling noise shared by draft and target; see NOTICE)

Developed by blueqat Inc.
