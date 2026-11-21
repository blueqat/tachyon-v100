# Inkling-2-Mini on Tachyon V100 Loop

Run [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (looped MoE transformer: 24 physical layers,
layers 6..17 executed three times = 48 executions, each pass restarting from the loop input with a depth-attention carry; no RoPE,
learned relative position bias; 128 experts top-8) on a single NVIDIA V100 32 GB with [Tachyon-v100](https://github.com/blueqat/tachyon-v100).
How the loop is implemented: [docs/LOOP.md](../LOOP.md). This page contains **no model weights**.

## Acceptable use (required reading)

The Inkling-2-Mini weights are released under Apache-2.0 **and** the
[Thinking Machines Lab Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/) (Model AUP).
By accessing, downloading or using the model materials you agree to be bound by the Model AUP. Tachyon-v100 does not change these terms;
if you build a product or service on Inkling-2-Mini with Tachyon, these obligations apply to you and to your users.
In particular (quoted from the Model AUP, last updated July 15, 2026; read the full policy):

- "If you permit others to use the Model Materials through your products or services, you are responsible for their compliance as well."
- "If you deploy the Model Materials in a product or service made available to end users, you agree to disclose to those users any known
  material limitations or risks of the AI system, and that they are interacting with an AI system wherever required by law or where that
  fact would be material to their understanding of the interaction."
- "Do not use the Model Materials to make automated (non-human) decisions that affect an individual's rights (including financial, legal,
  employment, healthcare, housing, insurance, or social services decisions)."
- "Do not use the Model Materials to engage in unauthorized or unlicensed professional practice or advising — including medical, legal,
  or financial advice — or to automate consequential decisions in these domains without appropriate human oversight."

The policy also prohibits, among others, illegal activity, harm to children, violence and weapons, discrimination, privacy violations and deception.

## Weights (download from upstream, unmodified)

| File | Repository | Size | Use |
|---|---|---|---|
| `Inkling-2-Mini-IQ4_NL.gguf` | [ggml-org/Inkling-2-Mini-GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) (GGUF conversion of the official weights by ggml-org, arch `inkling2_loop`) | 17.6 GB | main model |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) | small | tokenizer / chat template |

Only the `inkling2_loop` GGUF layout (24 blocks, as published by ggml-org) is supported. GGUFs with arch `inkling2` (48 expanded blocks) are a
different, incompatible layout. The expert `down` matrices are padded from 1152 to 1280 columns at load time (zeros; results are unchanged),
so the model takes 18.05 GB of VRAM for weights.

```sh
huggingface-cli download ggml-org/Inkling-2-Mini-GGUF Inkling-2-Mini-IQ4_NL.gguf --local-dir weights/inkling2mini
huggingface-cli download thinkingmachines/Inkling-2-Mini tokenizer.json tokenizer_config.json chat_template.jinja --local-dir weights/inkling2mini/hf
# inkcheck is built by `cargo build --release` (src/bin/inkcheck.rs): greedy or teacher-forced runs with top-20 logprobs (m1-golden/1 JSON),
# --ref <reference.json> for teacher forcing, --bench N to time N decode tokens. Prompts are token-id lists in the same format (tools/m1_golden.py).
CUDA_VISIBLE_DEVICES=0 target/release/inkcheck weights/inkling2mini/Inkling-2-Mini-IQ4_NL.gguf prompts.json --out ours.json --n 32 --bench 128
```

### Server

`tachyon serve` detects the `inkling2_loop` GGUF and uses the Inkling engine (OpenAI-compatible `/v1/chat/completions`, text only):

```sh
CUDA_VISIBLE_DEVICES=0 target/release/tachyon serve --gguf weights/inkling2mini/Inkling-2-Mini-IQ4_NL.gguf --hf weights/inkling2mini/hf \
  --model-name inkling-2-mini --kv8 --keys keys.json --host 127.0.0.1 --port 8080 --ctx 32768 --max-seqs 8 --max-running 8
```

- Chat template: Inkling's own format. `reasoning_effort` accepts `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` or a number in [0, 0.99].
  As for the other models, the server does not reason unless asked (`--thinking-default off` is the default): send `reasoning_effort`
  or `chat_template_kwargs: {"enable_thinking": true}`, or start the server with `--thinking-default on` (effort `medium`).
  When reasoning is off, Tachyon's server opens the answer section in the prompt itself, so the model skips reasoning (the model alone would not; see Limitations).
  Reasoning, answer and tool calls are returned separately (`reasoning_content`, `content`, `tool_calls`).
- `--kv8`: int8 KV cache for the global-layer executions; top-1 tokens identical to fp16 KV at 310/310 positions (mean |logprob diff| 0.013).
  The verification and engine benchmark tables above use fp16 KV; the HTTP numbers below use int8 KV.
- Prefix cache per account (KV of the global layers, the sliding-window ring and the convolution state are saved at the last message boundary).
- Over HTTP (same GPU, int8 KV, streaming): 1 user 57 tok/s, TTFT 0.99 s for a 1.3K-token prompt and 8.6 s for 9.9K tokens; 4 / 8 concurrent 128 / 160 tok/s total.

### Long context

Synthetic retrieval tests (needle, multi-key, multi-value, variable tracking, title), English and Japanese, document in the system message,
served by Tachyon with int8 KV: all five tasks 1.0 at 8K and 32K; at 64K and 128K every task stays at 0.75-1.0 (English variable tracking 0.4 at 64K).
Although the global layers have no relative-position term beyond 1024 tokens, retrieval did not break down up to 128K.
For questions over long documents, keep reasoning on: with reasoning off the model often answered that the document did not contain information
that it did contain (15 of 28 questions in our internal test); with `reasoning_effort: low` or `medium` this did not happen.
The server therefore adds a reasoning floor: a no-reasoning request whose prompt exceeds `--ink-nothink-floor-tokens` (default 4096) or that
carries documents is run with `--ink-nothink-floor-effort` (default `low`), a reasoning budget of `--ink-nothink-floor-budget` tokens (default 2048)
and `max_tokens` raised by the same amount. Set `--ink-nothink-floor-tokens 0` to turn it off.
Note that, at temperature 0, outputs with and without prefix-cache reuse can diverge after a few dozen tokens (different numerical path for the
remaining short prefill); we saw no quality difference.

## Verification

Reference: the Inkling implementation of the transformers pull request #51050 run on CPU in float32 (`tools/ink_ref.py`), with the official
weights (experts are NVFP4 / MXFP8 in the official files; the reference dequantizes them to BF16 and computes in f32, and does not quantize
activations). 5 prompts (English continuation, Japanese, code, arithmetic, a 147-token English passage), temperature 0, up to 32 tokens each
(two prompts stop early: 107 positions), compared with `tools/m1_compare.py`. "Near-tie" = the two top tokens of the comparison side differ by less than 0.1 in logprob.

| Comparison | Top-1 agreement | Not a near-tie | Mean / max abs. difference of the top-1 logprob |
|---|---|---|---|
| **Tachyon (IQ4_NL, KV fp16) vs reference**, teacher forcing on all 107 reference tokens | **104/107 (97.2%)** | 1 (the same position and token as llama.cpp) | 0.022 / 0.526 |
| llama.cpp (same IQ4_NL) vs reference, free running until first difference | 78/80 (97.5%) | 1 | 0.022 / 0.255 |
| Tachyon vs llama.cpp, same weights, free running until first difference | 88/89 (98.9%) | 0 | 0.0064 / 0.172 |
| Tachyon with vs without CUDA Graph | identical output files (tokens and top-20 logprobs; same md5) | | |
| Tachyon 4 / 8 / 16 / 32 concurrent vs one at a time (temperature 0) | 139/139, 340/340, 673/673, 1291/1291 tokens identical | | |

The remaining difference against the reference is mostly from re-quantizing the NVFP4 experts to IQ4_NL (llama.cpp diverges at the same place);
we have not separated the causes.

## Benchmarks

One Tesla V100-SXM3-32GB (driver 570.86, CUDA 12.5), IQ4_NL, temperature 0, `inkcheck`, internal measurements, **preliminary**.
llama.cpp = upstream master 29b52d398 on the same GPU and weights (flash attention cannot be enabled for this model in llama.cpp because of the position bias).

| | Tachyon | llama.cpp |
|---|---|---|
| Decode, 1 user (no draft; CUDA Graph) | **62-64 tok/s** | 36.9 tok/s |
| TTFT, 1K-token prompt | 0.69 s | 1.51 s |
| TTFT, 8K-token prompt | **6.72 s** (1,219 tok/s) | 13.1 s (625 tok/s) |
| TTFT, 32K-token prompt | 37.6 s; decode afterwards 49.7 tok/s | — |
| 4 / 8 concurrent streams (total) | 154 / 196 tok/s | 95 / 134 tok/s |
| 16 / 32 concurrent streams (total) | 238 / 270 tok/s | — |

Concurrent runs use short prompts (5 prompts cycled, 64 tokens each). VRAM: weights 18.05 GB + per-sequence state
(fp16 KV of the 8 global-layer executions = 32 KB/token, plus fixed window and convolution state: 0.39 GB at 4K, 4.55 GB at 128K) + workspace;
one 128K-token sequence fits in 23.2 GB total (fp16 KV, one sequence; little room is left for more).

## Limitations

- **Not ported yet: the vision tower and the MTP / block-draft head** (a 3-layer DSpark-style block drafter, not a standard MTP head). Text only, no speculative decoding.
- Experts are read from the IQ4_NL GGUF (NVFP4 re-quantized to 4-bit), not from the official NVFP4 weights.
- Prefill attention uses f32 CUDA cores (not tensor cores yet); with many concurrent streams, decoding does not yet group identical experts.
- Without the server's skip (for example with other servers or a raw template), `reasoning_effort: none` alone does not stop the model from writing
  a reasoning section, so a small `max_tokens` can return an empty answer.
- Not an official Thinking Machines Lab release.

## License

- Tachyon-v100: Apache-2.0, Copyright 2026 blueqat Inc.
- Inkling-2-Mini weights: Apache-2.0 plus the Model AUP above, released by Thinking Machines Lab, Inc. The GGUF conversion is by ggml-org. Not redistributed here.
