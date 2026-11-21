# SenseNova-Loop-9B on Tachyon V100 Loop

Run [sensenova/SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B) (looped transformer: 32 physical layers,
layers 4..27 executed twice = 56 executions, two-stream mHC inside the loop, Gated DeltaNet + full attention, MTP head, vision tower)
on a single NVIDIA V100 32 GB with [Tachyon-v100](https://github.com/blueqat/tachyon-v100). How the loop is implemented: [docs/LOOP.md](../LOOP.md).
This page contains **no model weights**; download them from the official repositories.

## Weights (official, unmodified)

| File | Repository | Size | Use |
|---|---|---|---|
| `SenseNova-Loop-9B-Q4_K_M.gguf` (or `-Q8_0.gguf`) | [sensenova/SenseNova-Loop-9B-GGUF](https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF) | 5.8 GB (9.8 GB) | main model + MTP head (`--gguf`) |
| `mmproj-SenseNova-Loop-9B-F16.gguf` | same | 0.92 GB | vision tower (`--mmproj`) |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | [sensenova/SenseNova-Loop-9B-AWQ](https://huggingface.co/sensenova/SenseNova-Loop-9B-AWQ) (or the main repo, which asks you to accept its terms) | small | `--hf <dir>` |

The tokenizer has the same vocabulary and merges as Qwen3.8 but a different pre-tokenizer regex; use the Loop files, not the Qwen ones.
The MTP head is inside the main GGUF (tensor block `blk.56`), so no `--mtp-gguf` is needed. The vocabulary (248,320 tokens) is the same as Qwen3.8,
so the reduced MTP output vocabulary `data/draft_vocab_q38.txt` from this repository applies (optional; without it the MTP head uses the full vocabulary).

```sh
huggingface-cli download sensenova/SenseNova-Loop-9B-GGUF SenseNova-Loop-9B-Q4_K_M.gguf mmproj-SenseNova-Loop-9B-F16.gguf --local-dir weights/loop9b
huggingface-cli download sensenova/SenseNova-Loop-9B-AWQ tokenizer.json tokenizer_config.json chat_template.jinja --local-dir weights/loop9b/hf
CUDA_VISIBLE_DEVICES=0 target/release/tachyon serve --gguf weights/loop9b/SenseNova-Loop-9B-Q4_K_M.gguf --hf weights/loop9b/hf \
  --mmproj weights/loop9b/mmproj-SenseNova-Loop-9B-F16.gguf --draft-vocab data/draft_vocab_q38.txt \
  --model-name sensenova-loop-9b --keys keys.json --host 127.0.0.1 --port 8080 --ctx 65536 --max-seqs 4 --max-running 4
```

Engine-only numbers: `CUDA_VISIBLE_DEVICES=0 target/release/tachyon bench --gguf weights/loop9b/SenseNova-Loop-9B-Q4_K_M.gguf --lens 512,2048,8192 --decode 64`.

## Verification

Reference: the official PyTorch code (`modeling_sensenova_loop.py`, github.com/OpenSenseNova/SenseNova-Loop at e19d541) run on CPU in float32
(`tools/loop_ref.py`), with weights restored from the official BF16 GGUF (`tools/loop_hf_from_gguf.py`; the 523 tensors common with the
official safetensors have identical values). 5 prompts (English continuation, Japanese, code, arithmetic, a 147-token English passage),
greedy, up to 32 tokens each (two prompts stop early at end of sequence: 32+9+32+2+32 = 107 positions), compared position by position with `tools/m1_compare.py` (teacher forcing on the reference tokens).

| Weights / path | Top-1 agreement | Mean / max abs. difference of the top-1 logprob |
|---|---|---|
| Q8_0, decode path | **107/107** | 0.004 / 0.062 |
| Q8_0, prefill path | **107/107** | 0.004 |
| Q4_K_M (both paths) | 101/107 (94.4%) | 0.042 / 0.73 (quantization) |

- Per layer (Q8_0, 5 tokens): relative difference of the residual after each of the 56 executions 0.4-1.1%, not growing with depth; KL at the last position 0.00036.
- MTP (k=3) at temperature 0 is token-identical to decoding without a draft (5 prompts x 128 tokens, with and without CUDA Graph).
- Server: JSON mode, image input (data URI), prefix-cache reuse, 1 vs 4 concurrent streams give identical temperature-0 outputs.

## Benchmarks

One Tesla V100-SXM3-32GB (driver 570.86, CUDA 12.5), Q4_K_M, temperature 0, one user unless noted, engine measurements (`tachyon bench` /
`tachyon spec`), internal measurements. **Preliminary**: re-measure on your hardware before quoting.

| | Q4_K_M | Q8_0 |
|---|---|---|
| Decode, 1 user, no draft | **55 tok/s** | 45 tok/s |
| Decode with MTP k=3, 1 user (tokens / time summed over 5 prompts, acceptance 49%, 2.45 tokens/step) | **98 tok/s** | not measured |
| Prefill 512 / 2K / 8K tokens | 1,901 / 2,039 / 1,985 tok/s | same as Q4_K_M within 1% |
| TTFT, 8K-token prompt | 4.1 s | 4.1 s |
| 4 / 8 / 16 concurrent short streams (total) | 197 / 296 / 391 tok/s | not measured |

KV cache (14 attention executions x 4 heads x 256): 56 KB/token in fp16; Gated DeltaNet state 42 executions: 88 MB per sequence.

## Limitations

- Image size limits differ from the official preprocessor (min 65,536 / max 16,777,216 pixels there; 8-4096 image tokens here). For ordinary image sizes the two agree; very small or very large images are resized differently.
- `--kv8` (int8 KV) runs but has not been compared numerically with fp16 KV for this model. The RAM prefix tier has not been tested with this model.
- Set `--model-name`; the engine default name is that of the first supported model.
- In our own (non-public) Japanese-language evaluation the model mixed simplified Chinese characters into Japanese text; check this for your use.
- Quality figures in the official model card were not reproduced by us.

## License

- Tachyon-v100: Apache-2.0, Copyright 2026 blueqat Inc.
- SenseNova-Loop-9B weights and code: Apache-2.0, released by SenseTime / OpenSenseNova (see the [LICENSE](https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF/blob/main/LICENSE) in the official repository). They are not redistributed here.
- This is not an official SenseNova release.
