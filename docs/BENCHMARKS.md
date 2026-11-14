# Benchmark method

All numbers in the README were measured on one V100-SXM3-32GB (900 GB/s HBM2), with the **same GGUF file** for Tachyon-v100 and llama.cpp.
They are preliminary: the GPU was shared with other jobs during some runs. Re-run on a dedicated GPU before quoting them.

## Engine only (no HTTP)

```sh
CUDA_VISIBLE_DEVICES=0 target/release/tachyon bench --gguf <model.gguf> --lens 547,8192 --decode 64 --repeat 3
llama-bench -m <model.gguf> -ngl 99 -fa 1 -p 547,8192 -n 64 -r 3
```

## End to end (HTTP, OpenAI API)

1. Start both servers on the same GPU, one at a time, with comparable settings:
   * `tachyon serve --no-auth --host 127.0.0.1 --port 8081 --gguf <model.gguf> --hf <tokenizer dir> [--mtp-gguf ...] --max-seqs 4`
   * `llama-server -m <model.gguf> --host 127.0.0.1 --port 8082 -ngl 99 -c 131072 -np 4 -ub 2048 -b 2048 -ctk q8_0 -ctv q8_0 -fa on`
2. `python3 tools/q36_bench.py 8081 tachyon 3 0.7` and `python3 tools/q36_bench.py 8082 llama 3 0.7`
   * `ja_prose`, `ja_chat`, `code`: one user, 400 tokens, thinking off. Speed = (tokens − 1) / (t_last − t_first), client clock, same formula for both.
   * `ttft1k`, `ttft8k`: ~1K / ~8K token prompts with a unique prefix per run (no prefix-cache hits); best of the runs.
3. Report: GPU model, driver, CUDA, llama.cpp commit, Tachyon commit, GGUF sha256, temperature, MTP on/off and k.

## Correctness

* `tools/m1_golden.py` makes "golden" files (prompts, greedy output, top-20 logprobs per position) with llama.cpp or the CPU f32 reference;
  `tools/m1_compare.py` and `tools/kl_top.py` compare Tachyon against them (top-1 agreement, KL on the top-20).
* With MTP, temperature-0 output must be token-identical to no-draft decoding (checked by `tachyon spec` and over HTTP with `--mtp-k 0`).
