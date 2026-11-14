# Building Tachyon-v100

## Requirements

| | Version | Notes |
|---|---|---|
| GPU | NVIDIA V100 (sm_70), 32 GB | 16 GB V100 is not enough for the 27B / 35B-A3B Q4 weights plus KV. Other GPUs are not targeted (the kernels are compiled for `sm_70` only). |
| OS | Linux x86_64 | tested on Ubuntu 22.04 |
| CUDA toolkit | 12.x (tested 12.5) | CUDA 13 removed offline compilation for Volta (sm_70). `nvcc` must be at `$CUDA_HOME/bin/nvcc` (default `/usr/local/cuda`). |
| Driver | one that supports your CUDA 12.x | |
| Rust | stable (tested 1.99) | `rustup` is fine |
| Host RAM | 64 GB+ recommended | the RAM tier for the KV / prefix cache is sized with `--host-cache-gib` |

## Build

```sh
export CUDA_HOME=/usr/local/cuda        # optional
cargo build --release                   # build.rs compiles kernels/*.cu with nvcc -O3 -arch=sm_70
ls target/release/tachyon target/release/tachyon-serve
```

Extra `nvcc` flags for experiments: `INFER_NVCC_FLAGS="-DFOO=1" cargo build --release`.

## Tests

```sh
cargo test --release --lib                         # CPU unit tests (KV manager, tokenizer, template, JSON constraint, sampler)
cargo test --release --test server_api             # HTTP API against the fake backend (no GPU)
TACHYON_HF_DIR=weights/q38-27b/hf cargo test --release --test server_api   # with the real tokenizer
```

GPU correctness checks (need a V100 and the GGUF):

```sh
python3 -m venv .venv && .venv/bin/pip install numpy gguf tokenizers jinja2 requests
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/kcheck.py --gguf weights/q38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf   # dequant kernels vs gguf-py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/gemm_check.py                                                   # GEMM / GEMV
.venv/bin/python tools/ref_qwen35.py --help                                                                    # CPU f32 reference, per layer
```
