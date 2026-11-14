#!/bin/bash
# 掛け算のカーネルの速さを測る（V100 で。GPU は CUDA_VISIBLE_DEVICES で選ぶ）
cd "$(dirname "$0")/.."
X=$(mktemp); trap 'rm -f $X ${X}.y' EXIT
.venv/bin/python -c "import numpy as np; np.random.default_rng(0).standard_normal(8*10240).astype(np.float32).tofile('$X')"
W=weights/iq3xxs
run() { ./target/release/kcheck gemv "$W/$1.bin" "$2" "$3" "$4" "$X" "$5" "$X.y" 2>&1 | tail -1; }
for B in 1 4 8; do run output.weight Q6_K 2560 248320 $B; done
for B in 1 8; do run blk.0.ffn_down_exps.weight IQ4_NL 640 1310720 $B; done
for B in 1 8; do run blk.0.ffn_gate_exps.weight IQ2_S 2560 327680 $B; done
run blk.0.ssm_out.weight Q6_K 6144 2560 1
run blk.2.attn_qkv.weight Q8_0 2560 10240 1
