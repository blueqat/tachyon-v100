# Looped transformers in Tachyon V100 Loop

A looped (recurrent-depth) transformer runs some of its decoder layers more than once per token, with the **same weights**.
The model gets more computation depth without more parameters: a 32-layer model can do 56 layer executions,
a 24-layer model 48. Both models supported here were *trained* this way; Tachyon does not add loops to a model that was not.

This page explains what that means for an inference engine and how Tachyon V100 Loop implements it on one NVIDIA V100.

## 1. Physical layers and executions

The engine separates two numbers:

- **physical layers**: the weight tensors in the GGUF file (each loaded to GPU memory once)
- **executions**: the order in which layers run for one token

```
SenseNova-Loop-9B (32 physical layers, 56 executions)

  exec:   0 1 2 3 | 4 ........ 27 | 4 ........ 27 | 28 29 30 31
          prelude | loop pass 1   | loop pass 2   | coda
                    \_______ same weights _______/

Inkling-2-Mini (24 physical layers, 48 executions)

  exec:   0 .. 5 | 6 ...... 17 | 6 ...... 17 | 6 ...... 17 | 18 .. 23
          prelude| pass 1      | pass 2      | pass 3      | coda
```

In Tachyon each execution is an entry in `Model.layers: Vec<Rc<LayerW>>`. Repeated executions point to the same `LayerW`
(reference-counted), so VRAM for weights does not grow with the number of passes.

**State is per execution, not per physical layer.** Every execution has its own attention KV cache and its own
recurrent state (Gated DeltaNet state or short-convolution state). Pass 2 of layer 10 does not see the keys of pass 1
of layer 10. This is what the official implementations do, and it falls out naturally if the engine indexes all
per-layer state by execution number: prefill, decode, CUDA Graph capture, MTP verification, prefix-cache reuse and
continuous batching are reused as they are. The positions (RoPE or position bias) are the same in every pass.

Cost per token is roughly proportional to the number of executions: Loop-9B reads its 24 looped layers twice,
about 1.75x the weight traffic of a plain 32-layer 9B model.

## 2. SenseNova-Loop-9B: two-stream mHC

The base block is the same hybrid as Qwen3.5/3.6/3.8 (Gated DeltaNet linear attention : full attention = 3 : 1,
hidden 4096, FFN 12288, 16 query heads / 4 KV heads, head dim 256, multi-section RoPE with YaRN factor 2 over 262,144).

Inside the loop (physical layers 4..27) the residual is split into **two streams** `a` and `b`
(manifold-constrained hyper-connections, mHC). On entry to layer 4 in each pass `a = b = h`; on exit from layer 27 `h = (a + b) / 2`.
Before both the attention and the FFN sub-block of each looped layer, a small router computes, per token:

```
r      = rsqrt(mean([a; b]^2) + 1e-6)
l      = W · ([a; b] · r) + bias                  W: f32 [8][2·4096]
x      = sigmoid(l0)·a + sigmoid(l1)·b            branch input
post   = 2·sigmoid(l2, l3)
M      = exp(Sinkhorn(l4..l7 as 2x2, 20 iterations in log space, row then column))
y      = branch(norm(x))                          attention or FFN
a'     = M00·a + M01·b + post0·y
b'     = M10·a + M11·b + post1·y
```

The Sinkhorn mixing matrix depends on the residual, so it is computed per token at run time (`kernels/mhc.cu`, f32),
in all three paths (prefill, decode, MTP verification). In the official GGUF the coda layers are
the tensor blocks `blk.52..55` (= physical layers 28..31, named by execution index) and the MTP head is `blk.56`; Tachyon reads them as they are.

## 3. Inkling-2-Mini: restart passes with depth attention

Inkling-2-Mini (24 layers, hidden 2304, MoE 128 experts top-8, no shared expert) loops layers 6..17 three times,
but each pass **restarts from the loop input** (the residual at the entry of layer 6) instead of continuing from the
previous pass's output; cross-pass information enters only through a *carry* (pass 1 has none):

- per layer, four learned queries (for q, k, v and the FFN input) attend over the depth axis to the previous passes'
  block states and outputs (depth attention);
- the result `m` is mixed into the sub-block input through a low-rank (64) gate:
  `(1 - tanh f)·h + tanh g·m`, with `f, g = forget/gain(down([rms h; rms m]))`.

Other parts of the architecture that the engine implements with its own kernels (`kernels/inkling.cu`, `src/inkling.rs`):

- **No RoPE.** Position enters as a learned bias: a per-token, per-head 16-dimensional vector (L2 norm clipped to 8)
  times a per-layer table `[16][1024]`, zero beyond distance 1024, so global layers have no relative-position term beyond distance 1024
  (only a position-dependent scale remains: their q and the bias are multiplied by `tau = 1 + 0.1·log(max(1, (p+1)/8192))`).
- 20 sliding-window layers (window 1024) and 4 global layers; per-token attention sink, sigmoid output gate,
  q/k L2 normalization, weightless RMSNorm on v.
- Short causal convolutions (4 taps, residual) on k, v and the attention and FFN outputs.
- Block attention residuals: every 6 physical layers a new residual block starts, and each sub-block input is a learned
  softmax mix of the accumulated block states.
- Sigmoid routing with a selection bias, a bounded SwiGLU `4·tanh(g/4)·sigmoid(g)·25·tanh(u/25)`, and
  `30·tanh(x/30)` soft-capped logits.

Because attention has an additive position bias, attention kernels that only support a causal mask cannot be used;
Tachyon's attention is its own code and handles the bias directly.

## 4. Training-free repetition does not help

`--repeat-layers START:END:N` repeats physical layers of an ordinary model N times (no mHC, separate state per execution).
We tried it on Qwen3.8-27B (UD-Q4_K_XL, one V100) over 16 settings (2-4 passes, 4-32 extra layers): next-token NLL on 48 texts
(48,854 tokens) was higher (worse) in every setting (+0.015 to +0.28 nats/token); on 214 reasoning problems (AIME 2025, MATH-500 level 4+,
and our own quantum-computing questions; 2 runs each, temperature 0.6) accuracy dropped from 90.2% to 84-88% for the three best settings;
single-user decode slowed roughly in proportion to the added layers (-6% for 4 layers, -11% for 8, -32% for 32). Looping helps only in models trained for it. The option is kept for research;
when it is off, outputs are bit-identical to an engine without it.

## 5. Where to look in the code

| What | Where |
|---|---|
| Loop config (`loop.*`, `mhc.*` GGUF keys), execution list, shared weights | `src/model.rs` (`LoopCfg`, `Config.lp`) |
| mHC router and merge | `kernels/mhc.cu` |
| Inkling-2-Mini model | `src/inkling.rs`, `kernels/inkling.cu`, `src/bin/inkcheck.rs` |
| CPU float32 references with the official code | `tools/loop_ref.py`, `tools/loop_hf_from_gguf.py`, `tools/ink_ref.py` |
| Comparison against a reference | `tools/m1_compare.py` |
