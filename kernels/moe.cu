// MoE（Qwen3.6-35B-A3B・GGUF arch qwen35moe）の小さなカーネル: ルーター・上位 k・SiLU·up・足し合わせ。
// 専門家の行列の掛け算は g3 の MoE（gemv3.cu の infer_g3_moe_gemv）。式は llama.cpp の build_moe_ffn（softmax・上位 8・重みの正規化）と
// qwen35moe の共有の専門家（sigmoid のゲート）と同じ:
//   logits = W_r x（f32）、p = softmax(logits)、上位 k の p を Σ で割った重み w_j、
//   出口 = Σ_j w_j · down_e(SiLU(gate_e x) · up_e x) + sigmoid(w_sg · x) · shared(x)
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {

// 小さい T（デコード・確かめ）: 1 warp = 専門家の行 1 本、T 本の x と内積（f32・決まった順）。k は 128 の倍数
__global__ void router_warp_kernel(const float* __restrict__ w, int k, int ne, const float* __restrict__ x, int64_t xld, int T,
                                   float* __restrict__ y) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = blockIdx.x * 8 + warp;
    if (e >= ne) return;
    const float4* wr = reinterpret_cast<const float4*>(w + int64_t(e) * k);
    for (int t = 0; t < T; ++t) {
        const float4* xr = reinterpret_cast<const float4*>(x + int64_t(t) * xld);
        float acc = 0.f;
        for (int i = lane; i < k / 4; i += 32) {
            const float4 a = __ldg(wr + i), b = __ldg(xr + i);
            acc = fmaf(a.x, b.x, acc);
            acc = fmaf(a.y, b.y, acc);
            acc = fmaf(a.z, b.z, acc);
            acc = fmaf(a.w, b.w, acc);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) y[int64_t(t) * ne + e] = acc;
    }
}

// 大きい T（プレフィル）: 64 語 × 64 行のタイル、k は 32 ずつ共有メモリへ。1 スレッド 4 語 × 4 行
__global__ void __launch_bounds__(256) router_tile_kernel(const float* __restrict__ w, int k, int ne, const float* __restrict__ x, int64_t xld,
                                                         int T, float* __restrict__ y) {
    __shared__ float xs[32][65];
    __shared__ float ws[32][65];
    const int t0 = blockIdx.x * 64, e0 = blockIdx.y * 64, tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < k; k0 += 32) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int idx = tid + 256 * r, kk = idx & 31, i = idx >> 5;
            const int t = t0 + i, e = e0 + i;
            xs[kk][i] = t < T ? __ldg(x + int64_t(t) * xld + k0 + kk) : 0.f;
            ws[kk][i] = e < ne ? __ldg(w + int64_t(e) * k + k0 + kk) : 0.f;
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < 32; ++kk) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = xs[kk][ty * 4 + i], b[i] = ws[kk][tx * 4 + i];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int t = t0 + ty * 4 + i;
        if (t >= T) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int e = e0 + tx * 4 + j;
            if (e < ne) y[int64_t(t) * ne + e] = acc[i][j];
        }
    }
}

// (値, 番号) の大きい方（同じ値なら番号の小さい方）
__device__ __forceinline__ void amax(float& v, int& i, float v2, int i2) {
    if (v2 > v || (v2 == v && i2 < i)) v = v2, i = i2;
}

// 1 ブロック（256）= 1 語: softmax → 上位 kk → 重みを正規化。ne ≤ 1024。
// 出力: ids・wts [T][kk]、デコードの形の仕事（je = ids、jr0 = t·kk + j、jcnt = 1、xidx = t）
__global__ void __launch_bounds__(256) topk_kernel(const float* __restrict__ lg, int64_t ld, int ne, int kk, int* __restrict__ ids,
                                                   float* __restrict__ wts, int* __restrict__ jr0, int* __restrict__ jcnt, int* __restrict__ xidx) {
    constexpr int PER = 4;
    __shared__ float rv[8];
    __shared__ int ri[8];
    __shared__ float sel_p[32];
    __shared__ int sel_i[32];
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float* l = lg + int64_t(t) * ld;
    float v[PER];
#pragma unroll
    for (int q = 0; q < PER; ++q) {
        const int e = tid + 256 * q;
        v[q] = e < ne ? l[e] : -INFINITY;
    }
    // 最大
    float m = -INFINITY;
#pragma unroll
    for (int q = 0; q < PER; ++q) m = fmaxf(m, v[q]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) rv[warp] = m;
    __syncthreads();
    m = rv[0];
    for (int i = 1; i < 8; ++i) m = fmaxf(m, rv[i]);
    __syncthreads();
    // Σ exp
    float p[PER], s = 0.f;
#pragma unroll
    for (int q = 0; q < PER; ++q) {
        p[q] = tid + 256 * q < ne ? expf(v[q] - m) : 0.f;
        s += p[q];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) rv[warp] = s;
    __syncthreads();
    float S = 0.f;
    for (int i = 0; i < 8; ++i) S += rv[i];
    __syncthreads();
    // 上位 kk（大きい順。同じ値は番号の小さい方を先に）
    for (int j = 0; j < kk; ++j) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
#pragma unroll
        for (int q = 0; q < PER; ++q) amax(bv, bi, v[q], tid + 256 * q);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float v2 = __shfl_xor_sync(0xffffffffu, bv, o);
            const int i2 = __shfl_xor_sync(0xffffffffu, bi, o);
            amax(bv, bi, v2, i2);
        }
        if (lane == 0) rv[warp] = bv, ri[warp] = bi;
        __syncthreads();
        if (tid == 0) {
            float b = rv[0];
            int bi2 = ri[0];
            for (int i = 1; i < 8; ++i) amax(b, bi2, rv[i], ri[i]);
            sel_i[j] = bi2;
            rv[0] = b;
            ri[0] = bi2;
        }
        __syncthreads();
        const int win = ri[0];
#pragma unroll
        for (int q = 0; q < PER; ++q)
            if (tid + 256 * q == win) {
                sel_p[j] = p[q] / S;
                v[q] = -INFINITY;
            }
        __syncthreads();
    }
    if (tid == 0) {
        float ws = 0.f;
        for (int j = 0; j < kk; ++j) ws += sel_p[j];
        ws = fmaxf(ws, 6.103515625e-5f);
        for (int j = 0; j < kk; ++j) {
            const int r = t * kk + j;
            ids[r] = sel_i[j];
            wts[r] = sel_p[j] / ws;
            if (jr0) jr0[r] = r;
            if (jcnt) jcnt[r] = 1;
            if (xidx) xidx[r] = t;
        }
    }
}

// h[r][i] = SiLU(g[r][i]) · u[r][i]（行は R 本、幅 n）
__global__ void swiglu_kernel(const float* __restrict__ g, const float* __restrict__ u, int64_t total, float* __restrict__ h) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float a = g[i];
    h[i] = a / (1.f + expf(-a)) * u[i];
}

// h[t][c] += Σ_j wts[t·kk + j] · o[行(t, j)][c] + sigmoid(sg[t]) · sh[t][c]。行(t, j) = rowmap[t·kk + j]（null なら t·kk + j）。sh が null なら共有は無し
__global__ void combine_kernel(float* __restrict__ h, int d, int T, int kk, const float* __restrict__ wts, const float* __restrict__ o,
                               const int* __restrict__ rowmap, const float* __restrict__ sh, const float* __restrict__ sg, int64_t sg_ld) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(T) * d) return;
    const int t = int(i / d), c = int(i % d);
    float s = 0.f;
    for (int j = 0; j < kk; ++j) {
        const int a = t * kk + j;
        const int r = rowmap ? rowmap[a] : a;
        s += wts[a] * o[int64_t(r) * d + c];
    }
    if (sh) {
        const float z = sg[int64_t(t) * sg_ld];
        s += sh[i] * (1.f / (1.f + expf(-z)));
    }
    h[i] += s;
}

// プレフィルの専門家ごとの並べ（GPU の上で・手元に戻らない。2026-11-14）: 1 ブロック。
// ids [r]（行 a = 語·kk + 枠 の専門家）→ 専門家の順の行: xidx[行] = 語、rowmap[a] = 行、
// grp [n_blk_max][3] = (専門家, 行の頭, 行の数 ≤ tokb)。余った仕事は行の数 0（GEMM が何もしない）
__global__ void __launch_bounds__(1024) group_kernel(const int* __restrict__ ids, int r, int kk, int ne, int tokb, int* __restrict__ xidx,
                                                     int* __restrict__ rowmap, int* __restrict__ grp, int n_blk_max) {
    extern __shared__ int sm[];
    int* cnt = sm;            // [ne]
    int* start = sm + ne;     // [ne]
    int* boff = sm + 2 * ne;  // [ne + 1]
    int* fill = sm + 3 * ne + 1;
    for (int e = threadIdx.x; e < ne; e += blockDim.x) cnt[e] = 0, fill[e] = 0;
    __syncthreads();
    for (int a = threadIdx.x; a < r; a += blockDim.x) atomicAdd(&cnt[ids[a]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int s = 0, b = 0;
        for (int e = 0; e < ne; ++e) {
            start[e] = s, boff[e] = b;
            s += cnt[e];
            b += (cnt[e] + tokb - 1) / tokb;
        }
        boff[ne] = b;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < ne; e += blockDim.x) {
        const int nb = boff[e + 1] - boff[e];
        for (int b = 0; b < nb && boff[e] + b < n_blk_max; ++b) {
            int* g = grp + 3 * (boff[e] + b);
            g[0] = e, g[1] = start[e] + b * tokb, g[2] = min(tokb, cnt[e] - b * tokb);
        }
    }
    for (int j = boff[ne] + threadIdx.x; j < n_blk_max; j += blockDim.x) grp[3 * j] = 0, grp[3 * j + 1] = 0, grp[3 * j + 2] = 0;
    for (int a = threadIdx.x; a < r; a += blockDim.x) {
        const int e = ids[a];
        const int p = start[e] + atomicAdd(&fill[e], 1);
        xidx[p] = a / kk;
        rowmap[a] = p;
    }
}

__global__ void lo16_kernel(const float* __restrict__ x, const __half* __restrict__ hi, int64_t n, __half* __restrict__ lo) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) lo[i] = __float2half_rn(x[i] - __half2float(hi[i]));
}

}  // namespace

extern "C" {

// lo = fp16(x − hi)（x の上 hi は fp16 に丸めた物。X = hi + lo で GEMM の入力の精度を上げる）
int infer_moe_lo16(const float* x, const void* hi, int64_t n, void* lo, cudaStream_t s) {
    if (n <= 0) return 0;
    lo16_kernel<<<unsigned((n + 255) / 256), 256, 0, s>>>(x, static_cast<const __half*>(hi), n, static_cast<__half*>(lo));
    return int(cudaGetLastError());
}

// プレフィルの専門家ごとの並べ（上の group_kernel）。n_blk_max ≥ ne + r / tokb なら必ず収まる
int infer_moe_group(const int* ids, int r, int kk, int ne, int tokb, int* xidx, int* rowmap, int* grp, int n_blk_max, cudaStream_t s) {
    if (r < 1 || ne < 1 || ne > 2048 || tokb < 1 || n_blk_max < ne + r / tokb) return -2;
    group_kernel<<<1, 1024, (4 * ne + 1) * sizeof(int), s>>>(ids, r, kk, ne, tokb, xidx, rowmap, grp, n_blk_max);
    return int(cudaGetLastError());
}

// y[t][e] = W[e] · x[t]（f32。W は [ne][k]、x の行の間隔 xld、y は [T][ne]）。k は 128 の倍数
int infer_moe_router(const float* w, int k, int ne, const float* x, int64_t xld, int T, float* y, cudaStream_t s) {
    if (k % 128 || ne < 1 || T < 1 || xld % 4) return -2;
    if (T <= 32) router_warp_kernel<<<(ne + 7) / 8, 256, 0, s>>>(w, k, ne, x, xld, T, y);
    else router_tile_kernel<<<dim3((T + 63) / 64, (ne + 63) / 64), 256, 0, s>>>(w, k, ne, x, xld, T, y);
    return int(cudaGetLastError());
}

// 語ごとに上位 kk（≤ 32）の専門家と重み。lg [T][ld] の先頭 ne（≤ 1024）個。jr0・jcnt・xidx は null でもよい（デコードの形の仕事）
int infer_moe_topk(const float* lg, int64_t ld, int ne, int kk, int T, int* ids, float* wts, int* jr0, int* jcnt, int* xidx, cudaStream_t s) {
    if (ne < 1 || ne > 1024 || kk < 1 || kk > 32 || kk > ne || T < 1) return -2;
    topk_kernel<<<T, 256, 0, s>>>(lg, ld, ne, kk, ids, wts, jr0, jcnt, xidx);
    return int(cudaGetLastError());
}

int infer_moe_swiglu(const float* g, const float* u, int64_t total, float* h, cudaStream_t s) {
    if (total <= 0) return 0;
    swiglu_kernel<<<unsigned((total + 255) / 256), 256, 0, s>>>(g, u, total, h);
    return int(cudaGetLastError());
}

int infer_moe_combine(float* h, int d, int T, int kk, const float* wts, const float* o, const int* rowmap, const float* sh, const float* sg,
                      int64_t sg_ld, cudaStream_t s) {
    if (T < 1) return 0;
    const int64_t n = int64_t(T) * d;
    combine_kernel<<<unsigned((n + 255) / 256), 256, 0, s>>>(h, d, T, kk, wts, o, rowmap, sh, sg, sg_ld);
    return int(cudaGetLastError());
}

}  // extern "C"
