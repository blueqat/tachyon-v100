// 量子化したままの重みで y = W x を計算する（1 文字ずつ出すとき用）。
// W は ggml の並び: 行 n（出力）ごとに k 要素（入力）が続く。行のバイト数は qtype_row_bytes。
// 1 つの warp が 1 行を受け持つ。lane = 8 かたまり × 4 sub。重みは 1 度だけ戻し、B 本の x に使い回す。
#include <cstdint>
#include <cuda_runtime.h>
#include "quant.cuh"

constexpr int ROWS_PER_BLOCK = 8;  // 1 ブロック 256 スレッド = 8 warp

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// x を K 方向に KT 要素ずつ共有メモリへ載せ、そのかたまりの範囲で内積を足していく。
constexpr int KT = 1024;  // B=8 でも 32KB

// 1 行ぶん、かたまり [c0, c1) の内積を B 本足し込む（x_s は共有メモリ上の x のこのかたまり）
template <int T, int B>
__device__ __forceinline__ void row_dot_tile(const uint8_t* row, int c0, int c1, const float* x_s, float acc[B]) {
    const int lane = threadIdx.x & 31;
    const int cl = lane >> 2, sub = lane & 3;
    for (int c = c0 + cl; c < c1; c += 8) {
        float v[8];
        deq8<T>(row, c, sub, v);
        // 8 要素は「4 個 + 4 個」。連続なら p0+4、iq4 系なら p0+16。どちらも float4 で読める
        const int p0 = pos8<T>(c, sub, 0) - c0 * 32, p1 = pos8<T>(c, sub, 4) - c0 * 32;
#pragma unroll
        for (int b = 0; b < B; ++b) {
            const float4 a = *reinterpret_cast<const float4*>(x_s + b * KT + p0);
            const float4 e = *reinterpret_cast<const float4*>(x_s + b * KT + p1);
            acc[b] += v[0] * a.x + v[1] * a.y + v[2] * a.z + v[3] * a.w + v[4] * e.x + v[5] * e.y + v[6] * e.z + v[7] * e.w;
        }
    }
}

// y[b, n] = sum_k W[n, k] x[b, k]
// ブロックは 8 warp。各 warp は ROWS_PER_WARP 行を受け持ち、k が長ければ KT ごとに x を載せ直す。
constexpr int ROWS_PER_WARP = 4;
template <int T, int B>
__global__ void __launch_bounds__(256) gemv_kernel(const uint8_t* __restrict__ w, int64_t row_bytes, int k, int n,
                                                   const float* __restrict__ x, float* __restrict__ y) {
    __shared__ __align__(16) float x_s[B * KT];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * ROWS_PER_BLOCK + warp) * ROWS_PER_WARP;
    float acc[ROWS_PER_WARP][B];
#pragma unroll
    for (int i = 0; i < ROWS_PER_WARP; ++i)
#pragma unroll
        for (int b = 0; b < B; ++b) acc[i][b] = 0.f;
    for (int k0 = 0; k0 < k; k0 += KT) {
        const int kt = min(KT, k - k0);
        __syncthreads();
        for (int i = threadIdx.x; i < B * kt; i += blockDim.x) x_s[(i / kt) * KT + i % kt] = x[int64_t(i / kt) * k + k0 + i % kt];
        __syncthreads();
#pragma unroll
        for (int i = 0; i < ROWS_PER_WARP; ++i) {
            const int r = r0 + i;
            if (r < n) row_dot_tile<T, B>(w + int64_t(r) * row_bytes, k0 / 32, (k0 + kt) / 32, x_s, acc[i]);
        }
    }
#pragma unroll
    for (int i = 0; i < ROWS_PER_WARP; ++i) {
        const int r = r0 + i;
#pragma unroll
        for (int b = 0; b < B; ++b) {
            const float v = warp_sum(acc[i][b]);
            if (lane == 0 && r < n) y[int64_t(b) * n + r] = v;
        }
    }
}

// 1 行ぶんの内積を B 本まとめて計算し、warp 全員が持つ（MoE 用。x は global から）
template <int T, int B>
__device__ __forceinline__ void row_dot(const uint8_t* row, int k, const float* x, int64_t x_stride, float acc[B]) {
    const int lane = threadIdx.x & 31;
    const int cl = lane >> 2, sub = lane & 3;
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0.f;
    for (int c = cl; c < k / 32; c += 8) {
        float v[8];
        deq8<T>(row, c, sub, v);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int p = pos8<T>(c, sub, j);
#pragma unroll
            for (int b = 0; b < B; ++b) acc[b] += v[j] * __ldg(x + b * x_stride + p);
        }
    }
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = warp_sum(acc[b]);
}

// MoE: 仕事 j（= トークン × 選ばれた枠）ごとに、専門家 ids[j] の行列で y[j] = W_e x[j / x_div]
//   gate/up: x はトークンの隠れ状態（x_div = topk）、down: x は仕事ごと（x_div = 1）
template <int T>
__global__ void moe_gemv_kernel(const uint8_t* __restrict__ w, int64_t expert_bytes, int64_t row_bytes, int k, int n,
                                const int32_t* __restrict__ ids, int x_div,
                                const float* __restrict__ x, float* __restrict__ y) {
    const int r = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5);
    const int j = blockIdx.y;
    if (r >= n) return;
    const uint8_t* row = w + int64_t(ids[j]) * expert_bytes + int64_t(r) * row_bytes;
    float acc[1];
    row_dot<T, 1>(row, k, x + int64_t(j / x_div) * k, k, acc);
    if ((threadIdx.x & 31) == 0) y[int64_t(j) * n + r] = acc[0];
}

// 確かめ用: 全部 f32 に戻す
template <int T>
__global__ void dequant_kernel(const uint8_t* __restrict__ w, int64_t row_bytes, int k, int64_t n_rows, float* __restrict__ y) {
    const int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t per_row = int64_t(k / 32) * 4;
    if (t >= per_row * n_rows) return;
    const int64_t r = t / per_row;
    const int c = int((t % per_row) >> 2), sub = int(t & 3);
    float v[8];
    deq8<T>(w + r * row_bytes, c, sub, v);
#pragma unroll
    for (int j = 0; j < 8; ++j) y[r * k + pos8<T>(c, sub, j)] = v[j];
}

#define FOR_TYPES(X) X(Q_F32) X(Q_F16) X(Q_BF16) X(Q_Q8_0) X(Q_Q5_1) X(Q_Q3_K) X(Q_Q4_K) X(Q_Q5_K) X(Q_Q6_K) \
    X(Q_IQ4_NL) X(Q_IQ4_XS) X(Q_IQ3_XXS) X(Q_IQ3_S) X(Q_IQ2_S) X(Q_Q2_0)

template <int T>
static void launch_gemv(const uint8_t* w, int64_t rb, int k, int n, const float* x, float* y, int b, cudaStream_t s) {
    const int rows_per_block = ROWS_PER_BLOCK * ROWS_PER_WARP;
    const dim3 grid((n + rows_per_block - 1) / rows_per_block), block(32 * ROWS_PER_BLOCK);
    switch (b) {
        case 1: gemv_kernel<T, 1><<<grid, block, 0, s>>>(w, rb, k, n, x, y); break;
        case 2: gemv_kernel<T, 2><<<grid, block, 0, s>>>(w, rb, k, n, x, y); break;
        case 4: gemv_kernel<T, 4><<<grid, block, 0, s>>>(w, rb, k, n, x, y); break;
        case 8: gemv_kernel<T, 8><<<grid, block, 0, s>>>(w, rb, k, n, x, y); break;
        default:  // それ以外の本数は 1 本ずつ
            for (int i = 0; i < b; ++i) gemv_kernel<T, 1><<<grid, block, 0, s>>>(w, rb, k, n, x + int64_t(i) * k, y + int64_t(i) * n);
    }
}

extern "C" {

// 戻り値: 0 = 成功、-1 = 知らない型、-2 = k が 32 の倍数でない、それ以外は cudaError
int infer_gemv(int qtype, const void* w, int k, int n, const float* x, float* y, int b, cudaStream_t s) {
    if (k % 32) return -2;
    const int64_t rb = qtype_row_bytes(qtype, k);
    const uint8_t* wp = static_cast<const uint8_t*>(w);
    switch (qtype) {
#define CASE(T) case T: launch_gemv<T>(wp, rb, k, n, x, y, b, s); break;
        FOR_TYPES(CASE)
#undef CASE
        default: return -1;
    }
    return int(cudaGetLastError());
}

int infer_moe_gemv(int qtype, const void* w, int k, int n, const int32_t* ids, int n_jobs, int x_div,
                   const float* x, float* y, cudaStream_t s) {
    if (k % 32) return -2;
    const int64_t rb = qtype_row_bytes(qtype, k), eb = rb * n;
    const uint8_t* wp = static_cast<const uint8_t*>(w);
    const dim3 grid((n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, n_jobs), block(32 * ROWS_PER_BLOCK);
    switch (qtype) {
#define CASE(T) case T: moe_gemv_kernel<T><<<grid, block, 0, s>>>(wp, eb, rb, k, n, ids, x_div, x, y); break;
        FOR_TYPES(CASE)
#undef CASE
        default: return -1;
    }
    return int(cudaGetLastError());
}

int infer_dequant(int qtype, const void* w, int k, int64_t n_rows, float* y, cudaStream_t s) {
    if (k % 32) return -2;
    const int64_t rb = qtype_row_bytes(qtype, k);
    const int64_t threads = int64_t(k / 32) * 4 * n_rows;
    const unsigned grid = unsigned((threads + 255) / 256);
    const uint8_t* wp = static_cast<const uint8_t*>(w);
    switch (qtype) {
#define CASE(T) case T: dequant_kernel<T><<<grid, 256, 0, s>>>(wp, rb, k, n_rows, y); break;
        FOR_TYPES(CASE)
#undef CASE
        default: return -1;
    }
    return int(cudaGetLastError());
}

int64_t infer_row_bytes(int qtype, int64_t k) { return qtype_row_bytes(qtype, k); }

}  // extern "C"
