// SenseNova-Loop の mHC（manifold-constrained hyper-connections・残差 2 本）。式は公式の modeling_sensenova_loop.py の MHCRouter と同じ:
//   flat = [a; b]（2d）、r = rsqrt(mean(flat²) + 1e-6)、logits = W·(flat·r) + bias（W は f32 [8][2d]、全部 f32）
//   pre = sigmoid(l0..1)、post = 2·sigmoid(l2..3)、mix = Sinkhorn(l4..7 を 2×2、対数のまま 行 → 列 を iters 回) を exp
//   枝の入口 h = (pre0·a + pre1·b)·pre_scale
//   合流: a' = m00·a + m01·b + post0·y、b' = m10·a + m11·b + post1·y（y = 枝の出口）。h = (a' + b')/2（最後の mHC の層の出口 = 平均）
// Sinkhorn は語ごと（logits が残差で決まる）なので、読み込み時に先に計算はできない。
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int RT = 256;  // 1 行 1 ブロック

__device__ __forceinline__ float lse2(float x, float y) {
    float m = fmaxf(x, y);
    return m + logf(expf(x - m) + expf(y - m));
}

__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void k_route(const float* __restrict__ a, const float* __restrict__ b, int64_t ld, int d, const float* __restrict__ W,
                        const float* __restrict__ bias, int iters, float pre_scale, float* __restrict__ h, int64_t h_ld,
                        float* __restrict__ rt) {
    const int row = blockIdx.x;
    const float* ar = a + row * ld;
    const float* br = b + row * ld;
    float acc[9];
#pragma unroll
    for (int j = 0; j < 9; ++j) acc[j] = 0.f;
    const int d2 = 2 * d;
    for (int i = threadIdx.x; i < d; i += RT) {
        float x = ar[i], y = br[i];
        acc[8] += x * x + y * y;
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] += W[(int64_t)j * d2 + i] * x + W[(int64_t)j * d2 + d + i] * y;
    }
    // ブロックの和（ワープ → 共有メモリ）
    __shared__ float sm[RT / 32][9];
    __shared__ float res[16];
#pragma unroll
    for (int j = 0; j < 9; ++j) {
        float v = acc[j];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        acc[j] = v;
    }
    const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < 9; ++j) sm[w][j] = acc[j];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float s[9];
        for (int j = 0; j < 9; ++j) {
            s[j] = 0.f;
            for (int q = 0; q < RT / 32; ++q) s[j] += sm[q][j];
        }
        const float r = rsqrtf(s[8] / (float)d2 + 1e-6f);
        float l[8];
        for (int j = 0; j < 8; ++j) l[j] = s[j] * r + bias[j];
        // 対数のまま Sinkhorn（行: (0,1)(2,3)、列: (0,2)(1,3)）
        float m0 = l[4], m1 = l[5], m2 = l[6], m3 = l[7];
        for (int it = 0; it < iters; ++it) {
            float z = lse2(m0, m1);
            m0 -= z;
            m1 -= z;
            z = lse2(m2, m3);
            m2 -= z;
            m3 -= z;
            z = lse2(m0, m2);
            m0 -= z;
            m2 -= z;
            z = lse2(m1, m3);
            m1 -= z;
            m3 -= z;
        }
        res[0] = sigm(l[0]);
        res[1] = sigm(l[1]);
        res[2] = 2.f * sigm(l[2]);
        res[3] = 2.f * sigm(l[3]);
        res[4] = expf(m0);
        res[5] = expf(m1);
        res[6] = expf(m2);
        res[7] = expf(m3);
        if (rt) {
            for (int j = 0; j < 6; ++j) rt[row * 8 + j] = res[2 + j];
        }
    }
    __syncthreads();
    const float p0 = res[0], p1 = res[1];
    float* hr = h + row * h_ld;
    for (int i = threadIdx.x; i < d; i += RT) hr[i] = (p0 * ar[i] + p1 * br[i]) * pre_scale;
}

__global__ void k_merge(float* __restrict__ a, float* __restrict__ b, int64_t ld, int d, const float* y, int64_t y_ld, const float* __restrict__ rt,
                        float* h, int64_t h_ld) {
    const int row = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;
    const float* r = rt + row * 8;
    const float po0 = r[0], po1 = r[1], m00 = r[2], m01 = r[3], m10 = r[4], m11 = r[5];
    const float x = a[row * ld + i], z = b[row * ld + i], yy = y[row * y_ld + i];
    const float na = m00 * x + m01 * z + po0 * yy;
    const float nb = m10 * x + m11 * z + po1 * yy;
    a[row * ld + i] = na;
    b[row * ld + i] = nb;
    if (h) h[row * h_ld + i] = (na + nb) * 0.5f;
}

}  // namespace

extern "C" {

// 語 t 行: a・b（f32 [t][ld]）→ h（枝の入口・f32 [t][h_ld]）と rt（f32 [t][8]: post0, post1, m00, m01, m10, m11）。W は f32 [8][2d]
int infer_mhc_route(const float* a, const float* b, int64_t ld, int d, const float* W, const float* bias, int t, int iters, float pre_scale,
                    float* h, int64_t h_ld, float* rt, cudaStream_t s) {
    if (t <= 0) return 0;
    if (iters < 1 || d <= 0) return -2;
    k_route<<<t, RT, 0, s>>>(a, b, ld, d, W, bias, iters, pre_scale, h, h_ld, rt);
    return (int)cudaGetLastError();
}

// 合流: a・b を y（枝の出口）と rt で更新。h が null でなければ h = (a + b)/2（y と同じ番地でもよい: 要素ごとに読んでから書く）
int infer_mhc_merge(float* a, float* b, int64_t ld, int d, const float* y, int64_t y_ld, const float* rt, int t, float* h, int64_t h_ld,
                    cudaStream_t s) {
    if (t <= 0) return 0;
    dim3 g((d + 255) / 256, t);
    k_merge<<<g, 256, 0, s>>>(a, b, ld, d, y, y_ld, rt, h, h_ld);
    return (int)cudaGetLastError();
}

}
