// 画像の符号化器（Qwen3.8-27B の mmproj・qwen3vl_merger）のカーネル。ViT 27 層（幅 1152・16 頭 × 72）＋ 2×2 をまとめる 2 層の MLP。
// 手本は llama.cpp の tools/mtmd/models/qwen3vl.cpp（式と順）。重みは F16、活性は f32 で持ち、行列の入力だけ fp16 にする（溜めは f32）。
//   infer_vis_gemm       Y[M,N] = X[M,K]·W^T + b（X fp16 [M][K]、W fp16 [N][K]、wmma 16×16×16。出力 f32・f32 に足す・fp16・GELU して fp16）
//   infer_vis_layernorm  y = LN(x)·w + b（x f32 [M][K] → y fp16）
//   infer_vis_rope_split qkv f32 [M][3·1152] に 2 次元の RoPE（llama.cpp の GGML_ROPE_TYPE_VISION）を掛け、q・k・v を fp16 [3][頭][M][80] に詰める
//   infer_vis_attn       因果でない Attention（頭ごと。O は fp16 [M][1152]）
//   infer_vis_f32_to_f16 / infer_vis_add
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

namespace {

constexpr int DH = 72;    // 頭の幅
constexpr int DHP = 80;   // 16 の倍数に詰めた幅
constexpr int NHEAD = 16;

__device__ __forceinline__ float gelu_tanh(float x) {
    // ggml_gelu と同じ式（tanh の近似）
    const float c = 0.7978845608028654f;  // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

// ---------------------------------------------------------------- GEMM
// ブロック 128×128、warp 8（4×2）で 1 warp 32×64（16×16 を 2×4）。K は 32 ずつ共有メモリへ
constexpr int GBM = 128, GBN = 128, GBK = 32, GPAD = 8;
// mode: 0 = f32 に書く、1 = f32 に足す、2 = fp16 に書く、3 = GELU して fp16 に書く
__global__ void __launch_bounds__(256) gemm_kernel(const __half* __restrict__ X, const __half* __restrict__ W, const float* __restrict__ bias,
                                                   void* __restrict__ Y, int M, int N, int K, int ldy, int mode) {
    __shared__ __align__(32) __half As[GBM][GBK + GPAD];
    __shared__ __align__(32) __half Bs[GBN][GBK + GPAD];
    const int tid = threadIdx.x, warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;  // 4 × 2
    const int m0 = blockIdx.y * GBM, n0 = blockIdx.x * GBN;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) wmma::fill_fragment(acc[i][j], 0.f);
    for (int k0 = 0; k0 < K; k0 += GBK) {
        // 1 スレッド 8 要素（16 B）× 2 回ずつ、A・B それぞれ 128 × 32
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int e = (tid + r * 256) * 8;  // 0..4095
            const int row = e / GBK, col = e % GBK;
            const int gk = k0 + col;
            uint4 va = make_uint4(0, 0, 0, 0), vb = make_uint4(0, 0, 0, 0);
            if (m0 + row < M && gk < K) va = *reinterpret_cast<const uint4*>(X + int64_t(m0 + row) * K + gk);
            if (n0 + row < N && gk < K) vb = *reinterpret_cast<const uint4*>(W + int64_t(n0 + row) * K + gk);
            *reinterpret_cast<uint4*>(&As[row][col]) = va;
            *reinterpret_cast<uint4*>(&Bs[row][col]) = vb;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < GBK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[4];
#pragma unroll
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(fa[i], &As[wm * 32 + i * 16][kk], GBK + GPAD);
#pragma unroll
            for (int j = 0; j < 4; ++j) wmma::load_matrix_sync(fb[j], &Bs[wn * 64 + j * 16][kk], GBK + GPAD);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        __syncthreads();
    }
    // 書き出し: warp ごとに 16×16 を共有メモリに置いてから（As を使い回す: 1 warp 16×16 f32 = 1KB、8 warp で 8KB ≦ As+Bs）
    float* st = reinterpret_cast<float*>(&As[0][0]) + warp * 256;
    const int lane = tid & 31;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            wmma::store_matrix_sync(st, acc[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) {
                const int r = m0 + wm * 32 + i * 16 + e / 16, c = n0 + wn * 64 + j * 16 + e % 16;
                if (r < M && c < N) {
                    float v = st[e] + (bias ? bias[c] : 0.f);
                    const int64_t o = int64_t(r) * ldy + c;
                    if (mode == 0) static_cast<float*>(Y)[o] = v;
                    else if (mode == 1) static_cast<float*>(Y)[o] += v;
                    else if (mode == 2) static_cast<__half*>(Y)[o] = __float2half(v);
                    else static_cast<__half*>(Y)[o] = __float2half(gelu_tanh(v));
                }
            }
            __syncwarp();
        }
}

// ---------------------------------------------------------------- LayerNorm（1 行 1 ブロック 256）
__device__ float block_sum(float v, float* sh) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < int(blockDim.x >> 5); ++i) t += sh[i];
    return t;
}

__global__ void layernorm_kernel(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b, int K, float eps,
                                 __half* __restrict__ y16, float* __restrict__ y32) {
    __shared__ float sh[32];
    const float* xr = x + int64_t(blockIdx.x) * K;
    float s = 0.f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) s += xr[i];
    const float mean = block_sum(s, sh) / K;
    float v = 0.f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        const float d = xr[i] - mean;
        v += d * d;
    }
    const float var = block_sum(v, sh) / K;
    const float sc = rsqrtf(var + eps);
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        const float o = (xr[i] - mean) * sc * w[i] + b[i];
        if (y16) y16[int64_t(blockIdx.x) * K + i] = __float2half(o);
        if (y32) y32[int64_t(blockIdx.x) * K + i] = o;
    }
}

// ---------------------------------------------------------------- RoPE（2 次元）と q・k・v の詰め直し
// 1 スレッド = (行, 頭, 対 i < 36)。対 (i, i+36) を回す。i < 18 は位置 py（周波数 θ^(i)）、18 ≤ i < 36 は px（θ^(i−18)）。
// θ = 10000^(−2/36)。v はそのまま。詰め直しは [3][頭][M][80]（72..79 は 0）
__global__ void rope_split_kernel(const float* __restrict__ qkv, const int* __restrict__ py, const int* __restrict__ px, int M, float theta_scale,
                                  __half* __restrict__ out) {
    const int64_t id = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int per = NHEAD * (DHP / 2);  // 1 行の (頭, 対) の数（対は 40 まで。36..39 は詰めの 0 と v の残り）
    if (id >= int64_t(M) * per) return;
    const int r = int(id / per), rest = int(id % per), h = rest / (DHP / 2), i = rest % (DHP / 2);
    const float* row = qkv + int64_t(r) * 3 * NHEAD * DH;
    const int64_t plane = int64_t(NHEAD) * M * DHP;
    __half* oq = out + (int64_t(h) * M + r) * DHP;
    __half* ok = oq + plane;
    __half* ov = oq + 2 * plane;
    if (i < DH / 2) {
        const int p = i < 18 ? py[r] : px[r];
        const int e = i < 18 ? i : i - 18;
        const float th = float(p) * powf(theta_scale, float(e));
        float sn, cs;
        sincosf(th, &sn, &cs);
        const float* q = row + h * DH;
        const float* k = row + NHEAD * DH + h * DH;
        const float q0 = q[i], q1 = q[i + DH / 2], k0 = k[i], k1 = k[i + DH / 2];
        oq[i] = __float2half(q0 * cs - q1 * sn);
        oq[i + DH / 2] = __float2half(q0 * sn + q1 * cs);
        ok[i] = __float2half(k0 * cs - k1 * sn);
        ok[i + DH / 2] = __float2half(k0 * sn + k1 * cs);
        const float* v = row + 2 * NHEAD * DH + h * DH;
        ov[i] = __float2half(v[i]);
        ov[i + DH / 2] = __float2half(v[i + DH / 2]);
    } else {
        // 72..79 の詰め（i = 36..39 → 列 72 + 2(i−36)、+1）
        const int c = DH + 2 * (i - DH / 2);
        oq[c] = oq[c + 1] = ok[c] = ok[c + 1] = ov[c] = ov[c + 1] = __float2half(0.f);
    }
}

// ---------------------------------------------------------------- Attention（因果でない・online softmax）
// ブロック = (頭, 128 個の問い)、warp 8 本で 16 個ずつ。鍵は 64 個ずつ共有メモリへ（8 warp で使い回す）。
// O・最大・和は 2 lane で 1 行（40 列ずつ）のレジスタに持つ。共有メモリは 96KB ちょうど
constexpr int AQ = 128, AK = 64, AW = 8;
__global__ void __launch_bounds__(256) attn_kernel(const __half* __restrict__ qkv16, int M, float scale, __half* __restrict__ out) {
    extern __shared__ __align__(32) uint8_t smem[];
    __half* Qs = reinterpret_cast<__half*>(smem);   // [128][80]
    __half* Ks = Qs + AQ * DHP;                     // [64][80]
    __half* Vs = Ks + AK * DHP;                     // [64][80]
    float* Ss = reinterpret_cast<float*>(Vs + AK * DHP);        // warp ごと [16][80]（S は [16][64]、PV の一時は [16][80]）
    __half* Ps = reinterpret_cast<__half*>(Ss + AW * 16 * DHP);  // warp ごと [16][64]
    const int h = blockIdx.y, q0 = blockIdx.x * AQ;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int64_t plane = int64_t(NHEAD) * M * DHP;
    const __half* Q = qkv16 + int64_t(h) * M * DHP;
    const __half* Kg = Q + plane;
    const __half* Vg = Q + 2 * plane;
    for (int e = tid; e < AQ * DHP / 8; e += AW * 32) {
        const int r = e / (DHP / 8), c = (e % (DHP / 8)) * 8;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (q0 + r < M) v = *reinterpret_cast<const uint4*>(Q + int64_t(q0 + r) * DHP + c);
        *reinterpret_cast<uint4*>(Qs + r * DHP + c) = v;
    }
    float* S = Ss + warp * 16 * DHP;
    __half* P = Ps + warp * 16 * AK;
    const int r = lane >> 1, half = lane & 1;
    float o[40];
#pragma unroll
    for (int c = 0; c < 40; ++c) o[c] = 0.f;
    float mx_r = -INFINITY, l_r = 0.f;
    for (int k0 = 0; k0 < M; k0 += AK) {
        __syncthreads();
        for (int e = tid; e < AK * DHP / 8; e += AW * 32) {
            const int rr = e / (DHP / 8), c = (e % (DHP / 8)) * 8;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (k0 + rr < M) {
                kv = *reinterpret_cast<const uint4*>(Kg + int64_t(k0 + rr) * DHP + c);
                vv = *reinterpret_cast<const uint4*>(Vg + int64_t(k0 + rr) * DHP + c);
            }
            *reinterpret_cast<uint4*>(Ks + rr * DHP + c) = kv;
            *reinterpret_cast<uint4*>(Vs + rr * DHP + c) = vv;
        }
        __syncthreads();
        {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sf[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) wmma::fill_fragment(sf[j], 0.f);
#pragma unroll
            for (int kk = 0; kk < DHP; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                wmma::load_matrix_sync(fa, Qs + warp * 16 * DHP + kk, DHP);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                    wmma::load_matrix_sync(fb, Ks + j * 16 * DHP + kk, DHP);
                    wmma::mma_sync(sf[j], fa, fb, sf[j]);
                }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) wmma::store_matrix_sync(S + j * 16, sf[j], AK, wmma::mem_row_major);
        }
        __syncwarp();
        float al;
        {
            float m = -INFINITY;
            for (int c = half * 32; c < half * 32 + 32; ++c) {
                const float v = k0 + c < M ? S[r * AK + c] * scale : -INFINITY;
                S[r * AK + c] = v;
                m = fmaxf(m, v);
            }
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
            const float mn = fmaxf(mx_r, m);
            float sum = 0.f;
            for (int c = half * 32; c < half * 32 + 32; ++c) {
                const float p = __expf(S[r * AK + c] - mn);
                sum += p;
                P[r * AK + c] = __float2half(p);
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            al = __expf(mx_r - mn);
            l_r = l_r * al + sum;
            mx_r = mn;
        }
        __syncwarp();
        {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> of[5];
#pragma unroll
            for (int j = 0; j < 5; ++j) wmma::fill_fragment(of[j], 0.f);
#pragma unroll
            for (int kk = 0; kk < AK; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                wmma::load_matrix_sync(fa, P + kk, AK);
#pragma unroll
                for (int j = 0; j < 5; ++j) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
                    wmma::load_matrix_sync(fb, Vs + kk * DHP + j * 16, DHP);
                    wmma::mma_sync(of[j], fa, fb, of[j]);
                }
            }
            __syncwarp();
#pragma unroll
            for (int j = 0; j < 5; ++j) wmma::store_matrix_sync(S + j * 16, of[j], DHP, wmma::mem_row_major);
            __syncwarp();
#pragma unroll
            for (int c = 0; c < 40; ++c) o[c] = o[c] * al + S[r * DHP + half * 40 + c];
            __syncwarp();
        }
    }
    const int q = q0 + warp * 16 + r;
    if (q < M) {
        const float inv = 1.f / l_r;
        for (int c = 0; c < 40; ++c) {
            const int d = half * 40 + c;
            if (d < DH) out[int64_t(q) * NHEAD * DH + h * DH + d] = __float2half(o[c] * inv);
        }
    }
}

__global__ void f32_to_f16_kernel(const float* __restrict__ x, int64_t n, __half* __restrict__ y) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}
__global__ void add_kernel(float* __restrict__ y, const float* __restrict__ x, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

}  // namespace

extern "C" {

int infer_vis_gemm(const void* x, const void* w, const float* bias, void* y, int m, int n, int k, int ldy, int mode, cudaStream_t s) {
    if (k % 8 || m <= 0 || n <= 0) return -2;
    dim3 grid((n + GBN - 1) / GBN, (m + GBM - 1) / GBM);
    gemm_kernel<<<grid, 256, 0, s>>>(static_cast<const __half*>(x), static_cast<const __half*>(w), bias, y, m, n, k, ldy, mode);
    return int(cudaGetLastError());
}

int infer_vis_layernorm(const float* x, const float* w, const float* b, int m, int k, float eps, void* y16, float* y32, cudaStream_t s) {
    layernorm_kernel<<<m, 256, 0, s>>>(x, w, b, k, eps, static_cast<__half*>(y16), y32);
    return int(cudaGetLastError());
}

int infer_vis_rope_split(const float* qkv, const int* py, const int* px, int m, float theta_scale, void* out, cudaStream_t s) {
    const int64_t n = int64_t(m) * NHEAD * (DHP / 2);
    rope_split_kernel<<<unsigned((n + 255) / 256), 256, 0, s>>>(qkv, py, px, m, theta_scale, static_cast<__half*>(out));
    return int(cudaGetLastError());
}

int64_t infer_vis_qkv16_bytes(int m) { return int64_t(3) * NHEAD * m * DHP * 2; }

int infer_vis_attn(const void* qkv16, int m, float scale, void* out, cudaStream_t s) {
    const int sm = (AQ + 2 * AK) * DHP * 2 + AW * 16 * DHP * 4 + AW * 16 * AK * 2;
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
        set = true;
    }
    dim3 grid((m + AQ - 1) / AQ, NHEAD);
    attn_kernel<<<grid, AW * 32, sm, s>>>(static_cast<const __half*>(qkv16), m, scale, static_cast<__half*>(out));
    return int(cudaGetLastError());
}

int infer_vis_f32_to_f16(const float* x, int64_t n, void* y, cudaStream_t s) {
    f32_to_f16_kernel<<<unsigned((n + 255) / 256), 256, 0, s>>>(x, n, static_cast<__half*>(y));
    return int(cudaGetLastError());
}

int infer_vis_add(float* y, const float* x, int64_t n, cudaStream_t s) {
    add_kernel<<<unsigned((n + 255) / 256), 256, 0, s>>>(y, x, n);
    return int(cudaGetLastError());
}

}  // extern "C"
