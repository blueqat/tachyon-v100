// Qwen3.8-27B（GGUF arch qwen35）の 1 歩をつなぐ小さなカーネル（src/model.rs から呼ぶ）。
//   embed:   語の番号 → token_embd の行を戻す（GGUF の並びのまま。倍率なし）
//   rmsnorm: 行ごとに x / sqrt(mean(x²) + eps) · w（f32 と f16 のどちらか・両方に書く）
//   f32→f16: プレフィルの GEMM の入力（X は fp16）
//   g3→f16:  g3 の並び（kernels/g3.cuh）の重みを fp16 [n][k] に戻す。プレフィルは戻した重みを GEMM の F16 の道で掛ける
//            （重みは g3 の並び 1 通りだけを GPU に置く。GGUF の並びと 2 通りは 32GB に入らない）
//   argmax:  行ごとの 1 位（同じ値なら番号の小さい方）
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "g3.cuh"

using namespace g3;

namespace {

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ---------------------------------------------------------------- 埋め込み
// 1 スレッド = 1 トークンのかたまり（32 要素）の sub（8 要素）。並びは kernels/gemv.cu の dequant_kernel と同じ
template <int T>
__global__ void embed_kernel(const uint8_t* __restrict__ w, int64_t rb, int k, const int* __restrict__ ids, int n_tok,
                             float* __restrict__ y) {
    const int64_t per_row = int64_t(k / 32) * 4;
    const int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= per_row * n_tok) return;
    const int r = int(t / per_row);
    const int c = int((t % per_row) >> 2), sub = int(t & 3);
    float v[8];
    deq8<T>(w + int64_t(ids[r]) * rb, c, sub, v);
#pragma unroll
    for (int j = 0; j < 8; ++j) y[int64_t(r) * k + pos8<T>(c, sub, j)] = v[j];
}

// ---------------------------------------------------------------- RMSNorm
// 1 ブロック（256 スレッド）= 1 行。2 乗の和は f32（スレッドごとに 20 個ほど足してから木で）
__global__ void __launch_bounds__(256) rmsnorm_kernel(const float* __restrict__ x, int64_t x_ld, const float* __restrict__ w, int k,
                                                     float eps, float* __restrict__ y, int64_t y_ld, __half* __restrict__ y16,
                                                     int64_t y16_ld) {
    __shared__ float red[8];
    const int r = blockIdx.x;
    const float* xr = x + int64_t(r) * x_ld;
    float ss = 0.f;
    for (int i = threadIdx.x; i < k; i += 256) {
        const float v = xr[i];
        ss = fmaf(v, v, ss);
    }
    ss = wsum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) tot += red[i];
    const float sc = 1.0f / sqrtf(tot / float(k) + eps);  // ggml の rms_norm と同じ形
    for (int i = threadIdx.x; i < k; i += 256) {
        const float v = (xr[i] * sc) * w[i];
        if (y) y[int64_t(r) * y_ld + i] = v;
        if (y16) y16[int64_t(r) * y16_ld + i] = __float2half_rn(v);
    }
}

__global__ void f32_to_f16_kernel(const float* __restrict__ x, int64_t n, __half* __restrict__ y) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half_rn(x[i]);
}

// ---------------------------------------------------------------- g3 → f16
// 1 スレッド = 1 行の 1 かたまり（32 要素）。並んだ 32 スレッドは同じタイル・同じかたまりの 32 の置き場（slot）
// → 重みの読みは 512 B 続き、書きは行ごとに 64 B。中身は gemv3.cu の g3_dequant_kernel と同じ（f32 で sc·q + o を 1 回だけ丸める）
template <int T>
__global__ void g3_deq16_kernel(const uint8_t* __restrict__ w, int k, int n, __half* __restrict__ y) {
    const int kb = k / 32;
    const int64_t id = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int n_tiles = (n + TILE - 1) / TILE;
    if (id >= int64_t(n_tiles) * kb * TILE) return;
    const int slot = int(id % TILE);
    const int64_t tj = id / TILE;
    const int j = int(tj % kb), tile = int(tj / kb);
    const int r = tile * TILE + row_of_slot(slot);
    const uint8_t* u = w + int64_t(tile) * tile_bytes(T, k) + int64_t(j / 8) * unit_bytes(T);
    const typename W<T>::C c = W<T>::ldc(u, j % 8, slot);
    const typename W<T>::S s = W<T>::lds(u, slot);
    if (r >= n) return;
    uint32_t a[16];
    W<T>::deq(c, a);
    float sc[W<T>::NG], o;
    W<T>::scale(s, j % 8, sc, o);
    __align__(16) __half out[32];
#pragma unroll
    for (int st = 0; st < 8; ++st) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t h = a[2 * st + (i >> 1)];
            const float q = __half2float(__ushort_as_half(uint16_t(i & 1 ? h >> 16 : h)));
            const int g = W<T>::NG == 2 ? st >> 2 : 0;
            out[st * 4 + i] = __float2half_rn(sc[g] * q + o);
        }
    }
    uint4* dst = reinterpret_cast<uint4*>(y + int64_t(r) * k + j * 32);
#pragma unroll
    for (int i = 0; i < 4; ++i) dst[i] = reinterpret_cast<const uint4*>(out)[i];
}

// ---------------------------------------------------------------- argmax
// 1 ブロック（1024 スレッド）= 1 行
__global__ void __launch_bounds__(1024) argmax_kernel(const float* __restrict__ x, int n, int64_t ld, int* __restrict__ out,
                                                     float* __restrict__ out_v) {
    __shared__ float sv[32];
    __shared__ int si[32];
    const float* xr = x + int64_t(blockIdx.x) * ld;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += 1024) {
        const float v = xr[i];
        if (v > bv) bv = v, bi = i;  // i は増えるだけなので、同じ値なら前の（小さい番号）が残る
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float v = __shfl_xor_sync(0xffffffffu, bv, o);
        const int j = __shfl_xor_sync(0xffffffffu, bi, o);
        if (v > bv || (v == bv && j < bi)) bv = v, bi = j;
    }
    if ((threadIdx.x & 31) == 0) sv[threadIdx.x >> 5] = bv, si[threadIdx.x >> 5] = bi;
    __syncthreads();
    if (threadIdx.x < 32) {
        bv = sv[threadIdx.x], bi = si[threadIdx.x];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float v = __shfl_xor_sync(0xffffffffu, bv, o);
            const int j = __shfl_xor_sync(0xffffffffu, bi, o);
            if (v > bv || (v == bv && j < bi)) bv = v, bi = j;
        }
        if (threadIdx.x == 0) {
            out[blockIdx.x] = bi == 0x7fffffff ? -1 : bi;  // 全部 NaN なら -1
            if (out_v) out_v[blockIdx.x] = bv;
        }
    }
}

// ================================================================ デコード（M2）の融かしたカーネル
// どれも M1 の道（rmsnorm → xprep、conv → gdn decode → xprep、rope）と同じ式・同じ足す順で、値はビットまで同じにする

__device__ __forceinline__ float silu_f(float x) { return x / (1.f + expf(-x)); }  // gdn.cu と同じ（expf）
__device__ __forceinline__ float sigm_f(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float softplus_f(float x) { return x > 20.f ? x : log1pf(expf(x)); }
__device__ __forceinline__ float4 ldf4(const float* p) { return *reinterpret_cast<const float4*>(p); }

// RMSNorm を g3 の x の下ごしらえ（xhl・xsi）に直に書く（xprep を 1 本省く）。1 ブロック（256）= 1 行。
// 2 乗の和は rmsnorm_kernel と同じ順なので、xprep(rmsnorm(x)) とビットまで同じ。y（f32）は要るときだけ
// 1 行を RMS_XP_NT スレッドで（M4c: 256 → 1024。行の数が少ない（B ≤ 8）ので 1 ブロックの読みの待ちを短く。確かめ 28.22→28.07ms/歩）。
// ⚠️ 1024 では 2 乗の和の順が rmsnorm_kernel（256）と変わるので、上の「ビットまで同じ」は 256 のときだけ（温度 0 の答えは前の版と同じことを口で確かめた）。
// TACHYON_RMS_NT=256 で前の幅
template <int RMS_XP_NT>
__global__ void __launch_bounds__(RMS_XP_NT) rmsnorm_xp_kernel(const float* __restrict__ x, int64_t x_ld, const float* __restrict__ w, int k,
                                                        float eps, float* __restrict__ y, int64_t y_ld, __half* __restrict__ xhl,
                                                        float2* __restrict__ xsi) {
    __shared__ float red[RMS_XP_NT / 32];
    const int r = blockIdx.x;
    const float* xr = x + int64_t(r) * x_ld;
    float ss = 0.f;
    for (int i = threadIdx.x; i < k; i += RMS_XP_NT) {
        const float v = xr[i];
        ss = fmaf(v, v, ss);
    }
    ss = wsum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < RMS_XP_NT / 32; ++i) tot += red[i];
    const float sc = 1.0f / sqrtf(tot / float(k) + eps);
    const int lane = threadIdx.x & 31;
    // 8 要素の組 g（k/8 は 32 の倍数）。prep8 は warp の全員で呼ぶ
    for (int g0 = threadIdx.x & ~31; g0 < k / 8; g0 += RMS_XP_NT) {
        const int g = g0 + lane;
        const float4 a = ldf4(xr + g * 8), b = ldf4(xr + g * 8 + 4);
        const float4 wa = ldf4(w + g * 8), wb = ldf4(w + g * 8 + 4);
        const float v[8] = {(a.x * sc) * wa.x, (a.y * sc) * wa.y, (a.z * sc) * wa.z, (a.w * sc) * wa.w,
                            (b.x * sc) * wb.x, (b.y * sc) * wb.y, (b.z * sc) * wb.z, (b.w * sc) * wb.w};
        if (y) {
            float* yr = y + int64_t(r) * y_ld + g * 8;
            *reinterpret_cast<float4*>(yr) = make_float4(v[0], v[1], v[2], v[3]);
            *reinterpret_cast<float4*>(yr + 4) = make_float4(v[4], v[5], v[6], v[7]);
        }
        __half* hi = xhl + int64_t(r) * 2 * k + g * 8;
        prep8(v, hi, hi + k, xsi + int64_t(r) * (k / 32) + g / 4, true, (lane & 3) == 0);
    }
}

// GDN の 1 歩（T = 1）: q・k の l2 正規化 → 状態の更新 → rms_norm × w × silu(z) → ssm_out の x の下ごしらえ（xhl・xsi、k = Hv·D）。
// c は conv（＋silu）の後で q・k はまだ正規化していない [B][c_ld]（conv は qkv の GEMV の出口で済ませてある: infer_g3_gemv_multi_conv）。
// 1 ブロック = (頭 h, 列 b)、512 スレッド。正規化は gdn.cu の conv_kernel、状態の更新と出口は decode_kernel と同じ式・同じ足す順。
// y（f32 の出口）は要るときだけ
constexpr int GS_NT = 512;
__device__ __forceinline__ float sum8f(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 1);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    return v + __shfl_xor_sync(0xffffffff, v, 4);
}
__device__ __forceinline__ float fma4f(float4 a, float4 b, float c) { return fmaf(a.w, b.w, fmaf(a.z, b.z, fmaf(a.y, b.y, fmaf(a.x, b.x, c)))); }
__device__ __forceinline__ float dot4f(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

__global__ void __launch_bounds__(GS_NT) gdn_step_kernel(const float* __restrict__ c, int64_t c_ld, const float* __restrict__ z, int64_t z_ld,
                                                         const float* __restrict__ alpha, const float* __restrict__ beta, int64_t ab_ld,
                                                         const float* __restrict__ ssm_a, const float* __restrict__ dt_bias,
                                                         const float* __restrict__ norm_w, float* __restrict__ state, const int* __restrict__ slots,
                                                         float* __restrict__ y, int64_t y_ld, __half* __restrict__ xhl, float2* __restrict__ xsi,
                                                         int Hk, int Hv, float eps) {
    constexpr int D = 128;
    __shared__ __align__(16) float qk_s[2 * D];
    __shared__ __align__(16) float o_s[D];
    __shared__ __align__(16) float out_s[D];
    __shared__ float red[8];
    const int h = blockIdx.x, b = blockIdx.y, kh = h % Hk, tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31, g = lane & 7, rq = lane >> 3;
    const int j0 = warp * 8 + rq;
    const int slot = slots ? slots[b] : b;
    const float* M = state + (int64_t(slot) * Hv + h) * D * D;
    float4 s[2][4];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int m = 0; m < 4; ++m) s[r][m] = ldf4(M + (j0 + 4 * r) * D + (m * 8 + g) * 4);
    const float* row = c + int64_t(b) * c_ld;
    // q・k の l2 正規化（conv_kernel と同じ: warp ごとの和を 4 つ順に足して rsqrtf(和 + eps)）
    float o = 0.f;
    if (tid < 2 * D) {
        o = row[(tid < D ? kh : Hk + kh) * D + (tid & (D - 1))];
        const float ss = wsum(o * o);
        if (lane == 0) red[warp] = ss;
    }
    const float al = alpha[int64_t(b) * ab_ld + h], bl = beta[int64_t(b) * ab_ld + h];
    const float v0 = row[(2 * Hk + h) * D + j0], v1 = row[(2 * Hk + h) * D + j0 + 4];
    const float4 zv = warp == 0 ? ldf4(z + int64_t(b) * z_ld + h * D + lane * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    const float ah = ssm_a[h], dh = dt_bias[h], scale = rsqrtf(float(D));
    const float4 nw = ldf4(norm_w + lane * 4);
    __syncthreads();
    if (tid < 2 * D) {
        const int q4 = tid < D ? 0 : 4;
        o *= rsqrtf(red[q4] + red[q4 + 1] + red[q4 + 2] + red[q4 + 3] + eps);
        qk_s[tid] = o;
    }
    __syncthreads();
    const float eg = expf(softplus_f(al + dh) * ah), be = sigm_f(bl);
    const float* qs = qk_s;
    const float* ks = qk_s + D;
    float4 k4[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) k4[m] = ldf4(ks + (m * 8 + g) * 4);
    float kv0 = 0.f, kv1 = 0.f;
#pragma unroll
    for (int m = 0; m < 4; ++m) kv0 = fma4f(s[0][m], k4[m], kv0), kv1 = fma4f(s[1][m], k4[m], kv1);
    kv0 = sum8f(kv0), kv1 = sum8f(kv1);
    const float d[2] = {(v0 - eg * kv0) * be, (v1 - eg * kv1) * be};
    float o0 = 0.f, o1 = 0.f;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const float4 q4 = ldf4(qs + (m * 8 + g) * 4);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float4& x = s[r][m];
            x.x = fmaf(k4[m].x, d[r], eg * x.x), x.y = fmaf(k4[m].y, d[r], eg * x.y);
            x.z = fmaf(k4[m].z, d[r], eg * x.z), x.w = fmaf(k4[m].w, d[r], eg * x.w);
        }
        o0 = fma4f(s[0][m], q4, o0), o1 = fma4f(s[1][m], q4, o1);
    }
    o0 = sum8f(o0), o1 = sum8f(o1);
    if (g == 0) o_s[j0] = o0 * scale, o_s[j0 + 4] = o1 * scale;
    // 状態を書き戻す（出口を待たない）
    float* Mo = state + (int64_t(slot) * Hv + h) * D * D;
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int m = 0; m < 4; ++m) *reinterpret_cast<float4*>(Mo + (j0 + 4 * r) * D + (m * 8 + g) * 4) = s[r][m];
    __syncthreads();
    if (warp == 0) {  // 出口: rms_norm(o) × w × silu(z)（decode_kernel と同じ）→ ssm_out の x の下ごしらえ
        const float4 ov = ldf4(&o_s[lane * 4]);
        const float rs = rsqrtf(wsum(dot4f(ov, ov)) / D + eps);
        const float4 ow = make_float4(ov.x * rs * nw.x * silu_f(zv.x), ov.y * rs * nw.y * silu_f(zv.y), ov.z * rs * nw.z * silu_f(zv.z),
                                      ov.w * rs * nw.w * silu_f(zv.w));
        *reinterpret_cast<float4*>(&out_s[lane * 4]) = ow;
        if (y) *reinterpret_cast<float4*>(y + int64_t(b) * y_ld + h * D + lane * 4) = ow;
        __syncwarp();
        const int K = Hv * D, l = lane & 15;
        float v[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = out_s[l * 8 + i];
        __half* hi = xhl + int64_t(b) * 2 * K + h * D + l * 8;
        prep8(v, hi, hi + K, xsi + int64_t(b) * (K / 32) + h * 4 + l / 4, lane < 16, (lane & 3) == 0);
    }
}

// RoPE の cos・sin の表（デコードの 1 歩で 16 層が同じ角度を使う）: tab[t][i] = (cos, sin)(pos · theta_scale^i)、i < n_rot/2。
// 角度と sincosf は attn.cu の rope_norm_kernel と同じ式（pos_mode 1: 文字だけ。区画 3 は位置 0）
// YaRN（attn.cu の rope_yarn_cs と同じ式・ggml-cuda の rope_yarn と同じ）
struct RopeYarn {
    float freq_scale, ext_factor, mscale, corr0, corr1;
};
static __device__ __forceinline__ void rope_yarn_cs(float theta_extrap, int i, const RopeYarn y, float* cs, float* sn) {
    const float theta_interp = y.freq_scale * theta_extrap;
    float theta = theta_interp, mscale = y.mscale;
    if (y.ext_factor != 0.0f) {
        const float r = (float(i) - y.corr0) / fmaxf(0.001f, y.corr1 - y.corr0);
        const float ramp_mix = (1.0f - fminf(1.0f, fmaxf(0.0f, r))) * y.ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / y.freq_scale);
    }
    *cs = cosf(theta) * mscale;
    *sn = sinf(theta) * mscale;
}

__global__ void rope_tab_kernel(const int* __restrict__ pos, int n, int n_rot, int s0, int s1, int s2, int s3, float theta_scale,
                                float2* __restrict__ tab, const bool yarn, const RopeYarn yp) {
    const int half = n_rot / 2;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n * half) return;
    const int t = id / half, i = id % half;
    const int sect = s0 + s1 + s2 + s3;
    int c = 0;
    if (sect > 0) {
        const int s = i % sect;
        if (s % 3 == 1 && s < 3 * s1) c = 1;
        else if (s % 3 == 2 && s < 3 * s2) c = 2;
        else if (s % 3 == 0 && s < 3 * s0) c = 0;
        else c = 3;
    }
    const int p = c == 3 ? 0 : pos[t];
    const float th = float(p) * powf(theta_scale, float(i));
    float sn, cs;
    if (yarn) rope_yarn_cs(th, i, yp, &cs, &sn);
    else sincosf(th, &sn, &cs);
    tab[id] = make_float2(cs, sn);
}

// q・k の RMSNorm ＋ RoPE（表から）、q を fp16 [n][H][D] へ、k・v を fp16 の KV の slot へ。rope_norm_kernel<false> と同じ計算
// （sincosf を表に置き換えただけ）。1 warp = 1 つの (トークン, 頭)、lane は 8 要素
struct QkRope {
    const float* q;
    const float* k;
    const float* v;
    int64_t q_ld, k_ld, v_ld;
    const float* q_norm;
    const float* k_norm;
    const float2* tab;  // [n][n_rot/2]
    const int* slot;
    __half* q_out;
    __half* kc;
    __half* vc;
    int n, h, hkv, n_rot;
    float eps;
};
__global__ void __launch_bounds__(256) qk_rope_kernel(const QkRope a) {
    constexpr int D = 256, PS = 64;
    const int lane = threadIdx.x & 31;
    const int per = a.h + 2 * a.hkv;
    const int64_t item = int64_t(blockIdx.x) * 8 + (threadIdx.x >> 5);
    if (item >= int64_t(a.n) * per) return;
    const int t = int(item / per), j = int(item % per);
    int kind, hd;
    const float* src;
    if (j < a.h) kind = 0, hd = j, src = a.q + t * a.q_ld + int64_t(j) * 2 * D;
    else if (j < a.h + a.hkv) kind = 1, hd = j - a.h, src = a.k + t * a.k_ld + int64_t(hd) * D;
    else kind = 2, hd = j - a.h - a.hkv, src = a.v + t * a.v_ld + int64_t(hd) * D;
    float x[8];
    {
        const float4 u = reinterpret_cast<const float4*>(src)[lane * 2];
        const float4 w = reinterpret_cast<const float4*>(src)[lane * 2 + 1];
        x[0] = u.x, x[1] = u.y, x[2] = u.z, x[3] = u.w, x[4] = w.x, x[5] = w.y, x[6] = w.z, x[7] = w.w;
    }
    if (kind < 2) {
        float ss = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) ss += x[e] * x[e];
        ss = wsum(ss);
        const float r = 1.0f / sqrtf(ss / D + a.eps);
        const float* w = kind == 0 ? a.q_norm : a.k_norm;
#pragma unroll
        for (int e = 0; e < 8; ++e) x[e] = x[e] * r * w[lane * 8 + e];
        const int half = a.n_rot / 2, hl = half / 8;
        const bool rot = lane * 8 < a.n_rot;
        const int partner = !rot ? lane : (lane < hl ? lane + hl : lane - hl);
        float xp[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) xp[e] = __shfl_sync(0xffffffff, x[e], partner);
        if (rot) {
            const bool first = lane < hl;
            const float2* tb = a.tab + int64_t(t) * half + (first ? lane : lane - hl) * 8;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float2 cs = tb[e];
                x[e] = first ? x[e] * cs.x - xp[e] * cs.y : xp[e] * cs.y + x[e] * cs.x;
            }
        }
    }
    __half2 hv[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) hv[e] = __floats2half2_rn(x[2 * e], x[2 * e + 1]);
    if (kind == 0) {
        *reinterpret_cast<uint4*>(a.q_out + (int64_t(t) * a.h + hd) * D + lane * 8) = *reinterpret_cast<uint4*>(hv);
        return;
    }
    const int s = a.slot[t];
    if (s < 0) return;
    const int64_t rw = (int64_t(s / PS) * a.hkv + hd) * PS + s % PS;
    *reinterpret_cast<uint4*>((kind == 1 ? a.kc : a.vc) + rw * D + lane * 8) = *reinterpret_cast<uint4*>(hv);
}

// argmax を 2 段で（1 段目: 行ごとに AM_P ブロックが切れの 1 位、2 段目: 行ごとに切れの 1 位をまとめる）。
// 同じ値なら番号の小さい方・NaN は無視・全部 NaN なら -1（argmax_kernel と同じ答え）
constexpr int AM_P = 40;
__device__ __forceinline__ void am_merge(float& bv, int& bi, float v, int j) {
    if (v > bv || (v == bv && j < bi)) bv = v, bi = j;
}
__global__ void __launch_bounds__(256) argmax1_kernel(const float* __restrict__ x, int n, int64_t ld, float2* __restrict__ part) {
    __shared__ float sv[8];
    __shared__ int si[8];
    const float* xr = x + int64_t(blockIdx.y) * ld;
    const int per = (n + AM_P - 1) / AM_P, lo = blockIdx.x * per, hi = min(n, lo + per);
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = lo + threadIdx.x; i < hi; i += 256) {
        const float v = xr[i];
        if (v > bv) bv = v, bi = i;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) am_merge(bv, bi, __shfl_xor_sync(0xffffffffu, bv, o), __shfl_xor_sync(0xffffffffu, bi, o));
    if ((threadIdx.x & 31) == 0) sv[threadIdx.x >> 5] = bv, si[threadIdx.x >> 5] = bi;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 8; ++w) am_merge(bv, bi, sv[w], si[w]);
        part[int64_t(blockIdx.y) * AM_P + blockIdx.x] = make_float2(bv, __int_as_float(bi));
    }
}
__global__ void argmax2_kernel(const float2* __restrict__ part, int* __restrict__ out, float* __restrict__ out_v) {
    const int r = blockIdx.x, lane = threadIdx.x;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = lane; i < AM_P; i += 32) {
        const float2 p = part[int64_t(r) * AM_P + i];
        am_merge(bv, bi, p.x, __float_as_int(p.y));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) am_merge(bv, bi, __shfl_xor_sync(0xffffffffu, bv, o), __shfl_xor_sync(0xffffffffu, bi, o));
    if (lane == 0) {
        out[r] = bi == 0x7fffffff ? -1 : bi;
        if (out_v) out_v[r] = bv;
    }
}

// 有限でない値（NaN・inf）の数（確かめ用）
__global__ void nonfinite_kernel(const float* __restrict__ x, int64_t n, int* __restrict__ cnt) {
    int c = 0;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
        c += !isfinite(x[i]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
    if ((threadIdx.x & 31) == 0 && c) atomicAdd(cnt, c);
}

}  // namespace

#define M_EMBED_TYPES(X) X(Q_F32) X(Q_F16) X(Q_Q8_0) X(Q_Q4_K) X(Q_Q5_K) X(Q_Q6_K) X(Q_IQ4_NL) X(Q_IQ4_XS) X(Q_Q3_K)
#define M_G3_TYPES(X) X(Q_Q4_K) X(Q_Q5_K) X(Q_Q6_K) X(Q_Q8_0) X(Q_IQ4_NL) X(Q_IQ4_XS)

extern "C" {

// y[t][k] = token_embd[ids[t]]（w は GGUF の並び・型 qtype）。戻り値: 0 = 成功、-1 = 型が無い、-2 = k が 32 の倍数でない
int infer_m_embed(int qtype, const void* w, int k, const int* ids, int n_tok, float* y, cudaStream_t s) {
    if (k % 32) return -2;
    if (n_tok <= 0) return 0;
    const int64_t rb = qtype_row_bytes(qtype, k);
    const int64_t threads = int64_t(k / 32) * 4 * n_tok;
    const unsigned grid = unsigned((threads + 255) / 256);
    const uint8_t* wp = static_cast<const uint8_t*>(w);
    switch (qtype) {
#define CASE(T) case T: embed_kernel<T><<<grid, 256, 0, s>>>(wp, rb, k, ids, n_tok, y); break;
        M_EMBED_TYPES(CASE)
#undef CASE
        default: return -1;
    }
    return int(cudaGetLastError());
}

// 行 r（< n_rows）ごとに RMSNorm × w。y（f32、行の間隔 y_ld）・y16（f16、y16_ld）は要る方だけ（null は書かない）
int infer_m_rmsnorm(const float* x, int64_t x_ld, const float* w, int k, int n_rows, float eps, float* y, int64_t y_ld, void* y16,
                    int64_t y16_ld, cudaStream_t s) {
    if (n_rows <= 0) return 0;
    rmsnorm_kernel<<<n_rows, 256, 0, s>>>(x, x_ld, w, k, eps, y, y_ld, static_cast<__half*>(y16), y16_ld);
    return int(cudaGetLastError());
}

int infer_m_f32_to_f16(const float* x, int64_t n, void* y, cudaStream_t s) {
    if (n <= 0) return 0;
    f32_to_f16_kernel<<<unsigned((n + 255) / 256), 256, 0, s>>>(x, n, static_cast<__half*>(y));
    return int(cudaGetLastError());
}

// g3 の並び（qtype は詰め直した後の型）→ f16 [n][k]。y は 16 バイトに揃える
int infer_m_g3_deq16(int qtype, const void* w, int k, int n, void* y, cudaStream_t s) {
    if (packed_type(qtype) != qtype) return -1;
    if (k % UK) return -2;
    if (reinterpret_cast<uintptr_t>(y) & 15) return -5;
    const int64_t threads = int64_t((n + TILE - 1) / TILE) * (k / 32) * TILE;
    const unsigned grid = unsigned((threads + 255) / 256);
    const uint8_t* wp = static_cast<const uint8_t*>(w);
    __half* yp = static_cast<__half*>(y);
    switch (qtype) {
#define CASE(T) case T: g3_deq16_kernel<T><<<grid, 256, 0, s>>>(wp, k, n, yp); break;
        M_G3_TYPES(CASE)
#undef CASE
        default: return -1;
    }
    return int(cudaGetLastError());
}

// 行ごとの 1 位: out[r] = argmax_i x[r][i]（i < n、行の間隔 ld）。out_v（null 可）に値も
int infer_m_argmax(const float* x, int n, int64_t ld, int n_rows, int* out, float* out_v, cudaStream_t s) {
    if (n_rows <= 0) return 0;
    argmax_kernel<<<n_rows, 1024, 0, s>>>(x, n, ld, out, out_v);
    return int(cudaGetLastError());
}

// ---------------------------------------------------------------- デコード（M2）
// RMSNorm を g3 の x の下ごしらえ xp（infer_g3_xprep と同じ並び、B = n_rows）に直に。y（f32）は null なら書かない。k は 256 の倍数
int infer_m_rmsnorm_xp(const float* x, int64_t x_ld, const float* w, int k, int n_rows, float eps, float* y, int64_t y_ld, void* xp,
                       cudaStream_t s) {
    if (k % 256 || (x_ld % 4) || (y && y_ld % 4)) return -2;
    if (n_rows <= 0) return 0;
    __half* xhl = static_cast<__half*>(xp);
    float2* xsi = reinterpret_cast<float2*>(static_cast<uint8_t*>(xp) + int64_t(n_rows) * k * 4);
    // 試しの切り替え（TACHYON_RMS_NT=256 で前の幅）
    static const int nt = [] { const char* e = getenv("TACHYON_RMS_NT"); return e && atoi(e) == 256 ? 256 : 1024; }();
    if (nt == 256) rmsnorm_xp_kernel<256><<<n_rows, 256, 0, s>>>(x, x_ld, w, k, eps, y, y_ld, xhl, xsi);
    else rmsnorm_xp_kernel<1024><<<n_rows, 1024, 0, s>>>(x, x_ld, w, k, eps, y, y_ld, xhl, xsi);
    return int(cudaGetLastError());
}

// GDN の 1 歩（B 列 × 1 語）。c は conv（＋silu）の後・q/k は正規化前 [B][c_ld]、state [slot][Hv][128][128]（slots が null なら slot = b）。
// 出口は ssm_out の x の下ごしらえ xp（k = Hv·128、B 本）。y（f32）は null なら書かない
int infer_m_gdn_step(const float* c, int64_t c_ld, const float* z, int64_t z_ld, const float* alpha, const float* beta, int64_t ab_ld,
                     const float* ssm_a, const float* dt_bias, const float* norm_w, float* state, const int* slots, float* y, int64_t y_ld,
                     void* xp, int B, int Hk, int Hv, float eps, cudaStream_t s) {
    if (Hv % Hk || c_ld % 4 || z_ld % 4 || (y && y_ld % 4) || B < 1) return -2;
    const int K = Hv * 128;
    __half* xhl = static_cast<__half*>(xp);
    float2* xsi = reinterpret_cast<float2*>(static_cast<uint8_t*>(xp) + int64_t(B) * K * 4);
    gdn_step_kernel<<<dim3(Hv, B), GS_NT, 0, s>>>(c, c_ld, z, z_ld, alpha, beta, ab_ld, ssm_a, dt_bias, norm_w, state, slots, y, y_ld, xhl,
                                                  xsi, Hk, Hv, eps);
    return int(cudaGetLastError());
}

// RoPE の表 tab[n][n_rot/2]（float2 = cos, sin）。theta_scale は attn.cu と同じく CPU で powf(theta, -2/n_rot)
// yarn: CPU の float[5]（freq_scale, ext_factor, mscale, corr0, corr1）。null か (1, 0, 1, …) なら前と同じ式
int infer_m_rope_tab(const int* pos, int n, int n_rot, const int* sections, float theta, const float* yarn, void* tab, cudaStream_t s) {
    if (n_rot % 16 || n <= 0) return n <= 0 ? 0 : -2;
    const float theta_scale = powf(theta, -2.0f / float(n_rot));
    const int items = n * (n_rot / 2);
    RopeYarn yp{1.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    if (yarn) yp = RopeYarn{yarn[0], yarn[1], yarn[2], yarn[3], yarn[4]};
    const bool on = yp.freq_scale != 1.0f || yp.ext_factor != 0.0f || yp.mscale != 1.0f;
    rope_tab_kernel<<<(items + 127) / 128, 128, 0, s>>>(pos, n, n_rot, sections[0], sections[1], sections[2], sections[3], theta_scale,
                                                         static_cast<float2*>(tab), on, yp);
    return int(cudaGetLastError());
}

// q・k の正規化と RoPE（表から）・q の書き出し・fp16 の KV への書き込み（infer_attn_rope の pos_mode 1・kv8 0 と同じ答え）
int infer_m_qk_rope(const float* q, const float* k, const float* v, int64_t q_ld, int64_t k_ld, int64_t v_ld, const float* q_norm,
                    const float* k_norm, const void* tab, const int* slot, void* q_out, void* kc, void* vc, int n, int h, int hkv, int n_rot,
                    float eps, cudaStream_t s) {
    if (n_rot % 16 || n_rot > 256 || q_ld % 4 || k_ld % 4 || v_ld % 4) return -2;
    if (n <= 0) return 0;
    QkRope a{q, k, v, q_ld, k_ld, v_ld, q_norm, k_norm, static_cast<const float2*>(tab), slot, static_cast<__half*>(q_out),
             static_cast<__half*>(kc), static_cast<__half*>(vc), n, h, hkv, n_rot, eps};
    const int64_t items = int64_t(n) * (h + 2 * hkv);
    qk_rope_kernel<<<unsigned((items + 7) / 8), 256, 0, s>>>(a);
    return int(cudaGetLastError());
}

// argmax の 2 段版（infer_m_argmax と同じ答え）。part は infer_m_argmax2_ws_bytes(n_rows) の作業場
int64_t infer_m_argmax2_ws_bytes(int n_rows) { return int64_t(n_rows) * AM_P * 8; }
int infer_m_argmax2(const float* x, int n, int64_t ld, int n_rows, void* part, int* out, float* out_v, cudaStream_t s) {
    if (n_rows <= 0) return 0;
    argmax1_kernel<<<dim3(AM_P, n_rows), 256, 0, s>>>(x, n, ld, static_cast<float2*>(part));
    argmax2_kernel<<<n_rows, 32, 0, s>>>(static_cast<const float2*>(part), out, out_v);
    return int(cudaGetLastError());
}

// 有限でない値の数を *cnt に足す（cnt は GPU の上。先に 0 にしておく）
int infer_m_nonfinite(const float* x, int64_t n, int* cnt, cudaStream_t s) {
    if (n <= 0) return 0;
    const int64_t blocks = std::min<int64_t>((n + 255) / 256, 1024);
    nonfinite_kernel<<<unsigned(blocks), 256, 0, s>>>(x, n, cnt);
    return int(cudaGetLastError());
}

}  // extern "C"
