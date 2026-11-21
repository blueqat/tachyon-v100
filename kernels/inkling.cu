// SPDX-License-Identifier: Apache-2.0
// Inkling-2-Mini（arch inkling2_loop）の小さなカーネル。手本は transformers PR #51050 の modeling_inkling2_loop.py。
// 行列の掛け算（GEMV・MoE の GEMV）は kernels/gemv.cu（GGUF の並びのまま）を使い、ここはその間の要素ごとの計算だけ。
// どれも f32。T（行）は「同時に計算する語」の数（デコードでは列の数、1 語ずつのプレフィルでは 1）。
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
// ブロックで和（sh は warp の数 = blockDim/32 個以上。1024 スレッドまで）
__device__ __forceinline__ float block_sum(float v, float* sh) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < int(blockDim.x >> 5); ++i) t += sh[i];
    return t;
}

// fp16 を 4 つ（8 バイト・8 バイトにそろった所）読んで f32 に
__device__ __forceinline__ float4 ld4h(const __half* p) {
    const uint2 u = *reinterpret_cast<const uint2*>(p);
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
    return make_float4(a.x, a.y, b.x, b.y);
}

// KV の置き場（fp16 か int8＋128 要素ごとの係数）。要素の番号 e（同じ並び）で引く
struct KvRef {
    const void* k;
    const void* v;
    const float* ks;  // int8 のときの係数 [e / 128]
    const float* vs;
    int kv8;
};
__device__ __forceinline__ float4 ld4_8(const int8_t* p, float sc) {
    const char4 c = *reinterpret_cast<const char4*>(p);
    return make_float4(c.x * sc, c.y * sc, c.z * sc, c.w * sc);
}
__device__ __forceinline__ float4 ld4k(const KvRef& r, int64_t e) {
    return r.kv8 ? ld4_8(static_cast<const int8_t*>(r.k) + e, r.ks[e >> 7]) : ld4h(static_cast<const __half*>(r.k) + e);
}
__device__ __forceinline__ float4 ld4v(const KvRef& r, int64_t e) {
    return r.kv8 ? ld4_8(static_cast<const int8_t*>(r.v) + e, r.vs[e >> 7]) : ld4h(static_cast<const __half*>(r.v) + e);
}

// ---------------------------------------------------------------- RMSNorm（重みは null なら無し）
__global__ void __launch_bounds__(1024) rms_kernel(const float* __restrict__ x, int64_t xld, const float* __restrict__ w, int k, float eps,
                                                  float* __restrict__ y, int64_t yld) {
    __shared__ float sh[32];
    const float* xr = x + blockIdx.x * xld;
    float s = 0.f;
    for (int i = threadIdx.x; i < k; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / k + eps);
    float* yr = y + blockIdx.x * yld;
    for (int i = threadIdx.x; i < k; i += blockDim.x) yr[i] = xr[i] * r * (w ? w[i] : 1.f);
}

// ---------------------------------------------------------------- 深さの Attention（block attention residual / message）
// 状態 s = 0..S-1（ptr[s] + row * ld）の RMSNorm（重みなし・eps 1e-6）を K = V に、q（[nq][d]）で softmax(q·s/√d) の重みで混ぜる。
// 出口 y[row][j][d]（j = 0..nq-1）。ブロック = (row, j)
constexpr int MIX_MAX = 16;
struct MixArgs {
    const float* ptr[MIX_MAX];
    int64_t ld[MIX_MAX];
    int S;
};
__global__ void __launch_bounds__(1024) mix_kernel(const MixArgs a, const float* __restrict__ q, int nq, int d, float* __restrict__ y, int64_t yld) {
    // 1 回の読みで全部の状態の 2 乗の和と q との内積を貯め、まとめて木で足す
    __shared__ float red[32][2 * MIX_MAX];
    __shared__ float p[MIX_MAX];
    const int row = blockIdx.x, j = blockIdx.y;
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    const float* qj = q + int64_t(j) * d;
    float ss[MIX_MAX], qs[MIX_MAX];
#pragma unroll
    for (int s = 0; s < MIX_MAX; ++s) ss[s] = qs[s] = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float qv = qj[i];
#pragma unroll
        for (int s = 0; s < MIX_MAX; ++s)
            if (s < a.S) {
                const float v = a.ptr[s][row * a.ld[s] + i];
                ss[s] += v * v;
                qs[s] += qv * v;
            }
    }
#pragma unroll
    for (int s = 0; s < MIX_MAX; ++s)
        if (s < a.S) {
            const float u = warp_sum(ss[s]), t = warp_sum(qs[s]);
            if (l == 0) {
                red[w][2 * s] = u;
                red[w][2 * s + 1] = t;
            }
        }
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -INFINITY, inv[MIX_MAX];
        for (int s = 0; s < a.S; ++s) {
            float u = 0.f, t = 0.f;
            for (int k = 0; k < int(blockDim.x >> 5); ++k) {
                u += red[k][2 * s];
                t += red[k][2 * s + 1];
            }
            inv[s] = rsqrtf(u / d + 1e-6f);
            p[s] = t * inv[s] * rsqrtf(float(d));
            m = fmaxf(m, p[s]);
        }
        float z = 0.f;
        for (int s = 0; s < a.S; ++s) z += (p[s] = __expf(p[s] - m));
        for (int s = 0; s < a.S; ++s) p[s] = p[s] / z * inv[s];
    }
    __syncthreads();
    float* yo = y + row * yld + int64_t(j) * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = 0.f;
        for (int s = 0; s < a.S; ++s) v += p[s] * a.ptr[s][row * a.ld[s] + i];
        yo[i] = v;
    }
}

// ---------------------------------------------------------------- 次の回への伝言の門（message gate）
// z[row] = [rms(h); rms(m)]（重みなし・eps は rms_norm_eps）
__global__ void __launch_bounds__(1024) cat2_rms_kernel(const float* __restrict__ h, int64_t hld, const float* __restrict__ m, int64_t mld, int d,
                                                       float eps, float* __restrict__ z) {
    __shared__ float sh[32];
    const float* src = blockIdx.y == 0 ? h + blockIdx.x * hld : m + blockIdx.x * mld;
    float s = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) s += src[i] * src[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / d + eps);
    float* zo = z + int64_t(blockIdx.x) * 2 * d + int64_t(blockIdx.y) * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) zo[i] = src[i] * r;
}
// y = (1 - tanh(f)) * h + tanh(g) * m
__global__ void ctrl_kernel(const float* __restrict__ h, int64_t hld, const float* __restrict__ m, int64_t mld, const float* __restrict__ g,
                            const float* __restrict__ f, int d, int T, float* __restrict__ y, int64_t yld) {
    const int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= int64_t(T) * d) return;
    const int r = int(t / d), i = int(t % d);
    const float hv = h[r * hld + i], mv = m[r * mld + i];
    y[r * yld + i] = (1.f - tanhf(f[t])) * hv + tanhf(g[t]) * mv;
}

// ---------------------------------------------------------------- 短い畳み込み（残差つき）と状態
// x [T][ch]（畳み込みの前）、st [T の列ごと][nc-1][ch]（古い順。列 r の状態は st + r * st_ld）、w [ch][nc]（ggml の ssm_conv の並び:
// w[c][j] は x[t - (nc-1) + j] に掛かる）。dyn [T][heads][nc]（null でなければ、頭ごとの「何語前か」の重みを足す）。
// y = (x + conv) * scale。状態は 1 語ずらす（x を最後に）。y と x は同じでもよい
__global__ void conv_kernel(const float* x, int64_t xld, float* __restrict__ st, int64_t st_ld, const int* __restrict__ lanes, const float* __restrict__ w,
                            int nc, const float* __restrict__ dyn, int64_t dld, int heads, int ch, int T, float scale, float* y, int64_t yld) {
    const int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= int64_t(T) * ch) return;
    const int r = int(t / ch), c = int(t % ch);
    float* s = st + (lanes ? lanes[r] : r) * st_ld;
    float win[8];
    for (int j = 0; j < nc - 1; ++j) win[j] = s[int64_t(j) * ch + c];
    const float xv = x[r * xld + c];
    win[nc - 1] = xv;
    float acc = 0.f;
    for (int j = 0; j < nc; ++j) acc += w[int64_t(c) * nc + j] * win[j];
    if (dyn) {
        const int h = c / (ch / heads);
        const float* dh = dyn + r * dld + h * nc;
        for (int lag = 0; lag < nc; ++lag) acc += dh[lag] * win[nc - 1 - lag];
    }
    for (int j = 0; j < nc - 2; ++j) s[int64_t(j) * ch + c] = win[j + 1];
    s[int64_t(nc - 2) * ch + c] = xv;
    y[r * yld + c] = (xv + acc) * scale;
}

// ---------------------------------------------------------------- q・k・v・r の下ごしらえ
// 頭ごとのベクトル（長さ dh）を: scale（null でなければ [heads][dh]）を掛け、L2 を thr 以下に（thr <= 0 なら RMSNorm・重みなし）、
// 最後に tau（語ごと・null なら 1）を掛ける。ブロック = (row, head)、32 スレッド
__global__ void headprep_kernel(float* __restrict__ x, int64_t xld, int dh, const float* __restrict__ scale, float thr, float eps,
                                const float* __restrict__ tau) {
    const int row = blockIdx.x, h = blockIdx.y, l = threadIdx.x;
    float* v = x + row * xld + int64_t(h) * dh;
    float buf[8];
    float ss = 0.f;
    const int per = (dh + 31) / 32;
    for (int u = 0; u < per; ++u) {
        const int i = l + 32 * u;
        float a = i < dh ? v[i] : 0.f;
        if (scale && i < dh) a *= scale[int64_t(h) * dh + i];
        buf[u] = a;
        ss += a * a;
    }
    ss = warp_sum(ss);
    float f;
    if (thr > 0.f) {
        f = fminf(thr / (sqrtf(ss) + 1e-6f), 1.f);
    } else {
        f = rsqrtf(ss / dh + eps);
    }
    if (tau) f *= tau[row];
    for (int u = 0; u < per; ++u) {
        const int i = l + 32 * u;
        if (i < dh) v[i] = buf[u] * f;
    }
}

// ---------------------------------------------------------------- デコードの Attention（相対位置の偏り・語ごとの sink）
// q [T][H][128]、r [T][H][nrel]（下ごしらえの後）、sink [T][H]、表 rb [extent][nrel]（ggml の [nrel, extent]）。
// 鍵と値は列ごとの置き場 kc/vc（列 r は + r * kv_ld、置き場の i 番 = [kvh][128] fp16）。置き場の i 番の語の位置は
//   ring = 0（全部）: i、ring = 1（窓）: pos - ((pos - i) mod cap)（cap = 窓の大きさ。i が今の語までに書かれた所だけ）
// 偏り = Σ_c r[c] * rb[dist][c]（0 <= dist < extent、外は 0）。logit = q·k / √128 + 偏り。sink の logit は sink[h]（値は 0）。
// 1 段目: ブロック = (row, head, split)、各 split は鍵の範囲 [s·chunk, …) を 8 warp で回す（warp ごとに online softmax）
constexpr int AT_NT = 256, AT_W = AT_NT / 32;
// 作業場の 1 つ = m, l, (空き 2), acc[128]（acc を 16 バイトにそろえる）
constexpr int PART_W = 4 + 128;
struct AttnDec {
    const float* q; int64_t qld;
    const float* r; int64_t rld;
    const float* sink; int64_t sld;
    const float* rb;
    KvRef kv; int64_t kv_ld; const int* lanes;
    const int* pos;     // [T] 今の語の位置
    int H, KVH, nrel, extent, cap, ring, window, chunk, n_split;
    float* part;        // [T][H][n_split][PART_W]（m, l, 空き, acc）
};
__global__ void __launch_bounds__(AT_NT) attn_dec_kernel(const AttnDec a) {
    const int row = blockIdx.x, h = blockIdx.y, sp = blockIdx.z;
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    const int p = a.pos[row];
    // 見る鍵の位置は [lo, p]（窓の層は lo = p - window + 1）。置き場の番号は ring なら 位置 mod cap
    const int lo = a.window > 0 ? max(0, p - a.window + 1) : 0;
    const int n_valid = p + 1 - lo;
    const int i0 = sp * a.chunk, i1 = min(n_valid, i0 + a.chunk);
    const int kvh = h / (a.H / a.KVH);
    const float* qh = a.q + row * a.qld + int64_t(h) * 128;
    const float4 qv = reinterpret_cast<const float4*>(qh)[l];
    const float* rh = a.r + row * a.rld + int64_t(h) * a.nrel;
    float rr = l < a.nrel ? rh[l] : 0.f;
    const int64_t lane_off = int64_t(a.lanes ? a.lanes[row] : row) * a.kv_ld;
    const float sc = rsqrtf(128.f);
    float m = -INFINITY, z = 0.f;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int i = i0 + w; i < i1; i += AT_W) {
        const int kp = lo + i;
        const int slot = a.ring ? kp % a.cap : kp;
        const int dist = p - kp;
        const int64_t e = lane_off + (int64_t(slot) * a.KVH + kvh) * 128 + l * 4;
        const float4 kv = ld4k(a.kv, e);
        float s = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        float b = 0.f;
        if (dist < a.extent && l < a.nrel) b = rr * a.rb[int64_t(dist) * a.nrel + l];
        s = warp_sum(s) * sc + warp_sum(b);
        const float mn = fmaxf(m, s);
        const float c0 = __expf(m - mn), c1 = __expf(s - mn);
        const float4 vv = ld4v(a.kv, e);
        acc.x = acc.x * c0 + c1 * vv.x;
        acc.y = acc.y * c0 + c1 * vv.y;
        acc.z = acc.z * c0 + c1 * vv.z;
        acc.w = acc.w * c0 + c1 * vv.w;
        z = z * c0 + c1;
        m = mn;
    }
    // warp の間で合わせる
    __shared__ float sm[AT_W], sz[AT_W];
    __shared__ float4 sacc[AT_W][32];
    if (l == 0) {
        sm[w] = m;
        sz[w] = z;
    }
    sacc[w][l] = acc;
    __syncthreads();
    if (w == 0) {
        float M = -INFINITY;
        for (int i = 0; i < AT_W; ++i) M = fmaxf(M, sm[i]);
        float Z = 0.f;
        float4 A = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int i = 0; i < AT_W; ++i) {
            if (sm[i] == -INFINITY) continue;
            const float c = __expf(sm[i] - M);
            Z += sz[i] * c;
            const float4 v = sacc[i][l];
            A.x += v.x * c;
            A.y += v.y * c;
            A.z += v.z * c;
            A.w += v.w * c;
        }
        float* out = a.part + ((int64_t(row) * a.H + h) * a.n_split + sp) * PART_W;
        if (l == 0) {
            out[0] = M;
            out[1] = Z;
        }
        reinterpret_cast<float4*>(out + 4)[l] = A;
    }
}
// 2 段目: split を合わせ、sink を足して y[row][h][128] に。ブロック = (row, head)、32 スレッド
__global__ void attn_combine_kernel(const AttnDec a, float* __restrict__ y, int64_t yld) {
    const int row = blockIdx.x, h = blockIdx.y, l = threadIdx.x;
    const float* part = a.part + (int64_t(row) * a.H + h) * a.n_split * PART_W;
    const float t = a.sink[row * a.sld + h];
    float M = t;
    for (int s = 0; s < a.n_split; ++s) M = fmaxf(M, part[s * PART_W]);
    float Z = __expf(t - M);
    float4 A = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int s = 0; s < a.n_split; ++s) {
        const float* ps = part + s * PART_W;
        if (ps[0] == -INFINITY) continue;
        const float c = __expf(ps[0] - M);
        Z += ps[1] * c;
        const float4 v = reinterpret_cast<const float4*>(ps + 4)[l];
        A.x += v.x * c;
        A.y += v.y * c;
        A.z += v.z * c;
        A.w += v.w * c;
    }
    const float iz = 1.f / Z;
    float4* yo = reinterpret_cast<float4*>(y + row * yld + int64_t(h) * 128);
    yo[l] = make_float4(A.x * iz, A.y * iz, A.z * iz, A.w * iz);
}

// ---------------------------------------------------------------- 要素ごと
// o *= sigmoid(g)
__global__ void gate_kernel(float* __restrict__ o, int64_t old, const float* __restrict__ g, int64_t gld, int n, int T) {
    const int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= int64_t(T) * n) return;
    const int r = int(t / n), i = int(t % n);
    o[r * old + i] *= 1.f / (1.f + __expf(-g[r * gld + i]));
}
// y = a + b
__global__ void add_kernel(const float* a, int64_t ald, const float* b, int64_t bld, int n, int T, float* y, int64_t yld) {
    const int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= int64_t(T) * n) return;
    const int r = int(t / n), i = int(t % n);
    y[r * yld + i] = a[r * ald + i] + b[r * bld + i];
}
// 語ごとの tau = 1 + alpha·log(max(1, (pos + 1) / floor))
__global__ void tau_kernel(const int* __restrict__ pos, int T, float alpha, float floor_, float* __restrict__ tau) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < T) tau[r] = 1.f + alpha * logf(fmaxf(1.f, float(pos[r] + 1) / floor_));
}

// ---------------------------------------------------------------- MoE
// 振り分け: s = sigmoid(logit)、上位 k は s + bias で選び、重みは s / Σs × scale。ブロック = 行（ne <= 256）
__global__ void __launch_bounds__(256) route_kernel(const float* __restrict__ lg, int64_t ld, const float* __restrict__ bias, int ne, int kk,
                                                    float scale, int* __restrict__ ids, float* __restrict__ wts) {
    __shared__ float s[256], key[256];
    const int row = blockIdx.x, i = threadIdx.x;
    if (i < ne) {
        s[i] = 1.f / (1.f + __expf(-lg[row * ld + i]));
        key[i] = s[i] + (bias ? bias[i] : 0.f);
    }
    __syncthreads();
    if (i == 0) {
        float sum = 0.f;
        int sel[32];
        for (int j = 0; j < kk; ++j) {
            int b = -1;
            float bv = -INFINITY;
            for (int e = 0; e < ne; ++e)
                if (key[e] > bv) {
                    bv = key[e];
                    b = e;
                }
            sel[j] = b;
            key[b] = -INFINITY;
            sum += s[b];
        }
        for (int j = 0; j < kk; ++j) {
            ids[row * kk + j] = sel[j];
            wts[row * kk + j] = s[sel[j]] / (sum + 1e-20f) * scale;
        }
    }
}
// 柔らかく抑えた SwiGLU: h = cap_g(g)·sigmoid(g)·cap_u(u)、cap_c(v) = tanh(v / c)·c。gu [R][2·ff]（前半 gate・後半 up）→ h [R][ff]
__global__ void situ_kernel(const float* __restrict__ gu, int ff, int R, float cg, float cu, float* __restrict__ h, int64_t hld) {
    const int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= int64_t(R) * ff) return;
    const int r = int(t / ff), i = int(t % ff);
    const float g = gu[int64_t(r) * 2 * ff + i], u = gu[int64_t(r) * 2 * ff + ff + i];
    h[r * hld + i] = tanhf(g / cg) * cg * (1.f / (1.f + __expf(-g))) * tanhf(u / cu) * cu;
}
// y[row] = Σ_j wts[row][j] · o[row·kk + j]
__global__ void moe_sum_kernel(const float* __restrict__ o, const float* __restrict__ wts, int d, int kk, int T, float* __restrict__ y, int64_t yld) {
    const int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= int64_t(T) * d) return;
    const int r = int(t / d), i = int(t % d);
    float v = 0.f;
    for (int j = 0; j < kk; ++j) v += wts[r * kk + j] * o[(int64_t(r) * kk + j) * d + i];
    y[r * yld + i] = v;
}
// 出口: logits = tanh(x / cap)·cap（cap > 0 のとき）。x [T][ld] の先頭 n だけ
__global__ void softcap_kernel(float* __restrict__ x, int64_t ld, int n, int T, float cap) {
    const int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= int64_t(T) * n) return;
    const int r = int(t / n), i = int(t % n);
    float& v = x[r * ld + i];
    v = tanhf(v / cap) * cap;
}

// ---------------------------------------------------------------- 小さな f32 の行列（ルーター・伝言の門の down）
// y[t][n] = W[n][k] · x[t][k]（k は 4 の倍数）。1 ブロック（256）= 1 行（行の数が少ない＝ルーター 128・down 64 なので、k を分けて読む）
__global__ void __launch_bounds__(256) gemv_f32_kernel(const float* __restrict__ W, int k, int n, const float* __restrict__ x, int64_t xld, int T,
                                                       float* __restrict__ y, int64_t yld) {
    __shared__ float sh[8][8];
    const int r = blockIdx.x, w = threadIdx.x >> 5, l = threadIdx.x & 31;
    const float4* wr = reinterpret_cast<const float4*>(W + int64_t(r) * k);
    float acc[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) acc[t] = 0.f;
    for (int i = threadIdx.x; i < k / 4; i += blockDim.x) {
        const float4 wv = wr[i];
#pragma unroll
        for (int t = 0; t < 8; ++t)
            if (t < T) {
                const float4 xv = reinterpret_cast<const float4*>(x + t * xld)[i];
                acc[t] += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
            }
    }
#pragma unroll
    for (int t = 0; t < 8; ++t)
        if (t < T) {
            const float v = warp_sum(acc[t]);
            if (l == 0) sh[t][w] = v;
        }
    __syncthreads();
    if (threadIdx.x < T) {
        float v = 0.f;
        for (int i = 0; i < 8; ++i) v += sh[threadIdx.x][i];
        y[threadIdx.x * yld + r] = v;
    }
}
// 伝言の門の後半を 1 つに: 行 i ごとに g = G[i]·c、f = F[i]·c（c は [T][rank]）、y = (1 - tanh f)·h + tanh g·m。1 warp = 1 行
__global__ void __launch_bounds__(256) ctrl2_kernel(const float* __restrict__ G, const float* __restrict__ F, int rank, const float* __restrict__ c,
                                                    const float* h, int64_t hld, const float* __restrict__ m, int64_t mld, int d, int T, float* y,
                                                    int64_t yld) {
    const int i = blockIdx.x * 8 + (threadIdx.x >> 5), l = threadIdx.x & 31;
    if (i >= d) return;
    for (int t = 0; t < T; ++t) {
        float g = 0.f, f = 0.f;
        for (int r = l; r < rank; r += 32) {
            const float cv = c[t * rank + r];
            g += G[int64_t(i) * rank + r] * cv;
            f += F[int64_t(i) * rank + r] * cv;
        }
        g = warp_sum(g);
        f = warp_sum(f);
        if (l == 0) {
            const float hv = h[t * hld + i], mv = m[t * mld + i];
            y[t * yld + i] = (1.f - tanhf(f)) * hv + tanhf(g) * mv;
        }
    }
}
// 振り分けを 1 warp で（ne <= 256・kk <= 32）
__global__ void route_warp_kernel(const float* __restrict__ lg, int64_t ld, const float* __restrict__ bias, int ne, int kk, float scale,
                                  int* __restrict__ ids, float* __restrict__ wts) {
    const int row = blockIdx.x, l = threadIdx.x;
    float s[8], key[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const int e = l + 32 * u;
        s[u] = e < ne ? 1.f / (1.f + __expf(-lg[row * ld + e])) : 0.f;
        key[u] = e < ne ? s[u] + (bias ? bias[e] : 0.f) : -INFINITY;
    }
    float sum = 0.f, mine = 0.f;
    int my_e = -1;
    for (int j = 0; j < kk; ++j) {
        float bv = -INFINITY;
        int be = 0x7fffffff, bu = 0;
#pragma unroll
        for (int u = 0; u < 8; ++u)
            if (key[u] > bv) {
                bv = key[u];
                be = l + 32 * u;
                bu = u;
            }
        // warp の最大（同点は番号の小さい方）
        float v = bv;
        int e = be;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float v2 = __shfl_xor_sync(0xffffffff, v, o);
            const int e2 = __shfl_xor_sync(0xffffffff, e, o);
            if (v2 > v || (v2 == v && e2 < e)) {
                v = v2;
                e = e2;
            }
        }
        float sv = 0.f;
        if (e == be) {
            sv = s[bu];
            key[bu] = -INFINITY;
        }
        sv = warp_sum(sv);
        sum += sv;
        if (l == j) {
            my_e = e;
            mine = sv;
        }
    }
    if (l < kk) {
        ids[row * kk + l] = my_e;
        wts[row * kk + l] = mine / (sum + 1e-20f) * scale;
    }
}

// ---------------------------------------------------------------- プレフィル（1 本の列・T 語をまとめて）
// 畳み込み: y[t] = (x[t] + Σ_j w[c][j]·x[t-(nc-1)+j] (+ 語ごとの重み)) · scale。t < nc-1 の前の語は状態 st（[nc-1][ch]、古い順）から。
// y と x は別の置き場（x は後の語の窓でも読むので）。状態は ink_conv_pf_state で後から更新する
__global__ void conv_pf_kernel(const float* __restrict__ x, int64_t xld, const float* __restrict__ st, const float* __restrict__ w, int nc,
                               const float* __restrict__ dyn, int64_t dld, int heads, int ch, int T, float scale, float* __restrict__ y, int64_t yld) {
    const int64_t id = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= int64_t(T) * ch) return;
    const int t = int(id / ch), c = int(id % ch);
    float win[8];
    for (int j = 0; j < nc; ++j) {
        const int src = t - (nc - 1) + j;
        win[j] = src >= 0 ? x[src * xld + c] : st[int64_t(src + nc - 1) * ch + c];
    }
    float acc = 0.f;
    for (int j = 0; j < nc; ++j) acc += w[int64_t(c) * nc + j] * win[j];
    if (dyn) {
        const int h = c / (ch / heads);
        const float* dh = dyn + t * dld + h * nc;
        for (int lag = 0; lag < nc; ++lag) acc += dh[lag] * win[nc - 1 - lag];
    }
    y[t * yld + c] = (win[nc - 1] + acc) * scale;
}
// 状態 = 最後の nc-1 語の入口（T < nc-1 なら古い状態をずらして足す）
__global__ void conv_pf_state_kernel(const float* __restrict__ x, int64_t xld, float* __restrict__ st, int nc, int ch, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    float old[8];
    for (int j = 0; j < nc - 1; ++j) old[j] = st[int64_t(j) * ch + c];
    for (int j = 0; j < nc - 1; ++j) {
        const int src = T - (nc - 1) + j;  // 新しい状態の j 番 = 位置 T-(nc-1)+j の入口
        st[int64_t(j) * ch + c] = src >= 0 ? x[src * xld + c] : old[src + nc - 1];
    }
}

// Attention（プレフィル）: 問い T 語（位置 pos[t]・昇順に続く）× 置き場の鍵。ブロック = (問いの 8 語, KV の頭 1 つ) × 同じ KV を見る問いの頭 G
// （= 8·G 本の問い）。鍵 32 語ずつ共有メモリに。1 warp = 問い PF_QW 本、lane = 鍵 1 つ（内積は lane の中で 128 次元）
constexpr int PF_TQ = 8, PF_TK = 32, PF_GMAX = 3;  // 共有メモリ 48KB に収める（Inkling は G = 3）
struct AttnPf {
    const float* q; int64_t qld;     // [T][H][128]
    const float* r; int64_t rld;     // [T][H][nrel]
    const float* sink; int64_t sld;  // [T][H]
    const float* rb;
    KvRef kv;
    const int* pos;
    int T, H, KVH, nrel, extent, cap, ring, window;
    float* y; int64_t yld;
};
__global__ void __launch_bounds__(256) attn_pf_kernel(const AttnPf a) {
    const int G = a.H / a.KVH;
    const int nv = PF_TQ * G;  // 問いの本数（<= 32）
    const int t0 = blockIdx.x * PF_TQ, kvh = blockIdx.y;
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __shared__ float Qs[PF_TQ * PF_GMAX][128];
    __shared__ float Rs[PF_TQ * PF_GMAX][32];
    __shared__ float Ks[PF_TK][129];
    __shared__ __align__(16) float Vs[PF_TK][128];
    __shared__ int qpos[PF_TQ * PF_GMAX];
    // 問い v = qi·G + g（語 t0+qi・頭 kvh·G+g）
    for (int i = threadIdx.x; i < nv * 128; i += blockDim.x) {
        const int v = i / 128, dd = i % 128, qi = v / G, g = v % G, t = t0 + qi;
        Qs[v][dd] = t < a.T ? a.q[t * a.qld + int64_t(kvh * G + g) * 128 + dd] * rsqrtf(128.f) : 0.f;
    }
    for (int i = threadIdx.x; i < nv * 32; i += blockDim.x) {
        const int v = i / 32, c = i % 32, qi = v / G, g = v % G, t = t0 + qi;
        Rs[v][c] = (t < a.T && c < a.nrel) ? a.r[t * a.rld + int64_t(kvh * G + g) * a.nrel + c] : 0.f;
    }
    if (threadIdx.x < nv) {
        const int qi = threadIdx.x / G, t = t0 + qi;
        qpos[threadIdx.x] = t < a.T ? a.pos[t] : -1;
    }
    __syncthreads();
    const int tl = min(a.T, t0 + PF_TQ) - 1;
    if (t0 >= a.T) return;
    const int p_first = a.pos[t0], p_last = a.pos[tl];
    const int lo = a.window > 0 ? max(0, p_first - a.window + 1) : 0;
    // warp ごとの問い（nv を 8 warp に割る）
    constexpr int QW = (PF_TQ * PF_GMAX + 7) / 8;
    float m[QW], z[QW];
    float4 o[QW];
#pragma unroll
    for (int u = 0; u < QW; ++u) {
        m[u] = -INFINITY;
        z[u] = 0.f;
        o[u] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    for (int k0 = lo; k0 <= p_last; k0 += PF_TK) {
        __syncthreads();
        for (int i = threadIdx.x; i < PF_TK * 32; i += blockDim.x) {
            const int j = i / 32, q4 = i % 32, kp = k0 + j;
            float4 kv = make_float4(0.f, 0.f, 0.f, 0.f), vv = kv;
            if (kp <= p_last) {
                const int slot = a.ring ? kp % a.cap : kp;
                const int64_t e = (int64_t(slot) * a.KVH + kvh) * 128 + q4 * 4;
                kv = ld4k(a.kv, e);
                vv = ld4v(a.kv, e);
            }
            Ks[j][q4 * 4 + 0] = kv.x;
            Ks[j][q4 * 4 + 1] = kv.y;
            Ks[j][q4 * 4 + 2] = kv.z;
            Ks[j][q4 * 4 + 3] = kv.w;
            reinterpret_cast<float4*>(Vs[j])[q4] = vv;
        }
        __syncthreads();
        const int kp = k0 + l;
#pragma unroll
        for (int u = 0; u < QW; ++u) {
            const int v = w * QW + u;
            if (v >= nv) break;
            const int pq = qpos[v];
            const int dist = pq - kp;
            const bool ok = pq >= 0 && dist >= 0 && (a.window <= 0 || dist < a.window);
            float s = -INFINITY;
            if (ok) {
                float acc = 0.f;
#pragma unroll 16
                for (int dd = 0; dd < 128; ++dd) acc += Qs[v][dd] * Ks[l][dd];
                if (dist < a.extent) {
                    const float* rbd = a.rb + int64_t(dist) * a.nrel;
                    for (int c = 0; c < a.nrel; ++c) acc += Rs[v][c] * rbd[c];
                }
                s = acc;
            }
            const float mx = warp_max(s);
            if (mx == -INFINITY) continue;  // この問いにはこの 32 語に見る鍵が無い
            const float mn = fmaxf(m[u], mx);
            const float c0 = __expf(m[u] - mn);
            const float pl = ok ? __expf(s - mn) : 0.f;
            z[u] = z[u] * c0 + warp_sum(pl);
            float4 acc4 = make_float4(o[u].x * c0, o[u].y * c0, o[u].z * c0, o[u].w * c0);
#pragma unroll 8
            for (int j = 0; j < PF_TK; ++j) {
                const float pj = __shfl_sync(0xffffffff, pl, j);
                const float4 vv = reinterpret_cast<const float4*>(Vs[j])[l];
                acc4.x += pj * vv.x;
                acc4.y += pj * vv.y;
                acc4.z += pj * vv.z;
                acc4.w += pj * vv.w;
            }
            o[u] = acc4;
            m[u] = mn;
        }
    }
    // sink を足して書く
#pragma unroll
    for (int u = 0; u < QW; ++u) {
        const int v = w * QW + u;
        if (v >= nv) break;
        const int qi = v / G, g = v % G, t = t0 + qi;
        if (t >= a.T) continue;
        const int h = kvh * G + g;
        const float sk = a.sink[t * a.sld + h];
        const float M = fmaxf(m[u], sk);
        const float c0 = __expf(m[u] - M);
        const float Z = z[u] * c0 + __expf(sk - M);
        const float iz = c0 / Z;
        reinterpret_cast<float4*>(a.y + t * a.yld + int64_t(h) * 128)[l] = make_float4(o[u].x * iz, o[u].y * iz, o[u].z * iz, o[u].w * iz);
    }
}

// KV の置き場に T 語ぶん書く（位置 pos[t]）
// KV に書く。ブロック = (語 t, 頭 h)・128 スレッド。kv_ld: 行ごとの置き場の間隔（要素。lanes があれば行 t は lanes[t] の列）
// int8 のときは頭のベクトル 128 ごとに absmax/127 の係数
struct KvOut {
    void* k;
    void* v;
    float* ks;
    float* vs;
    int kv8;
};
__global__ void __launch_bounds__(128) kvwrite_pf_kernel(const float* __restrict__ k, int64_t kld, const float* __restrict__ v, int64_t vld, int KVH, int T,
                                                         KvOut o, int64_t kv_ld, const int* __restrict__ lanes, const int* __restrict__ pos, int cap,
                                                         int ring) {
    __shared__ float mx[2][4];
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    const int p = pos[t];
    const int slot = ring ? p % cap : p;
    const int64_t e = int64_t(lanes ? lanes[t] : t) * kv_ld + (int64_t(slot) * KVH + h) * 128 + i;
    const float kv = k[t * kld + h * 128 + i], vv = v[t * vld + h * 128 + i];
    if (!o.kv8) {
        static_cast<__half*>(o.k)[e] = __float2half(kv);
        static_cast<__half*>(o.v)[e] = __float2half(vv);
        return;
    }
    float a = fabsf(kv), b = fabsf(vv);
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, s));
        b = fmaxf(b, __shfl_xor_sync(0xffffffff, b, s));
    }
    if ((i & 31) == 0) {
        mx[0][i >> 5] = a;
        mx[1][i >> 5] = b;
    }
    __syncthreads();
    const float ma = fmaxf(fmaxf(mx[0][0], mx[0][1]), fmaxf(mx[0][2], mx[0][3]));
    const float mb = fmaxf(fmaxf(mx[1][0], mx[1][1]), fmaxf(mx[1][2], mx[1][3]));
    const float sa = ma > 0.f ? ma / 127.f : 1.f, sb = mb > 0.f ? mb / 127.f : 1.f;
    static_cast<int8_t*>(o.k)[e] = int8_t(__float2int_rn(kv / sa));
    static_cast<int8_t*>(o.v)[e] = int8_t(__float2int_rn(vv / sb));
    if (i == 0) {
        o.ks[e >> 7] = sa;
        o.vs[e >> 7] = sb;
    }
}
// MoE の足し合わせ（専門家ごとに並べ替えた行から）: y[t] = Σ_j wts[t][j] · o[rowmap[t·kk + j]]
__global__ void moe_sum_map_kernel(const float* __restrict__ o, const int* __restrict__ rowmap, const float* __restrict__ wts, int d, int kk, int T,
                                   float* __restrict__ y, int64_t yld) {
    const int64_t id = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= int64_t(T) * d) return;
    const int t = int(id / d), i = int(id % d);
    float v = 0.f;
    for (int j = 0; j < kk; ++j) v += wts[t * kk + j] * o[int64_t(rowmap[t * kk + j]) * d + i];
    y[t * yld + i] = v;
}
// 小さな f32 の行列を T 語に（ブロック = (行, 語 8 つ)）
__global__ void __launch_bounds__(256) gemm_f32_kernel(const float* __restrict__ W, int k, int n, const float* __restrict__ x, int64_t xld, int T,
                                                       float* __restrict__ y, int64_t yld) {
    __shared__ float sh[8][8];
    const int r = blockIdx.x, tb = blockIdx.y * 8, w = threadIdx.x >> 5, l = threadIdx.x & 31;
    const float4* wr = reinterpret_cast<const float4*>(W + int64_t(r) * k);
    float acc[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) acc[t] = 0.f;
    for (int i = threadIdx.x; i < k / 4; i += blockDim.x) {
        const float4 wv = wr[i];
#pragma unroll
        for (int t = 0; t < 8; ++t)
            if (tb + t < T) {
                const float4 xv = reinterpret_cast<const float4*>(x + (tb + t) * xld)[i];
                acc[t] += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
            }
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const float v = warp_sum(acc[t]);
        if (l == 0) sh[t][w] = v;
    }
    __syncthreads();
    if (threadIdx.x < 8 && tb + threadIdx.x < T) {
        float v = 0.f;
        for (int i = 0; i < 8; ++i) v += sh[threadIdx.x][i];
        y[(tb + threadIdx.x) * yld + r] = v;
    }
}

// 伝言の門の後半（プレフィル）: 32 語 × 64 行のタイル。c [T][rank]、G・F [d][rank]（rank <= 64）
__global__ void __launch_bounds__(256) ctrl2_tile_kernel(const float* __restrict__ G, const float* __restrict__ F, int rank,
                                                         const float* __restrict__ c, const float* h, int64_t hld, const float* __restrict__ m,
                                                         int64_t mld, int d, int T, float* y, int64_t yld) {
    __shared__ float cs[32][65], gs[64][65], fs[64][65];
    const int t0 = blockIdx.x * 32, i0 = blockIdx.y * 64;
    for (int e = threadIdx.x; e < 32 * 64; e += 256) {
        const int tt = e / 64, rr = e % 64;
        cs[tt][rr] = (t0 + tt < T && rr < rank) ? c[(t0 + tt) * rank + rr] : 0.f;
    }
    for (int e = threadIdx.x; e < 64 * 64; e += 256) {
        const int ii = e / 64, rr = e % 64;
        const bool ok = i0 + ii < d && rr < rank;
        gs[ii][rr] = ok ? G[int64_t(i0 + ii) * rank + rr] : 0.f;
        fs[ii][rr] = ok ? F[int64_t(i0 + ii) * rank + rr] : 0.f;
    }
    __syncthreads();
    // スレッド = 行 1 つ × 語 8 つ（64 行 × 4 組）
    const int ii = threadIdx.x % 64, tg = threadIdx.x / 64;
    float g[8], f[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) g[u] = f[u] = 0.f;
    for (int rr = 0; rr < 64; ++rr) {
        const float gv = gs[ii][rr], fv = fs[ii][rr];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const float cv = cs[tg * 8 + u][rr];
            g[u] += gv * cv;
            f[u] += fv * cv;
        }
    }
    const int i = i0 + ii;
    if (i >= d) return;
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        const int t = t0 + tg * 8 + u;
        if (t >= T) break;
        const float hv = h[t * hld + i], mv = m[t * mld + i];
        y[t * yld + i] = (1.f - tanhf(f[u])) * hv + tanhf(g[u]) * mv;
    }
}

inline unsigned blocks(int64_t n, int nt = 256) { return unsigned((n + nt - 1) / nt); }

}  // namespace

extern "C" {

int ink_rms(const float* x, int64_t xld, const float* w, int k, int T, float eps, float* y, int64_t yld, cudaStream_t s) {
    rms_kernel<<<T, 1024, 0, s>>>(x, xld, w, k, eps, y, yld);
    return int(cudaGetLastError());
}

// ptrs/lds は S 個（<= 16）。q [nq][d]、y [T][nq][d]（行の間隔 yld）
int ink_mix(const float* const* ptrs, const int64_t* lds, int S, const float* q, int nq, int d, int T, float* y, int64_t yld, cudaStream_t s) {
    if (S < 1 || S > MIX_MAX) return -1;
    MixArgs a{};
    for (int i = 0; i < S; ++i) {
        a.ptr[i] = ptrs[i];
        a.ld[i] = lds[i];
    }
    a.S = S;
    mix_kernel<<<dim3(T, nq), 1024, 0, s>>>(a, q, nq, d, y, yld);
    return int(cudaGetLastError());
}

int ink_cat2_rms(const float* h, int64_t hld, const float* m, int64_t mld, int d, int T, float eps, float* z, cudaStream_t s) {
    cat2_rms_kernel<<<dim3(T, 2), 1024, 0, s>>>(h, hld, m, mld, d, eps, z);
    return int(cudaGetLastError());
}

int ink_ctrl(const float* h, int64_t hld, const float* m, int64_t mld, const float* g, const float* f, int d, int T, float* y, int64_t yld,
             cudaStream_t s) {
    ctrl_kernel<<<blocks(int64_t(T) * d), 256, 0, s>>>(h, hld, m, mld, g, f, d, T, y, yld);
    return int(cudaGetLastError());
}

int ink_conv(const float* x, int64_t xld, float* st, int64_t st_ld, const int* lanes, const float* w, int nc, const float* dyn, int64_t dld, int heads, int ch,
             int T, float scale, float* y, int64_t yld, cudaStream_t s) {
    if (nc < 2 || nc > 8) return -1;
    conv_kernel<<<blocks(int64_t(T) * ch), 256, 0, s>>>(x, xld, st, st_ld, lanes, w, nc, dyn, dld, heads, ch, T, scale, y, yld);
    return int(cudaGetLastError());
}

int ink_headprep(float* x, int64_t xld, int heads, int dh, int T, const float* scale, float thr, float eps, const float* tau, cudaStream_t s) {
    if (dh > 256) return -1;
    headprep_kernel<<<dim3(T, heads), 32, 0, s>>>(x, xld, dh, scale, thr, eps, tau);
    return int(cudaGetLastError());
}

// 作業場 part は T·H·n_split·PART_W 個の f32（ink_attn_part_floats）
int64_t ink_attn_part_floats(int T, int H, int n_split) { return int64_t(T) * H * n_split * PART_W; }

int ink_attn_dec(const float* q, int64_t qld, const float* r, int64_t rld, const float* sink, int64_t sld, const float* rb, KvRef kv,
                 int64_t kv_ld, const int* lanes, const int* pos, int T, int H, int KVH, int nrel, int extent, int cap, int ring, int window,
                 int max_keys, int chunk, float* part, float* y, int64_t yld, cudaStream_t s) {
    if (nrel > 32 || H % KVH || chunk < 1) return -1;
    AttnDec a{q, qld, r, rld, sink, sld, rb, kv, kv_ld, lanes, pos, H, KVH, nrel, extent, cap, ring, window, chunk, (max_keys + chunk - 1) / chunk, part};
    if (a.n_split < 1) a.n_split = 1;
    attn_dec_kernel<<<dim3(T, H, a.n_split), AT_NT, 0, s>>>(a);
    attn_combine_kernel<<<dim3(T, H), 32, 0, s>>>(a, y, yld);
    return int(cudaGetLastError());
}

int ink_gate(float* o, int64_t old, const float* g, int64_t gld, int n, int T, cudaStream_t s) {
    gate_kernel<<<blocks(int64_t(T) * n), 256, 0, s>>>(o, old, g, gld, n, T);
    return int(cudaGetLastError());
}

int ink_add(const float* a, int64_t ald, const float* b, int64_t bld, int n, int T, float* y, int64_t yld, cudaStream_t s) {
    add_kernel<<<blocks(int64_t(T) * n), 256, 0, s>>>(a, ald, b, bld, n, T, y, yld);
    return int(cudaGetLastError());
}

int ink_tau(const int* pos, int T, float alpha, float floor_, float* tau, cudaStream_t s) {
    tau_kernel<<<blocks(T), 256, 0, s>>>(pos, T, alpha, floor_, tau);
    return int(cudaGetLastError());
}

int ink_route(const float* lg, int64_t ld, const float* bias, int ne, int kk, int T, float scale, int* ids, float* wts, cudaStream_t s) {
    if (ne > 256 || kk > 32) return -1;
    route_kernel<<<T, 256, 0, s>>>(lg, ld, bias, ne, kk, scale, ids, wts);
    return int(cudaGetLastError());
}

// h の行の間隔は hld（down を 256 の倍数の k にそろえるため ff より広くてよい。広い所は呼ぶ側が 0 にしておく）
int ink_situ(const float* gu, int ff, int R, float cg, float cu, float* h, int64_t hld, cudaStream_t s) {
    situ_kernel<<<blocks(int64_t(R) * ff), 256, 0, s>>>(gu, ff, R, cg, cu, h, hld);
    return int(cudaGetLastError());
}

int ink_moe_sum(const float* o, const float* wts, int d, int kk, int T, float* y, int64_t yld, cudaStream_t s) {
    moe_sum_kernel<<<blocks(int64_t(T) * d), 256, 0, s>>>(o, wts, d, kk, T, y, yld);
    return int(cudaGetLastError());
}

int ink_gemv_f32(const float* W, int k, int n, const float* x, int64_t xld, int T, float* y, int64_t yld, cudaStream_t s) {
    if (k % 4 || T < 1 || T > 8) return -1;
    gemv_f32_kernel<<<n, 256, 0, s>>>(W, k, n, x, xld, T, y, yld);
    return int(cudaGetLastError());
}

int ink_ctrl2(const float* G, const float* F, int rank, const float* c, const float* h, int64_t hld, const float* m, int64_t mld, int d, int T,
              float* y, int64_t yld, cudaStream_t s) {
    if (T > 8 && rank <= 64) {
        ctrl2_tile_kernel<<<dim3((T + 31) / 32, (d + 63) / 64), 256, 0, s>>>(G, F, rank, c, h, hld, m, mld, d, T, y, yld);
        return int(cudaGetLastError());
    }
    ctrl2_kernel<<<(d + 7) / 8, 256, 0, s>>>(G, F, rank, c, h, hld, m, mld, d, T, y, yld);
    return int(cudaGetLastError());
}

int ink_route_warp(const float* lg, int64_t ld, const float* bias, int ne, int kk, int T, float scale, int* ids, float* wts, cudaStream_t s) {
    if (ne > 256 || kk > 32 || kk > ne) return -1;  // kk > ne だと同じ専門家を選び続ける（見直しの指摘）
    route_warp_kernel<<<T, 32, 0, s>>>(lg, ld, bias, ne, kk, scale, ids, wts);
    return int(cudaGetLastError());
}

int ink_conv_pf(const float* x, int64_t xld, const float* st, const float* w, int nc, const float* dyn, int64_t dld, int heads, int ch, int T,
                float scale, float* y, int64_t yld, cudaStream_t s) {
    if (nc < 2 || nc > 8 || x == y) return -1;
    conv_pf_kernel<<<blocks(int64_t(T) * ch), 256, 0, s>>>(x, xld, st, w, nc, dyn, dld, heads, ch, T, scale, y, yld);
    return int(cudaGetLastError());
}

int ink_conv_pf_state(const float* x, int64_t xld, float* st, int nc, int ch, int T, cudaStream_t s) {
    conv_pf_state_kernel<<<blocks(ch), 256, 0, s>>>(x, xld, st, nc, ch, T);
    return int(cudaGetLastError());
}

int ink_attn_pf(const float* q, int64_t qld, const float* r, int64_t rld, const float* sink, int64_t sld, const float* rb, KvRef kv,
                const int* pos, int T, int H, int KVH, int nrel, int extent, int cap, int ring, int window, float* y, int64_t yld,
                cudaStream_t s) {
    if (nrel > 32 || H % KVH || H / KVH > PF_GMAX) return -1;
    AttnPf a{q, qld, r, rld, sink, sld, rb, kv, pos, T, H, KVH, nrel, extent, cap, ring, window, y, yld};
    attn_pf_kernel<<<dim3((T + PF_TQ - 1) / PF_TQ, KVH), 256, 0, s>>>(a);
    return int(cudaGetLastError());
}

int ink_kvwrite_pf(const float* k, int64_t kld, const float* v, int64_t vld, int n, int T, KvOut o, int64_t kv_ld, const int* lanes, const int* pos,
                   int cap, int ring, cudaStream_t s) {
    if (n % 128) return -1;
    kvwrite_pf_kernel<<<dim3(T, n / 128), 128, 0, s>>>(k, kld, v, vld, n / 128, T, o, kv_ld, lanes, pos, cap, ring);
    return int(cudaGetLastError());
}

int ink_moe_sum_map(const float* o, const int* rowmap, const float* wts, int d, int kk, int T, float* y, int64_t yld, cudaStream_t s) {
    moe_sum_map_kernel<<<blocks(int64_t(T) * d), 256, 0, s>>>(o, rowmap, wts, d, kk, T, y, yld);
    return int(cudaGetLastError());
}

int ink_gemm_f32(const float* W, int k, int n, const float* x, int64_t xld, int T, float* y, int64_t yld, cudaStream_t s) {
    if (k % 4 || T < 1) return -1;
    gemm_f32_kernel<<<dim3(n, (T + 7) / 8), 256, 0, s>>>(W, k, n, x, xld, T, y, yld);
    return int(cudaGetLastError());
}

int ink_softcap(float* x, int64_t ld, int n, int T, float cap, cudaStream_t s) {
    softcap_kernel<<<blocks(int64_t(T) * n), 256, 0, s>>>(x, ld, n, T, cap);
    return int(cudaGetLastError());
}

}  // extern "C"
