// Gated DeltaNet（線形 Attention）と因果 conv1d。llama.cpp（unsloth）の qwen35（Qwen3.8-27B）の線形層
// （src/models/qwen35.cpp の build_layer_attn_linear）と同じ計算。手本は tools/gdn_ref.py、答え合わせは tools/gdncheck.py。
//
//   conv:    qkv → 幅 4 の因果畳み込み（状態 3 列を回す）→ silu → q・k を頭ごとに l2 正規化
//   decode:  1 トークンずつ状態を進める（state はレジスタ）→ 出口の rms_norm × w × silu(z)
//            T = 1 は decode_kernel、T > 1（下書きの確かめ）は decode_multi_kernel ＋ norm_kernel
//   prefill: 64 トークンの塊ごと（prep で塊の中を並べて解き、scan で塊を順に進め、out で塊の中の分と正規化）
// 数値は fp32（状態は llama.cpp と同じく f32。prefill の塊の中の g の累積 G だけ double）。
//
// 並び（C の順）: c（conv の後）[B][T][2*Hk*D + Hv*D]、z [B][T][Hv*D]、alpha・beta [B][T][Hv]、
//   state [slot][Hv][D(j: v)][D(i: k)]（ggml と同じ転置。S[i][j] = state[j][i]）、conv の状態 [slot][C][3]、y [B][T][Hv*D]
// v の頭 h は q・k の頭 h % Hk を使う（ggml の並び）。
#include <cstdint>
#include <cuda_runtime.h>
#include <initializer_list>

namespace {

constexpr int D = 128;   // 頭の幅（key・value とも）
constexpr int CS = 64;   // 塊の長さ

__device__ __forceinline__ float silu(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float dot4(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
__device__ __forceinline__ float fma4(float4 a, float4 b, float c) { return fmaf(a.w, b.w, fmaf(a.z, b.z, fmaf(a.y, b.y, fmaf(a.x, b.x, c)))); }
__device__ __forceinline__ float4 ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }
// float4 で読み書きする置き場は 16 バイト揃い（null は通す）
inline bool al16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }
__device__ __forceinline__ void st4(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }

// ---- 畳み込み ----
// 1 ブロック = 128 通り（= 1 頭）× TT トークン。1 スレッドが 1 通りを受け持つ。
// 状態を読むのも書くのも、トークン 0 を含むブロックだけ（ほかのブロックは x から直前 3 つを読む）。
// slots_out・n_valid は decode と同じ（新しい状態 = [古い状態, x の前 n_valid[b] 個] の最後の 3 つ）。y が null なら状態だけ。
template <int TT>
__global__ void __launch_bounds__(128) conv_kernel(const float* __restrict__ x, int64_t x_ld, float* __restrict__ st,
                                                   const int* __restrict__ slots, const int* __restrict__ slots_out,
                                                   const int* __restrict__ n_valid, const float* __restrict__ w,
                                                   float* __restrict__ y, int T, int C, int n_norm, float eps) {
    __shared__ float red[2][4];
    const int ch = blockIdx.x * 128 + threadIdx.x;
    const int t0 = blockIdx.y * TT, b = blockIdx.z;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n = n_valid ? min(n_valid[b], T) : T;
    const float* xb = x + int64_t(b) * T * x_ld + ch;
    float* yb = y + int64_t(b) * T * C + ch;
    const int slot = slots ? slots[b] : b;
    const float* s = st + (int64_t(slot) * C + ch) * 3;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f;
    if (t0 == 0) s0 = s[0], s1 = s[1], s2 = s[2];
    if (y) {
        const float4 wv = ld4(w + int64_t(ch) * 4);
        float p0, p1, p2;  // 直前 3 つ（古い順）
        if (t0 == 0) {
            p0 = s0, p1 = s1, p2 = s2;
        } else {
            p0 = xb[int64_t(t0 - 3) * x_ld], p1 = xb[int64_t(t0 - 2) * x_ld], p2 = xb[int64_t(t0 - 1) * x_ld];
        }
        const bool norm = blockIdx.x * 128 < n_norm;  // ブロックの中で同じ
        const int t1 = min(t0 + TT, n);
        for (int t = t0; t < t1; ++t) {
            const float xt = xb[int64_t(t) * x_ld];
            float o = silu(wv.x * p0 + wv.y * p1 + wv.z * p2 + wv.w * xt);
            p0 = p1, p1 = p2, p2 = xt;
            if (norm) {
                const float ss = warp_sum(o * o);
                if (lane == 0) red[t & 1][warp] = ss;
                __syncthreads();
                o *= rsqrtf(red[t & 1][0] + red[t & 1][1] + red[t & 1][2] + red[t & 1][3] + eps);
            }
            yb[int64_t(t) * C] = o;
        }
    }
    if (t0 == 0) {  // 新しい状態 = [古い状態, x] の最後の 3 つ
        float nv[3];
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int p = n + j;
            nv[j] = p < 3 ? (p == 0 ? s0 : p == 1 ? s1 : s2) : xb[int64_t(p - 3) * x_ld];
        }
        float* so = st + (int64_t(slots_out ? slots_out[b] : slot) * C + ch) * 3;
        so[0] = nv[0], so[1] = nv[1], so[2] = nv[2];
    }
}

// ---- 1 トークンずつ（decode と、下書きの確かめのような短い列） ----
// 1 ブロック = (頭 h, 列 b)、512 スレッド。状態の行 j（v の列）を 8 lane で持ち（lane g は i = (m·8 + g)·4 + e）、
// warp は 4 行ずつ 2 組（8 行）。和は 8 lane の中だけ（shfl 3 段で 4 行ぶん同時に）。T トークンの間、状態はレジスタのまま。
// 次のトークンの q・k は計算の間に共有メモリへ先に読む（同期は 1 トークン 1 回）。
// 1 行の更新（ggml の gated_delta_net と同じ順）: kv = M[j]·k、d = (v[j] - e^g kv) β、M[j] = e^g M[j] + d k、o[j] = M[j]·q / √D
// slots_out があれば新しい状態はそちらへ（下書きの確かめで元の状態を残す）。n_valid があれば列 b は前の n_valid[b] トークンだけ
// 進める（受け入れた分だけ確定する）。y が null なら出口を書かない。
constexpr int DEC_NT = 512;

__device__ __forceinline__ float sum8(float v) {  // 8 lane（隣どうし）の和
    v += __shfl_xor_sync(0xffffffff, v, 1);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    return v + __shfl_xor_sync(0xffffffff, v, 4);
}

__global__ void __launch_bounds__(DEC_NT) decode_kernel(
    const float* __restrict__ c, int64_t c_ld, const float* __restrict__ z, int64_t z_ld,
    const float* __restrict__ alpha, const float* __restrict__ beta, int64_t ab_ld,
    const float* __restrict__ ssm_a, const float* __restrict__ dt_bias, const float* __restrict__ norm_w,
    float* __restrict__ state, const int* __restrict__ slots, const int* __restrict__ slots_out,
    const int* __restrict__ n_valid, float* __restrict__ y, int64_t y_ld, int T, int Hk, int Hv, float eps) {
    __shared__ __align__(16) float o_s[2][D];
    __shared__ __align__(16) float qk_s[2][2 * D];  // q・k（トークンの偶奇で 2 枚）
    const int h = blockIdx.x, b = blockIdx.y, kh = h % Hk, tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31, g = lane & 7, rq = lane >> 3;
    const int j0 = warp * 8 + rq;  // この lane の行は j0 と j0 + 4
    const int n = n_valid ? min(n_valid[b], T) : T;
    const int slot = slots ? slots[b] : b;
    const float* M = state + (int64_t(slot) * Hv + h) * D * D;
    float4 s[2][4];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int m = 0; m < 4; ++m) s[r][m] = ld4(M + (j0 + 4 * r) * D + (m * 8 + g) * 4);
    const float ah = ssm_a[h], dh = dt_bias[h], scale = rsqrtf(float(D));
    const float4 nw = ld4(norm_w + lane * 4);
    const float* cb = c + int64_t(b) * T * c_ld;
    // 次のトークンの値は 1 トークン先に読んでおく: q・k は 64 スレッドがレジスタへ読み、同期の前に共有メモリへ。
    // alpha・beta・v（と warp 0 は z）は各スレッドのレジスタへ
    float4 qk_n = make_float4(0.f, 0.f, 0.f, 0.f), z_n = qk_n;
    float a_n = 0.f, b_n = 0.f, v0_n = 0.f, v1_n = 0.f;
    auto fetch = [&](int t) {
        const int64_t bt = int64_t(b) * T + t;
        const float* row = cb + int64_t(t) * c_ld;
        if (tid < 64) qk_n = ld4(row + ((tid >> 5) ? Hk + kh : kh) * D + (tid & 31) * 4);
        a_n = alpha[bt * ab_ld + h], b_n = beta[bt * ab_ld + h];
        v0_n = row[(2 * Hk + h) * D + j0], v1_n = row[(2 * Hk + h) * D + j0 + 4];
        if (warp == 0 && y) z_n = ld4(z + bt * z_ld + h * D + lane * 4);
    };
    auto put_qk = [&](int t) {
        if (tid < 64) st4(&qk_s[t & 1][(tid >> 5) * D + (tid & 31) * 4], qk_n);
    };
    if (n > 0) fetch(0), put_qk(0);
    __syncthreads();
    for (int t = 0; t < n; ++t) {
        const int64_t bt = int64_t(b) * T + t;
        const float al = a_n, bl = b_n, v0 = v0_n, v1 = v1_n;
        const float4 zv = z_n;
        if (t + 1 < n) fetch(t + 1);
        const float eg = expf(softplus(al + dh) * ah), be = sigm(bl);
        const float* qs = qk_s[t & 1];
        const float* ks = qs + D;
        float4 k4[4];
#pragma unroll
        for (int m = 0; m < 4; ++m) k4[m] = ld4(ks + (m * 8 + g) * 4);
        float kv0 = 0.f, kv1 = 0.f;
#pragma unroll
        for (int m = 0; m < 4; ++m) kv0 = fma4(s[0][m], k4[m], kv0), kv1 = fma4(s[1][m], k4[m], kv1);
        kv0 = sum8(kv0), kv1 = sum8(kv1);
        const float d[2] = {(v0 - eg * kv0) * be, (v1 - eg * kv1) * be};
        float o0 = 0.f, o1 = 0.f;
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const float4 q4 = ld4(qs + (m * 8 + g) * 4);
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                float4& x = s[r][m];
                x.x = fmaf(k4[m].x, d[r], eg * x.x), x.y = fmaf(k4[m].y, d[r], eg * x.y);
                x.z = fmaf(k4[m].z, d[r], eg * x.z), x.w = fmaf(k4[m].w, d[r], eg * x.w);
            }
            o0 = fma4(s[0][m], q4, o0), o1 = fma4(s[1][m], q4, o1);
        }
        o0 = sum8(o0), o1 = sum8(o1);
        if (g == 0) o_s[t & 1][j0] = o0 * scale, o_s[t & 1][j0 + 4] = o1 * scale;
        if (t + 1 < n) put_qk(t + 1);
        __syncthreads();  // o_s・qk_s は 2 枚を交互に使うので、同期は 1 トークン 1 回でよい
        if (warp == 0 && y) {  // 出口: rms_norm(o) × w × silu(z)
            const float4 ov = ld4(&o_s[t & 1][lane * 4]);
            const float rs = rsqrtf(warp_sum(dot4(ov, ov)) / D + eps);
            st4(y + bt * y_ld + h * D + lane * 4,
                make_float4(ov.x * rs * nw.x * silu(zv.x), ov.y * rs * nw.y * silu(zv.y),
                            ov.z * rs * nw.z * silu(zv.z), ov.w * rs * nw.w * silu(zv.w)));
        }
    }
    float* Mo = state + (int64_t(slots_out ? slots_out[b] : slot) * Hv + h) * D * D;
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int m = 0; m < 4; ++m) st4(Mo + (j0 + 4 * r) * D + (m * 8 + g) * 4, s[r][m]);
}

// ---- T > 1 の 1 トークンずつ（下書きの確かめ） ----
// decode_kernel と同じ計算を、頭を 4 ブロック（各 32 行・4 warp）に分け、トークンごとのブロックの同期をなくした版
// （warp は自分の 8 行だけで閉じる。q・k は global から直に読み、次のトークンの分は計算の間にレジスタへ）。
// o（√D で割ったもの）を y に置き、出口の正規化は norm_kernel が後でまとめて。B が小さいとき SM を広く使える
constexpr int DM_WARPS = 4, DM_PARTS = D / (DM_WARPS * 8);

__global__ void __launch_bounds__(DM_WARPS * 32) decode_multi_kernel(  // レジスタ 151・1 SM に 3 ブロック（4 に絞ると溢れて遅い）
    const float* __restrict__ c, int64_t c_ld, const float* __restrict__ alpha, const float* __restrict__ beta, int64_t ab_ld,
    const float* __restrict__ ssm_a, const float* __restrict__ dt_bias, float* __restrict__ state, const int* __restrict__ slots,
    const int* __restrict__ slots_out, const int* __restrict__ n_valid, float* __restrict__ o, int64_t o_ld, int T, int Hk,
    int Hv) {
    const int h = blockIdx.x / DM_PARTS, part = blockIdx.x % DM_PARTS, b = blockIdx.y, kh = h % Hk;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane & 7, rq = lane >> 3;
    const int j0 = (part * DM_WARPS + warp) * 8 + rq;  // この lane の行は j0 と j0 + 4
    const int n = n_valid ? min(n_valid[b], T) : T;
    const int slot = slots ? slots[b] : b;
    const float* M = state + (int64_t(slot) * Hv + h) * D * D;
    float4 s[2][4];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int m = 0; m < 4; ++m) s[r][m] = ld4(M + (j0 + 4 * r) * D + (m * 8 + g) * 4);
    const float ah = ssm_a[h], dh = dt_bias[h], scale = rsqrtf(float(D));
    const float* cb = c + int64_t(b) * T * c_ld;
    struct Tok {
        float4 q[4], k[4];
        float a, b, v0, v1;
    };
    auto fetch = [&](Tok& x, int t) {
        const float* row = cb + int64_t(t) * c_ld;
#pragma unroll
        for (int m = 0; m < 4; ++m) x.q[m] = ld4(row + kh * D + (m * 8 + g) * 4), x.k[m] = ld4(row + (Hk + kh) * D + (m * 8 + g) * 4);
        const int64_t ab = (int64_t(b) * T + t) * ab_ld + h;
        x.a = alpha[ab], x.b = beta[ab];
        x.v0 = row[(2 * Hk + h) * D + j0], x.v1 = row[(2 * Hk + h) * D + j0 + 4];
    };
    auto step = [&](const Tok& x, int t) {
        const float eg = expf(softplus(x.a + dh) * ah), be = sigm(x.b);
        float p0[4], p1[4];  // 和の鎖を短く（m ごとに別々に足してから）
#pragma unroll
        for (int m = 0; m < 4; ++m) p0[m] = dot4(s[0][m], x.k[m]), p1[m] = dot4(s[1][m], x.k[m]);
        const float kv0 = sum8((p0[0] + p0[1]) + (p0[2] + p0[3])), kv1 = sum8((p1[0] + p1[1]) + (p1[2] + p1[3]));
        const float d[2] = {(x.v0 - eg * kv0) * be, (x.v1 - eg * kv1) * be};
#pragma unroll
        for (int m = 0; m < 4; ++m) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                float4& e = s[r][m];
                e.x = fmaf(x.k[m].x, d[r], eg * e.x), e.y = fmaf(x.k[m].y, d[r], eg * e.y);
                e.z = fmaf(x.k[m].z, d[r], eg * e.z), e.w = fmaf(x.k[m].w, d[r], eg * e.w);
            }
            p0[m] = dot4(s[0][m], x.q[m]), p1[m] = dot4(s[1][m], x.q[m]);
        }
        if (o) {
            const float o0 = sum8((p0[0] + p0[1]) + (p0[2] + p0[3])), o1 = sum8((p1[0] + p1[1]) + (p1[2] + p1[3]));
            if (g == 0) {
                float* op = o + (int64_t(b) * T + t) * o_ld + h * D + j0;
                op[0] = o0 * scale, op[4] = o1 * scale;
            }
        }
    };
    // 2 トークンずつ（レジスタの置き場を 2 つ交互に。次の分を計算の間に読む）
    Tok xa, xb;
    if (n > 0) fetch(xa, 0);
    for (int t = 0; t < n; t += 2) {
        if (t + 1 < n) fetch(xb, t + 1);
        step(xa, t);
        if (t + 1 >= n) break;
        if (t + 2 < n) fetch(xa, t + 2);
        step(xb, t + 1);
    }
    float* Mo = state + (int64_t(slots_out ? slots_out[b] : slot) * Hv + h) * D * D;
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int m = 0; m < 4; ++m) st4(Mo + (j0 + 4 * r) * D + (m * 8 + g) * 4, s[r][m]);
}

// 出口の正規化だけ（decode_multi_kernel の後）: y = rms_norm(o) × w × silu(z)。1 warp = (行 bt, 頭 h) の 128 個
__global__ void __launch_bounds__(256) norm_kernel(float* __restrict__ y, int64_t y_ld, const float* __restrict__ z, int64_t z_ld,
                                                   const float* __restrict__ norm_w, const int* __restrict__ n_valid, int T,
                                                   int rows, int Hv, float eps) {
    const int idx = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (idx >= rows * Hv) return;
    const int bt = idx / Hv, h = idx % Hv;
    if (n_valid && bt % T >= n_valid[bt / T]) return;
    float* yp = y + int64_t(bt) * y_ld + h * D + lane * 4;
    const float4 ov = ld4(yp), zv = ld4(z + int64_t(bt) * z_ld + h * D + lane * 4), nw = ld4(norm_w + lane * 4);
    const float rs = rsqrtf(warp_sum(dot4(ov, ov)) / D + eps);
    st4(yp, make_float4(ov.x * rs * nw.x * silu(zv.x), ov.y * rs * nw.y * silu(zv.y), ov.z * rs * nw.z * silu(zv.z),
                        ov.w * rs * nw.w * silu(zv.w)));
}

// ---- プレフィル 1: 塊の中（塊どうしは並べて計算できる） ----
// 1 ブロック = (塊, 頭 h, 列 b)、256 スレッド、共有メモリ 43KB（1 SM に 2 ブロック）。G = 塊の中の g の累積、
//   L[i][j] = β_i (k_i·k_j) e^{G_i-G_j}（i > j）、A[i][j] = (q_i·k_j) e^{G_i-G_j} / √D（i ≥ j）
//   T = (I + L)^{-1} を前進代入で作り（64×64・安い）、W = T (k β e^G)、U = T (v β) は GEMM で
// 書き出し（scan・out が読みやすい並び）: W^T [D][CS]、Q'^T = (q e^G/√D)^T [D][CS]、U [CS][D]、A^T [CS][CS]、
//   e^G と e^{G_last-G}（各 [CS]）
// G は double で足す: |g| が大きい頭では G_i - G_j を f32 で引くと桁が落ちる
constexpr int KS = 68;                   // k・q の半分（64 列）を置く行の幅（float4 で読むので 4 の倍数）
constexpr int NLP = CS * (CS - 1) / 2;   // L の下三角（i > j）だけを i(i-1)/2 + j に詰めたもの
constexpr int PREP_SMEM = CS * 8 + 2 * CS * 4 + 2 * CS * KS * 4 + NLP * 4;

__global__ void __launch_bounds__(256, 2) prep_kernel(
    const float* __restrict__ c, int64_t c_ld, const float* __restrict__ alpha, const float* __restrict__ beta, int64_t ab_ld,
    const float* __restrict__ ssm_a, const float* __restrict__ dt_bias, float* __restrict__ ws_w, float* __restrict__ ws_q,
    float* __restrict__ ws_u, float* __restrict__ ws_a, float* __restrict__ ws_g, int T, int Hk, int Hv) {
    extern __shared__ __align__(16) double smd[];
    double* gs = smd;                               // G（double）
    float* bs = reinterpret_cast<float*>(gs + CS);  // β
    float* es = bs + CS;                            // e^G
    float* ks = es + CS;                            // k の半分 [CS][KS] → 後で T^T
    float* qs = ks + CS * KS;                       // q の半分 [CS][KS] → 後で右辺の半分
    float* Lp = qs + CS * KS;                       // L（詰めたもの）
    const int ci = blockIdx.x, h = blockIdx.y, b = blockIdx.z, kh = h % Hk;
    const int nc = gridDim.x, tid = threadIdx.x, t0 = ci * CS;
    const int64_t base = (int64_t(b) * Hv + h) * nc + ci;
    const float* rows = c + int64_t(b) * T * c_ld;
    const float scale = rsqrtf(float(D));
    const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
    if (tid < CS) {  // g と β。足りない所（T の後）は 0（ggml の pad と同じ）
        const int t = t0 + tid;
        double g = 0.0;
        float be = 0.f;
        if (t < T) {
            const int64_t ab = (int64_t(b) * T + t) * ab_ld + h;
            g = softplus(alpha[ab] + dt_bias[h]) * ssm_a[h];
            be = sigm(beta[ab]);
        }
        // 64 個の累積: warp ごとに足してから、warp 0 の合計を warp 1 に足す
        const int lane = tid & 31;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const double u = __shfl_up_sync(0xffffffff, g, o);
            if (lane >= o) g += u;
        }
        gs[tid] = g;
        bs[tid] = be;
    }
    __syncthreads();
    if (tid >= 32 && tid < CS) gs[tid] += gs[31];
    __syncthreads();
    if (tid < CS) es[tid] = expf(float(gs[tid]));
    // k k^T と q k^T を 4×4 ずつ。d は 64 ずつ 2 回に分けて共有メモリへ
    const int ti = tid >> 4, tj = tid & 15;
    float ak[4][4] = {}, aq[4][4] = {};
    for (int hd = 0; hd < 2; ++hd) {
        if (hd) __syncthreads();
        for (int e = tid; e < CS * 16; e += 256) {
            const int i = e >> 4, d = (e & 15) * 4, t = t0 + i;
            const float* r = rows + int64_t(t) * c_ld + hd * 64 + d;
            st4(qs + i * KS + d, t < T ? ld4(r + kh * D) : z4);
            st4(ks + i * KS + d, t < T ? ld4(r + (Hk + kh) * D) : z4);
        }
        __syncthreads();
        {  // Q'^T（scan が読む）
            float* Qt = ws_q + base * D * CS + hd * 64 * CS;
            for (int e = tid; e < 64 * CS; e += 256) {
                const int d = e >> 6, i = e & (CS - 1);
                Qt[e] = qs[i * KS + d] * es[i] * scale;
            }
        }
#pragma unroll 4
        for (int d = 0; d < 64; d += 4) {
            float4 ka[4], qa[4], kj[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                ka[u] = ld4(ks + (ti + 16 * u) * KS + d);
                qa[u] = ld4(qs + (ti + 16 * u) * KS + d);
                kj[u] = ld4(ks + (tj + 16 * u) * KS + d);
            }
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int v = 0; v < 4; ++v) ak[u][v] = fma4(ka[u], kj[v], ak[u][v]), aq[u][v] = fma4(qa[u], kj[v], aq[u][v]);
        }
    }
    {
        float* At = ws_a + base * CS * CS;
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int i = ti + 16 * u, j = tj + 16 * v;
                const float dec = i >= j ? expf(float(gs[i] - gs[j])) : 0.f;
                if (i > j) Lp[i * (i - 1) / 2 + j] = bs[i] * ak[u][v] * dec;
                At[j * CS + i] = aq[u][v] * dec * scale;
            }
    }
    // T = (I + L)^{-1}（下三角）を前進代入で: T[i][j] = -Σ_{j≤k<i} L[i][k] T[k][j]（i > j）。
    // 1 列を 4 スレッド（k を 4 つおき）で持つ。行 i は i % 4 のスレッドが書き、後でそれを読むのも同じスレッド
    // なので、列の中の同期は shfl だけでよい。置き場は T^T [j][i]（GEMM で T[t..t+3][k] を float4 で読む）
    float* Tt = ks;            // T^T [CS][KS]
    float* Rh = ks + CS * KS;  // 右辺の半分 [CS][KS]（qs の所）
    __syncthreads();           // ks・qs を使い終わった（Lp も揃った）
    for (int e = tid; e < CS * KS; e += 256) Tt[e] = e / KS == e % KS ? 1.f : 0.f;
    __syncthreads();
    {
        const int j = tid >> 2, p = tid & 3;
        float* Tj = Tt + j * KS;
        for (int i = 1; i < CS; ++i) {  // 全スレッドで同じ回数（shfl のため）
            const float* Li = Lp + i * (i - 1) / 2;
            float a = 0.f;
            for (int k = (j & ~3) + p; k < i; k += 4) a = fmaf(Li[k], Tj[k], a);
            a += __shfl_xor_sync(0xffffffff, a, 1);
            a += __shfl_xor_sync(0xffffffff, a, 2);
            if (i > j && (i & 3) == p) Tj[i] = -a;
        }
    }
    // W = T (k β e^G)、U = T (v β) を 64 列ずつ 4 回（W の前半・後半、U の前半・後半）。1 スレッド 4 行 × 4 列
    const int tg4 = (tid >> 4) * 4, dg4 = (tid & 15) * 4;
    for (int pass = 0; pass < 4; ++pass) {
        const bool isw = pass < 2;
        const int hd = pass & 1, off = (isw ? (Hk + kh) * D : (2 * Hk + h) * D) + hd * 64;
        __syncthreads();  // T がそろった・前の回の Rh を使い終わった
        for (int e = tid; e < CS * 16; e += 256) {
            const int i = e >> 4, d = (e & 15) * 4, t = t0 + i;
            const float4 x = t < T ? ld4(rows + int64_t(t) * c_ld + off + d) : z4;
            const float f = isw ? bs[i] * es[i] : bs[i];
            st4(Rh + i * KS + d, make_float4(x.x * f, x.y * f, x.z * f, x.w * f));
        }
        __syncthreads();
        float acc[4][4] = {};
        for (int k = 0; k < tg4 + 4; ++k) {  // T は下三角
            const float4 tv = ld4(Tt + k * KS + tg4), rv = ld4(Rh + k * KS + dg4);
            const float a4[4] = {tv.x, tv.y, tv.z, tv.w}, r4[4] = {rv.x, rv.y, rv.z, rv.w};
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[u][q] = fmaf(a4[u], r4[q], acc[u][q]);
        }
        if (isw) {
            float* Wt = ws_w + base * D * CS + (hd * 64 + dg4) * CS + tg4;
#pragma unroll
            for (int q = 0; q < 4; ++q) st4(Wt + q * CS, make_float4(acc[0][q], acc[1][q], acc[2][q], acc[3][q]));
        } else {
            float* U = ws_u + base * CS * D + tg4 * D + hd * 64 + dg4;
#pragma unroll
            for (int u = 0; u < 4; ++u) st4(U + u * D, make_float4(acc[u][0], acc[u][1], acc[u][2], acc[u][3]));
        }
    }
    if (tid < CS) {
        ws_g[base * 2 * CS + tid] = es[tid];
        ws_g[base * 2 * CS + CS + tid] = expf(float(gs[CS - 1] - gs[tid]));
    }
}

// ---- プレフィル 2: 塊を順に進める ----
// 1 ブロック = (v の列 32 本, 頭 h, 列 b)、256 スレッド。列どうしは独立なので分けて並べる。
// スレッド (rg, jg) = (tid/8, tid%8) は S の 4×4（i = rg·4..、j = jg·4..）をレジスタに持つ。塊ごとに:
//   1. S を共有メモリ Ss へ写し、[W ; Q'] S（128 行 × 32 列）を 4 行 × 4 列ずつ（rg < 16 は W の行、16 以上は Q' の行）。
//      W^T・Q'^T（prep が [i][t] で書いたもの）は 16 行（i）ずつ切れ端にして共有メモリへ
//   2. vn = U - W S（ws_u に上書き・共有メモリへ）、o' = Q' S（y に書く。塊の中の A vn は out_kernel で足す）
//   3. S = e^{G_last} S + K'^T vn（K' = k e^{G_last-G} を 16 トークンずつ切れ端に）
// 切れ端は 1 枚で、次の分は計算の間にレジスタへ先に読む。共有メモリ 32KB・レジスタ 80 以下で 1 SM に 3 ブロック
// （B=1 の 192 ブロックが一度に載る）
constexpr int SC_V = 32, SC_PI = 16, SC_NP = D / SC_PI;  // 列の数、切れ端の行の数、段 1 の切れ端の数
constexpr int SC_SMEM = (D * SC_V + CS * SC_V + SC_PI * D) * 4;
static_assert(2 * CS == D, "段 1 の切れ端は [W^T | Q'^T] の 128 列");

__global__ void __launch_bounds__(256, 3) scan_kernel(
    const float* __restrict__ c, int64_t c_ld, const float* __restrict__ ws_w, const float* __restrict__ ws_q,
    float* __restrict__ ws_u, const float* __restrict__ ws_g, float* __restrict__ state, const int* __restrict__ slots,
    float* __restrict__ o, int64_t o_ld, int T, int Hk, int Hv) {
    extern __shared__ __align__(16) float sm[];
    float* Ss = sm;              // S [D][SC_V]
    float* vs = Ss + D * SC_V;   // vn [CS][SC_V]
    float* pc = vs + CS * SC_V;  // 切れ端 [SC_PI][D]
    const int cb = blockIdx.x, h = blockIdx.y, b = blockIdx.z, kh = h % Hk;
    const int tid = threadIdx.x, rg = tid >> 3, jg = tid & 7, i0 = rg * 4, j0 = jg * 4, c0 = cb * SC_V;
    const int nc = (T + CS - 1) / CS;
    float* M = state + (int64_t(slots ? slots[b] : b) * Hv + h) * D * D;  // M[j][i] = S[i][j]
    float st[4][4];  // st[r][q] = S[i0 + r][c0 + j0 + q]
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const float4 v = ld4(M + int64_t(c0 + j0 + q) * D + i0);
        st[0][q] = v.x, st[1][q] = v.y, st[2][q] = v.z, st[3][q] = v.w;
    }
    const float* rows = c + int64_t(b) * T * c_ld;
    const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 pf[2];  // 次の切れ端（2048 個 = 1 スレッド float4 2 個）
    auto fetch = [&](int ci, int p) {  // p < SC_NP: [W^T | Q'^T] の i = 16p..、それ以降: K' の t = 16(p - SC_NP)..
        const int64_t base = (int64_t(b) * Hv + h) * nc + ci;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int e = tid + r * 256, ii = e >> 5, x = (e & 31) * 4;
            if (p < SC_NP) {
                const int64_t off = base * D * CS + (p * SC_PI + ii) * CS;
                pf[r] = x < CS ? ld4(ws_w + off + x) : ld4(ws_q + off + x - CS);
            } else {
                const int t = (p - SC_NP) * SC_PI + ii, tt = ci * CS + t;
                const float f = ws_g[base * 2 * CS + CS + t];
                const float4 k = tt < T ? ld4(rows + int64_t(tt) * c_ld + (Hk + kh) * D + x) : z4;
                pf[r] = make_float4(k.x * f, k.y * f, k.z * f, k.w * f);
            }
        }
    };
    auto put = [&]() {
#pragma unroll
        for (int r = 0; r < 2; ++r) st4(pc + (tid + r * 256) * 4, pf[r]);
    };
    fetch(0, 0);
    for (int ci = 0; ci < nc; ++ci) {
        const int64_t base = (int64_t(b) * Hv + h) * nc + ci;
        const int t0 = ci * CS;
        float* U = ws_u + base * CS * D + c0 + j0;
        float4 pu[4];
        if (rg < 16) {
#pragma unroll
            for (int r = 0; r < 4; ++r) pu[r] = ld4(U + (i0 + r) * D);
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) st4(Ss + (i0 + r) * SC_V + j0, make_float4(st[r][0], st[r][1], st[r][2], st[r][3]));
        // 段 1: [W ; Q'] S
        float acc[4][4] = {};
        for (int p = 0; p < SC_NP; ++p) {
            __syncthreads();
            put();
            __syncthreads();
            fetch(ci, p + 1);
#pragma unroll
            for (int ii = 0; ii < SC_PI; ++ii) {
                const float4 a = ld4(pc + ii * D + i0), s4 = ld4(Ss + (p * SC_PI + ii) * SC_V + j0);
                const float av[4] = {a.x, a.y, a.z, a.w}, sv[4] = {s4.x, s4.y, s4.z, s4.w};
#pragma unroll
                for (int r = 0; r < 4; ++r)
#pragma unroll
                    for (int q = 0; q < 4; ++q) acc[r][q] = fmaf(av[r], sv[q], acc[r][q]);
            }
        }
        // 段 2: vn と o'
        if (rg < 16) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const float4 v = make_float4(pu[r].x - acc[r][0], pu[r].y - acc[r][1], pu[r].z - acc[r][2], pu[r].w - acc[r][3]);
                st4(vs + (i0 + r) * SC_V + j0, v);
                st4(U + (i0 + r) * D, v);
            }
        } else {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int t = t0 + i0 - CS + r;
                if (t < T) st4(o + (int64_t(b) * T + t) * o_ld + h * D + c0 + j0, make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]));
            }
        }
        // 段 3: S = e^{G_last} S + K'^T vn
        const float e_last = ws_g[base * 2 * CS + CS - 1];
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int q = 0; q < 4; ++q) st[r][q] *= e_last;
        for (int p = SC_NP; p < SC_NP + CS / SC_PI; ++p) {
            __syncthreads();
            put();
            __syncthreads();
            if (p + 1 < SC_NP + CS / SC_PI) fetch(ci, p + 1);
            else if (ci + 1 < nc) fetch(ci + 1, 0);
#pragma unroll
            for (int tt = 0; tt < SC_PI; ++tt) {
                const float4 k4 = ld4(pc + tt * D + i0), v4 = ld4(vs + ((p - SC_NP) * SC_PI + tt) * SC_V + j0);
                const float kv[4] = {k4.x, k4.y, k4.z, k4.w}, vv[4] = {v4.x, v4.y, v4.z, v4.w};
#pragma unroll
                for (int r = 0; r < 4; ++r)
#pragma unroll
                    for (int q = 0; q < 4; ++q) st[r][q] = fmaf(kv[r], vv[q], st[r][q]);
            }
        }
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) st4(M + int64_t(c0 + j0 + q) * D + i0, make_float4(st[0][q], st[1][q], st[2][q], st[3][q]));
}

// ---- プレフィル 3: 塊の中の A vn を足して出口の正規化（塊どうし並べて計算できる） ----
// 1 ブロック = (塊, 頭 h, 列 b)、256 スレッド = 16 組 × 4 トークン ・ 16 組 × 8 列。共有メモリ 48KB（1 SM に 2 ブロック）
//   y = rms_norm(o' + A vn) × w × silu(z)
constexpr int OUT_SMEM = (CS * D + CS * CS) * 4;
__global__ void __launch_bounds__(256, 2) out_kernel(const float* __restrict__ ws_u, const float* __restrict__ ws_a,
                                                     const float* __restrict__ z, int64_t z_ld, const float* __restrict__ norm_w,
                                                     float* __restrict__ y, int64_t y_ld, int T, int Hv, float eps) {
    extern __shared__ __align__(16) float sm[];
    float* vs = sm;              // vn [CS][D]
    float* at = vs + CS * D;     // A^T [j][t]
    const int ci = blockIdx.x, h = blockIdx.y, b = blockIdx.z, nc = gridDim.x;
    const int tid = threadIdx.x, tg = tid >> 4, cg = tid & 15, t0 = ci * CS;
    const int64_t base = (int64_t(b) * Hv + h) * nc + ci;
    const float* VN = ws_u + base * CS * D;
    const float* A = ws_a + base * CS * CS;
#pragma unroll 8
    for (int e = tid * 4; e < CS * D; e += 1024) st4(vs + e, ld4(VN + e));
#pragma unroll 4
    for (int e = tid * 4; e < CS * CS; e += 1024) st4(at + e, ld4(A + e));
    const int tn = min(CS, T - t0);
    const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
    float acc[4][8];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const int t = tg * 4 + u;
        const float* yp = y + (int64_t(b) * T + t0 + t) * y_ld + h * D + cg * 8;
        const float4 a = t < tn ? ld4(yp) : z4, e = t < tn ? ld4(yp + 4) : z4;
        acc[u][0] = a.x, acc[u][1] = a.y, acc[u][2] = a.z, acc[u][3] = a.w;
        acc[u][4] = e.x, acc[u][5] = e.y, acc[u][6] = e.z, acc[u][7] = e.w;
    }
    __syncthreads();
    for (int j = 0; j < tg * 4 + 4; ++j) {  // A は下三角
        const float4 a = ld4(at + j * CS + tg * 4);
        const float4 v0 = ld4(vs + j * D + cg * 8), v1 = ld4(vs + j * D + cg * 8 + 4);
        const float av[4] = {a.x, a.y, a.z, a.w}, vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int q = 0; q < 8; ++q) acc[u][q] = fmaf(av[u], vv[q], acc[u][q]);
    }
    const float4 w0 = ld4(norm_w + cg * 8), w1 = ld4(norm_w + cg * 8 + 4);
    const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        float ss = 0.f;
#pragma unroll
        for (int q = 0; q < 8; ++q) ss = fmaf(acc[u][q], acc[u][q], ss);
#pragma unroll
        for (int x = 1; x < 16; x <<= 1) ss += __shfl_xor_sync(0xffffffff, ss, x);
        const int t = tg * 4 + u;
        if (t < tn) {
            const float rs = rsqrtf(ss / D + eps);
            const int64_t bt = int64_t(b) * T + t0 + t;
            const float* zp = z + bt * z_ld + h * D + cg * 8;
            const float4 za = ld4(zp), zb = ld4(zp + 4);
            const float zv[8] = {za.x, za.y, za.z, za.w, zb.x, zb.y, zb.z, zb.w};
            float r[8];
#pragma unroll
            for (int q = 0; q < 8; ++q) r[q] = acc[u][q] * rs * wv[q] * silu(zv[q]);
            float* yp = y + bt * y_ld + h * D + cg * 8;
            st4(yp, make_float4(r[0], r[1], r[2], r[3]));
            st4(yp + 4, make_float4(r[4], r[5], r[6], r[7]));
        }
    }
}

int64_t ws_floats(int B, int T, int Hv) {
    const int64_t n = int64_t(B) * Hv * ((T + CS - 1) / CS);
    return n * (3 * CS * D + CS * CS + 2 * CS);
}

}  // namespace

extern "C" {

// 戻り値: 0 = 成功、-2 = 大きさが合わない、それ以外は cudaError
int infer_gdn_conv(const float* x, int64_t x_ld, float* conv_s, const int* slots, const int* slots_out, const int* n_valid,
                   const float* w, float* y, int B, int T, int C, int n_norm, float eps, cudaStream_t s) {
    if (C % 128 || n_norm % 128 || T < 1 || !al16(w)) return -2;
    constexpr int TT = 64;
    const dim3 grid(C / 128, y ? (T + TT - 1) / TT : 1, B);
    conv_kernel<TT><<<grid, 128, 0, s>>>(x, x_ld, conv_s, slots, slots_out, n_valid, w, y, T, C, n_norm, eps);
    return int(cudaGetLastError());
}

// 1 トークンずつ T 回（状態はレジスタのまま）。decode は T = 1、下書きの確かめは T = 1 + 下書きの数
int infer_gdn_decode(const float* c, int64_t c_ld, const float* z, int64_t z_ld,
                     const float* alpha, const float* beta, int64_t ab_ld,
                     const float* ssm_a, const float* dt_bias, const float* norm_w,
                     float* state, const int* slots, const int* slots_out, const int* n_valid, float* y, int64_t y_ld,
                     int B, int T, int Hk, int Hv, float eps, cudaStream_t s) {
    if (Hv % Hk || c_ld % 4 || z_ld % 4 || y_ld % 4 || T < 1) return -2;
    if (!al16(c) || !al16(z) || !al16(y) || !al16(state) || !al16(norm_w) || int64_t(B) * T * Hv > INT32_MAX) return -2;
    if (T == 1) {
        decode_kernel<<<dim3(Hv, B), DEC_NT, 0, s>>>(c, c_ld, z, z_ld, alpha, beta, ab_ld, ssm_a, dt_bias, norm_w, state, slots,
                                                     slots_out, n_valid, y, y_ld, T, Hk, Hv, eps);
    } else {  // 頭を 4 ブロックに分けてトークンごとの同期なし → 出口の正規化をまとめて
        decode_multi_kernel<<<dim3(Hv * DM_PARTS, B), DM_WARPS * 32, 0, s>>>(c, c_ld, alpha, beta, ab_ld, ssm_a, dt_bias, state,
                                                                             slots, slots_out, n_valid, y, y_ld, T, Hk, Hv);
        if (y) norm_kernel<<<(B * T * Hv + 7) / 8, 256, 0, s>>>(y, y_ld, z, z_ld, norm_w, n_valid, T, B * T, Hv, eps);
    }
    return int(cudaGetLastError());
}

int64_t infer_gdn_prefill_ws_bytes(int B, int T, int Hv) { return ws_floats(B, T, Hv) * 4; }

// 塊ごと（64）。ws は infer_gdn_prefill_ws_bytes の大きさ。y は o の置き場にも使う
int infer_gdn_prefill(const float* c, int64_t c_ld, const float* z, int64_t z_ld,
                      const float* alpha, const float* beta, int64_t ab_ld,
                      const float* ssm_a, const float* dt_bias, const float* norm_w,
                      float* state, const int* slots, float* y, int64_t y_ld, float* ws,
                      int B, int T, int Hk, int Hv, float eps, cudaStream_t s) {
    if (Hv % Hk || c_ld % 4 || z_ld % 4 || y_ld % 4 || T < 1) return -2;
    if (!al16(c) || !al16(z) || !al16(y) || !al16(state) || !al16(norm_w) || !al16(ws)) return -2;
    // 1 SM に 2〜3 ブロック載せるため、共有メモリを多めに割り振る（どれも 48KB 以下なので上限の許しは要らない）
    for (const void* f : {(const void*)prep_kernel, (const void*)scan_kernel, (const void*)out_kernel})
        if (const cudaError_t e = cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, 100)) return int(e);
    const int nc = (T + CS - 1) / CS;
    const int64_t n = int64_t(B) * Hv * nc;
    float* ws_w = ws;
    float* ws_q = ws_w + n * CS * D;
    float* ws_u = ws_q + n * CS * D;
    float* ws_a = ws_u + n * CS * D;
    float* ws_g = ws_a + n * CS * CS;
    prep_kernel<<<dim3(nc, Hv, B), 256, PREP_SMEM, s>>>(c, c_ld, alpha, beta, ab_ld, ssm_a, dt_bias, ws_w, ws_q, ws_u, ws_a, ws_g,
                                                        T, Hk, Hv);
    scan_kernel<<<dim3(D / SC_V, Hv, B), 256, SC_SMEM, s>>>(c, c_ld, ws_w, ws_q, ws_u, ws_g, state, slots, y, y_ld, T, Hk, Hv);
    out_kernel<<<dim3(nc, Hv, B), 256, OUT_SMEM, s>>>(ws_u, ws_a, z, z_ld, norm_w, y, y_ld, T, Hv, eps);
    return int(cudaGetLastError());
}

}  // extern "C"
