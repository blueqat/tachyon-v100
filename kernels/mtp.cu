// MTP（下書きの 1 段。Qwen3.8-27B の blk.64）の小さなカーネル。大きな行列は g3（gemv3.cu）、Attention は attn.cu。
//   infer_mtp_prep     語の埋め込み（GGUF の行をその場で戻す）と前の h を、それぞれ RMSNorm × 重み（enorm・hnorm）して
//                      [e ; hh] に並べる（eh_proj の入力。llama.cpp の ggml_concat(e_norm, h_norm, 0) と同じ並び）。h が null の行は 0
//   infer_mtp_rmsnorm  行ごとの RMSNorm × 重み（ggml の rms_norm と同じ: x * (1/sqrt(mean(x²) + eps)) * w）
//   infer_mtp_gather   行を選んで詰める（logits を出す行だけ出力の GEMV に渡す）
//   infer_mtp_topk     行ごとの上位 k（k ≤ 32。logit の大きい順・同点は小さい id）と logsumexp。行を split に分けて → まとめる、の 2 段
//   infer_mtp_copy_rows 行の写し（前の h を升に覚える。番地の表は GPU の上）
#include <cfloat>
#include <climits>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "quant.cuh"

namespace {

// ---------------------------------------------------------------- 足し算（ブロック全体）

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// blockDim.x は 32 の倍数（1024 まで）。sh は 32 個以上
__device__ float block_sum(float v, float* sh) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    v = l < int(blockDim.x >> 5) ? sh[l] : 0.f;
    return warp_sum(v);
}

// ---------------------------------------------------------------- 埋め込み＋norm＋並べ

template <int T>
__device__ void deq_row(const uint8_t* row, int k, float* out) {
    const int n = (k / 32) * 4;  // (かたまり c, sub) の数。1 つで 8 要素
    for (int it = threadIdx.x; it < n; it += blockDim.x) {
        const int c = it >> 2, sub = it & 3;
        float v[8];
        deq8<T>(row, c, sub, v);
#pragma unroll
        for (int j = 0; j < 8; ++j) out[pos8<T>(c, sub, j)] = v[j];
    }
}

// 1 ブロック = 1 行。shared は k 個の f32（戻した埋め込み）
template <int T>
__global__ void __launch_bounds__(256) prep_kernel(const uint8_t* __restrict__ embd, int64_t rb, const int* __restrict__ tok,
                                                   const float* const* __restrict__ hp, const float* __restrict__ enorm,
                                                   const float* __restrict__ hnorm, int k, float eps, float* __restrict__ out) {
    extern __shared__ float e[];
    __shared__ float red[32];
    const int r = blockIdx.x;
    deq_row<T>(embd + int64_t(tok[r]) * rb, k, e);
    __syncthreads();
    float s = 0.f;
    for (int i = threadIdx.x; i < k; i += blockDim.x) s += e[i] * e[i];
    s = block_sum(s, red);
    const float ie = 1.0f / sqrtf(s / float(k) + eps);
    const float* h = hp[r];
    float sh = 0.f;
    if (h)
        for (int i = threadIdx.x; i < k; i += blockDim.x) sh += h[i] * h[i];
    sh = block_sum(sh, red);
    const float ih = 1.0f / sqrtf(sh / float(k) + eps);
    float* o = out + int64_t(r) * 2 * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        o[i] = e[i] * ie * enorm[i];
        o[k + i] = h ? h[i] * ih * hnorm[i] : 0.f;
    }
}

__global__ void __launch_bounds__(256) rmsnorm_kernel(const float* __restrict__ x, int64_t ldx, const float* __restrict__ w, int k,
                                                      float eps, float* __restrict__ y, int64_t ldy) {
    __shared__ float red[32];
    const float* xr = x + int64_t(blockIdx.x) * ldx;
    float s = 0.f;
    for (int i = threadIdx.x; i < k; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, red);
    const float sc = 1.0f / sqrtf(s / float(k) + eps);
    float* yr = y + int64_t(blockIdx.x) * ldy;
    for (int i = threadIdx.x; i < k; i += blockDim.x) yr[i] = xr[i] * sc * w[i];
}

__global__ void gather_kernel(const float* __restrict__ src, int64_t ld, const int* __restrict__ idx, int k, float* __restrict__ dst) {
    const float* s = src + int64_t(idx[blockIdx.x]) * ld;
    float* d = dst + int64_t(blockIdx.x) * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) d[i] = s[i];
}

__global__ void copy_rows_kernel(float* const* __restrict__ dst, const float* const* __restrict__ src, int k) {
    const float* s = src[blockIdx.x];
    float* d = dst[blockIdx.x];
    for (int i = threadIdx.x; i < k; i += blockDim.x) d[i] = s ? s[i] : 0.f;
}

// ---------------------------------------------------------------- 上位 k と logsumexp

__device__ __forceinline__ bool better(float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); }

// 大きい順に並べた K 個（レジスタ）。足すのは末尾に入れて上へ泡立てる（添字が決まっているので配列はレジスタに残る）
template <int K>
struct TopK {
    float v[K];
    int id[K];
    __device__ void init() {
#pragma unroll
        for (int j = 0; j < K; ++j) v[j] = -FLT_MAX, id[j] = INT_MAX;
    }
    __device__ void push(float x, int i) {
        if (!better(x, i, v[K - 1], id[K - 1])) return;
        v[K - 1] = x, id[K - 1] = i;
#pragma unroll
        for (int t = K - 1; t > 0; --t)
            if (better(v[t], id[t], v[t - 1], id[t - 1])) {
                const float tv = v[t];
                v[t] = v[t - 1], v[t - 1] = tv;
                const int ti = id[t];
                id[t] = id[t - 1], id[t - 1] = ti;
            }
    }
};

// 走る (max, Σexp(x − max)) を合わせる
__device__ __forceinline__ void lse_merge(float& m, float& s, float m2, float s2) {
    if (m2 == -INFINITY) return;
    if (m == -INFINITY) { m = m2, s = s2; return; }
    if (m2 > m) { s = s * __expf(m - m2) + s2, m = m2; }
    else s += s2 * __expf(m2 - m);
}

// ブロックの中の 1 番を選ぶ（値の大きい順・同点は小さい id）。shv・shi は 32 個以上
__device__ void block_best(float& v, int& id, float* shv, int* shi) {
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffffu, v, o);
        const int i2 = __shfl_xor_sync(0xffffffffu, id, o);
        if (better(v2, i2, v, id)) v = v2, id = i2;
    }
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) shv[w] = v, shi[w] = id;
    __syncthreads();
    if (l < int(blockDim.x >> 5)) v = shv[l], id = shi[l];
    else v = -FLT_MAX, id = INT_MAX;
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffffu, v, o);
        const int i2 = __shfl_xor_sync(0xffffffffu, id, o);
        if (better(v2, i2, v, id)) v = v2, id = i2;
    }
}

// 1 段目: grid (nsplit, rows)。行 r の列 [c0, c1) の上位 K と (max, Σexp) を作業場へ
template <int K, int NT>
__global__ void __launch_bounds__(NT) topk1_kernel(const float* __restrict__ x, int n, int64_t ld, int nsplit, float* __restrict__ wv,
                                                   int* __restrict__ wi, float2* __restrict__ wl) {
    __shared__ float sv[NT * K];
    __shared__ int si[NT * K];
    __shared__ float shv[32], shm[32], shs[32];
    __shared__ int shi[32];
    const int r = blockIdx.y, sp = blockIdx.x;
    const int per = (n + nsplit - 1) / nsplit;
    const int c0 = sp * per, c1 = min(n, c0 + per);
    const float* xr = x + int64_t(r) * ld;
    TopK<K> t;
    t.init();
    float m = -INFINITY, s = 0.f;
    for (int c = c0 + threadIdx.x; c < c1; c += NT) {
        const float v = xr[c];
        t.push(v, c);
        if (v == -INFINITY) continue;  // exp(-inf - -inf) の NaN を避ける
        if (v > m) s = s * __expf(m - v) + 1.f, m = v;
        else s += __expf(v - m);
    }
    // (max, Σexp) をまとめる
    {
        float mm = m, ss = s;
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            const float m2 = __shfl_xor_sync(0xffffffffu, mm, o), s2 = __shfl_xor_sync(0xffffffffu, ss, o);
            lse_merge(mm, ss, m2, s2);
        }
        const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
        if (l == 0) shm[w] = mm, shs[w] = ss;
        __syncthreads();
        if (threadIdx.x == 0) {
            float M = -INFINITY, S = 0.f;
            for (int i = 0; i < NT / 32; ++i) lse_merge(M, S, shm[i], shs[i]);
            wl[int64_t(r) * nsplit + sp] = make_float2(M, S);
        }
    }
#pragma unroll
    for (int j = 0; j < K; ++j) sv[threadIdx.x * K + j] = t.v[j], si[threadIdx.x * K + j] = t.id[j];
    __syncthreads();
    // K 回: 各スレッドの先頭のうち 1 番を選び、選ばれたスレッドは次へ
    int head = 0;
    for (int j = 0; j < K; ++j) {
        float v = head < K ? sv[threadIdx.x * K + head] : -FLT_MAX;
        int id = head < K ? si[threadIdx.x * K + head] : INT_MAX;
        const int my = id;
        block_best(v, id, shv, shi);
        if (my == id && id != INT_MAX) ++head;
        if (threadIdx.x == 0) {
            wv[(int64_t(r) * nsplit + sp) * K + j] = v;
            wi[(int64_t(r) * nsplit + sp) * K + j] = id;
        }
    }
}

// 2 段目: grid (rows)。nsplit × K 個の候補から上位 K と logsumexp
template <int K>
__global__ void __launch_bounds__(1024) topk2_kernel(const float* __restrict__ wv, const int* __restrict__ wi, const float2* __restrict__ wl,
                                                     int nsplit, int kout, int* __restrict__ oi, float* __restrict__ ov,
                                                     float* __restrict__ olse) {
    __shared__ float shv[32];
    __shared__ int shi[32];
    const int r = blockIdx.x;
    const int nc = nsplit * K;
    // 1 スレッド 1 候補（nc ≤ 1024）
    float v = -FLT_MAX;
    int id = INT_MAX;
    if (int(threadIdx.x) < nc) v = wv[int64_t(r) * nc + threadIdx.x], id = wi[int64_t(r) * nc + threadIdx.x];
    if (threadIdx.x == 0) {
        float M = -INFINITY, S = 0.f;
        for (int i = 0; i < nsplit; ++i) {
            const float2 q = wl[int64_t(r) * nsplit + i];
            lse_merge(M, S, q.x, q.y);
        }
        olse[r] = M + logf(S);
    }
    for (int j = 0; j < kout; ++j) {
        float bv = v;
        int bi = id;
        block_best(bv, bi, shv, shi);
        if (id == bi) v = -FLT_MAX, id = INT_MAX;  // 選ばれた候補を外す
        if (threadIdx.x == 0) oi[int64_t(r) * kout + j] = bi == INT_MAX ? -1 : bi, ov[int64_t(r) * kout + j] = bv;
    }
}

template <int K, int NT>
int topk_launch(const float* x, int n, int64_t ld, int rows, int nsplit, int kout, void* ws, int* oi, float* ov, float* olse,
                cudaStream_t s) {
    float* wv = static_cast<float*>(ws);
    int* wi = reinterpret_cast<int*>(wv + int64_t(rows) * nsplit * K);
    float2* wl = reinterpret_cast<float2*>(wi + int64_t(rows) * nsplit * K);
    topk1_kernel<K, NT><<<dim3(nsplit, rows), NT, 0, s>>>(x, n, ld, nsplit, wv, wi, wl);
    topk2_kernel<K><<<rows, 1024, 0, s>>>(wv, wi, wl, nsplit, kout, oi, ov, olse);
    return int(cudaGetLastError());
}

}  // namespace

extern "C" {

// 戻り値: 0 = 成功、-1 = 型を知らない、-2 = 形が合わない、それ以外は cudaError
// out [rows][2k]: [RMSNorm(embd[tok]) * enorm ; RMSNorm(h) * hnorm]。hp は GPU の上の行の番地の表（null の行は 0）
int infer_mtp_prep(int qtype, const void* embd, int64_t row_bytes, const int* tok, const float* const* hp, const float* enorm,
                   const float* hnorm, int k, int rows, float eps, float* out, cudaStream_t s) {
    if (k % 256 || rows <= 0) return -2;
    const uint8_t* w = static_cast<const uint8_t*>(embd);
    const size_t sh = size_t(k) * 4;
    switch (qtype) {
#define CASE(T) case T: prep_kernel<T><<<rows, 256, sh, s>>>(w, row_bytes, tok, hp, enorm, hnorm, k, eps, out); break;
        CASE(Q_F32) CASE(Q_F16) CASE(Q_BF16) CASE(Q_Q8_0) CASE(Q_Q3_K) CASE(Q_Q4_K) CASE(Q_Q5_K) CASE(Q_Q6_K) CASE(Q_IQ4_NL)
        CASE(Q_IQ4_XS)
#undef CASE
        default: return -1;
    }
    return int(cudaGetLastError());
}

int infer_mtp_rmsnorm(const float* x, int64_t ldx, const float* w, int k, int rows, float eps, float* y, int64_t ldy, cudaStream_t s) {
    if (rows <= 0) return 0;
    rmsnorm_kernel<<<rows, 256, 0, s>>>(x, ldx, w, k, eps, y, ldy);
    return int(cudaGetLastError());
}

int infer_mtp_gather(const float* src, int64_t ld, const int* idx, int k, int rows, float* dst, cudaStream_t s) {
    if (rows <= 0) return 0;
    gather_kernel<<<rows, 256, 0, s>>>(src, ld, idx, k, dst);
    return int(cudaGetLastError());
}

// 前の段の上位 1 位の語を、次の段の入力の語に（GPU の上で。段をまたいで手元に戻らずに続ける）
// dst[r] = map ? map[top[src[r] * kout]] : top[src[r] * kout]（候補が無い -1 は 0 に）
__global__ void feed_kernel(const int* top, int kout, const int* src, const int* map, int n, int* dst) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    int id = top[int64_t(src[r]) * kout];
    if (id < 0) id = 0;
    else if (map) id = map[id];
    dst[r] = id;
}

int infer_mtp_feed(const int* top, int kout, const int* src, const int* map, int n, int* dst, cudaStream_t s) {
    if (n <= 0) return 0;
    feed_kernel<<<(n + 127) / 128, 128, 0, s>>>(top, kout, src, map, n, dst);
    return int(cudaGetLastError());
}

int infer_mtp_copy_rows(float* const* dst, const float* const* src, int k, int rows, cudaStream_t s) {
    if (rows <= 0) return 0;
    copy_rows_kernel<<<rows, 256, 0, s>>>(dst, src, k);
    return int(cudaGetLastError());
}

// ---------------------------------------------------------------- 結び付けの下書き（TACHYON_HASH_RNG・A1）
// 下書きの語を Gumbel-max で: argmax_i [logit_i * invt + G(seed, pos, TAG_SAMPLE=0, id_i)]。G は本体の抜き取り（src/spec.rs の
// philox4x32_10・uniform・gumbel）と同じ式（Philox4x32-10・鍵 (seed, pos, 0, 語)・53 ビットの一様・−ln(−ln U)）。
// 本体も同じ雑音で取るので、P と Q が近いほど同じ語になる（受け取りが上がる）。出す語の分布は本体の P のまま（本体は自分の P で取る）。
// 行 r の結果は oi[r * kout] に書く（上位 k の 1 位の番号を置き換える。値 ov・lse は置き換えない＝確からしさの線はもとの Q のまま）。
struct GParam {
    unsigned long long seed;
    unsigned long long pos;
    float invt;  // 0 ならこの行は置き換えない
    int on;
};

__device__ __forceinline__ void philox(uint32_t c[4], uint32_t k0, uint32_t k1) {
#pragma unroll
    for (int r = 0; r < 10; ++r) {
        if (r > 0) {
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }
        const uint64_t p0 = uint64_t(0xD2511F53u) * c[0];
        const uint64_t p1 = uint64_t(0xCD9E8D57u) * c[2];
        const uint32_t n0 = uint32_t(p1 >> 32) ^ c[1] ^ k0, n1 = uint32_t(p1), n2 = uint32_t(p0 >> 32) ^ c[3] ^ k1, n3 = uint32_t(p0);
        c[0] = n0; c[1] = n1; c[2] = n2; c[3] = n3;
    }
}

__device__ __forceinline__ double gumbel_d(unsigned long long seed, unsigned long long pos, uint32_t id) {
    uint32_t c[4] = {uint32_t(pos), uint32_t(pos >> 32), id, 0u};
    philox(c, uint32_t(seed), uint32_t(seed >> 32));
    const uint64_t bits = ((uint64_t(c[0]) << 32) | c[1]) >> 11;
    const double u = (double(bits) + 0.5) * (1.0 / 9007199254740992.0);
    return -log(-log(u));
}

// float の大小を unsigned の大小に（atomicMax 用）
__device__ __forceinline__ uint32_t ord(float f) {
    const uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__global__ void gumbel_init_kernel(unsigned long long* key, int rows) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) key[r] = 0ull;
}

// grid (nsplit, rows)。鍵 = ord(値) << 32 | (0xFFFFFFFF − 番号)（同点は小さい番号）
__global__ void __launch_bounds__(256) gumbel_max_kernel(const float* __restrict__ x, int n, int64_t ld, const int* __restrict__ map,
                                                         const GParam* __restrict__ gp, unsigned long long* __restrict__ key) {
    const int r = blockIdx.y;
    const GParam g = gp[r];
    if (!g.on) return;
    const int per = (n + gridDim.x - 1) / gridDim.x;
    const int b = blockIdx.x * per, e = min(n, b + per);
    const float* xr = x + int64_t(r) * ld;
    unsigned long long best = 0ull;
    for (int i = b + threadIdx.x; i < e; i += blockDim.x) {
        const float l = xr[i];
        if (!(l > -FLT_MAX)) continue;  // -inf・nan は選ばない
        const uint32_t id = map ? uint32_t(map[i]) : uint32_t(i);
        const float v = float(double(l) * double(g.invt) + gumbel_d(g.seed, g.pos, id));
        const unsigned long long k = (uint64_t(ord(v)) << 32) | uint64_t(0xFFFFFFFFu - uint32_t(i));
        best = k > best ? k : best;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        const unsigned long long t = __shfl_xor_sync(0xffffffffu, best, o);
        best = t > best ? t : best;
    }
    if ((threadIdx.x & 31) == 0 && best) atomicMax(key + r, best);
}

__global__ void gumbel_out_kernel(const unsigned long long* key, const GParam* gp, int rows, int kout, int* oi) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || !gp[r].on || !key[r]) return;
    oi[int64_t(r) * kout] = int(0xFFFFFFFFu - uint32_t(key[r] & 0xFFFFFFFFull));
}

// x [rows][ld]（下書きの logits。map があれば番号 i の語は map[i]）。gp [rows]、ws は rows × 8 バイト以上
int infer_mtp_gumbel(const float* x, int n, int64_t ld, int rows, const int* map, const void* gp, void* ws, int kout, int* oi, cudaStream_t s) {
    if (rows <= 0) return 0;
    if (n <= 0 || kout < 1) return -2;
    auto* key = static_cast<unsigned long long*>(ws);
    const auto* g = static_cast<const GParam*>(gp);
    gumbel_init_kernel<<<(rows + 63) / 64, 64, 0, s>>>(key, rows);
    int ns = (320 + rows - 1) / rows;
    ns = ns < 1 ? 1 : (ns > 64 ? 64 : ns);
    while (ns > 1 && n / ns < 512) --ns;
    gumbel_max_kernel<<<dim3(ns, rows), 256, 0, s>>>(x, n, ld, map, g, key);
    gumbel_out_kernel<<<(rows + 63) / 64, 64, 0, s>>>(key, g, rows, kout, oi);
    return int(cudaGetLastError());
}

// 確かめ用: 1 つの (seed, pos, id) の G（手元の spec::gumbel と同じか）
__global__ void gumbel_probe_kernel(unsigned long long seed, unsigned long long pos, const uint32_t* ids, int n, double* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = gumbel_d(seed, pos, ids[i]);
}
int infer_mtp_gumbel_probe(unsigned long long seed, unsigned long long pos, const uint32_t* ids, int n, double* out, cudaStream_t s) {
    if (n <= 0) return 0;
    gumbel_probe_kernel<<<(n + 127) / 128, 128, 0, s>>>(seed, pos, ids, n, out);
    return int(cudaGetLastError());
}

// 行を分ける数（行が少ないときは多く分けて SM を埋める）
int infer_mtp_topk_split(int n, int rows) {
    int ns = (160 + rows - 1) / rows;
    ns = ns < 1 ? 1 : (ns > 32 ? 32 : ns);
    while (ns > 1 && n / ns < 1024) --ns;
    return ns;
}

// 作業場の大きさ（バイト）。k は 32 まで（中では 1・8・32 のどれかに切り上げる）
int64_t infer_mtp_topk_ws_bytes(int n, int rows, int k) {
    const int K = k <= 1 ? 1 : (k <= 8 ? 8 : 32);
    const int ns = infer_mtp_topk_split(n, rows);
    return int64_t(rows) * ns * K * 8 + int64_t(rows) * ns * 8;
}

// x [rows][ld] の各行の上位 k（oi [rows][k]・ov [rows][k]）と logsumexp（olse [rows]）
int infer_mtp_topk(const float* x, int n, int64_t ld, int rows, int k, void* ws, int64_t ws_bytes, int* oi, float* ov, float* olse,
                   cudaStream_t s) {
    if (k < 1 || k > 32 || rows <= 0 || n <= 0) return -2;
    if (ws_bytes < infer_mtp_topk_ws_bytes(n, rows, k)) return -3;
    const int ns = infer_mtp_topk_split(n, rows);
    if (k <= 1) return topk_launch<1, 256>(x, n, ld, rows, ns, k, ws, oi, ov, olse, s);
    if (k <= 8) return topk_launch<8, 256>(x, n, ld, rows, ns, k, ws, oi, ov, olse, s);
    return topk_launch<32, 128>(x, n, ld, rows, ns, k, ws, oi, ov, olse, s);
}

}  // extern "C"
