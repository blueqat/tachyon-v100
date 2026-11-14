// 普通の Attention（qwen35 の 4 層に 1 つ。llama.cpp の llama_model_qwen35::graph::build_layer_attn と同じ計算）。
//
//   q_full = x·attn_q（頭ごとに [q 256 | gate 256]）、k = x·attn_k、v = x·attn_v
//   q = rope(rms_norm(q)·q_norm)、k = rope(rms_norm(k)·k_norm)  … eps 1e-6、IMRoPE（qwen3vl 型の交互の M-RoPE）を先頭 n_rot=64 次元に
//   o = softmax(q k^T / sqrt(256)) v（因果、GQA は q の頭 h が KV の頭 h / (H/HKV)）
//   out = o * sigmoid(gate)  → attn_output へ
//
// 並び
//   q_full [N][q_ld]（頭 h の q は h*2D から、gate は h*2D + D から）、k [N][k_ld]、v [N][v_ld]（f32）
//   q（rope 後）[N][H][D] fp16
//   KV（層ごと）: kc, vc [n_pages][HKV][PS=64][D] fp16、または int8 ＋ 係数 ks, vs [n_pages][HKV][PS][D/32] f32
//                （トークン×頭×32 要素ごと。llama.cpp の q8_0 と同じ粒度。トークンごと 1 つだと外れ値で誤差が 2.5 倍）
//   slot[t] = 物理ページ × 64 + ページ内の位置（-1 は書かない）
//   block_table [n_seq][bt_ld]: 列の論理ページ → 物理ページ
//   q_start [n_seq+1]: 列 i の新しいトークンは q の行 q_start[i]..q_start[i+1]-1。kv_len[i] は書き込んだ後の KV の長さ。
//   新しいトークン j の位置は kv_len - q_len + j（その位置までの key を見る）。プレフィル・続きの塊・デコード・下書きの確かめを同じ形で
//
// Attention のカーネル（FA1 型、wmma m16n16k16、fp16 で掛けて fp32 で溜める）
//   1 ブロック = 1 つの (列, KV の頭, 行の塊, split)。行は「トークン × 同じ KV を見る q の頭 6 つ」を平らに並べたもの
//   （GQA の 6 頭で K/V の読みを 1 回にする。デコードの 1 トークンは 6 行、下書き 3 トークンは 18 行）
//   KV は C::BK（16 か 32）トークンずつ共有メモリへ。次の塊はレジスタに先読みしておく（V100 は cp.async が無い）
//   int8 の KV は共有メモリに入れるときに 32 要素ごとの係数を掛けて fp16 に戻す（int8 の CfgD・CfgV は 2 塊先まで先読み）
//   S = Q K^T（CfgD・CfgP は Q を wmma の塊のままレジスタに持つ。Q の読みは Volta では bank が必ず 2 重にぶつかるため）
//   → 共有メモリ → softmax（行ごとに走る max・和、fp32）→ P（fp16）→ O += P V（O はレジスタの wmma の塊）
//   O の行ごとの掛け直しは、塊の要素が何行目かを最初に load_matrix_sync で調べておいて使う
//   長い文脈で塊が少ないときは KV を kv_chunk ごとに分け（split-K）、作業場に (o, m, l) を書いて attn_combine で合わせる
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

namespace {

constexpr int D = 256;        // 頭の幅
constexpr int PS = 64;        // ページのトークン数
constexpr int BK = 32;        // split の幅の単位（どの形の KV の塊 C::BK も割り切る）
// 共有メモリの行の幅（half）。wmma の読みの bank のぶつかりが型ごとに違うので別々に（ncu で選んだ値）
#ifndef ATTN_PADQ
#define ATTN_PADQ 8
#endif
#ifndef ATTN_PADK
#define ATTN_PADK 8
#endif
#ifndef ATTN_PADV
#define ATTN_PADV 16
#endif
#ifndef ATTN_PADS
#define ATTN_PADS 4
#endif
constexpr int LDQ = D + ATTN_PADQ;  // Q（matrix_a・row_major）
constexpr int LDK = D + ATTN_PADK;  // K（matrix_b・col_major）
constexpr int LDV = D + ATTN_PADV;  // V（matrix_b・row_major）
constexpr float LOG2E = 1.4426950408889634f;
constexpr int QB = D / 32;    // int8 の係数の数（1 行 = 1 トークン × 1 頭あたり）

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

}  // namespace

// ---------------------------------------------------------------- q/k の正規化・RoPE・KV への書き込み

struct AttnRopeParams {
    const float* q;  // q_full [n][q_ld]。null なら q は飛ばす（K/V を書くだけ）
    const float* k;  // [n][k_ld]
    const float* v;  // [n][v_ld]
    int64_t q_ld, k_ld, v_ld;
    const float* q_norm;  // [D]
    const float* k_norm;  // [D]
    const int32_t* pos;   // pos_mode 1: [n]（文字だけ: t=h=w=pos、e=0）、4: [n][4]（t, h, w, e）
    const int32_t* slot;  // [n]
    void* q_out;          // half [n][H][D]
    void* kc;
    void* vc;
    float* ks;  // int8 のときの係数 [n_pages][HKV][PS][D/32]
    float* vs;
    int32_t n, h, hkv, n_rot;
    int32_t sections[4];
    int32_t pos_mode, kv8;
    float eps, theta;
};

namespace {

// 1 warp = 1 つの (トークン, 頭)。頭は q が H 個、k が HKV 個、v が HKV 個。lane は 8 要素（d = lane*8..+7）
template <bool KV8>
__global__ void __launch_bounds__(256) rope_norm_kernel(const AttnRopeParams a, float theta_scale) {
    const int lane = threadIdx.x & 31;
    const int nq = a.q ? a.h : 0;  // q が null なら K/V を書くだけ
    const int per = nq + 2 * a.hkv;
    const int64_t item = int64_t(blockIdx.x) * 8 + (threadIdx.x >> 5);
    if (item >= int64_t(a.n) * per) return;
    const int t = int(item / per), j = int(item % per) + (a.h - nq);
    int kind, hd;  // 0 = q, 1 = k, 2 = v
    const float* src;
    if (j < a.h) {
        kind = 0, hd = j, src = a.q + t * a.q_ld + int64_t(j) * 2 * D;
    } else if (j < a.h + a.hkv) {
        kind = 1, hd = j - a.h, src = a.k + t * a.k_ld + int64_t(hd) * D;
    } else {
        kind = 2, hd = j - a.h - a.hkv, src = a.v + t * a.v_ld + int64_t(hd) * D;
    }
    float x[8];
    {
        const float4 u = reinterpret_cast<const float4*>(src)[lane * 2];
        const float4 w = reinterpret_cast<const float4*>(src)[lane * 2 + 1];
        x[0] = u.x, x[1] = u.y, x[2] = u.z, x[3] = u.w, x[4] = w.x, x[5] = w.y, x[6] = w.z, x[7] = w.w;
    }
    if (kind < 2) {
        // RMSNorm（ggml: 1/sqrtf(mean + eps) を掛けてから重み）
        float ss = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) ss += x[e] * x[e];
        ss = warp_sum(ss);
        const float r = 1.0f / sqrtf(ss / D + a.eps);
        const float* w = kind == 0 ? a.q_norm : a.k_norm;
#pragma unroll
        for (int e = 0; e < 8; ++e) x[e] = x[e] * r * w[lane * 8 + e];
        // RoPE（NEOX 型: i と i + n_rot/2 の組）。角度は ggml-cuda と同じ pos * powf(theta_scale, i)
        const int half = a.n_rot / 2, hl = half / 8;  // 前半を持つ lane の数
        const bool rot = lane * 8 < a.n_rot;
        const int partner = !rot ? lane : (lane < hl ? lane + hl : lane - hl);
        float xp[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) xp[e] = __shfl_sync(0xffffffff, x[e], partner);
        if (rot) {
            const bool first = lane < hl;
            const int sect = a.sections[0] + a.sections[1] + a.sections[2] + a.sections[3];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int i = (first ? lane : lane - hl) * 8 + e;
                int c = 0;  // どの位置（t, h, w, e）を使うか。IMRoPE: 区画の中で t, h, w を交互に
                if (sect > 0) {
                    const int s = i % sect;
                    if (s % 3 == 1 && s < 3 * a.sections[1]) c = 1;
                    else if (s % 3 == 2 && s < 3 * a.sections[2]) c = 2;
                    else if (s % 3 == 0 && s < 3 * a.sections[0]) c = 0;
                    else c = 3;
                }
                const int p = a.pos_mode == 4 ? a.pos[int64_t(t) * 4 + c] : (c == 3 ? 0 : a.pos[t]);
                const float th = float(p) * powf(theta_scale, float(i));
                float sn, cs;
                sincosf(th, &sn, &cs);
                x[e] = first ? x[e] * cs - xp[e] * sn : xp[e] * sn + x[e] * cs;
            }
        }
    }
    if (kind == 0) {
        __half2 hv[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) hv[e] = __floats2half2_rn(x[2 * e], x[2 * e + 1]);
        __half* dst = static_cast<__half*>(a.q_out) + (int64_t(t) * a.h + hd) * D + lane * 8;
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<uint4*>(hv);
        return;
    }
    const int s = a.slot[t];
    if (s < 0) return;
    const int64_t row = (int64_t(s / PS) * a.hkv + hd) * PS + s % PS;  // ページ内の (頭, 位置) の行
    void* cache = kind == 1 ? a.kc : a.vc;
    if (!KV8) {
        __half2 hv[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) hv[e] = __floats2half2_rn(x[2 * e], x[2 * e + 1]);
        *reinterpret_cast<uint4*>(static_cast<__half*>(cache) + row * D + lane * 8) = *reinterpret_cast<uint4*>(hv);
    } else {  // int8: 32 要素（lane 4 つ）ごとに係数 = その中の最大 / 127（llama.cpp の q8_0 と同じ粒度）
        float am = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) am = fmaxf(am, fabsf(x[e]));
        am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 1));
        am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 2));
        const float sc = am / 127.f;
        uint32_t w[2] = {0, 0};
        if (am > 0.f) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int qv = max(-127, min(127, __float2int_rn(x[e] / sc)));
                w[e / 4] |= uint32_t(qv & 0xff) << (8 * (e % 4));
            }
        }
        *reinterpret_cast<uint2*>(static_cast<int8_t*>(cache) + row * D + lane * 8) = make_uint2(w[0], w[1]);
        if (lane % 4 == 0) (kind == 1 ? a.ks : a.vs)[row * QB + lane / 4] = sc;
    }
}

}  // namespace

// ---------------------------------------------------------------- Attention

struct AttnParams {
    const void* q;  // half [n_tok][H][D]
    const float* gate;  // gate の先頭（q_full + D）。null なら掛けない
    int64_t gate_ld, gate_hs;  // トークンごと・頭ごとの幅（要素）。q_full なら H*2D と 2D
    const void* kc;
    const void* vc;
    const float* ks;
    const float* vs;
    const int32_t* block_table;  // [n_seq][bt_ld]
    const int32_t* q_start;      // [n_seq+1]
    const int32_t* kv_len;       // [n_seq]
    void* out;                   // [n_tok][H*D] f32 か fp16
    float* ws;                   // split の作業場（infer_attn_ws_bytes の大きさ）
    int64_t ws_bytes;
    int32_t bt_ld, n_seq, n_tok, max_q_len;
    int32_t h, hkv, kv8, out_f16;
    int32_t kv_chunk;  // 0 = 分けない、> 0 = この幅（トークン、32 の倍数に切り上げ）ごとに分ける、-1 = 自動
    int32_t max_kv;    // 列の kv_len の最大（自動の split と作業場の大きさに使う。⚠️ 本当の最大より小さいと後ろの key を見落とす）
    float scale;       // 0 なら 1/sqrt(D)
    // 列ごとの kv_len の CPU の写し [n_seq]（任意。null なら全部の列が max_kv とみなす）。あれば自動の split の幅を列の長さで決める
    // （短い列を細かく分けすぎず、長い列を分け足りなくしない）。幅の選び方にだけ使い、split の数と作業場は max_kv で数える
    // （GPU の kv_len と同じ値を渡す。ずれても速さが変わるだけ。0 以下か max_kv より大きい値があれば -2）
    const int32_t* kv_len_host;
};

namespace {

// ROWS: 行の塊、NW: warp の数、BK: KV の塊（トークン）、ORB×OCB: O を 1 warp が受け持つ 16×16 の塊の数（行×列）
// QREG: Q を wmma の塊のままレジスタに持つ（Q の row_major の読みは Volta では bank が必ず 2 重にぶつかるので、
//       塊ごとに読み直さない）。warp は (行の塊, D の 1/KS) を受け持ち、その行の key の塊を全部計算する
template <int ROWS_, int NW, int BK_, int ORB_, int OCB_, bool QREG_>
struct Cfg {
    static constexpr int ROWS = ROWS_, BK = BK_, ORB = ORB_, OCB = OCB_;
    static constexpr bool QREG = QREG_;
    static constexpr int NT = NW * 32;
    static constexpr int MINB = NT == 128 ? 2 : 1;     // 1 SM に載せたいブロックの数
    static constexpr int F = QREG ? ROWS / 16 : (ROWS / 16) * (BK / 16);  // S の受け持ちの数（D 方向を除く）
    static constexpr int KS = NW / F;                  // S を D 方向に分ける数
    static constexpr int NQF = D / KS / 16;            // QREG: 1 warp の Q の塊の数
    static constexpr int OWC = D / 16 / OCB;           // O の warp の並び（列）
    static constexpr int TPR = NT / ROWS;              // softmax で 1 行を受け持つスレッド数
    static constexpr int CPT = BK / TPR;               // 1 スレッドの列数
    static constexpr int LDS = BK + ATTN_PADS;         // S の行（float）
    static constexpr int LDP = 2 * LDS;                // P の行（half）。P は S の 0 番の上に同じ行の幅で書く
    static constexpr int LDO = D + 4;                  // 終わりに O を置く行（float）
    static constexpr int SM_Q = ROWS * LDQ * 2, SM_KV = BK * (LDK + LDV) * 2, SM_S = KS * ROWS * LDS * 4;
    static constexpr int SM_X = ROWS * 3 * 4;  // alpha・(m, l)
    static constexpr bool QALIAS = QREG && SM_S >= SM_Q;  // Q は最初にレジスタへ移すだけなので S の場所を借りる
    static constexpr int SMEM = (QALIAS ? 0 : SM_Q) + SM_KV + SM_S + SM_X;
    static_assert(NW % F == 0 && KS >= 1 && (D / KS) % 16 == 0, "S の割り振り");
    static_assert((ROWS / 16 / ORB) * OWC == NW, "O の割り振り");
    static_assert(TPR >= 1 && TPR <= 32 && 32 % TPR == 0 && BK % TPR == 0 && CPT % 4 == 0, "softmax の割り振り");
    static_assert(ROWS * LDO * 4 <= (QALIAS ? 0 : SM_Q) + SM_KV, "O の置き場");
    static_assert(BK * LDK * 2 >= 256 * 4, "行の表を調べる置き場（K の場所）");
    static_assert(PS % BK == 0 && BK % 16 == 0 && 2 * BK <= NT, "KV の塊");
};

// KV の塊（BK トークン × D）をレジスタに先読みし、後で共有メモリへ。
// int8 は 32 要素ごとの係数を掛けて fp16 にする: 値 + 1152 の fp16 の並びを byte_perm で作って 1152 を引き（整数のまま正確）、
// 係数を fp16 で掛ける
template <class C, bool KV8>
struct KVLoad {
    static constexpr int BK = C::BK, NT = C::NT;
    static constexpr int VPT = KV8 ? 16 : 8;       // 16 バイトの束 1 つの要素数
    static constexpr int NVEC = BK * D / VPT;     // 塊 1 つの束の数
    static constexpr int NV = NVEC / NT;          // 1 スレッドの束の数
    static_assert(NVEC % NT == 0, "束の割り振り");
    uint4 k[NV], v[NV];
    float sk[KV8 ? NV : 1], sv[KV8 ? NV : 1];  // int8: 束の係数（束 16 要素は 32 要素の係数の半分）

    // 塊 j の K（KV = 0）か V（KV = 1）をレジスタへ
    template <int KV>
    __device__ __forceinline__ void load(const AttnParams& a, int seq, int g, int j, int kvl) {
        const int p0 = j * BK;  // 塊の最初のトークン（BK は PS を割るので 1 ページに収まる）
        const int phys = a.block_table[int64_t(seq) * a.bt_ld + p0 / PS];
        const int64_t row0 = (int64_t(phys) * a.hkv + g) * PS + p0 % PS;
        const void* src = KV == 0 ? a.kc : a.vc;
        const float* scs = KV == 0 ? a.ks : a.vs;
        uint4* r = KV == 0 ? k : v;
        float* rs = KV == 0 ? sk : sv;
#pragma unroll
        for (int i = 0; i < NV; ++i) {
            const int vi = threadIdx.x + i * NT, tok = vi / (D / VPT);
            if (p0 + tok < kvl) {
                const int64_t off = row0 * D + int64_t(vi) * VPT;  // 塊は連続（BK*D 要素）
                r[i] = KV8 ? *reinterpret_cast<const uint4*>(static_cast<const int8_t*>(src) + off)
                           : *reinterpret_cast<const uint4*>(static_cast<const __half*>(src) + off);
                if (KV8) rs[i] = scs[(row0 + tok) * QB + (vi % (D / VPT)) / 2];
            } else {  // 長さの外（ページの残りはごみかもしれない。NaN を持ち込まない）
                r[i] = make_uint4(0, 0, 0, 0);
                if (KV8) rs[i] = 0.f;
            }
        }
    }
    __device__ __forceinline__ void load(const AttnParams& a, int seq, int g, int j, int kvl) {
        load<0>(a, seq, g, j, kvl);
        load<1>(a, seq, g, j, kvl);
    }
    __device__ __forceinline__ static void put(__half* dst, const uint4& r, float s) {
        if (!KV8) {
            *reinterpret_cast<uint4*>(dst) = r;
        } else {
            const uint32_t w[4] = {r.x ^ 0x80808080u, r.y ^ 0x80808080u, r.z ^ 0x80808080u, r.w ^ 0x80808080u};
            const __half2 bias = __float2half2_rn(1152.f);  // 0x64XX = 1024 + XX、XX = 値 + 128
            // 係数は fp16 で掛ける。係数が fp16 の正規化数より小さい（6.1e-5 未満）塊は値が全部 7.8e-3 未満なので、
            // 精度が落ちても足りなくなるのは 1e-5 ほど（他の値に比べて無視できる）
            const __half2 s2 = __float2half2_rn(s);
            uint32_t h[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                uint32_t b = __byte_perm(w[e / 2], 0x64646464u, e % 2 ? 0x4342 : 0x4140);
                const __half2 x = __hmul2(__hsub2(*reinterpret_cast<__half2*>(&b), bias), s2);
                h[e] = *reinterpret_cast<const uint32_t*>(&x);
            }
            reinterpret_cast<uint4*>(dst)[0] = make_uint4(h[0], h[1], h[2], h[3]);
            reinterpret_cast<uint4*>(dst)[1] = make_uint4(h[4], h[5], h[6], h[7]);
        }
    }
    __device__ __forceinline__ void store_k(__half* Ks) const {
#pragma unroll
        for (int i = 0; i < NV; ++i) {
            const int vi = threadIdx.x + i * NT, tok = vi / (D / VPT), c = vi % (D / VPT);
            put(Ks + tok * LDK + c * VPT, k[i], KV8 ? sk[i] : 0.f);
        }
    }
    __device__ __forceinline__ void store_v(__half* Vs) const {
#pragma unroll
        for (int i = 0; i < NV; ++i) {
            const int vi = threadIdx.x + i * NT, tok = vi / (D / VPT), c = vi % (D / VPT);
            put(Vs + tok * LDV + c * VPT, v[i], KV8 ? sv[i] : 0.f);
        }
    }
};

template <class C, bool KV8>
__global__ void __launch_bounds__(C::NT, C::MINB) attn_kernel(const AttnParams a, float scale_log2) {
    constexpr int ROWS = C::ROWS, BK = C::BK;
    extern __shared__ __align__(16) uint8_t sm[];
    __half* Ks = reinterpret_cast<__half*>(sm + (C::QALIAS ? 0 : C::SM_Q));
    __half* Vs = Ks + BK * LDK;
    float* Ss = reinterpret_cast<float*>(Vs + BK * LDV);
    __half* Qs = C::QALIAS ? reinterpret_cast<__half*>(Ss) : reinterpret_cast<__half*>(sm);
    __half* Ps = reinterpret_cast<__half*>(Ss);
    float* alpha_s = Ss + C::KS * ROWS * C::LDS;
    float* ml_s = alpha_s + ROWS;

    const int tid = threadIdx.x, warp = tid >> 5;
    const int seq = blockIdx.z, g = blockIdx.y % a.hkv, split = blockIdx.y / a.hkv;
    const int G = a.h / a.hkv;
    const int qs = a.q_start[seq], ql = a.q_start[seq + 1] - qs, kvl = a.kv_len[seq];
    const int rows_total = ql * G, r0 = (gridDim.x - 1 - blockIdx.x) * ROWS;  // 因果で長い塊（後ろ）から先に
    if (r0 >= rows_total) return;
    const int pos0 = kvl - ql;  // 新しいトークン 0 の位置
    const int kv_end = pos0 + (min(r0 + ROWS, rows_total) - 1) / G + 1;
    int kv_lo = 0, kv_hi = kv_end;
    if (a.kv_chunk > 0) {
        kv_lo = split * a.kv_chunk;
        if (kv_lo >= kv_end) return;
        kv_hi = min(kv_end, kv_lo + a.kv_chunk);
    }
    const int j0 = kv_lo / BK, j1 = (kv_hi + BK - 1) / BK;

    // Q の塊（行 r = トークン r/G の、頭 g*G + r%G）
    for (int i = tid; i < ROWS * (D / 8); i += C::NT) {
        const int r = i / (D / 8), c = i % (D / 8), row = r0 + r;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (row < rows_total) {
            const int64_t qi = (int64_t(qs) + row / G) * a.h + g * G + row % G;
            val = reinterpret_cast<const uint4*>(static_cast<const __half*>(a.q) + qi * D)[c];
        }
        *reinterpret_cast<uint4*>(Qs + r * LDQ + c * 8) = val;
    }
    // 塊（accumulator）の要素 e が何行目か: 値 = 行番号 の 16×16 を読ませて調べる（4 ビットずつ詰める。置き場はまだ空の K）
    float* probe = reinterpret_cast<float*>(Ks);
    for (int i = tid; i < 256; i += C::NT) probe[i] = float(i / 16);
    __syncthreads();
    uint32_t rowmap = 0;
    {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> f;
        wmma::load_matrix_sync(f, probe, 16, wmma::mem_row_major);
        static_assert(decltype(f)::num_elements <= 8, "行の表は 8 要素まで");
#pragma unroll
        for (int e = 0; e < f.num_elements; ++e) rowmap |= uint32_t(f.x[e]) << (4 * e);
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o[C::ORB][C::OCB];
#pragma unroll
    for (int i = 0; i < C::ORB; ++i)
#pragma unroll
        for (int jj = 0; jj < C::OCB; ++jj) wmma::fill_fragment(o[i][jj], 0.f);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fq[C::QREG ? C::NQF : 1];
    if constexpr (C::QREG) {
        const int rb = warp % C::F, kq = warp / C::F;
#pragma unroll
        for (int i = 0; i < C::NQF; ++i) wmma::load_matrix_sync(fq[i], Qs + rb * 16 * LDQ + kq * (D / C::KS) + i * 16, LDQ);
    }

    // softmax の受け持ち: 行 srow の列 sc0..sc0+CPT-1。m・l はこの行の TPR スレッドが同じ値を持つ
    const int srow = tid / C::TPR, sc0 = (tid % C::TPR) * C::CPT;
    const int my_row = r0 + srow;
    const bool row_ok = my_row < rows_total;
    const int my_pos = pos0 + my_row / G;
    float m_run = -INFINITY, l_run = 0.f;

    // 先読み: 普通は 1 塊先（cur に j+1 を読む）。int8 の CfgD・CfgV は 1 塊の仕事が軽く読みの待ちが見えるので
    // 2 塊先まで（cur が j+1 を持ち、nx に j+2 を読む。レジスタを 2 組使い、ループは 2 つずつ）
    constexpr bool PF2 = KV8 && !C::QREG;
    KVLoad<C, KV8> LA, LB;
    LA.load(a, seq, g, j0, kvl);
    __syncthreads();  // 行の表を読み終わってから S を使う
    LA.store_k(Ks);
    LA.store_v(Vs);
    __syncthreads();
    if (PF2 && j0 + 1 < j1) LA.load(a, seq, g, j0 + 1, kvl);

    const int ow_r = warp / C::OWC, ow_c = warp % C::OWC;
    auto step = [&](const int j, KVLoad<C, KV8>& cur, KVLoad<C, KV8>& nx) {
        const bool nxt = j + 1 < j1;
        // 次の塊を先読み（V も S の前に出す。S の後に出すとレジスタは減るが読みの待ちが見えて遅い: 8K で 16.6 → 18.2ms）
        if (PF2) {
            if (j + 2 < j1) nx.load(a, seq, g, j + 2, kvl);
        } else if (nxt) {
            cur.load(a, seq, g, j + 1, kvl);
        }
        if constexpr (C::QREG) {  // S = Q K^T（warp ごとに 16 行 × BK、D の 1/KS 分。Q はレジスタ）
            const int rb = warp % C::F, kq = warp / C::F;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> s[BK / 16];
#pragma unroll
            for (int cb = 0; cb < BK / 16; ++cb) wmma::fill_fragment(s[cb], 0.f);
#pragma unroll
            for (int i = 0; i < C::NQF; ++i) {
#pragma unroll
                for (int cb = 0; cb < BK / 16; ++cb) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                    wmma::load_matrix_sync(fb, Ks + cb * 16 * LDK + kq * (D / C::KS) + i * 16, LDK);
                    wmma::mma_sync(s[cb], fq[i], fb, s[cb]);
                }
            }
#pragma unroll
            for (int cb = 0; cb < BK / 16; ++cb)
                wmma::store_matrix_sync(Ss + kq * ROWS * C::LDS + rb * 16 * C::LDS + cb * 16, s[cb], C::LDS, wmma::mem_row_major);
        } else {  // S = Q K^T（warp ごとに 16×16 の塊 1 つ、D を KS に分けた 1 つ分）
            const int f = warp % C::F, kq = warp / C::F;
            const int rb = f / (BK / 16), cb = f % (BK / 16);
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> s;
            wmma::fill_fragment(s, 0.f);
#pragma unroll 4
            for (int kk = kq * (D / C::KS); kk < (kq + 1) * (D / C::KS); kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fa, Qs + rb * 16 * LDQ + kk, LDQ);
                wmma::load_matrix_sync(fb, Ks + cb * 16 * LDK + kk, LDK);
                wmma::mma_sync(s, fa, fb, s);
            }
            wmma::store_matrix_sync(Ss + kq * ROWS * C::LDS + rb * 16 * C::LDS + cb * 16, s, C::LDS, wmma::mem_row_major);
        }
        __syncthreads();
        if (nxt) cur.store_k(Ks);  // K_j はもう要らない
        {  // softmax（log2 の目盛り）
            float sv[C::CPT];
#pragma unroll
            for (int c = 0; c < C::CPT; c += 4) {  // float4 で読む（行の幅 LDS ≡ 16B mod 128B なので bank がぶつからない）
                float4 u = *reinterpret_cast<const float4*>(Ss + srow * C::LDS + sc0 + c);
#pragma unroll
                for (int kq = 1; kq < C::KS; ++kq) {
                    const float4 w = *reinterpret_cast<const float4*>(Ss + kq * ROWS * C::LDS + srow * C::LDS + sc0 + c);
                    u.x += w.x, u.y += w.y, u.z += w.z, u.w += w.w;
                }
                sv[c] = u.x, sv[c + 1] = u.y, sv[c + 2] = u.z, sv[c + 3] = u.w;
            }
            float mx = -INFINITY;
#pragma unroll
            for (int c = 0; c < C::CPT; ++c) {
                const int col = sc0 + c, kp = j * BK + col;
                const float s = sv[c];
                const bool ok = row_ok && kp < kv_hi && kp <= my_pos;
                sv[c] = ok ? s * scale_log2 : -INFINITY;
                mx = fmaxf(mx, sv[c]);
            }
#pragma unroll
            for (int o2 = C::TPR / 2; o2 > 0; o2 >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o2));
            const float m_new = fmaxf(m_run, mx);
            const float m_use = m_new == -INFINITY ? 0.f : m_new;
            const float alpha = exp2f(m_run - m_use);
            float ps = 0.f;
#pragma unroll
            for (int c = 0; c < C::CPT; ++c) {
                sv[c] = exp2f(sv[c] - m_use);
                ps += sv[c];
            }
#pragma unroll
            for (int o2 = C::TPR / 2; o2 > 0; o2 >>= 1) ps += __shfl_xor_sync(0xffffffff, ps, o2);
            l_run = l_run * alpha + ps;
            m_run = m_new;
            __syncwarp();  // 同じ行の S を読み終わってから P を上書き
#pragma unroll
            for (int c = 0; c < C::CPT; c += 2)
                *reinterpret_cast<__half2*>(Ps + srow * C::LDP + sc0 + c) = __floats2half2_rn(sv[c], sv[c + 1]);
            if (tid % C::TPR == 0) alpha_s[srow] = alpha;
        }
        __syncthreads();
        {  // O = O * alpha + P V
#pragma unroll
            for (int i = 0; i < C::ORB; ++i) {
                const float* al = alpha_s + (ow_r * C::ORB + i) * 16;
#pragma unroll
                for (int jj = 0; jj < C::OCB; ++jj)
#pragma unroll
                    for (int e = 0; e < o[i][jj].num_elements; ++e) o[i][jj].x[e] *= al[(rowmap >> (4 * e)) & 15];
            }
#pragma unroll
            for (int kk = 0; kk < BK; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[C::ORB];
#pragma unroll
                for (int i = 0; i < C::ORB; ++i)
                    wmma::load_matrix_sync(fa[i], Ps + (ow_r * C::ORB + i) * 16 * C::LDP + kk, C::LDP);
#pragma unroll
                for (int jj = 0; jj < C::OCB; ++jj) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
                    wmma::load_matrix_sync(fb, Vs + kk * LDV + (ow_c * C::OCB + jj) * 16, LDV);
#pragma unroll
                    for (int i = 0; i < C::ORB; ++i) wmma::mma_sync(o[i][jj], fa[i], fb, o[i][jj]);
                }
            }
        }
        __syncthreads();
        if (nxt) cur.store_v(Vs);
    };
    if constexpr (PF2) {
        for (int j = j0; j < j1; j += 2) {
            step(j, LA, LB);
            if (j + 1 < j1) step(j + 1, LB, LA);
        }
    } else {
        for (int j = j0; j < j1; ++j) step(j, LA, LA);
    }

    // 書き出し: O を共有メモリ（Q・K・V の場所）に置き、行ごとに 1/l と gate を掛けて出す（split なら作業場へ）
    if (tid % C::TPR == 0) ml_s[2 * srow] = m_run, ml_s[2 * srow + 1] = l_run;
    float* Os = reinterpret_cast<float*>(sm);
#pragma unroll
    for (int i = 0; i < C::ORB; ++i)
#pragma unroll
        for (int jj = 0; jj < C::OCB; ++jj)
            wmma::store_matrix_sync(Os + (ow_r * C::ORB + i) * 16 * C::LDO + (ow_c * C::OCB + jj) * 16, o[i][jj], C::LDO,
                                    wmma::mem_row_major);
    __syncthreads();
    const int64_t ws_rows = int64_t(a.n_tok) * a.h;
    for (int i = tid; i < ROWS * (D / 4); i += C::NT) {
        const int r = i / (D / 4), c4 = i % (D / 4), row = r0 + r;
        if (row >= rows_total) continue;
        const int64_t tok = int64_t(qs) + row / G;
        const int h = g * G + row % G;
        const int64_t orow = tok * a.h + h;
        float4 val = *reinterpret_cast<const float4*>(Os + r * C::LDO + c4 * 4);
        if (a.kv_chunk > 0) {
            float* wo = a.ws + (int64_t(split) * ws_rows + orow) * D;
            reinterpret_cast<float4*>(wo)[c4] = val;
            if (c4 == 0) {
                float* wml = a.ws + int64_t(gridDim.y / a.hkv) * ws_rows * D + (int64_t(split) * ws_rows + orow) * 2;
                wml[0] = ml_s[2 * r], wml[1] = ml_s[2 * r + 1];
            }
            continue;
        }
        const float il = 1.f / ml_s[2 * r + 1];
        val.x *= il, val.y *= il, val.z *= il, val.w *= il;
        if (a.gate) {
            const float4 gt = *reinterpret_cast<const float4*>(a.gate + tok * a.gate_ld + h * a.gate_hs + c4 * 4);
            val.x /= 1.f + expf(-gt.x), val.y /= 1.f + expf(-gt.y), val.z /= 1.f + expf(-gt.z), val.w /= 1.f + expf(-gt.w);
        }
        if (a.out_f16) {
            __half2 hv[2] = {__floats2half2_rn(val.x, val.y), __floats2half2_rn(val.z, val.w)};
            reinterpret_cast<uint2*>(static_cast<__half*>(a.out) + orow * D)[c4] = *reinterpret_cast<uint2*>(hv);
        } else {
            reinterpret_cast<float4*>(static_cast<float*>(a.out) + orow * D)[c4] = val;
        }
    }
}

// split の合わせ: 1 ブロック = 1 つの (トークン, 頭)。256 スレッド = 64 組（float4 で 4 要素）× 4（split を 4 つおきに受け持つ）
// split が多くても（B=1 の長い文脈で 40 ほど）読みが 1 本の鎖にならないように、split の方向にも並べる
__global__ void __launch_bounds__(256) attn_combine_kernel(const AttnParams a, int n_split) {
    __shared__ float4 acc_s[3][64];
    __shared__ float l_s[3][64];
    const int tid = threadIdx.x, c4 = tid % 64, sg = tid / 64;
    const int64_t row = blockIdx.x;
    const int64_t ws_rows = int64_t(a.n_tok) * a.h;
    const int tok = int(row / a.h), h = int(row % a.h);
    int lo = 0, hi = a.n_seq - 1;  // tok を含む列（q_start[s] <= tok の最後）
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (a.q_start[mid] <= tok) lo = mid;
        else hi = mid - 1;
    }
    const int ql = a.q_start[lo + 1] - a.q_start[lo];
    const int pos = a.kv_len[lo] - ql + (tok - a.q_start[lo]);
    const int ns = min(n_split, (pos + a.kv_chunk) / a.kv_chunk);  // この行が見る split（どれも 1 つ以上の key がある）
    const float* wml = a.ws + int64_t(n_split) * ws_rows * D;
    float M = -INFINITY;
#pragma unroll 8
    for (int s = 0; s < ns; ++s) M = fmaxf(M, wml[(s * ws_rows + row) * 2]);
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    float L = 0.f;
#pragma unroll 4
    for (int s = sg; s < ns; s += 4) {
        const float w = exp2f(wml[(s * ws_rows + row) * 2] - M);
        L += w * wml[(s * ws_rows + row) * 2 + 1];
        const float4 u = reinterpret_cast<const float4*>(a.ws + (s * ws_rows + row) * D)[c4];
        acc.x += w * u.x, acc.y += w * u.y, acc.z += w * u.z, acc.w += w * u.w;
    }
    if (sg > 0) acc_s[sg - 1][c4] = acc, l_s[sg - 1][c4] = L;
    __syncthreads();
    if (sg > 0) return;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const float4 u = acc_s[i][c4];
        acc.x += u.x, acc.y += u.y, acc.z += u.z, acc.w += u.w;
        L += l_s[i][c4];
    }
    const float il = 1.f / L;
    float y[4] = {acc.x * il, acc.y * il, acc.z * il, acc.w * il};
    if (a.gate) {
        const float4 gt = *reinterpret_cast<const float4*>(a.gate + int64_t(tok) * a.gate_ld + int64_t(h) * a.gate_hs + c4 * 4);
        y[0] /= 1.f + expf(-gt.x), y[1] /= 1.f + expf(-gt.y), y[2] /= 1.f + expf(-gt.z), y[3] /= 1.f + expf(-gt.w);
    }
    if (a.out_f16) {
        __half2 hv[2] = {__floats2half2_rn(y[0], y[1]), __floats2half2_rn(y[2], y[3])};
        reinterpret_cast<uint2*>(static_cast<__half*>(a.out) + row * D)[c4] = *reinterpret_cast<uint2*>(hv);
    } else {
        reinterpret_cast<float4*>(static_cast<float*>(a.out) + row * D)[c4] = make_float4(y[0], y[1], y[2], y[3]);
    }
}

}  // namespace

// ---------------------------------------------------------------- 起動

namespace {

using CfgD = Cfg<16, 4, 32, 1, 4, true>;   // デコード（1 列 6 行 = 1〜2 トークン）。Q はレジスタ。44KB → 1 SM に 2 ブロック
using CfgV = Cfg<32, 4, 16, 2, 4, false>;  // 下書きの確かめ（3〜5 トークン）。KV の塊を 16 にして 39KB → 1 SM に 2 ブロック
using CfgP = Cfg<64, 8, 32, 2, 4, true>;  // プレフィル・続きの塊・長い確かめ。Q はレジスタ。86KB → 1 SM に 1 ブロック

// GPU ごとに 1 度だけ聞く・取る（2 枚で動かすとき、どの GPU でも正しく）。cudaGetDevice は今のスレッドの GPU
constexpr int MAXDEV = 64;
int cur_dev() {
    int dev = 0;
    cudaGetDevice(&dev);
    return dev >= 0 && dev < MAXDEV ? dev : 0;
}
int n_sm() {
    static std::atomic<int> n[MAXDEV] = {};
    const int dev = cur_dev();
    int v = n[dev].load(std::memory_order_relaxed);
    if (v <= 0) {
        if (cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || v <= 0) v = 80;
        n[dev].store(v, std::memory_order_relaxed);
    }
    return v;
}

struct Plan {
    int cfg, chunk, n_split;  // cfg 0 = CfgD、1 = CfgV、2 = CfgP
};

// 列の長さ（kv_len_host か、無ければ全部 max_kv）から数える: 幅 c で分けたときのブロック数（頭 1 つあたり）
struct Lens {
    const int32_t* p;  // null なら全部 lmax
    int n, lmax;
    int64_t sum() const {
        if (!p) return int64_t(n) * lmax;
        int64_t s = 0;
        for (int i = 0; i < n; ++i) s += std::max(p[i], 0);
        return s;
    }
    int64_t blocks(int c) const {  // Σ ceil(L_i / c)（長さ 0 の列もブロックは 1 つ起きる）
        if (!p) return int64_t(n) * std::max(1, (lmax + c - 1) / c);
        int64_t b = 0;
        for (int i = 0; i < n; ++i) b += std::max(1, (p[i] + c - 1) / c);
        return b;
    }
};

// 形と split の幅を決める（split の幅 c はどの列にも同じ。列 i は ceil(L_i / c) 個に分かれ、どのブロックも c トークン以下）。
// 列ごとに行の塊が 1〜2 つ（デコード・確かめ）でブロックが常駐できる数に届かなければ、1 波に収まるいちばん細かい幅にする
//   （列の長さで数えるので、短い列は細かく分けず、長い列を分け足りなくしない。全部の列が同じ長さなら前と同じ
//   「1 波になるまで分ける」。塊 2 つの確かめは前の「常駐数の半分に届かないときだけ」と同じ数になる）。
// 塊 1 つで届くなら、波の端数が小さくなる幅にする（長い列の 1..8 等分と、仕事を常駐数で割った幅を比べる。見積もりは
//   「波の数 × ブロックの平均の仕事」と「いちばん長いブロック」の大きい方）。
// 行の塊が多い（プレフィル）ときは、ブロックが常駐数の半分にも届かないときだけ 1 波まで分ける。split の幅は 32 の倍数で 64 以上
Plan plan(const AttnParams& a) {
    const int G = a.h / a.hkv, rows = a.max_q_len * G;
    const int cfg = rows <= CfgD::ROWS ? 0 : rows <= CfgV::ROWS ? 1 : 2;
    const int R = cfg == 0 ? CfgD::ROWS : cfg == 1 ? CfgV::ROWS : CfgP::ROWS;
    const int tiles = (rows + R - 1) / R;
    Lens ln{a.kv_len_host, a.n_seq, a.max_kv};
    if (a.kv_len_host) {
        int m = 1;
        for (int i = 0; i < a.n_seq; ++i) m = std::max(m, a.kv_len_host[i]);
        ln.lmax = m;
    }
    const int lmax = ln.lmax;
    auto round = [](int64_t c) { return int(std::max<int64_t>(2 * BK, (c + BK - 1) / BK * BK)); };
    int chunk = a.kv_chunk;
    if (chunk < 0) {
        const int64_t hb = int64_t(a.hkv) * tiles;  // 列 1 つ・split 1 つあたりのブロック
        const int64_t base = int64_t(a.n_seq) * hb;
        const int64_t resident = int64_t(n_sm()) * (cfg == 0 ? CfgD::MINB : cfg == 1 ? CfgV::MINB : CfgP::MINB);
        chunk = 0;
        if (tiles <= 2 && base < resident) {  // 行の塊 2 つまで = 確かめ（q_len ≦ 21）も、どの塊もほぼ全部の key を見る
            // 1 波に収まるいちばん細かい幅（ブロック数は幅が広いほど減るので二分探索）
            int lo = 2 * BK, hi = round(lmax);
            if (ln.blocks(lo) * hb > resident) {
                while (hi - lo > BK) {
                    const int mid = (lo + hi) / 2 / BK * BK;
                    if (ln.blocks(mid) * hb <= resident) hi = mid;
                    else lo = mid;
                }
                lo = hi;
            }
            chunk = lo < lmax ? lo : 0;
        } else if (tiles == 1) {
            const double need = cfg == 2 ? 0.9 : 0.75;  // 2 ブロック載る形は少し空いても HBM を使い切る
            const double work = double(ln.sum()) * hb;  // 全部のブロックの仕事（トークン）
            // 見積もり: max(いちばん長いブロック, 波の数 × 平均の仕事)。全部の列が同じ長さなら「波の数 × 幅」
            auto est = [&](int c) {
                const int64_t b = ln.blocks(c) * hb, w = (b + resident - 1) / resident;
                return std::max(double(std::min(c, lmax)), double(w) * work / double(b));
            };
            const double ideal = work / double(resident);
            int best_c = lmax;
            double best = est(lmax);
            bool done = ideal / best >= need;
            for (int n = 2; n <= 8 && !done; ++n) {
                const int c = round((int64_t(lmax) + n - 1) / n);
                const double t = est(c);
                if (t < best - 1e-9) best = t, best_c = c;
                if (ideal / t >= need) done = true;
            }
            // 長い列が全体の仕事を決めるとき（列の長さがばらばら）は、仕事を常駐数で割った幅も見る
            const int cb = round(int64_t(ideal));
            if (!done && cb < best_c && est(cb) < best - 1e-9) best_c = cb;
            chunk = best_c < lmax ? best_c : 0;
        } else if (2 * base <= resident) {
            const int ns = int(resident / base);
            if (ns > 1) {
                chunk = round((int64_t(lmax) + ns - 1) / ns);
                if (chunk >= lmax) chunk = 0;
            }
        }
    } else if (chunk > 0) {
        chunk = (chunk + BK - 1) / BK * BK;
    }
    // split の数は max_kv で数える（kv_len_host は幅の選び方にだけ使う。写しが GPU の kv_len とずれても速さが変わるだけで、
    // 答えは max_kv ≧ 本当の最大なら正しい）
    return {cfg, chunk, chunk > 0 ? (a.max_kv + chunk - 1) / chunk : 1};
}

int64_t ws_need(const AttnParams& a, const Plan& p) {
    return p.chunk > 0 ? int64_t(p.n_split) * a.n_tok * a.h * (D + 2) * 4 : 0;
}

template <class C, bool KV8>
void launch_attn(const AttnParams& a, const Plan& p, cudaStream_t s) {
    auto kern = attn_kernel<C, KV8>;
    // 共有メモリの上限を上げる（形ごと・GPU ごとに 1 回。cudaFuncSetAttribute は今の GPU にだけ効く）
    static std::atomic<bool> init[MAXDEV] = {};
    const int dev = cur_dev();
    if (!init[dev].load(std::memory_order_acquire)) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
        cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        init[dev].store(true, std::memory_order_release);
    }
    AttnParams b = a;
    b.kv_chunk = p.chunk;
    const int G = a.h / a.hkv;
    const dim3 grid((a.max_q_len * G + C::ROWS - 1) / C::ROWS, a.hkv * p.n_split, a.n_seq);
    const float scale = a.scale > 0.f ? a.scale : 1.0f / sqrtf(float(D));
    kern<<<grid, C::NT, C::SMEM, s>>>(b, scale * LOG2E);
}

__global__ void fill_kernel(void* p, int64_t n, int kind, uint32_t seed, float scale) {
    for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        uint32_t x = uint32_t(i) * 2654435761u ^ (uint32_t(i >> 32) * 40503u) ^ seed;
        x ^= x >> 15, x *= 2246822519u, x ^= x >> 13, x *= 3266489917u, x ^= x >> 16;
        const float u = (x >> 8) * (1.0f / 16777216.0f) * 2.f - 1.f;  // [-1, 1)
        if (kind == 0) static_cast<__half*>(p)[i] = __float2half_rn(u * scale);
        else if (kind == 1) static_cast<int8_t*>(p)[i] = int8_t(__float2int_rn(u * 127.f));
        else static_cast<float*>(p)[i] = u * scale;
    }
}

}  // namespace

extern "C" {

// 戻り値: 0 = 成功、-2 = 形が合わない、それ以外は cudaError
int infer_attn_rope(const AttnRopeParams* p, cudaStream_t s) {
    const AttnRopeParams& a = *p;
    if (a.n_rot % 16 || a.n_rot > D || a.h <= 0 || a.hkv <= 0 || a.q_ld % 4 || a.k_ld % 4 || a.v_ld % 4) return -2;
    if (a.n <= 0) return 0;
    const float theta_scale = powf(a.theta, -2.0f / float(a.n_rot));  // ggml-cuda と同じく CPU 側で
    const int64_t items = int64_t(a.n) * ((a.q ? a.h : 0) + 2 * a.hkv);
    if ((items + 7) / 8 > 0x7fffffff) return -2;
    const unsigned grid = unsigned((items + 7) / 8);
    if (a.kv8) rope_norm_kernel<true><<<grid, 256, 0, s>>>(a, theta_scale);
    else rope_norm_kernel<false><<<grid, 256, 0, s>>>(a, theta_scale);
    return int(cudaGetLastError());
}

// split の作業場の大きさ（バイト）。0 なら要らない
int64_t infer_attn_ws_bytes(const AttnParams* p) { return ws_need(*p, plan(*p)); }

// 選んだ split の幅（トークン、0 = 分けない）。速さの記録用
int infer_attn_chunk(const AttnParams* p) { return plan(*p).chunk; }

// 戻り値: 0 = 成功、-2 = 形が合わない、-3 = 作業場が足りない、それ以外は cudaError
int infer_attn(const AttnParams* p, cudaStream_t s) {
    const AttnParams& a = *p;
    if (a.h <= 0 || a.hkv <= 0 || a.h % a.hkv || a.max_q_len <= 0 || a.n_seq <= 0 || a.n_seq > 65535 || a.max_kv <= 0) return -2;
    if (a.gate && (a.gate_ld % 4 || a.gate_hs % 4)) return -2;
    if (a.kv_len_host)
        for (int i = 0; i < a.n_seq; ++i)
            if (a.kv_len_host[i] <= 0 || a.kv_len_host[i] > a.max_kv) return -2;  // max_kv が本当の最大より小さい（後ろの key を見落とす）
    const Plan pl = plan(a);
    if (pl.n_split > 65535 / a.hkv) return -2;
    const int64_t need = ws_need(a, pl);
    if (need > 0 && (!a.ws || a.ws_bytes < need)) return -3;
    const bool kv8 = a.kv8 != 0;
    if (pl.cfg == 0) kv8 ? launch_attn<CfgD, true>(a, pl, s) : launch_attn<CfgD, false>(a, pl, s);
    else if (pl.cfg == 1) kv8 ? launch_attn<CfgV, true>(a, pl, s) : launch_attn<CfgV, false>(a, pl, s);
    else kv8 ? launch_attn<CfgP, true>(a, pl, s) : launch_attn<CfgP, false>(a, pl, s);
    if (pl.chunk > 0) {
        AttnParams b = a;
        b.kv_chunk = pl.chunk;
        const int64_t rows = int64_t(a.n_tok) * a.h;
        attn_combine_kernel<<<unsigned(rows), 256, 0, s>>>(b, pl.n_split);
    }
    return int(cudaGetLastError());
}

// 試し用のでたらめ（kind 0 = fp16、1 = int8（-127..127）、2 = f32。値は [-scale, scale)）
int infer_attn_fill(void* p, int64_t n, int kind, uint32_t seed, float scale, cudaStream_t s) {
    fill_kernel<<<1024, 256, 0, s>>>(p, n, kind, seed, scale);
    return int(cudaGetLastError());
}

}  // extern "C"
