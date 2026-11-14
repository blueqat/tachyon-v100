// デコード用 GEMV の 3 版（並びと考え方は g3.cuh の頭）。密なモデル（Qwen3.8-27B）向け。
//   1) infer_g3_repack: GGUF の並び → 自前の並び（読み込むときに 1 度だけ。Q3_K・IQ3_S は Q6_K に直す）
//   2) infer_g3_xprep:  x（f32）→ f16 の上下と、32 要素ごとの (Σx, 2 のべきの倍率の逆数)（何本の GEMV にも使い回す）
//   3) infer_g3_gemv:   y = W x（B = 1..32 本。add なら y += W x）。infer_g3_gemv_multi は同じ x を掛ける行列を 4 つまで 1 回で。
//      足し込みは既定で「決まった順」（毎回同じ答え）: _ws 版に作業場（infer_g3_ws_bytes）を渡すと k をブロックで分けても速いまま
//      （infer_g3_set_det(0) か INFER_G3_DET=0 で atomicAdd の道。少し速いが下の桁が毎回揺れる）
//   4) infer_g3_swiglu: h = SiLU(W_gate x) * (W_up x)（gate と up を 1 回で。down に渡す x の下ごしらえまで書ける）
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include "g3.cuh"

using namespace g3;

namespace {

// ================================================================ 詰め直し
__device__ void put_nib_k(uint8_t* dst, const int* q) {  // K 系: 32 要素の下 4 ビット → 16 B（nib_elem の順）
    uint32_t w[4];
#pragma unroll
    for (int ww = 0; ww < 4; ++ww) {
        uint32_t v = 0;
#pragma unroll
        for (int p = 0; p < 8; ++p) v |= uint32_t(q[8 * ww + nib_elem(p)] & 0xf) << (4 * p);
        w[ww] = v;
    }
    *reinterpret_cast<uint4*>(dst) = make_uint4(w[0], w[1], w[2], w[3]);
}
__device__ void put_nib_iq4(uint8_t* dst, const int* q) {  // IQ4: 32 要素の表の番号 → 16 B（そのままの順）
    uint32_t w[4];
#pragma unroll
    for (int ww = 0; ww < 4; ++ww) {
        uint32_t v = 0;
#pragma unroll
        for (int p = 0; p < 8; ++p) v |= uint32_t(q[8 * ww + p] & 0xf) << (4 * p);
        w[ww] = v;
    }
    *reinterpret_cast<uint4*>(dst) = make_uint4(w[0], w[1], w[2], w[3]);
}
// 要素 e の上のビットの位置（回し量 r）
__device__ __forceinline__ int hpos(int e, int r) {
    const int ee = e & 7, t = (ee & 3) >> 1, which = ee & 1;
    return (4 + 4 * t + 16 * which + r) & 31;
}
__device__ uint32_t h5_word(const int* q) {  // 32 要素の 5 ビット目 → 1 語
    uint32_t h = 0;
    for (int e = 0; e < 32; ++e) {
        const int ww = e >> 3, hh = (e & 7) >> 2;
        h |= uint32_t((q[e] >> 4) & 1) << hpos(e, rot5(2 * ww + hh));
    }
    return h;
}
__device__ uint2 h6_words(const int* q) {  // 32 要素の上 2 ビット → 2 語
    uint32_t h[2] = {0, 0};
    for (int e = 0; e < 32; ++e) {
        const int ww = e >> 3, hh = (e & 7) >> 2;
        const int b = hpos(e, rot6(2 * (ww & 1) + hh));
        const uint32_t v = (q[e] >> 4) & 3;
        h[ww >> 1] |= (v & 1) << b;
        h[ww >> 1] |= (v >> 1) << ((b + 1) & 31);
    }
    return make_uint2(h[0], h[1]);
}
__device__ __forceinline__ uint32_t rd32(const uint8_t* p) { return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24); }
__device__ __forceinline__ uint16_t rd16(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }

// Q6_K の並びに直す元（Q6_K・Q3_K・IQ3_S）: 単位 1 つの q6（0..63）を かたまり c ごとに、scale 16 個と d
__device__ void q6_chunk(int T, const uint8_t* b, int c, int* q) {
    if (T == Q_Q6_K) {
        const int nn = c >> 2, quarter = c & 3;
        for (int l = 0; l < 32; ++l) {
            const int lo = (b[64 * nn + 32 * (quarter & 1) + l] >> (4 * (quarter >> 1))) & 0xf;
            const int hi = (b[128 + 32 * nn + l] >> (2 * quarter)) & 3;
            q[l] = lo | (hi << 4);
        }
    } else if (T == Q_Q3_K) {
        // hmask(32) qs(64) scales(12) d。値 = ((qs >> 2j) & 3) - (hmask のビットが無ければ 4)
        const int n = c >> 2, j = c & 3;
        for (int l = 0; l < 32; ++l) {
            const int lo = (b[32 + 32 * n + l] >> (2 * j)) & 3;
            const int hb = (b[l] >> (4 * n + j)) & 1;
            q[l] = lo - (hb ? 0 : 4) + 32;
        }
    } else {  // IQ3_S: d(2) qs(64) qh(8) signs(32) scales(4)。値 = ±格子の値
        const uint8_t* qs = b + 2 + 8 * c;
        const int qh = b[66 + c];
        const uint8_t* sg = b + 74 + 4 * c;
        for (int l = 0; l < 32; ++l) {
            const int l8 = l >> 3, j8 = l & 7, h = j8 >> 2;
            const int idx = qs[2 * l8 + h] | ((qh << (8 - 2 * l8 - h)) & 256);
            const int v = (iq3s_grid[idx] >> (8 * (j8 & 3))) & 0xff;
            q[l] = ((sg[l8] >> j8) & 1 ? -v : v) + 32;
        }
    }
}
__device__ void q6_scales(int T, const uint8_t* b, int8_t* sc, uint16_t& d) {
    if (T == Q_Q6_K) {
        for (int i = 0; i < 16; ++i) sc[i] = int8_t(b[192 + i]);
        d = rd16(b + 208);
    } else if (T == Q_Q3_K) {
        uint32_t aux[4] = {rd32(b + 96), rd32(b + 100), rd32(b + 104), 0};
        const uint32_t km1 = 0x03030303u, km2 = 0x0f0f0f0fu, tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & km2) | (((tmp >> 4) & km1) << 4);
        aux[3] = ((aux[1] >> 4) & km2) | (((tmp >> 6) & km1) << 4);
        aux[0] = (aux[0] & km2) | (((tmp >> 0) & km1) << 4);
        aux[1] = (aux[1] & km2) | (((tmp >> 2) & km1) << 4);
        for (int i = 0; i < 16; ++i) sc[i] = int8_t(int((aux[i / 4] >> (8 * (i % 4))) & 0xff) - 32);
        d = rd16(b + 108);
    } else {
        for (int i = 0; i < 16; ++i) sc[i] = int8_t(1 + 2 * ((b[106 + i / 4] >> (4 * ((i / 2) & 1))) & 0xf));
        d = rd16(b);
    }
}

// 1 スレッド = 1 行 × 1 単位。src は GGUF の並び（行ごとに続く）、dst は自前の並び
template <int T>
__global__ void repack_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, int k, int n) {
    constexpr int P = T == Q_Q3_K || T == Q_IQ3_S ? Q_Q6_K : T;  // 詰め直した後の型
    const int nu = k / UK;
    const int n_tiles = (n + TILE - 1) / TILE;
    const int64_t id = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= int64_t(n_tiles) * TILE * nu) return;
    const int u = int(id % nu);
    const int r = int(id / nu);  // 切り上げた分も含む行
    const int tile = r / TILE, slot = slot_of_row(r % TILE);
    uint8_t* o = dst + int64_t(tile) * tile_bytes(P, k) + int64_t(u) * unit_bytes(P);
    const bool real = r < n;
    const uint8_t* s = src + (real ? int64_t(r) * qtype_row_bytes(T, k) : 0);
    int q[32];
    if (!real) {
        for (int i = 0; i < 32; ++i) q[i] = 0;
    }
    if (T == Q_Q8_0) {
        for (int c = 0; c < 8; ++c) {
            const uint8_t* b = s + (u * 8 + c) * 34;
            __align__(16) uint8_t v[32];
            for (int i = 0; i < 32; ++i) v[i] = real ? uint8_t(int8_t(b[2 + i]) + 128) : 0;
            *reinterpret_cast<uint4*>(o + c * 1024 + slot * 16) = *reinterpret_cast<const uint4*>(v);
            *reinterpret_cast<uint4*>(o + c * 1024 + 512 + slot * 16) = *reinterpret_cast<const uint4*>(v + 16);
            *reinterpret_cast<uint16_t*>(o + 8192 + slot * 16 + c * 2) = real ? rd16(b) : 0;
        }
    } else if (T == Q_IQ4_NL) {
        for (int c = 0; c < 8; ++c) {
            const uint8_t* b = s + (u * 8 + c) * 18;
            if (real)
                for (int l = 0; l < 32; ++l) q[l] = l < 16 ? b[2 + l] & 0xf : b[2 + l - 16] >> 4;
            put_nib_iq4(o + c * 512 + slot * 16, q);
            *reinterpret_cast<uint16_t*>(o + 4096 + slot * 16 + c * 2) = real ? rd16(b) : 0;
        }
    } else if (T == Q_IQ4_XS) {
        const uint8_t* b = s + u * 136;
        for (int c = 0; c < 8; ++c) {
            if (real)
                for (int l = 0; l < 32; ++l) q[l] = l < 16 ? b[8 + 16 * c + l] & 0xf : b[8 + 16 * c + l - 16] >> 4;
            put_nib_iq4(o + c * 512 + slot * 16, q);
        }
        *reinterpret_cast<uint2*>(o + 4096 + slot * 8) = real ? make_uint2(rd32(b), rd32(b + 4)) : make_uint2(0, 0);
    } else if (T == Q_Q4_K || T == Q_Q5_K) {
        const int bs = T == Q_Q4_K ? 144 : 176;
        const uint8_t* b = s + u * bs;
        const uint8_t* qs = b + (T == Q_Q4_K ? 16 : 48);
        for (int ib = 0; ib < 8; ++ib) {
            if (real)
                for (int l = 0; l < 32; ++l) {
                    int v = (qs[32 * (ib >> 1) + l] >> (4 * (ib & 1))) & 0xf;
                    if (T == Q_Q5_K) v |= ((b[16 + l] >> ib) & 1) << 4;
                    q[l] = v;
                }
            put_nib_k(o + ib * 512 + slot * 16, q);
            if (T == Q_Q5_K) *reinterpret_cast<uint32_t*>(o + 4096 + ib * 128 + slot * 4) = h5_word(q);
        }
        const uint4 hd = real ? make_uint4(rd32(b), rd32(b + 4), rd32(b + 8), rd32(b + 12)) : make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(o + (T == Q_Q4_K ? 4096 : 5120) + slot * 16) = hd;
    } else {  // Q6_K（Q6_K・Q3_K・IQ3_S から）
        const uint8_t* b = s + u * (T == Q_Q6_K ? 210 : 110);
        for (int c = 0; c < 8; ++c) {
            if (real) q6_chunk(T, b, c, q);
            put_nib_k(o + c * 512 + slot * 16, q);
            *reinterpret_cast<uint2*>(o + 4096 + c * 256 + slot * 8) = h6_words(q);
        }
        __align__(16) int8_t sc[16];
        uint16_t d = 0;
        if (real) q6_scales(T, b, sc, d);
        else
            for (int i = 0; i < 16; ++i) sc[i] = 0;
        *reinterpret_cast<uint4*>(o + 6144 + slot * 16) = *reinterpret_cast<const uint4*>(sc);
        *reinterpret_cast<uint16_t*>(o + 6656 + slot * 2) = d;
    }
}

// 確かめ用: 自前の並びから f32 に戻す（1 スレッド = 1 行 × 1 かたまり）。y は [n][k]
template <int T>
__global__ void g3_dequant_kernel(const uint8_t* __restrict__ w, int k, int n, float* __restrict__ y) {
    const int kb = k / 32;
    const int64_t id = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= int64_t(n) * kb) return;
    const int r = int(id / kb), j = int(id % kb);
    const int tile = r / TILE, slot = slot_of_row(r % TILE);
    const uint8_t* u = w + int64_t(tile) * tile_bytes(T, k) + int64_t(j / 8) * unit_bytes(T);
    const typename W<T>::C c = W<T>::ldc(u, j % 8, slot);
    const typename W<T>::S s = W<T>::lds(u, slot);
    uint32_t a[16];
    W<T>::deq(c, a);
    float sc[W<T>::NG], o;
    W<T>::scale(s, j % 8, sc, o);
    for (int st = 0; st < 8; ++st) {
        for (int i = 0; i < 4; ++i) {
            const uint32_t h = a[2 * st + (i >> 1)];
            const float q = __half2float(__ushort_as_half(uint16_t(i & 1 ? h >> 16 : h)));
            const int g = W<T>::NG == 2 ? st >> 2 : 0;
            y[int64_t(r) * k + j * 32 + st * 4 + i] = sc[g] * q + o;
        }
    }
}

// ================================================================ x の下ごしらえ
// x[b][k]（f32）→ xhl[b][2][k]（f16 の上・下）、xsi[b][k/32]（float2: 上+下の和、倍率の逆数）
// 倍率は 32 要素（かたまり）ごとの 2 のべきで、max|x|·倍率 < 2^13（下の f16 が非正規数に落ちないように）。
// かたまりごとなので、x を作る側（RMSNorm・SiLU*up など）がそのまま書ける（全体の最大を待たない。g3::prep8）
// 1 warp = 1 トークンの 256 要素
__global__ void xprep_kernel(const float* __restrict__ x, int k, int B, __half* __restrict__ xhl, float2* __restrict__ xsi) {
    const int64_t w = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31, nu = k / UK;
    if (w >= int64_t(B) * nu) return;  // warp ごと抜ける
    const int t = int(w / nu), u = int(w % nu);
    const float4* xr = reinterpret_cast<const float4*>(x + int64_t(t) * k + u * UK + lane * 8);
    const float4 p = __ldg(xr), q = __ldg(xr + 1);
    const float v[8] = {p.x, p.y, p.z, p.w, q.x, q.y, q.z, q.w};
    __half* hi = xhl + int64_t(t) * 2 * k + u * UK + lane * 8;
    prep8(v, hi, hi + k, xsi + int64_t(t) * (k / 32) + u * 8 + lane / 4, true, (lane & 3) == 0);
}

// ================================================================ GEMV
// NB = 1 warp が扱う本数の上限（2・4 は HL、8・16 は TC）
//   HL: B の列 = (トークン, 上/下) で 1 枚に 4 トークン、1 歩 1 回の mma。D の 2 列（上と下）を足してから倍率を掛ける
//   TC: B の列 = トークンで 1 枚に 8 トークン、上と下で 2 回の mma（同じ A・同じ溜め）。NB/8 枚
//   B > G3_NBMAX は G3_NBMAX 本ずつの組（TG 組）に分け、組ごとに別の warp が同じ重みを読む（2 回目以降は L2 に当たる）
// ブロック = 8 warp = タイル × WK（k を分ける）× TG、1 SM に 1 ブロック（レジスタを溢れさせない方が速い）。
// x は k のひと切れずつ共有メモリに置き、ブロックの全員で使い回す。重みは各 warp が R かたまり先まで読んでおく（ひと切れの境もまたいで）。
template <int NB> struct Mode {
    static constexpr bool HL = NB <= 4;
    static constexpr int NT = HL ? 1 : NB / 8;          // B の枚数
    static constexpr int OUT = NB <= 2 ? 2 : HL ? 4 : NB;  // 1 スレッドが持つ (行, トークン) の数
};
// 出力 j の、タイルの中の行とトークン
template <int NB> __device__ __forceinline__ int out_row(int lane, int j) {
    return d_row0(lane) + (Mode<NB>::HL ? (j & 1) * 2 : ((j >> 1) & 1) * 2);
}
template <int NB> __device__ __forceinline__ int out_tok(int lane, int j) {
    if (Mode<NB>::HL) return ((lane >> 1) & 1) + (j >> 1) * 2;
    const int i = j & 7;
    return (j >> 3) * 8 + 2 * ((lane >> 1) & 1) + (i & 1) + (i >> 2) * 4;
}

struct XArg {
    const __half* xhl;   // [B][2][k]
    const float2* xsi;   // [B][k/32]（Σx, 倍率の逆数）
    int k, B;
    // MoE（infer_g3_moe_gemv）: 本 t の x は xhl・xsi の行 xidx[t]（null なら t）。専門家ごとに選ばれた語を写さずに引く
    const int* xidx;
};
// 共有メモリの x: 行 r = t*2 + p（p = 上・下）の頭 = r*RS バイト。その後ろに (Σx, 倍率の逆数) [t][CK]
// RS ≡ 16（mod 128）で、8 lane の 16 B 読みが別々のバンクに当たる
__host__ __device__ inline int x_rs(int cmax) { return cmax * 512 + 16; }
__host__ __device__ inline int x_smem(int B, int cmax) { return 2 * B * x_rs(cmax) + B * cmax * 8 * 8; }

#ifndef G3_SB
#define G3_SB 8
#endif
// ひと切れ（単位 [a, b)）を共有メモリへ（ブロックの全員で）
__device__ __forceinline__ void stage_x(uint8_t* sm, int RS, int CK, const XArg& xa, int a, int b) {
    // 行 2B 本 × per 個（16 B）をブロックの全員で。読みを G3_SB 個ずつ先に出してから書く
    const int per = (b - a) * 32;  // 1 行の 16 B の数（256 要素 = 512 B）
    const int n = 2 * xa.B * per;
    const uint4* src = reinterpret_cast<const uint4*>(xa.xhl + int64_t(a) * UK);
    const int rstride = xa.k / 8;  // 行の間（16 B 単位）
    for (int i0 = threadIdx.x; i0 < n; i0 += blockDim.x * G3_SB) {
        uint4 v[G3_SB];
        int d[G3_SB];
#pragma unroll
        for (int j = 0; j < G3_SB; ++j) {
            const int i = i0 + j * blockDim.x;
            if (i < n) {
                const int r = i / per, c = i - r * per;
                const int rr = xa.xidx ? 2 * __ldg(xa.xidx + (r >> 1)) + (r & 1) : r;
                v[j] = ld16(src + int64_t(rr) * rstride + c);
                d[j] = r * RS + c * 16;
            }
        }
#pragma unroll
        for (int j = 0; j < G3_SB; ++j)
            if (i0 + j * blockDim.x < n) *reinterpret_cast<uint4*>(sm + d[j]) = v[j];
    }
    float2* xs = reinterpret_cast<float2*>(sm + 2 * xa.B * RS);
    const int kb = xa.k / 32, per2 = (b - a) * 8, n2 = xa.B * per2;
    for (int i = threadIdx.x; i < n2; i += blockDim.x) {
        const int t = i / per2, c = i - t * per2;
        const int tt = xa.xidx ? __ldg(xa.xidx + t) : t;
        xs[t * CK + c] = __ldg(xa.xsi + int64_t(tt) * kb + a * 8 + c);
    }
}

// 1 かたまり（32 要素）ぶん: 戻して、倍率を配り、mma で掛けて、倍率を掛けて acc に足す
//   x の B（上・下）と (Σx, 倍率の逆数) は共有メモリの bp・xp から（jl はひと切れの中のかたまり番号）
template <int T, int NB>
__device__ __forceinline__ void chunk_mma(const uint32_t (&a)[16], const typename W<T>::S& s, int c, const uint8_t* const (&bp)[Mode<NB>::NT][2],
                                          const float2* const (&xp)[Mode<NB>::NT][4], int jl, float (&acc)[Mode<NB>::OUT], int lane) {
    using WT = W<T>;
    using M = Mode<NB>;
    constexpr int NG = WT::NG, NT = M::NT;
    float sc[NG], o;
    WT::scale(s, c, sc, o);
    // D の行 r0 は lane&~2 の、r0+2 は lane|2 の A の行（自分の A の行はどちらか一方）
    float s0[NG], s2[NG], o0 = 0.f, o2 = 0.f;
#pragma unroll
    for (int g = 0; g < NG; ++g) {
        const float x = __shfl_xor_sync(0xffffffffu, sc[g], 2);
        s0[g] = (lane & 2) ? x : sc[g];
        s2[g] = (lane & 2) ? sc[g] : x;
    }
    if (WT::HAS_O) {
        const float x = __shfl_xor_sync(0xffffffffu, o, 2);
        o0 = (lane & 2) ? x : o;
        o2 = (lane & 2) ? o : x;
    }
#pragma unroll
    for (int ct = 0; ct < NT; ++ct) {
        float g[NG][8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {  // 2 歩ずつ
            const int gg = NG == 2 ? i >> 1 : 0;
            const bool first = NG == 2 ? (i & 1) == 0 : i == 0;
            const uint4 bh = *reinterpret_cast<const uint4*>(bp[ct][0] + jl * 64 + 16 * i);
            if (first) mma884z(g[gg], a[4 * i], a[4 * i + 1], bh.x, bh.y);
            else mma884(g[gg], a[4 * i], a[4 * i + 1], bh.x, bh.y);
            if (!M::HL) {
                const uint4 bl = *reinterpret_cast<const uint4*>(bp[ct][1] + jl * 64 + 16 * i);
                mma884(g[gg], a[4 * i], a[4 * i + 1], bl.x, bl.y);
                mma884(g[gg], a[4 * i + 2], a[4 * i + 3], bh.z, bh.w);
                mma884(g[gg], a[4 * i + 2], a[4 * i + 3], bl.z, bl.w);
            } else {
                mma884(g[gg], a[4 * i + 2], a[4 * i + 3], bh.z, bh.w);
            }
        }
        if (M::HL) {
            // g: 0,1 = (r0, tA) の上下 / 2,3 = (r0+2, tA) / 4,5 = (r0, tA+2) / 6,7 = (r0+2, tA+2)
            float2 xv[2];
            xv[0] = xp[0][0][jl];
            if (M::OUT > 2) xv[1] = xp[0][1][jl];
#pragma unroll
            for (int jo = 0; jo < M::OUT; ++jo) {
                float v = 0.f;
#pragma unroll
                for (int gg = 0; gg < NG; ++gg) v += ((jo & 1) ? s2[gg] : s0[gg]) * (g[gg][2 * jo] + g[gg][2 * jo + 1]);
                if (WT::HAS_O) v += ((jo & 1) ? o2 : o0) * xv[jo >> 1].x;
                acc[jo] = fmaf(v, xv[jo >> 1].y, acc[jo]);
            }
        } else {
            // g[i]: 行 r0 + 2·((i>>1)&1)、トークン c0 + (i&1) + 4·(i>>2)
            float2 xv[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) xv[q] = xp[ct][q][jl];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const bool r2 = (i >> 1) & 1;
                const float2 xq = xv[(i & 1) + 2 * (i >> 2)];
                float v = 0.f;
#pragma unroll
                for (int gg = 0; gg < NG; ++gg) v += (r2 ? s2[gg] : s0[gg]) * g[gg][i];
                if (WT::HAS_O) v += (r2 ? o2 : o0) * xq.x;
                acc[ct * 8 + i] = fmaf(v, xq.y, acc[ct * 8 + i]);
            }
        }
    }
}

#ifndef G3_RTC
#define G3_RTC 4
#endif
// ブロックが受け持つ k（単位 [ku0, ku0 + nub)）を ns 切れに分けたときの、切れ s の範囲と、その中の warp kw の範囲
struct KSplit {
    int ku0, nub, ns, wk;
    __device__ __forceinline__ int sa(int s) const { return ku0 + s * nub / ns; }
    __device__ __forceinline__ int wa(int s, int kw) const { const int a = sa(s), n = sa(s + 1) - a; return a + kw * n / wk; }
};

// 試し（-DG3_TIMING）: ブロックごとの時刻（globaltimer, ns。V100 では 1µs 刻み）[入り・最初の x を置き終わり・計算の終わり（いちばん遅い warp）]
#ifdef G3_TIMING
__device__ unsigned long long g3_tm[4096][3];
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ int tm_slot() { return min(int(blockIdx.x + gridDim.x * blockIdx.y), 4095); }
#define G3_TM(i) do { if (threadIdx.x == 0) g3_tm[tm_slot()][i] = gtime(); } while (0)
#define G3_TM_MAX(i) do { if ((threadIdx.x & 31) == 0) atomicMax(&g3_tm[tm_slot()][i], gtime()); } while (0)
#else
#define G3_TM(i) do {} while (0)
#define G3_TM_MAX(i) do {} while (0)
#endif

// 1 warp: 1 タイル（32 行）× 受け持ちの単位 × NB 本。acc に内積を足す。
// ひと切れごとにブロックの全員で x を共有メモリへ置く（valid でない warp も置くのは手伝う）
template <int T, int NB>
__device__ __forceinline__ void warp_run(const uint8_t* tile, bool valid, const KSplit& ks, int kw, int tb, const XArg& xa, uint8_t* sm, int RS,
                                         int CK, bool stage, float (&acc)[Mode<NB>::OUT], int lane) {
    using WT = W<T>;
    using M = Mode<NB>;
    // TC はレジスタが足りないので先読みを 4 かたまりまで
    constexpr int NT = M::NT, R = M::HL ? WT::R : (WT::R < G3_RTC ? WT::R : G3_RTC), UB = unit_bytes(T);
    // 先読みは次の単位の頭までしか回さない（c + R - 8 < 8）。R > 8 は答えが壊れる（M4 の試しで 誤差 1.5・速く見えた）
    static_assert(R >= 1 && R <= 8, "先読みのかたまりの数 R は 1..=8");
    const int cb = b_col(lane);
    const float2* xsb = reinterpret_cast<const float2*>(sm + 2 * xa.B * RS);
    const uint8_t* bp[NT][2];
    const float2* xp[NT][4];
    if (M::HL) {
        const int t = min(cb >> 1, xa.B - 1);
        bp[0][0] = bp[0][1] = sm + (t * 2 + (cb & 1)) * RS;
        const int ta = (lane >> 1) & 1;
        xp[0][0] = xsb + min(ta, xa.B - 1) * CK;
        xp[0][1] = xsb + min(ta + 2, xa.B - 1) * CK;
        xp[0][2] = xp[0][3] = xp[0][0];
    } else {
        const int c0 = 2 * ((lane >> 1) & 1);
#pragma unroll
        for (int ct = 0; ct < NT; ++ct) {
            const int t = min(tb + ct * 8 + cb, xa.B - 1);
            bp[ct][0] = sm + t * 2 * RS;
            bp[ct][1] = bp[ct][0] + RS;
#pragma unroll
            for (int q = 0; q < 4; ++q) xp[ct][q] = xsb + min(tb + ct * 8 + c0 + (q & 1) + (q >> 1) * 4, xa.B - 1) * CK;
        }
    }
    typename WT::C ring[R];
    typename WT::S s_cur, s_nxt;
    // 最初の単位の重みを読み始めてから x を置く
    int u = ks.wa(0, kw);
    const uint8_t* up = tile + int64_t(u) * UB;
    if (valid) {
#pragma unroll
        for (int c = 0; c < R; ++c) ring[c] = WT::ldc(up, c, lane);
        s_cur = WT::lds(up, lane);
    }
    for (int s = 0; s < ks.ns; ++s) {
        const int a = ks.sa(s), ue = ks.wa(s, kw + 1);
        const int head = s + 1 < ks.ns ? ks.wa(s + 1, kw) : -1;  // 次の切れの自分の頭
        if (stage) {
            __syncthreads();  // 前の切れ（または前の行列）を皆が使い終わるまで待つ
            stage_x(sm, RS, CK, xa, a, ks.sa(s + 1));
            __syncthreads();
            if (s == 0) G3_TM(1);
        }
        if (!valid) continue;
        for (; u < ue; ++u) {
            // 次の単位（この切れの中か、次の切れの自分の頭）
            const int un = u + 1 < ue ? u + 1 : head;
            const bool more = un >= 0;
            const uint8_t* np = tile + int64_t(more ? un : u) * UB;
            if (more) s_nxt = WT::lds(np, lane);
            const int jb = (u - a) * 8;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                // 戻してから同じ置き場に次を読む（写しを作らない）
                uint32_t av[16];
                WT::deq(ring[c % R], av);
                if (c + R < 8) ring[c % R] = WT::ldc(up, c + R, lane);
                else if (more) ring[c % R] = WT::ldc(np, c + R - 8, lane);
                chunk_mma<T, NB>(av, s_cur, c, bp, xp, jb + c, acc, lane);
            }
            s_cur = s_nxt;
            up = np;
        }
        u = head >= 0 ? head : u;
    }
}

template <int NB>
__device__ __forceinline__ void warp_run_any(int type, const uint8_t* tile, bool valid, const KSplit& ks, int kw, int tb, const XArg& xa, uint8_t* sm,
                                             int RS, int CK, bool stage, float (&acc)[Mode<NB>::OUT], int lane) {
#define RUN(T) warp_run<T, NB>(tile, valid, ks, kw, tb, xa, sm, RS, CK, stage, acc, lane)
#ifdef G3_ONLY
    RUN(G3_ONLY);
    return;
#endif
    switch (type) {
        case Q_Q4_K: RUN(Q_Q4_K); break;
        case Q_Q5_K: RUN(Q_Q5_K); break;
        case Q_Q6_K: RUN(Q_Q6_K); break;
        case Q_Q8_0: RUN(Q_Q8_0); break;
        case Q_IQ4_NL: RUN(Q_IQ4_NL); break;
        case Q_IQ4_XS: RUN(Q_IQ4_XS); break;
    }
#undef RUN
}

// 1 回の起動でやること。行列は同じ x を掛ける（4 つまで）。swiglu なら m[0] = gate、m[1] = up で h を m[0].y へ
struct Mat {
    const uint8_t* w;
    float* y;  // [B][n]
    int type, n;
};
struct Job {
    Mat m[4];
    int t0[5];  // 行列ごとのタイルの通し番号の始め（t0[nm] = 合計）
    int nm, ntiles, wk, swiglu;
    int tg;  // 8 本ずつの組の数（1・2・4）
    int ns, cmax;  // ブロックの k を何切れに分けるか、1 切れの単位の数の上限
    int ksplit, add;  // k をブロックで何個に分けるか（blockIdx.y）、y に足し込むか（add なら atomicAdd。ksplit > 1 は add のときだけ）
    XArg xa;
    void* hp;  // swiglu: down 用の x の下ごしらえも書くなら（infer_g3_xprep と同じ並び、k = n）
    // 決まった順の足し込み（ksplit > 1 のときだけ。null なら atomicAdd）: 部分和 part[ブロックの列][切れ][warp][OUT][32]、
    // 数え板 cnt[ブロックの列]（0 で始まり、最後に来たブロックが 0 に戻す）
    float* part;
    int* cnt;
    // 行列 0 の出口に GDN の因果 conv1d（幅 4）＋ silu を掛ける（デコード T = 1。gdn.cu の conv_kernel と同じ式）。null なら掛けない。
    // cw [n][4]、cst（状態 [slot][n][3]、新しい状態を書き戻す）、cslot [B]（null なら slot = トークン）
    const float* cw;
    float* cst;
    const int* cslot;
};

// conv1d（幅 4、状態 3 列）＋ silu を 1 つの (通り ch, トークン t) に。状態は [s1, s2, x] に進める
__device__ __forceinline__ float conv_silu(const Job& jb, int ch, int t, float xt) {
    const int slot = jb.cslot ? jb.cslot[t] : t;
    float* st = jb.cst + (int64_t(slot) * jb.m[0].n + ch) * 3;
    const float p0 = st[0], p1 = st[1], p2 = st[2];
    const float4 wv = *reinterpret_cast<const float4*>(jb.cw + int64_t(ch) * 4);
    const float o = wv.x * p0 + wv.y * p1 + wv.z * p2 + wv.w * xt;
    st[0] = p1, st[1] = p2, st[2] = xt;
    return o / (1.f + expf(-o));
}

__device__ __forceinline__ void put(float* p, float v, int add) {
    if (add) atomicAdd(p, v);
    else *p = v;
}

// 1 SM に 1 ブロック（8 warp）でレジスタを全部使う方が、2 ブロック（16 warp）で溢れるより速い（B=1 で 1 割）
#ifndef G3_MINB
#define G3_MINB 1
#endif
#ifndef G3_MINB8
#define G3_MINB8 1
#endif
template <int NB>
__device__ __forceinline__ void g3_body(const Job& jb) {
    using M = Mode<NB>;
    constexpr int OUT = M::OUT;
    extern __shared__ __align__(16) uint8_t sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wk = jb.wk, TG = jb.tg;
    const int nu = jb.xa.k / UK;
    const int RS = x_rs(jb.cmax), CK = jb.cmax * 8;
    const int B = jb.xa.B;
#ifdef G3_TIMING
    if (threadIdx.x == 0) g3_tm[tm_slot()][2] = 0;
#endif
    G3_TM(0);
    if (jb.swiglu) {
        // warp 1 本が 1 タイルの gate と up を続けて（同じ lane に同じ行が来るので、そのまま SiLU(gate)*up）
        constexpr int NBW = NB < 8 ? 8 : NB;  // 1 組の本数
        const int tpb = 8 / TG, tgi = warp % TG, slot = warp / TG, tb = tgi * NBW;
        const int tile = blockIdx.x * tpb + slot < jb.ntiles ? blockIdx.x * tpb + slot : -1;
        const KSplit ks{0, nu, jb.ns, 1};
        float ag[OUT], au[OUT];
#pragma unroll
        for (int j = 0; j < OUT; ++j) ag[j] = au[j] = 0.f;
        const int t0 = max(tile, 0);
        warp_run_any<NB>(jb.m[0].type, jb.m[0].w + int64_t(t0) * tile_bytes(jb.m[0].type, jb.xa.k), tile >= 0, ks, 0, tb, jb.xa, sm, RS, CK, true,
                         ag, lane);
        warp_run_any<NB>(jb.m[1].type, jb.m[1].w + int64_t(t0) * tile_bytes(jb.m[1].type, jb.xa.k), tile >= 0, ks, 0, tb, jb.xa, sm, RS, CK,
                         jb.ns > 1, au, lane);
        G3_TM_MAX(2);
        // h は x の置き場の後ろに [枠][トークン][32 行]
        const int n = jb.m[0].n, TGT = NBW * TG;
        float* hs = reinterpret_cast<float*>(sm + x_smem(B, jb.cmax));
#pragma unroll
        for (int j = 0; j < OUT; ++j) {
            const int r = out_row<NB>(lane, j), t = tb + out_tok<NB>(lane, j), row = tile * TILE + r;
            const float h = ag[j] / (1.f + __expf(-ag[j])) * au[j];
            if (tile >= 0 && row < n && t < B) {
                if (jb.m[0].y) jb.m[0].y[int64_t(t) * n + row] = h;
                if (jb.hp) hs[(slot * TGT + t) * TILE + r] = h;
            }
        }
        if (!jb.hp) return;
        // down の x の下ごしらえ: タイル（32 行）= h の 1 かたまり。4 lane で (タイル, トークン) の 32 要素（8 要素ずつ）
        __syncthreads();
        const int kb = n / 32;
        __half* hx = static_cast<__half*>(jb.hp);
        float2* hsi = reinterpret_cast<float2*>(static_cast<uint8_t*>(jb.hp) + int64_t(B) * n * 4);
        for (int base = warp * 8; base < tpb * B; base += 64) {  // warp の全員で回す（prep8 が shfl を使う）
            const int pr = min(base + (lane >> 2), tpb * B - 1);
            const int ts = pr / B, t = pr - ts * B, tl = blockIdx.x * tpb + ts;
            const bool ok = base + (lane >> 2) < tpb * B && tl < jb.ntiles;
            float v[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = hs[(ts * TGT + t) * TILE + (lane & 3) * 8 + i];
            __half* hi = hx + int64_t(t) * 2 * n + tl * TILE + (lane & 3) * 8;
            prep8(v, hi, hi + n, hsi + int64_t(t) * kb + tl, ok, (lane & 3) == 0);
        }
        return;
    }
    // warp = ((タイル枠 × wk) + kw) × TG + 組
    int mi, tile, kw, tgi;
    {
        tgi = warp % TG;
        kw = (warp / TG) % wk;
        const int gt = blockIdx.x * (8 / (wk * TG)) + warp / (TG * wk);
        mi = (jb.nm > 1 && gt >= jb.t0[1]) + (jb.nm > 2 && gt >= jb.t0[2]) + (jb.nm > 3 && gt >= jb.t0[3]);
        const int t0 = mi == 0 ? 0 : mi == 1 ? jb.t0[1] : mi == 2 ? jb.t0[2] : jb.t0[3];
        tile = gt < jb.ntiles ? gt - t0 : -1;
        if (tile < 0) mi = 0;
    }
    float acc[OUT];
#pragma unroll
    for (int j = 0; j < OUT; ++j) acc[j] = 0.f;
    const Mat m = mi == 0 ? jb.m[0] : mi == 1 ? jb.m[1] : mi == 2 ? jb.m[2] : jb.m[3];
    const int ku0 = blockIdx.y * nu / jb.ksplit, ku1 = (blockIdx.y + 1) * nu / jb.ksplit;
    const KSplit ks{ku0, ku1 - ku0, jb.ns, wk};
    const uint8_t* tp = m.w + int64_t(max(tile, 0)) * tile_bytes(m.type, jb.xa.k);
    const int tb = tgi * (NB < 8 ? 8 : NB);  // この warp のトークンの始め
    warp_run_any<NB>(m.type, tp, tile >= 0, ks, kw, tb, jb.xa, sm, RS, CK, true, acc, lane);
    G3_TM_MAX(2);
    if (wk > 1) {
        // k を分けた warp の溜めを kw = 0 の warp に kw の順で足す（共有メモリは x の置き場を使い回す）
        float* red = reinterpret_cast<float*>(sm);
        __syncthreads();
#pragma unroll
        for (int j = 0; j < OUT; ++j) red[(warp * OUT + j) * 32 + lane] = acc[j];
        __syncthreads();
        if (kw == 0) {
#pragma unroll
            for (int j = 0; j < OUT; ++j)
                for (int w = 1; w < wk; ++w) acc[j] += red[((warp + w * TG) * OUT + j) * 32 + lane];
        }
    }
    const bool outw = kw == 0 && tile >= 0;  // 書き出す warp
    if (!jb.part) {
        if (!outw) return;
#pragma unroll
        for (int j = 0; j < OUT; ++j) {
            const int row = tile * TILE + out_row<NB>(lane, j), t = tb + out_tok<NB>(lane, j);
            if (row < m.n && t < B) put(m.y + int64_t(t) * m.n + row, mi == 0 && jb.cw ? conv_silu(jb, row, t, acc[j]) : acc[j], jb.add);
        }
        return;
    }
    // 決まった順の足し込み: 切れ（blockIdx.y）ごとの部分和を作業場に置き、同じ列（blockIdx.x）で最後に来たブロックが
    // 切れ 0, 1, … の順に足してから y に足す（どのブロックが最後でも足す順は同じ → 毎回同じ答え）
    constexpr int PWS = OUT * 32;  // 1 warp の部分和の数
    const float* pr = jb.part + (int64_t(blockIdx.x) * jb.ksplit * 8 + warp) * PWS + lane;  // 切れ s は + s·8·PWS
    if (outw) {
        float* pw = jb.part + ((int64_t(blockIdx.x) * jb.ksplit + blockIdx.y) * 8 + warp) * PWS + lane;
#pragma unroll
        for (int j = 0; j < OUT; ++j) pw[j * 32] = acc[j];
    }
    __threadfence();  // 部分和を他のブロックに見せてから数える
    __syncthreads();
    __shared__ int last;
    if (threadIdx.x == 0) {
        const int l = atomicAdd(jb.cnt + blockIdx.x, 1) == jb.ksplit - 1;
        if (l) {
            jb.cnt[blockIdx.x] = 0;  // 列の全員が数え終わっているので、次の起動のために 0 に戻す
            __threadfence();
        }
        last = l;
    }
    __syncthreads();
    if (!last || !outw) return;
    // 切れの順に足す。読みは切れの間で待ち合わせないよう先に出す（他のブロックが書いた所なので L1 を通さない）
    float v[OUT];
#pragma unroll
    for (int j = 0; j < OUT; ++j) v[j] = 0.f;
#pragma unroll 4
    for (int s = 0; s < jb.ksplit; ++s) {
#pragma unroll
        for (int j = 0; j < OUT; ++j) v[j] += __ldcg(pr + int64_t(s) * 8 * PWS + j * 32);
    }
    // y に足すのはこのブロックだけ（1 要素に 1 回）なので atomicAdd でも答えは同じ（読みの往復を待たない）
#pragma unroll
    for (int j = 0; j < OUT; ++j) {
        const int row = tile * TILE + out_row<NB>(lane, j), t = tb + out_tok<NB>(lane, j);
        if (row < m.n && t < B) atomicAdd(m.y + int64_t(t) * m.n + row, v[j]);
    }
}

template <int NB>
__global__ void __launch_bounds__(256, NB <= 4 ? G3_MINB : G3_MINB8) g3_kernel(const Job jb) {
    g3_body<NB>(jb);
}

// MoE（3.6・qwen35moe）: 仕事 z = blockIdx.z ごとに専門家 e[z] の行列（w + e·es）で、x の行 r0[z]..+cnt[z] を掛けて y の同じ行へ。
// x の行は xidx があれば xidx[r0 + i]（語の番号。専門家ごとに選ばれた語を写さずに引く）、無ければ r0 + i。k は分けない（add なし）
struct MoeJobs {
    const int* e;
    const int* r0;
    const int* cnt;
    const int* xidx;
    int64_t es[4];
};
template <int NB>
__global__ void __launch_bounds__(256, NB <= 4 ? G3_MINB : G3_MINB8) g3_moe_kernel(Job jb, const MoeJobs mj) {
    const int z = blockIdx.z;
    const int cnt = __ldg(mj.cnt + z);
    if (cnt <= 0) return;  // ブロックごと抜ける（__syncthreads の前）
    const int e = __ldg(mj.e + z), r0 = __ldg(mj.r0 + z);
    const int k = jb.xa.k;
    jb.xa.B = cnt;
    if (mj.xidx) jb.xa.xidx = mj.xidx + r0;
    else {
        jb.xa.xhl += int64_t(r0) * 2 * k;
        jb.xa.xsi += int64_t(r0) * (k / 32);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (i < jb.nm) {
            jb.m[i].w += int64_t(e) * mj.es[i];
            if (jb.m[i].y) jb.m[i].y += int64_t(r0) * jb.m[i].n;
        }
    }
    g3_body<NB>(jb);
}

// 1 warp が受け持つ本数（G3_NBMAX = 8・16・32）。それより多い本数は G3_NBMAX 本ずつの組に分ける
#ifndef G3_NBMAX
#define G3_NBMAX 16
#endif
int nb_class(int B) { return B <= 2 ? 2 : B <= 4 ? 4 : B <= 8 || G3_NBMAX == 8 ? 8 : B <= 16 || G3_NBMAX == 16 ? 16 : 32; }
int nb_out(int nb) { return nb <= 2 ? 2 : nb <= 4 ? 4 : nb; }
int tg_of(int B) { const int nb = nb_class(B) < 8 ? 8 : nb_class(B); return (B + nb - 1) / nb <= 1 ? 1 : (B + nb - 1) / nb <= 2 ? 2 : 4; }

// 1 SM に 1 ブロックなので、ブロックは 80 個（1 回で全部乗る）までにする
constexpr int SMS = 80;
// x を置く共有メモリの上限（1 ブロック）
constexpr int X_BUDGET = 88 * 1024;

template <int N>
void launch_nb(const Job& jb, dim3 grid, int smem, cudaStream_t s) {
    if (smem > 48 * 1024) cudaFuncSetAttribute(g3_kernel<N>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    g3_kernel<N><<<grid, 256, smem, s>>>(jb);
}

// 決まった順の足し込みの作業場: 数え板（ブロックの列ごと、CNT_MAX 列まで）＋ 部分和 [列][切れ][warp][OUT][32]
constexpr int CNT_MAX = 128;
constexpr int64_t CNT_BYTES = CNT_MAX * 4;
int64_t det_ws_bytes(int gx, int ksp, int nb) { return CNT_BYTES + int64_t(gx) * ksp * 8 * nb_out(nb) * 32 * 4; }

// 足し込みの順を決めるか（1 = 決める・既定、0 = atomicAdd）。-1 はまだ環境変数 INFER_G3_DET を読んでいない
std::atomic<int> g_det{-1};
bool det_mode() {
    int d = g_det.load(std::memory_order_relaxed);
    if (d < 0) {
        const char* e = getenv("INFER_G3_DET");
        d = e && *e ? atoi(e) != 0 : 1;
        g_det.store(d, std::memory_order_relaxed);
    }
    return d != 0;
}

// 起動の形
struct Plan3 {
    int wk, ksp, ns, cmax, gx, smem;
    bool det;  // k の切れの部分和を作業場に置き、決まった順で足す
};

// wk（0 なら形から）と、k を何切れに分けるかを決める。
//   ふつう: warp 1 本 = 1 タイルの k 全部（wk = 1、ブロックの 8 タイルで x を使い回す）。ブロックが 80 に届かなければ
//   足し込みのときは k をブロックで分け（ksp）、そうでなければ 1 タイルを wk 本の warp で分ける（どちらも 80 を超えない範囲）
//   swiglu: warp 1 本が 1 タイルの gate と up を続けて
// 守ること: ブロック y の k は単位 [y·nu/ksp, (y+1)·nu/ksp)（nu/ksp か、その +1）。それを ns 切れに分けた
// どの切れにも wk 単位以上（warp kw の受け持ちが空だと、先読みした重みが次の切れの頭とずれて答えが黙って狂う）、
// 1 切れの x（いちばん大きい切れ = cmax 単位）は X_BUDGET に収める。収まらなければ ksp → wk の順に下げる
// （wk・ksp は答えの値を変えない。wk = 1・ksp = 1 なら 1 単位ずつの切れで必ず収まる）
// det: 足し込みで ksp > 1 のとき、作業場（ws_bytes）が足りれば決まった順で足す。足りなければ ksp = 1 で、wk を形から選ぶなら
// 足し込みでない時と同じく 1 タイルを warp で分けて SM を埋める（ブロックの中の足す順は決まっているので毎回同じ。atomicAdd の道より遅い）
int plan_g3(int ntiles, int nu, int B, int wk_req, bool add, bool swiglu, bool hp, bool det, int64_t ws_bytes, Plan3& p) {
    const int nb = nb_class(B), tg = tg_of(B);
    auto blocks = [&](int w) { return (ntiles * tg * w + 7) / 8; };
    // k をブロックで分けてよいか: 足し込みだけ。決まった順なら作業場が要る（無ければ k は 1 タイルを warp で分けるだけ。足す順はブロックの中で決まる）
    const bool can_ksp = add && !swiglu && (!det || ws_bytes >= det_ws_bytes(SMS, 1, nb));
    int wk = swiglu ? 1 : wk_req;
    if (wk == 0) {
        wk = 1;
        if (!can_ksp)
            while (wk * 2 * tg <= 8 && wk * 2 <= nu && blocks(wk * 2) <= SMS) wk *= 2;
    }
    if ((wk != 1 && wk != 2 && wk != 4 && wk != 8) || wk > nu || wk * tg > 8) return -4;
    const int hsb = hp ? 8 * max(8, nb) * TILE * 4 : 0;  // swiglu の h（[枠][トークン][32 行]）
    const int red = 8 * nb_out(nb) * 32 * 4;
    for (;; wk /= 2) {
        const int gx = blocks(wk);
        int ksp = 1;
        if (can_ksp) {
            while (gx * ksp * 2 <= SMS && nu / (ksp * 2) >= max(3, wk) && ksp * 2 * B <= 64) ksp *= 2;
#ifdef G3_EXPERIMENT
            if (getenv("G3_KSP")) ksp = max(1, min(atoi(getenv("G3_KSP")), nu));
#endif
            // 決まった順は作業場が要る（列は CNT_MAX まで）。無ければ k をブロックで分けない
            if (det && ksp > 1 && (gx > CNT_MAX || ws_bytes < det_ws_bytes(gx, ksp, nb))) ksp = 1;
        }
        for (;; ksp /= 2) {
            const int lo = nu / ksp, hi = (nu + ksp - 1) / ksp;  // ブロックの単位の数の最小・最大
            int ns = 1;
#ifdef G3_EXPERIMENT
            if (getenv("G3_NS")) ns = max(1, min(atoi(getenv("G3_NS")), lo / max(wk, 1)));
#endif
            while (ns < hi && x_smem(B, (hi + ns - 1) / ns) + hsb > X_BUDGET) ++ns;
            const int cmax = (hi + ns - 1) / ns;
            if (lo / ns >= wk && x_smem(B, cmax) + hsb <= X_BUDGET) {
                p = Plan3{wk, ksp, ns, cmax, gx, max(x_smem(B, cmax) + hsb, red), det && add && ksp > 1};
                return 0;
            }
            if (ksp == 1) break;
        }
        if (wk == 1) return -4;  // 来ない（1 単位ずつの切れは必ず収まる）
    }
}

int launch(Job& jb, cudaStream_t s, void* ws, int64_t ws_bytes) {
    const int B = jb.xa.B, nu = jb.xa.k / UK, nb = nb_class(B);
    Plan3 p;
    if (int e = plan_g3(jb.ntiles, nu, B, jb.wk, jb.add, jb.swiglu, jb.hp != nullptr, det_mode(), ws ? ws_bytes : 0, p)) return e;
    jb.tg = tg_of(B);
    jb.wk = p.wk;
    jb.ksplit = p.ksp;
    jb.ns = p.ns;
    jb.cmax = p.cmax;
    jb.cnt = p.det ? static_cast<int*>(ws) : nullptr;
    jb.part = p.det ? reinterpret_cast<float*>(static_cast<uint8_t*>(ws) + CNT_BYTES) : nullptr;
    const dim3 grid(p.gx, p.ksp);
    const int smem = p.smem;
    switch (nb) {
        case 2: launch_nb<2>(jb, grid, smem, s); break;
        case 4: launch_nb<4>(jb, grid, smem, s); break;
#if G3_NBMAX >= 16
        case 16: launch_nb<16>(jb, grid, smem, s); break;
#endif
#if G3_NBMAX >= 32
        case 32: launch_nb<32>(jb, grid, smem, s); break;
#endif
        default: launch_nb<8>(jb, grid, smem, s); break;
    }
    return int(cudaGetLastError());
}

XArg xarg(const void* xp, int k, int B) {
    XArg a;
    a.xhl = static_cast<const __half*>(xp);
    a.xsi = reinterpret_cast<const float2*>(static_cast<const uint8_t*>(xp) + int64_t(B) * k * 4);
    a.k = k;
    a.B = B;
    a.xidx = nullptr;
    return a;
}

template <int N>
void launch_moe_nb(const Job& jb, const MoeJobs& mj, dim3 grid, int smem, cudaStream_t s) {
    if (smem > 48 * 1024) cudaFuncSetAttribute(g3_moe_kernel<N>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    g3_moe_kernel<N><<<grid, 256, smem, s>>>(jb, mj);
}


}  // namespace

#define G3_SRC_TYPES(X) X(Q_Q4_K) X(Q_Q5_K) X(Q_Q6_K) X(Q_Q8_0) X(Q_IQ4_NL) X(Q_IQ4_XS) X(Q_Q3_K) X(Q_IQ3_S)
#define G3_TYPES(X) X(Q_Q4_K) X(Q_Q5_K) X(Q_Q6_K) X(Q_Q8_0) X(Q_IQ4_NL) X(Q_IQ4_XS)

extern "C" {

// 詰め直した後の型（Q3_K・IQ3_S → Q6_K）。扱わない型は -1
int infer_g3_type(int qtype) { return packed_type(qtype); }

// 自前の並びの大きさ。扱わない型・k が 256 の倍数でなければ -1
int64_t infer_g3_bytes(int qtype, int k, int n) {
    const int p = packed_type(qtype);
    if (p < 0 || k % UK) return -1;
    return mat_bytes(p, k, n);
}

// GGUF の並び（src、[n][k]）→ 自前の並び（dst、infer_g3_bytes の大きさ）。どちらも GPU の上
int infer_g3_repack(int qtype, const void* src, void* dst, int k, int n, cudaStream_t s) {
    if (packed_type(qtype) < 0) return -1;
    if (k % UK) return -2;
    const int64_t threads = int64_t((n + TILE - 1) / TILE) * TILE * (k / UK);
    const unsigned grid = unsigned((threads + 127) / 128);
    const uint8_t* sp = static_cast<const uint8_t*>(src);
    uint8_t* dp = static_cast<uint8_t*>(dst);
    switch (qtype) {
#define CASE(T) case T: repack_kernel<T><<<grid, 128, 0, s>>>(sp, dp, k, n); break;
        G3_SRC_TYPES(CASE)
#undef CASE
    }
    return int(cudaGetLastError());
}

// 確かめ用: 自前の並び（qtype は詰め直した後の型）→ f32 [n][k]
int infer_g3_dequant(int qtype, const void* w, int k, int n, float* y, cudaStream_t s) {
    if (packed_type(qtype) != qtype) return -1;
    if (k % UK) return -2;
    const int64_t threads = int64_t(n) * (k / 32);
    const unsigned grid = unsigned((threads + 127) / 128);
    const uint8_t* wp = static_cast<const uint8_t*>(w);
    switch (qtype) {
#define CASE(T) case T: g3_dequant_kernel<T><<<grid, 128, 0, s>>>(wp, k, n, y); break;
        G3_TYPES(CASE)
#undef CASE
    }
    return int(cudaGetLastError());
}

// x の下ごしらえの置き場の大きさ（バイト）: xhl[B][2][k] f16、xsi[B][k/32] float2（Σx, 倍率の逆数）
int64_t infer_g3_xprep_bytes(int k, int B) { return int64_t(B) * k * 4 + int64_t(B) * (k / 32) * 8; }

int infer_g3_xprep(const float* x, int k, int B, void* xp, cudaStream_t s) {
    if (k % UK) return -2;
    __half* xhl = static_cast<__half*>(xp);
    float2* xsi = reinterpret_cast<float2*>(static_cast<uint8_t*>(xp) + int64_t(B) * k * 4);
    const int64_t warps = int64_t(B) * (k / UK);
    xprep_kernel<<<unsigned((warps + 7) / 8), 256, 0, s>>>(x, k, B, xhl, xsi);
    return int(cudaGetLastError());
}

// 同じ x を掛ける行列 nm 個（4 つまで）: y_i[b][n_i] = W_i x[b]（b < B ≤ 32）。w は自前の並び（qtype は詰め直した後の型）、
// xp は infer_g3_xprep の結果（同じ k・B）。wk が 0 なら形から選ぶ（1・2・4・8。収まらなければ下げる。答えの値は変わらない）。
// add = 1 なら y += W x（残差に直接足す用。n が小さい行列は k をブロックで分けて SM を埋める）。
// 足し込みの順（infer_g3_set_det）: 決める（既定）なら、k を分けた部分和を ws（infer_g3_ws_bytes 以上・最初に 0 で埋める。
// 1 つの stream に 1 つ）に置いて決まった順で足す（毎回同じ答え）。ws が null なら k をブロックで分けない（同じ答えだが遅い）。
// 決めないなら atomicAdd（速いが、k を分けると足す順が毎回変わり、下の桁が揺れる）
int infer_g3_gemv_multi_ws(int nm, const int* qtype, const void* const* w, const int* n, float* const* y, int k, const void* xp, int B,
                           int wk, int add, void* ws, int64_t ws_bytes, cudaStream_t s) {
    if (nm < 1 || nm > 4) return -3;
    if (B < 1 || B > 32) return -3;
    if (k % UK) return -2;
    Job jb{};
    jb.nm = nm;
    jb.t0[0] = 0;
    for (int i = 0; i < nm; ++i) {
        if (packed_type(qtype[i]) != qtype[i]) return -1;
        jb.m[i] = Mat{static_cast<const uint8_t*>(w[i]), y[i], qtype[i], n[i]};
        jb.t0[i + 1] = jb.t0[i] + (n[i] + TILE - 1) / TILE;
    }
    jb.ntiles = jb.t0[nm];
    jb.wk = wk;
    jb.swiglu = 0;
    jb.add = add;
    jb.xa = xarg(xp, k, B);
    return launch(jb, s, ws, ws_bytes);
}

// infer_g3_gemv_multi（add = 0）の行列 0 の出口に GDN の因果 conv1d（幅 4）＋ silu を掛けて y_0 に書く（デコード T = 1 で conv を融かす）。
// conv_w [n_0][4]、conv_st は状態 [slot][n_0][3]（書き戻す）、slots [B]（null なら slot = b）。値は gdn.cu の conv の silu までと同じ
int infer_g3_gemv_multi_conv(int nm, const int* qtype, const void* const* w, const int* n, float* const* y, int k, const void* xp, int B,
                             const float* conv_w, float* conv_st, const int* slots, cudaStream_t s) {
    if (nm < 1 || nm > 4) return -3;
    if (B < 1 || B > 32) return -3;
    if (k % UK) return -2;
    if (!conv_w || !conv_st || (reinterpret_cast<uintptr_t>(conv_w) & 15)) return -2;
    Job jb{};
    jb.nm = nm;
    jb.t0[0] = 0;
    for (int i = 0; i < nm; ++i) {
        if (packed_type(qtype[i]) != qtype[i]) return -1;
        jb.m[i] = Mat{static_cast<const uint8_t*>(w[i]), y[i], qtype[i], n[i]};
        jb.t0[i + 1] = jb.t0[i] + (n[i] + TILE - 1) / TILE;
    }
    jb.ntiles = jb.t0[nm];
    jb.wk = 0;
    jb.swiglu = 0;
    jb.add = 0;
    jb.xa = xarg(xp, k, B);
    jb.cw = conv_w;
    jb.cst = conv_st;
    jb.cslot = slots;
    return launch(jb, s, nullptr, 0);
}

int infer_g3_gemv_multi(int nm, const int* qtype, const void* const* w, const int* n, float* const* y, int k, const void* xp, int B, int wk,
                        int add, cudaStream_t s) {
    return infer_g3_gemv_multi_ws(nm, qtype, w, n, y, k, xp, B, wk, add, nullptr, 0, s);
}

int infer_g3_gemv_ws(int qtype, const void* w, int k, int n, const void* xp, float* y, int B, int wk, int add, void* ws, int64_t ws_bytes,
                     cudaStream_t s) {
    return infer_g3_gemv_multi_ws(1, &qtype, &w, &n, &y, k, xp, B, wk, add, ws, ws_bytes, s);
}

int infer_g3_gemv(int qtype, const void* w, int k, int n, const void* xp, float* y, int B, int wk, int add, cudaStream_t s) {
    return infer_g3_gemv_multi_ws(1, &qtype, &w, &n, &y, k, xp, B, wk, add, nullptr, 0, s);
}

// 決まった順の足し込みの作業場の大きさ（どの形・B でも足りる上限。約 1.3MB）
int64_t infer_g3_ws_bytes(void) { return det_ws_bytes(SMS, 1, G3_NBMAX); }

// 試し: g3_tm を n ブロックぶん手元へ（-DG3_TIMING でないときは -1）
int infer_g3_timing(unsigned long long* out, int n) {
#ifdef G3_TIMING
    return int(cudaMemcpyFromSymbol(out, g3_tm, size_t(min(n, 4096)) * 3 * 8));
#else
    (void)out, (void)n;
    return -1;
#endif
}

// 足し込みの順を決めるか（1 = 決める、0 = atomicAdd）。前の値を返す。最初の値は環境変数 INFER_G3_DET（無ければ 1）
int infer_g3_set_det(int on) {
    const int old = det_mode();
    g_det.store(on != 0, std::memory_order_relaxed);
    return old;
}

// 確かめ用: 1 つの行列の起動の形。out = {wk, ksp, ns, cmax, gx, smem, det}。戻り値は infer_g3_gemv と同じ（0 か -4 など）
int infer_g3_plan(int k, int n, int B, int wk, int add, int64_t ws_bytes, int* out) {
    if (B < 1 || B > 32) return -3;
    if (k % UK) return -2;
    Plan3 p;
    if (int e = plan_g3((n + TILE - 1) / TILE, k / UK, B, wk, add, false, false, det_mode(), ws_bytes, p)) return e;
    const int v[7] = {p.wk, p.ksp, p.ns, p.cmax, p.gx, p.smem, p.det};
    for (int i = 0; i < 7; ++i) out[i] = v[i];
    return 0;
}

// MoE の GEMV（3.6・qwen35moe）: 仕事 j（< n_jobs）ごとに y_i[r0[j] + t][n_i] = W_i[e[j]] x[行 t]（t < cnt[j] ≤ bmax ≤ 32、行列 nm ≤ 4 個）。
// W_i は専門家を続けて並べた自前の並び（専門家 e の頭 = w_i + e·estride_i バイト）。x は xp（infer_g3_xprep の結果・rx 行）の行
// xidx[r0[j] + t]（xidx が null なら r0[j] + t）。cnt[j] が 0 の仕事は何もしない。e・r0・cnt・xidx は GPU の上（CUDA Graph に取れる）
int infer_g3_moe_gemv(int nm, const int* qtype, const void* const* w, const int64_t* estride, const int* n, float* const* y, int k,
                      const void* xp, int rx, const int* xidx, const int* je, const int* jr0, const int* jcnt, int n_jobs, int bmax,
                      cudaStream_t s) {
    if (nm < 1 || nm > 4) return -3;
    if (bmax < 1 || bmax > 32 || rx < 1 || n_jobs < 1 || n_jobs > 65535) return -3;
    if (k % UK) return -2;
    Job jb{};
    MoeJobs mj{};
    jb.nm = nm;
    jb.t0[0] = 0;
    for (int i = 0; i < nm; ++i) {
        if (packed_type(qtype[i]) != qtype[i]) return -1;
        jb.m[i] = Mat{static_cast<const uint8_t*>(w[i]), y[i], qtype[i], n[i]};
        jb.t0[i + 1] = jb.t0[i] + (n[i] + TILE - 1) / TILE;
        mj.es[i] = estride[i];
    }
    mj.e = je, mj.r0 = jr0, mj.cnt = jcnt, mj.xidx = xidx;
    jb.ntiles = jb.t0[nm];
    jb.swiglu = 0;
    jb.add = 0;
    jb.xa = xarg(xp, k, rx);
    jb.xa.B = bmax;  // 起動の形は bmax で（仕事ごとの本数は kernel が cnt に替える）
    // wk は「仕事の数 × タイル」で SM が埋まるように選ぶ。k は分けない
    Plan3 p;
    if (int e = plan_g3(jb.ntiles * n_jobs, k / UK, bmax, 0, false, false, false, false, 0, p)) return e;
    const int tg = tg_of(bmax);
    jb.tg = tg;
    jb.wk = p.wk;
    jb.ksplit = 1;
    jb.ns = p.ns;
    jb.cmax = p.cmax;
    const int gx = (jb.ntiles * tg * p.wk + 7) / 8;
    const dim3 grid(gx, 1, n_jobs);
    switch (nb_class(bmax)) {
        case 2: launch_moe_nb<2>(jb, mj, grid, p.smem, s); break;
        case 4: launch_moe_nb<4>(jb, mj, grid, p.smem, s); break;
#if G3_NBMAX >= 16
        case 16: launch_moe_nb<16>(jb, mj, grid, p.smem, s); break;
#endif
#if G3_NBMAX >= 32
        case 32: launch_moe_nb<32>(jb, mj, grid, p.smem, s); break;
#endif
        default: launch_moe_nb<8>(jb, mj, grid, p.smem, s); break;
    }
    return int(cudaGetLastError());
}

// h[b][n] = SiLU(W_gate x[b]) * (W_up x[b])。gate・up は自前の並び（型は別々でよい）。
// h（f32）と hp（down に渡す x の下ごしらえ。infer_g3_xprep_bytes(n, B) の大きさ。n は 256 の倍数）はどちらかだけでもよい
int infer_g3_swiglu(int tg, const void* wg, int tu, const void* wu, int k, int n, const void* xp, float* h, void* hp, int B, cudaStream_t s) {
    if (B < 1 || B > 32) return -3;
    if (k % UK || (hp && n % UK)) return -2;
    if (packed_type(tg) != tg || packed_type(tu) != tu) return -1;
    Job jb{};
    jb.nm = 2;
    jb.m[0] = Mat{static_cast<const uint8_t*>(wg), h, tg, n};
    jb.m[1] = Mat{static_cast<const uint8_t*>(wu), nullptr, tu, n};
    jb.t0[0] = 0;
    jb.t0[1] = jb.t0[2] = jb.ntiles = (n + TILE - 1) / TILE;
    jb.wk = 0;
    jb.swiglu = 1;
    jb.hp = hp;
    jb.xa = xarg(xp, k, B);
    return launch(jb, s, nullptr, 0);
}

}  // extern "C"
