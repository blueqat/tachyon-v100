// デコード用 GEMV の 3 版（g3）: 重みを自前の並びに詰め直し、Volta の mma（m8n8k4・f16 を掛けて f32 で溜める）で掛ける。
//
// ■ 詰め直した並び（単位 = 32 行 × 256 要素。n は 32 の倍数に切り上げ、足りない行は 0）
//   32 行を 1 タイルとし、タイルごとに k 方向の単位を順に並べる。単位の中は「面」に分け、各行のぶんは置き場 slot
//   （= その行を A として受け持つ lane）の順に詰める → warp の 1 回の読みが 1 つの面の続き（512 B など）になる。
//   Q4_K  （4608 B）: Q[8 かたまり][32×16] S[32×16]（GGUF の頭 16 B: d・dmin・scales[12] のまま）
//   Q5_K  （5632 B）: Q[8][32×16] H[8][32×4]（5 ビット目）S[32×16]
//   Q6_K  （6720 B）: Q[8][32×16] H[8][32×8]（上 2 ビット）SC[32×16]（int8 の scale）D[32×2]
//   Q8_0  （8704 B）: Q[8][2][32×16]（q+128 の uint8。要素 0-15 / 16-31）D[32×16]（かたまりごとの f16 × 8）
//   IQ4_NL（4608 B）: Q[8][32×16]（表の番号）D[32×16]（f16 × 8）
//   IQ4_XS（4352 B）: Q[8][32×16] S[32×8]（GGUF の頭 8 B: d・scales_h・scales_l[4] のまま）
//   Q3_K・IQ3_S は詰め直すときに Q6_K の並びに直す（値は変わらない。大きさは 110 B → 210 B / 256 要素）
// ■ 4 ビットの詰め方
//   K 系: 1 語（8 要素 e0..e7）の nibble p に要素 {0,2,4,6,1,3,5,7}[p] を置く。lop3 で (w & 0x000F000F) | 0x64006400 とすると
//     half2 で (e0,e1)、0x00F000F0 で 16×(e2,e3)、w>>8 で (e4,e5)・16×(e6,e7) が出る＝ mma の A の 4 要素が k の順に並ぶ。
//     上のビット（Q5 は 1、Q6 は 2）は、決まった量だけ回すと同じ位置に来るように置く（rot5・rot6）。
//   IQ4: nibble p に要素 p（そのままの順）。prmt で表（kvalues + 128 の 16 バイト）を引き、0x64 を挟んで half2 にする。
// ■ 掛け算: A は整数 q（f16 で正確）、x は f16 の上と下（hi + lo、x の f32 をほぼそのまま表す）。
//   積は正確・和は f32。かたまりごとの倍率（d·sc など）は f32 で後から掛ける → f32 並みの精度。
#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include "quant.cuh"

namespace g3 {

constexpr int TILE = 32;  // 1 タイル = 32 行
constexpr int UK = 256;   // 1 単位 = 256 要素（かたまり 8 つ）

// タイルの中の行 r ↔ 置き場 slot（mma の A の行の割り当て: quad-pair = (lane>>2)&3、その中の行 = (lane&3)+4*(lane>>4)）
__host__ __device__ inline int slot_of_row(int r) { const int qp = r >> 3, a = r & 7; return (a & 3) + 4 * qp + 16 * (a >> 2); }
__host__ __device__ inline int row_of_slot(int l) { return 8 * ((l >> 2) & 3) + (l & 3) + 4 * (l >> 4); }

// 詰め直した後の型（Q3_K・IQ3_S は Q6_K に直す）。扱わない型は -1
__host__ __device__ inline int packed_type(int t) {
    switch (t) {
        case Q_Q4_K: case Q_Q5_K: case Q_Q6_K: case Q_Q8_0: case Q_IQ4_NL: case Q_IQ4_XS: return t;
        case Q_Q3_K: case Q_IQ3_S: return Q_Q6_K;
    }
    return -1;
}
__host__ __device__ constexpr int unit_bytes(int t) {
    return t == Q_Q4_K ? 4608 : t == Q_Q5_K ? 5632 : t == Q_Q6_K ? 6720 : t == Q_Q8_0 ? 8704 : t == Q_IQ4_NL ? 4608
         : t == Q_IQ4_XS ? 4352 : -1;
}
// 1 タイル（32 行 × k）のバイト数と、1 行列（n 行）のバイト数（t は詰め直した後の型）
__host__ __device__ inline int64_t tile_bytes(int t, int k) { return int64_t(k / UK) * unit_bytes(t); }
__host__ __device__ inline int64_t mat_bytes(int t, int k, int n) { return int64_t((n + TILE - 1) / TILE) * tile_bytes(t, k); }

// K 系の 4 ビットの nibble の位置 p に置く要素（1 語 8 要素の中）
__host__ __device__ inline int nib_elem(int p) { return p < 4 ? 2 * p : 2 * (p - 4) + 1; }
// 上のビットを置く回し量。添え字は (語 ww, 前後 h) の組
__host__ __device__ inline int rot5(int ci) { return ci < 4 ? ci : ci + 4; }           // ci = 2*ww + h → 0,1,2,3,8,9,10,11
__host__ __device__ inline int rot6(int ci) { return (ci & 1) * 2 + (ci >> 1) * 8; }   // ci = 2*(ww&1) + h → 0,2,8,10

// ---------------------------------------------------------------- 読み
__device__ __forceinline__ uint4 ld16(const void* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }
__device__ __forceinline__ uint2 ld8(const void* p) { return __ldg(reinterpret_cast<const uint2*>(p)); }
__device__ __forceinline__ uint32_t ld4(const void* p) { return __ldg(reinterpret_cast<const uint32_t*>(p)); }
__device__ __forceinline__ uint32_t ld2(const void* p) { return __ldg(reinterpret_cast<const uint16_t*>(p)); }
// 重みの読み。G3_WL1 = 0 なら L1 に置かない（ld.global.cg）、1 なら L1 を通す（ld.global.nc。同じタイルを
// 別の組の warp も読むとき L1 に当たる）。x は共有メモリにあるので L1 を汚しても困らない
#ifndef G3_WL1
#define G3_WL1 0
#endif
#if G3_WL1
__device__ __forceinline__ uint4 wl16(const void* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }
__device__ __forceinline__ uint2 wl8(const void* p) { return __ldg(reinterpret_cast<const uint2*>(p)); }
__device__ __forceinline__ uint32_t wl4(const void* p) { return __ldg(reinterpret_cast<const unsigned int*>(p)); }
__device__ __forceinline__ uint32_t wl2(const void* p) { return __ldg(reinterpret_cast<const unsigned short*>(p)); }
#else
__device__ __forceinline__ uint4 wl16(const void* p) { return __ldcg(reinterpret_cast<const uint4*>(p)); }
__device__ __forceinline__ uint2 wl8(const void* p) { return __ldcg(reinterpret_cast<const uint2*>(p)); }
__device__ __forceinline__ uint32_t wl4(const void* p) { return __ldcg(reinterpret_cast<const unsigned int*>(p)); }
__device__ __forceinline__ uint32_t wl2(const void* p) { return __ldcg(reinterpret_cast<const unsigned short*>(p)); }
#endif

// (a & MASK) | c を 1 命令で（c をレジスタに置かないと nvcc が 2 命令に分ける）
template <uint32_t MASK>
__device__ __forceinline__ uint32_t and_or(uint32_t a, uint32_t c) {
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "n"(MASK), "r"(c));
    return r;
}
// 0x64006400 をレジスタに（定数のままだと畳み込まれる）
__device__ __forceinline__ uint32_t magic64() {
    uint32_t r;
    asm("mov.b32 %0, 0x64006400;" : "=r"(r));
    return r;
}

__device__ __forceinline__ uint32_t hadd2u(uint32_t a, float c) {
    const __half2 r = __hadd2(*reinterpret_cast<const __half2*>(&a), __float2half2_rn(c));
    return *reinterpret_cast<const uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t hfma2u(uint32_t a, float m, float c) {
    const __half2 r = __hfma2(*reinterpret_cast<const __half2*>(&a), __float2half2_rn(m), __float2half2_rn(c));
    return *reinterpret_cast<const uint32_t*>(&r);
}
__device__ __forceinline__ float hbits2f(uint32_t h) { return __half2float(__ushort_as_half(uint16_t(h))); }
__device__ __forceinline__ uint32_t rotr(uint32_t h, int r) { return __funnelshift_r(h, h, r); }

// K 系: 4 ビット（＋上のビット）の 1 語 → A の 4 レジスタ（2 歩ぶん）。t0・t1 は上のビットを置いた 0x6400.. の語
template <int BIAS>
__device__ __forceinline__ void deq_word(uint32_t w, uint32_t t0a, uint32_t t1a, uint32_t t0b, uint32_t t1b, uint32_t* a) {
    const uint32_t w8 = w >> 8;
    a[0] = hadd2u(and_or<0x000F000Fu>(w, t0a), -1024.f - BIAS);
    a[1] = hfma2u(and_or<0x00F000F0u>(w, t1a), 1.f / 16, -64.f - BIAS);
    a[2] = hadd2u(and_or<0x000F000Fu>(w8, t0b), -1024.f - BIAS);
    a[3] = hfma2u(and_or<0x00F000F0u>(w8, t1b), 1.f / 16, -64.f - BIAS);
}

// IQ4: 1 語（nibble p = 要素 p）→ A の 4 レジスタ。表は kvalues_iq4nl + 128 を 16 バイトに詰めたもの
// （__byte_perm は選び番号の下 3 ビットだけを見る。4 ビット目で表の前半・後半を選ぶ。llama.cpp の get_int_from_table_16 と同じ考え）
__device__ __forceinline__ void deq_iq4_word(uint32_t w, uint32_t* a) {
    constexpr uint32_t T0 = 0x3f2d1801u, T1 = 0x766a5d4fu, T2 = 0xa6998d81u, T3 = 0xf1d9c5b5u;
    const uint32_t sel = 0x32103210u | ((w & 0x88888888u) >> 1);
    uint32_t t[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t ws = w >> (16 * i);
        t[i] = __byte_perm(__byte_perm(T0, T1, ws), __byte_perm(T2, T3, ws), sel >> (16 * i));
    }
    a[0] = hadd2u(__byte_perm(t[0], 0x64646464u, 0x4140), -1152.f);
    a[1] = hadd2u(__byte_perm(t[0], 0x64646464u, 0x4342), -1152.f);
    a[2] = hadd2u(__byte_perm(t[1], 0x64646464u, 0x4140), -1152.f);
    a[3] = hadd2u(__byte_perm(t[1], 0x64646464u, 0x4342), -1152.f);
}

// K 系の 6 ビットの scale と min（S の語 y・z・w = scales[0..11]）
__device__ __forceinline__ void kscale(const uint4& S, int ib, float& s, float& o) {
    const int i = (ib & 3) * 8;
    const uint32_t by = (S.y >> i) & 0xff, bz = (S.z >> i) & 0xff, bw = (S.w >> i) & 0xff;
    uint32_t sc, mn;
    if (ib < 4) { sc = by & 63; mn = bz & 63; }
    else { sc = (bw & 0xf) | ((by >> 6) << 4); mn = (bw >> 4) | ((bz >> 6) << 4); }
    s = hbits2f(S.x) * float(sc);
    o = -hbits2f(S.x >> 16) * float(mn);
}

// ---------------------------------------------------------------- 型ごと
// C: 1 かたまり（32 要素）ぶんの自分の行の生データ、S: 1 単位（256 要素）ぶんの倍率
// ldc(単位の先頭, かたまり c, lane)・lds(単位の先頭, lane)・deq(C, A の 16 レジスタ = 8 歩)・
// scale(S, c, s[NG], o): かたまり c の倍率（NG = 2 なら 16 要素ごと）と、Σx に掛ける o
// R: 先読みするかたまりの数（レジスタの量で決める）
template <int T> struct W;
#ifndef G3_R6
#define G3_R6 4
#endif
#ifndef G3_R5
#define G3_R5 8
#endif
#ifndef G3_R4
#define G3_R4 8
#endif
#ifndef G3_R8
#define G3_R8 4
#endif

template <> struct W<Q_Q4_K> {
    static constexpr int NG = 1, R = G3_R4;
    static constexpr bool HAS_O = true;
    struct C { uint4 q; };
    struct S { uint4 v; };
    __device__ static C ldc(const uint8_t* u, int c, int lane) { return {wl16(u + c * 512 + lane * 16)}; }
    __device__ static S lds(const uint8_t* u, int lane) { return {wl16(u + 4096 + lane * 16)}; }
    __device__ static void deq(const C& x, uint32_t* a) {
        const uint32_t w[4] = {x.q.x, x.q.y, x.q.z, x.q.w};
        const uint32_t m = magic64();
#pragma unroll
        for (int ww = 0; ww < 4; ++ww) deq_word<0>(w[ww], m, m, m, m, a + 4 * ww);
    }
    __device__ static void scale(const S& s, int c, float* sc, float& o) { kscale(s.v, c, sc[0], o); }
};

template <> struct W<Q_Q5_K> {
    static constexpr int NG = 1, R = G3_R5;
    static constexpr bool HAS_O = true;
    struct C { uint4 q; uint32_t h; };
    struct S { uint4 v; };
    __device__ static C ldc(const uint8_t* u, int c, int lane) { return {wl16(u + c * 512 + lane * 16), wl4(u + 4096 + c * 128 + lane * 4)}; }
    __device__ static S lds(const uint8_t* u, int lane) { return {wl16(u + 5120 + lane * 16)}; }
    __device__ static void deq(const C& x, uint32_t* a) {
        const uint32_t w[4] = {x.q.x, x.q.y, x.q.z, x.q.w};
        const uint32_t m = magic64();
#pragma unroll
        for (int ww = 0; ww < 4; ++ww) {
            const uint32_t ha = rotr(x.h, rot5(2 * ww)), hb = rotr(x.h, rot5(2 * ww + 1));
            deq_word<0>(w[ww], and_or<0x00100010u>(ha, m), and_or<0x01000100u>(ha, m), and_or<0x00100010u>(hb, m),
                        and_or<0x01000100u>(hb, m), a + 4 * ww);
        }
    }
    __device__ static void scale(const S& s, int c, float* sc, float& o) { kscale(s.v, c, sc[0], o); }
};

template <> struct W<Q_Q6_K> {
    static constexpr int NG = 2, R = G3_R6;  // 16 要素ごとに scale
    static constexpr bool HAS_O = false;
    struct C { uint4 q; uint2 h; };
    struct S { uint4 sc; uint32_t d; };
    __device__ static C ldc(const uint8_t* u, int c, int lane) { return {wl16(u + c * 512 + lane * 16), wl8(u + 4096 + c * 256 + lane * 8)}; }
    __device__ static S lds(const uint8_t* u, int lane) { return {wl16(u + 6144 + lane * 16), wl2(u + 6656 + lane * 2)}; }
    __device__ static void deq(const C& x, uint32_t* a) {
        const uint32_t w[4] = {x.q.x, x.q.y, x.q.z, x.q.w};
        const uint32_t m = magic64();
#pragma unroll
        for (int ww = 0; ww < 4; ++ww) {
            const uint32_t hw = ww < 2 ? x.h.x : x.h.y;
            const uint32_t ha = rotr(hw, rot6(2 * (ww & 1))), hb = rotr(hw, rot6(2 * (ww & 1) + 1));
            deq_word<32>(w[ww], and_or<0x00300030u>(ha, m), and_or<0x03000300u>(ha, m), and_or<0x00300030u>(hb, m),
                         and_or<0x03000300u>(hb, m), a + 4 * ww);
        }
    }
    __device__ static void scale(const S& s, int c, float* sc, float& o) {
        const float dd = hbits2f(s.d);
        const uint32_t wd = c < 2 ? s.sc.x : c < 4 ? s.sc.y : c < 6 ? s.sc.z : s.sc.w;
        const uint32_t b = wd >> (16 * (c & 1));
        sc[0] = dd * float(int8_t(b & 0xff));
        sc[1] = dd * float(int8_t((b >> 8) & 0xff));
        o = 0.f;
    }
};

template <> struct W<Q_Q8_0> {
    static constexpr int NG = 1, R = G3_R8;
    static constexpr bool HAS_O = false;
    struct C { uint4 q0, q1; };
    struct S { uint4 d; };
    __device__ static C ldc(const uint8_t* u, int c, int lane) { return {wl16(u + c * 1024 + lane * 16), wl16(u + c * 1024 + 512 + lane * 16)}; }
    __device__ static S lds(const uint8_t* u, int lane) { return {wl16(u + 8192 + lane * 16)}; }
    __device__ static void deq(const C& x, uint32_t* a) {
        const uint32_t w[8] = {x.q0.x, x.q0.y, x.q0.z, x.q0.w, x.q1.x, x.q1.y, x.q1.z, x.q1.w};
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            a[2 * s] = hadd2u(__byte_perm(w[s], 0x64646464u, 0x4140), -1152.f);
            a[2 * s + 1] = hadd2u(__byte_perm(w[s], 0x64646464u, 0x4342), -1152.f);
        }
    }
    __device__ static void scale(const S& s, int c, float* sc, float& o) {
        const uint32_t wd = c < 2 ? s.d.x : c < 4 ? s.d.y : c < 6 ? s.d.z : s.d.w;
        sc[0] = hbits2f(wd >> (16 * (c & 1)));
        o = 0.f;
    }
};

template <> struct W<Q_IQ4_NL> {
    static constexpr int NG = 1, R = G3_R4;
    static constexpr bool HAS_O = false;
    struct C { uint4 q; };
    struct S { uint4 d; };
    __device__ static C ldc(const uint8_t* u, int c, int lane) { return {wl16(u + c * 512 + lane * 16)}; }
    __device__ static S lds(const uint8_t* u, int lane) { return {wl16(u + 4096 + lane * 16)}; }
    __device__ static void deq(const C& x, uint32_t* a) {
        deq_iq4_word(x.q.x, a);
        deq_iq4_word(x.q.y, a + 4);
        deq_iq4_word(x.q.z, a + 8);
        deq_iq4_word(x.q.w, a + 12);
    }
    __device__ static void scale(const S& s, int c, float* sc, float& o) {
        const uint32_t wd = c < 2 ? s.d.x : c < 4 ? s.d.y : c < 6 ? s.d.z : s.d.w;
        sc[0] = hbits2f(wd >> (16 * (c & 1)));
        o = 0.f;
    }
};

template <> struct W<Q_IQ4_XS> {
    static constexpr int NG = 1, R = G3_R4;
    static constexpr bool HAS_O = false;
    struct C { uint4 q; };
    struct S { uint2 v; };  // x = d | scales_h << 16、y = scales_l[4]
    __device__ static C ldc(const uint8_t* u, int c, int lane) { return {wl16(u + c * 512 + lane * 16)}; }
    __device__ static S lds(const uint8_t* u, int lane) { return {wl8(u + 4096 + lane * 8)}; }
    __device__ static void deq(const C& x, uint32_t* a) { W<Q_IQ4_NL>::deq(W<Q_IQ4_NL>::C{x.q}, a); }
    __device__ static void scale(const S& s, int c, float* sc, float& o) {
        const int ls = int((s.v.y >> (4 * c)) & 0xf) | int(((s.v.x >> (16 + 2 * c)) & 3) << 4);
        sc[0] = hbits2f(s.v.x) * float(ls - 32);
        o = 0.f;
    }
};

// ---------------------------------------------------------------- mma
__device__ __forceinline__ void mma884(float* d, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
// C = 0 で始める（溜めを 0 にする手間を省く）
__device__ __forceinline__ void mma884z(float* d, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    const float z = 0.f;
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
        "{%12,%12,%12,%12,%12,%12,%12,%12};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "f"(z));
}

// ---------------------------------------------------------------- x の下ごしらえ（作る側のカーネルからも使う）
// 並び（B 本・k 要素）: xhl[b][2][k]（f16 の上・下）、続けて xsi[b][k/32]（float2: 上+下の和、倍率の逆数）
// 倍率は 32 要素ごとの 2 のべきで max|x|·倍率 < 2^13（下の f16 が非正規数に落ちないように）
// 1 lane = 8 要素、4 lane（lane & ~3 が同じ）= 1 かたまり。v は倍率を掛ける前の値、hi・lo は自分の 8 要素の置き場、
// si はかたまりの (Σx, 倍率の逆数)。shfl を使うので warp の 32 lane 全員で呼ぶ（書かない lane は put = false）
__device__ __forceinline__ void prep8(const float (&v)[8], __half* hi, __half* lo, float2* si, bool put, bool put_si) {
    float m = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) m = fmaxf(m, fabsf(v[i]));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
    int e = 0;
    frexpf(m, &e);
    e = max(e, -100);
    const float sc = m > 0.f ? ldexpf(1.f, 13 - e) : 1.f;
    __align__(16) __half h[8], l[8];
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float y = v[i] * sc;
        h[i] = __float2half_rn(y);
        l[i] = __float2half_rn(y - __half2float(h[i]));
        sum += __half2float(h[i]) + __half2float(l[i]);
    }
    if (put) {
        *reinterpret_cast<uint4*>(hi) = *reinterpret_cast<const uint4*>(h);
        *reinterpret_cast<uint4*>(lo) = *reinterpret_cast<const uint4*>(l);
    }
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    if (put && put_si) *si = make_float2(sum, 1.f / sc);
}

// スレッドが B で受け持つ列（0..7）と、D で持つ行（タイルの中の 0..31。もう 1 つは +2）
__device__ __forceinline__ int b_col(int lane) { return (lane & 3) + 4 * (lane >> 4); }
__device__ __forceinline__ int d_row0(int lane) { return 8 * ((lane >> 2) & 3) + (lane & 1) + 4 * (lane >> 4); }

}  // namespace g3
