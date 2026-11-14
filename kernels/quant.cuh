// SPDX-License-Identifier: Apache-2.0 AND MIT
// Copyright 2026 blueqat Inc.
// Portions derived from ggml (llama.cpp), MIT License, Copyright (c) 2023-2026 The ggml authors. See THIRD_PARTY_LICENSES.md.
// GGUF の量子化ブロックを、32 要素の「かたまり」単位で戻す。
// 1 つのかたまりを 4 本のスレッド（sub = 0..3）で分け、各スレッドが 8 要素を戻す。
// 置き方は llama.cpp の CPU 版 dequantize_row_*（ggml-quants.c, MIT）と同じ並び。
#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include "tables.cuh"

// ggml の型番号
enum QType : int {
    Q_F32 = 0, Q_F16 = 1, Q_Q5_1 = 7, Q_Q8_0 = 8, Q_Q3_K = 11, Q_Q4_K = 12, Q_Q5_K = 13, Q_Q6_K = 14, Q_IQ3_XXS = 18,
    Q_IQ4_NL = 20, Q_IQ3_S = 21, Q_IQ2_S = 22, Q_IQ4_XS = 23, Q_BF16 = 30, Q_Q2_0 = 42,
};

// 1 行のバイト数（k 要素）
__host__ __device__ inline int64_t qtype_row_bytes(int t, int64_t k) {
    switch (t) {
        case Q_F32: return k * 4;
        case Q_F16: case Q_BF16: return k * 2;
        case Q_Q8_0: return k / 32 * 34;
        case Q_IQ4_NL: return k / 32 * 18;
        case Q_Q2_0: return k / 64 * 18;
        case Q_Q5_1: return k / 32 * 24;
        case Q_Q3_K: return k / 256 * 110;
        case Q_Q4_K: return k / 256 * 144;
        case Q_Q5_K: return k / 256 * 176;
        case Q_Q6_K: return k / 256 * 210;
        case Q_IQ3_XXS: return k / 256 * 98;
        case Q_IQ3_S: return k / 256 * 110;
        case Q_IQ2_S: return k / 256 * 82;
        case Q_IQ4_XS: return k / 256 * 136;
    }
    return -1;
}

__device__ __forceinline__ float h2f(const uint8_t* p) {
    return __half2float(*reinterpret_cast<const __half*>(p));
}
__device__ __forceinline__ float bf2f(uint16_t b) {
    return __uint_as_float(uint32_t(b) << 16);
}

// iq4 系と Q5_1 は、8 要素の位置が「4 個 + 16 先の 4 個」になる。それ以外は連続 8 個。
template <int T> struct Pattern { static constexpr bool iq4 = false; };
template <> struct Pattern<Q_IQ4_NL> { static constexpr bool iq4 = true; };
template <> struct Pattern<Q_IQ4_XS> { static constexpr bool iq4 = true; };
template <> struct Pattern<Q_Q5_1> { static constexpr bool iq4 = true; };

// かたまり c（行の先頭から 32 要素単位）の sub 番目の 8 要素を v に戻す。
// 連続なら位置は c*32 + sub*8 + j、iq4 なら c*32 + sub*4 + (j<4 ? j : j+12)。
template <int T> __device__ __forceinline__ void deq8(const uint8_t* row, int c, int sub, float v[8]);

template <> __device__ __forceinline__ void deq8<Q_F32>(const uint8_t* row, int c, int sub, float v[8]) {
    const float* p = reinterpret_cast<const float*>(row) + c * 32 + sub * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = p[j];
}
template <> __device__ __forceinline__ void deq8<Q_F16>(const uint8_t* row, int c, int sub, float v[8]) {
    const __half* p = reinterpret_cast<const __half*>(row) + c * 32 + sub * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = __half2float(p[j]);
}
template <> __device__ __forceinline__ void deq8<Q_BF16>(const uint8_t* row, int c, int sub, float v[8]) {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(row) + c * 32 + sub * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = bf2f(p[j]);
}
template <> __device__ __forceinline__ void deq8<Q_Q8_0>(const uint8_t* row, int c, int sub, float v[8]) {
    const uint8_t* b = row + c * 34;
    const float d = h2f(b);
    const int8_t* q = reinterpret_cast<const int8_t*>(b + 2) + sub * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = d * q[j];
}
template <> __device__ __forceinline__ void deq8<Q_Q2_0>(const uint8_t* row, int c, int sub, float v[8]) {
    // 64 要素で 1 ブロック（d と 16 バイト）。値は (code - 1) * d
    const uint8_t* b = row + (c >> 1) * 18;
    const float d = h2f(b);
    const uint8_t* q = b + 2 + (c & 1) * 8 + sub * 2;
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = float(int((q[j >> 2] >> ((j & 3) * 2)) & 3) - 1) * d;
}
template <> __device__ __forceinline__ void deq8<Q_IQ4_NL>(const uint8_t* row, int c, int sub, float v[8]) {
    const uint8_t* b = row + c * 18;
    const float d = h2f(b);
    const uint8_t* q = b + 2 + sub * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = d * kvalues_iq4nl[q[j] & 0xf];
        v[j + 4] = d * kvalues_iq4nl[q[j] >> 4];
    }
}
template <> __device__ __forceinline__ void deq8<Q_IQ4_XS>(const uint8_t* row, int c, int sub, float v[8]) {
    // 256 要素で 1 ブロック: d(2) scales_h(2) scales_l(4) qs(128)
    const uint8_t* b = row + (c >> 3) * 136;
    const int ib = c & 7;
    const float d = h2f(b);
    const uint16_t sh = *reinterpret_cast<const uint16_t*>(b + 2);
    const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
    const float dl = d * (ls - 32);
    const uint8_t* q = b + 8 + 16 * ib + sub * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = dl * kvalues_iq4nl[q[j] & 0xf];
        v[j + 4] = dl * kvalues_iq4nl[q[j] >> 4];
    }
}
template <> __device__ __forceinline__ void deq8<Q_Q6_K>(const uint8_t* row, int c, int sub, float v[8]) {
    // 256 要素で 1 ブロック: ql(128) qh(64) scales(16) d(2)
    const uint8_t* b = row + (c >> 3) * 210;
    const int n = (c & 7) >> 2, quarter = c & 3;  // 128 の半分のどちらか、その中の 32 の 4 区切り
    const float d = h2f(b + 208);
    const uint8_t* ql = b + n * 64 + (quarter & 1) * 32;
    const uint8_t* qh = b + 128 + n * 32;
    const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192) + n * 8 + 2 * quarter;
    const int shift_l = (quarter >> 1) * 4, shift_h = 2 * quarter;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int l = sub * 8 + j;
        const int q = int(((ql[l] >> shift_l) & 0xf) | (((qh[l] >> shift_h) & 3) << 4)) - 32;
        v[j] = d * sc[l / 16] * q;
    }
}
template <> __device__ __forceinline__ void deq8<Q_IQ3_XXS>(const uint8_t* row, int c, int sub, float v[8]) {
    // 256 要素で 1 ブロック: d(2) qs(64) scales_and_signs(32)
    const uint8_t* b = row + (c >> 3) * 98;
    const int ib = c & 7;
    const float d = h2f(b);
    const uint8_t* qs = b + 2 + 8 * ib;
    const uint8_t* ss = b + 2 + 64 + 4 * ib;
    const uint32_t aux = uint32_t(ss[0]) | (uint32_t(ss[1]) << 8) | (uint32_t(ss[2]) << 16) | (uint32_t(ss[3]) << 24);
    const float db = d * (0.5f + (aux >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux >> (7 * sub)) & 127];
    const uint32_t g1 = iq3xxs_grid[qs[2 * sub]], g2 = iq3xxs_grid[qs[2 * sub + 1]];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = db * float((g1 >> (8 * j)) & 0xff) * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
        v[j + 4] = db * float((g2 >> (8 * j)) & 0xff) * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
    }
}
template <> __device__ __forceinline__ void deq8<Q_IQ3_S>(const uint8_t* row, int c, int sub, float v[8]) {
    // 256 要素で 1 ブロック: d(2) qs(64) qh(8) signs(32) scales(4)。32 要素ごとに格子番号 8 個・上の位 8 ビット・符号 4 バイト
    const uint8_t* b = row + (c >> 3) * 110;
    const int ib = c & 7;
    const float d = h2f(b);
    const uint8_t* qs = b + 2 + 8 * ib;
    const uint8_t qh = b[2 + 64 + ib];
    const uint8_t s = b[2 + 64 + 8 + 4 * ib + sub];
    const float db = d * (1 + 2 * ((b[2 + 64 + 8 + 32 + ib / 2] >> (4 * (ib & 1))) & 0xf));
    const uint32_t g1 = iq3s_grid[qs[2 * sub] | ((qh << (8 - 2 * sub)) & 256)];
    const uint32_t g2 = iq3s_grid[qs[2 * sub + 1] | ((qh << (7 - 2 * sub)) & 256)];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = db * float((g1 >> (8 * j)) & 0xff) * (s & kmask_iq2xs[j] ? -1.f : 1.f);
        v[j + 4] = db * float((g2 >> (8 * j)) & 0xff) * (s & kmask_iq2xs[j + 4] ? -1.f : 1.f);
    }
}
template <> __device__ __forceinline__ void deq8<Q_IQ2_S>(const uint8_t* row, int c, int sub, float v[8]) {
    // 256 要素で 1 ブロック: d(2) qs(64 = 32 の格子番号 + 32 の符号) qh(8) scales(8)
    const uint8_t* b = row + (c >> 3) * 82;
    const int ib = c & 7;
    const float d = h2f(b);
    const uint8_t* qs = b + 2 + 4 * ib;
    const uint8_t* signs = b + 2 + 32 + 4 * ib;
    const uint8_t qh = b[2 + 64 + ib];
    const uint8_t sc = b[2 + 64 + 8 + ib];
    const float dl = d * (0.5f + ((sc >> (4 * (sub / 2))) & 0xf)) * 0.25f;
    const uint64_t g = iq2s_grid[qs[sub] | ((qh << (8 - 2 * sub)) & 0x300)];
    const uint8_t s = signs[sub];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = dl * float((g >> (8 * j)) & 0xff) * (s & kmask_iq2xs[j] ? -1.f : 1.f);
}

template <> __device__ __forceinline__ void deq8<Q_Q3_K>(const uint8_t* row, int c, int sub, float v[8]) {
    // 256 要素で 1 ブロック: hmask(32) qs(64) scales(12) d(2)。16 要素ごとに 6 ビットの scale（-32 して使う）
    // かたまり ib の要素 l は qs[32*(ib/4) + l] の 2*(ib%4) ビット目から 2 ビット、hmask[l] の ib ビット目が 0 なら -4
    const uint8_t* b = row + (c >> 3) * 110;
    const int ib = c & 7;
    const float d = h2f(b + 108);
    const int is = 2 * ib + (sub >> 1);  // 16 要素の区切り
    const uint8_t* sc = b + 96;
    const int ls = ((sc[is & 7] >> (4 * (is >> 3))) & 0xf) | (((sc[8 + (is & 3)] >> (2 * (is >> 2))) & 3) << 4);
    const float dl = d * (ls - 32);
    // 1 バイトずつ読むと読みの命令で詰まる（出力 [5120→248320] で 228GB/s）ので 2 バイトずつ。
    // ブロックは 110 バイトなので 2 バイト境界までしか揃わない
    const uint16_t* hm = reinterpret_cast<const uint16_t*>(b + sub * 8);
    const uint16_t* q = reinterpret_cast<const uint16_t*>(b + 32 + 32 * (ib >> 2) + sub * 8);
    const int sh = 2 * (ib & 3);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t ql = uint32_t(q[j]) >> sh, h = uint32_t(hm[j]) >> ib;
        // 下 2 ビット + 上の 1 ビット×4 - 4（上のビットが 0 なら -4）
        v[2 * j] = dl * (int((ql & 3) | ((h & 1) << 2)) - 4);
        v[2 * j + 1] = dl * (int(((ql >> 8) & 3) | (((h >> 8) & 1) << 2)) - 4);
    }
}

// K 系の 6 ビットの scale と min（ggml の get_scale_min_k4）
__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
template <> __device__ __forceinline__ void deq8<Q_Q4_K>(const uint8_t* row, int c, int sub, float v[8]) {
    // 256 要素で 1 ブロック: d(2) dmin(2) scales(12) qs(128)。32 要素ごとに scale と min
    const uint8_t* b = row + (c >> 3) * 144;
    const int ib = c & 7;
    int sc, mn;
    scale_min_k4(ib, b + 4, sc, mn);
    const float d = h2f(b) * sc, m = h2f(b + 2) * mn;
    const uint8_t* q = b + 16 + 32 * (ib >> 1) + sub * 8;
    const int sh = (ib & 1) * 4;
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = d * ((q[j] >> sh) & 0xf) - m;
}
template <> __device__ __forceinline__ void deq8<Q_Q5_K>(const uint8_t* row, int c, int sub, float v[8]) {
    // 256 要素で 1 ブロック: d(2) dmin(2) scales(12) qh(32) qs(128)
    const uint8_t* b = row + (c >> 3) * 176;
    const int ib = c & 7;
    int sc, mn;
    scale_min_k4(ib, b + 4, sc, mn);
    const float d = h2f(b) * sc, m = h2f(b + 2) * mn;
    const uint8_t* qh = b + 16 + sub * 8;
    const uint8_t* q = b + 48 + 32 * (ib >> 1) + sub * 8;
    const int sh = (ib & 1) * 4;
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = d * (((q[j] >> sh) & 0xf) + ((qh[j] >> ib) & 1 ? 16 : 0)) - m;
}
template <> __device__ __forceinline__ void deq8<Q_Q5_1>(const uint8_t* row, int c, int sub, float v[8]) {
    // 32 要素で 1 ブロック: d(2) m(2) qh(4) qs(16)。値は x*d + m
    const uint8_t* b = row + c * 24;
    const float d = h2f(b), m = h2f(b + 2);
    const uint32_t qh = uint32_t(b[4]) | (uint32_t(b[5]) << 8) | (uint32_t(b[6]) << 16) | (uint32_t(b[7]) << 24);
    const uint8_t* q = b + 8;
#pragma unroll
    for (int j0 = 0; j0 < 4; ++j0) {
        const int j = sub * 4 + j0;
        v[j0] = float((q[j] & 0xf) | (((qh >> j) << 4) & 0x10)) * d + m;
        v[j0 + 4] = float((q[j] >> 4) | ((qh >> (j + 12)) & 0x10)) * d + m;
    }
}

// 8 要素の行内の位置
template <int T> __device__ __forceinline__ int pos8(int c, int sub, int j) {
    return Pattern<T>::iq4 ? c * 32 + sub * 4 + (j < 4 ? j : j + 12) : c * 32 + sub * 8 + j;
}
