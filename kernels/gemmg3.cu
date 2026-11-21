// プレフィル用の GEMM（g3 の並びの重みを直接読む）: Y[M,N] = X[M,K] · W^T。
//   W は g3 の並び（kernels/g3.cuh。32 行 × 256 要素の単位を面に分けた並び）のまま。fp16 の写し・戻しの置き場は要らない。
//   X は fp16 の [M][K]（行の間隔 ldx）。精度を上げる道（XL）では X = 上 + 下 の 2 つの fp16（同じ ldx）を掛けて足す。
// ブロック = 8 warp・W の 256 行（g3 のタイル 8 つ）× TOKB 語。どの warp も「作る」と「掛ける」の両方をやる:
//   作る: warp w がタイル w の次のかたまり（32 要素）を読み（lane = g3 の置き場 slot = 1 行。512 B 続きの読み）、
//         fp16 に戻して（K 系は f32 で q·d − m を 1 回だけ丸める）共有メモリの段へ。X の次の段もブロックの全員で置く
//   掛ける: 今の段を mma.m8n8k4（fp16 × fp16 → f32）で。warp は FM×16 語 × FN×16 行（A = X、B = W）
//   段は 2 つ（交互）。1 かたまりごとに __syncthreads 1 回。重みの生のバイトは 2 かたまり先までレジスタに読んでおく
// 型は warp ごと（タイルごと）に実行時に選ぶ → 型の違う行列（gate と up、qkv と z と α・β）を 1 回の起動で。
// 書き出しは f32・f16・f32 に足す（残差）・SwiGLU（silu(gate)·up を f16 で）。K は分けない（毎回同じ答え）。
#include <atomic>
#include <cstdint>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "g3.cuh"

using namespace g3;

// 1 = 型によらず全部 f32 で倍率を掛けてから fp16 に丸める（既定は K 系だけ。ほかは half2 で 1 回掛ける）
#ifndef G3G_F32ALL
#define G3G_F32ALL 0
#endif

namespace {

enum Mode : int { M_F32 = 0, M_F16 = 1, M_ADD = 2, M_SWIGLU = 3, M_SWIGLU_HL = 4 };
constexpr int MAXMAT = 4;
constexpr int NW = 8;          // 1 ブロックの warp 数
constexpr int NTH = NW * 32;
constexpr int BR = NW * TILE;  // 1 ブロックの W の行（段の行）= 256

struct Mat {
    const uint8_t* w;  // g3 の並び
    void* y;           // 出力（行の間隔 ldy 要素）
    int type, n, ldy, tiles, t0;  // tiles = ceil(n/32)、t0 = 通しのタイル番号の始め
};
struct Args {
    Mat mat[MAXMAT];
    const __half* x;
    const __half* xlo;  // XL の道だけ
    int nmat, mode, k, m, ldx, ntiles;
};
// MoE の専門家ごと: ブロックの行 y = 仕事 grp[3y..3y+2] = (専門家 e, 行の頭 m0, 行の数 mr)。
// W_i = mat[i].w + e·es[i]。X の行 r は xidx[r]（null なら r）。Y の行は r のまま（専門家の順に並んだ行）。
// ⭐ 密な道（27B）は Args のまま・MoE の分岐なしで作る: 1 つの型に混ぜた 11-14 の版で、密なプレフィルの g3 GEMM が 5.6% 遅くなった（GPU1 の nsys・8K 語）
struct ArgsM : Args {
    const int* grp;
    const int* xidx;
    int64_t es[MAXMAT];
};
template <class A> constexpr bool IS_M = std::is_same_v<A, ArgsM>;

// タイルの組: ブロックの語 TOKB、語の向きの warp WT（行の向きは NW/WT）、warp の FM×16 語 × FN×16 行
template <int TOKB_, int WT_, int FM_, int FN_>
struct Cfg {
    static constexpr int TOKB = TOKB_, WT = WT_, WR = NW / WT, FM = FM_, FN = FN_;
    static constexpr int TOKW = TOKB / WT;
    static constexpr int WST = BR * 64;    // 段の W（256 行 × 32 要素 × 2 B）
    static constexpr int XST = TOKB * 64;  // 段の X（上か下の 1 つぶん）
    static constexpr int XV = (TOKB * 4 + NTH - 1) / NTH;  // X の 16 B をスレッドあたり何個
    static_assert(TOKW == FM * 16, "語の割り振り");
    static_assert(WR * FN * 16 == BR, "行の割り振り");
    static_assert(FN % 2 == 0, "SwiGLU は gate と up を半分ずつ");
    template <bool XL> static constexpr int smem() { return 2 * (WST + XST * (XL ? 2 : 1)); }
};
using CA = Cfg<128, 2, 4, 4>;  // 大きな M（warp 64 語 × 64 行）
using CB = Cfg<64, 1, 4, 2>;   // M ≦ 64（warp 64 語 × 32 行）
using CC = Cfg<32, 1, 2, 2>;   // M ≦ 32
using CD = Cfg<16, 1, 1, 2>;   // M ≦ 16

// 段の中の (行 r, 16 バイトの番号 c) の位置。16 バイトの番号を行の bit1・bit3 で XOR（gemm.cu と同じ）:
// 掛ける側の読み（quad-pair の 4 行）も、作る側の書き（lane 0..7 = 行 {0-3, 8-11}）も bank がぶつからない
__device__ __forceinline__ int swz(int r) { return ((r >> 1) & 1) | (((r >> 3) & 1) << 1); }
__device__ __forceinline__ int soff(int r, int c) { return r * 64 + ((c ^ swz(r)) << 4); }

__device__ __forceinline__ __half2 u2h2(uint32_t u) {
    __half2 h;
    *reinterpret_cast<uint32_t*>(&h) = u;
    return h;
}
__device__ __forceinline__ uint32_t h22u(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ uint32_t pack2(float a, float b) { return h22u(__floats2half2_rn(a, b)); }

// かたまり ci（単位の中の 0..7）を fp16 の重み 32 個（16 語、k の順）に: w = sc·q + o
template <int T>
__device__ __forceinline__ void deq_scaled(const typename W<T>::C& c, const typename W<T>::S& s, int ci, uint32_t (&o16)[16]) {
    using WT = W<T>;
    uint32_t a[16];
    WT::deq(c, a);
    float sc[WT::NG], o;
    WT::scale(s, ci, sc, o);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int g = WT::NG == 2 ? i >> 3 : 0;
        if (WT::HAS_O || G3G_F32ALL) {
            const float2 f = __half22float2(u2h2(a[i]));
            o16[i] = pack2(fmaf(sc[g], f.x, o), fmaf(sc[g], f.y, o));
        } else {
            o16[i] = h22u(__hmul2(u2h2(a[i]), __float2half2_rn(sc[g])));
        }
    }
}

// 本体（作る側の型 T が決まった後）。acc[i][j] = 語の区画 i × 行の区画 j の 16×16
template <class C, int T, bool XL, class A>
__device__ __forceinline__ void mainloop(const A& a, uint8_t* sm, int warp, int lane, const uint8_t* tp, bool tv, int m0, int mr,
                                         const int (&brow)[C::FN], float (&acc)[C::FM][C::FN][8]) {
    using WT = W<T>;
    using CW = typename WT::C;
    using SW = typename WT::S;
    constexpr int STG = C::WST + C::XST * (XL ? 2 : 1);
    const int nk = a.k / 32;
    const int64_t ub = unit_bytes(T);
    const int tid = threadIdx.x;
    const int srow = warp * 32 + row_of_slot(lane);  // 作る側: この lane が戻す段の行

    // X: スレッドの受け持ちの 16 B（語 row・16 B の番号 c）
    const __half* xs[C::XV];
    int xo[C::XV];
    bool xw[C::XV];
#pragma unroll
    for (int v = 0; v < C::XV; ++v) {
        const int idx = tid + v * NTH, row = idx >> 2, c = idx & 3;
        xw[v] = idx < C::TOKB * 4;
        int xr_ = xw[v] && row < mr ? m0 + row : 0;
        if constexpr (IS_M<A>) {
            if (xw[v] && row < mr && a.xidx) xr_ = a.xidx[m0 + row];
        }
        xs[v] = xw[v] && row < mr ? a.x + int64_t(xr_) * a.ldx + c * 8 : nullptr;
        xo[v] = soff(row, c);
    }
    const int64_t lo_off = XL ? a.xlo - a.x : 0;
    uint4 xr[C::XV], xl[C::XV];
    auto ldx = [&](int t) {
#pragma unroll
        for (int v = 0; v < C::XV; ++v) {
            xr[v] = xs[v] ? ld16(xs[v] + t * 32) : make_uint4(0, 0, 0, 0);
            if (XL) xl[v] = xs[v] ? ld16(xs[v] + lo_off + t * 32) : make_uint4(0, 0, 0, 0);
        }
    };
    auto stx = [&](uint8_t* st) {
#pragma unroll
        for (int v = 0; v < C::XV; ++v)
            if (xw[v]) {
                *reinterpret_cast<uint4*>(st + C::WST + xo[v]) = xr[v];
                if (XL) *reinterpret_cast<uint4*>(st + C::WST + C::XST + xo[v]) = xl[v];
            }
    };
    // W: かたまり t の生のバイト（と、その単位の倍率）
    auto ldw = [&](int t, CW& c, SW& s) {
        if (tv) {
            const uint8_t* up = tp + int64_t(t >> 3) * ub;
            c = WT::ldc(up, t & 7, lane);
            s = WT::lds(up, lane);
        }
    };
    auto stw_all = [&](const CW& c, const SW& s, int t, uint8_t* st) {
        uint32_t o16[16];
        deq_scaled<T>(c, s, t & 7, o16);
#pragma unroll
        for (int cc = 0; cc < 4; ++cc)
            *reinterpret_cast<uint4*>(st + soff(srow, cc)) = make_uint4(o16[4 * cc], o16[4 * cc + 1], o16[4 * cc + 2], o16[4 * cc + 3]);
    };

    // 掛ける側の読みの位置
    const int q = (lane >> 2) & 3, l4 = (lane & 3) + 4 * (lane >> 4);
    const int wt = warp / C::WR;
    const int ar = wt * C::TOKW + (q >> 1) * 8 + l4;
    const int sa = swz(ar), sb = swz((q & 1) * 8 + l4);

    CW cA{}, cB{};
    SW sA{}, sB{};
    // 始め: かたまり 0 を段 0 に、1・2 をレジスタに、X の段 1 も
    ldw(0, cA, sA);
    ldx(0);
    stw_all(cA, sA, 0, sm);
    stx(sm);
    if (nk > 1) ldw(1, cA, sA);
    if (nk > 2) ldw(2, cB, sB);
    if (nk > 1) ldx(1);
    __syncthreads();

    // 1 段: 段 t を掛けながら、かたまり t+1（c・s）を戻して次の段へ。終わったら c に t+3 を読む
    auto step = [&](int t, CW& c, SW& s) {
        const uint8_t* st = sm + (t & 1) * STG;
        uint8_t* nx = sm + ((t + 1) & 1) * STG;
        const bool more = t + 1 < nk;
        uint32_t o16[16];
        if (more) deq_scaled<T>(c, s, (t + 1) & 7, o16);
#pragma unroll
        for (int cc = 0; cc < 4; ++cc) {
            uint4 fa[C::FM], fl[C::FM], fb[C::FN];
#pragma unroll
            for (int i = 0; i < C::FM; ++i) {
                const int o = C::WST + (ar + 16 * i) * 64 + ((cc ^ sa) << 4);
                fa[i] = *reinterpret_cast<const uint4*>(st + o);
                if (XL) fl[i] = *reinterpret_cast<const uint4*>(st + o + C::XST);
            }
#pragma unroll
            for (int j = 0; j < C::FN; ++j) fb[j] = *reinterpret_cast<const uint4*>(st + brow[j] * 64 + ((cc ^ sb) << 4));
            if (more)
                *reinterpret_cast<uint4*>(nx + soff(srow, cc)) = make_uint4(o16[4 * cc], o16[4 * cc + 1], o16[4 * cc + 2], o16[4 * cc + 3]);
#pragma unroll
            for (int i = 0; i < C::FM; ++i)
#pragma unroll
                for (int j = 0; j < C::FN; ++j) mma884(acc[i][j], fa[i].x, fa[i].y, fb[j].x, fb[j].y);
#pragma unroll
            for (int i = 0; i < C::FM; ++i)
#pragma unroll
                for (int j = 0; j < C::FN; ++j) mma884(acc[i][j], fa[i].z, fa[i].w, fb[j].z, fb[j].w);
            if (XL) {
#pragma unroll
                for (int i = 0; i < C::FM; ++i)
#pragma unroll
                    for (int j = 0; j < C::FN; ++j) mma884(acc[i][j], fl[i].x, fl[i].y, fb[j].x, fb[j].y);
#pragma unroll
                for (int i = 0; i < C::FM; ++i)
#pragma unroll
                    for (int j = 0; j < C::FN; ++j) mma884(acc[i][j], fl[i].z, fl[i].w, fb[j].z, fb[j].w);
            }
        }
        if (more) stx(nx);
        if (t + 3 < nk) ldw(t + 3, c, s);
        if (t + 2 < nk) ldx(t + 2);
        __syncthreads();
    };
    // nk は 8 の倍数（k は 256 の倍数）: かたまり t+1 は t が偶数なら cA、奇数なら cB
    for (int t = 0; t < nk; t += 2) {
        step(t, cA, sA);
        step(t + 1, cB, sB);
    }
}

template <class C, bool XL, class A>
__global__ void __launch_bounds__(NTH, 1) g3gemm_kernel(const __grid_constant__ A a) {
    extern __shared__ __align__(16) uint8_t sm[];
    constexpr int FM = C::FM, FN = C::FN;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bool hl = IS_M<A> && a.mode == M_SWIGLU_HL;
    const bool sw = a.mode == M_SWIGLU || hl;
    const int bx = blockIdx.x;
    int m0 = blockIdx.y * C::TOKB, mr = min(C::TOKB, a.m - m0), ge = 0;
    if constexpr (IS_M<A>) {
        if (a.grp) {
            ge = a.grp[3 * blockIdx.y], m0 = a.grp[3 * blockIdx.y + 1], mr = a.grp[3 * blockIdx.y + 2];
            if (mr <= 0) return;  // 余りの仕事（ブロックの全員が同じ所で抜ける）
        }
    }
    const int mend = IS_M<A> ? m0 + mr : a.m;  // 密な道は前と同じ a.m（m0 + mr を持つとレジスタが溢れた）

    // 作る側: この warp が戻すタイル（行列 pm のタイル pt）
    int pm = 0, pt;
    if (sw) {
        pm = warp < 4 ? 0 : 1;
        pt = bx * 4 + (warp & 3);
    } else {
        const int g = bx * NW + warp;
        while (pm + 1 < a.nmat && g >= a.mat[pm].t0 + a.mat[pm].tiles) ++pm;
        pt = g - a.mat[pm].t0;
    }
    const int ptype = a.mat[pm].type;
    const bool tv = pt < a.mat[pm].tiles;
    int64_t eoff = 0;
    if constexpr (IS_M<A>) {
        if (a.grp) eoff = int64_t(ge) * a.es[pm];
    }
    const uint8_t* tp = a.mat[pm].w + eoff + int64_t(tv ? pt : 0) * tile_bytes(ptype, a.k);

    // 掛ける側: 行の区画 j の段の行（SwiGLU は前半 j が gate、後半が up の同じ行）
    const int q = (lane >> 2) & 3, l4 = (lane & 3) + 4 * (lane >> 4);
    const int wr = warp % C::WR, wt = warp / C::WR;
    int brow[FN], rblk[FN];
#pragma unroll
    for (int j = 0; j < FN; ++j) {
        if (sw) {
            const int jj = j % (FN / 2);
            rblk[j] = wr * (FN / 2) * 16 + 16 * jj;  // gate の段の行（0..127）
            brow[j] = (j < FN / 2 ? 0 : BR / 2) + rblk[j] + (q & 1) * 8 + l4;
        } else {
            rblk[j] = wr * FN * 16 + 16 * j;
            brow[j] = rblk[j] + (q & 1) * 8 + l4;
        }
    }

    float acc[FM][FN][8];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

    switch (ptype) {
#define CASE(T) \
    case T: mainloop<C, T, XL, A>(a, sm, warp, lane, tp, tv, m0, mr, brow, acc); break;
        CASE(Q_Q4_K) CASE(Q_Q5_K) CASE(Q_Q6_K) CASE(Q_Q8_0) CASE(Q_IQ4_NL) CASE(Q_IQ4_XS)
#undef CASE
    }

    // 書き出し: 区画の要素 e（mma.m8n8k4 の f32 の並び）は 語 (q>>1)*8 + (lane&1) + (e&2) + 4*(lane>>4)、
    // 行 (q&1)*8 + (e&4) + (lane&2) + (e&1)。e と e+1 は続いた 2 行 → 2 つずつ書く
    const int er = (q >> 1) * 8 + (lane & 1) + 4 * (lane >> 4), ec = (q & 1) * 8 + (lane & 2);
    const int tok0 = m0 + wt * C::TOKW + er;
    const int mode = sw ? M_F16 : a.mode;
    constexpr int JN = FN;  // SwiGLU は前半だけ書く
#pragma unroll
    for (int j = 0; j < JN; ++j) {
        if (sw && j >= FN / 2) break;
        int mi = 0, row0;
        if (sw) {
            row0 = (bx * 4 + (rblk[j] >> 5)) * TILE + (rblk[j] & 31) + ec;
        } else {
            const int g = bx * NW + (rblk[j] >> 5);
            while (mi + 1 < a.nmat && g >= a.mat[mi].t0 + a.mat[mi].tiles) ++mi;
            row0 = (g - a.mat[mi].t0) * TILE + (rblk[j] & 31) + ec;
        }
        const Mat& mt = a.mat[mi];
#pragma unroll
        for (int i = 0; i < FM; ++i)
#pragma unroll
            for (int e = 0; e < 8; e += 2) {
                const int tok = tok0 + 16 * i + (e & 2), row = row0 + (e & 4);
                if (tok >= mend || row >= mt.n) continue;
                float v0 = acc[i][j][e], v1 = acc[i][j][e + 1];
                if (sw) {
                    const float u0 = acc[i][j + FN / 2][e], u1 = acc[i][j + FN / 2][e + 1];
                    v0 = v0 / (1.f + __expf(-v0)) * u0;
                    v1 = v1 / (1.f + __expf(-v1)) * u1;
                }
                const int64_t o = int64_t(tok) * mt.ldy + row;
                const bool two = row + 1 < mt.n;
                if (hl) {
                    // SwiGLU の答えを上・下の 2 つの fp16 に（上 = mat[0].y、下 = mat[1].y。次の GEMM の XL の X）
                    const __half h0 = __float2half_rn(v0), h1 = __float2half_rn(v1);
                    __half* yp = static_cast<__half*>(a.mat[0].y) + o;
                    __half* lp = static_cast<__half*>(a.mat[1].y) + o;
                    if (two) {
                        *reinterpret_cast<__half2*>(yp) = __halves2half2(h0, h1);
                        *reinterpret_cast<__half2*>(lp) = __floats2half2_rn(v0 - __half2float(h0), v1 - __half2float(h1));
                    } else {
                        yp[0] = h0, lp[0] = __float2half_rn(v0 - __half2float(h0));
                    }
                } else if (mode == M_F16) {
                    __half* yp = static_cast<__half*>(mt.y) + o;
                    if (two) *reinterpret_cast<__half2*>(yp) = __floats2half2_rn(v0, v1);
                    else yp[0] = __float2half_rn(v0);
                } else {
                    float* yp = static_cast<float*>(mt.y) + o;
                    if (two) {
                        float2 p = make_float2(v0, v1);
                        if (mode == M_ADD) {
                            const float2 b = *reinterpret_cast<const float2*>(yp);
                            p.x += b.x, p.y += b.y;
                        }
                        *reinterpret_cast<float2*>(yp) = p;
                    } else {
                        yp[0] = mode == M_ADD ? yp[0] + v0 : v0;
                    }
                }
            }
    }
}

constexpr int MAXDEV = 64;
int cur_dev() {
    int dev = 0;
    cudaGetDevice(&dev);
    return dev >= 0 && dev < MAXDEV ? dev : 0;
}

template <class C, bool XL, class A>
int launch(const A& a, cudaStream_t s) {
    auto f = g3gemm_kernel<C, XL, A>;
    constexpr int SMEM = C::template smem<XL>();
    static_assert(SMEM <= 96 * 1024, "共有メモリは 96KB まで");
    static std::atomic<bool> attr[MAXDEV] = {};
    const int dev = cur_dev();
    if (!attr[dev].load(std::memory_order_acquire)) {
        cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        attr[dev].store(true, std::memory_order_release);
    }
    const bool sw = a.mode == M_SWIGLU || a.mode == M_SWIGLU_HL;
    const int nbx = sw ? (a.mat[0].tiles + 3) / 4 : (a.ntiles + NW - 1) / NW;
    if (nbx == 0) return 0;
    int nby = (a.m + C::TOKB - 1) / C::TOKB;
    if constexpr (IS_M<A>) {
        if (a.grp) nby = a.m;  // grp のときの a.m = 仕事の数
    }  // grp のときの a.m = 仕事の数
    if (nby == 0) return 0;
    const dim3 grid(nbx, nby);
    f<<<grid, NTH, SMEM, s>>>(a);
    return int(cudaGetLastError());
}

template <bool XL, class A>
int run_cfg(int cfg, const A& a, cudaStream_t s) {
    switch (cfg) {
        case 0: return launch<CA, XL, A>(a, s);
        case 1: return launch<CB, XL, A>(a, s);
        case 2: return launch<CC, XL, A>(a, s);
        case 3: return launch<CD, XL, A>(a, s);
    }
    return -4;
}

// M からタイルの組を選ぶ（V100 で測って決めた）
int auto_cfg(int m) {
    if (m <= 16) return 3;
    if (m <= 32) return 2;
    if (m <= 64) return 1;
    return 0;
}

}  // namespace

extern "C" {

// 何本かの行列（同じ X、4 本まで）を 1 回で: Y_i[m][ldys[i]] = X[m][k] · W_i^T。W_i は g3 の並び（qtypes は詰め直した後の型）。
// mode: 0 = f32、1 = f16、2 = f32 の Y に足す（残差）、3 = SwiGLU（nmat = 2: W_0 = gate・W_1 = up・同じ n、Y_0 に silu(g)·u を f16 で）。
// xlo: null でなければ X = x + xlo（fp16 の上・下）を掛ける（掛け算は 2 倍）。cfg: -1 で M から自動。ldys: null なら n。
// 戻り値: 0 = 成功、-1 = 型が無い、-2 = k が 256 の倍数でない、-3 = 引数が変、-4 = cfg が無い、-5 = 揃っていない、それ以外は cudaError
int infer_g3_gemm_ex(int nmat, const int* qtypes, const void* const* ws, const int* ns, void* const* ys, const int* ldys, int k, int m,
                     const void* x, const void* xlo, int ldx, int mode, int cfg, cudaStream_t s) {
    if (nmat < 1 || nmat > MAXMAT || mode < M_F32 || mode > M_SWIGLU) return -3;
    if (mode == M_SWIGLU && (nmat != 2 || ns[0] != ns[1])) return -3;
    if (k % UK) return -2;
    if ((reinterpret_cast<uintptr_t>(x) & 15) || (ldx & 7) || (xlo && (reinterpret_cast<uintptr_t>(xlo) & 15))) return -5;
    if (m <= 0) return 0;
    Args a{};
    int nt = 0;
    for (int i = 0; i < nmat; ++i) {
        if (packed_type(qtypes[i]) != qtypes[i]) return -1;
        if (reinterpret_cast<uintptr_t>(ws[i]) & 15) return -5;
        const int ldy = ldys ? ldys[i] : ns[i];
        if (ldy & 1) return -5;
        a.mat[i] = Mat{static_cast<const uint8_t*>(ws[i]), ys[i], qtypes[i], ns[i], ldy, (ns[i] + TILE - 1) / TILE, nt};
        nt += a.mat[i].tiles;
    }
    a.x = static_cast<const __half*>(x);
    a.xlo = static_cast<const __half*>(xlo);
    a.nmat = nmat, a.mode = mode, a.k = k, a.m = m, a.ldx = ldx, a.ntiles = nt;
    if (cfg < 0) cfg = auto_cfg(m);
    return xlo ? run_cfg<true>(cfg, a, s) : run_cfg<false>(cfg, a, s);
}

// MoE のプレフィル（専門家ごとの GEMM・Tensor Core）: 仕事 j < n_blk = (専門家 e, 行の頭 m0, 行の数 mr ≦ tokb) を grp[3j..]（GPU の上）に。
// Y_i[m0 + t][·] = X[xidx[m0 + t]] · W_i[e]^T（t < mr）。W_i[e] = ws[i] + e·es[i]。tokb はタイルの組の語の数（16・32・64・128）。
// mode は infer_g3_gemm_ex と同じ（3 = SwiGLU で gate・up を 1 回に）＋ 4 = SwiGLU の答えを上（ys[0]）・下（ys[1]）の fp16 に。
// xlo: null でなければ X = x + xlo（上・下の fp16。前の GEMV の道と同じ精度）。xidx が null なら X の行は m0 + t
int infer_g3_moe_gemm(int nmat, const int* qtypes, const void* const* ws, const int64_t* es, const int* ns, void* const* ys, const int* ldys,
                      int k, const void* x, const void* xlo, int ldx, const int* xidx, const int* grp, int n_blk, int tokb, int mode,
                      cudaStream_t s) {
    if (nmat < 1 || nmat > MAXMAT || mode < M_F32 || mode > M_SWIGLU_HL || !grp) return -3;
    if ((mode == M_SWIGLU || mode == M_SWIGLU_HL) && (nmat != 2 || ns[0] != ns[1])) return -3;
    if (xlo && (reinterpret_cast<uintptr_t>(xlo) & 15)) return -5;
    if (k % UK) return -2;
    if ((reinterpret_cast<uintptr_t>(x) & 15) || (ldx & 7)) return -5;
    if (n_blk <= 0) return 0;
    if (n_blk > 65535) return -3;
    ArgsM a{};
    int nt = 0;
    for (int i = 0; i < nmat; ++i) {
        if (packed_type(qtypes[i]) != qtypes[i]) return -1;
        if ((reinterpret_cast<uintptr_t>(ws[i]) & 15) || (es[i] & 15)) return -5;
        const int ldy = ldys ? ldys[i] : ns[i];
        if (ldy & 1) return -5;
        a.mat[i] = Mat{static_cast<const uint8_t*>(ws[i]), ys[i], qtypes[i], ns[i], ldy, (ns[i] + TILE - 1) / TILE, nt};
        a.es[i] = es[i];
        nt += a.mat[i].tiles;
    }
    a.x = static_cast<const __half*>(x);
    a.xlo = static_cast<const __half*>(xlo);
    a.nmat = nmat, a.mode = mode, a.k = k, a.m = n_blk, a.ldx = ldx, a.ntiles = nt;
    a.grp = grp, a.xidx = xidx;
    int cfg = tokb >= 128 ? 0 : tokb >= 64 ? 1 : tokb >= 32 ? 2 : 3;
    return xlo ? run_cfg<true>(cfg, a, s) : run_cfg<false>(cfg, a, s);
}

// 1 本: Y[m][n] = X[m][k] · W^T（mode 0・1・2）
int infer_g3_gemm(int qtype, const void* w, int k, int n, int m, const void* x, int ldx, void* y, int mode, cudaStream_t s) {
    return infer_g3_gemm_ex(1, &qtype, &w, &n, &y, nullptr, k, m, x, nullptr, ldx, mode, -1, s);
}

// gate と up をまとめて: Y[m][n] = silu(X·Wg^T) · (X·Wu^T)（f16）。型は別々でよい
int infer_g3_gemm_swiglu(int tg, const void* wg, int tu, const void* wu, int k, int n, int m, const void* x, int ldx, void* y,
                         cudaStream_t s) {
    const int ts[2] = {tg, tu}, ns[2] = {n, n};
    const void* ws[2] = {wg, wu};
    void* ys[2] = {y, y};
    return infer_g3_gemm_ex(2, ts, ws, ns, ys, nullptr, k, m, x, nullptr, ldx, M_SWIGLU, -1, s);
}

}  // extern "C"
