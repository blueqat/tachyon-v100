// プレフィル用の行列×行列: Y[M,N] = X[M,K] · W^T（Tensor Core の mma.m8n8k4、fp16 で掛けて fp32 で溜める）。
// W は ggml の並びのまま（N 行 × K 要素、行のバイト数は qtype_row_bytes）。X は fp16 の [M, K]（行の間隔 ldx）。
// ブロックの warp を 2 つに分ける（V100 は cp.async が無いので、読みと戻しを掛け算から切り離す）:
//   作る側（PW warp）: 各 warp が受け持つ行（BN/PW 行）の量子化のバイトを「単位」（K 系は 256 要素）ごとに
//     16 バイトずつまとめて読み（次の単位はレジスタに先読み）、共有メモリに置いて、そこから deqs（deq8 と同じ並び）で fp16 に戻す。X も詰める
//   掛ける側（CW warp）: 詰め終わった段を mma.m8n8k4 で掛けるだけ。段の受け渡しは名前付きの barrier（bar.sync / bar.arrive）
// 型は作る側の warp ごとに実行時に選ぶ → 型の違う行列（gate と up、qkv と gate）を 1 回の起動で掛けられる。
// 書き出しは f32・f16・f32 に足す（残差）・SwiGLU（silu(gate) * up を f16 で）。
// M が小さく列のタイルが少ないときは K を分けて（split-K）作業場に部分和を書き、決まった順に足す（毎回同じ答え）。
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "quant.cuh"

// 測って選んだ作り（-D で変えて比べられる）
#ifndef GEMM_CONSUMER
#define GEMM_CONSUMER 2  // 掛ける側の読みの先出し: 1 = 束ごと、2 = 2 束ずつ、3 = 1 束先を出し続ける
#endif
#ifndef GEMM_XPAIR
#define GEMM_XPAIR 1  // 作る側の X は 2 組のレジスタに交互に（写さない）
#endif
#ifndef GEMM_KQ_F32
#define GEMM_KQ_F32 1  // K 系（Q4_K・Q5_K）の戻しは fp32 で 1 回だけ丸める（half2 と速さは同じで、GEMM の誤差が 1/3）
#endif

namespace {

// 量子化の「単位」: 作る側が 1 度にまとめて読む幅。K は要素、B はバイト、O は単位の頭の 16 バイトからのずれの最大
// （揃いが 16 なら 0、8 なら 8、2 なら 14）。共有メモリには 16 バイトに揃えた窓をそのまま置き、戻すときにずれを足して読む
// （揃え直すと、届いたばかりの読みをすぐ使うので待ちが出た）
template <int T> struct Unit;
template <> struct Unit<Q_Q4_K> { static constexpr int K = 256, B = 144, O = 0; };
template <> struct Unit<Q_Q5_K> { static constexpr int K = 256, B = 176, O = 0; };
template <> struct Unit<Q_Q6_K> { static constexpr int K = 256, B = 210, O = 14; };
template <> struct Unit<Q_IQ4_XS> { static constexpr int K = 256, B = 136, O = 8; };
template <> struct Unit<Q_IQ4_NL> { static constexpr int K = 256, B = 144, O = 14; };
template <> struct Unit<Q_Q8_0> { static constexpr int K = 128, B = 136, O = 14; };
template <> struct Unit<Q_Q3_K> { static constexpr int K = 256, B = 110, O = 14; };
template <> struct Unit<Q_IQ3_S> { static constexpr int K = 256, B = 110, O = 14; };
template <> struct Unit<Q_F16> { static constexpr int K = 64, B = 128, O = 0; };  // k が 64 の倍数
constexpr int RAWROW = 224;  // 共有メモリの 1 行（いちばん大きい Q6_K の 210 + 14 を 16 に切り上げ）
constexpr int KALIGN = 256;  // K と split-K の切れ目はこの倍数（どの型の単位でも割り切れる）

#define GEMM_TYPES(X) X(Q_Q4_K) X(Q_Q5_K) X(Q_Q6_K) X(Q_IQ4_XS) X(Q_IQ4_NL) X(Q_Q8_0) X(Q_Q3_K) X(Q_IQ3_S) X(Q_F16)

__host__ __device__ constexpr bool known_type(int t) {
    return t == Q_Q4_K || t == Q_Q5_K || t == Q_Q6_K || t == Q_IQ4_XS || t == Q_IQ4_NL || t == Q_Q8_0 || t == Q_Q3_K ||
           t == Q_IQ3_S || t == Q_F16;
}

// 書き出しの形
enum Mode : int { M_F32 = 0, M_F16 = 1, M_ADD = 2, M_SWIGLU = 3 };

constexpr int MAXMAT = 4;
struct Mat {
    const uint8_t* w;  // n 行 × k 要素
    int64_t rb;        // 1 行のバイト数
    void* y;           // 出力（行の間隔 ldy 要素）
    int type, n, ldy, tiles;  // tiles = 列のタイル数
};
struct Args {
    Mat mat[MAXMAT];
    const __half* x;
    float* ws;  // split-K の作業場（分けごとに m × n の f32）
    int nmat, mode, k, m, ldx, kchunk;  // kchunk = 1 つの分けの要素数
};

// タイルの形。BM×BN をブロック、WM×WN を掛ける側の 1 warp が受け持つ。BK は 32（deq8 のかたまり 1 つ）
template <int BM_, int BN_, int BK_, int WM_, int WN_, int PW_, int ST_>
struct Tile {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, WM = WM_, WN = WN_, PW = PW_, ST = ST_;
    static constexpr int WARPS_N = BN / WN, CW = (BM / WM) * WARPS_N;  // 掛ける側の warp 数
    static constexpr int CT = CW * 32, PT = PW * 32, NT = CT + PT;
    static constexpr int RPW = BN / PW;               // 作る側の 1 warp の行数
    static constexpr int STAGE = (BM + BN) * 64;      // 1 段のバイト（1 行 32 要素 = 64 バイト、並びは soff）
    static constexpr int RING = ST * STAGE;           // 段の輪のバイト
    static constexpr int SMEM = RING + BN * RAWROW;   // 全部のバイト
    static constexpr int XVEC = BM * 4, XV = (XVEC + PT - 1) / PT;  // X の 16 バイトの束の数と、作る側の 1 スレッドあたり
    static constexpr int WT = RPW * 4 / 32;           // 作る側の 1 lane が 1 段で戻す 8 要素の数
    static_assert(BK == 32 && BM % WM == 0 && BN % WN == 0 && WM % 16 == 0 && WN % 16 == 0, "タイルの形");
    static_assert(RPW % 16 == 0 && BM % 16 == 0, "作る側の割り振り（16 行ずつ）");
    static_assert(WARPS_N % 2 == 0 && (BN / 2) % RPW == 0, "SwiGLU は gate と up を半分ずつ");
    static_assert(BM * BN * 2 + CW * 16 * 20 * 4 <= RING, "書き出しの置き場が足りない");
    static_assert(SMEM <= 96 * 1024, "共有メモリは 96KB まで");
    static_assert(2 * ST + 1 <= 15, "名前付きの barrier は 16 個まで（15 は掛ける側だけの待ち合わせ）");
};

// 名前付きの barrier: 段 s が詰まった = FULL(s)、空いた = EMPTY(s)。0 は __syncthreads が使う
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }

__device__ __forceinline__ uint32_t pack2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// 段の中の (行 r, 16 バイトの番号 c) のバイト位置。16 バイトの番号を行の bit 1 と bit 3 で XOR する:
// 掛ける側の読み（quad-pair の 4 行 / 8 列）も作る側の書き（8 lane が行 {0-3, 8-11}）も bank がぶつからない
__device__ __forceinline__ int swz(int r) { return ((r >> 1) & 1) | (((r >> 3) & 1) << 1); }
__device__ __forceinline__ int soff(int r, int c) { return r * 64 + ((c ^ swz(r)) << 4); }
// 作る側の 16 バイトの束の番号 j → (行, 16 バイトの番号)。8 本の lane が行 {0,1,2,3,8,9,10,11}（+4・+16…）を受け持つ
__device__ __forceinline__ void jpos(int j, int& r, int& c) {
    c = (j >> 3) & 3;
    r = (j & 3) | ((j & 4) << 1) | (((j >> 5) & 1) << 2) | ((j >> 6) << 4);
}

// mma.m8n8k4（Volta の Tensor Core の素の命令）。warp の 4 つの quad-pair がそれぞれ 8×8×4 を掛ける
__device__ __forceinline__ void mma884(float* d, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// 共有メモリの単位の 1 行の間隔（バイト）: 16 の倍数で、語の数が 32 で割って 8 か 24 余る
// （同じ sub を読む 8 行 {0-3, 8-11} が bank をちょうど 2 回ずつ使う）
__host__ __device__ constexpr int raw_stride(int b) {
    int w = (b + 15) / 16 * 4;
    while (w % 16 != 8) w += 4;
    return w * 4;
}

// ---- 共有メモリの単位（b が単位の頭。揃いは型の揃いのまま）から 8 要素を fp16 に戻す。並びは deq8<T> と同じ
// （連続 8 個、iq4 系は「4 個 + 16 先の 4 個」）。計算は half2: バイト b を PRMT 1 回で fp16 の 1024 + b（ちょうど）にし、
// 引いてから掛ける（I2F・F2F を使わない。値は q*d - m を 1 回丸めたものと 1 ulp 以内）
__device__ __forceinline__ __half2 u2h2(uint32_t u) {
    __half2 h;
    *reinterpret_cast<uint32_t*>(&h) = u;
    return h;
}
__device__ __forceinline__ uint32_t h22u(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
constexpr uint32_t K1024 = 0x64006400u, K1056 = 0x64206420u, K1152 = 0x64806480u;  // 1024・1024+32・1024+128
// 4 バイトの組（要素 0〜3）を half2 2 つの (1024 + b) に通して f を掛け、uint4 の半分に
template <class F>
__device__ __forceinline__ uint4 pack8h(uint32_t lo, uint32_t hi, F f) {
    return make_uint4(h22u(f(u2h2(__byte_perm(lo, 0x64646464u, 0x4140)))), h22u(f(u2h2(__byte_perm(lo, 0x64646464u, 0x4342)))),
                      h22u(f(u2h2(__byte_perm(hi, 0x64646464u, 0x4140)))), h22u(f(u2h2(__byte_perm(hi, 0x64646464u, 0x4342)))));
}
__device__ __forceinline__ uint32_t lds16(const uint8_t* p) { return *reinterpret_cast<const uint16_t*>(p); }
__device__ __forceinline__ uint32_t lds32(const uint8_t* p) { return *reinterpret_cast<const uint32_t*>(p); }
__device__ __forceinline__ uint2 lds64(const uint8_t* p) { return *reinterpret_cast<const uint2*>(p); }
// 2 バイト揃いの 4 バイト
__device__ __forceinline__ uint32_t lds32a2(const uint8_t* p) { return lds16(p) | (lds16(p + 2) << 16); }
// fp16 の d を (d, d) に
__device__ __forceinline__ __half2 dup_lo(uint32_t w) { return u2h2(__byte_perm(w, 0, 0x1010)); }

// iq4nl の表（+128 して符号なしに）を 4 つの語に置き、番号 q（4 ビット × 4 本、下位 16 ビット）を引いて 4 バイトに
__device__ __forceinline__ uint32_t iq4_lookup4(uint32_t q) {
    // kvalues_iq4nl + 128 = {1, 24, 45, 63, 79, 93, 106, 118, 129, 141, 153, 166, 181, 197, 217, 241}
    constexpr uint32_t t0 = 0x3f2d1801u, t1 = 0x766a5d4fu, t2 = 0xa6998d81u, t3 = 0xf1d9c5b5u;
    const uint32_t sel = q & 0x7777u;
    const uint32_t lo = __byte_perm(t0, t1, sel), hi = __byte_perm(t2, t3, sel);
    return __byte_perm(lo, hi, 0x3210u | ((q & 0x8888u) >> 1));
}
// 4 バイトの下位 4 ビット（要素 0〜3）と上位 4 ビット（要素 4〜7）の表引き
__device__ __forceinline__ void iq4_lookup8(uint32_t w, uint32_t& lo, uint32_t& hi) {
    const uint32_t a = iq4_lookup4(w), b = iq4_lookup4(w >> 16);  // [b0.lo b0.hi b1.lo b1.hi]、[b2.lo …]
    lo = __byte_perm(a, b, 0x6420);
    hi = __byte_perm(a, b, 0x7531);
}

// K 系の 6 ビットの scale と min（かたまり c、hd = 単位の頭 16 バイト: d dmin scales[12]）→ (d*sc, dmin*mn) の half2
__device__ __forceinline__ void ksm(const uint4& hd, int c, uint32_t& sc, uint32_t& mn) {
    const int sh = (c & 3) * 8;
    if (c < 4) {
        sc = (hd.y >> sh) & 63;
        mn = (hd.z >> sh) & 63;
    } else {
        sc = ((hd.w >> sh) & 0xf) | ((((hd.y >> sh) >> 6) & 3) << 4);
        mn = (((hd.w >> sh) >> 4) & 0xf) | ((((hd.z >> sh) >> 6) & 3) << 4);
    }
}
__device__ __forceinline__ __half2 kscale2(const uint4& hd, int c) {
    uint32_t sc, mn;
    ksm(hd, c, sc, mn);
    return __hmul2(u2h2(hd.x), __hsub2(u2h2(K1024 | sc | (mn << 16)), u2h2(K1024)));
}

// K 系の fp32 の道（GEMM_KQ_F32=1）: q*d - m を fp32 で計算して 1 回だけ丸める。half2 の道は d*sc・dmin*mn を
// fp16 に丸めるので、32 要素で揃った誤差が乗る（Q5_K の GEMM で二乗平均 2e-4 → 7e-4）
// バイトから float は 2^23 を足した数のビットを作って引く（I2F を使わない）
__device__ __forceinline__ float u8f(uint32_t w, int k) {
    return __int_as_float(__byte_perm(w, 0x4B000000u, 0x7540 | k)) - 8388608.f;
}
__device__ __forceinline__ uint4 pack8k(uint32_t lo, uint32_t hi, float d, float m) {
    auto f = [&](uint32_t w, int k) { return fmaf(d, u8f(w, k), -m); };
    return make_uint4(pack2(f(lo, 0), f(lo, 1)), pack2(f(lo, 2), f(lo, 3)), pack2(f(hi, 0), f(hi, 1)), pack2(f(hi, 2), f(hi, 3)));
}
__device__ __forceinline__ void kscale(const uint4& hd, int c, float& d, float& m) {
    uint32_t sc, mn;
    ksm(hd, c, sc, mn);
    const float2 dd = __half22float2(u2h2(hd.x));
    d = dd.x * (__int_as_float(0x4B000000u | sc) - 8388608.f);
    m = dd.y * (__int_as_float(0x4B000000u | mn) - 8388608.f);
}

template <int T> __device__ __forceinline__ uint4 deqs(const uint8_t* b, int c, int sub);

template <> __device__ __forceinline__ uint4 deqs<Q_Q4_K>(const uint8_t* b, int c, int sub) {
    // d(2) dmin(2) scales(12) qs(128)
    const uint2 q = lds64(b + 16 + 32 * (c >> 1) + sub * 8);
    const int sh = (c & 1) * 4;
#if GEMM_KQ_F32
    float d, m;
    kscale(*reinterpret_cast<const uint4*>(b), c, d, m);
    return pack8k((q.x >> sh) & 0x0f0f0f0fu, (q.y >> sh) & 0x0f0f0f0fu, d, m);
#else
    const __half2 dm = kscale2(*reinterpret_cast<const uint4*>(b), c);
    const __half2 d = __low2half2(dm), nm = __hneg2(__high2half2(dm));
    return pack8h((q.x >> sh) & 0x0f0f0f0fu, (q.y >> sh) & 0x0f0f0f0fu,
                  [&](__half2 h) { return __hfma2(__hsub2(h, u2h2(K1024)), d, nm); });
#endif
}
template <> __device__ __forceinline__ uint4 deqs<Q_Q5_K>(const uint8_t* b, int c, int sub) {
    // d(2) dmin(2) scales(12) qh(32) qs(128)
    const uint2 h = lds64(b + 16 + sub * 8);
    const uint2 q = lds64(b + 48 + 32 * (c >> 1) + sub * 8);
    const int sh = (c & 1) * 4;
    const uint32_t lo = ((q.x >> sh) & 0x0f0f0f0fu) | (((h.x >> c) & 0x01010101u) << 4);
    const uint32_t hi = ((q.y >> sh) & 0x0f0f0f0fu) | (((h.y >> c) & 0x01010101u) << 4);
#if GEMM_KQ_F32
    float d, m;
    kscale(*reinterpret_cast<const uint4*>(b), c, d, m);
    return pack8k(lo, hi, d, m);
#else
    const __half2 dm = kscale2(*reinterpret_cast<const uint4*>(b), c);
    const __half2 d = __low2half2(dm), nm = __hneg2(__high2half2(dm));
    return pack8h(lo, hi, [&](__half2 v) { return __hfma2(__hsub2(v, u2h2(K1024)), d, nm); });
#endif
}
template <> __device__ __forceinline__ uint4 deqs<Q_Q6_K>(const uint8_t* b, int c, int sub) {
    // ql(128) qh(64) scales(16) d(2)。scale は符号付き 8 ビット → +128 して 1024+128 を引く。
    // b は 2 バイト揃いなので 8 バイトは 4 バイト揃いの語 3 つから 0 か 16 ビットずらして作る
    const int n = c >> 2, qq = c & 3;
    const int sh = int(reinterpret_cast<uintptr_t>(b) & 2) * 8;
    const uint8_t* b4 = b - (reinterpret_cast<uintptr_t>(b) & 2);
    auto ld8 = [&](int off) {
        const uint32_t w0 = lds32(b4 + off), w1 = lds32(b4 + off + 4), w2 = lds32(b4 + off + 8);
        return make_uint2(__funnelshift_r(w0, w1, sh), __funnelshift_r(w1, w2, sh));
    };
    const uint2 l = ld8(n * 64 + (qq & 1) * 32 + sub * 8);
    const uint2 h = ld8(128 + n * 32 + sub * 8);
    const uint32_t sc = (b[192 + n * 8 + 2 * qq + (sub >> 1)] ^ 0x80u) | 0x6400u;
    const __half2 d = __hmul2(dup_lo(lds16(b + 208)), __hsub2(dup_lo(sc), u2h2(K1152)));
    const int sl = (qq >> 1) * 4, shh = 2 * qq;
    const uint32_t lo = ((l.x >> sl) & 0x0f0f0f0fu) | (((h.x >> shh) & 0x03030303u) << 4);
    const uint32_t hi = ((l.y >> sl) & 0x0f0f0f0fu) | (((h.y >> shh) & 0x03030303u) << 4);
    return pack8h(lo, hi, [&](__half2 v) { return __hmul2(__hsub2(v, u2h2(K1056)), d); });
}
template <> __device__ __forceinline__ uint4 deqs<Q_IQ4_XS>(const uint8_t* b, int c, int sub) {
    // d(2) scales_h(2) scales_l(4) qs(128)
    const uint2 hd = lds64(b);
    const uint32_t ls = ((hd.y >> (4 * c)) & 0xf) | (((hd.x >> (16 + 2 * c)) & 3) << 4);
    const __half2 dl = __hmul2(dup_lo(hd.x), __hsub2(dup_lo(0x6400u | ls), u2h2(K1056)));
    uint32_t lo, hi;
    iq4_lookup8(lds32(b + 8 + 16 * c + 4 * sub), lo, hi);
    return pack8h(lo, hi, [&](__half2 v) { return __hmul2(__hsub2(v, u2h2(K1152)), dl); });
}
template <> __device__ __forceinline__ uint4 deqs<Q_IQ4_NL>(const uint8_t* b, int c, int sub) {
    // 32 要素で d(2) qs(16)
    const uint8_t* p = b + 18 * c;
    const __half2 d = dup_lo(lds16(p));
    uint32_t lo, hi;
    iq4_lookup8(lds32a2(p + 2 + 4 * sub), lo, hi);
    return pack8h(lo, hi, [&](__half2 v) { return __hmul2(__hsub2(v, u2h2(K1152)), d); });
}
template <> __device__ __forceinline__ uint4 deqs<Q_Q8_0>(const uint8_t* b, int c, int sub) {
    // 32 要素で d(2) qs(32)。符号付きのバイトは最上位ビットを反転して +128 にする
    const uint8_t* p = b + 34 * c;
    const __half2 d = dup_lo(lds16(p));
    const uint32_t lo = lds32a2(p + 2 + 8 * sub) ^ 0x80808080u, hi = lds32a2(p + 6 + 8 * sub) ^ 0x80808080u;
    return pack8h(lo, hi, [&](__half2 v) { return __hmul2(__hsub2(v, u2h2(K1152)), d); });
}
template <> __device__ __forceinline__ uint4 deqs<Q_Q3_K>(const uint8_t* b, int c, int sub) {
    // hmask(32) qs(64) scales(12) d(2)。かたまり c（n = c/4 番目の 128、j = c%4）の要素 e は
    // q = (qs[32n + e] >> 2j) & 3、上のビット h = (hmask[e] >> c) & 1、値 = d * (scale[2c + e/16] - 32) * (q + 4h - 4)。
    // 16 個の 6 ビットの scale は下 4 ビットが scales[s]（s < 8）か scales[s-8] の上 4 ビット、上 2 ビットが scales[8 + s%4] の
    // 2*(s/4) ビット目から。b は 2 バイト揃い（Q6_K と同じく 4 バイトの語 3 つからずらして 8 バイト）
    const int n = c >> 2, j = c & 3, s = 2 * c + (sub >> 1);
    const int sh = int(reinterpret_cast<uintptr_t>(b) & 2) * 8;
    const uint8_t* b4 = b - (reinterpret_cast<uintptr_t>(b) & 2);
    auto ld8 = [&](int off) {
        const uint32_t w0 = lds32(b4 + off), w1 = lds32(b4 + off + 4), w2 = lds32(b4 + off + 8);
        return make_uint2(__funnelshift_r(w0, w1, sh), __funnelshift_r(w1, w2, sh));
    };
    const uint2 h = ld8(sub * 8);
    const uint2 q = ld8(32 + 32 * n + sub * 8);
    const int lo4 = s < 8 ? (b[96 + s] & 0xf) : (b[96 + s - 8] >> 4);
    const int hi2 = (b[104 + (s & 3)] >> (2 * (s >> 2))) & 3;
    const float dl = __half2float(__ushort_as_half(uint16_t(lds16(b + 108)))) * float((lo4 | (hi2 << 4)) - 32);
    const uint32_t lo = ((q.x >> (2 * j)) & 0x03030303u) | (((h.x >> c) & 0x01010101u) << 2);
    const uint32_t hi = ((q.y >> (2 * j)) & 0x03030303u) | (((h.y >> c) & 0x01010101u) << 2);
    return pack8k(lo, hi, dl, 4.f * dl);
}
template <> __device__ __forceinline__ uint4 deqs<Q_IQ3_S>(const uint8_t* b, int c, int sub) {
    // d(2) qs(64) qh(8) signs(32) scales(4)。格子の表（iq3s_grid、512 語）を引くので quant.cuh の deq8 をそのまま使う
    // （b は単位の頭・2 バイト揃い、c は単位の中のかたまり。値 = d·(1 + 2·scale)·格子の値·符号 は f32 で正確で、fp16 に 1 回だけ丸める。
    // この GGUF では blk.14.ffn_down の 1 本だけなので速さは詰めていない）
    float v[8];
    deq8<Q_IQ3_S>(b, c, sub, v);
    return make_uint4(pack2(v[0], v[1]), pack2(v[2], v[3]), pack2(v[4], v[5]), pack2(v[6], v[7]));
}
template <> __device__ __forceinline__ uint4 deqs<Q_F16>(const uint8_t* b, int c, int sub) {
    return *reinterpret_cast<const uint4*>(b + (c * 32 + sub * 8) * 2);
}

// 作る側: この warp の行 [0, nvalid) は wrow0 + r*rb から。K の範囲 [kb, ke) を段に詰める。
// ring は段の輪、raw はこの warp の単位の置き場
template <int T, class C>
__device__ __forceinline__ void produce(const uint8_t* __restrict__ wrow0, int64_t rb, int nvalid, const __half* __restrict__ x,
                                        int ldx, int m0, int mr, int kb, int ke, uint8_t* ring, uint8_t* raw, int pw) {
    using U = Unit<T>;
    constexpr int RPW = C::RPW, NW = (U::B + U::O + 15) / 16, RS = raw_stride(NW * 16);
    static_assert(RS <= RAWROW, "単位が共有メモリの行に入らない");
    constexpr int LR = (RPW * NW + 31) / 32;  // 1 lane が 1 単位で読む 16 バイトの数
    constexpr int SPU = U::K / 32;            // 1 単位の段数
    const int lane = threadIdx.x & 31, pt = threadIdx.x - C::CT;
    const int nk = (ke - kb) / 32;
    int u = kb / U::K;
    const int u_end = ke / U::K;

    // 単位 uu を読む: lane は (行, 16 バイトの窓の番号) を受け持つ（並んだ lane が同じ行の続きを読む）。
    // 窓は単位の頭を 16 バイトに切り下げた所から。要る所（単位の終わりより前）だけ読む
    uint4 rr[LR];  // n より先の行は読まないので 0 のまま（戻した値はその列にしか効かず、書き出さない）
#pragma unroll
    for (int v = 0; v < LR; ++v) rr[v] = make_uint4(0, 0, 0, 0);
    auto load_raw = [&](int uu) {
#pragma unroll
        for (int v = 0; v < LR; ++v) {
            const int j = lane + 32 * v, r = j / NW, i = j % NW;
            if (j < RPW * NW && r < nvalid) {
                const uint8_t* st = wrow0 + r * rb + int64_t(uu) * U::B;
                const int o = int(reinterpret_cast<uintptr_t>(st) & 15);
                if (16 * i < o + U::B) rr[v] = __ldg(reinterpret_cast<const uint4*>(st - o) + i);
            }
        }
    };
    // 行 r の単位の頭のずれ
    auto roff = [&](int r) { return int(reinterpret_cast<uintptr_t>(wrow0 + r * rb + int64_t(u) * U::B) & 15); };
    auto store_raw = [&]() {
#pragma unroll
        for (int v = 0; v < LR; ++v) {
            const int j = lane + 32 * v;
            if (j < RPW * NW) *reinterpret_cast<uint4*>(raw + (j / NW) * RS + (j % NW) * 16) = rr[v];
        }
    };

    // X の 1 段（BM 行 × 32 要素 = 16 バイト × 4）
    const __half* xsrc[C::XV];
    int xoff[C::XV];
#pragma unroll
    for (int v = 0; v < C::XV; ++v) {
        const int idx = pt + v * C::PT;
        int row, c;
        jpos(idx, row, c);
        xsrc[v] = idx < C::XVEC && row < mr ? x + int64_t(m0 + row) * ldx + kb + c * 8 : nullptr;
        xoff[v] = soff(row, c);
    }
    auto load_x = [&](int t, uint4* xr) {
#pragma unroll
        for (int v = 0; v < C::XV; ++v)
            xr[v] = xsrc[v] ? __ldg(reinterpret_cast<const uint4*>(xsrc[v] + t * 32)) : make_uint4(0, 0, 0, 0);
    };

    load_raw(u);
    store_raw();
    __syncwarp();
    if (u + 1 < u_end) load_raw(u + 1);
#if GEMM_XPAIR
    // X は 2 組のレジスタに交互に読む（段 t を詰めたらその組に段 t+2 を読む）。組を写すと読みの届くのを待ってしまう。
    // nk は 8 の倍数なので 2 段ずつ回せる
    uint4 xa[C::XV], xb[C::XV];
    load_x(0, xa);
    load_x(1, xb);
    auto stage = [&](int t, uint4 (&xr)[C::XV]) {
        const int s = t % C::ST, cu = t % SPU;  // cu = 単位の中のかたまり番号
        uint4 wr[C::WT];
#pragma unroll
        for (int v = 0; v < C::WT; ++v) {
            int r, sub;
            jpos(lane + 32 * v, r, sub);
            wr[v] = deqs<T>(raw + r * RS + (U::O ? roff(r) : 0), cu, sub);
        }
        if (t >= C::ST) bar_sync(1 + C::ST + s, C::NT);
        uint8_t* xs = ring + s * C::STAGE;
        uint8_t* ws = xs + C::BM * 64 + pw * C::RPW * 64;
#pragma unroll
        for (int v = 0; v < C::XV; ++v)
            if (pt + v * C::PT < C::XVEC) *reinterpret_cast<uint4*>(xs + xoff[v]) = xr[v];
        if (t + 2 < nk) load_x(t + 2, xr);
#pragma unroll
        for (int v = 0; v < C::WT; ++v) {
            int r, sub;
            jpos(lane + 32 * v, r, sub);
            if (Pattern<T>::iq4) {
                const int h = (sub & 1) * 8;
                *reinterpret_cast<uint2*>(ws + soff(r, sub >> 1) + h) = make_uint2(wr[v].x, wr[v].y);
                *reinterpret_cast<uint2*>(ws + soff(r, 2 + (sub >> 1)) + h) = make_uint2(wr[v].z, wr[v].w);
            } else {
                *reinterpret_cast<uint4*>(ws + soff(r, sub)) = wr[v];
            }
        }
        bar_arrive(1 + s, C::NT);
        if (t % SPU == SPU - 1 && t + 1 < nk) {
            __syncwarp();
            store_raw();
            __syncwarp();
            ++u;
            if (u + 1 < u_end) load_raw(u + 1);
        }
    };
    for (int t = 0; t < nk; t += 2) {
        stage(t, xa);
        stage(t + 1, xb);
    }
#else
    uint4 xr[C::XV];
    load_x(0, xr);
    for (int t = 0; t < nk; ++t) {
        const int s = t % C::ST, cu = t % SPU;  // cu = 単位の中のかたまり番号
        // W を共有メモリの単位から戻す（束 j = 行 × sub）
        uint4 wr[C::WT];
#pragma unroll
        for (int v = 0; v < C::WT; ++v) {
            int r, sub;
            jpos(lane + 32 * v, r, sub);
            wr[v] = deqs<T>(raw + r * RS + (U::O ? roff(r) : 0), cu, sub);
        }
        // 次の段の X を読み始める（先に出すと戻しの命令の並びが崩れて遅くなった）
        uint4 xn[C::XV];
        if (t + 1 < nk) load_x(t + 1, xn);
        if (t >= C::ST) bar_sync(1 + C::ST + s, C::NT);
        uint8_t* xs = ring + s * C::STAGE;
        uint8_t* ws = xs + C::BM * 64 + pw * C::RPW * 64;
#pragma unroll
        for (int v = 0; v < C::XV; ++v)
            if (pt + v * C::PT < C::XVEC) *reinterpret_cast<uint4*>(xs + xoff[v]) = xr[v];
#pragma unroll
        for (int v = 0; v < C::WT; ++v) {
            int r, sub;
            jpos(lane + 32 * v, r, sub);
            if (Pattern<T>::iq4) {  // 4 個 + 16 先の 4 個 → 16 バイトの番号 sub/2 と 2+sub/2 の、前か後ろの 8 バイト
                const int h = (sub & 1) * 8;
                *reinterpret_cast<uint2*>(ws + soff(r, sub >> 1) + h) = make_uint2(wr[v].x, wr[v].y);
                *reinterpret_cast<uint2*>(ws + soff(r, 2 + (sub >> 1)) + h) = make_uint2(wr[v].z, wr[v].w);
            } else {
                *reinterpret_cast<uint4*>(ws + soff(r, sub)) = wr[v];
            }
        }
        bar_arrive(1 + s, C::NT);
#pragma unroll
        for (int v = 0; v < C::XV; ++v) xr[v] = xn[v];
        if (t % SPU == SPU - 1 && t + 1 < nk) {  // 単位の終わり: 先読みした次の単位を置く
            __syncwarp();
            store_raw();
            __syncwarp();
            ++u;
            if (u + 1 < u_end) load_raw(u + 1);
        }
    }
#endif
}

template <class C>
__global__ void __launch_bounds__(C::NT, 1) gemm_kernel(const __grid_constant__ Args a) {
    extern __shared__ __align__(16) uint8_t smem[];
    constexpr int FM = C::WM / 16, FN = C::WN / 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bool swiglu = a.mode == M_SWIGLU;

    // このブロックの行列と列
    int bx = blockIdx.x, mi = 0;
    while (mi + 1 < a.nmat && bx >= a.mat[mi].tiles) bx -= a.mat[mi].tiles, ++mi;
    const int bn = swiglu ? C::BN / 2 : C::BN;  // 出力の列の幅
    const int n0 = bx * bn, nout = a.mat[mi].n;
    const int m0 = blockIdx.y * C::BM, mr = min(C::BM, a.m - m0);
    const int kb = blockIdx.z * a.kchunk, ke = min(a.k, kb + a.kchunk);
    const int nk = (ke - kb) / 32;

    if (warp >= C::CW) {
        // 作る側: warp pw は行 [pw*RPW, (pw+1)*RPW)。SwiGLU なら前半は gate（mat 0）、後半は up（mat 1）
        const int pw = warp - C::CW, r0 = pw * C::RPW;
        const Mat& mt = a.mat[swiglu ? (r0 < C::BN / 2 ? 0 : 1) : mi];
        const int row0 = n0 + (swiglu ? r0 % (C::BN / 2) : r0);
        const int nvalid = max(0, min(C::RPW, mt.n - row0));
        const uint8_t* wrow0 = mt.w + int64_t(row0) * mt.rb;
        uint8_t* raw = smem + C::RING + r0 * RAWROW;
        switch (mt.type) {
#define CASE(T) \
    case T: produce<T, C>(wrow0, mt.rb, nvalid, a.x, a.ldx, m0, mr, kb, ke, smem, raw, pw); break;
            GEMM_TYPES(CASE)
#undef CASE
        }
        __syncthreads();
        return;
    }

    // 掛ける側: 16×16 の区画を mma.m8n8k4 1 回（k 4 つぶん）で。quad-pair q は区画の
    // 行 (q>>1)*8 から 8 行・列 (q&1)*8 から 8 列。lane は A の行 / B の列を 1 つ持つ（k の 4 要素 = 8 バイト）
    const int wm = warp / C::WARPS_N, wn = warp % C::WARPS_N;
    const int q = (lane >> 2) & 3, l4 = (lane & 3) + 4 * (lane >> 4);
    const int ar = wm * C::WM + (q >> 1) * 8 + l4, br = wn * C::WN + (q & 1) * 8 + l4;
    const int sa = swz(ar), sb = swz(br);  // 16 足しても変わらない
    float acc[FM][FN][8];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;
    // 段 xs の k 8 つ（16 バイト）の束 c を読む・掛ける
    auto ldf = [&](const uint8_t* xs, int c, uint4* fa, uint4* fb) {
        const uint8_t* ws = xs + C::BM * 64;
#pragma unroll
        for (int i = 0; i < FM; ++i) fa[i] = *reinterpret_cast<const uint4*>(xs + (ar + 16 * i) * 64 + ((c ^ sa) << 4));
#pragma unroll
        for (int j = 0; j < FN; ++j) fb[j] = *reinterpret_cast<const uint4*>(ws + (br + 16 * j) * 64 + ((c ^ sb) << 4));
    };
    auto mmaf = [&](const uint4* fa, const uint4* fb) {
#pragma unroll
        for (int i = 0; i < FM; ++i)
#pragma unroll
            for (int j = 0; j < FN; ++j) mma884(acc[i][j], fa[i].x, fa[i].y, fb[j].x, fb[j].y);
#pragma unroll
        for (int i = 0; i < FM; ++i)
#pragma unroll
            for (int j = 0; j < FN; ++j) mma884(acc[i][j], fa[i].z, fa[i].w, fb[j].z, fb[j].w);
    };
#if GEMM_CONSUMER == 3
    // 読みを 1 束先に出し続ける（段の境目も: 次の段の束 0 は、この段の最後の束を掛ける前に読む）
    uint4 fa0[FM], fb0[FN], fa1[FM], fb1[FN];
    bar_sync(1, C::NT);
    ldf(smem, 0, fa0, fb0);
    for (int t = 0; t < nk; ++t) {
        const int s = t % C::ST;
        const uint8_t* xs = smem + s * C::STAGE;
        ldf(xs, 1, fa1, fb1);
        mmaf(fa0, fb0);
        ldf(xs, 2, fa0, fb0);
        mmaf(fa1, fb1);
        ldf(xs, 3, fa1, fb1);
        mmaf(fa0, fb0);
        if (t + 1 < nk) {
            const int s1 = (t + 1) % C::ST;
            bar_sync(1 + s1, C::NT);
            ldf(smem + s1 * C::STAGE, 0, fa0, fb0);
        }
        mmaf(fa1, fb1);
        if (t + C::ST < nk) bar_arrive(1 + C::ST + s, C::NT);
    }
#else
    for (int t = 0; t < nk; ++t) {
        const int s = t % C::ST;
        bar_sync(1 + s, C::NT);
        const uint8_t* xs = smem + s * C::STAGE;
#if GEMM_CONSUMER == 2
        // 2 束ずつ読んでから掛ける（2 つ目の読みが 1 つ目の掛け算に隠れる）。命令が L0 に収まるよう、それ以上は展開しない
#pragma unroll 1
        for (int c = 0; c < 4; c += 2) {
            uint4 fa0[FM], fb0[FN], fa1[FM], fb1[FN];
            ldf(xs, c, fa0, fb0);
            ldf(xs, c + 1, fa1, fb1);
            mmaf(fa0, fb0);
            mmaf(fa1, fb1);
        }
#else
#pragma unroll 1
        for (int c = 0; c < 4; ++c) {
            uint4 fa0[FM], fb0[FN];
            ldf(xs, c, fa0, fb0);
            mmaf(fa0, fb0);
        }
#endif
        if (t + C::ST < nk) bar_arrive(1 + C::ST + s, C::NT);
    }
#endif
    __syncthreads();

    // SwiGLU: gate 側の warp（wn が前半）が acc をレジスタの並びのまま置き、相方の up 側（同じ並び）が silu(g) * u にする
    int ocol = n0 + wn * C::WN;  // この warp の出力の列の頭
    if (swiglu) {
        float* xch = reinterpret_cast<float*>(smem);
        constexpr int HN = C::WARPS_N / 2;
        const bool gate = wn < HN;
        float* g = xch + (wm * HN + wn % HN) * (FM * FN * 8 * 32);
        if (gate) {
#pragma unroll
            for (int i = 0; i < FM; ++i)
#pragma unroll
                for (int j = 0; j < FN; ++j)
#pragma unroll
                    for (int e = 0; e < 8; ++e) g[((i * FN + j) * 8 + e) * 32 + lane] = acc[i][j][e];
        }
        bar_sync(15, C::CT);
        if (gate) return;
#pragma unroll
        for (int i = 0; i < FM; ++i)
#pragma unroll
            for (int j = 0; j < FN; ++j)
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float gv = g[((i * FN + j) * 8 + e) * 32 + lane];
                    acc[i][j][e] = gv / (1.f + __expf(-gv)) * acc[i][j][e];
                }
        ocol = n0 + (wn - HN) * C::WN;
    }

    // 書き出し: 16×16 ずつ warp の置き場に出し、行と列の端を見ながら y へ。
    // 要素 e の区画の中の位置（mma.m8n8k4 の f32 の並び）: 行 (q>>1)*8 + (lane&1) + (e&2) + 4*(lane>>4)、
    // 列 (q&1)*8 + (e&4) + (lane&2) + (e&1)
    const bool split = gridDim.z > 1;
    const int mode = split ? M_F32 : a.mode == M_SWIGLU ? M_F16 : a.mode;
    const int ldy = split ? nout : a.mat[mi].ldy;
    void* y = split ? static_cast<void*>(a.ws + int64_t(blockIdx.z) * a.m * nout) : a.mat[mi].y;
    float* sc = reinterpret_cast<float*>(smem + C::BM * C::BN * 2) + warp * 16 * 20;
    const int er = (q >> 1) * 8 + (lane & 1) + 4 * (lane >> 4), ec = (q & 1) * 8 + (lane & 2);
    const int r = lane >> 1, c8 = (lane & 1) * 8;
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) {
#pragma unroll
            for (int e = 0; e < 8; e += 2)
                *reinterpret_cast<float2*>(sc + (er + (e & 2)) * 20 + ec + (e & 4)) = make_float2(acc[i][j][e], acc[i][j][e + 1]);
            __syncwarp();
            const int tr = wm * C::WM + i * 16 + r, gc = ocol + j * 16 + c8;
            if (tr < mr && gc < nout) {
                const float* s = sc + r * 20 + c8;
                const int64_t o = int64_t(m0 + tr) * ldy + gc;
                const bool full = gc + 8 <= nout;
                if (mode == M_F16) {
                    __half* yp = static_cast<__half*>(y) + o;
                    if (full && (reinterpret_cast<uintptr_t>(yp) & 15) == 0) {
                        *reinterpret_cast<uint4*>(yp) =
                            make_uint4(pack2(s[0], s[1]), pack2(s[2], s[3]), pack2(s[4], s[5]), pack2(s[6], s[7]));
                    } else {
                        for (int e = 0; e < 8 && gc + e < nout; ++e) yp[e] = __float2half_rn(s[e]);
                    }
                } else {
                    float* yp = static_cast<float*>(y) + o;
                    if (full && (reinterpret_cast<uintptr_t>(yp) & 15) == 0) {
                        float4 v0 = make_float4(s[0], s[1], s[2], s[3]), v1 = make_float4(s[4], s[5], s[6], s[7]);
                        if (mode == M_ADD) {
                            const float4 p0 = reinterpret_cast<float4*>(yp)[0], p1 = reinterpret_cast<float4*>(yp)[1];
                            v0.x += p0.x, v0.y += p0.y, v0.z += p0.z, v0.w += p0.w;
                            v1.x += p1.x, v1.y += p1.y, v1.z += p1.z, v1.w += p1.w;
                        }
                        reinterpret_cast<float4*>(yp)[0] = v0;
                        reinterpret_cast<float4*>(yp)[1] = v1;
                    } else {
                        for (int e = 0; e < 8 && gc + e < nout; ++e) yp[e] = mode == M_ADD ? yp[e] + s[e] : s[e];
                    }
                }
            }
            __syncwarp();
        }
}


// split-K の部分和を、分けの順に足して y へ（毎回同じ順なので同じ答え）
__global__ void splitk_sum(const float* __restrict__ ws, int splits, int m, int n, void* __restrict__ y, int ldy, int mode) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t mn = int64_t(m) * n;
    if (i >= mn) return;
    float a = ws[i];
    for (int z = 1; z < splits; ++z) a += ws[z * mn + i];
    const int64_t o = (i / n) * ldy + i % n;
    if (mode == M_F16) static_cast<__half*>(y)[o] = __float2half_rn(a);
    else if (mode == M_ADD) static_cast<float*>(y)[o] += a;
    else static_cast<float*>(y)[o] = a;
}

// タイルの組（cfg 番号）。Tile<BM, BN, BK, WM, WN, 作る側の warp, 段>。掛ける側はどれも 4 warp。
// 1 は欠番: 掛ける側 8 warp × 64×32 は 384 スレッドで 168 レジスタに収まらず遅かった。256×64（戻す W が半分）も遅かった
using C0 = Tile<128, 128, 32, 64, 64, 4, 4>;  // 大きな M
using C2 = Tile<64, 128, 32, 32, 64, 4, 4>;   // 中くらい
using C3 = Tile<32, 128, 32, 32, 32, 8, 4>;   // 小さな M
using C4 = Tile<16, 128, 32, 16, 32, 8, 4>;   // M ≦ 16
using C5 = Tile<64, 128, 32, 32, 64, 8, 4>;   // 中くらいで作る側を 8 warp に

// 今のスレッドの GPU の番号（表の添え字に使う）
constexpr int MAXDEV = 64;
int cur_dev() {
    int dev = 0;
    cudaGetDevice(&dev);
    return dev >= 0 && dev < MAXDEV ? dev : 0;
}

// SM の数（GPU ごとに 1 度だけ聞く）
int sm_count() {
    static std::atomic<int> n[MAXDEV] = {};
    const int dev = cur_dev();
    int v = n[dev].load(std::memory_order_relaxed);
    if (v <= 0) {
        if (cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || v <= 0) v = 80;
        n[dev].store(v, std::memory_order_relaxed);
    }
    return v;
}

template <class C>
int launch(Args& a, int splits, cudaStream_t s) {
    auto f = gemm_kernel<C>;
    // 48KB を超える共有メモリの許しは GPU ごと（2 枚で動かすとき）に 1 度だけ取る
    static std::atomic<bool> attr[MAXDEV] = {};
    const int dev = cur_dev();
    if (!attr[dev].load(std::memory_order_acquire)) {
        cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
        attr[dev].store(true, std::memory_order_release);
    }
    const int bn = a.mode == M_SWIGLU ? C::BN / 2 : C::BN;
    int tiles = 0;
    for (int i = 0; i < a.nmat; ++i) a.mat[i].tiles = (a.mat[i].n + bn - 1) / bn, tiles += a.mat[i].tiles;
    if (a.mode == M_SWIGLU) tiles = a.mat[0].tiles;
    // K の分け: KALIGN の倍数で切る
    const int units = a.k / KALIGN;
    splits = std::max(1, std::min(splits, units));
    a.kchunk = (units + splits - 1) / splits * KALIGN;
    splits = (a.k + a.kchunk - 1) / a.kchunk;
    if (tiles == 0) return 0;
    const dim3 grid(tiles, (a.m + C::BM - 1) / C::BM, splits);
    f<<<grid, C::NT, C::SMEM, s>>>(a);
    if (splits > 1) {
        const int64_t mn = int64_t(a.m) * a.mat[0].n;
        splitk_sum<<<unsigned((mn + 255) / 256), 256, 0, s>>>(a.ws, splits, a.m, a.mat[0].n, a.mat[0].y, a.mat[0].ldy, a.mode);
    }
    return int(cudaGetLastError());
}

int run_cfg(int cfg, Args& a, int splits, cudaStream_t s) {
    switch (cfg) {
        case 0: return launch<C0>(a, splits, s);
        case 2: return launch<C2>(a, splits, s);
        case 3: return launch<C3>(a, splits, s);
        case 4: return launch<C4>(a, splits, s);
        case 5: return launch<C5>(a, splits, s);
    }
    return -4;
}

int cfg_bm(int cfg) {
    switch (cfg) {
        case 0: return 128;
        case 2: case 5: return 64;
        case 3: return 32;
        case 4: return 16;
    }
    return 128;
}

// M からタイルの組を選ぶ（V100 で測って決めた）
int auto_cfg(int m) {
    if (m <= 16) return 4;
    if (m <= 32) return 3;
    if (m <= 64) return 5;
    return 0;
}

// 自動の split-K: 波（SM の数ずつのブロック。1 SM に 1 ブロック）の端数で空く SM が減る分け数を選ぶ。
// 時間の見積もり = 波の数 / 分け数 × 1 ブロックの時間 + 部分和の書き・読み（分けると増える）。
// 1 ブロックは 1 SM で約 0.7 TFLOPS、部分和は 600 GB/s で見る。1 つの分けに 4 単位（1024 要素）以上残す
int auto_splits(int cfg, int k, int n, int m) {
    const int sms = sm_count(), bm = cfg_bm(cfg);
    const int blocks = ((n + 127) / 128) * ((m + bm - 1) / bm);
    const int most = std::min(8, k / (4 * KALIGN));
    const double tb = 2.0 * k * 128 * bm / 0.7e12;
    int best = 1;
    double bc = double((blocks + sms - 1) / sms) * tb;
    for (int s = 2; s <= most; ++s) {
        const double c = double((blocks * s + sms - 1) / sms) / s * tb + double(m) * n * 4 * (2 * s + 1) / 600e9;
        if (c < bc) best = s, bc = c;
    }
    return best;
}

int check_mat(int qtype, const void* w, int k) {
    if (!known_type(qtype)) return -1;
    if (k % KALIGN) return -2;
    if (reinterpret_cast<uintptr_t>(w) & 15) return -5;
    return 0;
}

}  // namespace

extern "C" {

// 自動で選ぶ split-K に要る作業場のバイト数（0 なら要らない）。cfg を決め打ちしてもよいよう、どのタイルの組でも足りる大きさ
int64_t infer_gemm_ws_bytes(int k, int n, int m) {
    if (k % KALIGN || m <= 0 || n <= 0) return 0;
    int sp = 1;
    for (int c : {0, 2, 3, 4, 5}) sp = std::max(sp, auto_splits(c, k, n, m));
    return sp > 1 ? int64_t(sp) * m * n * 4 : 0;
}

// Y[m, n] = X[m, k] · W^T。X は fp16（行の間隔 k）。mode: 0 = Y を f32、1 = f16、2 = f32 の Y に足す（残差）。
// cfg: タイルの組（-1 で M から自動）。splits: K の分け数（0 で自動、1 で分けない）。
// ws / ws_bytes: split-K の作業場（足りなければ分けない）。
// 戻り値: 0 = 成功、-1 = 知らない型、-2 = k が 256 の倍数でない、-3 = mode が変、-4 = 知らない cfg、
// -5 = w が 16 バイトに揃っていない、それ以外は cudaError
int infer_gemm_ex(int qtype, const void* w, int k, int n, int m, const void* x, void* y, int mode, int cfg, int splits,
                  void* ws, int64_t ws_bytes, cudaStream_t s) {
    if (int e = check_mat(qtype, w, k)) return e;
    if (mode < M_F32 || mode > M_ADD) return -3;
    if (m <= 0 || n <= 0) return 0;
    if (cfg < 0) cfg = auto_cfg(m);
    if (splits <= 0) splits = auto_splits(cfg, k, n, m);
    if (ws == nullptr || ws_bytes < int64_t(splits) * m * n * 4) splits = 1;
    Args a{};
    a.mat[0] = Mat{static_cast<const uint8_t*>(w), qtype_row_bytes(qtype, k), y, qtype, n, n, 0};
    a.nmat = 1, a.mode = mode, a.k = k, a.m = m, a.ldx = k;
    a.x = static_cast<const __half*>(x);
    a.ws = static_cast<float*>(ws);
    return run_cfg(cfg, a, splits, s);
}

int infer_gemm(int qtype, const void* w, int k, int n, int m, const void* x, void* y, int mode, void* ws, int64_t ws_bytes,
               cudaStream_t s) {
    return infer_gemm_ex(qtype, w, k, n, m, x, y, mode, -1, 0, ws, ws_bytes, s);
}

// gate と up をまとめて: Y[m, n] = silu(X·Wg^T) * (X·Wu^T)（fp16）。型は別々でよい
int infer_gemm_swiglu(int tg, const void* wg, int tu, const void* wu, int k, int n, int m, const void* x, void* y, int cfg,
                      cudaStream_t s) {
    if (int e = check_mat(tg, wg, k)) return e;
    if (int e = check_mat(tu, wu, k)) return e;
    if (m <= 0 || n <= 0) return 0;
    if (cfg < 0) cfg = auto_cfg(m);
    Args a{};
    a.mat[0] = Mat{static_cast<const uint8_t*>(wg), qtype_row_bytes(tg, k), y, tg, n, n, 0};
    a.mat[1] = Mat{static_cast<const uint8_t*>(wu), qtype_row_bytes(tu, k), y, tu, n, n, 0};
    a.nmat = 2, a.mode = M_SWIGLU, a.k = k, a.m = m, a.ldx = k;
    a.x = static_cast<const __half*>(x);
    return run_cfg(cfg, a, 1, s);
}

// 同じ X に何本かの行列（qkv と gate、q・k・v など）を 1 回の起動で。行列 i は Y_i[m, ns[i]]（mode は全部同じ、split-K なし）
int infer_gemm_multi(int nmat, const int* qtypes, const void* const* ws, const int* ns, void* const* ys, int k, int m,
                     const void* x, int mode, int cfg, cudaStream_t s) {
    if (nmat < 1 || nmat > MAXMAT) return -3;
    if (mode < M_F32 || mode > M_ADD) return -3;
    if (m <= 0) return 0;
    if (cfg < 0) cfg = auto_cfg(m);
    Args a{};
    for (int i = 0; i < nmat; ++i) {
        if (int e = check_mat(qtypes[i], ws[i], k)) return e;
        a.mat[i] = Mat{static_cast<const uint8_t*>(ws[i]), qtype_row_bytes(qtypes[i], k), ys[i], qtypes[i], ns[i], ns[i], 0};
    }
    a.nmat = nmat, a.mode = mode, a.k = k, a.m = m, a.ldx = k;
    a.x = static_cast<const __half*>(x);
    return run_cfg(cfg, a, 1, s);
}

}  // extern "C"
