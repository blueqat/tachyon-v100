// 抜き取りの候補を GPU で（口の CPU の抜き取りが 1 語ごとに logits 全部（248K 語・1MB）を写して並べていたのを、上位 K 語だけにする）。
// 行ごとに: [0, n_valid) の上位 K 語（値の大きい順・同じ値は番号の小さい順＝sampler.rs の Key と同じ。NaN は -inf）、
// 最大 m と z = Σ exp((x − m)/t)（f64。top_k 無しの top_p の分母）。罰・温度・top_k・top_p・min_p・引くのは手元（sampler::prepare_cands）。
//   段 1: 行を CH 語ずつの塊に分け、塊ごとに共有メモリで並べて上位 K と (塊の最大, 塊の z) を作業場へ
//   段 2: 行ごとに塊の上位 K を全部並べて上位 K、z を合わせる
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace {

constexpr int CH = 8192;   // 段 1 の塊の語数（並べる数）
constexpr int NS = 8192;   // 段 2 で並べる数の上限（塊の数 × K）
constexpr int NT = 1024;

struct KV {
    float v;
    uint32_t i;
};

// 値の順を保つ符号なしの鍵（大きい値ほど大きい。-inf は小さい。-0 は +0 より小さい＝Rust の total_cmp と同じ）
__device__ __forceinline__ uint32_t okey(float v) {
    const uint32_t f = __float_as_uint(v);
    return (f & 0x80000000u) ? ~f : (f | 0x80000000u);
}

// a が b より前（鍵の大きい方・同じなら小さい番号）＝sampler.rs の Key（total_cmp・番号）と同じ順
__device__ __forceinline__ bool before(const KV& a, const KV& b) {
    const uint32_t ka = okey(a.v), kb = okey(b.v);
    return ka > kb || (ka == kb && a.i < b.i);
}

// 共有メモリの n（2 のべき）個を大きい順に（bitonic）
__device__ void bsort(KV* s, int n) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const bool desc = (i & k) == 0;
                    KV a = s[i], b = s[ixj];
                    // desc なら前に大きい方
                    if (desc ? before(b, a) : before(a, b)) {
                        s[i] = b;
                        s[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
}

constexpr int TB = 1024;  // 上位 K を並べる置き場（K と同じ値の語の数がこれを超えたら全部を並べる道へ）

// 共有メモリの s[0..n)（n は 2 のべき・n ≥ TB）から上位 K（≤ 256）を大きい順に t[0..K) へ。
// 鍵の上の桁から 8 ビットずつ K 番目の鍵 T を決め（基数で選ぶ）、T 以上の語だけを t に集めて並べる。
// T と同じ値の語が多すぎて t に入らなければ s を全部並べる（遅いが同じ答え）
__device__ void topk_block(KV* s, int n, int K, KV* t) {
    __shared__ uint32_t hist[256];
    __shared__ uint32_t sh_prefix, sh_need, sh_cnt;
    if (threadIdx.x == 0) {
        sh_prefix = 0;
        sh_need = K;
    }
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        const uint32_t pre = sh_prefix;
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const uint32_t k = okey(s[i].v);
            if ((k & hi_mask) == (pre & hi_mask)) atomicAdd(&hist[(k >> shift) & 255], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            uint32_t need = sh_need, b = 255;
            for (;; --b) {
                if (hist[b] >= need || b == 0) break;
                need -= hist[b];
            }
            sh_prefix = pre | (b << shift);
            sh_need = need;
        }
        __syncthreads();
    }
    const uint32_t T = sh_prefix;  // K 番目の鍵
    if (threadIdx.x == 0) sh_cnt = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (okey(s[i].v) >= T) {
            const uint32_t j = atomicAdd(&sh_cnt, 1u);
            if (j < TB) t[j] = s[i];
        }
    }
    __syncthreads();
    const uint32_t c = sh_cnt;
    if (c > TB) {
        bsort(s, n);
        for (int i = threadIdx.x; i < K; i += blockDim.x) t[i] = s[i];
        __syncthreads();
        return;
    }
    int m = 1;
    while (m < (int)c) m <<= 1;
    for (int i = c + threadIdx.x; i < m; i += blockDim.x) t[i] = KV{-INFINITY, 0xffffffffu};
    __syncthreads();
    bsort(t, m);
}

__device__ double block_sum(double x, double* red) {
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = x;
    __syncthreads();
    double s = 0;
    if (threadIdx.x == 0) {
        for (int w = 0; w < (int)(blockDim.x >> 5); ++w) s += red[w];
        red[0] = s;
    }
    __syncthreads();
    s = red[0];
    __syncthreads();
    return s;
}

__global__ void __launch_bounds__(NT) stage1(const float* __restrict__ lg, int64_t ld, int n, int K, const float* __restrict__ temps, KV* __restrict__ wk,
                                             float2* __restrict__ wmz_f, double* __restrict__ wz) {
    extern __shared__ KV s[];
    __shared__ double red[32];
    __shared__ float smax[32];
    const int c = blockIdx.x, r = blockIdx.y, nc = gridDim.x;
    const float* x = lg + int64_t(r) * ld;
    const int a = c * CH;
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < CH; i += blockDim.x) {
        const int g = a + i;
        float v = g < n ? x[g] : -INFINITY;
        if (v != v) v = -INFINITY;
        s[i] = KV{v, g < n ? uint32_t(g) : 0xffffffffu};
        mx = fmaxf(mx, v);
    }
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if ((threadIdx.x & 31) == 0) smax[threadIdx.x >> 5] = mx;
    __syncthreads();
    mx = smax[0];
    for (int w = 1; w < (int)(blockDim.x >> 5); ++w) mx = fmaxf(mx, smax[w]);
    float t = temps[r];
    if (!(t > 0.f)) t = 1.f;
    double z = 0;
    if (mx > -INFINITY)
        for (int i = threadIdx.x; i < CH; i += blockDim.x)
            if (s[i].v > -INFINITY) z += exp(double((s[i].v - mx) / t));
    z = block_sum(z, red);
    KV* tt = s + CH;
    topk_block(s, CH, K, tt);
    KV* o = wk + (int64_t(r) * nc + c) * K;
    for (int i = threadIdx.x; i < K; i += blockDim.x) o[i] = tt[i];
    if (threadIdx.x == 0) {
        wmz_f[int64_t(r) * nc + c] = make_float2(mx, t);
        wz[int64_t(r) * nc + c] = z;
    }
}

__global__ void __launch_bounds__(NT) stage2(int nc, int K, const KV* __restrict__ wk, const float2* __restrict__ wmz_f, const double* __restrict__ wz,
                                             uint32_t* __restrict__ ids, float* __restrict__ vals, double* __restrict__ mz) {
    extern __shared__ KV s[];
    __shared__ double red[32];
    const int r = blockIdx.x;
    const int m = nc * K;
    int n2 = TB;
    while (n2 < m) n2 <<= 1;
    for (int i = threadIdx.x; i < n2; i += blockDim.x) s[i] = i < m ? wk[int64_t(r) * m + i] : KV{-INFINITY, 0xffffffffu};
    __syncthreads();
    // 行の最大と z（塊の z を行の最大に合わせて足す。足す順は塊の順）
    float mx = -INFINITY, t = 1.f;
    for (int c = 0; c < nc; ++c) mx = fmaxf(mx, wmz_f[int64_t(r) * nc + c].x);
    t = wmz_f[int64_t(r) * nc].y;
    double z = 0;
    if (threadIdx.x == 0 && mx > -INFINITY)
        for (int c = 0; c < nc; ++c) {
            const float mb = wmz_f[int64_t(r) * nc + c].x;
            if (mb > -INFINITY) z += wz[int64_t(r) * nc + c] * exp(double((mb - mx) / t));
        }
    (void)red;
    KV* tt = s + n2;
    topk_block(s, n2, K, tt);
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        ids[int64_t(r) * K + i] = tt[i].i;
        vals[int64_t(r) * K + i] = tt[i].v;
    }
    if (threadIdx.x == 0) {
        mz[2 * r] = mx;
        mz[2 * r + 1] = z;
    }
}

int n_chunks(int n) { return (n + CH - 1) / CH; }

}  // namespace

extern "C" {

// 作業場の大きさ（rows 行・語彙 n・上位 K）。0 なら扱えない形（K > 256 か、塊の数 × K が 8192 を超える）
int64_t infer_cand_ws_bytes(int rows, int n, int K) {
    const int nc = n_chunks(n);
    if (K < 1 || K > 256 || int64_t(nc) * K > NS || rows < 1) return 0;
    return int64_t(rows) * nc * (int64_t(K) * sizeof(KV) + sizeof(float2) + sizeof(double)) + 256;
}

// lg [rows][ld] の各行の [0, n) の上位 K → ids・vals [rows][K]、mz [rows][2]（最大・z）。temps [rows]（GPU の上。z の温度。0 以下は 1）
int infer_cand_topk(const float* lg, int64_t ld, int rows, int n, int K, const float* temps, void* ws, int64_t ws_bytes, uint32_t* ids, float* vals,
                    double* mz, cudaStream_t s) {
    const int64_t need = infer_cand_ws_bytes(rows, n, K);
    if (need == 0 || ws_bytes < need) return -2;
    const int nc = n_chunks(n);
    uint8_t* p = static_cast<uint8_t*>(ws);
    KV* wk = reinterpret_cast<KV*>(p);
    p += int64_t(rows) * nc * K * sizeof(KV);
    double* wz = reinterpret_cast<double*>((reinterpret_cast<uintptr_t>(p) + 15) & ~uintptr_t(15));
    float2* wmz = reinterpret_cast<float2*>(wz + int64_t(rows) * nc);
    const int sm1 = (CH + TB) * sizeof(KV);
    cudaFuncSetAttribute(stage1, cudaFuncAttributeMaxDynamicSharedMemorySize, sm1);
    stage1<<<dim3(nc, rows), NT, sm1, s>>>(lg, ld, n, K, temps, wk, wmz, wz);
    int n2 = TB;
    while (n2 < nc * K) n2 <<= 1;
    const int sm2 = (n2 + TB) * sizeof(KV);
    cudaFuncSetAttribute(stage2, cudaFuncAttributeMaxDynamicSharedMemorySize, sm2);
    stage2<<<rows, NT, sm2, s>>>(nc, K, wk, wmz, wz, ids, vals, mz);
    return int(cudaGetLastError());
}

}  // extern "C"
