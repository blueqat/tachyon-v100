// KV の頁を写す・詰めるカーネル（kv-tier。src/kv.rs から呼ぶ）。
//  - kv_copy:  多数の (src, dst, bytes) を 1 回の起動でまとめて写す。src/dst は GPU でも固定した RAM（zero-copy）でもよい
//  - kv_quant / kv_dequant: fp16 の層頁 [K: H×T×D][V: H×T×D] ⇄ int8 の層頁 [K i8][V i8][係数 f16: 2×H×T]（行 = 1 トークン 1 ヘッド）
//  - kv_fill / kv_hash: 確かめ用（決まった模様で埋める・頁ごとの 64 ビットの指紋）
// 記述子は起動の引数で渡す（4KB まで）→ 呼んだ後すぐ手元の配列を捨ててよい。
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

struct KvCopyDesc {
    const char* src;
    char* dst;
    unsigned long long bytes;
};
constexpr int COPY_MAX = 160;  // 160×24B = 3840B（引数は 4KB まで）
struct KvCopyBatch {
    KvCopyDesc d[COPY_MAX];
    int n;
};
static_assert(sizeof(KvCopyBatch) + 16 <= 4096, "起動の引数は 4KB まで");
constexpr int CHUNK = 64 * 1024;  // 1 ブロックが受け持つ幅
constexpr int UNROLL = 8;         // 1 スレッドが先に読んでおく 16B の数（PCIe の往復を重ねる）

// 仕事 = (記述子, CHUNK の番号)。ブロックの数を絞って（grid-stride）、SM を計算に残しつつ PCIe を埋める
__global__ void __launch_bounds__(256) kv_copy_k(const KvCopyBatch b, long long nch) {
  for (long long item = blockIdx.x; item < b.n * nch; item += gridDim.x) {
    const int i = item / nch;
    const KvCopyDesc d = b.d[i];
    const unsigned long long beg = (unsigned long long)(item % nch) * CHUNK;
    if (beg >= d.bytes) continue;
    const unsigned long long end = min(beg + (unsigned long long)CHUNK, d.bytes);
    if ((((uintptr_t)d.src | (uintptr_t)d.dst | beg | end) & 15) == 0) {
        const uint4* s = reinterpret_cast<const uint4*>(d.src + beg);
        uint4* t = reinterpret_cast<uint4*>(d.dst + beg);
        const int n = int((end - beg) >> 4);
        for (int j0 = threadIdx.x; j0 < n; j0 += 256 * UNROLL) {
            uint4 r[UNROLL];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const int j = j0 + u * 256;
                if (j < n) r[u] = s[j];
            }
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const int j = j0 + u * 256;
                if (j < n) t[j] = r[u];
            }
        }
    } else {
        for (unsigned long long j = beg + threadIdx.x; j < end; j += 256) d.dst[j] = d.src[j];
    }
  }
}

// fp16 ⇄ int8 の層頁。1 warp が 1 行（D 要素）。D は 256 の倍数。
struct KvConvDesc {
    const char* src;
    char* dst;
};
constexpr int CONV_MAX = 250;  // 250×16B = 4000B
struct KvConvBatch {
    KvConvDesc d[CONV_MAX];
    int n;
};
static_assert(sizeof(KvConvBatch) + 16 <= 4096, "起動の引数は 4KB まで");

__global__ void __launch_bounds__(256) kv_quant_k(const KvConvBatch b, int rows, int D) {
  const int ng = (rows + 7) / 8;
  for (int item = blockIdx.x; item < b.n * ng; item += gridDim.x) {
    const int i = item / ng;
    const int row = (item % ng) * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= rows) continue;
    const __half* x = reinterpret_cast<const __half*>(b.d[i].src) + (int64_t)row * D;
    int8_t* q = reinterpret_cast<int8_t*>(b.d[i].dst) + (int64_t)row * D;
    __half* sc = reinterpret_cast<__half*>(b.d[i].dst + (int64_t)rows * D) + row;
    float amax = 0.f;
    for (int c = lane * 8; c < D; c += 256) {
        const uint4 v = *reinterpret_cast<const uint4*>(x + c);
        const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const float2 f = __half22float2(h[u]);
            amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    const __half s = __float2half(amax / 127.f);
    const float inv = amax > 0.f ? 1.f / __half2float(s) : 0.f;  // 係数は f16 に丸めた後の値で割る（戻しと揃える）
    for (int c = lane * 8; c < D; c += 256) {
        const uint4 v = *reinterpret_cast<const uint4*>(x + c);
        const __half2* h = reinterpret_cast<const __half2*>(&v);
        uint2 o;
        int8_t* ob = reinterpret_cast<int8_t*>(&o);
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const float2 f = __half22float2(h[u]);
            ob[2 * u] = (int8_t)max(-127, min(127, __float2int_rn(f.x * inv)));
            ob[2 * u + 1] = (int8_t)max(-127, min(127, __float2int_rn(f.y * inv)));
        }
        *reinterpret_cast<uint2*>(q + c) = o;
    }
    if (lane == 0) *sc = s;
  }
}

__global__ void __launch_bounds__(256) kv_dequant_k(const KvConvBatch b, int rows, int D) {
  const int ng = (rows + 7) / 8;
  for (int item = blockIdx.x; item < b.n * ng; item += gridDim.x) {
    const int i = item / ng;
    const int row = (item % ng) * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= rows) continue;
    const int8_t* q = reinterpret_cast<const int8_t*>(b.d[i].src) + (int64_t)row * D;
    const float s = __half2float(reinterpret_cast<const __half*>(b.d[i].src + (int64_t)rows * D)[row]);
    __half* y = reinterpret_cast<__half*>(b.d[i].dst) + (int64_t)row * D;
    for (int c = lane * 8; c < D; c += 256) {
        const uint2 v = *reinterpret_cast<const uint2*>(q + c);
        const int8_t* qb = reinterpret_cast<const int8_t*>(&v);
        uint4 o;
        __half2* h = reinterpret_cast<__half2*>(&o);
#pragma unroll
        for (int u = 0; u < 4; ++u) h[u] = __floats2half2_rn(qb[2 * u] * s, qb[2 * u + 1] * s);
        *reinterpret_cast<uint4*>(y + c) = o;
    }
  }
}

// 確かめ用
__device__ __forceinline__ uint64_t mix64(uint64_t z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
constexpr int PTR_MAX = 480;
struct KvPtrBatch {
    char* p[PTR_MAX];
    int n;
};
static_assert(sizeof(KvPtrBatch) + 32 <= 4096, "起動の引数は 4KB まで");
constexpr long long GRID_MAX = 1 << 16;  // grid-stride なので、これ以上は要らない

// mode 0: でたらめなビット。mode 1: fp16 の値（-4〜4 くらい。int8 に詰める試し用）
// 頁 i（呼んだ配列の中の番号）の中身は (seed, i) だけで決まる
__global__ void __launch_bounds__(256) kv_fill_k(const KvPtrBatch b, int base, unsigned long long bytes, unsigned long long seed, int mode) {
    const int i = blockIdx.y;
    if (i >= b.n) return;
    uint32_t* w = reinterpret_cast<uint32_t*>(b.p[i]);
    const unsigned long long n = bytes / 4;
    const uint64_t s = mix64(seed + (uint64_t)(base + i) * 0x632be59bd9b4e019ull);
    for (unsigned long long j = (unsigned long long)blockIdx.x * 256 + threadIdx.x; j < n; j += (unsigned long long)gridDim.x * 256) {
        const uint64_t r = mix64(s ^ j);
        if (mode == 0) {
            w[j] = (uint32_t)r;
        } else {
            const float a = ((int)(r & 0xffff) - 32768) / 8192.f, c = ((int)((r >> 16) & 0xffff) - 32768) / 8192.f;
            const __half2 h = __floats2half2_rn(a, c);
            w[j] = *reinterpret_cast<const uint32_t*>(&h);
        }
    }
}

// 頁ごとの指紋: Σ mix64(語 ⊕ 位置) を 64 ビットで足す（順番によらず、位置には敏感）
__global__ void __launch_bounds__(256) kv_hash_k(const KvPtrBatch b, unsigned long long bytes, unsigned long long* out, int out0) {
    const int i = blockIdx.y;
    if (i >= b.n) return;
    const uint32_t* w = reinterpret_cast<const uint32_t*>(b.p[i]);
    const unsigned long long n = bytes / 4;
    uint64_t acc = 0;
    for (unsigned long long j = (unsigned long long)blockIdx.x * 256 + threadIdx.x; j < n; j += (unsigned long long)gridDim.x * 256)
        acc += mix64(((uint64_t)w[j] << 32) ^ j);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(out + out0 + i, (unsigned long long)acc);
}

// 試し用: 流れを cycles だけ止める（転送が計算の流れを待つかを確かめる）
__global__ void kv_spin_k(long long cycles) {
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) {
    }
}

extern "C" {

int kv_spin(long long cycles, cudaStream_t s) {
    kv_spin_k<<<1, 1, 0, s>>>(cycles);
    return (int)cudaGetLastError();
}

// d は手元（CPU）の配列。COPY_MAX ずつ起動する。blocks = 1 回の起動のブロック数の上限（0 = 仕事の数だけ）
int kv_copy(const KvCopyDesc* d, int n, int blocks, cudaStream_t s) {
    for (int i0 = 0; i0 < n; i0 += COPY_MAX) {
        KvCopyBatch b;
        b.n = min(COPY_MAX, n - i0);
        unsigned long long mx = 0;
        for (int j = 0; j < b.n; ++j) {
            b.d[j] = d[i0 + j];
            mx = mx > b.d[j].bytes ? mx : b.d[j].bytes;
        }
        if (mx == 0) continue;
        const long long nch = (long long)((mx + CHUNK - 1) / CHUNK);
        const long long work = nch * b.n;
        const long long g = blocks > 0 ? min((long long)blocks, work) : min(work, GRID_MAX);
        kv_copy_k<<<(unsigned)g, 256, 0, s>>>(b, nch);
    }
    return (int)cudaGetLastError();
}

static int conv(bool q, const KvConvDesc* d, int n, int rows, int D, int blocks, cudaStream_t s) {
    if (D % 256 != 0 || rows <= 0) return -1;
    for (int j = 0; j < n; ++j)  // uint4 で読み書きする
        if ((((uintptr_t)d[j].src | (uintptr_t)d[j].dst) & 15) != 0) return -2;
    for (int i0 = 0; i0 < n; i0 += CONV_MAX) {
        KvConvBatch b;
        b.n = min(CONV_MAX, n - i0);
        for (int j = 0; j < b.n; ++j) b.d[j] = d[i0 + j];
        const long long work = (long long)((rows + 7) / 8) * b.n;
        const unsigned g = (unsigned)(blocks > 0 ? min((long long)blocks, work) : min(work, GRID_MAX));
        if (q)
            kv_quant_k<<<g, 256, 0, s>>>(b, rows, D);
        else
            kv_dequant_k<<<g, 256, 0, s>>>(b, rows, D);
    }
    return (int)cudaGetLastError();
}
// rows = 2×H×T（K と V の行を全部）。src/dst は層頁の頭
int kv_quant(const KvConvDesc* d, int n, int rows, int D, int blocks, cudaStream_t s) { return conv(true, d, n, rows, D, blocks, s); }
int kv_dequant(const KvConvDesc* d, int n, int rows, int D, int blocks, cudaStream_t s) { return conv(false, d, n, rows, D, blocks, s); }

int kv_fill(char* const* p, int n, unsigned long long bytes, unsigned long long seed, int mode, cudaStream_t s) {
    for (int i0 = 0; i0 < n; i0 += PTR_MAX) {
        KvPtrBatch b;
        b.n = min(PTR_MAX, n - i0);
        for (int j = 0; j < b.n; ++j) b.p[j] = p[i0 + j];
        const unsigned gx = (unsigned)min(64ull, (bytes / 4 + 255) / 256 + 1);
        kv_fill_k<<<dim3(gx, b.n), 256, 0, s>>>(b, i0, bytes, seed, mode);
    }
    return (int)cudaGetLastError();
}

// out は GPU 上の u64 × n（0 で埋めてから足す）
int kv_hash(char* const* p, int n, unsigned long long bytes, unsigned long long* out, cudaStream_t s) {
    cudaMemsetAsync(out, 0, sizeof(unsigned long long) * n, s);
    for (int i0 = 0; i0 < n; i0 += PTR_MAX) {
        KvPtrBatch b;
        b.n = min(PTR_MAX, n - i0);
        for (int j = 0; j < b.n; ++j) b.p[j] = p[i0 + j];
        const unsigned gx = (unsigned)min(64ull, (bytes / 4 + 255) / 256 + 1);
        kv_hash_k<<<dim3(gx, b.n), 256, 0, s>>>(b, bytes, out, i0);
    }
    return (int)cudaGetLastError();
}
}
