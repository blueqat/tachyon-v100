#!/usr/bin/env python3
"""Qwen3.8-27B（GGUF arch qwen35）の CPU 参照実装（numpy・float32）。

llama.cpp（unslothai/llama.cpp b12209-mix-b84c2bf の src/models/qwen35.cpp・delta-net-base.cpp）と同じ計算を、
GGUF から直接読んだ重み（gguf-py の dequantize）でたどる。CUDA エンジンの答え合わせの手本。
重みは 1 層ずつ下請けの process で並べて戻し（/dev/shm に書く）、使い終わったら捨てる。式は docs/qwen35-math.md。

使い方（V100 の機械で。GPU は使わない）:
  CUDA_VISIBLE_DEVICES= .venv/bin/python tools/ref_qwen35.py \
      --tokens 785,6722,315,9625,374 --out logs/ref.npz
    → 層ごとの残差 h[il]（層の出口 = llama.cpp の l_out-il）・最後の位置のロジットを .npz に
  --detail 0,3     その層の途中の値（llama.cpp の節の名前: attn_norm-0・linear_attn_out-0・Qcur-3 など）も入れる
  --layers 4       先頭 4 層だけ（速い確かめ。ロジットは出さない）
  --kv-f16         K・V を fp16 に丸める（llama.cpp の KV キャッシュと同じ。ref_compare.py は既定でこれ）
  別のスクリプトからは Ref(...).forward() を呼ぶ（tools/ref_compare.py）
"""
import os
# 下請けの process は BLAS を 1 本に（numpy を読む前に決める）
if os.environ.get('REF_Q35_WORKER'):
    for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[_k] = '1'
else:
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')
import argparse
import json
import multiprocessing as mp
import re
import struct
import sys
import time

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType as QT
from gguf.quants import dequantize

DEFAULT_GGUF = os.path.join(os.environ.get("TACHYON_WEIGHTS", "weights"), "q38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf")
F32 = np.float32


# ---------------- GGUF（目次は自前で読む。gguf-py の reader は語彙で 30 秒かかるため） ----------------

def _gguf_index(path, want_meta):
    """目次を読む。返り値 (meta, tensors{name: (offset, type, ne, nbytes)})。中身は読まない"""
    f = open(path, 'rb', buffering=1 << 20)
    rd = lambda n: f.read(n)
    u32 = lambda: struct.unpack('<I', rd(4))[0]
    u64 = lambda: struct.unpack('<Q', rd(8))[0]
    sstr = lambda: rd(u64()).decode('utf-8', 'replace')
    scal = {0: ('<B', 1), 1: ('<b', 1), 2: ('<H', 2), 3: ('<h', 2), 4: ('<I', 4), 5: ('<i', 4),
            6: ('<f', 4), 7: ('<?', 1), 10: ('<Q', 8), 11: ('<q', 8), 12: ('<d', 8)}
    npt = {0: np.uint8, 1: np.int8, 2: np.uint16, 3: np.int16, 4: np.uint32, 5: np.int32,
           6: np.float32, 7: np.bool_, 10: np.uint64, 11: np.int64, 12: np.float64}

    def val(t):
        if t in scal:
            fmt, n = scal[t]
            return struct.unpack(fmt, rd(n))[0]
        if t == 8:
            return sstr()
        if t == 9:
            et, n = u32(), u64()
            if et in npt:
                return np.frombuffer(rd(n * np.dtype(npt[et]).itemsize), npt[et]).tolist()
            return [val(et) for _ in range(n)]
        raise ValueError(f'gguf: 知らない型 {t}')

    assert rd(4) == b'GGUF', path
    ver = u32()
    assert ver >= 2, ver
    n_t, n_kv = u64(), u64()
    meta = {}
    for _ in range(n_kv):
        k = sstr()
        t = u32()
        v = val(t)
        if want_meta and not k.startswith('tokenizer.ggml.'):
            meta[k] = v
        elif k in ('tokenizer.ggml.eos_token_id', 'tokenizer.ggml.bos_token_id'):
            meta[k] = v
    align = meta.get('general.alignment', 32)
    ts = []
    for _ in range(n_t):
        name = sstr()
        nd = u32()
        ne = tuple(u64() for _ in range(nd))
        ty = u32()
        off = u64()
        ts.append((name, ne, ty, off))
    pos = f.tell()
    data0 = (pos + align - 1) // align * align
    f.close()
    tens = {}
    for name, ne, ty, off in ts:
        bs, tsz = GGML_QUANT_SIZES[QT(ty)]
        nel = int(np.prod(ne))
        tens[name] = (data0 + off, ty, ne, nel // bs * tsz)
    return meta, tens


class Gguf:
    """分かれた束（-0000i-of-0000n）もまとめて扱う。mmap せず pread で要る所だけ読む（RSS を増やさない）"""

    def __init__(self, path=None, index=None):
        if index is None:
            m = re.match(r'(.*)-(\d{5})-of-(\d{5})\.gguf$', path)
            paths = [path] if not m else [f'{m.group(1)}-{i:05d}-of-{m.group(3)}.gguf'
                                           for i in range(1, int(m.group(3)) + 1)]
            self.meta, self.t = {}, {}
            for i, p in enumerate(paths):
                meta, tens = _gguf_index(p, i == 0)
                if i == 0:
                    self.meta = meta
                for k, v in tens.items():
                    self.t[k] = (p,) + v
        else:
            self.meta, self.t = index
        self.fds = {}

    def index(self):
        return (self.meta, self.t)

    def _fd(self, p):
        if p not in self.fds:
            self.fds[p] = os.open(p, os.O_RDONLY)
        return self.fds[p]

    def _pread(self, p, off, n):
        buf = np.empty(n, np.uint8)
        mv = memoryview(buf)
        got = 0
        fd = self._fd(p)
        while got < n:
            k = os.preadv(fd, [mv[got:]], off + got)
            if k <= 0:
                raise IOError(f'{p}: 読めない {off + got}')
            got += k
        return buf

    def shape(self, name):
        return self.t[name][3]

    def n_rows(self, name):
        ne = self.shape(name)
        return int(np.prod(ne[1:])) if len(ne) > 1 else 1

    def _rowinfo(self, name):
        p, off, ty, ne, nb = self.t[name]
        bs, tsz = GGML_QUANT_SIZES[QT(ty)]
        return p, off, QT(ty), ne[0] // bs * tsz, ne[0]

    @staticmethod
    def _deq(raw, q, ncol):
        # raw (rows, row_bytes) uint8 → (rows, ncol) float32
        if q == QT.F32:
            return raw.view(np.float32).reshape(raw.shape[0], ncol).copy()
        if q == QT.F16:
            return raw.view(np.float16).reshape(raw.shape[0], ncol).astype(F32)
        if q == QT.BF16:
            return (raw.view(np.uint16).astype(np.uint32) << 16).view(F32).reshape(raw.shape[0], ncol)
        return dequantize(raw, q).astype(F32, copy=False).reshape(raw.shape[0], ncol)

    def rows(self, name, r0=0, n=None):
        """ne0 を 1 行として r0 から n 行。numpy の形 (n, ne0)"""
        p, off, q, rb, ncol = self._rowinfo(name)
        if n is None:
            n = self.n_rows(name) - r0
        raw = self._pread(p, off + r0 * rb, n * rb).reshape(n, rb)
        return self._deq(raw, q, ncol)

    def gather(self, name, idx):
        """行を飛び飛びに引く（token_embd）。idx の形 + (ne0,)"""
        p, off, q, rb, ncol = self._rowinfo(name)
        idx = np.asarray(idx, np.int64)
        u, inv = np.unique(idx.ravel(), return_inverse=True)
        fd = self._fd(p)
        raw = np.empty((len(u), rb), np.uint8)
        for i, r in enumerate(u):
            raw[i] = np.frombuffer(os.pread(fd, rb, off + int(r) * rb), np.uint8)
        return self._deq(raw, q, ncol)[inv].reshape(idx.shape + (ncol,))

    def mat(self, name):
        """そのまま全部。numpy の形は ne を逆にした物（[ne0, ne1] → (ne1, ne0)）"""
        return self.rows(name).reshape(tuple(reversed(self.shape(name))))


# ---------------- 形の決まり（GGUF の qwen35.* から） ----------------

class HP:
    def __init__(self, meta):
        arch = meta['general.architecture']
        g = lambda k, d=None: meta.get(f'{arch}.{k}', d)
        self.arch = arch
        self.n_layer_all = g('block_count')                     # 65（最後の 1 つは MTP）
        self.n_nextn = g('nextn_predict_layers', 0)
        self.n_layer = self.n_layer_all - self.n_nextn          # 64
        self.n_embd = g('embedding_length')                     # 5120
        # 3.6（qwen35moe）: FFN は専門家 n_exp のうち n_used 個（幅 n_ff_exp）＋共有の専門家（幅 n_ff = 共有の幅）
        self.moe = arch == 'qwen35moe'
        self.n_ff = g('expert_shared_feed_forward_length') if self.moe else g('feed_forward_length')   # 17408（3.6 は 512）
        self.n_exp = g('expert_count', 0)
        self.n_used = g('expert_used_count', 0)
        self.n_ff_exp = g('expert_feed_forward_length', 0)
        self.n_head = g('attention.head_count')                 # 24
        self.n_head_kv = g('attention.head_count_kv')           # 4
        self.head_dim = g('attention.key_length')               # 256
        assert g('attention.value_length', self.head_dim) == self.head_dim
        self.eps = F32(g('attention.layer_norm_rms_epsilon'))   # 1e-6
        self.n_rot = g('rope.dimension_count')                  # 64
        self.sections = (list(g('rope.dimension_sections', [])) + [0] * 4)[:4]   # [11, 11, 10, 0]
        self.rope_base = g('rope.freq_base')                    # 1e7
        self.d_conv = g('ssm.conv_kernel')                      # 4
        self.d_state = g('ssm.state_size')                      # 128 = head_k_dim = head_v_dim
        self.n_k_heads = g('ssm.group_count')                   # 16
        self.n_v_heads = g('ssm.time_step_rank')                # 48
        self.d_inner = g('ssm.inner_size')                      # 6144 = 48·128
        self.fa_interval = g('full_attention_interval', 4)
        assert self.d_inner == self.n_v_heads * self.d_state
        assert self.n_head % self.n_head_kv == 0 and self.n_v_heads % self.n_k_heads == 0
        assert self.n_rot % 2 == 0 and sum(self.sections) * 2 <= self.n_rot

    def is_recr(self, il):
        # 3 つが線形（GDN）、4 つ目ごとに普通の Attention（il = 3, 7, 11, …）
        return il < self.n_layer and (il + 1) % self.fa_interval != 0


# ---------------- 小さな計算（ggml の op と同じ） ----------------

def sigmoid(x):
    with np.errstate(over='ignore'):
        return (F32(1) / (F32(1) + np.exp(-x))).astype(F32, copy=False)


def silu(x):
    return x * sigmoid(x)


def softplus(x):
    # ggml: x > 20 なら x、それ以外 log1p(exp(x))
    with np.errstate(over='ignore'):
        return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20)))).astype(F32)


def rms_norm(x, w, eps):
    """最後の軸で RMSNorm（2 乗の和は倍精度。ggml の CPU と同じ）して w を掛ける"""
    ms = np.mean(np.square(x, dtype=np.float64), axis=-1, keepdims=True)
    y = x * (1.0 / np.sqrt(ms + float(eps))).astype(F32)
    return y if w is None else y * w


def l2_norm(x, eps):
    # build_gdn_l2_norm: rms_norm(x, eps/n) / sqrt(n) = x / sqrt(Σx² + eps)
    ss = np.sum(np.square(x, dtype=np.float64), axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(ss + float(eps))).astype(F32)


def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return (e / np.sum(e, axis=axis, keepdims=True, dtype=np.float64)).astype(F32)


def rope_cache(pos, hp):
    """IMRoPE（qwen3vl 型の交互の M-RoPE）の cos・sin。文字だけなので位置は [p, p, p, 0]。返り値 (N, n_rot/2)。
    sections [11, 11, 10, 0] は 32 組を t/h/w に交互に割り当てる（e に当たる組は無い）→ 文字だけなら普通の NEOX と同じ"""
    half = hp.n_rot // 2
    sec = hp.sections
    sect = sum(sec)
    comp = []
    for i in range(half):
        s = i % sect
        if s % 3 == 1 and s < 3 * sec[1]:
            comp.append(1)
        elif s % 3 == 2 and s < 3 * sec[2]:
            comp.append(2)
        elif s % 3 == 0 and s < 3 * sec[0]:
            comp.append(0)
        else:
            comp.append(3)
    p = np.asarray(pos, np.int64)
    pos4 = np.stack([p, p, p, np.zeros_like(p)], axis=1)
    # ggml の CPU と同じく θ_i = p · base^(-2i/n_rot) を float で掛け重ねて作る
    ts = F32(np.power(F32(hp.rope_base), F32(-2.0 / hp.n_rot)))
    scale = np.empty(half, F32)
    v = F32(1)
    for i in range(half):
        scale[i] = v
        v = F32(v * ts)
    theta = pos4[:, comp].astype(F32) * scale
    return np.cos(theta).astype(F32), np.sin(theta).astype(F32)


def rope_apply(x, cs):
    """x (N, H, D) の先頭 n_rot 次元を NEOX 型（i と i+n_rot/2 の組）で回す。残り D-n_rot 次元はそのまま"""
    c, s = cs
    half = c.shape[1]
    c, s = c[:, None, :], s[:, None, :]
    y = x.copy()
    x0, x1 = x[..., :half], x[..., half:2 * half]
    y[..., :half] = x0 * c - x1 * s
    y[..., half:2 * half] = x0 * s + x1 * c
    return y


def dconv(x, w, state=None):
    """深さ方向の因果的な畳み込み（ggml_ssm_conv）。x (T, C)、w (C, K)、state (K-1, C) は前の K-1 語（無ければ 0）。
    out[t] = Σ_k w[:, k] · xx[t + k]、xx = [state; x]。返り値 (out, 新しい state = xx の最後の K-1 行)"""
    T, C = x.shape
    K = w.shape[1]
    if state is None:
        state = np.zeros((K - 1, C), F32)
    xx = np.concatenate([state, x], axis=0)
    out = np.zeros((T, C), F32)
    for k in range(K):
        out += xx[k:k + T] * w[:, k]
    return out, xx[-(K - 1):].copy()


# ---------------- 下請け（重みの戻しと GDN の走査を process に分ける） ----------------

_W = None


def _winit(index):
    global _W
    _W = Gguf(index=index)


def _deq_job(a):
    """name の行 r..r+n を戻し、/dev/shm の path（形 shape の float32）の行 dst から書く"""
    name, r, n, path, shape, dst = a
    out = np.memmap(path, F32, 'r+', shape=shape)
    out[dst:dst + n] = _W.rows(name, r, n)
    out.flush()
    del out
    return n


def gdn_scan(a):
    """Gated DeltaNet の走査（ggml_gated_delta_net と同じ順番）。
    q, k (T, Hv', D) は L2 正規化済みで v の頭ごとに並べた物（v の頭 h には q・k の頭 h % Hk）、
    v (T, Hv', D)、g・beta (T, Hv')。S[h] は (D_k, D_v)（llama.cpp の記憶では転置して [h][v][k] の順）。
    S0 (Hv', D, D) は前の状態（無ければ 0）。返り値 o (T, Hv', D)、最後の S"""
    q, k, v, g, beta, S0 = a
    T, Hv, D = v.shape
    scale = F32(1.0 / np.sqrt(D))
    S = np.zeros((Hv, D, D), F32) if S0 is None else S0.astype(F32).copy()
    o = np.empty((T, Hv, D), F32)
    eg = np.exp(g).astype(F32)
    for t in range(T):
        kt = k[t]                                            # (Hv, D)
        qt = q[t]
        S *= eg[t][:, None, None]                            # S ← S · exp(g)
        kv = np.matmul(kt[:, None, :], S)[:, 0, :]           # Σ_i S[i, j] k[i]
        dlt = (v[t] - kv) * beta[t][:, None]                 # δ = β (v − Sᵀk)
        S += kt[:, :, None] * dlt[:, None, :]                # S ← S + k δᵀ
        o[t] = np.matmul(qt[:, None, :], S)[:, 0, :] * scale  # o = Sᵀq / √D
    return o, S


def gdn_core(qkv, z, beta, g, W, segs, hp, mapper=None):
    """GDN の射影より後ろ（行列の掛け算なし）: 畳み込み → silu → L2Norm → 走査 → 頭ごとの RMSNorm × silu(z)。
    qkv (N, 10240)、z (N, 6144)、beta・g (N, 48)。返り値は llama.cpp の節の名前の dict（final_output が ssm_out の入力）"""
    N = qkv.shape[0]
    D, Hk, Hv = hp.d_state, hp.n_k_heads, hp.n_v_heads
    conv = np.empty_like(qkv)
    for a, b in segs:
        conv[a:b], _ = dconv(qkv[a:b], W['ssm_conv1d'])
    conv = silu(conv)
    kd = Hk * D
    q = l2_norm(conv[:, :kd].reshape(N, Hk, D), hp.eps)
    k = l2_norm(conv[:, kd:2 * kd].reshape(N, Hk, D), hp.eps)
    v = conv[:, 2 * kd:].reshape(N, Hv, D)
    # 長い列は頭を分けて並べる（頭どうしは独立）
    jobs, where = [], []
    for a, b in segs:
        ng = 16 if b - a > 256 else (4 if b - a > 32 else 1)
        for hs in np.array_split(np.arange(Hv), ng):
            kh = hs % Hk                                   # v の頭 h は q・k の頭 h % Hk（Qwen3.5 系の並び）
            jobs.append((q[a:b][:, kh], k[a:b][:, kh], v[a:b][:, hs], g[a:b][:, hs], beta[a:b][:, hs], None))
            where.append((a, b, hs))
    o = np.empty((N, Hv, D), F32)
    for (a, b, hs), (oo, _) in zip(where, (mapper or (lambda f, js: [f(j) for j in js]))(gdn_scan, jobs)):
        o[a:b, hs] = oo
    # 出口: 頭ごとに RMSNorm（ssm_norm は 128 個を全部の頭で共有）して silu(z) を掛ける
    out = rms_norm(o, W['ssm_norm'], hp.eps) * silu(z.reshape(N, Hv, D))
    return dict(conv_output_silu=conv, q_conv_predelta=q, k_conv_predelta=k, v_conv_predelta=v, attn_output=o,
                final_output=out)


def attn_core(q, k, v, gate, segs, hp, kv_f16):
    """RoPE 後の q (N, 24, 256)・k (N, 4, 256)、v (N, 4, 256)、gate (N, 6144) → 因果の softmax Attention と sigmoid の gate。
    返り値 (o (N, 24, 256), o·σ(gate) (N, 6144))"""
    N = q.shape[0]
    H, Hkv, D = hp.n_head, hp.n_head_kv, hp.head_dim
    if kv_f16:
        k = k.astype(np.float16).astype(F32)
        v = v.astype(np.float16).astype(F32)
    scale = F32(1.0 / np.sqrt(D))
    o = np.empty((N, H, D), F32)
    r = H // Hkv                                          # q の頭 h は KV の頭 h // 6 を使う（ggml の broadcast）
    for a, b in segs:
        T = b - a
        mask = np.triu(np.full((T, T), -np.inf, F32), 1)
        for h in range(H):
            s = (q[a:b, h] @ k[a:b, h // r].T) * scale + mask
            o[a:b, h] = softmax(s, axis=1) @ v[a:b, h // r]
    return o, o.reshape(N, H * D) * sigmoid(gate)


# ---------------- 本体 ----------------

class Ref:
    def __init__(self, path=DEFAULT_GGUF, workers=16, log=print):
        t0 = time.time()
        self.G = Gguf(path)
        self.hp = HP(self.G.meta)
        assert self.hp.arch in ('qwen35', 'qwen35moe'), self.hp.arch
        self.log = log
        self.workers = workers
        self.pool = None
        self._shm = []
        self._seq = 0
        log(f'目次 {len(self.G.t)} 個 {time.time() - t0:.1f}s  層 {self.hp.n_layer}（＋MTP {self.hp.n_nextn}）')

    def start_pool(self):
        if self.pool is None and self.workers > 0:
            os.environ['REF_Q35_WORKER'] = '1'
            ctx = mp.get_context('spawn')
            self.pool = ctx.Pool(self.workers, initializer=_winit, initargs=(self.G.index(),))
            del os.environ['REF_Q35_WORKER']

    def close(self):
        self._free()
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def _free(self):
        for p in self._shm:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
        self._shm = []

    def _shared(self, shape):
        """下請けが書ける float32 の置き場（/dev/shm のファイルを memmap）"""
        path = f'/dev/shm/ref_q35_{os.getpid()}_{self._seq}'
        self._seq += 1
        with open(path, 'wb') as f:
            f.truncate(int(np.prod(shape)) * 4)
        self._shm.append(path)
        return path, np.memmap(path, F32, 'r+', shape=shape)

    def _map(self, fn, jobs):
        if self.pool is None:
            if _W is None:
                _winit(self.G.index())
            return [fn(j) for j in jobs]
        return self.pool.map(fn, jobs, chunksize=1)

    def load(self, names, r0=0, n=None, chunk=4 << 20):
        """大きな重みを下請けで並べて戻す。返り値 {name: (行数, ne0) float32}（/dev/shm の上。_free で捨てる）。
        r0・n で行の範囲を切れる（出力層）"""
        out, jobs = {}, []
        for name in names:
            ncol = self.G.shape(name)[0]
            nr = self.G.n_rows(name) - r0 if n is None else n
            if self.pool is None:
                out[name] = self.G.rows(name, r0, nr)
                continue
            path, out[name] = self._shared((nr, ncol))
            step = max(1, chunk // ncol)
            jobs += [(name, r0 + i, min(step, nr - i), path, (nr, ncol), i) for i in range(0, nr, step)]
        if jobs:
            self._map(_deq_job, jobs)
        return out

    # --- 層の重み ---
    def layer_w(self, il):
        G, hp = self.G, self.hp
        p = f'blk.{il}.'
        W = {'attn_norm': G.mat(p + 'attn_norm.weight'), 'post_norm': G.mat(p + 'post_attention_norm.weight')}
        if hp.is_recr(il):
            big = ['attn_qkv', 'attn_gate', 'ssm_out']
            for k in ('ssm_alpha', 'ssm_beta', 'ssm_conv1d', 'ssm_norm'):
                W[k] = G.mat(p + k + '.weight')
            W['ssm_dt'] = G.mat(p + 'ssm_dt.bias')
            W['ssm_a'] = G.mat(p + 'ssm_a')
            assert (W['ssm_a'] < 0).all(), f'{p}ssm_a は −exp(A_log) のはず（正なら S が育ち続ける）'
        else:
            big = ['attn_q', 'attn_k', 'attn_v', 'attn_output']
            for k in ('attn_q_norm', 'attn_k_norm'):
                W[k] = G.mat(p + k + '.weight')
        if hp.moe:
            W['ffn_gate_inp'] = G.mat(p + 'ffn_gate_inp.weight')              # (n_exp, n_embd)
            W['ffn_gate_inp_shexp'] = G.mat(p + 'ffn_gate_inp_shexp.weight')  # (n_embd,)
            big += ['ffn_gate_shexp', 'ffn_up_shexp', 'ffn_down_shexp', 'ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps']
        else:
            big += ['ffn_gate', 'ffn_up', 'ffn_down']
        for k, v in self.load([p + k + '.weight' for k in big]).items():
            W[k[len(p):-len('.weight')]] = v
        return W

    # --- 線形 Attention（Gated DeltaNet） ---
    def gdn(self, x, W, segs, keep):
        qkv = x @ W['attn_qkv'].T                          # (N, 2·Hk·D + Hv·D) = [q | k | v]
        z = x @ W['attn_gate'].T                           # (N, Hv·D)
        beta = sigmoid(x @ W['ssm_beta'].T)                # (N, Hv)
        alpha = x @ W['ssm_alpha'].T                       # (N, Hv)
        g = softplus(alpha + W['ssm_dt']) * W['ssm_a']     # (N, Hv)  ssm_a = −exp(A_log) なので g ≤ 0
        d = gdn_core(qkv, z, beta, g, W, segs, self.hp, self._map)
        y = d['final_output'].reshape(x.shape[0], -1) @ W['ssm_out'].T
        keep(dict(linear_attn_qkv_mixed=qkv, z=z, beta_sigmoid=beta, alpha=alpha, gate=g, linear_attn_out=y, **d))
        return y

    # --- 普通の Attention（gate 付き・GQA・因果マスク） ---
    def attn(self, x, W, segs, poss, kv_f16, keep):
        hp = self.hp
        N = x.shape[0]
        H, Hkv, D = hp.n_head, hp.n_head_kv, hp.head_dim
        qf = (x @ W['attn_q'].T).reshape(N, H, 2 * D)        # 頭ごとに [q(256) | gate(256)]
        q = rms_norm(qf[:, :, :D], W['attn_q_norm'], hp.eps)
        gate = qf[:, :, D:].reshape(N, H * D)
        k = rms_norm((x @ W['attn_k'].T).reshape(N, Hkv, D), W['attn_k_norm'], hp.eps)
        v = (x @ W['attn_v'].T).reshape(N, Hkv, D)
        keep(dict(Qcur_full=qf.reshape(N, 2 * H * D), Qcur_normed=q, Kcur_normed=k, Vcur=v, gate_reshaped=gate))
        cs = rope_cache(np.concatenate(poss), hp)
        q = rope_apply(q, cs)
        k = rope_apply(k, cs)
        keep(dict(Qcur=q, Kcur=k))
        o, og = attn_core(q, k, v, gate, segs, hp, kv_f16)
        y = og @ W['attn_output'].T
        keep(dict(attn_pregate=o.reshape(N, H * D), attn_gated=og, attn_output=y))
        return y

    # --- MoE の FFN（3.6）: llama.cpp の build_moe_ffn（softmax・上位 k・重みを Σ で割る）＋共有の専門家（sigmoid のゲート） ---
    def moe_ffn(self, x, W, keep):
        hp = self.hp
        N, d = x.shape
        ff, ne, kk = hp.n_ff_exp, hp.n_exp, hp.n_used
        lg = x @ W['ffn_gate_inp'].T                              # (N, ne)
        pr = softmax(lg, axis=1)
        sel = np.argsort(-pr, axis=1, kind='stable')[:, :kk]
        w = np.take_along_axis(pr, sel, 1)
        w = w / np.maximum(w.sum(1, keepdims=True), F32(6.103515625e-5))
        keep(dict(ffn_moe_logits=lg, ffn_moe_topk=sel.astype(F32), ffn_moe_weights_norm=w))
        g3 = W['ffn_gate_exps'].reshape(ne, ff, d)
        u3 = W['ffn_up_exps'].reshape(ne, ff, d)
        d3 = W['ffn_down_exps'].reshape(ne, d, ff)
        y = np.zeros((N, d), F32)
        outs = np.zeros((N, kk, d), F32)
        for e in np.unique(sel):
            ti, ji = np.nonzero(sel == e)
            xe = x[ti]
            h = silu(xe @ g3[e].T) * (xe @ u3[e].T)
            outs[ti, ji] = h @ d3[e].T
        for j in range(kk):                                       # 枠の順に足す（ggml と同じ順）
            y = y + w[:, j:j + 1] * outs[:, j]
        sh = (silu(x @ W['ffn_gate_shexp'].T) * (x @ W['ffn_up_shexp'].T)) @ W['ffn_down_shexp'].T
        y = y + sh * sigmoid(x @ W['ffn_gate_inp_shexp'])[:, None]
        return y

    # --- 全体 ---
    def forward(self, seqs, logit_pos=None, dump_pos=None, detail=(), n_layers=None, kv_f16=False):
        """seqs: トークン列のリスト（どれも位置 0 から、互いに独立）。logit_pos[i]: 列 i でロジットを出す位置。
        dump_pos[i]: 層ごとの残差を残す位置（None で残さない）。detail: 途中の値を全部残す層。
        返り値: 列ごとの dict（logits (P, V)・h (L, P', n_embd)・emb・result_norm・detail）"""
        hp, G, log = self.hp, self.G, self.log
        L = hp.n_layer if n_layers is None else n_layers
        seqs = [np.asarray(s, np.int64) for s in seqs]
        lens = [len(s) for s in seqs]
        offs = np.cumsum([0] + lens)
        segs = [(int(offs[i]), int(offs[i + 1])) for i in range(len(seqs))]
        poss = [np.arange(n) for n in lens]
        N = int(offs[-1])
        tok = np.concatenate(seqs)
        if dump_pos is None:
            dump_pos = [None] * len(seqs)
        outs = [dict(tokens=s, h=[], detail={}) for s in seqs]
        self.start_pool()
        t0 = time.time()
        res = G.gather('token_embd.weight', tok)                          # (N, 5120)
        for o, (a, b) in zip(outs, segs):
            o['emb'] = res[a:b]
        log(f'埋め込み {time.time() - t0:.1f}s  N={N}')

        def keeper(il):
            def keep(d):
                if il not in detail:
                    return
                for k, v in d.items():
                    for o, (a, b) in zip(outs, segs):
                        o['detail'][f'{k}-{il}'] = v[a:b] if (np.ndim(v) >= 1 and v.shape[0] == N) else v
            return keep

        for il in range(L):
            tl = time.time()
            keep = keeper(il)
            W = self.layer_w(il)
            tw = time.time() - tl
            cur = rms_norm(res, W['attn_norm'], hp.eps)
            keep({'attn_norm': cur})
            if hp.is_recr(il):
                y = self.gdn(cur, W, segs, keep)
            else:
                y = self.attn(cur, W, segs, poss, kv_f16, keep)
            res = res + y
            keep({'attn_residual': res})
            cur = rms_norm(res, W['post_norm'], hp.eps)
            keep({'attn_post_norm': cur})
            if hp.moe:
                y = self.moe_ffn(cur, W, keep)
            else:
                # FFN: down( silu(gate x) · (up x) )
                hdn = silu(cur @ W['ffn_gate'].T) * (cur @ W['ffn_up'].T)
                y = hdn @ W['ffn_down'].T
            keep({'ffn_out': y})
            res = res + y
            keep({'l_out': res})
            for o, (a, b), dp in zip(outs, segs, dump_pos):
                if dp is not None:
                    o['h'].append(res[a:b][dp])
            del W
            self._free()
            log(f'層 {il:2d} {"GDN " if hp.is_recr(il) else "attn"} {time.time() - tl:5.1f}s'
                f'（戻し {tw:4.1f}s）  |res| {float(np.abs(res).max()):.1f}')
        for o in outs:
            o['h'] = np.stack(o['h']) if o['h'] else np.zeros((0,))
        if L == hp.n_layer and logit_pos is not None:
            idx = np.concatenate([np.asarray(lp, np.int64) + a for lp, (a, b) in zip(logit_pos, segs)])
            hn = rms_norm(res[idx], G.mat('output_norm.weight'), hp.eps)
            V = G.n_rows('output.weight')
            logits = np.empty((len(idx), V), F32)
            step = 1 << 15
            for r0 in range(0, V, step):
                n = min(step, V - r0)
                logits[:, r0:r0 + n] = hn @ self.load(['output.weight'], r0, n)['output.weight'].T
                self._free()
            k = 0
            for o, lp in zip(outs, logit_pos):
                o['logit_pos'] = np.asarray(lp, np.int64)
                o['result_norm'] = hn[k:k + len(lp)]
                o['logits'] = logits[k:k + len(lp)]
                k += len(lp)
        log(f'合計 {time.time() - t0:.1f}s')
        return outs


def save_npz(path, o):
    d = {k: v for k, v in o.items() if k != 'detail'}
    for k, v in o['detail'].items():
        d['detail.' + k] = v
    np.savez(path, **d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gguf', default=DEFAULT_GGUF)
    ap.add_argument('--tokens', required=True, help='カンマ区切りのトークン ID（位置 0 から）')
    ap.add_argument('--out', required=True)
    ap.add_argument('--logits', default='last', help='last | all | カンマ区切りの位置')
    ap.add_argument('--dump', default='all', help='層ごとの残差を残す位置: all | last:N | none')
    ap.add_argument('--detail', default='', help='途中の値も残す層（カンマ区切り）')
    ap.add_argument('--layers', type=int, default=None, help='先頭の何層だけ通すか')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--kv-f16', action='store_true', help='K・V を fp16 に丸める（llama.cpp の KV キャッシュと同じ）')
    a = ap.parse_args()
    toks = [int(x) for x in a.tokens.split(',') if x.strip()]
    T = len(toks)
    lp = {'last': [T - 1], 'all': list(range(T))}.get(a.logits)
    if lp is None:
        lp = [int(x) for x in a.logits.split(',')]
    dp = None if a.dump == 'none' else (np.arange(T) if a.dump == 'all' else np.arange(T)[-int(a.dump.split(':')[1]):])
    det = {int(x) for x in a.detail.split(',') if x.strip()}
    ref = Ref(a.gguf, workers=a.workers, log=lambda s: print(s, file=sys.stderr, flush=True))
    try:
        o = ref.forward([toks], [lp], [dp], det, a.layers, a.kv_f16)[0]
    finally:
        ref.close()
    if dp is not None:
        o['h_pos'] = dp
    save_npz(a.out, o)
    if 'logits' in o:
        lg = o['logits'][-1]
        top = np.argsort(-lg)[:10]
        lse = np.log(np.sum(np.exp(lg - lg.max()), dtype=np.float64)) + lg.max()
        print(json.dumps({'pos': int(lp[-1]), 'top': [[int(t), round(float(lg[t] - lse), 4)] for t in top]}))


if __name__ == '__main__':
    main()
