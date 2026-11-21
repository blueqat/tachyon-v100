"""普通の Attention（qwen35 の 4 層に 1 つ）の numpy の手本と、カーネル（target/release/attncheck）の答え合わせ。

手本の計算は llama.cpp の llama_model_qwen35::graph::build_layer_attn（unsloth b12209）と同じ:
  q_full = x·attn_q を頭ごとに [q 256 | gate 256] に分ける
  q = rope(rms_norm(q)·q_norm)、k = rope(rms_norm(k)·k_norm)、v はそのまま（eps 1e-6）
  rope: IMRoPE（ggml_rope_multi・LLAMA_ROPE_TYPE_IMROPE）を先頭 n_rot=64 次元に、NEOX 型の組（i, i+32）。
        角度は ggml-cuda と同じ pos·powf(powf(theta, -2/n_rot), i)。区画 [11,11,10,0] で t・h・w を交互に（文字だけなら全部 pos）
  o = softmax(q k^T / sqrt(256)) v（因果、GQA: q の頭 h は KV の頭 h // 6）
  out = o · sigmoid(gate)  → attn_output

  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/attn_ref.py [quick|all|long|mixed] [--gguf <GGUF>]
  （mixed は長さのばらばらな列を混ぜた組だけ。quick・all にも入っている）
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/attn_ref.py lcpp [--dump <dir>]   … llama.cpp の途中の値と（下）

判定（どれも 誤差 = max|GPU - 手本| / max|手本|）:
  rope: GPU の q と KV の中身 vs float64 の手本（fp16 に丸める分 ~5e-4）。int8（32 要素ごとの係数）は 1 要素の誤差が 係数/2
        （＋ f32 の角度の揺れで丸めが隣に転ぶ分 127·(1e-4 + 1.5e-8·位置)）以内
  attn: GPU が書いた q・KV（fp16／int8 を戻した値）から float64 で計算した手本 vs GPU の out < 2e-3
  参考: 全部 float64（fp16 に丸めない）の手本との差も出す
lcpp: llama.cpp（CPU）が本物の文で書き出した 16 層ぶんの射影（Qcur_full・Kcur_normed・Vcur）をカーネルに通し、
      llama.cpp 自身の Qcur・Kcur（rope 後）と、llama.cpp の f32 の q・k・v から float64 で計算した Attention と比べる
      （fp16 は max < 2e-3、int8 の KV は量子化の分を見込んで L2 < 3e-2・max < 6e-2）。llama.cpp 自身の attn_pregate・attn_gated との差も出す
"""
import os
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BIN = str(ROOT / "target/release/attncheck")
GGUF = os.path.join(os.environ.get("TACHYON_WEIGHTS", "weights"), "q38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf")
F32 = np.float32
D, PAGE = 256, 64
TOL = 2e-3


class HP:
    h, hkv, n_rot, sections, theta, eps = 24, 4, 64, (11, 11, 10, 0), 1e7, 1e-6
    yarn = None  # (freq_scale, ext_factor, mscale, corr0, corr1)。None は YaRN なし


def yarn_params(factor, orig, n_rot=64, base=1e7, attn_factor=1.0, beta_fast=32.0, beta_slow=1.0):
    """ggml_rope_yarn_corr_dims と同じ（src/attn.rs の Yarn::new）"""
    c = lambda b: n_rot * math.log(orig / (b * 2 * math.pi)) / (2 * math.log(base))
    return (1.0 / factor, 1.0, attn_factor, max(0.0, math.floor(c(beta_fast))), min(n_rot - 1.0, math.ceil(c(beta_slow))))


def hf_yarn_inv_freq(factor, orig, n_rot=64, base=1e7, beta_fast=32.0, beta_slow=1.0):
    """HF transformers の _compute_yarn_parameters（truncate=True）を float64 で。返り値 (inv_freq, attention_factor)"""
    pos_freqs = base ** (np.arange(0, n_rot, 2, dtype=np.float64) / n_rot)
    extra, inter = 1.0 / pos_freqs, 1.0 / (factor * pos_freqs)
    fcd = lambda r: (n_rot * math.log(orig / (r * 2 * math.pi))) / (2 * math.log(base))
    low, high = max(math.floor(fcd(beta_fast)), 0), min(math.ceil(fcd(beta_slow)), n_rot - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(n_rot // 2, dtype=np.float64) - low) / (high - low), 0, 1)
    ef = 1 - ramp
    return inter * (1 - ef) + extra * ef, 0.1 * math.log(factor) + 1.0


# ---------------------------------------------------------------- 手本

def rms_norm(x, w, eps):
    ms = np.mean(np.square(x, dtype=np.float64), axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * w


def rope_comp(n_rot, sections):
    """i = 0..n_rot/2-1 がどの位置（0 t, 1 h, 2 w, 3 e）を使うか（ggml の IMRoPE）"""
    sect = sum(sections)
    comp = []
    for i in range(n_rot // 2):
        if sect == 0:
            comp.append(0)
            continue
        s = i % sect
        if s % 3 == 1 and s < 3 * sections[1]:
            comp.append(1)
        elif s % 3 == 2 and s < 3 * sections[2]:
            comp.append(2)
        elif s % 3 == 0 and s < 3 * sections[0]:
            comp.append(0)
        else:
            comp.append(3)
    return np.array(comp)


def rope_cs(pos4, hp):
    """pos4 (N, 4) → cos, sin (N, n_rot/2)。角度は f32 で pos·powf(theta_scale, i)（ggml-cuda と同じ）"""
    half = hp.n_rot // 2
    ts = np.power(F32(hp.theta), F32(-2.0 / hp.n_rot), dtype=F32)
    freq = np.power(ts, np.arange(half, dtype=F32), dtype=F32)
    ang = (pos4[:, rope_comp(hp.n_rot, hp.sections)].astype(F32) * freq).astype(F32)
    if hp.yarn is not None:  # ggml の rope_yarn（角度は f32 のまま混ぜる。大きさは float64）
        fs, ext, msc, c0, c1 = hp.yarn
        y = (np.arange(half, dtype=F32) - F32(c0)) / F32(max(0.001, c1 - c0))
        mix = ((1 - np.clip(y, 0, 1)) * F32(ext)).astype(F32)
        ang = (ang * F32(fs) * (1 - mix) + ang * mix).astype(F32)
        if ext != 0:
            msc = msc * (1 + 0.1 * math.log(1 / fs))
        a = ang.astype(np.float64)
        return np.cos(a) * msc, np.sin(a) * msc
    a = ang.astype(np.float64)
    return np.cos(a), np.sin(a)


def rope(x, cs, n_rot):
    """x (N, H, D) の先頭 n_rot 次元を NEOX 型（i と i+n_rot/2）で回す"""
    c, s = cs[0][:, None, :], cs[1][:, None, :]
    half = n_rot // 2
    y = x.copy()
    x0, x1 = x[..., :half], x[..., half:n_rot]
    y[..., :half] = x0 * c - x1 * s
    y[..., half:n_rot] = x0 * s + x1 * c
    return y


def pos4_of(pos, pos_mode):
    if pos_mode == 4:
        return pos.reshape(-1, 4)
    p = pos.astype(np.int64)
    return np.stack([p, p, p, np.zeros_like(p)], axis=1)


def prep_q(qf, qnw, pos4, hp):
    """attn_q の出力 → (q, gate)。float64"""
    qg = qf.astype(np.float64).reshape(-1, hp.h, 2 * D)
    return rope(rms_norm(qg[:, :, :D], qnw, hp.eps), rope_cs(pos4, hp), hp.n_rot), qg[:, :, D:]


def prep_kv(kf, vf, knw, pos4, hp):
    """attn_k・attn_v の出力 → (k, v)。float64"""
    k = rope(rms_norm(kf.astype(np.float64).reshape(-1, hp.hkv, D), knw, hp.eps), rope_cs(pos4, hp), hp.n_rot)
    return k, vf.astype(np.float64).reshape(-1, hp.hkv, D)


def attention(q, gate, k, v, qpos, hp, chunk=256, f32mm=False):
    """q, gate (T, H, D)・k, v (L, HKV, D)・qpos (T,) → out (T, H·D)。因果: key ≤ qpos"""
    T, L = q.shape[0], k.shape[0]
    G = hp.h // hp.hkv
    scale = 1.0 / np.sqrt(D)
    out = np.empty((T, hp.h, D))
    mm = (lambda a, b: (a.astype(F32) @ b.astype(F32)).astype(np.float64)) if f32mm else (lambda a, b: a @ b)
    for g in range(hp.hkv):
        kg, vg = k[:, g], v[:, g]
        for t0 in range(0, T, chunk):
            t1 = min(T, t0 + chunk)
            qq = q[t0:t1, g * G:(g + 1) * G].reshape(-1, D)            # (n·G, D)
            kend = int(qpos[t1 - 1]) + 1
            s = mm(qq, kg[:kend].T) * scale
            pr = np.repeat(qpos[t0:t1], G)
            s[np.arange(kend)[None, :] > pr[:, None]] = -np.inf
            s -= s.max(axis=1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(axis=1, keepdims=True)
            out[t0:t1, g * G:(g + 1) * G] = mm(p, vg[:kend]).reshape(t1 - t0, G, D)
    return (out / (1.0 + np.exp(-gate))).reshape(T, hp.h * D)


def rel(got, ref):
    return float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30))


def rel2(got, ref):
    return float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-30))


# ---------------------------------------------------------------- 1 つの組を GPU で

def norm_weights(gguf_path, rng):
    """本物の blk.3.attn_{q,k}_norm（無ければでたらめ）"""
    try:
        from gguf import GGUFReader
        r = GGUFReader(gguf_path)
        w = {t.name: np.array(t.data, dtype=F32) for t in r.tensors if t.name in ("blk.3.attn_q_norm.weight", "blk.3.attn_k_norm.weight")}
        return w["blk.3.attn_q_norm.weight"], w["blk.3.attn_k_norm.weight"], "blk.3 の本物"
    except Exception as e:  # noqa: BLE001
        return (1 + 0.3 * rng.standard_normal(D)).astype(F32), (1 + 0.3 * rng.standard_normal(D)).astype(F32), f"でたらめ（{e}）"


def place(lens, rng):
    """列ごとのページ（物理ページをでたらめに混ぜ、使わないページも 7 つ混ぜる）"""
    npp = [(l + PAGE - 1) // PAGE for l in lens]
    n_pages = sum(npp) + 7
    perm = rng.permutation(n_pages).astype(int)
    pages, o = [], 0
    for n in npp:
        pages.append([int(x) for x in perm[o:o + n]])
        o += n
    return pages, n_pages


def gpu_run(name, seqs, pages, n_pages, qf, kf, vf, pos, nw, hp=HP, kv8=False, out_f16=False, kv_chunk=-1, pos_mode=1,
            gate=True, host_lens=True):
    """attncheck case を走らせる。返り値 (gq (NQ, H, D), gout (NQ, H·D), kc, vc, ks, vs, 1 行の記録)。
    host_lens: 列の長さの CPU の写し（kv_len_host）を渡して自動の split を列ごとの長さで決める（False は max_kv だけで決める前のやり方）"""
    qnw, knw = nw
    case = dict(h=hp.h, hkv=hp.hkv, n_rot=hp.n_rot, sections=list(hp.sections), theta=hp.theta, eps=hp.eps,
                yarn=list(hp.yarn) if hp.yarn is not None else None,
                kv8=int(kv8), out_f16=int(out_f16), kv_chunk=kv_chunk, pos_mode=pos_mode, n_pages=n_pages,
                gate=int(gate), host_lens=int(host_lens), seqs=[dict(kv_len=l, q_len=q, pages=p) for (l, q), p in zip(seqs, pages)])
    ks = vs = None
    with tempfile.TemporaryDirectory(dir="/dev/shm") as d:
        dp = Path(d)
        (dp / "case.json").write_text(json.dumps(case))
        for nm, a in (("qf", qf), ("kf", kf), ("vf", vf), ("qnw", qnw), ("knw", knw)):
            np.ascontiguousarray(a, dtype=F32).tofile(dp / f"{nm}.f32")
        pos.astype(np.int32).tofile(dp / "pos.i32")
        r = subprocess.run([BIN, "case", d], capture_output=True, text=True)
        if r.returncode:
            print(r.stdout, r.stderr)
            raise SystemExit(f"{name}: attncheck が失敗")
        info = r.stderr.strip().splitlines()[-1]
        gq = np.fromfile(dp / "q.f32", F32).reshape(-1, hp.h, D)
        gout = np.fromfile(dp / "out.f32", F32).reshape(-1, hp.h * D)
        if kv8:
            kc = np.fromfile(dp / "kc.bin", np.int8).reshape(n_pages, hp.hkv, PAGE, D)
            vc = np.fromfile(dp / "vc.bin", np.int8).reshape(n_pages, hp.hkv, PAGE, D)
            ks = np.fromfile(dp / "ks.f32", F32).reshape(n_pages, hp.hkv, PAGE, D // 32)
            vs = np.fromfile(dp / "vs.f32", F32).reshape(n_pages, hp.hkv, PAGE, D // 32)
        else:
            kc = np.fromfile(dp / "kc.bin", np.float16).reshape(n_pages, hp.hkv, PAGE, D)
            vc = np.fromfile(dp / "vc.bin", np.float16).reshape(n_pages, hp.hkv, PAGE, D)
    return gq, gout, kc, vc, ks, vs, info


def gather_kv(kc, vc, ks, vs, pg, l):
    """列の KV（GPU が書いたもの、int8 は戻した値）を (l, HKV, D) で"""
    sl = np.array([pg[p // PAGE] * PAGE + p % PAGE for p in range(l)])
    gk = kc[sl // PAGE, :, sl % PAGE].astype(np.float64)
    gv = vc[sl // PAGE, :, sl % PAGE].astype(np.float64)
    if ks is not None:  # 32 要素ごとの係数
        gk = gk * np.repeat(ks[sl // PAGE, :, sl % PAGE], 32, axis=-1)
        gv = gv * np.repeat(vs[sl // PAGE, :, sl % PAGE], 32, axis=-1)
    return gk, gv, sl


def run_case(name, seqs, rng, nw, hp=HP, kv8=False, out_f16=False, kv_chunk=-1, pos_mode=1, gate=True,
             xscale=1.0, f32mm=False, host_lens=True, pos_off=0, rope_tol=2e-3):
    """seqs: [(kv_len, q_len)]。全トークンの射影の出力をでたらめに作り、GPU と手本を比べる"""
    t_start = time.time()
    lens = [l for l, _ in seqs]
    N, NQ = sum(lens), sum(q for _, q in seqs)
    qf = (xscale * rng.standard_normal((NQ, hp.h * 2 * D))).astype(F32)   # q は新しいトークンの分だけ
    kf = (xscale * rng.standard_normal((N, hp.hkv * D))).astype(F32)
    vf = rng.standard_normal((N, hp.hkv * D)).astype(F32)
    pos = np.concatenate([np.arange(l) + pos_off for l in lens]).astype(np.int32)
    if pos_mode == 4:  # 画像のような t・h・w が別々の位置
        p4 = np.stack([pos, pos + rng.integers(0, 50, N), pos // 2 + rng.integers(0, 9, N), rng.integers(0, 5, N)], axis=1)
        pos = p4.astype(np.int32).reshape(-1)
    pages, n_pages = place(lens, rng)
    qnw, knw = nw
    gq, gout, kc, vc, ks, vs, info = gpu_run(name, seqs, pages, n_pages, qf, kf, vf, pos, nw, hp, kv8, out_f16, kv_chunk,
                                             pos_mode, gate, host_lens)
    t_gpu = time.time()

    p4 = pos4_of(pos, pos_mode)
    is_new = np.concatenate([np.arange(l) >= l - ql for l, ql in seqs])
    q, g = prep_q(qf, qnw.astype(np.float64), p4[is_new], hp)
    k, v = prep_kv(kf, vf, knw.astype(np.float64), p4, hp)
    res = dict(rope=0.0, kv=0.0, attn=0.0, full=0.0)
    off, qo = 0, 0
    res["rope"] = rel(gq, q)
    ref_a, ref_f = [], []
    worst_int8 = 0.0
    for (l, ql), pg in zip(seqs, pages):
        gk, gv, sl = gather_kv(kc, vc, ks, vs, pg, l)
        if kv8:
            ksc, vsc = ks[sl // PAGE, :, sl % PAGE], vs[sl // PAGE, :, sl % PAGE]       # (l, HKV, D/32)
            for ref, gg, sc in ((k[off:off + l], gk, ksc), (v[off:off + l], gv, vsc)):
                amax = np.abs(ref.reshape(l, hp.hkv, D // 32, 32)).max(axis=-1)
                assert np.allclose(sc, amax / 127, rtol=1e-3, atol=1e-12), f"{name}: int8 の係数"
                worst_int8 = max(worst_int8, float((np.abs(gg - ref) / (np.repeat(sc, 32, axis=-1) + 1e-30)).max()))
        res["kv"] = max(res["kv"], rel(gk, k[off:off + l]), rel(gv, v[off:off + l]))
        qpos = np.arange(l - ql, l)
        gate_s = g[qo:qo + ql] if gate else np.full((ql, hp.h, D), 1e30)
        ref_a.append(attention(gq[qo:qo + ql].astype(np.float64), gate_s, gk, gv, qpos, hp, f32mm=f32mm))
        ref_f.append(attention(q[qo:qo + ql], gate_s, k[off:off + l], v[off:off + l], qpos, hp, f32mm=f32mm))
        off += l
        qo += ql
    ref_a, ref_f = np.concatenate(ref_a), np.concatenate(ref_f)
    res["attn"] = rel(gout, ref_a)
    res["attn_l2"] = rel2(gout, ref_a)
    res["full"] = rel(gout, ref_f)
    out_tol = TOL + (5e-4 if out_f16 else 0)
    # int8: 丸めの 0.5 に、f32 の RoPE の角度の揺れ（powf の 1〜2 ulp × 位置）で丸めが隣に転ぶ分を足す
    int8_tol = 0.5 + 127 * (1e-4 + 1.5e-8 * max(lens))
    good = res["rope"] < rope_tol and res["attn"] < out_tol and (worst_int8 <= int8_tol if kv8 else res["kv"] < rope_tol)
    kvs = f"int8 誤差/係数 {worst_int8:.3f}" if kv8 else f"kv {res['kv']:.1e}"
    print(f"{'OK' if good else 'NG'} {name:24s} rope {res['rope']:.1e}  {kvs}  attn {res['attn']:.1e} (L2 {res['attn_l2']:.1e})"
          f"  全部 f64 との差 {res['full']:.1e}  [{info.split(': ', 1)[1]}]  GPU {t_gpu - t_start:.1f}s 手本 {time.time() - t_gpu:.1f}s",
          flush=True)
    return good


def run_lcpp(dump, gguf_path, kv8=False):
    """llama.cpp（CPU、unsloth b12209）が本物の文（31 語）で書き出した途中の値で、普通の Attention の 16 層を 1 層ずつ比べる。
    dump: tools/lcpp_dump の出力（Qcur_full・Kcur_normed・Vcur・Qcur・Kcur・attn_pregate・attn_gated の -<層>.npy）。
    k の生の射影は Kcur の名前が上書きされて残らないので、kf = Kcur_normed / k_norm（大きさ 1 の行。正規化し直しても同じ）。
    k_norm が 0 の次元（層 43 にある）は値が分からないが掛ける重みが 0 なので、行の 2 乗和が D になるように埋める。
    手本は llama.cpp の f32 の Qcur・Kcur・Vcur から float64 で計算した Attention（llama.cpp 自身の CPU の Attention は
    fp16 で溜めるので 1〜3e-3 ずれる。その差も出す）"""
    from gguf import GGUFReader
    r = GGUFReader(gguf_path)
    W = {t.name: np.array(t.data, dtype=F32) for t in r.tensors if t.name.endswith(("attn_q_norm.weight", "attn_k_norm.weight"))}
    dp = Path(dump)
    layers = sorted(int(f.stem.split("-")[1]) for f in dp.glob("attn_gated-*.npy"))
    rng = np.random.default_rng(7)
    oks, worst = [], {}
    # int8 は量子化そのものの誤差（numpy で同じ量子化をした Attention でも L2 0.5〜1.2%・max 1.3e-2）を見込んで、
    # L2 < 3e-2・max < 6e-2 で大きな間違いだけを見る。fp16 は max < 2e-3
    tol = 6e-2 if kv8 else 2e-3
    print(f"--- KV {'int8（32 要素ごと）' if kv8 else 'fp16'}: llama.cpp の値と、llama.cpp の f32 の q・k・v から float64 で計算した手本")
    for il in layers:
        ld = lambda n: np.load(dp / f"{n}-{il}.npy").astype(np.float64)  # noqa: E731
        qf, kn, vv = ld("Qcur_full"), ld("Kcur_normed"), ld("Vcur")
        T = qf.shape[0]
        qnw, knw = W[f"blk.{il}.attn_q_norm.weight"], W[f"blk.{il}.attn_k_norm.weight"].astype(np.float64)
        miss = knw == 0
        kf = np.where(miss, 0.0, kn / np.where(miss, 1.0, knw))
        if miss.any():
            ss = (kf ** 2).sum(-1, keepdims=True)
            kf = np.where(miss, np.sqrt(np.maximum(D - ss, 0.0) / miss.sum()), kf)
        pos = np.arange(T, dtype=np.int32)
        pages, n_pages = place([T], rng)
        args = ([(T, T)], pages, n_pages, qf, kf.reshape(T, -1), vv.reshape(T, -1), pos, (qnw, knw.astype(F32)))
        gq, gout, kc, vc, ks, vs, _ = gpu_run(f"lcpp {il}", *args, kv8=kv8)
        gout_ng = gpu_run(f"lcpp {il}", *args, kv8=kv8, gate=False)[1]
        gk, gv, _ = gather_kv(kc, vc, ks, vs, pages[0], T)
        q32, k32 = ld("Qcur"), ld("Kcur")
        gate = qf.reshape(T, HP.h, 2 * D)[:, :, D:]
        qpos = np.arange(T)
        ref = attention(q32, gate, k32, vv, qpos, HP)
        ref_pre = attention(q32, np.full_like(gate, 1e30), k32, vv, qpos, HP)
        e = dict(q=rel(gq, ld("Qcur")), k=rel(gk, k32), v=rel(gv, vv), pre=rel(gout_ng, ref_pre), out=rel(gout, ref),
                 out_l2=rel2(gout, ref), lcpp_pre=rel(ld("attn_pregate"), ref_pre), lcpp_out=rel(ld("attn_gated"), ref),
                 vs_lcpp=rel(gout, ld("attn_gated")))
        for kk, vl in e.items():
            worst[kk] = max(worst.get(kk, 0.0), vl)
        good = (e["q"] < 2e-3 and e["k"] < (1e-2 if kv8 else 2e-3) and e["pre"] < tol and e["out"] < tol
                and e["out_l2"] < (3e-2 if kv8 else 2e-3))
        oks.append(good)
        print(f"{'OK' if good else 'NG'} 層 {il:2d}（T={T}）  Qcur {e['q']:.1e}  Kcur {e['k']:.1e}  Vcur {e['v']:.1e}  "
              f"手本と: pregate {e['pre']:.1e} gated {e['out']:.1e}（L2 {e['out_l2']:.1e}）  "
              f"llama.cpp と手本: {e['lcpp_pre']:.1e} / {e['lcpp_out']:.1e}  GPU と llama.cpp: {e['vs_lcpp']:.1e}", flush=True)
    print("最大: " + "  ".join(f"{k} {v:.1e}" for k, v in worst.items()))
    return oks


def main():
    args = sys.argv[1:]
    which = args[0] if args and not args[0].startswith("--") else "quick"
    gg = args[args.index("--gguf") + 1] if "--gguf" in args else GGUF
    if which == "lcpp":
        dump = args[args.index("--dump") + 1] if "--dump" in args else str(ROOT / "logs/lcpp31")
        oks = run_lcpp(dump, gg) + run_lcpp(dump, gg, kv8=True)
        print(f"{sum(oks)}/{len(oks)} OK")
        sys.exit(0 if all(oks) else 1)
    rng = np.random.default_rng(1234)
    nw = norm_weights(gg, rng)
    print(f"q_norm・k_norm: {nw[2]}（q_norm 平均 {nw[0].mean():.3f} 最大 {np.abs(nw[0]).max():.3f}、k_norm 平均 {nw[1].mean():.3f} 最大 {np.abs(nw[1]).max():.3f}）")
    nw = nw[:2]
    oks = []
    mix = [(547, 547), (1300, 300), (5000, 1), (777, 3), (64, 1), (1, 1), (65, 2), (129, 17)]
    dec = [(int(x), 1) for x in rng.integers(1, 9000, 8)] + [(64, 1), (65, 1), (1, 1)]
    ver2 = [(int(x), 2) for x in rng.integers(2, 3000, 5)]
    ver5 = [(int(x), int(q)) for x, q in zip(rng.integers(5, 6000, 9), [3, 4, 5, 1, 2, 3, 5, 4, 3])]   # 18〜30 行 → CfgV
    ver17 = [(int(x), int(q)) for x, q in zip(rng.integers(17, 5000, 6), [17, 9, 6, 1, 17, 10])]       # → CfgP（2 塊も）
    # 長さのばらばらな列を混ぜる: 自動の split の幅は列の長さ（kv_len_host）で決める。host_lens=False は max_kv だけで決める前のやり方。
    # 長い列は細かく・短い列は分けずに、どちらでも同じ答え
    mdec = [(32000, 1), (1000, 1), (64, 1), (1, 1), (5000, 1), (300, 1), (17, 1), (8192, 1)]
    mver = [(20000, 3), (100, 5), (3000, 1), (1, 1), (12000, 2), (65, 4)]
    mp17 = [(20000, 17), (500, 9), (64, 17), (3000, 6)]
    mmany = [(30000, 1)] + [(int(x), 1) for x in rng.integers(1, 2000, 47)]  # 48 列 × 4 頭 ＞ 常駐数 160（波の端数の道）
    if which == "sharp":  # 尖った softmax（本物の文の attention の偏りに近づける）で 1 列の確かめ
        for mul in (3, 6, 10):
            for l, q, ch in [(8003, 3, -1), (8003, 3, 224), (8003, 3, 256), (8003, 3, 64), (8003, 1, -1), (8004, 3, -1)]:
                oks.append(run_case(f"×{mul} kv {l} q {q} split {ch}", [(l, q)], rng, (nw[0] * mul, nw[1] * mul), kv_chunk=ch, host_lens=False))
    if which == "one":  # 1 列だけの確かめ（M2 の MTP: 本体は kv_len_host を渡さない＝host_lens=False）
        for l, q in [(8003, 3), (8004, 3), (6000, 3), (8003, 2), (8003, 4), (2051, 3), (16003, 3), (8003, 5)]:
            for hl in (False, True):
                oks.append(run_case(f"1 列 kv {l} q {q}{'' if hl else '（host なし）'}", [(l, q)], rng, nw, host_lens=hl))
    if which in ("quick", "all", "mixed"):
        oks.append(run_case("混ぜた長さ decode", mdec, rng, nw))
        oks.append(run_case("混ぜた長さ decode（前）", mdec, rng, nw, host_lens=False))
        oks.append(run_case("混ぜた長さ verify 1〜5", mver, rng, nw))
        oks.append(run_case("混ぜた長さ verify CfgP", mp17, rng, nw))
        oks.append(run_case("混ぜた長さ 48 列", mmany, rng, nw))
        # int8 は 9000 まで（係数の確かめ rtol 1e-3 が、3 万を超える位置では f32 の RoPE の角度の揺れで足りなくなる。split とは関係ない）
        oks.append(run_case("混ぜた長さ decode kv8", [(min(l, 9000), q) for l, q in mdec], rng, nw, kv8=True))
    if which in ("quick", "all"):
        oks.append(run_case("mix", mix, rng, nw))
        oks.append(run_case("mix split=128", mix, rng, nw, kv_chunk=128))
        oks.append(run_case("decode 自動 split", dec, rng, nw))
        oks.append(run_case("decode split=64", dec, rng, nw, kv_chunk=64))
        oks.append(run_case("verify 2 トークン", ver2, rng, nw))
        oks.append(run_case("verify 3〜5 トークン", ver5, rng, nw))
        oks.append(run_case("verify 〜17 トークン", ver17, rng, nw))
        oks.append(run_case("mix kv8", mix, rng, nw, kv8=True))
        oks.append(run_case("decode kv8", dec, rng, nw, kv8=True))
        oks.append(run_case("verify 3〜5 kv8 split=96", ver5, rng, nw, kv8=True, kv_chunk=96))
    if which == "all":
        oks.append(run_case("mix out fp16", mix, rng, nw, out_f16=True))
        oks.append(run_case("mix gate なし", mix, rng, nw, gate=False))
        oks.append(run_case("imrope t/h/w 別々", [(300, 300), (900, 5), (70, 1)], rng, nw, pos_mode=4))
        oks.append(run_case("尖った softmax ×3", mix, rng, (nw[0] * 3, nw[1] * 3), kv_chunk=256))
        oks.append(run_case("verify 2 kv8 split=64", ver2, rng, nw, kv8=True, kv_chunk=64))
        oks.append(run_case("B=16 decode 20K〜32K", [(int(x), 1) for x in rng.integers(20000, 32768, 16)], rng, nw, f32mm=True))
    if which in ("long", "all"):
        oks.append(run_case("prefill T=8192", [(8192, 8192)], rng, nw, f32mm=True))
        oks.append(run_case("prefill T=8192 kv8", [(8192, 8192)], rng, nw, kv8=True, f32mm=True))
        oks.append(run_case("続き 2048 @ 30000", [(32000, 2048)], rng, nw, f32mm=True))
    if which == "yarn":  # YaRN（factor 4・元 262144）: HF の式と ggml の式が同じか、GPU の rope と attention が手本と合うか
        fs, ext, msc, c0, c1 = yarn_params(4.0, 262144)
        inv, af = hf_yarn_inv_freq(4.0, 262144)
        base_f = 1e7 ** (-np.arange(0, 64, 2) / 64)
        mix = (1 - np.clip((np.arange(32) - c0) / max(0.001, c1 - c0), 0, 1)) * ext
        gg_f = base_f * fs * (1 - mix) + base_f * mix
        d_inv = float(np.max(np.abs(gg_f / inv - 1)))
        d_af = abs(msc * (1 + 0.1 * math.log(1 / fs)) - af)
        print(f"YaRN corr {c0}..{c1}  mscale {af:.6f}  HF と ggml の周波数の差 {d_inv:.1e}・大きさの差 {d_af:.1e}")
        oks.append(d_inv < 1e-12 and d_af < 1e-12)
        class HY(HP):
            yarn = (fs, ext, msc, c0, c1)
        oks.append(run_case("yarn mix", mix_cases(rng), rng, nw, hp=HY))
        # 70 万〜100 万の位置では f32 の角度（pos·powf）の 1ulp の違い（CUDA の powf と numpy）で rope の差が 1e-2 まで出る。
        # YaRN なしでも同じだけ出る（下の「比べ」）ので、ここだけ rope の許しを 3e-2 に。attention の出力は 1e-4 のまま
        oks.append(run_case("yarn なし @ 1M（比べ）", [(4000, 3), (777, 3)], rng, nw, pos_off=1000000, rope_tol=3e-2))
        oks.append(run_case("yarn decode @ 700K", [(3000, 1), (900, 1), (5000, 1)], rng, nw, hp=HY, pos_off=700000, rope_tol=3e-2))
        oks.append(run_case("yarn verify 3 @ 1M", [(4000, 3), (777, 3)], rng, nw, hp=HY, pos_off=1000000, rope_tol=3e-2))
        oks.append(run_case("yarn decode @ 20K", [(3000, 1), (900, 1), (5000, 1)], rng, nw, hp=HY, pos_off=20000))
        oks.append(run_case("yarn kv8 @ 4K", [(3000, 1), (2000, 5)], rng, nw, hp=HY, kv8=True, pos_off=4000))
    print(f"{sum(oks)}/{len(oks)} OK")
    sys.exit(0 if all(oks) else 1)


def mix_cases(rng):
    return [(547, 547), (1300, 300), (5000, 1), (777, 3), (64, 1), (1, 1), (65, 2), (129, 17)]


if __name__ == "__main__":
    main()
