"""プレフィルの GEMM（kernels/gemm.cu の infer_gemm / infer_gemm_swiglu / infer_gemm_multi）の答え合わせと速さ。
手本は gguf-py の dequantize と numpy（float64）。重みは GGUF の中を直接読む（写さない）。X は fp16。

  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/gemm_check.py check  [gguf]   … 全部の型・タイルの組・書き出しの形
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/gemm_check.py bench  [gguf]   … 形 × M ごとの TFLOPS
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/gemm_check.py model  [gguf]   … 64 層ぶんの GEMM の時間（TTFT の見積もり）
  （--cfg N でタイルの組を決め打ち、--m 16,64 で M を絞る、--shape 名前 で形を絞る、check は --only 正規表現 で行列を絞る）
"""
import argparse
import collections
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as T
from gguf import GGUFReader
from gguf.quants import dequantize

ROOT = Path(__file__).resolve().parent.parent
KCHECK = str(ROOT / "target/release/kcheck")
GGUF = os.path.join(os.environ.get("TACHYON_WEIGHTS", "weights"), "q38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf")
TOL = 2e-3  # 合格の線: max|y - 手本| / max|手本|

# 確かめる行列（テンソル名, 使う行数）。型ごとに 1 つ以上。行数は端（タイルに割り切れない列）も見るため半端に
CHECKS = [
    ("blk.0.attn_qkv.weight", 999),   # Q5_K 5120→10240
    ("blk.1.attn_qkv.weight", 999),   # Q4_K 5120→10240
    ("blk.1.ssm_out.weight", 999),    # Q6_K 6144→5120
    ("blk.0.ffn_gate.weight", 999),   # IQ4_XS 5120→17408
    ("blk.0.ffn_down.weight", 600),   # IQ4_XS 17408→5120
    ("blk.1.ffn_down.weight", 600),   # IQ4_NL 17408→5120
    ("blk.3.ffn_down.weight", 600),   # Q5_K 17408→5120
    ("blk.0.ssm_out.weight", 999),    # Q8_0 6144→5120
    ("blk.0.ssm_alpha.weight", 48),   # Q8_0 5120→48
    ("blk.0.ffn_up.weight", 999),     # Q3_K 5120→17408
    ("blk.14.ffn_down.weight", 600),  # IQ3_S 17408→5120（この GGUF で 1 本だけ）
    ("F16:5120:777", 777),            # 乱数の F16
]
MS_CHECK = [1, 16, 37, 64, 200, 547]

# 速さを測る形（名前, 種類, テンソル）。種類: gemm / swiglu（gate,up）/ multi（,区切り）
SHAPES = [
    ("gate_up", "swiglu", "blk.3.ffn_gate.weight,blk.3.ffn_up.weight"),      # Q4_K+Q5_K 5120→2×17408
    ("gate_up_iq4", "swiglu", "blk.0.ffn_gate.weight,blk.1.ffn_up.weight"),  # IQ4_XS 2 本
    ("ffn_up", "gemm", "blk.3.ffn_up.weight"),                               # Q5_K 5120→17408
    ("ffn_gate_iq4", "gemm", "blk.0.ffn_gate.weight"),                       # IQ4_XS 5120→17408
    ("ffn_down", "gemm", "blk.3.ffn_down.weight"),                           # Q5_K 17408→5120
    ("ffn_down_iq4", "gemm", "blk.0.ffn_down.weight"),                       # IQ4_XS 17408→5120
    ("ffn_down_q6", "gemm", "blk.56.ffn_down.weight"),                       # Q6_K 17408→5120
    ("ffn_down_iq3s", "gemm", "blk.14.ffn_down.weight"),                     # IQ3_S 17408→5120
    ("lin_in", "multi", "blk.0.attn_qkv.weight,blk.0.attn_gate.weight,blk.0.ssm_alpha.weight,blk.0.ssm_beta.weight"),
    ("attn_qkv", "gemm", "blk.0.attn_qkv.weight"),                           # Q5_K 5120→10240
    ("ssm_out", "gemm", "blk.4.ssm_out.weight"),                             # Q5_K 6144→5120
    ("full_in", "multi", "blk.3.attn_q.weight,blk.3.attn_k.weight,blk.3.attn_v.weight"),
    ("attn_output", "gemm", "blk.3.attn_output.weight"),                     # Q6_K 6144→5120
]
MS = [16, 64, 256, 512, 2048, 8192]


class Weights:
    """GGUF からテンソルの (位置, 型, 形, バイト数) を引く"""

    def __init__(self, path):
        self.path = path
        self.t = {t.name: (int(t.data_offset), t.tensor_type.name, [int(x) for x in t.shape], int(t.n_bytes))
                  for t in GGUFReader(path).tensors}

    def get(self, name):
        off, ty, shape, nb = self.t[name]
        return off, ty, shape[0], int(np.prod(shape[1:])), nb // int(np.prod(shape[1:]))

    def raw(self, name, rows=None):
        off, ty, k, n, rb = self.get(name)
        rows = n if rows is None else rows
        return np.fromfile(self.path, dtype=np.uint8, count=rows * rb, offset=off)

    def ref(self, name, rows=None):
        """戻した重み [rows, k]（float64）"""
        off, ty, k, n, rb = self.get(name)
        rows = n if rows is None else rows
        return dequantize(self.raw(name, rows), T[ty]).reshape(rows, k).astype(np.float64)


def kcheck(*args, env=None):
    e = dict(os.environ, **(env or {}))
    r = subprocess.run([KCHECK, *map(str, args)], capture_output=True, text=True, env=e)
    if r.returncode:
        raise RuntimeError(f"kcheck {' '.join(map(str, args))}\n{r.stderr[-3000:]}")
    return r.stderr.strip()


def err(y, ref):
    return float(np.abs(y - ref).max() / (np.abs(ref).max() + 1e-30)), float(np.sqrt(((y - ref) ** 2).mean() / ((ref ** 2).mean() + 1e-30)))


def env_of(cfg=-1, y=None, splits=None):
    e = {"INFER_GEMM_CFG": str(cfg)}
    if y:
        e["INFER_GEMM_Y"] = y
    if splits is not None:
        e["INFER_GEMM_SPLITS"] = str(splits)
    return e


def line(ok, label, e, r):
    print(f"{'OK' if ok else 'NG'} {label:70s} 最大 {e:.1e} 二乗平均 {r:.1e}", flush=True)


def cmd_check(a):
    W = Weights(a.gguf)
    rng = np.random.default_rng(0)
    ok = True
    with tempfile.TemporaryDirectory(dir=ROOT / "logs") as d:
        tmp = Path(d)
        for name, rows in CHECKS:
            if a.only and not re.search(a.only, name):
                continue
            if name.startswith("F16:"):
                _, k, n = name.split(":")
                k, n = int(k), int(n)
                wf = (rng.standard_normal((n, k)) * 0.05).astype(np.float16)
                wf.tofile(tmp / "w.bin")
                spec, ty, ref_w = str(tmp / "w.bin"), "F16", wf.astype(np.float64)
            else:
                off, ty, k, n, rb = W.get(name)
                n = min(n, rows)
                spec, ref_w = f"{W.path}@{off}", W.ref(name, n)
            for m in MS_CHECK:
                x = rng.standard_normal((m, k)).astype(np.float16)
                x.tofile(tmp / "x.f16")
                ref = x.astype(np.float64) @ ref_w.T
                runs = [(-1, None, None)]
                if m in (37, 547):
                    runs += [(c, None, None) for c in (0, 2, 3, 4, 5)]
                if m == 64:
                    runs += [(-1, "f16", None), (2, None, 3), (4, None, 5), (0, None, 2)]
                for cfg, yk, sp in runs:
                    kcheck("gemm", spec, ty, k, n, m, tmp / "x.f16", tmp / "y.f32", env=env_of(cfg, yk, sp))
                    y = np.fromfile(tmp / "y.f32", dtype=np.float32).reshape(m, n)
                    e, r = err(y, ref)
                    good = e < TOL and np.isfinite(y).all()
                    ok &= good
                    line(good, f"{name} {ty} k={k} n={n} m={m} cfg={cfg} y={yk or 'f32'} splits={sp if sp else 'auto'}", e, r)
            # 残差に足す（split-K でも）
            m = 64
            x = rng.standard_normal((m, k)).astype(np.float16)
            x.tofile(tmp / "x.f16")
            y0 = rng.standard_normal((m, n)).astype(np.float32)
            y0.tofile(tmp / "y0.f32")
            ref = y0.astype(np.float64) + x.astype(np.float64) @ ref_w.T
            for sp in (1, 4):
                kcheck("gemm", spec, ty, k, n, m, tmp / "x.f16", tmp / "y.f32", env=env_of(-1, f"add:{tmp / 'y0.f32'}", sp))
                e, r = err(np.fromfile(tmp / "y.f32", dtype=np.float32).reshape(m, n), ref)
                ok &= e < TOL
                line(e < TOL, f"{name} {ty} y+= m={m} splits={sp}", e, r)
        ok &= check_fused(W, tmp, rng, a.only)
    print("全部 OK" if ok else "⛔ NG あり")
    return 0 if ok else 1


def check_fused(W, tmp, rng, only=""):
    ok = True
    # SwiGLU（gate と up の型が違う組も）
    for g, u in [("blk.3.ffn_gate.weight", "blk.3.ffn_up.weight"),     # Q4_K + Q5_K
                 ("blk.0.ffn_gate.weight", "blk.1.ffn_up.weight"),     # IQ4_XS + IQ4_XS
                 ("blk.58.ffn_gate.weight", "blk.59.ffn_up.weight"),   # Q6_K + Q6_K
                 ("blk.17.ffn_gate.weight", "blk.28.ffn_up.weight"),   # IQ4_NL + IQ4_NL
                 ("blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"),     # IQ4_XS + Q3_K
                 ("blk.13.ffn_gate.weight", "blk.13.ffn_up.weight"),   # Q3_K + IQ4_XS
                 ("blk.14.ffn_gate.weight", "blk.14.ffn_up.weight"),   # IQ4_XS + Q3_K
                 ("blk.14.ffn_down.weight", "blk.3.ffn_down.weight")]: # IQ3_S + Q5_K（17408→5120。作る側の型の確かめ用で、モデルには無い組）
        if only and not re.search(only, g + "," + u):
            continue
        _, tg, k, n, _ = W.get(g)
        tu = W.get(u)[1]
        wg, wu = W.ref(g), W.ref(u)
        for m in (5, 64, 300):
            x = rng.standard_normal((m, k)).astype(np.float16)
            x.tofile(tmp / "x.f16")
            xg = x.astype(np.float64) @ wg.T
            ref = xg / (1 + np.exp(-xg)) * (x.astype(np.float64) @ wu.T)
            for cfg in ([-1] if m != 64 else [-1, 0, 2, 3, 4, 5]):
                kcheck("gguf", W.path, g, "swiglu", u, m, tmp / "x.f16", tmp / "y.f32", env=env_of(cfg))
                e, r = err(np.fromfile(tmp / "y.f32", dtype=np.float32).reshape(m, n), ref)
                good = e < TOL
                ok &= good
                line(good, f"swiglu {g}({tg}) {u}({tu}) m={m} cfg={cfg}", e, r)
        del wg, wu
    # 同じ X で何本か
    for names in ["blk.3.attn_q.weight,blk.3.attn_k.weight,blk.3.attn_v.weight",
                  "blk.0.attn_qkv.weight,blk.0.attn_gate.weight,blk.0.ssm_alpha.weight,blk.0.ssm_beta.weight",
                  "blk.14.ffn_down.weight,blk.3.ffn_down.weight"]:  # IQ3_S + Q5_K（型の確かめ用）
        if only and not re.search(only, names):
            continue
        ns = names.split(",")
        refs = [W.ref(nm) for nm in ns]
        k = refs[0].shape[1]
        for m in (7, 130):
            x = rng.standard_normal((m, k)).astype(np.float16)
            x.tofile(tmp / "x.f16")
            for cfg, yk in [(-1, None), (0, "f16"), (2, None)]:
                kcheck("gguf", W.path, names, "multi", m, tmp / "x.f16", tmp / "y.f32", env=env_of(cfg, yk))
                y = np.fromfile(tmp / "y.f32", dtype=np.float32)
                pos = 0
                for nm, rw in zip(ns, refs):
                    n = rw.shape[0]
                    e, r = err(y[pos:pos + m * n].reshape(m, n), x.astype(np.float64) @ rw.T)
                    pos += m * n
                    good = e < TOL
                    ok &= good
                    line(good, f"multi {nm}({W.get(nm)[1]}) n={n} m={m} cfg={cfg} y={yk or 'f32'}", e, r)
    return ok


def parse_ms(msg):
    m = re.search(r"([\d.]+) ms, ([\d.]+) TFLOPS", msg)
    return float(m.group(1)), float(m.group(2))


def xfile(k, m):
    """速さ用の X（logs/x{k}.f16 に 8192 行ぶん作っておく）"""
    p = ROOT / "logs" / f"x{k}.f16"
    if not p.exists() or p.stat().st_size < m * k * 2:
        np.random.default_rng(0).standard_normal((max(m, 8192), k)).astype(np.float16).tofile(p)
    return p


def timed(W, kind, names, m, cfg=-1, reps=None, tries=2):
    """(ms, TFLOPS)。測る値がぶれることがあるので、tries 回測っていちばん速いもの"""
    ns = names.split(",")
    _, _, k, n, _ = W.get(ns[0])
    flops = 2 * m * k * (sum(W.get(nm)[3] for nm in ns))
    reps = reps or max(3, min(100, int(3e11 / flops)))
    x = xfile(k, m)
    best = None
    for _ in range(tries):
        if kind == "gemm":
            msg = kcheck("gguf", W.path, ns[0], "gemm", m, x, "-", reps, env=env_of(cfg))
        elif kind == "swiglu":
            msg = kcheck("gguf", W.path, ns[0], "swiglu", ns[1], m, x, "-", reps, env=env_of(cfg))
        else:
            msg = kcheck("gguf", W.path, names, "multi", m, x, "-", reps, env=env_of(cfg, "f16"))
        r = parse_ms(msg)
        best = r if best is None or r[0] < best[0] else best
    return best


def cmd_bench(a):
    W = Weights(a.gguf)
    ms = [int(v) for v in a.m.split(",")] if a.m else MS
    print(f"{'形':14s} {'型':14s} {'k':>5s} {'n':>6s} " + " ".join(f"{'M=' + str(m):>16s}" for m in ms))
    for sname, kind, names in SHAPES:
        if a.shape and not re.search(a.shape, sname):
            continue
        ns = names.split(",")
        k = W.get(ns[0])[2]
        n = sum(W.get(nm)[3] for nm in ns)
        tys = "+".join(sorted(set(W.get(nm)[1] for nm in ns)))
        cells = []
        for m in ms:
            t, tf = timed(W, kind, names, m, a.cfg)
            cells.append(f"{tf:6.1f}T {t:7.3f}ms")
        print(f"{sname:14s} {tys:14s} {k:5d} {n:6d} " + " ".join(f"{c:>16s}" for c in cells), flush=True)
    return 0


def cmd_model(a):
    """64 層ぶん: 線形 Attention の層は (qkv+gate+alpha+beta) multi・ssm_out、普通の層は (q+k+v) multi・attn_output、
    全層で gate/up の SwiGLU・down。同じ (種類, 型の組) は 1 度だけ測って数を掛ける。出力（語彙）は GEMV で最後の 1 語だけ"""
    W = Weights(a.gguf)
    ms = [int(v) for v in a.m.split(",")] if a.m else [547, 8192]
    jobs = collections.Counter()
    rep = {}
    for i in range(64):
        b = f"blk.{i}."
        if b + "attn_qkv.weight" in W.t:
            items = [("multi", ",".join(b + s for s in ["attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight", "ssm_beta.weight"])),
                     ("gemm", b + "ssm_out.weight")]
        else:
            items = [("multi", ",".join(b + s for s in ["attn_q.weight", "attn_k.weight", "attn_v.weight"])),
                     ("gemm", b + "attn_output.weight")]
        items += [("swiglu", b + "ffn_gate.weight," + b + "ffn_up.weight"), ("gemm", b + "ffn_down.weight")]
        for kind, names in items:
            key = (kind, re.sub(r"blk\.\d+\.", "", names), tuple(W.get(nm)[1] for nm in names.split(",")))
            jobs[key] += 1
            rep.setdefault(key, names)
    supported = {"Q3_K", "IQ3_S", "Q4_K", "Q5_K", "Q6_K", "IQ4_XS", "IQ4_NL", "Q8_0", "F16"}
    for m in ms:
        total, flops_all, missing = 0.0, 0.0, []
        rows = []
        for key, cnt in sorted(jobs.items(), key=lambda kv: -kv[1]):
            kind, shape, tys = key
            names = rep[key]
            ns = names.split(",")
            fl = 2 * m * W.get(ns[0])[2] * sum(W.get(nm)[3] for nm in ns)  # swiglu は gate と up の 2 本ぶん
            if not set(tys) <= supported:
                missing.append((key, cnt, fl))
                continue
            t, tf = timed(W, kind, names, m, a.cfg, reps=3 if m >= 4096 else 10)
            total += t * cnt
            flops_all += fl * cnt
            rows.append(f"  {kind:6s} {shape:70s} {'+'.join(tys):22s} ×{cnt:2d} {t:8.3f} ms {tf:5.1f}T")
        # 戻しの無い型（IQ3_S）は、測れた分の平均の速さで見込む
        tf_avg = flops_all / (total * 1e-3)
        est = sum(fl * cnt for _, cnt, fl in missing) / tf_avg * 1e3
        print(f"M={m}:")
        print("\n".join(rows))
        for key, cnt, fl in missing:
            print(f"  {key[0]:6s} {key[1]:70s} {'+'.join(key[2]):22s} ×{cnt:2d} （型が無い・平均の速さで見込む）")
        print(f"  合計 {total + est:.1f} ms（うち見込み {est:.1f} ms）、平均 {tf_avg / 1e12:.1f} TFLOPS、"
              f"{m / ((total + est) * 1e-3):.0f} tok/s（GEMM だけ）", flush=True)
    # 出力（語彙 248320）: プレフィルでは最後の 1 語のロジットだけ要る → GEMV（kernels/gemv.cu の infer_gemv、B=1）
    k = W.get("output.weight")[2]
    xf = ROOT / "logs" / f"x{k}.f32"
    if not xf.exists():
        np.random.default_rng(0).standard_normal(k).astype(np.float32).tofile(xf)
    best = None
    for _ in range(3):
        msg = kcheck("gguf", W.path, "output.weight", "gemv", 0, "all", xf, 1, ROOT / "logs" / "lmhead.f32")
        t = float(re.search(r"([\d.]+) ms", msg).group(1))
        best = t if best is None else min(best, t)
    print(f"出力 output.weight {W.get('output.weight')[1]} [{k}→{W.get('output.weight')[3]}] 最後の 1 語の GEMV: {best:.3f} ms")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["check", "bench", "model"])
    p.add_argument("gguf", nargs="?", default=GGUF)
    p.add_argument("--cfg", type=int, default=-1)
    p.add_argument("--m", default="")
    p.add_argument("--shape", default="")
    p.add_argument("--only", default="", help="check: 名前がこの正規表現に合う行列・組だけ（例 'blk\\.1[34]\\.'）")
    a = p.parse_args()
    (ROOT / "logs").mkdir(exist_ok=True)
    sys.exit({"check": cmd_check, "bench": cmd_bench, "model": cmd_model}[a.cmd](a))


if __name__ == "__main__":
    main()
