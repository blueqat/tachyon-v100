"""g3（kernels/gemv3.cu）の答え合わせと速さ。手本は gguf-py の dequantize と numpy の float64。
Rust の target/release/g3check が GGUF から名前で重みを取り、GPU で詰め直して掛ける。

  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/g3check.py deq      … 並びの確かめ（戻した値が一致するか）
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/g3check.py gemv     … 掛け算の答え合わせ（B = 1..32）と速さ
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/g3check.py swiglu   … SiLU(gate)*up をまとめて 1 回で
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/g3check.py multi    … 同じ x の行列（qkv・gate・alpha・beta など）を 1 回で
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/g3check.py slices   … 半端な形 × wk の決め打ち × 足し込み（k の切れ方の確かめ）
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/g3check.py all      … 上の 5 つ（と、GPU を使わない起動の形の確かめ）
--quick で数を減らす。重みは G3_GGUF（既定 weights/q38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf）
足し込みは既定で「決まった順」（INFER_G3_DET=1）: 4 回くり返した答えがビットまで同じことも確かめる。
INFER_G3_DET=0 で atomicAdd の道（答えの誤差だけ見る）
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as T
from gguf import GGUFReader
from gguf.quants import dequantize

ROOT = Path(__file__).resolve().parent.parent
BIN = os.environ.get("G3_BIN", str(ROOT / "target/release/g3check"))
GGUF = os.environ.get("G3_GGUF", os.path.join(os.environ.get("TACHYON_WEIGHTS", "weights"), "q38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf"))
# 型 → (1 ブロックの要素数, バイト数)
TYPE_SIZE = {"Q8_0": (32, 34), "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210), "IQ4_NL": (32, 18),
             "IQ4_XS": (256, 136), "Q3_K": (256, 110), "IQ3_S": (256, 110)}

# 確かめるテンソル（全部の型・形を通す）
CASES = [
    "blk.3.ffn_gate.weight",      # Q4_K  5120→17408
    "blk.3.ffn_up.weight",        # Q5_K  5120→17408
    "blk.3.ffn_down.weight",      # Q5_K  17408→5120
    "blk.0.ffn_gate.weight",      # IQ4_XS
    "blk.0.ffn_down.weight",      # IQ4_XS 17408→5120
    "blk.1.ffn_down.weight",      # IQ4_NL 17408→5120
    "blk.0.ffn_up.weight",        # Q3_K（Q6_K に直す）
    "blk.14.ffn_down.weight",     # IQ3_S（Q6_K に直す）
    "blk.64.ffn_gate.weight",     # Q6_K  5120→17408
    "blk.3.attn_output.weight",   # Q6_K  6144→5120
    "blk.0.ssm_out.weight",       # Q8_0  6144→5120
    "blk.0.ssm_alpha.weight",     # Q8_0  5120→48（32 の倍数でない行数）
    "blk.0.attn_qkv.weight",      # Q5_K  5120→10240
    "blk.3.attn_k.weight",        # Q4_K  5120→1024
    "output.weight",              # Q6_K  5120→248320
]
# 半端な形（行列の左上 [n × k]）。k の単位の数が k の分け方で割り切れず、ブロックの k が 1 単位短くなる所。
# 2026-11-10 の見直しで、wk ≥ 2 と足し込みで黙って違う答え（k=3840 n=1024 B=12・16・17・32）、
# wk を形から選んで起動の誤り（k=768 n=100 B=31・32、k=3840 n=1000 B=7〜32。共有メモリが 96KB を超える）が出た
SLICES = [
    ("blk.3.ffn_gate.weight", 3840, 1024),      # Q4_K
    ("blk.0.ssm_out.weight", 3840, 1024),       # Q8_0
    ("blk.3.attn_output.weight", 3840, 1024),   # Q6_K
    ("blk.0.ffn_gate.weight", 3840, 1024),      # IQ4_XS
    ("blk.1.ffn_down.weight", 3840, 1000),      # IQ4_NL（n が 32 で割り切れない）
    ("blk.0.ffn_up.weight", 768, 100),          # Q3_K → Q6_K
    ("blk.3.ffn_down.weight", 5376, 1000),      # Q5_K（21 単位）
    ("blk.0.ssm_alpha.weight", 5120, 48),       # Q8_0 本物の形: 前は B=10・11 で wk=8 の切れが 96KB を超えて起動の誤り
    ("blk.3.attn_k.weight", 5120, 1024),        # Q4_K 本物の形（同じく B=10・11）
]
SLICE_BS = [7, 10, 11, 12, 16, 17, 20, 31, 32]
DET = os.environ.get("INFER_G3_DET", "1") not in ("", "0")

QUICK = ["blk.3.ffn_gate.weight", "blk.3.ffn_down.weight", "blk.0.ffn_gate.weight", "blk.1.ffn_down.weight",
         "blk.0.ffn_up.weight", "blk.14.ffn_down.weight", "blk.3.attn_output.weight", "blk.0.ssm_alpha.weight"]


def run(*args, env=None):
    r = subprocess.run([BIN, *map(str, args)], capture_output=True, text=True, env=dict(os.environ, **(env or {})))
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    return r.stdout.strip()


def det_ok(line):
    """決まった順なら、足し込みを 4 回くり返した答えがビットまで同じ（g3check gemv の「4 回同じ」）"""
    return not DET or "4 回同じ" in line


_tensors = None


def tensors():
    """名前 → (位置, 型名, k, n)"""
    global _tensors
    if _tensors is None:
        _tensors = {}
        r = GGUFReader(GGUF)
        for t in r.tensors:
            sh = [int(s) for s in t.shape]
            _tensors[t.name] = (int(t.data_offset), t.tensor_type.name, sh[0], int(np.prod(sh[1:])))
    return _tensors


def ref_rows(name, r0, r1):
    off, typ, k, n = tensors()[name]
    blk, bs = TYPE_SIZE[typ]
    rb = k // blk * bs
    mm = np.memmap(GGUF, dtype=np.uint8, mode="r", offset=off + r0 * rb, shape=((r1 - r0) * rb,))
    return dequantize(np.array(mm), T[typ]).reshape(r1 - r0, k).astype(np.float64)


def windows(n):
    w = [(0, min(n, 96))]
    if n > 256:
        w += [(n // 2 - 48, n // 2 + 48), (n - 96, n)]
    return w


def errs(y, yr):
    """(最大の誤差 / 最大の値, 2 ノルムの相対誤差)"""
    return np.abs(y - yr).max() / (np.abs(yr).max() + 1e-30), np.linalg.norm(y - yr) / (np.linalg.norm(yr) + 1e-30)


def cmd_deq(names):
    ok = True
    with tempfile.TemporaryDirectory(dir=ROOT) as d:
        tmp = Path(d)
        for name in names:
            off, typ, k, n = tensors()[name]
            rows = min(n, 80)  # 80 行 = 2.5 タイル（切り上げも確かめる）
            run("deq", GGUF, name, rows, tmp / "d.f32")
            got = np.fromfile(tmp / "d.f32", dtype=np.float32).reshape(rows, k)
            ref = ref_rows(name, 0, rows)
            e = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30)
            good = e < 1e-6
            ok &= good
            print(f"{'OK' if good else 'NG'} {name:28s} {typ:6s} k={k} 戻し {e:.1e}", flush=True)
    return ok


def cmd_gemv(names, Bs):
    rng = np.random.default_rng(0)
    ok = True
    with tempfile.TemporaryDirectory(dir=ROOT) as d:
        tmp = Path(d)
        for name in names:
            off, typ, k, n = tensors()[name]
            refs = [(r0, r1, ref_rows(name, r0, r1)) for r0, r1 in windows(n)]
            for B in Bs:
                x = rng.standard_normal((B, k)).astype(np.float32)
                x.tofile(tmp / "x.f32")
                line = run("gemv", GGUF, name, tmp / "x.f32", B, tmp / "y.f32")
                y = np.fromfile(tmp / "y.f32", dtype=np.float32).reshape(B, n)
                yr = np.concatenate([x.astype(np.float64) @ w.T for _, _, w in refs], axis=1)
                yg = np.concatenate([y[:, r0:r1] for r0, r1, _ in refs], axis=1)
                em, en = errs(yg, yr)
                ya = np.fromfile(tmp / "y.f32.add", dtype=np.float32).reshape(B, n)
                ema, ena = errs(np.concatenate([ya[:, r0:r1] for r0, r1, _ in refs], axis=1), yr)
                em, en = max(em, ema), max(en, ena)
                eo = ""
                if (tmp / "y.f32.old").exists():
                    yo = np.fromfile(tmp / "y.f32.old", dtype=np.float32).reshape(B, n)
                    eo = f" 2 版 {errs(np.concatenate([yo[:, r0:r1] for r0, r1, _ in refs], axis=1), yr)[0]:.1e}"
                    (tmp / "y.f32.old").unlink()
                good = em < 1e-4 and en < 1e-4 and det_ok(line)
                ok &= good
                print(f"{'OK' if good else 'NG'} 誤差 {em:.1e}（ノルム {en:.1e}、足し込みも）{eo} | {line}", flush=True)
    return ok


def cmd_slices(cases, Bs, wks=(0, 2, 4, 8)):
    """半端な形（行列の左上 [n × k]）× B × wk の決め打ち。答え（足し込みも）と、決まった順なら 4 回同じか"""
    rng = np.random.default_rng(3)
    ok = True
    env = {"G3CHECK_NOBENCH": "1"}
    with tempfile.TemporaryDirectory(dir=ROOT) as d:
        tmp = Path(d)
        for name, k, n in cases:
            w = ref_rows(name, 0, n)[:, :k]
            for B in Bs:
                x = rng.standard_normal((B, k)).astype(np.float32)
                x.tofile(tmp / "x.f32")
                yr = x.astype(np.float64) @ w.T
                for wk in wks:
                    if wk > k // 256 or wk * (1 if B <= 16 else 2) > 8:
                        continue  # 使えない wk（-4 で返る）
                    try:
                        line = run("gemv", GGUF, name, tmp / "x.f32", B, tmp / "y.f32", wk, k, n, env=env)
                    except RuntimeError as e:
                        ok = False
                        print(f"NG {name} k={k} n={n} B={B} wk={wk}: 起動の誤り {str(e).strip().splitlines()[-1][:200]}", flush=True)
                        continue
                    y = np.fromfile(tmp / "y.f32", dtype=np.float32).reshape(B, n)
                    ya = np.fromfile(tmp / "y.f32.add", dtype=np.float32).reshape(B, n)
                    em, en = errs(y, yr)
                    ema, ena = errs(ya, yr)
                    good = max(em, ema) < 1e-4 and max(en, ena) < 1e-4 and det_ok(line)
                    ok &= good
                    print(f"{'OK' if good else 'NG'} 誤差 {em:.1e}・足し込み {ema:.1e}（ノルム {max(en, ena):.1e}）| {line.split(': ', 1)[-1]}"
                          f" | {name} k={k} n={n} B={B} wk={wk}", flush=True)
    return ok


def cmd_plan():
    """起動の形（k の切れ方）を全部の形で（GPU は使わない）"""
    r = subprocess.run([BIN, "plan"], capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    print(r.stdout.strip(), r.stderr.strip()[-2000:], flush=True)
    return r.returncode == 0


def cmd_swiglu(pairs, Bs):
    rng = np.random.default_rng(1)
    ok = True
    with tempfile.TemporaryDirectory(dir=ROOT) as d:
        tmp = Path(d)
        for layer in pairs:
            g, u = f"blk.{layer}.ffn_gate.weight", f"blk.{layer}.ffn_up.weight"
            _, tg, k, n = tensors()[g]
            tu = tensors()[u][1]
            ws = windows(n)
            rg = [ref_rows(g, r0, r1) for r0, r1 in ws]
            ru = [ref_rows(u, r0, r1) for r0, r1 in ws]
            for B in Bs:
                x = rng.standard_normal((B, k)).astype(np.float32)
                x.tofile(tmp / "x.f32")
                line = run("swiglu", GGUF, g, u, tmp / "x.f32", B, tmp / "h.f32")
                h = np.fromfile(tmp / "h.f32", dtype=np.float32).reshape(B, n)
                x64 = x.astype(np.float64)
                hr = []
                for a, b in zip(rg, ru):
                    gg, uu = x64 @ a.T, x64 @ b.T
                    hr.append(gg / (1 + np.exp(-gg)) * uu)
                hr = np.concatenate(hr, axis=1)
                hg = np.concatenate([h[:, r0:r1] for r0, r1 in ws], axis=1)
                em, en = errs(hg, hr)
                # down に渡す下ごしらえ（hi + lo）× 倍率から戻した h も同じか
                hq = np.fromfile(tmp / "h.f32.hp", dtype=np.float32).reshape(B, n)
                emq, enq = errs(np.concatenate([hq[:, r0:r1] for r0, r1 in ws], axis=1), hr)
                em, en = max(em, emq), max(en, enq)
                good = em < 1e-4 and en < 1e-4
                ok &= good
                print(f"{'OK' if good else 'NG'} 層 {layer} {tg}+{tu} 誤差 {em:.1e}（ノルム {en:.1e}、下ごしらえも）| {line}", flush=True)
    return ok


def cmd_multi(groups, Bs):
    """同じ x の行列をまとめて 1 回で（線形の層の qkv・gate・alpha・beta、普通の層の q・k・v）"""
    rng = np.random.default_rng(2)
    ok = True
    with tempfile.TemporaryDirectory(dir=ROOT) as d:
        tmp = Path(d)
        for names in groups:
            k = tensors()[names[0]][2]
            refs = []
            for name in names:
                n = tensors()[name][3]
                refs.append([(r0, r1, ref_rows(name, r0, r1)) for r0, r1 in windows(n)])
            for B in Bs:
                x = rng.standard_normal((B, k)).astype(np.float32)
                x.tofile(tmp / "x.f32")
                line = run("multi", GGUF, tmp / "x.f32", B, tmp / "y", *names)
                em = en = 0.0
                for i, (name, rf) in enumerate(zip(names, refs)):
                    n = tensors()[name][3]
                    y = np.fromfile(tmp / f"y.{i}", dtype=np.float32).reshape(B, n)
                    yr = np.concatenate([x.astype(np.float64) @ w.T for _, _, w in rf], axis=1)
                    yg = np.concatenate([y[:, r0:r1] for r0, r1, _ in rf], axis=1)
                    a, b = errs(yg, yr)
                    em, en = max(em, a), max(en, b)
                good = em < 1e-4 and en < 1e-4
                ok &= good
                print(f"{'OK' if good else 'NG'} {'+'.join(n.split('.')[2] for n in names)} 誤差 {em:.1e}（ノルム {en:.1e}）| {line}", flush=True)
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    quick = "--quick" in sys.argv
    names = QUICK if quick else CASES
    Bs = [1, 3, 8, 32] if quick else [1, 2, 3, 4, 5, 8, 12, 16, 24, 32]
    pairs = [0, 3, 13] if quick else [0, 1, 3, 13, 14, 17, 28, 59]
    ok = True
    if cmd in ("deq", "all"):
        ok &= cmd_deq(names)
    if cmd in ("gemv", "all"):
        ok &= cmd_gemv(names, Bs)
    if cmd in ("swiglu", "all"):
        ok &= cmd_swiglu(pairs, Bs)
    if cmd in ("multi", "all"):
        groups = [["blk.%d.%s.weight" % (l, t) for t in ("attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta")] for l in (0, 21)]
        groups.append(["blk.3.%s.weight" % t for t in ("attn_q", "attn_k", "attn_v")])
        ok &= cmd_multi(groups[:2] if quick else groups, Bs)
    if cmd in ("slices", "all"):
        ok &= cmd_plan()
        ok &= cmd_slices(SLICES[:3] + SLICES[-1:] if quick else SLICES, [10, 12, 17, 32] if quick else SLICE_BS)
    if cmd not in ("deq", "gemv", "swiglu", "multi", "slices", "all"):
        print(__doc__)
        sys.exit(2)
    print("全部 OK" if ok else "⛔ NG あり")
    sys.exit(0 if ok else 1)
