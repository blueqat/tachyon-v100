"""自前のカーネル（target/release/kcheck）の答え合わせ。手本は gguf-py の dequantize と numpy。

  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/kcheck.py --gguf <GGUF（分割なら最初の束）> [テンソル名の正規表現]
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/kcheck.py <目次.json> <重みの置き場> [テンソル名の正規表現]   … 取り出した .bin で
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/kcheck.py --rand Q3_K,IQ3_S   … でたらめなバイト（全部のビットの組を通す）
  --rows N|all を前に付けると確かめる行数を変える（既定 64。all でテンソル全部）
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as T
from gguf.quants import dequantize

KCHECK = str(Path(__file__).resolve().parent.parent / "target/release/kcheck")
ROWS = 64  # 1 つのテンソルで確かめる行数
# 自前のカーネルが知っている型（kernels/gemv.cu の FOR_TYPES）
KERNEL_TYPES = {"F32", "F16", "BF16", "Q8_0", "Q5_1", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_NL", "IQ4_XS", "IQ3_XXS", "IQ3_S", "IQ2_S", "Q2_0"}


def run(*args):
    r = subprocess.run([KCHECK, *map(str, args)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    return r


class BinSrc:
    """取り出した .bin（tools/gguf_remote.py --fetch）"""

    def __init__(self, idx, wdir, tmp):
        self.idx, self.wdir, self.wb = json.loads(Path(idx).read_text()), Path(wdir), tmp / "w.bin"

    def tensors(self):
        return [t for t in self.idx["tensors"] if (self.wdir / (t["name"] + ".bin")).exists()]

    def raw(self, t, rows):
        n_all = int(np.prod(t["shape"][1:])) if len(t["shape"]) > 1 else 1
        raw = np.fromfile(self.wdir / (t["name"] + ".bin"), dtype=np.uint8)
        part = raw[: rows * (raw.size // n_all)]
        part.tofile(self.wb)
        return part

    def dequant(self, t, rows, out):
        return run("dequant", self.wb, t["type"], t["shape"][0], rows, out)

    def gemv(self, t, rows, x, B, out):
        return run("gemv", self.wb, t["type"], t["shape"][0], rows, x, B, out)


# でたらめなブロックの大きさと fp16 の係数の位置（NaN・inf を避けて有限の値に置き換える）
RAND_BLOCK = {"Q3_K": (256, 110, [108]), "IQ3_S": (256, 110, [0])}


class RandSrc:
    """でたらめなバイト（kcheck dequant / gemv に .bin で渡す）"""

    def __init__(self, types, tmp, rng, k=4096):
        self.types, self.wb, self.rng, self.k = types, tmp / "w.bin", rng, k

    def tensors(self):
        return [{"name": f"rand.{t}", "type": t, "shape": [self.k, 256]} for t in self.types]

    def raw(self, t, rows):
        qk, bs, halves = RAND_BLOCK[t["type"]]
        nb = rows * self.k // qk
        b = self.rng.integers(0, 256, (nb, bs), dtype=np.uint8)
        for o in halves:
            b[:, o : o + 2] = self.rng.uniform(-1, 1, nb).astype(np.float16).view(np.uint8).reshape(nb, 2)
        b.reshape(-1).tofile(self.wb)
        return b.reshape(-1)

    def dequant(self, t, rows, out):
        return run("dequant", self.wb, t["type"], self.k, rows, out)

    def gemv(self, t, rows, x, B, out):
        return run("gemv", self.wb, t["type"], self.k, rows, x, B, out)


class GgufSrc:
    """GGUF から名前でじかに（kcheck gguf …）"""

    def __init__(self, path, tmp):
        self.path, self.wb = path, tmp / "w.bin"

    def tensors(self):
        out = []
        for line in run("gguf", self.path).stdout.splitlines():
            name, typ, shape = line.split("\t")
            out.append({"name": name, "type": typ, "shape": [int(d) for d in shape.split(",")]})
        return out

    def raw(self, t, rows):
        run("gguf", self.path, t["name"], "raw", 0, rows, self.wb)
        return np.fromfile(self.wb, dtype=np.uint8)

    def dequant(self, t, rows, out):
        return run("gguf", self.path, t["name"], "dequant", 0, rows, out)

    def gemv(self, t, rows, x, B, out):
        return run("gguf", self.path, t["name"], "gemv", 0, rows, x, B, out)


def check(t, src, tmp, rng, max_rows=ROWS):
    k = t["shape"][0]
    n_all = int(np.prod(t["shape"][1:])) if len(t["shape"]) > 1 else 1
    rows = n_all if max_rows is None else min(max_rows, n_all)
    part = src.raw(t, rows)
    ref = dequantize(part, T[t["type"]]).reshape(rows, k).astype(np.float64)
    # 1) 戻すだけ
    src.dequant(t, rows, tmp / "d.f32")
    got = np.fromfile(tmp / "d.f32", dtype=np.float32).reshape(rows, k)
    e_deq = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12)
    # 2) 掛け算（4 本まとめて）
    B = 4
    x = rng.standard_normal((B, k)).astype(np.float32)
    x.tofile(tmp / "x.f32")
    src.gemv(t, rows, tmp / "x.f32", B, tmp / "y.f32")
    y = np.fromfile(tmp / "y.f32", dtype=np.float32).reshape(B, rows)
    yref = x.astype(np.float64) @ ref.T
    e_mv = np.abs(y - yref).max() / (np.abs(yref).max() + 1e-12)
    ok = e_deq < 1e-6 and e_mv < 1e-4
    # 戻しがビットまで同じか（手本は f32 で計算している）
    exact = "完全一致" if np.array_equal(got.view(np.uint32), ref.astype(np.float32).view(np.uint32)) else "ずれあり"
    print(f"{'OK ' if ok else 'NG '} {t['name']:40s} {t['type']:8s} k={k:5d} 行 {rows:5d} 戻し {e_deq:.1e}（{exact}） 掛け {e_mv:.1e}", flush=True)
    return ok


def main():
    a = sys.argv[1:]
    rng = np.random.default_rng(0)
    seen, ok = set(), True
    max_rows = ROWS
    if a[0] == "--rows":
        max_rows, a = (None if a[1] == "all" else int(a[1])), a[2:]
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        if a[0] == "--rand":
            src, pat = RandSrc(a[1].split(","), tmp, rng), "."
        elif a[0] == "--gguf":
            src, pat = GgufSrc(a[1], tmp), a[2] if len(a) > 2 else "."
        else:
            src, pat = BinSrc(a[0], a[1], tmp), a[2] if len(a) > 2 else "."
        pat = re.compile(pat)
        for t in src.tensors():
            if not pat.search(t["name"]):
                continue
            if t["shape"][0] % 32:  # ssm_a などの短いベクトルは掛け算に使わない
                continue
            key = (t["type"], re.sub(r"blk\.\d+\.", "", t["name"]))
            if key in seen:
                continue
            seen.add(key)
            if t["type"] not in KERNEL_TYPES or t["type"] not in T.__members__:
                print(f"--  {t['name']:40s} {t['type']:8s} 手本かカーネルが無いので飛ばす")
                continue
            ok &= check(t, src, tmp, rng, max_rows)
    print("全部 OK" if ok else "⛔ NG あり")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
