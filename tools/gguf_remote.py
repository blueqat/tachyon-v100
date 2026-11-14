"""GGUF の目次（メタデータとテンソルの一覧）を HTTP Range で読む。重みの本体は落とさない。

使い方:
  .venv/bin/python tools/gguf_remote.py <URL> [--out 目次.json]
  .venv/bin/python tools/gguf_remote.py <URL> --fetch <テンソル名の正規表現> --dest weights/
"""
import argparse
import json
import re
import struct
import sys
from pathlib import Path

import requests

GGUF_TYPES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
# ggml の型 → (1 ブロックの要素数, 1 ブロックのバイト数)
GGML = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
    39: ("MXFP4", 32, 17), 41: ("Q1_0", 128, 18), 42: ("Q2_0", 64, 18), 43: ("F8_E4M3", 1, 1),
}


class RangeReader:
    """必要な分だけ Range で先読みしながら順に読む。"""

    def __init__(self, url, chunk=8 << 20):
        self.url, self.chunk, self.buf, self.base, self.pos = url, chunk, b"", 0, 0
        self.s = requests.Session()

    def _ensure(self, n):
        while self.pos + n > self.base + len(self.buf):
            start = self.base + len(self.buf)
            r = self.s.get(self.url, headers={"Range": f"bytes={start}-{start + self.chunk - 1}"}, timeout=120)
            r.raise_for_status()
            if not r.content:
                raise EOFError(self.url)
            self.buf += r.content

    def read(self, n):
        self._ensure(n)
        off = self.pos - self.base
        self.pos += n
        return self.buf[off:off + n]

    def unpack(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self):
        return self.read(self.unpack("Q")).decode("utf-8", "replace")


def read_value(r, t):
    if t == 8:
        return r.string()
    if t == 9:
        et, n = r.unpack("I"), r.unpack("Q")
        vals = [read_value(r, et) for _ in range(n)]
        # 語彙のような長い配列は、長さと先頭だけ残す
        return vals if n <= 64 else {"len": n, "head": vals[:8]}
    return r.unpack(GGUF_TYPES[t])


def read_index(url):
    r = RangeReader(url)
    if r.read(4) != b"GGUF":
        raise ValueError("GGUF ではない")
    version, n_tensors, n_kv = r.unpack("I"), r.unpack("Q"), r.unpack("Q")
    meta = {}
    for _ in range(n_kv):
        k = r.string()
        meta[k] = read_value(r, r.unpack("I"))
    align = meta.get("general.alignment", 32)
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.unpack("I")
        shape = [r.unpack("Q") for _ in range(nd)]
        typ, off = r.unpack("I"), r.unpack("Q")
        tname, blk, bsz = GGML.get(typ, (f"type{typ}", 1, 0))
        n = 1
        for d in shape:
            n *= d
        tensors.append({"name": name, "shape": shape, "type": tname, "offset": off, "bytes": n // blk * bsz})
    data_start = (r.pos + align - 1) // align * align
    return {"url": url, "version": version, "data_start": data_start, "meta": meta, "tensors": tensors}


def fetch(idx, pattern, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    total = 0
    for t in idx["tensors"]:
        if not re.search(pattern, t["name"]):
            continue
        a = idx["data_start"] + t["offset"]
        out = dest / (t["name"] + ".bin")
        if out.exists() and out.stat().st_size == t["bytes"]:
            continue
        with s.get(idx["url"], headers={"Range": f"bytes={a}-{a + t['bytes'] - 1}"}, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(out, "wb") as f:
                for c in r.iter_content(8 << 20):
                    f.write(c)
        total += t["bytes"]
        print(f"{t['name']} {t['type']} {t['shape']} {t['bytes'] / 1e6:.1f}MB", file=sys.stderr)
    print(f"合計 {total / 1e9:.2f}GB", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--out")
    ap.add_argument("--fetch")
    ap.add_argument("--dest", default="weights")
    a = ap.parse_args()
    idx = read_index(a.url)
    if a.out:
        Path(a.out).write_text(json.dumps(idx, ensure_ascii=False, indent=1))
    if a.fetch:
        fetch(idx, a.fetch, a.dest)
    if not a.out and not a.fetch:
        json.dump(idx, sys.stdout, ensure_ascii=False, indent=1)
