#!/usr/bin/env python3
"""MTP の下書き用に語彙を絞る（FR-Spec と同じ考え: よく出る語だけで lm_head を掛ける）。語の数を数えて上位 N 個の番号を書く。

  python3 tools/draft_vocab.py --tokenizer <tokenizer.json> --out data/draft_vocab_q38_32k.txt [--n 32768]

数える文（種類ごとに同じ重みで足す。golden の文（golden/m1/src）と試しの台本は使わない）:
  ja: --ja-glob の .md（日本語の文書。data/draft_vocab_q38.txt は手元の日本語の文書で数えた）
  py: Python の標準ライブラリ（/usr/lib/python3.*。英語の docstring を含む）、rs: ~/.cargo/registry の .rs、c: /usr/include と CUDA の頭書き、en: ~/.cargo/registry の .md と /usr/share/doc の README・.txt（英語の文）
added token（248044 以降）と 1 バイトの語（256 個）は必ず入れる。出力は番号を 1 行に 1 つ（小さい順）
"""
import argparse
import collections
import glob
import os

from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=32768)
ap.add_argument("--mb", type=float, default=15.0, help="種類ごとの文の上限（MB）")
ap.add_argument("--ja-glob", default="docs/**/*.md", help="日本語の文（glob・recursive）")
a = ap.parse_args()

tok = Tokenizer.from_file(a.tokenizer)
H = os.path.expanduser("~")
cats = {
    "ja": sorted(glob.glob(a.ja_glob, recursive=True)) if a.ja_glob else [],
    "py": sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True)),
    "rs": sorted(glob.glob(f"{H}/.cargo/registry/src/**/*.rs", recursive=True)),
    "c": sorted(glob.glob("/usr/include/**/*.h", recursive=True)) + sorted(glob.glob("/usr/local/cuda*/include/**/*.h*", recursive=True)),
    "en": sorted(glob.glob(f"{H}/.cargo/registry/src/**/*.md", recursive=True)) + sorted(glob.glob("/usr/share/doc/**/README*", recursive=True))
    + sorted(glob.glob("/usr/share/doc/**/*.txt", recursive=True)),
}
total = collections.Counter()
for name, files in cats.items():
    c = collections.Counter()
    used = 0
    for f in files:
        if used > a.mb * 1e6:
            break
        if f.endswith(".gz") or not os.path.isfile(f):
            continue
        try:
            t = open(f, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        if not t.strip():
            continue
        used += len(t.encode())
        for i in range(0, len(t), 200_000):
            c.update(tok.encode(t[i:i + 200_000], add_special_tokens=False).ids)
    n = sum(c.values())
    print(f"{name}: {used / 1e6:.1f} MB・{n} 語・異なる語 {len(c)}")
    for k, v in c.items():
        total[k] += v / n
vocab = tok.get_vocab_size()
keep = set(i for i in range(248044, vocab))
# 1 バイトの語（byte-level BPE の 256 個の文字）
for i in range(vocab):
    s = tok.id_to_token(i)
    if s is not None and len(s) == 1:
        keep.add(i)
for k, _ in total.most_common():
    if len(keep) >= a.n:
        break
    keep.add(k)
ids = sorted(keep)
cov = sum(total[i] for i in ids) / sum(total.values())
print(f"{len(ids)} 語・数えた語の {100 * cov:.2f}% を覆う")
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, "w") as f:
    f.write("\n".join(map(str, ids)) + "\n")
