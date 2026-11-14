#!/usr/bin/env python3
"""3.6（qwen35moe）の MoE の FFN を numpy で計算して、tachyon layers の .npz と層ごとに突き合わせる。

  tachyon layers --gguf <3.6> --prompt-tokens <json> --npz o.npz --layers 4 --detail 0,1,2,3 [--chunk 64]
  python tools/moe_ref.py <3.6.gguf> o.npz [層,...]

入口は .npz の detail.attn_post_norm-<il>（b = post_norm(h)）、比べる相手は h[il] − detail.attn_residual-<il>（FFN が足した分）。
式は llama.cpp の build_moe_ffn（softmax・上位 k・重みを Σ で割る）＋共有の専門家（sigmoid のゲート）。重みは gguf の dequantize（f32）で、
x は量子化しない（f32 の手本）。専門家の選び方も比べる（1 語ごとの上位 k の集合の一致）。
"""
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


def main():
    gpath, npz = sys.argv[1], sys.argv[2]
    z = np.load(npz)
    r = GGUFReader(gpath)
    T = {t.name: t for t in r.tensors}
    kk = int(r.fields['qwen35moe.expert_used_count'].parts[-1][0])
    layers = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else None
    names = [k for k in z.files if k.startswith('detail.attn_post_norm-')]
    ils = sorted(int(k.rsplit('-', 1)[1]) for k in names)
    if layers:
        ils = [i for i in ils if i in layers]
    layer0 = int(z['layer0'][0])

    def deq(t):
        return dequantize(t.data, t.tensor_type).astype(np.float32)

    ok = True
    for il in ils:
        b = z[f'detail.attn_post_norm-{il}'].astype(np.float64)        # [n][d]
        res = z[f'detail.attn_residual-{il}'].astype(np.float64)
        got = z['h'][il - layer0].astype(np.float64) - res
        n, d = b.shape
        p = f'blk.{il}.'
        wr = deq(T[p + 'ffn_gate_inp.weight']).reshape(-1, d).astype(np.float64)       # [ne][d]
        ws = deq(T[p + 'ffn_gate_inp_shexp.weight']).reshape(d).astype(np.float64)
        lg = b @ wr.T
        pr = np.exp(lg - lg.max(1, keepdims=True))
        pr /= pr.sum(1, keepdims=True)
        sel = np.argsort(-pr, axis=1, kind='stable')[:, :kk]
        w = np.take_along_axis(pr, sel, 1)
        w /= np.maximum(w.sum(1, keepdims=True), 6.103515625e-5)
        tg, tu, td = T[p + 'ffn_gate_exps.weight'], T[p + 'ffn_up_exps.weight'], T[p + 'ffn_down_exps.weight']
        cache = {}

        def ex(e):
            if e not in cache:
                g = dequantize(tg.data[e], tg.tensor_type).astype(np.float64)   # [ff][d]
                u = dequantize(tu.data[e], tu.tensor_type).astype(np.float64)
                dn = dequantize(td.data[e], td.tensor_type).astype(np.float64)  # [d][ff]
                cache[e] = (g, u, dn)
            return cache[e]

        out = np.zeros((n, d))
        for t in range(n):
            for j in range(kk):
                g, u, dn = ex(int(sel[t, j]))
                a = g @ b[t]
                h = a / (1 + np.exp(-a)) * (u @ b[t])
                out[t] += w[t, j] * (dn @ h)
        sg, su, sd = (deq(T[p + x]).astype(np.float64) for x in ('ffn_gate_shexp.weight', 'ffn_up_shexp.weight', 'ffn_down_shexp.weight'))
        a = b @ sg.T
        hs = a / (1 + np.exp(-a)) * (b @ su.T)
        out += (hs @ sd.T) / (1 + np.exp(-(b @ ws)))[:, None]
        rel = np.linalg.norm(got - out) / np.linalg.norm(out)
        cos = (got * out).sum() / np.linalg.norm(got) / np.linalg.norm(out)
        rows = np.linalg.norm(got - out, axis=1) / np.linalg.norm(out, axis=1)
        # 専門家の選び: 1 位〜k 位の確率の差がごく小さい所は入れ替わってよい
        gap = np.sort(pr, 1)[:, ::-1]
        near = (gap[:, kk - 1] - gap[:, kk]) < 1e-4
        print(f'層 {il:2d}: {n} 語  FFN の相対誤差 {rel:.2e}（行の最大 {rows.max():.2e}・位置 {rows.argmax()}）  cos {cos:.7f}  '
              f'k 位と k+1 位が僅差の語 {near.sum()}')
        if rel > 5e-3 and rows.max() > 2e-2:
            ok = False
    print('OK' if ok else 'NG')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
