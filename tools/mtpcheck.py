#!/usr/bin/env python3
"""MTP の GPU（target/release/mtpcheck、src/mtp.rs）と CPU 参照（tools/ref_mtp.py run の出力）の答え合わせ。

  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/mtpcheck.py [--out logs/mtp] [--src main,file] [--k 1,2,3]

列 p<i> ごと・重み（main・file）ごとに:
  teacher  全位置の MTP の行を 32 行ずつ（KV を埋めながら）。h'（rel L2）、上位 20 の logits（同じ語の差 / 参照の最大）、lse、
           argmax の一致（外れは参照の 1 位と 2 位の差が NEAR 未満の所だけ許す）、最後の 2 行の logits 全部（rel L2）
  sim      温度 0 の下書きの回し方（spec::start・step と mtp::MtpHead。本体は参照の記録を再生）。歩の列
           （pos0・id_last・drafts・n_acc・new）が参照と全部同じか。下書きの行の h'・上位 20・lse、頭 4 行の logits 全部
合格の線: rel L2 < TOL（h'・logits）、argmax の外れは僅差だけ、sim の歩は全部一致（僅差で分かれたら、そこまでを比べて「僅差」と出す）
終わりに「全部 OK」。NG があれば終了コード 1
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
BIN = os.path.join(ROOT, 'target', 'release', 'mtpcheck')
TOL = 5e-3
NEAR = 0.05
TOPK = 20


def rel(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def rd(path, dt, shape=None):
    a = np.fromfile(path, dt)
    return a.reshape(shape) if shape is not None else a


def run(*args):
    r = subprocess.run([BIN, *map(str, args)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f'{BIN} {" ".join(map(str, args))}\n{r.stderr[-3000:]}')
    return r.stderr


def top_cmp(ri, rv, gi, gv):
    """上位 20 の比べ: 同じ語の logit の差の最大 / 参照の |logit| の最大、と参照の上位 5 が GPU の上位 20 に入っているか"""
    err, miss = 0.0, 0
    for a in range(ri.shape[0]):
        g = dict(zip(gi[a].tolist(), gv[a].tolist()))
        sc = float(np.abs(rv[a]).max()) + 1e-30
        for t, v in zip(ri[a].tolist(), rv[a].tolist()):
            if t in g:
                err = max(err, abs(g[t] - v) / sc)
        miss += sum(1 for t in ri[a][:5].tolist() if t not in g)
    return err, miss


def argmax_cmp(ri, rv, gi):
    """argmax が違う行と、そこでの参照の (1 位 − 2 位)"""
    bad = []
    for a in range(ri.shape[0]):
        if ri[a, 0] != gi[a, 0]:
            bad.append((a, float(rv[a, 0] - rv[a, 1])))
    return bad


def check_teacher(d, src, n, e, res):
    o = f'{d}/{src}'
    run('teacher', d, src)
    r_h = rd(f'{o}/teacher_hout.f32', np.float32, (n, e))
    g_h = rd(f'{o}/gpu_teacher_hout.f32', np.float32, (n, e))
    r_i = rd(f'{o}/teacher_topi.i32', np.int32, (n, TOPK))
    r_v = rd(f'{o}/teacher_topv.f32', np.float32, (n, TOPK))
    g_i = rd(f'{o}/gpu_teacher_topi.i32', np.int32, (n, TOPK))
    g_v = rd(f'{o}/gpu_teacher_topv.f32', np.float32, (n, TOPK))
    r_l = rd(f'{o}/teacher_lse.f32', np.float32)
    g_l = rd(f'{o}/gpu_teacher_lse.f32', np.float32)
    r_f = rd(f'{o}/teacher_logits.f32', np.float32)
    g_f = rd(f'{o}/gpu_teacher_logits.f32', np.float32)
    rows_h = [rel(g_h[p], r_h[p]) for p in range(n)]
    te, miss = top_cmp(r_i, r_v, g_i, g_v)
    bad = argmax_cmp(r_i, r_v, g_i)
    x = dict(h_rel_max=max(rows_h), h_rel_mean=float(np.mean(rows_h)), top_err=te, top5_miss=miss,
             lse_err=float(np.abs(g_l - r_l).max()), logits_rel=rel(g_f, r_f), argmax_diff=bad, n=n)
    ok = x['h_rel_max'] < TOL and x['logits_rel'] < TOL and te < TOL and all(gap < NEAR for _, gap in bad)
    res.append((f'{os.path.basename(d)} {src} teacher', ok, x))


def check_sim(d, src, k, n, e, res):
    o = f'{d}/{src}'
    run('sim', d, src, k)
    meta = json.load(open(f'{o}/meta.json'))
    rs = meta['sims'][str(k)]
    gs = json.load(open(f'{o}/gpu_sim{k}.json'))
    # 歩の列
    first = None
    for i, (a, b) in enumerate(zip(rs, gs)):
        if (a['pos0'], a['id_last'], list(a['drafts']), a['n_acc'], a['new']) != \
                (b['pos0'], b['id_last'], list(b['drafts']), b['n_acc'], b['new']):
            first = i
            break
    if first is None and len(rs) != len(gs):
        first = min(len(rs), len(gs))
    nr = sum(len(s['drafts']) for s in (rs if first is None else rs[:first]))
    r_i = rd(f'{o}/sim{k}_topi.i32', np.int32).reshape(-1, TOPK)
    r_v = rd(f'{o}/sim{k}_topv.f32', np.float32).reshape(-1, TOPK)
    g_i = rd(f'{o}/gpu_sim{k}_topi.i32', np.int32).reshape(-1, TOPK)
    g_v = rd(f'{o}/gpu_sim{k}_topv.f32', np.float32).reshape(-1, TOPK)
    r_h = rd(f'{o}/sim{k}_hout.f32', np.float32).reshape(-1, e)
    g_h = rd(f'{o}/gpu_sim{k}_hout.f32', np.float32).reshape(-1, e)
    r_l = rd(f'{o}/sim{k}_lse.f32', np.float32)
    g_l = rd(f'{o}/gpu_sim{k}_lse.f32', np.float32)
    r_f = rd(f'{o}/sim{k}_logits.f32', np.float32)
    g_f = rd(f'{o}/gpu_sim{k}_logits.f32', np.float32)
    nr = min(nr, r_i.shape[0], g_i.shape[0])
    nf = min(r_f.size, g_f.size)
    te, miss = top_cmp(r_i[:nr], r_v[:nr], g_i[:nr], g_v[:nr])
    bad = argmax_cmp(r_i[:nr], r_v[:nr], g_i[:nr])
    hrel = [rel(g_h[p], r_h[p]) for p in range(nr)]
    acc_r = sum(s['n_acc'] for s in rs)
    acc_g = sum(s['n_acc'] for s in gs)
    x = dict(steps_ref=len(rs), steps_gpu=len(gs), first_diff=first, rows=nr, h_rel_max=max(hrel) if hrel else 0.0,
             top_err=te, top5_miss=miss, lse_err=float(np.abs(g_l[:nr] - r_l[:nr]).max()) if nr else 0.0,
             logits_rel=rel(g_f[:nf], r_f[:nf]) if nf else 0.0, argmax_diff=bad,
             accepted_ref=f'{acc_r}/{sum(len(s["drafts"]) for s in rs)}', accepted_gpu=f'{acc_g}/{sum(len(s["drafts"]) for s in gs)}')
    near = False
    if first is not None:
        # 分かれた歩の下書きの行の、参照の 1 位と 2 位の差（その行は r_i の nr 番目から）
        j0 = nr
        nd = len(rs[first]['drafts']) if first < len(rs) else 0
        gaps = [float(r_v[j, 0] - r_v[j, 1]) for j in range(j0, min(j0 + nd, r_v.shape[0]))]
        x['diff_step'] = dict(ref=rs[first] if first < len(rs) else None, gpu=gs[first] if first < len(gs) else None, gaps=gaps)
        near = bool(gaps) and min(gaps) < NEAR
    ok = x['h_rel_max'] < TOL and x['logits_rel'] < TOL and te < TOL and all(g < NEAR for _, g in bad) and (first is None or near)
    res.append((f'{os.path.basename(d)} {src} sim k={k}' + ('（僅差で分かれた）' if first is not None and near else ''), ok, x))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(ROOT, 'logs', 'mtp'))
    ap.add_argument('--src', default='main,file')
    ap.add_argument('--k', default='1,2,3')
    ap.add_argument('--prompts', default=None, help='p0,p2 など（既定は全部）')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    res = []
    dirs = sorted(x for x in os.listdir(a.out) if x.startswith('p') and os.path.exists(f'{a.out}/{x}/meta.json'))
    if a.prompts:
        dirs = [x for x in dirs if x in a.prompts.split(',')]
    for dd in dirs:
        d = os.path.join(a.out, dd)
        meta = json.load(open(f'{d}/meta.json'))
        n, e = meta['n'], meta['n_embd']
        for src in a.src.split(','):
            if not os.path.exists(f'{d}/{src}/meta.json'):
                continue
            check_teacher(d, src, n, e, res)
            for k in [int(x) for x in a.k.split(',')]:
                if str(k) in json.load(open(f'{d}/{src}/meta.json'))['sims']:
                    check_sim(d, src, k, n, e, res)
    ng = 0
    for name, ok, x in res:
        ng += not ok
        brief = {k: (round(v, 7) if isinstance(v, float) else v) for k, v in x.items() if k not in ('diff_step',)}
        print(f'{"OK" if ok else "NG"}  {name}: {json.dumps(brief, ensure_ascii=False)}')
        if 'diff_step' in x:
            print(f'      分かれた歩: {json.dumps(x["diff_step"], ensure_ascii=False)}')
    if a.json:
        json.dump([dict(name=n_, ok=o_, **x_) for n_, o_, x_ in res], open(a.json, 'w'), ensure_ascii=False, indent=1)
    print('全部 OK' if ng == 0 else f'NG {ng} 件')
    sys.exit(1 if ng else 0)


if __name__ == '__main__':
    main()
