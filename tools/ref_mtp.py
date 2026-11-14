#!/usr/bin/env python3
"""Qwen3.8-27B の MTP（blk.64）の CPU 参照（numpy・float32）と、llama.cpp の draft-mtp の進め方の手本。

源（unslothai/llama.cpp b12209-mix-b84c2bf）:
  src/models/qwen35.cpp            graph（本体の h_nextn）・graph_mtp（MTP の 1 段）
  common/speculative.cpp           common_speculative_impl_draft_mtp（process・draft・accept）
  tools/server/server-context.cpp  下書きの後の seq_rm(ctx_dft, pos0, -1)・受け取りの後の seq_rm(pos_next, -1)
式は docs/qwen35-math.md §7（この参照で確かめた形）。

  本体の位置 p の h_nextn[p] = RMSNorm(h_L[p], output_norm)（= result_norm。norm の後）
  MTP の 1 行（位置 p・語 x・状態 h）:
    e  = RMSNorm(token_embd[x], nextn.enorm)        hh = RMSNorm(h, nextn.hnorm)
    x0 = nextn.eh_proj · [e ; hh]                    （10240 → 5120。e が前）
    x1 = x0 + attn_output · (Attn(…) ⊙ σ(gate))      （普通の Attention 1 層。MTP だけの KV。式は本体の 16 層と同じ）
    x2 = x1 + ffn_down · (silu(ffn_gate·y) ⊙ ffn_up·y)、y = RMSNorm(x1, post_attention_norm)
    h' = RMSNorm(x2, nextn.shared_head_norm)          （MTP の h_nextn。下書きをつなぐときの次の h）
    logits = output · h'
  進め方（llama-server の --spec-type draft-mtp。p_min 0・n_max k・下書きは argmax）:
    process  本体が通した語ごとに MTP の行を 1 つ: 位置 p の行は (x_p, h_nextn[p-1])。列の頭（p = 0）は h = 0
    draft    最後の語 id_last（位置 n）と h_nextn[n-1] → d1、(d1, h'_1) を位置 n+1 で → d2 …
             下書きの後、MTP の KV の位置 n 以降を消す（確かめの後の process が本体の h で書き直す）
    accept   受け取った所までの行を残す（後ろは seq_rm で消える）。次の pending = 本体の h_nextn[n + 受け取った数]

使い方（V100 の機械。GPU は使わない）:
  CUDA_VISIBLE_DEVICES= .venv/bin/python tools/ref_mtp.py run --prompts 0,2,3,5,6 --out logs/mtp
      本体を CPU 参照で通し（tools/ref_qwen35.py。K・V は f16 に丸める）、h_nextn と本体の argmax を取る。
      列は data/refcmp/qwen35-server.json の文＋llama-server が温度 0 で出した 32 語。MTP は 2 つの重みで:
        main  本体の GGUF の blk.64、埋め込みと出力も本体のもの
        file  別ファイル mtp-Qwen3.8-27B-Q4_0.gguf の blk.64、埋め込みと出力も別ファイルの Q3_K
              （llama-server の -md で測った 52.6 tok/s はこちら。ログ logs/q27.log の「loading draft model」）
      teacher: 列の全位置で MTP の行（KV を埋める）→ 行ごとの上位 20 の logits・h'
      sim:     温度 0 の下書き k = 1・2・3 の回り（下書き → 本体の argmax で確かめ → 追いつき）
      → logs/mtp/p<i>/ に生の f32/i32 と meta.json（tools/mtpcheck.py が GPU と比べる）
  CUDA_VISIBLE_DEVICES= .venv/bin/python tools/ref_mtp.py lcpp --dump logs/lcpp_mtp/main
      tools/lcpp_mtp.cpp（llama.cpp そのもの・CPU）の MTP の節と比べる（入力は llama.cpp の本体の h_nextn）
  .venv/bin/python tools/ref_mtp.py stats --out logs/mtp
      受け取りの数（teacher の 1 語目の当たり・sim の平均の長さ）
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_qwen35 as R  # noqa: E402
from ref_qwen35 import F32, rms_norm, rope_apply, rope_cache, sigmoid, silu  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
MAIN = R.DEFAULT_GGUF
MTPF = os.path.join(os.environ.get("TACHYON_WEIGHTS", "weights"), "q38-27b/mtp-Qwen3.8-27B-Q4_0.gguf")
SERVER = os.path.join(ROOT, 'data', 'refcmp', 'qwen35-server.json')
TOPK = 20


# ---------------- MTP だけの KV（位置で引く。列ごと） ----------------

class MtpKV:
    """K（rope の後）と V を位置の順に持つ。位置 p の行を書くと p から後ろは捨てる（llama.cpp の seq_rm と同じ形）"""

    def __init__(self, hp, f16=True):
        self.K = np.zeros((0, hp.n_head_kv, hp.head_dim), F32)
        self.V = np.zeros((0, hp.n_head_kv, hp.head_dim), F32)
        self.f16 = f16

    def __len__(self):
        return self.K.shape[0]

    def truncate(self, n):
        self.K, self.V = self.K[:n], self.V[:n]

    def put(self, pos, k, v):
        pos = np.asarray(pos)
        assert (np.diff(pos) == 1).all(), '行の位置は続いていること'
        assert pos[0] <= len(self), f'KV に穴が開く（{pos[0]} > {len(self)}）'
        if self.f16:  # llama.cpp の KV キャッシュは f16
            k = k.astype(np.float16).astype(F32)
            v = v.astype(np.float16).astype(F32)
        self.truncate(int(pos[0]))
        self.K = np.concatenate([self.K, k])
        self.V = np.concatenate([self.V, v])

    def attend(self, q, pos, scale):
        """q (n, H, D)。行 i は KV の位置 0..pos[i] を見る（因果）。GQA は q の頭 h → KV の頭 h // (H/Hkv)"""
        n, H, D = q.shape
        Hkv = self.K.shape[1]
        r = H // Hkv
        T = len(self)
        mask = np.where(np.arange(T)[None, :] <= np.asarray(pos)[:, None], F32(0), F32(-np.inf))  # (n, T)
        o = np.empty((n, H, D), F32)
        for g in range(Hkv):
            qq = q[:, g * r:(g + 1) * r].reshape(n * r, D)
            s = (qq @ self.K[:, g].T) * scale + np.repeat(mask, r, axis=0)
            s = s - s.max(axis=1, keepdims=True)
            e = np.exp(s)
            p = (e / e.sum(axis=1, keepdims=True, dtype=np.float64)).astype(F32)
            o[:, g * r:(g + 1) * r] = (p @ self.V[:, g]).reshape(n, r, D)
        return o


# ---------------- MTP の 1 段 ----------------

class MtpRef:
    """blk.64 の重みと出力の行列（f32 で全部持つ。出力は 5GB）。
    src='main': 本体の GGUF の blk.64、埋め込み・出力も本体の物。src='file': 別ファイルの物（埋め込み・出力も別ファイル）"""

    def __init__(self, src='main', main=MAIN, mtpf=MTPF, workers=16, log=print, kv_f16=True):
        t0 = time.time()
        self.src = src
        self.path = main if src == 'main' else mtpf
        self.ref = R.Ref(self.path, workers=workers, log=log)
        self.G, self.hp, self.log, self.kv_f16 = self.ref.G, self.ref.hp, log, kv_f16
        hp = self.hp
        assert hp.n_nextn == 1, hp.n_nextn
        il = hp.n_layer
        p = f'blk.{il}.'
        G = self.G
        W = {}
        for k, n in [('attn_norm', 'attn_norm'), ('post_norm', 'post_attention_norm'), ('attn_q_norm', 'attn_q_norm'),
                     ('attn_k_norm', 'attn_k_norm'), ('enorm', 'nextn.enorm'), ('hnorm', 'nextn.hnorm'),
                     ('head_norm', 'nextn.shared_head_norm')]:
            W[k] = G.mat(p + n + '.weight').astype(F32)
        big = [('eh_proj', 'nextn.eh_proj'), ('attn_q', 'attn_q'), ('attn_k', 'attn_k'), ('attn_v', 'attn_v'),
               ('attn_output', 'attn_output'), ('ffn_gate', 'ffn_gate'), ('ffn_up', 'ffn_up'), ('ffn_down', 'ffn_down')]
        self.ref.start_pool()
        got = self.ref.load([p + n + '.weight' for _, n in big])
        for k, n in big:
            W[k] = np.array(got[p + n + '.weight'])     # /dev/shm から自分の RAM へ
        self.ref._free()
        # 埋め込みと出力（nextn.embed_tokens・shared_head_head があればそれ。この 2 つの GGUF には無い）
        self.embd_name = p + 'nextn.embed_tokens.weight' if (p + 'nextn.embed_tokens.weight') in G.t else 'token_embd.weight'
        self.out_name = p + 'nextn.shared_head_head.weight' if (p + 'nextn.shared_head_head.weight') in G.t else 'output.weight'
        V = G.n_rows(self.out_name)
        self.Wout = np.empty((V, hp.n_embd), F32)
        step = 1 << 15
        for r0 in range(0, V, step):
            n = min(step, V - r0)
            self.Wout[r0:r0 + n] = self.ref.load([self.out_name], r0, n)[self.out_name]
            self.ref._free()
        self.W = W
        self.types = {n: R.QT(G.t[p + n + '.weight'][2]).name for _, n in big}
        self.types['token_embd'] = R.QT(G.t[self.embd_name][2]).name
        self.types['output'] = R.QT(G.t[self.out_name][2]).name
        log(f'MTP {src}: {self.path} の重み {time.time() - t0:.1f}s  型 {self.types}')

    def close(self):
        self.ref.close()

    def embed(self, toks):
        return self.G.gather(self.embd_name, np.asarray(toks, np.int64)).astype(F32)

    def logits(self, hout):
        return hout @ self.Wout.T

    def forward(self, kv, toks, pos, h, want_logits=True, keep=None):
        """1 回の MTP（行は 1 つの列の続いた位置）。KV に書き、(h' (n, 5120), logits (n, V) か None) を返す。
        keep(d) に途中の値（llama.cpp の節の名前。-64 は付けない）を渡す"""
        hp, W, eps = self.hp, self.W, self.hp.eps
        n = len(toks)
        H, Hkv, D = hp.n_head, hp.n_head_kv, hp.head_dim
        pos = np.asarray(pos, np.int64)
        h = np.asarray(h, F32).reshape(n, hp.n_embd)
        e = self.embed(toks)
        en = rms_norm(e, W['enorm'], eps)
        hn = rms_norm(h, W['hnorm'], eps)
        cat = np.concatenate([en, hn], axis=1)                        # ggml_concat(e_norm, h_norm, 0)
        x0 = cat @ W['eh_proj'].T
        a = rms_norm(x0, W['attn_norm'], eps)
        qf = (a @ W['attn_q'].T).reshape(n, H, 2 * D)                 # 頭ごとに [q | gate]
        q = rms_norm(qf[:, :, :D], W['attn_q_norm'], eps)
        gate = qf[:, :, D:].reshape(n, H * D)
        k = rms_norm((a @ W['attn_k'].T).reshape(n, Hkv, D), W['attn_k_norm'], eps)
        v = (a @ W['attn_v'].T).reshape(n, Hkv, D)
        cs = rope_cache(pos, hp)
        qr = rope_apply(q, cs)
        kr = rope_apply(k, cs)
        kv.put(pos, kr, v)
        o = kv.attend(qr, pos, F32(1.0 / np.sqrt(D)))
        og = o.reshape(n, H * D) * sigmoid(gate)
        ao = og @ W['attn_output'].T
        x1 = x0 + ao
        b = rms_norm(x1, W['post_norm'], eps)
        f = (silu(b @ W['ffn_gate'].T) * (b @ W['ffn_up'].T)) @ W['ffn_down'].T
        x2 = x1 + f
        hout = rms_norm(x2, W['head_norm'], eps)
        if keep is not None:
            keep(dict(mtp_tok_embd=e, mtp_enorm=en, mtp_hnorm=hn, mtp_concat=cat, mtp_eh_proj=x0, mtp_attn_norm=a,
                      mtp_Qcur_full=qf.reshape(n, 2 * H * D), mtp_Qcur_normed=q, mtp_gate=gate, mtp_Kcur_normed=k,
                      mtp_Vcur=v, mtp_Qcur=qr, mtp_Kcur=kr, mtp_attn_pregate=o.reshape(n, H * D), mtp_attn_gated=og,
                      mtp_attn_out=ao, mtp_attn_residual=x1, mtp_attn_post_norm=b, mtp_ffn_out=f, mtp_post_ffn=x2,
                      h_nextn=hout))
        return hout, (self.logits(hout) if want_logits else None)


def topk(lg, k=TOPK):
    """行ごとに (上位 k の id（logit の大きい順、同点は小さい id）, その logit, logsumexp)"""
    lg = np.atleast_2d(lg)
    n = lg.shape[0]
    ids = np.empty((n, k), np.int32)
    vals = np.empty((n, k), F32)
    lse = np.empty(n, F32)
    for i in range(n):
        r = lg[i]
        c = np.argpartition(-r, k)[:k]
        c = c[np.lexsort((c, -r[c]))]
        ids[i], vals[i] = c, r[c]
        m = r.max()
        lse[i] = m + np.log(np.sum(np.exp((r - m).astype(np.float64))))
    return ids, vals, lse


def h_in_rows(hn, toks_pos0, n, pending=None):
    """位置 pos0.. の n 行の h: 1 行目は pending（無ければ h_nextn[pos0-1]、pos0 = 0 なら 0）、残りは h_nextn[pos0..]"""
    pos0 = toks_pos0
    first = pending if pending is not None else (np.zeros(hn.shape[1], F32) if pos0 == 0 else hn[pos0 - 1])
    return np.vstack([first[None], hn[pos0:pos0 + n - 1]]) if n > 1 else first[None]


def teacher(m, S, hn, keep=None, n_full=2):
    """列の全位置で MTP の行（位置 p は (S[p], h_nextn[p-1])、p = 0 は h = 0）。1 回で通す"""
    N = len(S)
    kv = MtpKV(m.hp, m.kv_f16)
    hout, lg = m.forward(kv, S, np.arange(N), h_in_rows(hn, 0, N), keep=keep)
    ids, vals, lse = topk(lg)
    return dict(hout=hout, topi=ids, topv=vals, lse=lse, argmax=ids[:, 0].copy(), logits=lg[-n_full:].copy())


def sim(m, S, hn, tgt_arg, n_prompt, k, log=print):
    """温度 0 の下書きの回り（llama-server の draft-mtp と同じ順）。本体は「記録した argmax」（S が本体の貪欲な続きの間だけ正しい）"""
    hp = m.hp
    kv = MtpKV(hp, m.kv_f16)
    # プロンプトの追いつき（本体のプロンプトの batch の後の process）
    m.forward(kv, S[:n_prompt], np.arange(n_prompt), h_in_rows(hn, 0, n_prompt), want_logits=False)
    pending = hn[n_prompt - 1]
    pos0 = n_prompt
    id_last = int(tgt_arg[n_prompt - 1])
    steps, rows = [], dict(topi=[], topv=[], lse=[], hout=[], logits=[])
    while pos0 <= len(S) - 1:
        # 確かめの行 pos0..pos0+kk の本体の出力が要る（記録は len(S)-1 まで）。最後は下書きなし（kk = 0）の 1 歩で記録の最後の行の語まで
        # （src/spec.rs の回し方で max_new = len(S) - n_prompt + 1 と同じ止まり方）
        kk = min(k, len(S) - 1 - pos0)
        # 下書き（1 段ごとに 1 行。1 段目は (id_last, pending)、次は (d, h')）
        drafts = []
        t, h = id_last, pending
        for j in range(kk):
            ho, lg = m.forward(kv, [t], [pos0 + j], h[None])
            ids, vals, lse = topk(lg)
            d = int(ids[0, 0])
            drafts.append(d)
            rows['topi'].append(ids[0])
            rows['topv'].append(vals[0])
            rows['lse'].append(lse[0])
            rows['hout'].append(ho[0])
            if len(rows['logits']) < 4:
                rows['logits'].append(lg[0].copy())
            t, h = d, ho[0]
        kv.truncate(pos0)                    # llama-server: 下書きの後に seq_rm(ctx_dft, pos0, -1)
        # 確かめ（温度 0）: 受け取った下書きまでは入力が S と同じなので、記録した本体の argmax がそのまま使える
        n = 0
        while n < kk and drafts[n] == int(tgt_arg[pos0 + n]):
            n += 1
        new = int(tgt_arg[pos0 + n])
        # process: 確かめた行を本体の h で（受け取った分まで。後ろは seq_rm で消えるので書かない）
        toks = [id_last] + drafts[:n]
        m.forward(kv, toks, np.arange(pos0, pos0 + n + 1), h_in_rows(hn, pos0, n + 1, pending), want_logits=False)
        steps.append(dict(pos0=int(pos0), id_last=int(id_last), drafts=drafts, n_acc=int(n), new=new))
        pending = hn[pos0 + n]
        id_last = new
        pos0 = pos0 + n + 1
    acc = [s['n_acc'] for s in steps]
    dr = [len(s['drafts']) for s in steps]
    log(f'  sim k={k}: {len(steps)} 歩・受け取り {sum(acc)}/{sum(dr)}（{sum(acc) / max(1, sum(dr)):.3f}）・'
        f'1 歩の平均 {1 + sum(acc) / max(1, len(steps)):.2f} 語')
    return steps, {kk: np.array(v) for kk, v in rows.items()}


# ---------------- 保存 ----------------

def wr(path, a, dt):
    np.ascontiguousarray(a, dtype=dt).tofile(path)


def save_seq(d, S, n_prompt, hn, tgt):
    os.makedirs(d, exist_ok=True)
    wr(f'{d}/tokens.i32', S, np.int32)
    wr(f'{d}/h.f32', hn, F32)
    wr(f'{d}/tgt_argmax.i32', tgt['argmax'], np.int32)
    wr(f'{d}/tgt_topi.i32', tgt['topi'], np.int32)
    wr(f'{d}/tgt_topv.f32', tgt['topv'], F32)
    wr(f'{d}/tgt_lse.f32', tgt['lse'], F32)


def save_mtp(d, src, m, tr, sims, nodes):
    o = f'{d}/{src}'
    os.makedirs(o, exist_ok=True)
    for k in ('hout', 'topv', 'lse', 'logits'):
        wr(f'{o}/teacher_{k}.f32', tr[k], F32)
    for k in ('topi', 'argmax'):
        wr(f'{o}/teacher_{k}.i32', tr[k], np.int32)
    meta = dict(src=src, path=m.path, types=m.types, embd=m.embd_name, out=m.out_name, n_full=int(tr['logits'].shape[0]),
                sims={})
    for k, (steps, rows) in sims.items():
        meta['sims'][str(k)] = steps
        for kk in ('topv', 'lse', 'hout', 'logits'):
            wr(f'{o}/sim{k}_{kk}.f32', rows[kk], F32)
        wr(f'{o}/sim{k}_topi.i32', rows['topi'], np.int32)
    meta['nodes'] = sorted(nodes)
    for name, v in nodes.items():
        wr(f'{o}/node.{name}.f32', v, F32)
    json.dump(meta, open(f'{o}/meta.json', 'w'), ensure_ascii=False, indent=1)


# ---------------- run ----------------

def cmd_run(a):
    log = lambda s: print(s, file=sys.stderr, flush=True)
    srv = json.load(open(a.server))
    idx = [int(x) for x in a.prompts.split(',')]
    seqs, nps = [], []
    for i in idx:
        it = srv[i]
        S = list(it['prompt']) + list(it['gen'])
        if a.max_gen is not None:
            S = S[:len(it['prompt']) + a.max_gen]
        seqs.append(S)
        nps.append(len(it['prompt']))
    log(f'列 {idx}: 長さ {[len(s) for s in seqs]}（文 {nps}）')
    # 1) 本体（CPU 参照・全 64 層）。全位置の result_norm（= h_nextn）と logits
    t0 = time.time()
    ref = R.Ref(a.gguf, workers=a.workers, log=log)
    try:
        outs = ref.forward(seqs, logit_pos=[np.arange(len(s)) for s in seqs], dump_pos=None, kv_f16=True)
    finally:
        ref.close()
    log(f'本体 {time.time() - t0:.0f}s')
    tgts = []
    for i, S, n_p, o in zip(idx, seqs, nps, outs):
        ids, vals, lse = topk(o['logits'])
        tgt = dict(argmax=ids[:, 0].copy(), topi=ids, topv=vals, lse=lse)
        # 本体の貪欲な続きか（S[p+1] == argmax[p]）。外れた所で切る（sim の確かめが記録で済む範囲）
        bad = [p for p in range(n_p - 1, len(S) - 1) if S[p + 1] != tgt['argmax'][p]]
        cut = len(S) if not bad else bad[0] + 1
        log(f'  p{i}: 続きの外れ {bad[:5]} → {cut} 語で切る')
        S = S[:cut]
        hn = o['result_norm'][:cut].astype(F32)
        for k in tgt:
            tgt[k] = tgt[k][:cut]
        d = os.path.join(a.out, f'p{i}')
        save_seq(d, S, n_p, hn, tgt)
        tgts.append((i, d, S, n_p, hn, tgt))
        del o['logits']
    mtp_phase(a, [(i, d, S, n_p, hn, tgt) for i, d, S, n_p, hn, tgt in tgts], log, t0)


def load_saved(a, log):
    """run の本体の段で書いた列（tokens・h・tgt_*）を読み直す（MTP だけやり直すとき）"""
    srv = json.load(open(a.server))
    out = []
    for i in [int(x) for x in a.prompts.split(',')]:
        d = os.path.join(a.out, f'p{i}')
        S = np.fromfile(f'{d}/tokens.i32', np.int32)
        N = len(S)
        hn = np.fromfile(f'{d}/h.f32', F32).reshape(N, -1)
        tgt = dict(argmax=np.fromfile(f'{d}/tgt_argmax.i32', np.int32),
                   topi=np.fromfile(f'{d}/tgt_topi.i32', np.int32).reshape(N, TOPK),
                   topv=np.fromfile(f'{d}/tgt_topv.f32', F32).reshape(N, TOPK),
                   lse=np.fromfile(f'{d}/tgt_lse.f32', F32))
        out.append((i, d, S.tolist(), len(srv[i]['prompt']), hn, tgt))
        log(f'  p{i}: {N} 語（文 {len(srv[i]["prompt"])}）を読み直した')
    return out


def cmd_mtp(a):
    log = lambda s: print(s, file=sys.stderr, flush=True)
    mtp_phase(a, load_saved(a, log), log, time.time())


def mtp_phase(a, tgts, log, t0):
    # 2) MTP（2 つの重み）
    srcs = a.src.split(',')
    for src in srcs:
        m = MtpRef(src, a.gguf, a.mtp, workers=a.workers, log=log)
        try:
            for i, d, S, n_p, hn, tgt in tgts:
                t1 = time.time()
                nodes = {}
                keep = (lambda dd: nodes.update(dd)) if len(S) <= a.node_max else None
                tr = teacher(m, S, hn, keep=keep)
                hit = float(np.mean(tr['argmax'][1:] == tgt['argmax'][1:]))
                # 本体の温度 1 での受け取りの見込み = P_本体(MTP の argmax)
                pt = []
                for p in range(1, len(S)):
                    w = np.where(tgt['topi'][p] == tr['argmax'][p])[0]
                    pt.append(float(np.exp(tgt['topv'][p][w[0]] - tgt['lse'][p])) if len(w) else 0.0)
                log(f'  p{i} {src}: teacher {len(S)} 行・1 語目の当たり {hit:.3f}（温度 1 の見込み {np.mean(pt):.3f}）'
                    f' {time.time() - t1:.0f}s')
                sims = {}
                for k in [int(x) for x in a.k.split(',')]:
                    sims[k] = sim(m, S, hn, tgt['argmax'], n_p, k, log=log)
                save_mtp(d, src, m, tr, sims, nodes)
                meta = dict(prompt=i, n=len(S), n_prompt=n_p, n_embd=m.hp.n_embd, topk=TOPK, eps=float(m.hp.eps),
                            gguf=a.gguf, mtp=a.mtp, srcs=srcs)
                json.dump(meta, open(f'{d}/meta.json', 'w'), indent=1)
        finally:
            m.close()
            del m
    log(f'合計 {time.time() - t0:.0f}s')


# ---------------- llama.cpp と比べる ----------------

def rel(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def cmd_lcpp(a):
    """tools/lcpp_mtp.cpp の出力（llama.cpp の CPU）と比べる。MTP の入力（本体の h_nextn・語）は llama.cpp の物を使う
    → 差は MTP の 1 段の中だけ。節は「llama.cpp の 1 つ前の節を入力に、参照の式だけを通す」比べもする"""
    log = lambda s: print(s, file=sys.stderr, flush=True)
    D = a.dump
    meta = json.load(open(f'{D}/meta.json'))
    S = np.fromfile(f'{D}/tokens.i32', np.int32)
    N = len(S)
    hT = np.fromfile(f'{D}/tgt_h.f32', F32).reshape(N, -1)
    m = MtpRef(meta['src'], a.gguf, a.mtp, workers=a.workers, log=log)
    res = dict(src=meta['src'], n=N)
    try:
        hp = m.hp
        # 本体の h_nextn: 参照（ref_qwen35）と llama.cpp（自分の参照の run の結果があれば）
        hr = os.path.join(a.out, f'p{a.prompt}', 'h.f32') if (a.out and a.prompt is not None) else None
        if hr and os.path.exists(hr):
            hR = np.fromfile(hr, F32).reshape(-1, hp.n_embd)
            n = min(N, hR.shape[0])
            res['tgt_h_rel_vs_ref_qwen35'] = rel(hR[:n], hT[:n])
        # teacher: llama.cpp の本体の h で
        nodes = {}
        kv = MtpKV(hp, True)
        hout, lg = m.forward(kv, S, np.arange(N), h_in_rows(hT, 0, N), keep=lambda d: nodes.update(d))
        L = {}
        call = f'{D}/call0'
        for f in sorted(os.listdir(call)):
            if f.endswith('.npy'):
                L[f[:-4]] = np.load(f'{call}/{f}')
        rows = []
        for name, v in nodes.items():
            ln = name + '-64' if name != 'h_nextn' else 'h_nextn'
            if ln not in L:
                continue
            x = L[ln].reshape(v.shape) if L[ln].size == v.size else None
            if x is None:
                rows.append((name, 'shape', L[ln].shape, v.shape))
                continue
            rows.append((name, rel(v, x)))
        res['nodes_teacher'] = rows
        # 式だけ（行列の掛け算なし）: llama.cpp の節を入力にして次の節を計算
        eps = hp.eps
        W = m.W
        def g(n):
            return L.get(n + '-64')
        only = {}
        if g('mtp_tok_embd') is not None and g('mtp_enorm') is not None:
            only['enorm'] = rel(rms_norm(g('mtp_tok_embd').reshape(N, -1), W['enorm'], eps), g('mtp_enorm').reshape(N, -1))
        if g('mtp_hnorm') is not None:
            only['hnorm'] = rel(rms_norm(h_in_rows(hT, 0, N), W['hnorm'], eps), g('mtp_hnorm').reshape(N, -1))
        if g('mtp_eh_proj') is not None and g('mtp_attn_norm') is not None:
            only['attn_norm'] = rel(rms_norm(g('mtp_eh_proj').reshape(N, -1), W['attn_norm'], eps), g('mtp_attn_norm').reshape(N, -1))
        if g('mtp_post_ffn') is not None and 'h_nextn' in L:
            only['shared_head_norm'] = rel(rms_norm(g('mtp_post_ffn').reshape(N, -1), W['head_norm'], eps), L['h_nextn'].reshape(N, -1))
        if g('mtp_Qcur_full') is not None and g('mtp_Qcur_normed') is not None:
            qf = g('mtp_Qcur_full').reshape(N, hp.n_head, 2 * hp.head_dim)
            only['q_norm'] = rel(rms_norm(qf[:, :, :hp.head_dim], W['attn_q_norm'], eps), g('mtp_Qcur_normed').reshape(N, hp.n_head, -1))
        # Attention の中だけ（rope・因果・GQA・softmax）: llama.cpp の rope 前の q・k・v から
        if all(g(n_) is not None for n_ in ('mtp_Qcur_normed', 'mtp_Kcur_normed', 'mtp_Vcur', 'mtp_attn_pregate')):
            H, Hkv, Dh = hp.n_head, hp.n_head_kv, hp.head_dim
            qn = g('mtp_Qcur_normed').reshape(N, H, Dh)
            kn = g('mtp_Kcur_normed').reshape(N, Hkv, Dh)
            vn = g('mtp_Vcur').reshape(N, Hkv, Dh)
            want = g('mtp_attn_pregate').reshape(N, H * Dh)

            def core(pos_rope, extra=0):
                cs = rope_cache(pos_rope, hp)
                k2 = MtpKV(hp, True)
                k2.put(np.arange(N), rope_apply(kn, cs), vn)
                return k2.attend(rope_apply(qn, cs), np.arange(N) + extra, F32(1.0 / np.sqrt(Dh))).reshape(N, -1)

            only['attn_core'] = rel(core(np.arange(N)), want)
            # 確かめの確かめ: rope の位置を全部 1 ずらしても同じ（相対）、因果を 1 つ先まで見ると違う
            only['attn_core_pos+1'] = rel(core(np.arange(N) + 1), want)
            only['attn_core_mask+1'] = rel(core(np.arange(N), extra=1), want)
        res['ops_only'] = only
        # logits と下書き（llama.cpp の teacher の logits は全行）
        lgL = np.fromfile(f'{D}/mtp_logits.f32', F32).reshape(N, -1)
        aL = lgL.argmax(axis=1)
        aR = lg.argmax(axis=1)
        res['teacher_logits_rel'] = rel(lg, lgL)
        res['teacher_argmax_agree'] = float(np.mean(aL == aR))
        res['teacher_argmax_diff_rows'] = [int(i) for i in np.where(aL != aR)[0]]
        hL = np.fromfile(f'{D}/mtp_h.f32', F32).reshape(N, -1)
        res['teacher_h_rel'] = rel(hout, hL)
        # 下書きのつなぎ（llama.cpp と同じ入力: id_last・本体の h_nextn[N-1] → d1 → (d1, h'_1) → d2 …）
        ch = json.load(open(f'{D}/chain.json'))
        hc = np.fromfile(f'{D}/chain_h.f32', F32).reshape(len(ch['drafts']), -1)
        lc = np.fromfile(f'{D}/chain_logits.f32', F32).reshape(len(ch['drafts']), -1)
        t, h = ch['id_last'], hT[N - 1]
        mine = []
        for j in range(len(ch['drafts'])):
            ho, l1 = m.forward(kv, [t], [N + j], h[None])
            d = int(l1[0].argmax())
            mine.append(dict(draft=d, logits_rel=rel(l1[0], lc[j]), h_rel=rel(ho[0], hc[j])))
            t, h = d, ho[0]
        res['chain_lcpp'] = ch['drafts']
        res['chain_ref'] = mine
    finally:
        m.close()
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.json:
        json.dump(res, open(a.json, 'w'), ensure_ascii=False, indent=1)


def cmd_stats(a):
    for d in sorted(os.listdir(a.out)):
        P = os.path.join(a.out, d)
        if not os.path.exists(f'{P}/meta.json'):
            continue
        meta = json.load(open(f'{P}/meta.json'))
        tg = np.fromfile(f'{P}/tgt_argmax.i32', np.int32)
        for src in meta['srcs']:
            mm = json.load(open(f'{P}/{src}/meta.json'))
            ta = np.fromfile(f'{P}/{src}/teacher_argmax.i32', np.int32)
            s = f'{d} {src}: teacher {np.mean(ta[1:] == tg[1:]):.3f}'
            for k, steps in mm['sims'].items():
                acc = sum(x['n_acc'] for x in steps)
                dr = sum(len(x['drafts']) for x in steps)
                s += f'  k={k} {acc}/{dr} 平均 {1 + acc / max(1, len(steps)):.2f}'
            print(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--prompts', default='0,2,3,5,6')
    r.add_argument('--server', default=SERVER)
    r.add_argument('--out', default=os.path.join(ROOT, 'logs', 'mtp'))
    r.add_argument('--gguf', default=MAIN)
    r.add_argument('--mtp', default=MTPF)
    r.add_argument('--src', default='main,file')
    r.add_argument('--k', default='1,2,3')
    r.add_argument('--max-gen', type=int, default=None)
    r.add_argument('--node-max', type=int, default=80, help='途中の値を残す列の長さの上限')
    r.add_argument('--workers', type=int, default=16)
    mm = sub.add_parser('mtp', help='run の本体の段の結果から MTP だけやり直す')
    mm.add_argument('--prompts', default='0,2,3,5,6')
    mm.add_argument('--server', default=SERVER)
    mm.add_argument('--out', default=os.path.join(ROOT, 'logs', 'mtp'))
    mm.add_argument('--gguf', default=MAIN)
    mm.add_argument('--mtp', default=MTPF)
    mm.add_argument('--src', default='main,file')
    mm.add_argument('--k', default='1,2,3')
    mm.add_argument('--node-max', type=int, default=80)
    mm.add_argument('--workers', type=int, default=16)
    l = sub.add_parser('lcpp')
    l.add_argument('--dump', required=True)
    l.add_argument('--gguf', default=MAIN)
    l.add_argument('--mtp', default=MTPF)
    l.add_argument('--out', default=os.path.join(ROOT, 'logs', 'mtp'), help='run の出力（本体の h の比べに使う）')
    l.add_argument('--prompt', type=int, default=None, help='run の列の番号（本体の h を参照と比べる）')
    l.add_argument('--json', default=None)
    l.add_argument('--workers', type=int, default=16)
    s = sub.add_parser('stats')
    s.add_argument('--out', default=os.path.join(ROOT, 'logs', 'mtp'))
    a = ap.parse_args()
    {'run': cmd_run, 'mtp': cmd_mtp, 'lcpp': cmd_lcpp, 'stats': cmd_stats}[a.cmd](a)


if __name__ == '__main__':
    main()
