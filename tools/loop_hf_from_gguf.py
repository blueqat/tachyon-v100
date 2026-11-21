#!/usr/bin/env python3
"""SenseNova-Loop-9B の公式 BF16 GGUF を、公式の modeling_sensenova_loop.py が読める HF の safetensors に戻す（文字の部分だけ）。

なぜ: 本体の repo（sensenova/SenseNova-Loop-9B）は同意の門（gated）がある。GGUF の repo は門が無いので、そこから作る。
     公式の PyTorch のコード（github.com/OpenSenseNova/SenseNova-Loop）を手本として動かすために、GGUF の変換
     （OpenSenseNova/llama.cpp 24ea7bb の conversion/qwen.py・sensenova_loop.py）を逆にたどる:
  - 実行の番号の blk → 物理の層（blk 0..27 = 層 0..27、blk 52..55 = 層 28..31。blk 56 は MTP で戻さない）
  - attn_qkv_gate = [in_proj_qkv; in_proj_z]、ssm_beta_alpha = [in_proj_b; in_proj_a]（行で繋いだ物を分ける）
  - V の頭の並び（tiled → HF の grouped）を戻す: in_proj_qkv の v の行・in_proj_z の行・in_proj_a/b の行・A_log・dt_bias・
    conv1d の v の行・out_proj の列
  - RMSNorm の重みは GGUF で +1 してある（linear_attn.norm 以外）→ −1。ssm_a = −exp(A_log) → A_log = log(−ssm_a)
  - mHC（hc_*_fn）・conv1d は f32 のまま
BF16 の行列はバイトのまま並べ替えるだけ（値は変えない）。

Usage (CPU only; python3 with numpy, gguf, safetensors):
  python3 tools/loop_hf_from_gguf.py weights/loop9b/SenseNova-Loop-9B-BF16.gguf weights/loop9b/hf-ref --hf-dir weights/loop9b/hf
  → <out>/model.safetensors（約 18GB）と config.json（AWQ の repo の物から quantization_config を外し、architectures を公式の名前に）
  --check-awq: 量子化されていない小さなテンソル（in_proj_a/b・A_log・dt_bias・conv1d・norm・mhc）を AWQ の repo と HTTP Range で突き合わせる
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
from gguf import GGUFReader

NK, NV, HK, HV = 16, 32, 128, 128  # linear_num_key_heads・value_heads・key_head_dim・value_head_dim
R = NV // NK


def untile(x, axis, hd):
    """_reorder_v_heads（grouped → tiled）の逆。axis の長さ = NV·hd"""
    x = np.moveaxis(x, axis, 0)
    sh = x.shape
    y = x.reshape(R, NK, hd, *sh[1:]).swapaxes(0, 1).reshape(sh)
    return np.moveaxis(y, 0, axis)


def phys_of_blk(b):
    if b < 28:
        return b
    if 52 <= b < 56:
        return b - 24
    return None


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("out")
    ap.add_argument("--hf-dir", default="weights/loop9b/hf", help="AWQ の repo の config.json の置き場")
    ap.add_argument("--check-awq", action="store_true")
    a = ap.parse_args()
    rd = GGUFReader(a.gguf)
    T = {t.name: t for t in rd.tensors}
    out = {}  # name -> (dtype str, np array)

    def arr(name):
        t = T[name]
        shape = tuple(int(x) for x in reversed(t.shape.tolist()))  # numpy の並び（外側から）
        tn = t.tensor_type.name
        if tn == "BF16":
            return "BF16", np.asarray(t.data).view(np.uint16).reshape(shape)
        if tn == "F32":
            return "F32", np.asarray(t.data).view(np.float32).reshape(shape)
        raise SystemExit(f"{name}: 型 {tn} は BF16・F32 のはず")

    def put(name, dt, x):
        out[name] = (dt, np.ascontiguousarray(x))

    def norm_minus1(dt, x):
        assert dt == "F32"
        return x - np.float32(1.0)

    P = "model.language_model."
    dt, x = arr("token_embd.weight")
    put(P + "embed_tokens.weight", dt, x)
    dt, x = arr("output.weight")
    put("lm_head.weight", dt, x)
    dt, x = arr("output_norm.weight")
    put(P + "norm.weight", "F32", norm_minus1(dt, x))
    cc_qk = 2 * NK * HK
    for name in T:
        if not name.startswith("blk."):
            continue
        b = int(name.split(".")[1])
        p = phys_of_blk(b)
        if p is None:
            continue
        suf = name.split(".", 2)[2]
        L = f"{P}layers.{p}."
        dt, x = arr(name)
        if suf == "attn_norm.weight":
            put(L + "input_layernorm.weight", "F32", norm_minus1(dt, x))
        elif suf == "post_attention_norm.weight":
            put(L + "post_attention_layernorm.weight", "F32", norm_minus1(dt, x))
        elif suf == "attn_qkv_gate.weight":
            n_qkv = cc_qk + NV * HV
            qkv, z = x[:n_qkv], x[n_qkv:]
            v = untile(qkv[cc_qk:], 0, HV)
            put(L + "linear_attn.in_proj_qkv.weight", dt, np.concatenate([qkv[:cc_qk], v], 0))
            put(L + "linear_attn.in_proj_z.weight", dt, untile(z, 0, HV))
        elif suf == "ssm_beta_alpha.weight":
            put(L + "linear_attn.in_proj_b.weight", dt, untile(x[:NV], 0, 1))
            put(L + "linear_attn.in_proj_a.weight", dt, untile(x[NV:], 0, 1))
        elif suf == "ssm_a":
            put(L + "linear_attn.A_log", "F32", untile(np.log(-x.astype(np.float64)).astype(np.float32), 0, 1))
        elif suf == "ssm_dt.bias":
            put(L + "linear_attn.dt_bias", dt, untile(x, 0, 1))
        elif suf == "ssm_conv1d.weight":
            # GGUF [4, 8192]（numpy (8192, 4)）→ HF [8192, 1, 4]
            v = untile(x[cc_qk:], 0, HV)
            put(L + "linear_attn.conv1d.weight", dt, np.concatenate([x[:cc_qk], v], 0)[:, None, :])
        elif suf == "ssm_norm.weight":
            put(L + "linear_attn.norm.weight", dt, x)
        elif suf == "ssm_out.weight":
            put(L + "linear_attn.out_proj.weight", dt, untile(x, 1, HV))
        elif suf in ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"):
            m = {"attn_q.weight": "q_proj", "attn_k.weight": "k_proj", "attn_v.weight": "v_proj", "attn_output.weight": "o_proj"}[suf]
            put(L + f"self_attn.{m}.weight", dt, x)
        elif suf in ("attn_q_norm.weight", "attn_k_norm.weight"):
            put(L + f"self_attn.{suf.split('_')[1]}_norm.weight", "F32", norm_minus1(dt, x))
        elif suf in ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"):
            put(L + f"mlp.{suf.split('.')[0].replace('ffn_', '')}_proj.weight", dt, x)
        elif suf in ("hc_attn_fn.weight", "hc_attn_fn.bias", "hc_ffn_fn.weight", "hc_ffn_fn.bias"):
            m = "mhc_attn" if "attn" in suf else "mhc_ffn"
            assert dt == "F32"
            put(L + f"{m}.{suf.rsplit('.', 1)[1]}", dt, x)
        else:
            raise SystemExit(f"知らないテンソル {name}")
    print(f"{len(out)} テンソル", file=sys.stderr)
    # 物理の層 32 × 揃っているか
    for p in range(32):
        n = sum(1 for k in out if k.startswith(f"{P}layers.{p}."))
        exp = (14 if (p + 1) % 4 else 11) + (4 if 4 <= p < 28 else 0)
        assert n == exp, (p, n, exp)

    if a.check_awq:
        check_awq(out)
        return

    os.makedirs(a.out, exist_ok=True)
    header, off = {}, 0
    names = sorted(out)
    for k in names:
        dt, x = out[k]
        header[k] = {"dtype": dt, "shape": list(x.shape), "data_offsets": [off, off + x.nbytes]}
        off += x.nbytes
    hb = json.dumps(header).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(os.path.join(a.out, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for k in names:
            f.write(out[k][1].tobytes())
    cfg = json.load(open(os.path.join(a.hf_dir, "config.json")))
    cfg.pop("quantization_config", None)
    cfg["architectures"] = ["SenseNovaLoopForConditionalGeneration"]
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        src = os.path.join(a.hf_dir, f)
        if os.path.exists(src) and not os.path.exists(os.path.join(a.out, f)):
            os.symlink(src, os.path.join(a.out, f))
    print(f"書いた {off / 2**30:.2f} GiB → {a.out}", file=sys.stderr)


def check_awq(out):
    """AWQ の repo の量子化されていないテンソルと、戻した物を比べる（HTTP Range で要る所だけ）"""
    import requests

    base = "https://huggingface.co/sensenova/SenseNova-Loop-9B-AWQ/resolve/main/"
    idx = requests.get(base + "model.safetensors.index.json", timeout=60).json()["weight_map"]
    heads = {}
    worst = 0.0
    n = 0
    for k, (dt, x) in sorted(out.items()):
        if not any(s in k for s in ("in_proj_a.", "in_proj_b.", "A_log", "dt_bias", "conv1d", "layernorm", "norm.weight", "mhc_")):
            continue
        if k not in idx:
            print("AWQ に無い", k)
            continue
        f = idx[k]
        if f not in heads:
            r = requests.get(base + f, headers={"Range": "bytes=0-7"}, timeout=60)
            hl = struct.unpack("<Q", r.content)[0]
            h = json.loads(requests.get(base + f, headers={"Range": f"bytes=8-{8 + hl - 1}"}, timeout=60).content)
            heads[f] = (8 + hl, h)
        d0, h = heads[f]
        e = h[k]
        s, t = e["data_offsets"]
        raw = requests.get(base + f, headers={"Range": f"bytes={d0 + s}-{d0 + t - 1}"}, timeout=120).content
        ref = np.frombuffer(raw, dtype=np.uint16).copy() if e["dtype"] == "BF16" else np.frombuffer(raw, dtype=np.float32)
        ref = bf16_to_f32(ref) if e["dtype"] == "BF16" else ref
        mine = bf16_to_f32(x) if dt == "BF16" else x
        ref = ref.reshape(e["shape"]).astype(np.float32)
        assert list(mine.shape) == list(ref.shape), (k, mine.shape, ref.shape)
        err = float(np.max(np.abs(mine - ref)) / (np.max(np.abs(ref)) + 1e-12))
        worst = max(worst, err)
        n += 1
        if err > 1e-2 or n <= 3:
            print(f"{k}: 相対の最大差 {err:.2e}（AWQ {e['dtype']}）")
    print(f"AWQ と {n} 本を比べた。相対の最大差の最大 {worst:.2e}")


if __name__ == "__main__":
    main()
