"""Gated DeltaNet のカーネル（target/release/gdncheck）の答え合わせ。手本は tools/gdn_ref.py（float64）。Qwen3.8-27B（qwen35）向け。

  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/gdncheck.py [random|layer0|long|all] [GGUF]

  random: でたらめな入力で decode（B=1・8・32、slots を逆順にした組も）・steps・seq・prefill（T の端を色々）・
          prefill の後に 1 トークンずつ（split）・下書きの確かめと受け入れた分の確定（verify。T=3・17）
  layer0: 本物の blk.0（ssm_*・attn_norm・attn_qkv・attn_gate・ssm_alpha・ssm_beta）と本物の語の埋め込みで。
          入力 = rms_norm(token_embd[語], attn_norm) を f32 で射影したもの（llama.cpp の層 0 の入口と同じ）。
          prefill T=547（状態 0 から）、prefill の後に decode（B=1・8・32、状態は本物の文で作ったもの）
  long:   T=4096 を塊ごと・1 トークンずつ（steps と seq）・手本で比べる（減衰の弱い頭も混ぜる）
判定: 誤差 = max|GPU - 手本| / max|手本| < 1e-3（y・状態・conv の状態・conv の出力）。
「1 トークンずつ ＝ 塊ごと」は GPU どうしでも比べる（steps と prefill）。
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gdn_ref as R  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BIN = str(ROOT / "target/release/gdncheck")
GGUF = os.path.join(os.environ.get("TACHYON_WEIGHTS", "weights"), "q38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf")
HF = os.path.join(os.environ.get("TACHYON_WEIGHTS", "weights"), "q38-27b/hf")
Hk, Hv, D = 16, 48, R.D
C = (2 * Hk + Hv) * D
TOL = 1e-3


def rel(got, ref):
    return float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30))


def rand_params(rng):
    return dict(
        a=-np.exp(rng.uniform(-3.5, 5.0, Hv)), dt=rng.uniform(-8.0, 2.5, Hv),
        norm=1.0 + 0.1 * rng.standard_normal(D), conv=0.2 * rng.standard_normal((C, 4)),
        Hk=Hk, Hv=Hv, eps=1e-6,
    )


def rand_inputs(rng, B, T, zero_state=False):
    return dict(
        qkv=rng.standard_normal((B, T, C)), z=rng.standard_normal((B, T, Hv * D)),
        alpha=rng.standard_normal((B, T, Hv)), beta=rng.standard_normal((B, T, Hv)),
        conv_s=rng.standard_normal((B, C, 3)),
        state=np.zeros((B, Hv, D, D)) if zero_state else 0.1 * rng.standard_normal((B, Hv, D, D)),
    )


def f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def run_gpu(p, x, mode, slots=None, n_steps=0, n_acc=None):
    """入力を書いて gdncheck を走らせ、出力を返す。slots[b] = 列 b が使う状態の番号。
    mode split は前の T - n_steps トークンを塊ごと、残り n_steps を 1 トークンずつ。
    mode verify は T トークンを確かめてから n_acc[b] トークンだけ確定（state_full・conv_full は T 全部を進めた状態）"""
    B, T = x["qkv"].shape[:2]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        st, cs = x["state"], x["conv_s"]
        if slots is not None:  # 状態を置き場の順に並べ替えて渡す
            inv = np.argsort(slots)
            st, cs = st[inv], cs[inv]
            np.asarray(slots, dtype=np.int32).tofile(d / "slots.i32")
        if n_acc is not None:
            np.asarray(n_acc, dtype=np.int32).tofile(d / "n_acc.i32")
        for k, v in [("qkv", x["qkv"]), ("z", x["z"]), ("alpha", x["alpha"]), ("beta", x["beta"]), ("conv_s", cs),
                     ("state", st), ("a", p["a"]), ("dt", p["dt"]), ("norm", p["norm"]), ("conv", p["conv"])]:
            f32(v).tofile(d / f"{k}.f32")
        r = subprocess.run([BIN, "run", str(d), mode, str(B), str(T), str(n_steps)], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr)
        out = dict(
            y=np.fromfile(d / "y.f32", np.float32).reshape(B, T, Hv * D),
            conv_s=np.fromfile(d / "conv_s_out.f32", np.float32).reshape(B, C, 3),
            state=np.fromfile(d / "state_out.f32", np.float32).reshape(B, Hv, D, D),
        )
        if (d / "c.f32").exists():
            out["c"] = np.fromfile(d / "c.f32", np.float32).reshape(B, T, C)
        if (d / "state_full.f32").exists():
            out["state_full"] = np.fromfile(d / "state_full.f32", np.float32).reshape(B, Hv, D, D)
            out["conv_full"] = np.fromfile(d / "conv_full.f32", np.float32).reshape(B, C, 3)
        if slots is not None:
            out["state"], out["conv_s"] = out["state"][slots], out["conv_s"][slots]
        return out


def run_ref(p, x, chunked=False):
    """手本（float64。入力は GPU と同じ f32 に丸めたもの）"""
    B, T = x["qkv"].shape[:2]
    p64 = {k: (np.asarray(f32(v), np.float64) if isinstance(v, np.ndarray) else v) for k, v in p.items()}
    g = lambda k, b: np.asarray(f32(x[k][b]), np.float64)
    out = dict(y=[], conv_s=[], state=[], c=[])
    for b in range(B):
        y, cs, st = R.linear_attn(g("qkv", b), g("z", b), g("alpha", b), g("beta", b), p64, g("conv_s", b), g("state", b), chunked)
        c, _ = R.conv1d_silu(g("qkv", b), g("conv_s", b), p64["conv"])
        c = c.reshape(T, -1, D)
        nq = 2 * Hk
        c[:, :nq] = R.l2norm(c[:, :nq], p["eps"])
        out["y"].append(y), out["conv_s"].append(cs), out["state"].append(st), out["c"].append(c.reshape(T, C))
    return {k: np.stack(v) for k, v in out.items()}


def compare(name, got, ref, keys=("y", "state", "conv_s", "c")):
    errs = {k: rel(got[k], ref[k]) for k in keys if k in got and k in ref}
    ok = all(e < TOL for e in errs.values()) and all(np.isfinite(got[k]).all() for k in errs)
    print(f"{'OK ' if ok else 'NG '} {name:46s} " + "  ".join(f"{k} {e:.1e}" for k, e in errs.items()), flush=True)
    return ok


def case_random():
    rng = np.random.default_rng(1)
    p = rand_params(rng)
    ok = True
    for B in (1, 8, 32):
        x = rand_inputs(rng, B, 1)
        ref = run_ref(p, x)
        ok &= compare(f"decode B={B}", run_gpu(p, x, "decode"), ref)
        if B == 8:
            ok &= compare(f"decode B={B} slots 逆順", run_gpu(p, x, "decode", slots=np.arange(B)[::-1].copy()), ref)
    for T in (1, 2, 3, 5, 17, 63, 64, 65, 130, 300):
        for zero in (False, True):
            B = 2
            x = rand_inputs(rng, B, T, zero_state=zero)
            ref = run_ref(p, x)
            tag = f"T={T} B={B}{' 状態0' if zero else ''}"
            ok &= compare(f"prefill {tag}", run_gpu(p, x, "prefill"), ref)
            if T <= 65:
                ok &= compare(f"steps   {tag}", run_gpu(p, x, "steps"), ref)
                ok &= compare(f"seq     {tag}", run_gpu(p, x, "seq"), ref)
            if T >= 3:
                ok &= compare(f"split   {tag}（塊 {T - 2}＋1 トークン×2）", run_gpu(p, x, "split", n_steps=2), ref)
    x = rand_inputs(rng, 4, 130)
    ok &= compare("prefill T=130 B=4 slots 3,0,2,1", run_gpu(p, x, "prefill", slots=np.array([3, 0, 2, 1])), run_ref(p, x))
    # 下書きの確かめ（MTP 2 つなら T=3、DSpine b16 なら T=17）→ 受け入れた分だけ確定
    for T, n_acc in ((3, [1, 2, 3, 3]), (17, [1, 5, 17, 9])):
        x = rand_inputs(rng, 4, T)
        ok &= compare_verify(f"verify  T={T} B=4 受け入れ {n_acc}", p, x, n_acc)
    return ok


def compare_verify(name, p, x, n_acc):
    """verify の答え合わせ: y と T 全部の状態は T トークン分の手本、確定した状態は n_acc[b] トークン分の手本と"""
    got = run_gpu(p, x, "verify", n_acc=n_acc)
    full = run_ref(p, x)
    acc = [run_ref(p, {k: v[b:b + 1, :n] if v.ndim == 3 and k not in ("conv_s",) else v[b:b + 1]
                       for k, v in x.items()}) for b, n in enumerate(n_acc)]
    ref = dict(y=full["y"], c=full["c"], state_full=full["state"], conv_full=full["conv_s"],
               state=np.concatenate([a["state"] for a in acc]), conv_s=np.concatenate([a["conv_s"] for a in acc]))
    return compare(name, got, ref, keys=("y", "c", "state_full", "conv_full", "state", "conv_s"))


def project(L, h):
    """attn_norm の後の隠れ状態 h [B, T, 5120] から qkv・z・alpha・beta（f32 の掛け算。手本も GPU もこの値を使う）"""
    h = f32(h)
    return dict(qkv=h @ L["w_qkv"].T, z=h @ L["w_gate"].T, alpha=h @ L["w_alpha"].T, beta=h @ L["w_beta"].T)


def text_tokens(n):
    """本物の文（このリポジトリの日本語の説明とコード）を Qwen3.8 の語に切ったもの"""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{HF}/tokenizer.json")
    text = "".join((ROOT / f).read_text() for f in ("docs/qwen35-math.md", "src/gdn.rs", "kernels/gdn.cu", "docs/API.md", "src/kv.rs"))
    ids = tok.encode(text).ids
    assert len(ids) >= n, len(ids)
    return np.asarray(ids[:n])


def case_layer0(path):
    t0 = time.time()
    G = R.Gguf(path)
    L = G.layer(0)
    p = {k: L[k] for k in ("a", "dt", "norm", "conv", "Hk", "Hv", "eps")}
    assert (p["Hk"], p["Hv"]) == (Hk, Hv), (p["Hk"], p["Hv"])
    print(f"blk.0（{G.arch}）: eps {p['eps']:.1e}、ssm_a {p['a'].min():.2f}〜{p['a'].max():.4f}、dt {p['dt'].min():.2f}〜{p['dt'].max():.2f}"
          f"（読み込み {time.time() - t0:.0f} 秒）")
    n_all = 547 + 8 * 65 + 32 * 17
    ids = text_tokens(n_all)
    emb = G.get("token_embd.weight", rows=ids)
    h = R.rms_norm(emb.astype(np.float64), L["attn_norm"], p["eps"])  # 層 0 の入口（attn_norm の後）
    X = project(L, h)
    print(f"  語 {n_all} 個。qkv の大きさ（std）{X['qkv'].std():.3f}、z {X['z'].std():.3f}、"
          f"alpha {X['alpha'].mean():.2f}±{X['alpha'].std():.2f}、beta {X['beta'].mean():.2f}±{X['beta'].std():.2f}")
    ok = True

    def take(a, b, B, T):  # 文の [a, a + B*T) を B 本 × T に
        return {k: v[a:a + B * T].reshape(B, T, -1) for k, v in X.items()}

    # 1) 文の頭から 547 語（状態 0。llama-server と同じ長さの依頼）
    x = take(0, 0, 1, 547)
    x["conv_s"], x["state"] = np.zeros((1, C, 3)), np.zeros((1, Hv, D, D))
    ref = run_ref(p, x)
    pre, stp = run_gpu(p, x, "prefill"), run_gpu(p, x, "steps")
    ok &= compare("layer0 prefill B=1 T=547", pre, ref)
    ok &= compare("layer0 steps   B=1 T=547", stp, ref)
    ok &= compare("layer0 seq     B=1 T=547", run_gpu(p, x, "seq"), ref)
    ok &= compare("layer0 split   B=1 T=547（塊 539＋1×8）", run_gpu(p, x, "split", n_steps=8), ref)
    ok &= compare("layer0 prefill と steps（GPU どうし）", pre, stp, keys=("y", "state", "conv_s"))
    # 2) 8 本 × 65 語: 64 語を塊ごとに読んでから 1 語 decode（B=8）
    x = take(547, 0, 8, 65)
    x["conv_s"], x["state"] = np.zeros((8, C, 3)), np.zeros((8, Hv, D, D))
    ok &= compare("layer0 split   B=8 T=65（塊 64＋decode 1）", run_gpu(p, x, "split", n_steps=1), run_ref(p, x))
    # 3) 32 本 × 17 語: 16 語を読んでから 1 語 decode（B=32、slots はばらばら）
    x = take(547 + 8 * 65, 0, 32, 17)
    x["conv_s"], x["state"] = np.zeros((32, C, 3)), np.zeros((32, Hv, D, D))
    perm = np.random.default_rng(5).permutation(32)
    ok &= compare("layer0 split   B=32 T=17（塊 16＋decode 1・slots）", run_gpu(p, x, "split", slots=perm, n_steps=1),
                  run_ref(p, x))
    return ok


def case_long():
    rng = np.random.default_rng(3)
    p = rand_params(rng)
    # 減衰の弱い頭も残す（長く覚える頭で誤差が溜まらないか）
    p["a"][::4] = -0.03
    T = 4096
    x = rand_inputs(rng, 1, T)
    t0 = time.time()
    ref = run_ref(p, x)
    print(f"  手本 {time.time() - t0:.0f} 秒")
    pre, stp, seq = run_gpu(p, x, "prefill"), run_gpu(p, x, "steps"), run_gpu(p, x, "seq")
    ok = compare(f"long prefill T={T}", pre, ref)
    ok &= compare(f"long steps   T={T}", stp, ref)
    ok &= compare(f"long seq     T={T}", seq, ref)
    ok &= compare(f"long prefill と steps（GPU どうし）", pre, stp, keys=("y", "state", "conv_s"))
    return ok


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    path = sys.argv[2] if len(sys.argv) > 2 else GGUF
    ok = True
    if what in ("random", "all"):
        ok &= case_random()
    if what in ("layer0", "all"):
        ok &= case_layer0(path)
    if what in ("long", "all"):
        ok &= case_long()
    print("全部 OK" if ok else "⛔ NG あり")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
