"""Gated DeltaNet（線形 Attention）の手本（numpy・float64）。Qwen3.8-27B（GGUF の arch qwen35）向け。

llama.cpp（unslothai/llama.cpp、b12228 で読んだ。比べる相手の llama-server は b12209）の
src/models/qwen35.cpp の build_layer_attn_linear と、そこから呼ぶ ggml の演算
（ggml_ssm_conv・silu・build_gdn_l2_norm・softplus・ggml_gated_delta_net・build_norm_gated）を
そのまま書き写したもの。kernels/gdn.cu の答え合わせに使う（tools/gdncheck.py）。

並び（すべて C の順。B = 列の本数、T = トークン数、C = 2*Hk*D + Hv*D = 10240）
  qkv    [B, T, C]                attn_qkv の出力。q（Hk 頭）・k（Hk 頭）・v（Hv 頭）の順
  z      [B, T, Hv*D]             attn_gate の出力（出口の gate）
  alpha  [B, T, Hv]               ssm_alpha の出力
  beta   [B, T, Hv]               ssm_beta の出力
  conv_w [C, 4]                   ssm_conv1d（GGUF の [4, C]）。w[c, 3] が今のトークンに掛かる
  conv_s [B, C, 3]                畳み込みの状態（古い順に直前 3 トークン。llama.cpp の r_l と同じ並び）
  state  [B, Hv, D, D]            ggml と同じ転置の並び: state[h, 列 j(v), i(k)] = S[i, j]（s_l と同じ並び）
  y      [B, T, Hv*D]             ssm_out に入る前の値（final_output）
v の頭 h が使う q・k の頭は h % Hk（ggml_repeat_4d は並べて繰り返す。CUDA の融合版も iq1 = h % neq1）。
qwen4exp（Flash-Next）との違い: 出口の gate が sigmoid(z) でなく silu(z)（build_norm_gated）。
"""
import numpy as np

D = 128   # 頭の幅（key・value とも。ssm.state_size）
CK = 4    # 畳み込みの幅（ssm.conv_kernel）
CS = 64   # 塊の長さ（ggml の build_delta_net_chunking と同じ）


def softplus(x):
    # ggml: x > 20 ならそのまま、それ以外は log1p(exp(x))
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def conv1d_silu(x, conv_s, conv_w):
    """ggml_ssm_conv（状態 3 列と入力をつないで幅 4 で畳む）＋ silu。
    x [T, C]、conv_s [C, 3]、conv_w [C, 4] → (y [T, C], 新しい状態 [C, 3])"""
    T = x.shape[0]
    xin = np.concatenate([conv_s.T, x], axis=0)  # [3+T, C]
    y = np.zeros_like(x)
    for j in range(CK):
        y += xin[j:j + T] * conv_w[:, j]
    return silu(y), xin[T:T + CK - 1].T.copy()


def l2norm(x, eps):
    """build_gdn_l2_norm: rms_norm(x, eps/n) / sqrt(n) = x / sqrt(sum(x^2) + eps)（最後の軸）"""
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + eps)


def rms_norm(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def gates(alpha, beta, ssm_a, dt_bias):
    """g = softplus(alpha + dt_bias) * ssm_a（ssm_a = -exp(A_log)、負）、b = sigmoid(beta)"""
    return softplus(alpha + dt_bias) * ssm_a, sigmoid(beta)


def split_qkv(c, Hk, Hv, eps):
    """conv の後 [T, C] を q・k（l2 正規化）・v に分け、q・k を v の頭の数に広げる（h % Hk）"""
    T = c.shape[0]
    q = l2norm(c[:, :Hk * D].reshape(T, Hk, D), eps)
    k = l2norm(c[:, Hk * D:2 * Hk * D].reshape(T, Hk, D), eps)
    v = c[:, 2 * Hk * D:].reshape(T, Hv, D)
    idx = np.arange(Hv) % Hk
    return q[:, idx], k[:, idx], v


def gdn_recurrent(q, k, v, g, b, state):
    """1 トークンずつ（ggml の gated_delta_net と同じ）。
    q, k, v [T, Hv, D]（q・k は l2 正規化済み）、g, b [T, Hv]、state [Hv, D(j), D(i)]
    → (o [T, Hv, D], 新しい state)"""
    M = state.copy()
    T = q.shape[0]
    o = np.zeros_like(v)
    scale = 1.0 / np.sqrt(D)
    for t in range(T):
        M *= np.exp(g[t])[:, None, None]
        kv = np.einsum("hji,hi->hj", M, k[t])
        d = (v[t] - kv) * b[t][:, None]
        M += d[:, :, None] * k[t][:, None, :]
        o[t] = np.einsum("hji,hi->hj", M, q[t]) * scale
    return o, M


def gdn_chunked(q, k, v, g, b, state):
    """塊ごと（CS = 64。ggml の build_delta_net_chunking と同じ式）。gdn_recurrent と同じ答えになる。
    kernels/gdn.cu の prefill はこの形。S は数式の向き S[i(k), j(v)]（= state の転置）"""
    T, Hv, _ = q.shape
    pad = (-T) % CS
    P = lambda a: np.concatenate([a, np.zeros((pad,) + a.shape[1:])], 0) if pad else a
    q, k, v, g, b = P(q), P(k), P(v), P(g), P(b)
    q = q / np.sqrt(D)
    S = state.transpose(0, 2, 1).copy()  # [Hv, i, j]
    o = np.zeros_like(v)
    tri_strict = np.tril(np.ones((CS, CS)), -1)
    tri_incl = np.tril(np.ones((CS, CS)))
    for c0 in range(0, T + pad, CS):
        sl = slice(c0, c0 + CS)
        for h in range(Hv):
            qc, kc, vc, gc, bc = q[sl, h], k[sl, h], v[sl, h], g[sl, h], b[sl, h]
            gcs = np.cumsum(gc)
            dec = np.exp(np.minimum(gcs[:, None] - gcs[None, :], 0.0))  # 下三角だけ使う（その所は ≤ 0）
            L = (bc[:, None] * (kc @ kc.T)) * dec * tri_strict     # β_i (k_i·k_j) e^{G_i-G_j}, i > j
            Aqk = (qc @ kc.T) * dec * tri_incl                      # (q_i·k_j) e^{G_i-G_j}, i ≥ j
            Tm = np.linalg.inv(np.eye(CS) + L)                      # (I + L)^-1
            U = Tm @ (vc * bc[:, None])
            W = Tm @ (kc * (bc * np.exp(gcs))[:, None])
            vn = U - W @ S[h]
            o[sl, h] = (qc * np.exp(gcs)[:, None]) @ S[h] + Aqk @ vn
            S[h] = S[h] * np.exp(gcs[-1]) + (kc * np.exp(gcs[-1] - gcs)[:, None]).T @ vn
    return o[:T], S.transpose(0, 2, 1).copy()


def gated_norm(o, z, norm_w, eps):
    """build_norm_gated（qwen35）: rms_norm(o) * w * silu(z)（頭ごと 128 で正規化）"""
    return rms_norm(o, norm_w, eps) * silu(z)


def linear_attn(qkv, z, alpha, beta, p, conv_s, state, chunked=False):
    """線形層の中身（ssm_out の手前まで）を 1 本の列について。
    qkv [T, C]、z [T, Hv*D]、alpha・beta [T, Hv]、p = dict(a, dt, norm, conv, Hk, Hv, eps)
    → (y [T, Hv*D], 新しい conv_s, 新しい state)"""
    Hk, Hv, eps = p["Hk"], p["Hv"], p["eps"]
    T = qkv.shape[0]
    c, conv_s2 = conv1d_silu(qkv, conv_s, p["conv"])
    q, k, v = split_qkv(c, Hk, Hv, eps)
    g, b = gates(alpha, beta, p["a"], p["dt"])
    o, state2 = (gdn_chunked if chunked else gdn_recurrent)(q, k, v, g, b, state)
    y = gated_norm(o, z.reshape(T, Hv, D), p["norm"], eps).reshape(T, Hv * D)
    return y, conv_s2, state2


class Gguf:
    """GGUF から線形層の重みと埋め込みの行を読む（mmap。f32 に戻す。手本は gguf-py の dequantize）"""

    def __init__(self, path):
        from gguf import GGUFReader
        self.r = GGUFReader(path)
        self.t = {x.name: x for x in self.r.tensors}
        self.arch = str(bytes(self.r.fields["general.architecture"].parts[-1]), "utf-8")

    def meta(self, key):
        return self.r.fields[f"{self.arch}.{key}"].parts[-1][0]

    def get(self, name, rows=None):
        """[n_rows, k] の f32。rows を渡すとその行だけ戻す（埋め込み用）"""
        from gguf.quants import dequantize
        x = self.t[name]
        shape = [int(n) for n in reversed(x.shape.tolist())]
        raw = np.asarray(x.data)
        if rows is not None:
            raw = raw.reshape(shape[0], -1)[np.asarray(rows)]
            shape = [len(rows)] + shape[1:]
        a = dequantize(raw, x.tensor_type)
        return np.asarray(a, dtype=np.float32).reshape(shape)

    def layer(self, il):
        """線形層 il の重み。Hk = ssm.group_count、Hv = ssm.time_step_rank"""
        g = lambda n: self.get(f"blk.{il}.{n}")
        assert int(self.meta("ssm.state_size")) == D and int(self.meta("ssm.conv_kernel")) == CK
        return dict(
            a=g("ssm_a"), dt=g("ssm_dt.bias"), norm=g("ssm_norm.weight"), conv=g("ssm_conv1d.weight"),
            attn_norm=g("attn_norm.weight"),
            w_alpha=g("ssm_alpha.weight"), w_beta=g("ssm_beta.weight"),
            w_qkv=g("attn_qkv.weight"), w_gate=g("attn_gate.weight"),
            Hk=int(self.meta("ssm.group_count")), Hv=int(self.meta("ssm.time_step_rank")),
            eps=float(self.meta("attention.layer_norm_rms_epsilon")),
        )


if __name__ == "__main__":
    # 自分の確かめ: 塊ごと＝1 トークンずつ
    rng = np.random.default_rng(0)
    Hk, Hv, T = 2, 6, 150
    q = l2norm(rng.standard_normal((T, Hv, D)), 1e-6)
    k = l2norm(rng.standard_normal((T, Hv, D)), 1e-6)
    v = rng.standard_normal((T, Hv, D))
    g = -np.abs(rng.standard_normal((T, Hv))) * 3
    b = sigmoid(rng.standard_normal((T, Hv)))
    s0 = rng.standard_normal((Hv, D, D)) * 0.1
    o1, s1 = gdn_recurrent(q, k, v, g, b, s0)
    o2, s2 = gdn_chunked(q, k, v, g, b, s0)
    print("chunked vs recurrent: o", np.abs(o1 - o2).max() / np.abs(o1).max(), "state", np.abs(s1 - s2).max() / np.abs(s1).max())
