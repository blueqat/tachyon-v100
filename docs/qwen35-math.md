# Qwen3.8-27B（GGUF arch `qwen35`）の計算 — CUDA エンジンの仕様

`tools/ref_qwen35.py`（numpy・float32）が実装している式そのもの。手本は llama.cpp
（unslothai/llama.cpp b12209-mix-b84c2bf の `src/models/qwen35.cpp`・`src/models/delta-net-base.cpp`・ggml-cpu の op）。
答え合わせの結果は末尾。

## 0. 形の決まり（GGUF の `qwen35.*`）

| 記号 | 値 | GGUF の鍵 |
|---|---|---|
| L | 64 層（block_count 65 = 64 ＋ MTP 1） | block_count・nextn_predict_layers |
| d | 5120 | embedding_length |
| d_ff | 17408 | feed_forward_length |
| V | 248320（入力と出力は別の行列。tie しない） | token_embd・output |
| ε | 1e-6（f32 にすると 9.99999997e-07） | attention.layer_norm_rms_epsilon |
| 普通の Attention | H = 24 頭・H_kv = 4・D_h = 256、RoPE は先頭 64 次元だけ、θ_base = 1e7、sections [11,11,10,0] | attention.*・rope.* |
| 線形 Attention（GDN） | key の頭 H_k = 16・value の頭 H_v = 48・D = 128（k も v も）、畳み込み K = 4、d_inner = 6144 | ssm.group_count・ssm.time_step_rank・ssm.state_size・ssm.conv_kernel・ssm.inner_size |
| 層の並び | `il % 4 == 3`（3, 7, 11, …, 63）が普通の Attention の 16 層、残り 48 層が GDN | full_attention_interval 4 |

重みの向き: GGUF の 2 次元テンソル `[ne0, ne1]` は「ne1 行 × ne0 列」。y = W·x は **行ごとに ne0（入力）方向の内積**。
量子化の塊（32 や 256 要素）も ne0 方向に並ぶ → GEMV は 1 行を頭から読めばよい。下では W ∈ R^{out×in} と書く。

型（UD-Q4_K_XL）: 大きな行列は Q5_K・IQ4_XS・Q4_K・Q6_K・Q3_K・IQ4_NL・IQ3_S・Q8_0 が層ごとに混ざる（`ref_qwen35.py` は全部 gguf-py の
dequantize で f32 に戻す）。norm・ssm_a・ssm_dt・ssm_conv1d は F32。`token_embd` Q4_K、`output` Q6_K。

## 1. 全体

```
h⁰_t = token_embd[tok_t]                         # (d,)。行を 1 本引いて戻すだけ（倍率なし）
for il in 0..63:
    a   = RMSNorm(h, attn_norm[il])               # 前の norm
    m   = GDN(a)        if il%4 != 3              # 線形 Attention（§3）
          Attn(a, pos)  if il%4 == 3              # 普通の Attention（§4）
    h   = h + m                                   # attn_residual
    b   = RMSNorm(h, post_attention_norm[il])
    h   = h + FFN(b)                              # l_out（ref の h[il]）
r      = RMSNorm(h, output_norm)                  # result_norm（MTP の入力 h もこれ）
logits = output · r                               # (V,)
```
- 層のあとに足す物は何も無い（hyper-connection・PLE・MoE は無い。密な FFN だけ）
- 残差の大きさ: |h| の最大は層 20 前後で 200 前後、最後は 1000 近くまで育つ（fp16 で溜めるなら要注意。残差は f32 で持つ）

## 2. 小さな op（ggml と同じ定義）

- `RMSNorm(x, w) = x / sqrt(mean(x²) + ε) · w`。平均は最後の軸（d・D_h・D）で、2 乗の和は倍精度（ggml の CPU）。
  GGUF の norm の重みは HF の `(1 + weight)` を足し込み済み（変換の `conversion/qwen.py` が `*norm.weight` に +1。
  値は blk.0.attn_norm 平均 0.97、output_norm 平均 1.94）→ そのまま掛ける。
  ⚠️ HF の safetensors から読むときは +1 する。`linear_attn.norm.weight`（= ssm_norm、平均 0.87）だけは足さない。
  `ssm_a` も変換で `−exp(A_log)` にしてある
- `silu(x) = x·σ(x)`、`σ(x) = 1/(1+e^{−x})`
- `softplus(x) = x（x > 20）、log(1 + eˣ)（それ以外）`
- `L2Norm(x) = x / sqrt(Σx² + ε)`（llama.cpp は `rms_norm(x, ε/n)/√n` と書くが同じ）

## 3. 線形 Attention = Gated DeltaNet（48 層）

重み（il 層）: `attn_qkv` [5120→10240]・`attn_gate` [5120→6144]・`ssm_alpha` [5120→48]・`ssm_beta` [5120→48]・
`ssm_dt.bias` (48)・`ssm_a` (48)・`ssm_conv1d` (10240 チャンネル × 4)・`ssm_norm` (128)・`ssm_out` [6144→5120]

1 語 t ごとに（a_t は §1 の norm 後の入力）:
```
c_t   = attn_qkv · a_t                 # (10240) = [ q 16×128 | k 16×128 | v 48×128 ]   (linear_attn_qkv_mixed)
z_t   = attn_gate · a_t                # (6144) = 48×128                                  (z)
β_t   = σ(ssm_beta · a_t)              # (48)                                             (beta_sigmoid)
g_t   = softplus(ssm_alpha · a_t + ssm_dt) · ssm_a        # (48)。ssm_a = −exp(A_log) < 0 なので g ≤ 0  (gate)
# 深さ方向の因果的な畳み込み（チャンネルごと、重み w[c, 0..3]、前の 3 語を覚える）
u_t[c] = Σ_{k=0..3} w[c,k] · c_{t−3+k}[c]      # 列の頭より前は 0（= 新しい列の畳み込みの状態は 0）
u_t    = silu(u_t)                                                                        (conv_output_silu)
q_t[i] = L2Norm(u_t[128i : 128i+128])          i = 0..15                                  (q_conv_predelta)
k_t[i] = L2Norm(u_t[2048+128i : …])            i = 0..15                                  (k_conv_predelta)
v_t[h] = u_t[4096+128h : …]                    h = 0..47                                  (v_conv_predelta)
# 頭ごとの再帰（h = 0..47、v の頭 h は q・k の頭 h mod 16 を使う ← 並びに注意）
S_h  ← S_h · exp(g_t[h])                       # S_h ∈ R^{128(k)×128(v)}、列の頭で 0
δ    = β_t[h] · (v_t[h] − S_hᵀ k_t[h mod 16])
S_h  ← S_h + k_t[h mod 16] · δᵀ
o_t[h] = S_hᵀ q_t[h mod 16] / √128                                                         (attn_output)
# 出口
y_t[h] = RMSNorm(o_t[h], ssm_norm) · silu(z_t[128h : 128h+128])   # norm は頭ごとに 128 個で、重みは全部の頭で共有  (final_output)
m_t    = ssm_out · concat_h y_t[h]                                                        (linear_attn_out)
```
- **頭の対応は `h mod 16`（タイル型）**。HF の重みは「k の頭ごとに v の頭 3 つ」の並び（`h // 3`）だが、
  llama.cpp の変換（`conversion/qwen.py` の `_LinearAttentionVReorderBase`）が v の頭をタイル型に並べ替えて GGUF に書いている:
  in_proj_qkv の v の行・in_proj_z の行・in_proj_a/b の行・A_log・dt_bias・conv1d の v のチャンネル・out_proj の列。
  GGUF の重みを使う限り `h mod 16`。HF から読むなら同じ並べ替えをするか `h // 3` にする
- 状態（1 列あたり・1 層あたり）: 畳み込み 3×10240 f32（120KB）、S は 48×128×128 f32（3MB）→ 48 層で約 151MB/人。
  llama.cpp は S を転置して `[h][v][k]`（k が連続）で持つ。どちらで持っても式は同じ
- **畳み込みの状態は「前の 3 語の入力 c」（畳み込む前・silu の前の qkv）**で、出力ではない。新しい列の頭で畳み込みの状態も S も 0。
  プレフィルを塊（ubatch 512 など）に分けたとき・デコードの 1 語ごとには、**同じ列の状態を次の塊・次の語へ持ち越す**
  （`ref_qwen35.py` は列を一度に通すので持ち越しは無い。`dconv(x, w, state)`・`gdn_scan` の S0 は持ち越しを受け取れる）
- 長い列のプレフィルは llama.cpp では 64 語ずつの塊の式（chunked、`build_delta_net_chunking`）か融合カーネル（`ggml_gated_delta_net`）だが、
  上の 1 語ずつの再帰と数学的に同じ（`ref_qwen35.py` は再帰で計算）。1/√128 は融合カーネルは o に、chunked・1 語版は先に q に掛ける（同じ）
- `ssm_a` は GGUF に −exp(A_log) として入っている（そのまま掛ける）。`ssm_dt` は bias（足す）

## 4. 普通の Attention（16 層）

重み: `attn_q` [5120→12288]・`attn_k` [5120→1024]・`attn_v` [5120→1024]・`attn_q_norm` (256)・`attn_k_norm` (256)・`attn_output` [6144→5120]
```
f_t      = attn_q · a_t                 # (12288) = 24 頭 × [ q(256) | gate(256) ]（頭ごとに q と gate が交互）  (Qcur_full)
q_t[h]   = RMSNorm(f_t[512h : 512h+256], attn_q_norm)        # 頭ごとに 256 個で norm、重みは全部の頭で共有
gate_t[h]= f_t[512h+256 : 512h+512]
k_t[j]   = RMSNorm((attn_k · a_t)[256j : 256j+256], attn_k_norm)   j = 0..3
v_t[j]   = (attn_v · a_t)[256j : 256j+256]
q, k に RoPE（下）                                                                         (Qcur・Kcur)
（llama.cpp は k_t・v_t を f16 の KV キャッシュに入れる）
o_t[h]   = Σ_{s≤t} softmax_s( q_t[h]·k_s[h//6] / 16 ) · v_s[h//6]      # 因果、GQA は h // 6（グループ型）  (attn_pregate)
m_t      = attn_output · concat_h( o_t[h] ⊙ σ(gate_t[h]) )                                (attn_gated → attn_output)
```
- 倍率は 1/√256 = 1/16（`qwen35.cpp`: `kq_scale = f_attention_scale == 0 ? 1/sqrt(n_embd_head) : f_attention_scale`、
  GGUF に attention.scale は無い）。softcap・sink・ALiBi は無い
- **GQA は `h // 6`**（q の頭 0〜5 が KV の頭 0）。GDN の `h mod 16` と向きが逆なので取り違えない
- KV キャッシュ: 16 層 × 4 頭 × 256 × (K,V) × fp16 = 64KB/語

### RoPE（IMRoPE、文字だけなら普通の NEOX と同じ）
- 頭の 256 次元のうち **先頭 64 次元だけ** 回す。組は (i, i+32)、i = 0..31（NEOX 型）。64〜255 次元はそのまま
- θ_i = p · base^{−2i/64}、base = 1e7（ggml の CPU は base^{−2/64} を f32 で i 回掛けて作る。GPU は powf。差は無視できる）
- `x'[i] = x[i]·cosθ_i − x[i+32]·sinθ_i`、`x'[i+32] = x[i]·sinθ_i + x[i+32]·cosθ_i`
- M-RoPE（画像）: 位置は (t, h, w, e) の 4 つ。IMRoPE は **交互**（連続のブロックではない）: 組 i は `i%3==0 かつ i<33 → t`、
  `i%3==1 かつ i<33 → h`、`i%3==2 かつ i<30 → w`、残りは e（sections [11,11,10,0] だと 32 組が t・h・w でちょうど埋まり、e に当たる組は無い。
  ggml-cpu `ggml_mrope_cache_init` の `is_imrope`）。**文字だけなら t = h = w = p なので上の普通の式になる**。画像を入れる時はここを分ける
- yarn・freq_factors は無い（attn_factor 1、ext_factor 0）

## 5. FFN（全 64 層、密）
```
FFN(b) = ffn_down · ( silu(ffn_gate · b) ⊙ (ffn_up · b) )      # [5120→17408]×2 と [17408→5120]
```
（swiglu の clamp は無い）

## 6. 出口
- `r = RMSNorm(h_L, output_norm)`、`logits = output · r`（output は Q6_K [5120→248320]）
- llama.cpp は出力する位置だけ最後の層で切る（最後の層の FFN を全部の語で計算しなくてよい）。結果は同じ

## 7. MTP（下書き、blk.64。2026-11-10 に llama.cpp そのものと答え合わせ済み。`tools/ref_mtp.py`・`tools/lcpp_mtp.cpp`・`src/mtp.rs`）
源: unsloth b12209 の `src/models/qwen35.cpp`（graph の h_nextn・graph_mtp）、`common/speculative.cpp`（common_speculative_impl_draft_mtp）、
`tools/server/server-context.cpp`（下書きの後・受け取りの後の seq_rm）。
入力は本体の位置 p の **h_nextn[p] = RMSNorm(h_L[p], output_norm)（norm の後 = §6 の r）** と語。
⚠️ `llama-cparams.h` の注釈（"hidden state before the final output norm"）は qwen35 には当たらない（graph が output_norm の後で h_nextn を取る）
```
MTP の 1 行（位置 p・語 x・状態 h）:
e   = RMSNorm(token_embd[x], nextn.enorm)        # nextn.embed_tokens があればそれ（この 2 つの GGUF には無い）
hh  = RMSNorm(h, nextn.hnorm)
x0  = nextn.eh_proj · concat(e, hh)               # [10240→5120]。e が前（ggml_concat(e_norm, h_norm, 0)）
x1  = x0 + Attn_mtp(RMSNorm(x0, attn_norm))      # §4 と同じ普通の Attention（q と gate・q/k norm・IMRoPE・GQA h//6・1/16・σ(gate)）。MTP だけの KV
x2  = x1 + FFN_mtp(RMSNorm(x1, post_attention_norm))
h'  = RMSNorm(x2, nextn.shared_head_norm)         # MTP の h_nextn。下書きをつなぐときの次の h
logits = output · h'                              # shared_head_head が無いので、その GGUF の output
```
進め方（llama-server の `--spec-type draft-mtp`。qwen35 は is_mem_shared も chain_heads も偽 = 1 つの頭・KV は自分の物）:
- **process**（本体が通した batch ごと。プロンプトの塊も確かめも）: 語 x_p ごとに MTP の行 (x_p, 位置 p, h = h_nextn[p-1])。列の頭 p = 0 は h = 0
  （pending_h の初め。llama-server は同じ枠を使った前の依頼の最後の h が残る）→ MTP の KV の位置 p は「本体の h_{p-1} と x_p」から作る
- **draft**: 1 段目 (id_last, 位置 n, h = h_nextn[n-1])、j 段目 (d_{j-1}, 位置 n+j-1, h = 前の段の h')。下書きは argmax（top-k 10 の 1 位）、
  p_min 0 なので毎回 k 語
- 下書きの後に seq_rm(ctx_dft, n, -1)（下書きの KV を消す）→ 本体が [id_last, d1..dk] を確かめる → process がその k+1 行を本体の h で書き直す →
  受け取り a 個なら seq_rm(n+a+1, -1)。次の pending = 本体の h_nextn[n+a]、次の 1 段目は位置 n+a+1
- 位置は語の位置（vLLM は h の位置 = 1 つ前に置く）。RoPE は相対なので、全部の行が同じだけずれても Attention は変わらない
- 測った 52.6 tok/s は `--spec-draft-n-max 2`（ログの mean len 2.13 = 1 + 76/67）
重み: 本体の GGUF の blk.64（Q6_K・Q8_0。0.35GB）と、別ファイル `mtp-Qwen3.8-27B-Q4_0.gguf`（blk.64 は Q6_K・Q4_K で 0.27GB、
token_embd・output は Q3_K で 0.55GB ずつ。arch は同じ qwen35）。**llama-server の 52.6 tok/s（`-md mtp-…gguf`、`logs/q27.log` の
「loading draft model」）は別ファイル**で、`-md` は別の model として読むので埋め込みも出力もそのファイルの Q3_K（本体の物は使わない）。
`-md` なしの draft-mtp は本体の GGUF の blk.64 で、埋め込みと出力も本体の物

答え合わせ（#3 の 55 語。llama.cpp を CPU だけで通した MTP の節と。入力の h は llama.cpp の本体の h_nextn）:
- 掛け算の無い所（enorm・hnorm・concat・attn_norm・q_norm・shared_head_norm）: 相対 3.6e-8〜5.6e-8（f32 の丸めだけ）。埋め込みの行は完全一致
- 掛け算 1 回で約 1%（llama.cpp の CPU は活性を Q8_K にしてから掛ける。§9 と同じ）、h' で 3.4%、logits で 2.7〜2.9%
- 行ごとの argmax: main 54/55・file 55/55。つなぎの 3 語（11・3750・430）は main・file とも llama.cpp と同じ
- Attention の中だけ（llama.cpp の rope 前の q・k・v から）8.2e-4。因果を 1 つ先まで見ると 0.51 → 位置 p の行は MTP の KV の 0..p を見る
- GPU（`src/mtp.rs`）と参照: h' ≤ 5.4e-4・logits ≤ 1.8e-4、argmax と下書きの回し方は全部同じ

## 8. 数値の扱いの違い（llama-server と比べるときに効く所）
- llama.cpp（CUDA）は重みが量子化された行列の掛け算で、**入力（活性）も Q8_1 に量子化**してから整数の内積をとる（MMVQ・MMQ）。
  バッチが大きいと f16 に戻して cuBLAS。参照（f32 のまま）との差はここが大半
- K・V は f16 のキャッシュ。Flash Attention（V100 でも tile/wmma 版が動く）は Q も f16 にする
- GDN の状態・畳み込みの状態は f32
- 参照の `--kv-f16`（ref_compare.py は既定）は K・V だけ f16 に丸める（Q は丸めない。§9 の Attention の中の 5e-4 はこの差）
- 参照の丸めの順は ggml と少し違う所がある（L2Norm を `x/sqrt(Σx²+ε)` で直接、RoPE の角度は ggml の CPU と同じ掛け重ね）。どれも f32 の丸めの範囲（§9）

## 9. 答え合わせ（2026-11-10。やり方は tools/ref_qwen35.py・tools/lcpp_dump.cpp）

**llama-server（V100・unsloth b12209・UD-Q4_K_XL）と、8 つの文 × 32 語（温度 0）**。参照は K・V を f16 に丸めて計算
| 文 | 語数 | 次の語の一致 | top-1 logprob の差 最大 / 平均 |
|---|---|---|---|
| #0 The capital of France is | 5 | 32/32 | 0.016 / 0.001 |
| #1 日本で一番高い山は | 5 | 32/32 | 0.045 / 0.010 |
| #2 def quicksort(arr): | 6 | 32/32 | 0.010 / 0.002 |
| #3 chat（考えなし） | 23 | 32/32 | 0.037 / 0.006 |
| #4 Once upon a time | 13 | 32/32 | 0.028 / 0.005 |
| #5 chat（考えあり・日本語） | 20 | 32/32 | 0.028 / 0.001 |
| #6 kernels/quant.cuh の頭 | 600 | 32/32 | 0.030 / 0.003 |
| #7 長い日本語の文書の頭 | 1500 | 32/32 | 0.051 / 0.005 |
| 計 | | **256/256（100%）** | **254 位置のうち 253 が 0.05 以内**、最大 0.0514、平均 0.0042（中央 0.0004・90% 0.014・99% 0.041） |
- 0.05 を超えた 1 つ（#7 の 8 語目）は 1 位の確率が 0.17（logprob −1.75）しかない所で、差は 0.0514
- **サーバー自身の揺れ**: 同じ列をデコードの道（1 語ずつ）とプレフィルの道（まとめて 1 回）で出させると、1 位が 1 か所入れ替わり、
  top-1 logprob の差は最大 0.072・平均 0.0048（`ref_compare.py prefill`）→ 参照とサーバーの差（最大 0.051・平均 0.0042）はサーバーの揺れより小さい
- 上位 20 の下の方（確率の小さい語）は差が大きい（最大 1.8・平均 0.12）。量子化した活性で内積をとる GPU 側の雑音が小さい logit ほど効く

**llama.cpp（同じ源を CPU だけでビルド、`tools/lcpp_dump.cpp`）の途中の値と、#3 の 31 語で**
- 行列の掛け算を含まない所（llama.cpp の節を入力に参照の関数だけを通す）: RMSNorm・softplus の gate・畳み込み＋silu・L2Norm・
  GDN の走査・頭ごとの norm×silu(z)・q_norm・RoPE は **相対誤差 1e-8〜2e-7（f32 の丸めだけ）**、Attention の中（softmax・GQA）は 5e-4
  （llama.cpp の CPU の Flash Attention が Q・K を f16 で掛けるため）→ **式は llama.cpp と同じ**
- 掛け算は 1 回で 7e-3 ずれる（llama.cpp の CPU は活性を Q8_K = 256 個ごとの 8bit にしてから掛ける。残差の外れ値が大きいので荒い）。
  層を重ねると残差の差は 1%（層 10）→ 3%（層 40）→ 3〜6%（層 54〜63）、ロジットは 2.5%
- この位置で llama.cpp の CPU は 1 位が入れ替わる（8964 −0.66）が、GPU のサーバー（16078 −0.661）と参照（16078 −0.658）は同じ
  → **CPU の llama.cpp より参照の方が GPU のサーバーに近い**。CUDA エンジンの手本は参照（f32）でよい
