[English](README.md) | 日本語

# Tachyon-v100

NVIDIA V100（sm_70）向けに手書きのCUDAカーネルを実装した、Rust製のスクラッチ構築LLM推論エンジンです。

## ステータス

研究プレビューです。現在のリリース: **Tachyon V100 Loop**（ループ型トランスフォーマーモデルを追加、[Loopモデル](#loopモデルtachyon-v100-loop)を参照）。

### Tachyon V100 Loopでの変更点

- Loopモデル: SenseNova-Loop-9B（`sensenova_loop`）とInkling-2-Mini（`inkling2_loop`）。`--repeat-layers`（研究用オプション、デフォルトではオフ）
- `--kv8`: int8 KVキャッシュ（32要素スケール、MTPヘッドはfp16のまま）
- YaRN RoPEスケーリング（`--rope-scaling yarn --rope-factor F --rope-orig N`、またはGGUFの`rope.scaling.*`キーから指定）
- 信頼済みプロキシ背後でのエンドユーザーごとのプレフィックスキャッシュ所有者（`--prefix-owner-accounts`、ヘッダー`x-tachyon-prefix-owner`、デフォルトではオフ）
  自前のプロキシだけが鍵を持つアカウントだけを登録し、ヘッダーはそのプロキシ自身が設定（上書き）してください。そのアカウントでヘッダーを送れるクライアントは、誰のキャッシュを共有するかを選べてしまいます
- MTPドラフトとターゲットで共有されるシード付きサンプリングノイズ（`TACHYON_HASH_RNG`、デフォルトではオフ）、RAMプレフィックスティアの修正
- Qwen3.8-27BとQwen3.6-35B-A3B: 社内のゴールデンプロンプト12本（greedy・64トークン、プレフィル経路での教師強制による上位20のlogprob）で、ループ対応を入れる前のエンジンとビット単位で同一の出力

## V100を選ぶ理由

V100はvLLM >= 0.20でサポートが打ち切られており、FlashAttention 2/3もV100に対応していません。V100にはbf16がなく、int8/fp8テンソルコアもなく、`cp.async`もありません。Tachyon-v100は独自のfp16カーネル（Volta `mma`）、独自のFA1スタイルのタイルドアテンション、そしてGGUFブロックを直接読み取る量子化GEMVを提供します。

## 対応モデル

モデルごとのページ（取り方・ベンチ・ライセンス）: [docs/models/qwen3.8-27b.md](docs/models/qwen3.8-27b.md)・[docs/models/qwen3.6-35b-a3b.md](docs/models/qwen3.6-35b-a3b.md)・[docs/models/sensenova-loop-9b.md](docs/models/sensenova-loop-9b.md)・[docs/models/inkling-2-mini.md](docs/models/inkling-2-mini.md)（Loopモデルは下記）

| モデル | GGUFファイル | 備考 |
|---|---|---|
| Qwen3.8-27B | `unsloth/Qwen3.8-27B-GGUF` UD-Q4_K_XL（17.5 GB）+ MTP Q4_0 + mmproj-F16 | Dense、Gated DeltaNet線形アテンションとフルアテンションの3:1構成、64層、MTPヘッド |
| Qwen3.6-35B-A3B | `ggml-org/Qwen3.6-35B-A3B-GGUF` Q4_K_M + `mtp-Qwen3.6-35B-A3B-Q8_0` + mmproj（F16。Q8_0のmmprojは`tools/mmproj_to_f16.py`で変換） | MoE 256エキスパート top-8 + 共有エキスパート、同一のGDN/アテンションブロック、MTPヘッド |

llama.cppエコシステムの無改造GGUFを読み込みます。

## Loopモデル（Tachyon V100 Loop）

ループ型トランスフォーマーは、レイヤースタックの一部をトークンごとに共有ウェイトで複数回実行します。TachyonはウェイトをVRAMに1回だけ保持し、各実行に独自のKVキャッシュとリカレント状態を割り当てるため、プレフィル、CUDA Graphデコード、MTP、プレフィックスキャッシュ、バッチ処理がそのまま動作します。
アーキテクチャの詳細: [docs/LOOP.md](docs/LOOP.md)。

| モデル | ウェイト（上流からダウンロード。ここでは再配布していません） | ループ | ページ |
|---|---|---|---|
| SenseNova-Loop-9B | [sensenova/SenseNova-Loop-9B-GGUF](https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF) Q4_K_MまたはQ8_0（MTPヘッドを内包）+ mmproj-F16。トークナイザーは[sensenova/SenseNova-Loop-9B-AWQ](https://huggingface.co/sensenova/SenseNova-Loop-9B-AWQ)から | レイヤー4..27をx2 = 56回実行、ツーストリームmHC（トークンごとのSinkhornミキシング） | [docs/models/sensenova-loop-9b.md](docs/models/sensenova-loop-9b.md) |
| Inkling-2-Mini | [ggml-org/Inkling-2-Mini-GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) IQ4_NL（ggml-orgによる公式ウェイトのGGUF変換、アーキテクチャ`inkling2_loop`）。トークナイザーは[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)から | MoE 128エキスパートtop-8。レイヤー6..17をx3 = 48回実行、depth-attentionキャリー付きのリスタートパス。RoPEなし（学習済み位置バイアス） | [docs/models/inkling-2-mini.md](docs/models/inkling-2-mini.md) |

Inkling-2-Miniは[Thinking Machines Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/)の対象でもあります（モデルを使用することで本ポリシーに同意したものとみなされます。他者にデプロイする場合、あなたはその利用者のコンプライアンスに責任を負い、必要な場合には既知の制限事項およびAIと対話していることをユーザーに伝えなければならず、財務、法務、雇用、医療、住宅、保険、社会サービスに関する決定など、人の権利に影響を与える自動意思決定に本モデルを使用してはなりません）。モデルページを参照してください（ポリシーの原文は英語で、英語版が正文です）。

```sh
# SenseNova-Loop-9B (Apache-2.0)
huggingface-cli download sensenova/SenseNova-Loop-9B-GGUF SenseNova-Loop-9B-Q4_K_M.gguf mmproj-SenseNova-Loop-9B-F16.gguf --local-dir weights/loop9b
huggingface-cli download sensenova/SenseNova-Loop-9B-AWQ tokenizer.json tokenizer_config.json chat_template.jinja --local-dir weights/loop9b/hf
CUDA_VISIBLE_DEVICES=0 target/release/tachyon serve --gguf weights/loop9b/SenseNova-Loop-9B-Q4_K_M.gguf --hf weights/loop9b/hf \
  --mmproj weights/loop9b/mmproj-SenseNova-Loop-9B-F16.gguf --draft-vocab data/draft_vocab_q38.txt \
  --model-name sensenova-loop-9b --keys keys.json --host 127.0.0.1 --port 8080

# Inkling-2-Mini (Apache-2.0 + Model AUP)
huggingface-cli download ggml-org/Inkling-2-Mini-GGUF Inkling-2-Mini-IQ4_NL.gguf --local-dir weights/inkling2mini
huggingface-cli download thinkingmachines/Inkling-2-Mini tokenizer.json tokenizer_config.json chat_template.jinja --local-dir weights/inkling2mini/hf
CUDA_VISIBLE_DEVICES=0 target/release/tachyon serve --gguf weights/inkling2mini/Inkling-2-Mini-IQ4_NL.gguf --hf weights/inkling2mini/hf \
  --model-name inkling-2-mini --kv8 --keys keys.json --host 127.0.0.1 --port 8080 --ctx 32768 --max-seqs 8 --max-running 8
```



検証済みの数値（Tesla V100-SXM3-32GB 1台、CUDA 12.5、temperature 0。予備的な社内測定値であり、詳細と条件はモデルページに記載）:

| | SenseNova-Loop-9B Q4_K_M | Inkling-2-Mini IQ4_NL |
|---|---|---|
| CPU f32の公式コードに対するTop-1一致率（5プロンプト） | Q8_0で107/107（Q4_K_Mで101/107） | 104/107（97.2%） |
| デコード、1ユーザー、ドラフトなし | 55 tok/s | 62-64 tok/s |
| MTPありデコード | 98 tok/s（k=3） | —（ドラフトヘッドは未移植） |
| TTFT、8Kトークンプロンプト | 4.1 s | 6.7 s |
| 同時ストリーム（合計） | 4 / 8 / 16で197 / 296 / 391 tok/s | 4 / 8 / 16 / 32で154 / 196 / 238 / 270 tok/s |

制限事項: Inkling-2-Miniのビジョンおよびそのブロックドラフト（MTP）ヘッドはまだ移植されていません。`--repeat-layers START:END:N`（ループ用に学習されていないモデルのレイヤーを繰り返す）は研究用のみに含まれています。Qwen3.8-27Bでは、試した16通り（2〜4回・4〜32層）のすべてで次トークンのNLLが悪化し、推論問題の正答率も下がりました（docs/LOOP.md、セクション4）。

## 機能

- OpenAI互換の`/v1/chat/completions`：ストリーミング、usage、`tools`/`tool_choice`、`response_format`のjson_object/json_schemaサブセット（トークンマスキング経由）、`chat_template_kwargs`（`enable_thinking`/`reasoning_effort`）、シンキングバジェット、`stop`、`seed`、`top_k`/`min_p`、presence/frequency/repetitionペナルティ、mmproj経由のdata URI画像入力
- `/v1/models`、`/health`
- APIキーはsha256で保存、アカウント単位のレート/同時実行制限付き
- アカウント単位で分離されたプレフィックスキャッシュ
- RAMティア付きページドKVキャッシュ
- 継続的バッチング（最大4〜8シーケンス）
- CUDA Graphデコード
- MTP投機的デコード（greedy出力はドラフトなしと同一）
- GPUサンプリング

非対応：`logprobs`、`n>1`、`/props`。単一GPUのみ（TP2は進行中）。

## ベンチマーク

単一のV100-SXM3-32GB、同一GGUF重み、`llama-server` = llama.cpp unslothビルド b12209、特記なき限り単一ユーザー。**数値は暫定値です**：共有GPU上での内部測定であり、リリース前に再測定予定です。

### Qwen3.8-27B UD-Q4_K_XL

| 指標 | Tachyon | llama-server |
|---|---|---|
| デコード、1ユーザー、greedy、ドラフトなし | 40 tok/s | 35-37 tok/s |
| MTPあり | 75 tok/s（k=3、12ゴールデンプロンプトの集計） | 51-53 tok/s |
| TTFT、550トークンプロンプト | 0.70 s | 1.0 s |
| TTFT、8Kプロンプト | 7.9 s | 11.4 s |
| 同時4短リクエスト | 合計107 tok/s | — |

HTTP、temp 0、400トークン、MTP：日本語散文 59、コード 105-122、ツール呼び出し 102-111 tok/s。temp 0.7、GPUサンプリング（MTP k=6）：日本語散文 61、日本語チャット 62、コード 102 tok/s。

### Qwen3.6-35B-A3B Q4_K_M

llama-server（社内の運用と同じ引数）：`-c 131072 -np 4 -ub/-b 2048`、KV q8_0。

| 指標 | Tachyon | llama-server |
|---|---|---|
| デコード、1ユーザー、temp 0、ドラフトなし | 120 tok/s | 99 tok/s |
| デコード、1ユーザー、temp 0.7、GPUサンプリング | 117 tok/s | 98.5 tok/s |
| TTFT、1Kプロンプト | 0.29 s | 0.80 s |
| TTFT、8Kプロンプト | 2.02 s | 3.05 s |
| 4ストリーム、temp 0.7 | 合計166.7 tok/s | 合計159 tok/s（np4） |
| 8ストリーム、temp 0.7 | 合計167 tok/s | 合計180 tok/s（np8） |
| JSONモード（`response_format`） | 187.5 tok/s | 59.3 tok/s（grammar） |

MTP k=6、temp 0：日本語散文 171、日本語チャット 153、コード 268 tok/s（llama-server + MTP は同じ条件ではまだ測っていません）。17KトークンプロンプトのTTFT 3.71 s（プレフィル 4,630 tok/s）、RAMプレフィックスティアありの場合、同一会話の次ターンは0.27 s。8ストリームではllama-serverが勝ちます。同一エキスパートのMoEバッチングは未実装です。

## 精度

- **27B**：llama-serverとのtop-1一致 740/742ポジション（相違2件はほぼ同点）；CPU f32リファレンス比 192/192。
- **3.6**：CPU f32リファレンス比 top-1 98.2%、平均KL 0.0006（llama-server：97.6%、0.0024）。

numpyとのカーネル単位チェック（`tools/kcheck.py`など）、CPU f32リファレンスとのレイヤー単位比較（`tools/ref_qwen35.py`）、`llama-server`とのゴールデン比較で検証済みです。

## ビルド

要件：

- Linux x86_64
- CUDA 12.x（12.5でテスト済み。CUDA 13はVoltaサポートを削除）、およびそれをサポートするドライバー
- Rust stable（1.99でテスト済み）

`nvcc`は`build.rs`から`-arch=sm_70`で呼び出されます；`CUDA_HOME`のデフォルトは`/usr/local/cuda`です。GPUは`CUDA_VISIBLE_DEVICES`で選択します。

```sh
cargo build --release
# -> target/release/tachyon
```

コマンド：

```
tachyon generate | layers | bench [--lens 547,8192 --decode 64] | spec --k 3 --mtp-gguf
tachyon serve --gguf ... --hf <tokenizer dir> --keys keys.json --port 8080 [--mtp-gguf] [--mmproj] [--max-seqs 4] [--ctx 65536]
```

## クイックスタート

重みは本リポジトリには含まれません。Hugging Faceからダウンロードしてください（Apache-2.0）：

```sh
# Qwen3.8-27B
huggingface-cli download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf MTP/mtp-Qwen3.8-27B-Q4_0.gguf mmproj-F16.gguf --local-dir weights/q38-27b
huggingface-cli download Qwen/Qwen3.8-27B tokenizer.json tokenizer_config.json chat_template.jinja --local-dir weights/q38-27b/hf
```

`keys.json`を作成します（各キーのsha256のみが保存されます；平文キーはテスト専用です）：

```sh
KEY=$(openssl rand -hex 24)            # keep this value; send it as the Bearer token
printf '{"accounts": {"local": {"keys_sha256": ["%s"], "rpm": 600, "max_concurrent": 4}}}\n' \
  "$(printf %s "$KEY" | sha256sum | cut -d' ' -f1)" > keys.json
```

サーブ：

```sh
CUDA_VISIBLE_DEVICES=0 target/release/tachyon serve \
  --gguf weights/q38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --hf weights/q38-27b/hf \
  --mtp-gguf weights/q38-27b/MTP/mtp-Qwen3.8-27B-Q4_0.gguf --draft-vocab data/draft_vocab_q38.txt \
  --mmproj weights/q38-27b/mmproj-F16.gguf \
  --keys keys.json --host 127.0.0.1 --port 8080 \
  --ctx 65536 --max-seqs 4 --max-running 4
```

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

Qwen3.6-35B-A3Bの場合は`ggml-org/Qwen3.6-35B-A3B-GGUF`（`Qwen3.6-35B-A3B-Q4_K_M.gguf`、`mtp-Qwen3.6-35B-A3B-Q8_0.gguf`）、
`Qwen/Qwen3.6-35B-A3B`のトークナイザーファイル、およびF16 mmproj（`unsloth/Qwen3.6-35B-A3B-GGUF`の`mmproj-F16.gguf`、またはQ8_0のものを`tools/mmproj_to_f16.py`で変換）を使用します。
MoEモデルは同じ`data/draft_vocab_q38.txt`を使用します（同一トークナイザー）。

## 最小再現ベンチマーク

1. `huggingface-cli`でGGUFをダウンロードします。
2. `cargo build --release`
3. `CUDA_VISIBLE_DEVICES=0 target/release/tachyon bench --gguf weights/q38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf --lens 547,8192 --decode 64`（エンジンのみ：長さごとのプレフィルtok/s、単一ユーザーデコードtok/s）
4. 同一GPU上のメインラインllama.cppと比較します：`llama-bench -m <same gguf> -p 547,8192 -n 64 -ngl 99 -fa 1`。
5. HTTP経由のエンドツーエンド（両サーバーで同一プロンプトと同一TTFT定義）：`tachyon serve --no-auth --host 127.0.0.1`と`llama-server`を2つのポートで起動し、それぞれに`python3 tools/q36_bench.py <port> <label> 3 0.7`を実行します（400トークン、3回実行；`docs/BENCHMARKS.md`参照）。

## リポジトリ構成

- `src/`：Rust（GGUFローダー、モデル、KVマネージャー、spec/MTP、トークナイザー、チャットテンプレート、ビジョン、サーバー）
- `kernels/`：CUDA
- `tools/`：リファレンス実装、カーネルチェック、変換、ベンチマーク
- `tests/`：フェイクバックエンドによるHTTP APIテスト
- `data/`：MTP lm_head用ドラフトボキャブラリ
- `docs/`

## ライセンス

Apache-2.0、Copyright 2026 blueqat Inc. サードパーティライセンスは`THIRD_PARTY_LICENSES.md`に記載しています。モデルの重みは**含まれず**、それぞれのライセンスに従います。

## 謝辞

- llama.cpp/ggml（量子化フォーマットおよびリファレンス）
- Qwenチーム
- unslothおよびggml-orgのGGUF変換
- Tencent DSpine（設計リファレンス）
- SenseTime / OpenSenseNova（SenseNova-Loop-9B とその参照コード）
- Thinking Machines Lab（Inkling-2-Mini）、その参照実装を公開した transformers・llama.cpp の貢献者、ggml-org（GGUF 変換）
- YASAKANI（下書きと本体で共有するシード付きサンプリングノイズの考え方。NOTICE を参照）

blueqat Inc.により開発されています。
