#!/usr/bin/env python3
"""3.6 の 1 人の速さを、llama-server と Tachyon の口で同じ測り方で（両方に同じ文・同じ TTFT の作り方）。
usage: q36_bench.py PORT LABEL [reps] [温度（既定 0.7）]

  ja_prose / ja_chat / code: 1 人・温度 0.7・top_p 0.95・400 語（考え無し）。速さ = (出た語 − 1) / (最後の語 − 最初の語)（手元の時計。両方同じ式）
  ttft1k / ttft8k: 入力 約 1K・8K 語（毎回違う頭で、使い回しを効かせない）、最初の語までの秒（小さい方）
1 行の JSON を出す。サーバーが timings（llama-server）を返せば、その predicted_per_second も *_srv に。
"""
import json
import random
import sys
import time
import urllib.request

port, label = sys.argv[1], sys.argv[2]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 2
TEMP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7
URL = f"http://127.0.0.1:{port}/v1/chat/completions"
S = "量子コンピュータは重ね合わせと干渉を使って計算する。古典計算機とは違う原理で動き、誤り訂正が大きな課題になっている。"  # ~50 語


def req(msgs, max_tokens, temp=None):
    temp = TEMP if temp is None else temp
    body = {"model": "x", "stream": True, "max_tokens": max_tokens, "temperature": temp, "top_p": 0.95,
            "chat_template_kwargs": {"enable_thinking": False}, "messages": msgs,
            "stream_options": {"include_usage": True}}
    t0 = time.time()
    ts, tim, usage = [], None, None
    r = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900)
    for line in r:
        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
            continue
        d = json.loads(line[6:])
        if d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
            ts.append(time.time())
        if d.get("timings"):
            tim = d["timings"]
        if d.get("usage"):
            usage = d["usage"]
    t1 = time.time()
    n = (usage or {}).get("completion_tokens") or len(ts)
    dec = (n - 1) / (ts[-1] - ts[0]) if len(ts) >= 2 and ts[-1] > ts[0] else None
    return {"ttft": (ts[0] if ts else t1) - t0, "n": n, "dec": dec, "srv": (tim or {}).get("predicted_per_second"),
            "pp": (usage or {}).get("prompt_tokens") or (tim or {}).get("prompt_n")}


def nonce():
    return f"[{random.random():.12f}] "


JA_PROSE = [{"role": "user", "content": "量子誤り訂正の表面符号について、学部生向けに日本語で800字程度で説明してください。"}]
JA_CHAT = [{"role": "user", "content": "週末に京都へ一泊で行きます。紅葉の時期なんですが、混雑を避けるコツはありますか？"},
           {"role": "assistant", "content": "紅葉の時期の京都は確かに混みますね。朝早い時間に有名どころを回るのが一番のコツです。どのあたりを回る予定ですか？"},
           {"role": "user", "content": "東山と嵐山を考えています。移動手段や回る順番、お昼を食べる場所のおすすめも含めて、気軽な感じで詳しく教えてください。"}]
CODE = [{"role": "user", "content": "Pythonで、CSVファイルを読み込んで列ごとの平均・中央値・標準偏差を計算し、結果をMarkdownの表で出力するスクリプトを書いてください。型ヒントとdocstring付きで。"}]

out = {"label": label, "temp": TEMP}
req([{"role": "user", "content": "こんにちは"}], 8)  # 温め
for name, m in (("ja_prose", JA_PROSE), ("ja_chat", JA_CHAT), ("code", CODE)):
    xs = [req([{"role": "system", "content": nonce()}] + m, 400) for _ in range(reps)]
    out[name] = round(sum(x["dec"] for x in xs) / len(xs), 1)
    out[name + "_n"] = [x["n"] for x in xs]
    if xs[0]["srv"]:
        out[name + "_srv"] = round(sum(x["srv"] for x in xs) / len(xs), 1)
for name, k in (("ttft1k", 33), ("ttft8k", 270)):
    xs = [req([{"role": "system", "content": nonce() + S * k}, {"role": "user", "content": "要約して"}], 16) for _ in range(reps)]
    out[name] = round(min(x["ttft"] for x in xs), 3)
    out[name + "_ptok"] = xs[0]["pp"]
print(json.dumps(out, ensure_ascii=False), flush=True)
