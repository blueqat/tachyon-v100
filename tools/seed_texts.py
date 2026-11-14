#!/usr/bin/env python3
"""同じ種（seed）・温度 0.7 で何本か生成して JSON に（GPU の抜き取りの有り無しで同じ文字になるかの比べ）。
usage: seed_texts.py PORT OUT.json [presence か -] [温度]"""
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
pres = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "-" else None
TEMP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7
P = ["量子誤り訂正の表面符号について、学部生向けに日本語で説明してください。",
     "週末に京都へ一泊で行きます。紅葉の時期の混雑を避けるコツを気軽に教えてください。",
     "Pythonで、CSVを読み込んで列ごとの平均と標準偏差をMarkdownの表で出すスクリプトを書いてください。",
     "Write a short story about a lighthouse keeper who finds a message in a bottle."]
res = []
for i, p in enumerate(P):
    for seed in (1, 2):
        body = {"model": "x", "max_tokens": 200, "temperature": TEMP, "top_p": 0.95, "seed": seed,
                "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": p}]}
        if pres is not None:
            body["presence_penalty"] = pres
        d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
        res.append({"i": i, "seed": seed, "text": d["choices"][0]["message"]["content"]})
json.dump(res, open(out, "w"), ensure_ascii=False)
print(out, len(res))
