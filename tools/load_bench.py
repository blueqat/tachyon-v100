"""同時の人数ごとの速さと、最初の 1 語までの時間（TTFT）を測る（OpenAI 互換の口なら何でも）。

  .venv/bin/python tools/load_bench.py --url http://127.0.0.1:8082 --users 1,8,16 --prompt-tokens 1000,8000 --out logs/load.jsonl

- 1 人ずつ stream で受け取り、最初の 1 語が来た時刻・最後の時刻・出したトークン数を測る
- デコードの速さ = (出したトークン数 - 1) / (最後 - 最初の 1 語)。同時 N 人の合計 = 全員の出したトークン数 / 一番遅い人が終わるまで
- Thinking は切って測る（答えの長さをそろえるため。考える長さの比べは effort_bench.py）
- 文は公開の文章を繰り返して作る（社内の文やお客さまの文は使わない）。文頭の使い回しが効かないよう、人ごとに先頭へ別の番号を入れる
"""
import argparse
import concurrent.futures as cf
import json
import os
import time

import requests

FILLER = ("量子コンピュータは、量子力学の重ね合わせともつれを使って計算する機械です。古典的なビットは 0 か 1 のどちらかですが、"
          "量子ビットは両方の状態を同時にとることができます。これを利用して、ある種の問題を速く解けると期待されています。")


def make_prompt(n_tokens, tag):
    # 日本語はおよそ 1 文字 1 トークン前後。多めに作って、指示を最後に置く
    body = (FILLER * (n_tokens // len(FILLER) + 1))[: int(n_tokens * 1.0)]
    return f"[{tag}]\n{body}\n\n上の文章の要点を、箇条書き 3 つでまとめてください。"


def one(url, prompt, max_tokens, thinking_kwargs, api_key=None):
    body = dict(model="x", messages=[dict(role="user", content=prompt)], max_tokens=max_tokens, temperature=0.7,
                top_p=0.95, stream=True, stream_options=dict(include_usage=True), chat_template_kwargs=thinking_kwargs)
    # 鍵（tachyon-serve の --keys）。環境変数 TACHYON_API_KEY か --api-key-file から（⛔ コマンド行に鍵を書かない）
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    t0 = time.time()
    first = last = None
    n = 0
    usage = {}
    with requests.post(url + "/v1/chat/completions", json=body, stream=True, timeout=3600, headers=headers) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data == b"[DONE]":
                break
            j = json.loads(data)
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content"):
                    now = time.time()
                    first = first or now
                    last = now
                    n += 1
    toks = usage.get("completion_tokens", n)
    return dict(ttft=(first - t0) if first else None, total=(last or time.time()) - t0,
                decode_tps=(toks - 1) / (last - first) if first and last and last > first else None,
                completion_tokens=toks, prompt_tokens=usage.get("prompt_tokens"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8082")
    ap.add_argument("--users", default="1,8,16")
    ap.add_argument("--prompt-tokens", default="1000")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--label", default="")
    ap.add_argument("--thinking", default='{"reasoning_effort":"low","clear_thinking":true,"enable_thinking":false}')
    ap.add_argument("--out", default="logs/load.jsonl")
    ap.add_argument("--api-key-file", default="", help="鍵が 1 行のファイル（無ければ環境変数 TACHYON_API_KEY）")
    a = ap.parse_args()
    kw = json.loads(a.thinking)
    key = open(a.api_key_file).read().strip() if a.api_key_file else os.environ.get("TACHYON_API_KEY")
    for pt in map(int, a.prompt_tokens.split(",")):
        for u in map(int, a.users.split(",")):
            prompts = [make_prompt(pt, f"{time.time_ns()}-{i}") for i in range(u)]
            t0 = time.time()
            with cf.ThreadPoolExecutor(u) as ex:
                rs = list(ex.map(lambda p: one(a.url, p, a.max_tokens, kw, key), prompts))
            wall = time.time() - t0
            toks = sum(r["completion_tokens"] for r in rs)
            ttfts = sorted(r["ttft"] for r in rs if r["ttft"] is not None)
            dec = [r["decode_tps"] for r in rs if r["decode_tps"]]
            row = dict(label=a.label, users=u, prompt_tokens=pt, wall=round(wall, 1), total_tps=round(toks / wall, 1),
                       per_user_decode_tps=round(sum(dec) / len(dec), 1) if dec else None,
                       ttft_p50=round(ttfts[len(ttfts) // 2], 2) if ttfts else None, ttft_max=round(ttfts[-1], 2) if ttfts else None,
                       prompt_tokens_real=rs[0]["prompt_tokens"])
            print(json.dumps(row, ensure_ascii=False), flush=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
