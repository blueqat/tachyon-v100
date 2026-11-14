"""OpenAI 互換の口の形を外から確かめる（requests だけ。GPU は使わない）。

  .venv/bin/python tools/api_smoke.py --url http://127.0.0.1:<port> --api-key-file logs/key.txt [--fake]

- --fake: tachyon-serve --fake（決まった語を流す）に向けるとき。[[raw]] などの印で中身まで確かめる
- なし: 本物のエンジン・llama-server に向けるとき（形だけ確かめる。中身はモデル次第）
- 鍵はファイルか環境変数 TACHYON_API_KEY（⛔ コマンド行に鍵を書かない）
最後に「全部 OK」、NG があれば終了コード 1。
"""
import argparse
import json
import os
import sys

import requests

NG = []


def check(cond, what):
    print(("OK  " if cond else "NG  ") + what, flush=True)
    if not cond:
        NG.append(what)


def sse(url, headers, body):
    """stream の data を全部（[DONE] を見たか）"""
    out, done = [], False
    with requests.post(url + "/v1/chat/completions", json=dict(body, stream=True), headers=headers, stream=True, timeout=600) as r:
        if r.status_code != 200:
            return r.status_code, [r.json()], False
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines(decode_unicode=False):
            if not line.startswith(b"data: "):
                continue
            d = line[6:]
            if d == b"[DONE]":
                done = True
                continue
            out.append(json.loads(d))
    return 200, out, done


def merge(chunks):
    r = c = ""
    calls, finish, usage = [], None, None
    for ch in chunks:
        if ch.get("usage"):
            usage = ch["usage"]
        for choice in ch.get("choices", []):
            d = choice.get("delta", {})
            r += d.get("reasoning_content") or ""
            c += d.get("content") or ""
            calls += d.get("tool_calls") or []
            finish = choice.get("finish_reason") or finish
    return r, c, calls, finish, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--api-key-file", default="")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    key = open(a.api_key_file).read().strip() if a.api_key_file else os.environ.get("TACHYON_API_KEY", "")
    h = {"Authorization": f"Bearer {key}"} if key else {}
    u = a.url.rstrip("/")

    r = requests.get(u + "/health", timeout=30)
    check(r.status_code == 200 and r.json().get("status") == "ok", f"/health → {r.status_code} {r.text[:80]}")
    r = requests.get(u + "/v1/models", headers=h, timeout=30)
    mid = r.json()["data"][0]["id"] if r.status_code == 200 else ""
    check(r.status_code == 200 and r.json()["object"] == "list" and mid, f"/v1/models → {mid}")
    if key:
        r = requests.get(u + "/v1/models", timeout=30)
        check(r.status_code == 401 and "error" in r.json(), f"鍵なしは 401 → {r.status_code}")

    # stream なし（考えあり）
    q = "[[echo]]こんにちは、世界" if a.fake else "日本の首都はどこですか。一言で。"
    body = dict(model=mid, messages=[dict(role="user", content=q)], max_tokens=256 if a.fake else 2048)
    r = requests.post(u + "/v1/chat/completions", json=body, headers=h, timeout=600)
    j = r.json()
    check(r.status_code == 200 and j["object"] == "chat.completion" and j["id"].startswith("chatcmpl"), f"chat → {r.status_code}")
    m = j["choices"][0]["message"]
    check(m["role"] == "assistant" and isinstance(m.get("content"), str), "message.role・content")
    check(bool(m.get("reasoning_content")), "考えは reasoning_content に分かれる")
    check("<think>" not in m["content"] and "</think>" not in m["content"], "content に考えの印が残らない")
    us = j["usage"]
    check(us["total_tokens"] == us["prompt_tokens"] + us["completion_tokens"] and "reasoning_tokens" in us.get("completion_tokens_details", {}), f"usage {us}")
    check(j["choices"][0]["finish_reason"] in ("stop", "length"), f"finish_reason {j['choices'][0]['finish_reason']}")
    if a.fake:
        check(m["content"] == "こんにちは、世界", f"答え {m['content']!r}")

    # stream（usage つき）。まとめると stream なしと同じ
    code, ch, done = sse(u, h, dict(body, stream_options=dict(include_usage=True)))
    rr, cc, _, fin, usage = merge(ch)
    check(code == 200 and done, "stream → [DONE] まで")
    check(ch and ch[0]["choices"][0]["delta"].get("role") == "assistant", "最初の塊は role")
    check(all(c["object"] == "chat.completion.chunk" and c["id"] == ch[0]["id"] for c in ch), "塊の object・id")
    check(usage is not None and usage["completion_tokens"] > 0, f"stream の usage {usage}")
    check(fin in ("stop", "length"), f"stream の finish_reason {fin}")
    if a.fake:
        check((rr, cc) == (m["reasoning_content"], m["content"]), "stream をまとめると stream なしと同じ")

    # 道具
    tools = [dict(type="function", function=dict(name="get_weather", description="Get the current weather of a city",
                                                 parameters=dict(type="object", properties=dict(city=dict(type="string")), required=["city"])))]
    q = ("[[raw]]Call it.\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>"
         if a.fake else "東京の今の天気を調べて。道具を使うこと。")
    tb = dict(model=mid, messages=[dict(role="user", content=q)], tools=tools, max_tokens=256 if a.fake else 4096)
    r = requests.post(u + "/v1/chat/completions", json=tb, headers=h, timeout=600)
    j = r.json()
    tc = j["choices"][0]["message"].get("tool_calls") or []
    check(r.status_code == 200 and j["choices"][0]["finish_reason"] == "tool_calls" and tc, f"道具の呼び出し → {j['choices'][0]['finish_reason']}")
    if tc:
        args = json.loads(tc[0]["function"]["arguments"])
        check(tc[0]["type"] == "function" and tc[0]["function"]["name"] == "get_weather" and "city" in args and tc[0]["id"], f"tool_calls[0] {tc[0]}")
    code, ch, done = sse(u, h, tb)
    _, _, calls, fin, _ = merge(ch)
    check(done and fin == "tool_calls" and calls and calls[0].get("index") == 0, f"stream の道具の呼び出し {calls[:1]}")

    # 誤りの形
    r = requests.post(u + "/v1/chat/completions", json=dict(body, chat_template_kwargs=dict(reasoning_effort="high")), headers=h, timeout=60)
    e = r.json().get("error", {})
    check(r.status_code == 400 and e.get("type") == "invalid_request_error" and e.get("param"), f"違う reasoning_effort → {r.status_code} {e.get('param')}")
    r = requests.post(u + "/v1/chat/completions", data="{x", headers=dict(h, **{"content-type": "application/json"}), timeout=60)
    check(r.status_code == 400 and "error" in r.json(), f"壊れた JSON → {r.status_code}")

    r = requests.get(u + "/metrics", timeout=30)
    check(r.status_code == 200 and "tachyon_requests_total" in r.text and "tachyon_ttft_seconds_count" in r.text, "/metrics")
    print("全部 OK" if not NG else f"NG {len(NG)} 件", flush=True)
    sys.exit(1 if NG else 0)


if __name__ == "__main__":
    main()
