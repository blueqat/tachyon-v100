"""語の切り分けと chat_template の手本（HF の tokenizers と jinja2）。target/release/tokcheck が答え合わせに使う JSON を書く。

  .venv/bin/python tools/tok_ref.py <hf の置き場> <出力.json> [chat_template.jinja か .gguf（中のテンプレートを使う）] [--llama <.gguf>]

--llama: llama.cpp の llama-tokenize（語の表だけ読む・CPU。置き場は $LLAMA_BIN か third_party/llama-unsloth）でも
同じ文を切って HF と比べる（比べるだけ。手本は HF）。

jinja2 の環境は transformers の apply_chat_template（utils/chat_template_utils.py）と同じにしてある:
ImmutableSandboxedEnvironment(trim_blocks, lstrip_blocks, loopcontrols, {% generation %})、
tojson は json.dumps(ensure_ascii=False)、raise_exception は TemplateError。
"""
import ast
import json
import os
import random
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import jinja2
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent


class AssistantTracker(Extension):
    # transformers と同じく {% generation %} を受け付ける（描く結果は中身そのまま）
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return jinja2.nodes.CallBlock(self.call_method("_generation_support"), [], [], body).set_lineno(lineno)

    def _generation_support(self, caller):
        return caller()


def byte_level_chars():
    # GPT-2 の bytes_to_unicode（バイト → 語の中の文字）
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def compile_template(src):
    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[AssistantTracker, jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string(src)


# ───────────── 道具の定義 ─────────────
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "指定した都市の今の天気を返す。Returns the current weather.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "都市名（例: 東京）"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                "days": {"type": "integer", "minimum": 1, "maximum": 7},
            },
            "required": ["city"],
        },
    },
}
SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": 'Search the web. Use "quotes" & <tags> carefully.',
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
                "filters": {
                    "type": "object",
                    "properties": {"lang": {"type": "string"}, "after": {"type": "string", "format": "date"}},
                },
                "safe": {"type": "boolean"},
                "score": {"type": "number", "default": 0.5, "minimum": 1e-7, "maximum": 1e20},
            },
            "required": ["query"],
        },
    },
}
RUN = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": "Python のコードを実行する\n複数行の説明\t(タブ)",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}, "timeout": {"type": ["number", "null"]}, "args": {"type": "array", "items": {"type": "string"}}},
            "required": ["code"],
        },
    },
}
PLAIN_TOOL = {"name": "calc", "description": "四則演算", "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}}}

CODE = '''def fib(n: int) -> int:
    """フィボナッチ数"""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

if __name__ == "__main__":
\tprint(fib(10))  # 55
'''
RUST = "fn main() {\r\n    let v: Vec<u32> = (0..10).map(|x| x * x).collect();\r\n    println!(\"{:?}\", v);\r\n}\r\n"
EMOJI = "家族 👨‍👩‍👧‍👦 旗 🇯🇵🇺🇸 肌 👍🏽👋🏿 合字 🧑‍💻 異体字 ❤️ ☺︎ 古い 😀😃😄 珍しい 🫠🪿🫨 数学 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝟘𝟙𝟚"
JA = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。"
JA_MIX = "ＧＰＵは「V100（32GB）」を２枚、ﾊﾝｶｸｶﾀｶﾅも混ぜる。全角スペース　と、合成文字 が゙ ㍿ ①②③ 〜 ～ ― —。"


def u(text):
    return {"role": "user", "content": text}


def a(content, reasoning=None, tool_calls=None):
    m = {"role": "assistant", "content": content}
    if reasoning is not None:
        m["reasoning_content"] = reasoning
    if tool_calls is not None:
        m["tool_calls"] = tool_calls
    return m


def tc(name, args, wrap=True):
    f = {"name": name, "arguments": args}
    return {"id": "call_" + name, "type": "function", "function": f} if wrap else f


def tool(content, name=None):
    m = {"role": "tool", "content": content}
    if name:
        m["tool_call_id"] = "call_" + name
    return m


SYS = {"role": "system", "content": "あなたは丁寧な日本語のアシスタントです。You are a helpful assistant."}
MULTI = [
    SYS,
    u("2+2 は？"),
    a("4 です。", reasoning="簡単な足し算。2+2=4。"),
    u("では 3*7 は？"),
    a("21 です。", reasoning="\n掛け算: 3*7 = 21\n"),
    u("最後に 10/4 は？"),
]


def chat_cases():
    c = []

    def add(name, messages, tools=None, roundtrip=False, openai_args=False, agp=True, **kw):
        c.append(dict(name=name, messages=messages, tools=tools, kwargs=kw, add_generation_prompt=agp,
                      roundtrip=roundtrip, openai_args=openai_args))

    add("en_simple", [u("Hello, how are you?")])
    add("ja_simple", [u(JA)])
    add("ja_system", [SYS, u("V100 で LLM を速く動かすコツを 3 つ教えて。")])
    add("effort_xhigh", [SYS, u("Explain quicksort.")], reasoning_effort="xhigh")
    add("effort_medium", [SYS, u("クイックソートを説明して。")], reasoning_effort="medium")
    add("effort_low", [u("Name three primes.")], reasoning_effort="low")
    add("effort_low_nosys", [u("素数を 3 つ")], reasoning_effort="low", enable_thinking=True)
    add("effort_bad_high", [u("hi")], reasoning_effort="high")
    add("effort_bad_case", [SYS, u("hi")], reasoning_effort="XHIGH")
    add("thinking_off", [SYS, u("こんにちは")], enable_thinking=False)
    add("thinking_off_effort_low", [u("hello")], enable_thinking=False, reasoning_effort="low")
    add("thinking_off_effort_bad", [u("hello")], enable_thinking=False, reasoning_effort="max")
    add("code_python", [u("このコードを説明して:\n```python\n" + CODE + "```")])
    add("code_rust_crlf", [SYS, u("Review:\r\n" + RUST)], reasoning_effort="medium")
    add("emoji", [u(EMOJI)])
    add("ja_fullwidth_mix", [u(JA_MIX)])
    add("multi_preserve_default", MULTI)
    add("multi_preserve_false", MULTI, preserve_thinking=False)
    add("multi_preserve_true", MULTI, preserve_thinking=True, reasoning_effort="low")
    add("multi_thinking_off", MULTI, enable_thinking=False, preserve_thinking=False)
    add("multi_last_assistant", MULTI + [a("2.5 です。", reasoning="10/4=2.5")], agp=False, roundtrip=True)
    add("tools_one", [u("東京の天気は？")], tools=[WEATHER])
    add("tools_sys", [SYS, u("Search for V100 benchmarks")], tools=[WEATHER, SEARCH, RUN], reasoning_effort="medium")
    add("tools_plain_def", [u("12*34")], tools=[PLAIN_TOOL])
    add("tools_call_and_result", [
        SYS, u("大阪と札幌の天気を調べて"),
        a("", reasoning="2 都市を調べる。", tool_calls=[tc("get_weather", {"city": "大阪"}), tc("get_weather", {"city": "札幌", "unit": "celsius", "days": 3})]),
        tool('{"temp": 21, "sky": "晴れ"}', "get_weather"), tool('{"temp": 8, "sky": "雪"}', "get_weather"),
    ], tools=[WEATHER])
    add("tools_call_content_before", [
        u("Find papers on Gated DeltaNet"),
        a("I'll search for that.", tool_calls=[tc("web_search", {"query": "Gated DeltaNet", "top_k": 3, "filters": {"lang": "en", "after": "2024-01-01"}, "safe": True, "score": 0.25})]),
        tool("[1] Gated Delta Networks (2024)"),
        a("Found one paper.", reasoning="一件見つかった"),
        u("Thanks. Run fib(10)."),
    ], tools=[SEARCH, RUN], reasoning_effort="low")
    add("tools_nonstring_args", [
        u("run it"),
        a(None, tool_calls=[tc("run_python", {"code": CODE, "timeout": 1e-05, "args": ["-v", "--n=10"]}),
                            tc("run_python", {"code": "print('<&>\"\\\\')", "timeout": None, "args": []}),
                            tc("web_search", {"query": "x", "top_k": 12345678901234, "score": 1e20, "filters": {}, "safe": False})]),
        tool("55"), tool("<&>\"\\"), tool("[]"),
    ], tools=[RUN, SEARCH])
    # 数の読み戻し（serde_json の float_roundtrip）: 既定の読み方では 1 ulp ずれて、描いた文が Python の json と変わった数
    # （1.602176634e-19 → 1.6021766340000001e-19、1.98847e30 → 1.9884700000000003e+30。2026-11-10 の見直し）
    add("tools_float_roundtrip", [
        u("電子の電荷と太陽の質量は？"),
        a("", tool_calls=[tc("calc_phys", {"e": 1.602176634e-19, "m_sun": 1.98847e30, "g": 6.6743e-11, "h": 6.62607015e-34,
                                           "c": 299792458.0, "na": 6.02214076e23, "tiny": 5e-324, "big": 1.7976931348623157e308,
                                           "third": 0.1 + 0.2, "list": [2.2250738585072014e-308, 9007199254740993.0, 1e23, 8.98846567431158e307]})]),
        tool("ok"),
    ], tools=[{"type": "function", "function": {"name": "calc_phys", "description": "物理定数で計算",
                                                "parameters": {"type": "object", "properties": {"e": {"type": "number", "default": 1.602176634e-19},
                                                                                                "m_sun": {"type": "number", "default": 1.98847e30}}}}}])
    add("tools_openai_string_args", [
        u("東京の天気"),
        a("", tool_calls=[tc("get_weather", {"city": "東京", "unit": "celsius"})]),
        tool("晴れ 18℃"),
    ], tools=[WEATHER], openai_args=True)
    add("tools_unwrapped_call", [u("calc"), a("", tool_calls=[tc("calc", {"expr": "1+2"}, wrap=False)]), tool("3")], tools=[PLAIN_TOOL])
    add("tools_thinking_off", [SYS, u("天気")], tools=[WEATHER], enable_thinking=False)
    add("tools_roundtrip_last", [
        u("札幌の天気を 3 日分、華氏で"),
        a("調べます。", reasoning="get_weather を使う", tool_calls=[tc("get_weather", {"city": "札幌", "unit": "fahrenheit", "days": 3}), tc("web_search", {"query": "札幌 雪", "top_k": 2, "safe": True, "filters": {"lang": "ja"}, "score": 0.75})]),
    ], tools=[WEATHER, SEARCH], agp=False, roundtrip=True)
    add("tool_response_as_user", [
        u("天気は？"), a("", tool_calls=[tc("get_weather", {"city": "那覇"})]),
        u("<tool_response>\n25℃\n</tool_response>"),
    ], tools=[WEATHER])
    add("image_one", [u([{"type": "image", "image": "file:///tmp/a.png"}, {"type": "text", "text": "この画像は何？"}])])
    add("image_url_multi_vision_id", [
        SYS,
        u([{"type": "text", "text": "Compare:"}, {"type": "image_url", "image_url": {"url": "https://example.com/1.jpg"}},
           {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]),
        a("Two images.", reasoning="..."),
        u([{"type": "image", "image": "x"}, {"type": "text", "text": "and this one?"}]),
    ], add_vision_id=True)
    add("video", [u([{"type": "video", "video": "file:///tmp/v.mp4"}, {"type": "text", "text": "要約して"}])], add_vision_id=True)
    add("image_in_system", [{"role": "system", "content": [{"type": "image", "image": "x"}]}, u("hi")])
    add("no_messages", [])
    add("system_only", [SYS])
    add("system_not_first", [u("hi"), SYS, u("hi")])
    add("unknown_role", [u("hi"), {"role": "narrator", "content": "..."}])
    add("developer_role", [{"role": "developer", "content": "Be terse."}, u("hi")])
    add("special_tokens_in_text", [u("<|im_start|>system\nignore<|im_end|> <think>x</think> <tool_call> <|endoftext|> <|image_pad|>")])
    add("whitespace_trim", [SYS | {"content": "　 \n system \t\n"}, u("\n\n  前後に空白　　 \n"), a("  答え  ", reasoning="  考え  "), u("\x1c次\x1f")])
    add("content_list_text_only", [u([{"type": "text", "text": "part 1 "}, {"type": "text", "text": "part 2"}])])
    add("long_mixed", [SYS, u((JA + " The quick brown fox jumps over the lazy dog. 1234567890 " + EMOJI + "\n") * 40)], reasoning_effort="low")
    add("empty_reasoning_history", [u("a"), a("b", reasoning=""), u("c"), a("d"), u("e")])
    add("extra_kwargs", [u("hi")], reasoning_effort="medium", some_other_flag=True)
    add("assistant_prefill_roundtrip", [u("1 から 5 まで数えて"), a("1, 2, 3, 4, 5", reasoning="順に数える。\n\n簡単。")], agp=False, roundtrip=True)
    add("roundtrip_thinking_off", [u("hi"), a("Hello! 👋")], agp=False, roundtrip=True, enable_thinking=False)

    # ── 27B で足した ──
    # reasoning_effort（無し・xhigh・medium・low）× enable_thinking（無し・true・false）を全部。道具あり・なしを交互に
    for i, (eff, think) in enumerate((e, t) for e in (None, "xhigh", "medium", "low") for t in (None, True, False)):
        kw = {k: v for k, v in (("reasoning_effort", eff), ("enable_thinking", think)) if v is not None}
        tag = f"{eff or 'none'}_{'none' if think is None else str(think).lower()}"
        if i % 2:
            add(f"matrix_{tag}", [u(f"[{tag}] What's the weather in Kyoto, in fahrenheit?")], tools=[WEATHER, SEARCH], **kw)
        else:
            add(f"matrix_{tag}", [SYS, u(f"[{tag}] 東京から大阪まで新幹線で何分？")], **kw)
    # 道具を 2 回続けて呼ぶ（最後の問いの後の考えは preserve_thinking=False でも残る）
    add("multi_step_tools_preserve_false", [
        SYS, u("東京と那覇の天気を比べて"),
        a("", reasoning="まず東京。", tool_calls=[tc("get_weather", {"city": "東京"})]), tool("晴れ 18℃", "get_weather"),
        a("", reasoning="次は那覇。", tool_calls=[tc("get_weather", {"city": "那覇"})]), tool("曇り 25℃", "get_weather"),
    ], tools=[WEATHER], preserve_thinking=False, reasoning_effort="medium")
    # 前の問いの道具と考えは消え、新しい問いだけ
    add("history_tools_then_new_query", [
        u("札幌の天気"), a("", reasoning="調べる", tool_calls=[tc("get_weather", {"city": "札幌"})]), tool("雪 -2℃"),
        a("札幌は雪で -2℃ です。", reasoning="結果をそのまま"), u("ありがとう。明日は？"),
    ], tools=[WEATHER], preserve_thinking=False, enable_thinking=False)
    add("ja_multi_turn_effort_medium", [
        SYS, u("日本の三名園は？"), a("兼六園・後楽園・偕楽園です。", reasoning="金沢・岡山・水戸。"),
        u("それぞれの県は？"), a("石川県・岡山県・茨城県です。", reasoning="兼六園=石川、後楽園=岡山、偕楽園=茨城。"),
        u("一番古いのは？　理由も一言で。"),
    ], reasoning_effort="medium")
    add("tools_emoji_args_xhigh", [
        u("寿司の店を探して 🍣"),
        a("", reasoning="検索する 🔍", tool_calls=[tc("web_search", {"query": "🍣 寿司 🇯🇵 \"best\" <omakase>", "filters": {"lang": "ja"}, "top_k": 0, "score": 0.0})]),
        tool("🍣 すし匠（★4.8）"),
    ], tools=[SEARCH], reasoning_effort="xhigh")
    add("roundtrip_tools_effort_low", [
        u("Run fib(20) and search for Gated DeltaNet"),
        a("Running both.", reasoning="Two calls.", tool_calls=[tc("run_python", {"code": CODE, "timeout": 2.5, "args": []}), tc("web_search", {"query": "Gated DeltaNet", "top_k": 1})]),
    ], tools=[RUN, SEARCH], agp=False, roundtrip=True, reasoning_effort="low")
    add("assistant_content_list", [u("hi"), {"role": "assistant", "content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "there"}], "reasoning_content": None}, u("again")])
    add("image_and_video_one_message", [u([{"type": "image_url", "image_url": {"url": "x"}}, {"type": "video", "video": "y"}, {"type": "image", "image": "z"}, {"type": "text", "text": "比べて"}])])
    # 誤りで止まる: system が 2 つ（HF は止める、unsloth はまとめる）・名前の無い呼び出し・JSON 文字列のままの arguments・問いが無い
    add("two_systems", [SYS, {"role": "developer", "content": "Answer in English."}, u("hi")])
    add("tool_call_no_name", [u("x"), a("", tool_calls=[{"type": "function", "function": {"arguments": {"expr": "1"}}}]), tool("1")], tools=[PLAIN_TOOL])
    add("tool_args_json_string_raw", [u("x"), a("", tool_calls=[tc("calc", '{"expr": "1+2"}')]), tool("3")], tools=[PLAIN_TOOL])
    add("only_tool_responses", [u("<tool_response>\nx\n</tool_response>")])
    return c


# ───────────── 切り分けだけの文 ─────────────
POOL = (
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    + list(" \t\n\r　\xa0 \x0b\x0c")
    + list(".,;:!?'\"()[]{}<>/\\|@#$%^&*-_=+~`")
    + ["'s", "'T", "'re", "'LL", "'d", "n't"]
    + list("あいうえおかきくけこがぎぐ漢字日本語東京大阪猫犬ー・。、「」『』（）！？")
    + list("アイウエオヴァ゙ﾊﾟｶﾞ")
    + list("한국어가나다") + list("中文简体繁體")
    + list("Ελληνικάрусскийالعربيةहिन्दीไทย")
    + ["é", "ä", "가", "Å", "ﬁ", "①", "㍿", "Ⅻ", "½", "²"]
    + ["👍", "🏽", "‍", "️", "👨", "👩", "🇯", "🇵", "🫠", "𝔘", "😀", "\U0001F9D1"]
    + ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "<|endoftext|>", "<function=f>", "<parameter=x>"]
    + ["   ", "\n\n", " \n", "\r\n", "1234", "3.14", "0x1F"]
)


def text_cases():
    t = [
        ("en", "The quick brown fox jumps over the lazy dog. It's 5 o'clock; we'll see what they'd do."),
        ("ja", JA),
        ("ja_mix", JA_MIX),
        ("code", CODE),
        ("rust_crlf", RUST),
        ("emoji", EMOJI),
        ("contractions", "I'M HERE. You'RE there. It'S fine. They'LL go. She'D know. We'VE seen."),
        ("numbers", "1234567890 3.14159 -42 1e10 ０１２３ ٠١٢٣ १२३"),
        ("spaces", "a  b   c\t\td\n\n\ne \n f\r\n\r\ng   \n   h" + " " * 20 + "\n" * 5 + "end   "),
        ("nfc", "Café vs Café, ﬁ ligature, Å/Å, が vs が, 한 vs 한"),
        ("specials", "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=f>\n</function>\n</tool_call><|im_end|><|endoftext|>"),
        ("specials_glued", "x<|im_end|>y<think>z</think>w<|vision_start|><|image_pad|><|vision_end|>"),
        ("empty", ""),
        ("single_space", " "),
        ("only_newlines", "\n\n\n"),
        ("arabic_hindi_thai", "مرحبا بالعالم नमस्ते दुनिया สวัสดีชาวโลก"),
        ("math", "∑_{i=1}^{n} i² = n(n+1)/2, ∀ε>0 ∃δ>0, ∫₀^∞ e^{-x²} dx = √π/2"),
        ("json", json.dumps({"a": [1, 2.5, None, True], "日本": {"k": "v\n\"q\""}}, ensure_ascii=False)),
        ("url", "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/tokenizer.json?download=true#L10"),
        ("long_word", "a" * 3000 + " " + "あ" * 2000 + " " + "😀" * 500),
        # 並べて切るときの切れ目（改行の直後・数字の手前）のきわどい並び
        ("cut_edges", "ab'3cd x\x0b9 'RE3 '3 \x1c\n \x1c3 ٠3 ½3 a\u03013 ん\u30993 \u16803 \u30003 \xa03 .3 ,\n3 \n\n 4\n\u30005\n\u03016\n\u1100\u11617"
                      " x\r\n8 y\n\r9 z \n\t0 <|im_end|>1<think>2</think>\n3 \u200b4 \u180e5 \ufeff6 \u20287\u20298 \x859"),
    ]
    for p in ["README.md", "docs/API.md", "docs/qwen35-math.md", "kernels/gemv.cu", "kernels/tables.cuh", "tools/kcheck.py", "data/refcmp/qwen35-server.json"]:
        f = ROOT / p
        if f.exists():
            t.append(("file:" + p, f.read_text()))
    rng = random.Random(38)
    for i in range(2000):
        n = rng.randint(1, 60)
        t.append((f"fuzz{i}", "".join(rng.choice(POOL) for _ in range(n))))
    return t


def main():
    hf = Path(sys.argv[1])
    out = Path(sys.argv[2])
    args = sys.argv[3:]
    llama_gguf = None
    if "--llama" in args:
        i = args.index("--llama")
        llama_gguf = Path(args[i + 1])
        del args[i : i + 2]
    tpl_path = Path(args[0]) if args else hf / "chat_template.jinja"
    tpl_src = tpl_path
    if tpl_path.suffix == ".gguf":
        # GGUF に入っているテンプレートを取り出して手本の横に置く（tokcheck は GGUF からじかに読む）
        from gguf import GGUFReader

        src = GGUFReader(str(tpl_path)).fields["tokenizer.chat_template"].contents()
        tpl_path = out.with_suffix(".jinja")
        tpl_path.write_text(src)
    tok = Tokenizer.from_file(str(hf / "tokenizer.json"))
    tpl = compile_template(tpl_path.read_text())

    def enc(s):
        ids = tok.encode(s, add_special_tokens=False).ids
        return dict(ids=ids, decoded=tok.decode(ids, skip_special_tokens=False), decoded_skip=tok.decode(ids, skip_special_tokens=True))

    chat = []
    for c in chat_cases():
        try:
            prompt = tpl.render(messages=c["messages"], tools=c["tools"], add_generation_prompt=c["add_generation_prompt"], **c["kwargs"])
            c["expect"] = dict(prompt=prompt, error=None, **enc(prompt))
        except Exception as e:
            # raised: テンプレートの raise_exception で止まった（文まで同じはず）。
            # それ以外（UndefinedError・items の TypeError など jinja2 の中の誤り）は止まることだけ比べる
            c["expect"] = dict(prompt=None, error=str(e), raised=type(e) is jinja2.exceptions.TemplateError)
        chat.append(c)

    texts = []
    t0 = time.perf_counter()
    nbytes = ntok = 0
    for name, s in text_cases():
        r = enc(s)
        nbytes += len(s.encode())
        ntok += len(r["ids"])
        assert r["decoded"] == unicodedata.normalize("NFC", s), name
        texts.append(dict(name=name, text=s, **r))
    dt = time.perf_counter() - t0

    rng = random.Random(7)
    n = tok.get_vocab_size(with_added_tokens=True)
    seqs = [[rng.randrange(n) for _ in range(rng.randint(1, 24))] for _ in range(300)]
    # 1 バイトの語で、壊れた UTF-8（途中で切れる・余る・長すぎる・サロゲート・範囲外）を作る
    b2u = byte_level_chars()
    byte_id = lambda b: tok.token_to_id(b2u[b])
    bad = [b"\xf0\x80\x80\x80", b"\xe3\x81", b"\xe3\x81A", b"\xed\xa0\x80", b"\xc0\xaf", b"\xf4\x90\x80\x80", b"\xff\xfe",
           b"\x80\x80\x80", b"a\xe3\x81\x82\xe3", b"\xf0\x9f\x98", b"\xf0\x9f\x98\x80\x80", b"\xe2\x82\xac\xe2"]
    for x in bad:
        seqs.append([byte_id(b) for b in x])
        seqs.append([byte_id(b) for b in x] + tok.encode("<|im_end|>ok", add_special_tokens=False).ids + [byte_id(x[0])])
    rand = [dict(ids=ids, decoded=tok.decode(ids, skip_special_tokens=False), decoded_skip=tok.decode(ids, skip_special_tokens=True)) for ids in seqs]

    llama = None
    if llama_gguf:
        named = [("chat:" + c["name"], c["expect"]["prompt"], c["expect"]["ids"]) for c in chat if c["expect"]["error"] is None]
        named += [("文:" + t["name"], t["text"], t["ids"]) for t in texts]
        t0 = time.perf_counter()
        got = llama_tokenize(llama_gguf, [x[1] for x in named], tok, out.with_suffix(".llama.txt"))
        diffs = [n for (n, _, want), g in zip(named, got) if g != want]
        llama = dict(gguf=str(llama_gguf), n=len(named), same=len(named) - len(diffs), diffs=diffs)
        print(f"llama.cpp（llama-tokenize {time.perf_counter() - t0:.1f}s）: {llama['same']}/{llama['n']} 件が HF と同じ"
              + (f"。違う: {', '.join(diffs[:20])}" if diffs else ""))
        for (n, s, want), g in zip(named, got):
            if g != want and n in diffs[:8]:
                i = next((k for k, (x, y) in enumerate(zip(g, want)) if x != y), min(len(g), len(want)))
                print(f"  {n}: {i} 語目から llama {g[i:i + 6]} {[tok.decode([x]) for x in g[i:i + 6]]}／HF {want[i:i + 6]} {[tok.decode([x]) for x in want[i:i + 6]]}")

    out.write_text(json.dumps(dict(template=str(tpl_src), chat=chat, texts=texts, random_ids=rand, llama=llama), ensure_ascii=False))
    nerr = sum(c["expect"]["error"] is not None for c in chat)
    nraise = sum(bool(c["expect"].get("raised")) for c in chat)
    print(f"手本: chat {len(chat)} 件（うち誤り {nerr}、raise_exception {nraise}）、文 {len(texts)} 件 {nbytes/1e6:.2f}MB {ntok} 語"
          f"（Python の encode+decode {dt:.2f}s）、でたらめな番号 {len(rand)} 件（壊れた UTF-8 {2 * len(bad)} 件を含む）→ {out}")


def llama_tokenize(gguf_path, texts, tok, tmp, sep="<|quad_end|>"):
    """llama.cpp の llama-tokenize で切る。めったに出ない sep でつないで 1 回で切り、sep の所で分け直す
    （added token は前の切り分けより先に抜き出される＝HF も llama.cpp も、つないでも各文の切り方は変わらない）"""
    bindir = Path(os.environ.get("LLAMA_BIN", Path("third_party/llama-unsloth")))
    assert not any(sep in t for t in texts)
    tmp.write_bytes(sep.join(texts).encode())
    try:
        # --no-escape: 既定では文の中の \n・\" などを置き換えてしまう
        r = subprocess.run([str(bindir / "llama-tokenize"), "-m", str(gguf_path), "-f", str(tmp), "--ids", "--no-bos", "--no-escape", "--log-disable"],
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=str(bindir)), capture_output=True, text=True, check=True)
    finally:
        tmp.unlink()
    ids = ast.literal_eval(r.stdout.strip().splitlines()[-1])
    sid = tok.token_to_id(sep)
    out = [[]]
    for i in ids:
        if i == sid:
            out.append([])
        else:
            out[-1].append(i)
    assert len(out) == len(texts), (len(out), len(texts))
    return out


if __name__ == "__main__":
    main()
