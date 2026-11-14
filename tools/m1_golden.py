#!/usr/bin/env python3
"""M1 の答え（golden）を作る: 12 の文 → llama-server の貪欲な 64 語・各位置の上位 20 の logprob・時間、
3 つの文は層ごとの値（CPU 参照 tools/ref_qwen35.py と、llama.cpp の CPU 版 tools/lcpp_dump.cpp）。
M1 の作り手と確かめ役がこれと突き合わせる（tools/m1_compare.py）。中身の形は下の一覧。

置き場（V100 の機械。git には入れない）: golden/m1/（環境変数 M1_GOLDEN で変える）
  src/                 文の元（ja.txt・en.txt・mixed_en.md は LLM の下書き、mixed.md はそれと repo の文書を継いだ物）
  prompts.json         12 の文（名前・説明・描いた文・語の番号）
  golden.json          llama-server の答え（schema m1-golden/1・mode free。tools/m1_compare.py が読む）
  golden-rerun.json    同じ物をもう一度（サーバー自身が毎回同じかの確かめ）
  golden-prefill.json  サーバー自身のプレフィルの道（mode forced。雑音の床）
  timings.json         llama-server の時間（プレフィル・デコード。n_probs なしで 3 回の中央値）
  ref-<名>.npz / ref.json    CPU 参照の層ごとの残差・result_norm・logits ／ 同じ位置の上位 20（mode forced）
  lcpp-<名>.npz / lcpp.json  llama.cpp（CPU）の層ごとの l_out・result_norm・logits ／ 同じ位置の上位 20
  ref-fill.json        golden の held の位置（probs なし）を埋める CPU 参照（fill で golden.json の top_fill に）
  layers.json          ref と lcpp の層ごとの開き
  tokcheck.json        自前の tokenizer（target/release/tokcheck）で同じ語になるかの手本
  MANIFEST.md          sha256 の表（manifest で作る）

使い方（V100 の機械で。重い物は RAM の上限の下で）:
  P=.venv/bin/python
  CUDA_VISIBLE_DEVICES= $P tools/m1_golden.py prompts                 # 文を作って語に切る（GPU なし・数秒）
  $P tools/m1_golden.py collect                                       # llama-server（127.0.0.1:8081）に 12 の文
  $P tools/m1_golden.py collect --out golden-rerun.json               # もう一度（毎回同じかを見る）
  $P tools/m1_golden.py prefill                                       # プレフィルの道（7 分）
  $P tools/m1_golden.py timing                                        # 時間だけ（n_probs 0・3 回）
  CUDA_VISIBLE_DEVICES= $P tools/m1_golden.py ref      # CPU 参照（5〜15 分）
  CUDA_VISIBLE_DEVICES= $P tools/m1_golden.py lcpp     # llama.cpp の CPU（要 target/lcpp_dump）
  CUDA_VISIBLE_DEVICES= $P tools/m1_golden.py ref --names chat_nothink,emoji --no-h --detail '' --out ref-fill.json
  $P tools/m1_golden.py fill ref-fill.json                            # held の位置に top_fill
  $P tools/m1_golden.py layers                                        # ref と lcpp の層ごとの開き
  CUDA_VISIBLE_DEVICES= $P tools/m1_golden.py tokcheck logs/tok_ref_27b.json  # tokcheck の手本を書く（tok_ref.py の出力に足す）
  $P tools/m1_golden.py manifest
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = Path(os.environ.get('M1_GOLDEN', 'golden/m1'))
HF = Path(os.path.join(os.environ.get("TACHYON_WEIGHTS", "weights"), "q38-27b/hf"))
GGUF = os.environ.get('M1_GGUF', os.path.join(os.environ.get("TACHYON_WEIGHTS", "weights"), "q38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf"))
SCHEMA = 'm1-golden/1'
N_PREDICT = int(os.environ.get('M1_N_PREDICT', '64'))  # 3.6 の 200 語の貪欲の比べで変える
N_PROBS = 20
REF_NAMES = ['en_short', 'ja_600', 'code']      # 層ごとの値も取る 3 つ
FIRST_SPECIAL = 248044                          # これより上は決まった語（素の文には入れない）
# 温度 0 でも念のため、罰則の類は全部切る（GGUF の既定で入っていても効かないように）
NEUTRAL = dict(repeat_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0, dry_multiplier=0.0)

# ---------------- 12 の文 ----------------

CODE = '''import heapq
from collections import defaultdict


def dijkstra(graph: dict[str, list[tuple[str, float]]], source: str) -> dict[str, float]:
    """Return the shortest distance from `source` to every reachable node.

    `graph[u]` is a list of `(v, weight)` pairs; all weights are non-negative.
    """
'''
MATH = ('Question: A bakery sells muffins for $2.75 each and cookies for $1.20 each. On Saturday it sold three times '
        'as many cookies as muffins, and the two items together brought in $381.00. How many muffins and how many '
        "cookies did the bakery sell?\nAnswer: Let's solve this step by step.\n")
EMOJI = ('Emoji and symbol glossary (keep the same format):\n'
         '🍣 = sushi\n🗻 = Mount Fuji\n🚄 = bullet train\n👨\u200d👩\u200d👧\u200d👦 = family: man, woman, girl, boy (ZWJ sequence)\n'
         '👍🏽 = thumbs up: medium skin tone\n🇯🇵 = flag: Japan\n1\ufe0f\u20e3 = keycap: 1\n'
         '∑ = n-ary summation, ∫ = integral, √ = square root, ∞ = infinity, ≈ ≠ ≤ ≥ → ⇒ ↔\n'
         '«guillemets» „Anführungszeichen“ ‘single’ — em dash, – en dash, … ellipsis\n'
         'ＡＢＣ１２３（全角） ｶﾀｶﾅ（半角） ①②③ ㈱ ℃ ¥ € £ ₩ ©®™\n'
         'Ελληνικά · Русский · 한국어 · العربية · हिन्दी · ไทย · 中文\n'
         'tab:\tend, backslash: C:\\Users\\name, HTML: &lt;b&gt;bold&lt;/b&gt; &amp; "quotes"\n'
         '🎉 = ')
MID_WORD = 'After three days of heated debate, the committee finally agreed to postpone the vote until the internatio'

CHAT = [
    {'role': 'system', 'content': 'あなたは社内の開発チームを手伝うアシスタントです。答えは簡潔に、必要ならコード例を添えてください。'},
    {'role': 'user', 'content': 'Rust で文字列を逆順にするいちばん簡単な方法は？'},
    {'role': 'assistant', 'content': '`s.chars().rev().collect::<String>()` です。`chars()` は Unicode のスカラー値ごとに区切るので、'
                                     'UTF-8 のバイト列を壊しません。'},
    {'role': 'user', 'content': '絵文字の 👨\u200d👩\u200d👧 みたいに、いくつかの文字が組み合わさったものはどうなりますか？'},
    {'role': 'assistant', 'content': 'スカラー値ごとに逆にすると、ZWJ（U+200D）でつながった絵文字はばらばらになります。'
                                     '見た目の 1 文字（書記素クラスタ）ごとに逆にするには `unicode-segmentation` クレートの '
                                     '`graphemes(true)` を使います。'},
    {'role': 'user', 'content': 'では、その方法で文字列を逆にする関数と、絵文字を含む文字列で試す main を書いて、動きを短く説明してください。'},
]
TOOLS = [
    {'type': 'function', 'function': {
        'name': 'get_weather', 'description': 'Get the current weather for a city.',
        'parameters': {'type': 'object', 'properties': {
            'city': {'type': 'string', 'description': 'City name, e.g. Sapporo'},
            'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}}, 'required': ['city']}}},
    {'type': 'function', 'function': {
        'name': 'web_search', 'description': 'Search the web and return the top results.',
        'parameters': {'type': 'object', 'properties': {
            'query': {'type': 'string'}, 'top_k': {'type': 'integer', 'default': 5}}, 'required': ['query']}}},
]
TOOL_CHAT = [
    {'role': 'system', 'content': 'You are a travel assistant. Use the tools when they help, and answer in English.'},
    {'role': 'user', 'content': "I'm flying to Sapporo tomorrow morning. What's the weather like there right now?"},
    {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call_1', 'type': 'function', 'function': {
        'name': 'get_weather', 'arguments': {'city': 'Sapporo', 'unit': 'celsius'}}}]},
    {'role': 'tool', 'tool_call_id': 'call_1',
     'content': '{"city": "Sapporo", "temp_c": -3, "condition": "light snow", "wind_kmh": 18}'},
    {'role': 'assistant', 'content': "It's -3 °C with light snow and a moderate wind in Sapporo right now, "
                                     'so pack a warm coat and boots.'},
    {'role': 'user', 'content': "Thanks! Could you also look up the dates and opening hours of this year's Sapporo Snow Festival?"},
]
# mixed.md の継ぎ方（src/mixed.md が無いときだけ使う。一度作ったら src/ の物が正。repo の物は 2026-11-10 の master 4f9350f）
#   (置き場, ファイル, 行の範囲 or None, コードの囲みの言葉 or None)。頭 6,000 語は英語の Markdown → 日本語の数式の文書 → Python → CUDA の途中
MIXED_PARTS = [('src', 'mixed_en.md', None, None), ('repo', 'docs/qwen35-math.md', (1, 92), None),
               ('repo', 'tools/ref_qwen35.py', (223, 330), 'python'), ('repo', 'kernels/gdn.cu', (1, 140), 'cuda'),
               ('repo', 'docs/qwen35-math.md', (93, 178), None)]


def prompt_defs(src):
    """(名前, 説明, 中身)。中身は raw（text・cut）か chat（messages・tools・kwargs）"""
    rd = lambda f: (src / f).read_text(encoding='utf-8')
    return [
        ('en_short', '短い英語（事実の続き）', dict(text='In 1905, Albert Einstein published four papers that')),
        ('ja_short', '短い日本語', dict(text='日本の四季の中で、私がいちばん好きなのは')),
        ('code', 'Python の関数の頭と docstring（続きは本体）', dict(text=CODE)),
        ('math', '算数の文章題（英語・答えの途中から）', dict(text=MATH)),
        ('ja_600', '日本語の随筆（src/ja.txt）の頭 600 語', dict(text=rd('ja.txt'), cut=600)),
        ('en_2000', '英語の随筆（src/en.txt）の頭 2,000 語', dict(text=rd('en.txt'), cut=2000)),
        ('mixed_6000', '混ざった文書（src/mixed.md: 英語の運用メモ＋日本語の数式の文書＋Python＋CUDA）の頭 6,000 語',
         dict(text=rd('mixed.md'), cut=6000)),
        ('chat_nothink', '3 回目の会話（日本語・考えなし enable_thinking=false）',
         dict(messages=CHAT, tools=None, kwargs={'enable_thinking': False})),
        ('chat_think', '同じ会話で考えあり（enable_thinking=true・reasoning_effort は既定の xhigh）',
         dict(messages=CHAT, tools=None, kwargs={'enable_thinking': True})),
        ('tool_call', '道具 2 つ・1 回呼んだ後の次の依頼（英語・考えなし）。続きは web_search の呼び出しのはず',
         dict(messages=TOOL_CHAT, tools=TOOLS, kwargs={'enable_thinking': False})),
        ('emoji', '絵文字（ZWJ・肌の色・旗・keycap）と記号・全角半角・いろいろな文字', dict(text=EMOJI)),
        ('mid_word', '単語の途中で終わる（internatio|nal）', dict(text=MID_WORD)),
    ]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def ids_sha(ids):
    return sha256_bytes(','.join(map(str, ids)).encode())


def make_mixed(src):
    parts, names = [], []
    for kind, f, rng, lang in MIXED_PARTS:
        body = ((src if kind == 'src' else ROOT) / f).read_text(encoding='utf-8')
        if rng:
            body = '\n'.join(body.split('\n')[rng[0] - 1:rng[1]])
        name = f + (f' L{rng[0]}-{rng[1]}' if rng else '')
        names.append(name)
        if lang:
            body = f'```{lang}\n{body.rstrip()}\n```\n'
        parts.append(body if kind == 'src' else f'\n\n---\n\n<!-- {name} -->\n\n' + body)
    t = '# Mixed document: operating notes, math spec, and code\n\n' + ''.join(parts)
    (src / 'mixed.md').write_text(t, encoding='utf-8')
    print(f'src/mixed.md を作った（{len(t)} 字）: ' + ' + '.join(names))


def load_tok():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import tok_ref
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(HF / 'tokenizer.json'))
    tpl = tok_ref.compile_template((HF / 'chat_template.jinja').read_text())
    return tok, tpl, tok_ref


def prompts(a):
    tok, tpl, _ = load_tok()
    src = GOLD / 'src'
    if not (src / 'mixed.md').exists():
        make_mixed(src)
    out = []
    for name, desc, d in prompt_defs(src):
        p = dict(name=name, desc=desc)
        if 'messages' in d:
            text = tpl.render(messages=d['messages'], tools=d['tools'], add_generation_prompt=True, **d['kwargs'])
            p.update(kind='chat', messages=d['messages'], tools=d['tools'], kwargs=d['kwargs'], add_generation_prompt=True)
        else:
            text = d['text']
            p.update(kind='raw')
        assert unicodedata.normalize('NFC', text) == text, f'{name}: NFC でない（llama.cpp は NFC をかけないので語が変わる）'
        ids = tok.encode(text, add_special_tokens=False).ids
        if p['kind'] == 'raw':
            bad = [i for i in ids if i >= FIRST_SPECIAL]
            assert not bad, f'{name}: 素の文に決まった語 {bad[:5]}'
        full = len(ids)
        if d.get('cut'):
            assert full >= d['cut'], f'{name}: {full} 語しかない（{d["cut"]} 語が要る）'
            ids = ids[:d['cut']]
            p.update(cut=d['cut'], n_tokens_full=full)
        dec = tok.decode(ids, skip_special_tokens=False)
        p.update(text=text, ids=ids, n_tokens=len(ids), ids_sha256=ids_sha(ids), decoded=dec,
                 ends_mid_utf8=dec.endswith('\ufffd'))
        out.append(p)
        print(f'{name:13s} {p["kind"]:4s} {len(ids):5d} 語' + (f'（元 {full}）' if d.get('cut') else '') +
              ('  ⚠️ UTF-8 の途中で切れる' if p['ends_mid_utf8'] else '') + f'  …{dec[-40:]!r}')
    meta = dict(schema='m1-prompts/1', made=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                tokenizer=dict(dir=str(HF), tokenizer_json_sha256=sha256_bytes((HF / 'tokenizer.json').read_bytes()),
                               chat_template_sha256=sha256_bytes((HF / 'chat_template.jinja').read_bytes())),
                src={f.name: sha256_bytes(f.read_bytes()) for f in sorted(src.iterdir()) if f.is_file()},
                prompts=out)
    GOLD.mkdir(parents=True, exist_ok=True)
    (GOLD / 'prompts.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f'→ {GOLD / "prompts.json"}（{len(out)} 件・計 {sum(p["n_tokens"] for p in out)} 語）')


# ---------------- llama-server ----------------

def post(url, body, timeout=3600):
    import requests
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get(url):
    import requests
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def utf8_cut(text):
    """llama.cpp の validate_utf8（tools/server の server-common）と同じ: 終わりで切れている多バイト文字の頭の位置"""
    n = len(text)
    for i in range(1, min(4, n) + 1):
        c = text[n - i]
        if (c & 0xE0) == 0xC0 and i < 2 or (c & 0xF0) == 0xE0 and i < 3 or (c & 0xF8) == 0xF0 and i < 4:
            return n - i
    return n


def piece_bytes(tok, tok_ref):
    b2u = tok_ref.byte_level_chars()
    u2b = {c: b for b, c in b2u.items()}
    added = {t.content for t in tok.get_added_tokens_decoder().values()} if hasattr(tok, 'get_added_tokens_decoder') else set()

    def f(i):
        s = tok.id_to_token(i)
        if s is None:
            return b''
        if s in added:
            return s.encode()               # 決まった語は ASCII（UTF-8 の切れ目には関わらない）
        return bytes(u2b[c] for c in s)
    return f


def held_positions(gen, pb):
    """サーバーが probs を落とす位置（その語を足した時点で生成文の終わりが UTF-8 の途中 → add_token を呼ばない）"""
    text, held = b'', []
    for j, t in enumerate(gen):
        text += pb(t)
        if utf8_cut(text) < len(text):
            held.append(j)
    return held


def server_meta(url):
    m = {}
    try:
        p = get(url + '/props')
        m['props'] = {k: v for k, v in p.items() if 'template' not in k}
    except Exception as e:      # 古い版は /props が無い
        m['props_error'] = str(e)
    try:
        m['models'] = get(url + '/v1/models')
    except Exception as e:
        m['models_error'] = str(e)
    return m


def gpu_snapshot():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,clocks.sm,clocks.mem',
                            '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
        return r.stdout.strip().splitlines()
    except Exception as e:
        return [str(e)]


def collect(a):
    P = json.loads((GOLD / 'prompts.json').read_text())['prompts']
    tok, _, tok_ref = load_tok()
    pb = piece_bytes(tok, tok_ref)
    url = a.url
    t_start = time.time()
    meta = server_meta(url)
    # 最初の 1 回は温め（CUDA graph・置き場の確保）。答えには使わない
    post(url + '/completion', dict(prompt=P[0]['ids'], n_predict=8, temperature=0.0, cache_prompt=False, **NEUTRAL))
    out, gs = [], None
    for p in P:
        body = dict(prompt=p['ids'], n_predict=N_PREDICT, temperature=0.0, n_probs=N_PROBS, cache_prompt=False,
                    stream=False, return_tokens=True, **NEUTRAL)
        t0 = time.time()
        r = post(url + '/completion', body)
        wall = time.time() - t0
        gen = r['tokens']
        probs = r['completion_probabilities']
        held = held_positions(gen, pb)
        top, k = [], 0
        for j, t in enumerate(gen):
            if j in held:
                top.append(None)
                continue
            assert probs[k]['id'] == t, (p['name'], j, probs[k]['id'], t)
            top.append([[q['id'], q['logprob']] for q in probs[k]['top_logprobs']])
            k += 1
        assert k == len(probs), (p['name'], k, len(probs), held)
        # 温度 0 なので出た語 = 上位 20 の 1 位（同点で並びが違うと崩れる → 数えて残す）
        not_top1 = [j for j, (t, tp) in enumerate(zip(gen, top)) if tp is not None and tp[0][0] != t]
        ev = r.get('tokens_evaluated')
        assert ev == len(p['ids']), (p['name'], ev, len(p['ids']))      # cache_prompt false なので全部読み直す
        gs = gs or r.get('generation_settings')
        out.append(dict(name=p['name'], prompt=p['ids'], gen=gen, top=top, held=held, not_top1=not_top1,
                        text=r.get('content'), stop_type=r.get('stop_type'), stopping_word=r.get('stopping_word'),
                        tokens_evaluated=ev, truncated=r.get('truncated'), timings=r.get('timings', {}), wall_s=wall))
        tm = r.get('timings', {})
        print(f'{p["name"]:13s} 文 {len(p["ids"]):5d} 語 → {len(gen):2d} 語（{r.get("stop_type")}）'
              f'  プレフィル {tm.get("prompt_ms", 0):8.1f}ms  デコード {tm.get("predicted_per_second", 0):5.1f} tok/s'
              f'  probs なし {len(held)}  {r.get("content", "")[:50]!r}', flush=True)
    # 語の切り分けをサーバーとも突き合わせる（llama.cpp は NFC をかけないが、文は NFC にしてある）
    tk = []
    for p in json.loads((GOLD / 'prompts.json').read_text())['prompts']:
        s = post(url + '/tokenize', {'content': p['text'], 'add_special': False, 'parse_special': True})['tokens']
        s = s[:p['cut']] if p.get('cut') else s
        tk.append(dict(name=p['name'], same=s == p['ids']))
    print('llama-server の /tokenize と同じ: ' + ', '.join(f'{x["name"]} {"OK" if x["same"] else "違う"}' for x in tk))
    res = dict(schema=SCHEMA, mode='free', source='llama-server（unsloth b12209-mix-b84c2bf・CUDA）',
               made=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               model=dict(path=GGUF, size=os.path.getsize(GGUF)),
               request=dict(n_predict=N_PREDICT, temperature=0.0, n_probs=N_PROBS, cache_prompt=False, return_tokens=True,
                            **NEUTRAL),
               server=dict(url=url, **meta, generation_settings=gs), gpu=gpu_snapshot(), load=os.getloadavg(),
               tokenize_check=tk, elapsed_s=time.time() - t_start, prompts=out)
    (GOLD / a.out).write_text(json.dumps(res, ensure_ascii=False))
    print(f'→ {GOLD / a.out}')


def prefill(a):
    """サーバー自身の揺れ: golden の列を「文＋出た語 j 個」として 1 回のプレフィルで通し、最後の位置の 1 位と上位 20 を取る
    （golden はデコードの道。こちらは mode forced の形 → m1_compare.py golden.json golden-prefill.json で雑音の床が分かる）。
    長い文（--long 語以上）は --stride ごとの位置だけ（取らない位置の gen は null）"""
    G = json.loads((GOLD / 'golden.json').read_text())
    url = a.url
    res = []
    t_all = time.time()
    for p in G['prompts']:
        P, gen = p['prompt'], p['gen']
        js = range(len(gen)) if len(P) < a.long else range(0, len(gen), a.stride)
        pg, pt = [None] * len(gen), [None] * len(gen)
        t0 = time.time()
        for j in js:
            r = post(url + '/completion', dict(prompt=P + gen[:j], n_predict=1, temperature=0.0, n_probs=N_PROBS,
                                               cache_prompt=False, stream=False, return_tokens=True, **NEUTRAL))
            t = r['tokens'][0] if r.get('tokens') else None
            pr = r.get('completion_probabilities') or []      # UTF-8 の途中の語だけだと probs が無い
            pg[j] = t
            pt[j] = [[q['id'], q['logprob']] for q in pr[0]['top_logprobs']] if pr and pr[0]['id'] == t else None
        res.append(dict(name=p['name'], prompt=P, gen=pg, top=pt))
        print(f'{p["name"]:13s} {len(js):2d} 位置 {time.time() - t0:6.1f}s', flush=True)
    out = dict(schema=SCHEMA, mode='forced', source='llama-server のプレフィルの道（文＋golden の語 j 個を 1 回で・n_predict 1）',
               made=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), long=a.long, stride=a.stride,
               elapsed_s=time.time() - t_all, prompts=res)
    (GOLD / 'golden-prefill.json').write_text(json.dumps(out, ensure_ascii=False))
    print(f'→ {GOLD / "golden-prefill.json"}')


def timing(a):
    """速さの基準: n_probs なし（上位 20 を数える手間を入れない）で 3 回、プレフィルの ms とデコードの tok/s の中央値"""
    P = json.loads((GOLD / 'prompts.json').read_text())['prompts']
    url = a.url
    post(url + '/completion', dict(prompt=P[0]['ids'], n_predict=8, temperature=0.0, cache_prompt=False, **NEUTRAL))
    runs = {p['name']: [] for p in P}
    for rep in range(a.reps):
        for p in P:
            r = post(url + '/completion', dict(prompt=p['ids'], n_predict=N_PREDICT, temperature=0.0, cache_prompt=False,
                                               stream=False, **NEUTRAL))
            runs[p['name']].append(r['timings'])
        print(f'{rep + 1} 回目 済み', flush=True)
    rows = []
    for p in P:
        ts = runs[p['name']]
        med = lambda k: statistics.median(t[k] for t in ts)
        rows.append(dict(name=p['name'], prompt_n=ts[0]['prompt_n'], prompt_ms=med('prompt_ms'),
                         prompt_per_second=med('prompt_per_second'), predicted_n=ts[0]['predicted_n'],
                         predicted_ms=med('predicted_ms'), predicted_per_second=med('predicted_per_second'), runs=ts))
        print(f'{p["name"]:13s} {ts[0]["prompt_n"]:5d} 語  プレフィル {med("prompt_ms"):8.1f}ms（{med("prompt_per_second"):7.1f} tok/s）'
              f'  デコード {ts[0]["predicted_n"]:2d} 語 {med("predicted_per_second"):5.1f} tok/s')
    res = dict(made=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), reps=a.reps, n_probs=0, cache_prompt=False,
               server=server_meta(url), gpu=gpu_snapshot(), load=os.getloadavg(), rows=rows)
    (GOLD / 'timings.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f'→ {GOLD / "timings.json"}')


# ---------------- 層ごとの値 ----------------

def forced_seqs(names):
    g = {p['name']: p for p in json.loads((GOLD / 'golden.json').read_text())['prompts']}
    out = []
    for n in names:
        p = g[n]
        P, G = len(p['prompt']), len(p['gen'])
        out.append((n, p['prompt'] + p['gen'][:G - 1], list(range(P - 1, P - 1 + G)), p))
    return out


def top_rows(logits, k=N_PROBS):
    """logits (G, V) → 各位置の上位 k の [id, logprob]（log_softmax は全部の語 V で。llama-server と同じ）"""
    import numpy as np
    lg = logits.astype(np.float64)
    m = lg.max(1, keepdims=True)
    lse = np.log(np.exp(lg - m).sum(1, keepdims=True)) + m
    lp = lg - lse
    rows = []
    for j in range(lp.shape[0]):
        top = np.argpartition(-lp[j], k)[:k]
        top = top[np.argsort(-lp[j][top], kind='stable')]
        rows.append([[int(t), float(lp[j, t])] for t in top])
    return rows


def ref(a):
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ref_qwen35 as R
    names = a.names.split(',') if a.names else REF_NAMES
    S = forced_seqs(names)
    det = {int(x) for x in a.detail.split(',') if x.strip()}
    r = R.Ref(GGUF, workers=a.workers, log=lambda m: print(m, flush=True))
    t0 = time.time()
    try:
        outs = r.forward([s for _, s, _, _ in S], [lp for _, _, lp, _ in S],
                         [None if a.no_h else np.arange(len(s)) for _, s, _, _ in S], det, kv_f16=True)
    finally:
        r.close()
    dt = time.time() - t0
    res = []
    for (n, s, lp, p), o in zip(S, outs):
        if not a.no_h:
            o['h_pos'] = np.arange(len(s))
        R.save_npz(GOLD / f'ref-{n}.npz', o)
        top = top_rows(o['logits'])
        res.append(dict(name=n, prompt=p['prompt'], gen=[t[0][0] for t in top], top=top))
        print(f'ref-{n}.npz  列 {len(s)} 語・h {o["h"].shape}・logits {o["logits"].shape}・途中の値 {len(o["detail"])} 個')
    out = dict(schema=SCHEMA, mode='forced', source='tools/ref_qwen35.py（numpy f32・gguf-py の戻し・K/V は f16 に丸め）',
               made=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), detail_layers=sorted(det), elapsed_s=dt, prompts=res)
    (GOLD / a.out).write_text(json.dumps(out, ensure_ascii=False))
    print(f'→ {GOLD / a.out}（{dt:.0f}s）')


def fill(a):
    """サーバーが probs を落とした位置（golden の held）に、CPU 参照（ref の forced の JSON）の上位 20 を top_fill として足す。
    m1_compare.py はそこで入れ替わったときの僅差の判定にだけ使う（logprob の差には使わない）"""
    path = GOLD / 'golden.json'
    G = json.loads(path.read_text())
    F = {p['name']: p for p in json.loads((GOLD / a.base).read_text())['prompts']}
    for p in G['prompts']:
        if not p['held']:
            continue
        f = F.get(p['name'])
        if f is None:
            print(f'{p["name"]}: {a.base} に無い（held {len(p["held"])} 位置は空のまま）')
            continue
        # forced なので文脈は golden の語の列そのもの（参照の 1 位が違っても文脈は分かれない）。位置の数だけ揃っていればよい
        assert f['prompt'] == p['prompt'] and len(f['top']) == len(p['gen']), p['name']
        tf = [None] * len(p['gen'])
        for j in p['held']:
            tf[j] = f['top'][j]
        p['top_fill'] = tf
        same = sum(f['top'][j][0][0] == p['gen'][j] for j in p['held'])
        gaps = [round(f['top'][j][0][1] - f['top'][j][1][1], 3) for j in p['held']]
        print(f'{p["name"]}: held {p["held"]} に top_fill（参照の 1 位が golden と同じ {same}/{len(p["held"])}・1 と 2 位の差 {gaps}）')
    G['top_fill_source'] = f'{a.base}（tools/ref_qwen35.py・numpy f32・K/V f16。mode forced で golden の語を流した）'
    path.write_text(json.dumps(G, ensure_ascii=False))
    print(f'→ {path}')


def lcpp(a):
    import tempfile
    import numpy as np
    names = a.names.split(',') if a.names else REF_NAMES
    exe = Path(a.exe)
    res = []
    t_all = time.time()
    for n, s, lp, p in forced_seqs(names):
        G = len(lp)
        tmp = tempfile.mkdtemp(prefix=f'lcpp_{n}_', dir='/dev/shm' if a.shm else None)
        t0 = time.time()
        r = subprocess.run([str(exe), GGUF, ','.join(map(str, s)), tmp, r'^(l_out-[0-9]+|result_norm)$', str(a.threads),
                            str(G)], capture_output=True, text=True)
        if r.returncode:
            print(r.stderr[-3000:])
            raise SystemExit(f'lcpp_dump に失敗（{n}）')
        dt = time.time() - t0
        T = len(s)
        ld = lambda f: np.load(os.path.join(tmp, f))
        L = 64
        h = np.stack([ld(f'l_out-{il}.npy').reshape(T, -1) for il in range(L - 1)])
        d = dict(tokens=np.asarray(s, np.int64), h=h, h_pos=np.arange(T), h_last=ld(f'l_out-{L - 1}.npy').reshape(G, -1),
                 out_pos=np.asarray(lp, np.int64), result_norm=ld('result_norm.npy').reshape(G, -1),
                 logits=ld('logits.npy').reshape(G, -1))
        np.savez(GOLD / f'lcpp-{n}.npz', **d)
        for f in os.listdir(tmp):
            os.unlink(os.path.join(tmp, f))
        os.rmdir(tmp)
        top = top_rows(d['logits'])
        res.append(dict(name=n, prompt=p['prompt'], gen=[t[0][0] for t in top], top=top))
        print(f'lcpp-{n}.npz  列 {T} 語・h {h.shape}・h_last {d["h_last"].shape}・logits {d["logits"].shape}（{dt:.0f}s）', flush=True)
    out = dict(schema=SCHEMA, mode='forced', source='llama.cpp の CPU 版（tools/lcpp_dump.cpp・活性は Q8_K・K/V は f16）',
               made=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), elapsed_s=time.time() - t_all, prompts=res)
    (GOLD / 'lcpp.json').write_text(json.dumps(out, ensure_ascii=False))
    print(f'→ {GOLD / "lcpp.json"}')


def layers(a):
    """2 つの手本（CPU 参照 f32 と llama.cpp の CPU）の層ごとの開き → layers.json。エンジンの層ごとの比べの物差しに"""
    import numpy as np
    names = a.names.split(',') if a.names else REF_NAMES
    f64 = lambda x: np.asarray(x, np.float64)
    rel = lambda x, y: float(np.linalg.norm((f64(x) - f64(y)).ravel()) / max(np.linalg.norm(f64(y).ravel()), 1e-30))

    def cos_min(x, y):
        x, y = f64(x), f64(y)
        return float(((x * y).sum(-1) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1))).min())
    out = []
    for n in names:
        r, lc = np.load(GOLD / f'ref-{n}.npz'), np.load(GOLD / f'lcpp-{n}.npz')
        assert (r['tokens'] == lc['tokens']).all(), n
        h, hl, op = r['h'], lc['h'], lc['out_pos']
        assert h.shape[0] == hl.shape[0] + 1 and (r['logit_pos'] == op).all(), (n, h.shape, hl.shape)  # 参照は 64 層、llama.cpp は 63 層＋最後
        per = [dict(layer=il, rel=rel(h[il], hl[il]), cos_min=cos_min(h[il], hl[il])) for il in range(hl.shape[0])]
        per.append(dict(layer=h.shape[0] - 1, rel=rel(h[-1][op], lc['h_last']), cos_min=cos_min(h[-1][op], lc['h_last'])))
        o = dict(name=n, T=int(h.shape[1]), layers=per, result_norm_rel=rel(r['result_norm'], lc['result_norm']),
                 logits_rel=rel(r['logits'], lc['logits']),
                 argmax_same=int((r['logits'].argmax(1) == lc['logits'].argmax(1)).sum()), positions=int(len(op)))
        out.append(o)
        pick = [0, 1, 2, 3, 7, 15, 31, 47, 62, 63]
        print(f'{n}（{o["T"]} 語）層の出口の相対誤差 ' + ' '.join(f'{il}:{per[il]["rel"]:.1e}' for il in pick) +
              f'  cos 最小 {min(p["cos_min"] for p in per):.5f}  result_norm {o["result_norm_rel"]:.1e}'
              f'  logits {o["logits_rel"]:.1e}  1 位が同じ {o["argmax_same"]}/{o["positions"]}')
    (GOLD / 'layers.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f'→ {GOLD / "layers.json"}')


# ---------------- 自前の tokenizer の確かめ・目録 ----------------

def tokcheck(a):
    """tok_ref.py の出力（chat・texts・random_ids の揃った物）に 12 の文を足して、tokcheck の手本にする"""
    tok, _, _ = load_tok()
    base = json.loads(Path(a.base).read_text())
    P = json.loads((GOLD / 'prompts.json').read_text())['prompts']
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    for p in P:
        ids = enc(p['text'])
        e = dict(ids=ids, decoded=tok.decode(ids, skip_special_tokens=False), decoded_skip=tok.decode(ids, skip_special_tokens=True))
        if p['kind'] == 'chat':
            base['chat'].append(dict(name='m1_' + p['name'], messages=p['messages'], tools=p['tools'], kwargs=p['kwargs'],
                                     add_generation_prompt=True, openai_args=False, roundtrip=False,
                                     expect=dict(prompt=p['text'], error=None, **e)))
        else:
            base['texts'].append(dict(name='m1_' + p['name'], text=p['text'], **e))
    out = GOLD / 'tokcheck.json'
    out.write_text(json.dumps(base, ensure_ascii=False))
    print(f'→ {out}（chat {len(base["chat"])}・文 {len(base["texts"])}）。確かめ: target/release/tokcheck {HF} {out}')


def manifest(a):
    rows = []
    for f in sorted(GOLD.rglob('*')):
        if f.is_file() and f.name != 'MANIFEST.md':
            h = hashlib.sha256()
            with open(f, 'rb') as fh:
                for b in iter(lambda: fh.read(1 << 24), b''):
                    h.update(b)
            rows.append((str(f.relative_to(GOLD)), f.stat().st_size, h.hexdigest()))
    t = '| ファイル | バイト | sha256 |\n|---|---:|---|\n' + ''.join(f'| `{n}` | {s:,} | `{h}` |\n' for n, s, h in rows)
    (GOLD / 'MANIFEST.md').write_text(t)
    print(t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['prompts', 'collect', 'prefill', 'timing', 'ref', 'fill', 'lcpp', 'layers', 'tokcheck',
                                     'manifest'])
    ap.add_argument('base', nargs='?', help='tokcheck: tok_ref.py の出力 JSON ／ fill: 置き場の中の ref の JSON（例 ref-fill.json）')
    ap.add_argument('--url', default='http://127.0.0.1:8081')
    ap.add_argument('--out', default=None, help='collect: 書き出す名前（既定 golden.json）／ ref: JSON の名前（既定 ref.json）')
    ap.add_argument('--no-h', action='store_true', help='ref: 層ごとの残差を残さない（top_fill 用）')
    ap.add_argument('--reps', type=int, default=3, help='timing: 回数')
    ap.add_argument('--long', type=int, default=1000, help='prefill: これ以上の長さの文は間引く')
    ap.add_argument('--stride', type=int, default=8, help='prefill: 間引くときの間隔')
    ap.add_argument('--names', default='', help='ref・lcpp: 文の名前（カンマ区切り。既定は ' + ','.join(REF_NAMES) + '）')
    ap.add_argument('--detail', default='0,3', help='ref: 途中の値も残す層（最初の GDN 0 と最初の Attention 3）')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--exe', default=str(ROOT / 'target' / 'lcpp_dump'))
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--shm', action='store_true', help='lcpp: 途中の .npy を /dev/shm に置く')
    a = ap.parse_args()
    if a.step in ('tokcheck', 'fill') and not a.base:
        ap.error(f'{a.step} には JSON の名前が要る')
    if a.out is None:
        a.out = 'ref.json' if a.step == 'ref' else 'golden.json'
    globals()[a.step](a)


if __name__ == '__main__':
    main()
