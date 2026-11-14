#!/usr/bin/env python3
"""M1 の答え合わせ: エンジンの出力（golden と同じ形の JSON）を llama-server の答え（golden.json）と比べて PASS／FAIL を出す。

使い方（numpy も要らない。どこでも動く）:
  python3 tools/m1_compare.py <golden.json> <engine.json> [-v] [--json 結果.json] [--allow-missing]
  python3 tools/m1_compare.py --selftest          # 判定の決まりを作り物の例で確かめる
  golden は golden/m1/golden.json（作り方は tools/m1_golden.py）

合格（全部満たす。数はオプションで変えられる）:
  1. 比べた位置のうち 1 位の語が同じ割合 >= 95%（--agree-min）。全部の文を合わせて数える
  2. 語が違う位置は、どれも llama.cpp 自身の 1 位と 2 位の logprob の差が 0.1 未満（--tie-gap）＝僅差の入れ替わりだけ
  3. 語が同じ位置で、1 位の logprob の差 |Δ| <= 0.05（--lp-tol）の位置が 99% 以上（--lp-rate-min）
  4. golden の文が全部ある（--allow-missing で外せる）・語の番号（prompt）が同じ・どの文も 1 位置以上比べた

JSON の形（schema "m1-golden/1"。golden もエンジンも同じ。ほかの鍵はあってよい）:
  {"schema": "m1-golden/1",
   "mode": "forced" | "free",       # 位置 j の文脈: forced = 文＋golden の gen[:j]（teacher forcing）、free = 文＋自分の gen[:j]
   "source": "何が出したか",
   "prompts": [{"name": "en_short",              # golden の名前と同じ（12 個。並びは問わない）
                "prompt": [248045, ...],          # 文の語の番号（golden と同じでなければ誤り）
                "gen": [ ... ],                   # 出力の位置 j ごとの 1 位の語（温度 0 の貪欲）。forced では null = その位置は出していない
                "top": [[[id, logprob], ...], ...]   # 位置 j ごとの上位 k（1 位から並べる。k >= 1、2 以上を勧める。golden は 20）か null
               }, ...]}
  - logprob は自然対数の log_softmax で、分母は出力の行の全部（248,320 語。tokenizer の外の [PAD…] 243 語も含む）＝ llama-server と同じ
  - golden（llama-server）は mode free。top が null の位置は、サーバーが UTF-8 の途中で切れる語の probs を落とした所
    （golden の "held"）。そこは "top_fill"（CPU 参照の上位 20。あれば）を 2. の僅差の判定にだけ使う
  - gen は 64 個まで（llama-server が <|im_end|> などで止めた文は短い。最後の語が止める語）
数え方:
  - forced: golden の全部の位置を比べる（エンジンの gen が null・足りない位置は「出していない」に数え、比べない）
  - free: 最初に違った位置までを比べる。そこから先は文脈が違うので比べない（「分かれた後」に数える）。
    最初の違いが僅差なら 2. は通る。M1 の判定には forced を勧める（64 位置を全部比べられる）
"""
import argparse
import json
import math
import sys

SCHEMA = 'm1-golden/1'


def lp_of(top, t):
    """上位の並び top の中の語 t の logprob（無ければ None）"""
    if not top:
        return None
    for i, lp in top:
        if i == t:
            return lp
    return None


def compare_prompt(g, e, mode, a):
    """1 つの文。返り値 dict（位置ごとの数と違った所の並び）"""
    gen_g, top_g = g['gen'], g['top']
    fill = g.get('top_fill') or [None] * len(gen_g)
    gen_e = e.get('gen') or []
    top_e = e.get('top') or []
    n = len(gen_g)
    r = dict(name=g['name'], n=n, compared=0, identical=0, first_mismatch=None, mismatches=[], missing=0, diverged=0,
             lp_n=0, lp_within=0, lp_max=0.0, lp_sum=0.0, lp_worst=None, lp_unknown=0, top_inconsistent=0,
             common_max=0.0)
    for j in range(n):
        te = gen_e[j] if j < len(gen_e) else None
        if mode == 'free' and r['first_mismatch'] is not None:
            r['diverged'] += 1
            continue
        if te is None:
            r['missing'] += 1
            continue
        r['compared'] += 1
        tg = gen_g[j]
        tpe = top_e[j] if j < len(top_e) else None
        tpg = top_g[j]
        if tpe and tpe[0][0] != te:
            r['top_inconsistent'] += 1          # エンジンの top の 1 位と gen が違う（並べ方の誤り？）
        if te == tg:
            r['identical'] += 1
            lg, le = lp_of(tpg, tg), lp_of(tpe, tg)
            if lg is None or le is None:
                r['lp_unknown'] += 1
            else:
                d = abs(le - lg)
                r['lp_n'] += 1
                r['lp_sum'] += d
                r['lp_within'] += d <= a.lp_tol
                if d >= r['lp_max']:
                    r['lp_max'], r['lp_worst'] = d, j
            # 参考: 両方の上位に入っている語の logprob の差の最大
            if tpg and tpe:
                de = dict((i, lp) for i, lp in tpe)
                cm = [abs(de[i] - lp) for i, lp in tpg if i in de]
                if cm:
                    r['common_max'] = max(r['common_max'], max(cm))
        else:
            if r['first_mismatch'] is None:
                r['first_mismatch'] = j
            src = 'golden'
            tp = tpg
            if not tp or len(tp) < 2:
                tp, src = fill[j] if j < len(fill) else None, 'fill'
            gap = (tp[0][1] - tp[1][1]) if tp and len(tp) >= 2 else None
            r['mismatches'].append(dict(pos=j, golden=tg, engine=te, gap=gap, gap_src=src if gap is not None else None,
                                        ok=gap is not None and gap < a.tie_gap,
                                        golden_top2=(tp[:2] if tp else None), engine_top2=(tpe[:2] if tpe else None),
                                        engine_lp_of_golden=lp_of(tpe, tg)))
    return r


def bad_tops(p):
    """top の並びの形の誤り（[id, logprob] の組で、logprob が大きい順）。返り値は誤りの説明の並び"""
    out = []
    for key in ('top', 'top_fill'):
        for j, tp in enumerate(p.get(key) or []):
            if tp is None:
                continue
            ok = isinstance(tp, list) and len(tp) >= 1 and all(
                isinstance(x, list) and len(x) == 2 and isinstance(x[0], int) and isinstance(x[1], (int, float)) for x in tp)
            if not ok:
                out.append(f'{key}[{j}] の形が [[id, logprob], ...] でない')
            elif any(tp[k][1] < tp[k + 1][1] for k in range(len(tp) - 1)):
                out.append(f'{key}[{j}] が logprob の大きい順でない')
            if len(out) >= 3:
                return out
    return out


def compare(gold, eng, a):
    if eng.get('schema') not in (None, SCHEMA):
        print(f'⚠️ schema が {eng.get("schema")!r}（{SCHEMA} のはず）')
    mode = eng.get('mode')
    if mode not in ('free', 'forced'):
        raise SystemExit(f'mode は free か forced（{mode!r}）')
    ge = {p['name']: p for p in eng['prompts']}
    rows, problems = [], []
    for g in gold['prompts']:
        e = ge.get(g['name'])
        if e is None:
            if not a.allow_missing:
                problems.append(f'{g["name"]}: エンジンの出力に無い')
            continue
        if e.get('prompt') is not None and e['prompt'] != g['prompt']:
            problems.append(f'{g["name"]}: 文の語の番号が golden と違う（{len(e["prompt"])} 語／{len(g["prompt"])} 語）')
            continue
        pm = e.get('mode', mode)
        bt = [f'golden の {x}' for x in bad_tops(g)] + [f'エンジンの {x}' for x in bad_tops(e)]
        if pm not in ('free', 'forced') or bt:
            problems.append(f'{g["name"]}: ' + ('; '.join(bt) if bt else f'mode が {pm!r}'))
            continue
        r = compare_prompt(g, e, pm, a)
        if r['compared'] == 0:
            problems.append(f'{g["name"]}: 比べた位置が 0')
        rows.append(r)
    for n in ge:
        if n not in {g['name'] for g in gold['prompts']}:
            print(f'⚠️ golden に無い文 {n}（無視）')
    tot = sum(r['compared'] for r in rows)
    same = sum(r['identical'] for r in rows)
    bad = [(r['name'], m) for r in rows for m in r['mismatches'] if not m['ok']]
    lpn = sum(r['lp_n'] for r in rows)
    lpw = sum(r['lp_within'] for r in rows)
    agree_rate = same / tot if tot else 0.0
    lp_rate = lpw / lpn if lpn else 0.0
    checks = dict(agree=agree_rate >= a.agree_min, flips_near_tie_only=not bad, logprob=lpn > 0 and lp_rate >= a.lp_rate_min,
                  complete=not problems)
    summ = dict(mode=mode, source=eng.get('source'), golden_source=gold.get('source'), prompts=len(rows),
                positions_golden=sum(r['n'] for r in rows), compared=tot, identical=same, agree_rate=agree_rate,
                mismatches=sum(len(r['mismatches']) for r in rows), unacceptable=len(bad),
                logprob_compared=lpn, logprob_within=lpw, logprob_within_rate=lp_rate,
                logprob_absdiff_max=max((r['lp_max'] for r in rows), default=0.0),
                logprob_absdiff_mean=sum(r['lp_sum'] for r in rows) / lpn if lpn else None,
                logprob_unknown=sum(r['lp_unknown'] for r in rows), missing=sum(r['missing'] for r in rows),
                diverged=sum(r['diverged'] for r in rows), top_inconsistent=sum(r['top_inconsistent'] for r in rows),
                criteria=dict(agree_min=a.agree_min, tie_gap=a.tie_gap, lp_tol=a.lp_tol, lp_rate_min=a.lp_rate_min),
                checks=checks, problems=problems, passed=all(checks.values()), per_prompt=rows)
    return summ


def report(s, verbose):
    print(f'mode {s["mode"]}  エンジン: {s["source"]}  ／ golden: {s["golden_source"]}')
    print(f'{"文":13s} {"位置":>4s} {"比べた":>6s} {"同じ":>5s} {"同じ%":>6s} {"最初の違い":>9s} {"違い(僅差)":>10s} '
          f'{"|Δlp|最大":>9s} {"平均":>7s} {"出さず":>5s} {"分岐後":>5s}')
    for r in s['per_prompt']:
        pct = 100.0 * r['identical'] / r['compared'] if r['compared'] else float('nan')
        nm = len(r['mismatches'])
        ok = sum(m['ok'] for m in r['mismatches'])
        fm = '-' if r['first_mismatch'] is None else str(r['first_mismatch'])
        mean = r['lp_sum'] / r['lp_n'] if r['lp_n'] else float('nan')
        print(f'{r["name"]:13s} {r["n"]:4d} {r["compared"]:6d} {r["identical"]:5d} {pct:6.1f} {fm:>9s} {f"{nm}({ok})":>10s} '
              f'{r["lp_max"]:9.4f} {mean:7.4f} {r["missing"]:5d} {r["diverged"]:5d}')
        if verbose:
            for m in r['mismatches']:
                g = 'なし' if m['gap'] is None else f'{m["gap"]:.4f}（{m["gap_src"]}）'
                print(f'    位置 {m["pos"]:2d}: golden {m["golden"]} ／ エンジン {m["engine"]}  llama の 1・2 位の差 {g}'
                      f' → {"僅差 OK" if m["ok"] else "✗"}  golden 上位 2 {m["golden_top2"]}  エンジン上位 2 {m["engine_top2"]}')
            if r['lp_worst'] is not None and r['lp_max'] > 0:
                print(f'    |Δlogprob| 最大は位置 {r["lp_worst"]}（{r["lp_max"]:.4f}）')
    c = s['criteria']
    print(f'計: 比べた {s["compared"]} 位置（golden {s["positions_golden"]}・出さず {s["missing"]}・分岐後 {s["diverged"]}）'
          f'  同じ {s["identical"]}（{100 * s["agree_rate"]:.2f}%）  違い {s["mismatches"]}（僅差でない {s["unacceptable"]}）')
    mean = s['logprob_absdiff_mean']
    print(f'    1 位の logprob: {s["logprob_compared"]} 位置のうち |Δ|<={c["lp_tol"]} が {s["logprob_within"]}'
          f'（{100 * s["logprob_within_rate"]:.2f}%）・最大 {s["logprob_absdiff_max"]:.4f}・平均 '
          f'{"nan" if mean is None else f"{mean:.4f}"}・分からない {s["logprob_unknown"]}'
          + (f'  ⚠️ top の 1 位と gen が違う {s["top_inconsistent"]}' if s['top_inconsistent'] else ''))
    for p in s['problems']:
        print(f'  ✗ {p}')
    k = s['checks']
    print(f'判定: 同じ>={100 * c["agree_min"]:.0f}% {"○" if k["agree"] else "✗"} ／ 違いは僅差（<{c["tie_gap"]}）だけ '
          f'{"○" if k["flips_near_tie_only"] else "✗"} ／ |Δlp|<={c["lp_tol"]} が>={100 * c["lp_rate_min"]:.0f}% '
          f'{"○" if k["logprob"] else "✗"} ／ そろっている {"○" if k["complete"] else "✗"}')
    print('PASS' if s['passed'] else 'FAIL')


# ---------------- 作り物で判定の決まりを確かめる ----------------

def selftest():
    a = argparse.Namespace(agree_min=0.95, tie_gap=0.1, lp_tol=0.05, lp_rate_min=0.99, allow_missing=False)

    def top(t, lp=-0.1, gap=1.0):
        return [[t, lp], [t + 1, lp - gap], [t + 2, lp - gap - 1]]

    def gold(n=64, gaps=None):
        gaps = gaps or {}
        gen = list(range(100, 100 + n))
        return dict(name='p', prompt=[1, 2, 3], gen=gen, top=[top(t, gap=gaps.get(j, 1.0)) for j, t in enumerate(gen)])

    def run(g, e, mode, others=()):
        G = dict(source='g', prompts=[g, *others])
        E = dict(schema=SCHEMA, mode=mode, source='e', prompts=[e, *others])
        return compare(G, E, a)

    def eng(g, flips=(), dlp=None, mode='forced'):
        dlp = dlp or {}
        gen, tp = list(g['gen']), [list(map(list, x)) for x in g['top']]
        for j in flips:
            gen[j] = g['gen'][j] + 1                      # 2 位の語を選んだ
            tp[j] = [[gen[j], tp[j][0][1] + 0.01], [g['gen'][j], tp[j][0][1]]]
            if mode == 'free':                             # 分かれた後は別の語の列
                gen[j + 1:] = [t + 7 for t in gen[j + 1:]]
                tp[j + 1:] = [top(t) for t in gen[j + 1:]]
        for j, d in dlp.items():
            tp[j][0][1] += d
        return dict(name='p', prompt=g['prompt'], gen=gen, top=tp)

    checks = []

    def ok(name, cond):
        checks.append((name, bool(cond)))

    big3 = [dict(gold(), name=f'r{i}') for i in range(3)]

    g = gold()
    s = run(g, eng(g), 'forced')
    ok('同じなら PASS・Δ 0', s['passed'] and s['agree_rate'] == 1 and s['logprob_absdiff_max'] == 0)
    g = gold(gaps={10: 0.05})
    s = run(g, eng(g, flips=[10]), 'forced')
    ok('forced: 僅差の 1 か所の入れ替わりは PASS（63/64 同じ）', s['passed'] and s['identical'] == 63 and s['unacceptable'] == 0)
    g = gold(gaps={10: 0.5})
    s = run(g, eng(g, flips=[10]), 'forced')
    ok('forced: 僅差でない入れ替わりは FAIL', not s['passed'] and s['unacceptable'] == 1)
    g = gold(gaps={10: 0.05})
    s = run(g, eng(g, flips=[10], mode='free'), 'free')
    r = s['per_prompt'][0]
    ok('free: 最初の違いまで比べ、分かれた後は数えない（この文だけなら 10/11 で FAIL）', not s['passed']
       and r['compared'] == 11 and r['diverged'] == 53 and r['first_mismatch'] == 10 and s['unacceptable'] == 0)
    s = run(g, eng(g, flips=[10], mode='free'), 'free', others=big3)
    ok('free: ほかの文が全部同じなら PASS（202/203）', s['passed'] and s['identical'] == 202)
    g = gold(gaps={0: 0.05})
    s = run(g, eng(g, flips=[0], mode='free'), 'free')
    ok('free: 1 文だけで頭から分かれると同じ 0% で FAIL', not s['passed'] and s['agree_rate'] == 0)
    g = gold()
    s = run(g, eng(g, dlp={5: 0.06}), 'forced')
    ok('|Δlp| 0.06 が 1/64（98.4%）は FAIL', not s['passed'] and s['logprob_within'] == 63 and not s['checks']['logprob'])
    big = [dict(gold(), name=f'q{i}') for i in range(2)]
    g = gold()
    s = run(g, eng(g, dlp={5: 0.06}), 'forced', others=big)
    ok('|Δlp| 0.06 が 1/192（99.5%）は PASS', s['passed'] and s['logprob_within'] == 191)
    g = gold()
    s = run(g, eng(g, dlp={5: -0.049}), 'forced')
    ok('|Δlp| 0.049 は内', s['passed'] and s['logprob_within'] == 64)
    g = gold()
    s = compare(dict(prompts=[g, dict(gold(), name='q')]), dict(mode='forced', prompts=[eng(g)]), a)
    ok('文が足りないと FAIL', not s['passed'] and not s['checks']['complete'])
    a.allow_missing = True
    s = compare(dict(prompts=[g, dict(gold(), name='q')]), dict(mode='forced', prompts=[eng(g)]), a)
    ok('--allow-missing なら通す', s['passed'])
    a.allow_missing = False
    g = gold()
    e = eng(g)
    e['prompt'] = [1, 2, 4]
    s = run(g, e, 'forced')
    ok('文の語の番号が違うと FAIL', not s['passed'])
    g = gold(gaps={10: 0.05})
    g['top'][10] = None                              # サーバーが probs を落とした位置
    e = eng(gold(gaps={10: 0.05}), flips=[10])
    s = run(g, e, 'forced')
    ok('golden の top が null で入れ替わると、差が分からないので FAIL', not s['passed'] and s['unacceptable'] == 1)
    g['top_fill'] = [None] * 64
    g['top_fill'][10] = top(g['gen'][10], gap=0.03)
    s = run(g, e, 'forced')
    ok('top_fill があれば僅差の判定に使う', s['passed'] and s['per_prompt'][0]['mismatches'][0]['gap_src'] == 'fill')
    g = gold()
    e = eng(g)
    e['gen'][3] = None
    s = run(g, e, 'forced')
    ok('forced: null の位置は出していないに数え、残りで判定', s['passed'] and s['missing'] == 1 and s['compared'] == 63)
    g = gold()
    e = eng(g)
    e['gen'] = e['gen'][:40]
    s = run(g, e, 'free')
    ok('free: 短い出力は残りを出していないに数える', s['passed'] and s['missing'] == 24)
    g = gold()
    e = eng(g)
    e['top'][7] = list(reversed(e['top'][7]))
    s = run(g, e, 'forced')
    ok('top が logprob の大きい順でないと FAIL', not s['passed'] and not s['checks']['complete'])
    g = gold()
    e = dict(eng(g), mode='teacher')
    s = run(g, e, 'forced')
    ok('文ごとの mode がおかしいと FAIL', not s['passed'])
    for n, c in checks:
        print(f'{"OK" if c else "NG"}  {n}')
    bad = [n for n, c in checks if not c]
    print(f'selftest: {len(checks) - len(bad)}/{len(checks)} OK')
    return not bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('golden', nargs='?')
    ap.add_argument('engine', nargs='?')
    ap.add_argument('--agree-min', type=float, default=0.95)
    ap.add_argument('--tie-gap', type=float, default=0.1)
    ap.add_argument('--lp-tol', type=float, default=0.05)
    ap.add_argument('--lp-rate-min', type=float, default=0.99)
    ap.add_argument('--allow-missing', action='store_true', help='golden の文の一部だけでも判定する')
    ap.add_argument('--json', default='', help='結果を JSON で書く')
    ap.add_argument('-v', '--verbose', action='store_true', help='違った位置を全部出す')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.golden or not a.engine:
        ap.error('golden.json と engine.json が要る')
    gold = json.load(open(a.golden, encoding='utf-8'))
    eng = json.load(open(a.engine, encoding='utf-8'))
    s = compare(gold, eng, a)
    report(s, a.verbose)
    if a.json:
        json.dump(s, open(a.json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    sys.exit(0 if s['passed'] else 1)


if __name__ == '__main__':
    main()
