#!/usr/bin/env python3
"""答え（golden の形の JSON・mode forced）2 つの、位置ごとの KL と 1 位の一致を文ごとに。
usage: kl_top.py <手本.json> <比べる物.json> [...]

KL(P_手本 ‖ P_比べる物) ≈ Σ_{i ∈ 手本の上位 N} p_手本(i)·(lp_手本(i) − lp_比べる物(i))。lp_比べる物 は比べる物の上位の中から（無ければその最小の値）。
比べる物の上位を広く（tachyon generate --top 100）取ると近似がよくなる。手本の上位に入らない分は数えない（上位 20 の確率の和は表に出す）。
"""
import json
import math
import sys


def load(p):
    return {x['name']: x for x in json.load(open(p))['prompts']}


def main():
    A = load(sys.argv[1])
    for path in sys.argv[2:]:
        B = load(path)
        tot_n = tot_same = 0
        tot_kl = []
        print(f'== {path}')
        print(f'{"文":14} {"位置":>5} {"1 位一致":>8} {"KL 平均":>9} {"KL 中央":>9} {"KL 最大":>9} {"上位の和":>8}')
        for name, a in A.items():
            if name not in B:
                continue
            b = B[name]
            n = min(len(a['top']), len(b['top']))
            same, kls, mass = 0, [], []
            for j in range(n):
                ta, tb = a['top'][j], b['top'][j]
                if not ta or not tb:
                    continue
                same += ta[0][0] == tb[0][0]
                lb = {int(i): float(l) for i, l in tb}
                floor = min(lb.values())
                kl = sum(math.exp(l) * (l - lb.get(int(i), floor)) for i, l in ta)
                kls.append(kl)
                mass.append(sum(math.exp(l) for _, l in ta))
            if not kls:
                continue
            ks = sorted(kls)
            print(f'{name:14} {len(kls):5d} {same / len(kls) * 100:7.1f}% {sum(kls) / len(kls):9.5f} {ks[len(ks) // 2]:9.5f} {ks[-1]:9.4f} '
                  f'{sum(mass) / len(mass):8.3f}')
            tot_n += len(kls)
            tot_same += same
            tot_kl += kls
        ks = sorted(tot_kl)
        print(f'{"計":14} {tot_n:5d} {tot_same / tot_n * 100:7.2f}% {sum(ks) / len(ks):9.5f} {ks[len(ks) // 2]:9.5f} {ks[-1]:9.4f}')


if __name__ == '__main__':
    main()
