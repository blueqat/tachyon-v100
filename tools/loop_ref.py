#!/usr/bin/env python3
"""SenseNova-Loop-9B の手本: 公式の PyTorch のコード（github.com/OpenSenseNova/SenseNova-Loop の
src/sensenova_loop/models/modeling_sensenova_loop.py）を CPU・f32 で動かし、温度 0 の貪欲の答えと各位置の上位 20 を出す。

重みは tools/loop_hf_from_gguf.py で公式の BF16 GGUF から戻した safetensors（値は BF16 のまま・計算は f32）。
出力は schema m1-golden/1・mode free（tools/m1_compare.py が読む。エンジンは tachyon generate --teacher decode で同じ語を入れて比べる）

Usage (CPU only, about 80 GB RAM; torch CPU build + transformers):
  CUDA_VISIBLE_DEVICES= python3 tools/loop_ref.py \
      --model weights/loop9b/hf-ref --src <clone of github.com/OpenSenseNova/SenseNova-Loop>/src \
      --prompts logs/loop9b/prompts.json --n 32 --out logs/loop9b/ref.json
  --layers-npz out.npz: 1 つ目の文の各実行の出口の残差（hidden_states。mHC の中は 2 本を平均に）も書く（層ごとの突き合わせ用）
"""
import argparse
import json
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--src", required=True, help="公式の repo の src（sensenova_loop パッケージの親）")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--only", default="")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers-npz", default="")
    a = ap.parse_args()
    sys.path.insert(0, a.src)
    import torch

    torch.set_num_threads(a.threads)
    from sensenova_loop.models.modeling_sensenova_loop import SenseNovaLoopConfig, SenseNovaLoopForConditionalGeneration

    t0 = time.time()
    cfg = SenseNovaLoopConfig.from_pretrained(a.model)
    model = SenseNovaLoopForConditionalGeneration.from_pretrained(a.model, config=cfg, dtype=torch.float32, device_map={"": "cpu"},
                                                                  attn_implementation="eager")
    model.eval()
    lm = model.model.language_model
    print(f"読み込み {time.time() - t0:.0f}s・実行の順 {len(lm._loop_plan)}（{lm._loop_plan[:6]}…）・mtp {model.mtp is not None}", file=sys.stderr)
    # mHC の重みが f32 のまま・読めたか（読めていなければ初期値のまま = bias の 0 以外は 0）
    w = lm.layers[4].mhc_attn.weight
    assert w.dtype == torch.float32 and float(w.abs().sum()) > 0, "mHC の重みが読めていない"
    cases = json.load(open(a.prompts))
    only = {int(x) for x in a.only.split(",") if x != ""}
    res = []
    for ci, c in enumerate(cases):
        if only and ci not in only:
            continue
        ids = torch.tensor([c["prompt"]], dtype=torch.long)
        gen, top = [], []
        t1 = time.time()
        with torch.no_grad():
            if a.layers_npz and not res:
                import numpy as np

                out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
                hs = out.hidden_states  # (埋め込み, 実行ごと 56 個)。mHC の中は [.., 2·d]（2 本を平らに）
                d = cfg.text_config.hidden_size
                h = []
                for x in hs[1:]:
                    x = x[0].float()
                    if x.shape[-1] == 2 * d:
                        x = 0.5 * (x[:, :d] + x[:, d:])
                    h.append(x.numpy())
                np.savez(a.layers_npz, tokens=np.array(c["prompt"]), emb=hs[0][0].float().numpy(), h=np.stack(h),
                         logits=out.logits[0].float().numpy())
            # 公式の推論（sensenova_loop/generation.py）と同じ model.generate の貪欲
            g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=a.n,
                               output_logits=True, return_dict_in_generate=True, eos_token_id=[248046, 248044], pad_token_id=248044)
            for lg in g.logits:
                lp = torch.log_softmax(lg[0].double(), -1)
                v, i = torch.topk(lp, a.top)
                top.append([[int(x), float(y)] for x, y in zip(i, v)])
            gen = [int(x) for x in g.sequences[0, ids.shape[1]:]]
            assert len(gen) == len(top) and all(t[0][0] == x for t, x in zip(top, gen)), "generate の語と logits の 1 位が合わない"
        res.append({"name": c.get("name", f"case{ci}"), "prompt": c["prompt"], "gen": gen, "top": top})
        print(f"#{ci} {c.get('name')} {len(c['prompt'])} 語 → {len(gen)} 語 {time.time() - t1:.0f}s", file=sys.stderr)
        json.dump({"schema": "m1-golden/1", "mode": "free",
                   "source": "公式 modeling_sensenova_loop.py（CPU・f32・重みは公式 BF16 GGUF から戻した物）", "prompts": res},
                  open(a.out, "w"))


if __name__ == "__main__":
    main()
