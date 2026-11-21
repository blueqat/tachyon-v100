#!/usr/bin/env python3
"""Inkling-2-Mini の手本: transformers の PR #51050（add-inkling2-mini・modeling_inkling2_loop.py）を CPU・f32 で動かし、
温度 0 の貪欲の答えと各位置の上位 20 を出す（tools/loop_ref.py と同じ形・schema m1-golden/1・mode free）。

重み: 公式の safetensors（専門家は NVFP4＝層 0..19・MXFP8＝層 20..23、ほかは BF16）。PR の読み込みが専門家を BF16 に戻すので、
専門家は BF16 のまま置き、使うときに f32 に上げて計算する（RAM のため。f32 で全部持つと専門家だけで 98GB）。
⚠️ つまり専門家の値は「NVFP4 を BF16 に丸めた物」（丸めは相対 2^-9 以下）。ほかの重み・計算は全部 f32。
公式の計算は活性も NVFP4 に量子化する（hf_quant_config の required_activation_quantization）が、手本ではしない＝理想の f32。

Usage (CPU only, about 90 GB RAM; torch CPU build). Put a checkout of the transformers pull request #51050 tree first on
PYTHONPATH (it needs Python >= 3.11 and a recent huggingface-hub); point --model at the official checkpoint
(thinkingmachines/Inkling-2-Mini):
  CUDA_VISIBLE_DEVICES= PYTHONPATH=<transformers PR checkout>/src python3 \
      tools/ink_ref.py --model weights/inkling2mini/official --prompts prompts.json --n 32 --out ref.json
  --layers-npz out.npz: 1 つ目の文の各層（実行）の出口の hidden_states も書く（層ごとの突き合わせ用）
"""
import argparse
import json
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--only", default="")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers-npz", default="")
    a = ap.parse_args()
    import torch
    import transformers
    from transformers.models.inkling2_loop import modeling_inkling2_loop as M

    torch.set_num_threads(a.threads)
    print(f"transformers {transformers.__version__} {transformers.__file__}", file=sys.stderr)
    t0 = time.time()
    model = M.Inkling2LoopForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": "cpu"},
                                                                   attn_implementation="eager")
    model.eval()
    # 専門家のほかを f32 に。専門家は BF16 のまま置き、forward で f32 に上げる
    n32 = 0
    for mod in model.modules():
        if isinstance(mod, M.Inkling2LoopExperts):
            continue
        for name, p in list(mod.named_parameters(recurse=False)):
            if p.dtype != torch.float32:
                p.data = p.data.float()
                n32 += p.numel()
        for name, b in list(mod.named_buffers(recurse=False)):
            if b is not None and b.is_floating_point() and b.dtype != torch.float32:
                setattr(mod, name, b.float())

    def experts_forward(self, hidden_states, top_k_index, top_k_weights):
        hs = hidden_states.float()
        out = torch.zeros_like(hs)
        for e in torch.unique(top_k_index).tolist():
            if e >= self.num_experts:
                continue
            pos, tok = torch.where((top_k_index == e).T)  # pos: k の位置, tok: 語
            x = hs[tok]
            gu = torch.nn.functional.linear(x, self.gate_up_proj[e].float())
            gate, up = gu.chunk(2, dim=-1)
            h = self.act_fn(gate, up).float()
            y = torch.nn.functional.linear(h, self.down_proj[e].float()) * top_k_weights[tok, pos, None].float()
            out.index_add_(0, tok, y)
        return out.to(hidden_states.dtype)

    M.Inkling2LoopExperts.forward = experts_forward
    lm = model.model.language_model
    tc = model.config.text_config
    print(f"読み込み {time.time() - t0:.0f}s・f32 にした {n32 / 1e9:.2f}B・層 {len(lm.layers)}・ループ {tc.loop_layer_index_start}..{tc.loop_layer_index_end}×{tc.loop_num_passes}",
          file=sys.stderr)
    # 読めたかの確かめ（初期値のままだと 0）
    assert float(lm.message_input.query.abs().sum()) > 0, "message の query が読めていない"
    assert float(lm.layers[6].mlp.experts.gate_up_proj[0].float().abs().sum()) > 0, "専門家が読めていない"
    cases = json.load(open(a.prompts))
    only = {int(x) for x in a.only.split(",") if x != ""}
    res = []
    eos = [200006, 200010]
    for ci, c in enumerate(cases):
        if only and ci not in only:
            continue
        ids = torch.tensor([c["prompt"]], dtype=torch.long)
        top = []
        t1 = time.time()
        with torch.no_grad():
            if a.layers_npz and not res:
                import numpy as np

                out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
                hs = [x[0].float().numpy() for x in (out.hidden_states or [])]
                np.savez(a.layers_npz, tokens=np.array(c["prompt"]), h=np.stack(hs) if hs else np.zeros(0), logits=out.logits[0].float().numpy())
            g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=a.n,
                               output_logits=True, return_dict_in_generate=True, eos_token_id=eos, pad_token_id=199999)
            for lg in g.logits:
                lp = torch.log_softmax(lg[0].double(), -1)
                v, i = torch.topk(lp, a.top)
                top.append([[int(x), float(y)] for x, y in zip(i, v)])
            gen = [int(x) for x in g.sequences[0, ids.shape[1]:]]
            assert len(gen) == len(top) and all(t[0][0] == x for t, x in zip(top, gen)), "generate の語と logits の 1 位が合わない"
        res.append({"name": c.get("name", f"case{ci}"), "prompt": c["prompt"], "gen": gen, "top": top})
        print(f"#{ci} {c.get('name')} {len(c['prompt'])} 語 → {len(gen)} 語 {time.time() - t1:.0f}s", file=sys.stderr)
        json.dump({"schema": "m1-golden/1", "mode": "free",
                   "source": "transformers PR #51050 modeling_inkling2_loop.py（CPU・f32・専門家は公式 NVFP4/MXFP8 を BF16 に戻して f32 で計算）",
                   "prompts": res}, open(a.out, "w"))


if __name__ == "__main__":
    main()
