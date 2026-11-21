//! Inkling-2-Mini の答え合わせと速さ（src/inkling.rs の 1 語ずつの道）。
//!
//!   inkcheck <.gguf> <prompts.json> --out out.json [--ref ref.json] [--n 32] [--ctx 4096] [--bench 128]
//!
//! - --ref があれば teacher forcing（mode forced）: 手本の gen を 1 語ずつ入れ、各位置の上位 20 の logprob を書く
//! - 無ければ温度 0 の貪欲（mode free）。<|content_model_end_sampling|>（200006）か <|end_message|>（200010）で止める
//! - --bench N: 1 つ目の文の後に N 語を書いて 1 語あたりの秒を出す（tok/s）
//!   出力は schema m1-golden/1（tools/m1_compare.py が読む）
use serde_json::{json, Value};
use std::time::Instant;
use tachyon_v100::gguf::Model;
use tachyon_v100::inkling::Ink;

fn top_logprobs(logits: &[f32], k: usize) -> Vec<(u32, f64)> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let z: f64 = logits.iter().map(|&x| ((x as f64) - m).exp()).sum();
    let lz = m + z.ln();
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.select_nth_unstable_by(k, |&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let mut top: Vec<usize> = idx[..k].to_vec();
    top.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap().then(a.cmp(&b)));
    top.into_iter().map(|i| (i as u32, logits[i] as f64 - lz)).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let opt = |name: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone());
    let gguf = &args[1];
    let prompts: Value = serde_json::from_str(&std::fs::read_to_string(&args[2]).unwrap()).unwrap();
    let out = opt("--out").expect("--out");
    let n: usize = opt("--n").map_or(32, |v| v.parse().unwrap());
    let ctx: usize = opt("--ctx").map_or(4096, |v| v.parse().unwrap());
    let bench: usize = opt("--bench").map_or(0, |v| v.parse().unwrap());
    let batch: Vec<usize> = opt("--batch").map_or(vec![], |v| v.split(',').map(|x| x.parse().unwrap()).collect());
    let ttft: Vec<usize> = opt("--ttft").map_or(vec![], |v| v.split(',').map(|x| x.parse().unwrap()).collect());
    let refj: Option<Value> = opt("--ref").map(|p| serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap());
    let t0 = Instant::now();
    let g = Model::open(gguf).unwrap();
    let mut ink = Ink::load(&g, ctx).unwrap();
    let mut seq = ink.new_seq(ctx);
    eprintln!("読み込み {:.1}s・重み {:.2}GB・列 {:.2}GB・実行 {}", t0.elapsed().as_secs_f64(), ink.weight_bytes as f64 / 1e9, seq.bytes() as f64 / 1e9,
              ink.cfg.n_slot());
    let eos = [200006u32, 200010];
    for &n in &ttft {
        // 最初の 1 語までの秒（プレフィル＋最初の logits）。語は 1 つ目の文を繰り返した物
        let base: Vec<u32> = prompts[4]["prompt"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let toks: Vec<u32> = base.iter().cycle().take(n).cloned().collect();
        seq.reset();
        let t1 = Instant::now();
        ink.prefill(&mut seq, &toks);
        let _ = ink.logits();
        let dt = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        let mut tk = 0u32;
        for _ in 0..32 {
            ink.step(&mut seq, tk);
            let l = ink.logits();
            tk = l.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
        }
        eprintln!("ttft {n} 語: {dt:.2}s（{:.0} tok/s）・その後の 32 語 {:.1} tok/s", n as f64 / dt, 32.0 / t2.elapsed().as_secs_f64());
    }
    let mut res = vec![];
    for (ci, p) in prompts.as_array().unwrap().iter().enumerate() {
        let name = p["name"].as_str().unwrap_or("case").to_string();
        let toks: Vec<u32> = p["prompt"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let forced: Option<Vec<u32>> = refj.as_ref().and_then(|r| {
            r["prompts"].as_array().unwrap().iter().find(|x| x["name"] == p["name"]).map(|x| x["gen"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect())
        });
        seq.reset();
        let t1 = Instant::now();
        if std::env::var("INK_PF").map_or(true, |v| v != "0") {
            ink.prefill(&mut seq, &toks);
        } else {
            for &t in &toks {
                ink.step(&mut seq, t);
            }
        }
        let mut lg = ink.logits();
        let t_pre = t1.elapsed().as_secs_f64();
        let (mut gen, mut top) = (vec![], vec![]);
        let steps = forced.as_ref().map_or(n, |f| f.len().min(n));
        for j in 0..steps {
            let tl = top_logprobs(&lg, 20);
            let am = tl[0].0;
            gen.push(am);
            top.push(tl.iter().map(|&(i, l)| json!([i, l])).collect::<Vec<_>>());
            let next = forced.as_ref().map_or(am, |f| f[j]);
            if forced.is_none() && eos.contains(&next) {
                break;
            }
            if j + 1 < steps {
                ink.step(&mut seq, next);
                lg = ink.logits();
            }
        }
        eprintln!("#{ci} {name} {} 語（{:.2}s）→ {} 語 {:.2}s", toks.len(), t_pre, gen.len(), t1.elapsed().as_secs_f64());
        res.push(json!({"name": name, "prompt": toks, "gen": gen, "top": top}));
        if ci == 0 && bench > 0 {
            let tb = Instant::now();
            let mut t = gen[0];
            for _ in 0..bench {
                ink.step(&mut seq, t);
                let l = ink.logits();
                t = l.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
            }
            let dt = tb.elapsed().as_secs_f64();
            eprintln!("bench: {bench} 語 {dt:.2}s → {:.1} tok/s（logits を毎回 CPU へ写す込み）", bench as f64 / dt);
        }
    }
    // 同時 B 本: lane ごとに文（5 つを繰り返す）を読み、n 語を 1 歩にまとめて書く。1 本ずつの答え（res の gen）と語が同じかも数える
    let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
    for &b in &batch {
        let mut bs = ink.new_batch(ctx, b);
        let ps = prompts.as_array().unwrap();
        let mut cur = vec![0u32; b];
        let mut gens: Vec<Vec<u32>> = vec![vec![]; b];
        for l in 0..b {
            let toks: Vec<u32> = ps[l % ps.len()]["prompt"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            ink.prefill_lane(&mut bs, l, &toks);
            cur[l] = argmax(&ink.logits());
            gens[l].push(cur[l]);
        }
        let steps = n.max(8);
        let tb = Instant::now();
        for _ in 1..steps {
            ink.step_batch(&mut bs, &cur);
            for l in 0..b {
                cur[l] = argmax(&ink.logits_row(l));
                gens[l].push(cur[l]);
            }
        }
        let dt = tb.elapsed().as_secs_f64();
        // 1 本ずつの答えと比べる（止める語の前まで）
        let (mut same, mut total) = (0usize, 0usize);
        if refj.is_none() {
            for l in 0..b {
                let single: Vec<u32> = res[l % res.len()]["gen"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
                for (j, &tk) in single.iter().enumerate() {
                    if j >= gens[l].len() {
                        break;
                    }
                    total += 1;
                    if gens[l][j] == tk {
                        same += 1;
                    } else {
                        break;
                    }
                }
            }
        }
        eprintln!("batch {b}: {} 歩 {dt:.2}s → 合計 {:.1} tok/s・1 本あたり {:.1} tok/s・1 本ずつと同じ語 {same}/{total}（最初の違いまで）",
                  steps - 1, (b * (steps - 1)) as f64 / dt, (steps - 1) as f64 / dt);
    }
    let mode = if refj.is_some() { "forced" } else { "free" };
    let o = json!({"schema": "m1-golden/1", "mode": mode, "source": "Tachyon-v100 inkling（1 語ずつ・GGUF の並びの GEMV・KV f32）", "prompts": res});
    std::fs::write(&out, serde_json::to_string(&o).unwrap()).unwrap();
}
