//! Tachyon-v100 の M1（Qwen3.8-27B を GPU 1 枚で・1 人・温度 0）。
//!
//!   tachyon generate --prompt-tokens <ids | ファイル> [--n 64] [--json out.json] [--teacher decode|prefill]
//!       貪欲に n 語。出力は golden（tools/m1_golden.py）と同じ形（schema m1-golden/1・mode free|forced・prompts の各々に
//!       name・prompt・gen・top（各位置の上位 20 の [語, logprob]）・text・timings）。
//!       入力のファイルが答え（gen つき）なら --teacher で答えの語を入れて各位置の予測を出す（decode: 1 語ずつのデコードの道、
//!       prefill: 文＋答えを 1 回のプレフィルの道で）。答えの無い入力に --teacher は使えない
//!   tachyon layers --prompt-tokens <ids | ファイル> --npz out.npz [--layers N] [--detail 0,3] [--dump all|last:N] [--logits last|all]
//!       層ごとの残差 h（tools/ref_qwen35.py の .npz と同じ名前: tokens・emb・h・h_pos・logit_pos・result_norm・logits・detail.*）
//!   tachyon bench [--prompt-tokens …] [--lens 5,547,1500] [--decode 64] [--repeat 2]
//!       プレフィルの tok/s（長さごと）とデコードの tok/s（1 人・argmax だけ）
//!   tachyon spec --prompt-tokens <ids | ファイル> [--k 2] [--n 64] [--json out.json] [--mtp-gguf <path>]
//!       M2: MTP（blk.64）の下書き k 語＋本体の確かめ（温度 0・1 人）。出力は generate と同じ形（top は空）に、受け取りの数と速さ。
//!       答えは下書きなしの貪欲と語ごとに同じはず（確かめの道の丸めの差で僅差の所が入れ替わることはある）
//!       --no-gemm-catchup: MTP の追いつきを M2 の g3 の GEMV（32 行ずつ）で（既定は GEMM で 512 行ずつ）
//!       --graph: 確かめを CUDA Graph の道で（Model::verify_step）・--draft-vocab <file>: MTP の lm_head をその語だけに（data/draft_vocab_q38.txt）
//!   tachyon serve --keys keys.json [--port 8080] [--ctx 65536] [--max-prompt 60000] [--max-queue 8] [--thinking-default off] …
//!       OpenAI 互換の口（src/server/。1 本ずつ・待ち行列・429）。既定は SERVE_DEFAULTS、ほかの引数は src/server/cli.rs の頭
//!
//!   共通: --gguf <path>・--ctx 16384（KV の語数）・--chunk 2048（プレフィルの塊）・--gemv-max 32（これ以下の語数は GEMV の道）・
//!         --deq-gemm（33 語以上のプレフィルを M1 の道＝重みを fp16 に戻してから GEMM で。既定は g3 の並びのまま掛ける）・
//!         --gdn-seq（GDN のプレフィルを 1 語ずつの道で）・--check-finite（層ごとに NaN を数える）・--only 0,2（入力の何番目だけ）
//!   GPU は CUDA_VISIBLE_DEVICES で選ぶ（中では 0 番）。重い物は RAM の上限（例: systemd-run --user --scope -p MemoryMax=100G）の下で
use tachyon_v100::gguf;
use tachyon_v100::model::{self, Dump, Model, Opt};
use tachyon_v100::npz;
use tachyon_v100::tokenizer::Tokenizer;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Instant;

/// 既定の本体（tachyon_v100::weights_path の下・TACHYON_WEIGHTS で置き場を替える）
fn default_gguf() -> String {
    tachyon_v100::weights_path("q38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf")
}

struct Args {
    cmd: String,
    kv: HashMap<String, String>,
}

impl Args {
    fn parse() -> Args {
        let mut it = std::env::args().skip(1);
        let cmd = it.next().unwrap_or_else(|| usage());
        let mut kv = HashMap::new();
        let v: Vec<String> = it.collect();
        let mut i = 0;
        while i < v.len() {
            let k = v[i].strip_prefix("--").unwrap_or_else(|| panic!("引数が変: {}", v[i])).to_string();
            if i + 1 < v.len() && !v[i + 1].starts_with("--") {
                kv.insert(k, v[i + 1].clone());
                i += 2;
            } else {
                kv.insert(k, "1".into());
                i += 1;
            }
        }
        Args { cmd, kv }
    }
    fn s(&self, k: &str) -> Option<&str> {
        self.kv.get(k).map(|s| s.as_str())
    }
    fn n(&self, k: &str, d: usize) -> usize {
        self.s(k).map(|s| s.parse().unwrap_or_else(|_| panic!("--{k} は数"))).unwrap_or(d)
    }
    fn flag(&self, k: &str) -> bool {
        self.kv.contains_key(k)
    }
}

fn usage() -> ! {
    eprintln!("使い方: tachyon generate|layers|bench …（src/bin/tachyon.rs の頭）");
    std::process::exit(2)
}

/// 1 つの入力（答えの語があれば gen、golden なら名前も）
struct Case {
    name: Option<String>,
    prompt: Vec<u32>,
    gen: Option<Vec<u32>>,
}

fn ids_of(v: &Value) -> Option<Vec<u32>> {
    let a = v.as_array()?;
    a.iter().map(|x| x.as_u64().map(|u| u as u32)).collect()
}

fn case_of(v: &Value) -> Option<Case> {
    if let Some(p) = ids_of(v) {
        return Some(Case { name: None, prompt: p, gen: None });
    }
    let o = v.as_object()?;
    let pick = |keys: &[&str]| keys.iter().find_map(|k| o.get(*k).and_then(ids_of));
    let prompt = pick(&["prompt", "prompt_tokens", "prompt_ids", "input_ids", "tokens"])?;
    let gen = pick(&["gen", "gen_tokens", "output_tokens", "completion_tokens", "generated", "tokens_out", "greedy"]);
    let name = o.get("name").and_then(|v| v.as_str()).map(String::from);
    Some(Case { name, prompt, gen })
}

/// 入力: "760,6511,314" か、ファイル（数を並べた文・JSON の語の列・その配列・答えの JSON）
fn load_cases(arg: &str) -> Vec<Case> {
    let is_ids = |s: &str| !s.trim().is_empty() && s.chars().all(|c| c.is_ascii_digit() || c == ',' || c.is_whitespace());
    let parse_ids = |s: &str| -> Vec<u32> { s.split(|c: char| c == ',' || c.is_whitespace()).filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect() };
    if is_ids(arg) && !std::path::Path::new(arg).exists() {
        return vec![Case { name: None, prompt: parse_ids(arg), gen: None }];
    }
    let text = std::fs::read_to_string(arg).unwrap_or_else(|e| panic!("{arg}: {e}"));
    if is_ids(&text) {
        return vec![Case { name: None, prompt: parse_ids(&text), gen: None }];
    }
    let v: Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{arg}: JSON でも語の列でもない: {e}"));
    fn walk(v: &Value, out: &mut Vec<Case>) {
        if let Some(c) = case_of(v) {
            out.push(c);
            return;
        }
        match v {
            Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            Value::Object(o) => {
                for k in ["cases", "prompts", "items", "results", "data", "golden"] {
                    if let Some(x) = o.get(k) {
                        walk(x, out);
                        return;
                    }
                }
            }
            _ => {}
        }
    }
    let mut out = vec![];
    walk(&v, &mut out);
    assert!(!out.is_empty(), "{arg}: 入力が 1 つも無い");
    out
}

fn opt_of(a: &Args) -> Opt {
    let d = Opt::default();
    Opt {
        gemv_max: a.n("gemv-max", d.gemv_max),
        chunk: a.n("chunk", d.chunk),
        gdn_seq: a.flag("gdn-seq"),
        check_finite: a.flag("check-finite"),
        deq_gemm: a.flag("deq-gemm"),
    }
}

fn gib(b: usize) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

fn open_model(a: &Args, layers: Option<std::ops::Range<usize>>) -> (gguf::Model, Model) {
    let path = a.s("gguf").map(String::from).unwrap_or_else(default_gguf);
    tachyon_v100::cuda::check(unsafe { tachyon_v100::cuda::ffi::cudaSetDevice(0) }, "cudaSetDevice");
    let (f0, tot) = model::mem_info();
    let g = gguf::Model::open(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let m = Model::load(&g, opt_of(a), layers).unwrap_or_else(|e| panic!("{e}"));
    let (f1, _) = model::mem_info();
    eprintln!(
        "読み込み {:.1}s: 層 {}..{}/{}（出力 {}）、GGUF {:.2} GiB → GPU {:.2} GiB・作業場 {:.2} GiB、GPU の使用 {:.2} GiB（全部 {:.1}）",
        m.info.seconds,
        m.layer0,
        m.layer0 + m.n_loaded(),
        m.cfg.n_layer,
        if m.has_head() { "あり" } else { "なし" },
        gib(m.info.bytes_gguf),
        gib(m.info.bytes_gpu),
        gib(m.info.scratch_bytes),
        gib(f0 - f1),
        gib(tot)
    );
    (g, m)
}

fn tokenizer_for(a: &Args) -> Option<Tokenizer> {
    let dir = match a.s("tokenizer") {
        Some(d) => d.to_string(),
        None => {
            let g = std::path::PathBuf::from(a.s("gguf").map(String::from).unwrap_or_else(default_gguf));
            g.parent()?.join("hf").to_string_lossy().into_owned()
        }
    };
    Tokenizer::from_dir(&dir).ok()
}

fn top_json(logits: &[f32], k: usize) -> Value {
    Value::Array(model::top_logprobs(logits, k).into_iter().map(|(i, lp)| json!([i, lp])).collect())
}

fn select<T>(v: Vec<T>, only: Option<&str>) -> Vec<(usize, T)> {
    let keep: Option<Vec<usize>> = only.map(|s| s.split(',').map(|x| x.trim().parse().unwrap()).collect());
    v.into_iter().enumerate().filter(|(i, _)| keep.as_ref().map_or(true, |k| k.contains(i))).collect()
}

/// 1 語のデコード（既定は M2 の decode: CUDA Graph・融かしたカーネル。--old-decode なら M1 の forward の道）
fn step1(m: &mut Model, seq: &mut model::Seq, prev: u32, old: bool) -> u32 {
    if old {
        m.forward(seq, &[prev], None);
        m.head(0, 1);
        m.argmax(1)[0]
    } else {
        m.decode(&mut [seq], &[prev])[0]
    }
}

// ---------------------------------------------------------------- generate

fn generate(a: &Args) {
    let cases = load_cases(a.s("prompt-tokens").unwrap_or_else(|| usage()));
    let n_gen = a.n("n", 64);
    let k_top = a.n("top", 20);
    let teacher = a.s("teacher").map(|s| if s == "1" { "decode".to_string() } else { s.to_string() });
    let ctx = a.n("ctx", 16384);
    let old_dec = a.flag("old-decode");
    let (_g, mut m) = open_model(a, None);
    let tok = tokenizer_for(a);
    let eos: Vec<u32> = if a.flag("ignore-eos") { vec![] } else { tok.as_ref().map(|t| t.eos.clone()).unwrap_or_else(|| vec![248046, 248044]) };
    let mut seq = m.new_seq(ctx);
    let (f, _) = model::mem_info();
    eprintln!("列の状態 {:.2} GiB（文脈 {} 語）、GPU の空き {:.2} GiB", gib(seq.bytes()), seq.max_ctx, gib(f));
    let mut out = vec![];
    for (ci, c) in select(cases, a.s("only")) {
        seq.reset();
        let p = c.prompt.len();
        let (mut gen, mut tops) = (vec![], vec![]);
        let prompt_ms: f64;
        let (mut dec_ms, mut n_dec) = (0f64, 0usize);
        let push = |m: &Model, gen: &mut Vec<u32>, tops: &mut Vec<Value>, t: u32| {
            gen.push(t);
            if k_top > 0 {
                tops.push(top_json(&m.logits(1), k_top));
            }
        };
        match teacher.as_deref() {
            None | Some("decode") => {
                let fed = if teacher.is_some() { Some(c.gen.as_ref().expect("--teacher には答えの語（gen）が要る")) } else { None };
                let want = fed.map(|g| g.len()).unwrap_or(n_gen);
                let t0 = Instant::now();
                let last = m.prefill(&mut seq, &c.prompt, None);
                m.head(last - 1, 1);
                let t = m.argmax(1)[0];
                prompt_ms = t0.elapsed().as_secs_f64() * 1e3;
                push(&m, &mut gen, &mut tops, t);
                for j in 1..want {
                    let prev = match fed {
                        Some(g) => g[j - 1],
                        None => {
                            if eos.contains(gen.last().unwrap()) {
                                break;
                            }
                            *gen.last().unwrap()
                        }
                    };
                    let t1 = Instant::now();
                    let t = step1(&mut m, &mut seq, prev, old_dec);
                    dec_ms += t1.elapsed().as_secs_f64() * 1e3;
                    n_dec += 1;
                    push(&m, &mut gen, &mut tops, t);
                }
            }
            Some("prefill") => {
                // 文＋答えの頭 G−1 語を 1 回のプレフィルで。位置 P−1 … P+G−2 の予測
                let g = c.gen.as_ref().expect("--teacher prefill には答えの語（gen）が要る");
                let mut all = c.prompt.clone();
                all.extend_from_slice(&g[..g.len() - 1]);
                let t0 = Instant::now();
                let mut done = 0;
                for ch in all.chunks(m.opt.chunk) {
                    m.forward(&mut seq, ch, None);
                    let (lo, hi) = ((p - 1).max(done), done + ch.len());
                    let mut r = lo;
                    while r < hi {
                        let n = (hi - r).min(model::GEMV_B_MAX);
                        m.head(r - done, n);
                        let am = m.argmax(n);
                        let lg = m.logits(n);
                        for i in 0..n {
                            gen.push(am[i]);
                            if k_top > 0 {
                                tops.push(top_json(&lg[i * m.cfg.vocab..(i + 1) * m.cfg.vocab], k_top));
                            }
                        }
                        r += n;
                    }
                    done += ch.len();
                }
                prompt_ms = t0.elapsed().as_secs_f64() * 1e3;
            }
            Some(x) => panic!("--teacher は decode か prefill（{x}）"),
        }
        let text = tok.as_ref().map(|t| t.decode(&gen, false)).unwrap_or_default();
        eprintln!(
            "#{ci} 文 {p} 語 → {} 語  プレフィル {prompt_ms:.0}ms（{:.0} tok/s）  デコード {:.1} tok/s  {:?}",
            gen.len(),
            p as f64 / prompt_ms * 1e3,
            if n_dec > 0 { n_dec as f64 / dec_ms * 1e3 } else { 0.0 },
            text.chars().take(60).collect::<String>()
        );
        let mut o = json!({
            "prompt": c.prompt,
            "gen": gen,
            "top": tops,
            "text": text,
            "timings": {
                "prompt_n": p, "prompt_ms": prompt_ms, "prompt_per_second": p as f64 / prompt_ms * 1e3,
                "predicted_n": n_dec, "predicted_ms": dec_ms,
                "predicted_per_second": if n_dec > 0 { n_dec as f64 / dec_ms * 1e3 } else { 0.0 },
            },
            "stop_type": if eos.contains(gen.last().unwrap()) && teacher.is_none() { "eos" } else { "limit" },
        });
        if let Some(n) = &c.name {
            o["name"] = json!(n);
        } else {
            o["name"] = json!(format!("p{ci}"));
        }
        if let Some(t) = &teacher {
            o["teacher"] = json!(t);
            o["fed"] = json!(c.gen);
        }
        out.push(o);
    }
    // golden（tools/m1_golden.py）と同じ形: mode は free（自分の語で続ける）か forced（答えの語を入れる = teacher forcing）
    let all = json!({
        "schema": "m1-golden/1",
        "mode": if teacher.is_some() { "forced" } else { "free" },
        "source": format!("tachyon-v100 M1（{}・{:?}）", teacher.as_deref().map_or("貪欲".to_string(), |t| format!("teacher {t}")), m.opt),
        "made": format!("{:?}", std::time::SystemTime::now()),
        "request": { "n_predict": n_gen, "temperature": 0.0, "n_probs": k_top },
        "prompts": out,
    });
    let s = serde_json::to_string(&all).unwrap();
    match a.s("json") {
        Some(p) => std::fs::write(p, s).unwrap_or_else(|e| panic!("{p}: {e}")),
        None => println!("{s}"),
    }
}

// ---------------------------------------------------------------- layers

fn layers(a: &Args) {
    let cases = load_cases(a.s("prompt-tokens").unwrap_or_else(|| usage()));
    let c = select(cases, a.s("only")).into_iter().next().expect("入力が無い").1;
    let toks = c.prompt;
    let t = toks.len();
    // --layers N（0..N）か a:b（層 a..b。a > 0 なら --h-in に層 a の入口の残差 [T][5120] の f32 の生のファイル）
    let range = a.s("layers").map(|s| match s.split_once(':') {
        Some((x, y)) => x.parse::<usize>().unwrap()..y.parse::<usize>().unwrap(),
        None => 0..s.parse::<usize>().unwrap(),
    });
    let h_in: Option<Vec<f32>> = a.s("h-in").map(|p| {
        let b = std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"));
        b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    });
    let out = a.s("npz").unwrap_or_else(|| usage()).to_string();
    let (_g, mut m) = open_model(a, range);
    assert!(m.layer0 == 0 || h_in.is_some(), "層 {} からなので --h-in が要る", m.layer0);
    let mut seq = m.new_seq(a.n("ctx", t.max(64)));
    let detail: Vec<usize> = a.s("detail").map(|s| s.split(',').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect()).unwrap_or_default();
    let mut dump = Dump { detail_layers: detail, ..Default::default() };
    let t0 = Instant::now();
    let last = m.prefill_from(&mut seq, &toks, h_in.as_deref(), Some(&mut dump));
    tachyon_v100::cuda::sync();
    eprintln!("{t} 語（{} 語ずつ {} 塊）・{} 層を {:.2}s（写しを含む）", m.opt.chunk, (t + m.opt.chunk - 1) / m.opt.chunk, m.n_loaded(),
              t0.elapsed().as_secs_f64());
    let d = m.cfg.d;
    // 層ごとの残差を残す位置
    let dump_pos: Vec<usize> = match a.s("dump").unwrap_or("all") {
        "all" => (0..t).collect(),
        s if s.starts_with("last:") => {
            let n: usize = s[5..].parse().unwrap();
            (t.saturating_sub(n)..t).collect()
        }
        s => panic!("--dump は all か last:N（{s}）"),
    };
    let mut h = Vec::with_capacity(dump.h.len() * dump_pos.len() * d);
    for hl in &dump.h {
        for &p in &dump_pos {
            h.extend_from_slice(&hl[p * d..(p + 1) * d]);
        }
    }
    let tokens: Vec<i64> = toks.iter().map(|&x| x as i64).collect();
    let h_pos: Vec<i64> = dump_pos.iter().map(|&x| x as i64).collect();
    let mut arrays: Vec<(String, Vec<usize>, Vec<f32>)> = vec![];
    let (mut logit_pos, mut rn, mut lg) = (vec![], vec![], vec![]);
    if m.has_head() {
        let pos: Vec<usize> = match a.s("logits").unwrap_or("last") {
            "last" => vec![t - 1],
            "all" => (0..t).collect(),
            s => s.split(',').map(|x| x.parse().unwrap()).collect(),
        };
        // h は最後の塊の分だけ作業場に残っている
        let first = t - last;
        for &p in &pos {
            assert!(p >= first, "--logits の位置 {p} は最後の塊（{first}..）より前（--chunk を大きく）");
            m.head(p - first, 1);
            rn.extend(m.result_norm(1));
            lg.extend(m.logits(1));
            logit_pos.push(p as i64);
        }
        let am = model::top_logprobs(&lg[lg.len() - m.cfg.vocab..], 10);
        eprintln!("最後の位置の上位: {:?}", am.iter().map(|(i, l)| (*i, (l * 1e4).round() / 1e4)).collect::<Vec<_>>());
    }
    for (name, (w, v)) in &dump.detail {
        arrays.push((format!("detail.{name}"), vec![v.len() / w, *w], v.clone()));
    }
    let layer0 = [m.layer0 as i64];
    let mut list: Vec<(&str, Vec<usize>, npz::Data)> = vec![
        ("tokens", vec![t], npz::Data::I64(&tokens)),
        ("h", vec![dump.h.len(), dump_pos.len(), d], npz::Data::F32(&h)),
        ("h_pos", vec![h_pos.len()], npz::Data::I64(&h_pos)),
        ("layer0", vec![1], npz::Data::I64(&layer0)),
    ];
    if m.layer0 == 0 {
        list.push(("emb", vec![t, d], npz::Data::F32(&dump.emb)));
    }
    if !logit_pos.is_empty() {
        list.push(("logit_pos", vec![logit_pos.len()], npz::Data::I64(&logit_pos)));
        list.push(("result_norm", vec![logit_pos.len(), d], npz::Data::F32(&rn)));
        list.push(("logits", vec![logit_pos.len(), m.cfg.vocab], npz::Data::F32(&lg)));
    }
    for (n, s, v) in &arrays {
        list.push((n.as_str(), s.clone(), npz::Data::F32(v)));
    }
    npz::write(&out, &list).unwrap_or_else(|e| panic!("{out}: {e}"));
    eprintln!("→ {out}");
}

// ---------------------------------------------------------------- bench

fn bench(a: &Args) {
    let lens: Vec<usize> = a.s("lens").unwrap_or("5,547,1500").split(',').map(|x| x.parse().unwrap()).collect();
    let n_dec = a.n("decode", 64);
    let rep = a.n("repeat", 2).max(1);
    let base: Vec<u32> = match a.s("prompt-tokens") {
        Some(p) => load_cases(p).into_iter().flat_map(|c| c.prompt).collect(),
        None => (0..4096u32).map(|i| 1000 + (i * 7919) % 150000).collect(),
    };
    let ctx = a.n("ctx", lens.iter().max().unwrap() + n_dec + 64);
    let old_dec = a.flag("old-decode");
    let (_g, mut m) = open_model(a, None);
    if let Some(bs) = a.s("batch") {
        return bench_batch(a, &mut m, &base, bs, n_dec, rep);
    }
    let mut seq = m.new_seq(ctx);
    let (f, _) = model::mem_info();
    eprintln!("列の状態 {:.2} GiB（文脈 {} 語）、GPU の空き {:.2} GiB", gib(seq.bytes()), seq.max_ctx, gib(f));
    let mut rows = vec![];
    for &l in &lens {
        let toks: Vec<u32> = base.iter().cycle().take(l).copied().collect();
        let (mut best_p, mut best_d) = (f64::MAX, f64::MAX);
        for _ in 0..rep {
            seq.reset();
            tachyon_v100::cuda::sync();
            let t0 = Instant::now();
            let last = m.prefill(&mut seq, &toks, None);
            m.head(last - 1, 1);
            let mut t = m.argmax(1)[0];
            let pm = t0.elapsed().as_secs_f64() * 1e3;
            let t1 = Instant::now();
            for _ in 0..n_dec {
                t = step1(&mut m, &mut seq, t, old_dec);
            }
            let dm = t1.elapsed().as_secs_f64() * 1e3;
            best_p = best_p.min(pm);
            best_d = best_d.min(dm);
        }
        let r = json!({
            "prompt_n": l, "prefill_ms": best_p, "prefill_tok_s": l as f64 / best_p * 1e3,
            "decode_n": n_dec, "decode_ms": best_d, "decode_tok_s": n_dec as f64 / best_d * 1e3,
            "decode_ms_per_tok": best_d / n_dec.max(1) as f64,
        });
        eprintln!(
            "文 {l:5} 語: プレフィル {best_p:8.1}ms（{:7.1} tok/s）  デコード {n_dec} 語 {:6.2}ms/語（{:5.1} tok/s）",
            l as f64 / best_p * 1e3,
            best_d / n_dec.max(1) as f64,
            n_dec as f64 / best_d * 1e3
        );
        rows.push(r);
    }
    let (f, tot) = model::mem_info();
    let o = json!({ "rows": rows, "gpu_used_gib": gib(tot - f), "opt": format!("{:?}", m.opt), "repeat": rep });
    let s = serde_json::to_string_pretty(&o).unwrap();
    match a.s("json") {
        Some(p) => std::fs::write(p, s).unwrap(),
        None => println!("{s}"),
    }
}

// ---------------------------------------------------------------- serve

/// serve の既定（書かなければ入れる）: 1 本ずつ・待ち 8 本・文脈 64K・入力 6 万語まで・考えなし・出力は既定 8K／最大 32K・
/// プレフィルは 2,048 語ずつ（M1 は塊ごとに重みを fp16 に戻すので、塊を小さくすると遅い）・短いプレフィルは GEMV の道
const SERVE_DEFAULTS: &[(&str, &str)] = &[
    ("ctx", "65536"),
    ("max-context", "65536"),
    ("max-prompt", "60000"),
    // M3（11-11）: 同時 4 本まで（KV の頁を 4 本で分け合う。合計 67,584 語）
    ("max-running", "4"),
    ("max-seqs", "4"),
    ("kv-tokens", "67584"),
    ("max-queue", "8"),
    ("max-inflight", "64"),
    ("step-tokens", "2048"),
    ("prefill-chunk", "512"),
    ("prefill-chunk-alone", "2048"),
    ("chunk", "2048"),
    // 短いプレフィルを GEMV の道で 32 語ずつ回すのはやめた（0）: g3 のままの GEMM（M2）の方が 40 語から速い（2026-11-11 M2c の実測: 64 語 212→137ms・128 語 423→230ms）
    ("gemv-split-max", "0"),
    ("default-max-tokens", "8192"),
    ("max-output", "32768"),
    ("thinking-default", "off"),
    // 26 → 28（M8: mmproj 0.85GiB を載せる。画像の符号化の作業場は依頼のときだけ 0.7GiB まで）
    ("max-vram-gib", "28"),
    // MTP の下書き 3 語（M2。0 で切る）
    ("mtp-k", "6"),
    // 4 語目から後ろの下書きは、MTP の 1 位の確率の積が 0.7 以上の時だけ（確かめが 5 語以上だと 1 歩が 4ms 重い）。
    // M2c の口越しの実測（温度 0）: k=3 と比べて コード 550 105.6→120.5・道具 102.7→110.4・コード 8K 91.1→97.2、日本語は同じ
    ("draft-min-p", "0"),
    ("draft-min-p-deep", "0.7"),
];

/// tachyon serve …: M1 のモデルを OpenAI 互換の口に載せる（引数は tachyon_v100::server::cli と同じ＋ --gguf・--ctx・--chunk・--max-vram-gib）
fn serve_main(rest: Vec<String>) -> i32 {
    let mut argv = vec!["serve".to_string()];
    argv.extend(rest.iter().cloned());
    let has = |k: &str| rest.iter().any(|x| x == &format!("--{k}"));
    for (k, v) in SERVE_DEFAULTS {
        if !has(k) {
            argv.push(format!("--{k}"));
            argv.push(v.to_string());
        }
    }
    if !has("gguf") {
        argv.extend(["--gguf".to_string(), default_gguf()]);
    }
    // MTP の下書きの語彙（絞った lm_head）: 書かなければ、本体の隣の draft_vocab.txt があれば使う（無ければ全部の語）。--draft-vocab none で切る
    if !has("draft-vocab") {
        if let Some(p) = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.join("draft_vocab.txt"))).filter(|p| p.is_file()) {
            argv.extend(["--draft-vocab".to_string(), p.display().to_string()]);
        }
    }
    // 重みの版は --gguf の物（<GGUF>.sha256 か計算）
    if !has("weights") && !has("weights-sha") {
        let i = argv.iter().position(|x| x == "--gguf").unwrap();
        let g = argv[i + 1].clone();
        argv.extend(["--weights".to_string(), g]);
    }
    let make: tachyon_v100::server::cli::MakeEngine = Box::new(|a: &tachyon_v100::server::cli::ServeArgs| {
        let d = Opt::default();
        let opt = Opt { gemv_max: a.n("gemv-max", d.gemv_max)?, chunk: a.n("chunk", 2048)?, gdn_seq: a.flag("gdn-seq"), check_finite: a.flag("check-finite"), deq_gemm: a.flag("deq-gemm") };
        let cfg = tachyon_v100::server::m1::M1Config {
            gguf: a.s("gguf").map(String::from).unwrap_or_else(default_gguf).into(),
            ctx: a.n("ctx", 65536)?,
            opt,
            max_vram_gib: a.s("max-vram-gib").map(|s| s.parse::<f64>().map_err(|_| "--max-vram-gib は数".to_string())).transpose()?.unwrap_or(0.0),
            gemv_split_max: a.n("gemv-split-max", 0)?,
            mtp_k: a.n("mtp-k", 0)?,
            prefix_cache: !a.flag("no-prefix-cache"),
            prefix_slots: a.n("prefix-slots", 8)?,
            prefix_per_account: a.n("prefix-per-account", 2)?,
            mmproj: a.s("mmproj").map(std::path::PathBuf::from),
            mtp_gguf: a.s("mtp-gguf").map(std::path::PathBuf::from),
            max_seqs: a.n("max-seqs", 1)?,
            kv_tokens: a.n("kv-tokens", 0)?,
            verify_rows: a.n("verify-rows", 8)?,
            draft_vocab: a.s("draft-vocab").filter(|p| *p != "none").map(std::path::PathBuf::from),
            verify_graph: !a.flag("no-verify-graph"),
            gemm_catchup: !a.flag("no-gemm-catchup"),
            draft_min_p: a.s("draft-min-p").map(|x| x.parse::<f32>().map_err(|_| "--draft-min-p は数".to_string())).transpose()?.unwrap_or(0.0),
            draft_min_p_deep: a.s("draft-min-p-deep").map(|x| x.parse::<f32>().map_err(|_| "--draft-min-p-deep は数".to_string())).transpose()?.unwrap_or(0.0),
            // 頭の控えの RAM の段（2026-11-14）: --host-cache-gib 24（0 か無しなら使わない）・--host-ckpt-every 4096・--host-min-tokens 2048（区切りで切ると短い依頼は 1 割ほど遅くなるので、短い物は取らない）・
            // --host-min-avail-gib 40（RAM の空きがこれを切ったら段を広げない）
            host_cache: {
                let gib = a.s("host-cache-gib").map(|x| x.parse::<f64>().map_err(|_| "--host-cache-gib は数".to_string())).transpose()?.unwrap_or(0.0);
                let avail = a.s("host-min-avail-gib").map(|x| x.parse::<f64>().map_err(|_| "--host-min-avail-gib は数".to_string())).transpose()?.unwrap_or(40.0);
                (gib > 0.0 && !a.flag("no-prefix-cache")).then(|| tachyon_v100::server::hostcache::HostCacheConfig {
                    cap_bytes: (gib * (1u64 << 30) as f64) as usize,
                    every: a.n("host-ckpt-every", 4096).unwrap_or(4096),
                    min_tokens: a.n("host-min-tokens", 2048).unwrap_or(2048),
                    min_avail: (avail * (1u64 << 30) as f64) as usize,
                    node: None,
                })
            },
        };
        Ok(tachyon_v100::server::m1::loader(cfg))
    });
    tachyon_v100::server::cli::main_with(&argv, Some(make))
}

/// 何人分もまとめたデコード: --batch 1,8,16（B ごとに B 本の列を --lens の最初の長さでプレフィルしてから、n_dec 歩まとめて）
fn bench_batch(a: &Args, m: &mut Model, base: &[u32], bs: &str, n_dec: usize, rep: usize) {
    let bs: Vec<usize> = bs.split(',').map(|x| x.parse().unwrap()).collect();
    let l = a.s("lens").map(|s| s.split(',').next().unwrap().parse().unwrap()).unwrap_or(13usize);
    let ctx = a.n("ctx", l + n_dec + 64);
    let mut rows = vec![];
    for &b in &bs {
        let mut seqs = m.new_seqs(b, ctx);
        let (f, _) = model::mem_info();
        let mut best = f64::MAX;
        let mut last: Vec<u32> = vec![];
        for _ in 0..rep {
            // 列ごとに違う文（同じ長さ）
            let mut toks = vec![];
            for (i, s) in seqs.iter_mut().enumerate() {
                s.reset();
                let p: Vec<u32> = base.iter().cycle().skip(i * 17).take(l).copied().collect();
                let last_n = m.prefill(s, &p, None);
                m.head(last_n - 1, 1);
                toks.push(m.argmax(1)[0]);
            }
            // 温め（グラフを取る）
            {
                let mut v: Vec<&mut model::Seq> = seqs.iter_mut().collect();
                toks = m.decode(&mut v, &toks);
            }
            let t1 = Instant::now();
            for _ in 1..n_dec {
                let mut v: Vec<&mut model::Seq> = seqs.iter_mut().collect();
                toks = m.decode(&mut v, &toks);
            }
            best = best.min(t1.elapsed().as_secs_f64() * 1e3);
            last = toks.clone();
        }
        let steps = (n_dec - 1).max(1) as f64;
        let ms = best / steps;
        eprintln!(
            "B={b:2}: 1 歩 {ms:6.2}ms  1 人 {:5.1} tok/s  合計 {:6.1} tok/s（文 {l} 語・{} 歩・GPU の空き {:.1} GiB）  最後の語 {:?}",
            1e3 / ms,
            b as f64 * 1e3 / ms,
            n_dec - 1,
            gib(f),
            &last[..last.len().min(4)]
        );
        rows.push(json!({ "batch": b, "prompt_n": l, "steps": n_dec - 1, "step_ms": ms, "per_user_tok_s": 1e3 / ms,
                          "total_tok_s": b as f64 * 1e3 / ms, "last": last }));
    }
    let st = m.decode_stats();
    let o = json!({ "batch_rows": rows, "captures": st.captures, "capture_ms": st.capture_ms, "repeat": rep });
    let s = serde_json::to_string_pretty(&o).unwrap();
    match a.s("json") {
        Some(p) => std::fs::write(p, s).unwrap(),
        None => println!("{s}"),
    }
}

// ---------------------------------------------------------------- spec（M2: MTP の下書き）

/// 語の番号を 1 行に 1 つ並べたファイル（# から後ろは読まない）
fn read_ids_file(p: &str) -> Vec<u32> {
    std::fs::read_to_string(p)
        .unwrap_or_else(|e| panic!("{p}: {e}"))
        .lines()
        .map(|l| l.split('#').next().unwrap().trim())
        .filter(|l| !l.is_empty())
        .map(|l| l.parse().unwrap_or_else(|_| panic!("{p}: 数でない行 {l:?}")))
        .collect()
}

fn spec_main(a: &Args) {
    use tachyon_v100::mtp::{MtpHead, MtpOpts};
    use tachyon_v100::spec::{self, Sampling, SpecSeq, Verify};
    let cases = load_cases(a.s("prompt-tokens").unwrap_or_else(|| usage()));
    let n_gen = a.n("n", 64);
    let k = a.n("k", 2);
    let max_p = cases.iter().map(|c| c.prompt.len()).max().unwrap_or(0);
    let ctx = a.n("ctx", (max_p + n_gen + 64).max(1024));
    let (g, mut m) = open_model(a, None);
    let tok = tokenizer_for(a);
    let eos: Vec<u32> = if a.flag("ignore-eos") { vec![] } else { tok.as_ref().map(|t| t.eos.clone()).unwrap_or_else(|| vec![248046, 248044]) };
    let gm;
    let mg: &gguf::Model = match a.s("mtp-gguf") {
        Some(p) => {
            gm = gguf::Model::open(p).unwrap_or_else(|e| panic!("{p}: {e}"));
            &gm
        }
        None => &g,
    };
    let share = if a.s("mtp-gguf").is_some() && a.flag("mtp-own-head") { Default::default() } else { m.mtp_shared(&g) };
    // --mtp-log <file>: 下書きの上位 2 つを記録して JSON 行で書く（分析用）
    let mtp_log = a.s("mtp-log");
    let mut head = MtpHead::load(mg, MtpOpts { slots: 1, max_ctx: ctx, topk: if mtp_log.is_some() { 2 } else { 1 }, kv_pages: 0 }, share)
        .unwrap_or_else(|e| panic!("MTP: {e}"));
    if mtp_log.is_some() {
        head.log = Some(vec![]);
    }
    head.gemm_catchup = !a.flag("no-gemm-catchup");
    head.draft_min_p = a.s("draft-min-p").map(|x| x.parse().expect("--draft-min-p は数")).unwrap_or(0.0);
    head.draft_min_p_deep = a.s("draft-min-p-deep").map(|x| x.parse().expect("--draft-min-p-deep は数")).unwrap_or(0.0);
    if let Some(p) = a.s("draft-vocab") {
        let ids = read_ids_file(p);
        head.set_draft_vocab(&g, &ids).unwrap_or_else(|e| panic!("下書きの語彙: {e}"));
        eprintln!("下書きの lm_head を {} 語に絞った（{p}）", ids.len());
    }
    let chunk = m.opt.chunk;
    let mut tg = model::ModelTarget::new(&mut m, ctx, k + 1);
    tg.graph = a.flag("graph");
    let (f, _) = model::mem_info();
    eprintln!("MTP を載せた（k = {k}・文脈 {ctx}）、GPU の空き {:.2} GiB", gib(f));
    let stop = |t: u32| eos.contains(&t);
    let mut out = vec![];
    let (mut all_dec, mut all_n, mut all_steps, mut all_acc, mut all_drafted) = (0f64, 0usize, 0usize, 0usize, 0usize);
    for (ci, c) in select(cases, a.s("only")) {
        tg.reset();
        spec::Drafter::free(&mut head, 0);
        let rep = a.n("repeat", 1).max(1);
        let mut best: Option<(f64, f64, SpecSeq, f64, f64, usize)> = None;
        let mut hist_best: Vec<usize> = vec![];
        for _ in 0..rep {
            tg.reset();
            spec::Drafter::free(&mut head, 0);
            tg.verify_ms = 0.0;
            tg.commit_ms = 0.0;
            tg.n_verify = 0;
            tg.acc_hist.clear();
            let mut s = SpecSeq::new(0, Sampling::GREEDY, 0, k, n_gen);
            let t0 = Instant::now();
            spec::start(&mut tg, &mut head, &mut s, &c.prompt, chunk, Verify::Greedy, &stop).unwrap_or_else(|e| panic!("start: {e}"));
            tachyon_v100::cuda::sync();
            let ttft = t0.elapsed().as_secs_f64() * 1e3;
            let t1 = Instant::now();
            while spec::step(&mut tg, &mut head, std::slice::from_mut(&mut s), Verify::Greedy, &stop).unwrap_or_else(|e| panic!("step: {e}")) > 0 {}
            tachyon_v100::cuda::sync();
            let dec = t1.elapsed().as_secs_f64() * 1e3;
            if best.as_ref().map_or(true, |b| dec < b.1) {
                best = Some((ttft, dec, s, tg.verify_ms, tg.commit_ms, tg.n_verify));
                hist_best = tg.acc_hist.clone();
            }
        }
        let (ttft, dec, s, vms, cms, nv) = best.unwrap();
        if let (Some(p), Some(l)) = (mtp_log, head.log.as_mut()) {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(p).unwrap();
            let full: Vec<u32> = c.prompt.iter().copied().chain(s.out.iter().copied()).collect();
            writeln!(f, "{}", json!({ "case": ci, "name": c.name, "prompt_len": c.prompt.len(), "full": full,
                "drafts": l.iter().map(|d| json!({ "pos": d.pos, "last": d.last, "tops": d.tops })).collect::<Vec<_>>() })).unwrap();
            l.clear();
        }
        let n_after = s.out.len().saturating_sub(1);
        let tps = if dec > 0.0 { n_after as f64 / dec * 1e3 } else { 0.0 };
        let acc_len = if s.n_steps > 0 { (s.n_accepted + s.n_steps) as f64 / s.n_steps as f64 } else { 0.0 };
        all_dec += dec;
        all_n += n_after;
        all_steps += s.n_steps;
        all_acc += s.n_accepted;
        all_drafted += s.n_drafted;
        let text = tok.as_ref().map(|t| t.decode(&s.out, false)).unwrap_or_default();
        eprintln!(
            "#{ci} 文 {} 語 → {} 語  最初の 1 語 {ttft:.0}ms  その後 {tps:.1} tok/s  歩 {}・下書き {}・受け取り {}（{:.0}%・1 歩 {acc_len:.2} 語）  確かめ {:.2}ms/歩・commit {:.2}ms/歩  {:?}",
            c.prompt.len(),
            s.out.len(),
            s.n_steps,
            s.n_drafted,
            s.n_accepted,
            100.0 * s.n_accepted as f64 / s.n_drafted.max(1) as f64,
            vms / nv.max(1) as f64,
            cms / nv.max(1) as f64,
            text.chars().take(50).collect::<String>()
        );
        let mut o = json!({
            "prompt": c.prompt, "gen": s.out, "top": [], "text": text,
            "timings": { "prompt_n": c.prompt.len(), "prompt_ms": ttft, "predicted_n": n_after, "predicted_ms": dec, "predicted_per_second": tps },
            "spec": { "k": k, "acc_hist": hist_best, "steps": s.n_steps, "drafted": s.n_drafted, "accepted": s.n_accepted, "tokens_per_step": acc_len,
                      "verify_ms_per_step": vms / nv.max(1) as f64, "commit_ms_per_step": cms / nv.max(1) as f64 },
            "stop_type": if stop(*s.out.last().unwrap()) { "eos" } else { "limit" },
        });
        if let Some(nm) = &c.name {
            o["name"] = json!(nm);
        }
        out.push(o);
    }
    let tot = json!({ "k": k, "tok_s": all_n as f64 / all_dec * 1e3, "steps": all_steps, "drafted": all_drafted, "accepted": all_acc,
                      "accept_rate": all_acc as f64 / all_drafted.max(1) as f64, "tokens_per_step": (all_acc + all_steps) as f64 / all_steps.max(1) as f64 });
    eprintln!("計: {:.1} tok/s・受け取り {:.1}%・1 歩 {:.2} 語", all_n as f64 / all_dec * 1e3, 100.0 * all_acc as f64 / all_drafted.max(1) as f64,
              (all_acc + all_steps) as f64 / all_steps.max(1) as f64);
    let o = json!({ "schema": "m1-golden/1", "mode": "free", "source": "tachyon spec", "prompts": out, "total": tot });
    let s = serde_json::to_string_pretty(&o).unwrap();
    match a.s("json") {
        Some(p) => std::fs::write(p, s).unwrap(),
        None => println!("{s}"),
    }
}

/// 確かめの道だけの試し（MTP なし）: 答え（gen つき）の文をプレフィルし、答えの語を T 語ずつ forward_verify → 各行の 1 位が答えの次の語か →
/// 受け取りの数をずらしながら commit_verify。--t 3,2,4 は T の回り方。外れた位置は 1・2 位の logit の差つきで出す
fn vcheck(a: &Args) {
    let cases = load_cases(a.s("prompt-tokens").unwrap_or_else(|| usage()));
    let ts: Vec<usize> = a.s("t").unwrap_or("3,2,4,3,5,2").split(',').map(|x| x.parse().unwrap()).collect();
    let tmax = *ts.iter().max().unwrap();
    let max_p = cases.iter().map(|c| c.prompt.len() + c.gen.as_ref().map_or(0, |g| g.len())).max().unwrap_or(0);
    let ctx = a.n("ctx", max_p + 64);
    let (_g, mut m) = open_model(a, None);
    m.enable_spec(tmax);
    let mut seq = m.new_seq_spec(ctx);
    let v = m.cfg.vocab;
    // --ref: 同じ文をもう 1 本の列で 1 語ずつ流し、確かめの各行のロジットと比べる（受け取りは全部）
    let with_ref = a.flag("ref");
    let mut rseq = if with_ref { Some(m.new_seq(ctx)) } else { None };
    let (mut n_rows, mut n_bad) = (0usize, 0usize);
    for (ci, c) in select(cases, a.s("only")) {
        let g = c.gen.as_ref().expect("答えの語（gen）が要る");
        seq.reset();
        if let Some(r) = rseq.as_mut() {
            r.reset();
            m.prefill(r, &c.prompt, None);
        }
        let last = m.prefill(&mut seq, &c.prompt, None);
        m.head(last - 1, 1);
        let first = m.argmax(1)[0];
        let mut bad = vec![];
        if first != g[0] {
            bad.push(format!("プレフィル {first}≠{}", g[0]));
        }
        let (mut idx, mut it) = (0usize, 0usize);
        while idx + 1 < g.len() {
            let t = ts[it % ts.len()].min(g.len() - idx);
            if t < 2 {
                break;
            }
            let pos0 = seq.pos;
            let want_dump = rseq.is_some() && it == 0;
            let attn_layers: Vec<usize> = (0..m.cfg.n_layer).filter(|&il| m.cfg.is_attn(il)).collect();
            let mut da = Dump { detail_layers: attn_layers.clone(), ..Default::default() };
            m.forward_verify_dump(&mut seq, &g[idx..idx + t], if want_dump { Some(&mut da) } else { None });
            m.head(0, t);
            let am = m.argmax(t);
            let lg = m.logits(t);
            for i in 0..t {
                if idx + i + 1 >= g.len() {
                    break;
                }
                n_rows += 1;
                let want = g[idx + i + 1];
                if am[i] != want {
                    let r = &lg[i * v..(i + 1) * v];
                    n_bad += 1;
                    bad.push(format!("位置 {}（pos0 {pos0}・T {t}・行 {i}）: {}≠{}  logit {:.4} / 答えの語 {:.4}", idx + i + 1, am[i], want, r[am[i] as usize], r[want as usize]));
                }
            }
            if let Some(r) = rseq.as_mut() {
                for i in 0..t {
                    let mut db = Dump { detail_layers: attn_layers.clone(), ..Default::default() };
                    m.forward(r, &g[idx + i..idx + i + 1], if want_dump { Some(&mut db) } else { None });
                    if want_dump {
                        // 行 i: 層ごとの残差の相対の差と、Attention の層の出口（attn_gated）の相対の差
                        let d = m.cfg.d;
                        let rel = |x: &[f32], y: &[f32]| {
                            let (mut num, mut den) = (0f64, 0f64);
                            for (a, b) in x.iter().zip(y) {
                                num = num.max((a - b).abs() as f64);
                                den = den.max(b.abs() as f64);
                            }
                            num / den.max(1e-30)
                        };
                        let mut line = format!("    行 {i}（pos {}）残差の差:", pos0 + i);
                        for li in 0..da.h.len() {
                            let x = &da.h[li][i * d..(i + 1) * d];
                            let y = &db.h[li][..d];
                            let e = rel(x, y);
                            if li % 4 == 3 || e > 1e-3 {
                                line += &format!(" {li}:{e:.1e}");
                            }
                        }
                        eprintln!("{line}");
                        let mut l2 = format!("    行 {i} attn_gated の差:");
                        for &il in &attn_layers {
                            let k = format!("attn_gated-{il}");
                            if let (Some(x), Some(y)) = (da.detail.get(&k), db.detail.get(&k)) {
                                let w = x.0;
                                l2 += &format!(" {il}:{:.1e}", rel(&x.1[i * w..(i + 1) * w], &y.1[..w]));
                            }
                        }
                        eprintln!("{l2}");
                    }
                    m.head(0, 1);
                    let rl = m.logits(1);
                    let mut md = 0f32;
                    for j in 0..v {
                        md = md.max((rl[j] - lg[i * v + j]).abs());
                    }
                    if md > a.s("ref-tol").map(|x| x.parse().unwrap()).unwrap_or(0.05) {
                        eprintln!("    位置 {}（pos {}・T {t}・行 {i}）: ロジットの差 最大 {md:.4}", idx + i + 1, pos0 + i);
                    }
                }
                m.commit_verify(&mut seq, t, t);
                idx += t;
                it += 1;
                continue;
            }
            let n_keep = (it * 7 + 3) % t + 1;
            m.commit_verify(&mut seq, t, n_keep);
            idx += n_keep;
            it += 1;
        }
        eprintln!("#{ci} 文 {} 語・答え {} 語: 外れ {}", c.prompt.len(), g.len(), bad.len());
        for b in bad.iter().take(8) {
            eprintln!("    {b}");
        }
    }
    eprintln!("計: 行 {n_rows}・外れ {n_bad}");
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.first().map(String::as_str) == Some("serve") {
        // 終わりは _exit で（exit の後始末＝CUDA の文脈を壊す atexit と、順番決めの糸がエンジン（GPU の置き場・グラフ・MTP）を
        // 片づけるのが重なると SEGV になる。11-11 08:28 に 1 回起きた）。ファイルに書く物は無く、ログは stderr を流してから
        tachyon_v100::server::util::hard_exit(serve_main(raw[1..].to_vec()));
    }
    let a = Args::parse();
    match a.cmd.as_str() {
        "generate" => generate(&a),
        "layers" => layers(&a),
        "bench" => bench(&a),
        "spec" => spec_main(&a),
        "vcheck" => vcheck(&a),
        "vision" => vision_main(&a),
        _ => usage(),
    }
}

/// 画像を符号化して llama.cpp の MTMD_DEBUG_EMBEDDINGS と同じ形（[i32 語][i32 幅][f32 …]）で書く:
/// tachyon vision --mmproj <gguf> --image <file> --out <bin> [--repeat N]
fn vision_main(a: &Args) {
    if let (Some(p), Some(path)) = (a.s("dump-decoded"), a.s("image")) {
        // 読んだ画素だけ（GPU を使わない）
        let img = tachyon_v100::vision::decode(&std::fs::read(path).unwrap()).unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(p, &img.px).unwrap();
        eprintln!("読んだ画素を {p} に（{} × {}・RGB）", img.w, img.h);
        return;
    }
    let mm_default = tachyon_v100::weights_path("q38-27b/mmproj-F16.gguf");
    let mm = a.s("mmproj").unwrap_or(&mm_default);
    let t0 = Instant::now();
    let v = tachyon_v100::vision::Vision::load(mm).unwrap_or_else(|e| panic!("{e}"));
    eprintln!("mmproj を載せた（{:.2}s・{:.2} GiB）", t0.elapsed().as_secs_f64(), gib(v.bytes));
    // --raw rainbow:64: llama-mtmd-debug の -p encode --image rainbow -n 64 と同じ絵（正規化しない 0..1）を直に
    if let Some(r) = a.s("raw") {
        let n: usize = r.split(':').nth(1).unwrap().parse().unwrap();
        let (cx, cy) = (n as f32 / 2.0, n as f32 / 2.0);
        let md = (cx * cx + cy * cy).sqrt();
        let mut px = vec![0f32; n * n * 3];
        for y in 0..n {
            for x in 0..n {
                let (dx, dy) = (x as f32 - cx, y as f32 - cy);
                let mut hue = dy.atan2(dx) / std::f32::consts::TAU;
                if hue < 0.0 {
                    hue += 1.0;
                }
                let sat = ((dx * dx + dy * dy).sqrt() / md).min(1.0);
                let h6 = hue * 6.0;
                let i6 = h6 as i32;
                let f = h6 - i6 as f32;
                let (p, q, t) = (1.0 - sat, 1.0 - sat * f, 1.0 - sat * (1.0 - f));
                let (r, g, b) = match i6 % 6 {
                    0 => (1.0, t, p),
                    1 => (q, 1.0, p),
                    2 => (p, 1.0, t),
                    3 => (p, q, 1.0),
                    4 => (t, p, 1.0),
                    _ => (1.0, p, q),
                };
                px[(y * n + x) * 3..(y * n + x) * 3 + 3].copy_from_slice(&[r, g, b]);
            }
        }
        let e = v.encode_f32(n, n, &px, std::ptr::null_mut()).unwrap_or_else(|e| panic!("{e}"));
        let emb = e.emb.to_f32();
        let sum: f64 = emb.iter().map(|&x| x as f64).sum();
        eprintln!("  {:14} sum {sum:14.6}  first {:.4} {:.4} {:.4}", "final", emb[0], emb[1], emb[2]);
        return;
    }
    let path = a.s("image").unwrap_or_else(|| usage());
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let img = tachyon_v100::vision::decode(&bytes).unwrap_or_else(|e| panic!("{e}"));
    let r = v.preprocess(&img);
    eprintln!("{} × {} → {} × {}", img.w, img.h, r.w, r.h);
    if let Some(p) = a.s("dump-resized") {
        std::fs::write(p, &r.px).unwrap();
    }
    let mut enc = None;
    for _ in 0..a.n("repeat", 1).max(1) {
        let t1 = Instant::now();
        enc = Some(v.encode(&img, std::ptr::null_mut()).unwrap_or_else(|e| panic!("{e}")));
        eprintln!("符号化 {:.1}ms", t1.elapsed().as_secs_f64() * 1e3);
    }
    let e = enc.unwrap();
    let emb = e.emb.to_f32();
    let n = e.nx * e.ny;
    eprintln!("{} 語（{} × {}）× {}", n, e.nx, e.ny, v.out_dim);
    if let Some(p) = a.s("out") {
        let mut b = Vec::with_capacity(8 + emb.len() * 4);
        b.extend_from_slice(&(n as i32).to_le_bytes());
        b.extend_from_slice(&(v.out_dim as i32).to_le_bytes());
        for x in &emb {
            b.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::write(p, b).unwrap();
    }
}
