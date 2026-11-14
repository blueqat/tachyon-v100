//! MTP（src/mtp.rs）の答え合わせと速さ。手本は tools/ref_mtp.py run の出力（logs/mtp/p<i>/）。比べるのは tools/mtpcheck.py。
//!
//!   mtpcheck teacher <dir> <main|file> [本体.gguf] [MTP.gguf]  … 列の全位置で MTP の行（32 行ずつ・全部の行で logits）
//!        → <dir>/<src>/gpu_teacher_{topi.i32,topv.f32,lse.f32,hout.f32,logits.f32}
//!   mtpcheck sim <dir> <main|file> <k> [本体.gguf] [MTP.gguf]  … 温度 0 の回し方（spec::start・step）。本体は記録の再生（Replay）
//!        → <dir>/<src>/gpu_sim<k>.json と gpu_sim<k>_{topi,topv,lse,hout,logits}
//!   mtpcheck bench <main|file> [本体.gguf] [MTP.gguf]           … 1 段・回しの時間（作った入力で）
//! main = 本体の GGUF の blk.64（埋め込み・出力も本体の物）、file = 別ファイル（埋め込み・出力も別ファイルの Q3_K）
use tachyon_v100::cuda::{self, DevBuf};
use tachyon_v100::gguf::Model;
use tachyon_v100::mtp::{DevRow, MtpHead, MtpOpts, Row, Shared, Trace};
use tachyon_v100::spec::{self, Drafter, RowOut, SeqStep, StepOut, Target, Verify};
use std::ptr::null;

const MAIN: &str = "q38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf"; // tachyon_v100::weights_path の下
const MTPF: &str = "q38-27b/mtp-Qwen3.8-27B-Q4_0.gguf";
const TOPK: usize = 20;

fn read_i32(p: &str) -> Vec<i32> {
    std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}")).chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn read_f32(p: &str) -> Vec<f32> {
    std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}")).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn write_f32(p: &str, v: &[f32]) {
    std::fs::write(p, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}
fn write_i32(p: &str, v: &[i32]) {
    std::fs::write(p, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}

fn open_head(src: &str, args: &[String], opts: MtpOpts) -> MtpHead {
    let main_default = tachyon_v100::weights_path(MAIN);
    let mtpf_default = tachyon_v100::weights_path(MTPF);
    let main = args.first().map(String::as_str).unwrap_or(&main_default);
    let mtpf = args.get(1).map(String::as_str).unwrap_or(&mtpf_default);
    let path = match src {
        "main" => main,
        "file" => mtpf,
        _ => panic!("src は main か file"),
    };
    let t0 = std::time::Instant::now();
    let g = Model::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let h = MtpHead::load(&g, opts, Shared::default()).unwrap_or_else(|e| panic!("MTP の読み込み: {e}"));
    eprintln!("MTP {src}: {path}・{:.2}GB・{:.1}s", h.vram_bytes() as f64 / 1e9, t0.elapsed().as_secs_f64());
    h
}

/// 記録した本体（ref_mtp.py の列）を再生する偽の本体。入力が記録の列と同じ間だけ、その位置の argmax と h を返す
struct Replay {
    s: Vec<u32>,
    arg: Vec<u32>,
    h: DevBuf,
    e: usize,
    n_rows: usize,
}

impl Replay {
    fn row(&self, p: usize) -> *const f32 {
        unsafe { self.h.f32p().add(p * self.e) as *const f32 }
    }
}

impl Target for Replay {
    type H = DevRow;
    fn step(&mut self, batch: &[SeqStep]) -> Result<StepOut<DevRow>, String> {
        let mut rows = vec![];
        let mut hs = vec![];
        for st in batch {
            let mut ok = true;
            let mut r = vec![];
            let mut h = vec![];
            for (i, &t) in st.tokens.iter().enumerate() {
                let p = st.pos0 + i;
                ok = ok && p < self.s.len() && t == self.s[p];
                if ok {
                    r.push(RowOut { argmax: self.arg[p], dist: None });
                    h.push(DevRow(self.row(p)));
                } else {
                    // 記録の外（外れた下書きの後ろ）。使われないはず
                    r.push(RowOut { argmax: u32::MAX, dist: None });
                    h.push(DevRow(null()));
                }
                self.n_rows += 1;
            }
            rows.push(r);
            hs.push(h);
        }
        Ok(StepOut { rows, h: hs })
    }
    fn commit(&mut self, _seq: spec::SeqId, pos0: usize, n_keep: usize) -> Result<(), String> {
        if pos0 + n_keep > self.s.len() {
            return Err(format!("再生: 記録の外まで確定 {pos0}+{n_keep}"));
        }
        Ok(())
    }
    fn free(&mut self, _seq: spec::SeqId) {}
}

struct Seq {
    s: Vec<u32>,
    h: Vec<f32>,
    arg: Vec<u32>,
    n_prompt: usize,
    e: usize,
}

fn load_seq(dir: &str) -> Seq {
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/meta.json")).unwrap()).unwrap();
    let e = meta["n_embd"].as_u64().unwrap() as usize;
    let s: Vec<u32> = read_i32(&format!("{dir}/tokens.i32")).into_iter().map(|x| x as u32).collect();
    let h = read_f32(&format!("{dir}/h.f32"));
    assert_eq!(h.len(), s.len() * e);
    let arg: Vec<u32> = read_i32(&format!("{dir}/tgt_argmax.i32")).into_iter().map(|x| x as u32).collect();
    Seq { s, h, arg, n_prompt: meta["n_prompt"].as_u64().unwrap() as usize, e }
}

fn put_top(rows: &[tachyon_v100::mtp::TopRow], topi: &mut Vec<i32>, topv: &mut Vec<f32>, lse: &mut Vec<f32>) {
    for t in rows {
        topi.extend(t.ids[..TOPK].iter().map(|&x| x as i32));
        topv.extend_from_slice(&t.vals[..TOPK]);
        lse.push(t.lse);
    }
}

fn cmd_teacher(dir: &str, src: &str, rest: &[String]) {
    let q = load_seq(dir);
    let n = q.s.len();
    let mut m = open_head(src, rest, MtpOpts { slots: 1, max_ctx: n.next_power_of_two().max(64), topk: 32, kv_pages: 0 });
    let hd = DevBuf::from_f32(&q.h);
    let (mut topi, mut topv, mut lse, mut hout, mut full) = (vec![], vec![], vec![], vec![], vec![]);
    let n_full = 2;
    for c0 in (0..n).step_by(tachyon_v100::mtp::MAX_ROWS) {
        let c1 = (c0 + tachyon_v100::mtp::MAX_ROWS).min(n);
        let rows: Vec<Row> = (c0..c1)
            .map(|p| Row { slot: 0, pos: p, token: q.s[p], h: if p == 0 { null() } else { unsafe { hd.f32p().add((p - 1) * q.e) as *const f32 } } })
            .collect();
        let want: Vec<usize> = (0..rows.len()).collect();
        let t = m.forward(&rows, &want, 32).unwrap();
        put_top(&t, &mut topi, &mut topv, &mut lse);
        hout.extend(m.hout_rows(rows.len()));
        for p in c0..c1 {
            if p + n_full >= n {
                full.extend(m.logits_row(p - c0));
            }
        }
    }
    let o = format!("{dir}/{src}");
    std::fs::create_dir_all(&o).unwrap();
    write_i32(&format!("{o}/gpu_teacher_topi.i32"), &topi);
    write_f32(&format!("{o}/gpu_teacher_topv.f32"), &topv);
    write_f32(&format!("{o}/gpu_teacher_lse.f32"), &lse);
    write_f32(&format!("{o}/gpu_teacher_hout.f32"), &hout);
    write_f32(&format!("{o}/gpu_teacher_logits.f32"), &full);
    eprintln!("teacher {n} 行 → {o}/gpu_teacher_*（forward {} 回）", m.n_forward);
}

fn cmd_sim(dir: &str, src: &str, k: usize, rest: &[String]) {
    let q = load_seq(dir);
    let n = q.s.len();
    let mut m = open_head(src, rest, MtpOpts { slots: 2, max_ctx: (n + 64).next_power_of_two(), topk: 32, kv_pages: 0 });
    m.trace = Some(Trace { full: 4, ..Default::default() });
    let mut tgt = Replay { s: q.s.clone(), arg: q.arg.clone(), h: DevBuf::from_f32(&q.h), e: q.e, n_rows: 0 };
    // ref_mtp.py の sim と同じ長さ: 最後の新しい語は記録の最後の行の argmax（位置 n）まで
    let max_new = n - q.n_prompt + 1;
    let mut seqs = vec![spec::SpecSeq::new(0, spec::Sampling::GREEDY, 0, k, max_new)];
    let stop = |_: u32| false;
    spec::start(&mut tgt, &mut m, &mut seqs[0], &q.s[..q.n_prompt], 64, Verify::Greedy, &stop).unwrap();
    let mut steps = vec![];
    while !seqs[0].done {
        let (pos0, id_last, drafts) = (seqs[0].pos, seqs[0].id_last, seqs[0].drafts.clone());
        spec::step(&mut tgt, &mut m, &mut seqs, Verify::Greedy, &stop).unwrap();
        let n_acc = seqs[0].pos - pos0 - 1;
        steps.push(serde_json::json!({"pos0": pos0, "id_last": id_last, "drafts": drafts, "n_acc": n_acc, "new": seqs[0].id_last}));
    }
    m.free(0);
    let tr = m.trace.take().unwrap();
    let o = format!("{dir}/{src}");
    std::fs::create_dir_all(&o).unwrap();
    let (mut topi, mut topv, mut lse) = (vec![], vec![], vec![]);
    put_top(&tr.rows, &mut topi, &mut topv, &mut lse);
    write_i32(&format!("{o}/gpu_sim{k}_topi.i32"), &topi);
    write_f32(&format!("{o}/gpu_sim{k}_topv.f32"), &topv);
    write_f32(&format!("{o}/gpu_sim{k}_lse.f32"), &lse);
    write_f32(&format!("{o}/gpu_sim{k}_hout.f32"), &tr.hout.concat());
    write_f32(&format!("{o}/gpu_sim{k}_logits.f32"), &tr.logits.concat());
    std::fs::write(format!("{o}/gpu_sim{k}.json"), serde_json::to_string(&steps).unwrap()).unwrap();
    eprintln!("sim k={k}: {}・本体の行 {}・MTP の forward {} 回 {} 行 → {o}/gpu_sim{k}*", seqs[0], tgt.n_rows, m.n_forward, m.n_rows);
}

/// f を何度も回した 1 回の時間（ms）。5 組の一番速い組
fn bench_ms(mut f: impl FnMut()) -> f32 {
    f();
    cuda::sync();
    let one = cuda::time_ms(&mut f).max(1e-3);
    let reps = ((20.0 / one) as usize).clamp(3, 200);
    let mut best = f32::MAX;
    for _ in 0..5 {
        best = best.min(cuda::time_ms(|| (0..reps).for_each(|_| f())) / reps as f32);
    }
    best
}

fn cmd_bench(src: &str, rest: &[String]) {
    let mut m = open_head(src, rest, MtpOpts { slots: 32, max_ctx: 8192, topk: 1, kv_pages: 0 });
    let e = m.dims.n_embd;
    // 作った h（本体の h_nextn は output_norm の後なので大きさ ~1。でたらめでよい）
    let mut rng = 12345u64;
    let hv: Vec<f32> = (0..64 * e)
        .map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
        })
        .collect();
    let hd = DevBuf::from_f32(&hv);
    let hp = |i: usize| unsafe { hd.f32p().add((i % 64) * e) as *const f32 };
    let ctx = 1024usize;
    // 文脈 ctx の KV を 32 升ぶん埋める（32 行ずつ）
    for s in 0..32 {
        for c0 in (0..ctx).step_by(32) {
            let rows: Vec<Row> = (c0..c0 + 32).map(|p| Row { slot: s, pos: p, token: (p * 7 + s) as u32 % 1000, h: hp(p) }).collect();
            m.forward(&rows, &[], 1).unwrap();
        }
    }
    cuda::sync();
    println!("| 形 | 1 回 | 備考 |\n|---|---|---|");
    // 1 段（B 列 × 1 行、logits あり）
    for b in [1usize, 2, 4, 8, 16, 32] {
        let rows: Vec<Row> = (0..b).map(|s| Row { slot: s, pos: ctx, token: 42, h: hp(s) }).collect();
        let want: Vec<usize> = (0..b).collect();
        let t = bench_ms(|| {
            m.forward(&rows, &want, 1).unwrap();
        });
        let no = bench_ms(|| {
            m.forward(&rows, &[], 1).unwrap();
        });
        println!("| 下書き 1 段 B={b}（文脈 {ctx}） | {t:.3}ms | logits なし {no:.3}ms・lm_head＋上位 {:.3}ms |", t - no);
    }
    // 追いつき＋頭（確かめの後: 受け取り a 個なら a+1 行 ＋ 頭 1 行）を B 列
    for (b, a) in [(1usize, 0usize), (1, 2), (8, 2), (16, 0)] {
        let mut rows = vec![];
        let mut want = vec![];
        for s in 0..b {
            for j in 0..=a {
                rows.push(Row { slot: s, pos: ctx + j, token: 42, h: hp(j) });
            }
            want.push(rows.len());
            rows.push(Row { slot: s, pos: ctx + a + 1, token: 43, h: hp(a + 1) });
        }
        let t = bench_ms(|| {
            m.forward(&rows, &want, 1).unwrap();
        });
        println!("| 追いつき {} 行＋頭 1 行 × B={b} | {t:.3}ms | {} 行 |", a + 1, rows.len());
    }
    // 回し方の 1 歩ぶん（run: 追いつき a+1 行 → k 段）
    for (b, k) in [(1usize, 1usize), (1, 2), (1, 3), (4, 2), (8, 2), (8, 3)] {
        let ids: Vec<spec::SeqId> = (100..100 + b as u32).collect();
        // 升を取って前の h を作る（頭の追いつき）
        let base: Vec<u32> = vec![1; ctx];
        let hrow: Vec<DevRow> = (0..ctx).map(|p| DevRow(hp(p))).collect();
        for &id in &ids {
            m.run(&[spec::Observe { seq: id, pos0: 0, tokens: &base, h: &hrow }], &[]).unwrap();
        }
        let toks = [5u32, 6, 7, 8];
        let na = k.min(2) + 1; // 受け取り min(k, 2) 個の後の形: [id_last, d1..] を本体の h で、続けて新しい語から k 段
        let mut cur = ctx;
        let t = bench_ms(|| {
            let obs: Vec<spec::Observe<DevRow>> = ids.iter().map(|&id| spec::Observe { seq: id, pos0: cur, tokens: &toks[..na], h: &hrow[..na] }).collect();
            let asks: Vec<spec::DraftAsk> = ids.iter().map(|&id| spec::DraftAsk { seq: id, id_last: 9, pos: cur + na, k, sample: None }).collect();
            m.run(&obs, &asks).unwrap();
            cur += na;
        });
        println!("| run: 追いつき {na} 行＋下書き k={k} × B={b} | {t:.3}ms | |");
        for &id in &ids {
            m.free(id);
        }
    }
    // プレフィルの追いつき（512 行 = 32 行 × 16 回）
    let rows: Vec<Row> = (0..512).map(|p| Row { slot: 0, pos: p, token: 7, h: hp(p) }).collect();
    let t = bench_ms(|| {
        for c in rows.chunks(32) {
            m.forward(c, &[], 1).unwrap();
        }
    });
    println!("| プレフィルの追いつき 512 行（32 行ずつ） | {t:.3}ms | {:.0} 行/秒 |", 512.0 / (t as f64 * 1e-3));
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("teacher") => cmd_teacher(&a[2], &a[3], &a[4..]),
        Some("sim") => cmd_sim(&a[2], &a[3], a[4].parse().unwrap(), &a[5..]),
        Some("bench") => cmd_bench(&a[2], &a[3..]),
        _ => {
            eprintln!("使い方: mtpcheck teacher <dir> <main|file> | sim <dir> <main|file> <k> | bench <main|file>");
            std::process::exit(2);
        }
    }
}
