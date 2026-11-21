//! GPU の抜き取りの候補（kernels/sample.cu・sampler::prepare_cands）が、手元の sampler（logits 全部）と同じ分布になるかの試し。
//!   samplecheck [--draws 20000] [--vocab 248320]
//! 作り物のロジットの行（尖った・平たい・僅差の山・NaN・同じ値）× 抜き取りの設定（温度・top_k・top_p・min_p・罰）ごとに:
//!   1. GPU の上位 256 語が手元の並べ替えと同じ並びか
//!   2. 同じ種・同じ位置で、手元（sample）と GPU の候補（sample_cands）が同じ語を出す割合と、候補で決められず手元に戻した割合
//!   3. 違う種で引いた 2 つの度数（GPU・手元）の χ² と、同じ手元どうし（違う種）の χ²（揺れの目安）
use tachyon_v100::cuda::{check, DevBuf};
use tachyon_v100::model::{RowCands, CAND_K};
use tachyon_v100::server::sampler::{self, History, SamplingParams};
use std::collections::HashMap;

extern "C" {
    fn infer_mtp_gumbel(
        x: *const f32, n: i32, ld: i64, rows: i32, map: *const i32, gp: *const std::ffi::c_void, ws: *mut std::ffi::c_void, kout: i32, oi: *mut i32,
        s: *mut std::ffi::c_void,
    ) -> i32;
    fn infer_cand_ws_bytes(rows: i32, n: i32, k: i32) -> i64;
    fn infer_cand_topk(
        lg: *const f32, ld: i64, rows: i32, n: i32, k: i32, temps: *const f32, ws: *mut std::ffi::c_void, ws_bytes: i64, ids: *mut u32, vals: *mut f32,
        mz: *mut f64, s: *mut std::ffi::c_void,
    ) -> i32;
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn normal(&mut self) -> f32 {
        let (a, b) = (self.f().max(1e-7), self.f());
        (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
    }
}

fn row(kind: &str, n: usize, rng: &mut Rng) -> Vec<f32> {
    let mut v: Vec<f32> = (0..n).map(|_| rng.normal() * 2.0 - 4.0).collect();
    match kind {
        "peaked" => {
            v[1234] = 18.0;
            v[777] = 15.5;
            v[99] = 14.0;
        }
        "flat" => {
            for i in 0..400 {
                v[(i * 613) % n] = 8.0 + rng.normal() * 0.3;
            }
        }
        "ties" => {
            for i in 0..60 {
                v[(i * 4099) % n] = 9.0 + (i % 3) as f32 * 0.5;
            }
        }
        "nan" => {
            v[5] = f32::NAN;
            v[6] = f32::NAN;
            v[42] = 12.0;
            v[43] = 11.5;
            v[44] = 11.5;
        }
        _ => {
            // 実物らしい形: 上位 30 語ほどに重み
            for i in 0..30 {
                v[(i * 7919 + 13) % n] = 10.0 - i as f32 * 0.25 + rng.normal() * 0.2;
            }
        }
    }
    v
}

fn chi2(a: &HashMap<u32, u32>, b: &HashMap<u32, u32>) -> (f64, usize) {
    let mut keys: Vec<u32> = a.keys().chain(b.keys()).copied().collect();
    keys.sort();
    keys.dedup();
    let (na, nb): (f64, f64) = (a.values().sum::<u32>() as f64, b.values().sum::<u32>() as f64);
    let (mut x, mut df) = (0.0, 0usize);
    // 少ない語はまとめる（期待 5 未満）
    let (mut ra, mut rb) = (0.0, 0.0);
    for k in keys {
        let (ca, cb) = (*a.get(&k).unwrap_or(&0) as f64, *b.get(&k).unwrap_or(&0) as f64);
        if ca + cb < 10.0 {
            ra += ca;
            rb += cb;
            continue;
        }
        let e = ca + cb;
        let (ea, eb) = (e * na / (na + nb), e * nb / (na + nb));
        x += (ca - ea).powi(2) / ea + (cb - eb).powi(2) / eb;
        df += 1;
    }
    if ra + rb > 0.0 {
        let e = ra + rb;
        let (ea, eb) = (e * na / (na + nb), e * nb / (na + nb));
        if ea > 0.0 && eb > 0.0 {
            x += (ra - ea).powi(2) / ea + (rb - eb).powi(2) / eb;
            df += 1;
        }
    }
    (x, df.saturating_sub(1))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str, d: usize| args.iter().position(|x| x == k).map(|i| args[i + 1].parse().unwrap()).unwrap_or(d);
    let draws = get("--draws", 20000);
    let n = get("--vocab", 248320);
    let valid = n - 77; // 終わりの [PAD] は選ばない
    check(unsafe { tachyon_v100::cuda::ffi::cudaSetDevice(0) }, "cudaSetDevice");
    let mut rng = Rng(0x1234_5678_9abc_def1);
    if args.iter().any(|x| x == "--gumbel") {
        std::process::exit(gumbel_check(&mut rng));
    }
    let kinds = ["real", "peaked", "flat", "ties", "nan"];
    let mut hist = History::new(&[11, 1234, 4099, 8198], &SamplingParams { repetition_penalty: 1.1, ..SamplingParams::greedy() });
    for t in [1234u32, 1234, 777, 42, 4099] {
        hist.push(t);
    }
    let base = SamplingParams { temperature: 0.7, top_p: 0.95, top_k: 20, seed: 1, ..SamplingParams::greedy() };
    let cfgs: Vec<(&str, SamplingParams)> = vec![
        ("t0.7 p0.95 k20", base.clone()),
        ("t0.7 p0.8 k20 presence1.5", SamplingParams { top_p: 0.8, presence_penalty: 1.5, ..base.clone() }),
        ("t1.0 p0.95 k0", SamplingParams { temperature: 1.0, top_k: 0, ..base.clone() }),
        ("t1.0 k40 min_p0.05", SamplingParams { temperature: 1.0, top_p: 1.0, top_k: 40, min_p: 0.05, ..base.clone() }),
        ("t0.7 k20 rep1.1 freq0.5", SamplingParams { repetition_penalty: 1.1, frequency_penalty: 0.5, ..base.clone() }),
        ("t0 (貪欲) presence1.5", SamplingParams { temperature: 0.0, presence_penalty: 1.5, ..base.clone() }),
        ("t1.0 k0 p1.0（全部・戻すはず）", SamplingParams { temperature: 1.0, top_k: 0, top_p: 1.0, ..base.clone() }),
    ];
    let rows: Vec<Vec<f32>> = kinds.iter().map(|k| row(k, n, &mut rng)).collect();
    let flat: Vec<f32> = rows.concat();
    let r = rows.len();
    let dl = DevBuf::from_f32(&flat);
    let wsb = unsafe { infer_cand_ws_bytes(r as i32, valid as i32, CAND_K as i32) } as usize;
    assert!(wsb > 0);
    let ws = DevBuf::new(wsb);
    let (di, dv, dm, dt) = (DevBuf::new(r * CAND_K * 4), DevBuf::new(r * CAND_K * 4), DevBuf::new(r * 16), DevBuf::new(r * 4));
    let mut all_ok = true;
    for (name, p) in &cfgs {
        let temps = vec![p.temperature; r];
        check(unsafe { tachyon_v100::cuda::ffi::cudaMemcpy(dt.ptr, temps.as_ptr().cast(), r * 4, 1) }, "h2d");
        let t0 = std::time::Instant::now();
        check(
            unsafe {
                infer_cand_topk(dl.f32p(), n as i64, r as i32, valid as i32, CAND_K as i32, dt.f32p(), ws.ptr, wsb as i64, di.ptr.cast(), dv.f32p(),
                                dm.ptr.cast(), std::ptr::null_mut())
            },
            "cand",
        );
        tachyon_v100::cuda::sync();
        let gpu_ms = t0.elapsed().as_secs_f64() * 1e3;
        let (ib, vb, mb) = (di.to_bytes(), dv.to_f32(), dm.to_bytes());
        for (ri, kind) in kinds.iter().enumerate() {
            let ids: Vec<u32> = ib[ri * CAND_K * 4..(ri + 1) * CAND_K * 4].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            let vals = vb[ri * CAND_K..(ri + 1) * CAND_K].to_vec();
            let m = f64::from_le_bytes(mb[ri * 16..ri * 16 + 8].try_into().unwrap()) as f32;
            let z = f64::from_le_bytes(mb[ri * 16 + 8..ri * 16 + 16].try_into().unwrap());
            let c = RowCands { ids: ids.clone(), vals, m, z };
            let lr = &rows[ri][..];
            // 1. 並び（NaN は -inf、同じ値は番号の小さい順）
            let mut idx: Vec<u32> = (0..valid as u32).collect();
            let key = |i: u32| { let x = lr[i as usize]; if x.is_nan() { f32::NEG_INFINITY } else { x } };
            idx.sort_by(|&a, &b| key(b).total_cmp(&key(a)).then(a.cmp(&b)));
            let order_ok = idx[..CAND_K] == ids[..];
            // z（全部の語）
            let t = if p.temperature > 0.0 { p.temperature } else { 1.0 };
            let mx = (0..valid).map(|i| key(i as u32)).fold(f32::NEG_INFINITY, f32::max);
            let zc: f64 = (0..valid).map(|i| key(i as u32)).filter(|x| *x > f32::NEG_INFINITY).map(|x| (((x - mx) / t) as f64).exp()).sum();
            // 2. 同じ種で同じ語か
            // 分布は (行・設定・罰の数え) で決まるので 1 回だけ作り、引くのは種と位置ごと（sample・sample_cands と同じ）
            let pc = sampler::prepare(lr, valid, p, &hist);
            let pg = sampler::prepare_cands(&c, p, &hist);
            let fb = if pg.is_none() { draws } else { 0 };
            let pg = pg.unwrap_or_else(|| sampler::prepare(lr, valid, p, &hist));
            let mut same = 0usize;
            let (mut hg, mut hc, mut hc2) = (HashMap::new(), HashMap::new(), HashMap::new());
            for d in 0..draws {
                let pos = d as u64;
                let cpu = pc.draw(sampler::uniform(p.seed, pos), None);
                let g = pg.draw(sampler::uniform(p.seed, pos), None);
                same += (g == cpu) as usize;
                *hg.entry(pg.draw(sampler::uniform(p.seed + 1000, pos), None)).or_insert(0u32) += 1;
                *hc.entry(cpu).or_insert(0u32) += 1;
                *hc2.entry(pc.draw(sampler::uniform(p.seed + 2000, pos), None)).or_insert(0u32) += 1;
            }
            let (x1, df1) = chi2(&hg, &hc);
            let (x2, df2) = chi2(&hc2, &hc);
            let ok = order_ok && same == draws && (m - mx).abs() == 0.0 && ((z - zc) / zc).abs() < 1e-7;
            all_ok &= ok;
            println!(
                "{name:32} {kind:6}  上位256 {}  max {}  z 相対差 {:.1e}  同じ種で同じ語 {same}/{draws}  手元へ戻した {fb}  χ²(GPU 対 手元・別の種) {x1:.1}/{df1}  χ²(手元 対 手元) {x2:.1}/{df2}  {}",
                if order_ok { "同じ" } else { "違う" },
                if m == mx { "同じ" } else { "違う" },
                ((z - zc) / zc).abs(),
                if ok { "OK" } else { "NG" }
            );
        }
        println!("  （候補の GPU の時間 {gpu_ms:.2}ms・{r} 行）");
    }
    println!("{}", if all_ok { "全部 OK" } else { "NG あり" });
    std::process::exit(if all_ok { 0 } else { 1 });
}


/// --gumbel: 結び付けの下書き（TACHYON_HASH_RNG）の GPU の Gumbel が手元の spec::gumbel と同じか・GPU の Gumbel-max の語が手元と同じか
fn gumbel_check(rng: &mut Rng) -> i32 {
    use tachyon_v100::mtp::{gpu_gumbel_probe, GParam};
    use tachyon_v100::spec;
    let mut bad = 0;
    // 1. 雑音そのもの
    let ids: Vec<u32> = (0..20000u32).map(|i| i.wrapping_mul(2654435761) % 248320).collect();
    let mut maxd = 0f64;
    for &(seed, pos) in &[(0u64, 0u64), (1, 7), (u64::MAX, 123456789), (0xDEAD_BEEF_1234, 1 << 40)] {
        let g = gpu_gumbel_probe(seed, pos, &ids);
        for (i, &id) in ids.iter().enumerate() {
            let c = spec::gumbel(seed, pos, spec::TAG_SAMPLE, id);
            maxd = maxd.max((g[i] - c).abs());
        }
    }
    println!("gumbel の雑音: GPU と手元の差の最大 {maxd:.3e}（20000 語 × 4 組）");
    if maxd > 1e-9 {
        bad += 1;
    }
    // 2. Gumbel-max の語（行 3 本・位置 300・語彙 60960、map あり・なし）
    let n = 60960usize;
    let rows = 3usize;
    let map: Vec<i32> = (0..n as i32).map(|i| ((i as i64 * 7919) % 248320) as i32).collect();
    let d_x = DevBuf::new(rows * n * 4);
    let d_map = DevBuf::from_i32(&map);
    let d_gp = DevBuf::new(rows * std::mem::size_of::<GParam>());
    let d_ws = DevBuf::new(rows * 8);
    let d_oi = DevBuf::new(rows * 4);
    let (mut same, mut tot) = (0, 0);
    for it in 0..300u64 {
        let x: Vec<f32> = (0..rows * n).map(|_| rng.normal() * 3.0).collect();
        let use_map = it % 2 == 1;
        let gp: Vec<GParam> = (0..rows).map(|r| GParam { seed: 42 + r as u64, pos: 1000 + it, invt: [1.0f32 / 0.7, 1.0, 1.0 / 1.3][r], on: 1 }).collect();
        let mut oi = vec![-1i32; rows];
        unsafe {
            use tachyon_v100::cuda::ffi::cudaMemcpy;
            check(cudaMemcpy(d_x.ptr, x.as_ptr().cast(), rows * n * 4, 1), "x");
            check(cudaMemcpy(d_gp.ptr, gp.as_ptr().cast(), rows * std::mem::size_of::<GParam>(), 1), "gp");
            check(
                infer_mtp_gumbel(
                    d_x.ptr as *const f32, n as i32, n as i64, rows as i32, if use_map { d_map.ptr as *const i32 } else { std::ptr::null() },
                    d_gp.ptr, d_ws.ptr, 1, d_oi.ptr as *mut i32, std::ptr::null_mut(),
                ),
                "gumbel",
            );
            check(cudaMemcpy(oi.as_mut_ptr().cast(), d_oi.ptr, rows * 4, 2), "oi");
        }
        for r in 0..rows {
            let mut best = (f64::NEG_INFINITY, 0usize);
            for i in 0..n {
                // GPU と同じ: f32 に丸めた値で比べ、同点は小さい番号
                let id = if use_map { map[i] as u32 } else { i as u32 };
                let v = (x[r * n + i] as f64 * gp[r].invt as f64 + spec::gumbel(gp[r].seed, gp[r].pos, spec::TAG_SAMPLE, id)) as f32 as f64;
                if v > best.0 {
                    best = (v, i);
                }
            }
            tot += 1;
            same += (oi[r] as usize == best.1) as usize;
        }
    }
    println!("Gumbel-max の語: GPU と手元が同じ {same}/{tot}（f32 の丸めの僅差を除けば全部同じはず）");
    if same * 1000 < tot * 995 {
        bad += 1;
    }
    println!("{}", if bad == 0 { "gumbel: PASS" } else { "gumbel: FAIL" });
    bad
}
