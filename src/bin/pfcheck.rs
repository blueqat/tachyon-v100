//! プレフィルの GEMM（kernels/gemmg3.cu。g3 の並びの重みを直接読む）の答え合わせと速さ。重みは GGUF から名前で取る。
//!
//!   pfcheck check <gguf> [--only 正規表現] [--m 1,16,37,...]  … 型ごと × M × タイルの組 × 書き出し（f32・f16・足し込み）・SwiGLU・multi・上下（XL）
//!   pfcheck bench <gguf> [--m 16,32,...] [--shape 名前]       … 形 × M の時間・TFLOPS。g3 の GEMV（M ≤ 32）・前の道（fp16 に戻して F16 の GEMM）とも
//!
//! 手本は g3 の並びを f32 に戻した重み（infer_g3_dequant。gguf-py の dequantize とビットまで同じ、g3check deq）と X の f64 の内積。
//! 線: max|y − 手本| / max|手本| < 2e-3（W を fp16 に丸めるぶん。X は fp16 の値をそのまま手本にも使う）
use tachyon_v100::cuda::{self, ffi, DevBuf};
use tachyon_v100::gguf::{Model, TensorView};
use std::ffi::c_void;
use std::ptr::{null, null_mut};

mod k {
    use std::ffi::c_void;
    extern "C" {
        pub fn infer_g3_gemm_ex(
            nmat: i32, qtypes: *const i32, ws: *const *const c_void, ns: *const i32, ys: *const *mut c_void, ldys: *const i32, k: i32, m: i32,
            x: *const c_void, xlo: *const c_void, ldx: i32, mode: i32, cfg: i32, s: *mut c_void,
        ) -> i32;
        pub fn infer_m_g3_deq16(qtype: i32, w: *const c_void, k: i32, n: i32, y: *mut c_void, s: *mut c_void) -> i32;
    }
}

const GGUF: &str = "q38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf"; // tachyon_v100::weights_path の下
const TOL: f64 = 2e-3;

// ---------------------------------------------------------------- 小道具

fn f16_bits(x: f32) -> u16 {
    // f32 → f16（偶数丸め）
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 255 {
        return s | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let e2 = e - 127 + 15;
    if e2 >= 31 {
        return s | 0x7c00;
    }
    if e2 <= 0 {
        if e2 < -10 {
            return s;
        }
        let mm = m | 0x80_0000;
        let sh = (14 - e2) as u32;
        let mut r = mm >> sh;
        let rem = mm & ((1 << sh) - 1);
        let half = 1 << (sh - 1);
        if rem > half || (rem == half && (r & 1) == 1) {
            r += 1;
        }
        return s | r as u16;
    }
    let mut r = ((e2 as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1) {
        r += 1;
    }
    s | r as u16
}
fn f16_f32(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as i32, (h & 0x3ff) as u32);
    let f = if e == 0 {
        m as f32 * (2f32).powi(-24)
    } else if e == 31 {
        if m != 0 { f32::NAN } else { f32::INFINITY }
    } else {
        (1.0 + m as f32 / 1024.0) * (2f32).powi(e - 15)
    };
    if s == 1 { -f } else { f }
}
fn d2h_f32(p: *const c_void, n: usize) -> Vec<f32> {
    let mut v = vec![0f32; n];
    cuda::check(unsafe { ffi::cudaMemcpy(v.as_mut_ptr().cast(), p, n * 4, 2) }, "D2H");
    v
}
fn d2h_f16(p: *const c_void, n: usize) -> Vec<f32> {
    let mut v = vec![0u16; n];
    cuda::check(unsafe { ffi::cudaMemcpy(v.as_mut_ptr().cast(), p, n * 2, 2) }, "D2H");
    v.into_iter().map(f16_f32).collect()
}
fn h2d_u16(v: &[u16]) -> DevBuf {
    DevBuf::from_bytes(unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), v.len() * 2) })
}

/// 決まった乱数（xorshift）。たまに大きな値（残差の外れ値に似せて）
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn x(&mut self) -> f32 {
        let v = self.next() + self.next() + self.next();
        if self.next() > 0.998 { v * 40.0 } else { v }
    }
}

/// 詰め直した重み（GGUF の行 0..n）と、その f32 の値（手本用）
struct G3 {
    buf: DevBuf,
    t: i32,
    k: usize,
    n: usize,
    name: String,
}
fn repack(tv: &TensorView, n: usize) -> G3 {
    let t = tv.qt() as i32;
    let k = tv.k();
    let n = n.min(tv.n_rows());
    let bytes = unsafe { ffi::infer_g3_bytes(t, k as i32, n as i32) };
    assert!(bytes > 0, "{}: g3 が扱わない型 {}", tv.name, tv.type_name());
    let src = DevBuf::from_bytes(tv.rows(0, n));
    let d = DevBuf::new(bytes as usize);
    cuda::check(unsafe { ffi::infer_g3_repack(t, src.ptr, d.ptr, k as i32, n as i32, null_mut()) }, "repack");
    cuda::sync();
    G3 { buf: d, t: unsafe { ffi::infer_g3_type(t) }, k, n, name: format!("{}[{}]({})", tv.name, n, tv.type_name()) }
}
fn deq_f32(w: &G3) -> Vec<f32> {
    let y = DevBuf::new(w.n * w.k * 4);
    cuda::check(unsafe { ffi::infer_g3_dequant(w.t, w.buf.ptr, w.k as i32, w.n as i32, y.f32p(), null_mut()) }, "g3 dequant");
    cuda::sync();
    y.to_f32()
}

/// 手本 ref[t][r] = Σ x[t][i] w[r][i]（f64、スレッドで分ける）
fn ref_gemm(x: &[f64], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut out = vec![0f64; m * n];
    let nth = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(8).min(32);
    let rows_per = (n + nth - 1) / nth;
    std::thread::scope(|s| {
        let mut parts: Vec<(usize, Vec<f64>)> = vec![];
        let hs: Vec<_> = (0..nth)
            .map(|ti| {
                let (r0, r1) = (ti * rows_per, ((ti + 1) * rows_per).min(n));
                s.spawn(move || {
                    let mut o = vec![0f64; m * (r1.saturating_sub(r0))];
                    for r in r0..r1 {
                        let wr = &w[r * k..(r + 1) * k];
                        for t in 0..m {
                            let xt = &x[t * k..(t + 1) * k];
                            let mut a = 0f64;
                            for i in 0..k {
                                a += xt[i] * wr[i] as f64;
                            }
                            o[t * (r1 - r0) + (r - r0)] = a;
                        }
                    }
                    (r0, o)
                })
            })
            .collect();
        for h in hs {
            parts.push(h.join().unwrap());
        }
        for (r0, o) in parts {
            let w = if m == 0 { 0 } else { o.len() / m };
            for t in 0..m {
                for j in 0..w {
                    out[t * n + r0 + j] = o[t * w + j];
                }
            }
        }
    });
    out
}

/// (max|y−手本|/max|手本|, 二乗平均の相対)
fn err(y: &[f32], r: &[f64]) -> (f64, f64) {
    let (mut md, mut mr, mut sd, mut sr) = (0f64, 0f64, 0f64, 0f64);
    for (a, b) in y.iter().zip(r) {
        let d = (*a as f64 - b).abs();
        if !d.is_finite() {
            return (f64::INFINITY, f64::INFINITY);
        }
        md = md.max(d);
        mr = mr.max(b.abs());
        sd += d * d;
        sr += b * b;
    }
    (md / mr.max(1e-30), (sd / sr.max(1e-30)).sqrt())
}

fn gemm_ex(ws: &[&G3], ys: &[*mut c_void], ldys: Option<&[i32]>, m: usize, x: &DevBuf, xlo: Option<&DevBuf>, mode: i32, cfg: i32) -> i32 {
    let ts: Vec<i32> = ws.iter().map(|w| w.t).collect();
    let ps: Vec<*const c_void> = ws.iter().map(|w| w.buf.ptr as *const c_void).collect();
    let ns: Vec<i32> = ws.iter().map(|w| w.n as i32).collect();
    unsafe {
        k::infer_g3_gemm_ex(ws.len() as i32, ts.as_ptr(), ps.as_ptr(), ns.as_ptr(), ys.as_ptr(), ldys.map(|l| l.as_ptr()).unwrap_or(null()),
                            ws[0].k as i32, m as i32, x.ptr, xlo.map(|b| b.ptr as *const c_void).unwrap_or(null()), ws[0].k as i32, mode, cfg,
                            null_mut())
    }
}

struct Args {
    pos: Vec<String>,
    kv: Vec<(String, String)>,
}
impl Args {
    fn parse() -> Args {
        let mut pos = vec![];
        let mut kv = vec![];
        let mut it = std::env::args().skip(1);
        while let Some(a) = it.next() {
            if let Some(k) = a.strip_prefix("--") {
                kv.push((k.to_string(), it.next().unwrap_or_default()));
            } else {
                pos.push(a);
            }
        }
        Args { pos, kv }
    }
    fn s(&self, k: &str) -> Option<&str> {
        self.kv.iter().rev().find(|(a, _)| a == k).map(|(_, v)| v.as_str())
    }
    fn list(&self, k: &str, d: &[usize]) -> Vec<usize> {
        self.s(k).map(|s| s.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or_else(|| d.to_vec())
    }
}

// ---------------------------------------------------------------- check

struct Xs {
    x16: DevBuf,
    lo16: DevBuf,
    xr: Vec<f64>,  // fp16 の値（上だけ）
    xf: Vec<f64>,  // 上 + 下（XL の手本）
}
fn make_x(m: usize, k: usize, seed: u64) -> Xs {
    let mut g = Rng(seed | 1);
    let mut hi = vec![0u16; m * k];
    let mut lo = vec![0u16; m * k];
    let (mut xr, mut xf) = (vec![0f64; m * k], vec![0f64; m * k]);
    for i in 0..m * k {
        let v = g.x();
        let h = f16_bits(v);
        let l = f16_bits(v - f16_f32(h));
        hi[i] = h;
        lo[i] = l;
        xr[i] = f16_f32(h) as f64;
        xf[i] = f16_f32(h) as f64 + f16_f32(l) as f64;
    }
    Xs { x16: h2d_u16(&hi), lo16: h2d_u16(&lo), xr, xf }
}

fn check(a: &Args) -> bool {
    let gdef = tachyon_v100::weights_path(GGUF);
    let path = a.pos.get(1).map(|s| s.as_str()).unwrap_or(&gdef);
    let g = Model::open(path).unwrap();
    let only = a.s("only").map(|s| s.to_string());
    let ms = a.list("m", &[1, 16, 33, 64, 100, 200, 547]);
    let mut ok = true;
    let mut worst = 0f64;
    let mut report = |what: &str, e: (f64, f64), ok: &mut bool| {
        let pass = e.0 < TOL;
        if !pass {
            *ok = false;
        }
        worst = worst.max(e.0);
        println!("{} {what}: max {:.2e} rms {:.2e}", if pass { "OK" } else { "NG" }, e.0, e.1);
    };
    // 1 本ずつ（型ごと）: [名前, 行数]
    let singles = [
        ("blk.0.attn_qkv.weight", 1000), // Q5_K
        ("blk.1.attn_qkv.weight", 1000), // Q4_K
        ("blk.1.ssm_out.weight", 1000),  // Q6_K（k = 6144）
        ("blk.0.ffn_gate.weight", 1000), // IQ4_XS
        ("blk.1.ffn_down.weight", 600),  // IQ4_NL（k = 17408）
        ("blk.0.ssm_out.weight", 1000),  // Q8_0
        ("blk.0.ssm_alpha.weight", 48),  // Q8_0（n = 48）
        ("blk.0.ffn_up.weight", 1000),   // Q3_K → Q6_K
        ("blk.14.ffn_down.weight", 600), // IQ3_S → Q6_K
    ];
    for (name, rows) in singles {
        if only.as_ref().map(|o| !name.contains(o.as_str())).unwrap_or(false) {
            continue;
        }
        let w = repack(&g.tensor(name), rows);
        let wf = deq_f32(&w);
        for &m in &ms {
            let xs = make_x(m, w.k, 7 + m as u64);
            let r = ref_gemm(&xs.xr, &wf, m, w.n, w.k);
            let cfgs: Vec<i32> = if m <= 64 { vec![-1, 0] } else { vec![-1, 1, 2] };
            for &cfg in &cfgs {
                // f32
                let y = DevBuf::new(m * w.n * 4);
                cuda::check(gemm_ex(&[&w], &[y.ptr], None, m, &xs.x16, None, 0, cfg), "gemm");
                cuda::sync();
                report(&format!("{} m={m} cfg={cfg} f32", w.name), err(&y.to_f32(), &r), &mut ok);
                if cfg == -1 {
                    // f16
                    let y16 = DevBuf::new(m * w.n * 2);
                    cuda::check(gemm_ex(&[&w], &[y16.ptr], None, m, &xs.x16, None, 1, cfg), "gemm f16");
                    cuda::sync();
                    report(&format!("{} m={m} f16", w.name), err(&d2h_f16(y16.ptr, m * w.n), &r), &mut ok);
                    // 足し込み（y0 + XW）
                    let mut gr = Rng(99);
                    let y0: Vec<f32> = (0..m * w.n).map(|_| gr.x()).collect();
                    let ya = DevBuf::from_f32(&y0);
                    cuda::check(gemm_ex(&[&w], &[ya.ptr], None, m, &xs.x16, None, 2, cfg), "gemm add");
                    cuda::sync();
                    let ra: Vec<f64> = r.iter().zip(&y0).map(|(a, b)| a + *b as f64).collect();
                    report(&format!("{} m={m} add", w.name), err(&ya.to_f32(), &ra), &mut ok);
                    // 上下（XL）: 手本は上 + 下
                    let rf = ref_gemm(&xs.xf, &wf, m, w.n, w.k);
                    cuda::check(gemm_ex(&[&w], &[y.ptr], None, m, &xs.x16, Some(&xs.lo16), 0, cfg), "gemm xl");
                    cuda::sync();
                    let e_hi = err(&y.to_f32(), &rf);
                    report(&format!("{} m={m} XL", w.name), e_hi, &mut ok);
                }
            }
        }
    }
    // SwiGLU（型の違う組・同じ組・Q3_K）
    let pairs = [("blk.3.ffn_gate.weight", "blk.3.ffn_up.weight", 778), ("blk.1.ffn_gate.weight", "blk.1.ffn_up.weight", 640),
                 ("blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", 334)];
    for (gn, un, rows) in pairs {
        if only.as_ref().map(|o| !gn.contains(o.as_str()) && !o.contains("swiglu")).unwrap_or(false) {
            continue;
        }
        let wg = repack(&g.tensor(gn), rows);
        let wu = repack(&g.tensor(un), rows);
        let (gf, uf) = (deq_f32(&wg), deq_f32(&wu));
        for &m in &ms {
            let xs = make_x(m, wg.k, 11 + m as u64);
            let rg = ref_gemm(&xs.xr, &gf, m, wg.n, wg.k);
            let ru = ref_gemm(&xs.xr, &uf, m, wu.n, wu.k);
            let r: Vec<f64> = rg.iter().zip(&ru).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect();
            for cfg in [-1, if m <= 64 { 0 } else { 1 }] {
                let y = DevBuf::new(m * wg.n * 2);
                cuda::check(gemm_ex(&[&wg, &wu], &[y.ptr, y.ptr], None, m, &xs.x16, None, 3, cfg), "swiglu");
                cuda::sync();
                report(&format!("swiglu {}+{} m={m} cfg={cfg}", wg.name, wu.name), err(&d2h_f16(y.ptr, m * wg.n), &r), &mut ok);
            }
        }
    }
    // multi（行列の境がブロックの中に来る形）: qkv 1000 行・z 334 行（ldy は偶数の決まり）・α・β
    if only.as_ref().map(|o| o.contains("multi")).unwrap_or(true) {
        let names = [("blk.0.attn_qkv.weight", 1000), ("blk.0.attn_gate.weight", 334), ("blk.0.ssm_alpha.weight", 48), ("blk.0.ssm_beta.weight", 48)];
        let ws: Vec<G3> = names.iter().map(|(n, r)| repack(&g.tensor(n), *r)).collect();
        let wfs: Vec<Vec<f32>> = ws.iter().map(deq_f32).collect();
        for &m in &ms {
            let xs = make_x(m, ws[0].k, 13 + m as u64);
            for cfg in [-1, if m <= 64 { 0 } else { 1 }] {
                // 出力は 1 つの置き場に並べる（行の間隔 = 合計の幅）: 実際の qkv・z のように別々でもよいが、ldy の確かめも兼ねる
                let tot: usize = ws.iter().map(|w| w.n).sum();
                let y = DevBuf::new(m * tot * 4);
                let mut off = 0;
                let mut ys = vec![];
                for w in &ws {
                    ys.push(unsafe { (y.ptr as *mut f32).add(off) } as *mut c_void);
                    off += w.n;
                }
                let ld = vec![tot as i32; ws.len()];
                let wr: Vec<&G3> = ws.iter().collect();
                cuda::check(gemm_ex(&wr, &ys, Some(&ld), m, &xs.x16, None, 0, cfg), "multi");
                cuda::sync();
                let yv = y.to_f32();
                let mut off = 0;
                for (w, wf) in ws.iter().zip(&wfs) {
                    let r = ref_gemm(&xs.xr, wf, m, w.n, w.k);
                    let yy: Vec<f32> = (0..m).flat_map(|t| yv[t * tot + off..t * tot + off + w.n].to_vec()).collect();
                    report(&format!("multi {} m={m} cfg={cfg}", w.name), err(&yy, &r), &mut ok);
                    off += w.n;
                }
            }
        }
    }
    // 毎回同じか（同じ入力で 3 回）
    {
        let w = repack(&g.tensor("blk.3.ffn_down.weight"), 5120);
        let m = 300;
        let xs = make_x(m, w.k, 5);
        let mut first: Option<Vec<f32>> = None;
        for _ in 0..3 {
            let y = DevBuf::new(m * w.n * 4);
            cuda::check(gemm_ex(&[&w], &[y.ptr], None, m, &xs.x16, None, 0, -1), "gemm");
            cuda::sync();
            let v = y.to_f32();
            match &first {
                None => first = Some(v),
                Some(f) => {
                    let same = f.iter().zip(&v).all(|(a, b)| a.to_bits() == b.to_bits());
                    println!("{} 毎回同じ（{}）", if same { "OK" } else { "NG" }, w.name);
                    if !same {
                        ok = false;
                    }
                }
            }
        }
    }
    println!("{}（最大の誤差 {worst:.2e}）", if ok { "全部 OK" } else { "NG あり" });
    ok
}

// ---------------------------------------------------------------- bench

/// 重みの写し（L2 に残らないよう 96MB 以上ぶん）
fn copies(b: &DevBuf) -> Vec<DevBuf> {
    let n = ((96usize << 20) / b.len.max(1) + 1).max(2);
    (0..n)
        .map(|_| {
            let d = DevBuf::new(b.len);
            cuda::check(unsafe { ffi::cudaMemcpy(d.ptr, b.ptr, b.len, 3) }, "D2D");
            d
        })
        .collect()
}
fn best_ms(mut f: impl FnMut(usize)) -> f32 {
    f(0);
    cuda::sync();
    let one = cuda::time_ms(|| f(0)).max(1e-3);
    let reps = ((20.0 / one) as usize).clamp(1, 50);
    let mut best = f32::MAX;
    for r in 0..5 {
        best = best.min(cuda::time_ms(|| (0..reps).for_each(|i| f(r * reps + i))) / reps as f32);
    }
    best
}

fn bench(a: &Args) {
    let gdef = tachyon_v100::weights_path(GGUF);
    let path = a.pos.get(1).map(|s| s.as_str()).unwrap_or(&gdef);
    let g = Model::open(path).unwrap();
    let ms = a.list("m", &[16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 2048]);
    let shape = a.s("shape").map(|s| s.to_string());
    let old = a.s("old").map(|s| s == "1").unwrap_or(true);
    let xl = a.s("xl").map(|s| s == "1").unwrap_or(false);
    // (名前, 種類, テンソル)。種類: gemm（n = 5120 は足し込み）・swiglu・multi
    let shapes: [(&str, &str, &[&str]); 9] = [
        ("gate_up", "swiglu", &["blk.3.ffn_gate.weight", "blk.3.ffn_up.weight"]),
        ("gate_up_iq4", "swiglu", &["blk.1.ffn_gate.weight", "blk.1.ffn_up.weight"]),
        ("ffn_down", "gemm", &["blk.3.ffn_down.weight"]),
        ("ffn_down_iq4", "gemm", &["blk.0.ffn_down.weight"]),
        ("ffn_down_q6", "gemm", &["blk.56.ffn_down.weight"]),
        ("lin_in", "multi", &["blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight"]),
        ("ssm_out", "gemm", &["blk.4.ssm_out.weight"]),
        ("full_in", "multi", &["blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight"]),
        ("attn_output", "gemm", &["blk.3.attn_output.weight"]),
    ];
    let mmax = *ms.iter().max().unwrap();
    println!("形 | M | 新 ms | TFLOPS | g3 GEMV ms | 前の道 ms（戻し＋F16 GEMM）");
    for (sname, kind, names) in shapes {
        if shape.as_ref().map(|s| s != sname).unwrap_or(false) {
            continue;
        }
        let ws: Vec<G3> = names.iter().map(|n| repack(&g.tensor(n), usize::MAX)).collect();
        let kk = ws[0].k;
        let ntot: usize = if kind == "swiglu" { ws[0].n * 2 } else { ws.iter().map(|w| w.n).sum() };
        let wc: Vec<Vec<DevBuf>> = ws.iter().map(|w| copies(&w.buf)).collect();
        let nc = wc[0].len();
        let xs = make_x(mmax, kk, 3);
        let yb = DevBuf::new(mmax * ntot * 4);
        let mode = match kind {
            "swiglu" => 3,
            "gemm" if ws[0].n == 5120 => 2,
            _ => 0,
        };
        // 前の道の置き場（fp16 [n][k] に戻す）
        let wdeq = if old { Some(DevBuf::new(ntot * kk * 2 + 4096)) } else { None };
        // GEMV の道の x の下ごしらえ（B ≤ 32）
        let xp = DevBuf::new(unsafe { ffi::infer_g3_xprep_bytes(kk as i32, 32) } as usize);
        let xf: Vec<f32> = xs.xr[..32.min(mmax) * kk].iter().map(|v| *v as f32).collect();
        let xfb = DevBuf::from_f32(&xf);
        cuda::check(unsafe { ffi::infer_g3_xprep(xfb.f32p(), kk as i32, 32.min(mmax) as i32, xp.ptr, null_mut()) }, "xprep");
        for &m in &ms {
            let run_new = |i: usize| {
                let ps: Vec<*const c_void> = wc.iter().map(|c| c[i % nc].ptr as *const c_void).collect();
                let ts: Vec<i32> = ws.iter().map(|w| w.t).collect();
                let ns: Vec<i32> = ws.iter().map(|w| w.n as i32).collect();
                let mut ys = vec![];
                let mut off = 0;
                for w in &ws {
                    ys.push(unsafe { (yb.ptr as *mut f32).add(off * m) } as *mut c_void);
                    if mode != 3 {
                        off += w.n;
                    }
                }
                let xlo = if xl { xs.lo16.ptr as *const c_void } else { null() };
                cuda::check(
                    unsafe {
                        k::infer_g3_gemm_ex(ws.len() as i32, ts.as_ptr(), ps.as_ptr(), ns.as_ptr(), ys.as_ptr(), null(), kk as i32, m as i32,
                                            xs.x16.ptr, xlo, kk as i32, mode, -1, null_mut())
                    },
                    "g3 gemm",
                );
            };
            let t_new = best_ms(run_new);
            let flop = 2.0 * m as f64 * ntot as f64 * kk as f64;
            let t_gemv = if m <= 32 {
                let f = |i: usize| {
                    let ps: Vec<*const c_void> = wc.iter().map(|c| c[i % nc].ptr as *const c_void).collect();
                    match kind {
                        "swiglu" => cuda::check(
                            unsafe {
                                ffi::infer_g3_swiglu(ws[0].t, ps[0], ws[1].t, ps[1], kk as i32, ws[0].n as i32, xp.ptr, yb.f32p(), null_mut(), m as i32,
                                                     null_mut())
                            },
                            "g3 swiglu",
                        ),
                        "multi" => {
                            let ts: Vec<i32> = ws.iter().map(|w| w.t).collect();
                            let ns: Vec<i32> = ws.iter().map(|w| w.n as i32).collect();
                            let mut ys = vec![];
                            let mut off = 0;
                            for w in &ws {
                                ys.push(unsafe { yb.f32p().add(off * m) });
                                off += w.n;
                            }
                            cuda::check(
                                unsafe {
                                    ffi::infer_g3_gemv_multi(ws.len() as i32, ts.as_ptr(), ps.as_ptr(), ns.as_ptr(), ys.as_ptr(), kk as i32, xp.ptr,
                                                             m as i32, 0, 0, null_mut())
                                },
                                "g3 multi",
                            )
                        }
                        _ => cuda::check(
                            unsafe {
                                ffi::infer_g3_gemv(ws[0].t, ps[0], kk as i32, ws[0].n as i32, xp.ptr, yb.f32p(), m as i32, 0, (mode == 2) as i32,
                                                   null_mut())
                            },
                            "g3 gemv",
                        ),
                    }
                };
                format!("{:.3}", best_ms(f))
            } else {
                "-".into()
            };
            let t_old = if let Some(wd) = &wdeq {
                let f = |i: usize| {
                    let mut off = 0usize;
                    let mut ps = vec![];
                    for (w, c) in ws.iter().zip(&wc) {
                        let p = unsafe { (wd.ptr as *mut u8).add(off) } as *mut c_void;
                        cuda::check(unsafe { k::infer_m_g3_deq16(w.t, c[i % nc].ptr, kk as i32, w.n as i32, p, null_mut()) }, "deq16");
                        ps.push(p as *const c_void);
                        off += (w.n * kk * 2 + 255) / 256 * 256;
                    }
                    let t16 = vec![1i32; ws.len()];
                    let ns: Vec<i32> = ws.iter().map(|w| w.n as i32).collect();
                    match kind {
                        "swiglu" => cuda::check(
                            unsafe { ffi::infer_gemm_swiglu(1, ps[0], 1, ps[1], kk as i32, ws[0].n as i32, m as i32, xs.x16.ptr, yb.ptr, -1, null_mut()) },
                            "gemm swiglu",
                        ),
                        _ => {
                            let mut ys = vec![];
                            let mut o2 = 0;
                            for w in &ws {
                                ys.push(unsafe { (yb.ptr as *mut f32).add(o2 * m) } as *mut c_void);
                                o2 += w.n;
                            }
                            cuda::check(
                                unsafe {
                                    ffi::infer_gemm_multi(ws.len() as i32, t16.as_ptr(), ps.as_ptr(), ns.as_ptr(), ys.as_ptr(), kk as i32, m as i32,
                                                          xs.x16.ptr, if mode == 2 { 2 } else { 0 }, -1, null_mut())
                                },
                                "gemm multi",
                            )
                        }
                    }
                };
                format!("{:.3}", best_ms(f))
            } else {
                "-".into()
            };
            println!("{sname} | {m} | {t_new:.3} | {:.1} | {t_gemv} | {t_old}", flop / (t_new as f64 * 1e-3) / 1e12);
        }
    }
}

fn main() {
    let a = Args::parse();
    match a.pos.first().map(|s| s.as_str()) {
        Some("check") => {
            if !check(&a) {
                std::process::exit(1);
            }
        }
        Some("bench") => bench(&a),
        _ => {
            eprintln!("使い方: pfcheck check|bench [gguf] [--m ..] [--only ..] [--shape ..]");
            std::process::exit(2);
        }
    }
}
