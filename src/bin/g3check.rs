//! g3（kernels/gemv3.cu）の確かめと速さ。重みは GGUF から名前で取る（mmap のまま GPU へ送り、GPU の上で詰め直す）。
//! 答え合わせは tools/g3check.py（gguf-py の dequantize と numpy の float64）。
//!
//!   g3check deq    <gguf> <名前> <行数> <出力.f32>                         … 詰め直して戻す（並びの確かめ）
//!   g3check gemv   <gguf> <名前> <x.f32> <B> <出力.f32> [wk] [k] [n]       … 答えと速さ（2 版と比べる。出力.old に 2 版）。
//!                  k・n を渡すと行列の左上 [n 行 × k 列] だけで（半端な形の確かめ用。x は [B][k]）。
//!                  足し込み（出力.add）は 4 回くり返して、毎回ビットまで同じかも出す
//!   g3check plan                                                            … 起動の形（k の切れ方）を全部の形で調べる（GPU は要らない）
//!   g3check swiglu <gguf> <gate の名前> <up の名前> <x.f32> <B> <出力.f32>  … SiLU(gate)*up をまとめて 1 回で
//!   g3check table  <gguf> <名前>...                                         … B = 1,2,4,8,16,32 の時間と GB/s（n = 5120 は足し込み）
//!   g3check swtable <gguf> <層>...                                          … 同じく SiLU(gate)*up（down の x の下ごしらえまで）
//!   g3check model  <gguf> [B,...]                                           … 1 トークンの GEMV を層ごとに全部（64 層＋出力）
use tachyon_v100::cuda::{self, ffi, DevBuf};
use tachyon_v100::gguf::{Model, TensorView};
use tachyon_v100::qtype::QType;
use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::OnceLock;

fn read_f32(p: &str) -> Vec<f32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn write_f32(p: &str, v: &[f32]) {
    std::fs::write(p, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}

/// 詰め直した重み
struct G3 {
    buf: DevBuf,
    t: i32, // 詰め直した後の型
    k: usize,
    n: usize,
    gguf_bytes: usize,
}

/// GGUF の行 0..n（mmap）→ GPU → 自前の並び
fn repack(tv: &TensorView, n: usize) -> G3 {
    repack_sub(tv, tv.k(), n)
}

/// 行列の左上 [n 行 × k 列]（k は 256 の倍数。どの型もブロックは 256 要素の中で閉じるので、行の頭の k 列ぶんのバイト）
fn repack_sub(tv: &TensorView, k: usize, n: usize) -> G3 {
    let t = tv.qt() as i32;
    assert!(k % 256 == 0 && k <= tv.k() && n <= tv.n_rows(), "{}: 左上 [{n} × {k}] は取れない", tv.name);
    let bytes = unsafe { ffi::infer_g3_bytes(t, k as i32, n as i32) };
    assert!(bytes > 0, "{}: g3 が扱わない型 {}", tv.name, tv.type_name());
    let rb = tv.row_bytes() * k / tv.k();
    let host: Vec<u8> = if k == tv.k() {
        tv.rows(0, n).to_vec()
    } else {
        (0..n).flat_map(|r| tv.rows(r, 1)[..rb].iter().copied()).collect()
    };
    let src = DevBuf::from_bytes(&host);
    let d = DevBuf::new(bytes as usize);
    cuda::check(unsafe { ffi::infer_g3_repack(t, src.ptr, d.ptr, k as i32, n as i32, null_mut()) }, "repack");
    cuda::sync();
    G3 { buf: d, t: unsafe { ffi::infer_g3_type(t) }, k, n, gguf_bytes: rb * n }
}

/// 決まった順の足し込みの作業場（プロセスに 1 つ。0 で埋めておく。カーネルが使い終わりに 0 に戻す）
fn det_ws() -> (*mut c_void, i64) {
    static WS: OnceLock<(usize, i64)> = OnceLock::new();
    let (p, n) = *WS.get_or_init(|| {
        let n = unsafe { ffi::infer_g3_ws_bytes() };
        let b = DevBuf::new(n as usize);
        cuda::check(unsafe { ffi::cudaMemsetAsync(b.ptr, 0, b.len, null_mut()) }, "memset");
        cuda::sync();
        let p = b.ptr as usize;
        std::mem::forget(b); // 最後まで使う
        (p, n)
    });
    (p as *mut c_void, n)
}

/// x の下ごしらえ（置き場つき）
struct XP {
    x: DevBuf,
    xp: DevBuf,
    k: usize,
    b: usize,
}
impl XP {
    fn new(x: &[f32], k: usize, b: usize) -> XP {
        assert!(x.len() >= b * k, "x が足りない");
        let xp = DevBuf::new(unsafe { ffi::infer_g3_xprep_bytes(k as i32, b as i32) } as usize);
        let s = XP { x: DevBuf::from_f32(&x[..b * k]), xp, k, b };
        s.prep();
        s
    }
    fn prep(&self) {
        cuda::check(unsafe { ffi::infer_g3_xprep(self.x.f32p(), self.k as i32, self.b as i32, self.xp.ptr, null_mut()) }, "xprep");
    }
}

fn gemv(w: &G3, ptr: *const c_void, x: &XP, y: &DevBuf, wk: i32, add: bool) {
    let (ws, wsb) = det_ws();
    cuda::check(
        unsafe { ffi::infer_g3_gemv_ws(w.t, ptr, w.k as i32, w.n as i32, x.xp.ptr, y.f32p(), x.b as i32, wk, add as i32, ws, wsb, null_mut()) },
        "g3 gemv",
    );
}
fn zero(y: &DevBuf) {
    cuda::check(unsafe { ffi::cudaMemsetAsync(y.ptr, 0, y.len, null_mut()) }, "memset");
}
/// 残差に足し込む行列（出力が隠れ状態の幅）は add で測る
fn adds(w: &G3) -> bool {
    w.n == 5120
}

/// 下ごしらえ（hi + lo）× 倍率の逆数 → f32（確かめ用）
fn unprep(xp: &DevBuf, k: usize, b: usize) -> Vec<f32> {
    let mut raw = vec![0u8; xp.len];
    cuda::check(unsafe { ffi::cudaMemcpy(raw.as_mut_ptr().cast(), xp.ptr, xp.len, 2) }, "cudaMemcpy D2H");
    let h16 = |i: usize| -> f32 {
        let v = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
        // f16 → f32
        let (s, e, m) = ((v >> 15) as u32, ((v >> 10) & 0x1f) as i32, (v & 0x3ff) as u32);
        let f = if e == 0 { m as f32 * (2f32).powi(-24) } else if e == 31 { f32::INFINITY } else { (1.0 + m as f32 / 1024.0) * (2f32).powi(e - 15) };
        if s == 1 { -f } else { f }
    };
    let si = b * k * 4;
    let mut out = vec![0f32; b * k];
    for t in 0..b {
        for i in 0..k {
            let c = t * (k / 32) + i / 32;
            let isc = f32::from_le_bytes(raw[si + c * 8 + 4..si + c * 8 + 8].try_into().unwrap());
            out[t * k + i] = (h16(t * 2 * k + i) + h16(t * 2 * k + k + i)) * isc;
        }
    }
    out
}

/// L2（6MB）に残らないよう、合わせて 96MB 以上になるまで写しを作る
fn copies(b: &DevBuf) -> Vec<DevBuf> {
    let n = (96usize << 20).div_ceil(b.len.max(1)).clamp(1, 64);
    (1..n)
        .map(|_| {
            let d = DevBuf::new(b.len);
            cuda::check(unsafe { ffi::cudaMemcpy(d.ptr, b.ptr, b.len, 3) }, "cudaMemcpy D2D");
            d
        })
        .collect()
}

/// f(i) を何度も回した 1 回あたりの時間（ms）。GPU を他の人と分け合っていても揺れにくいよう、5 回測っていちばん速いもの
fn bench(mut f: impl FnMut(usize)) -> f32 {
    f(0);
    cuda::sync();
    let one = cuda::time_ms(|| f(0)).max(1e-3);
    let reps = ((10.0 / one) as usize).clamp(10, 1000);
    let mut best = f32::MAX;
    for r in 0..5 {
        best = best.min(cuda::time_ms(|| (0..reps).for_each(|i| f(r * reps + i))) / reps as f32);
    }
    best
}

fn gbs(bytes: usize, ms: f32) -> f64 {
    bytes as f64 / (ms as f64 * 1e-3) / 1e9
}

/// 詰め直した重みを L2 に残らない数だけ並べて、i 回目に使うものを返す
fn rotating(w: &G3) -> (Vec<DevBuf>, impl Fn(&[DevBuf], usize) -> *const c_void + '_) {
    let c = copies(&w.buf);
    let p0 = w.buf.ptr as *const c_void;
    (c, move |c: &[DevBuf], i: usize| if i % (c.len() + 1) == 0 { p0 } else { c[i % (c.len() + 1) - 1].ptr as *const c_void })
}

fn time_gemv(w: &G3, b: usize, wk: i32, add: bool) -> f32 {
    let mut rng = 12345u64;
    let x: Vec<f32> = (0..b * w.k)
        .map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect();
    let xp = XP::new(&x, w.k, b);
    let y = DevBuf::new(b * w.n * 4);
    let (c, pick) = rotating(w);
    bench(|i| gemv(w, pick(&c, i), &xp, &y, wk, add))
}

/// 起動の形（kernels/gemv3.cu の plan_g3）を、カーネルの k の切り方（KSplit）と同じ計算で確かめる。
/// どのブロック・どの切れにも wk 単位以上（warp ごとに 1 単位以上）、切れは cmax 以下、共有メモリは 96KB 以下、
/// k をブロックで分けるのは足し込みだけ・80 ブロックまで。順を決める・決めない × 作業場あり・なし で全部
fn plan_check() -> bool {
    let wsb = unsafe { ffi::infer_g3_ws_bytes() };
    let mut tiles: Vec<usize> = (1..=100).collect();
    tiles.extend([128, 160, 192, 320, 384, 544, 7760]);
    let (mut n_ok, mut n_bad, mut n_skip) = (0usize, 0usize, 0usize);
    let mut lowered = 0usize;
    for det in [1, 0] {
        unsafe { ffi::infer_g3_set_det(det) };
        for &ws in &[wsb, 0] {
            for nu in 1..=80usize {
                for &nt in &tiles {
                    for b in 1..=32usize {
                        for wk in [0, 1, 2, 4, 8] {
                            for add in [0, 1] {
                                let mut p = [0i32; 7];
                                let r = unsafe { ffi::infer_g3_plan((nu * 256) as i32, (nt * 32) as i32, b as i32, wk, add, ws, p.as_mut_ptr()) };
                                if r != 0 {
                                    // 渡した wk が使えない（nu より大きい・8 本ずつの組と合わせて 8 warp を超える）ときだけ -4
                                    let tg = if b <= 16 { 1 } else { 2 };
                                    if r == -4 && wk != 0 && (wk as usize > nu || wk as usize * tg > 8) {
                                        n_skip += 1;
                                        continue;
                                    }
                                    n_bad += 1;
                                    if n_bad <= 20 {
                                        eprintln!("NG 形が無い: nu={nu} tiles={nt} B={b} wk={wk} add={add} det={det} ws={ws} → {r}");
                                    }
                                    continue;
                                }
                                let [pwk, ksp, ns, cmax, gx, smem, pdet] = p.map(|v| v as usize);
                                let mut bad = vec![];
                                if ![1, 2, 4, 8].contains(&pwk) || (wk > 0 && pwk > wk as usize) {
                                    bad.push(format!("wk {pwk}"));
                                }
                                if wk > 0 && pwk < wk as usize {
                                    lowered += 1;
                                }
                                if ksp > 1 && (add == 0 || gx * ksp > 80) {
                                    bad.push(format!("ksp {ksp} gx {gx}"));
                                }
                                if pdet != 0 && (det == 0 || ws == 0 || ksp == 1) || (det == 1 && add == 1 && ksp > 1 && pdet == 0) {
                                    bad.push(format!("det {pdet}"));
                                }
                                if smem > 96 * 1024 - 64 {
                                    bad.push(format!("smem {smem}"));
                                }
                                for y in 0..ksp {
                                    let (ku0, ku1) = (y * nu / ksp, (y + 1) * nu / ksp);
                                    let nb = ku1 - ku0;
                                    for s in 0..ns {
                                        let (a0, a1) = (ku0 + s * nb / ns, ku0 + (s + 1) * nb / ns);
                                        let len = a1 - a0;
                                        if len > cmax {
                                            bad.push(format!("切れ {y}/{s} が {len} > cmax {cmax}"));
                                        }
                                        for kw in 0..pwk {
                                            if kw * len / pwk == (kw + 1) * len / pwk {
                                                bad.push(format!("ブロック {y} 切れ {s}（{len} 単位）の warp {kw} が空"));
                                                break;
                                            }
                                        }
                                    }
                                }
                                if bad.is_empty() {
                                    n_ok += 1;
                                } else {
                                    n_bad += 1;
                                    if n_bad <= 20 {
                                        eprintln!("NG nu={nu} tiles={nt} B={b} wk={wk} add={add} det={det} ws={ws}: {:?} → {}", p, bad.join("・"));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    unsafe { ffi::infer_g3_set_det(1) };
    println!("起動の形: {n_ok} 組 OK・NG {n_bad} 組（使えない wk {n_skip} 組、収まらず wk を下げた {lowered} 組）");
    n_bad == 0
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.get(1).map(|s| s.as_str()) == Some("plan") {
        std::process::exit(if plan_check() { 0 } else { 1 });
    }
    let dev: i32 = std::env::var("INFER_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    cuda::check(unsafe { ffi::cudaSetDevice(dev) }, "cudaSetDevice");
    let m = Model::open(&a[2]).unwrap_or_else(|e| panic!("{e}"));
    match a[1].as_str() {
        "deq" => {
            let tv = m.tensor(&a[3]);
            let rows: usize = a[4].parse().unwrap();
            let w = repack(&tv, rows);
            let y = DevBuf::new(w.k * rows * 4);
            cuda::check(unsafe { ffi::infer_g3_dequant(w.t, w.buf.ptr, w.k as i32, rows as i32, y.f32p(), null_mut()) }, "g3 dequant");
            cuda::sync();
            write_f32(&a[5], &y.to_f32());
        }
        "gemv" => {
            let tv = m.tensor(&a[3]);
            let b: usize = a[5].parse().unwrap();
            let wk: i32 = a.get(7).map(|s| s.parse().unwrap()).unwrap_or(0);
            let k: usize = a.get(8).map(|s| s.parse().unwrap()).unwrap_or(tv.k());
            let n: usize = a.get(9).map(|s| s.parse().unwrap()).unwrap_or(tv.n_rows());
            let w = repack_sub(&tv, k, n);
            let xp = XP::new(&read_f32(&a[4]), w.k, b);
            let y = DevBuf::new(b * w.n * 4);
            gemv(&w, w.buf.ptr, &xp, &y, wk, false);
            cuda::sync();
            write_f32(&a[6], &y.to_f32());
            // 足し込み（k をブロックで分ける）も答えを出す（0 に足す）
            zero(&y);
            gemv(&w, w.buf.ptr, &xp, &y, wk, true);
            cuda::sync();
            write_f32(&format!("{}.add", a[6]), &y.to_f32());
            // 0 でない y（残差のつもり）に 4 回足し直して、毎回ビットまで同じか（0 からだと切れ 2 つは足す順によらず同じになる）
            let y0: Vec<f32> = (0..b * w.n).map(|i| ((i * 7919 % 1999) as f32 - 999.0) * 1e-3).collect();
            let mut outs = vec![];
            for _ in 0..4 {
                cuda::check(unsafe { ffi::cudaMemcpy(y.ptr, y0.as_ptr().cast(), y.len, 1) }, "cudaMemcpy H2D");
                gemv(&w, w.buf.ptr, &xp, &y, wk, true);
                cuda::sync();
                outs.push(y.to_bytes());
            }
            let same = outs.iter().all(|v| *v == outs[0]);
            let mut plan = [0i32; 7];
            let (_, wsb) = det_ws();
            cuda::check(unsafe { ffi::infer_g3_plan(w.k as i32, w.n as i32, b as i32, wk, 1, wsb, plan.as_mut_ptr()) }, "plan");
            let det = if plan[6] != 0 { "決まった順" } else if plan[1] > 1 { "atomicAdd" } else { "分けない" };
            let plan_s = format!("足し込み wk={} ksp={} 切れ={}（{det}）、4 回{}", plan[0], plan[1], plan[2], if same { "同じ" } else { "違う" });
            if std::env::var_os("G3CHECK_NOBENCH").is_some() {
                // 答えだけ（半端な形の確かめ用。速さは測らない）
                println!("{} {} k={} n={} B={b} wk={wk}: {plan_s}", tv.name, tv.type_name(), w.k, w.n);
                return;
            }
            let (c, pick) = rotating(&w);
            let ms = bench(|i| gemv(&w, pick(&c, i), &xp, &y, wk, false));
            let ms_add = bench(|i| gemv(&w, pick(&c, i), &xp, &y, wk, true));
            let ms_prep = bench(|_| xp.prep());
            drop(c);
            // 2 版（kernels/gemv.cu）。扱う型で B ≤ 8 のときだけ
            let qt = tv.qt();
            let old = if qt != QType::Q3K && qt != QType::IQ3S && b <= 8 {
                let dw = DevBuf::from_bytes(tv.bytes);
                let yo = DevBuf::new(b * w.n * 4);
                let co = copies(&dw);
                let run = |i: usize| {
                    let p = if i % (co.len() + 1) == 0 { dw.ptr } else { co[i % (co.len() + 1) - 1].ptr };
                    cuda::check(unsafe { ffi::infer_gemv(qt as i32, p, w.k as i32, w.n as i32, xp.x.f32p(), yo.f32p(), b as i32, null_mut()) }, "gemv")
                };
                run(0);
                cuda::sync();
                write_f32(&format!("{}.old", a[6]), &yo.to_f32());
                let ms_old = bench(run);
                format!("| 2 版 {:.4} ms {:.0} GB/s | {:.2} 倍", ms_old, gbs(w.gguf_bytes, ms_old), ms_old / ms)
            } else {
                String::new()
            };
            println!(
                "{} {} k={} n={} B={b}: g3 {:.4} ms {:.0} GB/s・足し込み {:.4} ms {:.0} GB/s（x の下ごしらえ {:.4} ms）{old} | {plan_s}",
                tv.name,
                tv.type_name(),
                w.k,
                w.n,
                ms,
                gbs(w.gguf_bytes, ms),
                ms_add,
                gbs(w.gguf_bytes, ms_add),
                ms_prep
            );
        }
        "swiglu" => {
            let (tg, tu) = (m.tensor(&a[3]), m.tensor(&a[4]));
            let b: usize = a[6].parse().unwrap();
            let (wg, wu) = (repack(&tg, tg.n_rows()), repack(&tu, tu.n_rows()));
            assert_eq!((wg.k, wg.n), (wu.k, wu.n), "gate と up の形が違う");
            let xp = XP::new(&read_f32(&a[5]), wg.k, b);
            let h = DevBuf::new(b * wg.n * 4);
            // down に渡す x の下ごしらえ（hp）も書く
            let hp = DevBuf::new(unsafe { ffi::infer_g3_xprep_bytes(wg.n as i32, b as i32) } as usize);
            let run = |pg: *const c_void, pu: *const c_void, hf: *mut f32, hq: *mut c_void| {
                cuda::check(
                    unsafe { ffi::infer_g3_swiglu(wg.t, pg, wu.t, pu, wg.k as i32, wg.n as i32, xp.xp.ptr, hf, hq, b as i32, null_mut()) },
                    "swiglu",
                )
            };
            run(wg.buf.ptr, wu.buf.ptr, h.f32p(), hp.ptr);
            cuda::sync();
            write_f32(&a[7], &h.to_f32());
            write_f32(&format!("{}.hp", a[7]), &unprep(&hp, wg.n, b));
            let (cg, pg) = rotating(&wg);
            let (cu, pu) = rotating(&wu);
            let ms = bench(|i| run(pg(&cg, i), pu(&cu, i), h.f32p(), null_mut()));
            let ms_hp = bench(|i| run(pg(&cg, i), pu(&cu, i), null_mut(), hp.ptr));
            let y = DevBuf::new(b * wg.n * 4);
            let ms2 = bench(|i| {
                gemv(&wg, pg(&cg, i), &xp, &y, 0, false);
                gemv(&wu, pu(&cu, i), &xp, &y, 0, false);
            });
            let bytes = wg.gguf_bytes + wu.gguf_bytes;
            println!(
                "swiglu {}+{} k={} n={} B={b}: まとめて {:.4} ms {:.0} GB/s（down の下ごしらえまで {:.4} ms {:.0} GB/s）| 別々の 2 回 {:.4} ms {:.0} GB/s",
                tg.type_name(),
                tu.type_name(),
                wg.k,
                wg.n,
                ms,
                gbs(bytes, ms),
                ms_hp,
                gbs(bytes, ms_hp),
                ms2,
                gbs(bytes, ms2)
            );
        }
        "table" => {
            for name in &a[3..] {
                let tv = m.tensor(name);
                let w = repack(&tv, tv.n_rows());
                let mut line = format!("{}\t{}\t{}\t{}{}", name, tv.type_name(), w.k, w.n, if adds(&w) { "（足し込み）" } else { "" });
                for b in [1usize, 2, 4, 8, 16, 32] {
                    let ms = time_gemv(&w, b, 0, adds(&w));
                    line += &format!("\t{:.4}ms {:.0}GB/s", ms, gbs(w.gguf_bytes, ms));
                }
                println!("{line}");
            }
        }
        "multi" => {
            // g3check multi <gguf> <x.f32> <B> <出力> <名前>... … 同じ x の行列をまとめて 1 回で（出力.0, 出力.1, ...）
            let b: usize = a[4].parse().unwrap();
            let ws: Vec<G3> = a[6..].iter().map(|n| { let t = m.tensor(n); repack(&t, t.n_rows()) }).collect();
            let k = ws[0].k;
            assert!(ws.iter().all(|w| w.k == k), "k が違う");
            let xp = XP::new(&read_f32(&a[3]), k, b);
            let ys: Vec<DevBuf> = ws.iter().map(|w| DevBuf::new(b * w.n * 4)).collect();
            let ts: Vec<i32> = ws.iter().map(|w| w.t).collect();
            let ps: Vec<*const c_void> = ws.iter().map(|w| w.buf.ptr as *const c_void).collect();
            let ns: Vec<i32> = ws.iter().map(|w| w.n as i32).collect();
            let yp: Vec<*mut f32> = ys.iter().map(|y| y.f32p()).collect();
            let run = || {
                cuda::check(
                    unsafe { ffi::infer_g3_gemv_multi(ws.len() as i32, ts.as_ptr(), ps.as_ptr(), ns.as_ptr(), yp.as_ptr(), k as i32, xp.xp.ptr, b as i32, 0, 0, null_mut()) },
                    "multi",
                )
            };
            run();
            cuda::sync();
            for (i, y) in ys.iter().enumerate() {
                write_f32(&format!("{}.{i}", a[5]), &y.to_f32());
            }
            let ms = bench(|_| run());
            let bytes: usize = ws.iter().map(|w| w.gguf_bytes).sum();
            println!("multi {} 個 k={k} B={b}: {:.4} ms {:.0} GB/s", ws.len(), ms, gbs(bytes, ms));
        }
        "swtable" => {
            // 層 i の SiLU(gate)*up（down の x の下ごしらえまで）を B = 1,2,4,8,16,32 で
            for l in &a[3..] {
                let (tg, tu) = (m.tensor(&format!("blk.{l}.ffn_gate.weight")), m.tensor(&format!("blk.{l}.ffn_up.weight")));
                let (wg, wu) = (repack(&tg, tg.n_rows()), repack(&tu, tu.n_rows()));
                let (cg, pg) = rotating(&wg);
                let (cu, pu) = rotating(&wu);
                let bytes = wg.gguf_bytes + wu.gguf_bytes;
                let mut line = format!("swiglu 層{l}\t{}+{}\t{}\t{}", tg.type_name(), tu.type_name(), wg.k, wg.n);
                for b in [1usize, 2, 4, 8, 16, 32] {
                    let x: Vec<f32> = (0..b * wg.k).map(|i| ((i * 7919 % 1000) as f32) / 1000.0 - 0.5).collect();
                    let xp = XP::new(&x, wg.k, b);
                    let hp = DevBuf::new(unsafe { ffi::infer_g3_xprep_bytes(wg.n as i32, b as i32) } as usize);
                    let ms = bench(|i| {
                        cuda::check(
                            unsafe {
                                ffi::infer_g3_swiglu(wg.t, pg(&cg, i), wu.t, pu(&cu, i), wg.k as i32, wg.n as i32, xp.xp.ptr, null_mut(), hp.ptr, b as i32, null_mut())
                            },
                            "swiglu",
                        )
                    });
                    line += &format!("\t{:.4}ms {:.0}GB/s", ms, gbs(bytes, ms));
                }
                println!("{line}");
            }
        }
        "scan" => {
            // scan <gguf> <名前> <B> <n,n,…>: 左上 n 行だけで、足し込みでない GEMV の速さ（ブロックの数の効き）
            let tv = m.tensor(&a[3]);
            let b: usize = a[4].parse().unwrap();
            for n in a[5].split(',').map(|v| v.parse::<usize>().unwrap()) {
                let w = repack_sub(&tv, tv.k(), n);
                let x: Vec<f32> = (0..b * w.k).map(|i| ((i * 7919 % 1000) as f32) / 1000.0 - 0.5).collect();
                let xp = XP::new(&x, w.k, b);
                let y = DevBuf::new(b * w.n * 4);
                let mut p = [0i32; 7];
                cuda::check(unsafe { ffi::infer_g3_plan(w.k as i32, w.n as i32, b as i32, 0, 0, 0, p.as_mut_ptr()) }, "plan");
                let (c, pick) = rotating(&w);
                let ms = bench(|i| gemv(&w, pick(&c, i), &xp, &y, 0, false));
                drop(c);
                println!("{} {} n={n} B={b}: gx {}・wk {} {:.4}ms {:.0}GB/s", tv.name, tv.type_name(), p[4], p[0], ms, gbs(w.gguf_bytes, ms));
                // -DG3_TIMING の版なら、1 回の起動のブロックごとの時刻
                for add in [false, true] {
                    // 本番と同じく x は直前に書いた（L2 にある）ものを読む
                    xp.prep();
                    gemv(&w, w.buf.ptr, &xp, &y, 0, add);
                    cuda::sync();
                    let mut pa = [0i32; 7];
                    let (_, wsb) = det_ws();
                    cuda::check(unsafe { ffi::infer_g3_plan(w.k as i32, w.n as i32, b as i32, 0, add as i32, if add { wsb } else { 0 }, pa.as_mut_ptr()) }, "plan");
                    let nbk = (pa[4] * pa[1]) as usize;
                    let mut tm = vec![0u64; nbk * 3];
                    if unsafe { g3_timing(tm.as_mut_ptr(), nbk as i32) } != 0 {
                        break;
                    }
                    let t0 = (0..nbk).map(|i| tm[3 * i]).min().unwrap();
                    let f = |v: u64| (v - t0) as f64 / 1e3;
                    let starts: Vec<f64> = (0..nbk).map(|i| f(tm[3 * i])).collect();
                    let staged: Vec<f64> = (0..nbk).map(|i| f(tm[3 * i + 1]) - f(tm[3 * i])).collect();
                    let ends: Vec<f64> = (0..nbk).map(|i| f(tm[3 * i + 2])).collect();
                    let mx = |v: &[f64]| v.iter().cloned().fold(f64::MIN, f64::max);
                    let mn = |v: &[f64]| v.iter().cloned().fold(f64::MAX, f64::min);
                    let av = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
                    println!(
                        "   {} ブロック {}（gx {}×ksp {}）: 入り 0〜{:.1}µs・x を置くまで 平均 {:.1}µs（最大 {:.1}）・計算の終わり {:.1}〜{:.1}µs（平均 {:.1}）",
                        if add { "足し込み" } else { "上書き" }, nbk, pa[4], pa[1], mx(&starts), av(&staged), mx(&staged), mn(&ends), mx(&ends), av(&ends)
                    );
                }
            }
        }
        "parts" => parts(&m, a[3..].iter().map(|v| v.parse().unwrap()).collect()),
        "model" => model(&m, a.get(3).map(|s| s.split(',').map(|v| v.parse().unwrap()).collect()).unwrap_or(vec![1, 2, 4, 8, 16, 32])),
        c => panic!("知らない動作 {c}"),
    }
}

/// 1 トークン（B 本）の GEMV を層ごとに全部回す。層ごとに GPU へ載せて詰め直し、測って捨てる（GPU は 1 層ぶん＋出力だけ使う）
/// 線形の層: x 下ごしらえ → qkv・gate・alpha・beta（1 回）→ x → ssm_out（残差に足し込み）→ x → SiLU(gate)*up → x → down（足し込み）
/// 普通の層: x → q・k・v（1 回）→ x → attn_output → x → SiLU(gate)*up（down の x の下ごしらえまで）→ down。最後に x → output
/// （x の下ごしらえ 3 回は、本番では RMSNorm・GDN/Attention の出力の側でまとめて書ける）
fn model(m: &Model, bs: Vec<usize>) {
    let n_layer = m.hp_u32("block_count").unwrap_or(65) as usize - 1; // 最後の 1 つは MTP
    let k = 5120usize;
    let xmax = 17408usize;
    let bmax = *bs.iter().max().unwrap();
    let mut rng = 99u64;
    let xs: Vec<f32> = (0..bmax * xmax)
        .map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect();
    let out_t = m.tensor("output.weight");
    let out_w = repack(&out_t, out_t.n_rows());
    let mut tot = vec![0f64; bs.len()];
    let mut bytes_tot = 0usize;
    let mut rows = vec![];
    for l in 0..n_layer {
        let ly = m.layer(l);
        let has = |s: &str| ly.tensors().any(|t| t.name.ends_with(s));
        let linear = has("ssm_out.weight");
        let attn: Vec<&str> = if linear {
            vec!["attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight", "ssm_beta.weight"]
        } else {
            vec!["attn_q.weight", "attn_k.weight", "attn_v.weight"]
        };
        let proj = if linear { "ssm_out.weight" } else { "attn_output.weight" };
        let wa: Vec<G3> = attn.iter().map(|s| { let t = ly.tensor(s); repack(&t, t.n_rows()) }).collect();
        let wo = { let t = ly.tensor(proj); repack(&t, t.n_rows()) };
        let (tg, tu, td) = (ly.tensor("ffn_gate.weight"), ly.tensor("ffn_up.weight"), ly.tensor("ffn_down.weight"));
        let (wg, wu, wd) = (repack(&tg, tg.n_rows()), repack(&tu, tu.n_rows()), repack(&td, td.n_rows()));
        let lbytes: usize = wa.iter().map(|w| w.gguf_bytes).sum::<usize>() + wo.gguf_bytes + wg.gguf_bytes + wu.gguf_bytes + wd.gguf_bytes;
        bytes_tot += lbytes;
        let mut line = format!("{l:2} {} {:6.1}MB", if linear { "線形" } else { "普通" }, lbytes as f64 / 1e6);
        for (bi, &b) in bs.iter().enumerate() {
            let x_in = XP::new(&xs, k, b);
            let x_o = XP::new(&xs, wo.k, b);
            let x_d = XP::new(&xs, wd.k, b);
            let ya: Vec<DevBuf> = wa.iter().map(|w| DevBuf::new(b * w.n * 4)).collect();
            let yo = DevBuf::new(b * k * 4);
            let yd = DevBuf::new(b * k * 4);
            let ts: Vec<i32> = wa.iter().map(|w| w.t).collect();
            let ps: Vec<*const c_void> = wa.iter().map(|w| w.buf.ptr as *const c_void).collect();
            let ns: Vec<i32> = wa.iter().map(|w| w.n as i32).collect();
            let yp: Vec<*mut f32> = ya.iter().map(|y| y.f32p()).collect();
            let step = || unsafe {
                x_in.prep();
                cuda::check(
                    ffi::infer_g3_gemv_multi(ts.len() as i32, ts.as_ptr(), ps.as_ptr(), ns.as_ptr(), yp.as_ptr(), k as i32, x_in.xp.ptr, b as i32, 0, 0, null_mut()),
                    "multi",
                );
                x_o.prep();
                gemv(&wo, wo.buf.ptr, &x_o, &yo, 0, true);
                x_in.prep();
                cuda::check(
                    ffi::infer_g3_swiglu(wg.t, wg.buf.ptr, wu.t, wu.buf.ptr, k as i32, wg.n as i32, x_in.xp.ptr, null_mut(), x_d.xp.ptr, b as i32, null_mut()),
                    "swiglu",
                );
                gemv(&wd, wd.buf.ptr, &x_d, &yd, 0, true);
            };
            let ms = bench(|_| step()) as f64;
            tot[bi] += ms;
            line += &format!(" B={b} {:.3}ms", ms);
        }
        println!("{line}");
        rows.push(line);
    }
    // 出力
    let mut line = format!("出力 {:.1}MB", out_w.gguf_bytes as f64 / 1e6);
    let mut out_ms = vec![];
    for &b in &bs {
        let x = XP::new(&xs, k, b);
        let y = DevBuf::new(b * out_w.n * 4);
        let ms = bench(|_| {
            x.prep();
            gemv(&out_w, out_w.buf.ptr, &x, &y, 0, false)
        }) as f64;
        out_ms.push(ms);
        line += &format!(" B={b} {:.3}ms", ms);
    }
    println!("{line}");
    bytes_tot += out_w.gguf_bytes;
    println!("合計（{n_layer} 層＋出力、{:.2} GB）:", bytes_tot as f64 / 1e9);
    for (bi, &b) in bs.iter().enumerate() {
        let t = tot[bi] + out_ms[bi];
        println!(
            "  B={b:2}: {:.2} ms/歩（GEMV だけ）→ 1 人 {:.1} tok/s・合計 {:.0} tok/s、実効 {:.0} GB/s",
            t,
            1e3 / t,
            1e3 / t * b as f64,
            bytes_tot as f64 / (t * 1e-3) / 1e9
        );
    }
}

extern "C" {
    #[link_name = "infer_g3_timing"]
    fn g3_timing(out: *mut u64, n: i32) -> i32;
}

/// 層 l の 4 回の起動（qkv など・出力の射影・swiglu・down）を 1 つずつ測る（B = 1・4）。ブロックの数は 8 タイル = 1 ブロックの見積もり
fn parts(m: &Model, ls: Vec<usize>) {
    let k = 5120usize;
    let xmax = 17408usize;
    let mut rng = 99u64;
    let xs: Vec<f32> = (0..4 * xmax)
        .map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect();
    for l in ls {
        let ly = m.layer(l);
        let linear = ly.tensors().any(|t| t.name.ends_with("ssm_out.weight"));
        let attn: Vec<&str> = if linear {
            vec!["attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight", "ssm_beta.weight"]
        } else {
            vec!["attn_q.weight", "attn_k.weight", "attn_v.weight"]
        };
        let proj = if linear { "ssm_out.weight" } else { "attn_output.weight" };
        let wa: Vec<G3> = attn.iter().map(|s| { let t = ly.tensor(s); repack(&t, t.n_rows()) }).collect();
        let wo = { let t = ly.tensor(proj); repack(&t, t.n_rows()) };
        let (tg, tu, td) = (ly.tensor("ffn_gate.weight"), ly.tensor("ffn_up.weight"), ly.tensor("ffn_down.weight"));
        let (wg, wu, wd) = (repack(&tg, tg.n_rows()), repack(&tu, tu.n_rows()), repack(&td, td.n_rows()));
        let tn = |s: &str| ly.tensor(s).type_name().to_string();
        let tys = format!("[{}|{}|{}+{}|{}]", attn.iter().map(|s| tn(s)).collect::<Vec<_>>().join(","), tn(proj), tn("ffn_gate.weight"), tn("ffn_up.weight"), tn("ffn_down.weight"));
        for b in [4usize] {
            let x_in = XP::new(&xs, k, b);
            let x_o = XP::new(&xs, wo.k, b);
            let x_d = XP::new(&xs, wd.k, b);
            let ya: Vec<DevBuf> = wa.iter().map(|w| DevBuf::new(b * w.n * 4)).collect();
            let yo = DevBuf::new(b * k * 4);
            let yd = DevBuf::new(b * k * 4);
            let ts: Vec<i32> = wa.iter().map(|w| w.t).collect();
            let ps: Vec<*const c_void> = wa.iter().map(|w| w.buf.ptr as *const c_void).collect();
            let ns: Vec<i32> = wa.iter().map(|w| w.n as i32).collect();
            let yp: Vec<*mut f32> = ya.iter().map(|y| y.f32p()).collect();
            let plan = |w: &G3| -> String {
                let mut p = [0i32; 7];
                let (ws, wsb) = det_ws();
                let _ = ws;
                let r = unsafe { ffi::infer_g3_plan(w.k as i32, w.n as i32, b as i32, 0, 1, wsb, p.as_mut_ptr()) };
                if r == 0 { format!("gx {}×ksp {}・wk {}", p[4], p[1], p[0]) } else { format!("plan {r}") }
            };
            let t_a = bench(|_| unsafe {
                cuda::check(ffi::infer_g3_gemv_multi(ts.len() as i32, ts.as_ptr(), ps.as_ptr(), ns.as_ptr(), yp.as_ptr(), k as i32, x_in.xp.ptr, b as i32, 0, 0, null_mut()), "multi")
            });
            let t_o = bench(|_| gemv(&wo, wo.buf.ptr, &x_o, &yo, 0, true));
            let t_s = bench(|_| unsafe {
                cuda::check(ffi::infer_g3_swiglu(wg.t, wg.buf.ptr, wu.t, wu.buf.ptr, k as i32, wg.n as i32, x_in.xp.ptr, null_mut(), x_d.xp.ptr, b as i32, null_mut()), "swiglu")
            });
            let t_d = bench(|_| gemv(&wd, wd.buf.ptr, &x_d, &yd, 0, true));
            let ba: usize = wa.iter().map(|w| w.gguf_bytes).sum();
            let nta: usize = wa.iter().map(|w| (w.n + 31) / 32).sum();
            let nts = (wg.n + 31) / 32;
            print!("{tys} ");
            println!(
                "層{l:2} {} B={b}: qkv {:.1}MB {:.3}ms {:.0}GB/s（{} タイル≈{} ブロック）| 射影 {}×{} {:.1}MB {:.3}ms {:.0}GB/s（{}）| swiglu {:.1}MB {:.3}ms {:.0}GB/s（{} タイル≈{} ブロック）| down {}×{} {:.1}MB {:.3}ms {:.0}GB/s（{}）",
                if linear { "線形" } else { "普通" },
                ba as f64 / 1e6, t_a, gbs(ba, t_a), nta, (nta + 7) / 8,
                wo.k, wo.n, wo.gguf_bytes as f64 / 1e6, t_o, gbs(wo.gguf_bytes, t_o), plan(&wo),
                (wg.gguf_bytes + wu.gguf_bytes) as f64 / 1e6, t_s, gbs(wg.gguf_bytes + wu.gguf_bytes, t_s), nts, (nts + 7) / 8,
                wd.k, wd.n, wd.gguf_bytes as f64 / 1e6, t_d, gbs(wd.gguf_bytes, t_d), plan(&wd)
            );
        }
    }
}
