//! カーネルの確かめ: テンソル（生のバイト）を GPU で戻す／掛けて、結果を f32 のファイルに書く。
//! 答え合わせは tools/kcheck.py（gguf-py の dequantize と numpy）。
//!
//!   kcheck dequant <bin> <型> <k> <行数> <出力>
//!   kcheck gemv    <bin> <型> <k> <n> <x.f32（B×k）> <B> <出力>      … 速さも測る
//!   kcheck moe     <bin> <型> <k> <n> <ids.i32> <x_div> <x.f32> <出力>
//!   kcheck gemm    <w> <型> <k> <n> <m> <x.f16（m×k）> <出力|-> [回数]   … プレフィルの GEMM（出力は f32 のファイル）
//!     <w> は「ファイル」か「ファイル@位置」。出力が - なら書かない（速さだけ）。回数を付けると速さも測る。
//!     環境変数 INFER_GEMM_CFG でタイルの組（既定 -1 = 自動）、INFER_GEMM_SPLITS で K の分け数（既定 0 = 自動）、
//!     INFER_GEMM_Y=f16 で Y を fp16 で受ける、INFER_GEMM_Y=add:<y0.f32> で y0 に足す（残差）
//!
//! GGUF から名前でじかに取る（取り出した .bin は要らない。型・k・n は目次から）:
//!   kcheck gguf <gguf>                                                  … テンソルの一覧（名前 型 形、タブ区切り）
//!   kcheck gguf <gguf> <名前> raw     <行の始め> <行数> <出力.bin>        … 生のバイト（手本を作る用。行数 all で最後まで）
//!   kcheck gguf <gguf> <名前> dequant <行の始め> <行数> <出力.f32>
//!   kcheck gguf <gguf> <名前> gemv    <行の始め> <行数> <x.f32> <B> <出力.f32>   … 速さも測る
//!   kcheck gguf <gguf> <名前> moe     <ids.i32> <x_div> <x.f32> <出力.f32>      … 3 次元の専門家 [k, n, 専門家]
//!   kcheck gguf <gguf> <名前> gemm    <m> <x.f16> <出力|-> [回数]          … 行列全部（環境変数は上の gemm と同じ）
//!   kcheck gguf <gguf> <gate> swiglu <up> <m> <x.f16> <出力|-> [回数]      … silu(X·gate^T) * (X·up^T)（fp16 で受けて f32 で書く）
//!   kcheck gguf <gguf> <名,名,…> multi <m> <x.f16> <出力|-> [回数]         … 同じ X で何本か（出力は行列の順に並べる）
use tachyon_v100::cuda::{self, ffi, DevBuf};
use tachyon_v100::gguf::{ggml_type_name, Model};
use tachyon_v100::qtype::QType;
use std::io::{Read, Seek, SeekFrom};
use std::ptr::null_mut;

fn read_f32(p: &str) -> Vec<f32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn write_f32(p: &str, v: &[f32]) {
    std::fs::write(p, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}

fn dequant(t: QType, w: &[u8], k: usize, n: usize, out: &str) {
    assert_eq!(w.len(), t.row_bytes(k) * n, "大きさが合わない");
    let dw = DevBuf::from_bytes(w);
    let y = DevBuf::new(k * n * 4);
    cuda::check(unsafe { ffi::infer_dequant(t as i32, dw.ptr, k as i32, n as i64, y.f32p(), null_mut()) }, "dequant");
    cuda::sync();
    write_f32(out, &y.to_f32());
}

fn gemv(t: QType, w: &[u8], k: usize, n: usize, xp: &str, b: usize, out: &str) {
    let x = read_f32(xp);
    assert!(x.len() >= b * k, "x が足りない");
    assert!(w.len() >= t.row_bytes(k) * n, "重みが足りない");
    let dw = DevBuf::from_bytes(w);
    let dx = DevBuf::from_f32(&x[..b * k]);
    let y = DevBuf::new(b * n * 4);
    let run = || cuda::check(unsafe { ffi::infer_gemv(t as i32, dw.ptr, k as i32, n as i32, dx.f32p(), y.f32p(), b as i32, null_mut()) }, "gemv");
    run();
    cuda::sync();
    write_f32(out, &y.to_f32());
    let reps = 50;
    let ms = cuda::time_ms(|| (0..reps).for_each(|_| run())) / reps as f32;
    let bytes = (t.row_bytes(k) * n) as f64;
    eprintln!("gemv {:?} k={k} n={n} B={b}: {:.3} ms, {:.0} GB/s", t, ms, bytes / (ms as f64 * 1e-3) / 1e9);
}

fn moe(t: QType, w: &[u8], k: usize, n: usize, idp: &str, x_div: usize, xp: &str, out: &str) {
    let ids: Vec<i32> = std::fs::read(idp).unwrap().chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    let n_exp = w.len() / (t.row_bytes(k) * n);
    assert!(ids.iter().all(|&e| (e as usize) < n_exp), "ids が専門家の数 {n_exp} を超える");
    let x = read_f32(xp);
    let dw = DevBuf::from_bytes(w);
    let did = DevBuf::from_i32(&ids);
    let dx = DevBuf::from_f32(&x);
    let y = DevBuf::new(ids.len() * n * 4);
    cuda::check(
        unsafe { ffi::infer_moe_gemv(t as i32, dw.ptr, k as i32, n as i32, did.ptr.cast(), ids.len() as i32, x_div as i32, dx.f32p(), y.f32p(), null_mut()) },
        "moe",
    );
    cuda::sync();
    write_f32(out, &y.to_f32());
}

/// 「ファイル」か「ファイル@位置」から len バイトを読む
fn read_at(spec: &str, len: usize) -> Vec<u8> {
    let (path, off) = match spec.rsplit_once('@') {
        Some((p, o)) => (p, o.parse::<u64>().expect("位置が数でない")),
        None => (spec, 0),
    };
    let mut f = std::fs::File::open(path).unwrap();
    f.seek(SeekFrom::Start(off)).unwrap();
    let mut v = vec![0u8; len];
    f.read_exact(&mut v).expect("ファイルが短い");
    v
}

fn env_i32(name: &str, d: i32) -> i32 {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}

/// fp16 のビット列を f32 に
fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as u32;
    let bits = match e {
        0 if m == 0 => s,
        0 => return if s != 0 { -(m as f32) } else { m as f32 } * (2f32).powi(-24),
        31 => s | 0x7f80_0000 | (m << 13),
        _ => s | (((e + 112) as u32) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// f を何回か走らせて 1 回あたりのミリ秒
fn bench(reps: usize, mut f: impl FnMut()) -> f32 {
    f();
    cuda::sync();
    cuda::time_ms(|| (0..reps).for_each(|_| f())) / reps as f32
}

/// GPU の fp16 を f32 に
fn f16_buf(y: &DevBuf) -> Vec<f32> {
    y.to_bytes().chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()
}

fn report(what: &str, reps: usize, flops: f64, run: impl FnMut()) {
    if reps > 0 {
        let ms = bench(reps, run) as f64;
        eprintln!("{what}: {ms:.4} ms, {:.2} TFLOPS", flops / (ms * 1e-3) / 1e12);
    }
}

/// Y[m, n] = X[m, k] · W^T（X は fp16 のファイル）。出力は f32 のファイル
fn gemm(t: QType, w: &[u8], k: usize, n: usize, m: usize, xp: &str, out: &str, reps: usize) {
    let cfg = env_i32("INFER_GEMM_CFG", -1);
    let splits = env_i32("INFER_GEMM_SPLITS", 0);
    let ym = std::env::var("INFER_GEMM_Y").unwrap_or_default();
    let mode = if ym == "f16" { 1 } else if ym.starts_with("add:") { 2 } else { 0 };
    assert!(w.len() >= t.row_bytes(k) * n, "重みが足りない");
    let dw = DevBuf::from_bytes(&w[..t.row_bytes(k) * n]);
    let dx = DevBuf::from_bytes(&read_at(xp, m * k * 2));
    let y = match ym.strip_prefix("add:") {
        Some(p) => DevBuf::from_bytes(&read_at(p, m * n * 4)),
        None => DevBuf::new(m * n * if mode == 1 { 2 } else { 4 }),
    };
    // split-K の作業場: 自動なら infer_gemm_ws_bytes、決め打ちならその分け数ぶん
    let wsb = if splits > 1 { splits as usize * m * n * 4 } else { (unsafe { ffi::infer_gemm_ws_bytes(k as i32, n as i32, m as i32) }) as usize };
    let ws = DevBuf::new(wsb);
    let run = || {
        cuda::check(
            unsafe {
                ffi::infer_gemm_ex(t as i32, dw.ptr, k as i32, n as i32, m as i32, dx.ptr, y.ptr, mode, cfg, splits, ws.ptr, ws.len as i64, null_mut())
            },
            "gemm",
        )
    };
    run();
    cuda::sync();
    if out != "-" {
        write_f32(out, &if mode == 1 { f16_buf(&y) } else { y.to_f32() });
    }
    if mode == 2 {
        return; // 足し込みは何度も走らせると値が変わるので測らない
    }
    report(&format!("gemm {t:?} k={k} n={n} m={m} cfg={cfg} splits={splits}"), reps, 2.0 * (m * n * k) as f64, run);
}

/// gate と up をまとめて（Y は fp16 で受ける）
fn swiglu(tg: QType, wg: &[u8], tu: QType, wu: &[u8], k: usize, n: usize, m: usize, xp: &str, out: &str, reps: usize) {
    let cfg = env_i32("INFER_GEMM_CFG", -1);
    let (dg, du) = (DevBuf::from_bytes(wg), DevBuf::from_bytes(wu));
    let dx = DevBuf::from_bytes(&read_at(xp, m * k * 2));
    let y = DevBuf::new(m * n * 2);
    let run = || {
        cuda::check(
            unsafe { ffi::infer_gemm_swiglu(tg as i32, dg.ptr, tu as i32, du.ptr, k as i32, n as i32, m as i32, dx.ptr, y.ptr, cfg, null_mut()) },
            "swiglu",
        )
    };
    run();
    cuda::sync();
    if out != "-" {
        write_f32(out, &f16_buf(&y));
    }
    report(&format!("swiglu {tg:?}+{tu:?} k={k} n={n} m={m} cfg={cfg}"), reps, 4.0 * (m * n * k) as f64, run);
}

/// 同じ X で何本か（Y は f32 か f16。INFER_GEMM_Y=f16）
fn multi(mats: &[(QType, &[u8], usize)], k: usize, m: usize, xp: &str, out: &str, reps: usize) {
    let cfg = env_i32("INFER_GEMM_CFG", -1);
    let mode = if std::env::var("INFER_GEMM_Y").map(|s| s == "f16").unwrap_or(false) { 1 } else { 0 };
    let dws: Vec<DevBuf> = mats.iter().map(|(_, w, _)| DevBuf::from_bytes(w)).collect();
    let ys: Vec<DevBuf> = mats.iter().map(|(_, _, n)| DevBuf::new(m * n * if mode == 1 { 2 } else { 4 })).collect();
    let dx = DevBuf::from_bytes(&read_at(xp, m * k * 2));
    let ts: Vec<i32> = mats.iter().map(|(t, _, _)| *t as i32).collect();
    let ns: Vec<i32> = mats.iter().map(|(_, _, n)| *n as i32).collect();
    let wp: Vec<*const std::ffi::c_void> = dws.iter().map(|d| d.ptr as *const _).collect();
    let yp: Vec<*mut std::ffi::c_void> = ys.iter().map(|d| d.ptr).collect();
    let run = || {
        cuda::check(
            unsafe {
                ffi::infer_gemm_multi(mats.len() as i32, ts.as_ptr(), wp.as_ptr(), ns.as_ptr(), yp.as_ptr(), k as i32, m as i32, dx.ptr, mode, cfg, null_mut())
            },
            "multi",
        )
    };
    run();
    cuda::sync();
    if out != "-" {
        let v: Vec<f32> = ys.iter().flat_map(|y| if mode == 1 { f16_buf(y) } else { y.to_f32() }).collect();
        write_f32(out, &v);
    }
    let n_all: usize = ns.iter().map(|&n| n as usize).sum();
    report(&format!("multi {:?} k={k} n={n_all} m={m} cfg={cfg}", mats.iter().map(|x| x.0).collect::<Vec<_>>()), reps, 2.0 * (m * n_all * k) as f64, run);
}

fn set_device() {
    let dev: i32 = std::env::var("INFER_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    cuda::check(unsafe { ffi::cudaSetDevice(dev) }, "cudaSetDevice");
}

/// kcheck gguf …
fn from_gguf(a: &[String]) {
    let m = Model::open(&a[2]).unwrap_or_else(|e| panic!("{e}"));
    if a.len() == 3 {
        for t in &m.tensors {
            let shape: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
            println!("{}\t{}\t{}", t.name, ggml_type_name(t.ggml_type), shape.join(","));
        }
        return;
    }
    let num = |i: usize| -> usize { a[i].parse().unwrap_or_else(|_| panic!("数でない: {}", a[i])) };
    let reps = |i: usize| a.get(i).map(|_| num(i)).unwrap_or(0);
    if a[4] == "multi" {
        let tvs: Vec<_> = a[3].split(',').map(|nm| m.tensor(nm)).collect();
        let k = tvs[0].k();
        assert!(tvs.iter().all(|t| t.k() == k), "k が揃っていない");
        let mats: Vec<_> = tvs.iter().map(|t| (t.qt(), t.bytes, t.n_rows())).collect();
        set_device();
        return multi(&mats, k, num(5), &a[6], &a[7], reps(8));
    }
    let tv = m.tensor(&a[3]);
    // 行の始めと行数（行数 all は最後まで）
    let rows = || {
        let r0 = num(5);
        (r0, if a[6] == "all" { tv.n_rows() - r0 } else { num(6) })
    };
    let (k, op) = (tv.k(), a[4].as_str());
    if op == "raw" {
        let (r0, n) = rows();
        std::fs::write(&a[7], tv.rows(r0, n)).unwrap();
        return;
    }
    let t = tv.qt();
    set_device();
    match op {
        "dequant" => {
            let (r0, n) = rows();
            dequant(t, tv.rows(r0, n), k, n, &a[7]);
        }
        "gemv" => {
            let (r0, n) = rows();
            gemv(t, tv.rows(r0, n), k, n, &a[7], num(8), &a[9]);
        }
        "gemm" => gemm(t, tv.bytes, k, tv.n_rows(), num(5), &a[6], &a[7], reps(8)),
        "swiglu" => {
            let up = m.tensor(&a[5]);
            assert_eq!((up.k(), up.n_rows()), (k, tv.n_rows()), "gate と up の形が違う");
            swiglu(t, tv.bytes, up.qt(), up.bytes, k, tv.n_rows(), num(6), &a[7], &a[8], reps(9));
        }
        "moe" => {
            assert_eq!(tv.shape.len(), 3, "{}: 3 次元（専門家）でない", tv.name);
            moe(t, tv.bytes, k, tv.shape[1], &a[5], num(6), &a[7], &a[8]);
        }
        m => panic!("知らない動作 {m}"),
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a[1] == "gguf" {
        return from_gguf(&a);
    }
    set_device();
    let t = QType::from_name(&a[3]).expect("知らない型");
    let k: usize = a[4].parse().unwrap();
    let n: usize = a[5].parse().unwrap();
    if a[1] == "gemm" {
        let reps = a.get(9).map(|s| s.parse().unwrap()).unwrap_or(0);
        let m: usize = a[6].parse().unwrap();
        return gemm(t, &read_at(&a[2], t.row_bytes(k) * n), k, n, m, &a[7], &a[8], reps);
    }
    let w = std::fs::read(&a[2]).unwrap();
    match a[1].as_str() {
        "dequant" => dequant(t, &w, k, n, &a[6]),
        "gemv" => gemv(t, &w, k, n, &a[6], a[7].parse().unwrap(), &a[8]),
        "moe" => moe(t, &w, k, n, &a[6], a[7].parse().unwrap(), &a[8], &a[9]),
        m => panic!("知らない動作 {m}"),
    }
}
