//! KV の段分け（src/kv.rs）の測りと確かめ。GPU 1 枚で（1 つの処理で GPU 6GB 未満・固定した RAM 64GB 未満）。
//!
//!   kvbench pcie  [--mb 256]                                  … 生の PCIe（16KB〜16MB・1 本ずつ／まとめて／カーネル・node1/node0/cudaHostAlloc/pageable・両方向）
//!   kvbench tier  [--dtype f16|int8] [--host-dtype f16|int8] [--zero-copy] [--tokens 1024,32768,65536] [--mode memcpy|kernel] [--chunk 8]
//!                                                             … 追い出し・戻し・入れ替え・線形 Attention の状態（151MB）
//!   kvbench bulk  [--dtype f16|int8] [--tokens 131072] [--gpu-mb 4096]  … GPU に入りきらない長さを流しで（GPU の枠は使い回す）
//!   kvbench check [--iters 1500] [--seed 1] [--gpu 40] [--host 160]  … でたらめな操作で中身を確かめる（fp16/int8 × memcpy/kernel）
//!   kvbench quant                                             … fp16 ⇄ int8 の頁の往復の誤差、int8 に詰めながら RAM へ出す速さ
//!   kvbench blocks [--mb 256]                                 … zero-copy のカーネルのブロック数ごとの速さ（KERNEL_BLOCKS を決める）
//!   kvbench pin [--gb 16]                                     … 固定した RAM を取る・返す時間と、置かれた NUMA
//!   kvbench interfere [--gb 1]                                … 転送と計算（HBM を読む重い仕事）が互いをどれだけ遅くするか
use tachyon_v100::kv::{self, ck, ffi, CopyMode, Dir, GpuLoc, HostArena, KvDtype, KvShape, KvTier, Seg, TierConfig};
use std::ffi::c_void;
use std::time::Instant;

fn arg<'a>(a: &'a [String], k: &str, d: &'a str) -> &'a str {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).map(|s| s.as_str()).unwrap_or(d)
}

struct Ev(ffi::Event);
impl Ev {
    fn new() -> Self {
        let mut e = std::ptr::null_mut();
        ck(unsafe { ffi::cudaEventCreateWithFlags(&mut e, 0) }, "cudaEventCreate");
        Ev(e)
    }
    fn rec(&self, s: ffi::Stream) {
        ck(unsafe { ffi::cudaEventRecord(self.0, s) }, "cudaEventRecord");
    }
    fn ms_since(&self, a: &Ev) -> f64 {
        unsafe { ffi::cudaEventSynchronize(self.0) };
        let mut ms = 0f32;
        ck(unsafe { ffi::cudaEventElapsedTime(&mut ms, a.0, self.0) }, "cudaEventElapsedTime");
        ms as f64
    }
}
impl Drop for Ev {
    fn drop(&mut self) {
        unsafe { ffi::cudaEventDestroy(self.0) };
    }
}

fn stream() -> ffi::Stream {
    let mut s = std::ptr::null_mut();
    ck(unsafe { ffi::cudaStreamCreateWithFlags(&mut s, 1) }, "cudaStreamCreate");
    s
}
fn dmalloc(n: usize) -> *mut u8 {
    let mut p = std::ptr::null_mut();
    ck(unsafe { ffi::cudaMalloc(&mut p, n) }, "cudaMalloc");
    p.cast()
}
fn sync(s: ffi::Stream) {
    ck(unsafe { ffi::cudaStreamSynchronize(s) }, "cudaStreamSynchronize");
}
fn gpu_mem() -> (f64, f64) {
    let (mut f, mut t) = (0, 0);
    unsafe { ffi::cudaMemGetInfo(&mut f, &mut t) };
    ((t - f) as f64 / 1e9, t as f64 / 1e9)
}
fn gbs(bytes: usize, ms: f64) -> f64 {
    bytes as f64 / (ms * 1e-3) / 1e9
}
fn human(n: usize) -> String {
    if n >= 1 << 30 {
        format!("{:.2}GB", n as f64 / (1u64 << 30) as f64)
    } else if n >= 1 << 20 {
        format!("{}MB", n >> 20)
    } else {
        format!("{}KB", n >> 10)
    }
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// cudaMemcpyAsync を n 本まとめて（size ずつ、場所をずらして）
fn memcpy_batch(dir: Dir, dev: *mut u8, host: *mut u8, size: usize, total: usize, s: ffi::Stream) {
    for i in 0..total / size {
        let (d, h) = unsafe { (dev.add(i * size), host.add(i * size)) };
        let r = unsafe {
            match dir {
                Dir::H2D => ffi::cudaMemcpyAsync(d.cast(), h as *const c_void, size, ffi::H2D, s),
                Dir::D2H => ffi::cudaMemcpyAsync(h.cast(), d as *const c_void, size, ffi::D2H, s),
            }
        };
        ck(r, "cudaMemcpyAsync");
    }
}
fn kernel_batch(dir: Dir, dev: *mut u8, hdev: *mut u8, size: usize, total: usize, blocks: i32, s: ffi::Stream) {
    let d: Vec<ffi::CopyDesc> = (0..total / size)
        .map(|i| {
            let (g, h) = unsafe { (dev.add(i * size), hdev.add(i * size)) };
            match dir {
                Dir::H2D => ffi::CopyDesc { src: h, dst: g, bytes: size as u64 },
                Dir::D2H => ffi::CopyDesc { src: g, dst: h, bytes: size as u64 },
            }
        })
        .collect();
    ck(unsafe { ffi::kv_copy(d.as_ptr(), d.len() as i32, blocks, s) }, "kv_copy");
}

fn pcie(a: &[String]) {
    let total: usize = arg(a, "--mb", "256").parse::<usize>().unwrap() << 20;
    let node = kv::gpu_numa_node(kv::current_device());
    let other = node.map(|n| 1 - n);
    println!("GPU の NUMA: {node:?}（固定した RAM はこの node に置く）");
    let dev = dmalloc(total * 2);
    let s1 = stream();
    let s2 = stream();
    let t0 = Instant::now();
    let h_near = HostArena::new(total * 2, node);
    let reg_s = t0.elapsed().as_secs_f64();
    println!(
        "固定 {}（mmap→mbind→cudaHostRegister）{:.3} 秒 = {:.1} GB/s、置かれた node {:?}",
        human(total * 2),
        reg_s,
        (total * 2) as f64 / reg_s / 1e9,
        h_near.node_histogram(64)
    );
    let (b, e) = (Ev::new(), Ev::new());
    // ── 大きさごと ──
    println!("\n| 1 本の大きさ | 向き | 1 本ずつ（同期）µs | 1 本ずつ GB/s | まとめて GB/s（最大／中央） | カーネル（zero-copy）GB/s |");
    println!("|---|---|---|---|---|---|");
    for &size in &[16usize << 10, 64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20] {
        for dir in [Dir::H2D, Dir::D2H] {
            // 1 本ずつ: 出して待つ（API と同期の手間も込み）
            let n1 = (total / size).clamp(8, 200);
            let mut lat = vec![];
            for i in 0..n1 + 3 {
                let off = (i % (total / size)) * size;
                let t = Instant::now();
                memcpy_batch(dir, unsafe { dev.add(off) }, unsafe { h_near.ptr.add(off) }, size, size, s1);
                sync(s1);
                if i >= 3 {
                    lat.push(t.elapsed().as_secs_f64() * 1e6);
                }
            }
            let l = median(lat);
            let mut mb = vec![];
            let mut kb = vec![];
            for _ in 0..5 {
                b.rec(s1);
                memcpy_batch(dir, dev, h_near.ptr, size, total, s1);
                e.rec(s1);
                mb.push(gbs(total, e.ms_since(&b)));
                b.rec(s1);
                kernel_batch(dir, dev, h_near.dev, size, total, kv::KERNEL_BLOCKS, s1);
                e.rec(s1);
                kb.push(gbs(total, e.ms_since(&b)));
            }
            let mx = |v: &Vec<f64>| v.iter().cloned().fold(0.0, f64::max);
            println!(
                "| {} | {:?} | {:.1} | {:.2} | {:.2} / {:.2} | {:.2} / {:.2} |",
                human(size),
                dir,
                l,
                size as f64 / (l * 1e-6) / 1e9,
                mx(&mb),
                median(mb.clone()),
                mx(&kb),
                median(kb.clone())
            );
        }
    }
    // ── 両方向を同時に（追い出しと読み込みが重なるとき）──
    let mut bi = vec![];
    for _ in 0..5 {
        b.rec(s1);
        ck(unsafe { ffi::cudaStreamWaitEvent(s2, b.0, 0) }, "wait");
        memcpy_batch(Dir::D2H, dev, h_near.ptr, 4 << 20, total, s1);
        memcpy_batch(Dir::H2D, unsafe { dev.add(total) }, unsafe { h_near.ptr.add(total) }, 4 << 20, total, s2);
        let e2 = Ev::new();
        e.rec(s1);
        e2.rec(s2);
        let ms = e.ms_since(&b).max(e2.ms_since(&b));
        bi.push(gbs(2 * total, ms));
    }
    println!("\n両方向同時（4MB ずつ、D2H {} ＋ H2D {}）: 合計 {:.2} GB/s（中央 {:.2}）", human(total), human(total), bi.iter().cloned().fold(0.0, f64::max), median(bi.clone()));

    // ── RAM の種類（4MB・まとめて）──
    println!("\n| RAM | H2D GB/s | D2H GB/s |");
    println!("|---|---|---|");
    let row = |name: &str, host: *mut u8| {
        let mut r = [vec![], vec![]];
        for _ in 0..5 {
            for (k, dir) in [Dir::H2D, Dir::D2H].into_iter().enumerate() {
                b.rec(s1);
                memcpy_batch(dir, dev, host, 4 << 20, total, s1);
                e.rec(s1);
                r[k].push(gbs(total, e.ms_since(&b)));
            }
        }
        let mx = |v: &Vec<f64>| v.iter().cloned().fold(0.0, f64::max);
        println!("| {name} | {:.2} | {:.2} |", mx(&r[0]), mx(&r[1]));
    };
    row(&format!("固定・node{}（GPU と同じ）", node.unwrap_or(-1)), h_near.ptr);
    drop(h_near);
    if let Some(o) = other {
        let h = HostArena::new(total, Some(o));
        let hist = h.node_histogram(64);
        row(&format!("固定・node{o}（隣の node。置かれた {hist:?}）"), h.ptr);
    }
    {
        let h = HostArena::cuda_alloc(total);
        row("cudaHostAlloc（node 指定なし）", h.ptr);
    }
    {
        let mut v = vec![1u8; total];
        row("固定しない（pageable）", v.as_mut_ptr());
    }
    let (used, tot) = gpu_mem();
    println!("\nGPU の使用 {used:.2}/{tot:.1} GB（この処理を含む全体）");
    unsafe { ffi::cudaFree(dev.cast()) };
}

fn parse_dtype(s: &str) -> KvDtype {
    match s {
        "f16" => KvDtype::F16,
        "int8" => KvDtype::Int8,
        _ => panic!("--dtype は f16 か int8"),
    }
}
fn parse_mode(s: &str) -> CopyMode {
    match s {
        "memcpy" => CopyMode::Memcpy,
        "kernel" => CopyMode::Kernel,
        _ => panic!("--mode は memcpy か kernel"),
    }
}

/// 列 s を n トークン分作り、全頁を模様で埋めて指紋を返す。
/// RAM 側が int8 なら、値の模様（fp16）で埋め、戻った後にあるべき指紋（GPU の中で int8 に詰めて戻した物）を返す
fn make_seq(t: &mut KvTier, n: usize, seed: u64, hs: &kv::Hasher) -> (u32, Vec<u64>) {
    let s = t.new_seq();
    t.append(s, n).unwrap();
    let tab = t.table(s).unwrap();
    let ptrs: Vec<*mut u8> = tab.iter().map(|&g| t.slot_ptr(g)).collect();
    let bb = t.shape.block_bytes();
    let conv = t.host_shape.dtype != t.shape.dtype;
    kv::fill_pages(&ptrs, bb, seed, conv as i32, t.compute);
    if !conv {
        return (s, hs.hash(&ptrs, bb, t.compute));
    }
    let (f, q) = (t.shape, t.host_shape);
    let (fl, ql, layers) = (f.layer_page_bytes(), q.layer_page_bytes(), f.layers);
    let (qs, fs) = (dmalloc(16 * q.block_bytes()), dmalloc(16 * bb));
    let mut out = vec![];
    for ch in tab.chunks(16) {
        let (mut dq, mut dd) = (vec![], vec![]);
        for (i, &g) in ch.iter().enumerate() {
            for l in 0..layers {
                let qp = unsafe { qs.add((i * layers + l) * ql) };
                dq.push(ffi::ConvDesc { src: t.layer_ptr(g, l), dst: qp });
                dd.push(ffi::ConvDesc { src: qp, dst: unsafe { fs.add(i * bb + l * fl) } });
            }
        }
        let (rows, dim) = (f.rows() as i32, f.head_dim as i32);
        ck(unsafe { ffi::kv_quant(dq.as_ptr(), dq.len() as i32, rows, dim, 0, t.compute) }, "kv_quant");
        ck(unsafe { ffi::kv_dequant(dd.as_ptr(), dd.len() as i32, rows, dim, 0, t.compute) }, "kv_dequant");
        let p: Vec<*mut u8> = (0..ch.len()).map(|i| unsafe { fs.add(i * bb) }).collect();
        out.extend(hs.hash(&p, bb, t.compute));
    }
    unsafe {
        ffi::cudaFree(qs.cast());
        ffi::cudaFree(fs.cast());
    }
    (s, out)
}
fn verify(t: &KvTier, s: u32, want: &[u64], hs: &kv::Hasher) -> bool {
    let tab = t.table(s).expect("揃っていない");
    let ptrs: Vec<*mut u8> = tab.iter().map(|&g| t.slot_ptr(g)).collect();
    hs.hash(&ptrs, t.shape.block_bytes(), t.compute) == want
}
fn scrub(t: &KvTier) {
    let n = t.mgr.n_gpu as usize * t.shape.block_bytes();
    ck(unsafe { ffi::cudaMemsetAsync(t.gpu_base.cast(), 0xA5, n, t.compute) }, "cudaMemset");
    sync(t.compute);
}

fn tier(a: &[String]) {
    let dtype = parse_dtype(arg(a, "--dtype", "f16"));
    let mode = parse_mode(arg(a, "--mode", "memcpy"));
    let chunk: usize = arg(a, "--chunk", "8").parse().unwrap();
    let toks: Vec<usize> = arg(a, "--tokens", "1024,32768,65536").split(',').map(|x| x.parse().unwrap()).collect();
    let host_dtype = Some(parse_dtype(arg(a, "--host-dtype", arg(a, "--dtype", "f16"))));
    let zc = a.iter().any(|x| x == "--zero-copy");
    let shape = KvShape::qwen38_27b(dtype);
    let bb = shape.block_bytes();
    let hb = KvShape { dtype: host_dtype.unwrap(), ..shape }.block_bytes();
    let hs = kv::Hasher::new(4096);
    println!(
        "GPU {dtype:?}・RAM {:?}{}・{mode:?}・頁 {}（{:.1}KB/トークン）・RAM の頁 {}・1 回の転送 {chunk} 頁",
        host_dtype.unwrap(),
        if zc { "（zero-copy で詰める）" } else { "" },
        human(bb),
        shape.bytes_per_token() / 1024.0,
        human(hb)
    );
    println!("\n| トークン | KV | 追い出し ms（GB/s） | 戻し ms（GB/s） | 2 回目の追い出し（RAM に写しあり）ms | 出すまでの手間 µs | 中身 |");
    println!("|---|---|---|---|---|---|---|");
    for &n in &toks {
        let np = shape.pages_for(n);
        let mut t = KvTier::new(TierConfig { shape, gpu_pages: np as u32 + 2, host_pages: np as u32 + 2, host_node: None, mode, host_dtype });
        t.mgr.chunk_pages = chunk;
        t.conv_zero_copy = zc;
        let (s, want) = make_seq(&mut t, n, 42, &hs);
        let bytes = np * bb;
        let mut ev = vec![];
        let mut re = vec![];
        let mut clean = vec![];
        let mut sub = vec![];
        let mut ok = true;
        let mut s = s;
        for rep in 0..3 {
            if rep > 0 {
                // RAM の写しがある列は追い出しが転送なしになるので、作り直す
                t.free_seq(s);
                s = make_seq(&mut t, n, 42, &hs).0;
            }
            sync(t.compute);
            let t0 = Instant::now();
            let tk = t.evict_seq(s).unwrap();
            sub.push(t0.elapsed().as_secs_f64() * 1e6);
            t.wait(tk);
            ev.push(t0.elapsed().as_secs_f64() * 1e3);
            scrub(&t);
            let t0 = Instant::now();
            let tk = t.prefetch(s).unwrap();
            t.wait(tk);
            re.push(t0.elapsed().as_secs_f64() * 1e3);
            ok &= verify(&t, s, &want, &hs);
            let t0 = Instant::now();
            let tk = t.evict_seq(s).unwrap();
            t.wait(tk);
            clean.push(t0.elapsed().as_secs_f64() * 1e3);
            let tk = t.prefetch(s).unwrap();
            t.wait(tk);
            ok &= verify(&t, s, &want, &hs);
        }
        t.free_seq(s);
        let (e, r) = (median(ev), median(re));
        println!(
            "| {n} | {} | {e:.1}（{:.2}） | {r:.1}（{:.2}） | {:.3} | {:.0} | {} |",
            human(bytes),
            gbs(bytes, e),
            gbs(bytes, r),
            median(clean),
            median(sub),
            if ok { "OK" } else { "NG" }
        );
        assert!(ok, "中身が違う");
        let (used, _) = gpu_mem();
        eprintln!("  GPU 使用（全体）{used:.2} GB");
    }

    // ── 入れ替え: A（RAM）を戻すために B（GPU）を追い出す。D2H と H2D が重なる ──
    let n = *toks.iter().find(|&&x| x >= 8192).unwrap_or(&toks[toks.len() - 1]);
    let n = n.min(32768);
    let np = shape.pages_for(n);
    // GPU の枠はちょうど 1 人分（B の頁が全部 RAM を通る。RAM 側 int8 なら、通らなかった頁は fp16 のまま＝指紋が違うので）
    let mut t = KvTier::new(TierConfig { shape, gpu_pages: np as u32, host_pages: 2 * np as u32 + 4, host_node: None, mode, host_dtype });
    t.mgr.chunk_pages = chunk;
    t.conv_zero_copy = zc;
    let (a_s, a_h) = make_seq(&mut t, n, 1, &hs);
    let tk = t.evict_seq(a_s).unwrap();
    t.wait(tk);
    let (b_s, b_h) = make_seq(&mut t, n, 2, &hs);
    sync(t.compute);
    let t0 = Instant::now();
    let tk = t.prefetch(a_s).unwrap(); // B を追い出しながら A を戻す
    t.wait(tk);
    t.sync_all();
    let swap = t0.elapsed().as_secs_f64() * 1e3;
    let ok1 = verify(&t, a_s, &a_h, &hs);
    // 次は B を戻す: A は RAM に同じ写しがあるので追い出しは転送なし
    let t0 = Instant::now();
    let tk = t.prefetch(b_s).unwrap();
    t.wait(tk);
    t.sync_all();
    let swap2 = t0.elapsed().as_secs_f64() * 1e3;
    let ok2 = verify(&t, b_s, &b_h, &hs);
    println!(
        "\n入れ替え {n} トークン（{}）: B を追い出しつつ A を戻す {swap:.1} ms（片道ずつなら約 2 倍）、A に写しがあるときの B の戻し {swap2:.1} ms、中身 {}",
        human(np * bb),
        if ok1 && ok2 { "OK" } else { "NG" }
    );
    assert!(ok1 && ok2);
    drop(t);

    // ── 線形 Attention の状態（48 層・1 人 約 151MB）──
    let sb = 151usize << 20;
    let mut t = KvTier::new(TierConfig { shape, gpu_pages: 2, host_pages: (sb / hb + 2) as u32, host_node: None, mode, host_dtype });
    let dev = dmalloc(sb);
    let back = dmalloc(sb);
    kv::fill_pages(&[dev], sb, 9, 0, std::ptr::null_mut());
    let h0 = hs.hash(&[dev], sb, std::ptr::null_mut());
    let mut sv = vec![];
    let mut ld = vec![];
    for _ in 0..5 {
        let t0 = Instant::now();
        let (bl, tk) = t.blob_save(dev, sb).unwrap();
        t.wait(tk);
        sv.push(t0.elapsed().as_secs_f64() * 1e3);
        let t0 = Instant::now();
        let tk = t.blob_load(bl, back);
        t.wait(tk);
        ld.push(t0.elapsed().as_secs_f64() * 1e3);
        assert_eq!(hs.hash(&[back], sb, std::ptr::null_mut()), h0, "状態が戻らない");
        t.blob_free(bl);
    }
    println!("線形 Attention の状態 151MB: RAM へ {:.1} ms・戻し {:.1} ms（中身 OK）", median(sv), median(ld));
    unsafe {
        ffi::cudaFree(dev.cast());
        ffi::cudaFree(back.cast());
    }
}

/// GPU に入りきらない長さを流しで: GPU の枠は使い回し、RAM の枠は全部別
fn bulk(a: &[String]) {
    let dtype = parse_dtype(arg(a, "--dtype", "f16"));
    let mode = parse_mode(arg(a, "--mode", "memcpy"));
    let n: usize = arg(a, "--tokens", "131072").parse().unwrap();
    let gpu_mb: usize = arg(a, "--gpu-mb", "4096").parse().unwrap();
    let chunk: usize = arg(a, "--chunk", "8").parse().unwrap();
    let shape = KvShape::qwen38_27b(dtype);
    let bb = shape.block_bytes();
    let np = shape.pages_for(n);
    let gp = ((gpu_mb << 20) / bb).min(np) as u32;
    let mut t = KvTier::new(TierConfig { shape, gpu_pages: gp, host_pages: np as u32, host_node: None, mode, host_dtype: None });
    let mut res = vec![];
    for dir in [Dir::D2H, Dir::H2D] {
        let mut ms = vec![];
        for _ in 0..3 {
            let plans: Vec<_> = (0..np)
                .collect::<Vec<_>>()
                .chunks(chunk)
                .map(|c| {
                    let segs = c.iter().map(|&i| Seg { gpu: GpuLoc::Slot(i as u32 % gp), host: i as u32, bytes: bb }).collect();
                    t.mgr.raw_plan(dir, segs)
                })
                .collect();
            sync(t.compute);
            let t0 = Instant::now();
            let tk = t.submit(kv::Job { plans, waits: vec![] });
            t.wait(tk);
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        res.push(median(ms));
    }
    let bytes = np * bb;
    println!(
        "{dtype:?} {n} トークン（{}、GPU の枠 {gp} を使い回し）: 追い出し {:.0} ms（{:.2} GB/s）・戻し {:.0} ms（{:.2} GB/s）",
        human(bytes),
        res[0],
        gbs(bytes, res[0]),
        res[1],
        gbs(bytes, res[1])
    );
}

fn check(a: &[String]) {
    let iters: usize = arg(a, "--iters", "1500").parse().unwrap();
    let seed: u64 = arg(a, "--seed", "1").parse().unwrap();
    let gpu: u32 = arg(a, "--gpu", "40").parse().unwrap();
    let host: u32 = arg(a, "--host", "160").parse().unwrap();
    for (dt, mode) in [(KvDtype::F16, CopyMode::Memcpy), (KvDtype::F16, CopyMode::Kernel), (KvDtype::Int8, CopyMode::Memcpy), (KvDtype::Int8, CopyMode::Kernel)] {
        let t0 = Instant::now();
        match kv::selftest(iters, dt, mode, seed, gpu, host) {
            Ok(r) => println!(
                "{dt:?} {mode:?} GPU {gpu}・RAM {host} 頁: OK  操作 {}・確かめた頁 {}・D2H {} 頁・H2D {} 頁・写しありで転送なし {}・COW {}・RAM の写しを捨てた {}（{:.1} 秒）",
                r.ops,
                r.verified_pages,
                r.stats.d2h_pages,
                r.stats.h2d_pages,
                r.stats.clean_evicts,
                r.stats.cow,
                r.stats.host_drops,
                t0.elapsed().as_secs_f64()
            ),
            Err(e) => {
                println!("{dt:?} {mode:?}: NG {e}");
                std::process::exit(1);
            }
        }
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as u32;
    let v = if e == 0 {
        (m as f32) * 2f32.powi(-24)
    } else if e == 31 {
        f32::INFINITY
    } else {
        (1.0 + m as f32 / 1024.0) * 2f32.powi(e - 15)
    };
    if s == 1 {
        -v
    } else {
        v
    }
}

fn quant(_a: &[String]) {
    let f = KvShape::qwen38_27b(KvDtype::F16);
    let q = KvShape::qwen38_27b(KvDtype::Int8);
    let nb = 64; // 64 頁 = 4096 トークン
    let (fl, ql) = (f.layer_page_bytes(), q.layer_page_bytes());
    let nl = nb * f.layers;
    let src = dmalloc(nl * fl);
    let qd = dmalloc(nl * ql);
    let back = dmalloc(nl * fl);
    kv::fill_pages(&[src], nl * fl, 5, 1, std::ptr::null_mut());
    let conv = |s: *mut u8, sl: usize, d: *mut u8, dl: usize| -> Vec<ffi::ConvDesc> {
        (0..nl).map(|i| ffi::ConvDesc { src: unsafe { s.add(i * sl) }, dst: unsafe { d.add(i * dl) } }).collect()
    };
    let rows = f.rows() as i32;
    let dim = f.head_dim as i32;
    let s = std::ptr::null_mut();
    let dq = conv(src, fl, qd, ql);
    let dd = conv(qd, ql, back, fl);
    ck(unsafe { ffi::kv_quant(dq.as_ptr(), nl as i32, rows, dim, 0, s) }, "kv_quant");
    ck(unsafe { ffi::kv_dequant(dd.as_ptr(), nl as i32, rows, dim, 0, s) }, "kv_dequant");
    sync(s);
    // CPU で比べる: 行ごとに |x - x'| ≤ amax/127/2 ＋ fp16 の丸め
    let mut a = vec![0u16; nl * fl / 2];
    let mut b = vec![0u16; nl * fl / 2];
    unsafe {
        ffi::cudaMemcpyAsync(a.as_mut_ptr().cast(), src as *const c_void, nl * fl, ffi::D2H, s);
        ffi::cudaMemcpyAsync(b.as_mut_ptr().cast(), back as *const c_void, nl * fl, ffi::D2H, s);
    }
    sync(s);
    let d = f.head_dim;
    let (mut worst, mut se, mut sx, mut bad) = (0f64, 0f64, 0f64, 0usize);
    for r in 0..a.len() / d {
        let xs: Vec<f32> = a[r * d..(r + 1) * d].iter().map(|&h| f16_to_f32(h)).collect();
        let ys: Vec<f32> = b[r * d..(r + 1) * d].iter().map(|&h| f16_to_f32(h)).collect();
        let amax = xs.iter().fold(0f32, |m, x| m.max(x.abs()));
        for (x, y) in xs.iter().zip(&ys) {
            let e = (x - y).abs() as f64;
            let lim = amax as f64 / 127.0 * 0.5 * 1.01 + (x.abs() as f64) * 1e-3 + 1e-6;
            if e > lim {
                bad += 1;
            }
            worst = worst.max(e / amax.max(1e-6) as f64);
            se += e * e;
            sx += (*x as f64) * (*x as f64);
        }
    }
    println!(
        "fp16 → int8 → fp16（{nb} 頁・{} 行）: 誤差の上限を超えた要素 {bad}、最大誤差／行の最大値 {:.5}（理論 1/254 = {:.5}）、SNR {:.1} dB",
        a.len() / d,
        worst,
        1.0 / 254.0,
        10.0 * (sx / se).log10()
    );
    assert_eq!(bad, 0);

    // int8 に詰めながら RAM へ（zero-copy で書く）・RAM の int8 を読みながら fp16 に戻す
    let host = HostArena::new(nl * ql, kv::gpu_numa_node(kv::current_device()));
    let st = stream();
    let to_h = conv(src, fl, host.dev, ql);
    let from_h = conv(host.dev, ql, back, fl);
    let (b0, e0) = (Ev::new(), Ev::new());
    let mut t1 = vec![];
    let mut t2 = vec![];
    for _ in 0..5 {
        b0.rec(st);
        ck(unsafe { ffi::kv_quant(to_h.as_ptr(), nl as i32, rows, dim, kv::KERNEL_BLOCKS, st) }, "kv_quant");
        e0.rec(st);
        t1.push(e0.ms_since(&b0));
        b0.rec(st);
        ck(unsafe { ffi::kv_dequant(from_h.as_ptr(), nl as i32, rows, dim, kv::KERNEL_BLOCKS, st) }, "kv_dequant");
        e0.rec(st);
        t2.push(e0.ms_since(&b0));
    }
    let (m1, m2) = (median(t1), median(t2));
    let fb = nl * fl;
    println!(
        "GPU fp16 → RAM int8（{}→{}）: {m1:.1} ms = fp16 換算 {:.2} GB/s（PCIe は {:.2} GB/s）／RAM int8 → GPU fp16: {m2:.1} ms = fp16 換算 {:.2} GB/s",
        human(fb),
        human(nl * ql),
        gbs(fb, m1),
        gbs(nl * ql, m1),
        gbs(fb, m2)
    );
    // 戻した物が GPU の中で詰めて戻した物と同じか
    let mut c = vec![0u16; nl * fl / 2];
    unsafe { ffi::cudaMemcpyAsync(c.as_mut_ptr().cast(), back as *const c_void, fb, ffi::D2H, st) };
    sync(st);
    assert!(c == b, "RAM を通した int8 の往復が GPU の中の往復と違う");
    println!("RAM を通した往復 = GPU の中の往復（同じビット）");
    unsafe {
        ffi::cudaFree(src.cast());
        ffi::cudaFree(qd.cast());
        ffi::cudaFree(back.cast());
    }
}

/// zero-copy のカーネルのブロック数ごとの速さ（写す・int8 に詰めながら・両方向同時）
fn blocks(a: &[String]) {
    let total: usize = arg(a, "--mb", "256").parse::<usize>().unwrap() << 20;
    let dev = dmalloc(total * 2);
    let host = HostArena::new(total * 2, kv::gpu_numa_node(kv::current_device()));
    let (s1, s2) = (stream(), stream());
    let (b, e, e2) = (Ev::new(), Ev::new(), Ev::new());
    let f = KvShape::qwen38_27b(KvDtype::F16);
    let q = KvShape::qwen38_27b(KvDtype::Int8);
    let (fl, ql) = (f.layer_page_bytes(), q.layer_page_bytes());
    let nl = total / fl;
    let (rows, dim) = (f.rows() as i32, f.head_dim as i32);
    let to_h: Vec<ffi::ConvDesc> = (0..nl).map(|i| ffi::ConvDesc { src: unsafe { dev.add(i * fl) }, dst: unsafe { host.dev.add(i * ql) } }).collect();
    let from_h: Vec<ffi::ConvDesc> = (0..nl).map(|i| ffi::ConvDesc { src: unsafe { host.dev.add(i * ql) }, dst: unsafe { dev.add(total + i * fl) } }).collect();
    kv::fill_pages(&[dev], total, 3, 1, s1);
    sync(s1);
    println!("| ブロック数 | 16KB H2D | 16KB D2H | 256KB H2D | 256KB D2H | 両方向同時 256KB（合計） | fp16→RAM int8（fp16 換算） | RAM int8→fp16（fp16 換算） |");
    println!("|---|---|---|---|---|---|---|---|");
    for &nb in &[4, 8, 16, 32, 64, 160, 0] {
        let mut cells = vec![];
        for &size in &[16usize << 10, 256 << 10] {
            for dir in [Dir::H2D, Dir::D2H] {
                let mut v = vec![];
                for _ in 0..3 {
                    b.rec(s1);
                    kernel_batch(dir, dev, host.dev, size, total, nb, s1);
                    e.rec(s1);
                    v.push(gbs(total, e.ms_since(&b)));
                }
                cells.push(format!("{:.2}", v.iter().cloned().fold(0.0, f64::max)));
            }
        }
        let mut v = vec![];
        for _ in 0..3 {
            b.rec(s1);
            ck(unsafe { ffi::cudaStreamWaitEvent(s2, b.0, 0) }, "wait");
            kernel_batch(Dir::D2H, dev, host.dev, 256 << 10, total, nb, s1);
            kernel_batch(Dir::H2D, unsafe { dev.add(total) }, unsafe { host.dev.add(total) }, 256 << 10, total, nb, s2);
            e.rec(s1);
            e2.rec(s2);
            v.push(gbs(2 * total, e.ms_since(&b).max(e2.ms_since(&b))));
        }
        cells.push(format!("{:.2}", v.iter().cloned().fold(0.0, f64::max)));
        for (k, d) in [&to_h, &from_h].into_iter().enumerate() {
            let mut v = vec![];
            for _ in 0..3 {
                b.rec(s1);
                let r = unsafe {
                    if k == 0 {
                        ffi::kv_quant(d.as_ptr(), nl as i32, rows, dim, nb, s1)
                    } else {
                        ffi::kv_dequant(d.as_ptr(), nl as i32, rows, dim, nb, s1)
                    }
                };
                ck(r, "conv");
                e.rec(s1);
                v.push(gbs(nl * fl, e.ms_since(&b)));
            }
            cells.push(format!("{:.2}", v.iter().cloned().fold(0.0, f64::max)));
        }
        println!("| {} | {} |", if nb == 0 { "全部".to_string() } else { nb.to_string() }, cells.join(" | "));
    }
    unsafe { ffi::cudaFree(dev.cast()) };
}

/// 転送が計算（HBM を読む重い仕事＝デコードの代わり）をどれだけ遅くするか、計算が転送をどれだけ遅くするか
fn interfere(a: &[String]) {
    let gb: usize = arg(a, "--gb", "1").parse().unwrap();
    let total = gb << 30;
    let pb = 4usize << 20;
    let np = 512; // 計算の代わり: 2GB を読んで指紋を取る（HBM を使い切る）
    let work = dmalloc(np * pb);
    let ptrs: Vec<*mut u8> = (0..np).map(|i| unsafe { work.add(i * pb) }).collect();
    let src = dmalloc(total);
    let host = HostArena::new(total, kv::gpu_numa_node(kv::current_device()));
    let hs = kv::Hasher::new(np);
    let (sc, st) = (stream(), stream());
    let (b, e, bt, et) = (Ev::new(), Ev::new(), Ev::new(), Ev::new());
    let run_compute = |reps: usize| {
        b.rec(sc);
        for _ in 0..reps {
            ck(unsafe { ffi::kv_hash(ptrs.as_ptr(), np as i32, pb as u64, hs_buf(&hs), sc) }, "kv_hash");
        }
        e.rec(sc);
    };
    // 単独
    run_compute(3);
    let alone = e.ms_since(&b) / 3.0;
    println!("計算の代わり（2GB を読む）単独: {alone:.2} ms/回 = {:.0} GB/s", gbs(np * pb, alone));
    let mut base = vec![];
    for _ in 0..3 {
        bt.rec(st);
        memcpy_batch(Dir::D2H, src, host.ptr, 4 << 20, total, st);
        et.rec(st);
        base.push(et.ms_since(&bt));
    }
    let base = median(base);
    println!("転送 {gb}GB D2H 単独（memcpy 4MB）: {base:.0} ms = {:.2} GB/s\n", gbs(total, base));
    println!("| 転送のやり方 | 計算 ms/回（単独比） | 転送 GB/s |");
    println!("|---|---|---|");
    for (name, nb) in [("memcpy 4MB（コピーエンジン）", -1), ("カーネル 256KB・32 ブロック", 32), ("カーネル 256KB・8 ブロック", 8), ("カーネル 256KB・全部", 0)] {
        let reps = 12;
        bt.rec(st);
        if nb < 0 {
            memcpy_batch(Dir::D2H, src, host.ptr, 4 << 20, total, st);
        } else {
            kernel_batch(Dir::D2H, src, host.dev, 256 << 10, total, nb, st);
        }
        et.rec(st);
        run_compute(reps);
        let c = e.ms_since(&b) / reps as f64;
        let t = et.ms_since(&bt);
        println!("| {name} | {c:.2}（{:.2} 倍） | {:.2} |", c / alone, gbs(total, t));
    }
    {
        // zero-copy で RAM を読む向き（H2D）も
        let reps = 12;
        bt.rec(st);
        kernel_batch(Dir::H2D, src, host.dev, 256 << 10, total, 32, st);
        et.rec(st);
        run_compute(reps);
        let c = e.ms_since(&b) / reps as f64;
        println!("| カーネル 256KB・32 ブロック（H2D・RAM を読む） | {c:.2}（{:.2} 倍） | {:.2} |", c / alone, gbs(total, et.ms_since(&bt)));
    }
    unsafe { ffi::cudaFree(src.cast()) };
    drop(host);
    // 段分けの口から（1GB の列を追い出す間に計算を回す）。fp16 のまま／RAM 側 int8（置き場で詰める・zero-copy で詰める）
    for (name, hd, zc) in [("KvTier fp16（memcpy）", KvDtype::F16, false), ("KvTier RAM int8（置き場で詰める）", KvDtype::Int8, false), ("KvTier RAM int8（zero-copy で詰める）", KvDtype::Int8, true)] {
        let shape = KvShape::qwen38_27b(KvDtype::F16);
        let n = total / shape.bytes_per_token() as usize;
        let np = shape.pages_for(n) as u32;
        let mut t = KvTier::new(TierConfig { shape, gpu_pages: np, host_pages: np, host_node: None, mode: CopyMode::Memcpy, host_dtype: Some(hd) });
        t.conv_zero_copy = zc;
        let s = t.new_seq();
        t.append(s, n).unwrap();
        let tab = t.table(s).unwrap();
        let p: Vec<*mut u8> = tab.iter().map(|&g| t.slot_ptr(g)).collect();
        kv::fill_pages(&p, shape.block_bytes(), 1, 1, std::ptr::null_mut());
        sync(std::ptr::null_mut());
        let reps = 40;
        let t0 = Instant::now();
        let tk = t.evict_seq(s).unwrap();
        run_compute(reps);
        t.wait(tk);
        let tt = t0.elapsed().as_secs_f64() * 1e3;
        let c = e.ms_since(&b) / reps as f64;
        println!("| {name} | {c:.2}（{:.2} 倍） | fp16 換算 {:.2}（{tt:.0} ms） |", c / alone, gbs(total, tt));
    }
    unsafe { ffi::cudaFree(work.cast()) };
}

/// 固定した RAM を取る・返す時間（起動のときの手間）
fn pin(a: &[String]) {
    let gb: usize = arg(a, "--gb", "16").parse().unwrap();
    let node = kv::gpu_numa_node(kv::current_device());
    for (name, how) in [("mmap→mbind→cudaHostRegister", 0), ("cudaHostAlloc", 1)] {
        let t0 = Instant::now();
        let h = if how == 0 { HostArena::new(gb << 30, node) } else { HostArena::cuda_alloc(gb << 30) };
        let ta = t0.elapsed().as_secs_f64();
        let hist = h.node_histogram(256);
        let t0 = Instant::now();
        drop(h);
        let tf = t0.elapsed().as_secs_f64();
        println!("{name} {gb}GB: 取る {ta:.2} 秒（{:.1} GB/s）・返す {tf:.2} 秒・node {hist:?}", gb as f64 / ta);
    }
}

fn hs_buf(_h: &kv::Hasher) -> *mut u64 {
    // 指紋の置き場（測りでは中身を見ない）
    thread_local!(static BUF: *mut u8 = dmalloc(4096 * 8));
    BUF.with(|b| (*b).cast())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dev: i32 = std::env::var("INFER_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    ck(unsafe { tachyon_v100::cuda::ffi::cudaSetDevice(dev) }, "cudaSetDevice");
    match a.get(1).map(|s| s.as_str()) {
        Some("pcie") => pcie(&a),
        Some("tier") => tier(&a),
        Some("bulk") => bulk(&a),
        Some("check") => check(&a),
        Some("quant") => quant(&a),
        Some("blocks") => blocks(&a),
        Some("pin") => pin(&a),
        Some("interfere") => interfere(&a),
        _ => eprintln!("kvbench pcie|tier|bulk|check|quant|blocks|pin|interfere（先頭の説明を見る）"),
    }
}
