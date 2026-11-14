//! 普通の Attention のカーネルの確かめと速さ（kernels/attn.cu）。答え合わせは tools/attn_ref.py（numpy）。
//!
//!   attncheck case <dir>
//!       dir/case.json と qf（新しい分だけ）kf vf（全トークン）qnw knw（.f32）・pos.i32 を読み、列ごとに「前の分（KV に書くだけ）」と
//!       「新しい分（KV に書いて Attention）」の 2 回に分けて rope → attn を走らせ、
//!       q.f32（新しい分の rope 後の q）・out.f32・kc.bin・vc.bin（int8 なら ks.f32・vs.f32 も）を書く
//!   attncheck bench [prefill|decode|verify|all|mixed] [kv8]
//!       prefill T=547・8192・32768、decode B=1・8・32 × 文脈 1K・8K・32K、verify（MTP の 3 トークン・DSpine の 17 トークン）の時間。
//!       mixed は長さのばらばらな列を混ぜて、split を max_kv で決める（前）と列の長さで決める（kv_len_host）を比べる
//!   attncheck prof <prefill T | decode B ctx> [kv8]   … 1 回だけ（ncu 用）
//!   attncheck twogpu [kv8]                             … 見えている GPU 全部で同じ形を回し、どの GPU でも動き同じ答えか
use tachyon_v100::attn::{self, AttnParams, RopeParams, Shape, D, PAGE};
use tachyon_v100::cuda::{self, ffi, DevBuf};
use serde_json::Value;
use std::ffi::c_void;
use std::ptr::null_mut;

fn read_bytes(p: &str) -> Vec<u8> {
    std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"))
}
fn read_f32(p: &str) -> Vec<f32> {
    read_bytes(p).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn read_i32(p: &str) -> Vec<i32> {
    read_bytes(p).chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn write_f32(p: &str, v: &[f32]) {
    std::fs::write(p, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}
fn download(b: &DevBuf) -> Vec<u8> {
    let mut v = vec![0u8; b.len];
    cuda::check(unsafe { ffi::cudaMemcpy(v.as_mut_ptr().cast(), b.ptr, b.len, 2) }, "cudaMemcpy D2H");
    v
}
fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            // 非正規化数
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            s | ((e2 as u32) << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | (((e - 15 + 127) as u32) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}
fn f16s_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()
}
fn ok(r: Result<(), String>) {
    if let Err(e) = r {
        panic!("{e}");
    }
}
fn set_device() {
    let dev: i32 = std::env::var("INFER_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    cuda::check(unsafe { ffi::cudaSetDevice(dev) }, "cudaSetDevice");
}

/// KV の置き場（1 層分）
struct Kv {
    kc: DevBuf,
    vc: DevBuf,
    ks: Option<DevBuf>,
    vs: Option<DevBuf>,
}

impl Kv {
    fn new(sh: &Shape, n_pages: usize, kv8: bool) -> Kv {
        let pb = sh.page_bytes(kv8) * n_pages;
        let sc = || if kv8 { Some(DevBuf::new(sh.page_scales() * n_pages * 4)) } else { None };
        Kv { kc: DevBuf::new(pb), vc: DevBuf::new(pb), ks: sc(), vs: sc() }
    }
    fn set_rope(&self, p: &mut RopeParams) {
        p.kc = self.kc.ptr;
        p.vc = self.vc.ptr;
        p.ks = self.ks.as_ref().map_or(null_mut(), |b| b.f32p());
        p.vs = self.vs.as_ref().map_or(null_mut(), |b| b.f32p());
        p.kv8 = self.ks.is_some() as i32;
    }
    fn set_attn(&self, p: &mut AttnParams) {
        p.kc = self.kc.ptr;
        p.vc = self.vc.ptr;
        p.ks = self.ks.as_ref().map_or(std::ptr::null(), |b| b.f32p());
        p.vs = self.vs.as_ref().map_or(std::ptr::null(), |b| b.f32p());
        p.kv8 = self.ks.is_some() as i32;
    }
}

/// attncheck case <dir>
fn run_case(dir: &str) {
    let j: Value = serde_json::from_slice(&read_bytes(&format!("{dir}/case.json"))).unwrap();
    let gi = |k: &str| j[k].as_i64().unwrap_or_else(|| panic!("case.json に {k} が無い"));
    let sh = Shape {
        h: gi("h") as usize,
        hkv: gi("hkv") as usize,
        n_rot: gi("n_rot") as usize,
        sections: {
            let s: Vec<i32> = j["sections"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect();
            [s[0], s[1], s[2], s[3]]
        },
        theta: j["theta"].as_f64().unwrap() as f32,
        eps: j["eps"].as_f64().unwrap() as f32,
    };
    let kv8 = gi("kv8") != 0;
    let out_f16 = gi("out_f16") != 0;
    let pos_mode = gi("pos_mode") as usize;
    let n_pages = gi("n_pages") as usize;
    let seqs = j["seqs"].as_array().unwrap();
    let (qw, kw, vw) = (sh.h * 2 * D, sh.hkv * D, sh.hkv * D);
    let qf = read_f32(&format!("{dir}/qf.f32"));
    let kf = read_f32(&format!("{dir}/kf.f32"));
    let vf = read_f32(&format!("{dir}/vf.f32"));
    let pos = read_i32(&format!("{dir}/pos.i32"));
    let qnw = DevBuf::from_f32(&read_f32(&format!("{dir}/qnw.f32")));
    let knw = DevBuf::from_f32(&read_f32(&format!("{dir}/knw.f32")));

    // 前の分と新しい分に分けて詰める
    struct Batch {
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        pos: Vec<i32>,
        slot: Vec<i32>,
    }
    let mut pre = Batch { q: vec![], k: vec![], v: vec![], pos: vec![], slot: vec![] };
    let mut new = Batch { q: vec![], k: vec![], v: vec![], pos: vec![], slot: vec![] };
    let (mut bt, mut q_start, mut kv_len) = (vec![], vec![0i32], vec![]);
    let bt_ld = seqs.iter().map(|s| s["pages"].as_array().unwrap().len()).max().unwrap();
    let (mut t0, mut tq, mut max_q, mut max_kv) = (0usize, 0usize, 0usize, 0usize);
    for s in seqs {
        let l = s["kv_len"].as_u64().unwrap() as usize;
        let ql = s["q_len"].as_u64().unwrap() as usize;
        let pages: Vec<i32> = s["pages"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect();
        assert!(pages.len() * PAGE >= l && ql <= l && ql > 0);
        for p in 0..l {
            let t = t0 + p;
            let b = if p < l - ql { &mut pre } else { &mut new };
            if p >= l - ql {
                b.q.extend_from_slice(&qf[tq * qw..(tq + 1) * qw]);
                tq += 1;
            }
            b.k.extend_from_slice(&kf[t * kw..(t + 1) * kw]);
            b.v.extend_from_slice(&vf[t * vw..(t + 1) * vw]);
            b.pos.extend_from_slice(&pos[t * pos_mode..(t + 1) * pos_mode]);
            b.slot.push(pages[p / PAGE] * PAGE as i32 + (p % PAGE) as i32);
        }
        t0 += l;
        let mut row = pages.clone();
        row.resize(bt_ld, 0);
        bt.extend(row);
        q_start.push(q_start.last().unwrap() + ql as i32);
        kv_len.push(l as i32);
        max_q = max_q.max(ql);
        max_kv = max_kv.max(l);
    }
    let kv = Kv::new(&sh, n_pages, kv8);
    // ページのごみ（読まれないはずの所）を NaN にしておく
    for b in [&kv.kc, &kv.vc] {
        cuda::check(unsafe { cudaMemset(b.ptr, 0xff, b.len) }, "memset");
    }
    let n_new = new.slot.len();
    let q_new = DevBuf::new(n_new * sh.h * D * 2);
    let run_rope = |b: &Batch, q_out: &DevBuf| -> Vec<DevBuf> {
        let n = b.slot.len();
        if n == 0 {
            return vec![];
        }
        let bufs = vec![
            DevBuf::from_f32(if b.q.is_empty() { &[0.0] } else { &b.q }),
            DevBuf::from_f32(&b.k),
            DevBuf::from_f32(&b.v),
            DevBuf::from_i32(&b.pos),
            DevBuf::from_i32(&b.slot),
        ];
        let mut p = RopeParams::new(&sh);
        p.q = if b.q.is_empty() { std::ptr::null() } else { bufs[0].f32p() };
        p.k = bufs[1].f32p();
        p.v = bufs[2].f32p();
        p.pos = bufs[3].ptr.cast();
        p.slot = bufs[4].ptr.cast();
        p.q_norm = qnw.f32p();
        p.k_norm = knw.f32p();
        p.q_out = q_out.ptr;
        p.n = n as i32;
        p.pos_mode = pos_mode as i32;
        kv.set_rope(&mut p);
        ok(attn::rope(&p, null_mut()));
        bufs
    };
    let q_pre = DevBuf::new(pre.slot.len().max(1) * sh.h * D * 2);
    run_rope(&pre, &q_pre);
    let nb = run_rope(&new, &q_new);
    cuda::sync();

    let d_bt = DevBuf::from_i32(&bt);
    let d_qs = DevBuf::from_i32(&q_start);
    let d_kl = DevBuf::from_i32(&kv_len);
    let out = DevBuf::new(n_new * sh.h * D * if out_f16 { 2 } else { 4 });
    let mut p = AttnParams::new(&sh);
    p.q = q_new.ptr;
    if gi("gate") != 0 {
        p.gate = unsafe { nb[0].f32p().add(D) };
    }
    kv.set_attn(&mut p);
    p.block_table = d_bt.ptr.cast();
    p.bt_ld = bt_ld as i32;
    p.q_start = d_qs.ptr.cast();
    p.kv_len = d_kl.ptr.cast();
    p.n_seq = seqs.len() as i32;
    p.n_tok = n_new as i32;
    p.max_q_len = max_q as i32;
    p.max_kv = max_kv as i32;
    p.out = out.ptr;
    p.out_f16 = out_f16 as i32;
    p.kv_chunk = gi("kv_chunk") as i32;
    // 列の長さの CPU の写し（自動の split を列ごとの長さで決める）。host_lens 0 なら渡さない（全部 max_kv とみなす前の決め方）
    if j["host_lens"].as_i64().unwrap_or(1) != 0 {
        p.kv_len_host = kv_len.as_ptr();
    }
    let wsb = attn::ws_bytes(&p);
    let ws = DevBuf::new(wsb);
    p.ws = ws.f32p();
    p.ws_bytes = wsb as i64;
    ok(attn::attn(&p, null_mut()));
    cuda::sync();
    eprintln!(
        "case {dir}: 列 {} 新しいトークン {n_new} max_q {max_q} max_kv {max_kv} split の幅 {}{} 作業場 {wsb}",
        seqs.len(),
        attn::chunk(&p),
        if p.kv_len_host.is_null() { "（長さは max_kv）" } else { "" }
    );

    write_f32(&format!("{dir}/q.f32"), &f16s_to_f32(&download(&q_new)));
    let o = download(&out);
    write_f32(&format!("{dir}/out.f32"), &if out_f16 { f16s_to_f32(&o) } else { o.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() });
    std::fs::write(format!("{dir}/kc.bin"), download(&kv.kc)).unwrap();
    std::fs::write(format!("{dir}/vc.bin"), download(&kv.vc)).unwrap();
    if kv8 {
        std::fs::write(format!("{dir}/ks.f32"), download(kv.ks.as_ref().unwrap())).unwrap();
        std::fs::write(format!("{dir}/vs.f32"), download(kv.vs.as_ref().unwrap())).unwrap();
    }
}

extern "C" {
    fn cudaMemset(p: *mut c_void, v: i32, n: usize) -> i32;
    fn cudaGetDeviceCount(n: *mut i32) -> i32;
}

/// でたらめな置き場で、列 b のページは b*npp.. を混ぜた物理ページ
fn shuffled_pages(n_seq: usize, npp: usize) -> Vec<i32> {
    let total = n_seq * npp;
    let mut perm: Vec<i32> = (0..total as i32).collect();
    let mut x: u64 = 0x9e3779b97f4a7c15;
    for i in (1..total).rev() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        perm.swap(i, (x % (i as u64 + 1)) as usize);
    }
    perm
}

/// reps 回ずつ 5 組測り、いちばん速い組の 1 回あたり（ms）。GPU を他と分け合っていても揺れにくい
fn best_ms(reps: usize, mut f: impl FnMut()) -> f32 {
    (0..5).map(|_| cuda::time_ms(|| (0..reps).for_each(|_| f())) / reps as f32).fold(f32::INFINITY, f32::min)
}

struct Bench {
    sh: Shape,
    kv8: bool,
}

impl Bench {
    /// n_seq 列 × q_len、文脈 ctx（書いた後の長さ）。でたらめな q・gate・KV で attn の時間（ms）と split の幅
    fn attn(&self, n_seq: usize, q_len: usize, ctx: usize, reps: usize, prof: bool) -> (f32, usize) {
        let (ms, ch, _) = self.attn_lens(&vec![ctx; n_seq], q_len, reps, prof, true);
        (ms, ch)
    }

    /// 列ごとの長さ lens（書いた後の長さ）× q_len。host なら kv_len_host を渡す（自動の split を列の長さで決める）。
    /// 返り値は (時間 ms, split の幅, 出力)。入力は種で決まるので、同じ形なら何度・どの GPU でも同じ入力
    fn attn_lens(&self, lens: &[usize], q_len: usize, reps: usize, prof: bool, host: bool) -> (f32, usize, Vec<f32>) {
        let sh = &self.sh;
        let n_seq = lens.len();
        let max_kv = *lens.iter().max().unwrap();
        let npp = max_kv.div_ceil(PAGE);
        let total: usize = lens.iter().map(|l| l.div_ceil(PAGE)).sum();
        let kv = Kv::new(sh, total, self.kv8);
        let n_tok = n_seq * q_len;
        let q = DevBuf::new(n_tok * sh.h * D * 2);
        let gate = DevBuf::new(n_tok * sh.h * 2 * D * 4);
        let out = DevBuf::new(n_tok * sh.h * D * 4);
        let el = |b: &DevBuf, s: usize| b.len / s;
        if self.kv8 {
            ok(unsafe { attn::fill(kv.kc.ptr, el(&kv.kc, 1), 1, 1, 1.0) }); // SAFETY: 書き先はこの直前に取った GPU の置き場で、要素の数はその大きさから数えた
            ok(unsafe { attn::fill(kv.vc.ptr, el(&kv.vc, 1), 1, 2, 1.0) }); // SAFETY: 書き先はこの直前に取った GPU の置き場で、要素の数はその大きさから数えた
            let (ks, vs) = (kv.ks.as_ref().unwrap(), kv.vs.as_ref().unwrap());
            ok(unsafe { attn::fill(ks.ptr, el(ks, 4), 2, 3, 0.01) }); // SAFETY: 書き先はこの直前に取った GPU の置き場で、要素の数はその大きさから数えた
            ok(unsafe { attn::fill(vs.ptr, el(vs, 4), 2, 4, 0.01) }); // SAFETY: 書き先はこの直前に取った GPU の置き場で、要素の数はその大きさから数えた
        } else {
            ok(unsafe { attn::fill(kv.kc.ptr, el(&kv.kc, 2), 0, 1, 1.0) }); // SAFETY: 書き先はこの直前に取った GPU の置き場で、要素の数はその大きさから数えた
            ok(unsafe { attn::fill(kv.vc.ptr, el(&kv.vc, 2), 0, 2, 1.0) }); // SAFETY: 書き先はこの直前に取った GPU の置き場で、要素の数はその大きさから数えた
        }
        ok(unsafe { attn::fill(q.ptr, el(&q, 2), 0, 5, 1.0) }); // SAFETY: 書き先はこの直前に取った GPU の置き場で、要素の数はその大きさから数えた
        ok(unsafe { attn::fill(gate.ptr, el(&gate, 4), 2, 6, 2.0) }); // SAFETY: 書き先はこの直前に取った GPU の置き場で、要素の数はその大きさから数えた
        // 列 i のページは物理ページを混ぜた並びから順に（block table の行の幅は最長の列）
        let perm = shuffled_pages(1, total);
        let mut bt = vec![0i32; n_seq * npp];
        let mut o = 0;
        for (i, l) in lens.iter().enumerate() {
            for pg in 0..l.div_ceil(PAGE) {
                bt[i * npp + pg] = perm[o];
                o += 1;
            }
        }
        let bt = DevBuf::from_i32(&bt);
        let qs: Vec<i32> = (0..=n_seq).map(|i| (i * q_len) as i32).collect();
        let d_qs = DevBuf::from_i32(&qs);
        let kl: Vec<i32> = lens.iter().map(|&l| l as i32).collect();
        let d_kl = DevBuf::from_i32(&kl);
        let mut p = AttnParams::new(sh);
        p.q = q.ptr;
        p.gate = unsafe { gate.f32p().add(D) };
        kv.set_attn(&mut p);
        p.block_table = bt.ptr.cast();
        p.bt_ld = npp as i32;
        p.q_start = d_qs.ptr.cast();
        p.kv_len = d_kl.ptr.cast();
        p.n_seq = n_seq as i32;
        p.n_tok = n_tok as i32;
        p.max_q_len = q_len as i32;
        p.max_kv = max_kv as i32;
        p.out = out.ptr;
        if host {
            p.kv_len_host = kl.as_ptr();
        }
        if let Some(c) = std::env::var("ATTN_CHUNK").ok().and_then(|s| s.parse().ok()) {
            p.kv_chunk = c;  // split の幅を決め打ち（調べる用）
        }
        let wsb = attn::ws_bytes(&p);
        let ws = DevBuf::new(wsb);
        p.ws = ws.f32p();
        p.ws_bytes = wsb as i64;
        ok(attn::attn(&p, null_mut()));
        cuda::sync();
        if prof {
            return (0.0, attn::chunk(&p), vec![]);
        }
        let first = DevBuf::to_f32(&out);
        let ms = if reps > 0 { best_ms(reps, || ok(attn::attn(&p, null_mut()))) } else { 0.0 };
        assert!(first.iter().all(|x| x.is_finite()), "出力に NaN・inf");
        (ms, attn::chunk(&p), first)
    }

    /// プレフィル T トークンの rope（q・k・v を書く）の時間（ms）
    fn rope(&self, t: usize, reps: usize) -> f32 {
        let sh = &self.sh;
        let kv = Kv::new(sh, t.div_ceil(PAGE), self.kv8);
        let qf = DevBuf::new(t * sh.h * 2 * D * 4);
        let kf = DevBuf::new(t * sh.hkv * D * 4);
        let vf = DevBuf::new(t * sh.hkv * D * 4);
        for (i, b) in [&qf, &kf, &vf].iter().enumerate() {
            ok(unsafe { attn::fill(b.ptr, b.len / 4, 2, 10 + i as u32, 3.0) }); // SAFETY: 書き先はこの直前に取った GPU の置き場で、要素の数はその大きさから数えた
        }
        let w = DevBuf::from_f32(&vec![1.0f32; D]);
        let pos: Vec<i32> = (0..t as i32).collect();
        let d_pos = DevBuf::from_i32(&pos);
        let d_slot = DevBuf::from_i32(&pos);
        let q_out = DevBuf::new(t * sh.h * D * 2);
        let mut p = RopeParams::new(sh);
        p.q = qf.f32p();
        p.k = kf.f32p();
        p.v = vf.f32p();
        p.q_norm = w.f32p();
        p.k_norm = w.f32p();
        p.pos = d_pos.ptr.cast();
        p.slot = d_slot.ptr.cast();
        p.q_out = q_out.ptr;
        p.n = t as i32;
        kv.set_rope(&mut p);
        ok(attn::rope(&p, null_mut()));
        cuda::sync();
        best_ms(reps, || ok(attn::rope(&p, null_mut())))
    }
}

/// 長さのばらばらな列を混ぜたデコード・確かめ: split を max_kv だけで決める（前の決め方）と、列の長さで決める（kv_len_host）の比べ
fn bench_mixed(b: &Bench) {
    let sh = &b.sh;
    let kvb = if b.kv8 { 1.0 + 4.0 / 32.0 } else { 2.0 };
    let mut x: u64 = 0x2545f4914f6cdd1d;
    let mut rnd = |lo: usize, hi: usize| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        lo + (x % (hi - lo) as u64) as usize
    };
    let r16: Vec<usize> = (0..16).map(|_| rnd(1024, 32768)).collect();
    let cases: Vec<(&str, Vec<usize>, usize)> = vec![
        ("B=8 32K+1K×7", [vec![32768], vec![1024; 7]].concat(), 1),
        ("B=8 32K,16K,…,256", vec![32768, 16384, 8192, 4096, 2048, 1024, 512, 256], 1),
        ("B=16 1K〜32K でたらめ", r16, 1),
        ("B=32 32K+2K×31", [vec![32768], vec![2048; 31]].concat(), 1),
        ("B=48 32K+1K×47", [vec![32768], vec![1024; 47]].concat(), 1),
        ("B=64 128K+1K×63", [vec![131072], vec![1024; 63]].concat(), 1),
        ("B=32 同じ 8K（変わらないはず）", vec![8192; 32], 1),
        ("verify3 B=32 32K+1K×31", [vec![32768], vec![1024; 31]].concat(), 3),
        ("verify17 B=8 32K+1K×7", [vec![32768], vec![1024; 7]].concat(), 17),
        ("verify17 B=4 同じ 8K（変わらないはず）", vec![8192; 4], 17),
        ("verify17 B=32 32K+1K×31", [vec![32768], vec![1024; 31]].concat(), 17),
    ];
    for (name, lens, ql) in cases {
        let reps = if lens.iter().sum::<usize>() > 300_000 { 5 } else { 20 };
        let (t0, c0, o0) = b.attn_lens(&lens, ql, reps, false, false);
        let (t1, c1, o1) = b.attn_lens(&lens, ql, reps, false, true);
        let bytes = (lens.iter().sum::<usize>() * sh.hkv * D * 2) as f64 * kvb;
        // split の切れ目が違うだけなので、答えは足す順と P（fp16）の丸めの違いの分しか変わらない（手本との合格の線は 2e-3）
        let mx = o0.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
        let diff = o0.iter().zip(&o1).fold(0f32, |m, (a, c)| m.max((a - c).abs())) / mx;
        println!(
            "mixed {name:28} q_len={ql:2}: 前（max_kv で）{:8.1} µs split {c0:5} → 列の長さで {:8.1} µs split {c1:5}（{:.2} 倍、{:4.0} GB/s、答えの差 {diff:.1e}）",
            t0 * 1e3,
            t1 * 1e3,
            t0 / t1,
            bytes / (t1 as f64 * 1e-3) / 1e9
        );
        assert!(diff < 1e-3, "{name}: split の決め方で答えが変わった（{diff:.1e}）");
    }
}

/// 見えている GPU 全部（CUDA_VISIBLE_DEVICES=5,7 など 2 枚以上）で同じ形を 1 回ずつ: プレフィル（CfgP・共有メモリ 86KB の許しが要る）・
/// 確かめ（CfgV）・デコード（CfgD・split）。どの GPU でも動き、答えがビットまで同じか（許し・SM の数は GPU ごとに取る）
fn two_gpu(kv8: bool) -> bool {
    let mut n = 0;
    cuda::check(unsafe { cudaGetDeviceCount(&mut n) }, "cudaGetDeviceCount");
    if n < 2 {
        eprintln!("GPU が {n} 枚しか見えない（CUDA_VISIBLE_DEVICES=5,7 のように 2 枚見せる）");
        return false;
    }
    let shapes: [(&str, Vec<usize>, usize); 3] =
        [("prefill T=547", vec![547], 547), ("verify3 B=8", vec![3000; 8], 3), ("decode B=4 混ぜた長さ", vec![9000, 64, 1500, 300], 1)];
    let mut first: Vec<Vec<f32>> = vec![];
    let mut good = true;
    // 0 → 1 → … → 0 の順（最初の GPU に戻っても動くか）
    for (round, dev) in (0..n).chain([0]).enumerate() {
        cuda::check(unsafe { ffi::cudaSetDevice(dev) }, "cudaSetDevice");
        let b = Bench { sh: Shape::QWEN38_27B, kv8 };
        for (i, (name, lens, ql)) in shapes.iter().enumerate() {
            let (_, ch, o) = b.attn_lens(lens, *ql, 0, false, true);
            let same = if round == 0 {
                first.push(o);
                true
            } else {
                first[i] == o
            };
            good &= same;
            println!("GPU {dev}（{} 回目）{name:24} split {ch:5}: {}", round + 1, if same { "OK（最初の GPU とビットまで同じ）" } else { "NG（答えが違う）" });
        }
    }
    good
}

fn bench(what: &str, kv8: bool) {
    let b = Bench { sh: Shape::QWEN38_27B, kv8 };
    let sh = &b.sh;
    let kvb = if kv8 { 1.0 + 4.0 / 32.0 } else { 2.0 };  // 1 要素のバイト（int8 は 32 要素ごとの係数込み）
    eprintln!("KV {}", if kv8 { "int8（32 要素ごとの係数）" } else { "fp16" });
    if what == "mixed" {
        return bench_mixed(&b);
    }
    if what == "prefill" || what == "all" {
        for t in [547usize, 8192, 32768] {
            let reps = if t > 8192 { 1 } else if t > 547 { 3 } else { 20 };
            let (ms, ch) = b.attn(1, t, t, reps, false);
            let fl = 4.0 * (sh.h * D) as f64 * (t * (t + 1) / 2) as f64;
            let rms = b.rope(t, reps);
            println!(
                "prefill T={t:6}: attn {ms:8.3} ms ({:5.1} TFLOPS, split {ch})  rope+KV 書き込み {rms:.3} ms  → 16 層で {:.1} ms",
                fl / (ms as f64 * 1e-3) / 1e12,
                16.0 * (ms + rms)
            );
        }
    }
    let qls: &[usize] = match what {
        "decode" => &[1],
        "verify" => &[3, 17],
        "all" => &[1, 3, 17],
        _ => &[],
    };
    for &ql in qls {
        for bsz in [1usize, 8, 32] {
            for ctx in [1024usize, 8192, 32768] {
                let (ms, ch) = b.attn(bsz, ql, ctx, if bsz * ctx > 100_000 { 5 } else { 50 }, false);
                let bytes = (bsz * ctx * sh.hkv * D * 2) as f64 * kvb;
                println!(
                    "{} B={bsz:2} q_len={ql} ctx={ctx:5}: {:8.1} µs  KV {:7.1} MB  {:6.0} GB/s  (split {ch})",
                    if ql == 1 { "decode" } else { "verify" },
                    ms * 1e3,
                    bytes / 1e6,
                    bytes / (ms as f64 * 1e-3) / 1e9
                );
            }
        }
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    set_device();
    let kv8 = a.iter().any(|s| s == "kv8");
    match a.get(1).map(|s| s.as_str()) {
        Some("case") => run_case(&a[2]),
        Some("bench") => bench(a.get(2).map_or("all", |s| s.as_str()), kv8),
        Some("twogpu") => std::process::exit(if two_gpu(kv8) { 0 } else { 1 }),
        Some("prof") => {
            let b = Bench { sh: Shape::QWEN38_27B, kv8 };
            let n = |i: usize| a[i].parse::<usize>().unwrap();
            match a[2].as_str() {
                "prefill" => b.attn(1, n(3), n(3), 1, true),
                "decode" => b.attn(n(3), 1, n(4), 1, true),
                m => panic!("知らない {m}"),
            };
        }
        _ => eprintln!("attncheck case <dir> | bench [prefill|decode|verify|all|mixed] [kv8] | prof <prefill T|decode B ctx> [kv8] | twogpu [kv8]"),
    }
}
