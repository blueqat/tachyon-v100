//! Gated DeltaNet のカーネルの確かめと速さ（kernels/gdn.cu）。答え合わせは tools/gdncheck.py（手本は tools/gdn_ref.py）。
//!
//!   gdncheck run <dir> <decode|steps|seq|prefill|split> <B> <T> [n]
//!       dir の qkv z alpha beta a dt norm conv conv_s state（.f32。slots.i32 は任意）を読み、
//!       y c conv_s_out state_out（.f32）を書く。
//!       decode = conv＋decode を 1 回（T = 1）、steps = それを T 回（1 トークンずつ別々に呼ぶ）、
//!       seq = conv(T)＋decode(T) を 1 回ずつ（1 つのカーネルの中で 1 トークンずつ）、prefill = conv(T)＋塊ごと、
//!       split = 前の T-n トークンを prefill、残りの n を steps（prefill の後に decode が続くか）、
//!       verify = 下書きの確かめ: T トークンを decode(T) で（新しい状態は別の slot へ）→ n_acc.i32 の数だけ元の slot で確定
//!                （state_full・conv_full = T トークン全部を進めた状態も書く）
//!   gdncheck bench   … decode B=1,8,32（T=1 と、下書きの確かめ用に T=17）、prefill（B=1 T=547・2048・8192、B=4 T=547）の時間
//!   gdncheck prof <B> <T> … conv＋prefill を 1 回（ncu・nsys 用）
use tachyon_v100::cuda::{self, ffi, DevBuf};
use tachyon_v100::gdn::{self, Io, Layer, Shape, D};
use std::ptr::{null, null_mut};

fn read_f32(p: &str) -> Vec<f32> {
    std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}")).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn write_f32(p: &str, v: &[f32]) {
    std::fs::write(p, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}

struct Bufs {
    qkv: DevBuf,
    c: DevBuf,
    z: DevBuf,
    alpha: DevBuf,
    beta: DevBuf,
    y: DevBuf,
    state: DevBuf,
    conv_s: DevBuf,
    slots: Option<DevBuf>,
    a: DevBuf,
    dt: DevBuf,
    norm: DevBuf,
    conv: DevBuf,
}

impl Bufs {
    fn layer(&self) -> Layer {
        Layer { a: self.a.f32p(), dt: self.dt.f32p(), norm: self.norm.f32p(), conv: self.conv.f32p() }
    }
    /// 全体（B 本 × T トークン）をそのまま渡す Io
    fn io(&self, sh: &Shape) -> Io {
        Io {
            qkv: self.qkv.f32p(), qkv_ld: sh.conv_ch(), c: self.c.f32p(), z: self.z.f32p(), z_ld: sh.out_dim(),
            alpha: self.alpha.f32p(), beta: self.beta.f32p(), ab_ld: sh.hv, y: self.y.f32p(), y_ld: sh.out_dim(),
            state: self.state.f32p(), conv_state: self.conv_s.f32p(),
            slots: self.slots.as_ref().map_or(null(), |s| s.ptr.cast()), slots_out: null(), n_valid: null(),
        }
    }
}

/// トークン t だけを見る Io（行の幅を T 行分にして、B 本 × 1 トークンに見せる）。c は [B][C] の別の置き場
fn io_step(sh: &Shape, bf: &Bufs, t_all: usize, t: usize, c_step: &DevBuf) -> Io {
    let (c, o, h) = (sh.conv_ch(), sh.out_dim(), sh.hv);
    unsafe {
        Io {
            qkv: bf.qkv.f32p().add(t * c), qkv_ld: t_all * c, c: c_step.f32p(), z: bf.z.f32p().add(t * o), z_ld: t_all * o,
            alpha: bf.alpha.f32p().add(t * h), beta: bf.beta.f32p().add(t * h), ab_ld: t_all * h,
            y: bf.y.f32p().add(t * o), y_ld: t_all * o, state: bf.state.f32p(), conv_state: bf.conv_s.f32p(),
            slots: bf.slots.as_ref().map_or(null(), |s| s.ptr.cast()), slots_out: null(), n_valid: null(),
        }
    }
}

fn run(dir: &str, mode: &str, b: usize, t: usize, n_steps: usize) {
    let sh = Shape::Q38_27B;
    let f = |n: &str| read_f32(&format!("{dir}/{n}.f32"));
    let slots = std::fs::read(format!("{dir}/slots.i32")).ok().map(|v| {
        let s: Vec<i32> = v.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
        DevBuf::from_i32(&s)
    });
    let mut bf = Bufs {
        qkv: DevBuf::from_f32(&f("qkv")), c: DevBuf::new(b * t * sh.conv_ch() * 4), z: DevBuf::from_f32(&f("z")),
        alpha: DevBuf::from_f32(&f("alpha")), beta: DevBuf::from_f32(&f("beta")), y: DevBuf::new(b * t * sh.out_dim() * 4),
        state: DevBuf::from_f32(&f("state")), conv_s: DevBuf::from_f32(&f("conv_s")), slots,
        a: DevBuf::from_f32(&f("a")), dt: DevBuf::from_f32(&f("dt")), norm: DevBuf::from_f32(&f("norm")),
        conv: DevBuf::from_f32(&f("conv")),
    };
    assert_eq!(bf.qkv.len, b * t * sh.conv_ch() * 4, "qkv の大きさ");
    let l = bf.layer();
    let s = null_mut();
    match mode {
        "decode" | "seq" => {
            assert!(mode == "seq" || t == 1, "decode は T = 1");
            let io = bf.io(&sh);
            gdn::conv(&sh, &l, &io, b, t, s);
            gdn::decode(&sh, &l, &io, b, t, s);
        }
        "steps" => {
            let c_step = DevBuf::new(b * sh.conv_ch() * 4);
            for i in 0..t {
                let io = io_step(&sh, &bf, t, i, &c_step);
                gdn::conv(&sh, &l, &io, b, 1, s);
                gdn::decode(&sh, &l, &io, b, 1, s);
            }
        }
        "verify" => {
            // 元の状態（slot b）は残し、T トークン全部を進めた状態は slot B+b へ（slots_out）。
            // そのあと受け入れた n_acc[b] トークンだけを slot b で確定する（conv の状態・線形状態とも。出口は書かない）
            assert!(bf.slots.is_none(), "verify は slots なし");
            let n_acc: Vec<i32> =
                std::fs::read(format!("{dir}/n_acc.i32")).unwrap().chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
            assert_eq!(n_acc.len(), b);
            let dup = |mut v: Vec<f32>| {
                let n = v.len();
                v.resize(2 * n, 0.0);
                v
            };
            bf.state = DevBuf::from_f32(&dup(f("state")));
            bf.conv_s = DevBuf::from_f32(&dup(f("conv_s")));
            let scratch: Vec<i32> = (b as i32..2 * b as i32).collect();
            let (sc, na) = (DevBuf::from_i32(&scratch), DevBuf::from_i32(&n_acc));
            let mut io = bf.io(&sh);
            io.slots_out = sc.ptr.cast();
            gdn::conv(&sh, &l, &io, b, t, s);
            gdn::decode(&sh, &l, &io, b, t, s);
            let mut io2 = bf.io(&sh);
            io2.n_valid = na.ptr.cast();
            io2.y = null_mut();
            gdn::conv_state_only(&sh, &l, &io2, b, t, s);
            gdn::decode(&sh, &l, &io2, b, t, s);
            cuda::sync();
            let (st, cs) = (bf.state.to_f32(), bf.conv_s.to_f32());
            let (sn, cn) = (st.len() / 2, cs.len() / 2);
            write_f32(&format!("{dir}/y.f32"), &bf.y.to_f32());
            write_f32(&format!("{dir}/c.f32"), &bf.c.to_f32());
            write_f32(&format!("{dir}/state_out.f32"), &st[..sn]);
            write_f32(&format!("{dir}/conv_s_out.f32"), &cs[..cn]);
            write_f32(&format!("{dir}/state_full.f32"), &st[sn..]);
            write_f32(&format!("{dir}/conv_full.f32"), &cs[cn..]);
            return;
        }
        "split" => {
            // 前の tp トークンだけを詰め直した置き場で prefill（状態は bf のものを使う）→ 残りを 1 トークンずつ
            assert!(n_steps >= 1 && n_steps < t, "split は 1 ≤ n < T");
            let tp = t - n_steps;
            let head = |v: &[f32], w: usize| -> Vec<f32> {
                (0..b).flat_map(|i| v[i * t * w..(i * t + tp) * w].iter().copied()).collect()
            };
            let (c, o, h) = (sh.conv_ch(), sh.out_dim(), sh.hv);
            let (qkv_p, z_p) = (DevBuf::from_f32(&head(&f("qkv"), c)), DevBuf::from_f32(&head(&f("z"), o)));
            let (al_p, be_p) = (DevBuf::from_f32(&head(&f("alpha"), h)), DevBuf::from_f32(&head(&f("beta"), h)));
            let (c_p, y_p) = (DevBuf::new(b * tp * c * 4), DevBuf::new(b * tp * o * 4));
            let mut io = bf.io(&sh);
            io.qkv = qkv_p.f32p();
            io.c = c_p.f32p();
            io.z = z_p.f32p();
            io.alpha = al_p.f32p();
            io.beta = be_p.f32p();
            io.y = y_p.f32p();
            let ws = DevBuf::new(gdn::prefill_ws_bytes(&sh, b, tp));
            gdn::conv(&sh, &l, &io, b, tp, s);
            gdn::prefill(&sh, &l, &io, ws.ptr, b, tp, s);
            let c_step = DevBuf::new(b * c * 4);
            for i in tp..t {
                let io = io_step(&sh, &bf, t, i, &c_step);
                gdn::conv(&sh, &l, &io, b, 1, s);
                gdn::decode(&sh, &l, &io, b, 1, s);
            }
            cuda::sync();
            let (mut y, yp) = (bf.y.to_f32(), y_p.to_f32());
            for i in 0..b {
                y[i * t * o..(i * t + tp) * o].copy_from_slice(&yp[i * tp * o..(i + 1) * tp * o]);
            }
            write_f32(&format!("{dir}/y.f32"), &y);
            write_f32(&format!("{dir}/conv_s_out.f32"), &bf.conv_s.to_f32());
            write_f32(&format!("{dir}/state_out.f32"), &bf.state.to_f32());
            return;
        }
        "prefill" => {
            let io = bf.io(&sh);
            let ws = DevBuf::new(gdn::prefill_ws_bytes(&sh, b, t));
            gdn::conv(&sh, &l, &io, b, t, s);
            gdn::prefill(&sh, &l, &io, ws.ptr, b, t, s);
        }
        m => panic!("知らない動作 {m}"),
    }
    cuda::sync();
    write_f32(&format!("{dir}/y.f32"), &bf.y.to_f32());
    if mode != "steps" {
        write_f32(&format!("{dir}/c.f32"), &bf.c.to_f32());
    }
    write_f32(&format!("{dir}/conv_s_out.f32"), &bf.conv_s.to_f32());
    write_f32(&format!("{dir}/state_out.f32"), &bf.state.to_f32());
}

/// 速さを測る用のでたらめな値（xorshift。おおよそ [-1, 1)）
fn rand_vec(n: usize, seed: u64, scale: f32, bias: f32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * scale + bias
        })
        .collect()
}

fn bufs_rand(sh: &Shape, b: usize, t: usize) -> Bufs {
    let (c, o, h) = (sh.conv_ch(), sh.out_dim(), sh.hv);
    Bufs {
        qkv: DevBuf::from_f32(&rand_vec(b * t * c, 1, 1.0, 0.0)), c: DevBuf::new(b * t * c * 4),
        z: DevBuf::from_f32(&rand_vec(b * t * o, 2, 1.0, 0.0)), alpha: DevBuf::from_f32(&rand_vec(b * t * h, 3, 1.0, 0.0)),
        beta: DevBuf::from_f32(&rand_vec(b * t * h, 4, 1.0, 0.0)), y: DevBuf::new(b * t * o * 4),
        state: DevBuf::from_f32(&rand_vec(b * sh.state_len(), 5, 0.1, 0.0)),
        conv_s: DevBuf::from_f32(&rand_vec(b * sh.conv_state_len(), 6, 1.0, 0.0)), slots: None,
        a: DevBuf::from_f32(&rand_vec(h, 7, 2.0, -3.0)), dt: DevBuf::from_f32(&rand_vec(h, 8, 1.0, -2.0)),
        norm: DevBuf::from_f32(&rand_vec(D, 9, 0.1, 1.0)), conv: DevBuf::from_f32(&rand_vec(c * 4, 10, 0.1, 0.0)),
    }
}

/// 1 回あたりの時間（ms）。reps 回ずつ 5 組測って一番速い組（ほかの仕事が GPU に混ざったときの揺れを除く）
fn time(reps: usize, mut f: impl FnMut()) -> f32 {
    f();
    cuda::sync();
    (0..5).map(|_| cuda::time_ms(|| (0..reps).for_each(|_| f())) / reps as f32).fold(f32::INFINITY, f32::min)
}

fn bench() {
    let sh = Shape::Q38_27B;
    let s = null_mut();
    let st_mb = sh.state_len() as f64 * 4.0 / 1e6;
    eprintln!("線形状態 {:.2} MB/列（fp32）。1 層ぶんの時間（Qwen3.8-27B は線形層 48）", st_mb);
    for b in [1usize, 8, 32] {
        let bf = bufs_rand(&sh, b, 1);
        let (l, io) = (bf.layer(), bf.io(&sh));
        let tc = time(200, || gdn::conv(&sh, &l, &io, b, 1, s));
        let td = time(200, || gdn::decode(&sh, &l, &io, b, 1, s));
        let both = time(200, || {
            gdn::conv(&sh, &l, &io, b, 1, s);
            gdn::decode(&sh, &l, &io, b, 1, s);
        });
        let gbs = 2.0 * b as f64 * st_mb * 1e6 / (td as f64 * 1e-3) / 1e9;
        eprintln!(
            "decode B={b:2}: conv {:.4} ms、gdn {:.4} ms（状態の読み書き {gbs:.0} GB/s）、続けて {:.4} ms → 48 層で {:.2} ms",
            tc, td, both, both * 48.0
        );
    }
    // 下書きの確かめ（DSpine b16 なら 1+16 トークン）を 1 回で: decode(T) と prefill(T)
    for b in [1usize, 8, 32] {
        let t = 17;
        let bf = bufs_rand(&sh, b, t);
        let (l, io) = (bf.layer(), bf.io(&sh));
        let ws = DevBuf::new(gdn::prefill_ws_bytes(&sh, b, t));
        let tc = time(100, || gdn::conv(&sh, &l, &io, b, t, s));
        let td = time(100, || gdn::decode(&sh, &l, &io, b, t, s));
        let tp = time(100, || gdn::prefill(&sh, &l, &io, ws.ptr, b, t, s));
        eprintln!("verify B={b:2} T={t}: conv {:.4} ms、1 トークンずつ（seq）{:.4} ms、塊ごと {:.4} ms", tc, td, tp);
    }
    for (b, t) in [(1usize, 547usize), (4, 547), (1, 2048), (1, 8192)] {
        let bf = bufs_rand(&sh, b, t);
        let (l, io) = (bf.layer(), bf.io(&sh));
        let ws = DevBuf::new(gdn::prefill_ws_bytes(&sh, b, t));
        let reps = if b * t > 1000 { 4 } else { 20 };
        let tc = time(reps, || gdn::conv(&sh, &l, &io, b, t, s));
        let tp = time(reps, || gdn::prefill(&sh, &l, &io, ws.ptr, b, t, s));
        let tq = time(reps, || gdn::decode(&sh, &l, &io, b, t, s));
        eprintln!(
            "prefill B={b} T={t:4}: conv {:.3} ms、塊ごと {:.3} ms（{:.0} tok/s、48 層で {:.1} ms）、1 トークンずつ（seq）{:.3} ms、作業場 {:.0} MB",
            tc, tp, (b * t) as f32 / (tc + tp) * 1e3, (tc + tp) * 48.0, tq, gdn::prefill_ws_bytes(&sh, b, t) as f64 / 1e6
        );
    }
}

/// プロファイラ（ncu・nsys）用: conv＋prefill を 1 回だけ
fn prof(b: usize, t: usize) {
    let sh = Shape::Q38_27B;
    let bf = bufs_rand(&sh, b, t);
    let (l, io) = (bf.layer(), bf.io(&sh));
    let ws = DevBuf::new(gdn::prefill_ws_bytes(&sh, b, t));
    gdn::conv(&sh, &l, &io, b, t, null_mut());
    gdn::prefill(&sh, &l, &io, ws.ptr, b, t, null_mut());
    cuda::sync();
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dev: i32 = std::env::var("INFER_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    cuda::check(unsafe { ffi::cudaSetDevice(dev) }, "cudaSetDevice");
    match a.get(1).map(String::as_str) {
        Some("run") => run(&a[2], &a[3], a[4].parse().unwrap(), a[5].parse().unwrap(), a.get(6).map_or(0, |n| n.parse().unwrap())),
        Some("bench") => bench(),
        Some("prof") => prof(a[2].parse().unwrap(), a[3].parse().unwrap()),
        _ => panic!("使い方: gdncheck run <dir> <decode|steps|seq|prefill|split> <B> <T> [n] ／ gdncheck bench ／ gdncheck prof <B> <T>"),
    }
}
