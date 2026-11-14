//! デコード（M2）: B 本（1..32）の列を 1 語ずつ同じ歩で進める。1 歩を CUDA Graph に取って流し直す。
//!
//!   入力（語・位置・KV の書き先・頁表・GDN の状態の番号）は固定した RAM → GPU の決まった置き場に 1 回で写し、
//!   グラフの中のカーネルはそこを読む（グラフは B と KV の長さの幅（BUCKET 語ごと）で取り直す。幅で Attention の split が決まる）。
//!   1 層のカーネル（GDN の層 7 本・普通の層 9〜10 本）:
//!     RMSNorm → x の下ごしらえ（rmsnorm_xp）→ qkv・z・α・β の GEMV（qkv の出口で conv1d＋silu）→ GDN の 1 歩（l2 正規化・状態・
//!     出口の正規化 → ssm_out の下ごしらえ）→ ssm_out（残差に足す・決まった順の k 分け）→ RMSNorm → SiLU(gate)·up（down の下ごしらえ）→ down
//!     普通の層は q・k・v の GEMV → q/k の正規化と RoPE（表は 1 歩に 1 回）・KV へ → Attention → xprep → attn_output
//!   値は M1 の道（forward の T = 1）と同じ式。違うのは Attention の split の幅（KV の幅で決める）だけ
//!   どの列の答えも、同じ B・同じ幅なら毎回ビットまで同じ（足し込みは決まった順）
use super::*;
use crate::cuda::ffi::Stream;

/// グラフを取り直す KV の幅（語）。Attention の split と max_kv はこの幅の上の端で決める
pub const BUCKET: usize = 256;

mod dffi {
    use crate::cuda::ffi::Stream;
    use std::ffi::c_void;
    extern "C" {
        pub fn cudaStreamCreateWithFlags(s: *mut Stream, flags: u32) -> i32;
        pub fn cudaStreamCreateWithPriority(s: *mut Stream, flags: u32, prio: i32) -> i32;
        pub fn cudaEventCreateWithFlags(e: *mut *mut c_void, flags: u32) -> i32;
        pub fn cudaEventRecord(e: *mut c_void, s: Stream) -> i32;
        pub fn cudaEventSynchronize(e: *mut c_void) -> i32;
        pub fn cudaEventDestroy(e: *mut c_void) -> i32;
        pub fn cudaStreamWaitEvent(s: Stream, e: *mut c_void, flags: u32) -> i32;
        pub fn cudaStreamDestroy(s: Stream) -> i32;
        pub fn cudaStreamSynchronize(s: Stream) -> i32;
        pub fn cudaStreamBeginCapture(s: Stream, mode: i32) -> i32;
        pub fn cudaStreamEndCapture(s: Stream, g: *mut *mut c_void) -> i32;
        pub fn cudaGraphInstantiateWithFlags(e: *mut *mut c_void, g: *mut c_void, flags: u64) -> i32;
        pub fn cudaGraphLaunch(e: *mut c_void, s: Stream) -> i32;
        pub fn cudaGraphExecDestroy(e: *mut c_void) -> i32;
        pub fn cudaGraphDestroy(g: *mut c_void) -> i32;
        pub fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, n: usize, kind: i32, s: Stream) -> i32;
        pub fn cudaMallocHost(p: *mut *mut c_void, n: usize) -> i32;
        pub fn cudaFreeHost(p: *mut c_void) -> i32;
        pub fn cudaMemsetAsync(p: *mut c_void, v: i32, n: usize, s: Stream) -> i32;

        pub fn infer_m_rmsnorm_xp(
            x: *const f32, x_ld: i64, w: *const f32, k: i32, n_rows: i32, eps: f32, y: *mut f32, y_ld: i64, xp: *mut c_void, s: Stream,
        ) -> i32;
        pub fn infer_m_gdn_step(
            c: *const f32, c_ld: i64, z: *const f32, z_ld: i64, alpha: *const f32, beta: *const f32, ab_ld: i64, ssm_a: *const f32,
            dt_bias: *const f32, norm_w: *const f32, state: *mut f32, slots: *const i32, y: *mut f32, y_ld: i64, xp: *mut c_void, b: i32,
            hk: i32, hv: i32, eps: f32, s: Stream,
        ) -> i32;
        pub fn infer_m_rope_tab(pos: *const i32, n: i32, n_rot: i32, sections: *const i32, theta: f32, tab: *mut c_void, s: Stream) -> i32;
        pub fn infer_m_qk_rope(
            q: *const f32, k: *const f32, v: *const f32, q_ld: i64, k_ld: i64, v_ld: i64, q_norm: *const f32, k_norm: *const f32,
            tab: *const c_void, slot: *const i32, q_out: *mut c_void, kc: *mut c_void, vc: *mut c_void, n: i32, h: i32, hkv: i32,
            n_rot: i32, eps: f32, s: Stream,
        ) -> i32;
        pub fn infer_m_argmax2_ws_bytes(n_rows: i32) -> i64;
        pub fn infer_m_argmax2(
            x: *const f32, n: i32, ld: i64, n_rows: i32, part: *mut c_void, out: *mut i32, out_v: *mut f32, s: Stream,
        ) -> i32;
        pub fn infer_g3_gemv_multi_conv(
            nm: i32, qtype: *const i32, w: *const *const c_void, n: *const i32, y: *const *mut f32, k: i32, xp: *const c_void, b: i32,
            conv_w: *const f32, conv_st: *mut f32, slots: *const i32, s: Stream,
        ) -> i32;
    }
}

const BMAX: usize = GEMV_B_MAX;
// 入力の置き場（i32）の並び: 語・位置・KV の書き先・列の範囲（B+1）・KV の長さ・GDN の状態の番号・頁表 [B][n_pages]
const O_IDS: usize = 0;
const O_POS: usize = BMAX;
const O_KVS: usize = 2 * BMAX;
const O_QS: usize = 3 * BMAX;
const O_KVL: usize = 4 * BMAX + 1;
const O_GS: usize = 5 * BMAX + 1;
/// 確かめ（何列か）: GDN の新しい状態の置き場の番号（仮の置き場）[B]
const O_GSO: usize = 6 * BMAX + 1;
const O_BT: usize = 7 * BMAX + 4; // 16 バイトに揃える

struct Graph {
    g: *mut c_void,
    e: *mut c_void,
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            dffi::cudaGraphExecDestroy(self.e);
            dffi::cudaGraphDestroy(self.g);
        }
    }
}

/// デコードの記録（取り直したグラフの数など）
#[derive(Clone, Copy, Debug, Default)]
pub struct DecodeStats {
    pub steps: usize,
    pub captures: usize,
    pub capture_ms: f64,
}

pub(crate) struct Dec {
    s: Stream,
    pool_id: u64,
    n_pages: usize,
    din: DevBuf,
    hin: *mut i32,
    hin_n: usize,
    hout: *mut i32,
    tab: DevBuf,
    g3ws: DevBuf,
    amws: DevBuf,
    /// (B か 確かめの印＋語数, KV の幅, 確かめの h の書き先)
    graphs: HashMap<(usize, usize, usize), Graph>,
    use_graph: bool,
    stats: DecodeStats,
    /// 確定（commit_multi）の流れ（NULL の流れと待ち合わせない）・確かめの終わりの印・確定の終わりの印・入力の置き場（確かめの入力と別）
    cs: Stream,
    vev: *mut c_void,
    cev: *mut c_void,
    cpend: bool,
    chin: *mut i32,
    cdin: DevBuf,
}

impl Drop for Dec {
    fn drop(&mut self) {
        unsafe {
            // 確定の流れがまだ走っていれば終わるまで待つ（chin・グラフ・流れを捨てる前に）
            if self.cpend {
                dffi::cudaEventSynchronize(self.cev);
                self.cpend = false;
            }
        }
        self.graphs.clear();
        unsafe {
            dffi::cudaFreeHost(self.hin.cast());
            dffi::cudaFreeHost(self.hout.cast());
            dffi::cudaFreeHost(self.chin.cast());
            dffi::cudaEventDestroy(self.cev);
            dffi::cudaEventDestroy(self.vev);
            dffi::cudaStreamDestroy(self.cs);
            dffi::cudaStreamDestroy(self.s);
        }
    }
}

fn host_alloc(n: usize) -> *mut c_void {
    let mut p = null_mut();
    check(unsafe { dffi::cudaMallocHost(&mut p, n.max(16)) }, "cudaMallocHost");
    p
}

impl Dec {
    fn new(cfg: &Config, pool: &Pool) -> Dec {
        let mut s = null_mut();
        // 0 = cudaStreamDefault（NULL の流れとは暗に順がつく: プレフィル（NULL）→ デコードの順が守られる）
        // 優先度は高く（M3: プレフィルが別の流れ（低い優先度・非同期）で走っている間も、デコードの歩のブロックを先に入れる）
        check(unsafe { dffi::cudaStreamCreateWithPriority(&mut s, 0, -1) }, "cudaStreamCreate");
        let hin_n = O_BT + BMAX * pool.n_pages;
        let g3ws = DevBuf::new(unsafe { ffi::infer_g3_ws_bytes() } as usize);
        zero(g3ws.ptr, g3ws.len);
        let use_graph = std::env::var("TACHYON_NO_GRAPH").map(|v| v.is_empty() || v == "0").unwrap_or(true);
        let mut cs = null_mut();
        check(unsafe { dffi::cudaStreamCreateWithPriority(&mut cs, 1, -1) }, "確定の流れ");
        let (mut cev, mut vev) = (null_mut(), null_mut());
        // 2 = cudaEventDisableTiming
        check(unsafe { dffi::cudaEventCreateWithFlags(&mut cev, 2) }, "確定の印");
        check(unsafe { dffi::cudaEventCreateWithFlags(&mut vev, 2) }, "確かめの印");
        Dec {
            cs,
            vev,
            cev,
            cpend: false,
            chin: host_alloc(2 * BMAX * 4).cast(),
            cdin: DevBuf::new(2 * BMAX * 4),
            s,
            pool_id: pool.id,
            n_pages: pool.n_pages,
            din: DevBuf::new(hin_n * 4),
            hin: host_alloc(hin_n * 4).cast(),
            hin_n,
            hout: host_alloc(BMAX * 4).cast(),
            tab: DevBuf::new(BMAX * cfg.attn.n_rot / 2 * 8),
            g3ws,
            amws: DevBuf::new(unsafe { dffi::infer_m_argmax2_ws_bytes(BMAX as i32) } as usize),
            graphs: HashMap::new(),
            use_graph,
            stats: DecodeStats::default(),
        }
    }
    fn i32p(&self, off: usize) -> *mut i32 {
        at(&self.din, off * 4)
    }
    /// デコードの流れに、前の確定を待たせる（GPU の上で。手元は待たない）
    fn order_after_commit(&mut self) {
        if self.cpend {
            check(unsafe { dffi::cudaStreamWaitEvent(self.s, self.cev, 0) }, "確定を待つ");
        }
    }
}

impl Model {
    /// 確定（commit_multi）が終わるまで手元で待つ（状態を NULL の流れ・プレフィルの流れで触る前に）
    pub fn sync_commit(&mut self) {
        if let Some(d) = self.dec.as_mut() {
            if d.cpend {
                check(unsafe { dffi::cudaEventSynchronize(d.cev) }, "確定の待ち");
                d.cpend = false;
            }
        }
    }
}

/// グラフに取る 1 歩の形（ポインタは全部 GPU の決まった置き場）
struct StepIo {
    s: Stream,
    b: usize,
    max_kv: usize,
    ids: *const i32,
    pos: *const i32,
    kvs: *const i32,
    qs: *const i32,
    kvl: *const i32,
    gs: *const i32,
    bt: *const i32,
    bt_ld: usize,
    tab: *mut c_void,
    g3ws: *mut c_void,
    g3ws_len: i64,
    amws: *mut c_void,
    // 置き場の頭（層 gi・ai の分は足して使う）
    state: *mut u8,
    conv: *mut u8,
    kc: *mut u8,
    vc: *mut u8,
    state_stride: usize,
    conv_stride: usize,
    kv_stride: usize,
}

impl Model {
    /// B 本の列を 1 語ずつ進める（tokens[i] は seqs[i] の次の語。どの列も同じ置き場（new_seqs）・違う slot）。
    /// 返り値は列ごとの 1 位の語。ロジットは作業場の logits [B][vocab] に残る（logits(B) で読める）。どの列も pos が 1 進む
    pub fn decode(&mut self, seqs: &mut [&mut Seq], tokens: &[u32]) -> Vec<u32> {
        self.decode_h(seqs, tokens, null_mut())
    }

    /// decode と同じ。hout が null でなければ、各列の h_nextn（output_norm の後）[B][d] をそこへ（MTP に見せる。毎回同じ番地にする）
    pub fn decode_h(&mut self, seqs: &mut [&mut Seq], tokens: &[u32], hout: *mut f32) -> Vec<u32> {
        let b = seqs.len();
        assert!(b >= 1 && b <= BMAX, "1 歩の列の数 {b} は 1..{BMAX}");
        assert_eq!(tokens.len(), b, "語の数と列の数");
        assert!(self.layer0 == 0 && self.layers.len() == self.cfg.n_layer && self.output.is_some(), "decode は全部の層と出力が要る");
        let pool = seqs[0].pool.clone();
        for (i, s) in seqs.iter().enumerate() {
            assert!(std::rc::Rc::ptr_eq(&s.pool, &pool), "decode の列は同じ置き場（new_seqs）から");
            assert!(s.pos + 1 <= s.max_ctx, "列 {i}: 文脈 {} が上限 {} を超える", s.pos + 1, s.max_ctx);
            assert!(seqs[..i].iter().all(|t| t.slot != s.slot), "同じ列が 2 回");
        }
        assert!(tokens.iter().all(|&x| (x as usize) < self.cfg.vocab), "語の番号が語彙の外");
        if self.dec.as_ref().map_or(true, |d| d.pool_id != pool.id) {
            self.dec = None; // 前の置き場のグラフを捨ててから
            self.dec = Some(Dec::new(&self.cfg, &pool));
        }
        let (n_gdn, np) = (pool.n_gdn, pool.n_pages);
        for s in seqs.iter_mut() {
            assert!(s.ensure(s.pos + 1), "KV の頁が足りない（列 {}・位置 {}）", s.slot, s.pos + 1);
        }
        let max_kv = seqs.iter().map(|s| s.pos + 1).max().unwrap();
        let bucket = ((max_kv + BUCKET - 1) / BUCKET * BUCKET).min(np * attn::PAGE);
        // 入力を固定した RAM に並べて 1 回で写す
        let dec = self.dec.as_mut().unwrap();
        dec.order_after_commit();
        let h = unsafe { std::slice::from_raw_parts_mut(dec.hin, dec.hin_n) };
        for (i, s) in seqs.iter().enumerate() {
            let p = s.pos;
            h[O_IDS + i] = tokens[i] as i32;
            // RoPE の位置（画像のある列はずれる。文字だけなら p）
            h[O_POS + i] = if s.media.is_empty() { p as i32 } else { s.rope_pos(p) };
            h[O_KVS + i] = s.kv_row(p);
            h[O_KVL + i] = (p + 1) as i32;
            h[O_GS + i] = (s.slot * n_gdn) as i32;
            for (j, &pg) in s.pages.iter().enumerate() {
                h[O_BT + i * np + j] = pg as i32;
            }
        }
        for i in 0..=b {
            h[O_QS + i] = i as i32;
        }
        let n_copy = O_BT + b * np;
        check(unsafe { dffi::cudaMemcpyAsync(dec.din.ptr, dec.hin.cast(), n_copy * 4, 1, dec.s) }, "入力の写し");
        let io = StepIo {
            s: dec.s,
            b,
            max_kv: bucket,
            ids: dec.i32p(O_IDS),
            pos: dec.i32p(O_POS),
            kvs: dec.i32p(O_KVS),
            qs: dec.i32p(O_QS),
            kvl: dec.i32p(O_KVL),
            gs: dec.i32p(O_GS),
            bt: dec.i32p(O_BT),
            bt_ld: np,
            tab: dec.tab.ptr,
            g3ws: dec.g3ws.ptr,
            g3ws_len: dec.g3ws.len as i64,
            amws: dec.amws.ptr,
            state: pool.state.ptr.cast(),
            conv: pool.conv.ptr.cast(),
            kc: pool.kc.ptr.cast(),
            vc: pool.vc.ptr.cast(),
            state_stride: pool.state_len * 4,
            conv_stride: pool.conv_len * 4,
            kv_stride: pool.total_pages * pool.page_bytes,
        };
        if dec.use_graph {
            let key = (b, bucket, hout as usize);
            if !dec.graphs.contains_key(&key) {
                let t0 = std::time::Instant::now();
                let mut g = null_mut();
                // 2 = cudaStreamCaptureModeRelaxed（起動の中の cudaFuncSetAttribute などを止めない）
                check(unsafe { dffi::cudaStreamBeginCapture(io.s, 2) }, "cudaStreamBeginCapture");
                self.enqueue_step(&io, hout);
                check(unsafe { dffi::cudaStreamEndCapture(io.s, &mut g) }, "cudaStreamEndCapture");
                let mut e = null_mut();
                check(unsafe { dffi::cudaGraphInstantiateWithFlags(&mut e, g, 0) }, "cudaGraphInstantiate");
                let dec = self.dec.as_mut().unwrap();
                // 使わなくなった幅のグラフは捨てる（同じ B の短い幅）
                if dec.graphs.len() >= 16 {
                    dec.graphs.retain(|k, _| k.0 != b || k.1 >= bucket);
                }
                dec.graphs.insert(key, Graph { g, e });
                dec.stats.captures += 1;
                dec.stats.capture_ms += t0.elapsed().as_secs_f64() * 1e3;
            }
            let dec = self.dec.as_ref().unwrap();
            check(unsafe { dffi::cudaGraphLaunch(dec.graphs[&key].e, io.s) }, "cudaGraphLaunch");
        } else {
            self.enqueue_step(&io, hout);
        }
        let am_ptr = self.sdec().am.ptr;
        let dec = self.dec.as_mut().unwrap();
        check(
            unsafe { dffi::cudaMemcpyAsync(dec.hout.cast(), am_ptr, b * 4, 2, dec.s) },
            "argmax の写し",
        );
        check(unsafe { dffi::cudaStreamSynchronize(dec.s) }, "decode の同期");
        dec.stats.steps += 1;
        let out = unsafe { std::slice::from_raw_parts(dec.hout, b) };
        for s in seqs.iter_mut() {
            s.pos += 1;
        }
        out.iter()
            .map(|&i| {
                assert!(i >= 0, "ロジットが全部 NaN");
                i as u32
            })
            .collect()
    }

    pub fn decode_stats(&self) -> DecodeStats {
        self.dec.as_ref().map(|d| d.stats).unwrap_or_default()
    }

    /// 1 歩のカーネルを io.s に積む（グラフに取るときもそのまま流すときも同じ）
    fn enqueue_step(&self, io: &StepIo, hout: *mut f32) {
        let cfg = self.cfg;
        let sc = self.sdec();
        let (d, ff, b, s) = (cfg.d, cfg.d_ff, io.b, io.s);
        let eps = cfg.eps;
        check(
            unsafe { kffi::infer_m_embed(self.embd_qt as i32, self.embd.ptr, d as i32, io.ids, b as i32, sc.h.f32p(), s) },
            "embed",
        );
        let sh = cfg.attn;
        check(unsafe { dffi::infer_m_rope_tab(io.pos, b as i32, sh.n_rot as i32, sh.sections.as_ptr(), sh.theta, io.tab, s) }, "rope tab");
        let norm_xp = |x: *const f32, w: &DevBuf, xp: &DevBuf| {
            check(
                unsafe { dffi::infer_m_rmsnorm_xp(x, d as i64, w.f32p(), d as i32, b as i32, eps, null_mut(), 0, xp.ptr, s) },
                "rmsnorm_xp",
            );
        };
        // 残差に足す（決まった順の k 分け。作業場は 1 つの流れに 1 つ）
        let add = |m: &Mat, xp: &DevBuf| {
            check(
                unsafe {
                    ffi::infer_g3_gemv_ws(m.qt, m.buf.ptr, m.k as i32, m.n as i32, xp.ptr, sc.h.f32p(), b as i32, 0, 1, io.g3ws, io.g3ws_len, s)
                },
                "g3 gemv add",
            );
        };
        let (mut gi, mut ai) = (0usize, 0usize);
        for l in &self.layers {
            norm_xp(sc.h.f32p(), &l.attn_norm, &sc.xp_d);
            match &l.mix {
                Mix::Gdn(w) => {
                    let g = cfg.gdn;
                    let (cc, ao, hv) = (g.conv_ch(), g.out_dim(), g.hv);
                    let mats = [&w.qkv, &w.gate, &w.alpha, &w.beta];
                    let ts: Vec<i32> = mats.iter().map(|m| m.qt).collect();
                    let ws: Vec<*const c_void> = mats.iter().map(|m| m.buf.ptr as *const c_void).collect();
                    let ns: Vec<i32> = mats.iter().map(|m| m.n as i32).collect();
                    let ys = [sc.c.f32p(), sc.z.f32p(), sc.alpha.f32p(), sc.beta.f32p()];
                    let conv_st: *mut f32 = unsafe { io.conv.add(gi * io.conv_stride).cast() };
                    check(
                        unsafe {
                            dffi::infer_g3_gemv_multi_conv(4, ts.as_ptr(), ws.as_ptr(), ns.as_ptr(), ys.as_ptr(), d as i32, sc.xp_d.ptr, b as i32,
                                                           w.conv.f32p(), conv_st, io.gs, s)
                        },
                        "g3 gemv multi conv",
                    );
                    let st: *mut f32 = unsafe { io.state.add(gi * io.state_stride).cast() };
                    check(
                        unsafe {
                            dffi::infer_m_gdn_step(sc.c.f32p(), cc as i64, sc.z.f32p(), ao as i64, sc.alpha.f32p(), sc.beta.f32p(), hv as i64,
                                                   w.a.f32p(), w.dt.f32p(), w.norm.f32p(), st, io.gs, null_mut(), 0, sc.xp_o.ptr, b as i32,
                                                   g.hk as i32, g.hv as i32, g.eps, s)
                        },
                        "gdn step",
                    );
                    add(&w.out, &sc.xp_o);
                    gi += 1;
                }
                Mix::Attn(w) => {
                    let (qw, kv, ao) = (cfg.q_full(), cfg.kv_dim(), cfg.attn_out());
                    let mats = [&w.q, &w.k, &w.v];
                    let ts: Vec<i32> = mats.iter().map(|m| m.qt).collect();
                    let ws: Vec<*const c_void> = mats.iter().map(|m| m.buf.ptr as *const c_void).collect();
                    let ns: Vec<i32> = mats.iter().map(|m| m.n as i32).collect();
                    let ys = [sc.qf.f32p(), sc.kk.f32p(), sc.vv.f32p()];
                    check(
                        unsafe {
                            ffi::infer_g3_gemv_multi(3, ts.as_ptr(), ws.as_ptr(), ns.as_ptr(), ys.as_ptr(), d as i32, sc.xp_d.ptr, b as i32, 0, 0, s)
                        },
                        "g3 gemv multi",
                    );
                    let kc: *mut c_void = unsafe { io.kc.add(ai * io.kv_stride).cast() };
                    let vc: *mut c_void = unsafe { io.vc.add(ai * io.kv_stride).cast() };
                    check(
                        unsafe {
                            dffi::infer_m_qk_rope(sc.qf.f32p(), sc.kk.f32p(), sc.vv.f32p(), qw as i64, kv as i64, kv as i64, w.q_norm.f32p(),
                                                  w.k_norm.f32p(), io.tab, io.kvs, sc.q16.ptr, kc, vc, b as i32, sh.h as i32, sh.hkv as i32,
                                                  sh.n_rot as i32, sh.eps, s)
                        },
                        "qk rope",
                    );
                    let mut p = AttnParams::new(&sh);
                    p.q = sc.q16.ptr;
                    p.gate = unsafe { sc.qf.f32p().add(attn::D) };
                    p.kc = kc;
                    p.vc = vc;
                    p.block_table = io.bt;
                    p.bt_ld = io.bt_ld as i32;
                    p.q_start = io.qs;
                    p.kv_len = io.kvl;
                    p.n_seq = b as i32;
                    p.n_tok = b as i32;
                    p.max_q_len = 1;
                    p.max_kv = io.max_kv as i32;
                    p.out = sc.y.ptr;
                    p.out_f16 = 0;
                    let wsb = attn::ws_bytes(&p);
                    assert!(wsb <= sc.attn_ws.len, "Attention の作業場が足りない（{wsb}）");
                    p.ws = sc.attn_ws.f32p();
                    p.ws_bytes = sc.attn_ws.len as i64;
                    attn::attn(&p, s).unwrap_or_else(|e| panic!("decode の Attention: {e}"));
                    check(unsafe { ffi::infer_g3_xprep(sc.y.f32p(), ao as i32, b as i32, sc.xp_o.ptr, s) }, "g3 xprep");
                    add(&w.o, &sc.xp_o);
                    ai += 1;
                }
            }
            if l.moe.is_some() {
                // MoE（3.6）: ルーターは f32 の b を読むので xa にも書く
                check(
                    unsafe {
                        dffi::infer_m_rmsnorm_xp(sc.h.f32p(), d as i64, l.post_norm.f32p(), d as i32, b as i32, eps, sc.xa.f32p(), d as i64,
                                                 sc.xp_d.ptr, s)
                    },
                    "rmsnorm_xp（MoE）",
                );
                self.moe_ffn(l, sc, b, false, s);
                continue;
            }
            norm_xp(sc.h.f32p(), &l.post_norm, &sc.xp_d);
            check(
                unsafe {
                    ffi::infer_g3_swiglu(l.gate.qt, l.gate.buf.ptr, l.up.qt, l.up.buf.ptr, d as i32, ff as i32, sc.xp_d.ptr, null_mut(),
                                         sc.xp_ff.ptr, b as i32, s)
                },
                "g3 swiglu",
            );
            add(&l.down, &sc.xp_ff);
        }
        // 出口
        let out = self.output.as_ref().unwrap();
        check(
            unsafe { dffi::infer_m_rmsnorm_xp(sc.h.f32p(), d as i64, self.out_norm.f32p(), d as i32, b as i32, eps, hout, d as i64, sc.xp_d.ptr, s) },
            "rmsnorm_xp（出口）",
        );
        check(
            unsafe { ffi::infer_g3_gemv(out.qt, out.buf.ptr, d as i32, out.n as i32, sc.xp_d.ptr, sc.logits.f32p(), b as i32, 0, 0, s) },
            "output gemv",
        );
        check(
            unsafe {
                dffi::infer_m_argmax2(sc.logits.f32p(), out.n as i32, out.n as i64, b as i32, io.amws, sc.am.ptr.cast(), sc.amv.f32p(), s)
            },
            "argmax",
        );
    }
}

// ---------------------------------------------------------------- 下書きの確かめ（M2・MTP）

/// 確かめのグラフの鍵の印（デコードの B ≤ 32 と重ならない）
const VERIFY_KEY: usize = 10_000;
/// 確かめのグラフを取り直す KV の幅（語）
const VERIFY_BUCKET: usize = 1024;

impl Model {
    /// 下書きの確かめ（1 列・T = tokens.len() 語、2..=tmax）を CUDA Graph で: forward_verify ＋ out_norm_rows（hout へ）＋ head（T 行）＋ argmax。
    /// 返り値は行ごとの 1 位の語（ロジットは作業場の logits [T][vocab] に残る）。GDN の新しい状態は仮の置き場へ（commit_verify で確定）。
    /// seq は new_seq_spec の列。hout は毎回同じ番地にする（グラフに取る）
    pub fn verify_step(&mut self, seq: &mut Seq, tokens: &[u32], hout: *mut f32) -> Vec<u32> {
        self.verify_multi(&mut [seq], &[tokens.to_vec()], hout).pop().unwrap()
    }

    /// 何列かの確かめを 1 回で（M3）: 列 b の T 語（どの列も同じ T。短い列は呼ぶ側で埋める）。行は [b][T] の順
    /// （logits・argmax・hout も）。commit は列ごとに commit_verify_rows（行 b·T から）
    pub fn verify_multi(&mut self, seqs: &mut [&mut Seq], toks: &[Vec<u32>], hout: *mut f32) -> Vec<Vec<u32>> {
        let nb = seqs.len();
        assert!(nb >= 1 && toks.len() == nb);
        let t = toks[0].len();
        assert!(toks.iter().all(|x| x.len() == t), "どの列も同じ語数");
        let rows = nb * t;
        assert!(self.layer0 == 0 && self.layers.len() == self.cfg.n_layer && self.output.is_some(), "verify は全部の層と出力が要る");
        let pool = seqs[0].pool.clone();
        assert!(pool.spec_state, "確かめは spec_state の置き場の列で");
        for (i, sq) in seqs.iter().enumerate() {
            assert!(std::rc::Rc::ptr_eq(&sq.pool, &pool), "同じ置き場の列");
            assert!(seqs[..i].iter().all(|o| o.slot != sq.slot), "同じ列が 2 回");
            assert!(sq.pos + t <= sq.max_ctx, "文脈 {} が上限 {} を超える", sq.pos + t, sq.max_ctx);
        }
        assert!(toks.iter().flatten().all(|&x| (x as usize) < self.cfg.vocab), "語の番号が語彙の外");
        {
            let b = self.spec.as_mut().expect("enable_spec を先に");
            assert!(t >= 2 && rows <= b.tmax, "確かめの行 {nb}×{t} は 2..={}", b.tmax);
        }
        if self.dec.as_ref().map_or(true, |d| d.pool_id != pool.id) {
            self.dec = None;
            self.dec = Some(Dec::new(&self.cfg, &pool));
        }
        let (np, n_gdn, ns) = (pool.n_pages, pool.n_gdn, pool.n_slots);
        for sq in seqs.iter_mut() {
            let need = sq.pos + t;
            assert!(sq.ensure(need), "KV の頁が足りない（位置 {need}）");
        }
        let max_kv = seqs.iter().map(|sq| sq.pos + t).max().unwrap();
        // 確かめは T が 2..=8 と変わるので、グラフを取り直す幅はデコードより粗く（VERIFY_BUCKET 語ごと）
        let bucket = ((max_kv + VERIFY_BUCKET - 1) / VERIFY_BUCKET * VERIFY_BUCKET).min(np * attn::PAGE);
        let dec = self.dec.as_mut().unwrap();
        dec.order_after_commit();
        let h = unsafe { std::slice::from_raw_parts_mut(dec.hin, dec.hin_n) };
        for (b, sq) in seqs.iter().enumerate() {
            for i in 0..t {
                let p = sq.pos + i;
                h[O_IDS + b * t + i] = toks[b][i] as i32;
                h[O_POS + b * t + i] = if sq.media.is_empty() { p as i32 } else { sq.rope_pos(p) };
                h[O_KVS + b * t + i] = sq.kv_row(p);
            }
            h[O_QS + b] = (b * t) as i32;
            h[O_KVL + b] = (sq.pos + t) as i32;
            h[O_GS + b] = (sq.slot * n_gdn) as i32;
            h[O_GSO + b] = ((ns + sq.slot) * n_gdn) as i32;
            for (j, &pg) in sq.pages.iter().enumerate() {
                h[O_BT + b * np + j] = pg as i32;
            }
        }
        h[O_QS + nb] = rows as i32;
        let n_copy = O_BT + nb * np;
        check(unsafe { dffi::cudaMemcpyAsync(dec.din.ptr, dec.hin.cast(), n_copy * 4, 1, dec.s) }, "入力の写し");
        let io = StepIo {
            s: dec.s,
            b: rows,
            max_kv: bucket,
            ids: dec.i32p(O_IDS),
            pos: dec.i32p(O_POS),
            kvs: dec.i32p(O_KVS),
            qs: dec.i32p(O_QS),
            kvl: dec.i32p(O_KVL),
            gs: dec.i32p(O_GS),
            bt: dec.i32p(O_BT),
            bt_ld: np,
            tab: dec.tab.ptr,
            g3ws: dec.g3ws.ptr,
            g3ws_len: dec.g3ws.len as i64,
            amws: dec.amws.ptr,
            state: pool.state.ptr.cast(),
            conv: pool.conv.ptr.cast(),
            kc: pool.kc.ptr.cast(),
            vc: pool.vc.ptr.cast(),
            state_stride: pool.state_len * 4,
            conv_stride: pool.conv_len * 4,
            kv_stride: pool.total_pages * pool.page_bytes,
        };
        let gso: *const i32 = dec.i32p(O_GSO);
        let use_graph = dec.use_graph;
        let key = (VERIFY_KEY + nb * 64 + t, bucket, hout as usize);
        if use_graph {
            if !dec.graphs.contains_key(&key) {
                let t0 = std::time::Instant::now();
                let mut g = null_mut();
                check(unsafe { dffi::cudaStreamBeginCapture(io.s, 2) }, "cudaStreamBeginCapture");
                self.enqueue_verify(&io, nb, t, gso, hout);
                check(unsafe { dffi::cudaStreamEndCapture(io.s, &mut g) }, "cudaStreamEndCapture");
                let mut e = null_mut();
                check(unsafe { dffi::cudaGraphInstantiateWithFlags(&mut e, g, 0) }, "cudaGraphInstantiate");
                let dec = self.dec.as_mut().unwrap();
                if dec.graphs.len() >= 128 {
                    dec.graphs.retain(|k, _| k.0 != key.0 || k.1 >= bucket);
                }
                dec.graphs.insert(key, Graph { g, e });
                dec.stats.captures += 1;
                dec.stats.capture_ms += t0.elapsed().as_secs_f64() * 1e3;
            }
            let dec = self.dec.as_ref().unwrap();
            check(unsafe { dffi::cudaGraphLaunch(dec.graphs[&key].e, io.s) }, "cudaGraphLaunch");
        } else {
            self.enqueue_verify(&io, nb, t, gso, hout);
        }
        let am_ptr = self.sdec().am.ptr;
        let dec = self.dec.as_mut().unwrap();
        check(unsafe { dffi::cudaMemcpyAsync(dec.hout.cast(), am_ptr, rows * 4, 2, dec.s) }, "argmax の写し");
        check(unsafe { dffi::cudaStreamSynchronize(dec.s) }, "verify の同期");
        dec.stats.steps += 1;
        let out = unsafe { std::slice::from_raw_parts(dec.hout, rows) };
        for sq in seqs.iter_mut() {
            sq.pos += t;
        }
        (0..nb)
            .map(|b| {
                out[b * t..(b + 1) * t]
                    .iter()
                    .map(|&i| {
                        assert!(i >= 0, "ロジットが全部 NaN");
                        i as u32
                    })
                    .collect()
            })
            .collect()
    }

    /// 確かめの 1 歩のカーネルを io.s に積む（1 列・T 語。GDN は 1 語ずつの道で、新しい状態は仮の置き場へ）
    fn enqueue_verify(&self, io: &StepIo, nb: usize, tq: usize, gso: *const i32, hout: *mut f32) {
        let cfg = self.cfg;
        let sc = self.sdec();
        let sb = self.spec.as_ref().unwrap();
        let (d, ff, t, s) = (cfg.d, cfg.d_ff, io.b, io.s);
        let eps = cfg.eps;
        check(unsafe { kffi::infer_m_embed(self.embd_qt as i32, self.embd.ptr, d as i32, io.ids, t as i32, sc.h.f32p(), s) }, "embed");
        let sh = cfg.attn;
        check(unsafe { dffi::infer_m_rope_tab(io.pos, t as i32, sh.n_rot as i32, sh.sections.as_ptr(), sh.theta, io.tab, s) }, "rope tab");
        let norm_xp = |x: *const f32, w: &DevBuf, y: *mut f32, xp: &DevBuf| {
            check(
                unsafe { dffi::infer_m_rmsnorm_xp(x, d as i64, w.f32p(), d as i32, t as i32, eps, y, d as i64, xp.ptr, s) },
                "rmsnorm_xp",
            );
        };
        let add = |m: &Mat, xp: &DevBuf| {
            check(
                unsafe {
                    ffi::infer_g3_gemv_ws(m.qt, m.buf.ptr, m.k as i32, m.n as i32, xp.ptr, sc.h.f32p(), t as i32, 0, 1, io.g3ws, io.g3ws_len, s)
                },
                "g3 gemv add",
            );
        };
        let (mut gi, mut ai) = (0usize, 0usize);
        for l in &self.layers {
            norm_xp(sc.h.f32p(), &l.attn_norm, null_mut(), &sc.xp_d);
            match &l.mix {
                Mix::Gdn(w) => {
                    let g = cfg.gdn;
                    let (cc, ao, hv) = (g.conv_ch(), g.out_dim(), g.hv);
                    let (y_qkv, y_a, y_b, y_c) = sb.hist(gi);
                    let mats = [&w.qkv, &w.gate, &w.alpha, &w.beta];
                    let ts: Vec<i32> = mats.iter().map(|m| m.qt).collect();
                    let ws: Vec<*const c_void> = mats.iter().map(|m| m.buf.ptr as *const c_void).collect();
                    let ns: Vec<i32> = mats.iter().map(|m| m.n as i32).collect();
                    let ys = [y_qkv, sc.z.f32p(), y_a, y_b];
                    check(
                        unsafe {
                            ffi::infer_g3_gemv_multi(4, ts.as_ptr(), ws.as_ptr(), ns.as_ptr(), ys.as_ptr(), d as i32, sc.xp_d.ptr, t as i32, 0, 0, s)
                        },
                        "g3 gemv multi",
                    );
                    let layer = gdn::Layer { a: w.a.f32p(), dt: w.dt.f32p(), norm: w.norm.f32p(), conv: w.conv.f32p() };
                    let mut gio = gdn::Io::empty();
                    gio.qkv = y_qkv;
                    gio.qkv_ld = cc;
                    gio.c = y_c;
                    gio.z = sc.z.f32p();
                    gio.z_ld = ao;
                    gio.alpha = y_a;
                    gio.beta = y_b;
                    gio.ab_ld = hv;
                    gio.y = sc.y.f32p();
                    gio.y_ld = ao;
                    // 列 b の状態は 置き場の頭＋層 gi ＋ 番号 gs[b]（slot·n_gdn）、新しい状態は gso[b]（仮の置き場）
                    gio.state = unsafe { io.state.add(gi * io.state_stride).cast() };
                    gio.conv_state = unsafe { io.conv.add(gi * io.conv_stride).cast() };
                    gio.slots = io.gs;
                    gio.slots_out = gso;
                    gdn::conv(&g, &layer, &gio, nb, tq, s);
                    gdn::decode(&g, &layer, &gio, nb, tq, s);
                    check(unsafe { ffi::infer_g3_xprep(sc.y.f32p(), ao as i32, t as i32, sc.xp_o.ptr, s) }, "g3 xprep");
                    add(&w.out, &sc.xp_o);
                    gi += 1;
                }
                Mix::Attn(w) => {
                    let (qw, kv, ao) = (cfg.q_full(), cfg.kv_dim(), cfg.attn_out());
                    let mats = [&w.q, &w.k, &w.v];
                    let ts: Vec<i32> = mats.iter().map(|m| m.qt).collect();
                    let ws: Vec<*const c_void> = mats.iter().map(|m| m.buf.ptr as *const c_void).collect();
                    let ns: Vec<i32> = mats.iter().map(|m| m.n as i32).collect();
                    let ys = [sc.qf.f32p(), sc.kk.f32p(), sc.vv.f32p()];
                    check(
                        unsafe {
                            ffi::infer_g3_gemv_multi(3, ts.as_ptr(), ws.as_ptr(), ns.as_ptr(), ys.as_ptr(), d as i32, sc.xp_d.ptr, t as i32, 0, 0, s)
                        },
                        "g3 gemv multi",
                    );
                    let kc: *mut c_void = unsafe { io.kc.add(ai * io.kv_stride).cast() };
                    let vc: *mut c_void = unsafe { io.vc.add(ai * io.kv_stride).cast() };
                    check(
                        unsafe {
                            dffi::infer_m_qk_rope(sc.qf.f32p(), sc.kk.f32p(), sc.vv.f32p(), qw as i64, kv as i64, kv as i64, w.q_norm.f32p(),
                                                  w.k_norm.f32p(), io.tab, io.kvs, sc.q16.ptr, kc, vc, t as i32, sh.h as i32, sh.hkv as i32,
                                                  sh.n_rot as i32, sh.eps, s)
                        },
                        "qk rope",
                    );
                    let mut p = AttnParams::new(&sh);
                    p.q = sc.q16.ptr;
                    p.gate = unsafe { sc.qf.f32p().add(attn::D) };
                    p.kc = kc;
                    p.vc = vc;
                    p.block_table = io.bt;
                    p.bt_ld = io.bt_ld as i32;
                    p.q_start = io.qs;
                    p.kv_len = io.kvl;
                    p.n_seq = nb as i32;
                    p.n_tok = t as i32;
                    p.max_q_len = tq as i32;
                    p.max_kv = io.max_kv as i32;
                    p.out = sc.y.ptr;
                    p.out_f16 = 0;
                    let wsb = attn::ws_bytes(&p);
                    assert!(wsb <= sc.attn_ws.len, "Attention の作業場が足りない（{wsb}）");
                    p.ws = sc.attn_ws.f32p();
                    p.ws_bytes = sc.attn_ws.len as i64;
                    attn::attn(&p, s).unwrap_or_else(|e| panic!("確かめの Attention: {e}"));
                    check(unsafe { ffi::infer_g3_xprep(sc.y.f32p(), ao as i32, t as i32, sc.xp_o.ptr, s) }, "g3 xprep");
                    add(&w.o, &sc.xp_o);
                    ai += 1;
                }
            }
            if l.moe.is_some() {
                norm_xp(sc.h.f32p(), &l.post_norm, sc.xa.f32p(), &sc.xp_d);
                self.moe_ffn(l, sc, t, false, s);
                continue;
            }
            norm_xp(sc.h.f32p(), &l.post_norm, null_mut(), &sc.xp_d);
            check(
                unsafe {
                    ffi::infer_g3_swiglu(l.gate.qt, l.gate.buf.ptr, l.up.qt, l.up.buf.ptr, d as i32, ff as i32, sc.xp_d.ptr, null_mut(),
                                         sc.xp_ff.ptr, t as i32, s)
                },
                "g3 swiglu",
            );
            add(&l.down, &sc.xp_ff);
        }
        // 出口: h_nextn（output_norm の後）を hout へ（MTP が読む）、logits [T][vocab]、行ごとの 1 位
        let out = self.output.as_ref().unwrap();
        norm_xp(sc.h.f32p(), &self.out_norm, hout, &sc.xp_d);
        check(
            unsafe { ffi::infer_g3_gemv(out.qt, out.buf.ptr, d as i32, out.n as i32, sc.xp_d.ptr, sc.logits.f32p(), t as i32, 0, 0, s) },
            "output gemv",
        );
        check(
            unsafe {
                dffi::infer_m_argmax2(sc.logits.f32p(), out.n as i32, out.n as i64, t as i32, io.amws, sc.am.ptr.cast(), sc.amv.f32p(), s)
            },
            "argmax",
        );
    }
}

// ---------------------------------------------------------------- 確定（M4: 何列かを 1 本のグラフで・自分の流れで）

/// 確定のグラフの鍵の印
const COMMIT_KEY: usize = 20_000;

impl Model {
    /// verify_multi の後の確定を何列か 1 回で（CUDA Graph・確定の流れ）: 列 b を、元の状態から控えの行 b·T.. の前 keeps[b] 語だけ進め直す。
    /// 式は commit_verify_rows と同じ。手元は待たない（次の確かめ・デコードは GPU の上で待つ。状態を別の流れで触る前は sync_commit）
    pub fn commit_multi(&mut self, seqs: &mut [&mut Seq], t: usize, keeps: &[usize]) {
        let nb = seqs.len();
        assert!(nb >= 1 && keeps.len() == nb);
        let pool = seqs[0].pool.clone();
        for (sq, &k) in seqs.iter().zip(keeps) {
            assert!(std::rc::Rc::ptr_eq(&sq.pool, &pool) && k >= 1 && k <= t && sq.pos >= t);
        }
        assert!(nb * t <= self.spec.as_ref().expect("enable_spec を先に").tmax);
        if self.dec.as_ref().map_or(true, |d| d.pool_id != pool.id) {
            self.dec = None;
            self.dec = Some(Dec::new(&self.cfg, &pool));
        }
        let n_gdn = pool.n_gdn;
        let dec = self.dec.as_mut().unwrap();
        // 前の確定（入力の写しを含む）が終わってから chin を書く
        if dec.cpend {
            check(unsafe { dffi::cudaEventSynchronize(dec.cev) }, "確定の待ち");
            dec.cpend = false;
        }
        let h = unsafe { std::slice::from_raw_parts_mut(dec.chin, 2 * BMAX) };
        for (b, sq) in seqs.iter().enumerate() {
            h[b] = (sq.slot * n_gdn) as i32;
            h[BMAX + b] = keeps[b] as i32;
        }
        // 確かめ（デコードの流れ）の後に、確定の流れで
        check(unsafe { dffi::cudaEventRecord(dec.vev, dec.s) }, "確かめの終わりの印");
        check(unsafe { dffi::cudaStreamWaitEvent(dec.cs, dec.vev, 0) }, "確かめを待つ");
        check(unsafe { dffi::cudaMemcpyAsync(dec.cdin.ptr, dec.chin.cast(), 2 * BMAX * 4, 1, dec.cs) }, "確定の入力の写し");
        let (s, gs, nv) = (dec.cs, dec.cdin.ptr as *const i32, unsafe { (dec.cdin.ptr as *const i32).add(BMAX) });
        let (st0, cv0) = (pool.state.ptr as *mut u8, pool.conv.ptr as *mut u8);
        let (sstride, cstride) = (pool.state_len * 4, pool.conv_len * 4);
        let key = (COMMIT_KEY + nb * 64 + t, 0, 0);
        let enqueue = |m: &Model| {
            let sb = m.spec.as_ref().unwrap();
            let sh = m.cfg.gdn;
            let mut gi = 0;
            for l in &m.layers {
                if let Mix::Gdn(w) = &l.mix {
                    let (qkv, a, b, c) = sb.hist(gi);
                    let layer = gdn::Layer { a: w.a.f32p(), dt: w.dt.f32p(), norm: w.norm.f32p(), conv: w.conv.f32p() };
                    let mut io = gdn::Io::empty();
                    io.qkv = qkv;
                    io.qkv_ld = sh.conv_ch();
                    io.c = c;
                    io.z = c; // y が null なので読まない
                    io.z_ld = sh.out_dim();
                    io.alpha = a;
                    io.beta = b;
                    io.ab_ld = sh.hv;
                    io.y = null_mut();
                    io.y_ld = sh.out_dim();
                    io.state = unsafe { st0.add(gi * sstride).cast() };
                    io.conv_state = unsafe { cv0.add(gi * cstride).cast() };
                    io.slots = gs;
                    io.n_valid = nv;
                    gdn::conv_state_only(&sh, &layer, &io, nb, t, s);
                    gdn::decode(&sh, &layer, &io, nb, t, s);
                    gi += 1;
                }
            }
        };
        if dec.use_graph {
            if !dec.graphs.contains_key(&key) {
                let mut g = null_mut();
                check(unsafe { dffi::cudaStreamBeginCapture(s, 2) }, "cudaStreamBeginCapture");
                enqueue(self);
                check(unsafe { dffi::cudaStreamEndCapture(s, &mut g) }, "cudaStreamEndCapture");
                let mut e = null_mut();
                check(unsafe { dffi::cudaGraphInstantiateWithFlags(&mut e, g, 0) }, "cudaGraphInstantiate");
                self.dec.as_mut().unwrap().graphs.insert(key, Graph { g, e });
            }
            let dec = self.dec.as_ref().unwrap();
            check(unsafe { dffi::cudaGraphLaunch(dec.graphs[&key].e, s) }, "cudaGraphLaunch");
        } else {
            enqueue(self);
        }
        let dec = self.dec.as_mut().unwrap();
        check(unsafe { dffi::cudaEventRecord(dec.cev, dec.cs) }, "確定の終わりの印");
        dec.cpend = true;
        for (sq, &k) in seqs.iter_mut().zip(keeps) {
            sq.pos = sq.pos - t + k;
        }
    }
}
