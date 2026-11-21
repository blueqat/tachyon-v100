// SPDX-License-Identifier: Apache-2.0
//! Inkling-2-Mini（GGUF の arch `inkling2_loop`・Thinking Machines Lab・Apache-2.0）の 1 語ずつの道（デコード）。
//!
//! 手本は transformers PR #51050 の modeling_inkling2_loop.py（tools/ink_ref.py）。形（GGUF の ggml-org 版）:
//! - 24 層（物理）。層 6..18 を 3 回（2・3 回目は前の回からの伝言を q・k・v・FFN の入口に混ぜる）＝ 48 実行。
//!   KV と畳み込みの状態は実行ごと（llama.cpp と同じ番号: 実行 il < 24 は物理 il、回 p >= 1 は 24 + (p-1)·12 + (ip-6)）
//! - RoPE は無い。位置は「語ごとの 16 次元 r × 距離ごとの表（1024）」の偏り（距離 1024 以上は 0）と、全体の層の log の倍率 tau
//! - 窓 1024 の層（20）と全体の層（5・11・17・23）。全体の層は q・k・v の入口を別々に混ぜ・別々に正規化し、k・v の畳み込みに語ごとの重みを足す
//! - 語ごとの sink（logit は sink_proj(x)、値は 0）、出口の sigmoid の門
//! - k・v・Attention の出口・FFN の出口に 4 語の短い畳み込み（残差つき・1/√(nc+1)、語ごとの重みありは 1/√(2nc+1)）
//! - 深さの Attention（block attention residual）: 6 層ごとの塊の入口を貯め、各層の入口で「貯めた物＋今」を学習した問いで混ぜる
//! - MoE 128 から 8（sigmoid＋偏りで選び、重みは sigmoid を足して 1 にして ×8）。活性は tanh で抑えた SwiGLU（gate 4・up 25）
//! - 出口の logits は 30·tanh(x/30)、語彙は 201,024 行のうち頭の 200,058
//!
//! 重みは GGUF のまま読み、密な行列・専門家は g3 の並びに詰め直す（GEMV／Tensor Core の GEMM）。KV は fp16。
//! デコードは列 B 本（≤ 32）を 1 歩に（CUDA Graph）、プレフィルは 1 本の列を 512 語ずつ。
use crate::cuda::{check, ffi, DevBuf};
use crate::gguf;
use crate::qtype::QType;
use std::ptr::null;

mod kffi {
    use crate::cuda::ffi::Stream;
    use std::ffi::c_void;
    extern "C" {
        pub fn infer_m_embed(qtype: i32, w: *const c_void, k: i32, ids: *const i32, n_tok: i32, y: *mut f32, s: Stream) -> i32;
        pub fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, n: usize, kind: i32, s: Stream) -> i32;
        pub fn cudaMemset(p: *mut c_void, v: i32, n: usize) -> i32;
        pub fn cudaStreamCreateWithFlags(s: *mut Stream, flags: u32) -> i32;
        pub fn cudaStreamSynchronize(s: Stream) -> i32;
        pub fn cudaStreamBeginCapture(s: Stream, mode: i32) -> i32;
        pub fn cudaStreamEndCapture(s: Stream, g: *mut *mut c_void) -> i32;
        pub fn cudaGraphInstantiateWithFlags(e: *mut *mut c_void, g: *mut c_void, flags: u64) -> i32;
        pub fn cudaGraphLaunch(e: *mut c_void, s: Stream) -> i32;
        pub fn ink_rms(x: *const f32, xld: i64, w: *const f32, k: i32, t: i32, eps: f32, y: *mut f32, yld: i64, s: Stream) -> i32;
        pub fn ink_mix(ptrs: *const *const f32, lds: *const i64, ns: i32, q: *const f32, nq: i32, d: i32, t: i32, y: *mut f32, yld: i64, s: Stream) -> i32;
        pub fn ink_cat2_rms(h: *const f32, hld: i64, m: *const f32, mld: i64, d: i32, t: i32, eps: f32, z: *mut f32, s: Stream) -> i32;
        pub fn ink_ctrl(h: *const f32, hld: i64, m: *const f32, mld: i64, g: *const f32, f: *const f32, d: i32, t: i32, y: *mut f32, yld: i64, s: Stream) -> i32;
        pub fn ink_conv(
            x: *const f32, xld: i64, st: *mut f32, st_ld: i64, lanes: *const i32, w: *const f32, nc: i32, dyn_: *const f32, dld: i64, heads: i32, ch: i32, t: i32,
            scale: f32, y: *mut f32, yld: i64, s: Stream,
        ) -> i32;
        pub fn ink_headprep(x: *mut f32, xld: i64, heads: i32, dh: i32, t: i32, scale: *const f32, thr: f32, eps: f32, tau: *const f32, s: Stream) -> i32;
        pub fn ink_attn_part_floats(t: i32, h: i32, n_split: i32) -> i64;
        pub fn ink_attn_dec(
            q: *const f32, qld: i64, r: *const f32, rld: i64, sink: *const f32, sld: i64, rb: *const f32, kv: super::KvRef, kv_ld: i64, lanes: *const i32,
            pos: *const i32, t: i32, h: i32, kvh: i32, nrel: i32, extent: i32, cap: i32, ring: i32, window: i32, max_keys: i32, chunk: i32, part: *mut f32, y: *mut f32,
            yld: i64, s: Stream,
        ) -> i32;
        pub fn ink_gate(o: *mut f32, old: i64, g: *const f32, gld: i64, n: i32, t: i32, s: Stream) -> i32;
        pub fn ink_add(a: *const f32, ald: i64, b: *const f32, bld: i64, n: i32, t: i32, y: *mut f32, yld: i64, s: Stream) -> i32;
        pub fn ink_tau(pos: *const i32, t: i32, alpha: f32, floor_: f32, tau: *mut f32, s: Stream) -> i32;
        pub fn ink_route(lg: *const f32, ld: i64, bias: *const f32, ne: i32, kk: i32, t: i32, scale: f32, ids: *mut i32, wts: *mut f32, s: Stream) -> i32;
        pub fn ink_situ(gu: *const f32, ff: i32, r: i32, cg: f32, cu: f32, h: *mut f32, hld: i64, s: Stream) -> i32;
        pub fn ink_gemv_f32(w: *const f32, k: i32, n: i32, x: *const f32, xld: i64, t: i32, y: *mut f32, yld: i64, s: Stream) -> i32;
        pub fn ink_ctrl2(g: *const f32, f: *const f32, rank: i32, c: *const f32, h: *const f32, hld: i64, m: *const f32, mld: i64, d: i32, t: i32, y: *mut f32, yld: i64, s: Stream) -> i32;
        pub fn ink_route_warp(lg: *const f32, ld: i64, bias: *const f32, ne: i32, kk: i32, t: i32, scale: f32, ids: *mut i32, wts: *mut f32, s: Stream) -> i32;
        pub fn ink_conv_pf(x: *const f32, xld: i64, st: *const f32, w: *const f32, nc: i32, dyn_: *const f32, dld: i64, heads: i32, ch: i32, t: i32, scale: f32, y: *mut f32, yld: i64, s: Stream) -> i32;
        pub fn ink_conv_pf_state(x: *const f32, xld: i64, st: *mut f32, nc: i32, ch: i32, t: i32, s: Stream) -> i32;
        pub fn ink_attn_pf(
            q: *const f32, qld: i64, r: *const f32, rld: i64, sink: *const f32, sld: i64, rb: *const f32, kv: super::KvRef, pos: *const i32, t: i32,
            h: i32, kvh: i32, nrel: i32, extent: i32, cap: i32, ring: i32, window: i32, y: *mut f32, yld: i64, s: Stream,
        ) -> i32;
        pub fn ink_kvwrite_pf(
            k: *const f32, kld: i64, v: *const f32, vld: i64, n: i32, t: i32, o: super::KvOut, kv_ld: i64, lanes: *const i32, pos: *const i32, cap: i32, ring: i32,
            s: Stream,
        ) -> i32;
        pub fn ink_moe_sum_map(o: *const f32, rowmap: *const i32, wts: *const f32, d: i32, kk: i32, t: i32, y: *mut f32, yld: i64, s: Stream) -> i32;
        pub fn ink_gemm_f32(w: *const f32, k: i32, n: i32, x: *const f32, xld: i64, t: i32, y: *mut f32, yld: i64, s: Stream) -> i32;
        pub fn infer_m_f32_to_f16(x: *const f32, n: i64, y: *mut c_void, s: Stream) -> i32;
        pub fn infer_g3_gemm_ex(
            nmat: i32, qtypes: *const i32, ws: *const *const c_void, ns: *const i32, ys: *const *mut c_void, ldys: *const i32, k: i32, m: i32,
            x: *const c_void, xlo: *const c_void, ldx: i32, mode: i32, cfg: i32, s: Stream,
        ) -> i32;
        pub fn infer_moe_group(
            ids: *const i32, r: i32, kk: i32, ne: i32, tokb: i32, xidx: *mut i32, rowmap: *mut i32, grp: *mut i32, n_blk_max: i32, s: Stream,
        ) -> i32;
        pub fn infer_g3_moe_gemm(
            nmat: i32, qtypes: *const i32, ws: *const *const c_void, es: *const i64, ns: *const i32, ys: *const *mut c_void, ldys: *const i32,
            k: i32, x: *const c_void, xlo: *const c_void, ldx: i32, xidx: *const i32, grp: *const i32, n_blk: i32, tokb: i32, mode: i32,
            s: Stream,
        ) -> i32;
        pub fn infer_g3_moe_gemv(
            nm: i32, qtype: *const i32, w: *const *const c_void, estride: *const i64, n: *const i32, y: *const *mut f32, k: i32,
            xp: *const c_void, rx: i32, xidx: *const i32, je: *const i32, jr0: *const i32, jcnt: *const i32, n_jobs: i32, bmax: i32,
            s: Stream,
        ) -> i32;
        pub fn ink_moe_sum(o: *const f32, wts: *const f32, d: i32, kk: i32, t: i32, y: *mut f32, yld: i64, s: Stream) -> i32;
        pub fn ink_softcap(x: *mut f32, ld: i64, n: i32, t: i32, cap: f32, s: Stream) -> i32;
    }
}

/// kernels/inkling.cu の KvRef・KvOut と同じ並び（fp16 か int8＋128 要素ごとの係数）
#[repr(C)]
#[derive(Clone, Copy)]
pub struct KvRef {
    k: *const std::ffi::c_void,
    v: *const std::ffi::c_void,
    ks: *const f32,
    vs: *const f32,
    kv8: i32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct KvOut {
    k: *mut std::ffi::c_void,
    v: *mut std::ffi::c_void,
    ks: *mut f32,
    vs: *mut f32,
    kv8: i32,
}

/// KV を int8 にするか（TACHYON_INK_KV8=1 か set_kv8。列を作る前に決める）
static KV8: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn set_kv8(on: bool) {
    KV8.store(on, std::sync::atomic::Ordering::SeqCst);
}
fn kv8_on() -> bool {
    KV8.load(std::sync::atomic::Ordering::SeqCst) || std::env::var("TACHYON_INK_KV8").is_ok_and(|v| v == "1")
}

/// このモデルの流れ（CUDA Graph に撮るため、既定の流れではなく自前の 1 本。Ink::load が作る）
static mut STREAM: ffi::Stream = std::ptr::null_mut();
#[allow(non_snake_case)]
#[inline]
fn S() -> ffi::Stream {
    unsafe { STREAM }
}

fn ck(code: i32, what: &str) {
    check(code, what);
}

/// GGUF の inkling2_loop.* から読む形
#[derive(Clone, Debug)]
pub struct InkCfg {
    pub d: usize,
    pub n_phys: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub hd: usize,
    pub n_rel: usize,
    pub extent: usize,
    pub n_swa: usize,
    pub swa: Vec<bool>,
    pub n_exp: usize,
    pub k_exp: usize,
    pub ff: usize,
    pub eps: f32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_floor: f32,
    pub log_alpha: f32,
    pub nc: usize,
    pub block: usize,
    pub loop_start: usize,
    pub loop_end: usize,
    pub passes: usize,
    pub vocab_padded: usize,
    pub vocab: usize,
    pub softcap: f32,
    pub exp_scale: f32,
    pub situ_g: f32,
    pub situ_u: f32,
}

impl InkCfg {
    pub fn from_gguf(g: &gguf::Model) -> Result<InkCfg, String> {
        if g.arch() != "inkling2_loop" {
            return Err(format!("arch {} は inkling2_loop でない", g.arch()));
        }
        let u = |key: &str| g.hp_u32(key).map(|v| v as usize).ok_or_else(|| format!("GGUF に {key} が無い"));
        let f = |key: &str| g.hp_f32(key).ok_or_else(|| format!("GGUF に {key} が無い"));
        let n_phys = u("block_count")?;
        let arr = |key: &str| g.hp_arr_i64(key).ok_or_else(|| format!("GGUF に {key} が無い"));
        let heads = arr("attention.head_count")?;
        let kvh = arr("attention.head_count_kv")?;
        if heads.iter().any(|&h| h != heads[0]) || kvh.iter().any(|&h| h != kvh[0]) {
            return Err("層ごとに頭の数が違う（この版は同じ数だけ）".into());
        }
        let pat = g.hp("attention.sliding_window_pattern").and_then(|v| v.as_arr()).ok_or("sliding_window_pattern が無い")?;
        let swa: Vec<bool> = pat.iter().map(|v| v.as_bool().unwrap_or(false)).collect();
        if swa.len() != n_phys {
            return Err(format!("sliding_window_pattern の長さ {} が層の数 {n_phys} と違う", swa.len()));
        }
        let c = InkCfg {
            d: u("embedding_length")?,
            n_phys,
            n_head: heads[0] as usize,
            n_kv: kvh[0] as usize,
            hd: u("attention.key_length")?,
            n_rel: u("attention.relative_length")?,
            extent: u("attention.relative_extent")?,
            n_swa: u("attention.sliding_window")?,
            swa,
            n_exp: u("expert_count")?,
            k_exp: u("expert_used_count")?,
            ff: u("expert_feed_forward_length")?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            qk_clip: f("attention.qk_clip")?,
            rel_clip: f("attention.relative_clip")?,
            log_floor: u("attention.log_scale_floor")? as f32,
            log_alpha: f("attention.log_scale_alpha")?,
            nc: u("shortconv.l_cache")?,
            block: u("attn_res.block_size")?,
            loop_start: u("loop.start")?,
            loop_end: u("loop.end")?,
            passes: u("loop.count")?,
            vocab_padded: u("vocab_size")?,
            vocab: u("vocab_size_unpadded")?,
            softcap: g.hp_f32("final_logit_softcapping").unwrap_or(0.0),
            exp_scale: g.hp_f32("expert_weights_scale").unwrap_or(1.0),
            situ_g: f("activation.situ_beta")?,
            situ_u: f("activation.situ_linear_beta")?,
        };
        if c.hd != 128 || c.n_rel > 32 || c.n_swa != c.extent || c.loop_start == 0 || c.loop_end > n_phys || c.passes < 1 {
            return Err(format!("この版が扱わない形: {c:?}"));
        }
        if g.hp("loop.varying").and_then(|v| v.as_bool()).unwrap_or(false) {
            return Err("loop.varying（回の数が変わる版）は扱わない".into());
        }
        Ok(c)
    }
    /// 実行（KV と畳み込みの状態の番号）の数
    pub fn n_slot(&self) -> usize {
        self.n_phys + (self.passes - 1) * (self.loop_end - self.loop_start)
    }
    fn qd(&self) -> usize {
        self.n_head * self.hd
    }
    fn kvd(&self) -> usize {
        self.n_kv * self.hd
    }
    /// 畳み込みの状態（1 実行ぶん）: k・v・Attention の出口・FFN の出口の (nc-1) 語
    fn conv_floats(&self) -> usize {
        (self.nc - 1) * (2 * self.kvd() + 2 * self.d)
    }
}

/// GPU の上の GGUF の並びのままの行列（qt は ggml の型番号。n 行 × k）
pub struct W {
    pub buf: DevBuf,
    pub qt: i32,
    pub k: usize,
    pub n: usize,
}

impl W {
    fn new(tv: gguf::TensorView) -> W {
        let qt = tv.qt();
        W { buf: DevBuf::from_bytes(tv.bytes), qt: qt as i32, k: tv.k(), n: tv.n_rows() }
    }
    fn gemv(&self, x: *const f32, y: *mut f32) {
        ck(unsafe { ffi::infer_gemv(self.qt, self.buf.ptr, self.k as i32, self.n as i32, x, y, 1, S()) }, "ink gemv");
    }
}

/// g3 の並びに詰め直した行列（kernels/gemv3.cu）。qt は詰め直した後の型。専門家の束なら es は 1 つぶんのバイト数。
/// k が 256 の倍数でない行列（専門家の down の 1152）は、行の後ろに 0 のかたまりを足して kp（256 の倍数）にそろえる
pub struct M3 {
    pub buf: DevBuf,
    pub qt: i32,
    pub k: usize,
    pub n: usize,
    pub es: i64,
}

impl M3 {
    /// rows_e: 専門家 1 つの行の数（束でなければ全部の行）
    fn new(tv: gguf::TensorView, rows_e: usize, stage: &DevBuf) -> M3 {
        let src = tv.qt();
        let qt = src as i32;
        let (k, n) = (tv.k(), tv.n_rows());
        let kp = k.div_ceil(256) * 256;
        let rb = tv.row_bytes();
        let rbp = unsafe { ffi::infer_row_bytes(qt, kp as i64) } as usize;
        assert!(kp == k || matches!(src, QType::Q8_0 | QType::IQ4NL), "{}: 0 のかたまりで k をそろえられない型 {}", tv.name, tv.type_name());
        let bytes = unsafe { ffi::infer_g3_bytes(qt, kp as i32, n as i32) };
        assert!(bytes > 0, "{}: g3 が扱わない型 {}（k {kp}）", tv.name, tv.type_name());
        let buf = DevBuf::new(bytes as usize);
        let tile = unsafe { ffi::infer_g3_bytes(qt, kp as i32, 32) } as usize;
        let step = (stage.len / rbp / 32 * 32).max(32);
        let mut host = vec![0u8; if kp == k { 0 } else { step * rbp }];
        let mut r0 = 0;
        while r0 < n {
            let nr = step.min(n - r0);
            let src_rows = tv.rows(r0, nr);
            if kp == k {
                k_h2d(stage.ptr, src_rows);
            } else {
                for r in 0..nr {
                    host[r * rbp..r * rbp + rb].copy_from_slice(&src_rows[r * rb..(r + 1) * rb]);
                    host[r * rbp + rb..(r + 1) * rbp].fill(0);
                }
                k_h2d(stage.ptr, &host[..nr * rbp]);
            }
            let dst = unsafe { buf.ptr.cast::<u8>().add(r0 / 32 * tile) };
            ck(unsafe { ffi::infer_g3_repack(qt, stage.ptr, dst.cast(), kp as i32, nr as i32, S()) }, "g3 repack");
            r0 += nr;
        }
        let es = if rows_e < n { unsafe { ffi::infer_g3_bytes(qt, kp as i32, rows_e as i32) } } else { 0 };
        M3 { buf, qt: unsafe { ffi::infer_g3_type(qt) }, k: kp, n: rows_e.min(n), es }
    }
}

fn k_h2d(dst: *mut std::ffi::c_void, src: &[u8]) {
    ck(unsafe { ffi::cudaMemcpy(dst, src.as_ptr().cast(), src.len(), 1) }, "ink H2D");
}

fn up_i32(b: &DevBuf, v: &[i32]) {
    assert!(v.len() * 4 <= b.len);
    ck(unsafe { ffi::cudaMemcpy(b.ptr, v.as_ptr().cast(), v.len() * 4, 1) }, "ink H2D i32");
}

fn f32buf(tv: gguf::TensorView) -> DevBuf {
    assert_eq!(tv.qtype, Some(QType::F32), "{}: f32 のはず（{}）", tv.name, tv.type_name());
    DevBuf::from_bytes(tv.bytes)
}

pub struct InkLayer {
    pub swa: bool,
    pub wq: M3,
    pub wk: M3,
    pub wv: M3,
    pub wo: M3,
    pub wgate: M3,
    pub wrel: M3,
    /// sink_proj を f32 [H][d] に戻した物（行が 24 しかないので、1 行 1 ブロックの f32 の GEMV で）
    pub wsink: DevBuf,
    pub k_dyn: Option<M3>,
    pub v_dyn: Option<M3>,
    pub q_clip: DevBuf,
    pub rel_bias: DevBuf,
    pub rel_clip: DevBuf,
    pub k_conv: DevBuf,
    pub v_conv: DevBuf,
    pub attn_post_conv: DevBuf,
    pub ffn_post_conv: DevBuf,
    /// 窓の層: [attn_norm]、全体の層: [q, k, v]
    pub norms: Vec<DevBuf>,
    /// 層 0 は無い。窓の層は 1 本・全体の層は 3 本（[n][d]）
    pub attn_res_q: Option<(DevBuf, usize)>,
    pub ffn_res_q: DevBuf,
    pub ffn_norm: DevBuf,
    /// f32 [n_exp][d]
    pub router: DevBuf,
    pub exp_bias: Option<DevBuf>,
    pub gate_up: M3,
    pub down: M3,
}

/// ループの層の伝言の門（q・k・v・FFN の 4 つ）
pub struct InkLoopW {
    /// [4][d]
    pub msg_q: DevBuf,
    /// f32 [rank][2d]・[d][rank]・[d][rank]
    pub down: Vec<DevBuf>,
    pub gain: Vec<DevBuf>,
    pub forget: Vec<DevBuf>,
    pub rank: usize,
}

/// 列の状態（lanes 本ぶん）: 実行ごとの KV（fp16 [lane][cap][kvh][128]）と畳み込みの状態（[lane][conv_floats]）。
/// 1 本の列は lanes = 1。デコードは lane 0..B を 1 歩にまとめる（step_batch）、プレフィルは lane ごと
pub struct InkSeq {
    pub lanes: usize,
    pub pos: Vec<usize>,
    pub max_ctx: usize,
    kc: Vec<DevBuf>,
    vc: Vec<DevBuf>,
    /// int8 のときの係数（128 要素ごと）。fp16 なら空
    ks: Vec<DevBuf>,
    vs: Vec<DevBuf>,
    pub kv8: bool,
    cap: Vec<usize>,
    conv: Vec<DevBuf>,
    conv_floats: usize,
    kvd: usize,
    /// B 本の 1 歩を撮った CUDA Graph（B ごと）
    graphs: std::collections::HashMap<usize, usize>,
}

impl InkSeq {
    pub fn reset(&mut self) {
        ck(unsafe { kffi::cudaStreamSynchronize(S()) }, "sync");
        self.pos.iter_mut().for_each(|p| *p = 0);
        for c in &self.conv {
            ck(unsafe { kffi::cudaMemset(c.ptr, 0, c.len) }, "conv の状態を 0 に");
        }
    }
    /// lane b だけ初めに戻す
    pub fn reset_lane(&mut self, b: usize) {
        ck(unsafe { kffi::cudaStreamSynchronize(S()) }, "sync");
        self.pos[b] = 0;
        for c in &self.conv {
            let p = unsafe { c.ptr.cast::<f32>().add(b * self.conv_floats) };
            ck(unsafe { kffi::cudaMemset(p.cast(), 0, self.conv_floats * 4) }, "conv の状態を 0 に");
        }
    }
    pub fn bytes(&self) -> usize {
        self.kc.iter().chain(&self.vc).chain(&self.ks).chain(&self.vs).chain(&self.conv).map(|b| b.len).sum()
    }

    fn eb(&self) -> usize {
        if self.kv8 { 1 } else { 2 }
    }

    /// 実行 il の lane の KV（読む側）
    fn kv_ref(&self, il: usize, lane: usize) -> KvRef {
        let o = lane * self.cap[il] * self.kvd;
        unsafe {
            KvRef {
                k: self.kc[il].ptr.cast::<u8>().add(o * self.eb()).cast(),
                v: self.vc[il].ptr.cast::<u8>().add(o * self.eb()).cast(),
                ks: if self.kv8 { self.ks[il].f32p().add(o / 128) } else { std::ptr::null() },
                vs: if self.kv8 { self.vs[il].f32p().add(o / 128) } else { std::ptr::null() },
                kv8: self.kv8 as i32,
            }
        }
    }
    fn kv_out(&self, il: usize, lane: usize) -> KvOut {
        let r = self.kv_ref(il, lane);
        KvOut { k: r.k as *mut _, v: r.v as *mut _, ks: r.ks as *mut _, vs: r.vs as *mut _, kv8: r.kv8 }
    }

    /// save_lane(_, len) が取る GPU の大きさ（バイト）
    pub fn snap_bytes(&self, len: usize) -> usize {
        let eb = self.eb();
        (0..self.kc.len())
            .map(|il| {
                let n = if self.cap[il] < self.max_ctx { self.cap[il] * self.kvd } else { len * self.kvd };
                2 * n * eb + if self.kv8 { 2 * (n / 128) * 4 } else { 0 } + self.conv_floats * 4
            })
            .sum()
    }

    /// lane の状態を写す（頭の使い回しの控え）。全体の層の KV は len 語まで、窓の層の輪と畳み込みの状態は全部
    pub fn save_lane(&self, lane: usize, len: usize) -> LaneSnap {
        ck(unsafe { kffi::cudaStreamSynchronize(S()) }, "sync");
        let eb = self.eb();
        let mut parts = vec![];
        for il in 0..self.kc.len() {
            let row = self.cap[il] * self.kvd;
            let n = if self.cap[il] < self.max_ctx { row } else { len * self.kvd };
            let off = lane * row;
            let take = |b: &DevBuf, e0: usize, ne: usize, es: usize| {
                let d = DevBuf::new((ne * es).max(16));
                ck(unsafe { ffi::cudaMemcpy(d.ptr, b.ptr.cast::<u8>().add(e0 * es).cast(), ne * es, 3) }, "snap");
                d
            };
            let k = take(&self.kc[il], off, n, eb);
            let v = take(&self.vc[il], off, n, eb);
            let (ks, vs) = if self.kv8 { (Some(take(&self.ks[il], off / 128, n / 128, 4)), Some(take(&self.vs[il], off / 128, n / 128, 4))) } else { (None, None) };
            let cv = take(&self.conv[il], lane * self.conv_floats, self.conv_floats, 4);
            parts.push((n, k, v, ks, vs, cv));
        }
        LaneSnap { len, parts }
    }

    /// 控えを lane に戻す（pos = 控えの長さ）
    pub fn restore_lane(&mut self, lane: usize, sn: &LaneSnap) {
        ck(unsafe { kffi::cudaStreamSynchronize(S()) }, "sync");
        let eb = self.eb();
        for (il, (n, k, v, ks, vs, cv)) in sn.parts.iter().enumerate() {
            let off = lane * self.cap[il] * self.kvd;
            let put = |dst: &DevBuf, e0: usize, src: &DevBuf, ne: usize, es: usize| {
                ck(unsafe { ffi::cudaMemcpy(dst.ptr.cast::<u8>().add(e0 * es).cast(), src.ptr, ne * es, 3) }, "restore");
            };
            put(&self.kc[il], off, k, *n, eb);
            put(&self.vc[il], off, v, *n, eb);
            if let (Some(a), Some(b)) = (ks, vs) {
                put(&self.ks[il], off / 128, a, n / 128, 4);
                put(&self.vs[il], off / 128, b, n / 128, 4);
            }
            put(&self.conv[il], lane * self.conv_floats, cv, self.conv_floats, 4);
        }
        self.pos[lane] = sn.len;
    }
}

/// lane の状態の写し（実行ごとに: 要素の数・K・V・係数・畳み込みの状態）
pub struct LaneSnap {
    pub len: usize,
    parts: Vec<(usize, DevBuf, DevBuf, Option<DevBuf>, Option<DevBuf>, DevBuf)>,
}

impl LaneSnap {
    pub fn bytes(&self) -> usize {
        self.parts.iter().map(|p| p.1.len + p.2.len + p.3.as_ref().map_or(0, |b| b.len) + p.4.as_ref().map_or(0, |b| b.len) + p.5.len).sum()
    }
}


struct Scratch {
    /// 行（語）の数の上限（プレフィルの 1 塊）
    tm: usize,
    x: DevBuf,
    pre: DevBuf,
    blk: Vec<DevBuf>,
    mixed: DevBuf,
    inp: DevBuf,
    q: DevBuf,
    kraw: DevBuf,
    kk: DevBuf,
    vraw: DevBuf,
    v: DevBuf,
    r: DevBuf,
    sink: DevBuf,
    g: DevBuf,
    dk: DevBuf,
    dv: DevBuf,
    att: DevBuf,
    oraw: DevBuf,
    o: DevBuf,
    ffn_in: DevBuf,
    m: DevBuf,
    lg: DevBuf,
    ids: DevBuf,
    wts: DevBuf,
    gu: DevBuf,
    h: DevBuf,
    o8: DevBuf,
    z: DevBuf,
    cg: DevBuf,
    msgs: DevBuf,
    logits: DevBuf,
    part: DevBuf,
    pos: DevBuf,
    tau: DevBuf,
    tok: DevBuf,
    /// デコードの行 → 列（lane）
    lanes: DevBuf,
    /// g3 の x の下ごしらえ（q・k・v の入口、Attention の出口、専門家の h）
    xp: Vec<DevBuf>,
    xp_o: DevBuf,
    xp_h: DevBuf,
    /// GEMM の x（fp16 [tm][3072 まで]）
    x16: DevBuf,
    /// MoE の仕事の並び。デコード（1 語）は固定の jr0・jcnt・xidx、プレフィルは手元で専門家ごとにまとめて書く
    jr0: DevBuf,
    jcnt: DevBuf,
    xidx: DevBuf,
    pje: DevBuf,
    pjr0: DevBuf,
    pjcnt: DevBuf,
    pxidx: DevBuf,
    rowmap: DevBuf,
    grp: DevBuf,
    h16: DevBuf,
    ffp: usize,
}


pub struct Ink {
    pub cfg: InkCfg,
    pub embd: W,
    pub tok_norm: DevBuf,
    pub out: M3,
    pub out_norm: DevBuf,
    pub out_res_q: DevBuf,
    pub layers: Vec<InkLayer>,
    pub loops: Vec<InkLoopW>,
    sc: Scratch,
    attn_chunk: usize,
    max_split: usize,
    /// デコードを 1 歩にまとめる本数の上限（g3 の GEMV の B）
    pub max_lanes: usize,
    pub weight_bytes: usize,
}

const BLK_MAX: usize = 8;
/// プレフィルの 1 塊の語の数（窓の層の輪は n_swa + これ）
pub const PF_CHUNK: usize = 512;
/// これより少ない語は GEMV（g3 の B の上限 32）、多ければ GEMM
const GEMV_B: usize = 32;
/// プレフィルの専門家の GEMM のタイルの語の数（kernels/gemmg3.cu の 16・32・64・128）
const MOE_TOKB: usize = 64;

impl Ink {
    pub fn load(g: &gguf::Model, max_ctx: usize) -> Result<Ink, String> {
        let cfg = InkCfg::from_gguf(g)?;
        unsafe {
            if STREAM.is_null() {
                let mut st: ffi::Stream = std::ptr::null_mut();
                ck(kffi::cudaStreamCreateWithFlags(&mut st, 0), "流れを作る");
                STREAM = st;
            }
        }
        let stage = DevBuf::new(64 << 20);
        let mut bytes = 0usize;
        let mut w = |tv: gguf::TensorView| {
            bytes += tv.bytes.len();
            W::new(tv)
        };
        let mut gb = 0usize;
        let mut m3 = |tv: gguf::TensorView, rows_e: usize| {
            let m = M3::new(tv, rows_e, &stage);
            gb += m.buf.len;
            m
        };
        let embd = w(g.tensor("token_embd.weight"));
        let ot = g.tensor("output.weight");
        let out = m3(ot, ot.n_rows());
        let mut layers = vec![];
        let mut loops = vec![];
        for il in 0..cfg.n_phys {
            let l = g.layer(il);
            let swa = cfg.swa[il];
            let norms = if swa {
                vec![f32buf(l.tensor("attn_norm.weight"))]
            } else {
                vec![f32buf(l.tensor("attn_q_inp_norm.weight")), f32buf(l.tensor("attn_k_inp_norm.weight")), f32buf(l.tensor("attn_v_inp_norm.weight"))]
            };
            let attn_res_q = l.get("attn_res_query.weight").map(|tv| {
                let nq = if tv.shape.len() > 1 { tv.shape[1] } else { 1 };
                (f32buf(tv), nq)
            });
            if il > 0 && attn_res_q.is_none() {
                return Err(format!("層 {il} に attn_res_query が無い"));
            }
            let ge = l.tensor("ffn_gate_up_exps.weight");
            let de = l.tensor("ffn_down_exps.weight");
            if ge.shape != [cfg.d, 2 * cfg.ff, cfg.n_exp] || de.shape != [cfg.ff, cfg.d, cfg.n_exp] {
                return Err(format!("層 {il} の専門家の形 {:?} {:?}", ge.shape, de.shape));
            }
            let mut dm = |name: &str| {
                let tv = l.tensor(name);
                let n = tv.n_rows();
                m3(tv, n)
            };
            let (wq, wk, wv, wo, wgate, wrel) =
                (dm("attn_q.weight"), dm("attn_k.weight"), dm("attn_v.weight"), dm("attn_output.weight"), dm("attn_gate.weight"), dm("attn_rel.weight"));
            let (k_dyn, v_dyn) = if swa { (None, None) } else { (Some(dm("attn_k_conv_proj.weight")), Some(dm("attn_v_conv_proj.weight"))) };
            let gate_up = m3(ge, 2 * cfg.ff);
            let down = m3(de, cfg.d);
            layers.push(InkLayer {
                swa,
                wq,
                wk,
                wv,
                wo,
                wgate,
                wrel,
                wsink: {
                    let tv = l.tensor("attn_sink_proj.weight");
                    let wq = W::new(tv);
                    let out = DevBuf::new(wq.k * wq.n * 4);
                    ck(unsafe { ffi::infer_dequant(wq.qt, wq.buf.ptr, wq.k as i32, wq.n as i64, out.f32p(), S()) }, "sink deq");
                    out
                },
                k_dyn,
                v_dyn,
                q_clip: f32buf(l.tensor("attn_q_clip.weight")),
                rel_bias: f32buf(l.tensor("attn_rel_bias.weight")),
                rel_clip: f32buf(l.tensor("attn_rel_clip.weight")),
                k_conv: f32buf(l.tensor("attn_k_conv.weight")),
                v_conv: f32buf(l.tensor("attn_v_conv.weight")),
                attn_post_conv: f32buf(l.tensor("attn_post_conv.weight")),
                ffn_post_conv: f32buf(l.tensor("ffn_post_conv.weight")),
                norms,
                attn_res_q,
                ffn_res_q: f32buf(l.tensor("ffn_res_query.weight")),
                ffn_norm: f32buf(l.tensor("ffn_norm.weight")),
                router: f32buf(l.tensor("ffn_gate_inp.weight")),
                exp_bias: l.get("exp_probs_b.bias").map(f32buf),
                gate_up,
                down,
            });
            if il >= cfg.loop_start && il < cfg.loop_end && cfg.passes > 1 {
                let names = ["q", "k", "v", "ffn"];
                let dn = l.tensor("loop_q_ctrl_down.weight");
                loops.push(InkLoopW {
                    msg_q: f32buf(l.tensor("loop_msg_query.weight")),
                    rank: dn.n_rows(),
                    down: names.iter().map(|n| f32buf(l.tensor(&format!("loop_{n}_ctrl_down.weight")))).collect(),
                    gain: names.iter().map(|n| f32buf(l.tensor(&format!("loop_{n}_ctrl_gain.weight")))).collect(),
                    forget: names.iter().map(|n| f32buf(l.tensor(&format!("loop_{n}_ctrl_forget.weight")))).collect(),
                });
            }
        }
        drop(m3);
        let d = cfg.d;
        let f = |n: usize| DevBuf::new(n * 4);
        let attn_chunk = 256;
        let max_split = max_ctx.div_ceil(attn_chunk).max(1);
        let part = unsafe { kffi::ink_attn_part_floats(GEMV_B as i32, cfg.n_head as i32, max_split as i32) } as usize;
        let n_loop = cfg.loop_end - cfg.loop_start;
        let ffp = cfg.ff.div_ceil(256) * 256;
        let xpb = |k: usize, b: usize| DevBuf::new(unsafe { ffi::infer_g3_xprep_bytes(k as i32, b as i32) } as usize);
        let kk = cfg.k_exp;
        let tm = PF_CHUNK;
        let r = tm * kk;
        let sc = Scratch {
            tm,
            x: f(tm * d),
            pre: f(tm * d),
            blk: (0..BLK_MAX).map(|_| f(tm * d)).collect(),
            mixed: f(tm * 3 * d),
            inp: f(3 * tm * d),
            q: f(tm * cfg.qd()),
            kraw: f(tm * cfg.kvd()),
            kk: f(tm * cfg.kvd()),
            vraw: f(tm * cfg.kvd()),
            v: f(tm * cfg.kvd()),
            r: f(tm * cfg.n_head * cfg.n_rel),
            sink: f(tm * cfg.n_head),
            g: f(tm * cfg.qd()),
            dk: f(tm * cfg.n_kv * cfg.nc),
            dv: f(tm * cfg.n_kv * cfg.nc),
            att: f(tm * cfg.qd()),
            oraw: f(tm * d),
            o: f(tm * d),
            ffn_in: f(tm * d),
            m: f(tm * d),
            lg: f(tm * cfg.n_exp),
            ids: f(r),
            wts: f(r),
            gu: f(r * 2 * cfg.ff),
            h: f(r * ffp),
            o8: f(r * d),
            z: f(tm * 2 * d),
            cg: f(tm * 256),
            msgs: f(n_loop * tm * 4 * d),
            logits: f(GEMV_B * cfg.vocab_padded),
            part: f(part),
            pos: f(tm),
            tau: f(tm),
            tok: f(tm),
            lanes: f(GEMV_B),
            xp: (0..3).map(|_| xpb(d, tm)).collect(),
            xp_o: xpb(cfg.qd(), GEMV_B),
            xp_h: xpb(ffp, r.max(GEMV_B * kk)),
            x16: DevBuf::new(tm * cfg.qd().max(d) * 2),
            jr0: DevBuf::from_i32(&(0..(GEMV_B * kk) as i32).collect::<Vec<_>>()),
            jcnt: DevBuf::from_i32(&vec![1; GEMV_B * kk]),
            xidx: DevBuf::from_i32(&(0..(GEMV_B * kk) as i32).map(|j| j / kk as i32).collect::<Vec<_>>()),
            pje: f(r),
            pjr0: f(r),
            pjcnt: f(r),
            pxidx: f(r),
            rowmap: f(r),
            grp: f(3 * (cfg.n_exp + r / MOE_TOKB)),
            h16: DevBuf::new(r * ffp * 2),
            ffp,
        };
        ck(unsafe { kffi::cudaMemset(sc.h.ptr, 0, sc.h.len) }, "h を 0 に");
        Ok(Ink {
            tok_norm: f32buf(g.tensor("token_embd_norm.weight")),
            out_norm: f32buf(g.tensor("output_norm.weight")),
            out_res_q: f32buf(g.tensor("output_res_query.weight")),
            cfg,
            embd,
            out,
            layers,
            loops,
            sc,
            attn_chunk,
            max_split,
            max_lanes: GEMV_B,
            weight_bytes: bytes + gb,
        })
    }

    pub fn new_seq(&self, max_ctx: usize) -> InkSeq {
        self.new_batch(max_ctx, 1)
    }

    /// lanes 本の列（デコードを lanes 本まで 1 歩にまとめられる）
    pub fn new_batch(&self, max_ctx: usize, lanes: usize) -> InkSeq {
        let c = &self.cfg;
        assert!(max_ctx <= self.max_split * self.attn_chunk, "max_ctx {max_ctx} が load の値より大きい");
        assert!(lanes >= 1 && lanes <= self.max_lanes, "lanes {lanes} は 1..{}", self.max_lanes);
        let (mut kc, mut vc, mut cap, mut conv, mut ks, mut vs) = (vec![], vec![], vec![], vec![], vec![], vec![]);
        let kv8 = kv8_on();
        for il in 0..c.n_slot() {
            let ip = self.phys(il);
            // 窓の層は輪（プレフィルの 1 塊が窓の古い鍵を消さないよう、窓 + 1 塊）
            let n = if c.swa[ip] { (c.n_swa + PF_CHUNK).min(max_ctx) } else { max_ctx };
            let e = lanes * n * c.kvd();
            kc.push(DevBuf::new(e * if kv8 { 1 } else { 2 }));
            vc.push(DevBuf::new(e * if kv8 { 1 } else { 2 }));
            if kv8 {
                ks.push(DevBuf::new(e / 128 * 4));
                vs.push(DevBuf::new(e / 128 * 4));
            }
            cap.push(n);
            conv.push(DevBuf::new(lanes * c.conv_floats() * 4));
        }
        let mut s = InkSeq { lanes, pos: vec![0; lanes], max_ctx, kc, vc, ks, vs, kv8, cap, conv, conv_floats: c.conv_floats(), kvd: c.kvd(), graphs: Default::default() };
        s.reset();
        s
    }

    /// 実行 il の物理の層
    pub fn phys(&self, il: usize) -> usize {
        let c = &self.cfg;
        if il < c.n_phys { il } else { c.loop_start + (il - c.n_phys) % (c.loop_end - c.loop_start) }
    }

    /// 深さの Attention: 状態（どれも [t][d]・行の間隔 d）を q（[nq][d]）で混ぜて y [t][nq][d]
    fn mix(&self, states: &[*const f32], q: *const f32, nq: usize, t: usize, y: *mut f32) {
        let d = self.cfg.d as i64;
        let lds = vec![d; states.len()];
        ck(unsafe { kffi::ink_mix(states.as_ptr(), lds.as_ptr(), states.len() as i32, q, nq as i32, d as i32, t as i32, y, nq as i64 * d, S()) }, "ink mix");
    }

    /// RMSNorm（重みつき）。x・y は [t][d]
    fn rms(&self, x: *const f32, w: *const f32, t: usize, y: *mut f32) {
        let d = self.cfg.d;
        ck(unsafe { kffi::ink_rms(x, d as i64, w, d as i32, t as i32, self.cfg.eps, y, d as i64, S()) }, "ink rms");
    }

    /// 伝言の門: y = (1 - tanh(F z))·h + tanh(G z)·m、z = down·[rms(h); rms(m)]。h・m・y は [t][d]（m の行の間隔は mld）
    fn control(&self, lw: &InkLoopW, c: usize, h: *const f32, m: *const f32, mld: usize, t: usize, y: *mut f32) {
        let sc = &self.sc;
        let d = self.cfg.d;
        let rk = lw.rank;
        unsafe {
            ck(kffi::ink_cat2_rms(h, d as i64, m, mld as i64, d as i32, t as i32, self.cfg.eps, sc.z.f32p(), S()), "ink cat2");
            if t <= 8 {
                ck(kffi::ink_gemv_f32(lw.down[c].f32p(), 2 * d as i32, rk as i32, sc.z.f32p(), 2 * d as i64, t as i32, sc.cg.f32p(), rk as i64, S()), "ink ctrl down");
            } else {
                ck(kffi::ink_gemm_f32(lw.down[c].f32p(), 2 * d as i32, rk as i32, sc.z.f32p(), 2 * d as i64, t as i32, sc.cg.f32p(), rk as i64, S()), "ink ctrl down");
            }
            ck(kffi::ink_ctrl2(lw.gain[c].f32p(), lw.forget[c].f32p(), rk as i32, sc.cg.f32p(), h, d as i64, m, mld as i64, d as i32, t as i32, y, d as i64, S()),
               "ink ctrl2");
        }
    }

    /// 小さな f32 の行列（ルーター・sink）: y [t][n] = W x
    fn f32mm(&self, w: &DevBuf, kd: usize, n: usize, x: *const f32, t: usize, y: *mut f32) {
        let e = unsafe {
            if t <= 8 {
                kffi::ink_gemv_f32(w.f32p(), kd as i32, n as i32, x, kd as i64, t as i32, y, n as i64, S())
            } else {
                kffi::ink_gemm_f32(w.f32p(), kd as i32, n as i32, x, kd as i64, t as i32, y, n as i64, S())
            }
        };
        ck(e, "ink f32mm");
    }

    fn xprep(&self, x: *const f32, kd: usize, b: usize, xp: &DevBuf) {
        ck(unsafe { ffi::infer_g3_xprep(x, kd as i32, b as i32, xp.ptr, S()) }, "ink xprep");
    }

    /// 同じ x（[t][k] f32）の行列を 4 つまで: ys[i] [t][n_i] = W_i x。t <= 32 は GEMV（xp に下ごしらえ）、多ければ fp16 の GEMM
    fn mm(&self, mats: &[&M3], ys: &[*mut f32], x: *const f32, t: usize, xp: &DevBuf) {
        let kd = mats[0].k;
        let ts: Vec<i32> = mats.iter().map(|m| m.qt).collect();
        let ns: Vec<i32> = mats.iter().map(|m| m.n as i32).collect();
        let ws: Vec<*const std::ffi::c_void> = mats.iter().map(|m| m.buf.ptr as *const std::ffi::c_void).collect();
        if t <= GEMV_B {
            self.xprep(x, kd, t, xp);
            ck(
                unsafe { ffi::infer_g3_gemv_multi(mats.len() as i32, ts.as_ptr(), ws.as_ptr(), ns.as_ptr(), ys.as_ptr(), kd as i32, xp.ptr, t as i32, 0, 0, S()) },
                "ink g3 gemv",
            );
        } else {
            let sc = &self.sc;
            ck(unsafe { kffi::infer_m_f32_to_f16(x, (t * kd) as i64, sc.x16.ptr, S()) }, "ink f16");
            let yv: Vec<*mut std::ffi::c_void> = ys.iter().map(|p| p.cast()).collect();
            ck(
                unsafe {
                    kffi::infer_g3_gemm_ex(mats.len() as i32, ts.as_ptr(), ws.as_ptr(), ns.as_ptr(), yv.as_ptr(), null(), kd as i32, t as i32, sc.x16.ptr,
                                           null(), kd as i32, 0, -1, S())
                },
                "ink g3 gemm",
            );
        }
    }

    /// 専門家の束の GEMV。仕事 j は専門家 je[j] の行 jr0[j]..+jcnt[j]（rx は x の行の数・xidx は仕事の行 → x の行、null なら同じ）
    #[allow(clippy::too_many_arguments)]
    fn moe(&self, m: &M3, xp: &DevBuf, rx: usize, xidx: *const i32, je: *const i32, jr0: *const i32, jcnt: *const i32, n_jobs: usize, bmax: usize,
           y: *mut f32) {
        let (q, w, es, n, ys) = ([m.qt], [m.buf.ptr as *const std::ffi::c_void], [m.es], [m.n as i32], [y]);
        ck(
            unsafe {
                kffi::infer_g3_moe_gemv(1, q.as_ptr(), w.as_ptr(), es.as_ptr(), n.as_ptr(), ys.as_ptr(), m.k as i32, xp.ptr, rx as i32, xidx, je, jr0, jcnt,
                                        n_jobs as i32, bmax as i32, S())
            },
            "ink g3 moe",
        );
    }

    fn copy(dst: *mut f32, src: *const f32, n: usize) {
        ck(unsafe { kffi::cudaMemcpyAsync(dst.cast(), src.cast(), n * 4, 3, S()) }, "ink copy");
    }

    /// 1 語を入れて logits（GPU の上、頭の vocab 個・softcap の後）を作る（lane 0）。seq.pos を 1 進める
    pub fn step(&mut self, seq: &mut InkSeq, tok: u32) {
        self.step_batch(seq, &[tok]);
    }

    /// lane 0..B に 1 語ずつ（B = toks.len()）。logits は行 b に
    pub fn step_batch(&mut self, seq: &mut InkSeq, toks: &[u32]) {
        let lanes: Vec<usize> = (0..toks.len()).collect();
        self.step_lanes(seq, &lanes, toks);
    }

    /// 列 lanes[r] に 1 語ずつ（行 r）。logits は行 r に。1 歩は CUDA Graph に撮って流す（B ごと。INK_GRAPH=0 で撮らない）
    pub fn step_lanes(&mut self, seq: &mut InkSeq, lanes: &[usize], toks: &[u32]) {
        let b = toks.len();
        assert!(b >= 1 && b <= seq.lanes && lanes.len() == b);
        for &l in lanes {
            assert!(l < seq.lanes && seq.pos[l] < seq.max_ctx, "lane {l} の文脈が一杯（{}）", seq.max_ctx);
        }
        let tk: Vec<i32> = toks.iter().map(|&t| t as i32).collect();
        let ps: Vec<i32> = lanes.iter().map(|&l| seq.pos[l] as i32).collect();
        let ln: Vec<i32> = lanes.iter().map(|&l| l as i32).collect();
        // 手元の Vec から写すので同期で（非同期だと写す前に Vec が消える）
        ck(unsafe { kffi::cudaStreamSynchronize(S()) }, "sync");
        for (buf, v) in [(&self.sc.tok, &tk), (&self.sc.pos, &ps), (&self.sc.lanes, &ln)] {
            ck(unsafe { ffi::cudaMemcpy(buf.ptr, v.as_ptr().cast(), b * 4, 1) }, "tok・pos・lanes");
        }
        let use_graph = std::env::var("INK_GRAPH").map_or(true, |v| v != "0");
        if !use_graph {
            self.forward(seq, b, None);
        } else {
            if !seq.graphs.contains_key(&b) {
                let mut g: *mut std::ffi::c_void = std::ptr::null_mut();
                let mut e: *mut std::ffi::c_void = std::ptr::null_mut();
                unsafe {
                    ck(kffi::cudaStreamBeginCapture(S(), 2), "graph begin");
                    self.forward(seq, b, None);
                    ck(kffi::cudaStreamEndCapture(S(), &mut g), "graph end");
                    ck(kffi::cudaGraphInstantiateWithFlags(&mut e, g, 0), "graph instantiate");
                }
                seq.graphs.insert(b, e as usize);
            }
            ck(unsafe { kffi::cudaGraphLaunch(seq.graphs[&b] as *mut std::ffi::c_void, S()) }, "graph launch");
        }
        for &l in lanes {
            seq.pos[l] += 1;
        }
    }

    /// lane 0 の文をまとめて読む（PF_CHUNK 語ずつ）。最後の語の logits が行 0 に残る
    pub fn prefill(&mut self, seq: &mut InkSeq, toks: &[u32]) {
        self.prefill_lane(seq, 0, toks);
    }

    /// lane の文をまとめて読む。最後の語の logits が行 0 に残る
    pub fn prefill_lane(&mut self, seq: &mut InkSeq, lane: usize, toks: &[u32]) {
        for ch in toks.chunks(PF_CHUNK) {
            assert!(seq.pos[lane] + ch.len() <= seq.max_ctx, "文脈が一杯（{}）", seq.max_ctx);
            let tk: Vec<i32> = ch.iter().map(|&t| t as i32).collect();
            let ps: Vec<i32> = (0..ch.len()).map(|i| (seq.pos[lane] + i) as i32).collect();
            ck(unsafe { kffi::cudaMemcpyAsync(self.sc.tok.ptr, tk.as_ptr().cast(), tk.len() * 4, 1, S()) }, "tok");
            ck(unsafe { kffi::cudaMemcpyAsync(self.sc.pos.ptr, ps.as_ptr().cast(), ps.len() * 4, 1, S()) }, "pos");
            self.forward(seq, ch.len(), Some(lane));
            seq.pos[lane] += ch.len();
        }
    }

    /// t 語の forward（tok・pos は GPU の上）。lane = Some(b): 列 b の t 語（プレフィル）、None: 列 0..t の 1 語ずつ（デコード・CUDA Graph に撮れる）
    fn forward(&self, seq: &InkSeq, t: usize, lane: Option<usize>) {
        let c = self.cfg.clone();
        let d = c.d;
        let sc = &self.sc;
        assert!(t <= sc.tm);
        ck(unsafe { kffi::infer_m_embed(self.embd.qt, self.embd.buf.ptr, d as i32, sc.tok.ptr.cast(), t as i32, sc.x.f32p(), S()) }, "ink embed");
        self.rms(sc.x.f32p(), self.tok_norm.f32p(), t, sc.x.f32p());
        ck(unsafe { kffi::ink_tau(sc.pos.ptr.cast(), t as i32, c.log_alpha, c.log_floor, sc.tau.f32p(), S()) }, "ink tau");
        // 塊の入口（深さの Attention の K・V）。blocks は sc.blk の番号
        let mut blocks: Vec<usize> = vec![];
        for il in 0..c.loop_start {
            self.layer(seq, il, il, &mut blocks, false, t, lane);
        }
        Self::copy(sc.pre.f32p(), sc.x.f32p(), t * d);
        let pre_blocks = blocks.clone();
        let n_loop = c.loop_end - c.loop_start;
        for pass in 0..c.passes {
            if pass > 0 {
                // 前の回の「最初の塊の後の塊」と出口から、この回の 12 層ぶんの伝言（q・k・v・FFN）を先に作る
                let bank: Vec<*const f32> =
                    blocks[pre_blocks.len() + 1..].iter().map(|&b| sc.blk[b].f32p() as *const f32).chain([sc.x.f32p() as *const f32]).collect();
                for j in 0..n_loop {
                    self.mix(&bank, self.loops[j].msg_q.f32p(), 4, t, unsafe { sc.msgs.f32p().add(j * sc.tm * 4 * d) });
                }
                Self::copy(sc.x.f32p(), sc.pre.f32p(), t * d);
                blocks = pre_blocks.clone();
            }
            for ip in c.loop_start..c.loop_end {
                let il = if pass == 0 { ip } else { c.n_phys + (pass - 1) * n_loop + (ip - c.loop_start) };
                self.layer(seq, il, ip, &mut blocks, pass > 0, t, lane);
            }
        }
        for il in c.loop_end..c.n_phys {
            self.layer(seq, il, il, &mut blocks, false, t, lane);
        }
        // 出口（プレフィルは最後の語だけ・デコードは全部の行）: 深さの Attention → RMSNorm → lm_head → softcap
        let (r0, nr) = if lane.is_some() { (t - 1, 1) } else { (0, t) };
        let off = r0 * d;
        let st: Vec<*const f32> = blocks.iter().map(|&b| unsafe { sc.blk[b].f32p().add(off) } as *const f32).chain([unsafe { sc.x.f32p().add(off) } as *const f32]).collect();
        self.mix(&st, self.out_res_q.f32p(), 1, nr, sc.m.f32p());
        self.rms(sc.m.f32p(), self.out_norm.f32p(), nr, sc.m.f32p());
        self.mm(&[&self.out], &[sc.logits.f32p()], sc.m.f32p(), nr, &sc.xp[0]);
        if c.softcap > 0.0 {
            ck(unsafe { kffi::ink_softcap(sc.logits.f32p(), c.vocab_padded as i64, c.vocab as i32, nr as i32, c.softcap, S()) }, "ink softcap");
        }
    }

    /// 実行 il（物理 ip）の 1 層・t 語。sc.x が入口で出口。伝言（msg）は sc.msgs の (ip - loop_start) 番
    #[allow(clippy::too_many_arguments)]
    fn layer(&self, seq: &InkSeq, il: usize, ip: usize, blocks: &mut Vec<usize>, msg: bool, t: usize, lane: Option<usize>) {
        let c = &self.cfg;
        let sc = &self.sc;
        let lw = &self.layers[ip];
        let d = c.d;
        let (qd, kvd) = (c.qd(), c.kvd());
        let tm = sc.tm;
        // 入口の混ぜ（層 0 は無し）
        let nq = lw.attn_res_q.as_ref().map_or(0, |a| a.1);
        let mut st: Vec<*const f32> = blocks.iter().map(|&b| sc.blk[b].f32p() as *const f32).collect();
        st.push(sc.x.f32p());
        let inp = |j: usize| unsafe { sc.inp.f32p().add(j * tm * d) };
        if let Some((q, _)) = &lw.attn_res_q {
            self.mix(&st, q.f32p(), nq, t, sc.mixed.f32p());
        }
        // 入口 j の頭と行の間隔（混ぜの出口は [t][nq][d]）
        let src = |j: usize| -> (*const f32, usize) {
            match &lw.attn_res_q {
                None => (sc.x.f32p() as *const f32, d),
                Some((_, n)) => (unsafe { sc.mixed.f32p().add(if *n == 1 { 0 } else { j * d }) } as *const f32, *n * d),
            }
        };
        let lp = if msg { Some(&self.loops[ip - c.loop_start]) } else { None };
        let msgp = |j: usize| unsafe { sc.msgs.f32p().add((ip - c.loop_start) * tm * 4 * d + j * d) };
        // q・k・v の入口: 伝言の門 → 正規化（窓の層で伝言なしは 3 つとも同じなので 1 回）
        let once = lw.swa && lp.is_none();
        for j in 0..if once { 1 } else { 3 } {
            let (sp, sld) = src(j);
            let nw = if lw.swa { &lw.norms[0] } else { &lw.norms[j] };
            if let Some(l) = lp {
                // 伝言の門は行の間隔 d の入口を要るので、混ぜの出口（間隔 nq·d）を先に inp に写す
                if sld != d {
                    self.copy_rows(inp(j), sp, sld, t);
                    self.control(l, j, inp(j), msgp(j), 4 * d, t, inp(j));
                } else {
                    self.control(l, j, sp, msgp(j), 4 * d, t, inp(j));
                }
                self.rms(inp(j), nw.f32p(), t, inp(j));
            } else {
                ck(unsafe { kffi::ink_rms(sp, sld as i64, nw.f32p(), d as i32, t as i32, c.eps, inp(j), d as i64, S()) }, "ink rms in");
            }
        }
        let (iq, ik, iv) = if once { (inp(0), inp(0), inp(0)) } else { (inp(0), inp(1), inp(2)) };
        let (xq, xk, xv) = (&sc.xp[0], &sc.xp[1], &sc.xp[2]);
        let (dk, dv) = match (&lw.k_dyn, &lw.v_dyn) {
            (Some(a), Some(b)) => {
                self.mm(&[&lw.wq, &lw.wgate, &lw.wrel], &[sc.q.f32p(), sc.g.f32p(), sc.r.f32p()], iq, t, xq);
                self.mm(&[&lw.wk, a], &[sc.kraw.f32p(), sc.dk.f32p()], ik, t, xk);
                self.mm(&[&lw.wv, b], &[sc.vraw.f32p(), sc.dv.f32p()], iv, t, xv);
                (sc.dk.f32p() as *const f32, sc.dv.f32p() as *const f32)
            }
            _ if once => {
                self.mm(&[&lw.wq, &lw.wgate, &lw.wrel, &lw.wk], &[sc.q.f32p(), sc.g.f32p(), sc.r.f32p(), sc.kraw.f32p()], iq, t, xq);
                self.mm(&[&lw.wv], &[sc.vraw.f32p()], iq, t, xq);
                (null(), null())
            }
            _ => {
                self.mm(&[&lw.wq, &lw.wgate, &lw.wrel], &[sc.q.f32p(), sc.g.f32p(), sc.r.f32p()], iq, t, xq);
                self.mm(&[&lw.wk], &[sc.kraw.f32p()], ik, t, xk);
                self.mm(&[&lw.wv], &[sc.vraw.f32p()], iv, t, xv);
                (null(), null())
            }
        };
        self.f32mm(&lw.wsink, d, c.n_head, iq, t, sc.sink.f32p());
        // 列の置き場: プレフィルは lane の頭（行は 1 本の列）、デコードは行 = lane（行の間隔つき）
        let cap = seq.cap[il];
        let kvrow = cap * kvd;
        let (lane0, kv_ld, st_ld) = match lane {
            Some(b) => (b, 0usize, 0usize),
            None => (0, kvrow, seq.conv_floats),
        };
        let conv = unsafe { seq.conv[il].f32p().add(lane0 * seq.conv_floats) };
        // KV（fp16 か int8）。デコードは lane 0 の頭から、行 r は lanes[r] の列（kv_ld = 1 列の要素の数）
        let (kvr, kvo) = (seq.kv_ref(il, lane0), seq.kv_out(il, lane0));
        let lanes_p: *const i32 = if lane.is_some() { null() } else { sc.lanes.ptr.cast() };
        let nc = c.nc;
        let (s_static, s_dyn) = (1.0 / ((nc + 1) as f32).sqrt(), 1.0 / ((2 * nc + 1) as f32).sqrt());
        let sk = if dk.is_null() { s_static } else { s_dyn };
        let st_k = conv;
        let st_v = unsafe { conv.add((nc - 1) * kvd) };
        let st_o = unsafe { conv.add(2 * (nc - 1) * kvd) };
        let st_f = unsafe { conv.add(2 * (nc - 1) * kvd + (nc - 1) * d) };
        let tau = if lw.swa { null() } else { sc.tau.f32p() as *const f32 };
        let ti = t as i32;
        let dyn_ld = (c.n_kv * nc) as i64;
        let ring = (cap < seq.max_ctx) as i32;
        let window = if lw.swa { c.n_swa as i32 } else { 0 };
        // 畳み込み（プレフィルは窓を塊の中から、デコードは列ごとの状態から）
        let convf = |x: *const f32, ch: usize, stp: *mut f32, w: *const f32, dynp: *const f32, dld: i64, heads: usize, scale: f32, y: *mut f32, what: &str| unsafe {
            if lane.is_some() {
                ck(kffi::ink_conv_pf(x, ch as i64, stp, w, nc as i32, dynp, dld, heads as i32, ch as i32, ti, scale, y, ch as i64, S()), what);
                ck(kffi::ink_conv_pf_state(x, ch as i64, stp, nc as i32, ch as i32, ti, S()), what);
            } else {
                ck(kffi::ink_conv(x, ch as i64, stp, st_ld as i64, lanes_p, w, nc as i32, dynp, dld, heads as i32, ch as i32, ti, scale, y, ch as i64, S()), what);
            }
        };
        convf(sc.kraw.f32p(), kvd, st_k, lw.k_conv.f32p(), dk, dyn_ld, c.n_kv, sk, sc.kk.f32p(), "ink conv k");
        convf(sc.vraw.f32p(), kvd, st_v, lw.v_conv.f32p(), dv, dyn_ld, c.n_kv, sk, sc.v.f32p(), "ink conv v");
        let rd = (c.n_head * c.n_rel) as i64;
        unsafe {
            ck(kffi::ink_headprep(sc.q.f32p(), qd as i64, c.n_head as i32, c.hd as i32, ti, lw.q_clip.f32p(), c.qk_clip, c.eps, tau, S()), "ink prep q");
            ck(kffi::ink_headprep(sc.kk.f32p(), kvd as i64, c.n_kv as i32, c.hd as i32, ti, null(), c.qk_clip, c.eps, null(), S()), "ink prep k");
            ck(kffi::ink_headprep(sc.v.f32p(), kvd as i64, c.n_kv as i32, c.hd as i32, ti, null(), 0.0, c.eps, null(), S()), "ink prep v");
            ck(kffi::ink_headprep(sc.r.f32p(), rd, c.n_head as i32, c.n_rel as i32, ti, lw.rel_clip.f32p(), c.rel_clip, c.eps, tau, S()), "ink prep r");
            // KV に書く（窓の層は輪: 位置 mod cap。位置は GPU の上の pos から）
            ck(kffi::ink_kvwrite_pf(sc.kk.f32p(), kvd as i64, sc.v.f32p(), kvd as i64, kvd as i32, ti, kvo, kv_ld as i64, lanes_p, sc.pos.ptr.cast(),
                                    cap as i32, ring, S()),
               "ink kv write");
            if lane.is_none() {
                // デコード: 鍵の数は GPU の上の pos から数える。split は最大の数で起こし、外の split は何もしない
                let max_keys = if lw.swa { c.n_swa } else { seq.max_ctx };
                ck(kffi::ink_attn_dec(sc.q.f32p(), qd as i64, sc.r.f32p(), rd, sc.sink.f32p(), c.n_head as i64, lw.rel_bias.f32p(), kvr, kv_ld as i64, lanes_p,
                                      sc.pos.ptr.cast(), ti, c.n_head as i32, c.n_kv as i32, c.n_rel as i32, c.extent as i32, cap as i32, ring, window,
                                      max_keys as i32, self.attn_chunk as i32, sc.part.f32p(), sc.att.f32p(), qd as i64, S()),
                   "ink attn");
            } else {
                ck(kffi::ink_attn_pf(sc.q.f32p(), qd as i64, sc.r.f32p(), rd, sc.sink.f32p(), c.n_head as i64, lw.rel_bias.f32p(), kvr, sc.pos.ptr.cast(),
                                     ti, c.n_head as i32, c.n_kv as i32, c.n_rel as i32, c.extent as i32, cap as i32, ring, window, sc.att.f32p(), qd as i64,
                                     S()),
                   "ink attn pf");
            }
            ck(kffi::ink_gate(sc.att.f32p(), qd as i64, sc.g.f32p(), qd as i64, qd as i32, ti, S()), "ink gate");
        }
        self.mm(&[&lw.wo], &[sc.oraw.f32p()], sc.att.f32p(), t, &sc.xp_o);
        convf(sc.oraw.f32p(), d, st_o, lw.attn_post_conv.f32p(), null(), 0, 1, s_static, sc.o.f32p(), "ink conv o");
        // 塊の切れ目: 入口を貯めて、Attention の出口から新しい残差を始める
        if ip % c.block == 0 {
            let b = (0..BLK_MAX).find(|b| !blocks.contains(b)).expect("塊の置き場が足りない");
            Self::copy(sc.blk[b].f32p(), sc.x.f32p(), t * d);
            blocks.push(b);
            Self::copy(sc.ffn_in.f32p(), sc.o.f32p(), t * d);
        } else {
            ck(unsafe { kffi::ink_add(sc.x.f32p(), d as i64, sc.o.f32p(), d as i64, d as i32, ti, sc.ffn_in.f32p(), d as i64, S()) }, "ink add");
        }
        // FFN の入口: 深さの Attention → 伝言の門 → 正規化
        let mut st: Vec<*const f32> = blocks.iter().map(|&b| sc.blk[b].f32p() as *const f32).collect();
        st.push(sc.ffn_in.f32p());
        self.mix(&st, lw.ffn_res_q.f32p(), 1, t, sc.m.f32p());
        if let Some(l) = lp {
            self.control(l, 3, sc.m.f32p(), msgp(3), 4 * d, t, sc.m.f32p());
        }
        self.rms(sc.m.f32p(), lw.ffn_norm.f32p(), t, sc.m.f32p());
        self.moe_ffn(lw, t, lane.is_none());
        convf(sc.oraw.f32p(), d, st_f, lw.ffn_post_conv.f32p(), null(), 0, 1, s_static, sc.o.f32p(), "ink conv f");
        unsafe {
            ck(kffi::ink_add(sc.ffn_in.f32p(), d as i64, sc.o.f32p(), d as i64, d as i32, ti, sc.x.f32p(), d as i64, S()), "ink add ffn");
        }
    }

    /// 行を写す（間隔 sld の t 行 → 間隔 d）
    fn copy_rows(&self, dst: *mut f32, src: *const f32, sld: usize, t: usize) {
        let d = self.cfg.d;
        for r in 0..t {
            Self::copy(unsafe { dst.add(r * d) }, unsafe { src.add(r * sld) }, d);
        }
    }

    /// MoE: ルーター（f32）→ 上位 8 → gate_up → 抑えた SwiGLU → down → 重みで足す。入口 sc.m・出口 sc.oraw（畳み込みの前）
    fn moe_ffn(&self, lw: &InkLayer, t: usize, decode: bool) {
        let c = &self.cfg;
        let sc = &self.sc;
        let (d, kk) = (c.d, c.k_exp);
        let r = t * kk;
        self.f32mm(&lw.router, d, c.n_exp, sc.m.f32p(), t, sc.lg.f32p());
        ck(
            unsafe {
                kffi::ink_route_warp(sc.lg.f32p(), c.n_exp as i64, lw.exp_bias.as_ref().map_or(null(), |b| b.f32p() as *const f32), c.n_exp as i32, kk as i32,
                                     t as i32, c.exp_scale, sc.ids.ptr.cast(), sc.wts.f32p(), S())
            },
            "ink route",
        );
        self.xprep(sc.m.f32p(), d, t, &sc.xp[0]);
        if decode || t == 1 {
            // デコードの形の仕事: 仕事 j = 行 j/kk の枠 j%kk（専門家 ids[j]）、出口の行 j。並びは固定（CUDA Graph に撮れる）
            assert!(t <= GEMV_B);
            let (je, jr0, jcnt) = (sc.ids.ptr as *const i32, sc.jr0.ptr as *const i32, sc.jcnt.ptr as *const i32);
            self.moe(&lw.gate_up, &sc.xp[0], t, sc.xidx.ptr.cast(), je, jr0, jcnt, r, 1, sc.gu.f32p());
            ck(unsafe { kffi::ink_situ(sc.gu.f32p(), c.ff as i32, r as i32, c.situ_g, c.situ_u, sc.h.f32p(), sc.ffp as i64, S()) }, "ink situ");
            self.xprep(sc.h.f32p(), sc.ffp, r, &sc.xp_h);
            self.moe(&lw.down, &sc.xp_h, r, null(), je, jr0, jcnt, r, 1, sc.o8.f32p());
            ck(unsafe { kffi::ink_moe_sum(sc.o8.f32p(), sc.wts.f32p(), d as i32, kk as i32, t as i32, sc.oraw.f32p(), d as i64, S()) }, "ink moe sum");
            return;
        }
        if std::env::var("INK_MOE_GEMV").map_or(true, |v| v != "1") {
            // プレフィル: 専門家ごとの GEMM（Tensor Core）。並べは GPU の上（infer_moe_group）
            let n_blk = (c.n_exp + r / MOE_TOKB) as i32;
            let tokb = MOE_TOKB as i32;
            unsafe {
                ck(kffi::infer_moe_group(sc.ids.ptr.cast(), r as i32, kk as i32, c.n_exp as i32, tokb, sc.pxidx.ptr.cast(), sc.rowmap.ptr.cast(),
                                         sc.grp.ptr.cast(), n_blk, S()),
                   "moe group");
                ck(kffi::infer_m_f32_to_f16(sc.m.f32p(), (t * d) as i64, sc.x16.ptr, S()), "moe x16");
                let (q, w, es, n, ys) = ([lw.gate_up.qt], [lw.gate_up.buf.ptr as *const std::ffi::c_void], [lw.gate_up.es], [lw.gate_up.n as i32],
                                         [sc.gu.ptr]);
                ck(kffi::infer_g3_moe_gemm(1, q.as_ptr(), w.as_ptr(), es.as_ptr(), n.as_ptr(), ys.as_ptr(), null(), d as i32, sc.x16.ptr, null(), d as i32,
                                           sc.pxidx.ptr.cast(), sc.grp.ptr.cast(), n_blk, tokb, 0, S()),
                   "moe gemm gate_up");
                ck(kffi::ink_situ(sc.gu.f32p(), c.ff as i32, r as i32, c.situ_g, c.situ_u, sc.h.f32p(), sc.ffp as i64, S()), "ink situ");
                ck(kffi::infer_m_f32_to_f16(sc.h.f32p(), (r * sc.ffp) as i64, sc.h16.ptr, S()), "moe h16");
                let (q, w, es, n, ys) = ([lw.down.qt], [lw.down.buf.ptr as *const std::ffi::c_void], [lw.down.es], [lw.down.n as i32], [sc.o8.ptr]);
                ck(kffi::infer_g3_moe_gemm(1, q.as_ptr(), w.as_ptr(), es.as_ptr(), n.as_ptr(), ys.as_ptr(), null(), sc.ffp as i32, sc.h16.ptr, null(),
                                           sc.ffp as i32, null(), sc.grp.ptr.cast(), n_blk, tokb, 0, S()),
                   "moe gemm down");
                ck(kffi::ink_moe_sum_map(sc.o8.f32p(), sc.rowmap.ptr.cast(), sc.wts.f32p(), d as i32, kk as i32, t as i32, sc.oraw.f32p(), d as i64, S()),
                   "ink moe sum map");
            }
            return;
        }
        // プレフィル: 専門家ごとに語をまとめる（手元で。仕事 = 専門家 1 つの最大 32 行）
        ck(unsafe { kffi::cudaStreamSynchronize(S()) }, "moe ids を待つ");
        let mut b = vec![0u8; r * 4];
        crate::cuda::check(unsafe { ffi::cudaMemcpy(b.as_mut_ptr().cast(), sc.ids.ptr, r * 4, 2) }, "ids D2H");
        let ids: Vec<usize> = b.chunks_exact(4).map(|x| i32::from_le_bytes(x.try_into().unwrap()) as usize).collect();
        let mut per: Vec<Vec<u32>> = vec![vec![]; c.n_exp];
        for (a, &e) in ids.iter().enumerate() {
            assert!(e < c.n_exp, "専門家の番号 {e}");
            per[e].push(a as u32);
        }
        let (mut xidx, mut rowmap) = (vec![0i32; r], vec![0i32; r]);
        let (mut je, mut jr0, mut jcnt) = (vec![], vec![], vec![]);
        let mut row = 0usize;
        for (e, list) in per.iter().enumerate() {
            for ch in list.chunks(GEMV_B) {
                je.push(e as i32);
                jr0.push(row as i32);
                jcnt.push(ch.len() as i32);
                for &a in ch {
                    xidx[row] = (a as usize / kk) as i32;
                    rowmap[a as usize] = row as i32;
                    row += 1;
                }
            }
        }
        let up = |buf: &DevBuf, v: &[i32]| ck(unsafe { ffi::cudaMemcpy(buf.ptr, v.as_ptr().cast(), v.len() * 4, 1) }, "moe H2D");
        up(&sc.pxidx, &xidx);
        up(&sc.rowmap, &rowmap);
        up(&sc.pje, &je);
        up(&sc.pjr0, &jr0);
        up(&sc.pjcnt, &jcnt);
        let nj = je.len();
        let (pje, pjr0, pjcnt) = (sc.pje.ptr as *const i32, sc.pjr0.ptr as *const i32, sc.pjcnt.ptr as *const i32);
        self.moe(&lw.gate_up, &sc.xp[0], t, sc.pxidx.ptr.cast(), pje, pjr0, pjcnt, nj, GEMV_B, sc.gu.f32p());
        ck(unsafe { kffi::ink_situ(sc.gu.f32p(), c.ff as i32, r as i32, c.situ_g, c.situ_u, sc.h.f32p(), sc.ffp as i64, S()) }, "ink situ");
        self.xprep(sc.h.f32p(), sc.ffp, r, &sc.xp_h);
        self.moe(&lw.down, &sc.xp_h, r, null(), pje, pjr0, pjcnt, nj, GEMV_B, sc.o8.f32p());
        ck(unsafe { kffi::ink_moe_sum_map(sc.o8.f32p(), sc.rowmap.ptr.cast(), sc.wts.f32p(), d as i32, kk as i32, t as i32, sc.oraw.f32p(), d as i64, S()) },
           "ink moe sum map");
    }

    /// 直前の step の logits（行 0・頭の vocab 個）を CPU へ
    pub fn logits(&self) -> Vec<f32> {
        self.logits_row(0)
    }

    /// 直前の step_lanes の行 0..n の logits（頭の vocab 個ずつ続けて）
    pub fn logits_rows(&self, n: usize) -> Vec<f32> {
        ck(unsafe { kffi::cudaStreamSynchronize(S()) }, "sync");
        let (v, vp) = (self.cfg.vocab, self.cfg.vocab_padded);
        let mut all = vec![0f32; n * vp];
        ck(unsafe { ffi::cudaMemcpy(all.as_mut_ptr().cast(), self.sc.logits.ptr, n * vp * 4, 2) }, "logits D2H");
        let mut out = Vec::with_capacity(n * v);
        for r in 0..n {
            out.extend_from_slice(&all[r * vp..r * vp + v]);
        }
        out
    }

    /// 直前の step_batch の行 b の logits
    pub fn logits_row(&self, b: usize) -> Vec<f32> {
        ck(unsafe { kffi::cudaStreamSynchronize(S()) }, "sync");
        let mut v = vec![0f32; self.cfg.vocab];
        let src = unsafe { self.sc.logits.f32p().add(b * self.cfg.vocab_padded) };
        ck(unsafe { ffi::cudaMemcpy(v.as_mut_ptr().cast(), src.cast(), self.cfg.vocab * 4, 2) }, "logits D2H");
        v
    }

    /// 直前の step の出口の残差（確かめ用）
    pub fn hidden(&self) -> Vec<f32> {
        ck(unsafe { kffi::cudaStreamSynchronize(S()) }, "sync");
        self.sc.x.to_f32()
    }
}
