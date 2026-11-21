//! MTP（下書きの 1 段。Qwen3.8-27B の blk.64）を GPU で。式は docs/qwen35-math.md §7、手本は tools/ref_mtp.py
//! （llama.cpp の graph_mtp・draft-mtp と節ごとに答え合わせ済み）。
//!
//!   e  = RMSNorm(token_embd[x], enorm)、hh = RMSNorm(h, hnorm)        infer_mtp_prep（埋め込みはその場で GGUF の行を戻す）
//!   x0 = eh_proj · [e ; hh]                                             g3（10240 → 5120）
//!   x1 = x0 + attn_output · Attn(RMSNorm(x0, attn_norm))                g3 multi（q・k・v）→ attn::rope → attn::attn → g3（足し込み）
//!   x2 = x1 + ffn_down · SwiGLU(RMSNorm(x1, post_attention_norm))       g3 swiglu（down の下ごしらえまで）→ g3（足し込み）
//!   h' = RMSNorm(x2, shared_head_norm)、logits = output · h'            g3（5120 → 248320）→ infer_mtp_topk
//!
//! 重みの出どころ: `load` に渡す GGUF の blk.{n_layer}（本体の GGUF なら Q6_K・Q8_0、別ファイル mtp-Qwen3.8-27B-Q4_0.gguf なら Q6_K・Q4_K）。
//! 埋め込みと出力は既定でその GGUF の物（別ファイルなら Q3_K。llama-server の -md と同じ）。`Shared` で本体の物を借りられる
//! （本体の GGUF の blk.64 を使うときは借りれば 1.8GB 浮く）。
//!
//! KV: MTP の Attention 1 層ぶん。attn.rs と同じ並び kc, vc [n_pages][4][64][256] fp16。升 s は頁 s*pages_per_slot.. を使う（決め打ち）。
//! 位置 p の行は KV の位置 p に書く。列の頭（p = 0）の行は h = 0（llama-server の pending_h の初め）。
//! 下書きの行も KV に書く（後ろの追いつきが本体の h で上書きする。Attention は kv_len までしか読まないので、残りかすは見えない）。
use crate::attn::{self, AttnParams, RopeParams};
use crate::cuda::{self, ffi, ffi::Stream, DevBuf};
use crate::gguf::{Model, TensorView};
use crate::spec::{self, DraftAsk, DraftOut, Drafter, Observe, SeqId};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::{null, null_mut};

mod kffi {
    use super::*;
    extern "C" {
        pub fn infer_mtp_prep(
            qtype: i32, embd: *const c_void, row_bytes: i64, tok: *const i32, hp: *const *const f32, enorm: *const f32, hnorm: *const f32,
            k: i32, rows: i32, eps: f32, out: *mut f32, s: Stream,
        ) -> i32;
        pub fn infer_mtp_rmsnorm(x: *const f32, ldx: i64, w: *const f32, k: i32, rows: i32, eps: f32, y: *mut f32, ldy: i64, s: Stream) -> i32;
        pub fn infer_mtp_gather(src: *const f32, ld: i64, idx: *const i32, k: i32, rows: i32, dst: *mut f32, s: Stream) -> i32;
        pub fn infer_mtp_feed(top: *const i32, kout: i32, src: *const i32, map: *const i32, n: i32, dst: *mut i32, s: Stream) -> i32;
        pub fn infer_mtp_copy_rows(dst: *const *mut f32, src: *const *const f32, k: i32, rows: i32, s: Stream) -> i32;
        pub fn infer_mtp_topk_ws_bytes(n: i32, rows: i32, k: i32) -> i64;
        pub fn infer_mtp_gumbel(
            x: *const f32, n: i32, ld: i64, rows: i32, map: *const i32, gp: *const c_void, ws: *mut c_void, kout: i32, oi: *mut i32, s: Stream,
        ) -> i32;
        pub fn infer_mtp_gumbel_probe(seed: u64, pos: u64, ids: *const u32, n: i32, out: *mut f64, s: Stream) -> i32;
        pub fn infer_g3_gemm_ex(
            nmat: i32, qtypes: *const i32, ws: *const *const c_void, ns: *const i32, ys: *const *mut c_void, ldys: *const i32, k: i32, m: i32,
            x: *const c_void, xlo: *const c_void, ldx: i32, mode: i32, cfg: i32, s: Stream,
        ) -> i32;
        pub fn infer_m_f32_to_f16(x: *const f32, n: i64, y: *mut c_void, s: Stream) -> i32;
        pub fn infer_mtp_topk(
            x: *const f32, n: i32, ld: i64, rows: i32, k: i32, ws: *mut c_void, ws_bytes: i64, oi: *mut i32, ov: *mut f32, olse: *mut f32,
            s: Stream,
        ) -> i32;
        // MoE（kernels/moe.cu・gemv3.cu。3.6 の MTP の層）
        pub fn infer_moe_router(w: *const f32, k: i32, ne: i32, x: *const f32, xld: i64, t: i32, y: *mut f32, s: Stream) -> i32;
        pub fn infer_moe_topk(
            lg: *const f32, ld: i64, ne: i32, kk: i32, t: i32, ids: *mut i32, wts: *mut f32, jr0: *mut i32, jcnt: *mut i32, xidx: *mut i32,
            s: Stream,
        ) -> i32;
        pub fn infer_moe_swiglu(g: *const f32, u: *const f32, total: i64, h: *mut f32, s: Stream) -> i32;
        pub fn infer_moe_combine(
            h: *mut f32, d: i32, t: i32, kk: i32, wts: *const f32, o: *const f32, rowmap: *const i32, sh: *const f32, sg: *const f32,
            sg_ld: i64, s: Stream,
        ) -> i32;
        pub fn infer_g3_moe_gemv(
            nm: i32, qtype: *const i32, w: *const *const c_void, estride: *const i64, n: *const i32, y: *const *mut f32, k: i32,
            xp: *const c_void, rx: i32, xidx: *const i32, je: *const i32, jr0: *const i32, jcnt: *const i32, n_jobs: i32, bmax: i32,
            s: Stream,
        ) -> i32;
        pub fn infer_moe_group(
            ids: *const i32, r: i32, kk: i32, ne: i32, tokb: i32, xidx: *mut i32, rowmap: *mut i32, grp: *mut i32, n_blk_max: i32, s: Stream,
        ) -> i32;
        pub fn infer_g3_moe_gemm(
            nmat: i32, qtypes: *const i32, ws: *const *const c_void, es: *const i64, ns: *const i32, ys: *const *mut c_void, ldys: *const i32,
            k: i32, x: *const c_void, xlo: *const c_void, ldx: i32, xidx: *const i32, grp: *const i32, n_blk: i32, tokb: i32, mode: i32,
            s: Stream,
        ) -> i32;
    }
}

/// MTP の層の MoE（3.6）: ルーター f32 [n_exp + 1][n_embd]（最後の行が共有のゲート）と、専門家を続けて並べた g3 の行列。
/// 共有の専門家は MtpWeights の ffn_gate・ffn_up・ffn_down
pub struct MtpMoe {
    pub router: DevBuf,
    pub gate: G3Mat,
    pub up: G3Mat,
    pub down: G3Mat,
    pub n_exp: usize,
    pub k: usize,
    pub ff: usize,
    es_gu: i64,
    es_d: i64,
    // 作業場（MAX_ROWS 行ぶん。行 R = MAX_ROWS·k）
    rl: DevBuf,
    ids: DevBuf,
    wts: DevBuf,
    jr0: DevBuf,
    jcnt: DevBuf,
    xidx: DevBuf,
    yg: DevBuf,
    yu: DevBuf,
    hh: DevBuf,
    hp: DevBuf,
    o: DevBuf,
    sh: DevBuf,
}

/// 1 回で流せる行の上限（g3 の B の上限）
pub const MAX_ROWS: usize = 32;

/// 結び付けの下書き（TACHYON_HASH_RNG）の 1 行の引数（kernels/mtp.cu の GParam と同じ並び・24 バイト）
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GParam {
    pub seed: u64,
    pub pos: u64,
    pub invt: f32,
    pub on: i32,
}
/// 段ごとの GParam の置き場（段 j は j*MAX_ROWS 行目から）
const G_STAGES: usize = 16;

/// 下書きの j 段目（位置 ask.pos + j + 1 の語）の結び付けの引数
fn gparam(ask: &DraftAsk, j: usize) -> GParam {
    match ask.gumbel {
        Some((seed, t)) if ask.sample.is_none() && t > 1e-5 => GParam { seed, pos: (ask.pos + j + 1) as u64, invt: 1.0 / t, on: 1 },
        _ => GParam::default(),
    }
}

/// 確かめ用: GPU の Gumbel（kernels/mtp.cu の gumbel_d）を ids について（手元の spec::gumbel と同じか）
pub fn gpu_gumbel_probe(seed: u64, pos: u64, ids: &[u32]) -> Vec<f64> {
    let n = ids.len();
    let d_ids = DevBuf::new(n * 4);
    let d_out = DevBuf::new(n * 8);
    let mut out = vec![0f64; n];
    unsafe {
        cuda::check(ffi::cudaMemcpy(d_ids.ptr, ids.as_ptr().cast(), n * 4, 1), "probe H2D");
        cuda::check(kffi::infer_mtp_gumbel_probe(seed, pos, d_ids.ptr as *const u32, n as i32, d_out.ptr as *mut f64, null_mut()), "probe");
        cuda::check(ffi::cudaMemcpy(out.as_mut_ptr().cast(), d_out.ptr, n * 8, 2), "probe D2H");
    }
    out
}
/// 追いつき（下書きを出さない行）を GEMM の道（forward_gemm）で 1 回に流す行の上限
pub const GEMM_ROWS: usize = 512;

/// GEMM の道の作業場（GEMM_ROWS 行。最初に要るときに取る）
struct Big {
    meta: DevBuf,
    cat: DevBuf,
    x16: DevBuf,
    x: DevBuf,
    a: DevBuf,
    qf: DevBuf,
    kk: DevBuf,
    vv: DevBuf,
    q16: DevBuf,
    o16: DevBuf,
    mid16: DevBuf,
}
/// 追いつきの MoE を専門家ごとの GEMM で（2026-11-14。GEMM_ROWS 行・行 R = GEMM_ROWS·k。最初に要るときに取る）
struct MoeBig {
    rl: DevBuf,
    ids: DevBuf,
    wts: DevBuf,
    xidx: DevBuf,
    rowmap: DevBuf,
    grp: DevBuf,
    h16: DevBuf,
    o: DevBuf,
    sh: DevBuf,
    a: DevBuf,
    x16: DevBuf,
    mid16: DevBuf,
}

/// 専門家ごとの GEMM の語のまとまり（本体の TACHYON_MOE_TOKB と同じ既定）
const MOE_TOKB: usize = 64;

fn mtp_moe_gemm_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TACHYON_MOE_GEMM").map_or(true, |v| v != "0"))
}

/// 上位 k の上限（infer_mtp_topk）
pub const MAX_TOPK: usize = 32;
/// 手元に戻らずに続けて流す段の数（1 段目を含む。4 段目からは確からしさで止めるので手元で）
const CHAIN: usize = 3;

/// forward の 1 回ぶんの形（meta に書いた後、カーネルを積むのに要る数）
struct Plan {
    b: usize,
    n_seq: usize,
    max_q: usize,
    max_kv: usize,
    nw: usize,
}

fn ck(code: i32, what: &str) -> Result<(), String> {
    match code {
        0 => Ok(()),
        c if c < 0 => Err(format!("{what}: 自前のカーネルのエラー {c}")),
        c => Err(format!("{what}: cudaError {c}")),
    }
}

/// g3 の並びに詰め直した行列（own が None なら借り物）
pub struct G3Mat {
    pub ptr: *const c_void,
    /// 詰め直した後の型
    pub t: i32,
    pub k: usize,
    pub n: usize,
    own: Option<DevBuf>,
}

impl G3Mat {
    /// GGUF の行 → GPU → 自前の並び（元の写しは捨てる）
    pub fn load(tv: &TensorView) -> Result<G3Mat, String> {
        let (t, k, n) = (tv.qt() as i32, tv.k(), tv.n_rows());
        let bytes = unsafe { ffi::infer_g3_bytes(t, k as i32, n as i32) };
        if bytes <= 0 {
            return Err(format!("{}: g3 が扱わない型 {}", tv.name, tv.type_name()));
        }
        let src = DevBuf::from_bytes(tv.bytes);
        let dst = DevBuf::new(bytes as usize);
        ck(unsafe { ffi::infer_g3_repack(t, src.ptr, dst.ptr, k as i32, n as i32, null_mut()) }, "g3 repack")?;
        cuda::sync();
        Ok(G3Mat { ptr: dst.ptr, t: unsafe { ffi::infer_g3_type(t) }, k, n, own: Some(dst) })
    }
    /// GGUF の行列の一部の行（rows の番号の順）だけを詰め直す（下書き用の小さな lm_head）
    pub fn load_rows(tv: &TensorView, rows: &[u32]) -> Result<G3Mat, String> {
        let (t, k) = (tv.qt() as i32, tv.k());
        let rb = tv.row_bytes();
        let mut raw = Vec::with_capacity(rows.len() * rb);
        for &r in rows {
            let r = r as usize;
            if r >= tv.n_rows() {
                return Err(format!("{}: 行 {r} が無い", tv.name));
            }
            raw.extend_from_slice(&tv.bytes[r * rb..(r + 1) * rb]);
        }
        let n = rows.len();
        let bytes = unsafe { ffi::infer_g3_bytes(t, k as i32, n as i32) };
        if bytes <= 0 {
            return Err(format!("{}: g3 が扱わない型 {}", tv.name, tv.type_name()));
        }
        let src = DevBuf::from_bytes(&raw);
        let dst = DevBuf::new(bytes as usize);
        ck(unsafe { ffi::infer_g3_repack(t, src.ptr, dst.ptr, k as i32, n as i32, null_mut()) }, "g3 repack")?;
        cuda::sync();
        Ok(G3Mat { ptr: dst.ptr, t: unsafe { ffi::infer_g3_type(t) }, k, n, own: Some(dst) })
    }
    /// 本体などが持つ詰め直し済みの行列を借りる（本体が先に消えないこと）
    pub fn borrow(ptr: *const c_void, t: i32, k: usize, n: usize) -> G3Mat {
        G3Mat { ptr, t, k, n, own: None }
    }
    pub fn bytes(&self) -> usize {
        self.own.as_ref().map_or(0, |b| b.len)
    }
}

/// 埋め込み（GGUF の並びのまま。行をその場で戻す）
pub struct Embd {
    pub ptr: *const c_void,
    pub qtype: i32,
    pub row_bytes: usize,
    pub k: usize,
    pub n: usize,
    own: Option<DevBuf>,
}

impl Embd {
    pub fn load(tv: &TensorView) -> Embd {
        let b = DevBuf::from_bytes(tv.bytes);
        Embd { ptr: b.ptr, qtype: tv.qt() as i32, row_bytes: tv.row_bytes(), k: tv.k(), n: tv.n_rows(), own: Some(b) }
    }
    pub fn borrow(ptr: *const c_void, qtype: i32, row_bytes: usize, k: usize, n: usize) -> Embd {
        Embd { ptr, qtype, row_bytes, k, n, own: None }
    }
    pub fn bytes(&self) -> usize {
        self.own.as_ref().map_or(0, |b| b.len)
    }
}

/// 本体から借りる物
#[derive(Default)]
pub struct Shared {
    pub embd: Option<Embd>,
    pub output: Option<G3Mat>,
}

/// 形（GGUF の qwen35.*）
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    pub eps: f32,
    /// MTP の層の番号（= 本体の層の数 64）
    pub il: usize,
}

pub struct MtpOpts {
    /// 同時に持つ列の数（升）
    pub slots: usize,
    /// 1 列の最大の長さ（MTP の KV の位置の上限）
    pub max_ctx: usize,
    /// 下書きの上位 k（1 = argmax だけ。Q から取るとき・記録するときは 32）
    pub topk: usize,
    /// KV の頁の数（0 なら slots × max_ctx/64。> 0 なら本体と同じ頁の番号を set_pages で渡す＝M3 の頁を配る置き場）
    pub kv_pages: usize,
}

impl Default for MtpOpts {
    fn default() -> Self {
        MtpOpts { slots: 8, max_ctx: 8192, topk: 1, kv_pages: 0 }
    }
}

/// 1 行の入力（升・位置・語・h の番地。null は 0）
#[derive(Clone, Copy, Debug)]
pub struct Row {
    pub slot: usize,
    pub pos: usize,
    pub token: u32,
    pub h: *const f32,
}

/// logits を出した行の上位 k と logsumexp
#[derive(Clone, Debug)]
pub struct TopRow {
    pub ids: Vec<u32>,
    pub vals: Vec<f32>,
    pub lse: f32,
}

/// 下書きの行の記録（答え合わせ用。`MtpHead::trace` を Some にすると run が足していく）
#[derive(Default)]
pub struct Trace {
    pub rows: Vec<TopRow>,
    pub hout: Vec<Vec<f32>>,
    /// 頭から full 行ぶんだけ logits を全部
    pub logits: Vec<Vec<f32>>,
    pub full: usize,
}

/// 下書きの記録（分析用。`MtpHead::log` を Some にすると run が足していく。上位 2 つを取るので頭の 3 段の続け流しは使わない）
#[derive(Clone, Debug, Default)]
pub struct DraftLog {
    /// 下書きの頭の位置（id_last の位置。段 j の語は pos + 1 + j に来るはず）
    pub pos: usize,
    pub last: u32,
    /// 段ごとの (1 位, 確率, 2 位, 確率)
    pub tops: Vec<(u32, f32, u32, f32)>,
}

/// GPU の上の h の 1 行（本体の出力など。spec::Target::H）
#[derive(Clone, Copy, Debug)]
pub struct DevRow(pub *const f32);

pub struct MtpWeights {
    pub eh_proj: G3Mat,
    pub attn_q: G3Mat,
    pub attn_k: G3Mat,
    pub attn_v: G3Mat,
    pub attn_output: G3Mat,
    pub ffn_gate: G3Mat,
    pub ffn_up: G3Mat,
    pub ffn_down: G3Mat,
    pub attn_norm: DevBuf,
    pub post_norm: DevBuf,
    pub q_norm: DevBuf,
    pub k_norm: DevBuf,
    pub enorm: DevBuf,
    pub hnorm: DevBuf,
    pub head_norm: DevBuf,
    pub embd: Embd,
    pub output: G3Mat,
    /// MoE の層（3.6。None なら密な FFN）
    pub moe: Option<MtpMoe>,
}

// 小さな入力の置き場（GPU の上で 1 本。歩ごとに 1 回で送る）
struct Meta {
    dev: DevBuf,
    host: Vec<u8>,
    tok: usize,
    pos: usize,
    slot: usize,
    hptr: usize,
    q_start: usize,
    kv_len: usize,
    bt: usize,
    gidx: usize,
    cdst: usize,
    csrc: usize,
    feed: usize,
}

impl Meta {
    fn new(r: usize, bt_ld: usize, slots: usize) -> Meta {
        let mut off = 0;
        let mut take = |n: usize| {
            let o = off;
            off += (n + 7) / 8 * 8;
            o
        };
        let tok = take(r * 4);
        let pos = take(r * 4);
        let slot = take(r * 4);
        let hptr = take(r * 8);
        let q_start = take((r + 1) * 4);
        let kv_len = take(r * 4);
        let gidx = take(r * 4);
        let cdst = take(slots.max(r) * 8);
        let csrc = take(slots.max(r) * 8);
        let feed = take(r * 4);
        // 頁表は最後（送るのは使った所まで）
        let bt = take(r * bt_ld * 4);
        Meta { dev: DevBuf::new(off), host: vec![0; off], tok, pos, slot, hptr, q_start, kv_len, bt, gidx, cdst, csrc, feed }
    }
    fn put_i32(&mut self, at: usize, v: &[i32]) {
        for (i, x) in v.iter().enumerate() {
            self.host[at + 4 * i..at + 4 * i + 4].copy_from_slice(&x.to_le_bytes());
        }
    }
    fn put_u64(&mut self, at: usize, v: &[u64]) {
        for (i, x) in v.iter().enumerate() {
            self.host[at + 8 * i..at + 8 * i + 8].copy_from_slice(&x.to_le_bytes());
        }
    }
    /// 頭から n バイトだけ送る（頁表は最後に置いてあるので、使った所まで）
    fn upload(&self, n: usize) {
        let n = n.min(self.host.len());
        cuda::check(unsafe { ffi::cudaMemcpy(self.dev.ptr, self.host.as_ptr().cast(), n, 1) }, "mtp meta H2D");
    }
    fn d(&self, at: usize) -> *mut u8 {
        unsafe { (self.dev.ptr as *mut u8).add(at) }
    }
}

/// MTP の 1 段と、その KV・作業場
pub struct MtpHead {
    pub w: MtpWeights,
    pub dims: Dims,
    pub ash: attn::Shape,
    pub opts: MtpOpts,
    pub pages_per_slot: usize,
    kc: DevBuf,
    vc: DevBuf,
    // 作業場（MAX_ROWS 行ぶん）
    cat: DevBuf,
    x: DevBuf,
    a: DevBuf,
    qf: DevBuf,
    kk: DevBuf,
    vv: DevBuf,
    q16: DevBuf,
    ao: DevBuf,
    hout: [DevBuf; 2],
    cur: usize,
    hsel: DevBuf,
    logits: DevBuf,
    xp_a: DevBuf,
    xp_b: DevBuf,
    attn_ws: DevBuf,
    topk_ws: DevBuf,
    top_i: DevBuf,
    top_v: DevBuf,
    top_l: DevBuf,
    meta: Meta,
    /// 続けて流す 2・3 段目の入力の置き場（chain）
    meta_ch: [Meta; CHAIN - 1],
    /// draft_vocab の語の番号（GPU の上。chain の feed が使う）
    draft_ids_dev: Option<DevBuf>,
    /// 頭の 3 段を手元に戻らず続けて流す（既定 true。TACHYON_NO_MTP_CHAIN=1 で切る）
    pub chain: bool,
    /// HCAP の影の測り（TACHYON_HCAP_SHADOW=1）: 下書きの段ごとに 2・3 番手を手元で数えて DraftOut::alts に（chain は切る・遅い。測るだけ）
    pub hcap_shadow: bool,
    /// 結び付けの下書きの引数（GPU。G_STAGES × MAX_ROWS 行）と作業場（行ごとに 8 バイト）
    gp_dev: DevBuf,
    gkey: DevBuf,
    /// 升ごとの前の h（本体の最後の行の h_nextn）
    pend: DevBuf,
    /// pend の位置（本体のどの位置の h か）
    pend_pos: Vec<Option<usize>>,
    /// 本体の h で書いた KV の長さ
    committed: Vec<usize>,
    slot_of: HashMap<SeqId, usize>,
    free_slots: Vec<usize>,
    pub trace: Option<Trace>,
    /// 下書き用に絞った lm_head（FR-Spec）と、その行の語の番号。Some なら logits はこの語だけ（argmax の下書きの道だけ）
    pub draft_vocab: Option<(G3Mat, Vec<u32>)>,
    big: Option<Big>,
    moe_big: Option<MoeBig>,
    /// 升ごとの頁表（kv_pages の置き場のとき）
    ext_pages: Vec<Vec<u32>>,
    /// 下書きを打ち切る確からしさ（MTP の 1 位の確率の積がこれ未満の語から後ろは出さない。0 なら k 語まで必ず）
    pub draft_min_p: f32,
    /// 4 語目から後ろの線（draft_min_p より大きければこちら）
    pub draft_min_p_deep: f32,
    /// 追いつきを GEMM の道で（既定 true）
    pub gemm_catchup: bool,
    /// 升ごとの画像（列の位置 start・nx・ny。RoPE の位置を本体と同じ決まりでずらす。画像の語は t の位置）
    media: Vec<Vec<(usize, usize, usize)>>,
    /// 下書きの記録（分析用）
    pub log: Option<Vec<DraftLog>>,
    /// 1 回の run の行の数（記録）
    pub n_forward: usize,
    pub n_rows: usize,
}

fn f32buf(v: &[f32]) -> DevBuf {
    DevBuf::from_f32(v)
}

fn norm_w(g: &Model, name: &str) -> Result<DevBuf, String> {
    let tv = g.get(name).ok_or_else(|| format!("テンソルが無い: {name}"))?;
    if tv.ggml_type != 0 {
        return Err(format!("{name}: F32 のはず（{}）", tv.type_name()));
    }
    let v: Vec<f32> = tv.bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    Ok(f32buf(&v))
}

impl MtpHead {
    /// g の blk.{n_layer}（MTP の層）を GPU へ。埋め込み・出力は share が無ければ g の物
    pub fn load(g: &Model, opts: MtpOpts, share: Shared) -> Result<MtpHead, String> {
        let hp = |k: &str| g.hp_u32(k).ok_or_else(|| format!("GGUF に {}.{k} が無い", g.arch()));
        let n_all = hp("block_count")? as usize;
        let n_nextn = g.hp_u32("nextn_predict_layers").unwrap_or(0) as usize;
        if n_nextn != 1 {
            return Err(format!("MTP の層が 1 つではない（{n_nextn}）"));
        }
        let il = n_all - n_nextn;
        let p = |s: &str| format!("blk.{il}.{s}");
        let n_embd = hp("embedding_length")? as usize;
        let n_head = hp("attention.head_count")? as usize;
        let n_head_kv = hp("attention.head_count_kv")? as usize;
        let head_dim = hp("attention.key_length")? as usize;
        let is_moe = g.arch() == "qwen35moe";
        // MoE（3.6）は共有の専門家の幅（密な FFN の道をそのまま共有の専門家に使う）
        let n_ff = if is_moe { hp("expert_shared_feed_forward_length")? } else { hp("feed_forward_length")? } as usize;
        let eps = g.hp_f32("attention.layer_norm_rms_epsilon").ok_or("eps が無い")?;
        let n_rot = hp("rope.dimension_count")? as usize;
        let theta = g.hp_f32("rope.freq_base").ok_or("rope.freq_base が無い")?;
        let sec = g.hp_arr_i64("rope.dimension_sections").unwrap_or_default();
        let mut sections = [0i32; 4];
        for (i, s) in sec.iter().take(4).enumerate() {
            sections[i] = *s as i32;
        }
        if head_dim != attn::D || n_head % n_head_kv != 0 {
            return Err(format!("attn の形と合わない（head {head_dim}・{n_head}/{n_head_kv}）"));
        }
        if opts.topk == 0 || opts.topk > MAX_TOPK || opts.slots == 0 || opts.max_ctx == 0 {
            return Err("MtpOpts が変".into());
        }
        let ash = attn::Shape { h: n_head, hkv: n_head_kv, n_rot, sections, theta, eps, yarn: attn::yarn_for(n_rot, theta, attn::gguf_rope_scaling(g)) };
        for n in ["nextn.embed_tokens.weight", "nextn.shared_head_head.weight"] {
            if g.get(&p(n)).is_some() {
                return Err(format!("{}: まだ扱っていない（この 2 つの GGUF には無い）", p(n)));
            }
        }
        let gm = |s: &str| -> Result<G3Mat, String> { G3Mat::load(&g.get(&p(s)).ok_or_else(|| format!("テンソルが無い: {}", p(s)))?) };
        let embd = match share.embd {
            Some(e) => e,
            None => Embd::load(&g.tensor("token_embd.weight")),
        };
        let output = match share.output {
            Some(o) => o,
            None => G3Mat::load(&g.tensor("output.weight"))?,
        };
        let w = MtpWeights {
            eh_proj: gm("nextn.eh_proj.weight")?,
            attn_q: gm("attn_q.weight")?,
            attn_k: gm("attn_k.weight")?,
            attn_v: gm("attn_v.weight")?,
            attn_output: gm("attn_output.weight")?,
            ffn_gate: gm(if is_moe { "ffn_gate_shexp.weight" } else { "ffn_gate.weight" })?,
            ffn_up: gm(if is_moe { "ffn_up_shexp.weight" } else { "ffn_up.weight" })?,
            ffn_down: gm(if is_moe { "ffn_down_shexp.weight" } else { "ffn_down.weight" })?,
            attn_norm: norm_w(g, &p("attn_norm.weight"))?,
            post_norm: norm_w(g, &p("post_attention_norm.weight"))?,
            q_norm: norm_w(g, &p("attn_q_norm.weight"))?,
            k_norm: norm_w(g, &p("attn_k_norm.weight"))?,
            enorm: norm_w(g, &p("nextn.enorm.weight"))?,
            hnorm: norm_w(g, &p("nextn.hnorm.weight"))?,
            head_norm: norm_w(g, &p("nextn.shared_head_norm.weight"))?,
            embd,
            output,
            moe: None,
        };
        let mut w = w;
        if is_moe {
            let n_exp = hp("expert_count")? as usize;
            let kk = hp("expert_used_count")? as usize;
            let ff = hp("expert_feed_forward_length")? as usize;
            let rt = g.get(&p("ffn_gate_inp.weight")).ok_or("ffn_gate_inp が無い")?;
            let rs = g.get(&p("ffn_gate_inp_shexp.weight")).ok_or("ffn_gate_inp_shexp が無い")?;
            if rt.ggml_type != 0 || rs.ggml_type != 0 || rt.k() != n_embd || rt.n_rows() != n_exp || rs.n_elems() != n_embd {
                return Err("MTP のルーターの形・型（f32 [n_exp][n_embd]）が違う".into());
            }
            let mut rb = rt.bytes.to_vec();
            rb.extend_from_slice(rs.bytes);
            let (ge, ue, de) = (gm("ffn_gate_exps.weight")?, gm("ffn_up_exps.weight")?, gm("ffn_down_exps.weight")?);
            if ge.k != n_embd || ge.n != n_exp * ff || ue.k != n_embd || ue.n != n_exp * ff || de.k != ff || de.n != n_exp * n_embd || ge.t != ue.t {
                return Err("MTP の専門家の形が違う".into());
            }
            let st = |m: &G3Mat, k: usize, n: usize| -> Result<i64, String> {
                let b = unsafe { ffi::infer_g3_bytes(m.t, k as i32, n as i32) };
                if b <= 0 { Err("g3 の大きさ".into()) } else { Ok(b) }
            };
            let (es_gu, es_d) = (st(&ge, n_embd, ff)?, st(&de, ff, n_embd)?);
            let r = MAX_ROWS;
            let rr = r * kk;
            let fb = |n: usize| DevBuf::new(n * 4);
            w.moe = Some(MtpMoe {
                router: DevBuf::from_bytes(&rb),
                gate: ge,
                up: ue,
                down: de,
                n_exp,
                k: kk,
                ff,
                es_gu,
                es_d,
                rl: fb(r * (n_exp + 1)),
                ids: fb(rr),
                wts: fb(rr),
                jr0: fb(rr),
                jcnt: fb(rr),
                xidx: fb(rr),
                yg: fb(rr * ff),
                yu: fb(rr * ff),
                hh: fb(rr * ff),
                hp: DevBuf::new(unsafe { ffi::infer_g3_xprep_bytes(ff as i32, rr as i32) } as usize),
                o: fb(rr * n_embd),
                sh: fb(r * n_embd),
            });
        }
        let w = w;
        let n_vocab = w.output.n;
        let dims = Dims { n_embd, n_head, n_head_kv, head_dim, n_ff, n_vocab, eps, il };
        // 形の確かめ
        let chk = |m: &G3Mat, k: usize, n: usize, nm: &str| {
            if m.k != k || m.n != n {
                Err(format!("{nm}: 形 {}×{} ≠ {k}×{n}", m.k, m.n))
            } else {
                Ok(())
            }
        };
        let (hd, kvd) = (n_head * head_dim, n_head_kv * head_dim);
        chk(&w.eh_proj, 2 * n_embd, n_embd, "eh_proj")?;
        chk(&w.attn_q, n_embd, 2 * hd, "attn_q")?;
        chk(&w.attn_k, n_embd, kvd, "attn_k")?;
        chk(&w.attn_v, n_embd, kvd, "attn_v")?;
        chk(&w.attn_output, hd, n_embd, "attn_output")?;
        chk(&w.ffn_gate, n_embd, n_ff, "ffn_gate")?;
        chk(&w.ffn_up, n_embd, n_ff, "ffn_up")?;
        chk(&w.ffn_down, n_ff, n_embd, "ffn_down")?;
        chk(&w.output, n_embd, n_vocab, "output")?;
        if w.embd.k != n_embd {
            return Err("token_embd の幅が違う".into());
        }
        let r = MAX_ROWS;
        let pages_per_slot = opts.max_ctx.div_ceil(attn::PAGE);
        let n_pages = if opts.kv_pages > 0 { opts.kv_pages } else { opts.slots * pages_per_slot };
        let page_bytes = ash.page_bytes(false);
        let kc = DevBuf::new(n_pages * page_bytes);
        let vc = DevBuf::new(n_pages * page_bytes);
        let xpb = |k: usize| unsafe { ffi::infer_g3_xprep_bytes(k as i32, r as i32) } as usize;
        let xp_a = DevBuf::new(xpb(2 * n_embd).max(xpb(hd)).max(xpb(n_embd)));
        let xp_b = DevBuf::new(xpb(n_ff));
        // Attention の作業場は最大の形で取っておく
        let mut ap = AttnParams::new(&ash);
        ap.n_seq = r as i32;
        ap.n_tok = r as i32;
        ap.max_q_len = r as i32;
        ap.max_kv = opts.max_ctx as i32;
        let mut wsb = 0usize;
        for ns in 1..=r {
            for nq in 1..=r / ns {
                ap.n_seq = ns as i32;
                ap.n_tok = (ns * nq) as i32;
                ap.max_q_len = nq as i32;
                wsb = wsb.max(attn::ws_bytes(&ap));
            }
        }
        // 上位 k の作業場は行の数で split の数が変わるので、1..32 行の最大で
        let tkb = (1..=r).map(|n| unsafe { kffi::infer_mtp_topk_ws_bytes(n_vocab as i32, n as i32, opts.topk as i32) }).max().unwrap() as usize;
        let topk_ws = DevBuf::new(tkb);
        let meta = Meta::new(r, pages_per_slot, opts.slots);
        let fb = |n: usize| DevBuf::new(n * 4);
        let slots = opts.slots;
        Ok(MtpHead {
            w,
            dims,
            ash,
            pages_per_slot,
            kc,
            vc,
            cat: fb(r * 2 * n_embd),
            x: fb(r * n_embd),
            a: fb(r * n_embd),
            qf: fb(r * 2 * hd),
            kk: fb(r * kvd),
            vv: fb(r * kvd),
            q16: DevBuf::new(r * hd * 2),
            ao: fb(r * hd),
            hout: [fb(r * n_embd), fb(r * n_embd)],
            cur: 0,
            hsel: fb(r * n_embd),
            logits: fb(r * n_vocab),
            xp_a,
            xp_b,
            attn_ws: DevBuf::new(wsb.max(16)),
            topk_ws,
            top_i: fb(r * MAX_TOPK),
            top_v: fb(r * MAX_TOPK),
            top_l: fb(CHAIN * r),
            meta,
            meta_ch: [Meta::new(r, pages_per_slot, opts.slots), Meta::new(r, pages_per_slot, opts.slots)],
            draft_ids_dev: None,
            chain: std::env::var("TACHYON_NO_MTP_CHAIN").map_or(true, |v| v != "1"),
            hcap_shadow: std::env::var("TACHYON_HCAP_SHADOW").map_or(false, |v| v == "1"),
            gp_dev: DevBuf::new(G_STAGES * MAX_ROWS * std::mem::size_of::<GParam>()),
            gkey: DevBuf::new(G_STAGES * MAX_ROWS * 8),
            pend: fb(slots * n_embd),
            pend_pos: vec![None; slots],
            committed: vec![0; slots],
            slot_of: HashMap::new(),
            free_slots: (0..slots).rev().collect(),
            opts,
            trace: None,
            draft_vocab: None,
            big: None,
            moe_big: None,
            ext_pages: vec![],
            draft_min_p: 0.0,
            draft_min_p_deep: 0.0,
            gemm_catchup: true,
            media: vec![vec![]; slots],
            log: None,
            n_forward: 0,
            n_rows: 0,
        })
    }

    /// 下書きの lm_head を ids の語だけに絞る（g の output.weight から。topk = 1 の argmax の下書きでだけ使う）
    pub fn set_draft_vocab(&mut self, g: &Model, ids: &[u32]) -> Result<(), String> {
        if self.opts.topk > 2 {
            return Err("語彙を絞るのは argmax の下書き（topk ≤ 2。2 は記録用）だけ".into());
        }
        if ids.is_empty() || ids.iter().any(|&i| i as usize >= self.dims.n_vocab) {
            return Err("下書きの語彙が変".into());
        }
        let mut u = ids.to_vec();
        u.sort_unstable();
        u.dedup();
        if u.len() != ids.len() {
            return Err("下書きの語彙に同じ語が 2 回ある".into());
        }
        let m = G3Mat::load_rows(&g.tensor("output.weight"), ids)?;
        if m.k != self.dims.n_embd {
            return Err("output.weight の幅が違う".into());
        }
        self.draft_ids_dev = Some(DevBuf::from_i32(&ids.iter().map(|&x| x as i32).collect::<Vec<_>>()));
        self.draft_vocab = Some((m, ids.to_vec()));
        Ok(())
    }

    /// GPU に持っている量（バイト。借り物は数えない）
    pub fn vram_bytes(&self) -> usize {
        let w = &self.w;
        let mats = [&w.eh_proj, &w.attn_q, &w.attn_k, &w.attn_v, &w.attn_output, &w.ffn_gate, &w.ffn_up, &w.ffn_down, &w.output];
        let mut s: usize = mats.iter().map(|m| m.bytes()).sum::<usize>() + w.embd.bytes() + self.draft_vocab.as_ref().map_or(0, |(m, _)| m.bytes());
        for b in [&self.kc, &self.vc, &self.cat, &self.x, &self.a, &self.qf, &self.kk, &self.vv, &self.q16, &self.ao, &self.hsel, &self.logits] {
            s += b.len;
        }
        let big = self.big.as_ref().map_or(0, |b| {
            [&b.meta, &b.cat, &b.x16, &b.x, &b.a, &b.qf, &b.kk, &b.vv, &b.q16, &b.o16, &b.mid16].iter().map(|x| x.len).sum::<usize>()
        });
        let moe = w.moe.as_ref().map_or(0, |m| {
            m.gate.bytes() + m.up.bytes() + m.down.bytes()
                + [&m.router, &m.rl, &m.ids, &m.wts, &m.jr0, &m.jcnt, &m.xidx, &m.yg, &m.yu, &m.hh, &m.hp, &m.o, &m.sh].iter().map(|x| x.len).sum::<usize>()
        });
        s + self.hout[0].len * 2 + self.xp_a.len + self.xp_b.len + self.attn_ws.len + self.topk_ws.len + self.pend.len + big + moe
    }

    /// 最後の forward の h' の行 i の番地
    pub fn hout_ptr(&self, i: usize) -> *const f32 {
        unsafe { self.hout[self.cur].f32p().add(i * self.dims.n_embd) }
    }

    /// 最後の forward の h' の行（答え合わせ用）
    pub fn hout_rows(&self, n: usize) -> Vec<f32> {
        let e = self.dims.n_embd;
        let mut v = vec![0f32; n * e];
        cuda::check(unsafe { ffi::cudaMemcpy(v.as_mut_ptr().cast(), self.hout[self.cur].ptr, n * e * 4, 2) }, "hout D2H");
        v
    }

    /// 最後の forward の logits の行（want の順の i 番目。答え合わせ用）
    pub fn logits_row(&self, i: usize) -> Vec<f32> {
        let v = self.dims.n_vocab;
        let mut out = vec![0f32; v];
        cuda::check(unsafe { ffi::cudaMemcpy(out.as_mut_ptr().cast(), self.logits.f32p().add(i * v).cast(), v * 4, 2) }, "logits D2H");
        out
    }

    /// 列 seq の画像（本体の Seq::media と同じ並び）。free で消える
    pub fn set_media(&mut self, seq: SeqId, spans: Vec<(usize, usize, usize)>) -> Result<(), String> {
        let s = self.slot(seq)?;
        self.media[s] = spans;
        Ok(())
    }

    /// 升 slot の位置 p の RoPE の位置（crate::model::Seq::rope_pos と同じ決まり）
    fn rope_pos(&self, slot: usize, p: usize) -> i32 {
        let mut delta = 0i64;
        for &(start, nx, ny) in &self.media[slot] {
            let n = nx * ny;
            if p < start {
                break;
            }
            if p < start + n {
                return (start as i64 + delta) as i32;
            }
            delta += nx.max(ny) as i64 - n as i64;
        }
        (p as i64 + delta) as i32
    }

    fn pend_ptr(&self, slot: usize) -> *mut f32 {
        unsafe { self.pend.f32p().add(slot * self.dims.n_embd) }
    }

    /// 1 回の MTP（rows ≤ 32 行）。行は升ごとに続いた位置で並べる（升の塊が Attention の列になる）。
    /// KV の位置 pos に書く。want の行だけ logits を出して上位 kout（≤ opts.topk）を返す
    pub fn forward(&mut self, rows: &[Row], want: &[usize], kout: usize) -> Result<Vec<TopRow>, String> {
        self.forward_g(rows, want, kout, None)
    }

    /// GParam を段 stage の置き場に送る（行は want の順）。どの行も off なら None
    fn put_g(&mut self, stage: usize, g: &[GParam]) -> Option<usize> {
        if g.is_empty() || !g.iter().any(|x| x.on != 0) || stage >= G_STAGES || g.len() > MAX_ROWS {
            return None;
        }
        let off = stage * MAX_ROWS;
        let sz = std::mem::size_of::<GParam>();
        unsafe {
            cuda::check(
                ffi::cudaMemcpy((self.gp_dev.ptr as *mut u8).add(off * sz).cast(), g.as_ptr().cast(), g.len() * sz, 1),
                "gparam H2D",
            );
        }
        Some(off)
    }

    /// forward に、want の行の下書きを Gumbel-max で選び直す引数（g。want と同じ数）を足した物
    pub fn forward_g(&mut self, rows: &[Row], want: &[usize], kout: usize, g: Option<&[GParam]>) -> Result<Vec<TopRow>, String> {
        if rows.is_empty() {
            return Ok(vec![]);
        }
        let goff = g.and_then(|g| self.put_g(0, g));
        let (p, n) = self.plan(0, rows, want, kout)?;
        self.meta.upload(n);
        self.enq(0, &p, kout, 0, None, goff)?;
        if want.is_empty() {
            return Ok(vec![]);
        }
        Ok(self.read_tops(&[(0, want.len())], kout).pop().unwrap())
    }

    fn meta_at(&mut self, mi: usize) -> &mut Meta {
        if mi == 0 {
            &mut self.meta
        } else {
            &mut self.meta_ch[mi - 1]
        }
    }

    /// 行を列に分けて meta（mi 番）の手元の写しに書く（送るのは呼ぶ側。送る長さを返す）
    fn plan(&mut self, mi: usize, rows: &[Row], want: &[usize], kout: usize) -> Result<(Plan, usize), String> {
        let b = rows.len();
        if b > MAX_ROWS || want.len() > b || kout == 0 || kout > self.opts.topk {
            return Err(format!("forward: 行 {b}・want {}・kout {kout}", want.len()));
        }
        // 列（升の塊）に分ける
        let mut q_start = vec![0i32];
        let mut kv_len = vec![];
        let mut bt: Vec<i32> = vec![];
        let mut max_q = 0usize;
        let mut max_kv = 0usize;
        let mut i = 0;
        while i < b {
            let mut j = i + 1;
            while j < b && rows[j].slot == rows[i].slot && rows[j].pos == rows[j - 1].pos + 1 {
                j += 1;
            }
            let r = rows[i];
            if r.slot >= self.opts.slots || rows[j - 1].pos >= self.opts.max_ctx {
                return Err(format!("forward: 升 {}・位置 {} が上限を超える", r.slot, rows[j - 1].pos));
            }
            if rows[..i].iter().any(|x| x.slot == r.slot) {
                return Err(format!("forward: 升 {} の行が離れている", r.slot));
            }
            q_start.push(j as i32);
            kv_len.push((rows[j - 1].pos + 1) as i32);
            bt.extend((0..self.pages_per_slot).map(|p| self.page_pad(r.slot, p) as i32));
            max_q = max_q.max(j - i);
            max_kv = max_kv.max(rows[j - 1].pos + 1);
            i = j;
        }
        let n_seq = kv_len.len();
        let tok: Vec<i32> = rows.iter().map(|r| r.token as i32).collect();
        if rows.iter().any(|r| r.token as usize >= self.w.embd.n) {
            return Err("forward: 語の番号が語彙を超える".into());
        }
        let pos: Vec<i32> = rows.iter().map(|r| self.rope_pos(r.slot, r.pos)).collect();
        let slot: Vec<i32> = rows.iter().map(|r| (self.page(r.slot, r.pos / attn::PAGE) * attn::PAGE + r.pos % attn::PAGE) as i32).collect();
        let hptr: Vec<u64> = rows.iter().map(|r| r.h as u64).collect();
        let gidx: Vec<i32> = want.iter().map(|&w| w as i32).collect();
        let m = self.meta_at(mi);
        let (mt, mp, ms, mh, mq, mk, mb, mg) = (m.tok, m.pos, m.slot, m.hptr, m.q_start, m.kv_len, m.bt, m.gidx);
        m.put_i32(mt, &tok);
        m.put_i32(mp, &pos);
        m.put_i32(ms, &slot);
        m.put_u64(mh, &hptr);
        m.put_i32(mq, &q_start);
        m.put_i32(mk, &kv_len);
        m.put_i32(mb, &bt);
        m.put_i32(mg, &gidx);
        Ok((Plan { b, n_seq, max_q, max_kv, nw: want.len() }, mb + bt.len() * 4))
    }

    /// meta（mi 番。送ってあること）の行を流す。上位 kout を top_* の toff 行目から書く。
    /// feed が Some(ftop) なら、まず top_i の ftop 行目から（meta の feed の行の順で）語を拾って入力の語にする
    fn enq(&mut self, mi: usize, p: &Plan, kout: usize, toff: usize, feed: Option<usize>, goff: Option<usize>) -> Result<(), String> {
        let d = self.dims;
        let (e, hd, kvd) = (d.n_embd, d.n_head * d.head_dim, d.n_head_kv * d.head_dim);
        let s: Stream = null_mut();
        let b = p.b;
        let m = if mi == 0 { &self.meta } else { &self.meta_ch[mi - 1] };
        let (mt, mp, ms, mh, mq, mk, mb, mg, mf) = (m.tok, m.pos, m.slot, m.hptr, m.q_start, m.kv_len, m.bt, m.gidx, m.feed);
        let mbase = m.dev.ptr as *mut u8;
        let md = |at: usize| unsafe { mbase.add(at) };
        if let Some(ftop) = feed {
            let map = self.draft_ids_dev.as_ref().filter(|_| self.trace.is_none()).map_or(null(), |x| x.ptr as *const i32);
            ck(
                unsafe { kffi::infer_mtp_feed((self.top_i.ptr as *const i32).add(ftop * kout), kout as i32, md(mf).cast(), map, b as i32, md(mt).cast(), s) },
                "mtp feed",
            )?;
        }
        let w = &self.w;
        let bi = b as i32;
        let g3 = |mat: &G3Mat, xp: &DevBuf, y: &DevBuf, add: bool, rows: i32| -> Result<(), String> {
            ck(unsafe { ffi::infer_g3_gemv(mat.t, mat.ptr, mat.k as i32, mat.n as i32, xp.ptr, y.f32p(), rows, 0, add as i32, s) }, "g3 gemv")
        };
        let xprep = |x: *const f32, k: usize, xp: &DevBuf, rows: i32| -> Result<(), String> {
            ck(unsafe { ffi::infer_g3_xprep(x, k as i32, rows, xp.ptr, s) }, "g3 xprep")
        };
        let rms = |x: &DevBuf, wt: &DevBuf, y: &DevBuf| -> Result<(), String> {
            ck(unsafe { kffi::infer_mtp_rmsnorm(x.f32p(), e as i64, wt.f32p(), e as i32, bi, d.eps, y.f32p(), e as i64, s) }, "rmsnorm")
        };
        // e・hh → [e ; hh] → eh_proj
        ck(
            unsafe {
                kffi::infer_mtp_prep(
                    w.embd.qtype, w.embd.ptr, w.embd.row_bytes as i64, md(mt).cast(), md(mh).cast(), w.enorm.f32p(), w.hnorm.f32p(), e as i32, bi,
                    d.eps, self.cat.f32p(), s,
                )
            },
            "mtp prep",
        )?;
        xprep(self.cat.f32p(), 2 * e, &self.xp_a, bi)?;
        g3(&w.eh_proj, &self.xp_a, &self.x, false, bi)?;
        // Attention
        rms(&self.x, &w.attn_norm, &self.a)?;
        xprep(self.a.f32p(), e, &self.xp_a, bi)?;
        {
            let qt = [w.attn_q.t, w.attn_k.t, w.attn_v.t];
            let wp = [w.attn_q.ptr, w.attn_k.ptr, w.attn_v.ptr];
            let ns = [w.attn_q.n as i32, w.attn_k.n as i32, w.attn_v.n as i32];
            let ys = [self.qf.f32p(), self.kk.f32p(), self.vv.f32p()];
            ck(
                unsafe { ffi::infer_g3_gemv_multi(3, qt.as_ptr(), wp.as_ptr(), ns.as_ptr(), ys.as_ptr(), e as i32, self.xp_a.ptr, bi, 0, 0, s) },
                "g3 qkv",
            )?;
        }
        let mut rp = RopeParams::new(&self.ash);
        rp.q = self.qf.f32p();
        rp.k = self.kk.f32p();
        rp.v = self.vv.f32p();
        rp.q_ld = (2 * hd) as i64;
        rp.k_ld = kvd as i64;
        rp.v_ld = kvd as i64;
        rp.q_norm = w.q_norm.f32p();
        rp.k_norm = w.k_norm.f32p();
        rp.pos = md(mp).cast();
        rp.slot = md(ms).cast();
        rp.q_out = self.q16.ptr;
        rp.kc = self.kc.ptr;
        rp.vc = self.vc.ptr;
        rp.n = bi;
        attn::rope(&rp, s)?;
        let mut ap = AttnParams::new(&self.ash);
        ap.q = self.q16.ptr;
        ap.gate = unsafe { self.qf.f32p().add(d.head_dim) };
        ap.gate_ld = (2 * hd) as i64;
        ap.gate_hs = (2 * d.head_dim) as i64;
        ap.kc = self.kc.ptr;
        ap.vc = self.vc.ptr;
        ap.block_table = md(mb).cast();
        ap.bt_ld = self.pages_per_slot as i32;
        ap.q_start = md(mq).cast();
        ap.kv_len = md(mk).cast();
        ap.n_seq = p.n_seq as i32;
        ap.n_tok = bi;
        ap.max_q_len = p.max_q as i32;
        ap.max_kv = p.max_kv as i32;
        ap.out = self.ao.ptr;
        ap.ws = self.attn_ws.f32p();
        ap.ws_bytes = self.attn_ws.len as i64;
        attn::attn(&ap, s)?;
        xprep(self.ao.f32p(), hd, &self.xp_a, bi)?;
        g3(&w.attn_output, &self.xp_a, &self.x, true, bi)?;
        // FFN
        if w.moe.is_some() {
            self.moe_ffn(self.x.f32p(), b, s)?;
        } else {
            rms(&self.x, &w.post_norm, &self.a)?;
            xprep(self.a.f32p(), e, &self.xp_a, bi)?;
            ck(
                unsafe {
                    ffi::infer_g3_swiglu(w.ffn_gate.t, w.ffn_gate.ptr, w.ffn_up.t, w.ffn_up.ptr, e as i32, d.n_ff as i32, self.xp_a.ptr, null_mut(), self.xp_b.ptr, bi, s)
                },
                "g3 swiglu",
            )?;
            g3(&w.ffn_down, &self.xp_b, &self.x, true, bi)?;
        }
        // h' と logits
        self.cur ^= 1;
        let ho = &self.hout[self.cur];
        rms(&self.x, &w.head_norm, ho)?;
        self.n_forward += 1;
        self.n_rows += b;
        if p.nw == 0 {
            return Ok(());
        }
        let nw = p.nw as i32;
        ck(unsafe { kffi::infer_mtp_gather(ho.f32p(), e as i64, md(mg).cast(), e as i32, nw, self.hsel.f32p(), s) }, "gather")?;
        xprep(self.hsel.f32p(), e, &self.xp_a, nw)?;
        let outm = match &self.draft_vocab {
            Some((m, _)) if self.trace.is_none() => m,
            _ => &w.output,
        };
        g3(outm, &self.xp_a, &self.logits, false, nw)?;
        let v = outm.n;
        ck(
            unsafe {
                kffi::infer_mtp_topk(
                    self.logits.f32p(), v as i32, v as i64, nw, kout as i32, self.topk_ws.ptr, self.topk_ws.len as i64,
                    (self.top_i.ptr as *mut i32).add(toff * kout), self.top_v.f32p().add(toff * kout), self.top_l.f32p().add(toff), s,
                )
            },
            "topk",
        )?;
        if let Some(go) = goff {
            let map = self.draft_ids_dev.as_ref().filter(|_| self.draft_vocab.is_some() && self.trace.is_none()).map_or(null(), |x| x.ptr as *const i32);
            let sz = std::mem::size_of::<GParam>();
            ck(
                unsafe {
                    kffi::infer_mtp_gumbel(
                        self.logits.f32p(), v as i32, v as i64, nw, map, (self.gp_dev.ptr as *const u8).add(go * sz).cast(),
                        (self.gkey.ptr as *mut u8).add(go * 8).cast(), kout as i32, (self.top_i.ptr as *mut i32).add(toff * kout), s,
                    )
                },
                "gumbel",
            )?;
        }
        Ok(())
    }

    /// 頭の n_ch 段（1 段目 = rows0 の want0 の行、2 段目から = 列ごとに 1 行）を、手元に戻らず 1 回で流す（argmax の下書き・kout = 1）。
    /// 2 段目からの行は run の段の決まりと同じ（k > j の列を head の順に）。語は前の段の 1 位を GPU の上で拾う
    fn chain_run(&mut self, rows0: &[Row], want0: &[usize], heads: &[Row], asks: &[DraftAsk], n_ch: usize) -> Result<Vec<Vec<TopRow>>, String> {
        let e = self.dims.n_embd;
        let (p0, n0) = self.plan(0, rows0, want0, 1)?;
        let mut plans = vec![(p0, n0)];
        let mut buf = self.cur ^ 1; // 1 段目が書く h' の置き場
        let mut prev: Vec<(usize, usize)> = (0..heads.len()).map(|hi| (hi, want0[hi])).collect(); // (列, 前の段の h' の行)
        // 段ごとの結び付けの引数（行は want の順 = 列の順）
        let mut goffs: Vec<Option<usize>> = vec![];
        let g0: Vec<GParam> = (0..heads.len()).map(|hi| gparam(&asks[hi], 0)).collect();
        goffs.push(self.put_g(0, &g0));
        for j in 1..n_ch {
            let live: Vec<(usize, usize, usize)> =
                prev.iter().enumerate().filter(|(_, &(hi, _))| asks[hi].k > j).map(|(ti, &(hi, hr))| (ti, hi, hr)).collect();
            if live.is_empty() {
                break;
            }
            let hb = self.hout[buf].f32p();
            let rows: Vec<Row> =
                live.iter().map(|&(_, hi, hr)| Row { slot: heads[hi].slot, pos: asks[hi].pos + j, token: 0, h: unsafe { hb.add(hr * e) } }).collect();
            let want: Vec<usize> = (0..rows.len()).collect();
            let (pj, nj) = self.plan(j, &rows, &want, 1)?;
            let gj: Vec<GParam> = live.iter().map(|&(_, hi, _)| gparam(&asks[hi], j)).collect();
            goffs.push(self.put_g(j, &gj));
            let src: Vec<i32> = live.iter().map(|&(ti, _, _)| ti as i32).collect();
            let m = self.meta_at(j);
            let f = m.feed;
            m.put_i32(f, &src);
            plans.push((pj, nj));
            prev = live.iter().enumerate().map(|(r, &(_, hi, _))| (hi, r)).collect();
            buf ^= 1;
        }
        for (j, (_, n)) in plans.iter().enumerate() {
            let n = *n;
            self.meta_at(j).upload(n);
        }
        let mut segs = vec![];
        let mut toff = 0;
        for (j, (p, _)) in plans.iter().enumerate() {
            let feed = if j == 0 { None } else { Some(segs.last().map_or(0, |&(o, _): &(usize, usize)| o)) };
            self.enq(j, p, 1, toff, feed, goffs[j])?;
            segs.push((toff, p.nw));
            toff += p.nw;
        }
        Ok(self.read_tops(&segs, 1))
    }

    /// 上位 kout を手元へ（segs: (toff, 行の数) ごとに）
    fn read_tops(&self, segs: &[(usize, usize)], kout: usize) -> Vec<Vec<TopRow>> {
        let n = segs.iter().map(|&(o, c)| o + c).max().unwrap_or(0);
        let mut ti = vec![0i32; n * kout];
        let mut tv = vec![0f32; n * kout];
        let mut tl = vec![0f32; n];
        unsafe {
            cuda::check(ffi::cudaMemcpy(ti.as_mut_ptr().cast(), self.top_i.ptr, n * kout * 4, 2), "topk D2H");
            cuda::check(ffi::cudaMemcpy(tv.as_mut_ptr().cast(), self.top_v.ptr, n * kout * 4, 2), "topk D2H");
            cuda::check(ffi::cudaMemcpy(tl.as_mut_ptr().cast(), self.top_l.ptr, n * 4, 2), "topk D2H");
        }
        let map = match &self.draft_vocab {
            Some((_, ids)) if self.trace.is_none() => Some(ids),
            _ => None,
        };
        let un = |x: i32| -> u32 {
            match map {
                Some(ids) if x >= 0 => ids[x as usize],
                _ => x as u32,
            }
        };
        segs.iter()
            .map(|&(o, c)| {
                (o..o + c)
                    .map(|r| TopRow { ids: ti[r * kout..(r + 1) * kout].iter().map(|&x| un(x)).collect(), vals: tv[r * kout..(r + 1) * kout].to_vec(), lse: tl[r] })
                    .collect()
            })
            .collect()
    }

    /// 追いつきの行（1 つの升の続いた位置、33..=GEMM_ROWS 行）を GEMM の道で（logits も h' も出さない。KV を書くだけ）。
    /// 式は forward と同じ（行列は g3 の並びのまま GEMM。活性は fp16 で掛ける）
    /// MoE の FFN（3.6 の MTP の層）: x [rows][n_embd] += Σ w_j·down_e(SiLU(gate_e b)·up_e b) + sigmoid(g·b)·shared(b)、b = post_norm(x)。
    /// MAX_ROWS 行ずつ「語 × 枠」の仕事で（手元を待たない）。式と部品は本体の Model::moe_ffn のデコードの形と同じ
    fn moe_ffn(&self, x: *mut f32, rows: usize, s: Stream) -> Result<(), String> {
        let w = &self.w;
        let m = w.moe.as_ref().unwrap();
        let d = self.dims;
        let (e, ne, kk, ff, sff) = (d.n_embd, m.n_exp, m.k, m.ff, d.n_ff);
        let mut c0 = 0;
        while c0 < rows {
            let b = (rows - c0).min(MAX_ROWS);
            let bi = b as i32;
            let xc = unsafe { x.add(c0 * e) };
            let nj = b * kk;
            ck(unsafe { kffi::infer_mtp_rmsnorm(xc, e as i64, w.post_norm.f32p(), e as i32, bi, d.eps, self.a.f32p(), e as i64, s) }, "rmsnorm")?;
            ck(unsafe { ffi::infer_g3_xprep(self.a.f32p(), e as i32, bi, self.xp_a.ptr, s) }, "g3 xprep")?;
            ck(unsafe { kffi::infer_moe_router(m.router.f32p(), e as i32, (ne + 1) as i32, self.a.f32p(), e as i64, bi, m.rl.f32p(), s) }, "moe router")?;
            ck(
                unsafe {
                    kffi::infer_moe_topk(m.rl.f32p(), (ne + 1) as i64, ne as i32, kk as i32, bi, m.ids.ptr.cast(), m.wts.f32p(), m.jr0.ptr.cast(),
                                         m.jcnt.ptr.cast(), m.xidx.ptr.cast(), s)
                },
                "moe topk",
            )?;
            ck(
                unsafe {
                    ffi::infer_g3_swiglu(w.ffn_gate.t, w.ffn_gate.ptr, w.ffn_up.t, w.ffn_up.ptr, e as i32, sff as i32, self.xp_a.ptr, null_mut(), self.xp_b.ptr,
                                         bi, s)
                },
                "g3 swiglu（共有）",
            )?;
            ck(unsafe { ffi::infer_g3_gemv(w.ffn_down.t, w.ffn_down.ptr, sff as i32, e as i32, self.xp_b.ptr, m.sh.f32p(), bi, 0, 0, s) }, "共有の down")?;
            let (gq, gw, ges, gn, gy) = ([m.gate.t, m.up.t], [m.gate.ptr, m.up.ptr], [m.es_gu, m.es_gu], [ff as i32, ff as i32], [m.yg.f32p(), m.yu.f32p()]);
            ck(
                unsafe {
                    kffi::infer_g3_moe_gemv(2, gq.as_ptr(), gw.as_ptr(), ges.as_ptr(), gn.as_ptr(), gy.as_ptr(), e as i32, self.xp_a.ptr, bi, m.xidx.ptr.cast(),
                                            m.ids.ptr.cast(), m.jr0.ptr.cast(), m.jcnt.ptr.cast(), nj as i32, 1, s)
                },
                "moe gate・up",
            )?;
            ck(unsafe { kffi::infer_moe_swiglu(m.yg.f32p(), m.yu.f32p(), (nj * ff) as i64, m.hh.f32p(), s) }, "moe swiglu")?;
            ck(unsafe { ffi::infer_g3_xprep(m.hh.f32p(), ff as i32, nj as i32, m.hp.ptr, s) }, "moe xprep")?;
            let (dq, dw, des, dn, dy) = ([m.down.t], [m.down.ptr], [m.es_d], [e as i32], [m.o.f32p()]);
            ck(
                unsafe {
                    kffi::infer_g3_moe_gemv(1, dq.as_ptr(), dw.as_ptr(), des.as_ptr(), dn.as_ptr(), dy.as_ptr(), ff as i32, m.hp.ptr, nj as i32, null(),
                                            m.ids.ptr.cast(), m.jr0.ptr.cast(), m.jcnt.ptr.cast(), nj as i32, 1, s)
                },
                "moe down",
            )?;
            ck(
                unsafe {
                    kffi::infer_moe_combine(xc, e as i32, bi, kk as i32, m.wts.f32p(), m.o.f32p(), null(), m.sh.f32p(), m.rl.f32p().add(ne), (ne + 1) as i64, s)
                },
                "moe combine",
            )?;
            c0 += b;
        }
        Ok(())
    }

    /// MoE の FFN を専門家ごとの GEMM で（追いつきの行 33..=GEMM_ROWS 行）。式は moe_ffn と同じ（本体の Model::moe_ffn のプレフィルの道と同じ部品）
    fn moe_ffn_gemm(&self, x: *mut f32, rows: usize, s: Stream) -> Result<(), String> {
        let w = &self.w;
        let m = w.moe.as_ref().unwrap();
        let b = self.moe_big.as_ref().unwrap();
        let d = self.dims;
        let (e, ne, kk, ff, sff) = (d.n_embd, m.n_exp, m.k, m.ff, d.n_ff);
        let (n, r) = (rows as i32, rows * kk);
        ck(unsafe { kffi::infer_mtp_rmsnorm(x, e as i64, w.post_norm.f32p(), e as i32, n, d.eps, b.a.f32p(), e as i64, s) }, "rmsnorm")?;
        ck(unsafe { kffi::infer_m_f32_to_f16(b.a.f32p(), (rows * e) as i64, b.x16.ptr, s) }, "f32→f16")?;
        ck(unsafe { kffi::infer_moe_router(m.router.f32p(), e as i32, (ne + 1) as i32, b.a.f32p(), e as i64, n, b.rl.f32p(), s) }, "moe router")?;
        ck(
            unsafe {
                kffi::infer_moe_topk(b.rl.f32p(), (ne + 1) as i64, ne as i32, kk as i32, n, b.ids.ptr.cast(), b.wts.f32p(), null_mut(), null_mut(),
                                     null_mut(), s)
            },
            "moe topk",
        )?;
        // 共有の専門家（GEMM・SwiGLU → down）
        let (ts, ws, ns, yv) = ([w.ffn_gate.t, w.ffn_up.t], [w.ffn_gate.ptr, w.ffn_up.ptr], [sff as i32, sff as i32], [b.mid16.ptr, b.mid16.ptr]);
        ck(
            unsafe { kffi::infer_g3_gemm_ex(2, ts.as_ptr(), ws.as_ptr(), ns.as_ptr(), yv.as_ptr(), null(), e as i32, n, b.x16.ptr, null(), e as i32, 3, -1, s) },
            "mtp gemm swiglu（共有）",
        )?;
        let (td, wd, nd, yd) = ([w.ffn_down.t], [w.ffn_down.ptr], [e as i32], [b.sh.ptr]);
        ck(
            unsafe { kffi::infer_g3_gemm_ex(1, td.as_ptr(), wd.as_ptr(), nd.as_ptr(), yd.as_ptr(), null(), sff as i32, n, b.mid16.ptr, null(), sff as i32, 0, -1, s) },
            "mtp gemm（共有の down）",
        )?;
        // 専門家ごと
        let n_blk = (ne + r / MOE_TOKB) as i32;
        ck(
            unsafe {
                kffi::infer_moe_group(b.ids.ptr.cast(), r as i32, kk as i32, ne as i32, MOE_TOKB as i32, b.xidx.ptr.cast(), b.rowmap.ptr.cast(),
                                      b.grp.ptr.cast(), n_blk, s)
            },
            "moe group",
        )?;
        let (gq, gw, ges, gn, gy) = ([m.gate.t, m.up.t], [m.gate.ptr, m.up.ptr], [m.es_gu, m.es_gu], [ff as i32, ff as i32], [b.h16.ptr, b.h16.ptr]);
        ck(
            unsafe {
                kffi::infer_g3_moe_gemm(2, gq.as_ptr(), gw.as_ptr(), ges.as_ptr(), gn.as_ptr(), gy.as_ptr(), null(), e as i32, b.x16.ptr, null(), e as i32,
                                        b.xidx.ptr.cast(), b.grp.ptr.cast(), n_blk, MOE_TOKB as i32, 3, s)
            },
            "mtp moe gemm gate・up",
        )?;
        let (dq, dw, des, dn, dy) = ([m.down.t], [m.down.ptr], [m.es_d], [e as i32], [b.o.ptr]);
        ck(
            unsafe {
                kffi::infer_g3_moe_gemm(1, dq.as_ptr(), dw.as_ptr(), des.as_ptr(), dn.as_ptr(), dy.as_ptr(), null(), ff as i32, b.h16.ptr, null(), ff as i32,
                                        null(), b.grp.ptr.cast(), n_blk, MOE_TOKB as i32, 0, s)
            },
            "mtp moe gemm down",
        )?;
        ck(
            unsafe {
                kffi::infer_moe_combine(x, e as i32, n, kk as i32, b.wts.f32p(), b.o.f32p(), b.rowmap.ptr.cast(), b.sh.f32p(), b.rl.f32p().add(ne), (ne + 1) as i64, s)
            },
            "moe combine",
        )
    }

    pub fn forward_gemm(&mut self, rows: &[Row]) -> Result<(), String> {
        let n = rows.len();
        if n == 0 {
            return Ok(());
        }
        if n > GEMM_ROWS {
            return Err(format!("forward_gemm: {n} 行は多い（{GEMM_ROWS} まで）"));
        }
        let r0 = rows[0];
        if rows.iter().enumerate().any(|(i, r)| r.slot != r0.slot || r.pos != r0.pos + i) {
            return Err("forward_gemm: 1 つの升の続いた位置だけ".into());
        }
        if r0.slot >= self.opts.slots || r0.pos + n > self.opts.max_ctx {
            return Err(format!("forward_gemm: 升 {}・位置 {} が上限を超える", r0.slot, r0.pos + n));
        }
        if rows.iter().any(|r| r.token as usize >= self.w.embd.n) {
            return Err("forward_gemm: 語の番号が語彙を超える".into());
        }
        let d = self.dims;
        let (e, hd, kvd, ff) = (d.n_embd, d.n_head * d.head_dim, d.n_head_kv * d.head_dim, d.n_ff);
        let pps = self.pages_per_slot;
        let gr = GEMM_ROWS;
        if mtp_moe_gemm_on() && self.moe_big.is_none() {
            if let Some(m) = self.w.moe.as_ref() {
                let rr = gr * m.k;
                self.moe_big = Some(MoeBig {
                    rl: DevBuf::new(gr * (m.n_exp + 1) * 4),
                    ids: DevBuf::new(rr * 4),
                    wts: DevBuf::new(rr * 4),
                    xidx: DevBuf::new(rr * 4),
                    rowmap: DevBuf::new(rr * 4),
                    grp: DevBuf::new(3 * (m.n_exp + rr) * 4),
                    h16: DevBuf::new(rr * m.ff * 2),
                    o: DevBuf::new(rr * e * 4),
                    sh: DevBuf::new(gr * e * 4),
                    a: DevBuf::new(gr * e * 4),
                    x16: DevBuf::new(gr * e * 2),
                    mid16: DevBuf::new(gr * ff * 2),
                });
            }
        }
        let kvrow: Vec<i32> = rows.iter().map(|r| (self.page(r.slot, r.pos / attn::PAGE) * attn::PAGE + r.pos % attn::PAGE) as i32).collect();
        let btv: Vec<i32> = (0..pps).map(|p| self.page_pad(r0.slot, p) as i32).collect();
        let rpos: Vec<i32> = rows.iter().map(|r| self.rope_pos(r.slot, r.pos)).collect();
        let big = self.big.get_or_insert_with(|| Big {
            meta: DevBuf::new(gr * 4 * 3 + gr * 8 + 16 + pps * 4 + 64),
            cat: DevBuf::new(gr * 2 * e * 4),
            x16: DevBuf::new(gr * 2 * e * 2),
            x: DevBuf::new(gr * e * 4),
            a: DevBuf::new(gr * e * 4),
            qf: DevBuf::new(gr * 2 * hd * 4),
            kk: DevBuf::new(gr * kvd * 4),
            vv: DevBuf::new(gr * kvd * 4),
            q16: DevBuf::new(gr * hd * 2),
            o16: DevBuf::new(gr * hd * 2),
            mid16: DevBuf::new(gr * ff * 2),
        });
        // 入力（語・位置・KV の書き先・h の番地・列の範囲・KV の長さ・頁表）を 1 回で
        let mut host = vec![0u8; big.meta.len];
        let (o_tok, o_pos, o_slot, o_h) = (0, gr * 4, gr * 8, gr * 12);
        let o_qs = gr * 20;
        let o_kl = o_qs + 8;
        let o_bt = o_qs + 16;
        let put = |host: &mut [u8], at: usize, v: i32| host[at..at + 4].copy_from_slice(&v.to_le_bytes());
        for (i, r) in rows.iter().enumerate() {
            put(&mut host, o_tok + 4 * i, r.token as i32);
            put(&mut host, o_pos + 4 * i, rpos[i]);
            put(&mut host, o_slot + 4 * i, kvrow[i]);
            host[o_h + 8 * i..o_h + 8 * i + 8].copy_from_slice(&(r.h as u64).to_le_bytes());
        }
        put(&mut host, o_qs, 0);
        put(&mut host, o_qs + 4, n as i32);
        put(&mut host, o_kl, (r0.pos + n) as i32);
        for p in 0..pps {
            put(&mut host, o_bt + 4 * p, btv[p]);
        }
        let used = o_bt + 4 * pps;
        cuda::check(unsafe { ffi::cudaMemcpy(big.meta.ptr, host.as_ptr().cast(), used, 1) }, "mtp big meta H2D");
        let md = |at: usize| unsafe { (big.meta.ptr as *mut u8).add(at) };
        let s: Stream = null_mut();
        let w = &self.w;
        let ni = n as i32;
        let f16 = |x: &DevBuf, cnt: usize, y: &DevBuf| ck(unsafe { kffi::infer_m_f32_to_f16(x.f32p(), cnt as i64, y.ptr, s) }, "f32→f16");
        let gemm = |mats: &[&G3Mat], ys: &[*mut c_void], x: *const c_void, k: usize, mode: i32| -> Result<(), String> {
            let ts: Vec<i32> = mats.iter().map(|m| m.t).collect();
            let wp: Vec<*const c_void> = mats.iter().map(|m| m.ptr).collect();
            let ns: Vec<i32> = mats.iter().map(|m| m.n as i32).collect();
            ck(
                unsafe {
                    kffi::infer_g3_gemm_ex(mats.len() as i32, ts.as_ptr(), wp.as_ptr(), ns.as_ptr(), ys.as_ptr(), null(), k as i32, ni, x, null(), k as i32,
                                           mode, -1, s)
                },
                "mtp g3 gemm",
            )
        };
        let rms = |x: &DevBuf, wt: &DevBuf, y: &DevBuf| -> Result<(), String> {
            ck(unsafe { kffi::infer_mtp_rmsnorm(x.f32p(), e as i64, wt.f32p(), e as i32, ni, d.eps, y.f32p(), e as i64, s) }, "rmsnorm")
        };
        // e・hh → [e ; hh] → eh_proj
        ck(
            unsafe {
                kffi::infer_mtp_prep(
                    w.embd.qtype, w.embd.ptr, w.embd.row_bytes as i64, md(o_tok).cast(), md(o_h).cast(), w.enorm.f32p(), w.hnorm.f32p(), e as i32, ni,
                    d.eps, big.cat.f32p(), s,
                )
            },
            "mtp prep",
        )?;
        f16(&big.cat, n * 2 * e, &big.x16)?;
        gemm(&[&w.eh_proj], &[big.x.ptr], big.x16.ptr, 2 * e, 0)?;
        // Attention
        rms(&big.x, &w.attn_norm, &big.a)?;
        f16(&big.a, n * e, &big.x16)?;
        gemm(&[&w.attn_q, &w.attn_k, &w.attn_v], &[big.qf.ptr, big.kk.ptr, big.vv.ptr], big.x16.ptr, e, 0)?;
        let mut rp = RopeParams::new(&self.ash);
        rp.q = big.qf.f32p();
        rp.k = big.kk.f32p();
        rp.v = big.vv.f32p();
        rp.q_ld = (2 * hd) as i64;
        rp.k_ld = kvd as i64;
        rp.v_ld = kvd as i64;
        rp.q_norm = w.q_norm.f32p();
        rp.k_norm = w.k_norm.f32p();
        rp.pos = md(o_pos).cast();
        rp.slot = md(o_slot).cast();
        rp.q_out = big.q16.ptr;
        rp.kc = self.kc.ptr;
        rp.vc = self.vc.ptr;
        rp.n = ni;
        attn::rope(&rp, s)?;
        let mut ap = AttnParams::new(&self.ash);
        ap.q = big.q16.ptr;
        ap.gate = unsafe { big.qf.f32p().add(d.head_dim) };
        ap.gate_ld = (2 * hd) as i64;
        ap.gate_hs = (2 * d.head_dim) as i64;
        ap.kc = self.kc.ptr;
        ap.vc = self.vc.ptr;
        ap.block_table = md(o_bt).cast();
        ap.bt_ld = pps as i32;
        ap.q_start = md(o_qs).cast();
        ap.kv_len = md(o_kl).cast();
        ap.n_seq = 1;
        ap.n_tok = ni;
        ap.max_q_len = ni;
        ap.max_kv = (r0.pos + n) as i32;
        ap.kv_chunk = 0;
        ap.out = big.o16.ptr;
        ap.out_f16 = 1;
        attn::attn(&ap, s)?;
        gemm(&[&w.attn_output], &[big.x.ptr], big.o16.ptr, hd, 2)?;
        // FFN（MoE は専門家ごとの GEMM で。TACHYON_MOE_GEMM=0 なら 32 行ずつデコードの形で）
        if w.moe.is_some() {
            let xp = big.x.f32p();
            if self.moe_big.is_some() && n > MAX_ROWS {
                self.moe_ffn_gemm(xp, n, s)?;
            } else {
                self.moe_ffn(xp, n, s)?;
            }
        } else {
            rms(&big.x, &w.post_norm, &big.a)?;
            f16(&big.a, n * e, &big.x16)?;
            gemm(&[&w.ffn_gate, &w.ffn_up], &[big.mid16.ptr, big.mid16.ptr], big.x16.ptr, e, 3)?;
            gemm(&[&w.ffn_down], &[big.x.ptr], big.mid16.ptr, ff, 2)?;
        }
        self.n_forward += 1;
        self.n_rows += n;
        Ok(())
    }

    /// 頁表の埋め（持っていない頁は 0。Attention は kv_len より後ろを読まない）
    fn page_pad(&self, slot: usize, idx: usize) -> usize {
        match self.ext_pages.get(slot) {
            Some(v) if !v.is_empty() => v.get(idx).copied().unwrap_or(0) as usize,
            _ => slot * self.pages_per_slot + idx,
        }
    }

    /// 升 slot の頁 idx の通し番号（書く所。set_pages で渡されていればその頁表、無ければ決め打ちの並び）
    fn page(&self, slot: usize, idx: usize) -> usize {
        match self.ext_pages.get(slot) {
            Some(v) if !v.is_empty() => *v.get(idx).unwrap_or_else(|| panic!("MTP の升 {slot} の頁 {idx} が無い（本体の頁を先に取ること）")) as usize,
            _ => slot * self.pages_per_slot + idx,
        }
    }

    /// 頁 page の MTP の KV の置き場（K・V。頭の控えを RAM の段に写す・戻す。hostcache）
    pub fn kv_page_spans(&self, page: u32) -> [(*mut u8, usize); 2] {
        let pb = self.ash.page_bytes(false);
        let off = page as usize * pb;
        assert!(off + pb <= self.kc.len);
        unsafe { [(self.kc.ptr.cast::<u8>().add(off), pb), (self.vc.ptr.cast::<u8>().add(off), pb)] }
    }

    /// 列 seq の頁表（本体の Seq と同じ頁の番号。M3）。MTP の KV の位置 p は頁 pages[p/64] に書く
    pub fn set_pages(&mut self, seq: SeqId, pages: &[u32]) -> Result<(), String> {
        let s = self.slot(seq)?;
        if self.opts.kv_pages == 0 || pages.iter().any(|&p| p as usize >= self.opts.kv_pages) {
            return Err("set_pages: kv_pages の置き場で、頁の番号はその中".into());
        }
        if self.ext_pages.len() <= s {
            self.ext_pages.resize(s + 1, vec![]);
        }
        self.ext_pages[s].clear();
        self.ext_pages[s].extend_from_slice(pages);
        Ok(())
    }

    /// 頭の使い回し: 列 seq の MTP の KV は 0..len が書いてある（前の依頼の物）として、続きを len から。h は位置 len-1 の本体の h_nextn
    pub fn set_prefix(&mut self, seq: SeqId, len: usize, h: *const f32) -> Result<(), String> {
        let s = self.slot(seq)?;
        if len == 0 || len > self.opts.max_ctx {
            return Err("set_prefix: 長さが変".into());
        }
        self.committed[s] = len;
        self.save_pend(&[(s, h, len - 1)])
    }

    /// 列の升（無ければ取る）
    fn slot(&mut self, seq: SeqId) -> Result<usize, String> {
        if let Some(&s) = self.slot_of.get(&seq) {
            return Ok(s);
        }
        let s = self.free_slots.pop().ok_or("MTP の升が足りない")?;
        self.slot_of.insert(seq, s);
        self.pend_pos[s] = None;
        self.committed[s] = 0;
        self.media[s].clear();
        Ok(s)
    }

    /// 升の前の h を写す（本体の行の番地から）。copy は (升, 番地, その h の位置)
    fn save_pend(&mut self, copy: &[(usize, *const f32, usize)]) -> Result<(), String> {
        if copy.is_empty() {
            return Ok(());
        }
        let dst: Vec<u64> = copy.iter().map(|&(s, _, _)| self.pend_ptr(s) as u64).collect();
        let src: Vec<u64> = copy.iter().map(|&(_, p, _)| p as u64).collect();
        let (cd, cs) = (self.meta.cdst, self.meta.csrc);
        // meta の残りを壊さないよう、写しの表だけ送る
        self.meta.put_u64(cd, &dst);
        self.meta.put_u64(cs, &src);
        let n = copy.len();
        unsafe {
            cuda::check(ffi::cudaMemcpy(self.meta.d(cd).cast(), self.meta.host[cd..].as_ptr().cast(), n * 8, 1), "pend H2D");
            cuda::check(ffi::cudaMemcpy(self.meta.d(cs).cast(), self.meta.host[cs..].as_ptr().cast(), n * 8, 1), "pend H2D");
        }
        ck(unsafe { kffi::infer_mtp_copy_rows(self.meta.d(cd).cast(), self.meta.d(cs).cast(), self.dims.n_embd as i32, n as i32, null_mut()) }, "pend copy")?;
        for &(s, _, p) in copy {
            self.pend_pos[s] = Some(p);
        }
        Ok(())
    }

    fn pick(&self, t: &TopRow, ask: &DraftAsk, j: usize) -> (u32, Option<spec::Dist>) {
        // 結び付け（ask.gumbel）なら t.ids[0] は GPU の Gumbel-max で選び直した語（enq の gumbel）
        match ask.sample {
            None => (t.ids[0], None),
            Some((smp, seed, coupled)) => {
                let k = t.ids.iter().position(|&x| x == u32::MAX).unwrap_or(t.ids.len());
                let q = spec::Dist::from_candidates(&t.ids[..k], &t.vals[..k], &smp);
                let tag = if coupled { spec::TAG_SAMPLE } else { spec::TAG_DRAFT };
                (spec::sample(&q, seed, (ask.pos + j + 1) as u64, tag), Some(q))
            }
        }
    }

    /// HCAP の影: 最後の forward の logits の row 行目（want の順）から 2・3 番手（下書きと同じ取り方: 結び付けなら Gumbel をのせた値）
    fn alts_of(&self, row: usize, ask: &DraftAsk, j: usize) -> [u32; 2] {
        let lg = self.logits_row(row);
        let n = match &self.draft_vocab {
            Some((m, _)) if self.trace.is_none() => m.n,
            _ => self.w.output.n,
        };
        let ids = self.draft_vocab.as_ref().filter(|_| self.trace.is_none()).map(|(_, ids)| ids);
        let id_of = |i: usize| ids.map_or(i as u32, |v| v[i]);
        let g = gparam(ask, j);
        // 3 番目に大きい logit（G は [−3.62, 37.43]: 53 ビットの一様から。上位 3 つに入れない値は計算しない＝結果は同じ）
        let mut l3 = [f32::NEG_INFINITY; 3];
        for &l in &lg[..n] {
            if l > l3[2] {
                l3[2] = l;
                if l3[2] > l3[1] {
                    l3.swap(1, 2);
                    if l3[1] > l3[0] {
                        l3.swap(0, 1);
                    }
                }
            }
        }
        let mx = l3[2];
        let mut best: [(f64, u32); 3] = [(f64::NEG_INFINITY, u32::MAX); 3];
        for i in 0..n {
            let l = lg[i];
            let v = if g.on != 0 {
                if ((l - mx) as f64) * (g.invt as f64) < -41.5 {
                    continue;
                }
                ((l as f64) * (g.invt as f64) + spec::gumbel(g.seed, g.pos, spec::TAG_SAMPLE, id_of(i))) as f32 as f64
            } else {
                l as f64
            };
            if v > best[2].0 {
                best[2] = (v, id_of(i));
                if best[2].0 > best[1].0 {
                    best.swap(1, 2);
                    if best[1].0 > best[0].0 {
                        best.swap(0, 1);
                    }
                }
            }
        }
        [best[1].1, best[2].1]
    }

    /// 下書きの行を記録する（tops と want は forward に渡した順。logits の行も want の順）
    fn record(&mut self, tops: &[TopRow], want: &[usize]) {
        if let Some(mut tr) = self.trace.take() {
            let e = self.dims.n_embd;
            let n = want.iter().copied().max().map_or(0, |m| m + 1);
            let ho = self.hout_rows(n);
            for (i, t) in tops.iter().enumerate() {
                tr.rows.push(t.clone());
                tr.hout.push(ho[want[i] * e..(want[i] + 1) * e].to_vec());
                if tr.logits.len() < tr.full {
                    tr.logits.push(self.logits_row(i));
                }
            }
            self.trace = Some(tr);
        }
    }
}

impl Drafter<DevRow> for MtpHead {
    fn run(&mut self, obs: &[Observe<DevRow>], asks: &[DraftAsk]) -> Result<Vec<DraftOut>, String> {
        // 1 段目の行: 追いつき（本体の h）と、下書きの頭（id_last と本体の最後の h）
        let mut rows: Vec<Row> = vec![];
        let mut newp: Vec<(usize, *const f32, usize)> = vec![];
        for o in obs {
            if o.tokens.is_empty() || o.h.len() != o.tokens.len() {
                return Err("追いつき: 語と h の数が合わない".into());
            }
            let s = self.slot(o.seq)?;
            if o.pos0 > self.committed[s] {
                return Err(format!("追いつき: 列 {} の KV に穴（書いた {}・pos0 {}）", o.seq, self.committed[s], o.pos0));
            }
            for (j, &tk) in o.tokens.iter().enumerate() {
                let p = o.pos0 + j;
                let h = if j > 0 {
                    o.h[j - 1].0
                } else if p == 0 {
                    null()
                } else if self.pend_pos[s] == Some(p - 1) {
                    self.pend_ptr(s) as *const f32
                } else {
                    return Err(format!("追いつき: 列 {} の位置 {p} の前の h が無い（{:?}）", o.seq, self.pend_pos[s]));
                };
                rows.push(Row { slot: s, pos: p, token: tk, h });
            }
            self.committed[s] = o.pos0 + o.tokens.len();
            newp.push((s, o.h.last().unwrap().0, o.pos0 + o.tokens.len() - 1));
        }
        let mut heads: Vec<Row> = vec![];
        for a in asks {
            let s = self.slot(a.seq)?;
            if a.pos != self.committed[s] || a.k == 0 {
                return Err(format!("下書き: 列 {} の位置 {} ≠ 書いた長さ {}（k {}）", a.seq, a.pos, self.committed[s], a.k));
            }
            let h = match newp.iter().rev().find(|x| x.0 == s) {
                Some(&(_, p, at)) if at + 1 == a.pos => p,
                Some(_) => return Err("下書き: 追いつきの最後と位置が合わない".into()),
                None if a.pos == 0 => null(),
                None if self.pend_pos[s] == Some(a.pos - 1) => self.pend_ptr(s) as *const f32,
                None => return Err(format!("下書き: 列 {} の前の h が無い", a.seq)),
            };
            heads.push(Row { slot: s, pos: a.pos, token: a.id_last, h });
        }
        if heads.len() > MAX_ROWS {
            return Err(format!("下書き: 1 回に {} 列は多い（{MAX_ROWS} まで）", heads.len()));
        }
        // 下書きの頭の行は、同じ升の追いつきの行のすぐ後ろに置く（升ごとに続いた位置で並べる決まり）
        let mut all: Vec<Row> = vec![];
        let mut head_at = vec![usize::MAX; heads.len()];
        let mut used = vec![false; heads.len()];
        let mut i = 0;
        while i < rows.len() {
            let mut j = i + 1;
            while j < rows.len() && rows[j].slot == rows[i].slot {
                j += 1;
            }
            all.extend_from_slice(&rows[i..j]);
            for (hi, h) in heads.iter().enumerate() {
                if !used[hi] && h.slot == rows[i].slot {
                    head_at[hi] = all.len();
                    all.push(*h);
                    used[hi] = true;
                }
            }
            i = j;
        }
        for (hi, h) in heads.iter().enumerate() {
            if !used[hi] {
                head_at[hi] = all.len();
                all.push(*h);
            }
        }
        let kout = if asks.iter().any(|a| a.sample.is_some()) || self.trace.is_some() || self.log.is_some() { self.opts.topk } else { 1 };
        // 32 行に収まれば 1 回で（下書きの頭の行だけ logits）。収まらなければ、追いつきを 32 行ずつ先に流し、頭の行を別に 1 回で
        let mut tops: Vec<TopRow> = vec![];
        let mut head_row: Vec<usize> = vec![0; heads.len()];
        // 頭の 3 段を手元に戻らず続けて流せるか（argmax の下書き・確からしさで止めない段だけ。記録の時は使わない）
        let kmax = asks.iter().map(|a| a.k).max().unwrap_or(0);
        let thr0 = if asks.iter().all(|a| a.sample.is_none()) { self.draft_min_p } else { 0.0 };
        let use_chain = self.chain && !self.hcap_shadow && kout == 1 && self.trace.is_none() && thr0 <= 0.0 && kmax >= 2 && !heads.is_empty();
        let mut pre: Vec<Vec<TopRow>> = vec![];
        if all.len() <= MAX_ROWS && use_chain {
            pre = self.chain_run(&all, &head_at, &heads, asks, kmax.min(CHAIN))?;
            tops = std::mem::take(&mut pre[0]);
            head_row = head_at.clone();
        } else if all.len() <= MAX_ROWS {
            let g0: Vec<GParam> = (0..heads.len()).map(|hi| gparam(&asks[hi], 0)).collect();
            tops = self.forward_g(&all, &head_at, kout, Some(&g0))?;
            head_row = head_at.clone();
            self.record(&tops, &head_at);
        } else {
            let body: Vec<Row> = all.iter().enumerate().filter(|(i, _)| !head_at.contains(i)).map(|(_, r)| *r).collect();
            // 升ごとに続いた位置の塊に分け、長い塊は GEMM の道で GEMM_ROWS 行ずつ（残りの短い塊は g3 の GEMV で 32 行ずつ）
            let mut i = 0;
            while i < body.len() {
                let mut j = i + 1;
                while j < body.len() && body[j].slot == body[i].slot && body[j].pos == body[j - 1].pos + 1 {
                    j += 1;
                }
                for c in body[i..j].chunks(GEMM_ROWS) {
                    if self.gemm_catchup && c.len() > MAX_ROWS {
                        self.forward_gemm(c)?;
                    } else {
                        for c2 in c.chunks(MAX_ROWS) {
                            self.forward(c2, &[], 1)?;
                        }
                    }
                }
                i = j;
            }
            if !heads.is_empty() && use_chain {
                let want: Vec<usize> = (0..heads.len()).collect();
                pre = self.chain_run(&heads, &want, &heads, asks, kmax.min(CHAIN))?;
                tops = std::mem::take(&mut pre[0]);
                head_row = want;
            } else if !heads.is_empty() {
                let want: Vec<usize> = (0..heads.len()).collect();
                let g0: Vec<GParam> = (0..heads.len()).map(|hi| gparam(&asks[hi], 0)).collect();
                tops = self.forward_g(&heads, &want, kout, Some(&g0))?;
                self.record(&tops, &want);
                head_row = want;
            }
        }
        // 本体の最後の h を升に覚える（1 段目が古い pend を読み終えた後）
        self.save_pend(&newp)?;
        // 2 段目から: 列ごとに 1 行（前の段の h' と下書きの語）
        let mut outs: Vec<DraftOut> = asks.iter().map(|_| DraftOut::default()).collect();
        let mut qs: Vec<Vec<spec::Dist>> = vec![vec![]; asks.len()];
        // tops は want の順（= head の順）
        // 下書きの確からしさ（MTP の 1 位の確率の積）。draft_min_p > 0 なら、これを下回る語から後ろは出さない（argmax の下書きだけ）
        let thr = if asks.iter().all(|a| a.sample.is_none()) { self.draft_min_p } else { 0.0 };
        let conf = |t: &TopRow| -> f32 { (t.vals[0] - t.lse).exp() };
        let mut cum = vec![1f32; asks.len()];
        let mut alive = vec![true; asks.len()];
        let top2 = |t: &TopRow| -> (u32, f32, u32, f32) {
            let p = |i: usize| t.vals.get(i).map_or(0.0, |v| (v - t.lse).exp());
            (t.ids[0], p(0), t.ids.get(1).copied().unwrap_or(u32::MAX), p(1))
        };
        let log_base = self.log.as_ref().map_or(0, |l| l.len());
        if let Some(l) = self.log.as_mut() {
            for (hi, a) in asks.iter().enumerate() {
                l.push(DraftLog { pos: a.pos, last: a.id_last, tops: vec![top2(&tops[hi])] });
            }
        }
        for (hi, a) in asks.iter().enumerate() {
            cum[hi] = conf(&tops[hi]);
            if thr > 0.0 && cum[hi] < thr {
                alive[hi] = false;
                continue;
            }
            let (tk, q) = self.pick(&tops[hi], a, 0);
            outs[hi].tokens.push(tk);
            if self.hcap_shadow {
                let al = self.alts_of(hi, a, 0);
                outs[hi].alts.push(al);
                outs[hi].conf.push(conf(&tops[hi]));
            }
            if let Some(q) = q {
                qs[hi].push(q);
            }
        }
        for j in 1..kmax {
            // この段を計算するかは、ここまでの確からしさで先に決める（計算してから捨てると MTP の 1 段ぶん無駄になる）
            let th_j = if j >= 3 { thr.max(self.draft_min_p_deep) } else { thr };
            let live: Vec<usize> = (0..asks.len()).filter(|&hi| asks[hi].k > j && alive[hi] && !(th_j > 0.0 && cum[hi] < th_j)).collect();
            if live.is_empty() {
                break;
            }
            let rows2: Vec<Row> = live
                .iter()
                .map(|&hi| Row { slot: heads[hi].slot, pos: asks[hi].pos + j, token: *outs[hi].tokens.last().unwrap(), h: self.hout_ptr(head_row[hi]) })
                .collect();
            let want: Vec<usize> = (0..rows2.len()).collect();
            let t2 = if j < pre.len() {
                // 続けて流した段（行の並びは同じ決まりで作ってある。数が違えば決まりが崩れている）
                if pre[j].len() != rows2.len() {
                    return Err(format!("chain: {j} 段目の行が {} ≠ {}", pre[j].len(), rows2.len()));
                }
                std::mem::take(&mut pre[j])
            } else {
                let gj: Vec<GParam> = live.iter().map(|&hi| gparam(&asks[hi], j)).collect();
                self.forward_g(&rows2, &want, kout, Some(&gj))?
            };
            self.record(&t2, &want);
            if let Some(l) = self.log.as_mut() {
                for (r, &hi) in live.iter().enumerate() {
                    l[log_base + hi].tops.push(top2(&t2[r]));
                }
            }
            for (r, &hi) in live.iter().enumerate() {
                cum[hi] *= conf(&t2[r]);
                // 4 語目から（確かめが 5 語以上）は GEMV の組が 8 本に変わって 1 歩が 4ms ほど重くなるので、別の線で
                let th = if j >= 3 { thr.max(self.draft_min_p_deep) } else { thr };
                if th > 0.0 && cum[hi] < th {
                    alive[hi] = false;
                    continue;
                }
                let (tk, q) = self.pick(&t2[r], &asks[hi], j);
                outs[hi].tokens.push(tk);
                if self.hcap_shadow {
                    let al = self.alts_of(r, &asks[hi], j);
                    outs[hi].alts.push(al);
                    outs[hi].conf.push(conf(&t2[r]));
                }
                if let Some(q) = q {
                    qs[hi].push(q);
                }
                head_row[hi] = r;
            }
        }
        for (hi, a) in asks.iter().enumerate() {
            if a.sample.is_some() {
                outs[hi].q = Some(std::mem::take(&mut qs[hi]));
            }
        }
        Ok(outs)
    }

    fn free(&mut self, seq: SeqId) {
        if let Some(s) = self.slot_of.remove(&seq) {
            self.media[s].clear();
            self.pend_pos[s] = None;
            self.committed[s] = 0;
            self.free_slots.push(s);
        }
    }
}
