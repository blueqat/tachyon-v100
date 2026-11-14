//! 普通の Attention（qwen35 の 4 層に 1 つ）を呼ぶ口（kernels/attn.cu）。
//!
//!   attn_q → q_full [N][H·2D]（頭ごとに q | gate）、attn_k → k [N][HKV·D]、attn_v → v [N][HKV·D]
//!   rope(): q・k の RMSNorm ＋ IMRoPE、q は fp16 [N][H][D] へ、k・v はページ化した KV の slot へ
//!   attn(): softmax(q k^T / 16) v × sigmoid(gate) → out [N][H·D]（→ attn_output）
//!
//! KV（層ごと）は kc, vc [n_pages][HKV][64][D] fp16、または int8 ＋ ks, vs [n_pages][HKV][64][D/32] f32（32 要素ごとの係数）。
//! 列は block_table（論理ページ → 物理ページ）と kv_len（書いた後の長さ）で表す。新しいトークンは列の最後の q_len 個。
use crate::cuda::ffi::Stream;
use std::ffi::c_void;
use std::ptr::{null, null_mut};

/// 頭の幅・ページのトークン数（カーネルの決め打ち）
pub const D: usize = 256;
pub const PAGE: usize = 64;

/// kernels/attn.cu の AttnRopeParams と同じ並び
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RopeParams {
    /// q_full [n][q_ld]。null なら K/V を書くだけ
    pub q: *const f32,
    pub k: *const f32,
    pub v: *const f32,
    pub q_ld: i64,
    pub k_ld: i64,
    pub v_ld: i64,
    pub q_norm: *const f32,
    pub k_norm: *const f32,
    /// pos_mode 1: [n]（文字だけ）、4: [n][4]（t, h, w, e）
    pub pos: *const i32,
    /// [n]、物理ページ × 64 + ページ内の位置（-1 は書かない）
    pub slot: *const i32,
    /// fp16 [n][H][D]
    pub q_out: *mut c_void,
    pub kc: *mut c_void,
    pub vc: *mut c_void,
    pub ks: *mut f32,
    pub vs: *mut f32,
    pub n: i32,
    pub h: i32,
    pub hkv: i32,
    pub n_rot: i32,
    pub sections: [i32; 4],
    pub pos_mode: i32,
    pub kv8: i32,
    pub eps: f32,
    pub theta: f32,
}

/// kernels/attn.cu の AttnParams と同じ並び
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AttnParams {
    /// fp16 [n_tok][H][D]（rope の q_out）
    pub q: *const c_void,
    /// gate の先頭（q_full + D）。null なら掛けない
    pub gate: *const f32,
    pub gate_ld: i64,
    pub gate_hs: i64,
    pub kc: *const c_void,
    pub vc: *const c_void,
    pub ks: *const f32,
    pub vs: *const f32,
    pub block_table: *const i32,
    pub q_start: *const i32,
    pub kv_len: *const i32,
    pub out: *mut c_void,
    pub ws: *mut f32,
    pub ws_bytes: i64,
    pub bt_ld: i32,
    pub n_seq: i32,
    pub n_tok: i32,
    pub max_q_len: i32,
    pub h: i32,
    pub hkv: i32,
    pub kv8: i32,
    pub out_f16: i32,
    /// 0 = 分けない、> 0 = この幅（トークン）ごとに split、-1 = 自動
    pub kv_chunk: i32,
    pub max_kv: i32,
    /// 0 なら 1/sqrt(D)
    pub scale: f32,
    /// 列ごとの kv_len の CPU の写し [n_seq]（任意。null なら全部 max_kv とみなす）。あれば自動の split の幅を列の長さで決める
    /// （短い列を細かく分けすぎず、長い列を分け足りなくしない）。幅の選び方にだけ使う（split の数と作業場は max_kv で数える）。
    /// GPU の kv_len と同じ値にし、ws_bytes と attn に同じものを渡す（0 以下か max_kv より大きい値があれば attn は「形が合わない」）
    pub kv_len_host: *const i32,
}

mod ffi {
    use super::*;
    extern "C" {
        pub fn infer_attn_rope(p: *const RopeParams, s: Stream) -> i32;
        pub fn infer_attn_ws_bytes(p: *const AttnParams) -> i64;
        pub fn infer_attn_chunk(p: *const AttnParams) -> i32;
        pub fn infer_attn(p: *const AttnParams, s: Stream) -> i32;
        pub fn infer_attn_fill(p: *mut c_void, n: i64, kind: i32, seed: u32, scale: f32, s: Stream) -> i32;
    }
}

/// 形（GGUF の qwen35.*）。Qwen3.8-27B は H 24・HKV 4・n_rot 64・sections [11,11,10,0]・theta 1e7・eps 1e-6
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub h: usize,
    pub hkv: usize,
    pub n_rot: usize,
    pub sections: [i32; 4],
    pub theta: f32,
    pub eps: f32,
}

impl Shape {
    pub const QWEN38_27B: Shape = Shape { h: 24, hkv: 4, n_rot: 64, sections: [11, 11, 10, 0], theta: 1e7, eps: 1e-6 };
    /// 1 ページ・1 層の K（または V）のバイト数
    pub fn page_bytes(&self, kv8: bool) -> usize {
        self.hkv * PAGE * D * if kv8 { 1 } else { 2 }
    }
    /// int8 の係数の 1 ページ・1 層の個数（f32。トークン×頭×32 要素ごと）
    pub fn page_scales(&self) -> usize {
        self.hkv * PAGE * (D / 32)
    }
}

impl RopeParams {
    pub fn new(sh: &Shape) -> Self {
        RopeParams {
            q: null(),
            k: null(),
            v: null(),
            q_ld: (sh.h * 2 * D) as i64,
            k_ld: (sh.hkv * D) as i64,
            v_ld: (sh.hkv * D) as i64,
            q_norm: null(),
            k_norm: null(),
            pos: null(),
            slot: null(),
            q_out: null_mut(),
            kc: null_mut(),
            vc: null_mut(),
            ks: null_mut(),
            vs: null_mut(),
            n: 0,
            h: sh.h as i32,
            hkv: sh.hkv as i32,
            n_rot: sh.n_rot as i32,
            sections: sh.sections,
            pos_mode: 1,
            kv8: 0,
            eps: sh.eps,
            theta: sh.theta,
        }
    }
}

impl AttnParams {
    pub fn new(sh: &Shape) -> Self {
        AttnParams {
            q: null(),
            gate: null(),
            gate_ld: (sh.h * 2 * D) as i64,
            gate_hs: (2 * D) as i64,
            kc: null(),
            vc: null(),
            ks: null(),
            vs: null(),
            block_table: null(),
            q_start: null(),
            kv_len: null(),
            out: null_mut(),
            ws: null_mut(),
            ws_bytes: 0,
            bt_ld: 0,
            n_seq: 0,
            n_tok: 0,
            max_q_len: 0,
            h: sh.h as i32,
            hkv: sh.hkv as i32,
            kv8: 0,
            out_f16: 0,
            kv_chunk: -1,
            max_kv: 0,
            scale: 0.0,
            kv_len_host: null(),
        }
    }
}

fn err(code: i32, what: &str) -> Result<(), String> {
    match code {
        0 => Ok(()),
        -2 => Err(format!("{what}: 形が合わない")),
        -3 => Err(format!("{what}: split の作業場が足りない")),
        c => Err(format!("{what}: cudaError {c}")),
    }
}

/// q・k の正規化と RoPE、q の書き出しと K/V の書き込み
// s（Stream）は CUDA の不透明な handle をそのまま CUDA の API に渡すだけで、Rust の側では参照外ししない（不正な値は CUDA が誤りで返す）。
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn rope(p: &RopeParams, s: Stream) -> Result<(), String> {
    err(unsafe { ffi::infer_attn_rope(p, s) }, "attn_rope")
}

/// split の作業場のバイト数（0 なら要らない）。max_kv・n_tok・max_q_len・n_seq・kv_chunk（と kv_len_host で選ぶ split の幅）で決まる
pub fn ws_bytes(p: &AttnParams) -> usize {
    unsafe { ffi::infer_attn_ws_bytes(p) as usize }
}

/// 選ばれる split の幅（トークン。0 = 分けない）
pub fn chunk(p: &AttnParams) -> usize {
    unsafe { ffi::infer_attn_chunk(p) as usize }
}

/// Attention（＋ split なら合わせ）
// s（Stream）は CUDA の不透明な handle をそのまま CUDA の API に渡すだけで、Rust の側では参照外ししない（不正な値は CUDA が誤りで返す）。
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn attn(p: &AttnParams, s: Stream) -> Result<(), String> {
    err(unsafe { ffi::infer_attn(p, s) }, "attn")
}

/// 試し用のでたらめ（kind 0 = fp16、1 = int8、2 = f32）
///
/// # Safety
/// `p` は GPU の上の、`n` 要素（kind の型で）書いてよい置き場を指すこと（GPU のカーネルがそこへ書く）。
pub unsafe fn fill(p: *mut c_void, n: usize, kind: i32, seed: u32, scale: f32) -> Result<(), String> {
    err(unsafe { ffi::infer_attn_fill(p, n as i64, kind, seed, scale, null_mut()) }, "attn_fill")
}
