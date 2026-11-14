//! Gated DeltaNet（線形 Attention）と因果 conv1d を呼ぶ口（kernels/gdn.cu）。
//! llama.cpp の qwen35（Qwen3.8-27B）の線形層のうち、射影（attn_qkv・attn_gate・ssm_alpha・ssm_beta・ssm_out の GEMV）の間に入る所。
//!
//!   qkv ─conv（＋silu・q/k の l2 正規化）→ c ─decode か prefill（＋rms_norm × w × silu(z)）→ y ─ssm_out→
//!
//! 並びは kernels/gdn.cu の頭の説明と同じ。ポインタはすべて GPU 上の f32（slots は i32）。
use crate::cuda::{check, ffi::Stream};
use std::ffi::c_void;
use std::ptr::null;

/// 頭の幅（key・value とも）
pub const D: usize = 128;

mod ffi {
    use super::Stream;
    extern "C" {
        pub fn infer_gdn_conv(
            x: *const f32, x_ld: i64, conv_s: *mut f32, slots: *const i32, slots_out: *const i32, n_valid: *const i32,
            w: *const f32, y: *mut f32, b: i32, t: i32, c: i32, n_norm: i32, eps: f32, s: Stream,
        ) -> i32;
        pub fn infer_gdn_decode(
            c: *const f32, c_ld: i64, z: *const f32, z_ld: i64, alpha: *const f32, beta: *const f32, ab_ld: i64,
            ssm_a: *const f32, dt_bias: *const f32, norm_w: *const f32, state: *mut f32, slots: *const i32,
            slots_out: *const i32, n_valid: *const i32, y: *mut f32, y_ld: i64, b: i32, t: i32, hk: i32, hv: i32, eps: f32,
            s: Stream,
        ) -> i32;
        pub fn infer_gdn_prefill_ws_bytes(b: i32, t: i32, hv: i32) -> i64;
        pub fn infer_gdn_prefill(
            c: *const f32, c_ld: i64, z: *const f32, z_ld: i64, alpha: *const f32, beta: *const f32, ab_ld: i64,
            ssm_a: *const f32, dt_bias: *const f32, norm_w: *const f32, state: *mut f32, slots: *const i32,
            y: *mut f32, y_ld: i64, ws: *mut f32, b: i32, t: i32, hk: i32, hv: i32, eps: f32, s: Stream,
        ) -> i32;
    }
}

/// 形（GGUF の qwen35.ssm.*: hk = group_count、hv = time_step_rank、D = state_size）
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub hk: usize,
    pub hv: usize,
    pub eps: f32,
}

impl Shape {
    /// Qwen3.8-27B: hk 16・hv 48・eps 1e-6（attention.layer_norm_rms_epsilon）
    pub const Q38_27B: Shape = Shape { hk: 16, hv: 48, eps: 1e-6 };
    /// conv の通り数 = q・k・v（attn_qkv の出力の幅）
    pub fn conv_ch(&self) -> usize {
        (2 * self.hk + self.hv) * D
    }
    /// 出口の幅（attn_gate・ssm_out の入力）
    pub fn out_dim(&self) -> usize {
        self.hv * D
    }
    /// 1 本の列の状態の大きさ（f32 の個数）: 線形状態と conv の状態
    pub fn state_len(&self) -> usize {
        self.hv * D * D
    }
    pub fn conv_state_len(&self) -> usize {
        self.conv_ch() * 3
    }
}

/// 層ごとの重み（GPU 上。ssm_a・ssm_dt.bias [hv]、ssm_norm [D]、ssm_conv1d [conv_ch][4]）
#[derive(Clone, Copy)]
pub struct Layer {
    pub a: *const f32,
    pub dt: *const f32,
    pub norm: *const f32,
    pub conv: *const f32,
}

/// 1 回分の入出力。行の幅（*_ld）は要素の数。b 本 × t トークンで、行 (b, t) は b*t + t 番目
pub struct Io {
    /// attn_qkv の出力 [b][t][x_ld]（先頭 conv_ch を使う）
    pub qkv: *const f32,
    pub qkv_ld: usize,
    /// conv の後（q・k は正規化済み）[b][t][conv_ch]。decode・prefill はここを読む
    pub c: *mut f32,
    pub z: *const f32,
    pub z_ld: usize,
    pub alpha: *const f32,
    pub beta: *const f32,
    pub ab_ld: usize,
    pub y: *mut f32,
    pub y_ld: usize,
    /// 状態の置き場: 線形 [slot][hv][D][D]、conv [slot][conv_ch][3]。slots が null なら slot = b
    pub state: *mut f32,
    pub conv_state: *mut f32,
    pub slots: *const i32,
    /// conv・decode の新しい状態の置き場（null なら slots と同じ）。下書きの確かめで元の状態を残すときに。
    /// ⚠️ ほかの列の slots と重ねない
    pub slots_out: *const i32,
    /// conv・decode で列 b は前の n_valid[b] トークンだけ進める（null なら t 全部）。受け入れた分だけ確定するときに
    pub n_valid: *const i32,
}

fn conv_to(sh: &Shape, l: &Layer, io: &Io, out: *mut f32, b: usize, t: usize, s: Stream) {
    let c = sh.conv_ch();
    let r = unsafe {
        ffi::infer_gdn_conv(io.qkv, io.qkv_ld as i64, io.conv_state, io.slots, io.slots_out, io.n_valid, l.conv, out, b as i32,
                            t as i32, c as i32, (2 * sh.hk * D) as i32, sh.eps, s)
    };
    check(r, "gdn conv");
}

/// conv（＋silu・q/k の l2 正規化）→ io.c。conv の状態は書き戻す（slots_out・n_valid に従う）
pub fn conv(sh: &Shape, l: &Layer, io: &Io, b: usize, t: usize, s: Stream) {
    conv_to(sh, l, io, io.c, b, t, s);
}

/// conv の状態だけを進める（io.c は書かない）。受け入れた分だけ確定するとき（io.n_valid）に
pub fn conv_state_only(sh: &Shape, l: &Layer, io: &Io, b: usize, t: usize, s: Stream) {
    conv_to(sh, l, io, std::ptr::null_mut(), b, t, s);
}

/// 1 トークンずつ t 回（状態はレジスタのまま）。decode は t = 1、下書きの確かめは t = 1 + 下書きの数。conv の後に呼ぶ。
/// io.y が null なら出口を書かず状態だけ進める
// s（Stream）は CUDA の不透明な handle をそのまま CUDA の API に渡すだけで、Rust の側では参照外ししない（不正な値は CUDA が誤りで返す）。
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn decode(sh: &Shape, l: &Layer, io: &Io, b: usize, t: usize, s: Stream) {
    let r = unsafe {
        ffi::infer_gdn_decode(io.c, sh.conv_ch() as i64, io.z, io.z_ld as i64, io.alpha, io.beta, io.ab_ld as i64, l.a, l.dt,
                              l.norm, io.state, io.slots, io.slots_out, io.n_valid, io.y, io.y_ld as i64, b as i32, t as i32,
                              sh.hk as i32, sh.hv as i32, sh.eps, s)
    };
    check(r, "gdn decode");
}

/// prefill の作業場の大きさ（バイト）
pub fn prefill_ws_bytes(sh: &Shape, b: usize, t: usize) -> usize {
    unsafe { ffi::infer_gdn_prefill_ws_bytes(b as i32, t as i32, sh.hv as i32) as usize }
}

/// 塊ごと（64）の prefill。conv の後に呼ぶ。ws は prefill_ws_bytes 以上。状態はその場で進める（slots_out・n_valid は使えない）
// s（Stream）は CUDA の不透明な handle をそのまま CUDA の API に渡すだけで、Rust の側では参照外ししない（不正な値は CUDA が誤りで返す）。
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn prefill(sh: &Shape, l: &Layer, io: &Io, ws: *mut c_void, b: usize, t: usize, s: Stream) {
    assert!(io.slots_out.is_null() && io.n_valid.is_null(), "prefill は slots_out・n_valid を使えない");
    let r = unsafe {
        ffi::infer_gdn_prefill(io.c, sh.conv_ch() as i64, io.z, io.z_ld as i64, io.alpha, io.beta, io.ab_ld as i64, l.a, l.dt,
                               l.norm, io.state, io.slots, io.y, io.y_ld as i64, ws.cast(), b as i32, t as i32, sh.hk as i32,
                               sh.hv as i32, sh.eps, s)
    };
    check(r, "gdn prefill");
}

impl Io {
    /// 全部 null（使う所だけ埋める）
    pub fn empty() -> Self {
        Io {
            qkv: null(), qkv_ld: 0, c: std::ptr::null_mut(), z: null(), z_ld: 0, alpha: null(), beta: null(), ab_ld: 0,
            y: std::ptr::null_mut(), y_ld: 0, state: std::ptr::null_mut(), conv_state: std::ptr::null_mut(), slots: null(),
            slots_out: null(), n_valid: null(),
        }
    }
}
