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
    /// YaRN（Yarn::OFF なら 1, 0, 1, 0, 0 で、カーネルは前と同じ式を通る）
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub mscale: f32,
    pub corr0: f32,
    pub corr1: f32,
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
    pub yarn: Yarn,
}

/// RoPE の YaRN（llama.cpp の ggml rope_yarn と HF の _compute_yarn_parameters と同じ式）。
///   角度 θ_i = pos·base^(−2i/n_rot)。補間 θ_i·freq_scale と外挿 θ_i を、組の番号 i の傾き
///   ramp(i) = 1 − clamp((i − corr0)/max(0.001, corr1 − corr0), 0, 1) で混ぜる: θ = θ_interp·(1 − ramp·ext) + θ_extrap·ramp·ext。
///   cos・sin に attn_factor·(1 + 0.1·ln(1/freq_scale)) を掛ける（q と k の両方に掛かるので logits は その 2 乗倍）。
///   ここの mscale は ggml の attn_factor（既定 1）で、(1 + 0.1·ln) はカーネルの中で掛ける（ggml と同じ分け方）。
///   corr0 = max(0, floor(c(beta_fast)))、corr1 = min(n_rot − 1, ceil(c(beta_slow)))、c(b) = n_rot·ln(orig/(2π·b))/(2·ln base)。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yarn {
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub mscale: f32,
    pub corr0: f32,
    pub corr1: f32,
}

impl Yarn {
    pub const OFF: Yarn = Yarn { freq_scale: 1.0, ext_factor: 0.0, mscale: 1.0, corr0: 0.0, corr1: 0.0 };
    /// カーネルに渡す float[5]
    pub fn as_arr(&self) -> [f32; 5] {
        [self.freq_scale, self.ext_factor, self.mscale, self.corr0, self.corr1]
    }
    pub fn is_off(&self) -> bool {
        *self == Yarn::OFF
    }
    /// factor（例 4）・orig（元の文脈。例 262144）・attn_factor（既定 1）・beta_fast 32・beta_slow 1
    pub fn new(factor: f32, orig: usize, n_rot: usize, base: f32, attn_factor: f32, beta_fast: f32, beta_slow: f32) -> Yarn {
        let fs = 1.0 / factor;
        let c = |b: f32| n_rot as f32 * (orig as f32 / (b * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln());
        let corr0 = c(beta_fast).floor().max(0.0);
        let corr1 = c(beta_slow).ceil().min(n_rot as f32 - 1.0);
        Yarn { freq_scale: fs, ext_factor: 1.0, mscale: attn_factor, corr0, corr1 }
    }
}

/// 処理全体の RoPE の伸ばし方（CLI の --rope-scaling yarn --rope-factor F --rope-orig N で、重みを読む前に決める）。
/// 決めていなければ GGUF の {arch}.rope.scaling.* を見る（llama.cpp の書き方）。どちらも無ければ OFF。
#[derive(Clone, Copy, Debug)]
pub struct RopeScaling {
    pub factor: f32,
    pub orig: usize,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

static ROPE_SCALING: std::sync::OnceLock<Option<RopeScaling>> = std::sync::OnceLock::new();
static KV8: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// 本体の KV を int8（32 要素ごとに f32 の係数 = 1 ベクトル 288 バイト、fp16 の 56%）にするか（CLI の --kv8。重みを読む前に決める）。
/// MTP の頭の KV は fp16 のまま。2 回目以降は無視
pub fn set_kv8(on: bool) -> bool {
    let _ = KV8.set(on);
    kv8()
}
pub fn kv8() -> bool {
    KV8.get().copied().unwrap_or(false)
}

/// 引数（"rope-scaling" 等の値を返す関数）から決める。2 回目以降は無視（最初の物が残る）
pub fn set_rope_scaling_from(get: &dyn Fn(&str) -> Option<String>) -> Result<Option<RopeScaling>, String> {
    let kind = get("rope-scaling");
    let rs = match kind.as_deref() {
        None | Some("none") => None,
        Some("yarn") => {
            let f = |k: &str, d: f32| -> Result<f32, String> {
                get(k).map(|v| v.parse::<f32>().map_err(|_| format!("--{k} は数"))).transpose().map(|v| v.unwrap_or(d))
            };
            let factor = f("rope-factor", 4.0)?;
            let orig = f("rope-orig", 262144.0)? as usize;
            if !(factor > 1.0) || orig == 0 {
                return Err("--rope-factor は 1 より大きく、--rope-orig は正の数".into());
            }
            Some(RopeScaling { factor, orig, attn_factor: f("rope-attn-factor", 1.0)?, beta_fast: f("yarn-beta-fast", 32.0)?, beta_slow: f("yarn-beta-slow", 1.0)? })
        }
        Some(x) => return Err(format!("--rope-scaling は yarn か none（{x}）")),
    };
    let _ = ROPE_SCALING.set(rs);
    Ok(*ROPE_SCALING.get().unwrap())
}

/// 形を作るときの YaRN。CLI で決めた物 → GGUF の rope.scaling.*（type = yarn のときだけ）→ OFF
pub fn yarn_for(n_rot: usize, base: f32, gguf_scaling: Option<RopeScaling>) -> Yarn {
    match ROPE_SCALING.get().copied().flatten().or(gguf_scaling) {
        Some(r) => Yarn::new(r.factor, r.orig, n_rot, base, r.attn_factor, r.beta_fast, r.beta_slow),
        None => Yarn::OFF,
    }
}

/// GGUF の {arch}.rope.scaling.type = "yarn" と factor・original_context_length・attn_factor（llama.cpp の書き方）
pub fn gguf_rope_scaling(m: &crate::gguf::Model) -> Option<RopeScaling> {
    if m.hp("rope.scaling.type")?.as_str()? != "yarn" {
        return None;
    }
    Some(RopeScaling {
        factor: m.hp_f32("rope.scaling.factor")?,
        orig: m.hp_u32("rope.scaling.original_context_length")? as usize,
        attn_factor: m.hp_f32("rope.scaling.attn_factor").unwrap_or(1.0),
        beta_fast: m.hp_f32("rope.scaling.yarn_beta_fast").unwrap_or(32.0),
        beta_slow: m.hp_f32("rope.scaling.yarn_beta_slow").unwrap_or(1.0),
    })
}

impl Shape {
    pub const QWEN38_27B: Shape = Shape { h: 24, hkv: 4, n_rot: 64, sections: [11, 11, 10, 0], theta: 1e7, eps: 1e-6, yarn: Yarn::OFF };
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
            freq_scale: sh.yarn.freq_scale,
            ext_factor: sh.yarn.ext_factor,
            mscale: sh.yarn.mscale,
            corr0: sh.yarn.corr0,
            corr1: sh.yarn.corr1,
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
