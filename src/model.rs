//! Qwen3.8-27B（GGUF arch qwen35）の全体の計算（M1: GPU 1 枚・1 人・温度 0）。式は docs/qwen35-math.md。
//!
//!   重み: 大きな行列は g3 の並び（kernels/g3.cuh）に詰め直して GPU に 1 通りだけ置く（Q3_K・IQ3_S は Q6_K に直る）。
//!         デコード（と短いプレフィル）は g3 の GEMV でそのまま掛ける。長いプレフィルは行列を 1 本ずつ fp16 に戻して
//!         GEMM（kernels/gemm.cu）の F16 の道で掛ける（GGUF の並びと 2 通りを持つと 32GB に入らない）。
//!         token_embd は GGUF の並びのまま（行を引いて戻す）、norm・ssm_a・ssm_dt・ssm_conv1d は f32
//!   1 歩: forward(列, 語) が語の行列 h [T][5120]（f32 の残差）を 64 層通す。T ≤ gemv_max なら GEMV（B = T）、それより多いと GEMM。
//!         GDN の層は src/gdn.rs、普通の Attention の層は src/attn.rs、残差には GEMV の add・GEMM の足し込みで直に足す
//!   状態: Seq（GDN の状態 48 層 × 48×128×128 f32・conv の状態 48 層 × 10240×3・KV 16 層 × [頁][4][64][256] fp16。
//!         M1 は 1 列なので頁表は 0, 1, 2, … のまま＝続いた KV）
use crate::attn::{self, AttnParams, RopeParams};
use crate::cuda::{check, ffi, DevBuf};
use crate::gdn;
use crate::gguf;
use crate::qtype::QType;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::{null, null_mut};

mod decode;
pub use decode::DecodeStats;
mod spec_target;
pub use spec_target::ModelTarget;

mod kffi {
    use crate::cuda::ffi::Stream;
    use std::ffi::c_void;
    extern "C" {
        pub fn infer_m_embed(qtype: i32, w: *const c_void, k: i32, ids: *const i32, n_tok: i32, y: *mut f32, s: Stream) -> i32;
        pub fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, n: usize, kind: i32, s: Stream) -> i32;
        pub fn cudaStreamWaitEvent(s: Stream, e: *mut c_void, flags: u32) -> i32;
        pub fn infer_m_rmsnorm(
            x: *const f32, x_ld: i64, w: *const f32, k: i32, n_rows: i32, eps: f32, y: *mut f32, y_ld: i64, y16: *mut c_void,
            y16_ld: i64, s: Stream,
        ) -> i32;
        pub fn infer_m_f32_to_f16(x: *const f32, n: i64, y: *mut c_void, s: Stream) -> i32;
        pub fn infer_g3_gemm_ex(
            nmat: i32, qtypes: *const i32, ws: *const *const c_void, ns: *const i32, ys: *const *mut c_void, ldys: *const i32, k: i32, m: i32,
            x: *const c_void, xlo: *const c_void, ldx: i32, mode: i32, cfg: i32, s: Stream,
        ) -> i32;
        pub fn infer_m_g3_deq16(qtype: i32, w: *const c_void, k: i32, n: i32, y: *mut c_void, s: Stream) -> i32;
        pub fn infer_m_argmax(x: *const f32, n: i32, ld: i64, n_rows: i32, out: *mut i32, out_v: *mut f32, s: Stream) -> i32;
        pub fn infer_m_nonfinite(x: *const f32, n: i64, cnt: *mut i32, s: Stream) -> i32;
        pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
        pub fn cudaMemset(p: *mut c_void, v: i32, n: usize) -> i32;
        pub fn cudaMemsetAsync(p: *mut c_void, v: i32, n: usize, s: Stream) -> i32;
        // mHC（kernels/mhc.cu・SenseNova-Loop）
        pub fn infer_mhc_route(
            a: *const f32, b: *const f32, ld: i64, d: i32, w: *const f32, bias: *const f32, t: i32, iters: i32, pre_scale: f32, h: *mut f32,
            h_ld: i64, rt: *mut f32, s: Stream,
        ) -> i32;
        pub fn infer_mhc_merge(
            a: *mut f32, b: *mut f32, ld: i64, d: i32, y: *const f32, y_ld: i64, rt: *const f32, t: i32, h: *mut f32, h_ld: i64, s: Stream,
        ) -> i32;
        pub fn cudaStreamCreateWithPriority(s: *mut Stream, flags: u32, prio: i32) -> i32;
        pub fn cudaStreamSynchronize(s: Stream) -> i32;
        pub fn cudaStreamQuery(s: Stream) -> i32;
        // MoE（kernels/moe.cu・gemv3.cu）
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
        pub fn infer_cand_ws_bytes(rows: i32, n: i32, k: i32) -> i64;
        pub fn infer_cand_topk(
            lg: *const f32, ld: i64, rows: i32, n: i32, k: i32, temps: *const f32, ws: *mut c_void, ws_bytes: i64, ids: *mut u32, vals: *mut f32,
            mz: *mut f64, s: Stream,
        ) -> i32;
        pub fn infer_moe_group(
            ids: *const i32, r: i32, kk: i32, ne: i32, tokb: i32, xidx: *mut i32, rowmap: *mut i32, grp: *mut i32, n_blk_max: i32, s: Stream,
        ) -> i32;
        pub fn infer_g3_moe_gemm(
            nmat: i32, qtypes: *const i32, ws: *const *const c_void, es: *const i64, ns: *const i32, ys: *const *mut c_void, ldys: *const i32,
            k: i32, x: *const c_void, xlo: *const c_void, ldx: i32, xidx: *const i32, grp: *const i32, n_blk: i32, tokb: i32, mode: i32,
            s: Stream,
        ) -> i32;
        pub fn infer_moe_lo16(x: *const f32, hi: *const c_void, n: i64, lo: *mut c_void, s: Stream) -> i32;
        pub fn infer_g3_moe_gemv(
            nm: i32, qtype: *const i32, w: *const *const c_void, estride: *const i64, n: *const i32, y: *const *mut f32, k: i32,
            xp: *const c_void, rx: i32, xidx: *const i32, je: *const i32, jr0: *const i32, jcnt: *const i32, n_jobs: i32, bmax: i32,
            s: Stream,
        ) -> i32;
    }
}

const S: ffi::Stream = std::ptr::null_mut();
/// ggml の F16 の型番号（GEMM の F16 の道）
const Q_F16: i32 = 1;
/// g3 の 1 回の本数の上限
pub const GEMV_B_MAX: usize = 32;

// ---------------------------------------------------------------- 形

/// GGUF の qwen35.* から読む形
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// 本体の層の数（block_count − nextn_predict_layers = 64）
    pub n_layer: usize,
    pub d: usize,
    pub d_ff: usize,
    pub vocab: usize,
    pub eps: f32,
    /// full_attention_interval（4）: il % 4 == 3 が普通の Attention
    pub interval: usize,
    pub attn: attn::Shape,
    pub gdn: gdn::Shape,
    /// MoE（Qwen3.6-35B-A3B・arch qwen35moe）。None なら密な FFN（Qwen3.8-27B・arch qwen35）
    pub moe: Option<MoeCfg>,
    /// 層の繰り返しと mHC（SenseNova-Loop・arch sensenova_loop）。None なら層は 1 回ずつ。
    /// Some のとき n_layer は「実行の数」（56）で、層 il の重みは物理の層 phys(il) のもの。KV・GDN の状態は実行ごとに別
    pub lp: Option<LoopCfg>,
}

/// SenseNova-Loop の形（GGUF の sensenova_loop.loop.*・mhc.*）。実行の順 = 0..start、start..end を count 回、end..n_phys。
/// 学習していないモデル（qwen35・qwen35moe）の層の繰り返し（`--repeat-layers S:E:N`・枝 relayer）にも使う: mHC なし（mhc_start = mhc_end = 0）・
/// 普通の残差・重みはどの実行も物理の blk から
#[derive(Clone, Copy, Debug)]
pub struct LoopCfg {
    pub start: usize,
    pub end: usize,
    pub count: usize,
    /// 物理の層の数（32）
    pub n_phys: usize,
    /// mHC の物理の層の範囲（4..28）。残差は 2 本、最初の層で h を 2 本に写し、最後の層の出口で平均に戻す
    pub mhc_start: usize,
    pub mhc_end: usize,
    /// Sinkhorn の回数（20）
    pub iters: usize,
    pub pre_scale: f32,
    /// SenseNova-Loop の GGUF は繰り返しの後ろの層を実行の番号の blk（52..55）に置く。層の繰り返し（relayer）は false（物理の blk）
    pub tail_blk_exec: bool,
}

/// 層の繰り返し（`--repeat-layers START:END:N`・環境変数 TACHYON_REPEAT_LAYERS）: 物理の層 START..END を合わせて N 回。重みを読む前に決める。
/// 2 回目以降は無視（最初の物が残る）。N = 1 は切と同じ
static REPEAT: std::sync::OnceLock<Option<(usize, usize, usize)>> = std::sync::OnceLock::new();

fn parse_repeat(v: &str) -> Result<Option<(usize, usize, usize)>, String> {
    let v = v.trim();
    if v.is_empty() || v == "none" || v == "0" {
        return Ok(None);
    }
    let p: Vec<&str> = v.split(':').collect();
    let n = |x: &str| x.trim().parse::<usize>().map_err(|_| format!("--repeat-layers は START:END:N（{v}）"));
    if p.len() != 3 {
        return Err(format!("--repeat-layers は START:END:N（{v}）"));
    }
    let (a, b, c) = (n(p[0])?, n(p[1])?, n(p[2])?);
    if a >= b || c == 0 {
        return Err(format!("--repeat-layers の範囲・回数が変（{v}）"));
    }
    Ok(if c == 1 { None } else { Some((a, b, c)) })
}

/// 引数の値（無ければ環境変数 TACHYON_REPEAT_LAYERS）で層の繰り返しを決める。返りは決まった値
pub fn set_repeat_layers(arg: Option<&str>) -> Result<Option<(usize, usize, usize)>, String> {
    let v = match arg {
        Some(a) => parse_repeat(a)?,
        None => match std::env::var("TACHYON_REPEAT_LAYERS") {
            Ok(e) => parse_repeat(&e)?,
            Err(_) => None,
        },
    };
    if REPEAT.set(v).is_err() && repeat_layers() != v {
        return Err(format!("層の繰り返しは 1 回だけ決める（前 {:?}・今 {v:?}）", repeat_layers()));
    }
    Ok(repeat_layers())
}

pub fn repeat_layers() -> Option<(usize, usize, usize)> {
    REPEAT.get().copied().flatten()
}

impl LoopCfg {
    fn extra(&self) -> usize {
        (self.count - 1) * (self.end - self.start)
    }
    /// 実行 il の物理の層
    pub fn phys(&self, il: usize) -> usize {
        let w = self.end - self.start;
        if il < self.start {
            il
        } else if il < self.end + self.extra() {
            self.start + (il - self.start) % w
        } else {
            il - self.extra()
        }
    }
    /// 実行 il の重みが GGUF のどの blk にあるか（繰り返しは最初の回の blk。最後の層は blk 52.. に実行の番号で置かれている）
    pub fn src_blk(&self, il: usize) -> usize {
        if !self.tail_blk_exec {
            return self.phys(il);
        }
        if il >= self.start && il < self.end + self.extra() {
            self.phys(il)
        } else {
            il
        }
    }
    pub fn is_mhc(&self, il: usize) -> bool {
        let p = self.phys(il);
        p >= self.mhc_start && p < self.mhc_end
    }
    /// mHC の最初の層（h を 2 本に写す）
    pub fn mhc_first(&self, il: usize) -> bool {
        self.phys(il) == self.mhc_start
    }
}

/// MoE の形（GGUF の qwen35moe.*）。FFN = 上位 k 個の専門家（softmax・重みを正規化）＋ 共有の専門家（sigmoid のゲート）
#[derive(Clone, Copy, Debug)]
pub struct MoeCfg {
    /// 専門家の数（256）
    pub n_exp: usize,
    /// 1 語で使う数（8）
    pub k: usize,
    /// 専門家の幅（512）
    pub ff: usize,
}

impl Config {
    pub fn from_gguf(m: &gguf::Model) -> Result<Config, String> {
        let is_moe = match m.arch() {
            "qwen35" | "sensenova_loop" => false,
            "qwen35moe" => true,
            a => return Err(format!("arch が qwen35・qwen35moe・sensenova_loop ではない: {a}")),
        };
        let u = |k: &str| m.hp_u32(k).map(|v| v as usize).ok_or_else(|| format!("{}.{k} が無い", m.arch()));
        let n_all = u("block_count")?;
        let nextn = m.hp_u32("nextn_predict_layers").unwrap_or(0) as usize;
        let eps = m.hp_f32("attention.layer_norm_rms_epsilon").ok_or("attention.layer_norm_rms_epsilon が無い")?;
        let sec = m.hp_arr_i64("rope.dimension_sections").unwrap_or_else(|| vec![0; 4]);
        let mut sections = [0i32; 4];
        for (i, v) in sec.iter().take(4).enumerate() {
            sections[i] = *v as i32;
        }
        let moe = if is_moe {
            Some(MoeCfg { n_exp: u("expert_count")?, k: u("expert_used_count")?, ff: u("expert_feed_forward_length")? })
        } else {
            None
        };
        let lp = if m.arch() == "sensenova_loop" { Some(Self::loop_cfg(m, n_all - nextn)?) } else { None };
        let mut cfg = Config {
            lp,
            n_layer: n_all - nextn,
            d: u("embedding_length")?,
            // MoE は共有の専門家の幅（密な FFN の作業場をそのまま共有の専門家に使う）
            d_ff: if is_moe { u("expert_shared_feed_forward_length")? } else { u("feed_forward_length")? },
            moe,
            vocab: m.tensor("output.weight").n_rows(),
            eps,
            interval: u("full_attention_interval")?,
            attn: attn::Shape {
                h: u("attention.head_count")?,
                hkv: u("attention.head_count_kv")?,
                n_rot: u("rope.dimension_count")?,
                sections,
                theta: m.hp_f32("rope.freq_base").ok_or("rope.freq_base が無い")?,
                eps,
                yarn: attn::yarn_for(u("rope.dimension_count")?, m.hp_f32("rope.freq_base").unwrap_or(1e7), attn::gguf_rope_scaling(m)),
            },
            gdn: gdn::Shape { hk: u("ssm.group_count")?, hv: u("ssm.time_step_rank")?, eps },
        };
        // カーネルの決め打ちと合うか
        if u("attention.key_length")? != attn::D || u("attention.value_length")? != attn::D {
            return Err(format!("head_dim が {} ではない", attn::D));
        }
        if u("ssm.state_size")? != gdn::D || u("ssm.conv_kernel")? != 4 || u("ssm.inner_size")? != cfg.gdn.out_dim() {
            return Err("ssm の形がカーネル（D 128・畳み込み 4）と合わない".into());
        }
        if cfg.attn.h % cfg.attn.hkv != 0 || cfg.d % 256 != 0 || cfg.d_ff % 256 != 0 {
            return Err(format!("形が変: {cfg:?}"));
        }
        if let Some(mc) = cfg.moe {
            if mc.ff % 256 != 0 || mc.k < 1 || mc.k > 32 || mc.n_exp > 1023 || cfg.d % 128 != 0 {
                return Err(format!("MoE の形がカーネルと合わない: {mc:?}"));
            }
        }
        if let Some((start, end, count)) = repeat_layers() {
            // 層の繰り返し（学習していない繰り返し・mHC なし）。境は Attention の組（interval 層）に揃える＝実行の番号 il % interval が物理の層と同じ種類
            if cfg.lp.is_some() {
                return Err("--repeat-layers は sensenova_loop と一緒に使えない".into());
            }
            let iv = cfg.interval;
            if start % iv != 0 || end % iv != 0 || end > cfg.n_layer {
                return Err(format!("--repeat-layers {start}:{end}:{count} は {iv} の倍数で 0..{} の中", cfg.n_layer));
            }
            let n_phys = cfg.n_layer;
            let lp = LoopCfg { start, end, count, n_phys, mhc_start: 0, mhc_end: 0, iters: 0, pre_scale: 1.0, tail_blk_exec: false };
            cfg.n_layer = n_phys + lp.extra();
            for il in 0..cfg.n_layer {
                assert_eq!(cfg.is_attn(il), cfg.is_attn(lp.phys(il)), "実行 {il} の層の種類");
            }
            cfg.lp = Some(lp);
        }
        if m.arch() == "sensenova_loop" {
            // 実行ごとの層の種類（GGUF の attention.recurrent_layers）が il % 4 の決まりと合うか
            let rec = m.hp("attention.recurrent_layers").and_then(|v| v.as_arr()).ok_or("attention.recurrent_layers が無い")?;
            for il in 0..cfg.n_layer {
                let r = rec.get(il).and_then(|v| v.as_bool()).ok_or("attention.recurrent_layers が短い")?;
                if r == cfg.is_attn(il) {
                    return Err(format!("実行 {il} の層の種類が full_attention_interval と合わない"));
                }
            }
        }
        Ok(cfg)
    }
    /// sensenova_loop.loop.*・mhc.*（llama.cpp の OpenSenseNova 版と同じ決まりで確かめる）
    fn loop_cfg(m: &gguf::Model, n_exec: usize) -> Result<LoopCfg, String> {
        let u = |k: &str| m.hp_u32(k).map(|v| v as usize).ok_or_else(|| format!("sensenova_loop.{k} が無い"));
        let (start, end, count) = (u("loop.start")?, u("loop.end")?, u("loop.count")?);
        if count == 0 || start >= end {
            return Err("loop の範囲・回数が変".into());
        }
        let extra = (count - 1) * (end - start);
        if n_exec < end + extra {
            return Err("loop の実行の数が block_count を超える".into());
        }
        let n_phys = n_exec - extra;
        let (ms, me, streams, iters) = (u("mhc.start")?, u("mhc.end")?, u("mhc.streams")?, u("mhc.iterations")?);
        let fixed = m.hp("mhc.fixed_identity").and_then(|v| v.as_bool()).ok_or("mhc.fixed_identity が無い")?;
        let pre_scale = m.hp_f32("mhc.pre_scale").ok_or("mhc.pre_scale が無い")?;
        if streams != 2 || fixed || iters == 0 || ms >= me || me > n_phys || !pre_scale.is_finite() || pre_scale <= 0.0 {
            return Err(format!("mHC の形が対応外（streams {streams}・fixed_identity {fixed}・iterations {iters}・{ms}..{me}）"));
        }
        // mHC は loop の境をまたがない（中にあるなら繰り返しの度に入って出る）
        if !(me <= start || (ms >= start && me <= end) || ms >= end) {
            return Err("mHC の範囲が loop の境をまたぐ".into());
        }
        Ok(LoopCfg { start, end, count, n_phys, mhc_start: ms, mhc_end: me, iters, pre_scale, tail_blk_exec: true })
    }
    pub fn is_attn(&self, il: usize) -> bool {
        il % self.interval == self.interval - 1
    }
    /// q_full（頭ごとに q | gate）の幅
    pub fn q_full(&self) -> usize {
        self.attn.h * 2 * attn::D
    }
    pub fn kv_dim(&self) -> usize {
        self.attn.hkv * attn::D
    }
    /// Attention の出口・GDN の出口の幅（どちらも 6144）
    pub fn attn_out(&self) -> usize {
        self.attn.h * attn::D
    }
}

// ---------------------------------------------------------------- 重み

/// GPU 上の重み行列（g3 の並び。qt は詰め直した後の型）
pub struct Mat {
    pub buf: DevBuf,
    pub qt: i32,
    pub src: QType,
    pub k: usize,
    pub n: usize,
}

pub struct GdnW {
    pub qkv: Mat,
    pub gate: Mat,
    pub alpha: Mat,
    pub beta: Mat,
    pub out: Mat,
    pub a: DevBuf,
    pub dt: DevBuf,
    pub norm: DevBuf,
    pub conv: DevBuf,
}

pub struct AttnW {
    pub q: Mat,
    pub k: Mat,
    pub v: Mat,
    pub o: Mat,
    pub q_norm: DevBuf,
    pub k_norm: DevBuf,
}

pub enum Mix {
    Gdn(GdnW),
    Attn(AttnW),
}

pub struct LayerW {
    pub attn_norm: DevBuf,
    pub post_norm: DevBuf,
    pub mix: Mix,
    /// 密な FFN（MoE の層では共有の専門家）
    pub gate: Mat,
    pub up: Mat,
    pub down: Mat,
    pub moe: Option<MoeW>,
    /// mHC の層（SenseNova-Loop の 4..28）: Attention の前・FFN の前の振り分け
    pub hc: Option<HcW>,
}

/// mHC の振り分けの重み（f32 [8][2d] と [8]。量子化しない）
pub struct HcW {
    pub attn_w: DevBuf,
    pub attn_b: DevBuf,
    pub ffn_w: DevBuf,
    pub ffn_b: DevBuf,
}

/// MoE の層の重み: ルーター（f32 [n_exp + 1][d]。最後の行が共有の専門家のゲート）と、専門家を続けて並べた g3 の行列
pub struct MoeW {
    pub router: DevBuf,
    pub gate: Mat,
    pub up: Mat,
    pub down: Mat,
    /// 専門家 1 つぶんのバイト数（gate・up は [ff][d]、down は [d][ff]）
    pub es_gu: i64,
    pub es_d: i64,
}

fn h2d(dst: *mut c_void, src: &[u8]) {
    check(unsafe { ffi::cudaMemcpy(dst, src.as_ptr().cast(), src.len(), 1) }, "cudaMemcpy H2D");
}
fn d2h(dst: &mut [u8], src: *const c_void) {
    check(unsafe { ffi::cudaMemcpy(dst.as_mut_ptr().cast(), src, dst.len(), 2) }, "cudaMemcpy D2H");
}
fn d2h_f32(src: *const f32, n: usize) -> Vec<f32> {
    let mut v = vec![0f32; n];
    check(unsafe { ffi::cudaMemcpy(v.as_mut_ptr().cast(), src.cast(), n * 4, 2) }, "cudaMemcpy D2H");
    v
}
fn zero(p: *mut c_void, n: usize) {
    check(unsafe { kffi::cudaMemset(p, 0, n) }, "cudaMemset");
}
/// バッファの頭から b バイト先
fn at<T>(b: &DevBuf, bytes: usize) -> *mut T {
    debug_assert!(bytes <= b.len);
    unsafe { b.ptr.cast::<u8>().add(bytes).cast() }
}

/// (空き, 全部) バイト
pub fn mem_info() -> (usize, usize) {
    let (mut f, mut t) = (0usize, 0usize);
    check(unsafe { kffi::cudaMemGetInfo(&mut f, &mut t) }, "cudaMemGetInfo");
    (f, t)
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
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

fn d2h_f16(src: *const c_void, n: usize) -> Vec<f32> {
    let mut b = vec![0u8; n * 2];
    d2h(&mut b, src);
    b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()
}

/// 読み込みの置き場（GGUF の行を何十 MB ずつ GPU に送って詰め直す）
struct Loader {
    stage: DevBuf,
    bytes_gguf: usize,
    bytes_gpu: usize,
}

impl Loader {
    fn mat(&mut self, tv: gguf::TensorView) -> Mat {
        self.mat_rows(tv, 0, tv.n_rows())
    }
    /// 行 r0..r0+n だけを 1 本の行列に（qkv と gate を 1 本に繋いだ attn_qkv_gate などを分ける）
    fn mat_rows(&mut self, tv: gguf::TensorView, r0_all: usize, n: usize) -> Mat {
        let src = tv.qt();
        let (qt, k) = (src as i32, tv.k());
        assert!(r0_all + n <= tv.n_rows(), "{}: 行 {r0_all}+{n} が外", tv.name);
        let bytes = unsafe { ffi::infer_g3_bytes(qt, k as i32, n as i32) };
        assert!(bytes > 0, "{}: g3 が扱わない型 {}（k {k}）", tv.name, tv.type_name());
        let buf = DevBuf::new(bytes as usize);
        let tile = unsafe { ffi::infer_g3_bytes(qt, k as i32, 32) } as usize;
        let rb = tv.row_bytes();
        let step = (self.stage.len / rb / 32 * 32).max(32);
        assert!(step * rb <= self.stage.len, "{}: 置き場が 32 行より小さい", tv.name);
        let mut r0 = 0;
        while r0 < n {
            let nr = step.min(n - r0);
            h2d(self.stage.ptr, tv.rows(r0_all + r0, nr));
            let dst: *mut c_void = at(&buf, r0 / 32 * tile);
            check(unsafe { ffi::infer_g3_repack(qt, self.stage.ptr, dst, k as i32, nr as i32, S) }, "g3 repack");
            r0 += nr;
        }
        self.bytes_gguf += n * rb;
        self.bytes_gpu += buf.len;
        Mat { buf, qt: unsafe { ffi::infer_g3_type(qt) }, src, k, n }
    }
    fn f32v(&mut self, tv: gguf::TensorView) -> DevBuf {
        assert_eq!(tv.qtype, Some(QType::F32), "{}: f32 のはず（{}）", tv.name, tv.type_name());
        self.bytes_gguf += tv.bytes.len();
        self.bytes_gpu += tv.bytes.len();
        DevBuf::from_bytes(tv.bytes)
    }
}

// ---------------------------------------------------------------- 列ごとの状態

/// 何本かの列の状態をまとめて置く所（M2: デコードで何本もまとめて 1 歩にするとき、どの列も同じ頭からの番号で引けるように）。
/// 並びは列ごとに続く: state [slot][GDN の層][hv][D][D] f32、conv [slot][GDN の層][conv_ch][3] f32、
/// kc・vc [slot][Attention の層][頁][4][64][256] fp16。1 本の列だけを見る道（プレフィル）は列の頭から今まで通りに使える
struct Pool {
    id: u64,
    n_slots: usize,
    n_gdn: usize,
    n_attn: usize,
    n_pages: usize,
    state_len: usize,
    conv_len: usize,
    page_bytes: usize,
    /// KV が int8 か（attn::kv8()）。int8 なら ks・vs [Attention の層][頁][4][64][8] f32 の係数を持つ
    kv8: bool,
    scale_bytes: usize,
    ks: DevBuf,
    vs: DevBuf,
    /// GDN の状態を 2 倍取った（下書きの確かめの仮の置き場つき）
    spec_state: bool,
    state: DevBuf,
    conv: DevBuf,
    /// KV: [Attention の層][頁の通し番号 total_pages][4][64][256] fp16。列は頁表（Seq::pages）で頁を引く
    kc: DevBuf,
    vc: DevBuf,
    total_pages: usize,
    /// 頁を列に配るか（M3）。false なら列 slot は頁 slot·n_pages.. を最初から持つ（前と同じ並び）
    paged: bool,
    free: std::cell::RefCell<Vec<u32>>,
}

static POOL_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// 1 本の列（会話）の状態。置き場は Pool の slot 番目
/// プロンプトの中の画像の行（位置 start から n = nx·ny 語。行の埋め込みは emb [n][d] の f32 の GPU の行）。
/// RoPE の位置は HF の Qwen3-VL の決まり: 画像の語 k は (t, h, w) = (b, b + k/nx, b + k%nx)（b = 画像の頭の位置 ＋ それまでのずれ）、
/// 画像の後ろの文字は b + max(nx, ny) から続く（列の位置と RoPE の位置のずれ = max(nx, ny) − n が画像ごとに足される）
#[derive(Clone)]
pub struct MediaSpan {
    pub start: usize,
    pub nx: usize,
    pub ny: usize,
    pub emb: std::rc::Rc<DevBuf>,
}

impl MediaSpan {
    pub fn n(&self) -> usize {
        self.nx * self.ny
    }
    fn shift(&self) -> i64 {
        self.nx.max(self.ny) as i64 - self.n() as i64
    }
}

pub struct Seq {
    /// 次の語の位置（ここまでの語の数）
    pub pos: usize,
    /// プロンプトの中の画像（start の順。無ければ空 = 今までと同じ位置）
    pub media: Vec<MediaSpan>,
    pub max_ctx: usize,
    n_pages: usize,
    /// 置き場の中の番号（Model::new_seqs で作った列どうしは同じ置き場を分け合う）
    pub slot: usize,
    pool: std::rc::Rc<Pool>,
    /// 頁表（位置 p は頁 pages[p / 64] の p % 64 行目）
    pages: Vec<u32>,
}

impl Seq {
    /// layers: 状態を持つ層の範囲（ふつうは 0..64）。1 本だけの置き場
    pub fn new(cfg: &Config, layers: std::ops::Range<usize>, max_ctx: usize) -> Seq {
        Self::new_n(cfg, layers, max_ctx, 1).pop().unwrap()
    }
    /// n 本の列（同じ置き場。デコードでまとめられる）
    pub fn new_n(cfg: &Config, layers: std::ops::Range<usize>, max_ctx: usize, n: usize) -> Vec<Seq> {
        Self::new_n_ex(cfg, layers, max_ctx, n, false)
    }
    /// spec_state なら GDN の状態・conv の状態を列の数の 2 倍取る（後ろ半分は下書きの確かめの仮の置き場。KV は増やさない）。
    /// 列 slot の層 gi の仮の置き場は、state_at(gi) から見て slot の番号で +n·n_gdn（spec::SPEC_SLOTS_OUT）
    pub fn new_n_ex(cfg: &Config, layers: std::ops::Range<usize>, max_ctx: usize, n: usize, spec_state: bool) -> Vec<Seq> {
        Self::new_pool(cfg, layers, max_ctx, n, spec_state, None)
    }
    /// 頁を配る置き場（M3）: n 本の列で KV を total_tokens 語ぶん分け合う（1 本は max_ctx まで）。頁は ensure で取り、free_pages で返す
    pub fn new_paged(cfg: &Config, layers: std::ops::Range<usize>, max_ctx: usize, n: usize, spec_state: bool, total_tokens: usize) -> Vec<Seq> {
        Self::new_pool(cfg, layers, max_ctx, n, spec_state, Some(total_tokens))
    }
    fn new_pool(cfg: &Config, layers: std::ops::Range<usize>, max_ctx: usize, n: usize, spec_state: bool, total: Option<usize>) -> Vec<Seq> {
        assert!(n >= 1);
        let ns = if spec_state { 2 * n } else { n };
        let n_gdn = layers.clone().filter(|&il| !cfg.is_attn(il)).count();
        let n_attn = layers.len() - n_gdn;
        let n_pages = (max_ctx + attn::PAGE - 1) / attn::PAGE;
        let kv8 = attn::kv8();
        let page_bytes = cfg.attn.page_bytes(kv8);
        let scale_bytes = if kv8 { cfg.attn.page_scales() * 4 } else { 0 };
        let total_pages = match total {
            Some(t) => (t + attn::PAGE - 1) / attn::PAGE,
            None => n * n_pages,
        };
        let (state_len, conv_len) = (cfg.gdn.state_len(), cfg.gdn.conv_state_len());
        let pool = std::rc::Rc::new(Pool {
            id: POOL_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            n_slots: n,
            n_gdn,
            n_attn,
            n_pages,
            state_len,
            conv_len,
            page_bytes,
            kv8,
            scale_bytes,
            ks: DevBuf::new(total_pages * n_attn.max(1) * scale_bytes),
            vs: DevBuf::new(total_pages * n_attn.max(1) * scale_bytes),
            state: DevBuf::new(ns * n_gdn.max(1) * state_len * 4),
            conv: DevBuf::new(ns * n_gdn.max(1) * conv_len * 4),
            spec_state,
            kc: DevBuf::new(total_pages * n_attn.max(1) * page_bytes),
            vc: DevBuf::new(total_pages * n_attn.max(1) * page_bytes),
            total_pages,
            paged: total.is_some(),
            free: std::cell::RefCell::new(if total.is_some() { (0..total_pages as u32).rev().collect() } else { vec![] }),
        });
        let v: Vec<Seq> = (0..n)
            .map(|slot| Seq {
                pos: 0,
                media: vec![],
                max_ctx: n_pages * attn::PAGE,
                n_pages,
                slot,
                pool: pool.clone(),
                pages: if total.is_some() { vec![] } else { ((slot * n_pages) as u32..((slot + 1) * n_pages) as u32).collect() },
            })
            .collect();
        zero(pool.state.ptr, pool.state.len);
        zero(pool.conv.ptr, pool.conv.len);
        v
    }
    /// GDN の層 gi（載せた中の何番目の GDN の層か）の状態の頭
    fn state_at(&self, gi: usize) -> *mut f32 {
        let p = &self.pool;
        at(&p.state, (self.slot * p.n_gdn + gi) * p.state_len * 4)
    }
    fn conv_at(&self, gi: usize) -> *mut f32 {
        let p = &self.pool;
        at(&p.conv, (self.slot * p.n_gdn + gi) * p.conv_len * 4)
    }
    /// Attention の層 ai の K・V の頭（頁は頁表で引く）
    fn kc_at(&self, ai: usize) -> *mut c_void {
        let p = &self.pool;
        at(&p.kc, ai * p.total_pages * p.page_bytes)
    }
    fn vc_at(&self, ai: usize) -> *mut c_void {
        let p = &self.pool;
        at(&p.vc, ai * p.total_pages * p.page_bytes)
    }
    /// int8 の KV の係数の頭（fp16 なら null）
    fn ks_at(&self, ai: usize) -> *mut f32 {
        let p = &self.pool;
        if p.kv8 { at(&p.ks, ai * p.total_pages * p.scale_bytes) } else { std::ptr::null_mut() }
    }
    fn vs_at(&self, ai: usize) -> *mut f32 {
        let p = &self.pool;
        if p.kv8 { at(&p.vs, ai * p.total_pages * p.scale_bytes) } else { std::ptr::null_mut() }
    }
    pub fn kv8(&self) -> bool {
        self.pool.kv8
    }
    /// 位置 p の KV の行の通し番号（頁 × 64 ＋ 頁の中の行）
    fn kv_row(&self, p: usize) -> i32 {
        (self.pages[p / attn::PAGE] as usize * attn::PAGE + p % attn::PAGE) as i32
    }
    /// 頁表（i32）
    fn page_table(&self) -> Vec<i32> {
        self.pages.iter().map(|&x| x as i32).collect()
    }
    /// 位置 upto（語の数）まで書けるよう頁を取る。頁が足りなければ false（取った分はそのまま）
    pub fn ensure(&mut self, upto: usize) -> bool {
        let need = (upto + attn::PAGE - 1) / attn::PAGE;
        if need > self.n_pages {
            return false;
        }
        let mut f = self.pool.free.borrow_mut();
        while self.pages.len() < need {
            match f.pop() {
                Some(pg) => self.pages.push(pg),
                None => return false,
            }
        }
        true
    }
    /// 頁を置き場に返す（頁を配る置き場だけ。列の状態は reset まで残る）
    pub fn free_pages(&mut self) {
        if self.pool.paged {
            self.pool.free.borrow_mut().extend(self.pages.drain(..));
        }
    }
    /// 頁を持ち出す（頭の使い回しの控えへ）・戻す
    pub fn take_pages(&mut self) -> Vec<u32> {
        assert!(self.pool.paged);
        std::mem::take(&mut self.pages)
    }
    pub fn give_pages(&mut self, pages: Vec<u32>) {
        assert!(self.pool.paged && self.pages.is_empty());
        self.pages = pages;
    }
    /// 置き場の空いた頁の数
    pub fn free_page_count(&self) -> usize {
        self.pool.free.borrow().len()
    }
    /// 頁を置き場に返す（列の外で持っていた頁）
    pub fn return_pages(&self, pages: Vec<u32>) {
        assert!(self.pool.paged);
        self.pool.free.borrow_mut().extend(pages);
    }
    /// 頁表の写し
    pub fn pages_vec(&self) -> Vec<u32> {
        self.pages.clone()
    }
    /// 頁を全部持ち出す（頁を配る置き場なら。そうでなければ空）
    pub fn take_pages_any(&mut self) -> Vec<u32> {
        if self.pool.paged {
            std::mem::take(&mut self.pages)
        } else {
            vec![]
        }
    }
    pub fn n_pages_held(&self) -> usize {
        self.pages.len()
    }
    pub fn total_pages(&self) -> usize {
        self.pool.total_pages
    }
    fn zero_state(&self) {
        let p = &self.pool;
        zero(self.state_at(0).cast(), p.n_gdn * p.state_len * 4);
        zero(self.conv_at(0).cast(), p.n_gdn * p.conv_len * 4);
    }
    /// GDN の状態と conv の状態の大きさ（バイト。この列の分）
    pub fn state_bytes(&self) -> (usize, usize) {
        let p = &self.pool;
        (p.n_gdn * p.state_len * 4, p.n_gdn * p.conv_len * 4)
    }
    /// GDN・conv の状態を写す（頭の使い回しの控え。dst は state_bytes の大きさ）
    pub fn save_state(&self, st: &DevBuf, cv: &DevBuf) {
        let (a, b) = self.state_bytes();
        assert!(st.len >= a && cv.len >= b);
        crate::cuda::check(unsafe { crate::cuda::ffi::cudaMemcpy(st.ptr, self.state_at(0).cast(), a, 3) }, "状態の控え");
        crate::cuda::check(unsafe { crate::cuda::ffi::cudaMemcpy(cv.ptr, self.conv_at(0).cast(), b, 3) }, "conv の控え");
    }
    /// 控えから戻し、位置を pos にする（KV の 0..pos は控えを取った後に書き換えていないこと）
    pub fn load_state(&mut self, st: &DevBuf, cv: &DevBuf, pos: usize) {
        let (a, b) = self.state_bytes();
        assert!(st.len >= a && cv.len >= b && pos <= self.max_ctx);
        crate::cuda::check(unsafe { crate::cuda::ffi::cudaMemcpy(self.state_at(0).cast(), st.ptr, a, 3) }, "状態の戻し");
        crate::cuda::check(unsafe { crate::cuda::ffi::cudaMemcpy(self.conv_at(0).cast(), cv.ptr, b, 3) }, "conv の戻し");
        self.pos = pos;
    }
    /// 頁 page の KV の置き場（Attention の層ごとに K・V の順。頭の控えを RAM の段に写す・戻す。hostcache）
    pub fn kv_page_spans(&self, page: u32) -> Vec<(*mut u8, usize)> {
        let p = &self.pool;
        let mut v = Vec::with_capacity(p.n_attn * 2);
        for ai in 0..p.n_attn {
            let off = (ai * p.total_pages + page as usize) * p.page_bytes;
            v.push((at::<u8>(&p.kc, off), p.page_bytes));
            v.push((at::<u8>(&p.vc, off), p.page_bytes));
            if p.kv8 {
                let so = (ai * p.total_pages + page as usize) * p.scale_bytes;
                v.push((at::<u8>(&p.ks, so), p.scale_bytes));
                v.push((at::<u8>(&p.vs, so), p.scale_bytes));
            }
        }
        v
    }
    /// この列の GDN の状態・conv の状態の置き場（state_bytes の大きさ。RAM の段に写す・戻す）
    pub fn state_spans(&self) -> [(*mut u8, usize); 2] {
        let (a, b) = self.state_bytes();
        [(self.state_at(0).cast(), a), (self.conv_at(0).cast(), b)]
    }
    /// 状態を外で戻した後に位置だけ置く（KV の 0..pos と GDN の状態は pos の物が入っていること）
    pub fn set_pos(&mut self, pos: usize) {
        assert!(pos <= self.max_ctx);
        self.pos = pos;
    }
    /// 列の頭に戻す（GDN の状態と conv の状態を 0 に。KV は位置で上書きされるのでそのまま）
    pub fn reset(&mut self) {
        self.zero_state();
        self.pos = 0;
        self.media.clear();
    }
    /// 位置 i の RoPE の (t, h, w, e)
    pub fn rope_pos4(&self, i: usize) -> [i32; 4] {
        let mut delta = 0i64;
        for sp in &self.media {
            if i < sp.start {
                break;
            }
            if i < sp.start + sp.n() {
                let k = i - sp.start;
                let b = sp.start as i64 + delta;
                return [b as i32, (b + (k / sp.nx) as i64) as i32, (b + (k % sp.nx) as i64) as i32, 0];
            }
            delta += sp.shift();
        }
        let p = (i as i64 + delta) as i32;
        [p, p, p, p]
    }
    /// 1 つの数にした RoPE の位置（デコード・確かめ・MTP。画像の語は t）
    pub fn rope_pos(&self, i: usize) -> i32 {
        self.rope_pos4(i)[0]
    }
    /// この列の分の置き場（バイト）
    pub fn bytes(&self) -> usize {
        let p = &self.pool;
        (p.state.len + p.conv.len + p.kc.len + p.vc.len + if p.kv8 { p.ks.len + p.vs.len } else { 0 }) / p.n_slots
    }
}

// ---------------------------------------------------------------- 途中の値を残す（layers の答え合わせ用）

/// forward の途中の値を手元に写す。h は層ごとの残差（ref_qwen35.py の h = llama.cpp の l_out）
#[derive(Default)]
pub struct Dump {
    pub emb: Vec<f32>,
    /// h[i] = 層 layer0 + i の出口の残差（行 × d を語の順に）
    pub h: Vec<Vec<f32>>,
    /// 途中の値も残す層
    pub detail_layers: Vec<usize>,
    /// "名前-層" → (1 行の幅, 値)
    pub detail: HashMap<String, (usize, Vec<f32>)>,
}

impl Dump {
    fn want(&self, il: usize) -> bool {
        self.detail_layers.contains(&il)
    }
    fn put(&mut self, name: &str, il: usize, w: usize, v: Vec<f32>) {
        let e = self.detail.entry(format!("{name}-{il}")).or_insert_with(|| (w, vec![]));
        e.1.extend(v);
    }
}

// ---------------------------------------------------------------- 作業場

#[derive(Clone, Copy, Debug)]
pub struct Opt {
    /// T がこれ以下なら g3 の GEMV（B = T）、超えたら GEMM（0 なら全部 GEMM。最大 32）
    pub gemv_max: usize,
    /// プレフィルの塊の語数（作業場の大きさもこれで決まる）
    pub chunk: usize,
    /// GDN の T > 1 を 1 語ずつの道（gdn::decode を T で）で。既定は塊ごと（gdn::prefill）
    pub gdn_seq: bool,
    /// 層ごとに NaN・inf を数える（遅い。確かめ用）
    pub check_finite: bool,
    /// GEMM の道で、行列を fp16 に戻してから掛ける（M1 の道）。既定は g3 の並びのまま掛ける（kernels/gemmg3.cu・M2）
    pub deq_gemm: bool,
}

impl Default for Opt {
    fn default() -> Self {
        Opt { gemv_max: GEMV_B_MAX, chunk: 2048, gdn_seq: false, check_finite: false, deq_gemm: false }
    }
}

struct Scratch {
    tc: usize,
    h: DevBuf,      // f32 [tc][d] 残差
    xa: DevBuf,     // f32 [tc][d] norm の出口
    x16: DevBuf,    // f16 [tc][d]
    mid16: DevBuf,  // f16 [tc][d_ff]（GEMM の SwiGLU の出口 = down の X）
    o16: DevBuf,    // f16 [tc][6144]（ssm_out・attn_output の X）
    xp_d: DevBuf,   // g3 の x の下ごしらえ（k = d、B ≤ 32）
    xp_o: DevBuf,   // （k = 6144）
    xp_ff: DevBuf,  // （k = d_ff。swiglu が書く）
    qkv: DevBuf,    // f32 [tc][10240]
    z: DevBuf,      // f32 [tc][6144]
    alpha: DevBuf,  // f32 [tc][48]
    beta: DevBuf,   // f32 [tc][48]
    c: DevBuf,      // f32 [tc][10240] conv の後
    y: DevBuf,      // f32 [tc][6144] GDN・Attention の出口
    qf: DevBuf,     // f32 [tc][12288]
    kk: DevBuf,     // f32 [tc][1024]
    vv: DevBuf,     // f32 [tc][1024]
    q16: DevBuf,    // f16 [tc][24][256]
    wdeq: DevBuf,   // f16 の重みの置き場（GEMM の道。いちばん大きいのは gate＋up）
    gemm_ws: DevBuf,
    gdn_ws: DevBuf,
    attn_ws: DevBuf,
    logits: DevBuf, // f32 [32][vocab]
    ids: DevBuf,    // i32 [tc]
    pos: DevBuf,    // i32 [tc]
    pos4: DevBuf,   // i32 [tc][4]（画像のある列の RoPE の t, h, w, e）
    slot: DevBuf,   // i32 [tc]
    bt: DevBuf,     // i32 [n_pages]
    qs: DevBuf,     // i32 [2]
    kl: DevBuf,     // i32 [1]
    am: DevBuf,     // i32 [32]
    amv: DevBuf,    // f32 [32]
    cnt: DevBuf,    // i32 [1]
    bt_pages: usize,
    /// MoE の層の作業場（3.6）
    moe: Option<MoeSc>,
    /// mHC の 2 本の残差（f32 [tc][d] ずつ）と振り分け（f32 [tc][8]: post0, post1, m00, m01, m10, m11）。SenseNova-Loop だけ
    hc_a: DevBuf,
    hc_b: DevBuf,
    hc_rt: DevBuf,
}

fn moe_gemm_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TACHYON_MOE_GEMM").map_or(true, |v| v != "0"))
}

fn moe_xl_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TACHYON_MOE_XL").is_ok_and(|v| v == "1"))
}

fn moe_tokb() -> usize {
    static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *T.get_or_init(|| match std::env::var("TACHYON_MOE_TOKB").ok().and_then(|v| v.parse::<usize>().ok()) {
        Some(x) if [16, 32, 64, 128].contains(&x) => x,
        _ => 64,
    })
}

/// MoE の作業場（tc 語まで。行 R = tc·k）
struct MoeSc {
    rl: DevBuf,     // f32 [tc][n_exp + 1] ルーターの出口（最後の列が共有のゲート）
    ids: DevBuf,    // i32 [R] 選んだ専門家（= デコードの形の仕事の専門家）
    wts: DevBuf,    // f32 [R]
    jr0: DevBuf,    // i32 [R] 仕事の行の頭
    jcnt: DevBuf,   // i32 [R] 仕事の行の数
    xidx: DevBuf,   // i32 [R] 行 → 語
    je: DevBuf,     // i32 [R] 仕事の専門家（専門家ごとにまとめる道）
    rowmap: DevBuf, // i32 [R] (語, 枠) → 行（専門家ごとにまとめる道）
    yg: DevBuf,     // f32 [R][ff]
    yu: DevBuf,     // f32 [R][ff]
    hh: DevBuf,     // f32 [R][ff]
    hp: DevBuf,     // down の x の下ごしらえ（k = ff、R 行）
    o: DevBuf,      // f32 [R][d] 専門家ごとの出口
    sh: DevBuf,     // f32 [tc][d] 共有の専門家の出口
    xpt: DevBuf,    // x の下ごしらえ（k = d、tc 行。プレフィルの道）
    grp: DevBuf,    // i32 [3][仕事] 専門家ごとの GEMM の仕事（専門家, 行の頭, 行の数）
    xlo: DevBuf,    // f16 [tc][d] x の下（X = x16 + xlo）
}

impl MoeSc {
    fn new(cfg: &Config, mc: &MoeCfg, tc: usize) -> MoeSc {
        let (d, r) = (cfg.d, tc * mc.k);
        let xpb = |k: usize, b: usize| unsafe { ffi::infer_g3_xprep_bytes(k as i32, b as i32) } as usize;
        MoeSc {
            rl: DevBuf::new(tc * (mc.n_exp + 1) * 4),
            ids: DevBuf::new(r * 4),
            wts: DevBuf::new(r * 4),
            jr0: DevBuf::new(r * 4),
            jcnt: DevBuf::new(r * 4),
            xidx: DevBuf::new(r * 4),
            je: DevBuf::new(r * 4),
            rowmap: DevBuf::new(r * 4),
            yg: DevBuf::new(r * mc.ff * 4),
            yu: DevBuf::new(r * mc.ff * 4),
            hh: DevBuf::new(r * mc.ff * 4),
            hp: DevBuf::new(xpb(mc.ff, r)),
            o: DevBuf::new(r * d * 4),
            sh: DevBuf::new(tc * d * 4),
            xpt: DevBuf::new(xpb(d, tc)),
            grp: DevBuf::new(3 * (mc.n_exp + r) * 4),
            xlo: DevBuf::new(tc * d * 2),
        }
    }
    fn bytes(&self) -> usize {
        [&self.rl, &self.ids, &self.wts, &self.jr0, &self.jcnt, &self.xidx, &self.je, &self.rowmap, &self.yg, &self.yu, &self.hh, &self.hp, &self.o,
         &self.sh, &self.xpt, &self.grp, &self.xlo]
            .iter()
            .map(|b| b.len)
            .sum()
    }
}

impl Scratch {
    fn new(cfg: &Config, tc: usize) -> Scratch {
        Self::new_ex(cfg, tc, false)
    }
    /// light: デコード・確かめだけの作業場（GEMM の道の置き場を取らない。M3 の別の流れ用）
    fn new_ex(cfg: &Config, tc: usize, light: bool) -> Scratch {
        let (d, ff, ao) = (cfg.d, cfg.d_ff, cfg.attn_out());
        let cc = cfg.gdn.conv_ch();
        let hv = cfg.gdn.hv;
        let xpb = |k: usize| unsafe { ffi::infer_g3_xprep_bytes(k as i32, GEMV_B_MAX as i32) } as usize;
        let wdeq = [2 * ff * d, (cc + ao + 2 * hv) * d, (cfg.q_full() + 2 * cfg.kv_dim()) * d, ff * d, ao * d]
            .into_iter()
            .max()
            .unwrap()
            * 2
            + 4 * 256;
        // split-K の作業場: 残差に足す GEMM（k = 6144・17408 → n = d）の M ごとの最大（足りなければ GEMM は分けないだけ）
        let gws = [(ao, d), (ff, d)]
            .iter()
            .flat_map(|&(k, n)| (1..=tc).map(move |m| unsafe { ffi::infer_gemm_ws_bytes(k as i32, n as i32, m as i32) }))
            .max()
            .unwrap_or(0)
            .max(0) as usize;
        let gdn_ws = gdn::prefill_ws_bytes(&cfg.gdn, 1, tc);
        let (wdeq, gws, gdn_ws) = if light { (16, 0, 16) } else { (wdeq, gws, gdn_ws) };
        Scratch {
            tc,
            h: DevBuf::new(tc * d * 4),
            xa: DevBuf::new(tc * d * 4),
            x16: DevBuf::new(tc * d * 2),
            mid16: DevBuf::new(tc * ff * 2),
            o16: DevBuf::new(tc * ao * 2),
            xp_d: DevBuf::new(xpb(d)),
            xp_o: DevBuf::new(xpb(ao)),
            xp_ff: DevBuf::new(xpb(ff)),
            qkv: DevBuf::new(tc * cc * 4),
            z: DevBuf::new(tc * ao * 4),
            alpha: DevBuf::new(tc * hv * 4),
            beta: DevBuf::new(tc * hv * 4),
            c: DevBuf::new(tc * cc * 4),
            y: DevBuf::new(tc * ao * 4),
            qf: DevBuf::new(tc * cfg.q_full() * 4),
            kk: DevBuf::new(tc * cfg.kv_dim() * 4),
            vv: DevBuf::new(tc * cfg.kv_dim() * 4),
            q16: DevBuf::new(tc * ao * 2),
            wdeq: DevBuf::new(wdeq),
            gemm_ws: DevBuf::new(gws),
            gdn_ws: DevBuf::new(gdn_ws),
            attn_ws: DevBuf::new(64 << 20),
            logits: DevBuf::new(GEMV_B_MAX * cfg.vocab * 4),
            ids: DevBuf::new(tc * 4),
            pos: DevBuf::new(tc * 4),
            pos4: DevBuf::new(tc * 16),
            slot: DevBuf::new(tc * 4),
            bt: DevBuf::new(4),
            qs: DevBuf::new(8),
            kl: DevBuf::new(4),
            am: DevBuf::new(GEMV_B_MAX * 4),
            amv: DevBuf::new(GEMV_B_MAX * 4),
            cnt: DevBuf::new(4),
            bt_pages: 0,
            moe: cfg.moe.as_ref().map(|mc| MoeSc::new(cfg, mc, tc)),
            hc_a: DevBuf::new(if cfg.lp.is_some() { tc * d * 4 } else { 16 }),
            hc_b: DevBuf::new(if cfg.lp.is_some() { tc * d * 4 } else { 16 }),
            hc_rt: DevBuf::new(if cfg.lp.is_some() { tc * 8 * 4 } else { 16 }),
        }
    }
    fn bytes(&self) -> usize {
        [
            &self.h, &self.xa, &self.x16, &self.mid16, &self.o16, &self.xp_d, &self.xp_o, &self.xp_ff, &self.qkv, &self.z,
            &self.alpha, &self.beta, &self.c, &self.y, &self.qf, &self.kk, &self.vv, &self.q16, &self.wdeq, &self.gemm_ws,
            &self.gdn_ws, &self.attn_ws, &self.logits, &self.hc_a, &self.hc_b, &self.hc_rt,
        ]
        .iter()
        .map(|b| b.len)
        .sum::<usize>()
            + self.moe.as_ref().map_or(0, |m| m.bytes())
    }
}

/// 確かめ用: TACHYON_ATTN_CHUNK=<幅>（0 = 分けない）で forward の Attention の split の幅を決め打ち。
/// TACHYON_ATTN_CHUNK_SMALL は 32 語以下の forward（デコード・確かめ）だけ
fn attn_chunk_env(t: usize) -> Option<i32> {
    let g = |k: &str| std::env::var(k).ok().and_then(|v| v.parse().ok());
    if t <= GEMV_B_MAX {
        if let Some(c) = g("TACHYON_ATTN_CHUNK_SMALL") {
            return Some(c);
        }
    }
    g("TACHYON_ATTN_CHUNK")
}

fn upload_i32(b: &DevBuf, v: &[i32]) {
    assert!(v.len() * 4 <= b.len);
    h2d(b.ptr, unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), v.len() * 4) });
}

// ---------------------------------------------------------------- モデル

/// 読み込みの記録
#[derive(Clone, Copy, Debug, Default)]
pub struct LoadInfo {
    pub seconds: f64,
    pub bytes_gguf: usize,
    pub bytes_gpu: usize,
    pub scratch_bytes: usize,
}

pub struct Model {
    pub cfg: Config,
    pub opt: Opt,
    /// 載せた最初の層（ふつう 0）。layers[i] は層 layer0 + i
    pub layer0: usize,
    /// 実行の順の層（SenseNova-Loop の繰り返しは同じ重みを Rc で指す）
    pub layers: Vec<std::rc::Rc<LayerW>>,
    embd: DevBuf,
    embd_qt: QType,
    out_norm: DevBuf,
    output: Option<Mat>,
    sc: Scratch,
    /// プレフィル（forward）の流れ（既定は NULL の流れ）。M3 では非同期の低い優先度の流れ
    st: ffi::Stream,
    /// デコード・確かめの作業場（M3: プレフィルと同時に走らせるとき。None なら sc を使う）
    sd: Option<Scratch>,
    pub info: LoadInfo,
    /// デコード（M2: B 本まとめて・CUDA Graph）の置き場。最初の decode で作る（src/model/decode.rs）
    dec: Option<decode::Dec>,
    /// 下書きの確かめ（M2: MTP）の置き場。enable_spec で作る（src/model/spec_target.rs）
    spec: Option<spec_target::SpecBufs>,
    /// 今の forward の列に画像がある（RoPE を sc.pos4 の (t, h, w, e) で）
    pos4_on: bool,
    /// 抜き取りの候補（GPU の上位 K。kernels/sample.cu）の置き場
    cand: CandBufs,
}

/// 抜き取りの候補の置き場（32 行・上位 CAND_K まで）
struct CandBufs {
    ws: DevBuf,
    ids: DevBuf,
    vals: DevBuf,
    mz: DevBuf,
    temps: DevBuf,
}

/// GPU で取る抜き取りの候補の数（sampler::prepare_cands。top_k はこれより小さい時だけ GPU の道）
pub const CAND_K: usize = 256;

/// 1 行の候補: 上位 K 語（大きい順）と、行の最大・z = Σ exp((x − 最大)/温度)
pub struct RowCands {
    pub ids: Vec<u32>,
    pub vals: Vec<f32>,
    pub m: f32,
    pub z: f64,
}

/// 行列の掛け方（1 回の forward で全部の行列に同じ）
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Path {
    Gemv,
    Gemm,
}

impl Model {
    /// GGUF を GPU に載せる。n_layers が Some(n) なら先頭 n 層だけ（出力の層は載せない。層の答え合わせ用）
    /// layers が Some(a..b) なら層 a..b だけ（層の答え合わせ用。埋め込みは a = 0 のときだけ、出力の層は b = 64 のときだけ載せる。
    /// a > 0 なら forward に入口の残差 h を渡す）
    pub fn load(g: &gguf::Model, opt: Opt, layers: Option<std::ops::Range<usize>>) -> Result<Model, String> {
        let t0 = std::time::Instant::now();
        let cfg = Config::from_gguf(g)?;
        let range = layers.unwrap_or(0..cfg.n_layer);
        let range = range.start.min(cfg.n_layer)..range.end.min(cfg.n_layer);
        if range.is_empty() {
            return Err(format!("層の範囲が空: {range:?}"));
        }
        let full = range.end == cfg.n_layer;
        let mut ld = Loader { stage: DevBuf::new(64 << 20), bytes_gguf: 0, bytes_gpu: 0 };
        let et = g.tensor("token_embd.weight");
        let embd_qt = et.qt();
        let embd = if range.start == 0 { DevBuf::from_bytes(et.bytes) } else { DevBuf::new(16) };
        if range.start == 0 {
            ld.bytes_gguf += et.bytes.len();
            ld.bytes_gpu += embd.len;
        }
        let mut layers: Vec<std::rc::Rc<LayerW>> = Vec::with_capacity(range.len());
        // SenseNova-Loop: 繰り返しの実行は最初の回の重みを指す（GGUF の blk も最初の回の番号にしか無い）
        let mut by_blk: HashMap<usize, std::rc::Rc<LayerW>> = HashMap::new();
        for il in range.clone() {
            let blk = cfg.lp.map_or(il, |lp| lp.src_blk(il));
            if let Some(w) = by_blk.get(&blk) {
                layers.push(w.clone());
                continue;
            }
            let l = g.layer(blk);
            let mix = if cfg.is_attn(il) {
                Mix::Attn(AttnW {
                    q: ld.mat(l.tensor("attn_q.weight")),
                    k: ld.mat(l.tensor("attn_k.weight")),
                    v: ld.mat(l.tensor("attn_v.weight")),
                    o: ld.mat(l.tensor("attn_output.weight")),
                    q_norm: ld.f32v(l.tensor("attn_q_norm.weight")),
                    k_norm: ld.f32v(l.tensor("attn_k_norm.weight")),
                })
            } else {
                // SenseNova-Loop の GGUF は qkv と gate（z）、β と α を 1 本に繋いである（llama.cpp の attn_qkv_gate・ssm_beta_alpha。前が qkv・β）
                let (qkv, gate) = match l.get("attn_qkv_gate.weight") {
                    Some(t) => {
                        let (cc, ao) = (cfg.gdn.conv_ch(), cfg.gdn.out_dim());
                        assert_eq!(t.n_rows(), cc + ao, "attn_qkv_gate の行の数");
                        (ld.mat_rows(t, 0, cc), ld.mat_rows(t, cc, ao))
                    }
                    None => (ld.mat(l.tensor("attn_qkv.weight")), ld.mat(l.tensor("attn_gate.weight"))),
                };
                let (alpha, beta) = match l.get("ssm_beta_alpha.weight") {
                    Some(t) => {
                        let hv = cfg.gdn.hv;
                        assert_eq!(t.n_rows(), 2 * hv, "ssm_beta_alpha の行の数");
                        (ld.mat_rows(t, hv, hv), ld.mat_rows(t, 0, hv))
                    }
                    None => (ld.mat(l.tensor("ssm_alpha.weight")), ld.mat(l.tensor("ssm_beta.weight"))),
                };
                Mix::Gdn(GdnW {
                    qkv,
                    gate,
                    alpha,
                    beta,
                    out: ld.mat(l.tensor("ssm_out.weight")),
                    a: ld.f32v(l.tensor("ssm_a")),
                    dt: ld.f32v(l.tensor("ssm_dt.bias")),
                    norm: ld.f32v(l.tensor("ssm_norm.weight")),
                    conv: ld.f32v(l.tensor("ssm_conv1d.weight")),
                })
            };
            let (gate, up, down, moe) = match cfg.moe {
                None => (ld.mat(l.tensor("ffn_gate.weight")), ld.mat(l.tensor("ffn_up.weight")), ld.mat(l.tensor("ffn_down.weight")), None),
                Some(mc) => {
                    // ルーター [n_exp][d] と共有のゲート [d] を 1 つの f32 [n_exp + 1][d] に
                    let r = l.tensor("ffn_gate_inp.weight");
                    let rs = l.tensor("ffn_gate_inp_shexp.weight");
                    assert_eq!(r.qtype, Some(QType::F32), "ffn_gate_inp は f32 のはず");
                    assert_eq!(rs.qtype, Some(QType::F32), "ffn_gate_inp_shexp は f32 のはず");
                    assert_eq!((r.k(), r.n_rows(), rs.n_elems()), (cfg.d, mc.n_exp, cfg.d), "ルーターの形");
                    let mut rb = Vec::with_capacity(r.bytes.len() + rs.bytes.len());
                    rb.extend_from_slice(r.bytes);
                    rb.extend_from_slice(rs.bytes);
                    ld.bytes_gguf += rb.len();
                    ld.bytes_gpu += rb.len();
                    let (ge, ue, de) = (l.tensor("ffn_gate_exps.weight"), l.tensor("ffn_up_exps.weight"), l.tensor("ffn_down_exps.weight"));
                    assert_eq!((ge.k(), ge.n_rows()), (cfg.d, mc.n_exp * mc.ff), "ffn_gate_exps の形");
                    assert_eq!((ue.k(), ue.n_rows()), (cfg.d, mc.n_exp * mc.ff), "ffn_up_exps の形");
                    assert_eq!((de.k(), de.n_rows()), (mc.ff, mc.n_exp * cfg.d), "ffn_down_exps の形");
                    // 専門家を続けて並べた [n_exp·n][k] を 1 本の行列として詰め直す（n は 32 の倍数なので専門家の境はタイルの境）
                    let (g, u, dn) = (ld.mat(ge), ld.mat(ue), ld.mat(de));
                    assert_eq!(g.qt, u.qt, "gate と up の型");
                    let es_gu = unsafe { ffi::infer_g3_bytes(g.src as i32, cfg.d as i32, mc.ff as i32) };
                    let es_d = unsafe { ffi::infer_g3_bytes(dn.src as i32, mc.ff as i32, cfg.d as i32) };
                    assert!(es_gu > 0 && es_d > 0 && es_gu as usize * mc.n_exp == g.buf.len && es_d as usize * mc.n_exp == dn.buf.len);
                    let moe = MoeW { router: DevBuf::from_bytes(&rb), gate: g, up: u, down: dn, es_gu, es_d };
                    (ld.mat(l.tensor("ffn_gate_shexp.weight")), ld.mat(l.tensor("ffn_up_shexp.weight")), ld.mat(l.tensor("ffn_down_shexp.weight")),
                     Some(moe))
                }
            };
            let hc = match cfg.lp {
                Some(lp) if lp.is_mhc(il) => {
                    let mut hv = |n: &str, len: usize| {
                        let t = l.tensor(n);
                        assert_eq!(t.n_elems(), len, "{n} の大きさ");
                        ld.f32v(t)
                    };
                    let d2 = 2 * cfg.d;
                    Some(HcW {
                        attn_w: hv("hc_attn_fn.weight", 8 * d2),
                        attn_b: hv("hc_attn_fn.bias", 8),
                        ffn_w: hv("hc_ffn_fn.weight", 8 * d2),
                        ffn_b: hv("hc_ffn_fn.bias", 8),
                    })
                }
                _ => None,
            };
            let w = std::rc::Rc::new(LayerW {
                attn_norm: ld.f32v(l.tensor("attn_norm.weight")),
                post_norm: ld.f32v(l.tensor("post_attention_norm.weight")),
                mix,
                gate,
                up,
                down,
                moe,
                hc,
            });
            by_blk.insert(blk, w.clone());
            layers.push(w);
        }
        if let Some(lp) = cfg.lp {
            // 途中から載せるとき、mHC の 2 本の残差は入口の h からは作れない（mHC の最初の層か、その外から始める）
            assert!(range.start == 0 || !lp.is_mhc(range.start) || lp.mhc_first(range.start), "mHC の途中の層 {} からは載せられない", range.start);
        }
        let out_norm = ld.f32v(g.tensor("output_norm.weight"));
        let output = if full { Some(ld.mat(g.tensor("output.weight"))) } else { None };
        crate::cuda::sync();
        drop(ld.stage);
        let mut opt = opt;
        opt.gemv_max = opt.gemv_max.min(GEMV_B_MAX);
        opt.chunk = opt.chunk.max(1);
        let sc = Scratch::new(&cfg, opt.chunk.max(GEMV_B_MAX));
        crate::cuda::sync();
        let info = LoadInfo {
            seconds: t0.elapsed().as_secs_f64(),
            bytes_gguf: ld.bytes_gguf,
            bytes_gpu: ld.bytes_gpu,
            scratch_bytes: sc.bytes(),
        };
        let cw = unsafe { kffi::infer_cand_ws_bytes(GEMV_B_MAX as i32, cfg.vocab as i32, CAND_K as i32) }.max(16) as usize;
        let cand = CandBufs {
            ws: DevBuf::new(cw),
            ids: DevBuf::new(GEMV_B_MAX * CAND_K * 4),
            vals: DevBuf::new(GEMV_B_MAX * CAND_K * 4),
            mz: DevBuf::new(GEMV_B_MAX * 16),
            temps: DevBuf::new(GEMV_B_MAX * 4),
        };
        Ok(Model { cfg, opt, layer0: range.start, layers, embd, embd_qt, out_norm, output, sc, st: S, sd: None, info, dec: None, spec: None, pos4_on: false, cand })
    }

    pub fn n_loaded(&self) -> usize {
        self.layers.len()
    }
    pub fn has_head(&self) -> bool {
        self.output.is_some()
    }
    pub fn new_seq(&self, max_ctx: usize) -> Seq {
        Seq::new(&self.cfg, self.layer0..self.layer0 + self.layers.len(), max_ctx)
    }
    /// n 本の列（同じ置き場。decode で何本でもまとめられる）
    pub fn new_seqs(&self, n: usize, max_ctx: usize) -> Vec<Seq> {
        Seq::new_n(&self.cfg, self.layer0..self.layer0 + self.layers.len(), max_ctx, n)
    }

    fn finite(&self, p: *const f32, n: usize, what: &str) {
        if !self.opt.check_finite {
            return;
        }
        zero(self.sc.cnt.ptr, 4);
        check(unsafe { kffi::infer_m_nonfinite(p, n as i64, self.sc.cnt.f32p().cast(), self.st) }, "nonfinite");
        let mut b = [0u8; 4];
        d2h(&mut b, self.sc.cnt.ptr);
        let c = i32::from_le_bytes(b);
        assert!(c == 0, "{what}: 有限でない値が {c} 個");
    }

    /// norm: h の t 行 → xa（f32）、GEMM の道なら x16（f16）にも
    fn norm(&self, w: &DevBuf, t: usize, path: Path) {
        let d = self.cfg.d;
        let y16 = if path == Path::Gemm { self.sc.x16.ptr } else { null_mut() };
        check(
            unsafe {
                kffi::infer_m_rmsnorm(self.sc.h.f32p(), d as i64, w.f32p(), d as i32, t as i32, self.cfg.eps, self.sc.xa.f32p(), d as i64,
                                      y16, d as i64, self.st)
            },
            "rmsnorm",
        );
    }

    /// g3 の x の下ごしらえ（f32 [t][k] → xp）
    fn xprep(&self, x: *const f32, k: usize, t: usize, xp: &DevBuf) {
        check(unsafe { ffi::infer_g3_xprep(x, k as i32, t as i32, xp.ptr, self.st) }, "g3 xprep");
    }

    /// 同じ x の行列（4 つまで）: ys[i] [t][n_i] = W_i x（f32）。GEMV なら xp、GEMM なら x16 を読む
    fn mm_multi(&self, mats: &[&Mat], ys: &[*mut f32], t: usize, path: Path, xp: &DevBuf, x16: *const c_void) {
        let k = mats[0].k;
        let ts: Vec<i32> = mats.iter().map(|m| m.qt).collect();
        let ns: Vec<i32> = mats.iter().map(|m| m.n as i32).collect();
        match path {
            Path::Gemv => {
                let ws: Vec<*const c_void> = mats.iter().map(|m| m.buf.ptr as *const c_void).collect();
                check(
                    unsafe {
                        ffi::infer_g3_gemv_multi(mats.len() as i32, ts.as_ptr(), ws.as_ptr(), ns.as_ptr(), ys.as_ptr(), k as i32, xp.ptr,
                                                 t as i32, 0, 0, self.st)
                    },
                    "g3 gemv multi",
                );
            }
            Path::Gemm if !self.opt.deq_gemm => {
                let ws: Vec<*const c_void> = mats.iter().map(|m| m.buf.ptr as *const c_void).collect();
                let yv: Vec<*mut c_void> = ys.iter().map(|p| p.cast()).collect();
                check(
                    unsafe {
                        kffi::infer_g3_gemm_ex(mats.len() as i32, ts.as_ptr(), ws.as_ptr(), ns.as_ptr(), yv.as_ptr(), null(), k as i32, t as i32,
                                               x16, null(), k as i32, 0, -1, self.st)
                    },
                    "g3 gemm multi",
                );
            }
            Path::Gemm => {
                let ws = self.deq16(mats);
                let t16 = vec![Q_F16; mats.len()];
                let yv: Vec<*mut c_void> = ys.iter().map(|p| p.cast()).collect();
                check(
                    unsafe {
                        ffi::infer_gemm_multi(mats.len() as i32, t16.as_ptr(), ws.as_ptr(), ns.as_ptr(), yv.as_ptr(), k as i32, t as i32,
                                              x16, 0, -1, self.st)
                    },
                    "gemm multi",
                );
            }
        }
    }

    /// 行列を fp16 [n][k] に戻して作業場に並べる（16 バイト揃え）。返り値はそれぞれの頭
    fn deq16(&self, mats: &[&Mat]) -> Vec<*const c_void> {
        let mut off = 0usize;
        let mut ps = vec![];
        for m in mats {
            let bytes = m.n * m.k * 2;
            assert!(off + bytes <= self.sc.wdeq.len, "fp16 の置き場が足りない");
            let p: *mut c_void = at(&self.sc.wdeq, off);
            check(unsafe { kffi::infer_m_g3_deq16(m.qt, m.buf.ptr, m.k as i32, m.n as i32, p, self.st) }, "g3 deq16");
            ps.push(p as *const c_void);
            off += (bytes + 255) / 256 * 256;
        }
        ps
    }

    /// 残差に足す: h [t][n] += W x。GEMV は x を f32 [t][k]（xp に下ごしらえ）、GEMM は x16（f16 [t][k]）
    fn mm_add(&self, m: &Mat, t: usize, path: Path, x: *const f32, xp: &DevBuf, x16: *const c_void) {
        assert_eq!(m.n, self.cfg.d, "残差に足す行列の出口は d のはず");
        match path {
            Path::Gemv => {
                self.xprep(x, m.k, t, xp);
                self.gemv_add(m, xp, t);
            }
            Path::Gemm if !self.opt.deq_gemm => {
                let (ws, ns, yv) = ([m.buf.ptr as *const c_void], [m.n as i32], [self.sc.h.ptr]);
                check(
                    unsafe {
                        kffi::infer_g3_gemm_ex(1, &m.qt, ws.as_ptr(), ns.as_ptr(), yv.as_ptr(), null(), m.k as i32, t as i32, x16, null(), m.k as i32,
                                               2, -1, self.st)
                    },
                    "g3 gemm add",
                );
            }
            Path::Gemm => {
                let w = self.deq16(&[m]);
                check(
                    unsafe {
                        ffi::infer_gemm(Q_F16, w[0], m.k as i32, m.n as i32, t as i32, x16, self.sc.h.ptr, 2, self.sc.gemm_ws.ptr,
                                        self.sc.gemm_ws.len as i64, self.st)
                    },
                    "gemm add",
                );
            }
        }
    }

    fn gemv_add(&self, m: &Mat, xp: &DevBuf, t: usize) {
        assert_eq!(m.n, self.cfg.d, "残差に足す行列の出口は d のはず");
        // wk = 0（形から選ぶ）: 足し込みのときは wk = 1 のまま（wk ≥ 2 と add = 1 の組は使わない）
        check(
            unsafe { ffi::infer_g3_gemv(m.qt, m.buf.ptr, m.k as i32, m.n as i32, xp.ptr, self.sc.h.f32p(), t as i32, 0, 1, self.st) },
            "g3 gemv add",
        );
    }

    /// FFN: h += down(silu(gate b) * up b)、b = post_norm(h)（xa・x16 に入っている）
    fn ffn(&self, l: &LayerW, t: usize, path: Path) {
        let (d, ff) = (self.cfg.d, self.cfg.d_ff);
        if l.moe.is_some() {
            if path == Path::Gemv {
                self.xprep(self.sc.xa.f32p(), d, t, &self.sc.xp_d);
            }
            self.moe_ffn(l, &self.sc, t, path == Path::Gemm, self.st);
            return;
        }
        match path {
            Path::Gemv => {
                self.xprep(self.sc.xa.f32p(), d, t, &self.sc.xp_d);
                check(
                    unsafe {
                        ffi::infer_g3_swiglu(l.gate.qt, l.gate.buf.ptr, l.up.qt, l.up.buf.ptr, d as i32, ff as i32, self.sc.xp_d.ptr,
                                             null_mut(), self.sc.xp_ff.ptr, t as i32, self.st)
                    },
                    "g3 swiglu",
                );
                self.gemv_add(&l.down, &self.sc.xp_ff, t);
            }
            Path::Gemm if !self.opt.deq_gemm => {
                let ts = [l.gate.qt, l.up.qt];
                let ws = [l.gate.buf.ptr as *const c_void, l.up.buf.ptr as *const c_void];
                let ns = [ff as i32, ff as i32];
                let yv = [self.sc.mid16.ptr, self.sc.mid16.ptr];
                check(
                    unsafe {
                        kffi::infer_g3_gemm_ex(2, ts.as_ptr(), ws.as_ptr(), ns.as_ptr(), yv.as_ptr(), null(), d as i32, t as i32, self.sc.x16.ptr,
                                               null(), d as i32, 3, -1, self.st)
                    },
                    "g3 gemm swiglu",
                );
                self.mm_add(&l.down, t, path, null(), &self.sc.xp_ff, self.sc.mid16.ptr);
            }
            Path::Gemm => {
                let w = self.deq16(&[&l.gate, &l.up]);
                check(
                    unsafe {
                        ffi::infer_gemm_swiglu(Q_F16, w[0], Q_F16, w[1], d as i32, ff as i32, t as i32, self.sc.x16.ptr, self.sc.mid16.ptr,
                                               -1, self.st)
                    },
                    "gemm swiglu",
                );
                self.mm_add(&l.down, t, path, null(), &self.sc.xp_ff, self.sc.mid16.ptr);
            }
        }
    }

    /// MoE の FFN（3.6）: h += Σ_j w_j·down_e(SiLU(gate_e b)·up_e b) + sigmoid(g·b)·shared(b)、b = post_norm(h)。
    /// 入口: sc.xa（f32 [t][d]）と、grouped でなければ sc.xp_d（b の下ごしらえ、t ≤ 32 行）、grouped なら sc.x16（f16。共有の専門家の GEMM）。
    /// grouped = false: 語 × 枠ごとに 1 行の仕事（GPU の上で決まる・CUDA Graph に取れる）。
    /// grouped = true（プレフィル）: 選ばれた専門家を手元に写し、専門家ごとに語を 32 本ずつまとめて掛ける（流れ s を 1 回待つ）
    pub(crate) fn moe_ffn(&self, l: &LayerW, sc: &Scratch, t: usize, grouped: bool, s: ffi::Stream) {
        let cfg = self.cfg;
        let mc = cfg.moe.expect("MoE の形");
        let w = l.moe.as_ref().unwrap();
        let ms = sc.moe.as_ref().expect("MoE の作業場");
        let (d, ff, ne, kk, sff) = (cfg.d, mc.ff, mc.n_exp, mc.k, cfg.d_ff);
        let r = t * kk;
        assert!(r * 4 <= ms.ids.len, "MoE の作業場（{t} 語）が足りない");
        // プレフィルの専門家の掛け算を GEMM で（2026-11-14。TACHYON_MOE_GEMM=0 で前の GEMV の道。語の数のまとまり TACHYON_MOE_TOKB）
        let use_gemm = grouped && moe_gemm_on();
        let tokb = moe_tokb();
        // ルーター（＋共有のゲート）→ 上位 k
        check(
            unsafe { kffi::infer_moe_router(w.router.f32p(), d as i32, (ne + 1) as i32, sc.xa.f32p(), d as i64, t as i32, ms.rl.f32p(), s) },
            "moe router",
        );
        check(
            unsafe {
                kffi::infer_moe_topk(ms.rl.f32p(), (ne + 1) as i64, ne as i32, kk as i32, t as i32, ms.ids.ptr.cast(), ms.wts.f32p(),
                                     ms.jr0.ptr.cast(), ms.jcnt.ptr.cast(), ms.xidx.ptr.cast(), s)
            },
            "moe topk",
        );
        // 共有の専門家 → ms.sh [t][d]
        if !grouped {
            check(
                unsafe {
                    ffi::infer_g3_swiglu(l.gate.qt, l.gate.buf.ptr, l.up.qt, l.up.buf.ptr, d as i32, sff as i32, sc.xp_d.ptr, null_mut(), sc.xp_ff.ptr,
                                         t as i32, s)
                },
                "g3 swiglu（共有）",
            );
            check(
                unsafe { ffi::infer_g3_gemv(l.down.qt, l.down.buf.ptr, sff as i32, d as i32, sc.xp_ff.ptr, ms.sh.f32p(), t as i32, 0, 0, s) },
                "g3 gemv（共有の down）",
            );
        } else {
            let ts = [l.gate.qt, l.up.qt];
            let ws = [l.gate.buf.ptr as *const c_void, l.up.buf.ptr as *const c_void];
            let ns = [sff as i32, sff as i32];
            let yv = [sc.mid16.ptr, sc.mid16.ptr];
            check(
                unsafe {
                    kffi::infer_g3_gemm_ex(2, ts.as_ptr(), ws.as_ptr(), ns.as_ptr(), yv.as_ptr(), null(), d as i32, t as i32, sc.x16.ptr, null(),
                                           d as i32, 3, -1, s)
                },
                "g3 gemm swiglu（共有）",
            );
            let (wd, nd, yd) = ([l.down.buf.ptr as *const c_void], [d as i32], [ms.sh.ptr]);
            check(
                unsafe {
                    kffi::infer_g3_gemm_ex(1, &l.down.qt, wd.as_ptr(), nd.as_ptr(), yd.as_ptr(), null(), sff as i32, t as i32, sc.mid16.ptr, null(),
                                           sff as i32, 0, -1, s)
                },
                "g3 gemm（共有の down）",
            );
        }
        if use_gemm {
            // 専門家ごとの GEMM（Tensor Core・mma.m8n8k4）: gate・up を SwiGLU で 1 回（f16 [R][ff] を yg に）→ down（f32 [R][d] を o に）
            // 並べは GPU の上で（手元に戻らない・流れを待たない）
            let n_blk = (ne + r / tokb) as i32;
            check(
                unsafe {
                    kffi::infer_moe_group(ms.ids.ptr.cast(), r as i32, kk as i32, ne as i32, tokb as i32, ms.xidx.ptr.cast(), ms.rowmap.ptr.cast(),
                                          ms.grp.ptr.cast(), n_blk, s)
                },
                "moe group",
            );
            let (gq, dq) = ([w.gate.qt, w.up.qt], [w.down.qt]);
            let gw = [w.gate.buf.ptr as *const c_void, w.up.buf.ptr as *const c_void];
            let (ges, des) = ([w.es_gu, w.es_gu], [w.es_d]);
            let (gn, dn) = ([ff as i32, ff as i32], [d as i32]);
            // 精度: 既定は x・h を fp16 1 つ（密な層の GEMM と同じ）。TACHYON_MOE_XL=1 で上・下の 2 つ（前の GEMV の道と同じ精度・約 1 割遅い）。
            // f32 の手本との KL は前の道と同じ（0.2510 → 0.2501）・前の道との 1 位の一致 98.4%（2026-11-14）
            let xl = moe_xl_on();
            let (h16, h16lo) = (ms.yg.ptr, ms.yu.ptr);
            let gy = if xl { [h16, h16lo] } else { [h16, h16] };
            if xl {
                check(unsafe { kffi::infer_moe_lo16(sc.xa.f32p(), sc.x16.ptr, (t * d) as i64, ms.xlo.ptr, s) }, "moe x の下");
            }
            check(
                unsafe {
                    kffi::infer_g3_moe_gemm(2, gq.as_ptr(), gw.as_ptr(), ges.as_ptr(), gn.as_ptr(), gy.as_ptr(), null(), d as i32, sc.x16.ptr,
                                            if xl { ms.xlo.ptr } else { null_mut() }, d as i32, ms.xidx.ptr.cast(), ms.grp.ptr.cast(), n_blk,
                                            tokb as i32, if xl { 4 } else { 3 }, s)
                },
                "g3 moe gemm gate・up",
            );
            let dw = [w.down.buf.ptr as *const c_void];
            let dy = [ms.o.ptr];
            check(
                unsafe {
                    kffi::infer_g3_moe_gemm(1, dq.as_ptr(), dw.as_ptr(), des.as_ptr(), dn.as_ptr(), dy.as_ptr(), null(), ff as i32, h16,
                                            if xl { h16lo } else { null_mut() }, ff as i32, null(), ms.grp.ptr.cast(), n_blk, tokb as i32, 0, s)
                },
                "g3 moe gemm down",
            );
            check(
                unsafe {
                    kffi::infer_moe_combine(sc.h.f32p(), d as i32, t as i32, kk as i32, ms.wts.f32p(), ms.o.f32p(), ms.rowmap.ptr.cast(), ms.sh.f32p(),
                                            ms.rl.f32p().add(ne), (ne + 1) as i64, s)
                },
                "moe combine",
            );
            return;
        }
        // 仕事の並び: デコードの形（行 = 語·k + 枠、x の行 = 語）か、専門家ごと（行は専門家の順、32 本ずつ）
        let (xp_x, rx, je, n_jobs, bmax, rowmap): (*const c_void, usize, *const i32, usize, usize, *const i32) = if !grouped {
            (sc.xp_d.ptr, t, ms.ids.ptr.cast(), r, 1, null())
        } else {
            self.xprep_s(sc.xa.f32p(), d, t, &ms.xpt, s);
            check(unsafe { kffi::cudaStreamSynchronize(s) }, "MoE の上位 k の待ち");
            let mut b = vec![0u8; r * 4];
            d2h(&mut b, ms.ids.ptr);
            let ids: Vec<usize> = b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as usize).collect();
            let mut per: Vec<Vec<u32>> = vec![vec![]; ne];
            for (a, &e) in ids.iter().enumerate() {
                assert!(e < ne, "専門家の番号 {e}");
                per[e].push(a as u32);
            }
            let (mut xidx, mut rowmap) = (vec![0i32; r], vec![0i32; r]);
            let (mut je, mut jr0, mut jcnt) = (vec![], vec![], vec![]);
            let mut row = 0usize;
            for (e, list) in per.iter().enumerate() {
                for ch in list.chunks(GEMV_B_MAX) {
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
            assert_eq!(row, r);
            upload_i32(&ms.xidx, &xidx);
            upload_i32(&ms.rowmap, &rowmap);
            upload_i32(&ms.je, &je);
            upload_i32(&ms.jr0, &jr0);
            upload_i32(&ms.jcnt, &jcnt);
            (ms.xpt.ptr, t, ms.je.ptr.cast(), je.len(), GEMV_B_MAX, ms.rowmap.ptr.cast())
        };
        let (gq, dq) = ([w.gate.qt, w.up.qt], [w.down.qt]);
        let gw = [w.gate.buf.ptr as *const c_void, w.up.buf.ptr as *const c_void];
        let (ges, des) = ([w.es_gu, w.es_gu], [w.es_d]);
        let (gn, dn) = ([ff as i32, ff as i32], [d as i32]);
        let gy = [ms.yg.f32p(), ms.yu.f32p()];
        check(
            unsafe {
                kffi::infer_g3_moe_gemv(2, gq.as_ptr(), gw.as_ptr(), ges.as_ptr(), gn.as_ptr(), gy.as_ptr(), d as i32, xp_x, rx as i32,
                                        ms.xidx.ptr.cast(), je, ms.jr0.ptr.cast(), ms.jcnt.ptr.cast(), n_jobs as i32, bmax as i32, s)
            },
            "g3 moe gate・up",
        );
        check(unsafe { kffi::infer_moe_swiglu(ms.yg.f32p(), ms.yu.f32p(), (r * ff) as i64, ms.hh.f32p(), s) }, "moe swiglu");
        check(unsafe { ffi::infer_g3_xprep(ms.hh.f32p(), ff as i32, r as i32, ms.hp.ptr, s) }, "moe xprep");
        let dw = [w.down.buf.ptr as *const c_void];
        let dy = [ms.o.f32p()];
        check(
            unsafe {
                kffi::infer_g3_moe_gemv(1, dq.as_ptr(), dw.as_ptr(), des.as_ptr(), dn.as_ptr(), dy.as_ptr(), ff as i32, ms.hp.ptr, r as i32, null(),
                                        je, ms.jr0.ptr.cast(), ms.jcnt.ptr.cast(), n_jobs as i32, bmax as i32, s)
            },
            "g3 moe down",
        );
        check(
            unsafe {
                kffi::infer_moe_combine(sc.h.f32p(), d as i32, t as i32, kk as i32, ms.wts.f32p(), ms.o.f32p(), rowmap, ms.sh.f32p(),
                                        ms.rl.f32p().add(ne), (ne + 1) as i64, s)
            },
            "moe combine",
        );
    }

    /// xprep を流れ s に
    fn xprep_s(&self, x: *const f32, k: usize, t: usize, xp: &DevBuf, s: ffi::Stream) {
        check(unsafe { ffi::infer_g3_xprep(x, k as i32, t as i32, xp.ptr, s) }, "g3 xprep");
    }

    /// 線形 Attention（GDN）の層: h += ssm_out(GDN(a))、a = attn_norm(h)（xa・x16）
    fn gdn_layer(&self, w: &GdnW, gi: usize, seq: &Seq, t: usize, path: Path, il: usize, dump: &mut Option<&mut Dump>) {
        let sh = self.cfg.gdn;
        let (cc, ao, hv) = (sh.conv_ch(), sh.out_dim(), sh.hv);
        let sc = &self.sc;
        if path == Path::Gemv {
            self.xprep(sc.xa.f32p(), self.cfg.d, t, &sc.xp_d);
        }
        // 下書きの確かめ（spec が動いている）: qkv・α・β・conv の後を層ごとの控えに書き、新しい状態は仮の置き場へ（commit で受け取った分だけ確定）
        let sp = self.spec.as_ref().filter(|b| b.active);
        let (y_qkv, y_a, y_b, y_c) = match sp {
            Some(b) => b.hist(gi),
            None => (sc.qkv.f32p(), sc.alpha.f32p(), sc.beta.f32p(), sc.c.f32p()),
        };
        self.mm_multi(&[&w.qkv, &w.gate, &w.alpha, &w.beta], &[y_qkv, sc.z.f32p(), y_a, y_b], t, path, &sc.xp_d, sc.x16.ptr);
        let layer = gdn::Layer { a: w.a.f32p(), dt: w.dt.f32p(), norm: w.norm.f32p(), conv: w.conv.f32p() };
        let mut io = gdn::Io::empty();
        io.qkv = y_qkv;
        io.qkv_ld = cc;
        io.c = y_c;
        io.z = sc.z.f32p();
        io.z_ld = ao;
        io.alpha = y_a;
        io.beta = y_b;
        io.ab_ld = hv;
        io.y = sc.y.f32p();
        io.y_ld = ao;
        io.state = seq.state_at(gi);
        io.conv_state = seq.conv_at(gi);
        if let Some(b) = sp {
            assert!(seq.pool.spec_state, "下書きの確かめは new_seq_spec の列で");
            assert!(t <= b.tmax, "確かめの語数 {t} が控え（{}）を超える", b.tmax);
            io.slots_out = b.slots_out_for(seq);
        }
        gdn::conv(&sh, &layer, &io, 1, t, self.st);
        if t == 1 || self.opt.gdn_seq || sp.is_some() {
            gdn::decode(&sh, &layer, &io, 1, t, self.st);
        } else {
            gdn::prefill(&sh, &layer, &io, sc.gdn_ws.ptr, 1, t, self.st);
        }
        if let Some(dp) = dump.as_deref_mut() {
            if dp.want(il) {
                dp.put("linear_attn_qkv_mixed", il, cc, d2h_f32(y_qkv, t * cc));
                dp.put("z", il, ao, d2h_f32(sc.z.f32p(), t * ao));
                dp.put("alpha", il, hv, d2h_f32(y_a, t * hv));
                dp.put("beta_raw", il, hv, d2h_f32(y_b, t * hv));
                dp.put("conv_out", il, cc, d2h_f32(y_c, t * cc));
                dp.put("final_output", il, ao, d2h_f32(sc.y.f32p(), t * ao));
            }
        }
        if path == Path::Gemm {
            check(unsafe { kffi::infer_m_f32_to_f16(sc.y.f32p(), (t * ao) as i64, sc.o16.ptr, self.st) }, "f32→f16");
        }
        self.mm_add(&w.out, t, path, sc.y.f32p(), &sc.xp_o, sc.o16.ptr);
    }

    /// 普通の Attention の層: h += attn_output(Attn(a))。位置・書く場所・頁表は forward が作業場に置いてある
    fn attn_layer(&self, w: &AttnW, ai: usize, seq: &Seq, t: usize, path: Path, il: usize, dump: &mut Option<&mut Dump>) {
        let cfg = self.cfg;
        let (qw, kv, ao) = (cfg.q_full(), cfg.kv_dim(), cfg.attn_out());
        let pos0 = seq.pos;
        let sc = &self.sc;
        if path == Path::Gemv {
            self.xprep(sc.xa.f32p(), cfg.d, t, &sc.xp_d);
        }
        self.mm_multi(&[&w.q, &w.k, &w.v], &[sc.qf.f32p(), sc.kk.f32p(), sc.vv.f32p()], t, path, &sc.xp_d, sc.x16.ptr);
        let mut r = RopeParams::new(&cfg.attn);
        r.q = sc.qf.f32p();
        r.k = sc.kk.f32p();
        r.v = sc.vv.f32p();
        r.q_norm = w.q_norm.f32p();
        r.k_norm = w.k_norm.f32p();
        r.pos = sc.pos.ptr.cast();
        if self.pos4_on {
            r.pos = sc.pos4.ptr.cast();
            r.pos_mode = 4;
        }
        r.slot = sc.slot.ptr.cast();
        r.q_out = sc.q16.ptr;
        r.kc = seq.kc_at(ai);
        r.vc = seq.vc_at(ai);
        r.ks = seq.ks_at(ai);
        r.vs = seq.vs_at(ai);
        r.kv8 = seq.kv8() as i32;
        r.n = t as i32;
        attn::rope(&r, self.st).unwrap_or_else(|e| panic!("層 {il}: {e}"));
        let mut p = AttnParams::new(&cfg.attn);
        p.q = sc.q16.ptr;
        p.gate = unsafe { sc.qf.f32p().add(attn::D) };
        p.kc = seq.kc_at(ai);
        p.vc = seq.vc_at(ai);
        p.ks = seq.ks_at(ai);
        p.vs = seq.vs_at(ai);
        p.kv8 = seq.kv8() as i32;
        p.block_table = sc.bt.ptr.cast();
        p.bt_ld = seq.n_pages as i32;
        p.q_start = sc.qs.ptr.cast();
        p.kv_len = sc.kl.ptr.cast();
        p.n_seq = 1;
        p.n_tok = t as i32;
        p.max_q_len = t as i32;
        p.max_kv = (pos0 + t) as i32;
        if let Some(c) = attn_chunk_env(t) {
            p.kv_chunk = c;
        }
        let gemm = path == Path::Gemm;
        p.out = if gemm { sc.o16.ptr } else { sc.y.ptr };
        p.out_f16 = gemm as i32;
        let wsb = attn::ws_bytes(&p);
        assert!(wsb <= sc.attn_ws.len, "Attention の作業場が足りない（{wsb}）");
        p.ws = sc.attn_ws.f32p();
        p.ws_bytes = sc.attn_ws.len as i64;
        attn::attn(&p, self.st).unwrap_or_else(|e| panic!("層 {il}: {e}"));
        if let Some(dp) = dump.as_deref_mut() {
            if dp.want(il) {
                dp.put("Qcur_full", il, qw, d2h_f32(sc.qf.f32p(), t * qw));
                dp.put("Kcur_raw", il, kv, d2h_f32(sc.kk.f32p(), t * kv));
                dp.put("Vcur", il, kv, d2h_f32(sc.vv.f32p(), t * kv));
                dp.put("Qcur", il, ao, d2h_f16(sc.q16.ptr, t * ao));
                let o = if gemm { d2h_f16(sc.o16.ptr, t * ao) } else { d2h_f32(sc.y.f32p(), t * ao) };
                dp.put("attn_gated", il, ao, o);
            }
        }
        self.mm_add(&w.o, t, path, sc.y.f32p(), &sc.xp_o, sc.o16.ptr);
    }

    /// mHC の層の頭（SenseNova-Loop）: 最初の mHC の層なら 2 本の残差 a = b = h。振り分け（W・bias）で h = 枝の入口、sc.hc_rt に post・mix
    pub(crate) fn hc_route(&self, sc: &Scratch, first: bool, w: &DevBuf, bias: &DevBuf, t: usize, s: ffi::Stream) {
        let d = self.cfg.d;
        let lp = self.cfg.lp.expect("mHC は SenseNova-Loop だけ");
        if first {
            for dst in [&sc.hc_a, &sc.hc_b] {
                check(unsafe { kffi::cudaMemcpyAsync(dst.ptr, sc.h.ptr, t * d * 4, 3, s) }, "mHC の 2 本に写す");
            }
        }
        check(
            unsafe {
                kffi::infer_mhc_route(sc.hc_a.f32p(), sc.hc_b.f32p(), d as i64, d as i32, w.f32p(), bias.f32p(), t as i32, lp.iters as i32,
                                      lp.pre_scale, sc.h.f32p(), d as i64, sc.hc_rt.f32p(), s)
            },
            "mHC route",
        );
    }

    /// 枝の計算（norm の後）の前に h を 0 に（枝の出口の行列は h に足し込むので、h = 枝の出口だけになる）
    pub(crate) fn hc_zero_h(&self, sc: &Scratch, t: usize, s: ffi::Stream) {
        check(unsafe { kffi::cudaMemsetAsync(sc.h.ptr, 0, t * self.cfg.d * 4, s) }, "mHC h を 0 に");
    }

    /// 枝の出口（h）を 2 本に合流。h = (a + b)/2（最後の mHC の層の出口・次の層が mHC でなければこれが残差）
    pub(crate) fn hc_merge(&self, sc: &Scratch, t: usize, s: ffi::Stream) {
        let d = self.cfg.d;
        check(
            unsafe {
                kffi::infer_mhc_merge(sc.hc_a.f32p(), sc.hc_b.f32p(), d as i64, d as i32, sc.h.f32p(), d as i64, sc.hc_rt.f32p(), t as i32,
                                      sc.h.f32p(), d as i64, s)
            },
            "mHC merge",
        );
    }

    /// tokens を seq の続き（位置 seq.pos から）として 64 層通す。終わると作業場の h [T][d] に最後の層の残差が残る（head で使う）。
    /// T は opt.chunk 以下（長い文は prefill で塊に分ける）
    pub fn forward(&mut self, seq: &mut Seq, tokens: &[u32], dump: Option<&mut Dump>) {
        self.forward_from(seq, tokens, None, dump)
    }

    /// forward の入口を選べる形: h_in が Some なら埋め込みの代わりにその残差 [T][d] から（層 layer0 の入口。層の答え合わせ用）
    pub fn forward_from(&mut self, seq: &mut Seq, tokens: &[u32], h_in: Option<&[f32]>, mut dump: Option<&mut Dump>) {
        // 前の forward がまだ流れていれば待つ（作業場の入力を上書きする前に）。NULL の流れの 0 埋め・写しも待つ
        self.sync_prefill();
        self.sync_legacy();
        self.sync_commit();
        let t = tokens.len();
        assert!(t >= 1 && t <= self.sc.tc, "1 回の語数 {t} が作業場（{}）を超える", self.sc.tc);
        assert!(seq.pos + t <= seq.max_ctx, "文脈 {} が上限 {} を超える", seq.pos + t, seq.max_ctx);
        let d = self.cfg.d;
        let path = if t <= self.opt.gemv_max { Path::Gemv } else { Path::Gemm };
        let ids: Vec<i32> = tokens.iter().map(|&x| x as i32).collect();
        assert!(ids.iter().all(|&x| x >= 0 && (x as usize) < self.cfg.vocab), "語の番号が語彙の外");
        upload_i32(&self.sc.ids, &ids);
        // Attention の位置・書く場所（頁表で引く）・頁表・列の範囲（16 層で同じ）
        assert!(seq.ensure(seq.pos + t), "KV の頁が足りない（位置 {}）", seq.pos + t);
        let pos: Vec<i32> = (0..t).map(|i| (seq.pos + i) as i32).collect();
        upload_i32(&self.sc.pos, &pos);
        // 画像のある列: RoPE の位置は (t, h, w, e) で（文字だけの列は今までと同じ道）
        self.pos4_on = !seq.media.is_empty();
        if self.pos4_on {
            let p4: Vec<i32> = (0..t).flat_map(|i| seq.rope_pos4(seq.pos + i)).collect();
            upload_i32(&self.sc.pos4, &p4);
        }
        let slots: Vec<i32> = (0..t).map(|i| seq.kv_row(seq.pos + i)).collect();
        upload_i32(&self.sc.slot, &slots);
        if self.sc.bt_pages < seq.n_pages {
            self.sc.bt = DevBuf::new(seq.n_pages * 4);
            self.sc.bt_pages = seq.n_pages;
        }
        upload_i32(&self.sc.bt, &seq.page_table());
        upload_i32(&self.sc.qs, &[0, t as i32]);
        upload_i32(&self.sc.kl, &[(seq.pos + t) as i32]);
        match h_in {
            Some(h) => {
                assert_eq!(h.len(), t * d, "入口の残差の大きさ");
                h2d(self.sc.h.ptr, unsafe { std::slice::from_raw_parts(h.as_ptr().cast(), h.len() * 4) });
            }
            None => {
                assert!(self.layer0 == 0, "層 {} から載せたので入口の残差が要る", self.layer0);
                check(
                    unsafe {
                        kffi::infer_m_embed(self.embd_qt as i32, self.embd.ptr, d as i32, self.sc.ids.ptr.cast(), t as i32, self.sc.h.f32p(), self.st)
                    },
                    "embed",
                );
                // 画像の行は、語の埋め込みの代わりに画像の符号化の行（プレフィルの流れの上で、埋め込みの後に）
                let (a0, a1) = (seq.pos, seq.pos + t);
                for sp in &seq.media {
                    let (a, b) = (sp.start.max(a0), (sp.start + sp.n()).min(a1));
                    if a < b {
                        check(
                            unsafe {
                                kffi::cudaMemcpyAsync(
                                    self.sc.h.f32p().add((a - a0) * d).cast(),
                                    (sp.emb.ptr as *const f32).add((a - sp.start) * d).cast(),
                                    (b - a) * d * 4,
                                    3,
                                    self.st,
                                )
                            },
                            "画像の行",
                        );
                    }
                }
            }
        }
        if let Some(dp) = dump.as_deref_mut() {
            dp.emb.extend(d2h_f32(self.sc.h.f32p(), t * d));
        }
        let (mut gi, mut ai) = (0usize, 0usize);
        for li in 0..self.layers.len() {
            let il = self.layer0 + li;
            let l = &self.layers[li];
            let lp = self.cfg.lp;
            if let Some(hw) = l.hc.as_ref() {
                self.hc_route(&self.sc, lp.unwrap().mhc_first(il), &hw.attn_w, &hw.attn_b, t, self.st);
            }
            self.norm(&l.attn_norm, t, path);
            if l.hc.is_some() {
                self.hc_zero_h(&self.sc, t, self.st);
            }
            if let Some(dp) = dump.as_deref_mut() {
                if dp.want(il) {
                    dp.put("attn_norm", il, d, d2h_f32(self.sc.xa.f32p(), t * d));
                }
            }
            match &l.mix {
                Mix::Gdn(w) => {
                    self.gdn_layer(w, gi, seq, t, path, il, &mut dump);
                    gi += 1;
                }
                Mix::Attn(w) => {
                    self.attn_layer(w, ai, seq, t, path, il, &mut dump);
                    ai += 1;
                }
            }
            if let Some(hw) = l.hc.as_ref() {
                self.hc_merge(&self.sc, t, self.st);
                self.hc_route(&self.sc, false, &hw.ffn_w, &hw.ffn_b, t, self.st);
            }
            if let Some(dp) = dump.as_deref_mut() {
                if dp.want(il) {
                    dp.put("attn_residual", il, d, d2h_f32(self.sc.h.f32p(), t * d));
                }
            }
            self.norm(&l.post_norm, t, path);
            if l.hc.is_some() {
                self.hc_zero_h(&self.sc, t, self.st);
            }
            if let Some(dp) = dump.as_deref_mut() {
                if dp.want(il) {
                    dp.put("attn_post_norm", il, d, d2h_f32(self.sc.xa.f32p(), t * d));
                }
            }
            self.ffn(l, t, path);
            if l.hc.is_some() {
                self.hc_merge(&self.sc, t, self.st);
            }
            self.finite(self.sc.h.f32p(), t * d, &format!("層 {il} の残差"));
            if let Some(dp) = dump.as_deref_mut() {
                if dp.h.len() <= li {
                    dp.h.resize(li + 1, vec![]);
                }
                dp.h[li].extend(d2h_f32(self.sc.h.f32p(), t * d));
            }
        }
        seq.pos += t;
    }

    /// プレフィルの流れに、別の流れの印（cudaEvent）を待たせる（画像の符号化の結果を読む前に）
    // ev（CUDA のイベント）は CUDA の不透明な handle をそのまま CUDA の API に渡すだけで、Rust の側では参照外ししない（不正な値は CUDA が誤りで返す）。
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn prefill_wait(&self, ev: *mut c_void) {
        check(unsafe { kffi::cudaStreamWaitEvent(self.st, ev, 0) }, "画像の待ち");
    }

    /// 長い文を opt.chunk 語ずつ forward（状態と KV は塊をまたいで持ち越す）。最後の塊の語数を返す（h の行の数）
    pub fn prefill(&mut self, seq: &mut Seq, tokens: &[u32], dump: Option<&mut Dump>) -> usize {
        self.prefill_from(seq, tokens, None, dump)
    }

    /// prefill の入口を選べる形（h_in は [T][d]、forward_from と同じ）
    pub fn prefill_from(&mut self, seq: &mut Seq, tokens: &[u32], h_in: Option<&[f32]>, mut dump: Option<&mut Dump>) -> usize {
        let (d, c) = (self.cfg.d, self.opt.chunk);
        let mut last = 0;
        for (i, ch) in tokens.chunks(c).enumerate() {
            let h = h_in.map(|h| &h[i * c * d..(i * c + ch.len()) * d]);
            self.forward_from(seq, ch, h, dump.as_deref_mut());
            last = ch.len();
        }
        last
    }

    /// 出口: 作業場の h の行 r0..r0+n（n ≤ 32）を output_norm → output。ロジットは作業場の logits [n][vocab] に置く
    pub fn head(&self, r0: usize, n: usize) {
        assert!(n >= 1 && n <= GEMV_B_MAX && r0 + n <= self.sc.tc);
        let out = self.output.as_ref().expect("出力の層を載せていない（--layers で一部だけ）");
        let d = self.cfg.d;
        let sc = &self.sc;
        let x = unsafe { sc.h.f32p().add(r0 * d) };
        check(
            unsafe {
                kffi::infer_m_rmsnorm(x, d as i64, self.out_norm.f32p(), d as i32, n as i32, self.cfg.eps, sc.xa.f32p(), d as i64, null_mut(),
                                      0, self.st)
            },
            "output_norm",
        );
        self.xprep(sc.xa.f32p(), d, n, &sc.xp_d);
        check(
            unsafe { ffi::infer_g3_gemv(out.qt, out.buf.ptr, d as i32, out.n as i32, sc.xp_d.ptr, sc.logits.f32p(), n as i32, 0, 0, self.st) },
            "output gemv",
        );
        self.finite(sc.logits.f32p(), n * out.n, "logits");
    }

    /// head の後: n 行それぞれの 1 位の語
    /// デコード・確かめの作業場（別に取っていればそれ）
    pub(crate) fn sdec(&self) -> &Scratch {
        self.sd.as_ref().unwrap_or(&self.sc)
    }

    /// プレフィルを別の流れで（M3）: 非同期・低い優先度の流れを作り、デコード・確かめの作業場を別に取る
    pub fn enable_async_prefill(&mut self) {
        if self.sd.is_some() {
            return;
        }
        let mut s: ffi::Stream = null_mut();
        // 1 = cudaStreamNonBlocking（NULL の流れと待ち合わせない）、優先度 0（低い）。TACHYON_SD_ONLY=1 なら流れは NULL のまま（確かめ用）
        if std::env::var("TACHYON_SD_ONLY").map_or(true, |v| v != "1") {
            // 優先度: 0 = 低い（デコードを先に。既定）、-1 = デコードと同じ（TACHYON_PREFILL_PRIO）
            let prio = std::env::var("TACHYON_PREFILL_PRIO").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            check(unsafe { kffi::cudaStreamCreateWithPriority(&mut s, 1, prio) }, "プレフィルの流れ");
        }
        self.st = s;
        self.sd = Some(Scratch::new_ex(&self.cfg, GEMV_B_MAX, true));
        self.dec = None; // デコードのグラフは新しい作業場で取り直す
    }

    /// プレフィルの流れを待つ
    pub fn sync_prefill(&self) {
        if !self.st.is_null() {
            check(unsafe { kffi::cudaStreamSynchronize(self.st) }, "プレフィルの流れの待ち");
        }
    }

    /// NULL の流れ（cudaMemset・D2D の cudaMemcpy・MTP など）に積んだ物を待つ。プレフィルの流れは NULL の流れと待ち合わせないので、
    /// 状態を 0 にした・控えから戻した直後にプレフィルを流す前に要る
    fn sync_legacy(&self) {
        if !self.st.is_null() {
            check(unsafe { kffi::cudaStreamSynchronize(null_mut()) }, "NULL の流れの待ち");
        }
    }

    /// 自分の流れ（NonBlocking）で KV・状態を読み書きする者（RAM の段の写し）の前に: プレフィルの流れと NULL の流れの両方を待つ。
    /// ⛔ sync_prefill だけでは足りない: 1 本だけ（max-seqs 1）だと非同期のプレフィルが無く、forward も reset の cudaMemset も
    ///    NULL の流れに乗るので、sync_prefill は何もせず、写しが終わっていないプレフィル・0 埋めと競る（2026-11-15 の A→B→A の取り違え。
    ///    b2865d8 で写しを自分の流れにしてから）。MTP の頭も NULL の流れなのでこれで待てる。確定の流れ（cs）は呼ぶ側が sync_commit で待つ
    pub fn sync_for_copy(&self) {
        self.sync_prefill();
        check(unsafe { kffi::cudaStreamSynchronize(null_mut()) }, "NULL の流れの待ち（写しの前）");
    }

    /// プレフィルの流れが空いたか（待たない）
    pub fn prefill_idle(&self) -> bool {
        self.st.is_null() || unsafe { kffi::cudaStreamQuery(self.st) } == 0
    }

    /// デコード・確かめの後: ロジット [n][vocab]（デコードの作業場から）
    pub fn logits_dec(&self, n: usize) -> Vec<f32> {
        d2h_f32(self.sdec().logits.f32p(), n * self.cfg.vocab)
    }

    /// 抜き取りの候補（GPU）: ロジットの行 0..n（dec ならデコード・確かめの作業場、でなければプレフィルの作業場）の [0, valid) の上位 CAND_K 語と最大・z。
    /// temps は行ごとの温度（z に使う）。扱えない形なら None（手元で logits 全部から）
    pub fn cands(&self, n: usize, dec: bool, valid: usize, temps: &[f32]) -> Option<Vec<RowCands>> {
        assert_eq!(temps.len(), n);
        if n == 0 || n > GEMV_B_MAX || self.output.is_none() {
            return None;
        }
        let v = self.cfg.vocab;
        let valid = if valid == 0 { v } else { valid.min(v) };
        let lg = if dec { self.sdec().logits.f32p() } else { self.sync_prefill(); self.sc.logits.f32p() };
        let c = &self.cand;
        h2d(c.temps.ptr, unsafe { std::slice::from_raw_parts(temps.as_ptr().cast(), n * 4) });
        let r = unsafe {
            kffi::infer_cand_topk(lg, v as i64, n as i32, valid as i32, CAND_K as i32, c.temps.f32p(), c.ws.ptr, c.ws.len as i64, c.ids.ptr.cast(),
                                  c.vals.f32p(), c.mz.ptr.cast(), S)
        };
        if r != 0 {
            return None;
        }
        let mut ib = vec![0u8; n * CAND_K * 4];
        let mut vb = vec![0u8; n * CAND_K * 4];
        let mut mb = vec![0u8; n * 16];
        d2h(&mut ib, c.ids.ptr);
        d2h(&mut vb, c.vals.ptr);
        d2h(&mut mb, c.mz.ptr);
        let u32s = |b: &[u8]| -> Vec<u32> { b.chunks_exact(4).map(|x| u32::from_le_bytes(x.try_into().unwrap())).collect() };
        let f32s = |b: &[u8]| -> Vec<f32> { b.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect() };
        let f64s: Vec<f64> = mb.chunks_exact(8).map(|x| f64::from_le_bytes(x.try_into().unwrap())).collect();
        let (ids, vals) = (u32s(&ib), f32s(&vb));
        Some(
            (0..n)
                .map(|r| RowCands {
                    ids: ids[r * CAND_K..(r + 1) * CAND_K].to_vec(),
                    vals: vals[r * CAND_K..(r + 1) * CAND_K].to_vec(),
                    m: f64s[2 * r] as f32,
                    z: f64s[2 * r + 1],
                })
                .collect(),
        )
    }

    pub fn argmax(&self, n: usize) -> Vec<u32> {
        self.sync_prefill();
        let v = self.cfg.vocab;
        check(
            unsafe { kffi::infer_m_argmax(self.sc.logits.f32p(), v as i32, v as i64, n as i32, self.sc.am.ptr.cast(), self.sc.amv.f32p(), self.st) },
            "argmax",
        );
        // argmax のカーネルはプレフィルの流れの上（NULL の流れの写しは待たない）→ 終わってから写す
        self.sync_prefill();
        let mut b = vec![0u8; n * 4];
        d2h(&mut b, self.sc.am.ptr);
        b.chunks_exact(4)
            .map(|c| {
                let i = i32::from_le_bytes(c.try_into().unwrap());
                assert!(i >= 0, "ロジットが全部 NaN");
                i as u32
            })
            .collect()
    }

    /// head の後: ロジット [n][vocab] を手元に
    pub fn logits(&self, n: usize) -> Vec<f32> {
        self.sync_prefill();
        d2h_f32(self.sc.logits.f32p(), n * self.cfg.vocab)
    }

    /// 作業場の h の行 r0..r0+n を手元に（最後の層の残差）
    pub fn hidden(&self, r0: usize, n: usize) -> Vec<f32> {
        self.sync_prefill();
        let d = self.cfg.d;
        d2h_f32(unsafe { self.sc.h.f32p().add(r0 * d) }, n * d)
    }

    /// result_norm（head の後の xa）の n 行
    pub fn result_norm(&self, n: usize) -> Vec<f32> {
        self.sync_prefill();
        d2h_f32(self.sc.xa.f32p(), n * self.cfg.d)
    }
}

/// log-softmax の上位 k: [(語, logprob)]（大きい順）。和は f64
pub fn top_logprobs(logits: &[f32], k: usize) -> Vec<(u32, f64)> {
    let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| if b > a { b } else { a }) as f64;
    let s: f64 = logits.iter().map(|&x| ((x as f64) - m).exp()).sum();
    let lse = m + s.ln();
    let mut idx: Vec<u32> = Vec::with_capacity(k + 1);
    for (i, &x) in logits.iter().enumerate() {
        if idx.len() < k || x > logits[*idx.last().unwrap() as usize] {
            let p = idx.partition_point(|&j| logits[j as usize] >= x);
            idx.insert(p, i as u32);
            idx.truncate(k);
        }
    }
    idx.into_iter().map(|i| (i, logits[i as usize] as f64 - lse)).collect()
}
