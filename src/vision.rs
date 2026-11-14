//! 画像の入力（Qwen3.8-27B の mmproj・qwen3vl_merger）: 画像を読み、縮め、ViT で符号化して本体の埋め込みの幅（5120）の行にする。
//! 手本は llama.cpp の tools/mtmd（mtmd-image.cpp の dyn_size・clip.cpp・models/qwen3vl.cpp）。式と並びはそこに合わせ、
//! 位置（本体に入れる時の M-RoPE）は HF の Qwen3-VL の決まりに合わせる（モデルが学習した形）。
//!
//! 1. 読む（jpeg・png・webp・gif の最初の 1 枚。向きの EXIF は見ない）→ RGB
//! 2. 縮める: 32 の倍数に（smart resize。語の数 8〜4096）。Pillow と同じ bicubic（a = −0.5・固定小数点）で、縦横比を変えて直に
//! 3. (x/255 − 0.5)/0.5 → 16×16 の区画に切る。区画の並びは 2×2 のまとまりごと（まとまりの行・列 → 中の dy・dx）
//! 4. 区画の埋め込み（2 つの重みの和）＋ bias ＋ 位置の埋め込み（48×48 を bilinear・角合わせで伸ばす）
//! 5. 27 層の ViT（LayerNorm・qkv・2 次元の RoPE・Attention・GELU の MLP）→ post LN → 4 区画ずつ 4608 → MLP（GELU）→ 5120
use crate::cuda::{self, ffi::Stream, DevBuf};
use crate::gguf;
use std::ffi::c_void;
use std::ptr::null_mut;

mod ffi {
    use super::*;
    extern "C" {
        pub fn infer_vis_gemm(x: *const c_void, w: *const c_void, bias: *const f32, y: *mut c_void, m: i32, n: i32, k: i32, ldy: i32, mode: i32, s: Stream) -> i32;
        pub fn infer_vis_layernorm(x: *const f32, w: *const f32, b: *const f32, m: i32, k: i32, eps: f32, y16: *mut c_void, y32: *mut f32, s: Stream) -> i32;
        pub fn infer_vis_rope_split(qkv: *const f32, py: *const i32, px: *const i32, m: i32, theta_scale: f32, out: *mut c_void, s: Stream) -> i32;
        pub fn infer_vis_qkv16_bytes(m: i32) -> i64;
        pub fn infer_vis_attn(qkv16: *const c_void, m: i32, scale: f32, out: *mut c_void, s: Stream) -> i32;
        pub fn infer_vis_f32_to_f16(x: *const f32, n: i64, y: *mut c_void, s: Stream) -> i32;
        pub fn cudaStreamSynchronize(s: Stream) -> i32;
        pub fn cudaStreamCreateWithPriority(s: *mut Stream, flags: u32, prio: i32) -> i32;
        pub fn cudaEventCreateWithFlags(e: *mut *mut c_void, flags: u32) -> i32;
        pub fn cudaEventRecord(e: *mut c_void, s: Stream) -> i32;
        pub fn cudaEventQuery(e: *mut c_void) -> i32;
        pub fn cudaEventSynchronize(e: *mut c_void) -> i32;
        pub fn cudaEventDestroy(e: *mut c_void) -> i32;
    }
}

/// 符号化の流れ（NULL の流れと待ち合わせない・優先度は低い）
pub fn stream_new() -> Stream {
    let mut s = null_mut();
    cuda::check(unsafe { ffi::cudaStreamCreateWithPriority(&mut s, 1, 0) }, "vis 流れ");
    s
}

/// 流れ s を待つ（誤りの後で、積んだ符号化の入力を捨てる前に）
// s（Stream）は CUDA の不透明な handle をそのまま CUDA の API に渡すだけで、Rust の側では参照外ししない（不正な値は CUDA が誤りで返す）。
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn stream_sync(s: Stream) {
    unsafe { ffi::cudaStreamSynchronize(s) };
}

/// 流れ s の今の所の印（終わったかは query・待つのは sync）
pub struct Mark(*mut c_void);

impl Mark {
    // s（Stream）は CUDA の不透明な handle をそのまま CUDA の API に渡すだけで、Rust の側では参照外ししない（不正な値は CUDA が誤りで返す）。
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn record(s: Stream) -> Mark {
        let mut e = null_mut();
        cuda::check(unsafe { ffi::cudaEventCreateWithFlags(&mut e, 2) }, "vis 印");
        cuda::check(unsafe { ffi::cudaEventRecord(e, s) }, "vis 印");
        Mark(e)
    }
    pub fn raw(&self) -> *mut c_void {
        self.0
    }
    pub fn done(&self) -> bool {
        unsafe { ffi::cudaEventQuery(self.0) == 0 }
    }
    pub fn wait(&self) {
        cuda::check(unsafe { ffi::cudaEventSynchronize(self.0) }, "vis 印の待ち");
    }
}

impl Drop for Mark {
    fn drop(&mut self) {
        unsafe {
            ffi::cudaEventSynchronize(self.0);
            ffi::cudaEventDestroy(self.0);
        }
    }
}

fn ck(code: i32, what: &str) -> Result<(), String> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("{what}: {code}"))
    }
}

// ---------------------------------------------------------------- 1〜2. 読む・縮める

/// RGB の画像（行ごと、1 画素 3 バイト）
#[derive(Clone)]
pub struct Rgb {
    pub w: usize,
    pub h: usize,
    pub px: Vec<u8>,
}

/// バイト（jpeg・png・webp・gif）→ RGB
pub fn decode(bytes: &[u8]) -> Result<Rgb, String> {
    // 小さなファイルが巨大な画像に開く物（圧縮爆弾）を止める: 1 辺 16384・戻す量 512MiB まで
    let mut r = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format().map_err(|e| format!("画像を読めない: {e}"))?;
    let mut lim = image::Limits::default();
    lim.max_image_width = Some(16384);
    lim.max_image_height = Some(16384);
    lim.max_alloc = Some(512 << 20);
    r.limits(lim);
    let img = r.decode().map_err(|e| format!("画像を読めない: {e}"))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    if w == 0 || h == 0 {
        return Err("画像の大きさが 0".into());
    }
    Ok(Rgb { w, h, px: rgb.into_raw() })
}

/// 縮めた後の大きさ（llama.cpp の calc_size_preserved_ratio・transformers の smart_resize）
pub fn target_size(w: usize, h: usize, align: usize, min_px: usize, max_px: usize) -> (usize, usize) {
    let a = align as f32;
    let round_by = |x: f32| ((x / a).round() as usize) * align;
    let ceil_by = |x: f32| ((x / a).ceil() as usize) * align;
    let floor_by = |x: f32| ((x / a).floor() as usize) * align;
    let (wf, hf) = (w as f32, h as f32);
    let mut wb = align.max(round_by(wf));
    let mut hb = align.max(round_by(hf));
    if max_px > 0 && hb * wb > max_px {
        let beta = (hf * wf / max_px as f32).sqrt();
        hb = align.max(floor_by(hf / beta));
        wb = align.max(floor_by(wf / beta));
    } else if min_px > 0 && hb * wb < min_px {
        let beta = (min_px as f32 / (hf * wf)).sqrt();
        hb = ceil_by(hf * beta);
        wb = ceil_by(wf * beta);
    }
    (wb, hb)
}

/// Pillow と同じ bicubic（a = −0.5、22 ビットの固定小数点、横 → 縦）
pub fn resize_bicubic(src: &Rgb, tw: usize, th: usize) -> Rgb {
    const PB: i32 = 22;
    let filter = |x: f64| -> f64 {
        let x = x.abs();
        let a = -0.5;
        if x < 1.0 {
            ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            (((x - 5.0) * x + 8.0) * x - 4.0) * a
        } else {
            0.0
        }
    };
    let weights = |ins: usize, outs: usize| -> (usize, Vec<(usize, usize)>, Vec<i32>) {
        let scale = ins as f64 / outs as f64;
        let fs = scale.max(1.0);
        let support = 2.0 * fs;
        let ksize = support.ceil() as usize * 2 + 1;
        let mut bounds = vec![(0usize, 0usize); outs];
        let mut kk = vec![0f64; outs * ksize];
        for xx in 0..outs {
            let center = (xx as f64 + 0.5) * scale;
            let ss = 1.0 / fs;
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = ((center + support + 0.5) as i64).min(ins as i64) as usize - xmin;
            let mut ww = 0.0;
            for x in 0..xmax {
                let w = filter((x as f64 + xmin as f64 - center + 0.5) * ss);
                kk[xx * ksize + x] = w;
                ww += w;
            }
            if ww != 0.0 {
                for x in 0..xmax {
                    kk[xx * ksize + x] /= ww;
                }
            }
            bounds[xx] = (xmin, xmax);
        }
        let fx = (1u64 << PB) as f64;
        let ki: Vec<i32> = kk.iter().map(|&w| (w * fx + if w < 0.0 { -0.5 } else { 0.5 }) as i32).collect();
        (ksize, bounds, ki)
    };
    let clip8 = |v: i32| -> u8 { v.clamp(0, 255) as u8 };
    let mut cur = src.clone();
    if tw != cur.w {
        let (ks, b, k) = weights(cur.w, tw);
        let mut out = vec![0u8; tw * cur.h * 3];
        for y in 0..cur.h {
            for xx in 0..tw {
                let (xmin, xcnt) = b[xx];
                let mut s = [1i32 << (PB - 1); 3];
                for x in 0..xcnt {
                    let p = &cur.px[(y * cur.w + xmin + x) * 3..];
                    for c in 0..3 {
                        s[c] = s[c].wrapping_add(p[c] as i32 * k[xx * ks + x]);
                    }
                }
                for c in 0..3 {
                    out[(y * tw + xx) * 3 + c] = clip8(s[c] >> PB);
                }
            }
        }
        cur = Rgb { w: tw, h: cur.h, px: out };
    }
    if th != cur.h {
        let (ks, b, k) = weights(cur.h, th);
        let row = cur.w * 3;
        let mut out = vec![0u8; row * th];
        for yy in 0..th {
            let (ymin, ycnt) = b[yy];
            for i in 0..row {
                let mut s = 1i32 << (PB - 1);
                for y in 0..ycnt {
                    s = s.wrapping_add(cur.px[(ymin + y) * row + i] as i32 * k[yy * ks + y]);
                }
                out[yy * row + i] = clip8(s >> PB);
            }
        }
        cur = Rgb { w: cur.w, h: th, px: out };
    }
    cur
}

/// 縦横比を保って縮め、tw × th の黒の上の真ん中に置く（llama.cpp の PAD_CEIL）
pub fn resize_pad(src: &Rgb, tw: usize, th: usize) -> Rgb {
    if src.w == tw && src.h == th {
        return src.clone();
    }
    // HF の Qwen3-VL も llama.cpp（b12209）も縦横比を変えて直に縮める（黒で埋めない。11-12 に埋めの形と比べて cos 0.94 → 0.9997）
    if std::env::var("TACHYON_VIS_PAD").map_or(true, |v| v != "ceil") {
        return resize_bicubic(src, tw, th);
    }
    let sw = tw as f32 / src.w as f32;
    let sh = th as f32 / src.h as f32;
    let s = sw.min(sh);
    let nw = ((src.w as f32 * s).ceil() as usize).min(tw);
    let nh = ((src.h as f32 * s).ceil() as usize).min(th);
    let r = resize_bicubic(src, nw, nh);
    let mut out = Rgb { w: tw, h: th, px: vec![0u8; tw * th * 3] };
    let (ox, oy) = ((tw - nw) / 2, (th - nh) / 2);
    for y in 0..nh {
        let d = ((oy + y) * tw + ox) * 3;
        out.px[d..d + nw * 3].copy_from_slice(&r.px[y * nw * 3..(y + 1) * nw * 3]);
    }
    out
}

/// Qwen3.8-27B の mmproj の決まり（区画 16・2×2 をまとめる・語の数 8〜4096）
pub const PATCH: usize = 16;
pub const MERGE: usize = 2;
pub const MIN_TOKENS: usize = 8;
pub const MAX_TOKENS: usize = 4096;

/// 模型に渡す大きさに縮める（w・h は 32 の倍数）。口の側でも使う（語の数 = (w/32)·(h/32) が先に分かる）
pub fn resize_for_model(img: &Rgb) -> Rgb {
    let a = PATCH * MERGE;
    let (tw, th) = target_size(img.w, img.h, a, MIN_TOKENS * a * a, MAX_TOKENS * a * a);
    resize_pad(img, tw, th)
}

/// base64（data URI の中身）を戻す。max バイトを超えたら Err
pub fn b64decode(s: &str, max: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' | b'\n' | b'\r' | b' ' | b'\t' => continue,
            _ => return Err("the data URI is not valid base64".into()),
        } as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            if out.len() > max {
                return Err(format!("the image is larger than {max} bytes"));
            }
        }
    }
    Ok(out)
}

/// data URI → 画像のバイト
pub fn data_uri_bytes(uri: &str, max: usize) -> Result<Vec<u8>, String> {
    let rest = uri.strip_prefix("data:").ok_or("not a data URI")?;
    let (head, data) = rest.split_once(',').ok_or("the data URI has no ','")?;
    if !head.ends_with(";base64") {
        return Err("the data URI must be base64 (data:image/...;base64,...)".into());
    }
    b64decode(data, max)
}

// ---------------------------------------------------------------- 符号化器

pub struct VisLayer {
    ln1_w: DevBuf,
    ln1_b: DevBuf,
    qkv_w: DevBuf,
    qkv_b: DevBuf,
    out_w: DevBuf,
    out_b: DevBuf,
    ln2_w: DevBuf,
    ln2_b: DevBuf,
    up_w: DevBuf,
    up_b: DevBuf,
    down_w: DevBuf,
    down_b: DevBuf,
}

pub struct Vision {
    pub d: usize,
    pub n_head: usize,
    pub d_ff: usize,
    pub patch: usize,
    pub merge: usize,
    pub out_dim: usize,
    pub eps: f32,
    pub min_tokens: usize,
    pub max_tokens: usize,
    patch_w: DevBuf,
    patch_b: DevBuf,
    /// 位置の埋め込み [48·48][1152]（手元。伸ばすのは手元で）
    pos: Vec<f32>,
    pos_side: usize,
    layers: Vec<VisLayer>,
    post_w: DevBuf,
    post_b: DevBuf,
    mm0_w: DevBuf,
    mm0_b: DevBuf,
    mm2_w: DevBuf,
    mm2_b: DevBuf,
    pub bytes: usize,
    /// 符号化の作業場（区画の数 cap まで。1 つの流れで順に使い回す。広げるときは流れを待つ）
    ws: std::cell::RefCell<Option<Ws>>,
}

struct Ws {
    cap: usize,
    x16: DevBuf,
    qkv: DevBuf,
    q16: DevBuf,
    ao: DevBuf,
}

/// 手元で区画に切って GPU に送った画像（符号化の入力。符号化が終わるまで持っておく）
pub struct Staged {
    np: usize,
    pub nx: usize,
    pub ny: usize,
    xin: DevBuf,
    pyd: DevBuf,
    pxd: DevBuf,
    /// 残差（位置の埋め込みで始まり、符号化の間に書き換わる）
    h: DevBuf,
}

fn f32s(t: &gguf::TensorView) -> Vec<f32> {
    match t.type_name().as_str() {
        "F32" => t.bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        "F16" => t.bytes.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap()))).collect(),
        x => panic!("{}: 型 {x} は読まない", t.name),
    }
}

pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            // 非正規数
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            (s << 31) | ((e2 as u32) << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 {
        (s << 31) | (0xff << 23) | (m << 13)
    } else {
        (s << 31) | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

pub fn f32_to_f16(f: f32) -> u16 {
    // 偶数への丸め
    let x = f.to_bits();
    let s = ((x >> 16) & 0x8000) as u16;
    let e = ((x >> 23) & 0xff) as i32;
    let m = x & 0x7f_ffff;
    if e == 255 {
        return s | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let e2 = e - 127 + 15;
    if e2 >= 31 {
        return s | 0x7c00;
    }
    if e2 <= 0 {
        if e2 < -10 {
            return s;
        }
        let m = m | 0x80_0000;
        let shift = (14 - e2) as u32;
        let half = 1u32 << (shift - 1);
        let r = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let r = if rem > half || (rem == half && r & 1 == 1) { r + 1 } else { r };
        return s | r as u16;
    }
    let r = m >> 13;
    let rem = m & 0x1fff;
    let mut v = ((e2 as u32) << 10) | r;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        v += 1;
    }
    s | v as u16
}

fn f16buf(t: &gguf::TensorView) -> DevBuf {
    assert_eq!(t.type_name(), "F16", "{}: F16 のはず", t.name);
    DevBuf::from_bytes(t.bytes)
}

fn f32buf(t: &gguf::TensorView) -> DevBuf {
    DevBuf::from_f32(&f32s(t))
}

/// 符号化の結果
pub struct Encoded {
    /// [n_tokens][out_dim]（f32、GPU）
    pub emb: DevBuf,
    /// 語の格子（まとめた後）: 横 nx・縦 ny（n_tokens = nx·ny、並びは行ごと）
    pub nx: usize,
    pub ny: usize,
}

impl Vision {
    pub fn load(path: &str) -> Result<Vision, String> {
        let g = gguf::Model::open(path).map_err(|e| format!("{path}: {e}"))?;
        let u = |k: &str| g.meta(k).and_then(|v| v.as_u32()).map(|x| x as usize).ok_or_else(|| format!("{path}: {k} が無い"));
        let d = u("clip.vision.embedding_length")?;
        let n_head = u("clip.vision.attention.head_count")?;
        let d_ff = u("clip.vision.feed_forward_length")?;
        let n_layer = u("clip.vision.block_count")?;
        let patch = u("clip.vision.patch_size")?;
        let merge = u("clip.vision.spatial_merge_size").unwrap_or(2);
        let out_dim = u("clip.vision.projection_dim")?;
        let eps = g.meta("clip.vision.attention.layer_norm_epsilon").and_then(|v| v.as_f32()).unwrap_or(1e-6);
        if d != 1152 || n_head != 16 || patch != 16 || merge != 2 {
            return Err(format!("{path}: 形が違う（幅 {d}・頭 {n_head}・区画 {patch}・まとめ {merge}）。qwen3vl_merger の 1152・16・16・2 だけ"));
        }
        let mut bytes = 0usize;
        // 区画の埋め込み: 2 つ（時間方向の 2 枚ぶん）の和。[1152][3·16·16]
        let p0 = f32s(&g.tensor("v.patch_embd.weight"));
        let p1 = f32s(&g.tensor("v.patch_embd.weight.1"));
        let pw16: Vec<u16> = p0.iter().zip(&p1).map(|(a, b)| f32_to_f16(a + b)).collect();
        let patch_w = DevBuf::from_bytes(unsafe { std::slice::from_raw_parts(pw16.as_ptr().cast(), pw16.len() * 2) });
        bytes += patch_w.len;
        let pos = f32s(&g.tensor("v.position_embd.weight"));
        let pos_side = ((pos.len() / d) as f64).sqrt() as usize;
        let mut layers = vec![];
        for l in 0..n_layer {
            let t = |s: &str| g.tensor(&format!("v.blk.{l}.{s}"));
            let ly = VisLayer {
                ln1_w: f32buf(&t("ln1.weight")),
                ln1_b: f32buf(&t("ln1.bias")),
                qkv_w: f16buf(&t("attn_qkv.weight")),
                qkv_b: f32buf(&t("attn_qkv.bias")),
                out_w: f16buf(&t("attn_out.weight")),
                out_b: f32buf(&t("attn_out.bias")),
                ln2_w: f32buf(&t("ln2.weight")),
                ln2_b: f32buf(&t("ln2.bias")),
                up_w: f16buf(&t("ffn_up.weight")),
                up_b: f32buf(&t("ffn_up.bias")),
                down_w: f16buf(&t("ffn_down.weight")),
                down_b: f32buf(&t("ffn_down.bias")),
            };
            bytes += [&ly.qkv_w, &ly.out_w, &ly.up_w, &ly.down_w].iter().map(|b| b.len).sum::<usize>();
            layers.push(ly);
        }
        let v = Vision {
            d,
            n_head,
            d_ff,
            patch,
            merge,
            out_dim,
            eps,
            min_tokens: 8,
            max_tokens: 4096,
            patch_w,
            patch_b: f32buf(&g.tensor("v.patch_embd.bias")),
            pos,
            pos_side,
            layers,
            post_w: f32buf(&g.tensor("v.post_ln.weight")),
            post_b: f32buf(&g.tensor("v.post_ln.bias")),
            mm0_w: f16buf(&g.tensor("mm.0.weight")),
            mm0_b: f32buf(&g.tensor("mm.0.bias")),
            mm2_w: f16buf(&g.tensor("mm.2.weight")),
            mm2_b: f32buf(&g.tensor("mm.2.bias")),
            bytes: 0,
            ws: std::cell::RefCell::new(None),
        };
        Ok(Vision { bytes: bytes + v.mm0_w.len + v.mm2_w.len, ..v })
    }

    /// 縮めた後の画像
    pub fn preprocess(&self, img: &Rgb) -> Rgb {
        resize_for_model(img)
    }

    /// 縮めた画像（resize_for_model の結果）を正規化して符号化
    pub fn encode_resized(&self, r: &Rgb, s: Stream) -> Result<Encoded, String> {
        let f: Vec<f32> = r.px.iter().map(|&v| (v as f32 / 255.0 - 0.5) / 0.5).collect();
        self.encode_f32(r.w, r.h, &f, s)
    }

    /// 画像 1 枚を符号化（s は GPU の流れ。手元は最後に待つ）
    pub fn encode(&self, img: &Rgb, s: Stream) -> Result<Encoded, String> {
        let r = self.preprocess(img);
        let f: Vec<f32> = r.px.iter().map(|&v| (v as f32 / 255.0 - 0.5) / 0.5).collect();
        self.encode_f32(r.w, r.h, &f, s)
    }

    /// 縮めて正規化した画像（行ごと・1 画素 RGB の f32）を符号化（手元で待つ）。w・h は 32 の倍数
    // s（Stream）は CUDA の不透明な handle をそのまま CUDA の API に渡すだけで、Rust の側では参照外ししない（不正な値は CUDA が誤りで返す）。
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn encode_f32(&self, w: usize, h: usize, px32: &[f32], s: Stream) -> Result<Encoded, String> {
        let st = self.stage(w, h, px32)?;
        let emb = self.run(&st, s)?;
        cuda::check(unsafe { ffi::cudaStreamSynchronize(s) }, "vis 待ち");
        Ok(Encoded { emb, nx: st.nx, ny: st.ny })
    }

    /// 縮めた画像（resize_for_model の結果）を区画に切って GPU に送る（カーネルは流さない。手元の仕事と同期の写しだけ）
    pub fn stage_resized(&self, r: &Rgb) -> Result<Staged, String> {
        let f: Vec<f32> = r.px.iter().map(|&v| (v as f32 / 255.0 - 0.5) / 0.5).collect();
        self.stage(r.w, r.h, &f)
    }

    pub fn stage(&self, w: usize, h: usize, px32: &[f32]) -> Result<Staged, String> {
        let (p, mg, d) = (self.patch, self.merge, self.d);
        if w % (p * mg) != 0 || h % (p * mg) != 0 || px32.len() != w * h * 3 {
            return Err(format!("画像の大きさ {w} × {h} が {} の倍数でない", p * mg));
        }
        let (pw, ph) = (w / p, h / p);
        let (nx, ny) = (pw / mg, ph / mg);
        let np = pw * ph;
        let kp = 3 * p * p;
        // 区画（2×2 のまとまりの順）と位置
        let mut xs = vec![0u16; np * kp];
        let mut py = vec![0i32; np];
        let mut px = vec![0i32; np];
        let mut posv = vec![0f32; np * d];
        let pe = self.interp_pos(pw, ph);
        let mut i = 0;
        for by in 0..ny {
            for bx in 0..nx {
                for dy in 0..mg {
                    for dx in 0..mg {
                        let (yy, xx) = (by * mg + dy, bx * mg + dx);
                        py[i] = yy as i32;
                        px[i] = xx as i32;
                        for c in 0..3 {
                            for ky in 0..p {
                                for kx in 0..p {
                                    xs[i * kp + c * p * p + ky * p + kx] = f32_to_f16(px32[((yy * p + ky) * w + xx * p + kx) * 3 + c]);
                                }
                            }
                        }
                        posv[i * d..(i + 1) * d].copy_from_slice(&pe[(yy * pw + xx) * d..(yy * pw + xx + 1) * d]);
                        i += 1;
                    }
                }
            }
        }
        Ok(Staged {
            np,
            nx,
            ny,
            xin: DevBuf::from_bytes(unsafe { std::slice::from_raw_parts(xs.as_ptr().cast(), xs.len() * 2) }),
            pyd: DevBuf::from_i32(&py),
            pxd: DevBuf::from_i32(&px),
            h: DevBuf::from_f32(&posv),
        })
    }

    /// 送った画像を流れ s で符号化（待たない）。結果 [nx·ny][out_dim]（f32）。st は流れ s が終わるまで捨てないこと
    // s（Stream）は CUDA の不透明な handle をそのまま CUDA の API に渡すだけで、Rust の側では参照外ししない（不正な値は CUDA が誤りで返す）。
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn run(&self, stg: &Staged, s: Stream) -> Result<DevBuf, String> {
        let (p, mg, d) = (self.patch, self.merge, self.d);
        let np = stg.np;
        let kp = 3 * p * p;
        let dbg = std::env::var("TACHYON_VIS_DEBUG").is_ok();
        let show = |name: &str, b: &DevBuf, n: usize, half: bool| {
            if !dbg {
                return;
            }
            unsafe { ffi::cudaStreamSynchronize(s) };
            let v: Vec<f32> = if half {
                let raw = b.to_bytes();
                raw[..n * 2].chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()
            } else {
                b.to_f32()[..n].to_vec()
            };
            let sum: f64 = v.iter().map(|&x| x as f64).sum();
            eprintln!("  {name:14} sum {sum:14.6}  first {:.4} {:.4} {:.4}", v[0], v[1], v[2]);
        };
        // 作業場（足りなければ、流れを待ってから広げる）
        let mut wsr = self.ws.borrow_mut();
        if wsr.as_ref().map_or(true, |w| w.cap < np) {
            unsafe { ffi::cudaStreamSynchronize(s) };
            *wsr = None;
            let cap = np.max(2048);
            *wsr = Some(Ws {
                cap,
                x16: DevBuf::new(cap * d.max(self.d_ff) * 2),
                qkv: DevBuf::new(cap * 3 * d * 4),
                q16: DevBuf::new(unsafe { ffi::infer_vis_qkv16_bytes(cap as i32) } as usize),
                ao: DevBuf::new(cap * d * 2),
            });
        }
        let ws = wsr.as_ref().unwrap();
        let (x16, qkv, q16, ao) = (&ws.x16, &ws.qkv, &ws.q16, &ws.ao);
        let (h, pyd, pxd) = (&stg.h, &stg.pyd, &stg.pxd);
        // 残差（f32）= 位置 ＋ 区画の埋め込み ＋ bias
        ck(unsafe { ffi::infer_vis_gemm(stg.xin.ptr, self.patch_w.ptr, self.patch_b.f32p(), h.ptr, np as i32, d as i32, kp as i32, d as i32, 1, s) }, "vis 区画")?;
        let dh = d / self.n_head;
        let theta_scale = 10000f32.powf(-2.0 / (dh as f32 / 2.0));
        let scale = 1.0 / (dh as f32).sqrt();
        let (m, di, ff) = (np as i32, d as i32, self.d_ff as i32);
        show("inp_pos_emb", h, np * d, false);
        for (li, ly) in self.layers.iter().enumerate() {
            let sh = dbg && (li == 0 || li == self.layers.len() - 1);
            unsafe {
                ck(ffi::infer_vis_layernorm(h.f32p(), ly.ln1_w.f32p(), ly.ln1_b.f32p(), m, di, self.eps, x16.ptr, null_mut(), s), "vis ln1")?;
                if sh { show(&format!("ln1-{li}"), x16, np * d, true); }
                ck(ffi::infer_vis_gemm(x16.ptr, ly.qkv_w.ptr, ly.qkv_b.f32p(), qkv.ptr, m, 3 * di, di, 3 * di, 0, s), "vis qkv")?;
                if sh { show(&format!("qkv-{li}"), qkv, np * 3 * d, false); }
                ck(ffi::infer_vis_rope_split(qkv.f32p(), pyd.ptr.cast(), pxd.ptr.cast(), m, theta_scale, q16.ptr, s), "vis rope")?;
                ck(ffi::infer_vis_attn(q16.ptr, m, scale, ao.ptr, s), "vis attn")?;
                if sh { show(&format!("kqv_out-{li}"), ao, np * d, true); }
                ck(ffi::infer_vis_gemm(ao.ptr, ly.out_w.ptr, ly.out_b.f32p(), h.ptr, m, di, di, di, 1, s), "vis out")?;
                if sh { show(&format!("ffn_inp-{li}"), h, np * d, false); }
                ck(ffi::infer_vis_layernorm(h.f32p(), ly.ln2_w.f32p(), ly.ln2_b.f32p(), m, di, self.eps, x16.ptr, null_mut(), s), "vis ln2")?;
                ck(ffi::infer_vis_gemm(x16.ptr, ly.up_w.ptr, ly.up_b.f32p(), qkv.ptr, m, ff, di, ff, 3, s), "vis up")?;
                if sh { show(&format!("ffn_gelu-{li}"), qkv, np * self.d_ff, true); }
                ck(ffi::infer_vis_gemm(qkv.ptr, ly.down_w.ptr, ly.down_b.f32p(), h.ptr, m, di, ff, di, 1, s), "vis down")?;
                if sh { show(&format!("layer_out-{li}"), h, np * d, false); }
            }
        }
        // post LN → [np/4][4608] → mm.0（GELU）→ mm.2
        let nt = stg.nx * stg.ny;
        let k4 = d * mg * mg;
        let emb = DevBuf::new(nt * self.out_dim * 4);
        unsafe {
            ck(ffi::infer_vis_layernorm(h.f32p(), self.post_w.f32p(), self.post_b.f32p(), m, di, self.eps, x16.ptr, null_mut(), s), "vis post")?;
            ck(ffi::infer_vis_gemm(x16.ptr, self.mm0_w.ptr, self.mm0_b.f32p(), qkv.ptr, nt as i32, k4 as i32, k4 as i32, k4 as i32, 3, s), "vis mm0")?;
            ck(ffi::infer_vis_gemm(qkv.ptr, self.mm2_w.ptr, self.mm2_b.f32p(), emb.ptr, nt as i32, self.out_dim as i32, k4 as i32, self.out_dim as i32, 0, s), "vis mm2")?;
        }
        let _ = ffi::infer_vis_f32_to_f16;
        Ok(emb)
    }

    /// 位置の埋め込みを pw × ph に伸ばす（bilinear・角合わせ。行は y·pw + x）
    fn interp_pos(&self, pw: usize, ph: usize) -> Vec<f32> {
        let (n, d) = (self.pos_side, self.d);
        let mut out = vec![0f32; pw * ph * d];
        let sx = if pw > 1 { (n - 1) as f32 / (pw - 1) as f32 } else { 0.0 };
        let sy = if ph > 1 { (n - 1) as f32 / (ph - 1) as f32 } else { 0.0 };
        for y in 0..ph {
            let fy = y as f32 * sy;
            let y0 = (fy.floor() as usize).min(n - 1);
            let y1 = (y0 + 1).min(n - 1);
            let wy = fy - y0 as f32;
            for x in 0..pw {
                let fx = x as f32 * sx;
                let x0 = (fx.floor() as usize).min(n - 1);
                let x1 = (x0 + 1).min(n - 1);
                let wx = fx - x0 as f32;
                let o = &mut out[(y * pw + x) * d..(y * pw + x + 1) * d];
                let r = |yy: usize, xx: usize| &self.pos[(yy * n + xx) * d..(yy * n + xx + 1) * d];
                let (a, b, c, e) = (r(y0, x0), r(y0, x1), r(y1, x0), r(y1, x1));
                for k in 0..d {
                    o[k] = (a[k] * (1.0 - wx) + b[k] * wx) * (1.0 - wy) + (c[k] * (1.0 - wx) + e[k] * wx) * wy;
                }
            }
        }
        out
    }
}
