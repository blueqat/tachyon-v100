//! GGUF の読み込み。分割された束（…-0000N-of-0000M.gguf）を全部 mmap し、メタデータとテンソルの目次を持つ。
//! 重みのバイトは mmap のまま渡す（写さない）。読むのは目次だけなので、開くのは速く RAM もほぼ食わない。
//!
//!   let m = Model::open("/…/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")?;
//!   let n = m.hp_u32("block_count");                 // qwen4exp.block_count
//!   let t = m.layer(0).tensor("attn_qkv.weight");     // blk.0.attn_qkv.weight
//!   infer_gemv(t.qt() as i32, …t.bytes…, t.k(), t.n_rows(), …)
use crate::qtype::QType;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::ops::Range;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn err<T>(s: impl Into<String>) -> Result<T> {
    Err(Error(s.into()))
}

// ---------------------------------------------------------------- ggml の型

/// ggml の型: (番号, 名前, 1 ブロックの要素数, 1 ブロックのバイト数)。tools/gguf_remote.py の表＋TQ・NVFP4
/// （Q8_1 は ggml の今の 36。gguf-py 0.19 の表は古い 40）
pub const GGML_TYPES: &[(u32, &str, usize, usize)] = &[
    (0, "F32", 1, 4), (1, "F16", 1, 2), (2, "Q4_0", 32, 18), (3, "Q4_1", 32, 20),
    (6, "Q5_0", 32, 22), (7, "Q5_1", 32, 24), (8, "Q8_0", 32, 34), (9, "Q8_1", 32, 36),
    (10, "Q2_K", 256, 84), (11, "Q3_K", 256, 110), (12, "Q4_K", 256, 144), (13, "Q5_K", 256, 176),
    (14, "Q6_K", 256, 210), (15, "Q8_K", 256, 292), (16, "IQ2_XXS", 256, 66), (17, "IQ2_XS", 256, 74),
    (18, "IQ3_XXS", 256, 98), (19, "IQ1_S", 256, 50), (20, "IQ4_NL", 32, 18), (21, "IQ3_S", 256, 110),
    (22, "IQ2_S", 256, 82), (23, "IQ4_XS", 256, 136), (24, "I8", 1, 1), (25, "I16", 1, 2),
    (26, "I32", 1, 4), (27, "I64", 1, 8), (28, "F64", 1, 8), (29, "IQ1_M", 256, 56), (30, "BF16", 1, 2),
    (34, "TQ1_0", 256, 54), (35, "TQ2_0", 256, 66), (39, "MXFP4", 32, 17), (40, "NVFP4", 64, 36), (41, "Q1_0", 128, 18),
    (42, "Q2_0", 64, 18), (43, "F8_E4M3", 1, 1),
];

/// 型番号 → (名前, 1 ブロックの要素数, 1 ブロックのバイト数)
pub fn ggml_type(id: u32) -> Option<(&'static str, usize, usize)> {
    GGML_TYPES.iter().find(|t| t.0 == id).map(|t| (t.1, t.2, t.3))
}

pub fn ggml_type_name(id: u32) -> String {
    ggml_type(id).map(|t| t.0.to_string()).unwrap_or_else(|| format!("type{id}"))
}

/// 型と形からバイト数（shape[0] がブロックの倍数でなければ None）
pub fn ggml_nbytes(id: u32, shape: &[usize]) -> Option<usize> {
    let (_, blk, bsz) = ggml_type(id)?;
    let k = *shape.first().unwrap_or(&1);
    if k % blk != 0 {
        return None;
    }
    let rows = shape.iter().skip(1).try_fold(1usize, |a, &d| a.checked_mul(d))?;
    (k / blk).checked_mul(bsz)?.checked_mul(rows)
}

// ---------------------------------------------------------------- メタデータの値

/// メタデータの値（GGUF の値の型 0〜12）
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Arr(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    /// 整数なら i128 で（符号つき・なしをまとめて比べられる）
    pub fn as_int(&self) -> Option<i128> {
        use Value::*;
        Some(match *self {
            U8(v) => v as i128,
            I8(v) => v as i128,
            U16(v) => v as i128,
            I16(v) => v as i128,
            U32(v) => v as i128,
            I32(v) => v as i128,
            U64(v) => v as i128,
            I64(v) => v as i128,
            _ => return None,
        })
    }
    pub fn as_u32(&self) -> Option<u32> {
        self.as_int().and_then(|v| u32::try_from(v).ok())
    }
    pub fn as_u64(&self) -> Option<u64> {
        self.as_int().and_then(|v| u64::try_from(v).ok())
    }
    pub fn as_i64(&self) -> Option<i64> {
        self.as_int().and_then(|v| i64::try_from(v).ok())
    }
    /// 数なら何でも f64 で
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v as f64),
            Value::F64(v) => Some(v),
            _ => self.as_int().map(|v| v as f64),
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        self.as_f64().map(|v| v as f32)
    }
    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(b) => Some(b),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(a) => Some(a),
            _ => None,
        }
    }
    pub fn type_name(&self) -> &'static str {
        use Value::*;
        match self {
            U8(_) => "u8",
            I8(_) => "i8",
            U16(_) => "u16",
            I16(_) => "i16",
            U32(_) => "u32",
            I32(_) => "i32",
            F32(_) => "f32",
            Bool(_) => "bool",
            Str(_) => "str",
            Arr(_) => "arr",
            U64(_) => "u64",
            I64(_) => "i64",
            F64(_) => "f64",
        }
    }
}

impl fmt::Display for Value {
    /// 長い配列（語彙など）は長さと先頭 8 つだけ
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use Value::*;
        match self {
            Str(s) if s.chars().count() > 80 => write!(f, "{:?}…（{} 字）", s.chars().take(80).collect::<String>(), s.chars().count()),
            Str(s) => write!(f, "{s:?}"),
            Arr(a) => {
                let head: Vec<String> = a.iter().take(if a.len() > 16 { 8 } else { 16 }).map(|v| v.to_string()).collect();
                if a.len() > 16 {
                    write!(f, "[{}, …]（{} 個）", head.join(", "), a.len())
                } else {
                    write!(f, "[{}]", head.join(", "))
                }
            }
            F32(v) => write!(f, "{v:e}"),
            F64(v) => write!(f, "{v:e}"),
            Bool(v) => write!(f, "{v}"),
            _ => write!(f, "{}", self.as_int().unwrap()),
        }
    }
}

// ---------------------------------------------------------------- 目次の読み

/// mmap の上を頭から読む
struct Rd<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        match self.pos.checked_add(n) {
            Some(e) if e <= self.b.len() => {
                let s = &self.b[self.pos..e];
                self.pos = e;
                Ok(s)
            }
            _ => err(format!("目次が途中で切れている（{} バイト目で {n} バイト読もうとした）", self.pos)),
        }
    }
    fn arr<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.take(N)?.try_into().unwrap())
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.arr()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.arr()?))
    }
    fn len(&mut self) -> Result<usize> {
        let n = self.u64()?;
        usize::try_from(n).map_err(|_| Error(format!("長さが大きすぎる: {n}")))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.len()?;
        let s = self.take(n)?;
        Ok(match std::str::from_utf8(s) {
            Ok(s) => s.to_owned(),
            Err(_) => String::from_utf8_lossy(s).into_owned(),
        })
    }
    fn value(&mut self, t: u32, depth: u32) -> Result<Value> {
        use Value::*;
        Ok(match t {
            0 => U8(self.arr::<1>()?[0]),
            1 => I8(self.arr::<1>()?[0] as i8),
            2 => U16(u16::from_le_bytes(self.arr()?)),
            3 => I16(i16::from_le_bytes(self.arr()?)),
            4 => U32(self.u32()?),
            5 => I32(i32::from_le_bytes(self.arr()?)),
            6 => F32(f32::from_le_bytes(self.arr()?)),
            7 => Bool(self.arr::<1>()?[0] != 0),
            8 => Str(self.string()?),
            9 => {
                if depth > 8 {
                    return err("配列の入れ子が深すぎる");
                }
                let et = self.u32()?;
                let n = self.len()?;
                // 1 要素は最低 1 バイト。残りより多いと言うなら壊れている
                if n > self.b.len() - self.pos {
                    return err(format!("配列の長さ {n} が残りのバイト数を超える"));
                }
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.value(et, depth + 1)?);
                }
                Arr(v)
            }
            10 => U64(self.u64()?),
            11 => I64(i64::from_le_bytes(self.arr()?)),
            12 => F64(f64::from_le_bytes(self.arr()?)),
            _ => return err(format!("知らない値の型 {t}")),
        })
    }
}

// ---------------------------------------------------------------- 束・テンソル・モデル

/// テンソルの目次の 1 行
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// ggml の順（shape[0] が 1 行の要素数 k、shape[1] が行、shape[2] が専門家）
    pub shape: Vec<usize>,
    pub ggml_type: u32,
    /// 自前のカーネルが知っている型なら
    pub qtype: Option<QType>,
    /// どの束か（0 始まり）
    pub shard: usize,
    /// その束のデータ領域の頭からの位置
    pub offset: usize,
    /// 型と形から計算したバイト数
    pub size: usize,
}

/// 1 つのファイル
pub struct Shard {
    pub path: PathBuf,
    pub version: u32,
    pub alignment: usize,
    /// データ領域の頭（ファイルの頭から）
    pub data_start: usize,
    /// この束のメタデータ（ファイルの順）
    pub meta: Vec<(String, Value)>,
    /// Model::tensors のうちこの束の分
    pub tensors: Range<usize>,
    map: Mmap,
}

impl Shard {
    pub fn file_len(&self) -> usize {
        self.map.len()
    }
    /// データ領域（テンソルが 1 つも無い束は空）
    pub fn data(&self) -> &[u8] {
        &self.map[self.data_start.min(self.map.len())..]
    }
    pub fn meta(&self, key: &str) -> Option<&Value> {
        self.meta.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    fn open(path: &Path, shard: usize, tensors: &mut Vec<TensorInfo>) -> Result<Shard> {
        let f = File::open(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        // 安全: 読むだけ。ファイルを書き換える者はいない前提（重みは読むだけの置き場）
        let map = unsafe { Mmap::map(&f) }.map_err(|e| Error(format!("{}: mmap: {e}", path.display())))?;
        let at = |e: Error| Error(format!("{}: {}", path.display(), e.0));
        let mut r = Rd { b: &map, pos: 0 };
        if r.take(4).map_err(at)? != b"GGUF" {
            return err(format!("{}: GGUF ではない", path.display()));
        }
        let version = r.u32().map_err(at)?;
        if !(2..=3).contains(&version) {
            return err(format!("{}: 版 {version} は読めない（2・3 だけ。大きい数ならビッグエンディアン）", path.display()));
        }
        let n_tensors = r.len().map_err(at)?;
        let n_kv = r.len().map_err(at)?;
        let mut meta = Vec::with_capacity(n_kv.min(1 << 16));
        for _ in 0..n_kv {
            let k = r.string().map_err(at)?;
            let t = r.u32().map_err(at)?;
            let v = r.value(t, 0).map_err(|e| at(Error(format!("{k}: {}", e.0))))?;
            meta.push((k, v));
        }
        let alignment = match meta.iter().find(|(k, _)| k == "general.alignment") {
            Some((_, v)) => v.as_u64().filter(|a| a.is_power_of_two()).ok_or_else(|| at(Error(format!("general.alignment が変: {v}"))))? as usize,
            None => 32,
        };
        let t0 = tensors.len();
        for _ in 0..n_tensors {
            let name = r.string().map_err(at)?;
            let nd = r.u32().map_err(at)?;
            if nd > 8 {
                return err(format!("{}: {name}: 次元が {nd}", path.display()));
            }
            let mut shape = Vec::with_capacity(nd as usize);
            for _ in 0..nd {
                shape.push(r.len().map_err(at)?);
            }
            let ggml_type = r.u32().map_err(at)?;
            let offset = r.len().map_err(at)?;
            let size = ggml_nbytes(ggml_type, &shape).ok_or_else(|| {
                at(Error(format!("{name}: 型 {} と形 {shape:?} から大きさが出せない", ggml_type_name(ggml_type))))
            })?;
            let qtype = QType::from_id(ggml_type);
            tensors.push(TensorInfo { name, shape, ggml_type, qtype, shard, offset, size });
        }
        let data_start = (r.pos + alignment - 1) / alignment * alignment;
        // 各テンソルがデータ領域に収まっているか（並びと隙間は ggufinfo が確かめる）
        let data_len = map.len().saturating_sub(data_start);
        for t in &tensors[t0..] {
            if t.offset % alignment != 0 {
                return err(format!("{}: {}: 位置 {} が {alignment} に揃っていない", path.display(), t.name, t.offset));
            }
            if t.offset.checked_add(t.size).map_or(true, |e| e > data_len) {
                return err(format!("{}: {}: 位置 {}＋{} がファイルの外（データ {data_len}）", path.display(), t.name, t.offset, t.size));
            }
        }
        Ok(Shard { path: path.to_owned(), version, alignment, data_start, meta, tensors: t0..tensors.len(), map })
    }
}

/// テンソル 1 つの見え方（バイトは mmap のまま）
#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a> {
    pub name: &'a str,
    pub ggml_type: u32,
    pub qtype: Option<QType>,
    pub shape: &'a [usize],
    pub bytes: &'a [u8],
}

impl<'a> TensorView<'a> {
    /// カーネルに渡す型（知らない型なら止める）
    pub fn qt(&self) -> QType {
        self.qtype.unwrap_or_else(|| panic!("{}: 型 {} は自前のカーネルが知らない", self.name, ggml_type_name(self.ggml_type)))
    }
    pub fn type_name(&self) -> String {
        ggml_type_name(self.ggml_type)
    }
    /// 1 行の要素数
    pub fn k(&self) -> usize {
        self.shape.first().copied().unwrap_or(1)
    }
    /// 行の数（shape[1..] の積。専門家も全部並べた数）
    pub fn n_rows(&self) -> usize {
        self.shape.iter().skip(1).product()
    }
    pub fn n_elems(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn row_bytes(&self) -> usize {
        ggml_nbytes(self.ggml_type, &self.shape[..1]).unwrap()
    }
    /// 行 r0..r0+n のバイト
    pub fn rows(&self, r0: usize, n: usize) -> &'a [u8] {
        let rb = self.row_bytes();
        assert!(r0.checked_add(n).map_or(false, |e| e <= self.n_rows()), "{}: 行 {r0}+{n} が {} 行を超える", self.name, self.n_rows());
        &self.bytes[r0 * rb..(r0 + n) * rb]
    }
    /// いちばん外の次元の i 番目（専門家の 3 次元なら専門家 i の [k, n]）
    pub fn slab(&self, i: usize) -> TensorView<'a> {
        let (last, inner) = self.shape.split_last().expect("0 次元");
        assert!(i < *last, "{}: {i} が外の次元 {last} を超える", self.name);
        let sz = self.bytes.len() / last;
        TensorView { shape: inner, bytes: &self.bytes[i * sz..(i + 1) * sz], ..*self }
    }
}

/// 分割された束をまとめたモデル
pub struct Model {
    pub shards: Vec<Shard>,
    /// 束の順、束の中はファイルの順
    pub tensors: Vec<TensorInfo>,
    by_name: HashMap<String, usize>,
    /// 層 i → その層のテンソル（tensors の添字）
    by_layer: HashMap<usize, Vec<usize>>,
    /// メタデータ（最初の束のもの）の添字
    meta_idx: HashMap<String, usize>,
}

/// "…-00002-of-00004.gguf" → ("…", 2, 4)
fn split_name(path: &Path) -> Option<(String, usize, usize)> {
    let s = path.to_str()?.strip_suffix(".gguf")?;
    let (head, count) = s.rsplit_once("-of-")?;
    let (prefix, no) = head.rsplit_once('-')?;
    if no.len() != 5 || count.len() != 5 {
        return None;
    }
    Some((prefix.to_owned(), no.parse().ok()?, count.parse().ok()?))
}

impl Model {
    /// GGUF を開く。split.count があれば、同じ名前の束を全部開く（どの束を渡してもよい）
    pub fn open(path: impl AsRef<Path>) -> Result<Model> {
        let path = path.as_ref();
        let mut tensors = vec![];
        let first = Shard::open(path, 0, &mut tensors)?;
        let count = match first.meta("split.count") {
            Some(v) => v.as_u64().ok_or_else(|| Error(format!("{}: split.count が整数でない: {v}", path.display())))? as usize,
            None => 1,
        };
        let mut shards = vec![];
        if count <= 1 {
            shards.push(first);
        } else {
            let (prefix, _, cnt) = split_name(path)
                .ok_or_else(|| Error(format!("{}: split.count={count} だが名前が …-0000N-of-0000M.gguf でない", path.display())))?;
            if cnt != count {
                return err(format!("{}: 名前の束の数 {cnt} と split.count {count} が違う", path.display()));
            }
            // 渡されたのが 1 つ目ならそのまま使う。2 つ目以降なら 1 つ目から開き直す
            let mut first = if split_name(path).map(|s| s.1) == Some(1) { Some(first) } else { None };
            if first.is_none() {
                tensors.clear();
            }
            for i in 0..count {
                let p = PathBuf::from(format!("{prefix}-{:05}-of-{count:05}.gguf", i + 1));
                let sh = match first.take() {
                    Some(sh) => sh,
                    None => Shard::open(&p, i, &mut tensors)?,
                };
                let no = sh.meta("split.no").and_then(|v| v.as_u64());
                let c = sh.meta("split.count").and_then(|v| v.as_u64());
                if no != Some(i as u64) || c != Some(count as u64) {
                    return err(format!("{}: split.no={no:?} split.count={c:?}（{i}/{count} のはず）", p.display()));
                }
                shards.push(sh);
            }
        }
        if let Some(n) = shards[0].meta("split.tensors.count").and_then(|v| v.as_u64()) {
            if n as usize != tensors.len() {
                return err(format!("split.tensors.count={n} だがテンソルは {} 個", tensors.len()));
            }
        }
        let mut by_name = HashMap::with_capacity(tensors.len());
        let mut by_layer: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, t) in tensors.iter().enumerate() {
            if by_name.insert(t.name.clone(), i).is_some() {
                return err(format!("テンソル {} が 2 つある", t.name));
            }
            if let Some(l) = t.name.strip_prefix("blk.").and_then(|s| s.split_once('.')).and_then(|(l, _)| l.parse().ok()) {
                by_layer.entry(l).or_default().push(i);
            }
        }
        let meta_idx = shards[0].meta.iter().enumerate().map(|(i, (k, _))| (k.clone(), i)).collect();
        Ok(Model { shards, tensors, by_name, by_layer, meta_idx })
    }

    // ---- メタデータ

    /// メタデータ（最初の束のもの）をファイルの順で
    pub fn meta_all(&self) -> &[(String, Value)] {
        &self.shards[0].meta
    }
    pub fn meta(&self, key: &str) -> Option<&Value> {
        self.meta_idx.get(key).map(|&i| &self.shards[0].meta[i].1)
    }
    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        self.meta(key)?.as_u32()
    }
    pub fn meta_u64(&self, key: &str) -> Option<u64> {
        self.meta(key)?.as_u64()
    }
    pub fn meta_i64(&self, key: &str) -> Option<i64> {
        self.meta(key)?.as_i64()
    }
    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        self.meta(key)?.as_f32()
    }
    pub fn meta_bool(&self, key: &str) -> Option<bool> {
        self.meta(key)?.as_bool()
    }
    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.meta(key)?.as_str()
    }
    pub fn meta_arr(&self, key: &str) -> Option<&[Value]> {
        self.meta(key)?.as_arr()
    }
    /// 整数の配列（1 つでも整数でなければ None）
    pub fn meta_arr_i64(&self, key: &str) -> Option<Vec<i64>> {
        self.meta_arr(key)?.iter().map(|v| v.as_i64()).collect()
    }
    pub fn meta_arr_u32(&self, key: &str) -> Option<Vec<u32>> {
        self.meta_arr(key)?.iter().map(|v| v.as_u32()).collect()
    }
    pub fn meta_arr_f32(&self, key: &str) -> Option<Vec<f32>> {
        self.meta_arr(key)?.iter().map(|v| v.as_f32()).collect()
    }
    pub fn meta_arr_str(&self, key: &str) -> Option<Vec<&str>> {
        self.meta_arr(key)?.iter().map(|v| v.as_str()).collect()
    }

    /// general.architecture（qwen4exp など）
    pub fn arch(&self) -> &str {
        self.meta_str("general.architecture").unwrap_or("")
    }
    /// "{arch}.{key}" の値（ハイパーパラメータ）
    pub fn hp(&self, key: &str) -> Option<&Value> {
        self.meta(&format!("{}.{key}", self.arch()))
    }
    pub fn hp_u32(&self, key: &str) -> Option<u32> {
        self.hp(key)?.as_u32()
    }
    pub fn hp_f32(&self, key: &str) -> Option<f32> {
        self.hp(key)?.as_f32()
    }
    pub fn hp_arr_i64(&self, key: &str) -> Option<Vec<i64>> {
        self.hp(key)?.as_arr()?.iter().map(|v| v.as_i64()).collect()
    }

    // ---- テンソル

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.by_name.get(name).map(|&i| &self.tensors[i])
    }
    pub fn view<'a>(&'a self, t: &'a TensorInfo) -> TensorView<'a> {
        let d = self.shards[t.shard].data();
        TensorView { name: &t.name, ggml_type: t.ggml_type, qtype: t.qtype, shape: &t.shape, bytes: &d[t.offset..t.offset + t.size] }
    }
    pub fn get(&self, name: &str) -> Option<TensorView<'_>> {
        self.info(name).map(|t| self.view(t))
    }
    /// 無ければ止める
    pub fn tensor(&self, name: &str) -> TensorView<'_> {
        self.get(name).unwrap_or_else(|| panic!("テンソルが無い: {name}"))
    }
    /// 全部の大きさ（バイト）
    pub fn total_bytes(&self) -> usize {
        self.tensors.iter().map(|t| t.size).sum()
    }

    // ---- 層

    /// {arch}.block_count（無ければ blk.N の最大＋1）
    pub fn n_layers(&self) -> usize {
        self.hp_u32("block_count").map(|n| n as usize).unwrap_or_else(|| self.by_layer.keys().max().map_or(0, |m| m + 1))
    }
    pub fn layer(&self, i: usize) -> Layer<'_> {
        Layer { model: self, idx: i }
    }
}

/// 層 i（blk.i.*）のテンソル
#[derive(Clone, Copy)]
pub struct Layer<'a> {
    pub model: &'a Model,
    pub idx: usize,
}

impl<'a> Layer<'a> {
    /// "attn_qkv.weight" → blk.i.attn_qkv.weight
    pub fn get(&self, suffix: &str) -> Option<TensorView<'a>> {
        self.model.get(&format!("blk.{}.{suffix}", self.idx))
    }
    pub fn tensor(&self, suffix: &str) -> TensorView<'a> {
        self.model.tensor(&format!("blk.{}.{suffix}", self.idx))
    }
    pub fn has(&self, suffix: &str) -> bool {
        self.get(suffix).is_some()
    }
    /// この層のテンソル（ファイルの順）
    pub fn tensors(&self) -> impl Iterator<Item = TensorView<'a>> + 'a {
        let m = self.model;
        m.by_layer.get(&self.idx).into_iter().flatten().map(move |&i| m.view(&m.tensors[i]))
    }
    /// blk.i. を除いた名前
    pub fn names(&self) -> Vec<&'a str> {
        let p = format!("blk.{}.", self.idx);
        self.tensors().map(|t| t.name.strip_prefix(p.as_str()).unwrap()).collect()
    }
    /// この層の大きさ（バイト）
    pub fn bytes(&self) -> usize {
        self.tensors().map(|t| t.bytes.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_names() {
        let p = Path::new("/w/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf");
        assert_eq!(split_name(p), Some(("/w/Qwen3.8-Flash-Next-UD-Q4_K_XL".into(), 2, 4)));
        assert_eq!(split_name(Path::new("/w/mmproj-F16.gguf")), None);
    }

    #[test]
    fn nbytes() {
        assert_eq!(ggml_nbytes(14, &[2560, 248320]), Some(2560 / 256 * 210 * 248320));
        assert_eq!(ggml_nbytes(42, &[640, 2560, 512]), Some(640 / 64 * 18 * 2560 * 512));
        assert_eq!(ggml_nbytes(12, &[100]), None);
        assert_eq!(ggml_nbytes(0, &[48]), Some(192));
    }

    /// 小さな GGUF を作って読む（値の型を全部・配列の入れ子・テンソル 2 つ）
    #[test]
    fn roundtrip() {
        fn s(b: &mut Vec<u8>, x: &str) {
            b.extend((x.len() as u64).to_le_bytes());
            b.extend(x.as_bytes());
        }
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(2u64.to_le_bytes()); // テンソル
        b.extend(5u64.to_le_bytes()); // kv
        s(&mut b, "general.architecture");
        b.extend(8u32.to_le_bytes());
        s(&mut b, "toy");
        s(&mut b, "toy.block_count");
        b.extend(4u32.to_le_bytes());
        b.extend(1u32.to_le_bytes());
        s(&mut b, "toy.eps");
        b.extend(6u32.to_le_bytes());
        b.extend(1e-6f32.to_le_bytes());
        s(&mut b, "toy.big");
        b.extend(9u32.to_le_bytes());
        b.extend(11u32.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        b.extend((-3i64).to_le_bytes());
        b.extend(23703573157769i64.to_le_bytes());
        s(&mut b, "toy.nest");
        b.extend(9u32.to_le_bytes());
        b.extend(9u32.to_le_bytes());
        b.extend(1u64.to_le_bytes());
        b.extend(8u32.to_le_bytes());
        b.extend(1u64.to_le_bytes());
        s(&mut b, "x");
        // テンソル: blk.0.w F32 [8, 2]、output Q8_0 [32, 1]
        s(&mut b, "blk.0.w");
        b.extend(2u32.to_le_bytes());
        b.extend(8u64.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        b.extend(0u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        s(&mut b, "output");
        b.extend(1u32.to_le_bytes());
        b.extend(32u64.to_le_bytes());
        b.extend(8u32.to_le_bytes());
        b.extend(64u64.to_le_bytes());
        while b.len() % 32 != 0 {
            b.push(0);
        }
        let ds = b.len();
        for i in 0..16 {
            b.extend((i as f32).to_le_bytes());
        }
        b.extend([7u8; 34]);
        let p = std::env::temp_dir().join(format!("gguf-test-{}.gguf", std::process::id()));
        std::fs::write(&p, &b).unwrap();
        let m = Model::open(&p).unwrap();
        std::fs::remove_file(&p).unwrap();
        assert_eq!(m.arch(), "toy");
        assert_eq!(m.n_layers(), 1);
        assert_eq!(m.hp_f32("eps"), Some(1e-6));
        assert_eq!(m.hp_arr_i64("big"), Some(vec![-3, 23703573157769]));
        assert_eq!(m.meta_arr("toy.nest").unwrap()[0].as_arr().unwrap()[0].as_str(), Some("x"));
        assert_eq!(m.shards[0].data_start, ds);
        let w = m.layer(0).tensor("w");
        assert_eq!((w.qt(), w.k(), w.n_rows(), w.row_bytes()), (QType::F32, 8, 2, 32));
        assert_eq!(&w.rows(1, 1)[..4], &8f32.to_le_bytes());
        assert_eq!(w.slab(1).bytes, w.rows(1, 1));
        assert_eq!(m.layer(0).names(), vec!["w"]);
        let o = m.tensor("output");
        assert_eq!((o.qt(), o.bytes.len(), o.bytes[0]), (QType::Q8_0, 34, 7));
    }
}
