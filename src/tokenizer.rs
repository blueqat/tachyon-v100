//! 語の切り分け（HF の tokenizer.json を tokenizers で読む）と、少しずつ文字に戻すこと。
//! Qwen3.8 は byte-level BPE（GPT-2 と同じ 256 文字の置き換え）。1 語が UTF-8 の途中で切れるので、
//! 戻すときは文字が揃うまでバイトを持っておく（DecodeStream）。
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// 1 つの持ち主（アカウント）が覚えておく塊の大きさの合計の既定（超えたらその持ち主の分だけ全部忘れる）
pub const CACHE_BYTES_PER_SCOPE: usize = 8 << 20;
/// 覚える持ち主の数の上限の既定（超えた新しい持ち主は覚えない。⛔ ほかの持ち主の覚えは決して捨てない）
pub const CACHE_MAX_SCOPES: usize = 256;
/// これより短い塊は覚えない（切り分けるほうが速い）
const CACHE_MIN: usize = 256;
/// これより長い塊は切ってよい所で分けて並べて切り分ける
const PAR_MIN: usize = 48 << 10;
const PAR_CHUNK: usize = 16 << 10;

/// 1 つの持ち主の覚え。⛔ 持ち主をまたいで引かない・捨てない（他の人の文を、当たりの速さから推し量らせないため。
/// 2026-11-10 の見直しの重大な指摘: 前は 64MB の 1 つの表を全部のアカウントで共有していた）
#[derive(Default)]
struct ScopeCache {
    map: HashMap<String, Vec<u32>>,
    bytes: usize,
    hits: u64,
    misses: u64,
    /// 枠を超えて全部忘れた回数
    resets: u64,
}

/// 持ち主ごとの覚えの数（試し用。中身は出さない）
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
    pub bytes: usize,
    pub resets: u64,
}

/// 持ち主 → その覚え（外の表は持ち主を引くときだけ握る。持ち主ごとに別の鍵なので、他の持ち主の切り分けを待たない）
struct Caches {
    scopes: Mutex<HashMap<String, Arc<Mutex<ScopeCache>>>>,
    bytes_per_scope: usize,
    max_scopes: usize,
}

impl Default for Caches {
    fn default() -> Self {
        Caches { scopes: Mutex::new(HashMap::new()), bytes_per_scope: CACHE_BYTES_PER_SCOPE, max_scopes: CACHE_MAX_SCOPES }
    }
}

/// 語の数の上限つきの切り分けの誤り
#[derive(Debug)]
pub enum EncodeError {
    /// 上限を超えた（at_least 語以上。そこで切り分けをやめた）
    TooLong { at_least: usize },
    Failed(Error),
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncodeError::TooLong { at_least } => write!(f, "語の数が上限を超えた（{at_least} 語以上）"),
            EncodeError::Failed(e) => write!(f, "{e}"),
        }
    }
}

/// 決まった役の語の番号（tokenizer.json の added_tokens から引く）
#[derive(Clone, Copy, Debug)]
pub struct SpecialIds {
    pub endoftext: u32,
    pub im_start: u32,
    pub im_end: u32,
    pub think: u32,
    pub think_end: u32,
    pub tool_call: u32,
    pub tool_call_end: u32,
    pub tool_response: u32,
    pub tool_response_end: u32,
    pub vision_start: u32,
    pub vision_end: u32,
    pub image_pad: u32,
    pub video_pad: u32,
}

/// Inkling-2（Thinking Machines）の役の語。Qwen の語（<|im_start|> など）は無い
#[derive(Clone, Copy, Debug)]
pub struct InkIds {
    pub message_user: u32,
    pub message_model: u32,
    pub message_system: u32,
    pub message_tool: u32,
    pub content_text: u32,
    pub content_thinking: u32,
    pub content_tool_error: u32,
    pub content_invoke_tool_json: u32,
    pub content_invoke_tool_text: u32,
    pub end_message: u32,
    pub end_sampling: u32,
}

impl InkIds {
    /// 回の頭の語（頭の使い回しの区切り）
    pub fn is_message_start(&self, t: u32) -> bool {
        t == self.message_user || t == self.message_model || t == self.message_system || t == self.message_tool
    }
}

pub struct Tokenizer {
    hf: tokenizers::Tokenizer,
    /// 文を区切って切り分けてよいか（added token の中に「<」・改行・数字が無いこと。Qwen3.8 は満たす）
    can_split: bool,
    /// 持ち主（アカウント）ごとに、<|im_start|> から始まる塊（会話の 1 回分）→ 語の番号。前の回は毎回同じなので使い回す
    cache: Caches,
    /// id → 文字に戻したときのバイト列（byte-level の置き換えを戻した後）
    bytes: Vec<Box<[u8]>>,
    /// id → special な added token か（skip_special で飛ばす）
    special: Vec<bool>,
    pub ids: SpecialIds,
    /// Inkling の語（tokenizer に <|message_model|> があるとき）
    pub ink: Option<InkIds>,
    /// 生成を止める語（generation_config.json の eos_token_id。無ければ <|im_end|> と <|endoftext|>）
    pub eos: Vec<u32>,
}

pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// GGUF の語の表（tokenizer.ggml.*）を tokenizer.json と突き合わせた結果（check_gguf）
#[derive(Clone, Debug, PartialEq)]
pub struct GgufVocab {
    /// GGUF の語の数（= 埋め込み・出力の行数）。tokenizer.json より多い分は [PAD…]（型 UNUSED・文字は空）
    pub n_vocab: usize,
    /// tokenizer.ggml.pre（前の切り分けの種類。Qwen3.8 は qwen35）
    pub pre: String,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub pad: Option<u32>,
    /// tokenizer.ggml.add_bos_token（無い GGUF もある。llama.cpp は gpt2 型では既定で付けない）
    pub add_bos: Option<bool>,
    pub n_merges: usize,
}

/// GPT-2 の bytes_to_unicode の逆（文字 → バイト）。256 個
fn byte_level_inverse() -> std::collections::HashMap<char, u8> {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.iter().zip(&cs).map(|(&b, &c)| (char::from_u32(c).unwrap(), b as u8)).collect()
}

impl Tokenizer {
    /// tokenizer.json のあるフォルダ（generation_config.json があれば止める語もそこから）
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, Error> {
        let dir = dir.as_ref();
        let mut t = Self::from_file(dir.join("tokenizer.json"))?;
        if let Ok(s) = std::fs::read_to_string(dir.join("generation_config.json")) {
            let g: serde_json::Value = serde_json::from_str(&s)?;
            let eos: Vec<u32> = match &g["eos_token_id"] {
                serde_json::Value::Array(a) => a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect(),
                v => v.as_u64().map(|v| vec![v as u32]).unwrap_or_default(),
            };
            if !eos.is_empty() {
                t.eos = eos;
            }
        }
        Ok(t)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        let hf = tokenizers::Tokenizer::from_file(path)?;
        let added = hf.get_added_tokens_decoder();
        let n = hf.get_vocab(true).values().max().map(|&m| m as usize + 1).unwrap_or(0);
        let inv = byte_level_inverse();
        let mut bytes = Vec::with_capacity(n);
        let mut special = vec![false; n];
        for id in 0..n as u32 {
            let b: Vec<u8> = match hf.id_to_token(id) {
                // HF の ByteLevel decoder と同じ: 全部の文字が表にあれば戻す、1 つでも無ければ文字列のまま
                Some(tok) => tok.chars().map(|c| inv.get(&c).copied()).collect::<Option<Vec<u8>>>().unwrap_or_else(|| tok.into_bytes()),
                None => vec![],
            };
            bytes.push(b.into_boxed_slice());
            if let Some(a) = added.get(&id) {
                special[id as usize] = a.special;
            }
        }
        let ink = match hf.token_to_id("<|message_model|>") {
            None => None,
            Some(_) => {
                let g = |s: &str| hf.token_to_id(s).ok_or_else(|| format!("語が無い: {s}"));
                Some(InkIds {
                    message_user: g("<|message_user|>")?,
                    message_model: g("<|message_model|>")?,
                    message_system: g("<|message_system|>")?,
                    message_tool: g("<|message_tool|>")?,
                    content_text: g("<|content_text|>")?,
                    content_thinking: g("<|content_thinking|>")?,
                    content_tool_error: g("<|content_tool_error|>")?,
                    content_invoke_tool_json: g("<|content_invoke_tool_json|>")?,
                    content_invoke_tool_text: g("<|content_invoke_tool_text|>")?,
                    end_message: g("<|end_message|>")?,
                    end_sampling: g("<|content_model_end_sampling|>")?,
                })
            }
        };
        // Inkling には Qwen の役の語が無い: 無い語は u32::MAX（どの語とも当たらない）にする
        let is_ink = ink.is_some();
        let id = |s: &str| match hf.token_to_id(s) {
            Some(v) => Ok(v),
            None if is_ink => Ok(u32::MAX),
            None => Err(format!("語が無い: {s}")),
        };
        let ids = SpecialIds {
            endoftext: id("<|endoftext|>")?,
            im_start: id("<|im_start|>")?,
            im_end: id("<|im_end|>")?,
            think: id("<think>")?,
            think_end: id("</think>")?,
            tool_call: id("<tool_call>")?,
            tool_call_end: id("</tool_call>")?,
            tool_response: id("<tool_response>")?,
            tool_response_end: id("</tool_response>")?,
            vision_start: id("<|vision_start|>")?,
            vision_end: id("<|vision_end|>")?,
            image_pad: id("<|image_pad|>")?,
            video_pad: id("<|video_pad|>")?,
        };
        let eos = match &ink {
            Some(k) => vec![k.end_sampling],
            None => vec![ids.im_end, ids.endoftext],
        };
        // added token は正規化・前の切り分けより先に文から抜き出される（leftmost-longest）。
        // added token の中（2 文字目以降）に「<」・改行・数字が無ければ、<|im_start|> の手前と is_cut の所は
        // added token の途中にならない → そこで分けて切っても同じ番号になる
        let can_split = added.values().all(|a| {
            !a.content.contains('\n') && !a.content.chars().skip(1).any(|c| c == '<' || c.is_ascii_digit()) && !a.lstrip && !a.rstrip && !a.single_word && !a.normalized
        });
        Ok(Self { hf, can_split: can_split && !is_ink, cache: Caches::default(), bytes, special, ids, ink, eos })
    }

    /// 文字列 → 語の番号。chat_template で描いた文をそのまま渡す前提で、
    /// 文中の <|im_start|> などは 1 語として切る（HF の add_special_tokens=False・split_special_tokens=False と同じ）。
    /// 長い塊は改行・数字の所で分けて並べて切る。覚えは使わない（encode_cached）。結果は encode_plain と同じ
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        self.encode_limited(None, text, usize::MAX).map_err(EncodeError::into_error)
    }

    /// encode と同じ結果で、会話の 1 回分（<|im_start|> から）ごとに**その持ち主（scope）の覚え**を使い回す。
    /// ⛔ 持ち主はアカウント（鍵の名前）にする: 持ち主の違う依頼の覚えは決して引かない・捨てない
    pub fn encode_cached(&self, scope: &str, text: &str) -> Result<Vec<u32>, Error> {
        self.encode_limited(Some(scope), text, usize::MAX).map_err(EncodeError::into_error)
    }

    /// 語の数に上限をつけて切る: max_tokens を超えたら、その回（<|im_start|> から）の終わりで切り分けをやめて TooLong。
    /// 長すぎる依頼を全部切り分けてから断るのではなく、超えた所で断る（余計に切るのは 1 回分の塊まで）。
    /// scope が Some なら、その持ち主の覚えを使う（encode_cached と同じ）
    pub fn encode_limited(&self, scope: Option<&str>, text: &str, max_tokens: usize) -> Result<Vec<u32>, EncodeError> {
        if !self.can_split {
            let v = self.encode_plain(text).map_err(EncodeError::Failed)?;
            return if v.len() > max_tokens { Err(EncodeError::TooLong { at_least: v.len() }) } else { Ok(v) };
        }
        let cache = scope.and_then(|s| self.cache.scope(s, true));
        let mut out = Vec::with_capacity((text.len() / 3).min(max_tokens.saturating_add(1)).min(1 << 22));
        for part in split_before(text, "<|im_start|>") {
            let done = out.len();
            let part_ids = |budget: usize| {
                self.encode_part(part, budget).map_err(|e| match e {
                    EncodeError::TooLong { at_least } => EncodeError::TooLong { at_least: done + at_least },
                    e => e,
                })
            };
            let budget = max_tokens.saturating_sub(done);
            match cache.as_ref().filter(|_| part.len() >= CACHE_MIN) {
                None => out.extend(part_ids(budget)?),
                Some(c) => {
                    let hit = lock(c).get(part);
                    match hit {
                        Some(ids) => out.extend_from_slice(&ids),
                        None => {
                            // 上限を超えて途中でやめた塊は覚えない
                            let ids = part_ids(budget)?;
                            out.extend_from_slice(&ids);
                            lock(c).put(part, ids, self.cache.bytes_per_scope);
                        }
                    }
                }
            }
            if out.len() > max_tokens {
                return Err(EncodeError::TooLong { at_least: out.len() });
            }
        }
        Ok(out)
    }

    /// 区切らず tokenizers にそのまま渡す（答え合わせ用）
    pub fn encode_plain(&self, text: &str) -> Result<Vec<u32>, Error> {
        Ok(self.hf.encode_fast(text, false)?.get_ids().to_vec())
    }

    /// 長い塊は切ってよい所（is_cut）で分けて、スレッドで並べて切る
    /// budget 語を超えたら、並べて切る波（糸の数 × 4 切れ＝最大 512KB）の終わりでやめて TooLong（at_least はそこまでの語数）
    fn encode_part(&self, text: &str, budget: usize) -> Result<Vec<u32>, EncodeError> {
        if text.len() < PAR_MIN {
            let v = self.encode_plain(text).map_err(EncodeError::Failed)?;
            return if v.len() > budget { Err(EncodeError::TooLong { at_least: v.len() }) } else { Ok(v) };
        }
        let pieces = split_safe(text, PAR_CHUNK);
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).clamp(1, 8);
        let mut out = vec![];
        for wave in pieces.chunks(threads * 4) {
            let per = wave.len().div_ceil(threads);
            let res: Vec<Result<Vec<u32>, Error>> = std::thread::scope(|sc| {
                let hs: Vec<_> = wave
                    .chunks(per)
                    .map(|ps| {
                        sc.spawn(move || -> Result<Vec<u32>, Error> {
                            let mut v = vec![];
                            for p in ps {
                                v.extend(self.encode_plain(p)?);
                            }
                            Ok(v)
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("切り分けのスレッドが落ちた".into()))).collect()
            });
            for r in res {
                out.extend(r.map_err(EncodeError::Failed)?);
            }
            if out.len() > budget {
                return Err(EncodeError::TooLong { at_least: out.len() });
            }
        }
        Ok(out)
    }

    /// 切れ目の規則の確かめ用: 切ってよい所（is_cut）を全部切って、1 つずつ切り分けてつなぐ（encode_plain と同じになるはず）
    #[doc(hidden)]
    pub fn encode_every_cut(&self, text: &str) -> Result<Vec<u32>, Error> {
        let mut v = vec![];
        for p in split_safe(text, 1) {
            v.extend(self.encode_plain(p)?);
        }
        Ok(v)
    }

    /// その持ち主の覚えの数（無ければ全部 0）
    pub fn cache_stats(&self, scope: &str) -> CacheStats {
        match self.cache.scope(scope, false) {
            Some(c) => {
                let c = lock(&c);
                CacheStats { hits: c.hits, misses: c.misses, entries: c.map.len(), bytes: c.bytes, resets: c.resets }
            }
            None => CacheStats::default(),
        }
    }

    /// 覚えている持ち主の数
    pub fn cache_scopes(&self) -> usize {
        lock(&self.cache.scopes).len()
    }

    /// 覚えの枠（持ち主ごとのバイト数・持ち主の数。0 なら覚えない）。起動のときに 1 回
    pub fn set_cache_limits(&mut self, bytes_per_scope: usize, max_scopes: usize) {
        self.cache.bytes_per_scope = bytes_per_scope;
        self.cache.max_scopes = max_scopes;
    }

    /// 全部の持ち主の覚えを忘れる
    pub fn clear_cache(&self) {
        lock(&self.cache.scopes).clear();
    }

    /// tokenizers の BPE が持つ語ごとの覚え（糸ごとに持ち、全部の依頼で共有）を使うか。
    /// ⛔ 口では切る: 糸を使い回すので、別のアカウントが最近使った語だけ少し速く切れる（1 語あたり µs の時間差。
    /// 測れるとは思えないが、アカウントをまたぐ状態を残さない）。tokcheck などの道具は既定（使う）のまま
    pub fn set_word_cache(&mut self, on: bool) {
        let mut m = self.hf.get_model().clone();
        m.resize_cache(if on { 10_000 } else { 0 });
        self.hf.with_model(m);
    }

    /// 語の番号 → 文字列（壊れた UTF-8 は U+FFFD。HF の decode と同じ）
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut b = vec![];
        for &id in ids {
            if !(skip_special && self.is_special(id)) {
                b.extend_from_slice(self.token_bytes(id));
            }
        }
        String::from_utf8_lossy(&b).into_owned()
    }

    /// 1 語のバイト列（知らない番号は空）
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        self.bytes.get(id as usize).map(|b| &b[..]).unwrap_or(&[])
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special.get(id as usize).copied().unwrap_or(false)
    }

    pub fn is_eos(&self, id: u32) -> bool {
        self.eos.contains(&id)
    }

    pub fn token_to_id(&self, s: &str) -> Option<u32> {
        self.hf.token_to_id(s)
    }

    /// 番号の数（added token を含む。Qwen3.8 は 248077）。埋め込み・出力の行数 248320 より小さく、
    /// 248077〜248319 は GGUF では [PAD…]（文字は空）→ サンプリングでは選ばない（logits を -inf に）
    pub fn vocab_size(&self) -> usize {
        self.bytes.len()
    }

    pub fn hf(&self) -> &tokenizers::Tokenizer {
        &self.hf
    }

    pub fn decode_stream(&self, skip_special: bool) -> DecodeStream<'_> {
        DecodeStream { tok: self, pending: vec![], skip_special }
    }

    /// GGUF の語の表が tokenizer.json と同じか確かめる（違えば誤り。1 秒ほど）。見ること:
    ///   語の文字が番号ごとに同じ・merges が順まで同じ・語の型（special な added → CONTROL、
    ///   ほかの added → CONTROL か USER_DEFINED（<|…|> の形は CONTROL になる）、残り NORMAL、余りの [PAD…] は UNUSED）・
    ///   bos/eos/pad が本物の語で、eos が止める語（self.eos）に入っている
    pub fn check_gguf(&self, m: &crate::gguf::Model) -> Result<GgufVocab, Error> {
        const NORMAL: i64 = 1;
        const CONTROL: i64 = 3;
        const USER: i64 = 4;
        const UNUSED: i64 = 5;
        let model = m.meta_str("tokenizer.ggml.model").unwrap_or("");
        if model != "gpt2" {
            return Err(format!("tokenizer.ggml.model が gpt2 でない（{model:?}）").into());
        }
        let toks = m.meta_arr_str("tokenizer.ggml.tokens").ok_or("tokenizer.ggml.tokens が無い")?;
        let types = m.meta_arr_i64("tokenizer.ggml.token_type").ok_or("tokenizer.ggml.token_type が無い")?;
        let n = self.vocab_size();
        if toks.len() < n || types.len() != toks.len() {
            return Err(format!("GGUF の語の数 {}（型 {}）が tokenizer.json の {n} より少ない・合わない", toks.len(), types.len()).into());
        }
        let added = self.hf.get_added_tokens_decoder();
        let mut bad = vec![];
        for (id, (&t, &ty)) in toks.iter().zip(&types).enumerate() {
            let want: &[i64] = match added.get(&(id as u32)) {
                _ if id >= n => &[UNUSED],
                Some(a) if a.special => &[CONTROL],
                Some(_) => &[CONTROL, USER],
                None => &[NORMAL],
            };
            let hf = self.hf.id_to_token(id as u32);
            if (id < n && hf.as_deref() != Some(t)) || !want.contains(&ty) {
                bad.push(format!("語 {id}: GGUF {t:?}（型 {ty}）／tokenizer.json {hf:?}"));
            }
        }
        // merges（tokenizer.json は "a b" か ["a", "b"]。tokenizers は順位の順に書き出す）
        let mj = serde_json::to_value(self.hf.get_model())?;
        let hm: Vec<String> = mj["merges"]
            .as_array()
            .ok_or("tokenizer.json に merges が無い")?
            .iter()
            .map(|v| match v {
                serde_json::Value::Array(p) => p.iter().map(|x| x.as_str().unwrap_or("")).collect::<Vec<_>>().join(" "),
                v => v.as_str().unwrap_or("").to_string(),
            })
            .collect();
        let gm = m.meta_arr_str("tokenizer.ggml.merges").ok_or("tokenizer.ggml.merges が無い")?;
        if gm.len() != hm.len() {
            bad.push(format!("merges の数: GGUF {}／tokenizer.json {}", gm.len(), hm.len()));
        } else if let Some(i) = gm.iter().zip(&hm).position(|(a, b)| *a != b.as_str()) {
            bad.push(format!("merges {i} 番目: GGUF {:?}／tokenizer.json {:?}", gm[i], hm[i]));
        }
        let v = GgufVocab {
            n_vocab: toks.len(),
            pre: m.meta_str("tokenizer.ggml.pre").unwrap_or("").to_string(),
            bos: m.meta_u32("tokenizer.ggml.bos_token_id"),
            eos: m.meta_u32("tokenizer.ggml.eos_token_id"),
            pad: m.meta_u32("tokenizer.ggml.padding_token_id"),
            add_bos: m.meta_bool("tokenizer.ggml.add_bos_token"),
            n_merges: gm.len(),
        };
        for (k, id) in [("bos", v.bos), ("eos", v.eos), ("pad", v.pad)] {
            if id.is_some_and(|i| i as usize >= n) {
                bad.push(format!("{k} {id:?} が本物の語でない（{n} 以上）"));
            }
        }
        if let Some(e) = v.eos.filter(|e| !self.is_eos(*e)) {
            bad.push(format!("GGUF の eos {e} が止める語 {:?} に無い", self.eos));
        }
        if !bad.is_empty() {
            return Err(format!("GGUF の語の表が tokenizer.json と違う（{} か所）: {}", bad.len(), bad[..bad.len().min(5)].join("; ")).into());
        }
        Ok(v)
    }
}

impl EncodeError {
    fn into_error(self) -> Error {
        match self {
            EncodeError::Failed(e) => e,
            e @ EncodeError::TooLong { .. } => e.to_string().into(),
        }
    }
}

/// 毒の付いた鍵でも中身はそのまま使う（別の糸が落ちても止めない）
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Caches {
    /// その持ち主の覚え。make なら無ければ作る（持ち主の数が上限なら作らない＝覚えずに切る。ほかの持ち主は捨てない）
    fn scope(&self, s: &str, make: bool) -> Option<Arc<Mutex<ScopeCache>>> {
        let mut m = lock(&self.scopes);
        if let Some(c) = m.get(s) {
            return Some(c.clone());
        }
        if !make || m.len() >= self.max_scopes || self.bytes_per_scope == 0 {
            return None;
        }
        let c = Arc::new(Mutex::new(ScopeCache::default()));
        m.insert(s.to_string(), c.clone());
        Some(c)
    }
}

impl ScopeCache {
    fn get(&mut self, k: &str) -> Option<Vec<u32>> {
        let r = self.map.get(k).cloned();
        if r.is_some() {
            self.hits += 1;
        } else {
            self.misses += 1;
        }
        r
    }

    /// 覚える。枠を超えるなら、この持ち主の分だけ全部忘れてから（ほかの持ち主には触らない）
    fn put(&mut self, k: &str, v: Vec<u32>, budget: usize) {
        let n = k.len() + v.len() * 4;
        if n > budget / 4 {
            return;
        }
        if self.bytes + n > budget {
            self.map.clear();
            self.bytes = 0;
            self.resets += 1;
        }
        self.bytes += n;
        self.map.insert(k.to_string(), v);
    }
}

/// pat の手前で分ける（pat は次の塊の頭に付く）
fn split_before<'t>(text: &'t str, pat: &str) -> Vec<&'t str> {
    let mut out = vec![];
    let mut start = 0;
    for (i, _) in text.match_indices(pat) {
        if i > start {
            out.push(&text[start..i]);
            start = i;
        }
    }
    out.push(&text[start..]);
    out
}

fn space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// prev と c の間で切ってよいか。前の切り分けの正規表現の 1 つの塊がそこを越えず、
/// 先読み（\s+(?!\S)）の結果も変わらず、NFC も変わらない所だけ:
///   ・改行の直後で、次が空白でない（\s*[\r\n]+ が改行までで終わる）
///   ・ASCII の数字の手前で、前が空白でない（\p{N} は 1 文字ずつの塊、数字と合わさる文字は無い）
fn is_cut(prev: char, c: char) -> bool {
    (prev == '\n' && !space(c)) || (c.is_ascii_digit() && !space(prev))
}

/// だいたい size バイトずつ、is_cut の所で分ける
fn split_safe(text: &str, size: usize) -> Vec<&str> {
    let mut out = vec![];
    let mut start = 0;
    let mut from = size;
    while from < text.len() {
        while !text.is_char_boundary(from) {
            from += 1;
        }
        let Some(prev) = text[..from].chars().next_back() else { break };
        let mut prev = prev;
        let Some(p) = text[from..].char_indices().find_map(|(i, c)| {
            let cut = is_cut(prev, c);
            prev = c;
            cut.then_some(from + i)
        }) else {
            break;
        };
        out.push(&text[start..p]);
        start = p;
        from = p + size;
    }
    out.push(&text[start..]);
    out
}

/// 1 語ずつ文字に戻す。UTF-8 の文字が途中で切れている間は出さずに持っておく。
/// step の出力を全部つなげて flush を足すと、decode(全部) と同じ文字列になる
pub struct DecodeStream<'a> {
    tok: &'a Tokenizer,
    pending: Vec<u8>,
    skip_special: bool,
}

impl DecodeStream<'_> {
    /// 1 語足して、確定した文字を返す（まだ揃わなければ空）
    pub fn step(&mut self, id: u32) -> String {
        if self.skip_special && self.tok.is_special(id) {
            return String::new();
        }
        self.pending.extend_from_slice(self.tok.token_bytes(id));
        let mut out = String::new();
        let mut done = 0;
        loop {
            match std::str::from_utf8(&self.pending[done..]) {
                Ok(s) => {
                    out.push_str(s);
                    done = self.pending.len();
                    break;
                }
                Err(e) => {
                    let ok = done + e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[done..ok]).unwrap());
                    match e.error_len() {
                        // 壊れたバイト: from_utf8_lossy と同じく 1 つの U+FFFD にして先へ
                        Some(n) => {
                            out.push('\u{FFFD}');
                            done = ok + n;
                        }
                        // 文字の途中で終わっている: 続きの語を待つ
                        None => {
                            done = ok;
                            break;
                        }
                    }
                }
            }
        }
        self.pending.drain(..done);
        out
    }

    /// 終わり: 持っているバイトを吐き出す（途中で切れた文字は U+FFFD）
    pub fn flush(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }

    /// まだ文字にしていないバイトの数
    pub fn pending_bytes(&self) -> usize {
        self.pending.len()
    }
}

#[cfg(test)]
mod tests {
    //! 切り分けの覚えがアカウントをまたがないこと（2026-11-10 の見直しの重大な指摘への答え）。
    //! 合成の tokenizer（1 バイト 1 語・tokenizer.json は src/server/fake.rs）で。GPU は使わない
    use super::*;
    use std::time::Instant;

    fn synth(tag: &str) -> Tokenizer {
        let dir = std::env::temp_dir().join(format!("tachyon-tokcache-{tag}-{}", std::process::id()));
        crate::server::fake::write_synthetic_tokenizer(&dir).unwrap();
        let mut t = Tokenizer::from_dir(&dir).unwrap();
        t.set_word_cache(false);
        t
    }

    /// 会話の 2 回分（どちらも覚える大きさ以上）。secret が他の人の文
    fn convo(secret: &str, n: usize) -> String {
        format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n", "共有の前置き。".repeat(n), secret.repeat(n))
    }

    #[test]
    fn cache_never_crosses_accounts() {
        let mut tok = synth("cross");
        tok.set_cache_limits(64 << 10, 4);
        let t = convo("顧客 A の秘密の議事録。", 40);
        let want = tok.encode_plain(&t).unwrap();
        // A が切る: 2 回分とも外れ（覚える）
        assert_eq!(tok.encode_cached("acct-a", &t).unwrap(), want);
        let a1 = tok.cache_stats("acct-a");
        assert_eq!((a1.hits, a1.misses, a1.entries), (0, 2, 2));
        // B が同じ文を切る: A の覚えには 1 つも当たらない（B の数は、A がいない場合と同じ）
        assert_eq!(tok.encode_cached("acct-b", &t).unwrap(), want);
        let b1 = tok.cache_stats("acct-b");
        assert_eq!((b1.hits, b1.misses, b1.entries), (0, 2, 2), "B が A の覚えに当たった");
        assert_eq!(tok.cache_stats("acct-a"), a1, "B の切り分けで A の覚えが変わった");
        let alone = synth("alone");
        alone.encode_cached("acct-b", &t).unwrap();
        assert_eq!(alone.cache_stats("acct-b"), b1, "B の覚えの動きが、A がいるかで変わる");
        // A の 2 回目は自分の覚えに当たる（覚えそのものは効いている）
        assert_eq!(tok.encode_cached("acct-a", &t).unwrap(), want);
        assert_eq!(tok.cache_stats("acct-a").hits, 2);
        // B が枠（64KB）を何度も超えるほど切っても、A の覚えは 1 つも捨てられない
        let a2 = tok.cache_stats("acct-a");
        for i in 0..40 {
            let x = convo(&format!("B の文 {i} 番。"), 60);
            assert_eq!(tok.encode_cached("acct-b", &x).unwrap(), tok.encode_plain(&x).unwrap());
        }
        assert!(tok.cache_stats("acct-b").resets >= 2, "B は自分の枠の中で何度も忘れたはず: {:?}", tok.cache_stats("acct-b"));
        assert_eq!(tok.cache_stats("acct-a"), a2, "B の切り分けで A の覚えが捨てられた");
        assert_eq!(tok.encode_cached("acct-a", &t).unwrap(), want);
        assert_eq!(tok.cache_stats("acct-a").hits, a2.hits + 2);
        // 持ち主の数の上限（4）を超えた新しい持ち主は覚えない。前からいる持ち主は捨てない
        for s in ["acct-c", "acct-d", "acct-e", "acct-f"] {
            assert_eq!(tok.encode_cached(s, &t).unwrap(), want);
        }
        assert_eq!(tok.cache_scopes(), 4);
        assert_eq!(tok.cache_stats("acct-e"), CacheStats::default());
        assert_eq!(tok.cache_stats("acct-a").entries, 2);
        // 覚えを使わない切り分け（encode）は、どの持ち主の覚えも見ない・作らない
        let before: Vec<CacheStats> = ["acct-a", "acct-b"].iter().map(|s| tok.cache_stats(s)).collect();
        assert_eq!(tok.encode(&t).unwrap(), want);
        assert_eq!(before, ["acct-a", "acct-b"].iter().map(|s| tok.cache_stats(s)).collect::<Vec<_>>());
    }

    /// 時間でも確かめる: A が覚えた文を B が切る時間は、B が初めて切る時間と同じくらい（当たりなら 1/10 以下になる）。
    /// 比べのために A 自身の 2 回目（当たり）が速いことも見る＝この試しは当たりを見分けられる
    #[test]
    fn cache_timing_does_not_leak_across_accounts() {
        let tok = synth("timing");
        let text = |i: usize, who: &str| convo(&format!("秘密 {who}{i}: 来期の値上げは 12%。"), 6000);
        let time = |scope: &str, s: &str| {
            let t0 = Instant::now();
            std::hint::black_box(tok.encode_cached(scope, s).unwrap());
            t0.elapsed().as_secs_f64()
        };
        let best = |f: &dyn Fn(usize) -> f64| (0..3).map(f).fold(f64::MAX, f64::min);
        // B が初めて切る（だれも覚えていない文）
        let cold = best(&|i| time("acct-b", &text(i, "cold")));
        // A が先に切った文を B が切る
        let after_a = best(&|i| {
            let s = text(i, "warm");
            time("acct-a", &s);
            time("acct-b", &s)
        });
        // A 自身の 2 回目
        let own_hit = best(&|i| time("acct-a", &text(i, "warm")));
        eprintln!(
            "切り分け（約 {}KB）: B が初めて {:.2}ms・A の後の B {:.2}ms・A の 2 回目 {:.3}ms",
            text(0, "x").len() / 1000,
            cold * 1e3,
            after_a * 1e3,
            own_hit * 1e3
        );
        assert!(own_hit < cold / 10.0, "当たりが速くない（この試しでは当たりを見分けられない）");
        assert!(after_a > cold / 3.0, "A が覚えた文を B が速く切れた（アカウントをまたいで当たっている）");
        // B の覚えの数（当たりは B 自身が前に送った共通の前置きの分だけ）は、A がいない tokenizer で B だけが同じ順に送ったときと同じ
        let alone = synth("timing-alone");
        for i in 0..3 {
            alone.encode_cached("acct-b", &text(i, "cold")).unwrap();
        }
        for i in 0..3 {
            alone.encode_cached("acct-b", &text(i, "warm")).unwrap();
        }
        assert_eq!(tok.cache_stats("acct-b"), alone.cache_stats("acct-b"), "B の覚えの動きが A の依頼で変わった");
    }

    /// BPE の語の覚えを切ったときの手間（本物の tokenizer で測るだけ。TACHYON_HF_DIR=… cargo test --release --lib word_cache_cost -- --ignored --nocapture）
    #[test]
    #[ignore]
    fn word_cache_cost() {
        let dir = std::env::var("TACHYON_HF_DIR").expect("TACHYON_HF_DIR");
        let text = std::env::var("TACHYON_TEXT").map(|p| std::fs::read_to_string(p).unwrap()).unwrap_or_else(|_| "The quick brown fox jumps over the lazy dog. 日本語の文も混ぜる。".repeat(4000));
        for on in [true, false] {
            let mut t = Tokenizer::from_dir(&dir).unwrap();
            t.set_word_cache(on);
            let mut best = f64::MAX;
            let mut n = 0;
            for _ in 0..3 {
                let t0 = Instant::now();
                // 1 回分の塊が 48KB 未満の会話（並べずに 1 本の糸で切る道＝語の覚えが効く道）
                for piece in text.as_bytes().chunks(40_000) {
                    n = t.encode(&String::from_utf8_lossy(piece)).unwrap().len();
                }
                best = best.min(t0.elapsed().as_secs_f64());
            }
            eprintln!("語の覚え {}: {:.1}KB を {:.1}ms（最後の塊 {n} 語）", if on { "あり" } else { "なし" }, text.len() as f64 / 1e3, best * 1e3);
        }
    }

    #[test]
    fn limited_stops_early() {
        let tok = synth("limit");
        let t = format!("{}{}", convo("短い。", 2), convo("長い長い本文。", 4000));
        let n = tok.encode_plain(&t).unwrap().len();
        assert_eq!(tok.encode_limited(None, &t, n).unwrap().len(), n);
        match tok.encode_limited(Some("acct-a"), &t, 100) {
            Err(EncodeError::TooLong { at_least }) => assert!(at_least > 100 && at_least <= n),
            r => panic!("上限で止まらない: {:?}", r.map(|v| v.len())),
        }
        // 1 つの大きな塊（<|im_start|> なし・2MB）でも、並べて切る波（最大 512KB）の所でやめる。
        // ⚠️ 切ってよい所（改行・数字）が無い 1 行は分けられないので全部切る（口では 1 つの message が 1MiB までなので、そこまで）
        let big = "長い長い本文。\n".repeat(100_000);
        let total = tok.encode(&big).unwrap().len();
        match tok.encode_limited(None, &big, 100) {
            Err(EncodeError::TooLong { at_least }) => assert!(at_least > 100 && at_least <= (600 << 10) && at_least < total, "{at_least} / {total}"),
            r => panic!("上限で止まらない: {:?}", r.map(|v| v.len())),
        }
        assert_eq!(tok.encode_limited(None, &big, total).unwrap(), tok.encode_plain(&big).unwrap());
    }
}
