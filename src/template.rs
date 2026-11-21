//! chat_template（Jinja）を minijinja で描くことと、モデルの出力を「考え・答え・道具の呼び出し」に分けること。
//! 描き方は HF transformers の apply_chat_template に合わせる:
//!   trim_blocks・lstrip_blocks、tojson は ensure_ascii=False・「, 」「: 」区切り（Python の json.dumps）、
//!   raise_exception、{% generation %} は素通し、str の startswith などは minijinja-contrib の pycompat。
//! 出力の分け方は tokenizer_config.json の response_template（Qwen3.8 の形）に合わせる。
use minijinja::value::{Kwargs, Serde, Value, ValueKind};
use minijinja::{Environment, ErrorKind};
use serde_json::{Map, Value as Json};
use std::collections::HashMap;
use std::fmt;
use std::path::Path;

/// Qwen3.8 の chat_template が受け付ける reasoning_effort（Flash-Next・27B で同じ。既定は xhigh）
pub const EFFORTS: [&str; 3] = ["xhigh", "medium", "low"];
/// Inkling-2 の chat_template の reasoning_effort（文字。数 0〜0.99 も受ける＝"num:<数>" で持つ）
pub const INK_EFFORTS: [&str; 7] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

/// テンプレートが Inkling の形（<|message_model|>）か。1 つのプロセスに 1 つのモデルなので全体で持つ
pub static INKLING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn is_inkling() -> bool {
    INKLING.load(std::sync::atomic::Ordering::Relaxed)
}

/// この口が受ける reasoning_effort の名前
pub fn efforts() -> &'static [&'static str] {
    if is_inkling() { &INK_EFFORTS } else { &EFFORTS }
}

/// reasoning_effort が受けられるか（Inkling は "num:0.5" の形の数も）
pub fn effort_ok(e: &str) -> bool {
    efforts().contains(&e) || (is_inkling() && e.strip_prefix("num:").and_then(|x| x.parse::<f64>().ok()).is_some_and(|x| (0.0..=0.99).contains(&x)))
}

/// Inkling で考えないか（"none"・数 0）
pub fn ink_effort_is_none(e: &str) -> bool {
    e == "none" || e.strip_prefix("num:").and_then(|x| x.parse::<f64>().ok()) == Some(0.0)
}

const NAME: &str = "chat";
/// raise_exception の印（minijinja の誤りの中から見分ける）
const RAISE: &str = "raise_exception: ";

#[derive(Debug, Clone, PartialEq)]
pub enum TemplateError {
    /// reasoning_effort が xhigh・medium・low 以外（API なら 400）
    BadEffort(String),
    /// テンプレートが raise_exception で止めた（依頼の誤り。API なら 400）
    Raised(String),
    /// 依頼の形がおかしい（kwargs の型、arguments が JSON の object でない など）
    BadRequest(String),
    /// テンプレートそのものの誤り（読めない・描けない）
    Template(String),
}

impl fmt::Display for TemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TemplateError::BadEffort(e) => write!(f, "reasoning_effort は {} のどれか（{e:?} は使えない）", efforts().join("・")),
            TemplateError::Raised(m) => write!(f, "{m}"),
            TemplateError::BadRequest(m) => write!(f, "{m}"),
            TemplateError::Template(m) => write!(f, "chat_template: {m}"),
        }
    }
}
impl std::error::Error for TemplateError {}

/// テンプレートに渡す値（OpenAI 互換の chat_template_kwargs に当たる）。None は「渡さない」（テンプレートの既定）
#[derive(Clone, Debug, Default)]
pub struct ChatOptions {
    pub add_generation_prompt: bool,
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<String>,
    pub preserve_thinking: Option<bool>,
    pub add_vision_id: Option<bool>,
    /// そのほかの kwargs（そのままテンプレートへ）
    pub extra: Map<String, Json>,
}

impl ChatOptions {
    /// chat_template_kwargs（JSON の object）から。null は「渡さない」扱い
    pub fn from_kwargs(kw: &Json, add_generation_prompt: bool) -> Result<Self, TemplateError> {
        let mut o = ChatOptions { add_generation_prompt, ..Default::default() };
        let Some(m) = kw.as_object() else {
            return if kw.is_null() { Ok(o) } else { Err(TemplateError::BadRequest("chat_template_kwargs は object".into())) };
        };
        let bad = |k: &str, t: &str| TemplateError::BadRequest(format!("chat_template_kwargs.{k} は {t}"));
        for (k, v) in m {
            if v.is_null() {
                continue;
            }
            match k.as_str() {
                "enable_thinking" => o.enable_thinking = Some(v.as_bool().ok_or_else(|| bad(k, "true/false"))?),
                "preserve_thinking" => o.preserve_thinking = Some(v.as_bool().ok_or_else(|| bad(k, "true/false"))?),
                "add_vision_id" => o.add_vision_id = Some(v.as_bool().ok_or_else(|| bad(k, "true/false"))?),
                "reasoning_effort" => {
                    o.reasoning_effort = Some(match v {
                        Json::Number(n) if is_inkling() => format!("num:{}", n.as_f64().unwrap_or(-1.0)),
                        _ => v.as_str().ok_or_else(|| bad(k, "文字列"))?.to_string(),
                    })
                }
                "messages" | "tools" | "add_generation_prompt" => return Err(bad(k, "ここでは渡せない")),
                _ => {
                    o.extra.insert(k.clone(), v.clone());
                }
            }
        }
        Ok(o)
    }

    /// 考えるか（テンプレートの既定は考える）
    pub fn thinking(&self) -> bool {
        self.enable_thinking != Some(false)
    }

    /// reasoning_effort の確かめ。テンプレートは考えるときしか見ないが、API では常に誤りにする
    pub fn validate(&self) -> Result<(), TemplateError> {
        match &self.reasoning_effort {
            Some(e) if !effort_ok(e) => Err(TemplateError::BadEffort(e.clone())),
            _ => Ok(()),
        }
    }
}

pub struct ChatTemplate {
    env: Environment<'static>,
}

impl ChatTemplate {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TemplateError> {
        let s = std::fs::read_to_string(path.as_ref()).map_err(|e| TemplateError::Template(format!("{}: {e}", path.as_ref().display())))?;
        Self::new(&s)
    }

    /// GGUF に入っているテンプレート（tokenizer.chat_template）。
    /// ⚠️ unsloth の本体の GGUF のは HF のと違う。HF の chat_template.jinja を正とする
    pub fn from_gguf(m: &crate::gguf::Model) -> Result<Self, TemplateError> {
        Self::new(m.meta_str("tokenizer.chat_template").ok_or_else(|| TemplateError::Template("GGUF に tokenizer.chat_template が無い".into()))?)
    }

    pub fn new(source: &str) -> Result<Self, TemplateError> {
        let terr = |e: minijinja::Error| TemplateError::Template(format!("{e:#}"));
        let mut env = Environment::new();
        env.set_syntax(minijinja::syntax::SyntaxConfig::builder().trim_blocks(true).lstrip_blocks(true).build().map_err(terr)?);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson);
        env.add_filter("trim", trim);
        env.add_function("raise_exception", |msg: String| -> Result<Value, minijinja::Error> {
            Err(minijinja::Error::new(ErrorKind::InvalidOperation, format!("{RAISE}{msg}")))
        });
        env.add_template_owned(NAME, strip_generation(source)).map_err(terr)?;
        if source.contains("<|message_model|>") {
            INKLING.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(Self { env })
    }

    /// messages（と tools）を描いて、モデルに渡す 1 つの文にする。
    /// messages は OpenAI 形を normalize_messages で直した物（tool_calls の arguments は object）
    pub fn render(&self, messages: &Json, tools: Option<&Json>, opt: &ChatOptions) -> Result<String, TemplateError> {
        opt.validate()?;
        self.render_raw(messages, tools, opt)
    }

    /// reasoning_effort を先に確かめずに描く（テンプレートそのままの振る舞い。答え合わせ用）
    pub fn render_raw(&self, messages: &Json, tools: Option<&Json>, opt: &ChatOptions) -> Result<String, TemplateError> {
        let mut ctx = opt.extra.clone();
        ctx.insert("messages".into(), messages.clone());
        ctx.insert("tools".into(), tools.cloned().unwrap_or(Json::Null));
        ctx.insert("documents".into(), Json::Null);
        ctx.insert("add_generation_prompt".into(), opt.add_generation_prompt.into());
        let mut put = |k: &str, v: Option<Json>| {
            if let Some(v) = v {
                ctx.insert(k.into(), v);
            }
        };
        put("enable_thinking", opt.enable_thinking.map(Json::from));
        put(
            "reasoning_effort",
            opt.reasoning_effort.clone().map(|e| match e.strip_prefix("num:").and_then(|x| x.parse::<f64>().ok()) {
                Some(x) => Json::from(x),
                None => Json::from(e),
            }),
        );
        put("preserve_thinking", opt.preserve_thinking.map(Json::from));
        put("add_vision_id", opt.add_vision_id.map(Json::from));
        let t = self.env.get_template(NAME).map_err(|e| TemplateError::Template(format!("{e:#}")))?;
        t.render(Value::from(Serde(&Json::Object(ctx)))).map_err(map_err)
    }
}

fn map_err(e: minijinja::Error) -> TemplateError {
    let mut cur: Option<&(dyn std::error::Error + 'static)> = Some(&e);
    while let Some(x) = cur {
        if let Some(m) = x.downcast_ref::<minijinja::Error>() {
            if let Some(msg) = m.detail().and_then(|d| d.strip_prefix(RAISE)) {
                return TemplateError::Raised(msg.to_string());
            }
        }
        cur = x.source();
    }
    TemplateError::Template(format!("{e:#}"))
}

/// HF の {% generation %}…{% endgeneration %}（学習用の印）は描く結果を変えないので、
/// 同じ空白の扱いのまま {% if true %}…{% endif %} に置き換える
fn strip_generation(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(i) = rest.find("{%") {
        let Some(j) = rest[i..].find("%}") else { break };
        let tag = &rest[i..i + j + 2];
        let inner = &tag[2..tag.len() - 2];
        let core = inner.trim_matches(|c: char| c == '-' || c == '+' || c.is_whitespace());
        out.push_str(&rest[..i]);
        match core {
            "generation" => out.push_str(&tag.replacen("generation", "if true", 1)),
            "endgeneration" => out.push_str(&tag.replacen("endgeneration", "endif", 1)),
            _ => out.push_str(tag),
        }
        rest = &rest[i + j + 2..];
    }
    out.push_str(rest);
    out
}

/// Python の str.isspace と同じ空白（Rust の is_whitespace に \x1c〜\x1f を足す）
fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// jinja2 の trim（= Python の str.strip）
fn trim(v: Value, chars: Option<String>) -> Value {
    let s = v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string());
    let t = match &chars {
        Some(cs) => s.trim_matches(|c| cs.contains(c)),
        None => s.trim_matches(py_space),
    };
    if v.is_safe() {
        Value::from_safe_string(t.to_string())
    } else {
        Value::from(t)
    }
}

/// HF の tojson: json.dumps(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False)
fn tojson(v: Value, kw: Kwargs) -> Result<Value, minijinja::Error> {
    let ensure_ascii = kw.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false);
    let sort_keys = kw.get::<Option<bool>>("sort_keys")?.unwrap_or(false);
    let indent: Option<String> = match kw.get::<Option<Value>>("indent")? {
        None => None,
        Some(i) if i.is_none() => None,
        Some(i) => Some(match i.as_str() {
            Some(s) => s.to_string(),
            None => " ".repeat(i.as_usize().ok_or_else(|| minijinja::Error::new(ErrorKind::InvalidOperation, "indent は数か文字列"))?),
        }),
    };
    let (item_sep, key_sep) = match kw.get::<Option<Vec<String>>>("separators")? {
        Some(s) if s.len() == 2 => (s[0].clone(), s[1].clone()),
        Some(_) => return Err(minijinja::Error::new(ErrorKind::InvalidOperation, "separators は 2 つ")),
        None if indent.is_some() => (",".into(), ": ".into()),
        None => (", ".into(), ": ".into()),
    };
    kw.assert_all_used()?;
    let mut w = PyJson { out: String::new(), ensure_ascii, sort_keys, indent, item_sep, key_sep };
    w.value(&v, 0)?;
    Ok(Value::from_safe_string(w.out))
}

/// Python の json.dumps と同じ文字を書く
struct PyJson {
    out: String,
    ensure_ascii: bool,
    sort_keys: bool,
    indent: Option<String>,
    item_sep: String,
    key_sep: String,
}

impl PyJson {
    fn newline(&mut self, depth: usize) {
        if let Some(ind) = &self.indent {
            self.out.push('\n');
            for _ in 0..depth {
                self.out.push_str(ind);
            }
        }
    }

    fn value(&mut self, v: &Value, depth: usize) -> Result<(), minijinja::Error> {
        match v.kind() {
            ValueKind::Undefined | ValueKind::None => self.out.push_str("null"),
            ValueKind::Bool => self.out.push_str(if v.is_true() { "true" } else { "false" }),
            ValueKind::Number => {
                if v.is_integer() {
                    self.out.push_str(&v.to_string());
                } else {
                    self.out.push_str(&py_float(f64::try_from(v.clone())?));
                }
            }
            ValueKind::String => self.string(v.as_str().unwrap_or_default()),
            ValueKind::Map => {
                let mut items: Vec<(String, Value)> = vec![];
                for k in v.try_iter()? {
                    let val = v.get_item(&k)?;
                    items.push((key_str(&k), val));
                }
                if self.sort_keys {
                    items.sort_by(|a, b| a.0.cmp(&b.0));
                }
                if items.is_empty() {
                    self.out.push_str("{}");
                    return Ok(());
                }
                self.out.push('{');
                for (i, (k, val)) in items.iter().enumerate() {
                    if i > 0 {
                        let s = self.item_sep.clone();
                        self.out.push_str(&s);
                    }
                    self.newline(depth + 1);
                    self.string(k);
                    let s = self.key_sep.clone();
                    self.out.push_str(&s);
                    self.value(val, depth + 1)?;
                }
                self.newline(depth);
                self.out.push('}');
            }
            ValueKind::Seq | ValueKind::Iterable => {
                let items: Vec<Value> = v.try_iter()?.collect();
                if items.is_empty() {
                    self.out.push_str("[]");
                    return Ok(());
                }
                self.out.push('[');
                for (i, val) in items.iter().enumerate() {
                    if i > 0 {
                        let s = self.item_sep.clone();
                        self.out.push_str(&s);
                    }
                    self.newline(depth + 1);
                    self.value(val, depth + 1)?;
                }
                self.newline(depth);
                self.out.push(']');
            }
            k => return Err(minijinja::Error::new(ErrorKind::InvalidOperation, format!("tojson: {k} は JSON にできない"))),
        }
        Ok(())
    }

    fn string(&mut self, s: &str) {
        self.out.push('"');
        for c in s.chars() {
            match c {
                '"' => self.out.push_str("\\\""),
                '\\' => self.out.push_str("\\\\"),
                '\n' => self.out.push_str("\\n"),
                '\r' => self.out.push_str("\\r"),
                '\t' => self.out.push_str("\\t"),
                '\u{8}' => self.out.push_str("\\b"),
                '\u{c}' => self.out.push_str("\\f"),
                c if (c as u32) < 0x20 || (self.ensure_ascii && (c as u32) > 0x7e) => {
                    let mut buf = [0u16; 2];
                    for u in c.encode_utf16(&mut buf) {
                        self.out.push_str(&format!("\\u{:04x}", u));
                    }
                }
                c => self.out.push(c),
            }
        }
        self.out.push('"');
    }
}

/// json.dumps が辞書の鍵を文字にする規則
fn key_str(k: &Value) -> String {
    match k.kind() {
        ValueKind::String => k.as_str().unwrap_or_default().to_string(),
        ValueKind::None | ValueKind::Undefined => "null".into(),
        ValueKind::Bool => (if k.is_true() { "true" } else { "false" }).into(),
        ValueKind::Number if !k.is_integer() => py_float(f64::try_from(k.clone()).unwrap_or(f64::NAN)),
        _ => k.to_string(),
    }
}

/// Python の repr(float)（json.dumps の数の書き方）
pub fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    let e = format!("{:e}", f.abs()); // 最短で戻る桁（"1.5e-5" など）
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let sign = if f.is_sign_negative() { "-" } else { "" };
    let body = if (-4..16).contains(&exp) {
        if exp >= 0 {
            let p = exp as usize + 1;
            if digits.len() <= p {
                format!("{digits}{}.0", "0".repeat(p - digits.len()))
            } else {
                format!("{}.{}", &digits[..p], &digits[p..])
            }
        } else {
            format!("0.{}{digits}", "0".repeat((-exp - 1) as usize))
        }
    } else {
        let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits.clone() };
        format!("{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    };
    format!("{sign}{body}")
}

/// OpenAI 形の messages をテンプレートの形に直す:
/// tool_calls の arguments（JSON の文字列）を object に、role "developer" を "system" に
pub fn normalize_messages(messages: &Json) -> Result<Json, TemplateError> {
    let bad = |m: String| TemplateError::BadRequest(m);
    let mut ms = messages.clone();
    let arr = ms.as_array_mut().ok_or_else(|| bad("messages は配列".into()))?;
    for m in arr.iter_mut() {
        let m = m.as_object_mut().ok_or_else(|| bad("message は object".into()))?;
        if m.get("role").and_then(Json::as_str) == Some("developer") {
            m.insert("role".into(), "system".into());
        }
        let Some(calls) = m.get_mut("tool_calls").and_then(Json::as_array_mut) else { continue };
        for c in calls {
            let f = if c.get("function").is_some_and(Json::is_object) { &mut c["function"] } else { c };
            let name = f.get("name").and_then(Json::as_str).unwrap_or("").to_string();
            if let Some(Json::String(s)) = f.get("arguments") {
                let v = if s.trim().is_empty() {
                    Json::Object(Map::new())
                } else {
                    match serde_json::from_str::<Json>(s) {
                        Ok(v @ Json::Object(_)) => v,
                        _ => return Err(bad(format!("tool_calls の {name} の arguments が JSON の object でない"))),
                    }
                };
                f["arguments"] = v;
            }
        }
    }
    Ok(ms)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaKind {
    Image,
    Video,
}

/// messages の中の画像・動画を、テンプレートが <|image_pad|>／<|video_pad|> を置く順に集める
pub fn collect_media(messages: &Json) -> Vec<(MediaKind, Json)> {
    let mut out = vec![];
    for m in messages.as_array().into_iter().flatten() {
        for it in m.get("content").and_then(Json::as_array).into_iter().flatten() {
            let Some(o) = it.as_object() else { continue };
            let ty = o.get("type").and_then(Json::as_str);
            if o.contains_key("image") || o.contains_key("image_url") || ty == Some("image") {
                out.push((MediaKind::Image, it.clone()));
            } else if o.contains_key("video") || ty == Some("video") {
                out.push((MediaKind::Video, it.clone()));
            }
        }
    }
    out
}

/// k 番目の pad 語を counts[k] 個に広げる（画像 1 枚が何語になるかは画像の側が決める）
pub fn expand_pads(ids: &[u32], pad: u32, counts: &[usize]) -> Result<Vec<u32>, String> {
    let n = ids.iter().filter(|&&i| i == pad).count();
    if n != counts.len() {
        return Err(format!("pad の数 {n} と画像の数 {} が合わない", counts.len()));
    }
    let mut out = Vec::with_capacity(ids.len() + counts.iter().sum::<usize>());
    let mut k = 0;
    for &i in ids {
        if i == pad {
            out.extend(std::iter::repeat(pad).take(counts[k]));
            k += 1;
        } else {
            out.push(i);
        }
    }
    Ok(out)
}

// ───────────── 出力を分ける ─────────────

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Map<String, Json>,
}

impl ToolCall {
    /// OpenAI 形（arguments は JSON の文字列）
    pub fn to_openai(&self, id: &str) -> Json {
        serde_json::json!({
            "id": id,
            "type": "function",
            "function": {"name": self.name, "arguments": Json::Object(self.arguments.clone()).to_string()},
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum OutEvent {
    Reasoning(String),
    Content(String),
    ToolCall(ToolCall),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParsedOutput {
    pub reasoning_content: String,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

impl ParsedOutput {
    pub fn add(&mut self, ev: OutEvent) {
        match ev {
            OutEvent::Reasoning(s) => self.reasoning_content.push_str(&s),
            OutEvent::Content(s) => self.content.push_str(&s),
            OutEvent::ToolCall(t) => self.tool_calls.push(t),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    Reasoning,
    Content,
    Tool,
    Done,
}

const THINK: &str = "<think>";
const THINK_END: &str = "</think>";
const TOOL: &str = "<tool_call>";
const TOOL_END: &str = "</tool_call>";
const ENDS: [&str; 2] = ["<|im_end|>", "<|endoftext|>"];

/// モデルの出力（文字に戻した物）を少しずつ受け取り、考え／答え／道具の呼び出しに分ける。
///   考え: <think> の後 〜 </think>（または <tool_call> の手前。<|im_end|> なら終わり）。前後の空白は捨てる
///   道具: <tool_call><function=名前><parameter=鍵>値</parameter>…</function></tool_call> を 1 つずつまとめて出す
///   答え: それ以外（<|im_end|>・<|endoftext|> で終わり）。前後の空白は捨てる
/// 印が 2 回の push にまたがっても大丈夫なように、印の頭かもしれない末尾は次まで持つ
pub struct OutputParser {
    mode: Mode,
    buf: String,
    /// 今の区切り（考え・答えのひと続き）で文字を出したか（先頭の空白を捨てる）
    started: bool,
    /// 末尾の空白（続きの文字が来たら出す、区切りが終われば捨てる）
    ws: String,
    /// まだ何も出していない（答えが <think> で始まったら考えに入る）
    fresh: bool,
    /// <tool_call> の手前の答えの状態（started・末尾の空白）。中身が呼び出しでなければ戻して続ける
    saved: Option<(bool, String)>,
    /// 道具名 → 引数名 → JSON schema の型
    types: HashMap<String, HashMap<String, Vec<String>>>,
}

impl OutputParser {
    /// in_reasoning: プロンプトが <think> を開いたまま終わっている（考えありの生成）
    pub fn new(in_reasoning: bool) -> Self {
        OutputParser {
            mode: if in_reasoning { Mode::Reasoning } else { Mode::Content },
            buf: String::new(),
            started: false,
            ws: String::new(),
            fresh: true,
            saved: None,
            types: HashMap::new(),
        }
    }

    /// 描いたプロンプトの終わりを見て決める
    pub fn for_prompt(prompt: &str) -> Self {
        let tail = &prompt[prompt.rfind("<|im_start|>").unwrap_or(0)..];
        Self::new(tail.rfind(THINK) > tail.rfind(THINK_END))
    }

    /// 道具の定義（OpenAI 形の tools）から引数の型を覚える。文字列以外の引数を JSON として読むため
    pub fn with_tools(mut self, tools: Option<&Json>) -> Self {
        for t in tools.and_then(Json::as_array).into_iter().flatten() {
            let f = t.get("function").unwrap_or(t);
            let Some(name) = f.get("name").and_then(Json::as_str) else { continue };
            let mut m = HashMap::new();
            for (k, p) in f.pointer("/parameters/properties").and_then(Json::as_object).into_iter().flatten() {
                let ty = match p.get("type") {
                    Some(Json::String(s)) => vec![s.clone()],
                    Some(Json::Array(a)) => a.iter().filter_map(Json::as_str).map(str::to_string).collect(),
                    _ => vec![],
                };
                m.insert(k.clone(), ty);
            }
            self.types.insert(name.to_string(), m);
        }
        self
    }

    pub fn push(&mut self, text: &str) -> Vec<OutEvent> {
        let mut ev = vec![];
        self.buf.push_str(text);
        self.run(&mut ev, false);
        ev
    }

    /// 生成が終わったとき（max_tokens で切れても、持っている物を出す）
    pub fn finish(&mut self) -> Vec<OutEvent> {
        let mut ev = vec![];
        self.run(&mut ev, true);
        ev
    }

    fn run(&mut self, ev: &mut Vec<OutEvent>, last: bool) {
        loop {
            match self.mode {
                Mode::Done => {
                    self.buf.clear();
                    return;
                }
                Mode::Reasoning | Mode::Content => {
                    let tags: &[&str] = match self.mode {
                        Mode::Reasoning => &[THINK_END, TOOL, ENDS[0], ENDS[1]],
                        _ if self.fresh => &[TOOL, ENDS[0], ENDS[1], THINK],
                        _ => &[TOOL, ENDS[0], ENDS[1]],
                    };
                    if let Some((i, tag)) = find_first(&self.buf, tags) {
                        let before = self.buf[..i].to_string();
                        self.buf.drain(..i + tag.len());
                        // <think> は答えの先頭（空白だけの後）に来たときだけ考えの始まり
                        if tag == THINK && !before.trim().is_empty() {
                            self.text(&format!("{before}{THINK}"), ev);
                            self.fresh = false;
                            continue;
                        }
                        self.text(&before, ev);
                        if tag == TOOL && self.mode == Mode::Content {
                            self.saved = Some((self.started, std::mem::take(&mut self.ws)));
                        }
                        self.end_segment();
                        self.mode = match tag {
                            THINK => Mode::Reasoning,
                            THINK_END => Mode::Content,
                            TOOL => Mode::Tool,
                            _ => Mode::Done,
                        };
                        continue;
                    }
                    let keep = if last { 0 } else { partial_suffix(&self.buf, tags) };
                    let n = self.buf.len() - keep;
                    let s: String = self.buf.drain(..n).collect();
                    self.text(&s, ev);
                    if last {
                        self.end_segment();
                    }
                    return;
                }
                Mode::Tool => {
                    let found = find_first(&self.buf, &[TOOL_END, TOOL, ENDS[0], ENDS[1]]);
                    let (block, close, next) = match found {
                        Some((i, tag)) => {
                            let b = self.buf[..i].to_string();
                            self.buf.drain(..i + tag.len());
                            let next = match tag {
                                TOOL_END => Mode::Content,
                                TOOL => Mode::Tool,
                                _ => Mode::Done,
                            };
                            (b, if tag == TOOL_END { TOOL_END } else { "" }, next)
                        }
                        None if last => (std::mem::take(&mut self.buf), "", Mode::Done),
                        None => return,
                    };
                    let saved = self.saved.take();
                    match self.parse_call(&block) {
                        Some(c) => {
                            self.fresh = false;
                            ev.push(OutEvent::ToolCall(c));
                        }
                        // <function=…> が無い: 道具の呼び出しではないので、手前の答えの続きとして出す
                        None => {
                            self.mode = Mode::Content;
                            if let Some((started, ws)) = saved {
                                self.started = started;
                                self.ws = ws;
                            }
                            self.text(&format!("{TOOL}{block}{close}"), ev);
                            if next != Mode::Content {
                                self.end_segment();
                            }
                        }
                    }
                    self.mode = next;
                }
            }
        }
    }

    /// 考え・答えの文字を出す（先頭の空白は捨て、末尾の空白は持っておく）
    fn text(&mut self, s: &str, ev: &mut Vec<OutEvent>) {
        let s = if self.started { s } else { s.trim_start_matches(py_space) };
        if s.is_empty() {
            return;
        }
        self.started = true;
        let body = s.trim_end_matches(py_space);
        if body.is_empty() {
            self.ws.push_str(s);
            return;
        }
        let out = format!("{}{body}", std::mem::take(&mut self.ws));
        self.ws.push_str(&s[body.len()..]);
        self.fresh = false;
        ev.push(match self.mode {
            Mode::Reasoning => OutEvent::Reasoning(out),
            _ => OutEvent::Content(out),
        });
    }

    fn end_segment(&mut self) {
        self.started = false;
        self.ws.clear();
    }

    /// <tool_call> の中身 1 つを読む（response_template の tool_calls と同じ形）
    fn parse_call(&self, block: &str) -> Option<ToolCall> {
        let b = block.trim_start_matches(py_space).strip_prefix("<function=")?;
        let end = b.find('>')?;
        let name = &b[..end];
        if name.is_empty() || name.contains('\n') {
            return None;
        }
        let body = &b[end + 1..];
        let types = self.types.get(name);
        let mut args = Map::new();
        let mut rest = body;
        while let Some(i) = rest.find("<parameter=") {
            let r = &rest[i + "<parameter=".len()..];
            let Some(ke) = r.find('>') else { break };
            let key = &r[..ke];
            if key.is_empty() || key.contains('\n') {
                rest = r;
                continue;
            }
            let v = &r[ke + 1..];
            let v = v.strip_prefix('\n').unwrap_or(v);
            let (vend, skip) = match find_first(v, &["</parameter>", "<parameter=", "</function>"]) {
                Some((j, "</parameter>")) => (j, j + "</parameter>".len()),
                Some((j, _)) => (j, j),
                None => (v.len(), v.len()),
            };
            let val = &v[..vend];
            let val = val.strip_suffix('\n').unwrap_or(val);
            args.insert(key.to_string(), convert_arg(val, types.and_then(|t| t.get(key))));
            rest = &v[skip..];
        }
        Some(ToolCall { name: name.to_string(), arguments: args })
    }
}

/// 引数の値（文字）を、定義の型に合わせて JSON にする。定義に無い引数は文字列のまま
fn convert_arg(v: &str, ty: Option<&Vec<String>>) -> Json {
    let Some(ty) = ty else { return Json::String(v.to_string()) };
    let has = |t: &str| ty.iter().any(|x| x == t);
    if has("string") {
        return Json::String(v.to_string());
    }
    let t = v.trim();
    if ty.is_empty() || has("object") || has("array") || has("integer") || has("number") || has("boolean") || has("null") {
        if let Ok(j) = serde_json::from_str::<Json>(t) {
            return j;
        }
        if has("boolean") {
            match t.to_ascii_lowercase().as_str() {
                "true" => return Json::Bool(true),
                "false" => return Json::Bool(false),
                _ => {}
            }
        }
    }
    Json::String(v.to_string())
}

/// いちばん前に出てくる印（同じ所なら tags の順）
fn find_first<'t>(s: &str, tags: &[&'t str]) -> Option<(usize, &'t str)> {
    tags.iter().filter_map(|t| s.find(t).map(|i| (i, *t))).min_by_key(|x| x.0)
}

/// s の末尾が、どれかの印の頭（途中まで）になっている長さ
fn partial_suffix(s: &str, tags: &[&str]) -> usize {
    let mut best = 0;
    for t in tags {
        for n in (1..t.len()).rev() {
            if n <= best {
                break;
            }
            if t.is_char_boundary(n) && s.ends_with(&t[..n]) {
                best = n;
                break;
            }
        }
    }
    best
}

/// まとめて分ける（stream しないとき）
pub fn parse_output(text: &str, in_reasoning: bool, tools: Option<&Json>) -> ParsedOutput {
    let mut p = OutputParser::new(in_reasoning).with_tools(tools);
    let mut out = ParsedOutput::default();
    p.push(text).into_iter().chain(p.finish()).for_each(|e| out.add(e));
    out
}
