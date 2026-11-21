//! 依頼（JSON）を読んで確かめる。誤りは 400（param と code つき。文は英語＝OpenAI の口に合わせる）。
//! 既定の値（docs/API.md 3・3.5）: reasoning_effort は medium を明示、preserve_thinking はふつうの会話は false・tools があれば true。
use super::sampler::SamplingParams;
use crate::template::{effort_ok, efforts, is_inkling, normalize_messages, ChatOptions, TemplateError};
use serde_json::{Map, Value as Json};

/// 依頼の大きさの上限（2026-11-10 の見直し: 描いて語に切る前に、バイトで早めに断る）
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limits {
    /// 1 つの message の文（content・reasoning_content・tool_calls の引数）のバイト数
    pub max_message_bytes: usize,
    pub max_messages: usize,
    /// tools を JSON にしたときのバイト数
    pub max_tools_bytes: usize,
    /// 画像（data URI の image_url）を受けるか・1 依頼の画像の数の上限
    pub allow_images: bool,
    pub max_images: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits { max_message_bytes: 1 << 20, max_messages: 4096, max_tools_bytes: 1 << 20, allow_images: false, max_images: 8 }
    }
}

/// 画像の part の data URI（{"type":"image_url","image_url":{"url":…}}・{"image_url":"…"}・{"type":"image","image":"…"}）
pub fn image_uri(po: &Map<String, Json>) -> Option<&str> {
    let v = po.get("image_url").or_else(|| po.get("image"))?;
    match v {
        Json::String(s) => Some(s),
        Json::Object(o) => o.get("url").and_then(Json::as_str),
        _ => None,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReqError {
    pub code: &'static str,
    pub param: Option<String>,
    pub message: String,
}

pub fn req_err(code: &'static str, param: &str, msg: impl Into<String>) -> ReqError {
    ReqError { code, param: (!param.is_empty()).then(|| param.to_string()), message: msg.into() }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ToolChoice {
    Auto,
    /// 道具を描かない（呼んでしまったら答えの文字として返す）
    None,
    /// ⚠️ 文法で縛る仕組みがまだ無いので Auto と同じ（描く道具も同じ）
    Required,
    /// その道具だけ描く（呼ぶのを縛るのではない）
    Named(String),
}

/// 試料の取り方（依頼に書かれた物だけ）
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SamplingInput {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i64>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub seed: Option<u64>,
}

impl SamplingInput {
    /// 書かれていない値はモデルカードの推奨（docs/API.md 3）: 考えあり temp 1.0・top_p 0.95・top_k 20、
    /// なし temp 0.7・top_p 0.8・top_k 20・presence 1.5。
    /// 温度 0（貪欲）を明示したときは罰の既定を入れない（llama-server と温度 0 で突き合わせるため）
    pub fn resolve(&self, thinking: bool, seed: u64) -> SamplingParams {
        let (t, tp, pp) = if thinking { (1.0, 0.95, 0.0) } else { (0.7, 0.8, 1.5) };
        let greedy = self.temperature.is_some_and(|t| t < 1e-5);
        SamplingParams {
            temperature: self.temperature.unwrap_or(t),
            top_p: self.top_p.unwrap_or(tp),
            top_k: match self.top_k {
                Some(k) if k <= 0 => 0,
                Some(k) => k.min(u32::MAX as i64) as u32,
                None => 20,
            },
            min_p: self.min_p.unwrap_or(0.0),
            presence_penalty: self.presence_penalty.unwrap_or(if greedy { 0.0 } else { pp }),
            frequency_penalty: self.frequency_penalty.unwrap_or(0.0),
            repetition_penalty: self.repetition_penalty.unwrap_or(1.0),
            seed: self.seed.unwrap_or(seed),
        }
    }
}

/// /v1/chat/completions の依頼（確かめた後）
#[derive(Clone, Debug)]
pub struct ChatParams {
    /// 依頼の model（返事の model にそのまま返す。前段・chat が送った名前。無い・変な物は None → 口の名前）
    pub model: Option<String>,
    /// normalize_messages 済み（tool_calls の arguments は object・developer は system）
    pub messages: Json,
    /// 空の配列は None
    pub tools: Option<Json>,
    pub tool_choice: ToolChoice,
    pub stream: bool,
    pub include_usage: bool,
    pub max_tokens: Option<usize>,
    pub sampling: SamplingInput,
    pub stop: Vec<String>,
    /// chat_template_kwargs（既定はまだ入れていない → apply_defaults）
    pub opts: ChatOptions,
    /// 先に計算した文書の番号（"document" か "documents": [1 つ]）
    pub document: Option<String>,
    /// 考えの上限（語）。thinking_budget_tokens・reasoning_budget_tokens・thinking_budget（上の段か chat_template_kwargs）。
    /// 考えがこの語数に達したら、考えを閉じる文と </think> を差し込んで答えへ（M2d）
    pub think_budget: Option<usize>,
    /// response_format の JSON の形（json_object・json_schema）。None は普通の文
    pub response_format: Option<std::sync::Arc<super::jsonc::Schema>>,
}

impl ChatParams {
    /// docs/API.md 3 の既定を入れる（文書を付けて tools が決まった後に呼ぶ）
    pub fn apply_defaults(&mut self, default_effort: &str, default_thinking: bool) {
        apply_opt_defaults(&mut self.opts, self.tools.is_some(), default_effort, default_thinking);
    }

    /// テンプレートに渡す tools（tool_choice で絞る）
    pub fn render_tools(&self) -> Option<Json> {
        match &self.tool_choice {
            ToolChoice::None => None,
            ToolChoice::Named(n) => {
                let t: Vec<Json> = self.tools.iter().flat_map(|t| t.as_array().cloned().unwrap_or_default()).filter(|t| tool_name(t) == Some(n.as_str())).collect();
                Some(Json::Array(t))
            }
            _ => self.tools.clone(),
        }
    }
}

/// enable_thinking（無ければ既定。serve の既定は false）・reasoning_effort（無ければ既定）・preserve_thinking（無ければ tools があるか）
pub fn apply_opt_defaults(opts: &mut ChatOptions, has_tools: bool, default_effort: &str, default_thinking: bool) {
    // 考えるのはテンプレートの既定なので、既定が「考える」なら何も書かない（前と同じ描き方）
    if opts.enable_thinking.is_none() && !default_thinking {
        opts.enable_thinking = Some(false);
    }
    if opts.reasoning_effort.is_none() {
        opts.reasoning_effort = Some(default_effort.to_string());
    }
    if opts.preserve_thinking.is_none() {
        opts.preserve_thinking = Some(has_tools);
    }
}

pub fn tool_name(t: &Json) -> Option<&str> {
    t.get("function").unwrap_or(t).get("name").and_then(Json::as_str)
}

pub fn parse_object(body: &[u8]) -> Result<Map<String, Json>, ReqError> {
    match serde_json::from_slice::<Json>(body) {
        Ok(Json::Object(o)) => Ok(o),
        Ok(_) => Err(req_err("invalid_json", "", "request body must be a JSON object")),
        Err(e) => Err(req_err("invalid_json", "", format!("request body is not valid JSON: {e}"))),
    }
}

pub fn parse_chat(body: &[u8]) -> Result<ChatParams, ReqError> {
    parse_chat_limited(body, &Limits::default())
}

pub fn parse_chat_limited(body: &[u8], lim: &Limits) -> Result<ChatParams, ReqError> {
    let o = parse_object(body)?;
    let messages = parse_messages_limited(o.get("messages"), lim)?;
    let tools = parse_tools_limited(o.get("tools"), lim)?;
    let tool_choice = parse_tool_choice(o.get("tool_choice"), tools.as_ref())?;
    let stream = opt_bool(&o, "stream")?.unwrap_or(false);
    let include_usage = match o.get("stream_options") {
        None | Some(Json::Null) => false,
        Some(Json::Object(so)) => opt_bool(so, "include_usage")?.unwrap_or(false),
        Some(_) => return Err(req_err("invalid_type", "stream_options", "stream_options must be an object")),
    };
    let max_tokens = match opt_uint(&o, "max_completion_tokens")? {
        Some(v) => Some(v),
        None => opt_uint(&o, "max_tokens")?,
    };
    if max_tokens == Some(0) {
        return Err(req_err("invalid_value", "max_tokens", "max_tokens must be at least 1"));
    }
    if let Some(n) = opt_uint(&o, "n")? {
        if n != 1 {
            return Err(req_err("unsupported_value", "n", "only n=1 is supported"));
        }
    }
    if opt_bool(&o, "logprobs")? == Some(true) || o.get("top_logprobs").is_some_and(|v| !v.is_null()) {
        return Err(req_err("unsupported_value", "logprobs", "logprobs are not supported yet"));
    }
    // JSON の形（json_object・json_schema の一部。src/server/jsonc.rs）。対応しない JSON Schema の語は unsupported_value
    let response_format = super::jsonc::from_response_format(o.get("response_format")).map_err(|(param, msg)| {
        let code = if msg.contains("not supported") { "unsupported_value" } else { "invalid_value" };
        req_err(code, &param, msg)
    })?;
    let sampling = SamplingInput {
        temperature: opt_num(&o, "temperature", 0.0, 2.0, false)?,
        top_p: opt_num(&o, "top_p", 0.0, 1.0, true)?,
        top_k: opt_int(&o, "top_k")?,
        min_p: opt_num(&o, "min_p", 0.0, 1.0, false)?,
        presence_penalty: opt_num(&o, "presence_penalty", -2.0, 2.0, false)?,
        frequency_penalty: opt_num(&o, "frequency_penalty", -2.0, 2.0, false)?,
        repetition_penalty: opt_num(&o, "repetition_penalty", 0.0, 10.0, true)?,
        seed: match o.get("seed") {
            None | Some(Json::Null) => None,
            Some(v) => Some(v.as_u64().or_else(|| v.as_i64().map(|x| x as u64)).ok_or_else(|| req_err("invalid_type", "seed", "seed must be an integer"))?),
        },
    };
    if let Some(k) = sampling.top_k {
        if k < -1 {
            return Err(req_err("invalid_value", "top_k", "top_k must be -1, 0 (disabled) or a positive integer"));
        }
    }
    let stop = parse_stop(o.get("stop"))?;
    let opts = parse_kwargs(&o, true)?;
    let document = parse_document(&o)?;
    let mut think_budget = None;
    for k in ["thinking_budget_tokens", "reasoning_budget_tokens", "thinking_budget"] {
        let v = match o.get(k) {
            Some(v) if !v.is_null() => Some((k.to_string(), v.clone())),
            _ => opts.extra.get(k).filter(|v| !v.is_null()).map(|v| (format!("chat_template_kwargs.{k}"), v.clone())),
        };
        if let Some((name, v)) = v {
            // -1 は上限なし（llama.cpp の reasoning_budget と同じ）
            match v.as_i64() {
                Some(-1) => {}
                Some(n) if n >= 0 => think_budget = Some(think_budget.map_or(n as usize, |b: usize| b.min(n as usize))),
                _ => return Err(req_err("invalid_value", &name, format!("{name} must be an integer >= 0 (or -1 for no limit)"))),
            }
        }
    }
    // model は名前を返すだけ（どの名前でも同じモデルで答える）。256 字までの見える文字だけ
    let model = o.get("model").and_then(Json::as_str).filter(|m| !m.is_empty() && m.len() <= 256 && !m.chars().any(char::is_control)).map(str::to_string);
    Ok(ChatParams { model, messages, tools, tool_choice, stream, include_usage, max_tokens, sampling, stop, opts, document, think_budget, response_format })
}

/// chat_template_kwargs（と上の段の reasoning_effort）。reasoning_effort は考えるかによらず確かめる（違えば 400）
pub fn parse_kwargs(o: &Map<String, Json>, add_generation_prompt: bool) -> Result<ChatOptions, ReqError> {
    let kw = o.get("chat_template_kwargs").unwrap_or(&Json::Null);
    let mut opts = ChatOptions::from_kwargs(kw, add_generation_prompt).map_err(|e| req_err("invalid_value", "chat_template_kwargs", template_msg(&e)))?;
    match o.get("reasoning_effort") {
        None | Some(Json::Null) => {}
        Some(Json::String(e)) => match &opts.reasoning_effort {
            Some(k) if k != e => {
                return Err(req_err("invalid_value", "reasoning_effort", "reasoning_effort and chat_template_kwargs.reasoning_effort differ"));
            }
            _ => opts.reasoning_effort = Some(e.clone()),
        },
        Some(Json::Number(n)) if is_inkling() => {
            let e = format!("num:{}", n.as_f64().unwrap_or(-1.0));
            match &opts.reasoning_effort {
                Some(k) if *k != e => return Err(req_err("invalid_value", "reasoning_effort", "reasoning_effort and chat_template_kwargs.reasoning_effort differ")),
                _ => opts.reasoning_effort = Some(e),
            }
        }
        Some(_) => return Err(req_err("invalid_type", "reasoning_effort", if is_inkling() { "reasoning_effort must be a string or a number in [0, 0.99]" } else { "reasoning_effort must be a string" })),
    }
    if let Some(e) = &opts.reasoning_effort {
        if !effort_ok(e) {
            return Err(req_err(
                "invalid_value",
                "chat_template_kwargs.reasoning_effort",
                format!("reasoning_effort must be one of {} (got {e:?})", efforts().join(", ")),
            ));
        }
    }
    Ok(opts)
}

/// テンプレートの誤り → 文（英語）
pub fn template_msg(e: &TemplateError) -> String {
    match e {
        TemplateError::BadEffort(v) => format!("reasoning_effort must be one of {} (got {v:?})", efforts().join(", ")),
        TemplateError::Raised(m) => m.clone(),
        TemplateError::BadRequest(m) => m.clone(),
        TemplateError::Template(m) => format!("could not render the conversation: {m}"),
    }
}

pub fn parse_messages(v: Option<&Json>) -> Result<Json, ReqError> {
    parse_messages_limited(v, &Limits::default())
}

/// 1 つの message の文のバイト数（content の文・parts の text・reasoning_content・tool_calls を JSON にした長さ）
fn message_bytes(o: &Map<String, Json>) -> usize {
    let s = |v: Option<&Json>| v.and_then(Json::as_str).map_or(0, str::len);
    let content = match o.get("content") {
        Some(Json::String(c)) => c.len(),
        Some(Json::Array(parts)) => parts.iter().map(|p| s(p.get("text"))).sum(),
        _ => 0,
    };
    let calls = o.get("tool_calls").filter(|c| !c.is_null()).map_or(0, |c| c.to_string().len());
    content + s(o.get("reasoning_content")) + calls
}

pub fn parse_messages_limited(v: Option<&Json>, lim: &Limits) -> Result<Json, ReqError> {
    let arr = match v {
        Some(Json::Array(a)) if !a.is_empty() => a,
        Some(Json::Array(_)) => return Err(req_err("invalid_value", "messages", "messages must not be empty")),
        None | Some(Json::Null) => return Err(req_err("missing_required_parameter", "messages", "messages is required")),
        Some(_) => return Err(req_err("invalid_type", "messages", "messages must be an array")),
    };
    if arr.len() > lim.max_messages {
        return Err(req_err("too_many_messages", "messages", format!("at most {} messages per request", lim.max_messages)));
    }
    let mut n_images = 0usize;
    for (i, m) in arr.iter().enumerate() {
        let p = format!("messages[{i}]");
        let o = m.as_object().ok_or_else(|| req_err("invalid_type", &p, "message must be an object"))?;
        if message_bytes(o) > lim.max_message_bytes {
            return Err(req_err("message_too_large", &p, format!("a message may have at most {} bytes of text", lim.max_message_bytes)));
        }
        let role = o.get("role").and_then(Json::as_str).ok_or_else(|| req_err("invalid_value", &format!("{p}.role"), "role is required"))?;
        if !matches!(role, "system" | "developer" | "user" | "assistant" | "tool") {
            return Err(req_err("invalid_value", &format!("{p}.role"), format!("unknown role {role:?}")));
        }
        match o.get("content") {
            None | Some(Json::Null) | Some(Json::String(_)) => {}
            Some(Json::Array(parts)) => {
                for (j, part) in parts.iter().enumerate() {
                    let pp = format!("{p}.content[{j}]");
                    let po = part.as_object().ok_or_else(|| req_err("invalid_type", &pp, "content part must be an object"))?;
                    let ty = po.get("type").and_then(Json::as_str);
                    let image = po.contains_key("image") || po.contains_key("image_url") || matches!(ty, Some("image" | "image_url"));
                    if image && lim.allow_images {
                        // 画像は data URI だけ（この口は URL を取りに行かない）
                        let uri = image_uri(po).ok_or_else(|| req_err("invalid_value", &pp, "an image part needs image_url.url (a data URI)"))?;
                        if !uri.starts_with("data:") {
                            return Err(req_err("unsupported_content", &pp, "images must be sent as data URIs (data:image/png;base64,...); this server does not fetch URLs"));
                        }
                        n_images += 1;
                        if n_images > lim.max_images {
                            return Err(req_err("too_many_images", &pp, format!("at most {} images per request", lim.max_images)));
                        }
                        continue;
                    }
                    let media = image || po.contains_key("video") || po.contains_key("input_audio");
                    if media || matches!(ty, Some("video" | "input_audio" | "file")) {
                        let what = if image { "image input is not enabled on this server" } else { "video/audio input is not supported yet" };
                        return Err(req_err("unsupported_content", &pp, what));
                    }
                    if !matches!(po.get("text"), Some(Json::String(_))) {
                        return Err(req_err("invalid_value", &pp, "content part must be {\"type\": \"text\", \"text\": \"...\"}"));
                    }
                }
            }
            Some(_) => return Err(req_err("invalid_type", &format!("{p}.content"), "content must be a string, an array of parts or null")),
        }
        if let Some(r) = o.get("reasoning_content") {
            if !(r.is_string() || r.is_null()) {
                return Err(req_err("invalid_type", &format!("{p}.reasoning_content"), "reasoning_content must be a string"));
            }
        }
        match o.get("tool_calls") {
            None | Some(Json::Null) => {}
            Some(Json::Array(calls)) => {
                if role != "assistant" {
                    return Err(req_err("invalid_value", &format!("{p}.tool_calls"), "only assistant messages can have tool_calls"));
                }
                for (j, c) in calls.iter().enumerate() {
                    let name = c.get("function").unwrap_or(c).get("name").and_then(Json::as_str).unwrap_or("");
                    if !valid_tool_name(name) {
                        return Err(req_err("invalid_value", &format!("{p}.tool_calls[{j}].function.name"), "tool call needs a valid function name"));
                    }
                }
            }
            Some(_) => return Err(req_err("invalid_type", &format!("{p}.tool_calls"), "tool_calls must be an array")),
        }
    }
    normalize_messages(&Json::Array(arr.clone())).map_err(|e| req_err("invalid_value", "messages", template_msg(&e)))
}

/// 道具の名前: 英数字と _ - . : だけ、1〜128 文字（<function=名前> に入れるので > や改行は困る）
pub fn valid_tool_name(n: &str) -> bool {
    !n.is_empty() && n.len() <= 128 && n.chars().all(|c| c.is_ascii_alphanumeric() || "_-.:".contains(c))
}

pub fn parse_tools(v: Option<&Json>) -> Result<Option<Json>, ReqError> {
    parse_tools_limited(v, &Limits::default())
}

pub fn parse_tools_limited(v: Option<&Json>, lim: &Limits) -> Result<Option<Json>, ReqError> {
    let arr = match v {
        None | Some(Json::Null) => return Ok(None),
        Some(Json::Array(a)) if a.is_empty() => return Ok(None),
        Some(Json::Array(a)) => a,
        Some(_) => return Err(req_err("invalid_type", "tools", "tools must be an array")),
    };
    if v.map_or(0, |t| t.to_string().len()) > lim.max_tools_bytes {
        return Err(req_err("tools_too_large", "tools", format!("tools may be at most {} bytes as JSON", lim.max_tools_bytes)));
    }
    for (i, t) in arr.iter().enumerate() {
        let p = format!("tools[{i}]");
        let o = t.as_object().ok_or_else(|| req_err("invalid_type", &p, "tool must be an object"))?;
        match o.get("type").and_then(Json::as_str) {
            Some("function") | None => {}
            Some(other) => return Err(req_err("unsupported_value", &format!("{p}.type"), format!("tool type {other:?} is not supported (only \"function\")"))),
        }
        let f = o.get("function").and_then(Json::as_object).ok_or_else(|| req_err("invalid_value", &format!("{p}.function"), "tool needs a function object"))?;
        let name = f.get("name").and_then(Json::as_str).unwrap_or("");
        if !valid_tool_name(name) {
            return Err(req_err("invalid_value", &format!("{p}.function.name"), "function name must be 1-128 chars of [A-Za-z0-9_-.:]"));
        }
        if let Some(ps) = f.get("parameters") {
            if !(ps.is_object() || ps.is_null()) {
                return Err(req_err("invalid_type", &format!("{p}.function.parameters"), "parameters must be a JSON schema object"));
            }
        }
    }
    let mut names: Vec<&str> = arr.iter().filter_map(tool_name).collect();
    names.sort_unstable();
    if names.windows(2).any(|w| w[0] == w[1]) {
        return Err(req_err("invalid_value", "tools", "tool names must be unique"));
    }
    Ok(Some(Json::Array(arr.clone())))
}

pub fn parse_tool_choice(v: Option<&Json>, tools: Option<&Json>) -> Result<ToolChoice, ReqError> {
    let has = |n: &str| tools.and_then(Json::as_array).is_some_and(|a| a.iter().any(|t| tool_name(t) == Some(n)));
    match v {
        None | Some(Json::Null) => Ok(ToolChoice::Auto),
        Some(Json::String(s)) => match s.as_str() {
            "auto" => Ok(ToolChoice::Auto),
            "none" => Ok(ToolChoice::None),
            "required" if tools.is_some() => Ok(ToolChoice::Required),
            "required" => Err(req_err("invalid_value", "tool_choice", "tool_choice \"required\" needs tools")),
            _ => Err(req_err("invalid_value", "tool_choice", "tool_choice must be \"auto\", \"none\", \"required\" or a function")),
        },
        Some(Json::Object(o)) => {
            if o.get("type").and_then(Json::as_str).is_some_and(|t| t != "function") {
                return Err(req_err("unsupported_value", "tool_choice", "tool_choice.type must be \"function\""));
            }
            let n = o.get("function").and_then(|f| f.get("name")).and_then(Json::as_str).unwrap_or("");
            if !has(n) {
                return Err(req_err("invalid_value", "tool_choice", "tool_choice names a function that is not in tools"));
            }
            Ok(ToolChoice::Named(n.to_string()))
        }
        Some(_) => Err(req_err("invalid_type", "tool_choice", "tool_choice must be a string or an object")),
    }
}

pub fn parse_stop(v: Option<&Json>) -> Result<Vec<String>, ReqError> {
    let v: Vec<String> = match v {
        None | Some(Json::Null) => vec![],
        Some(Json::String(s)) => vec![s.clone()],
        Some(Json::Array(a)) => a.iter().map(|x| x.as_str().map(str::to_string).ok_or_else(|| req_err("invalid_type", "stop", "stop must be a string or an array of strings"))).collect::<Result<_, _>>()?,
        Some(_) => return Err(req_err("invalid_type", "stop", "stop must be a string or an array of strings")),
    };
    if v.len() > 16 {
        return Err(req_err("invalid_value", "stop", "at most 16 stop strings"));
    }
    if v.iter().any(|s| s.len() > 1024) {
        return Err(req_err("invalid_value", "stop", "stop strings must be at most 1024 bytes"));
    }
    Ok(v.into_iter().filter(|s| !s.is_empty()).collect())
}

fn parse_document(o: &Map<String, Json>) -> Result<Option<String>, ReqError> {
    let one = match (o.get("document"), o.get("documents")) {
        (None | Some(Json::Null), None | Some(Json::Null)) => return Ok(None),
        (Some(Json::String(s)), None | Some(Json::Null)) => s.clone(),
        (None | Some(Json::Null), Some(Json::Array(a))) if a.len() == 1 && a[0].is_string() => a[0].as_str().unwrap_or_default().to_string(),
        (None | Some(Json::Null), Some(Json::Array(a))) if a.is_empty() => return Ok(None),
        _ => return Err(req_err("invalid_value", "document", "document must be one document id (\"document\": \"doc-…\" or \"documents\": [\"doc-…\"])")),
    };
    Ok(Some(one))
}

pub fn opt_bool(o: &Map<String, Json>, k: &str) -> Result<Option<bool>, ReqError> {
    match o.get(k) {
        None | Some(Json::Null) => Ok(None),
        Some(Json::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(req_err("invalid_type", k, format!("{k} must be true or false"))),
    }
}

fn opt_uint(o: &Map<String, Json>, k: &str) -> Result<Option<usize>, ReqError> {
    match o.get(k) {
        None | Some(Json::Null) => Ok(None),
        Some(v) => v.as_u64().map(|x| Some(x.min(usize::MAX as u64) as usize)).ok_or_else(|| req_err("invalid_type", k, format!("{k} must be a non-negative integer"))),
    }
}

fn opt_int(o: &Map<String, Json>, k: &str) -> Result<Option<i64>, ReqError> {
    match o.get(k) {
        None | Some(Json::Null) => Ok(None),
        Some(v) => v.as_i64().map(Some).ok_or_else(|| req_err("invalid_type", k, format!("{k} must be an integer"))),
    }
}

/// 数（lo_open なら lo は含まない）
fn opt_num(o: &Map<String, Json>, k: &str, lo: f64, hi: f64, lo_open: bool) -> Result<Option<f32>, ReqError> {
    match o.get(k) {
        None | Some(Json::Null) => Ok(None),
        Some(v) => {
            let x = v.as_f64().ok_or_else(|| req_err("invalid_type", k, format!("{k} must be a number")))?;
            let ok = x.is_finite() && x <= hi && if lo_open { x > lo } else { x >= lo };
            if !ok {
                let l = if lo_open { "(" } else { "[" };
                return Err(req_err("invalid_value", k, format!("{k} must be in {l}{lo}, {hi}]")));
            }
            Ok(Some(x as f32))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(v: Json) -> Result<ChatParams, ReqError> {
        parse_chat(v.to_string().as_bytes())
    }

    #[test]
    fn defaults_and_effort() {
        let mut p = parse(json!({"messages": [{"role": "user", "content": "hi"}]})).unwrap();
        assert!(!p.stream && p.tools.is_none() && p.opts.reasoning_effort.is_none());
        p.apply_defaults("medium", true);
        assert_eq!(p.opts.reasoning_effort.as_deref(), Some("medium"));
        assert_eq!(p.opts.preserve_thinking, Some(false));
        let tools = json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]);
        let mut p = parse(json!({"messages": [{"role": "user", "content": "hi"}], "tools": tools})).unwrap();
        p.apply_defaults("medium", true);
        assert_eq!(p.opts.preserve_thinking, Some(true));
        // 明示した値はそのまま
        let mut p = parse(json!({"messages": [{"role": "user", "content": "hi"}], "tools": tools, "chat_template_kwargs": {"preserve_thinking": false, "reasoning_effort": "low"}})).unwrap();
        p.apply_defaults("medium", true);
        assert_eq!((p.opts.preserve_thinking, p.opts.reasoning_effort.as_deref()), (Some(false), Some("low")));
        // 違う effort は考えなしでも 400
        let e = parse(json!({"messages": [{"role": "user", "content": "hi"}], "chat_template_kwargs": {"enable_thinking": false, "reasoning_effort": "high"}})).unwrap_err();
        assert_eq!(e.param.as_deref(), Some("chat_template_kwargs.reasoning_effort"));
        let e = parse(json!({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "max"})).unwrap_err();
        assert_eq!(e.code, "invalid_value");
        let p = parse(json!({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "xhigh"})).unwrap();
        assert_eq!(p.opts.reasoning_effort.as_deref(), Some("xhigh"));
    }

    #[test]
    fn sampling_defaults() {
        let s = SamplingInput::default();
        let a = s.resolve(true, 5);
        assert_eq!((a.temperature, a.top_p, a.top_k, a.presence_penalty, a.seed), (1.0, 0.95, 20, 0.0, 5));
        let b = s.resolve(false, 5);
        assert_eq!((b.temperature, b.top_p, b.top_k, b.presence_penalty), (0.7, 0.8, 20, 1.5));
        let g = SamplingInput { temperature: Some(0.0), ..Default::default() }.resolve(false, 5);
        assert_eq!((g.presence_penalty, g.is_greedy()), (0.0, true));
        let k = SamplingInput { top_k: Some(-1), seed: Some(9), ..Default::default() }.resolve(true, 5);
        assert_eq!((k.top_k, k.seed), (0, 9));
    }

    #[test]
    fn errors() {
        let bad = |v: Json| parse(v).unwrap_err();
        assert_eq!(bad(json!({})).code, "missing_required_parameter");
        assert_eq!(bad(json!({"messages": []})).code, "invalid_value");
        assert_eq!(bad(json!({"messages": [{"role": "robot", "content": "x"}]})).code, "invalid_value");
        assert_eq!(bad(json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]})).code, "unsupported_content");
        let m = json!([{"role": "user", "content": "hi"}]);
        assert_eq!(bad(json!({"messages": m, "temperature": 3})).param.as_deref(), Some("temperature"));
        assert_eq!(bad(json!({"messages": m, "top_p": 0})).param.as_deref(), Some("top_p"));
        assert_eq!(bad(json!({"messages": m, "n": 2})).code, "unsupported_value");
        assert_eq!(bad(json!({"messages": m, "logprobs": true})).code, "unsupported_value");
        assert_eq!(bad(json!({"messages": m, "max_tokens": 0})).param.as_deref(), Some("max_tokens"));
        assert_eq!(bad(json!({"messages": m, "max_tokens": -5})).code, "invalid_type");
        assert_eq!(bad(json!({"messages": m, "stop": [1]})).param.as_deref(), Some("stop"));
        assert_eq!(bad(json!({"messages": m, "tools": [{"type": "function", "function": {"name": "a b"}}]})).code, "invalid_value");
        assert_eq!(bad(json!({"messages": m, "tool_choice": "required"})).param.as_deref(), Some("tool_choice"));
        assert_eq!(bad(json!({"messages": m, "tool_choice": {"type": "function", "function": {"name": "x"}}})).param.as_deref(), Some("tool_choice"));
        assert!(parse(json!({"messages": m, "response_format": {"type": "json_object"}})).unwrap().response_format.is_some());
        assert!(parse(json!({"messages": m, "response_format": {"type": "text"}})).unwrap().response_format.is_none());
        let e = bad(json!({"messages": m, "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object", "properties": {"a": {"type": "string", "pattern": "^a"}}}}}}));
        assert_eq!(e.code, "unsupported_value");
        assert_eq!(e.param.as_deref(), Some("response_format.json_schema.schema"));
        assert_eq!(bad(json!({"messages": m, "response_format": {"type": "xml"}})).code, "unsupported_value");
        let t1 = json!([{"type": "function", "function": {"name": "a"}}]);
        assert_eq!(bad(json!({"messages": m, "tools": t1, "tool_choice": {"type": "custom", "function": {"name": "a"}}})).param.as_deref(), Some("tool_choice"));
        assert_eq!(bad(json!({"messages": m, "chat_template_kwargs": {"enable_thinking": "yes"}})).param.as_deref(), Some("chat_template_kwargs"));
        assert_eq!(parse_chat(b"{not json").unwrap_err().code, "invalid_json");
        assert_eq!(parse_chat(b"[1]").unwrap_err().code, "invalid_json");
        // 通る物
        let p = parse(json!({"messages": m, "stop": "END", "max_completion_tokens": 7, "max_tokens": 9, "seed": -1, "stream": true, "stream_options": {"include_usage": true}, "document": "doc-1"})).unwrap();
        assert_eq!((p.stop, p.max_tokens, p.sampling.seed, p.stream, p.include_usage, p.document.as_deref()), (vec!["END".to_string()], Some(7), Some(u64::MAX), true, true, Some("doc-1")));
        let t = json!([{"type": "function", "function": {"name": "a"}}, {"type": "function", "function": {"name": "b"}}]);
        let p = parse(json!({"messages": m, "tools": t, "tool_choice": {"type": "function", "function": {"name": "b"}}})).unwrap();
        assert_eq!(p.render_tools().unwrap().as_array().unwrap().len(), 1);
        let p = parse(json!({"messages": m, "tools": t, "tool_choice": "none"})).unwrap();
        assert!(p.render_tools().is_none() && p.tools.is_some());
    }
}
