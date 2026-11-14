//! POST /v1/chat/completions（stream あり・なし）。
//! 流れ: 鍵 → 回数・同時の数 → 依頼を読む → （文書を付ける）→ 既定を入れる → テンプレートを描いて語に切る（別の糸）→
//!       文脈の長さ（超えたら 400）→ Backend に入れる（一杯なら 429）→ 出た語を文字に → 考え／答え／道具の呼び出し → 返す。
use super::backend::{CacheScope, DocRef, FinishReason, GenEvent, GenRequest, GenStream, MediaInput, PrefixHints, Priority};
use super::docs::DocRecord;
use super::metrics::RequestRecord;
use super::output::{prompt_tail, tool_call_text, OutputPipe};
use super::request::{self, ChatParams, ToolChoice};
use super::sampler::SamplingParams;
use super::util::{rand_id, rand_u64, unix_secs};
use super::{submit_error, Account, ApiError, AppState};
use crate::template::{ChatOptions, OutEvent};
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json as AxJson;
use serde_json::{json, Value as Json};
use std::convert::Infallible;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

pub const EP_CHAT: &str = "chat";

/// テンプレートを描いて語に切った物
pub(crate) struct Prepared {
    /// 最後の回（出力の分け方に要る所だけ）
    pub tail: String,
    pub ids: Vec<u32>,
    pub system_end: Option<usize>,
    pub turn_start: Option<usize>,
    pub user_start: Option<usize>,
}

/// 最後の発言の回の頭（turn_start の 1 つ前の `<|im_start|>`）。system の回なら None（system_end と同じ所）
pub(crate) fn user_start_of(ids: &[u32], turn_start: Option<usize>, im_start: u32) -> Option<usize> {
    let ts = turn_start?;
    ids[..ts].iter().rposition(|&t| t == im_start).filter(|&i| i > 0)
}

/// 描く・切る（長い文は時間がかかるので spawn_blocking の中で呼ぶ）。
/// scope: 切り分けの覚えの持ち主（⛔ アカウントの名前。アカウントをまたいで覚えを使わない）。
/// max_tokens を超えたら、そこで切り分けをやめて 400（長すぎる依頼を全部切ってから断らない）
pub(crate) fn prepare(st: &AppState, scope: &str, max_tokens: usize, messages: &Json, tools: Option<&Json>, opts: &ChatOptions) -> Result<Prepared, ApiError> {
    let text = st.tpl.render(messages, tools, opts).map_err(|e| template_error(&e))?;
    let ids = st.tok.encode_limited(Some(scope), &text, max_tokens).map_err(|e| match e {
        crate::tokenizer::EncodeError::TooLong { at_least } => ApiError::bad(
            "context_length_exceeded",
            Some("messages"),
            format!("the prompt is at least {at_least} tokens; at most {max_tokens} tokens are allowed per request (prompts are not truncated)"),
        ),
        crate::tokenizer::EncodeError::Failed(e) => ApiError::internal("tokenizer_error", format!("tokenizer: {e}")),
    })?;
    let system_end = if text.starts_with("<|im_start|>system\n") { ids.iter().skip(1).position(|&t| t == st.tok.ids.im_start).map(|i| i + 1) } else { None };
    let turn_start = ids.iter().rposition(|&t| t == st.tok.ids.im_start).filter(|&i| i > 0);
    let user_start = user_start_of(&ids, turn_start, st.tok.ids.im_start);
    Ok(Prepared { tail: prompt_tail(&text).to_string(), ids, system_end, turn_start, user_start })
}

/// messages の画像を読み・縮め、prompt の <|image_pad|> 1 語ずつを画像の語の数に広げる（spawn_blocking の中で）。
/// 画像の順はテンプレートが pad を置く順（template::collect_media）
pub(crate) fn prepare_media(st: &AppState, messages: &Json, prep: &mut Prepared) -> Result<Vec<Arc<MediaInput>>, ApiError> {
    use crate::template::{collect_media, expand_pads, MediaKind};
    let media = collect_media(messages);
    if media.is_empty() {
        return Ok(vec![]);
    }
    let bad = |m: String| ApiError::bad("invalid_image", Some("messages"), m);
    let mut imgs = vec![];
    for (k, (kind, part)) in media.iter().enumerate() {
        if *kind != MediaKind::Image {
            return Err(ApiError::bad("unsupported_content", Some("messages"), "video input is not supported yet"));
        }
        let uri = part.as_object().and_then(request::image_uri).ok_or_else(|| bad(format!("image {k}: image_url.url is missing")))?;
        let bytes = crate::vision::data_uri_bytes(uri, st.cfg.max_image_bytes).map_err(|e| bad(format!("image {k}: {e}")))?;
        let img = crate::vision::decode(&bytes).map_err(|e| bad(format!("image {k}: {e}")))?;
        if img.w > 16384 || img.h > 16384 {
            return Err(bad(format!("image {k}: {} x {} pixels is too large (at most 16384 per side)", img.w, img.h)));
        }
        imgs.push(crate::vision::resize_for_model(&img));
    }
    let a = crate::vision::PATCH * crate::vision::MERGE;
    let counts: Vec<usize> = imgs.iter().map(|r| (r.w / a) * (r.h / a)).collect();
    let pad = st.tok.ids.image_pad;
    let ids = expand_pads(&prep.ids, pad, &counts).map_err(|e| ApiError::internal("media_mismatch", e))?;
    // 広げた後の pad の続きの頭 = 画像 k の start
    let mut out = vec![];
    let mut i = 0;
    for (k, r) in imgs.into_iter().enumerate() {
        while i < ids.len() && ids[i] != pad {
            i += 1;
        }
        let (nx, ny) = (r.w / a, r.h / a);
        out.push(Arc::new(MediaInput { start: i, nx, ny, w: r.w, h: r.h, rgb: r.px }));
        i += counts[k];
    }
    // 頭の印を広げた後の位置に（system には画像を置けない＝テンプレートが止める ので system_end は変わらない）
    prep.ids = ids;
    prep.turn_start = prep.ids.iter().rposition(|&t| t == st.tok.ids.im_start).filter(|&i| i > 0);
    prep.user_start = user_start_of(&prep.ids, prep.turn_start, st.tok.ids.im_start);
    Ok(out)
}

pub(crate) fn template_error(e: &crate::template::TemplateError) -> ApiError {
    use crate::template::TemplateError as T;
    let param = match e {
        T::BadEffort(_) => Some("chat_template_kwargs.reasoning_effort"),
        T::Raised(_) => Some("messages"),
        _ => None,
    };
    ApiError::bad(if matches!(e, T::BadEffort(_)) { "invalid_value" } else { "invalid_messages" }, param, request::template_msg(e))
}

pub async fn chat_completions(State(st): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    let t0 = Instant::now();
    let acct = match st.authenticate(&headers) {
        Ok(a) => a,
        Err(e) => return st.reject(None, EP_CHAT, e),
    };
    match chat_inner(&st, &acct, t0, &body).await {
        Ok(r) => r,
        Err(e) => st.reject(Some(&acct), EP_CHAT, e),
    }
}

/// 依頼に文書を付ける（文書の messages を前に、tools は文書の物）。見えない文書は 404
fn attach_document(st: &AppState, acct: &Account, p: &mut ChatParams) -> Result<Option<Arc<DocRecord>>, ApiError> {
    let Some(id) = p.document.clone() else { return Ok(None) };
    let d = st.docs.get(&id, acct).ok_or_else(|| ApiError::not_found("document_not_found", format!("document {id:?} not found")))?;
    let mut m = d.messages.as_array().cloned().unwrap_or_default();
    m.extend(p.messages.as_array().cloned().unwrap_or_default());
    p.messages = Json::Array(m);
    match (&p.tools, &d.tools) {
        (Some(a), Some(b)) if a != b => return Err(ApiError::bad("document_mismatch", Some("tools"), "tools differ from the tools the document was created with (leave tools out to use the document's)")),
        (Some(_), None) => return Err(ApiError::bad("document_mismatch", Some("tools"), "the document was created without tools; tools would change its prefix")),
        (None, Some(t)) => p.tools = Some(t.clone()),
        _ => {}
    }
    Ok(Some(d))
}

/// 考えの上限に達したときに差し込む文（Qwen の model card の推奨と同じ）
const THINK_CLOSE: &str = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n";

async fn chat_inner(st: &Arc<AppState>, acct: &Arc<Account>, t0: Instant, body: &[u8]) -> Result<Response, ApiError> {
    st.check_ready()?;
    acct.take_rate().map_err(|wait| ApiError::limit("rate_limit_exceeded", wait, format!("rate limit for this key reached ({} requests/min)", acct.rpm.unwrap_or(0))))?;
    let guard = acct.enter().ok_or_else(|| ApiError::limit("too_many_concurrent_requests", 1, format!("this key may run {} requests at once", acct.max_concurrent.unwrap_or(0))))?;
    // 口の同時の数は先に取る（描く・語に切る間も数える。取って確かめるので上限を超えない）
    let slot = InflightSlot::take(st).ok_or_else(|| ApiError::limit("server_busy", 1, "the server is busy; retry later"))?;
    let mut p = request::parse_chat_limited(body, &st.cfg.chat_limits())?;
    let doc = attach_document(st, acct, &mut p)?;
    p.apply_defaults(&st.cfg.default_effort, st.cfg.default_thinking);
    let thinking = p.opts.thinking();
    let max_ctx = st.max_context_for(acct);
    let max_prompt = st.max_prompt_for(acct);
    let (st2, msgs, tools, opts, scope) = (st.clone(), p.messages.clone(), p.render_tools(), p.opts.clone(), acct.name.clone());
    let (prep, media) = tokio::task::spawn_blocking(move || -> Result<(Prepared, Vec<Arc<MediaInput>>), ApiError> {
        let mut prep = prepare(&st2, &scope, max_prompt, &msgs, tools.as_ref(), &opts)?;
        let media = if st2.cfg.images { prepare_media(&st2, &msgs, &mut prep)? } else { vec![] };
        Ok((prep, media))
    })
    .await
    .map_err(|e| ApiError::internal("prepare_failed", e.to_string()))??;
    if !media.is_empty() && doc.is_some() {
        return Err(ApiError::bad("unsupported_content", Some("messages"), "images cannot be used together with a document"));
    }
    if let Some(d) = &doc {
        if !prep.ids.starts_with(&d.tokens) {
            return Err(ApiError::bad(
                "document_mismatch",
                Some("chat_template_kwargs"),
                "this request does not start with the document's tokens (reasoning_effort / enable_thinking / preserve_thinking must match the document's chat_template_kwargs, and messages must not start with a system message)",
            ));
        }
    }
    let n = prep.ids.len();
    if n > max_prompt {
        return Err(ApiError::bad("context_length_exceeded", Some("messages"), format!("the prompt is {n} tokens; this key allows at most {max_prompt} tokens per request (prompts are not truncated)")));
    }
    let want = p.max_tokens.unwrap_or(st.cfg.default_max_tokens).min(st.cfg.max_output).min(acct.max_tokens.unwrap_or(usize::MAX)).max(1);
    let room = max_ctx - n;
    if room < want.min(st.cfg.min_room) {
        return Err(ApiError::bad("context_length_exceeded", Some("messages"), format!("the prompt is {n} tokens; only {room} of {max_ctx} tokens are left for the output")));
    }
    let max_tokens = want.min(room);
    let id = rand_id("chatcmpl-", 24);
    let sampling = p.sampling.resolve(thinking, rand_u64());
    let req = GenRequest {
        id: id.clone(),
        scope: CacheScope::for_account(&acct.name, acct.internal),
        prompt: prep.ids,
        max_tokens,
        sampling,
        stop_ids: st.stop_ids.clone(),
        hints: PrefixHints {
            system_end: prep.system_end,
            document: doc.as_ref().map(|d| DocRef { id: d.id.clone(), tokens: d.tokens.len() }),
            turn_start: prep.turn_start,
            user_start: prep.user_start,
            pin: None,
        },
        priority: Priority::Normal,
        // 考えの上限は、プロンプトが <think> を開いたまま終わる（考えて書き始める）ときだけ
        think_budget: match p.think_budget {
            Some(b) if thinking && super::output::prompt_opens_think(&prep.tail) => Some(super::backend::ThinkBudget {
                budget: b,
                close: st.tok.encode(THINK_CLOSE).map_err(|e| ApiError::internal("tokenizer_error", format!("tokenizer: {e}")))?,
                end: st.tok.ids.think_end,
            }),
            _ => None,
        },
        // JSON の形: 考えて書き始めるなら </think> の後から絞る
        constraint: p.response_format.clone().map(|sc| super::jsonc::Matcher::new(sc, st.toktab(), thinking && super::output::prompt_opens_think(&prep.tail))),
        media,
    };
    let gen = st.backend.submit(req).map_err(submit_error)?;
    let mut track = Track::new(st, &id, acct, guard, slot, t0, n, p.stream);
    track.max_tokens = max_tokens;
    let ctx = Ctx { id, model: p.model.clone().unwrap_or_else(|| st.model_id.clone()), created: unix_secs(), tail: prep.tail, tools: p.tools.clone(), tool_choice: p.tool_choice.clone(), stops: p.stop.clone(), include_usage: p.include_usage };
    Ok(if p.stream { stream(st.clone(), ctx, track, gen) } else { collect(st, ctx, track, gen).await })
}

/// 口の同時の数の 1 つ分（drop で返す）
pub(crate) struct InflightSlot(Arc<AppState>);

impl InflightSlot {
    pub(crate) fn take(st: &Arc<AppState>) -> Option<Self> {
        if st.inflight.fetch_add(1, Ordering::SeqCst) >= st.cfg.max_inflight {
            st.inflight.fetch_sub(1, Ordering::SeqCst);
            return None;
        }
        Some(InflightSlot(st.clone()))
    }
}

impl Drop for InflightSlot {
    fn drop(&mut self) {
        self.0.inflight.fetch_sub(1, Ordering::SeqCst);
    }
}

/// 出力を作るのに要る物
struct Ctx {
    id: String,
    /// 返事の model（依頼の名前。無ければ口の名前）
    model: String,
    created: u64,
    tail: String,
    tools: Option<Json>,
    tool_choice: ToolChoice,
    stops: Vec<String>,
    include_usage: bool,
}

/// 依頼 1 つの記録。drop したときに数とログを残す（途中で切られた依頼は 499・client_closed）
pub(crate) struct Track {
    st: Arc<AppState>,
    id: String,
    account: Arc<Account>,
    stream: bool,
    t0: Instant,
    t_started: Option<Instant>,
    t_first: Option<Instant>,
    t_last: Option<Instant>,
    prompt: usize,
    cached: usize,
    completion: usize,
    reasoning: usize,
    /// 受け取った語の数（途中で切られたときの completion）
    streamed: usize,
    max_tokens: usize,
    finish: &'static str,
    status: u16,
    error: Option<String>,
    done: bool,
    _guard: super::auth::InflightGuard,
    _slot: InflightSlot,
}

impl Track {
    #[allow(clippy::too_many_arguments)]
    fn new(st: &Arc<AppState>, id: &str, account: &Arc<Account>, guard: super::auth::InflightGuard, slot: InflightSlot, t0: Instant, prompt: usize, stream: bool) -> Self {
        Track {
            st: st.clone(),
            id: id.to_string(),
            account: account.clone(),
            stream,
            t0,
            t_started: None,
            t_first: None,
            t_last: None,
            prompt,
            cached: 0,
            completion: 0,
            reasoning: 0,
            streamed: 0,
            max_tokens: 0,
            finish: "client_closed",
            status: 499,
            error: None,
            done: false,
            _guard: guard,
            _slot: slot,
        }
    }

    fn started(&mut self, cached: usize) {
        self.t_started.get_or_insert_with(Instant::now);
        self.cached = cached.min(self.prompt);
    }

    fn tokens(&mut self, n: usize) {
        self.streamed += n;
        let now = Instant::now();
        self.t_first.get_or_insert(now);
        self.t_last = Some(now);
        self.st.metrics.generated(n);
    }

    fn finish(&mut self, pipe: &OutputPipe, reason: &'static str) {
        self.completion = pipe.n_tokens;
        self.reasoning = pipe.n_reasoning;
        self.finish = reason;
        self.status = 200;
        self.done = true;
    }

    fn fail(&mut self, pipe: &OutputPipe, msg: &str) {
        self.completion = pipe.n_tokens;
        self.reasoning = pipe.n_reasoning;
        self.finish = "error";
        self.status = 500;
        self.error = Some(msg.chars().take(200).collect());
        self.done = true;
    }

    fn secs(a: Instant, b: Option<Instant>) -> Option<f64> {
        b.map(|b| b.saturating_duration_since(a).as_secs_f64())
    }

    fn usage(&self) -> Json {
        json!({
            "prompt_tokens": self.prompt, "completion_tokens": self.completion, "total_tokens": self.prompt + self.completion,
            "prompt_tokens_details": {"cached_tokens": self.cached},
            "completion_tokens_details": {"reasoning_tokens": self.reasoning},
        })
    }

    /// llama-server の timings と同じ名前（tools/effort_bench.py が読む）
    fn timings(&self) -> Json {
        let ms = |a: Option<Instant>, b: Option<Instant>| match (a, b) {
            (Some(a), Some(b)) => b.saturating_duration_since(a).as_secs_f64() * 1000.0,
            _ => 0.0,
        };
        let started = self.t_started.or(self.t_first);
        let prompt_ms = ms(started, self.t_first);
        let predicted_ms = ms(self.t_first, self.t_last);
        let pn = self.prompt.saturating_sub(self.cached);
        json!({
            "cache_n": self.cached, "prompt_n": pn, "prompt_ms": prompt_ms,
            "prompt_per_second": if prompt_ms > 0.0 { pn as f64 * 1000.0 / prompt_ms } else { 0.0 },
            "predicted_n": self.completion, "predicted_ms": predicted_ms,
            "predicted_per_second": if predicted_ms > 0.0 { self.completion.saturating_sub(1) as f64 * 1000.0 / predicted_ms } else { 0.0 },
            "queue_ms": ms(Some(self.t0), started),
        })
    }
}

impl Drop for Track {
    fn drop(&mut self) {
        if !self.done {
            // 客が切った: 計算した分は数える（考えの数は分からない）
            self.completion = self.completion.max(self.streamed);
        }
        let total = self.t0.elapsed().as_secs_f64();
        let decode_tps = match (self.t_first, self.t_last) {
            (Some(a), Some(b)) if self.completion >= 2 && b > a => Some((self.completion - 1) as f64 / (b - a).as_secs_f64()),
            _ => None,
        };
        let ttft = Self::secs(self.t0, self.t_first);
        let queue = Self::secs(self.t0, self.t_started);
        self.st.metrics.request_done(&RequestRecord {
            account: &self.account.name,
            endpoint: EP_CHAT,
            status: self.status,
            finish: self.finish,
            prompt: self.prompt,
            cached: self.cached,
            completion: self.completion,
            reasoning: self.reasoning,
            ttft,
            queue,
            total,
            decode_tps,
        });
        // ⛔ 中身は残さない: 誰が・いつ・何語・どれだけかかったか・どう終わったか
        let ms = |x: Option<f64>| x.map(|s| (s * 1000.0).round() as u64);
        self.st.log_json(json!({
            "ev": "chat", "id": self.id, "account": self.account.name, "status": self.status, "stream": self.stream,
            "prompt": self.prompt, "cached": self.cached, "completion": self.completion, "reasoning": self.reasoning,
            "max_tokens": self.max_tokens, "finish": self.finish, "queue_ms": ms(queue), "ttft_ms": ms(ttft),
            "ms": (total * 1000.0).round() as u64, "tps": decode_tps.map(|x| (x * 10.0).round() / 10.0), "error": self.error,
        }));
    }
}

/// どう終わったか
#[derive(Clone, Copy, PartialEq)]
enum End {
    Stop,
    Length,
}

/// 終わりに残っていた出力の扱い: length で切れたなら、閉じていない道具の呼び出しは答えの文字に（壊れた引数で道具を呼ばせない）
fn tail_choice(end: End, tc: &ToolChoice) -> &ToolChoice {
    if end == End::Length {
        &ToolChoice::None
    } else {
        tc
    }
}

fn finish_reason(end: End, n_calls: usize) -> &'static str {
    match end {
        End::Length => "length",
        End::Stop if n_calls > 0 => "tool_calls",
        End::Stop => "stop",
    }
}

/// 出力の出来事 → API の形（tool_choice "none" で呼んだ物は答えの文字に戻す）
enum Piece {
    Reasoning(String),
    Content(String),
    Call(Json),
}

fn piece(e: OutEvent, tc: &ToolChoice) -> Piece {
    match e {
        OutEvent::Reasoning(s) => Piece::Reasoning(s),
        OutEvent::Content(s) => Piece::Content(s),
        OutEvent::ToolCall(c) if *tc == ToolChoice::None => Piece::Content(tool_call_text(&c)),
        OutEvent::ToolCall(c) => Piece::Call(c.to_openai(&rand_id("call_", 24))),
    }
}

/// 語の流れを最後まで読む（stream しないとき）。途中で切られたら（future が drop）Backend も取り消される
async fn collect(st: &Arc<AppState>, ctx: Ctx, mut track: Track, mut gen: GenStream) -> Response {
    let mut pipe = OutputPipe::new(&st.tok, &ctx.tail, ctx.tools.as_ref(), ctx.stops.clone(), st.stop_ids.clone());
    let (mut reasoning, mut content, mut calls) = (String::new(), String::new(), Vec::<Json>::new());
    let take = |evs: Vec<OutEvent>, tc: &ToolChoice, r: &mut String, c: &mut String, k: &mut Vec<Json>| {
        for e in evs {
            match piece(e, tc) {
                Piece::Reasoning(s) => r.push_str(&s),
                Piece::Content(s) => c.push_str(&s),
                Piece::Call(j) => k.push(j),
            }
        }
    };
    let end = loop {
        match gen.recv().await {
            Some(GenEvent::Started { cached }) => track.started(cached),
            Some(GenEvent::Tokens(ids)) => {
                track.tokens(ids.len());
                for id in ids {
                    take(pipe.push(id), &ctx.tool_choice, &mut reasoning, &mut content, &mut calls);
                    if pipe.is_done() {
                        break;
                    }
                }
                if pipe.is_done() {
                    break End::Stop;
                }
            }
            Some(GenEvent::Done(FinishReason::Stop(id))) => {
                pipe.stop_token(id);
                break End::Stop;
            }
            Some(GenEvent::Done(FinishReason::Length)) => break End::Length,
            Some(GenEvent::Error(m)) => {
                track.fail(&pipe, &m);
                return ApiError::internal("backend_error", m).into_response();
            }
            None => {
                track.fail(&pipe, "backend closed the stream");
                return ApiError::internal("backend_error", "the engine stopped this request").into_response();
            }
        }
    };
    drop(gen); // 止めの文字列で止めたときの取り消し
    take(pipe.finish(), tail_choice(end, &ctx.tool_choice), &mut reasoning, &mut content, &mut calls);
    let fr = finish_reason(end, calls.len());
    track.finish(&pipe, fr);
    let mut msg = json!({"role": "assistant", "content": if calls.is_empty() || !content.is_empty() { Json::from(content) } else { Json::Null }});
    if !reasoning.is_empty() {
        msg["reasoning_content"] = reasoning.into();
    }
    if !calls.is_empty() {
        msg["tool_calls"] = calls.into();
    }
    let body = json!({
        "id": ctx.id, "object": "chat.completion", "created": ctx.created, "model": ctx.model,
        "system_fingerprint": st.fingerprint,
        "choices": [{"index": 0, "message": msg, "logprobs": null, "finish_reason": fr}],
        "usage": track.usage(), "timings": track.timings(),
    });
    let mut r = AxJson(body).into_response();
    if let Ok(v) = HeaderValue::from_str(&ctx.id) {
        r.headers_mut().insert("x-request-id", v);
    }
    r
}

/// SSE で流す（中身は別の task。客が切ったら送れなくなる → task が終わる → Backend も取り消される）
fn stream(st: Arc<AppState>, ctx: Ctx, track: Track, gen: GenStream) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(256);
    let id = ctx.id.clone();
    let keep = Duration::from_secs(st.cfg.keepalive_secs.max(1));
    tokio::spawn(stream_task(st, ctx, track, gen, tx));
    let mut r = Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::new().interval(keep)).into_response();
    r.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    if let Ok(v) = HeaderValue::from_str(&id) {
        r.headers_mut().insert("x-request-id", v);
    }
    r
}

async fn stream_task(st: Arc<AppState>, ctx: Ctx, mut track: Track, mut gen: GenStream, tx: mpsc::Sender<Result<Event, Infallible>>) {
    let head = |choices: Json| json!({"id": ctx.id, "object": "chat.completion.chunk", "created": ctx.created, "model": ctx.model, "system_fingerprint": st.fingerprint, "choices": choices});
    let delta = |d: Json| Event::default().data(head(json!([{"index": 0, "delta": d, "logprobs": null, "finish_reason": null}])).to_string());
    // 送れなければ客が切った: そのまま終わる（Track の drop で 499 が残る・gen の drop で取り消し）。
    // つないだまま読まない客も、stall_timeout の間 送れなければ切る（エンジンの升を持ち続けさせない）
    let stall = st.cfg.stall_timeout;
    macro_rules! send {
        ($e:expr) => {
            match tokio::time::timeout(stall, tx.send(Ok($e))).await {
                Ok(Ok(())) => {}
                _ => return,
            }
        };
    }
    send!(delta(json!({"role": "assistant", "content": ""})));
    let mut pipe = OutputPipe::new(&st.tok, &ctx.tail, ctx.tools.as_ref(), ctx.stops.clone(), st.stop_ids.clone());
    let mut n_calls = 0usize;
    let out = |evs: Vec<OutEvent>, tc: &ToolChoice, n_calls: &mut usize| -> Vec<Event> {
        evs.into_iter()
            .map(|e| match piece(e, tc) {
                Piece::Reasoning(s) => delta(json!({"reasoning_content": s})),
                Piece::Content(s) => delta(json!({"content": s})),
                Piece::Call(mut j) => {
                    j["index"] = (*n_calls).into();
                    *n_calls += 1;
                    delta(json!({"tool_calls": [j]}))
                }
            })
            .collect()
    };
    let end = loop {
        let ev = tokio::select! {
            ev = gen.recv() => ev,
            _ = tx.closed() => return,
        };
        match ev {
            Some(GenEvent::Started { cached }) => track.started(cached),
            Some(GenEvent::Tokens(ids)) => {
                track.tokens(ids.len());
                let mut evs = vec![];
                for id in ids {
                    evs.extend(pipe.push(id));
                    if pipe.is_done() {
                        break;
                    }
                }
                for e in out(evs, &ctx.tool_choice, &mut n_calls) {
                    send!(e);
                }
                if pipe.is_done() {
                    break End::Stop;
                }
            }
            Some(GenEvent::Done(FinishReason::Stop(id))) => {
                pipe.stop_token(id);
                break End::Stop;
            }
            Some(GenEvent::Done(FinishReason::Length)) => break End::Length,
            Some(GenEvent::Error(m)) => {
                track.fail(&pipe, &m);
                let e = ApiError::internal("backend_error", m);
                send!(Event::default().data(e.body().to_string()));
                send!(Event::default().data("[DONE]"));
                return;
            }
            None => {
                track.fail(&pipe, "backend closed the stream");
                let e = ApiError::internal("backend_error", "the engine stopped this request");
                send!(Event::default().data(e.body().to_string()));
                send!(Event::default().data("[DONE]"));
                return;
            }
        }
    };
    drop(gen);
    for e in out(pipe.finish(), tail_choice(end, &ctx.tool_choice), &mut n_calls) {
        send!(e);
    }
    let fr = finish_reason(end, n_calls);
    track.finish(&pipe, fr);
    let mut last = head(json!([{"index": 0, "delta": {}, "logprobs": null, "finish_reason": fr}]));
    last["timings"] = track.timings();
    send!(Event::default().data(last.to_string()));
    if ctx.include_usage {
        let mut u = head(json!([]));
        u["usage"] = track.usage();
        send!(Event::default().data(u.to_string()));
    }
    send!(Event::default().data("[DONE]"));
}

/// 見張りの結果
#[derive(Clone, Debug)]
pub(crate) enum Probe {
    /// 答えた（かかった ms・語の数）
    Answered(u64, usize),
    /// 待ち行列が一杯か時間切れだが、直近 10 秒に語を出している（混んでいるだけ。落とさない）
    Busy,
}

/// /health?probe=1: 短い依頼を実際に通して、1 語でも出るか（見張り用。考えなし・2 語まで）
pub(crate) async fn probe(st: &Arc<AppState>) -> Result<Probe, String> {
    let t0 = Instant::now();
    let msgs = json!([{"role": "user", "content": "Reply with OK."}]);
    let opts = ChatOptions { add_generation_prompt: true, enable_thinking: Some(false), reasoning_effort: Some(st.cfg.default_effort.clone()), preserve_thinking: Some(false), ..Default::default() };
    let st2 = st.clone();
    let prep = tokio::task::spawn_blocking(move || prepare(&st2, "probe:health", 4096, &msgs, None, &opts)).await.map_err(|e| e.to_string())?.map_err(|e| e.message)?;
    let req = GenRequest {
        id: rand_id("probe-", 16),
        scope: CacheScope { namespace: "_health".into(), account: "_health".into() },
        prompt: prep.ids,
        max_tokens: 2,
        sampling: SamplingParams::greedy(),
        stop_ids: st.stop_ids.clone(),
        hints: PrefixHints::default(),
        priority: Priority::Normal,
        think_budget: None,
        media: vec![],
        constraint: None,
    };
    // 動いているか: 直近 10 秒に語を出した・Backend の計算の数（プレフィルも）が増えた
    let p0 = st.backend.stats().progress;
    let moved = || st.metrics.recent_tokens() > 0 || st.backend.stats().progress != p0;
    let mut gen = match st.backend.submit(req) {
        Ok(g) => g,
        Err(super::SubmitError::QueueFull { .. }) => {
            // 一杯: 3 秒のうちに計算が進めば混んでいるだけ
            for _ in 0..30 {
                if moved() {
                    return Ok(Probe::Busy);
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            return Err("queue full and no progress for 3 s".into());
        }
        Err(e) => return Err(format!("{e:?}")),
    };
    // 止める語で終わったときも 1 語と数える（すぐ <|im_end|> を出すのも答え）
    let wait = async move {
        let mut n = 0;
        while let Some(ev) = gen.recv().await {
            match ev {
                GenEvent::Tokens(t) => n += t.len(),
                GenEvent::Done(FinishReason::Stop(_)) => return Ok(n + 1),
                GenEvent::Done(FinishReason::Length) => return Ok(n),
                GenEvent::Error(m) => return Err(m),
                GenEvent::Started { .. } => {}
            }
        }
        Err("the engine closed the probe".to_string())
    };
    let res = tokio::time::timeout(st.cfg.probe_timeout, wait).await;
    match res {
        Ok(Ok(n)) => Ok(Probe::Answered(t0.elapsed().as_millis() as u64, n)),
        Ok(Err(m)) => Err(m),
        Err(_) if moved() => Ok(Probe::Busy),
        Err(_) => Err(format!("no answer within {:?}", st.cfg.probe_timeout)),
    }
}
