//! OpenAI 互換の口（HTTP・SSE。axum 0.8＋tokio）。
//!
//! 形: 口（この mod）→ `Backend`（依頼を受けて語を流す trait。backend.rs）→ エンジン。
//!   - `POST /v1/chat/completions`（stream あり・なし・tools・reasoning_content・chat_template_kwargs・usage）
//!   - `GET /v1/models`（id に重みの版＝sha の先頭 8 桁）・`GET /health`（読み込み中 503／準備完了 200。`?probe=1` で 1 語答えさせる）
//!   - `GET /metrics`（Prometheus。アカウントごとの語の数・待ち行列・TTFT・tok/s・当たり率。数だけ）
//!   - `POST/GET/DELETE /v1/documents`（文書の先計算。docs/API.md 3.7）・`POST /admin/drain`
//! 試しは `fake::FakeBackend`（決まった語を流す）。M1 のモデルは `engine::Engine`（1 歩）を実装して `SimpleScheduler` に載せる。
//! ⛔ 鍵と依頼の中身はログにも /metrics にも出さない（誰が・いつ・何語だけ）。
pub mod auth;
pub mod backend;
pub mod chat;
pub mod cli;
pub mod docs;
pub mod engine;
pub mod jsonc;
pub mod fake;
pub mod hostcache;
pub mod m1;
pub mod metrics;
pub mod output;
pub mod prewarm;
pub mod request;
pub mod sampler;
pub mod util;

pub use auth::{Account, Auth, KeysFile};
pub use backend::*;
pub use engine::{Engine, EngineInfo, ItemOut, SchedConfig, SimpleScheduler, StepItem};

use crate::template::{ChatOptions, ChatTemplate, EFFORTS};
use crate::tokenizer::Tokenizer;
use axum::extract::{DefaultBodyLimit, Path, Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json as AxJson, Router};
use metrics::{Gauges, Metrics};
use serde_json::{json, Value as Json};
use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use util::LogSink;

/// 口の設定（既定は docs/API.md 3・3.5）
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// /v1/models の id の頭（id = "<model_name>-<sha 8 桁>"）
    pub model_name: String,
    /// 重みの sha256（16 進。先頭 8 桁を id に）
    pub weights_sha: String,
    /// 1 依頼の語の上限（入力＋出力）。入力がこれを超えたら 400（切り詰めない）
    pub max_context: usize,
    /// 入力（プロンプト）の語の上限（max_context とこの小さい方。serve の既定は 60,000）。超えたら 400。
    /// 切り分けはこの数を超えた所でやめる（長すぎる依頼を全部切ってから断らない）
    pub max_prompt_tokens: usize,
    /// 1 つの message（content・reasoning_content・tool_calls の合計）のバイト数の上限（超えたら 400 message_too_large）
    pub max_message_bytes: usize,
    /// messages の数の上限
    pub max_messages: usize,
    /// tools（JSON にしたとき）のバイト数の上限
    pub max_tools_bytes: usize,
    /// enable_thinking を書かない依頼に入れる値（serve の既定は false＝考えない。docs/API.md 3 の元の決めは true）
    pub default_thinking: bool,
    /// max_tokens を書かない依頼の出力の上限
    pub default_max_tokens: usize,
    /// 出力の上限（これより大きい max_tokens は切り詰める）
    pub max_output: usize,
    /// 出力に残すべき最小の語数（足りなければ 400。考える余地が無い）
    pub min_room: usize,
    /// 口で同時に持つ生成の依頼の上限（超えたら 429。待ち行列の上限は Backend）
    pub max_inflight: usize,
    pub max_body_bytes: usize,
    /// 画像の入力（--mmproj を渡したとき）。1 依頼の画像の数・1 枚の data URI を戻したバイト数の上限
    pub images: bool,
    pub max_images: usize,
    pub max_image_bytes: usize,
    /// reasoning_effort を書かない依頼に入れる値（docs/API.md 3: medium）
    pub default_effort: String,
    /// llama.cpp と同じく FIM の 3 語（<|fim_pad|>・<|repo_name|>・<|file_sep|>）でも止める
    pub fim_stops: bool,
    /// SSE の keep-alive（読み込みが長い間も切れないように）
    pub keepalive_secs: u64,
    pub max_documents_per_account: usize,
    pub max_document_tokens_per_account: usize,
    pub probe_timeout: Duration,
    /// 見張りの結果を使い回す間
    pub probe_cache: Duration,
    /// stream でこの間 送れなければ（つないだまま読まない客）切る
    pub stall_timeout: Duration,
    /// /metrics にも admin の鍵を求める（既定は鍵なし＝127.0.0.1 の Prometheus 用）
    pub metrics_key: bool,
    /// SIGTERM の後、走っている依頼が自分で終わるのを待つ間（過ぎたら取り消す）
    pub shutdown_grace: Duration,
    /// SIGTERM からこれだけ経ったら、何が残っていても終わる（systemd の止めを 10 秒以内に）
    pub exit_deadline: Duration,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            model_name: "qwen3.8-27b".into(),
            weights_sha: "0".repeat(64),
            max_context: 262_144,
            max_prompt_tokens: 262_144,
            max_message_bytes: 1 << 20,
            max_messages: 4096,
            max_tools_bytes: 1 << 20,
            default_thinking: false,
            default_max_tokens: 32_768,
            max_output: 131_072,
            min_room: 1024,
            max_inflight: 1024,
            // 本文は読む前に 4MiB で切る（2026-11-10 の見直し: 64MiB だと 1 依頼で何 GB も使えた）
            max_body_bytes: 4 << 20,
            images: false,
            max_images: 8,
            max_image_bytes: 16 << 20,
            default_effort: "medium".into(),
            fim_stops: true,
            keepalive_secs: 15,
            max_documents_per_account: 64,
            max_document_tokens_per_account: 4 << 20,
            probe_timeout: Duration::from_secs(30),
            probe_cache: Duration::from_secs(5),
            stall_timeout: Duration::from_secs(120),
            metrics_key: false,
            shutdown_grace: Duration::from_secs(2),
            exit_deadline: Duration::from_secs(8),
        }
    }
}

impl ServerConfig {
    /// 依頼を読むときの大きさの上限（request.rs）
    pub fn limits(&self) -> request::Limits {
        request::Limits { max_message_bytes: self.max_message_bytes, max_messages: self.max_messages, max_tools_bytes: self.max_tools_bytes, allow_images: false, max_images: self.max_images }
    }
    /// chat の依頼（画像を受けるのはここだけ。文書には入れない）
    pub fn chat_limits(&self) -> request::Limits {
        request::Limits { allow_images: self.images, ..self.limits() }
    }
}

/// 口の持ち物（全部の依頼で共有）
pub struct AppState {
    pub cfg: ServerConfig,
    pub backend: Arc<dyn Backend>,
    pub tok: Arc<Tokenizer>,
    pub tpl: Arc<ChatTemplate>,
    pub auth: Auth,
    pub metrics: Metrics,
    pub docs: docs::DocRegistry,
    /// 止める語（generation_config の eos ＋ fim_stops なら FIM の 3 語）
    pub stop_ids: Vec<u32>,
    pub model_id: String,
    pub fingerprint: String,
    pub log: LogSink,
    pub draining: AtomicBool,
    pub inflight: AtomicUsize,
    pub started: u64,
    /// 見張りの結果（5 秒は使い回す・同時に 1 つだけ。鍵なしの /health で推論を連打させない）
    probe_cache: tokio::sync::Mutex<Option<(Instant, Result<chat::Probe, String>)>>,
    /// JSON の形の絞りに使う語の表（最初の依頼で作る）
    toktab: std::sync::OnceLock<Arc<jsonc::TokTable>>,
}

/// 止める語: 学習の側（generation_config.json の eos_token_id = <|im_end|>・<|endoftext|>）に、
/// llama.cpp が EOG に足す FIM の 3 語を足す（比べる相手の llama-server と同じ所で止まるように。番号は tokenizer から引く）
pub fn stop_ids(tok: &Tokenizer, fim: bool) -> Vec<u32> {
    let mut v = tok.eos.clone();
    if fim {
        for s in ["<|fim_pad|>", "<|repo_name|>", "<|file_sep|>"] {
            if let Some(id) = tok.token_to_id(s) {
                if !v.contains(&id) {
                    v.push(id);
                }
            }
        }
    }
    v
}

impl AppState {
    /// JSON の形の絞りの語の表（止める語は口と同じ）
    pub fn toktab(&self) -> Arc<jsonc::TokTable> {
        self.toktab.get_or_init(|| Arc::new(jsonc::TokTable::new(&self.tok, &self.stop_ids))).clone()
    }

    pub fn new(cfg: ServerConfig, backend: Arc<dyn Backend>, tok: Arc<Tokenizer>, tpl: Arc<ChatTemplate>, auth: Auth, log: LogSink) -> Result<Arc<Self>, String> {
        if !EFFORTS.contains(&cfg.default_effort.as_str()) {
            return Err(format!("default_effort は {} のどれか", EFFORTS.join("・")));
        }
        if cfg.weights_sha.len() < 8 || !cfg.weights_sha.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err("weights_sha は 16 進（8 桁以上）".into());
        }
        // テンプレートが描けるか（起動時に 1 回。後の描けない誤りは依頼のせい＝400 にできる）
        let probe = json!([{"role": "user", "content": "ping"}]);
        let opt = ChatOptions { add_generation_prompt: true, reasoning_effort: Some(cfg.default_effort.clone()), ..Default::default() };
        tpl.render(&probe, None, &opt).map_err(|e| format!("chat_template が描けない: {e}"))?;
        let sha8 = cfg.weights_sha[..8].to_ascii_lowercase();
        Ok(Arc::new(AppState {
            stop_ids: stop_ids(&tok, cfg.fim_stops),
            model_id: format!("{}-{sha8}", cfg.model_name),
            fingerprint: format!("tachyon-{sha8}"),
            cfg,
            backend,
            tok,
            tpl,
            auth,
            metrics: Metrics::new(),
            docs: docs::DocRegistry::default(),
            log,
            draining: AtomicBool::new(false),
            inflight: AtomicUsize::new(0),
            started: util::unix_secs(),
            probe_cache: tokio::sync::Mutex::new(None),
            toktab: std::sync::OnceLock::new(),
        }))
    }

    /// ログ 1 行（JSON）。⛔ 中身・鍵は入れない
    pub fn log_json(&self, mut v: Json) {
        if let Some(o) = v.as_object_mut() {
            o.insert("t".into(), util::utc_now().into());
        }
        (self.log)(&v.to_string());
    }

    pub fn authenticate(&self, h: &HeaderMap) -> Result<Arc<Account>, ApiError> {
        self.auth.check(h).map_err(|e| match e {
            auth::AuthError::Missing => ApiError::auth("missing_api_key", "an API key is required (Authorization: Bearer <key>)"),
            auth::AuthError::Invalid => ApiError::auth("invalid_api_key", "invalid API key"),
        })
    }

    /// 受け付けられる状態か（読み込み中・止めている間は 503）
    pub fn check_ready(&self) -> Result<(), ApiError> {
        if self.draining.load(Ordering::SeqCst) {
            return Err(ApiError::unavailable("draining", 30, "the server is draining; retry on another instance"));
        }
        match self.backend.status() {
            Status::Ready => Ok(()),
            Status::Loading { .. } => Err(ApiError::unavailable("loading", 10, "the model is loading")),
            Status::Failed(_) => Err(ApiError::unavailable("backend_failed", 60, "the engine failed to start")),
        }
    }

    /// その鍵の 1 依頼の語の上限（全体・エンジン・鍵の小さい方）
    pub fn max_context_for(&self, a: &Account) -> usize {
        let mut m = self.cfg.max_context;
        if let Some(x) = self.backend.max_context() {
            m = m.min(x);
        }
        if let Some(x) = a.max_context {
            m = m.min(x);
        }
        m
    }

    /// その鍵の 1 依頼の入力の語の上限（max_context_for と max_prompt_tokens の小さい方）
    pub fn max_prompt_for(&self, a: &Account) -> usize {
        self.max_context_for(a).min(self.cfg.max_prompt_tokens)
    }

    /// 断った依頼を数えてログに残し、誤りの返事を作る
    pub fn reject(&self, acct: Option<&Account>, endpoint: &'static str, e: ApiError) -> Response {
        let name = acct.map(|a| a.name.as_str()).unwrap_or("-");
        self.metrics.rejected(name, endpoint, e.status.as_u16(), e.code);
        // 誤りの文（message）は依頼の中身を含みうるので残さない。code と param（欄の名前）だけ
        self.log_json(json!({"ev": "reject", "endpoint": endpoint, "account": name, "status": e.status.as_u16(), "code": e.code, "param": e.param}));
        e.into_response()
    }
}

/// OpenAI 形の誤り {"error": {"message", "type", "param", "code"}}
#[derive(Clone, Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub kind: &'static str,
    pub code: &'static str,
    pub param: Option<String>,
    pub message: String,
    pub retry_after: Option<u64>,
}

impl ApiError {
    fn new(status: StatusCode, kind: &'static str, code: &'static str, msg: impl Into<String>) -> Self {
        ApiError { status, kind, code, param: None, message: msg.into(), retry_after: None }
    }
    pub fn bad(code: &'static str, param: Option<&str>, msg: impl Into<String>) -> Self {
        ApiError { param: param.map(str::to_string), ..Self::new(StatusCode::BAD_REQUEST, "invalid_request_error", code, msg) }
    }
    pub fn auth(code: &'static str, msg: impl Into<String>) -> Self {
        Self::new(StatusCode::UNAUTHORIZED, "authentication_error", code, msg)
    }
    pub fn forbidden(code: &'static str, msg: impl Into<String>) -> Self {
        Self::new(StatusCode::FORBIDDEN, "permission_error", code, msg)
    }
    pub fn not_found(code: &'static str, msg: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, "not_found_error", code, msg)
    }
    pub fn limit(code: &'static str, retry_after: u64, msg: impl Into<String>) -> Self {
        ApiError { retry_after: Some(retry_after), ..Self::new(StatusCode::TOO_MANY_REQUESTS, "rate_limit_error", code, msg) }
    }
    pub fn unavailable(code: &'static str, retry_after: u64, msg: impl Into<String>) -> Self {
        ApiError { retry_after: Some(retry_after), ..Self::new(StatusCode::SERVICE_UNAVAILABLE, "unavailable_error", code, msg) }
    }
    pub fn not_implemented(code: &'static str, msg: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_IMPLEMENTED, "not_implemented_error", code, msg)
    }
    pub fn internal(code: &'static str, msg: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", code, msg)
    }
    pub fn body(&self) -> Json {
        json!({"error": {"message": self.message, "type": self.kind, "param": self.param, "code": self.code}})
    }
}

impl From<request::ReqError> for ApiError {
    fn from(e: request::ReqError) -> Self {
        ApiError { param: e.param, ..ApiError::new(StatusCode::BAD_REQUEST, "invalid_request_error", e.code, e.message) }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut r = (self.status, AxJson(self.body())).into_response();
        if let Some(s) = self.retry_after {
            r.headers_mut().insert("retry-after", HeaderValue::from(s));
        }
        r
    }
}

/// Backend の受け付けの誤り → 返事
pub fn submit_error(e: SubmitError) -> ApiError {
    match e {
        SubmitError::QueueFull { retry_after } => ApiError::limit("queue_full", retry_after, "the server is busy (queue full); retry later"),
        SubmitError::NotReady => ApiError::unavailable("not_ready", 10, "the engine is not ready"),
        SubmitError::TooLong { max } => ApiError::bad("context_length_exceeded", Some("messages"), format!("the engine accepts at most {max} tokens per request")),
        SubmitError::Invalid(m) => ApiError::bad("invalid_request", None, m),
        SubmitError::Unsupported => ApiError::not_implemented("unsupported", "this backend does not support that"),
        SubmitError::Internal(m) => ApiError::internal("backend_error", m),
    }
}

pub fn router(st: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics_text))
        .route("/v1/models", get(models))
        .route("/v1/models/{id}", get(model_one))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/v1/documents", post(docs::create).get(docs::list))
        .route("/v1/documents/batch", post(docs::create_batch))
        .route("/v1/documents/{id}", get(docs::get_one).delete(docs::delete))
        .route("/admin/drain", post(admin_drain))
        .route("/admin/prewarm", post(admin_prewarm))
        .fallback(not_found)
        .layer(DefaultBodyLimit::max(st.cfg.max_body_bytes))
        .layer(axum::middleware::map_response_with_state(st.clone(), json_rejections))
        .with_state(st)
}

/// axum が本文を読む所で断った返事（413 本文が大きすぎる・415 など、文字だけの返事）を OpenAI 形の JSON に
async fn json_rejections(State(st): State<Arc<AppState>>, r: Response) -> Response {
    let s = r.status();
    let is_json = r.headers().get(axum::http::header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).is_some_and(|v| v.starts_with("application/json"));
    if is_json || !(s.is_client_error() || s.is_server_error()) || r.headers().contains_key("retry-after") {
        return r;
    }
    let (code, msg) = match s {
        StatusCode::PAYLOAD_TOO_LARGE => ("request_too_large", format!("the request body is larger than {} bytes", st.cfg.max_body_bytes)),
        StatusCode::UNSUPPORTED_MEDIA_TYPE => ("unsupported_media_type", "unsupported content type".to_string()),
        StatusCode::METHOD_NOT_ALLOWED => ("method_not_allowed", "method not allowed".to_string()),
        _ => ("bad_request", "the request could not be read".to_string()),
    };
    let kind = if s.is_server_error() { "server_error" } else { "invalid_request_error" };
    (s, AxJson(json!({"error": {"message": msg, "type": kind, "param": null, "code": code}}))).into_response()
}

/// 待ち受ける（shutdown が終わったら、新しい接続を断って走っている依頼を待つ）
pub async fn serve(listener: tokio::net::TcpListener, st: Arc<AppState>, shutdown: impl Future<Output = ()> + Send + 'static) -> std::io::Result<()> {
    let addr = listener.local_addr()?;
    st.log_json(json!({"ev": "listen", "addr": addr.to_string(), "model": st.model_id, "auth": !st.auth.is_open(), "stop_ids": st.stop_ids}));
    axum::serve(listener, router(st)).with_graceful_shutdown(shutdown).await
}

/// 止める手順（SIGTERM の後。試しからも呼ぶ）: 受付を止め（503）、走っている依頼を最大 grace 待ち、
/// 残っていれば Backend に取り消させて（客には誤りの塊か 500）、それが返るのを最大 3 秒待つ
pub async fn drain_and_cancel(st: &Arc<AppState>, grace: Duration) {
    st.draining.store(true, Ordering::SeqCst);
    st.log_json(json!({"ev": "drain", "inflight": st.inflight.load(Ordering::SeqCst)}));
    let t0 = Instant::now();
    while st.inflight.load(Ordering::SeqCst) > 0 && t0.elapsed() < grace {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let left = st.inflight.load(Ordering::SeqCst);
    st.backend.shutdown();
    if left > 0 {
        st.log_json(json!({"ev": "cancel_all", "inflight": left}));
    }
    let t1 = Instant::now();
    while st.inflight.load(Ordering::SeqCst) > 0 && t1.elapsed() < Duration::from_secs(3) {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    st.log_json(json!({"ev": "drained", "inflight": st.inflight.load(Ordering::SeqCst), "ms": t0.elapsed().as_millis() as u64}));
}

/// SIGTERM・Ctrl-C を待って drain_and_cancel。⭐ 合図から exit_deadline（既定 8 秒）経ったら、何が残っていても
/// プロセスを終える（長いプレフィルの途中・読み込みの途中・切れない接続があっても、systemd の止めが 10 秒以内に終わる）
pub async fn shutdown_signal(st: Arc<AppState>) {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = term => {},
    }
    let deadline = st.cfg.exit_deadline;
    let log = st.log.clone();
    std::thread::Builder::new()
        .name("tachyon-exit".into())
        .spawn(move || {
            std::thread::sleep(deadline);
            log(&json!({"ev": "exit_deadline", "secs": deadline.as_secs_f64(), "t": util::utc_now()}).to_string());
            util::hard_exit(0);
        })
        .ok();
    drain_and_cancel(&st, st.cfg.shutdown_grace).await;
}

async fn not_found() -> Response {
    ApiError::not_found("not_found", "unknown endpoint").into_response()
}

async fn health(State(st): State<Arc<AppState>>, Query(q): Query<HashMap<String, String>>) -> Response {
    let s = st.backend.stats();
    match st.backend.status() {
        // state: loading・ready・draining・failed（status は前からの名前。準備完了は llama-server と同じ "ok"）
        Status::Loading { progress, what } => {
            (StatusCode::SERVICE_UNAVAILABLE, AxJson(json!({"status": "loading", "state": "loading", "progress": progress, "what": what}))).into_response()
        }
        Status::Failed(m) => (StatusCode::INTERNAL_SERVER_ERROR, AxJson(json!({"status": "error", "state": "failed", "message": m}))).into_response(),
        Status::Ready if st.draining.load(Ordering::SeqCst) => {
            (StatusCode::SERVICE_UNAVAILABLE, AxJson(json!({"status": "draining", "state": "draining", "inflight": st.inflight.load(Ordering::SeqCst)}))).into_response()
        }
        Status::Ready => {
            let mut body = json!({"status": "ok", "state": "ready", "model": st.model_id, "waiting": s.waiting, "running": s.running, "inflight": st.inflight.load(Ordering::SeqCst),
                                  "max_waiting": s.max_waiting, "max_context": st.cfg.max_context.min(st.backend.max_context().unwrap_or(usize::MAX)),
                                  "max_prompt_tokens": st.cfg.max_prompt_tokens});
            if let Some(h) = hostcache::STATS.lock().ok().and_then(|g| g.clone()) {
                body["prefix_host"] = h;
            }
            if q.get("probe").is_some_and(|v| v != "0") {
                let res = {
                    let mut c = st.probe_cache.lock().await;
                    match &*c {
                        Some((t, r)) if t.elapsed() < st.cfg.probe_cache => r.clone(),
                        _ => {
                            let r = chat::probe(&st).await;
                            *c = Some((Instant::now(), r.clone()));
                            r
                        }
                    }
                };
                match res {
                    Ok(chat::Probe::Answered(ms, n)) => {
                        body["probe"] = "answered".into();
                        body["probe_ms"] = ms.into();
                        body["probe_tokens"] = n.into();
                    }
                    Ok(chat::Probe::Busy) => body["probe"] = "busy".into(),
                    Err(m) => return (StatusCode::SERVICE_UNAVAILABLE, AxJson(json!({"status": "probe_failed", "message": m}))).into_response(),
                }
            }
            (StatusCode::OK, AxJson(body)).into_response()
        }
    }
}

fn model_json(st: &AppState) -> Json {
    json!({
        "id": st.model_id, "object": "model", "created": st.started, "owned_by": "tachyon-v100",
        "root": st.cfg.model_name, "weights_sha256": st.cfg.weights_sha,
        "max_context": st.cfg.max_context.min(st.backend.max_context().unwrap_or(usize::MAX)), "max_output": st.cfg.max_output,
    })
}

async fn models(State(st): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(e) = st.authenticate(&headers) {
        return st.reject(None, "models", e);
    }
    AxJson(json!({"object": "list", "data": [model_json(&st)]})).into_response()
}

async fn model_one(State(st): State<Arc<AppState>>, headers: HeaderMap, Path(id): Path<String>) -> Response {
    if let Err(e) = st.authenticate(&headers) {
        return st.reject(None, "models", e);
    }
    if id != st.model_id && id != st.cfg.model_name {
        return ApiError::not_found("model_not_found", format!("model {id:?} not found; available: {}", st.model_id)).into_response();
    }
    AxJson(model_json(&st)).into_response()
}

async fn metrics_text(State(st): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if st.cfg.metrics_key {
        match st.authenticate(&headers) {
            Ok(a) if a.admin => {}
            Ok(a) => return st.reject(Some(&a), "metrics", ApiError::forbidden("not_admin", "this key cannot read /metrics")),
            Err(e) => return st.reject(None, "metrics", e),
        }
    }
    let ready = st.backend.status() == Status::Ready;
    let g = Gauges {
        ready,
        draining: st.draining.load(Ordering::SeqCst),
        inflight: st.inflight.load(Ordering::SeqCst),
        stats: st.backend.stats(),
        accounts: st.auth.accounts.iter().map(|a| (a.name.clone(), a.inflight())).collect(),
        documents: st.docs.per_account(),
    };
    ([("content-type", "text/plain; version=0.0.4; charset=utf-8")], st.metrics.render(&g)).into_response()
}

/// 受付を止める（{"drain": false} で戻す）。admin の鍵だけ
/// 頭の先計算の札を読み直す（--prewarm-dir の *.json。admin の鍵）。待たずに返る（進みはログの prewarm・/health の prefix_host）
async fn admin_prewarm(State(st): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let acct = match st.authenticate(&headers) {
        Ok(a) => a,
        Err(e) => return st.reject(None, "admin", e),
    };
    if !acct.admin {
        return st.reject(Some(&acct), "admin", ApiError::forbidden("not_admin", "this key cannot use /admin"));
    }
    let Some(d) = prewarm::dir() else {
        return st.reject(Some(&acct), "admin", ApiError::bad("invalid_request", None, "the server was started without --prewarm-dir"));
    };
    st.log_json(json!({"ev": "admin_prewarm", "account": acct.name}));
    tokio::spawn(prewarm::run(st.clone()));
    AxJson(json!({"started": true, "dir": d.display().to_string()})).into_response()
}

async fn admin_drain(State(st): State<Arc<AppState>>, headers: HeaderMap, body: axum::body::Bytes) -> Response {
    let acct = match st.authenticate(&headers) {
        Ok(a) => a,
        Err(e) => return st.reject(None, "admin", e),
    };
    if !acct.admin {
        return st.reject(Some(&acct), "admin", ApiError::forbidden("not_admin", "this key cannot use /admin"));
    }
    // 本文なし（か {}）で止める、{"drain": false} で戻す。読めない本文は 400（間違って止めない）
    let on = if body.iter().all(u8::is_ascii_whitespace) {
        true
    } else {
        match serde_json::from_slice::<Json>(&body).ok().as_ref().map(|v| v.get("drain")) {
            Some(None) => true,
            Some(Some(Json::Bool(b))) => *b,
            _ => return st.reject(Some(&acct), "admin", ApiError::bad("invalid_value", Some("drain"), "body must be empty or {\"drain\": true|false}")),
        }
    };
    st.draining.store(on, Ordering::SeqCst);
    st.log_json(json!({"ev": "admin_drain", "account": acct.name, "drain": on}));
    AxJson(json!({"draining": on, "inflight": st.inflight.load(Ordering::SeqCst)})).into_response()
}
