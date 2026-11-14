//! 文書を先に計算して置いておく口（docs/API.md 3.7）。
//!   POST /v1/documents      {"messages": [system・文書の回…], "tools"?, "chat_template_kwargs"?, "name"?, "priority"?: "low", "shared"?}
//!   POST /v1/documents/batch {"documents": [上と同じ物…], "priority"?: "low"}（夜の一括）
//!   GET  /v1/documents[/{id}]・DELETE /v1/documents/{id}
//! 会話の依頼に "document": "doc-…" を付けると、文書の messages を前に足して描き、文書の語の分は Backend が使い回す。
//! 文書の語 = 文書の messages の後に user の回が来たときの、その回の手前まで（テンプレートを描いて切る）。
//! 持ち主: お客さまの文書はそのアカウントだけ。社内の文書（shared）は社内の鍵どうしで見える（docs/API.md 3.6）。
use super::backend::{CacheScope, DocJob, DocState, PrefixHints, Priority, SubmitError};
use super::request::{self, parse_kwargs, parse_messages_limited, parse_tools_limited, req_err};
use super::util::{rand_id, unix_secs};
use super::{Account, ApiError, AppState};
use crate::template::ChatOptions;
use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::Json as AxJson;
use serde_json::{json, Map, Value as Json};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

pub const EP_DOCS: &str = "documents";


pub struct DocRecord {
    pub id: String,
    pub name: Option<String>,
    pub account: String,
    /// 社内の鍵どうしで見える
    pub shared: bool,
    pub messages: Json,
    pub tools: Option<Json>,
    /// 描いたときの chat_template_kwargs（既定を入れた後）
    pub kwargs: Json,
    pub tokens: Vec<u32>,
    pub created: u64,
    pub priority: Priority,
}

#[derive(Default)]
pub struct DocRegistry {
    map: Mutex<BTreeMap<String, Arc<DocRecord>>>,
}

fn visible(d: &DocRecord, a: &Account) -> bool {
    d.account == a.name || (d.shared && a.internal)
}

impl DocRegistry {
    /// アカウントの枠（数・語の数）の中なら入れる。枠を超えるなら Err(今の数, 今の語の数)。数えると入れるを 1 つの鍵の中で
    pub fn insert_within(&self, d: DocRecord, max_docs: usize, max_tokens: usize) -> Result<Arc<DocRecord>, (usize, usize)> {
        let mut m = self.map.lock().unwrap();
        let (n, t) = m.values().filter(|x| x.account == d.account).fold((0, 0), |(n, t), x| (n + 1, t + x.tokens.len()));
        if n + 1 > max_docs || t + d.tokens.len() > max_tokens {
            return Err((n, t));
        }
        let d = Arc::new(d);
        m.insert(d.id.clone(), d.clone());
        Ok(d)
    }

    /// その鍵から見える文書
    pub fn get(&self, id: &str, a: &Account) -> Option<Arc<DocRecord>> {
        self.map.lock().unwrap().get(id).filter(|d| visible(d, a)).cloned()
    }

    pub fn list(&self, a: &Account) -> Vec<Arc<DocRecord>> {
        self.map.lock().unwrap().values().filter(|d| visible(d, a)).cloned().collect()
    }

    pub fn remove(&self, id: &str) -> Option<Arc<DocRecord>> {
        self.map.lock().unwrap().remove(id)
    }

    /// そのアカウントの文書の数と語の数
    pub fn usage(&self, account: &str) -> (usize, usize) {
        let m = self.map.lock().unwrap();
        m.values().filter(|d| d.account == account).fold((0, 0), |(n, t), d| (n + 1, t + d.tokens.len()))
    }

    pub fn per_account(&self) -> Vec<(String, usize, usize)> {
        let mut by: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for d in self.map.lock().unwrap().values() {
            let e = by.entry(d.account.clone()).or_default();
            e.0 += 1;
            e.1 += d.tokens.len();
        }
        by.into_iter().map(|(a, (n, t))| (a, n, t)).collect()
    }
}

/// 文書の語（文書の messages の後に user の回が来たときの、その回の手前まで）
pub(crate) fn render_prefix(st: &AppState, scope: &str, max_tokens: usize, messages: &Json, tools: Option<&Json>, opts: &ChatOptions) -> Result<(Vec<u32>, Option<usize>), ApiError> {
    // 文書の後ろに仮の user の回を置いて描き、その回の頭で切る（印は毎回変える＝文書の中の文字とぶつからない）
    let sentinel = rand_id("TACHYON_DOCUMENT_BOUNDARY_", 20);
    let mut m = messages.as_array().cloned().unwrap_or_default();
    m.push(json!({"role": "user", "content": sentinel}));
    let o = ChatOptions { add_generation_prompt: false, ..opts.clone() };
    let text = st.tpl.render(&Json::Array(m), tools, &o).map_err(|e| super::chat::template_error(&e))?;
    let cut = text.rfind(&format!("<|im_start|>user\n{sentinel}")).ok_or_else(|| ApiError::internal("document_render", "could not find the end of the document in the rendered prompt"))?;
    let ids = st.tok.encode_limited(Some(scope), &text[..cut], max_tokens).map_err(|e| match e {
        crate::tokenizer::EncodeError::TooLong { at_least } => {
            ApiError::bad("context_length_exceeded", Some("messages"), format!("the document is at least {at_least} tokens; this key allows at most {max_tokens}"))
        }
        crate::tokenizer::EncodeError::Failed(e) => ApiError::internal("tokenizer_error", format!("tokenizer: {e}")),
    })?;
    let system_end = if text.starts_with("<|im_start|>system\n") { ids.iter().skip(1).position(|&t| t == st.tok.ids.im_start).map(|i| i + 1) } else { None };
    Ok((ids, system_end))
}

struct DocSpec {
    messages: Json,
    tools: Option<Json>,
    opts: ChatOptions,
    name: Option<String>,
    priority: Priority,
    shared: bool,
}

fn parse_spec(o: &Map<String, Json>, acct: &Account, default_priority: Priority, cfg: &super::ServerConfig) -> Result<DocSpec, ApiError> {
    let messages = parse_messages_limited(o.get("messages"), &cfg.limits())?;
    let tools = parse_tools_limited(o.get("tools"), &cfg.limits())?;
    let mut opts = parse_kwargs(o, false)?;
    request::apply_opt_defaults(&mut opts, tools.is_some(), &cfg.default_effort, cfg.default_thinking);
    let name = match o.get("name") {
        None | Some(Json::Null) => None,
        Some(Json::String(s)) if s.len() <= 256 => Some(s.clone()),
        Some(_) => return Err(req_err("invalid_value", "name", "name must be a string of at most 256 bytes").into()),
    };
    let priority = match o.get("priority").and_then(Json::as_str) {
        None => default_priority,
        Some("low") => Priority::Low,
        Some("normal") => Priority::Normal,
        Some(_) => return Err(req_err("invalid_value", "priority", "priority must be \"normal\" or \"low\"").into()),
    };
    let shared = request::opt_bool(o, "shared")?.unwrap_or(acct.internal);
    if shared && !acct.internal {
        return Err(req_err("invalid_value", "shared", "only internal accounts can share documents").into());
    }
    Ok(DocSpec { messages, tools, opts, name, priority, shared })
}

fn kwargs_json(o: &ChatOptions) -> Json {
    let mut m = o.extra.clone();
    if let Some(v) = o.enable_thinking {
        m.insert("enable_thinking".into(), v.into());
    }
    if let Some(v) = &o.reasoning_effort {
        m.insert("reasoning_effort".into(), v.clone().into());
    }
    if let Some(v) = o.preserve_thinking {
        m.insert("preserve_thinking".into(), v.into());
    }
    Json::Object(m)
}

fn state_json(s: Option<DocState>) -> Json {
    match s {
        None => json!({"status": "unknown"}),
        Some(DocState::Queued) => json!({"status": "queued"}),
        Some(DocState::Computing { done, total }) => json!({"status": "computing", "progress": {"done": done, "total": total}}),
        Some(DocState::Ready { tier }) => json!({"status": "ready", "tier": tier}),
        Some(DocState::Failed(m)) => json!({"status": "failed", "error": m}),
    }
}

fn doc_json(d: &DocRecord, s: Option<DocState>) -> Json {
    let mut j = json!({
        "id": d.id, "object": "document", "name": d.name, "account": d.account, "shared": d.shared,
        "tokens": d.tokens.len(), "created": d.created, "chat_template_kwargs": d.kwargs, "has_tools": d.tools.is_some(),
        "priority": if d.priority == Priority::Low { "low" } else { "normal" },
    });
    if let (Some(o), Json::Object(extra)) = (j.as_object_mut(), state_json(s)) {
        o.extend(extra);
    }
    j
}

async fn create_one(st: &Arc<AppState>, acct: &Arc<Account>, spec: DocSpec) -> Result<Json, ApiError> {
    let (n_docs, n_tok) = st.docs.usage(&acct.name);
    let max_docs = acct.max_documents.unwrap_or(st.cfg.max_documents_per_account);
    if n_docs >= max_docs {
        return Err(ApiError::limit("document_limit", 60, format!("this account already has {n_docs} documents (limit {max_docs}); delete some first")));
    }
    let max_ctx = st.max_context_for(acct);
    let (st2, msgs, tools, opts, scope) = (st.clone(), spec.messages.clone(), spec.tools.clone(), spec.opts.clone(), acct.name.clone());
    let (ids, system_end) = tokio::task::spawn_blocking(move || render_prefix(&st2, &scope, max_ctx, &msgs, tools.as_ref(), &opts))
        .await
        .map_err(|e| ApiError::internal("prepare_failed", e.to_string()))??;
    if ids.is_empty() {
        return Err(ApiError::bad("invalid_value", Some("messages"), "the document renders to no tokens"));
    }
    if ids.len() > max_ctx {
        return Err(ApiError::bad("context_length_exceeded", Some("messages"), format!("the document is {} tokens; this key allows at most {max_ctx}", ids.len())));
    }
    if n_tok + ids.len() > st.cfg.max_document_tokens_per_account {
        return Err(ApiError::limit("document_limit", 60, format!("documents of this account would exceed {} tokens", st.cfg.max_document_tokens_per_account)));
    }
    let id = rand_id("doc-", 24);
    let job = DocJob {
        id: id.clone(),
        scope: CacheScope::for_account(&acct.name, acct.internal),
        tokens: ids.clone(),
        priority: spec.priority,
        hints: PrefixHints { system_end, document: None, turn_start: None, ..Default::default() },
    };
    let state = st.backend.doc_create(job).map_err(|e| match e {
        SubmitError::Unsupported => ApiError::not_implemented("documents_unsupported", "this engine cannot precompute documents yet"),
        other => super::submit_error(other),
    })?;
    let rec = st.docs.insert_within(
        DocRecord {
            id: id.clone(),
            name: spec.name,
        account: acct.name.clone(),
        shared: spec.shared,
        messages: spec.messages,
        tools: spec.tools,
        kwargs: kwargs_json(&spec.opts),
            tokens: ids,
            created: unix_secs(),
            priority: spec.priority,
        },
        max_docs,
        st.cfg.max_document_tokens_per_account,
    );
    let rec = match rec {
        Ok(r) => r,
        Err((n, t)) => {
            // 同じアカウントが同時に作って枠を超えた: 先計算は取り消す
            st.backend.doc_delete(&id);
            return Err(ApiError::limit("document_limit", 60, format!("this account has {n} documents / {t} tokens; the limit is {max_docs} / {}", st.cfg.max_document_tokens_per_account)));
        }
    };
    st.log_json(json!({"ev": "document", "op": "create", "id": rec.id, "account": acct.name, "tokens": rec.tokens.len(), "shared": rec.shared}));
    Ok(doc_json(&rec, Some(state)))
}

fn begin(st: &AppState, headers: &HeaderMap) -> Result<Arc<Account>, Response> {
    st.authenticate(headers).map_err(|e| st.reject(None, EP_DOCS, e))
}

pub async fn create(State(st): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    let acct = match begin(&st, &headers) {
        Ok(a) => a,
        Err(r) => return r,
    };
    let r: Result<Json, ApiError> = async {
        st.check_ready()?;
        acct.take_rate().map_err(|w| ApiError::limit("rate_limit_exceeded", w, "rate limit for this key reached"))?;
        let o = request::parse_object(&body)?;
        let spec = parse_spec(&o, &acct, Priority::Normal, &st.cfg)?;
        create_one(&st, &acct, spec).await
    }
    .await;
    match r {
        Ok(j) => {
            st.metrics.counted(&acct.name, EP_DOCS, 200);
            AxJson(j).into_response()
        }
        Err(e) => st.reject(Some(&acct), EP_DOCS, e),
    }
}

/// まとめて（既定の優先度は low＝普通の依頼の合間に）。1 件ずつ結果か誤りを返す
pub async fn create_batch(State(st): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    let acct = match begin(&st, &headers) {
        Ok(a) => a,
        Err(r) => return r,
    };
    let r: Result<Json, ApiError> = async {
        st.check_ready()?;
        acct.take_rate().map_err(|w| ApiError::limit("rate_limit_exceeded", w, "rate limit for this key reached"))?;
        let o = request::parse_object(&body)?;
        let prio = match o.get("priority").and_then(Json::as_str) {
            None | Some("low") => Priority::Low,
            Some("normal") => Priority::Normal,
            Some(_) => return Err(ApiError::bad("invalid_value", Some("priority"), "priority must be \"normal\" or \"low\"")),
        };
        let items = o.get("documents").and_then(Json::as_array).filter(|a| !a.is_empty()).ok_or_else(|| ApiError::bad("invalid_value", Some("documents"), "documents must be a non-empty array"))?;
        if items.len() > 256 {
            return Err(ApiError::bad("invalid_value", Some("documents"), "at most 256 documents per batch"));
        }
        let mut out = vec![];
        for (i, it) in items.iter().enumerate() {
            let res = match it.as_object() {
                Some(io) => match parse_spec(io, &acct, prio, &st.cfg) {
                    Ok(spec) => create_one(&st, &acct, spec).await,
                    Err(e) => Err(e),
                },
                None => Err(ApiError::bad("invalid_type", Some(format!("documents[{i}]").as_str()), "document must be an object")),
            };
            out.push(match res {
                Ok(j) => j,
                Err(e) => {
                    st.metrics.rejected(&acct.name, EP_DOCS, e.status.as_u16(), e.code);
                    e.body()
                }
            });
        }
        Ok(json!({"object": "list", "data": out}))
    }
    .await;
    match r {
        Ok(j) => {
            st.metrics.counted(&acct.name, EP_DOCS, 200);
            AxJson(j).into_response()
        }
        Err(e) => st.reject(Some(&acct), EP_DOCS, e),
    }
}

pub async fn list(State(st): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let acct = match begin(&st, &headers) {
        Ok(a) => a,
        Err(r) => return r,
    };
    let data: Vec<Json> = st.docs.list(&acct).iter().map(|d| doc_json(d, st.backend.doc_state(&d.id))).collect();
    AxJson(json!({"object": "list", "data": data})).into_response()
}

pub async fn get_one(State(st): State<Arc<AppState>>, headers: HeaderMap, Path(id): Path<String>) -> Response {
    let acct = match begin(&st, &headers) {
        Ok(a) => a,
        Err(r) => return r,
    };
    match st.docs.get(&id, &acct) {
        Some(d) => AxJson(doc_json(&d, st.backend.doc_state(&d.id))).into_response(),
        None => st.reject(Some(&acct), EP_DOCS, ApiError::not_found("document_not_found", format!("document {id:?} not found"))),
    }
}

/// 消せるのは持ち主のアカウントと admin の鍵（社内の共有の文書）
pub async fn delete(State(st): State<Arc<AppState>>, headers: HeaderMap, Path(id): Path<String>) -> Response {
    let acct = match begin(&st, &headers) {
        Ok(a) => a,
        Err(r) => return r,
    };
    let Some(d) = st.docs.get(&id, &acct) else {
        return st.reject(Some(&acct), EP_DOCS, ApiError::not_found("document_not_found", format!("document {id:?} not found")));
    };
    if d.account != acct.name && !acct.admin {
        return st.reject(Some(&acct), EP_DOCS, ApiError::forbidden("not_owner", "only the owner (or an admin key) can delete this document"));
    }
    st.docs.remove(&id);
    st.backend.doc_delete(&id);
    st.log_json(json!({"ev": "document", "op": "delete", "id": id, "account": acct.name}));
    st.metrics.counted(&acct.name, EP_DOCS, 200);
    AxJson(json!({"id": id, "object": "document", "deleted": true})).into_response()
}
