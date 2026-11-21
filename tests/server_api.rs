//! OpenAI 互換の口の試し（HTTP で。GPU は使わない）。
//!   cargo test --release --test server_api                                         # 合成の tokenizer（1 バイト 1 語）
//!   TACHYON_HF_DIR=weights/q38-27b/hf cargo test --release --test server_api   # 本物の tokenizer
//! テンプレートは tests/data/qwen38/chat_template.jinja（HF の Qwen/Qwen3.8-27B と同じ。sha256 af16f654…）
use tachyon_v100::server::engine::EngineLoader;
use tachyon_v100::server::fake::{self, FakeBackend, FakeConfig, FakeEngine, DEFAULT_ANSWER, DEFAULT_THOUGHT};
use tachyon_v100::server::util::LogSink;
use tachyon_v100::server::{self, AppState, Auth, Backend, EngineInfo, KeysFile, SchedConfig, ServerConfig, SimpleScheduler, Status};
use tachyon_v100::template::ChatTemplate;
use tachyon_v100::tokenizer::Tokenizer;
use reqwest::header::HeaderMap;
use serde_json::{json, Value as Json};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

const K_INT: &str = "sk-test-internal-000000000001";
const K_INT2: &str = "sk-test-internal-000000000002";
const K_CUST: &str = "sk-test-customer-00000000001";
const K_CUST2: &str = "sk-test-customer-00000000002";
const K_LIM: &str = "sk-test-limited-000000000001";
const K_CONC: &str = "sk-test-concurrent-000000001";
const K_ISO_A: &str = "sk-test-isolation-00000000a";
const K_ISO_B: &str = "sk-test-isolation-00000000b";
const ALL_KEYS: [&str; 8] = [K_INT, K_INT2, K_CUST, K_CUST2, K_LIM, K_CONC, K_ISO_A, K_ISO_B];

static TOK: OnceLock<Arc<Tokenizer>> = OnceLock::new();

fn tok() -> Arc<Tokenizer> {
    TOK.get_or_init(|| {
        let t = match std::env::var("TACHYON_HF_DIR") {
            Ok(d) => Tokenizer::from_dir(d).unwrap(),
            Err(_) => {
                let dir = std::env::temp_dir().join(format!("tachyon-test-tok-{}", std::process::id()));
                fake::write_synthetic_tokenizer(&dir).unwrap();
                Tokenizer::from_dir(&dir).unwrap()
            }
        };
        Arc::new(t)
    })
    .clone()
}

fn tpl() -> Arc<ChatTemplate> {
    Arc::new(ChatTemplate::from_file(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/qwen38/chat_template.jinja")).unwrap())
}

fn keys() -> KeysFile {
    serde_json::from_value(json!({"accounts": {
        "internal": {"keys": [K_INT], "internal": true, "admin": true},
        "internal2": {"keys": [K_INT2], "internal": true},
        "cust": {"keys": [K_CUST], "max_context": 2000},
        "cust2": {"keys": [K_CUST2]},
        "limited": {"keys": [K_LIM], "rpm": 2},
        "conc": {"keys": [K_CONC], "max_concurrent": 1},
        "iso-a": {"keys": [K_ISO_A]},
        "iso-b": {"keys": [K_ISO_B]},
    }}))
    .unwrap()
}

/// 試しの口の設定: 考えありを既定に（docs/API.md 3 の元の決め。serve の既定＝考えなしは serve_default_* の試しで見る）
fn cfg() -> ServerConfig {
    ServerConfig { default_thinking: true, ..ServerConfig::default() }
}

struct Srv {
    base: String,
    c: reqwest::Client,
    st: Arc<AppState>,
    fake: Option<Arc<FakeBackend>>,
    logs: Arc<Mutex<Vec<String>>>,
}

async fn start_with(cfg: ServerConfig, backend: Arc<dyn Backend>, fake: Option<Arc<FakeBackend>>) -> Srv {
    let logs = Arc::new(Mutex::new(vec![]));
    let l2 = logs.clone();
    let sink: LogSink = Arc::new(move |s: &str| l2.lock().unwrap().push(s.to_string()));
    let st = AppState::new(cfg, backend, tok(), tpl(), Auth::from_keys(keys()).unwrap(), sink).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let st2 = st.clone();
    tokio::spawn(async move { server::serve(listener, st2, std::future::pending()).await.unwrap() });
    Srv { base: format!("http://{addr}"), c: reqwest::Client::builder().no_proxy().build().unwrap(), st, fake, logs }
}

async fn start(fc: FakeConfig) -> Srv {
    let fake = FakeBackend::new(tok(), fc);
    start_with(cfg(), fake.clone(), Some(fake)).await
}

impl Srv {
    fn fake(&self) -> &FakeBackend {
        self.fake.as_ref().unwrap()
    }

    async fn post(&self, path: &str, key: Option<&str>, body: &Json) -> (u16, HeaderMap, Json) {
        self.post_raw(path, key, body.to_string()).await
    }

    async fn post_raw(&self, path: &str, key: Option<&str>, body: String) -> (u16, HeaderMap, Json) {
        let mut r = self.c.post(format!("{}{path}", self.base)).header("content-type", "application/json").body(body);
        if let Some(k) = key {
            r = r.bearer_auth(k);
        }
        let resp = r.send().await.unwrap();
        let (s, h) = (resp.status().as_u16(), resp.headers().clone());
        let text = resp.text().await.unwrap();
        (s, h, serde_json::from_str(&text).unwrap_or(Json::String(text)))
    }

    async fn get(&self, path: &str, key: Option<&str>) -> (u16, Json) {
        let mut r = self.c.get(format!("{}{path}", self.base));
        if let Some(k) = key {
            r = r.bearer_auth(k);
        }
        let resp = r.send().await.unwrap();
        let s = resp.status().as_u16();
        let text = resp.text().await.unwrap();
        (s, serde_json::from_str(&text).unwrap_or(Json::String(text)))
    }

    async fn delete(&self, path: &str, key: &str) -> (u16, Json) {
        let resp = self.c.delete(format!("{}{path}", self.base)).bearer_auth(key).send().await.unwrap();
        let s = resp.status().as_u16();
        (s, resp.json_text().await)
    }

    async fn chat(&self, key: &str, body: &Json) -> (u16, Json) {
        let (s, _, j) = self.post("/v1/chat/completions", Some(key), body).await;
        (s, j)
    }

    /// stream: SSE の data を全部（[DONE] を見たか）
    async fn sse(&self, key: &str, body: &Json) -> (u16, Vec<Json>, bool) {
        let mut b = body.clone();
        b["stream"] = true.into();
        let mut resp = self.c.post(format!("{}/v1/chat/completions", self.base)).bearer_auth(key).header("content-type", "application/json").body(b.to_string()).send().await.unwrap();
        let s = resp.status().as_u16();
        if s != 200 {
            return (s, vec![resp.json_text().await], false);
        }
        assert!(resp.headers()["content-type"].to_str().unwrap().starts_with("text/event-stream"));
        let (mut buf, mut out, mut done) = (Vec::<u8>::new(), vec![], false);
        while let Some(chunk) = resp.chunk().await.unwrap() {
            buf.extend_from_slice(&chunk);
            while let Some(i) = buf.windows(2).position(|w| w == b"\n\n") {
                let ev = String::from_utf8(buf[..i].to_vec()).unwrap();
                buf.drain(..i + 2);
                for line in ev.lines() {
                    if let Some(d) = line.strip_prefix("data: ") {
                        assert!(!done, "[DONE] の後に data");
                        if d == "[DONE]" {
                            done = true;
                        } else {
                            out.push(serde_json::from_str(d).unwrap());
                        }
                    }
                }
            }
        }
        (s, out, done)
    }

    async fn metrics(&self) -> String {
        self.c.get(format!("{}/metrics", self.base)).send().await.unwrap().text().await.unwrap()
    }
}

trait JsonText {
    async fn json_text(self) -> Json;
}
impl JsonText for reqwest::Response {
    async fn json_text(self) -> Json {
        let t = self.text().await.unwrap();
        serde_json::from_str(&t).unwrap_or(Json::String(t))
    }
}

fn user(s: &str) -> Json {
    json!({"messages": [{"role": "user", "content": s}]})
}

fn with(mut b: Json, k: &str, v: Json) -> Json {
    b[k] = v;
    b
}

/// stream の塊 → (考え, 答え, 道具の呼び出し, finish_reason, usage)
fn merge(chunks: &[Json]) -> (String, String, Vec<Json>, Option<String>, Option<Json>) {
    let (mut r, mut c, mut t, mut f, mut u) = (String::new(), String::new(), vec![], None, None);
    for ch in chunks {
        assert_eq!(ch["object"], "chat.completion.chunk", "{ch}");
        if let Some(us) = ch.get("usage") {
            assert_eq!(ch["choices"], json!([]));
            u = Some(us.clone());
        }
        for choice in ch["choices"].as_array().unwrap() {
            let d = &choice["delta"];
            r.push_str(d["reasoning_content"].as_str().unwrap_or(""));
            c.push_str(d["content"].as_str().unwrap_or(""));
            t.extend(d["tool_calls"].as_array().cloned().unwrap_or_default());
            if let Some(fr) = choice["finish_reason"].as_str() {
                assert!(f.is_none(), "finish_reason が 2 回");
                f = Some(fr.to_string());
            }
        }
    }
    (r, c, t, f, u)
}

async fn wait_for(what: &str, ms: u64, f: impl Fn() -> bool) {
    let t0 = Instant::now();
    while !f() {
        assert!(t0.elapsed() < Duration::from_millis(ms), "待ちきれない: {what}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

fn prompt_text(srv: &Srv) -> String {
    tok().decode(&srv.fake().last().unwrap().prompt, false)
}

fn tid(s: &str) -> u32 {
    tok().token_to_id(s).unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn health_models_auth() {
    let cfg = ServerConfig { weights_sha: "1234abcd".to_string() + &"0".repeat(56), ..cfg() };
    let fake = FakeBackend::new(tok(), FakeConfig::default());
    let srv = start_with(cfg, fake.clone(), Some(fake)).await;
    let (s, h) = srv.get("/health", None).await;
    assert_eq!((s, h["status"].as_str()), (200, Some("ok")));
    assert_eq!(h["model"], "qwen3.8-27b-1234abcd");
    // /v1/models は鍵が要る
    let (s, e) = srv.get("/v1/models", None).await;
    assert_eq!((s, e["error"]["code"].as_str(), e["error"]["type"].as_str()), (401, Some("missing_api_key"), Some("authentication_error")));
    let (s, m) = srv.get("/v1/models", Some(K_CUST)).await;
    assert_eq!(s, 200);
    assert_eq!(m["data"][0]["id"], "qwen3.8-27b-1234abcd");
    assert_eq!(m["data"][0]["weights_sha256"].as_str().unwrap().len(), 64);
    assert_eq!(srv.get("/v1/models/qwen3.8-27b-1234abcd", Some(K_CUST)).await.0, 200);
    assert_eq!(srv.get("/v1/models/other", Some(K_CUST)).await.0, 404);
    // 違う鍵・鍵なし
    let (s, _, e) = srv.post("/v1/chat/completions", Some("sk-wrong-key-0000000000000"), &user("hi")).await;
    assert_eq!((s, e["error"]["code"].as_str()), (401, Some("invalid_api_key")));
    let (s, _, _) = srv.post("/v1/chat/completions", None, &user("hi")).await;
    assert_eq!(s, 401);
    // 読み込み中は 503（Retry-After つき）、失敗は /health 500
    srv.fake().set_status(Some(Status::Loading { progress: 0.5, what: "weights".into() }));
    let (s, h) = srv.get("/health", None).await;
    assert_eq!((s, h["status"].as_str()), (503, Some("loading")));
    let (s, hd, e) = srv.post("/v1/chat/completions", Some(K_INT), &user("hi")).await;
    assert_eq!((s, e["error"]["code"].as_str()), (503, Some("loading")));
    assert!(hd.contains_key("retry-after"));
    srv.fake().set_status(Some(Status::Failed("boom".into())));
    assert_eq!(srv.get("/health", None).await.0, 500);
    srv.fake().set_status(None);
    // 見張り: 実際に 1 語答えさせる
    let (s, h) = srv.get("/health?probe=1", None).await;
    assert_eq!((s, h["status"].as_str()), (200, Some("ok")), "{h}");
    assert!(h["probe_tokens"].as_u64().unwrap() >= 1);
    // 知らない口
    assert_eq!(srv.get("/v1/nothing", Some(K_INT)).await.0, 404);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn non_stream_thinking_split_and_defaults() {
    let srv = start(FakeConfig::default()).await;
    let (s, j) = srv.chat(K_INT, &user("Hello?")).await;
    assert_eq!(s, 200, "{j}");
    assert_eq!(j["object"], "chat.completion");
    assert!(j["id"].as_str().unwrap().starts_with("chatcmpl-"));
    let m = &j["choices"][0]["message"];
    assert_eq!((m["role"].as_str(), m["reasoning_content"].as_str(), m["content"].as_str()), (Some("assistant"), Some(DEFAULT_THOUGHT), Some(DEFAULT_ANSWER)));
    assert_eq!(j["choices"][0]["finish_reason"], "stop");
    // usage はエンジンに渡した語・出た語と合う
    let rec = srv.fake().last().unwrap();
    let plan = fake::plan(&tok(), &rec.prompt);
    let think_end = plan.tokens.iter().position(|&t| t == tok().ids.think_end).unwrap();
    let u = &j["usage"];
    assert_eq!(u["prompt_tokens"].as_u64().unwrap() as usize, rec.prompt.len());
    assert_eq!(u["completion_tokens"].as_u64().unwrap() as usize, plan.tokens.len());
    assert_eq!(u["completion_tokens_details"]["reasoning_tokens"].as_u64().unwrap() as usize, think_end + 1);
    assert_eq!(u["total_tokens"].as_u64().unwrap() as usize, rec.prompt.len() + plan.tokens.len());
    assert!(j["timings"]["predicted_n"].as_u64().is_some());
    // 既定: medium（文を足さない）・考えあり・推奨の試料の取り方・止める語（eos ＋ FIM の 3 語）
    let p = prompt_text(&srv);
    assert!(p.ends_with("<|im_start|>assistant\n<think>\n") && !p.contains("Reasoning effort"), "{p}");
    assert_eq!((rec.sampling.temperature, rec.sampling.top_p, rec.sampling.top_k, rec.sampling.presence_penalty), (1.0, 0.95, 20, 0.0));
    for s in ["<|im_end|>", "<|endoftext|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>"] {
        assert!(rec.stop_ids.contains(&tid(s)), "{s}");
    }
    assert_eq!(rec.stop_ids.len(), 5);
    assert_eq!(rec.max_tokens, 32_768);
    assert_eq!(rec.scope.namespace, "internal");
    // 考えなし: 空の考えの枠・考えなしの推奨（presence 1.5）
    let (s, j) = srv.chat(K_INT, &with(user("Hello?"), "chat_template_kwargs", json!({"enable_thinking": false}))).await;
    assert_eq!(s, 200);
    let m = &j["choices"][0]["message"];
    assert!(m.get("reasoning_content").is_none());
    assert_eq!(m["content"], DEFAULT_ANSWER);
    assert_eq!(j["usage"]["completion_tokens_details"]["reasoning_tokens"], 0);
    assert!(prompt_text(&srv).ends_with("<think>\n\n</think>\n\n"));
    let rec = srv.fake().last().unwrap();
    assert_eq!((rec.sampling.temperature, rec.sampling.top_p, rec.sampling.presence_penalty), (0.7, 0.8, 1.5));
    // 書いた値はそのまま・温度 0 なら罰の既定は入れない・種
    let b = with(with(with(user("Hello?"), "chat_template_kwargs", json!({"enable_thinking": false})), "temperature", json!(0)), "seed", json!(42));
    assert_eq!(srv.chat(K_INT, &b).await.0, 200);
    let rec = srv.fake().last().unwrap();
    assert_eq!((rec.sampling.temperature, rec.sampling.presence_penalty, rec.sampling.seed), (0.0, 0.0, 42));
    // xhigh・low は system に 1 文、上の段の reasoning_effort も受ける
    assert_eq!(srv.chat(K_INT, &with(user("Hi"), "chat_template_kwargs", json!({"reasoning_effort": "xhigh"}))).await.0, 200);
    assert!(prompt_text(&srv).starts_with("<|im_start|>system\nReasoning effort is set to xhigh."));
    assert_eq!(srv.chat(K_INT, &with(user("Hi"), "reasoning_effort", json!("low"))).await.0, 200);
    assert!(prompt_text(&srv).contains("Reasoning effort is set to low."));
    // 違う effort は 400（考えなしでも）
    for kw in [json!({"reasoning_effort": "high"}), json!({"reasoning_effort": "max", "enable_thinking": false})] {
        let (s, e) = srv.chat(K_INT, &with(user("Hi"), "chat_template_kwargs", kw)).await;
        assert_eq!((s, e["error"]["param"].as_str(), e["error"]["code"].as_str()), (400, Some("chat_template_kwargs.reasoning_effort"), Some("invalid_value")), "{e}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn response_model_is_request_model() {
    // 返事の model は依頼の model（前段・chat が送った名前）をそのまま。無ければ口の名前
    let srv = start(FakeConfig::default()).await;
    let (s, j) = srv.chat(K_INT, &with(user("Hi"), "model", json!("qwen3.8:27b"))).await;
    assert_eq!((s, j["model"].as_str()), (200, Some("qwen3.8:27b")));
    let (s, chunks, done) = srv.sse(K_INT, &with(with(user("Hi"), "model", json!("qwen3.8:27b")), "stream_options", json!({"include_usage": true}))).await;
    assert!(s == 200 && done && !chunks.is_empty());
    assert!(chunks.iter().all(|c| c["model"] == "qwen3.8:27b"), "{chunks:?}");
    let mut b = user("Hi");
    b.as_object_mut().unwrap().remove("model");
    let (s, j) = srv.chat(K_INT, &b).await;
    assert_eq!((s, j["model"].as_str()), (200, Some("qwen3.8-27b-00000000")));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stream_matches_non_stream() {
    let srv = start(FakeConfig::default()).await;
    for body in [user("Hello?"), user("[[echo]]日本語のテスト🎌 と English"), with(user("[[echo]]絵文字👍"), "chat_template_kwargs", json!({"enable_thinking": false}))] {
        let (s, j) = srv.chat(K_INT, &body).await;
        assert_eq!(s, 200);
        let (s, chunks, done) = srv.sse(K_INT, &with(body.clone(), "stream_options", json!({"include_usage": true}))).await;
        assert!(s == 200 && done);
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        let id = chunks[0]["id"].as_str().unwrap();
        assert!(chunks.iter().all(|c| c["id"] == id && c["model"] == j["model"]));
        let (r, c, t, f, u) = merge(&chunks);
        let m = &j["choices"][0]["message"];
        assert_eq!(r, m["reasoning_content"].as_str().unwrap_or(""));
        assert_eq!(c, m["content"].as_str().unwrap());
        assert!(t.is_empty());
        assert_eq!(f.as_deref(), Some("stop"));
        let u = u.expect("include_usage なのに usage が無い");
        assert_eq!(u["completion_tokens"], j["usage"]["completion_tokens"]);
        assert_eq!(u["prompt_tokens"], j["usage"]["prompt_tokens"]);
        assert_eq!(u["completion_tokens_details"], j["usage"]["completion_tokens_details"]);
        // 終わりの塊（finish_reason つき）に timings
        assert!(chunks.iter().any(|c| c["choices"][0]["finish_reason"] == "stop" && c["timings"].is_object()));
    }
    let (_, j) = srv.chat(K_INT, &user("[[echo]]日本語のテスト🎌 と English")).await;
    assert_eq!(j["choices"][0]["message"]["content"], "日本語のテスト🎌 と English");
    // include_usage なしなら usage の塊は無い
    let (_, chunks, _) = srv.sse(K_INT, &user("Hello?")).await;
    assert!(merge(&chunks).4.is_none());
}

fn weather_tools() -> Json {
    json!([{"type": "function", "function": {"name": "get_weather", "description": "Get the weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}},
        {"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}])
}

const CALL: &str = "[[raw]]I need the weather.\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tool_calls_and_preserve_thinking() {
    let srv = start(FakeConfig::default()).await;
    let body = with(user(CALL), "tools", weather_tools());
    let (s, j) = srv.chat(K_INT, &body).await;
    assert_eq!(s, 200, "{j}");
    let ch = &j["choices"][0];
    assert_eq!(ch["finish_reason"], "tool_calls");
    let m = &ch["message"];
    assert_eq!((m["content"].clone(), m["reasoning_content"].as_str()), (Json::Null, Some("I need the weather.")));
    let tc = &m["tool_calls"][0];
    assert!(tc["id"].as_str().unwrap().starts_with("call_"));
    assert_eq!((tc["type"].as_str(), tc["function"]["name"].as_str()), (Some("function"), Some("get_weather")));
    let args: Json = serde_json::from_str(tc["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args, json!({"city": "Tokyo", "days": 3}));
    let p = prompt_text(&srv);
    assert!(p.contains("# Tools") && p.contains("\"name\": \"get_weather\"") && p.contains("\"name\": \"search\""));
    // stream
    let (s, chunks, done) = srv.sse(K_INT, &body).await;
    assert!(s == 200 && done);
    let (r, c, t, f, _) = merge(&chunks);
    assert_eq!((r.as_str(), c.as_str(), f.as_deref()), ("I need the weather.", "", Some("tool_calls")));
    assert_eq!(t.len(), 1);
    assert_eq!((t[0]["index"].as_u64(), t[0]["function"]["name"].as_str()), (Some(0), Some("get_weather")));
    assert_eq!(serde_json::from_str::<Json>(t[0]["function"]["arguments"].as_str().unwrap()).unwrap(), args);
    // 道具の手番を続ける（arguments は JSON の文字列・tool の結果）
    let conv = json!({"tools": weather_tools(), "messages": [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "", "reasoning_content": "OLD_THOUGHT", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        {"role": "user", "content": "and tomorrow?"},
    ]});
    assert_eq!(srv.chat(K_INT, &conv).await.0, 200);
    let p = prompt_text(&srv);
    // tools があれば preserve_thinking の既定は true → 前の回の考えが残る
    assert!(p.contains("OLD_THOUGHT") && p.contains("<tool_response>\nsunny\n</tool_response>") && p.contains("<parameter=city>\nTokyo\n</parameter>"), "{p}");
    let mut no_tools = conv.clone();
    no_tools.as_object_mut().unwrap().remove("tools");
    assert_eq!(srv.chat(K_INT, &no_tools).await.0, 200);
    assert!(!prompt_text(&srv).contains("OLD_THOUGHT"), "tools なしは preserve_thinking=false が既定");
    assert_eq!(srv.chat(K_INT, &with(no_tools.clone(), "chat_template_kwargs", json!({"preserve_thinking": true}))).await.0, 200);
    assert!(prompt_text(&srv).contains("OLD_THOUGHT"));
    assert_eq!(srv.chat(K_INT, &with(conv.clone(), "chat_template_kwargs", json!({"preserve_thinking": false}))).await.0, 200);
    assert!(!prompt_text(&srv).contains("OLD_THOUGHT"));
    // tool_choice "none": 道具を描かない。呼んでしまったら答えの文字
    let (s, j) = srv.chat(K_INT, &with(body.clone(), "tool_choice", json!("none"))).await;
    assert_eq!(s, 200);
    assert!(!prompt_text(&srv).contains("# Tools"));
    assert_eq!(j["choices"][0]["finish_reason"], "stop");
    assert!(j["choices"][0]["message"]["content"].as_str().unwrap().starts_with("<tool_call>\n<function=get_weather>"));
    assert!(j["choices"][0]["message"].get("tool_calls").is_none());
    // 名前を指した tool_choice: その道具だけ描く
    assert_eq!(srv.chat(K_INT, &with(body.clone(), "tool_choice", json!({"type": "function", "function": {"name": "search"}}))).await.0, 200);
    let p = prompt_text(&srv);
    assert!(p.contains("\"name\": \"search\"") && !p.contains("\"name\": \"get_weather\""));
    // max_tokens で呼び出しの途中で切れたら、壊れた引数で道具を呼ばせない（答えの文字・finish_reason length）
    let cut = tok().encode("I need the weather.\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTok").unwrap().len();
    for stream in [false, true] {
        let b = with(body.clone(), "max_tokens", json!(cut));
        let (f, c, calls) = if stream {
            let (_, ch, _) = srv.sse(K_INT, &b).await;
            let (_, c, t, f, _) = merge(&ch);
            (f.unwrap(), c, t.len())
        } else {
            let (s, j) = srv.chat(K_INT, &b).await;
            assert_eq!(s, 200);
            let m = &j["choices"][0]["message"];
            (j["choices"][0]["finish_reason"].as_str().unwrap().to_string(), m["content"].as_str().unwrap_or("").to_string(), m["tool_calls"].as_array().map_or(0, |a| a.len()))
        };
        assert_eq!((f.as_str(), calls), ("length", 0), "stream={stream}");
        assert!(c.starts_with("<tool_call>\n<function=get_weather>"), "{c}");
    }
    // 誤り
    let (s, e) = srv.chat(K_INT, &with(user("x"), "tools", json!([{"type": "function", "function": {"name": "bad name"}}]))).await;
    assert_eq!((s, e["error"]["param"].as_str()), (400, Some("tools[0].function.name")));
    let (s, _) = srv.chat(K_INT, &with(body.clone(), "tool_choice", json!({"type": "function", "function": {"name": "nope"}}))).await;
    assert_eq!(s, 400);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stops_lengths_and_cancel() {
    let srv = start(FakeConfig::default()).await;
    // max_tokens → length
    for stream in [false, true] {
        let b = with(with(user("[[len:50]]"), "max_tokens", json!(10)), "chat_template_kwargs", json!({"enable_thinking": false}));
        let (content, fr, usage) = if stream {
            let (_, ch, _) = srv.sse(K_INT, &with(b, "stream_options", json!({"include_usage": true}))).await;
            let (_, c, _, f, u) = merge(&ch);
            (c, f.unwrap(), u.unwrap())
        } else {
            let (s, j) = srv.chat(K_INT, &b).await;
            assert_eq!(s, 200);
            (j["choices"][0]["message"]["content"].as_str().unwrap().to_string(), j["choices"][0]["finish_reason"].as_str().unwrap().to_string(), j["usage"].clone())
        };
        assert_eq!((content.as_str(), fr.as_str()), ("x".repeat(10).as_str(), "length"));
        assert_eq!(usage["completion_tokens"], 10);
        assert_eq!(srv.fake().last().unwrap().max_tokens, 10);
    }
    // 止めの文字列: 手前まで・stop・残りは取り消す
    for stream in [false, true] {
        let b = with(user("[[delay:2]][[raw]]alpha beta STOP gamma delta epsilon zeta eta theta"), "stop", json!(["STOP", "never"]));
        let b = with(b, "chat_template_kwargs", json!({"enable_thinking": false}));
        let (c, f) = if stream {
            let (_, ch, done) = srv.sse(K_INT, &b).await;
            assert!(done);
            let (_, c, _, f, _) = merge(&ch);
            (c, f.unwrap())
        } else {
            let (_, j) = srv.chat(K_INT, &b).await;
            (j["choices"][0]["message"]["content"].as_str().unwrap().to_string(), j["choices"][0]["finish_reason"].as_str().unwrap().to_string())
        };
        assert_eq!((c.as_str(), f.as_str()), ("alpha beta", "stop"));
        let rec = srv.fake().last().unwrap();
        wait_for("止めの文字列の後の取り消し", 3000, || rec.cancelled.load(Ordering::SeqCst)).await;
    }
    // 止める語: FIM の語（llama.cpp と同じ）と <|endoftext|>。その語は出さない
    for (raw, want) in [("before<|fim_pad|>after", "before"), ("one<|endoftext|>two", "one"), ("a<|file_sep|>b", "a"), ("c<|repo_name|>d", "c")] {
        let b = with(user(&format!("[[raw]]{raw}")), "chat_template_kwargs", json!({"enable_thinking": false}));
        let (s, j) = srv.chat(K_INT, &b).await;
        assert_eq!(s, 200);
        assert_eq!((j["choices"][0]["message"]["content"].as_str(), j["choices"][0]["finish_reason"].as_str()), (Some(want), Some("stop")));
        let n_want = tok().encode(want).unwrap().len() + 1;
        assert_eq!(j["usage"]["completion_tokens"].as_u64().unwrap() as usize, n_want);
    }
    // 客が stream を途中で切る → Backend も取り消し・499 で数える
    let mut resp = srv
        .c
        .post(format!("{}/v1/chat/completions", srv.base))
        .bearer_auth(K_INT2)
        .body(with(user("[[len:100000]][[delay:2]]"), "stream", json!(true)).to_string())
        .send()
        .await
        .unwrap();
    let _ = resp.chunk().await.unwrap();
    let _ = resp.chunk().await.unwrap();
    drop(resp);
    let rec = srv.fake().last().unwrap();
    wait_for("切った stream の取り消し", 3000, || rec.cancelled.load(Ordering::SeqCst)).await;
    let st = srv.st.clone();
    wait_for("inflight が 0 に", 3000, || st.inflight.load(Ordering::SeqCst) == 0).await;
    assert!(rec.sent.load(Ordering::SeqCst) < 100_000);
    let m = srv.metrics().await;
    assert!(m.contains("tachyon_requests_total{account=\"internal2\",endpoint=\"chat\",code=\"499\"} 1"));
    let done: u64 = m.lines().find_map(|l| l.strip_prefix("tachyon_completion_tokens_total{account=\"internal2\"} ")).unwrap().parse().unwrap();
    assert!(done >= 2, "切った依頼の語が数えられていない: {done}");
    // stream なしでも、客が待ちきれずに切ったら取り消す（書く物が無い間でも気づく）
    let short = reqwest::Client::builder().no_proxy().timeout(Duration::from_millis(300)).build().unwrap();
    let r = short.post(format!("{}/v1/chat/completions", srv.base)).bearer_auth(K_INT2).body(user("[[len:100000]][[delay:2]]").to_string()).send().await;
    assert!(r.is_err(), "時間切れになるはず");
    let rec = srv.fake().last().unwrap();
    wait_for("stream なしの切断の取り消し", 3000, || rec.cancelled.load(Ordering::SeqCst)).await;
    wait_for("inflight が 0 に", 3000, || st.inflight.load(Ordering::SeqCst) == 0).await;
    // 失敗する Backend
    let (s, e) = srv.chat(K_INT, &user("[[error:3]]")).await;
    assert_eq!((s, e["error"]["type"].as_str()), (500, Some("server_error")));
    let (s, ch, done) = srv.sse(K_INT, &user("[[error:3]]")).await;
    assert!(s == 200 && done && ch.last().unwrap()["error"]["code"] == "backend_error");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn limits_and_bad_requests() {
    let srv = start(FakeConfig::default()).await;
    // 鍵ごとの 1 分の回数（rpm 2）→ 3 回目は 429 と Retry-After
    for _ in 0..2 {
        assert_eq!(srv.chat(K_LIM, &user("hi")).await.0, 200);
    }
    let (s, h, e) = srv.post("/v1/chat/completions", Some(K_LIM), &user("hi")).await;
    assert_eq!((s, e["error"]["code"].as_str(), e["error"]["type"].as_str()), (429, Some("rate_limit_exceeded"), Some("rate_limit_error")));
    let ra: u64 = h["retry-after"].to_str().unwrap().parse().unwrap();
    assert!((1..=60).contains(&ra));
    // 鍵ごとの同時の数（1）
    let slow = with(user("[[len:200]][[delay:5]]"), "stream", json!(true));
    let resp = srv.c.post(format!("{}/v1/chat/completions", srv.base)).bearer_auth(K_CONC).body(slow.to_string()).send().await.unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let (s, e) = srv.chat(K_CONC, &user("hi")).await;
    assert_eq!((s, e["error"]["code"].as_str()), (429, Some("too_many_concurrent_requests")));
    drop(resp);
    let st = srv.st.clone();
    wait_for("conc の依頼が終わる", 3000, || st.inflight.load(Ordering::SeqCst) == 0).await;
    assert_eq!(srv.chat(K_CONC, &user("hi")).await.0, 200);
    // 鍵ごとの文脈の上限（cust は 2000 語）→ 400。切り詰めない
    let long = "日本語の長い文。".repeat(800);
    let (s, e) = srv.chat(K_CUST, &user(&long)).await;
    assert_eq!((s, e["error"]["code"].as_str(), e["error"]["param"].as_str()), (400, Some("context_length_exceeded"), Some("messages")), "{e}");
    // 出力の余地が無い（min_room 1024 語）→ 400、max_tokens を小さく書けば通り、残りで切られる
    let mid = "日本語の長い文。".repeat(1);
    let n_short = {
        assert_eq!(srv.chat(K_CUST, &user(&mid)).await.0, 200);
        srv.fake().last().unwrap().prompt.len()
    };
    assert_eq!(srv.fake().last().unwrap().max_tokens, 2000 - n_short);
    let unit = tok().encode("あいうえおかきくけこ").unwrap().len();
    let pad = "あいうえおかきくけこ".repeat((2000 - n_short - 500) / unit);
    let (s, e) = srv.chat(K_CUST, &user(&format!("{mid}{pad}"))).await;
    assert_eq!((s, e["error"]["code"].as_str()), (400, Some("context_length_exceeded")), "{e}");
    let (s, _) = srv.chat(K_CUST, &with(user(&format!("{mid}{pad}")), "max_tokens", json!(100))).await;
    assert_eq!(s, 200);
    assert_eq!(srv.fake().last().unwrap().max_tokens, 100);
    // 全体の上限 256K（262,144 語）→ 400
    // BPE が繰り返しの継ぎ目でまとめても 262,144 語を超えるように多めに
    // 1 つの message は 1MiB まで（本物の tokenizer だと 1 つに入らないので 3 つに分ける）
    let huge = "日本語の文。".repeat(110_000 / tok().encode("日本語の文。").unwrap().len() + 10);
    let three = json!({"messages": [{"role": "user", "content": huge}, {"role": "assistant", "content": "ok"}, {"role": "user", "content": huge}, {"role": "assistant", "content": "ok"}, {"role": "user", "content": huge}]});
    let (s, e) = srv.chat(K_INT, &three).await;
    assert_eq!((s, e["error"]["code"].as_str()), (400, Some("context_length_exceeded")), "{e}");
    assert!(e["error"]["message"].as_str().unwrap().contains("262144"));
    // 形の誤り
    let bad = |b: &str| srv.post_raw("/v1/chat/completions", Some(K_INT), b.to_string());
    let (s, _, e) = bad("{not json").await;
    assert_eq!((s, e["error"]["code"].as_str()), (400, Some("invalid_json")));
    for (b, param) in [
        (json!({}), "messages"),
        (json!({"messages": "hi"}), "messages"),
        (with(user("x"), "n", json!(2)), "n"),
        (with(user("x"), "temperature", json!(5)), "temperature"),
        (with(user("x"), "top_p", json!(1.5)), "top_p"),
        (with(user("x"), "max_tokens", json!(0)), "max_tokens"),
        (with(user("x"), "logprobs", json!(true)), "logprobs"),
        (with(user("x"), "stop", json!(5)), "stop"),
        (with(user("x"), "chat_template_kwargs", json!({"enable_thinking": "yes"})), "chat_template_kwargs"),
        (json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:,"}}]}]}), "messages[0].content[0]"),
        (json!({"messages": [{"role": "user", "content": "a"}, {"role": "system", "content": "late"}]}), "messages"),
    ] {
        let (s, e) = srv.chat(K_INT, &b).await;
        assert_eq!((s, e["error"]["param"].as_str()), (400, Some(param)), "{b} → {e}");
        assert_eq!(e["error"]["type"], "invalid_request_error");
    }
    // 壊れた Authorization は 401 invalid_api_key（鍵なしとは分ける）
    let r = srv.c.post(format!("{}/v1/chat/completions", srv.base)).header("authorization", "Basic abc").body(user("x").to_string()).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 401);
    assert_eq!(r.json_text().await["error"]["code"], "invalid_api_key");
    // 受付を止める（admin の鍵だけ）。読めない本文では止めない
    assert_eq!(srv.post_raw("/admin/drain", Some(K_INT), "{oops".into()).await.0, 400);
    assert_eq!(srv.post("/admin/drain", Some(K_INT), &json!({"drain": "yes"})).await.0, 400);
    assert_eq!(srv.chat(K_INT, &user("hi")).await.0, 200);
    assert_eq!(srv.post("/admin/drain", Some(K_INT2), &json!({})).await.0, 403);
    assert_eq!(srv.post("/admin/drain", Some(K_INT), &json!({})).await.0, 200);
    let (s, e) = srv.chat(K_INT, &user("hi")).await;
    assert_eq!((s, e["error"]["code"].as_str()), (503, Some("draining")));
    assert_eq!(srv.get("/health", None).await.0, 503);
    assert_eq!(srv.post("/admin/drain", Some(K_INT), &json!({"drain": false})).await.0, 200);
    assert_eq!(srv.chat(K_INT, &user("hi")).await.0, 200);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn metrics_key_and_stall() {
    // /metrics に admin の鍵を求める設定
    let fake = FakeBackend::new(tok(), FakeConfig { delay: Duration::ZERO, ..Default::default() });
    let cfg = ServerConfig { metrics_key: true, stall_timeout: Duration::from_secs(1), ..cfg() };
    let srv = start_with(cfg, fake.clone(), Some(fake.clone())).await;
    assert_eq!(srv.get("/metrics", None).await.0, 401);
    assert_eq!(srv.get("/metrics", Some(K_INT2)).await.0, 403);
    let r = srv.c.get(format!("{}/metrics", srv.base)).bearer_auth(K_INT).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200);
    // つないだまま読まない客: 送れない時間が stall_timeout を超えたら切る（依頼が終わり 499 で残る）
    let b = with(user("[[len:300000]]"), "stream", json!(true));
    let resp = srv.c.post(format!("{}/v1/chat/completions", srv.base)).bearer_auth(K_INT).body(b.to_string()).send().await.unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let st = srv.st.clone();
    wait_for("読まない客の依頼が終わる", 15_000, || st.inflight.load(Ordering::SeqCst) == 0).await;
    let m = srv.c.get(format!("{}/metrics", srv.base)).bearer_auth(K_INT).send().await.unwrap().text().await.unwrap();
    assert!(m.contains("tachyon_requests_total{account=\"internal\",endpoint=\"chat\",code=\"499\"} 1"), "{m}");
    drop(resp);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn queue_full_is_429() {
    let fake = FakeBackend::new(tok(), FakeConfig { max_running: 1, max_waiting: 1, ..Default::default() });
    let cfg = ServerConfig { probe_cache: Duration::ZERO, ..cfg() };
    let srv = start_with(cfg, fake.clone(), Some(fake)).await;
    let slow = with(user("[[len:400]][[delay:5]]"), "stream", json!(true));
    let a = srv.c.post(format!("{}/v1/chat/completions", srv.base)).bearer_auth(K_INT).body(slow.to_string()).send().await.unwrap();
    let fk = srv.fake.clone().unwrap();
    wait_for("1 本目が走る", 3000, || fk.stats().running == 1).await;
    let b = srv.c.post(format!("{}/v1/chat/completions", srv.base)).bearer_auth(K_INT).body(slow.to_string()).send().await.unwrap();
    wait_for("2 本目が待つ", 3000, || fk.stats().waiting == 1).await;
    let (s, h, e) = srv.post("/v1/chat/completions", Some(K_INT), &user("hi")).await;
    assert_eq!((s, e["error"]["code"].as_str()), (429, Some("queue_full")));
    assert!(h.contains_key("retry-after"));
    let m = srv.metrics().await;
    assert!(m.contains("tachyon_queue_length 1") && m.contains("tachyon_running_requests 1"), "{m}");
    assert!(m.contains("tachyon_rejected_total{account=\"internal\",endpoint=\"chat\",reason=\"queue_full\"} 1"));
    // 見張り: 待ち行列が一杯でも語が出ていれば「混んでいるだけ」（200・busy）。落とさせない
    let (s, h) = srv.get("/health?probe=1", None).await;
    assert_eq!((s, h["probe"].as_str()), (200, Some("busy")), "{h}");
    drop((a, b));
    let fk = srv.fake.clone().unwrap();
    wait_for("空く", 3000, || fk.stats().running == 0 && fk.stats().waiting == 0).await;
    let (s, h) = srv.get("/health?probe=1", None).await;
    assert_eq!((s, h["probe"].as_str()), (200, Some("answered")), "{h}");
}

/// 待ち行列にいる間（まだ何も書いていない）に客が切っても、keep-alive で気づいて待ちから外す（長いプレフィルを無駄にしない）
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn queued_disconnect_is_noticed() {
    let fake = FakeBackend::new(tok(), FakeConfig { max_running: 1, max_waiting: 4, ..Default::default() });
    let cfg = ServerConfig { keepalive_secs: 1, ..cfg() };
    let srv = start_with(cfg, fake.clone(), Some(fake.clone())).await;
    let slow = with(user("[[len:3000]][[delay:3]]"), "stream", json!(true));
    let mut a = srv.c.post(format!("{}/v1/chat/completions", srv.base)).bearer_auth(K_INT).body(slow.to_string()).send().await.unwrap();
    wait_for("1 本目が走る", 3000, || fake.stats().running == 1).await;
    let b = srv.c.post(format!("{}/v1/chat/completions", srv.base)).bearer_auth(K_INT2).body(slow.to_string()).send().await.unwrap();
    wait_for("2 本目が待つ", 3000, || fake.stats().waiting == 1).await;
    let rec_b = fake.last().unwrap();
    drop(b);
    // 1 本目は読み続ける（走っている人は切られない）
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_millis(3500) && !rec_b.cancelled.load(Ordering::SeqCst) {
        let _ = tokio::time::timeout(Duration::from_millis(50), a.chunk()).await;
    }
    assert!(rec_b.cancelled.load(Ordering::SeqCst), "待っている間に切った依頼が外れない");
    assert_eq!(fake.stats().waiting, 0);
    assert_eq!(fake.stats().running, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn concurrency_100() {
    let srv = start(FakeConfig { max_running: 32, max_waiting: 128, delay: Duration::from_millis(2), ..Default::default() }).await;
    let srv = Arc::new(srv);
    let t0 = Instant::now();
    let mut hs = vec![];
    for i in 0..100 {
        let srv = srv.clone();
        hs.push(tokio::spawn(async move {
            let want = format!("req-{i}-日本語-{}", "x".repeat(i % 7));
            let b = user(&format!("[[echo]]{want}"));
            let got = if i % 2 == 0 {
                let (s, ch, done) = srv.sse(K_INT, &b).await;
                assert!(s == 200 && done, "{i}: {s}");
                let (r, c, _, f, _) = merge(&ch);
                assert_eq!((r.as_str(), f.as_deref()), (DEFAULT_THOUGHT, Some("stop")));
                c
            } else {
                let (s, j) = srv.chat(K_INT, &b).await;
                assert_eq!(s, 200, "{i}: {j}");
                j["choices"][0]["message"]["content"].as_str().unwrap().to_string()
            };
            assert_eq!(got, want, "{i} の答えが混ざった");
        }));
    }
    for h in hs {
        h.await.unwrap();
    }
    let el = t0.elapsed();
    let st = srv.st.clone();
    wait_for("inflight が 0 に", 3000, || st.inflight.load(Ordering::SeqCst) == 0).await;
    let m = srv.metrics().await;
    assert!(m.contains("tachyon_requests_total{account=\"internal\",endpoint=\"chat\",code=\"200\"} 100"), "{m}");
    assert!(m.contains("tachyon_ttft_seconds_count 100") && m.contains("tachyon_inflight_requests 0") && m.contains("tachyon_queue_length 0"));
    let max_run = srv.fake().records().len();
    assert_eq!(max_run, 100);
    eprintln!("100 本（32 本ずつ走る・1 語 2ms）: {:.2} 秒", el.as_secs_f64());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn metrics_logs_and_cache_scope() {
    let srv = start(FakeConfig::default()).await;
    let marker = "SECRET-MARKER-こんにちは-42";
    let long = format!("{marker} {}", "共有する長い前置きの文。".repeat(40));
    // 社内どうしは文頭を共有、お客さまはアカウントの中だけ（docs/API.md 3.6）
    let (_, a) = srv.chat(K_INT, &user(&long)).await;
    let (_, b) = srv.chat(K_INT2, &user(&long)).await;
    let (_, c) = srv.chat(K_CUST2, &user(&long)).await;
    let (_, d) = srv.chat(K_CUST2, &user(&long)).await;
    let cached = |j: &Json| j["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap();
    assert_eq!(cached(&a), 0);
    assert!(cached(&b) >= 64, "社内の 2 本目は当たる: {b}");
    assert_eq!(cached(&c), 0, "お客さまは社内の文頭に当たらない");
    assert!(cached(&d) >= 64);
    let recs = srv.fake().records();
    assert_eq!(recs[2].scope.namespace, "acct:cust2");
    assert_eq!(recs[1].scope.namespace, "internal");
    // system の終わりの位置（状態の写しの手がかり）
    let sys = json!({"messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "hi"}]});
    assert_eq!(srv.chat(K_INT, &sys).await.0, 200);
    let rec = srv.fake().last().unwrap();
    let se = rec.hints.system_end.unwrap();
    assert_eq!(rec.prompt[se], tok().ids.im_start);
    assert!(tok().decode(&rec.prompt[..se], false).ends_with("You are terse.<|im_end|>\n"));
    // 数は出るが、中身・鍵は /metrics にもログにも出ない
    let _ = srv.chat(K_CUST, &user(&format!("{marker} [[error:1]]"))).await;
    let _ = srv.chat(K_CUST, &with(user(marker), "temperature", json!(9))).await;
    let _ = srv.post("/v1/chat/completions", Some("sk-not-a-real-key-SECRET-0000"), &user(marker)).await;
    let m = srv.metrics().await;
    for want in [
        "tachyon_prompt_tokens_total{account=\"internal\"}",
        "tachyon_completion_tokens_total{account=\"cust2\"}",
        "tachyon_reasoning_tokens_total{account=\"internal2\"}",
        "tachyon_cached_prompt_tokens_total{account=\"cust2\"}",
        "tachyon_prefix_cache_hit_ratio{account=\"internal2\"}",
        "tachyon_ttft_seconds_bucket{le=\"0.5\"}",
        "tachyon_decode_tokens_per_second_count",
        "tachyon_queue_length 0",
        "tachyon_tokens_per_second",
        "tachyon_requests_total{account=\"cust\",endpoint=\"chat\",code=\"500\"} 1",
        "tachyon_rejected_total{account=\"cust\",endpoint=\"chat\",reason=\"invalid_value\"} 1",
        "tachyon_rejected_total{account=\"-\",endpoint=\"chat\",reason=\"invalid_api_key\"} 1",
        "tachyon_ready 1",
    ] {
        assert!(m.contains(want), "{want} が無い:\n{m}");
    }
    let logs = srv.logs.lock().unwrap().join("\n");
    assert!(logs.contains("\"ev\":\"chat\"") && logs.contains("\"account\":\"cust2\"") && logs.contains("\"ev\":\"reject\""));
    for bad in [marker, "SECRET", "こんにちは", "共有する長い前置き"].iter().chain(ALL_KEYS.iter()) {
        assert!(!m.contains(bad), "/metrics に {bad}");
        assert!(!logs.contains(bad), "ログに {bad}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn documents() {
    let srv = start(FakeConfig::default()).await;
    let text = format!("DOCUMENT: {}", "議事録の本文。決まったこと: 来週までに試す。".repeat(30));
    let doc_body = json!({"name": "minutes", "messages": [{"role": "system", "content": "Answer from the document."}, {"role": "user", "content": text}]});
    let (s, _, d) = srv.post("/v1/documents", Some(K_INT), &doc_body).await;
    assert_eq!(s, 200, "{d}");
    let id = d["id"].as_str().unwrap().to_string();
    assert!(id.starts_with("doc-"));
    assert_eq!((d["status"].as_str(), d["shared"].as_bool(), d["object"].as_str()), (Some("ready"), Some(true), Some("document")));
    assert_eq!(d["chat_template_kwargs"]["reasoning_effort"], "medium");
    let n_doc = d["tokens"].as_u64().unwrap() as usize;
    assert!(n_doc > 64);
    // 見える人: 社内の鍵どうし。お客さまからは見えない
    let (_, l) = srv.get("/v1/documents", Some(K_INT2)).await;
    assert_eq!(l["data"][0]["id"], id.as_str());
    assert_eq!(srv.get("/v1/documents", Some(K_CUST)).await.1["data"], json!([]));
    assert_eq!(srv.get(&format!("/v1/documents/{id}"), Some(K_CUST)).await.0, 404);
    assert_eq!(srv.get(&format!("/v1/documents/{id}"), Some(K_INT2)).await.0, 200);
    // 文書を付けた会話: 文書の messages が前に付き、文書の分は使い回し
    let ask = with(user("[[echo]]what was decided?"), "document", json!(id));
    let (s, j) = srv.chat(K_INT2, &ask).await;
    assert_eq!(s, 200, "{j}");
    assert_eq!(j["choices"][0]["message"]["content"], "what was decided?");
    let rec = srv.fake().last().unwrap();
    assert_eq!(rec.hints.document.as_ref().map(|d| (d.id.as_str(), d.tokens)), Some((id.as_str(), n_doc)));
    assert!(j["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap() as usize >= n_doc);
    let p = tok().decode(&rec.prompt, false);
    assert!(p.starts_with("<|im_start|>system\nAnswer from the document.<|im_end|>\n<|im_start|>user\nDOCUMENT:") && p.contains("what was decided?"));
    // 文書と違う設定（xhigh は system に 1 文足す）→ 400
    let (s, e) = srv.chat(K_INT2, &with(ask.clone(), "reasoning_effort", json!("xhigh"))).await;
    assert_eq!((s, e["error"]["code"].as_str()), (400, Some("document_mismatch")));
    // お客さまの文書はそのアカウントだけ・共有はできない
    let (s, _, e) = srv.post("/v1/documents", Some(K_CUST), &with(doc_body.clone(), "shared", json!(true))).await;
    assert_eq!((s, e["error"]["param"].as_str()), (400, Some("shared")));
    let (s, _, cd) = srv.post("/v1/documents", Some(K_CUST2), &doc_body).await;
    assert_eq!((s, cd["shared"].as_bool()), (200, Some(false)));
    let cid = cd["id"].as_str().unwrap();
    assert_eq!(srv.get(&format!("/v1/documents/{cid}"), Some(K_CUST)).await.0, 404);
    assert_eq!(srv.get(&format!("/v1/documents/{cid}"), Some(K_INT)).await.0, 404);
    assert_eq!(srv.chat(K_CUST, &with(user("hi"), "document", json!(cid))).await.0, 404);
    let (s, j) = srv.chat(K_CUST2, &with(user("hi"), "document", json!(cid))).await;
    assert_eq!(s, 200);
    assert_eq!(srv.fake().last().unwrap().scope.namespace, "acct:cust2");
    assert!(j["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap() as usize >= n_doc);
    // 一括（既定は low）。1 件ずつ結果か誤り
    let (s, _, bj) = srv.post("/v1/documents/batch", Some(K_INT), &json!({"documents": [doc_body.clone(), {"messages": []}]})).await;
    assert_eq!(s, 200);
    assert_eq!((bj["data"][0]["priority"].as_str(), bj["data"][1]["error"]["code"].as_str()), (Some("low"), Some("invalid_value")));
    // 消す: 持ち主（か admin）だけ
    assert_eq!(srv.delete(&format!("/v1/documents/{id}"), K_INT2).await.0, 403);
    let (s, dj) = srv.delete(&format!("/v1/documents/{id}"), K_INT).await;
    assert_eq!((s, dj["deleted"].as_bool()), (200, Some(true)));
    assert_eq!(srv.get(&format!("/v1/documents/{id}"), Some(K_INT)).await.0, 404);
    assert_eq!(srv.chat(K_INT, &ask).await.0, 404);
    let m = srv.metrics().await;
    assert!(m.contains("tachyon_documents{account=\"cust2\"} 1"), "{m}");
}

/// SimpleScheduler（M1 のモデルが載る道）を FakeEngine で
fn engine_srv_loader(return_tokens: bool, max_ctx: usize, log: Arc<Mutex<Option<Arc<Mutex<fake::FakeEngineLog>>>>>) -> EngineLoader {
    engine_srv_loader_multi(return_tokens, 1, max_ctx, log)
}

fn engine_srv_loader_multi(return_tokens: bool, multi: usize, max_ctx: usize, log: Arc<Mutex<Option<Arc<Mutex<fake::FakeEngineLog>>>>>) -> EngineLoader {
    engine_srv_loader_ex(return_tokens, multi, false, max_ctx, log)
}

fn engine_srv_loader_ex(return_tokens: bool, multi: usize, reuse: bool, max_ctx: usize, log: Arc<Mutex<Option<Arc<Mutex<fake::FakeEngineLog>>>>>) -> EngineLoader {
    Box::new(move |progress: &dyn Fn(f32, &str)| {
        progress(0.5, "fake");
        let mut e = FakeEngine::new(tok(), EngineInfo { max_seqs: 8, max_ctx, max_step_tokens: 4096 }, return_tokens);
        e.multi = multi;
        e.reuse = reuse;
        *log.lock().unwrap() = Some(e.log.clone());
        Ok(Box::new(e) as Box<dyn tachyon_v100::server::Engine>)
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn engine_scheduler_path() {
    // (logits を返す)・(1 語を返す)・(下書きの受け取りの真似で 3 語まで返す): どれも同じ答え・同じ止まり方
    for (return_tokens, multi) in [(false, 1), (true, 1), (true, 3)] {
        let log = Arc::new(Mutex::new(None));
        let sc = SchedConfig { max_running: 8, max_waiting: 64, step_tokens: 512, prefill_chunk: 64, prefill_chunk_alone: 256, valid_vocab: tok().vocab_size() };
        let sched = SimpleScheduler::spawn(engine_srv_loader_multi(return_tokens, multi, 8192, log.clone()), sc);
        let s2 = sched.clone();
        wait_for("エンジンの読み込み", 5000, || s2.status() == Status::Ready).await;
        let srv = Arc::new(start_with(cfg(), sched.clone(), None).await);
        let elog = log.lock().unwrap().clone().unwrap();
        // 8 本を同時に（温度 0）。1 歩にまとまり、答えは混ざらない
        let mut hs = vec![];
        for i in 0..8 {
            let srv = srv.clone();
            hs.push(tokio::spawn(async move {
                let want = format!("answer number {i} です");
                let b = with(user(&format!("[[delay:3]][[echo]]{want}")), "temperature", json!(0));
                let (s, j) = if i % 2 == 0 {
                    srv.chat(K_INT, &b).await
                } else {
                    let (s, ch, _) = srv.sse(K_INT, &b).await;
                    let (r, c, _, f, _) = merge(&ch);
                    (s, json!({"choices": [{"message": {"content": c, "reasoning_content": r}, "finish_reason": f}]}))
                };
                assert_eq!(s, 200, "{j}");
                assert_eq!(j["choices"][0]["message"]["content"], want.as_str());
                assert_eq!(j["choices"][0]["message"]["reasoning_content"], DEFAULT_THOUGHT);
                assert_eq!(j["choices"][0]["finish_reason"], "stop");
            }));
        }
        for h in hs {
            h.await.unwrap();
        }
        let g = elog.lock().unwrap().clone();
        assert!(g.max_batch > 1, "まとめて回していない: {g:?}");
        // 長い入力は塊で（1 歩の語は予算 512 まで）
        let long = format!("[[echo]]ok {}", "長い入力の文。".repeat(300));
        let (s, j) = srv.chat(K_INT, &with(with(user(&long), "temperature", json!(0)), "max_tokens", json!(4))).await;
        assert_eq!((s, j["choices"][0]["finish_reason"].as_str()), (200, Some("length")));
        assert_eq!(j["usage"]["completion_tokens"], 4);
        assert!(elog.lock().unwrap().max_step_tokens <= 512);
        // 止める語・止めの文字列（取り消し → release）・失敗
        let b = with(user("[[raw]]keep<|fim_pad|>drop"), "temperature", json!(0));
        let (_, j) = srv.chat(K_INT, &with(b, "chat_template_kwargs", json!({"enable_thinking": false}))).await;
        assert_eq!(j["choices"][0]["message"]["content"], "keep");
        let b = with(with(user("[[delay:2]][[raw]]aaa STOP bbb ccc ddd eee fff"), "stop", json!("STOP")), "temperature", json!(0));
        let (_, j) = srv.chat(K_INT, &with(b, "chat_template_kwargs", json!({"enable_thinking": false}))).await;
        assert_eq!(j["choices"][0]["message"]["content"], "aaa");
        let (s, _) = srv.chat(K_INT, &user("[[error:2]]")).await;
        assert_eq!(s, 500);
        let s2 = sched.clone();
        wait_for("全部 release", 3000, || s2.stats().running == 0).await;
        let rel = elog.lock().unwrap().released.len();
        assert!(rel >= 12, "release が足りない: {rel}");
        // エンジンの文脈の上限（8192）を超える入力は口で 400
        let huge = "日本語の文。".repeat(12_000 / tok().encode("日本語の文。").unwrap().len() + 10);
        let (s, e) = srv.chat(K_INT, &user(&huge)).await;
        assert_eq!((s, e["error"]["code"].as_str()), (400, Some("context_length_exceeded")));
        let (_, m) = srv.get("/v1/models", Some(K_INT)).await;
        assert_eq!(m["data"][0]["max_context"], 8192);
        // 文書の先計算はこの順番決めには無い → 501
        let (s, _, e) = srv.post("/v1/documents", Some(K_INT), &json!({"messages": [{"role": "user", "content": "doc"}]})).await;
        assert_eq!((s, e["error"]["code"].as_str()), (501, Some("documents_unsupported")));
    }
}

/// M2d: 考えの上限で考えを閉じて答えへ／前の依頼の頭の使い回しはアカウントごと・答えは同じ
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn engine_think_budget_and_prefix_reuse() {
    for multi in [1, 3] {
        let log = Arc::new(Mutex::new(None));
        let sc = SchedConfig { max_running: 1, max_waiting: 16, step_tokens: 512, prefill_chunk: 64, prefill_chunk_alone: 256, valid_vocab: tok().vocab_size() };
        let sched = SimpleScheduler::spawn(engine_srv_loader_ex(true, multi, true, 8192, log.clone()), sc);
        let s2 = sched.clone();
        wait_for("エンジンの読み込み", 5000, || s2.status() == Status::Ready).await;
        let srv = Arc::new(start_with(cfg(), sched.clone(), None).await);
        let think = |b: Json| with(b, "chat_template_kwargs", json!({"enable_thinking": true}));
        // 上限なし: 考えは DEFAULT_THOUGHT のまま
        let (s, j) = srv.chat(K_INT, &think(with(user("[[echo]]ok"), "temperature", json!(0)))).await;
        assert_eq!(s, 200, "{j}");
        assert_eq!(j["choices"][0]["message"]["reasoning_content"], DEFAULT_THOUGHT);
        // 上限 3 語: 考えを閉じる文が差し込まれ、考えは短く、答えが続く
        for k in ["thinking_budget_tokens", "reasoning_budget_tokens"] {
            let (s, j) = srv.chat(K_INT, &think(with(with(user("[[echo]]ok"), "temperature", json!(0)), k, json!(3)))).await;
            assert_eq!(s, 200, "{j}");
            let r = j["choices"][0]["message"]["reasoning_content"].as_str().unwrap_or("");
            assert!(r.contains("Considering the limited time by the user"), "{k}: {r:?}");
            assert!(r.len() < DEFAULT_THOUGHT.len() + 120, "{k}: 考えが長い {r:?}");
            // 偽のエンジンは予定の続きを語の数で引くので、差し込んだ後は予定の終わり（<|im_end|>）に来る。考えが閉じて止まればよい
            assert!(!r.contains("</think>") && j["choices"][0]["finish_reason"] == "stop", "{j}");
        }
        // 上限の値の誤りは 400
        let (s, _) = srv.chat(K_INT, &with(user("hi"), "thinking_budget_tokens", json!("x"))).await;
        assert_eq!(s, 400);
        // 頭の使い回し: 同じアカウントで会話を伸ばすと cached > 0・答えは使い回さないときと同じ
        let cached = |j: &Json| j["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap();
        let m1 = json!([{"role": "user", "content": format!("[[echo]]first {}", "長い入力の文。".repeat(100))}]);
        let b1 = with(json!({"messages": m1.clone()}), "temperature", json!(0));
        let (_, j1) = srv.chat(K_INT, &b1).await;
        let mut m2 = m1.as_array().unwrap().clone();
        m2.push(json!({"role": "assistant", "content": j1["choices"][0]["message"]["content"].clone()}));
        m2.push(json!({"role": "user", "content": "[[echo]]second turn"}));
        let b2 = with(json!({"messages": m2}), "temperature", json!(0));
        // 別のアカウントには使い回さない（iso-a/b は切り分けの覚えの試しが数えるので使わない）
        let (_, jb) = srv.chat(K_CUST2, &b2).await;
        assert_eq!(cached(&jb), 0, "別のアカウントに前の依頼の頭が当たった");
        let (_, j1b) = srv.chat(K_INT, &b1).await;
        assert_eq!(cached(&j1b), 0);
        let (_, ja) = srv.chat(K_INT, &b2).await;
        assert!(cached(&ja) > 0, "同じアカウントで頭を使い回していない: {ja}");
        assert_eq!(ja["choices"][0]["message"]["content"], jb["choices"][0]["message"]["content"]);
        assert_eq!(ja["choices"][0]["message"]["content"], "second turn");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn engine_loading_and_failure() {
    let log = Arc::new(Mutex::new(None));
    // 読み込みは試しが門を開けるまで終わらない（混んだ機械でも時間に頼らない）
    let gate = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let slow: EngineLoader = {
        let inner = engine_srv_loader(false, 4096, log);
        let gate = gate.clone();
        Box::new(move |p: &dyn Fn(f32, &str)| {
            p(0.25, "waiting for the test");
            while !gate.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(2));
            }
            inner(p)
        })
    };
    let sched = SimpleScheduler::spawn(slow, SchedConfig { valid_vocab: tok().vocab_size(), ..Default::default() });
    let srv = start_with(cfg(), sched.clone(), None).await;
    let (s, h) = srv.get("/health", None).await;
    assert_eq!((s, h["status"].as_str()), (503, Some("loading")));
    let (s, e) = srv.chat(K_INT, &user("hi")).await;
    assert_eq!((s, e["error"]["code"].as_str()), (503, Some("loading")));
    gate.store(true, Ordering::SeqCst);
    let s2 = sched.clone();
    wait_for("読み込み", 5000, || s2.status() == Status::Ready).await;
    assert_eq!(srv.get("/health", None).await.0, 200);
    // 歩が 3 回続けて失敗したら Failed（/health 500 → 見張りが起こし直す）。それまでの依頼は 500
    for i in 0..3 {
        let (s, _) = srv.chat(K_INT, &user("[[stepfail]]")).await;
        assert_eq!(s, 500, "{i}");
    }
    let s2 = sched.clone();
    wait_for("Failed", 3000, || matches!(s2.status(), Status::Failed(_))).await;
    let (s, h) = srv.get("/health", None).await;
    assert_eq!((s, h["status"].as_str()), (500, Some("error")));
    assert!(h["message"].as_str().unwrap().contains("3 times"));
    let bad: EngineLoader = Box::new(|_: &dyn Fn(f32, &str)| Err("no GPU".to_string()));
    let sched = SimpleScheduler::spawn(bad, SchedConfig::default());
    let s2 = sched.clone();
    wait_for("失敗", 3000, || matches!(s2.status(), Status::Failed(_))).await;
    let srv = start_with(cfg(), sched, None).await;
    let (s, h) = srv.get("/health", None).await;
    assert_eq!((s, h["status"].as_str(), h["message"].as_str()), (500, Some("error"), Some("no GPU")));
    let (s, e) = srv.chat(K_INT, &user("hi")).await;
    assert_eq!((s, e["error"]["code"].as_str()), (503, Some("backend_failed")));
}

// ---------------------------------------------------------------- serve の既定

/// serve の既定（ServerConfig::default）: 考えない。依頼で enable_thinking: true を書けば考える。/health の state
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn serve_default_thinking_off_by_default() {
    let fake = FakeBackend::new(tok(), FakeConfig::default());
    let srv = start_with(ServerConfig::default(), fake.clone(), Some(fake)).await;
    let (s, h) = srv.get("/health", None).await;
    assert_eq!((s, h["status"].as_str(), h["state"].as_str()), (200, Some("ok"), Some("ready")), "{h}");
    let (s, j) = srv.chat(K_INT, &user("Hello?")).await;
    assert_eq!(s, 200, "{j}");
    let m = &j["choices"][0]["message"];
    assert!(m.get("reasoning_content").is_none(), "考えないはず: {j}");
    assert_eq!(m["content"], DEFAULT_ANSWER);
    assert!(prompt_text(&srv).ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "{}", prompt_text(&srv));
    let sp = srv.fake().last().unwrap().sampling;
    assert_eq!((sp.temperature, sp.top_p, sp.top_k, sp.presence_penalty), (0.7, 0.8, 20, 1.5), "考えなしの推奨の値");
    assert_eq!(j["usage"]["completion_tokens_details"]["reasoning_tokens"], 0);
    // 書けば考える（値は考えありの推奨）
    let (s, j) = srv.chat(K_INT, &with(user("Hello?"), "chat_template_kwargs", json!({"enable_thinking": true}))).await;
    assert_eq!(s, 200);
    assert_eq!(j["choices"][0]["message"]["reasoning_content"], DEFAULT_THOUGHT);
    assert_eq!(srv.fake().last().unwrap().sampling.temperature, 1.0);
    // 道具があっても考えない（preserve_thinking は道具ありの既定で true のまま）
    let (s, _) = srv.chat(K_INT, &with(user("weather?"), "tools", weather_tools())).await;
    assert_eq!(s, 200);
    assert!(prompt_text(&srv).ends_with("<think>\n\n</think>\n\n"));
}

/// 大きさの上限は描く・切る前に: 本文 4MiB（413）・1 つの message 1MiB・messages の数・入力の語の上限（切り分けを途中でやめる）
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn serve_default_byte_and_token_limits() {
    let fake = FakeBackend::new(tok(), FakeConfig::default());
    let c = ServerConfig { max_context: 65_536, max_prompt_tokens: 60_000, ..ServerConfig::default() };
    let srv = start_with(c, fake.clone(), Some(fake)).await;
    // 本文の上限（既定 4MiB）を超える → 413（OpenAI 形の JSON）。読み切らずに断る。
    // 試しは上限 64KiB の口で（大きな本文を書いている途中で切られると、客の側が返事を読めないことがあるため）
    assert_eq!(ServerConfig::default().max_body_bytes, 4 << 20);
    let small = FakeBackend::new(tok(), FakeConfig::default());
    let srv_small = start_with(ServerConfig { max_body_bytes: 64 << 10, ..ServerConfig::default() }, small.clone(), Some(small)).await;
    let big = json!({"messages": [{"role": "user", "content": "a".repeat(100 << 10)}]}).to_string();
    let (s, _, e) = srv_small.post_raw("/v1/chat/completions", Some(K_INT), big).await;
    assert_eq!((s, e["error"]["code"].as_str(), e["error"]["type"].as_str()), (413, Some("request_too_large"), Some("invalid_request_error")), "{e}");
    assert!(e["error"]["message"].as_str().unwrap().contains("65536"), "{e}");
    assert_eq!(srv_small.chat(K_INT, &user("hi")).await.0, 200);
    // 1 つの message が 1MiB を超える → 400（描かない）
    let one = json!({"messages": [{"role": "user", "content": "a".repeat((1 << 20) + 1)}]});
    let (s, e) = srv.chat(K_INT, &one).await;
    assert_eq!((s, e["error"]["code"].as_str(), e["error"]["param"].as_str()), (400, Some("message_too_large"), Some("messages[0]")), "{e}");
    let parts = json!({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x", "reasoning_content": "r".repeat((1 << 20) + 1)}, {"role": "user", "content": "hi"}]});
    assert_eq!(srv.chat(K_INT, &parts).await.1["error"]["param"], "messages[1]");
    // messages が多すぎる
    let many: Vec<Json> = (0..5000).map(|i| json!({"role": if i % 2 == 0 { "user" } else { "assistant" }, "content": "x"})).collect();
    let (s, e) = srv.chat(K_INT, &json!({"messages": many})).await;
    assert_eq!((s, e["error"]["code"].as_str()), (400, Some("too_many_messages")));
    // 入力の語の上限 60,000: 超えたら 400。長い回が先にあれば、その回で切り分けをやめる（at least N 語、N は全体より小さい）
    let unit = tok().encode("日本語の文。").unwrap().len();
    let chunk = "日本語の文。".repeat(70_000 / unit + 10);
    let n_one = tok().encode(&chunk).unwrap().len();
    assert!(n_one > 60_000 && chunk.len() < 1 << 20);
    let body = json!({"messages": [{"role": "user", "content": chunk}, {"role": "assistant", "content": "ok"}, {"role": "user", "content": chunk}, {"role": "assistant", "content": "ok"}, {"role": "user", "content": chunk}]});
    let (s, e) = srv.chat(K_INT, &body).await;
    assert_eq!((s, e["error"]["code"].as_str()), (400, Some("context_length_exceeded")), "{e}");
    let msg = e["error"]["message"].as_str().unwrap();
    assert!(msg.contains("60000"), "{msg}");
    let at_least: usize = msg.split("at least ").nth(1).and_then(|r| r.split(' ').next()).and_then(|n| n.parse().ok()).expect(msg);
    assert!(at_least < 2 * n_one, "全部切ってから断っている（{at_least} 語）");
    // ちょうど入る長さは通る（出力の余地は 65,536 − 入力）
    let fit = "日本語の文。".repeat(50_000 / unit);
    let (s, _) = srv.chat(K_INT, &with(user(&fit), "max_tokens", json!(16))).await;
    assert_eq!(s, 200);
    // 文書も同じ上限（本文の大きさ）
    let (s, _, e) = srv.post("/v1/documents", Some(K_INT), &one).await;
    assert_eq!((s, e["error"]["code"].as_str()), (400, Some("message_too_large")));
}

/// 切り分けの覚えはアカウントごと（口を通して）: B が A と同じ会話を送っても、A の覚えに当たらない
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn serve_default_tokenizer_cache_is_per_account() {
    let srv = start(FakeConfig::default()).await;
    let secret = format!("A 社の秘密の議事録: {}", "来期の値上げは 12% とする。".repeat(40));
    let conv = json!({"messages": [
        {"role": "system", "content": format!("社内の決まり。{}", "長い前置き。".repeat(40))},
        {"role": "user", "content": secret},
        {"role": "assistant", "content": "要点を読みました。"},
        {"role": "user", "content": "まとめて。"},
    ]});
    let t = srv.st.tok.clone();
    assert_eq!(t.cache_stats("iso-b"), Default::default());
    assert_eq!(srv.chat(K_ISO_A, &conv).await.0, 200);
    assert_eq!(srv.chat(K_ISO_A, &conv).await.0, 200);
    let a = t.cache_stats("iso-a");
    assert!(a.hits >= 2 && a.entries >= 2, "A は自分の覚えに当たる: {a:?}");
    assert_eq!(srv.chat(K_ISO_B, &conv).await.0, 200);
    let b = t.cache_stats("iso-b");
    assert_eq!(b.hits, 0, "B が A の覚えに当たった: {b:?}");
    assert!(b.misses >= 2);
    assert_eq!(t.cache_stats("iso-a"), a, "B の依頼で A の覚えが変わった");
    // 語の番号は同じ（覚えの有無で答えは変わらない）
    let recs = srv.fake().records();
    let n = recs.len();
    assert_eq!(recs[n - 1].prompt, recs[n - 2].prompt);
}

/// SIGTERM の後の手順（drain_and_cancel）: 走っている依頼（stream）と待っている依頼を数秒で終わらせる。M1 と同じ道（SimpleScheduler・1 本ずつ）
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn serve_default_shutdown_cancels_inflight() {
    let log = Arc::new(Mutex::new(None));
    let sc = SchedConfig { max_running: 1, max_waiting: 8, step_tokens: 2048, prefill_chunk: 2048, prefill_chunk_alone: 2048, valid_vocab: tok().vocab_size() };
    let sched = SimpleScheduler::spawn(engine_srv_loader(false, 8192, log), sc);
    let s2 = sched.clone();
    wait_for("エンジンの読み込み", 5000, || s2.status() == Status::Ready).await;
    let srv = Arc::new(start_with(cfg(), sched.clone(), None).await);
    // 長い生成（1 語 5ms × 10 万語）を stream で 1 本、待ち行列に 2 本
    let long = with(with(user("[[len:100000]][[delay:5]]"), "max_tokens", json!(100_000)), "temperature", json!(0));
    let s1 = srv.clone();
    let l1 = long.clone();
    let h1 = tokio::spawn(async move { s1.sse(K_INT, &l1).await });
    let st = srv.st.clone();
    wait_for("1 本目が走る", 5000, || st.backend.stats().running == 1).await;
    let mut queued = vec![];
    for _ in 0..2 {
        let s = srv.clone();
        let b = long.clone();
        queued.push(tokio::spawn(async move { s.chat(K_INT, &b).await }));
    }
    wait_for("2 本が待つ", 5000, || st.backend.stats().waiting == 2).await;
    let t0 = Instant::now();
    server::drain_and_cancel(&srv.st, Duration::from_millis(300)).await;
    let el = t0.elapsed();
    assert!(el < Duration::from_secs(4), "止めに {el:?}");
    assert_eq!(srv.st.inflight.load(Ordering::SeqCst), 0);
    let (s, chunks, done) = h1.await.unwrap();
    assert_eq!(s, 200);
    assert!(done, "stream が [DONE] で閉じていない");
    let err = chunks.iter().find(|c| c.get("error").is_some()).expect("止めの誤りの塊が無い");
    assert!(err["error"]["message"].as_str().unwrap().contains("shutting down"), "{err}");
    for q in queued {
        let (s, e) = q.await.unwrap();
        assert_eq!(s, 500, "{e}");
        assert!(e["error"]["message"].as_str().unwrap().contains("shutting down"), "{e}");
    }
    // 新しい依頼は受けない
    assert_eq!(srv.chat(K_INT, &user("hi")).await.0, 503);
    let (s, h) = srv.get("/health", None).await;
    assert_eq!((s, h["state"].as_str()), (503, Some("draining")));
    let logs = srv.logs.lock().unwrap().join("\n");
    assert!(logs.contains("\"ev\":\"cancel_all\"") && logs.contains("\"ev\":\"drained\""), "{logs}");
}

/// --prefix-owner-accounts（2026-11-15）: 載ったアカウントは x-tachyon-prefix-owner の利用者ごとに頭の使い回しを分ける。
/// ⛔ 別の利用者・札なしの依頼には当てない。載っていないアカウントの札は読まない（前と同じ）。形が変なら 400
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn prefix_owner_isolation() {
    let fake = FakeBackend::new(tok(), FakeConfig::default());
    let srv = start_with(ServerConfig { prefix_owner_accounts: vec!["cust2".into()], ..cfg() }, fake.clone(), Some(fake)).await;
    let long = format!("SECRET-OWNER-A {}", "利用者 A だけの長い前置き。".repeat(40));
    let (ua, ub) = ("a".repeat(32), "b".repeat(32));
    let go = |key: &'static str, owner: Option<String>, body: Json| {
        let (base, c) = (srv.base.clone(), srv.c.clone());
        async move {
            let mut r = c.post(format!("{base}/v1/chat/completions")).bearer_auth(key).header("content-type", "application/json").body(body.to_string());
            if let Some(o) = owner {
                r = r.header("x-tachyon-prefix-owner", o);
            }
            let resp = r.send().await.unwrap();
            let s = resp.status().as_u16();
            (s, serde_json::from_str::<Json>(&resp.text().await.unwrap()).unwrap())
        }
    };
    let cached = |j: &Json| j["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or_else(|| panic!("{j}"));
    let (s, a1) = go(K_CUST2, Some(ua.clone()), user(&long)).await;
    assert_eq!(s, 200, "{a1}");
    assert_eq!(cached(&a1), 0);
    let (_, a2) = go(K_CUST2, Some(ua.clone()), user(&long)).await;
    assert!(cached(&a2) >= 64, "同じ利用者の 2 本目は当たる: {a2}");
    let (_, b1) = go(K_CUST2, Some(ub.clone()), user(&long)).await;
    assert_eq!(cached(&b1), 0, "⛔ 別の利用者に A の頭が当たった");
    let (_, n1) = go(K_CUST2, None, user(&long)).await;
    assert_eq!(cached(&n1), 0, "⛔ 札なしの依頼に A の頭が当たった");
    let recs = srv.fake().records();
    assert_eq!(recs[0].scope.owner, format!("cust2#{ua}"));
    assert_eq!(recs[0].scope.account, "cust2", "数はアカウントで数える");
    assert_eq!(recs[3].scope.owner, "cust2");
    // 載っていないアカウントの札は読まない（同じアカウントの 2 本目は札が違っても当たる＝前と同じ）
    let short = format!("SECRET-OWNER-C {}", "載っていないアカウントの前置き。".repeat(12));
    let (_, c1) = go(K_CUST, Some(ua.clone()), user(&short)).await;
    let (_, c2) = go(K_CUST, Some(ub.clone()), user(&short)).await;
    assert_eq!(cached(&c1), 0);
    assert!(cached(&c2) >= 64, "載っていないアカウントは札を見ない: {c2}");
    assert_eq!(srv.fake().records().last().unwrap().scope.owner, "cust");
    // 形が変な札は 400（⛔ 黙ってアカウント全体に落とさない）
    for bad in ["A".repeat(32), "a".repeat(15), "a".repeat(65), "x".repeat(32), "someone@example.com".into()] {
        let (s, j) = go(K_CUST2, Some(bad.clone()), user("hi")).await;
        assert_eq!(s, 400, "{bad}: {j}");
        assert_eq!(j["error"]["param"], "x-tachyon-prefix-owner");
    }
    // ログには札の頭 8 字だけ
    let logs = srv.logs.lock().unwrap().join("\n");
    assert!(logs.contains("\"owner\":\"aaaaaaaa\"") && !logs.contains(&ua), "{logs}");
}
