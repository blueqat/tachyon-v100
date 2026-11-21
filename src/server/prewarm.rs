//! 頭の先計算の札（prewarm: よく使う前置きを起動時に計算しておく）。
//!
//! --prewarm-dir DIR の *.json を起動の後（エンジンが ready になってから）に読み、頭を計算して RAM の段（hostcache）に
//! pinned（追い出さない）・public（どのアカウントでも当たる）で置く。POST /admin/prewarm で読み直す（admin の鍵）。
//! 札の形: {"name": "my-system-prompt-ja", "public": true, "prefix_only": true, "messages": [{"role": "system", "content": "..."}]}
//!   prefix_only = system の中身の頭だけ（後ろに何かが続く）。テンプレートで描いて、その中身の終わりまでを語に切り、
//!   終わりの MARGIN 語を落とす（後ろに続く文字と語の切れ目がまたがっても、当たる所が残るように）
//!   prefix_only でなければ、文書（docs.rs）と同じ「messages の後に user の回が来る」所まで
//! ⛔ 札に入れてよいのは、誰に見えてもよい文だけ（public で全部のアカウントに当たる）。利用者の文を入れない。
use super::backend::{CacheScope, FinishReason, GenEvent, GenRequest, PrefixHints, Priority, Status};
use super::sampler::SamplingParams;
use super::util::rand_id;
use super::AppState;
use crate::template::ChatOptions;
use serde_json::{json, Value as Json};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

const MARGIN: usize = 8;

static DIR: OnceLock<PathBuf> = OnceLock::new();
static RUNNING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_dir(d: &str) {
    let _ = DIR.set(PathBuf::from(d));
}

pub fn dir() -> Option<&'static PathBuf> {
    DIR.get()
}

/// 札 1 つを語に（描いて切る）
fn render(st: &AppState, doc: &Json) -> Result<(String, Vec<u32>), String> {
    let name = doc.get("name").and_then(Json::as_str).ok_or("name が無い")?.to_string();
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)) || name.len() > 64 {
        return Err("name は英数字と -_. で 64 字まで".into());
    }
    if doc.get("public").and_then(Json::as_bool) != Some(true) {
        return Err("public: true の札だけ（全部のアカウントに当たる文だけを置く）".into());
    }
    let msgs = doc.get("messages").and_then(Json::as_array).ok_or("messages が無い")?.clone();
    // 依頼と同じ既定で描く（chat.rs の apply_defaults と同じ）: 27B のテンプレートは enable_thinking・reasoning_effort で system の頭に
    // 「Reasoning effort is set to …」を足すので、既定を入れずに描くと札の語が依頼と頭からずれて当たらない（11-14 に GPU1 で 0 回だった）。
    // 考えあり・effort 違いの依頼に当てたいときは札に chat_template_kwargs を書く（その形ごとに 1 枚）
    let kw = doc.get("chat_template_kwargs").cloned().unwrap_or(Json::Null);
    let mut opts = ChatOptions::from_kwargs(&kw, false).map_err(|e| format!("chat_template_kwargs: {e:?}"))?;
    super::request::apply_opt_defaults(&mut opts, false, &st.cfg.default_effort, st.cfg.default_thinking);
    let sentinel = rand_id("TACHYON_PREWARM_END_", 20);
    let text = if doc.get("prefix_only").and_then(Json::as_bool) == Some(true) {
        let mut m = msgs;
        let last = m.last_mut().ok_or("messages が空")?;
        let c = last.get("content").and_then(Json::as_str).ok_or("prefix_only の最後の回の content は文字列")?.to_string();
        last["content"] = Json::String(format!("{c}{sentinel}"));
        m.push(json!({"role": "user", "content": "x"}));
        let text = st.tpl.render(&Json::Array(m), None, &opts).map_err(|e| format!("template: {e:?}"))?;
        let cut = text.find(&sentinel).ok_or("描いた中に印が無い")?;
        text[..cut].to_string()
    } else {
        let mut m = msgs;
        m.push(json!({"role": "user", "content": sentinel}));
        let text = st.tpl.render(&Json::Array(m), None, &opts).map_err(|e| format!("template: {e:?}"))?;
        let cut = text.rfind(&format!("<|im_start|>user\n{sentinel}")).ok_or("描いた中に印が無い")?;
        text[..cut].to_string()
    };
    let mut ids = st.tok.encode_limited(Some("_prewarm"), &text, st.cfg.max_context).map_err(|e| format!("tokenizer: {e:?}"))?;
    if doc.get("prefix_only").and_then(Json::as_bool) == Some(true) {
        ids.truncate(ids.len().saturating_sub(MARGIN));
    }
    if ids.len() < 64 {
        return Err(format!("短すぎる（{} 語）", ids.len()));
    }
    Ok((name, ids))
}

/// 札を全部計算する（1 つずつ。ほかの依頼と同じ待ち行列に低い優先で入る）。同時に 2 つは走らせない
pub async fn run(st: Arc<AppState>) {
    let Some(d) = dir().cloned() else { return };
    if RUNNING.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return;
    }
    // エンジンの読み込みを待つ（最大 15 分）
    let t0 = Instant::now();
    while !matches!(st.backend.status(), Status::Ready) {
        if t0.elapsed() > Duration::from_secs(900) {
            st.log_json(json!({"ev": "prewarm_skip", "why": "engine not ready"}));
            RUNNING.store(false, std::sync::atomic::Ordering::SeqCst);
            return;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    let mut files: Vec<PathBuf> = match std::fs::read_dir(&d) {
        Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "json")).collect(),
        Err(e) => {
            st.log_json(json!({"ev": "prewarm_skip", "why": format!("{}: {e}", d.display())}));
            RUNNING.store(false, std::sync::atomic::Ordering::SeqCst);
            return;
        }
    };
    files.sort();
    let (mut ok, mut bad) = (0, 0);
    for f in files {
        let fname = f.file_name().map(|x| x.to_string_lossy().to_string()).unwrap_or_default();
        let doc: Json = match std::fs::read_to_string(&f).map_err(|e| e.to_string()).and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string())) {
            Ok(v) => v,
            Err(e) => {
                st.log_json(json!({"ev": "prewarm_failed", "file": fname, "error": e}));
                bad += 1;
                continue;
            }
        };
        let st2 = st.clone();
        let r = tokio::task::spawn_blocking(move || render(&st2, &doc)).await.map_err(|e| e.to_string()).and_then(|x| x);
        let (name, ids) = match r {
            Ok(x) => x,
            Err(e) => {
                st.log_json(json!({"ev": "prewarm_failed", "file": fname, "error": e}));
                bad += 1;
                continue;
            }
        };
        let n = ids.len();
        let t1 = Instant::now();
        let req = GenRequest {
            id: rand_id("prewarm-", 16),
            scope: CacheScope::fixed("_prewarm"),
            prompt: ids,
            max_tokens: 1,
            sampling: SamplingParams::greedy(),
            stop_ids: st.stop_ids.clone(),
            hints: PrefixHints { pin: Some(name.clone()), ..Default::default() },
            priority: Priority::Low,
            think_budget: None,
            media: vec![],
            constraint: None,
        };
        let res = match st.backend.submit(req) {
            Ok(mut g) => {
                let mut out = Err("no result".to_string());
                while let Some(ev) = g.recv().await {
                    match ev {
                        GenEvent::Done(FinishReason::Stop(_)) | GenEvent::Done(FinishReason::Length) => {
                            out = Ok(());
                            break;
                        }
                        GenEvent::Error(m) => {
                            out = Err(m);
                            break;
                        }
                        _ => {}
                    }
                }
                out
            }
            Err(e) => Err(format!("{e:?}")),
        };
        match res {
            Ok(()) => {
                ok += 1;
                st.log_json(json!({"ev": "prewarm", "name": name, "tokens": n, "ms": t1.elapsed().as_millis() as u64}));
            }
            Err(e) => {
                bad += 1;
                st.log_json(json!({"ev": "prewarm_failed", "name": name, "error": e}));
            }
        }
    }
    st.log_json(json!({"ev": "prewarm_done", "ok": ok, "failed": bad, "secs": t0.elapsed().as_secs()}));
    RUNNING.store(false, std::sync::atomic::Ordering::SeqCst);
}
