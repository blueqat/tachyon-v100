//! 試し用の中身（GPU は使わない）:
//!   - `FakeBackend`: 決まった語を流す Backend（待ち行列・同時の数・取り消し・文頭の当たり・文書も真似る）
//!   - `FakeEngine`: 決まった語に山のある logits を返す Engine（SimpleScheduler と CPU の sampler を通す）
//!   - `write_synthetic_tokenizer`: 小さな tokenizer.json（byte-level・merges なし・Qwen3.8 と同じ added token を 256 番から）
//! 何を出すかは、プロンプトの最後の user の回の文で決める（`plan`）:
//!   `[[raw]]<文>`   その文をそのまま出す（`</think>`・`<tool_call>`・`<|fim_pad|>` なども書ける）＋ <|im_end|>
//!   `[[echo]]`      user の文（印を除く）を答えにする
//!   `[[len:N]]`     同じ語を N 個（止める語なし。max_tokens で止まる）
//!   `[[delay:MS]]`  1 語ごとに MS ミリ秒待つ
//!   `[[error:N]]`   N 語の後に失敗する
//!   `[[stepfail]]`  FakeEngine の 1 歩が丸ごと失敗する（Engine::step が Err）
//!   なし            短い考え（考えありのとき）＋「Hello from the fake backend.」
use super::backend::*;
use super::engine::{Engine, EngineInfo, ItemOut, SeqId, StepItem};
use super::output::prompt_opens_think;
use super::sampler::SamplingParams;
use crate::tokenizer::Tokenizer;
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Semaphore};

pub const DEFAULT_ANSWER: &str = "Hello from the fake backend.";
pub const DEFAULT_THOUGHT: &str = "The user wrote something. I will answer briefly.";

/// 出す物の予定
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plan {
    pub tokens: Vec<u32>,
    pub delay: Option<Duration>,
    pub error_after: Option<usize>,
    pub step_fail: bool,
}

fn directive(user: &str, name: &str) -> Option<usize> {
    let key = format!("[[{name}:");
    let i = user.find(&key)? + key.len();
    let j = user[i..].find("]]")? + i;
    user[i..j].trim().parse().ok()
}

/// プロンプトの最後の user の回 → 出す語
pub fn plan(tok: &Tokenizer, prompt: &[u32]) -> Plan {
    let text = tok.decode(prompt, false);
    let thinking = prompt_opens_think(&text);
    let user = text.rfind("<|im_start|>user\n").map(|i| &text[i + "<|im_start|>user\n".len()..]).unwrap_or("");
    let user = user.split("<|im_end|>").next().unwrap_or("");
    let delay = directive(user, "delay").map(|ms| Duration::from_millis(ms as u64));
    let error_after = directive(user, "error");
    let eos = tok.ids.im_end;
    let enc = |s: &str| tok.encode(s).unwrap_or_default();
    let tokens = if let Some(i) = user.find("[[raw]]") {
        let mut t = enc(&user[i + "[[raw]]".len()..]);
        t.push(eos);
        t
    } else if let Some(n) = directive(user, "len") {
        let x = enc("x").first().copied().unwrap_or(0);
        vec![x; n]
    } else {
        let answer = if user.contains("[[echo]]") {
            let mut s = user.replace("[[echo]]", "");
            for d in ["delay", "error"] {
                while let Some(i) = s.find(&format!("[[{d}:")) {
                    let j = s[i..].find("]]").map(|j| i + j + 2).unwrap_or(s.len());
                    s.replace_range(i..j, "");
                }
            }
            s.trim().to_string()
        } else {
            DEFAULT_ANSWER.to_string()
        };
        let mut t = if thinking { enc(&format!("{DEFAULT_THOUGHT}\n</think>\n\n{answer}")) } else { enc(&answer) };
        t.push(eos);
        t
    };
    Plan { tokens, delay, error_after, step_fail: user.contains("[[stepfail]]") }
}

#[derive(Clone, Debug)]
pub struct FakeConfig {
    /// 1 語ごとの間（依頼で [[delay:MS]] を書けばそちら）
    pub delay: Duration,
    /// 最初の語の前の間（プレフィルの真似）
    pub first_delay: Duration,
    pub max_running: usize,
    pub max_waiting: usize,
    /// これだけ経つまで Loading
    pub load_time: Duration,
    /// 文書の先計算にかかる時間
    pub doc_time: Duration,
    /// 文頭の当たりの真似（同じ namespace の前の依頼との共通の頭を 64 語に切り下げて cached に）
    pub prefix_cache: bool,
}

impl Default for FakeConfig {
    fn default() -> Self {
        FakeConfig {
            delay: Duration::from_millis(1),
            first_delay: Duration::ZERO,
            max_running: 32,
            max_waiting: 64,
            load_time: Duration::ZERO,
            doc_time: Duration::ZERO,
            prefix_cache: true,
        }
    }
}

/// 受け取った依頼の記録（試しで中身を確かめる用。⚠️ FakeBackend だけ。本物は中身を残さない）
#[derive(Clone, Debug)]
pub struct FakeRecord {
    pub id: String,
    pub scope: CacheScope,
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    pub stop_ids: Vec<u32>,
    pub hints: PrefixHints,
    pub priority: Priority,
    pub cached: usize,
    /// 受け取り側が途中で drop した
    pub cancelled: Arc<AtomicBool>,
    /// 送った語の数
    pub sent: Arc<AtomicUsize>,
}

struct FakeDoc {
    scope: CacheScope,
    tokens: Vec<u32>,
    ready_at: Instant,
}

pub struct FakeBackend {
    tok: Arc<Tokenizer>,
    cfg: FakeConfig,
    sem: Arc<Semaphore>,
    waiting: Arc<AtomicUsize>,
    running: Arc<AtomicUsize>,
    /// 送った語の数（Stats.progress）
    progress: Arc<AtomicUsize>,
    created: Instant,
    status_override: Mutex<Option<Status>>,
    records: Mutex<Vec<FakeRecord>>,
    /// namespace → 前の依頼の語（新しい順に 64 本まで）
    cache: Mutex<HashMap<String, Vec<Vec<u32>>>>,
    docs: Mutex<HashMap<String, FakeDoc>>,
    /// shutdown が呼ばれた（走っている依頼は次の語で Error・新しい依頼は NotReady）
    stopped: Arc<AtomicBool>,
}

impl FakeBackend {
    pub fn new(tok: Arc<Tokenizer>, cfg: FakeConfig) -> Arc<Self> {
        Arc::new(FakeBackend {
            sem: Arc::new(Semaphore::new(cfg.max_running.max(1))),
            tok,
            cfg,
            waiting: Arc::new(AtomicUsize::new(0)),
            running: Arc::new(AtomicUsize::new(0)),
            progress: Arc::new(AtomicUsize::new(0)),
            created: Instant::now(),
            status_override: Mutex::new(None),
            records: Mutex::new(vec![]),
            cache: Mutex::new(HashMap::new()),
            docs: Mutex::new(HashMap::new()),
            stopped: Arc::new(AtomicBool::new(false)),
        })
    }

    /// 状態を決め打ちする（None で戻す）
    pub fn set_status(&self, s: Option<Status>) {
        *self.status_override.lock().unwrap() = s;
    }

    pub fn records(&self) -> Vec<FakeRecord> {
        self.records.lock().unwrap().clone()
    }

    pub fn last(&self) -> Option<FakeRecord> {
        self.records.lock().unwrap().last().cloned()
    }

    fn cached_for(&self, req: &GenRequest) -> usize {
        let mut best = 0;
        if let Some(d) = &req.hints.document {
            let docs = self.docs.lock().unwrap();
            if let Some(fd) = docs.get(&d.id) {
                // 文書は同じ namespace で、先頭が同じで、計算が終わっていれば使える
                if fd.scope.namespace == req.scope.namespace && req.prompt.starts_with(&fd.tokens) && Instant::now() >= fd.ready_at {
                    best = fd.tokens.len();
                }
            }
        }
        if self.cfg.prefix_cache {
            let mut c = self.cache.lock().unwrap();
            let v = c.entry(req.scope.namespace.clone()).or_default();
            for p in v.iter() {
                let lcp = p.iter().zip(&req.prompt).take_while(|(a, b)| a == b).count();
                // 頁（64 語）に切り下げ、最後の 1 語は必ず計算する
                best = best.max((lcp.min(req.prompt.len().saturating_sub(1)) / 64) * 64);
            }
            v.insert(0, req.prompt.clone());
            v.truncate(64);
        }
        best
    }
}

impl Backend for FakeBackend {
    fn status(&self) -> Status {
        if let Some(s) = self.status_override.lock().unwrap().clone() {
            return s;
        }
        let el = self.created.elapsed();
        if el < self.cfg.load_time {
            return Status::Loading { progress: el.as_secs_f32() / self.cfg.load_time.as_secs_f32(), what: "fake weights".into() };
        }
        Status::Ready
    }

    fn shutdown(&self) {
        self.stopped.store(true, Ordering::SeqCst);
    }

    fn submit(&self, req: GenRequest) -> Result<GenStream, SubmitError> {
        if self.status() != Status::Ready || self.stopped.load(Ordering::SeqCst) {
            return Err(SubmitError::NotReady);
        }
        // 待ち = まだ走っていない依頼。走る枠が空いていれば待ちには数えない
        let busy = self.running.load(Ordering::SeqCst) + self.waiting.load(Ordering::SeqCst);
        if busy >= self.cfg.max_running + self.cfg.max_waiting {
            return Err(SubmitError::QueueFull { retry_after: 1 });
        }
        let cached = self.cached_for(&req);
        let pl = plan(&self.tok, &req.prompt);
        let rec = FakeRecord {
            id: req.id.clone(),
            scope: req.scope.clone(),
            prompt: req.prompt.clone(),
            max_tokens: req.max_tokens,
            sampling: req.sampling.clone(),
            stop_ids: req.stop_ids.clone(),
            hints: req.hints.clone(),
            priority: req.priority,
            cached,
            cancelled: Arc::new(AtomicBool::new(false)),
            sent: Arc::new(AtomicUsize::new(0)),
        };
        let (cancelled, sent, progress, stopped) = (rec.cancelled.clone(), rec.sent.clone(), self.progress.clone(), self.stopped.clone());
        self.records.lock().unwrap().push(rec);
        let (tx, rx) = mpsc::unbounded_channel();
        let (sem, waiting, running) = (self.sem.clone(), self.waiting.clone(), self.running.clone());
        let (delay, first) = (pl.delay.unwrap_or(self.cfg.delay), self.cfg.first_delay);
        waiting.fetch_add(1, Ordering::SeqCst);
        tokio::spawn(async move {
            let permit = tokio::select! {
                p = sem.acquire_owned() => p,
                _ = tx.closed() => {
                    waiting.fetch_sub(1, Ordering::SeqCst);
                    cancelled.store(true, Ordering::SeqCst);
                    return;
                }
            };
            waiting.fetch_sub(1, Ordering::SeqCst);
            running.fetch_add(1, Ordering::SeqCst);
            let _permit = permit;
            // 送れない（受け取り側が drop した）・間の途中で切られた → 取り消し
            let ok = async {
                tx.send(GenEvent::Started { cached }).ok()?;
                if !first.is_zero() {
                    tokio::select! { _ = tokio::time::sleep(first) => {}, _ = tx.closed() => return None }
                }
                for (i, &t) in pl.tokens.iter().enumerate() {
                    if stopped.load(Ordering::SeqCst) {
                        tx.send(GenEvent::Error("the server is shutting down".into())).ok()?;
                        return Some(());
                    }
                    if pl.error_after == Some(i) {
                        tx.send(GenEvent::Error("fake engine failure".into())).ok()?;
                        return Some(());
                    }
                    if i >= req.max_tokens {
                        tx.send(GenEvent::Done(FinishReason::Length)).ok()?;
                        return Some(());
                    }
                    if !delay.is_zero() {
                        tokio::select! { _ = tokio::time::sleep(delay) => {}, _ = tx.closed() => return None }
                    }
                    if req.stop_ids.contains(&t) {
                        tx.send(GenEvent::Done(FinishReason::Stop(t))).ok()?;
                        return Some(());
                    }
                    tx.send(GenEvent::Tokens(vec![t])).ok()?;
                    sent.fetch_add(1, Ordering::SeqCst);
                    progress.fetch_add(1, Ordering::SeqCst);
                }
                // 予定を出し切った（[[len:N]] で max_tokens より短い）
                tx.send(GenEvent::Done(FinishReason::Length)).ok()?;
                Some(())
            }
            .await;
            if ok.is_none() {
                cancelled.store(true, Ordering::SeqCst);
            }
            running.fetch_sub(1, Ordering::SeqCst);
        });
        Ok(rx)
    }

    fn stats(&self) -> Stats {
        Stats {
            waiting: self.waiting.load(Ordering::SeqCst),
            running: self.running.load(Ordering::SeqCst),
            max_running: self.cfg.max_running,
            max_waiting: self.cfg.max_waiting,
            progress: self.progress.load(Ordering::SeqCst) as u64,
            extra: vec![("fake_records".into(), self.records.lock().unwrap().len() as f64)],
        }
    }

    fn doc_create(&self, job: DocJob) -> Result<DocState, SubmitError> {
        if self.status() != Status::Ready {
            return Err(SubmitError::NotReady);
        }
        let ready_at = Instant::now() + self.cfg.doc_time;
        let st = if self.cfg.doc_time.is_zero() { DocState::Ready { tier: "ram".into() } } else { DocState::Queued };
        self.docs.lock().unwrap().insert(job.id, FakeDoc { scope: job.scope, tokens: job.tokens, ready_at });
        Ok(st)
    }

    fn doc_state(&self, id: &str) -> Option<DocState> {
        let docs = self.docs.lock().unwrap();
        let d = docs.get(id)?;
        let now = Instant::now();
        Some(if now >= d.ready_at {
            DocState::Ready { tier: "ram".into() }
        } else {
            let left = d.ready_at - now;
            let frac = 1.0 - left.as_secs_f64() / self.cfg.doc_time.as_secs_f64().max(1e-9);
            DocState::Computing { done: (d.tokens.len() as f64 * frac) as usize, total: d.tokens.len() }
        })
    }

    fn doc_delete(&self, id: &str) {
        self.docs.lock().unwrap().remove(id);
    }
}

/// FakeEngine の記録（まとめた数・release した列）
#[derive(Clone, Debug, Default)]
pub struct FakeEngineLog {
    pub steps: usize,
    pub max_batch: usize,
    pub max_step_tokens: usize,
    pub released: Vec<SeqId>,
}

struct FakeSeq {
    toks: Vec<u32>,
    /// (入力の長さ, 予定)
    plan: Option<(usize, Plan)>,
}

/// 決まった語に山のある logits（10.0、ほかは 0）を返す。return_tokens なら語をそのまま返す
pub struct FakeEngine {
    tok: Arc<Tokenizer>,
    info: EngineInfo,
    n_vocab: usize,
    return_tokens: bool,
    /// return_tokens のとき、1 歩に何語まで返すか（下書きを受け取るエンジンの真似。ItemOut::Tokens）
    pub multi: usize,
    /// 頭の使い回しの真似（Engine::reuse）: 同じ scope の前の依頼のプロンプトが頭にあれば、その長さを使い回す
    pub reuse: bool,
    last: Option<(String, Vec<u32>)>,
    seqs: HashMap<SeqId, FakeSeq>,
    pub log: Arc<Mutex<FakeEngineLog>>,
}

impl FakeEngine {
    pub fn new(tok: Arc<Tokenizer>, info: EngineInfo, return_tokens: bool) -> Self {
        let n_vocab = tok.vocab_size();
        FakeEngine { tok, info, n_vocab, return_tokens, multi: 1, reuse: false, last: None, seqs: HashMap::new(), log: Arc::new(Mutex::new(FakeEngineLog::default())) }
    }
}

impl Engine for FakeEngine {
    fn info(&self) -> EngineInfo {
        self.info.clone()
    }

    fn step(&mut self, items: &[StepItem]) -> Result<Vec<ItemOut>, String> {
        let mut out = Vec::with_capacity(items.len());
        let mut delay = Duration::ZERO;
        for it in items {
            let s = self.seqs.entry(it.seq).or_insert(FakeSeq { toks: vec![], plan: None });
            if s.toks.len() != it.pos {
                return Err(format!("列 {} の位置が合わない（持っている {}・来た {}）", it.seq, s.toks.len(), it.pos));
            }
            s.toks.extend_from_slice(&it.tokens);
            if !it.want {
                out.push(ItemOut::None);
                continue;
            }
            if s.plan.is_none() {
                s.plan = Some((s.toks.len(), plan(&self.tok, &s.toks)));
            }
            let (pl, p) = s.plan.as_ref().unwrap();
            if p.step_fail {
                return Err("fake step failure".into());
            }
            let k = s.toks.len() - pl;
            if p.error_after == Some(k) {
                out.push(ItemOut::Error("fake engine failure".into()));
                continue;
            }
            delay = delay.max(p.delay.unwrap_or(Duration::ZERO));
            let next = p.tokens.get(k).copied().unwrap_or_else(|| self.tok.ids.im_end);
            if self.return_tokens && self.multi > 1 {
                // 予定の続きを multi 語まで（終わりの語で止める）。最後の語の手前までは計算済みとして持つ
                let mut v = vec![next];
                // 失敗の予定（error_after）の手前で止める（次の歩でその位置に来る）
                while v.len() < self.multi && *v.last().unwrap() != self.tok.ids.im_end && p.error_after != Some(k + v.len()) {
                    v.push(p.tokens.get(k + v.len()).copied().unwrap_or(self.tok.ids.im_end));
                }
                let n = v.len();
                s.toks.extend_from_slice(&v[..n - 1]);
                out.push(ItemOut::Tokens(v));
                continue;
            }
            out.push(if self.return_tokens {
                ItemOut::Token(next)
            } else {
                let mut l = vec![0f32; self.n_vocab];
                l[next as usize] = 10.0;
                ItemOut::Logits(l)
            });
        }
        let mut g = self.log.lock().unwrap();
        g.steps += 1;
        g.max_batch = g.max_batch.max(items.len());
        g.max_step_tokens = g.max_step_tokens.max(items.iter().map(|i| i.tokens.len()).sum());
        drop(g);
        if !delay.is_zero() {
            std::thread::sleep(delay);
        }
        Ok(out)
    }

    fn release(&mut self, seq: SeqId) {
        self.seqs.remove(&seq);
        self.log.lock().unwrap().released.push(seq);
    }

    fn reuse(&mut self, seq: SeqId, scope: &str, prompt: &[u32], snap: Option<usize>) -> usize {
        if !self.reuse {
            return 0;
        }
        let l = match &self.last {
            Some((sc, p)) if sc == scope && p.len() < prompt.len() && prompt.starts_with(p) => p.len(),
            _ => 0,
        };
        let at = snap.filter(|&s| s > 0 && s < prompt.len()).unwrap_or(prompt.len());
        self.last = Some((scope.to_string(), prompt[..at].to_vec()));
        if l > 0 {
            self.seqs.insert(seq, FakeSeq { toks: prompt[..l].to_vec(), plan: None });
        }
        l
    }
}

/// Qwen3.8 の added token（tokenizer.json の 248044〜248076 と同じ並び・同じ special）
pub const QWEN38_ADDED: &[(&str, bool)] = &[
    ("<|endoftext|>", true),
    ("<|im_start|>", true),
    ("<|im_end|>", true),
    ("<|object_ref_start|>", true),
    ("<|object_ref_end|>", true),
    ("<|box_start|>", true),
    ("<|box_end|>", true),
    ("<|quad_start|>", true),
    ("<|quad_end|>", true),
    ("<|vision_start|>", true),
    ("<|vision_end|>", true),
    ("<|vision_pad|>", true),
    ("<|image_pad|>", true),
    ("<|video_pad|>", true),
    ("<tool_call>", false),
    ("</tool_call>", false),
    ("<|fim_prefix|>", false),
    ("<|fim_middle|>", false),
    ("<|fim_suffix|>", false),
    ("<|fim_pad|>", false),
    ("<|repo_name|>", false),
    ("<|file_sep|>", false),
    ("<tool_response>", false),
    ("</tool_response>", false),
    ("<think>", false),
    ("</think>", false),
    ("<|audio_start|>", true),
    ("<|audio_end|>", true),
    ("<tts_pad>", true),
    ("<tts_text_bos>", true),
    ("<tts_text_eod>", true),
    ("<tts_text_bos_single>", true),
    ("<|audio_pad|>", true),
];

/// GPT-2 の bytes_to_unicode（バイト → 文字）
fn bytes_to_unicode() -> Vec<char> {
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
    let mut out = vec!['\0'; 256];
    for (b, c) in bs.iter().zip(&cs) {
        out[*b as usize] = char::from_u32(*c).unwrap();
    }
    out
}

/// 試し用の小さな tokenizer.json と generation_config.json を dir に書く（1 バイト = 1 語・added token は 256 番から）。
/// ⚠️ 語の番号は本物と違う（<|im_end|> = 258）。口の中は番号を必ず tokenizer から引くので、それを確かめるのにも使う
pub fn write_synthetic_tokenizer(dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let b2u = bytes_to_unicode();
    let mut vocab = serde_json::Map::new();
    for (b, c) in b2u.iter().enumerate() {
        vocab.insert(c.to_string(), json!(b));
    }
    let added: Vec<serde_json::Value> = QWEN38_ADDED
        .iter()
        .enumerate()
        .map(|(i, (s, sp))| json!({"id": 256 + i, "content": s, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": sp}))
        .collect();
    let pre = r#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#;
    let j = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": pre}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}]},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "", "end_of_word_suffix": "",
                  "fuse_unk": false, "byte_fallback": false, "ignore_merges": false, "vocab": vocab, "merges": []},
    });
    std::fs::write(dir.join("tokenizer.json"), serde_json::to_string(&j)?)?;
    let id = |s: &str| 256 + QWEN38_ADDED.iter().position(|(t, _)| *t == s).unwrap();
    std::fs::write(dir.join("generation_config.json"), json!({"eos_token_id": [id("<|im_end|>"), id("<|endoftext|>")]}).to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_tokenizer_roundtrip() {
        let dir = std::env::temp_dir().join(format!("tachyon-synth-{}", std::process::id()));
        write_synthetic_tokenizer(&dir).unwrap();
        let tok = Tokenizer::from_dir(&dir).unwrap();
        assert_eq!(tok.vocab_size(), 256 + QWEN38_ADDED.len());
        assert_eq!(tok.ids.im_end, 258);
        assert_eq!(tok.eos, vec![258, 256]);
        let s = "<|im_start|>user\n日本語 and </think> <tool_call><|im_end|>";
        let ids = tok.encode(s).unwrap();
        assert_eq!(tok.decode(&ids, false), s);
        assert!(ids.contains(&tok.ids.think_end) && ids.contains(&tok.ids.tool_call) && ids[0] == tok.ids.im_start);
        let p = plan(&tok, &tok.encode("<|im_start|>user\n[[raw]]a</think>b<|im_end|>\n<|im_start|>assistant\n<think>\n").unwrap());
        assert_eq!(tok.decode(&p.tokens, false), "a</think>b<|im_end|>");
        let p = plan(&tok, &tok.encode("<|im_start|>user\n[[len:5]][[delay:3]]<|im_end|>\n").unwrap());
        assert_eq!((p.tokens.len(), p.delay), (5, Some(Duration::from_millis(3))));
        let p = plan(&tok, &tok.encode("<|im_start|>user\n[[echo]] hi [[delay:1]]<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n").unwrap());
        assert_eq!(tok.decode(&p.tokens, false), "hi<|im_end|>");
        std::fs::remove_dir_all(&dir).ok();
    }
}
