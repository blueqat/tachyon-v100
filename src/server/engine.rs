//! エンジン（1 歩の計算）との境目 `Engine` と、それを回す素朴な順番決め `SimpleScheduler`（Backend の実装）。
//!
//! M1 のモデル（src/model.rs）はこの trait を実装すれば口に載る:
//!   - `step`: 列ごとに「今回足す語」（プレフィルの塊か、デコードの 1 語）を受け取り、`want` の列は最後の位置の次の語の
//!     logits（CPU の sampler が選ぶ）か、エンジンが選んだ語を返す。列の状態（KV・GDN）はエンジンが列の番号で持つ
//!   - `release`: 列が終わった・取り消された（状態と KV を空ける）
//! SimpleScheduler がやること: 待ち行列（上限を超えたら 429）・継続まとめ（走っている人のデコード 1 語＋残りの予算でプレフィルの塊）・
//! 止める語・max_tokens・エンジンの文脈の上限・取り消し（受け取り側が drop したら release）。
//! ⚠️ 文頭の使い回し・頁つき KV・追い出しは無い（本物の順番決め sched.rs で Backend を直に実装して置き換える）
use super::backend::{Backend, FinishReason, GenEvent, GenRequest, GenStream, Priority, Stats, Status, SubmitError, ThinkBudget};
use super::sampler::{self, History, SamplingParams};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc as smpsc, Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;

/// デコードの人がいるときに、プロンプトの最後に残す塊の語数
const FINAL_TAIL: usize = 32;

/// 列（1 つの依頼）の番号。順番決めが振る
pub type SeqId = u64;

#[derive(Clone, Debug)]
pub struct EngineInfo {
    /// 同時に持てる列の数（GDN の状態の升の数）
    pub max_seqs: usize,
    /// 1 列の語の上限（KV の大きさ）
    pub max_ctx: usize,
    /// 1 歩で流せる語の数の上限（作業場の大きさ）
    pub max_step_tokens: usize,
}

/// 1 歩の中の 1 列
#[derive(Clone, Debug)]
pub struct StepItem {
    pub seq: SeqId,
    /// この列で計算済みの語の数（tokens の最初の語の位置）。0 なら新しい列
    pub pos: usize,
    /// 今回足す語（プレフィルの塊か、デコードの 1 語）
    pub tokens: Vec<u32>,
    /// 最後の語の次の語を返すか（プレフィルの途中の塊は false）
    pub want: bool,
    /// エンジンが自分で選ぶときの設定（logits を返すなら見なくてよい）
    pub sampling: Arc<SamplingParams>,
    /// 罰の計算に使う、これまでの語の数え（罰があるデコードの語のときだけ。下書きを自分で受け取るエンジンが使う）
    pub hist: Option<History>,
    /// response_format の JSON の形（この語の後に出す語を絞る。want の列だけ）。エンジンは候補・logits にかけ、下書きは受け取るたびに進める
    pub constraint: Option<Arc<super::jsonc::Matcher>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ItemOut {
    /// want = false の列
    None,
    /// 次の語の logits（長さは語彙。SimpleScheduler の CPU の sampler が選ぶ）
    Logits(Vec<f32>),
    /// エンジンが選んだ語
    Token(u32),
    /// エンジンが選んだ何語か（下書きの受け取り: 受け取った下書き＋新しい 1 語）。エンジンは最後の語の手前まで計算済み
    /// （次の歩の入力は最後の語、位置は pos + tokens.len() + 返した数 − 1）
    Tokens(Vec<u32>),
    /// この列だけ失敗（ほかの列は続ける）
    Error(String),
}

pub trait Engine: Send {
    fn info(&self) -> EngineInfo;
    /// 1 歩。返す Vec は items と同じ長さ・同じ順。Err は全部の列の失敗（走っている依頼は全部 Error で終わる）
    fn step(&mut self, items: &[StepItem]) -> Result<Vec<ItemOut>, String>;
    fn release(&mut self, seq: SeqId);
    /// 新しい列を始める前に: 同じ scope（アカウント）の前の依頼の頭を使い回せるなら、使い回した語の数（< prompt.len()）。
    /// 返した数の位置から流す（エンジンはその位置まで計算済みの状態にしておく）。既定は使い回さない
    /// 受け入れの前に: この列（プロンプト prompt_len 語・出力の上限まで合わせて max_total 語）を今受けられるか。
    /// true なら列の場所と KV の頁を予約する（false なら待たせる）。既定は受ける
    fn admit(&mut self, _seq: SeqId, _prompt_len: usize, _max_total: usize) -> bool {
        true
    }
    /// この列のプレフィルの塊がまだ流れているか（true なら順番決めはこの列の次の塊を待つ。ほかの人のデコードは進める）
    fn busy(&mut self, _seq: SeqId) -> bool {
        false
    }
    /// snap: この列で控えを取る位置（最後の回の頭。None ならプロンプトの終わり）
    fn reuse(&mut self, _seq: SeqId, _scope: &str, _prompt: &[u32], _snap: Option<usize>) -> usize {
        0
    }
    /// reuse と同じ（頭の印を全部渡す。RAM の段・先計算の札はこちらを見る）。既定は reuse(turn_start)
    fn reuse_hints(&mut self, seq: SeqId, scope: &str, prompt: &[u32], hints: &super::backend::PrefixHints) -> usize {
        self.reuse(seq, scope, prompt, hints.turn_start)
    }
    /// 受け入れた列（admit の後・最初の step の前）に画像を付ける（符号化して prompt の pad の所に置く）。
    /// 画像のある列は reuse しない（頭の控えに入れない）。既定は画像を受けない
    fn attach_media(&mut self, _seq: SeqId, media: &[Arc<super::backend::MediaInput>]) -> Result<(), String> {
        if media.is_empty() {
            Ok(())
        } else {
            Err("this engine does not take images".into())
        }
    }
}

impl Engine for Box<dyn Engine> {
    fn info(&self) -> EngineInfo {
        (**self).info()
    }
    fn step(&mut self, items: &[StepItem]) -> Result<Vec<ItemOut>, String> {
        (**self).step(items)
    }
    fn release(&mut self, seq: SeqId) {
        (**self).release(seq)
    }
    fn reuse(&mut self, seq: SeqId, scope: &str, prompt: &[u32], snap: Option<usize>) -> usize {
        (**self).reuse(seq, scope, prompt, snap)
    }
    fn reuse_hints(&mut self, seq: SeqId, scope: &str, prompt: &[u32], hints: &super::backend::PrefixHints) -> usize {
        (**self).reuse_hints(seq, scope, prompt, hints)
    }
    fn admit(&mut self, seq: SeqId, prompt_len: usize, max_total: usize) -> bool {
        (**self).admit(seq, prompt_len, max_total)
    }
    fn busy(&mut self, seq: SeqId) -> bool {
        (**self).busy(seq)
    }
    fn attach_media(&mut self, seq: SeqId, media: &[Arc<super::backend::MediaInput>]) -> Result<(), String> {
        (**self).attach_media(seq, media)
    }
}

/// 重みを読んでエンジンを作る（順番決めの糸の中で呼ぶ。読み込み中は /health が 503 loading）。
/// 引数は進み具合を知らせる関数（0〜1・何をしているか）
pub type EngineLoader = Box<dyn FnOnce(&dyn Fn(f32, &str)) -> Result<Box<dyn Engine>, String> + Send>;

#[derive(Clone, Debug)]
pub struct SchedConfig {
    /// 同時に走らせる列（エンジンの max_seqs とこの小さい方）
    pub max_running: usize,
    /// 待ち行列の上限（超えたら 429。docs/API.md 3.5 の例 64）
    pub max_waiting: usize,
    /// 1 歩の語の予算（デコード＋プレフィルの塊。design-27b 5 章: 512）
    pub step_tokens: usize,
    /// デコードの人がいるときのプレフィルの塊（256）
    pub prefill_chunk: usize,
    /// デコードの人がいないときの塊（2048。最初の 1 語を早く）
    pub prefill_chunk_alone: usize,
    /// 選んでよい語の番号の上限（tokenizer の語の数 248,077。GGUF の余りの [PAD…] は選ばない）
    pub valid_vocab: usize,
}

impl Default for SchedConfig {
    fn default() -> Self {
        SchedConfig { max_running: 32, max_waiting: 64, step_tokens: 512, prefill_chunk: 256, prefill_chunk_alone: 2048, valid_vocab: 0 }
    }
}

enum Cmd {
    Submit(GenRequest, mpsc::UnboundedSender<GenEvent>),
    Stop,
}

struct Shared {
    status: Mutex<Status>,
    waiting: AtomicUsize,
    running: AtomicUsize,
    max_running: AtomicUsize,
    max_ctx: AtomicUsize,
    max_waiting: usize,
    steps: AtomicU64,
    /// shutdown が呼ばれた（新しい依頼を受けない）
    stopping: AtomicBool,
    step_tokens: AtomicU64,
}

pub struct SimpleScheduler {
    tx: Mutex<smpsc::Sender<Cmd>>,
    sh: Arc<Shared>,
}

impl SimpleScheduler {
    /// 順番決めの糸を起こす（loader はその糸の中で重みを読む）
    pub fn spawn(loader: EngineLoader, cfg: SchedConfig) -> Arc<Self> {
        let (tx, rx) = smpsc::channel();
        let sh = Arc::new(Shared {
            status: Mutex::new(Status::Loading { progress: 0.0, what: "starting".into() }),
            waiting: AtomicUsize::new(0),
            running: AtomicUsize::new(0),
            max_running: AtomicUsize::new(cfg.max_running),
            max_ctx: AtomicUsize::new(0),
            max_waiting: cfg.max_waiting,
            steps: AtomicU64::new(0),
            stopping: AtomicBool::new(false),
            step_tokens: AtomicU64::new(0),
        });
        let sh2 = sh.clone();
        std::thread::Builder::new().name("tachyon-sched".into()).spawn(move || run(loader, cfg, rx, sh2)).expect("順番決めの糸を起こせない");
        Arc::new(SimpleScheduler { tx: Mutex::new(tx), sh })
    }
}

impl Drop for SimpleScheduler {
    fn drop(&mut self) {
        if let Ok(tx) = self.tx.lock() {
            let _ = tx.send(Cmd::Stop);
        }
    }
}

impl Backend for SimpleScheduler {
    fn status(&self) -> Status {
        self.sh.status.lock().unwrap().clone()
    }

    fn submit(&self, req: GenRequest) -> Result<GenStream, SubmitError> {
        if self.status() != Status::Ready || self.sh.stopping.load(Ordering::SeqCst) {
            return Err(SubmitError::NotReady);
        }
        let max_ctx = self.sh.max_ctx.load(Ordering::SeqCst);
        if max_ctx > 0 && req.prompt.len() >= max_ctx {
            return Err(SubmitError::TooLong { max: max_ctx });
        }
        if self.sh.waiting.fetch_add(1, Ordering::SeqCst) >= self.sh.max_waiting {
            self.sh.waiting.fetch_sub(1, Ordering::SeqCst);
            return Err(SubmitError::QueueFull { retry_after: 1 });
        }
        let (tx, rx) = mpsc::unbounded_channel();
        if self.tx.lock().unwrap().send(Cmd::Submit(req, tx)).is_err() {
            self.sh.waiting.fetch_sub(1, Ordering::SeqCst);
            return Err(SubmitError::Internal("the scheduler thread has stopped".into()));
        }
        Ok(rx)
    }

    fn stats(&self) -> Stats {
        Stats {
            waiting: self.sh.waiting.load(Ordering::SeqCst),
            running: self.sh.running.load(Ordering::SeqCst),
            max_running: self.sh.max_running.load(Ordering::SeqCst),
            max_waiting: self.sh.max_waiting,
            progress: self.sh.step_tokens.load(Ordering::Relaxed),
            extra: vec![("steps".into(), self.sh.steps.load(Ordering::Relaxed) as f64), ("step_tokens".into(), self.sh.step_tokens.load(Ordering::Relaxed) as f64)],
        }
    }

    fn max_context(&self) -> Option<usize> {
        match self.sh.max_ctx.load(Ordering::SeqCst) {
            0 => None,
            n => Some(n),
        }
    }

    /// 糸に止めを送る（今の歩が終わった所で、走っている依頼・待っている依頼を全部 Error で終えて糸が終わる）
    fn shutdown(&self) {
        self.sh.stopping.store(true, Ordering::SeqCst);
        if let Ok(tx) = self.tx.lock() {
            let _ = tx.send(Cmd::Stop);
        }
    }
}

struct Job {
    req: GenRequest,
    tx: mpsc::UnboundedSender<GenEvent>,
}

struct Seq {
    id: SeqId,
    prompt: Vec<u32>,
    /// 計算済みの語の数
    pos: usize,
    /// 最後に出した語（次の歩の入力）
    last: u32,
    n_gen: usize,
    max_tokens: usize,
    stop_ids: Vec<u32>,
    sampling: Arc<SamplingParams>,
    hist: History,
    tx: mpsc::UnboundedSender<GenEvent>,
    /// 考えの上限（考えている間だけ Some）と、考えの語の数
    think: Option<ThinkBudget>,
    n_think: usize,
    /// 次の歩で差し込む語（考えを閉じる文）。出力にも出す
    forced: Vec<u32>,
    /// response_format の JSON の形（出した語で進める）
    cons: Option<super::jsonc::Matcher>,
}

impl Seq {
    /// 次の歩の絞り: 差し込む語（考えを閉じる文）があれば、それを流した後の状態
    fn cons_after_forced(&self) -> Option<Arc<super::jsonc::Matcher>> {
        let mut m = self.cons.clone()?;
        for &t in &self.forced {
            m.advance(t);
        }
        Some(Arc::new(m))
    }

    fn prefilled(&self) -> bool {
        self.pos >= self.prompt.len()
    }
}

fn run(loader: EngineLoader, cfg: SchedConfig, rx: smpsc::Receiver<Cmd>, sh: Arc<Shared>) {
    let progress = |p: f32, what: &str| *sh.status.lock().unwrap() = Status::Loading { progress: p, what: what.to_string() };
    let mut eng = match loader(&progress) {
        Ok(e) => e,
        Err(m) => {
            // 読み込みの失敗（--max-vram-gib を超えた等）は /health 500 のまま口が生き続ける＝systemd は「動いている」と見て起こし直さない
            // （11-13 に --max-vram-gib 24 の 27B が 7 分 /health を待たせたのはこれ）。TACHYON_EXIT_ON_LOAD_FAIL=1 なら 71 で終わる＝Restart=on-failure が拾う
            eprintln!("{}", serde_json::json!({"ev": "engine_load_failed", "error": m}));
            if std::env::var("TACHYON_EXIT_ON_LOAD_FAIL").is_ok_and(|v| v == "1") {
                super::util::hard_exit(71);
            }
            *sh.status.lock().unwrap() = Status::Failed(m);
            return;
        }
    };
    let info = eng.info();
    let max_running = cfg.max_running.min(info.max_seqs).max(1);
    let step_budget = cfg.step_tokens.min(info.max_step_tokens).max(1);
    sh.max_running.store(max_running, Ordering::SeqCst);
    sh.max_ctx.store(info.max_ctx, Ordering::SeqCst);
    *sh.status.lock().unwrap() = Status::Ready;

    let mut waiting: VecDeque<Job> = VecDeque::new();
    let mut running: Vec<Seq> = vec![];
    let mut next_id: SeqId = 1;
    // 歩が続けて失敗したら（CUDA の誤りは戻らないことが多い）Failed にして /health を 500 に＝見張りが起こし直す
    let mut fails = 0;
    let fail_all = |running: &mut Vec<Seq>, eng: &mut Box<dyn Engine>, m: &str| {
        for s in running.drain(..) {
            let _ = s.tx.send(GenEvent::Error(m.to_string()));
            eng.release(s.id);
            sh.running.fetch_sub(1, Ordering::SeqCst);
        }
    };
    loop {
        // 1. 取り込み（何も無ければ待つ）
        if running.is_empty() && waiting.is_empty() {
            match rx.recv() {
                Ok(Cmd::Submit(req, tx)) => waiting.push_back(Job { req, tx }),
                Ok(Cmd::Stop) | Err(_) => return,
            }
        }
        loop {
            match rx.try_recv() {
                Ok(Cmd::Submit(req, tx)) => waiting.push_back(Job { req, tx }),
                Ok(Cmd::Stop) => {
                    fail_all(&mut running, &mut eng, "the server is shutting down");
                    for j in waiting.drain(..) {
                        let _ = j.tx.send(GenEvent::Error("the server is shutting down".into()));
                        sh.waiting.fetch_sub(1, Ordering::SeqCst);
                    }
                    // 後から来た依頼（shutdown の前に submit を通った物）も断る
                    while let Ok(c) = rx.try_recv() {
                        if let Cmd::Submit(_, tx) = c {
                            let _ = tx.send(GenEvent::Error("the server is shutting down".into()));
                            sh.waiting.fetch_sub(1, Ordering::SeqCst);
                        }
                    }
                    return;
                }
                Err(smpsc::TryRecvError::Empty) => break,
                Err(smpsc::TryRecvError::Disconnected) => {
                    if running.is_empty() && waiting.is_empty() {
                        return;
                    }
                    break;
                }
            }
        }
        // 2. 取り消された依頼を外す（受け取り側が drop した）
        waiting.retain(|j| {
            let keep = !j.tx.is_closed();
            if !keep {
                sh.waiting.fetch_sub(1, Ordering::SeqCst);
            }
            keep
        });
        running.retain(|s| {
            let keep = !s.tx.is_closed();
            if !keep {
                eng.release(s.id);
                sh.running.fetch_sub(1, Ordering::SeqCst);
            }
            keep
        });
        // 3. 受け入れ（普通の依頼が先、low は普通の依頼が待っていないときだけ）
        while running.len() < max_running {
            let Some(i) = waiting.iter().position(|j| j.req.priority == Priority::Normal).or(if waiting.is_empty() { None } else { Some(0) }) else { break };
            if waiting[i].req.prompt.is_empty() || waiting[i].req.prompt.len() >= info.max_ctx {
                let j = waiting.remove(i).unwrap();
                sh.waiting.fetch_sub(1, Ordering::SeqCst);
                let _ = j.tx.send(GenEvent::Error(format!("the prompt must be 1..{} tokens", info.max_ctx)));
                continue;
            }
            // エンジンが受けられるか（列の場所・KV の頁）。受けられなければ、走っている人が終わるまで待つ（順番は変えない）
            {
                let r = &waiting[i].req;
                if !eng.admit(next_id, r.prompt.len(), r.prompt.len() + r.max_tokens.max(1)) {
                    if running.is_empty() {
                        // 誰も走っていないのに受けられない＝この依頼は入らない
                        let j = waiting.remove(i).unwrap();
                        sh.waiting.fetch_sub(1, Ordering::SeqCst);
                        let _ = j.tx.send(GenEvent::Error("the request does not fit in the engine's KV cache".into()));
                        continue;
                    }
                    break;
                }
            }
            let j = waiting.remove(i).unwrap();
            sh.waiting.fetch_sub(1, Ordering::SeqCst);
            // 画像のある依頼: 符号化して列に付ける（頭の使い回しはしない）
            let cached = if !j.req.media.is_empty() {
                if let Err(e) = eng.attach_media(next_id, &j.req.media) {
                    eng.release(next_id);
                    let _ = j.tx.send(GenEvent::Error(format!("image: {e}")));
                    next_id += 1;
                    continue;
                }
                0
            } else {
                // 頭の使い回しは持ち主ごと（scope.owner＝アカウント、--prefix-owner-accounts のアカウントは利用者ごと。社内どうしでも分ける）
                eng.reuse_hints(next_id, &j.req.scope.owner, &j.req.prompt, &j.req.hints).min(j.req.prompt.len() - 1)
            };
            if j.tx.send(GenEvent::Started { cached }).is_err() {
                eng.release(next_id);
                continue;
            }
            let sampling = Arc::new(j.req.sampling.clone());
            let think = j.req.think_budget.clone();
            running.push(Seq {
                cons: j.req.constraint.clone(),
                think,
                n_think: 0,
                forced: vec![],
                id: next_id,
                hist: History::new(&j.req.prompt, &sampling),
                prompt: j.req.prompt,
                pos: cached,
                last: 0,
                n_gen: 0,
                max_tokens: j.req.max_tokens.max(1),
                stop_ids: j.req.stop_ids,
                sampling,
                tx: j.tx,
            });
            next_id += 1;
            sh.running.fetch_add(1, Ordering::SeqCst);
        }
        if running.is_empty() {
            continue;
        }
        // 4. 1 歩を組む: デコードの人は 1 語ずつ（飢えさせない）、残りの予算でプレフィルの塊
        let n_dec = running.iter().filter(|s| s.prefilled()).count();
        let chunk = if n_dec > 0 { cfg.prefill_chunk } else { cfg.prefill_chunk_alone }.max(1);
        // デコードの人が予算を使い切っても、プレフィルの人を飢えさせない（1 歩に少なくとも 64 語）
        let mut budget = step_budget.saturating_sub(n_dec);
        if n_dec < running.len() {
            budget = budget.max(chunk.min(64));
        }
        let mut items = vec![];
        let mut idx = vec![];
        for (k, s) in running.iter().enumerate() {
            if s.prefilled() {
                let hist = if s.sampling.has_penalty() { Some(s.hist.clone()) } else { None };
                // 考えの上限: 前の語に続けて、考えを閉じる文を 1 歩で流す（エンジンにはプレフィルの続きと同じ）
                let mut tokens = vec![s.last];
                tokens.extend_from_slice(&s.forced);
                items.push(StepItem { seq: s.id, pos: s.pos, tokens, want: true, sampling: s.sampling.clone(), hist, constraint: s.cons_after_forced() });
                idx.push(k);
            } else if budget > 0 && !eng.busy(s.id) {
                let rem = s.prompt.len() - s.pos;
                let mut n = rem.min(chunk).min(budget);
                // ほかの人がデコードしている間は、プロンプトの最後の 32 語を別の塊に残す（最後の塊はエンジンが答えを待って返すので、
                // 大きいとその間ほかの人の書き出しが止まる。前の塊は待たずに流せる）
                if n_dec > 0 && rem > FINAL_TAIL && n > rem - FINAL_TAIL {
                    n = rem - FINAL_TAIL;
                }
                budget -= n;
                let end = s.pos + n;
                items.push(StepItem {
                    seq: s.id,
                    pos: s.pos,
                    tokens: s.prompt[s.pos..end].to_vec(),
                    want: end == s.prompt.len(),
                    sampling: s.sampling.clone(),
                    hist: None,
                    constraint: if end == s.prompt.len() { s.cons.clone().map(Arc::new) } else { None },
                });
                idx.push(k);
            }
        }
        if items.is_empty() {
            // 予算が 0（デコードの人が予算より多い）でもデコードは入るので、ここには来ないはず
            std::thread::sleep(Duration::from_millis(1));
            continue;
        }
        // 5. 計算
        let outs = match eng.step(&items) {
            Ok(o) if o.len() == items.len() => o,
            r => {
                let m = match r {
                    Ok(o) => format!("engine returned {} outputs for {} items", o.len(), items.len()),
                    Err(m) => m,
                };
                fail_all(&mut running, &mut eng, &m);
                fails += 1;
                if fails >= 3 {
                    *sh.status.lock().unwrap() = Status::Failed(format!("engine step failed {fails} times in a row: {m}"));
                    for j in waiting.drain(..) {
                        let _ = j.tx.send(GenEvent::Error("the engine has failed".into()));
                        sh.waiting.fetch_sub(1, Ordering::SeqCst);
                    }
                    return;
                }
                continue;
            }
        };
        fails = 0;
        sh.steps.fetch_add(1, Ordering::Relaxed);
        sh.step_tokens.fetch_add(items.iter().map(|i| i.tokens.len() as u64).sum(), Ordering::Relaxed);
        // 6. 後始末
        let mut done = vec![false; running.len()];
        for ((it, out), &k) in items.iter().zip(outs).zip(&idx) {
            let s = &mut running[k];
            s.pos += it.tokens.len();
            if !it.want {
                continue;
            }
            // 差し込んだ語を出力へ（考えはここで閉じた）
            if s.prefilled() && !s.forced.is_empty() {
                let f = std::mem::take(&mut s.forced);
                s.n_gen += f.len();
                for &t in &f {
                    s.hist.push(t);
                    if let Some(m) = &mut s.cons {
                        m.advance(t);
                    }
                }
                if s.tx.send(GenEvent::Tokens(f)).is_err() {
                    done[k] = true;
                    continue;
                }
            }
            let toks = match out {
                ItemOut::Token(t) => vec![t],
                ItemOut::Tokens(v) if !v.is_empty() => {
                    // 受け取った下書きの分もエンジンは計算済み
                    s.pos += v.len() - 1;
                    v
                }
                ItemOut::Tokens(_) => {
                    let _ = s.tx.send(GenEvent::Error("the engine returned no token".into()));
                    done[k] = true;
                    continue;
                }
                ItemOut::Logits(l) => match &s.cons {
                    Some(m) => vec![sampler::prepare_masked(&l, cfg.valid_vocab, &s.sampling, &s.hist, m).pick(&s.sampling, s.pos as u64)],
                    None => vec![sampler::sample(&l, cfg.valid_vocab, &s.sampling, &s.hist, s.pos as u64)],
                },
                ItemOut::None => {
                    let _ = s.tx.send(GenEvent::Error("the engine returned no token".into()));
                    done[k] = true;
                    continue;
                }
                ItemOut::Error(m) => {
                    let _ = s.tx.send(GenEvent::Error(m));
                    done[k] = true;
                    continue;
                }
            };
            // 何語か来たら（下書き）、前から順に: 止める語・max_tokens で終わったら後ろは捨てる（列は release する）
            let n_toks = toks.len();
            for (j, tok) in toks.into_iter().enumerate() {
                s.n_gen += 1;
                // JSON の形: 出せない語が来たら（エンジンの誤り）ここで止める（壊れた JSON を黙って返さない）
                if let Some(m) = &mut s.cons {
                    if !m.allows(tok) || !m.advance(tok) {
                        let _ = s.tx.send(GenEvent::Error("internal error: the output left the response_format (please report)".into()));
                        done[k] = true;
                        break;
                    }
                }
                if s.stop_ids.contains(&tok) {
                    let _ = s.tx.send(GenEvent::Done(FinishReason::Stop(tok)));
                    done[k] = true;
                    break;
                }
                if s.tx.send(GenEvent::Tokens(vec![tok])).is_err() {
                    done[k] = true;
                    break;
                }
                s.hist.push(tok);
                s.last = tok;
                if let Some(tb) = &s.think {
                    if tok == tb.end {
                        s.think = None;
                    } else {
                        s.n_think += 1;
                        // 上限に達した: 次の歩で考えを閉じる文を差し込む（この歩の残りの語＝受け取った下書きは出してから）
                        if s.n_think >= tb.budget && j + 1 == n_toks && s.pos + tb.close.len() + 2 < info.max_ctx {
                            s.forced = tb.close.clone();
                            s.think = None;
                        }
                    }
                }
                // 次の歩でこの語を位置 pos に入れるので、pos が上限なら終わり（下書きの途中の語は、エンジンが計算済みなので pos を見ない）
                if s.n_gen >= s.max_tokens || (j + 1 == n_toks && s.pos >= info.max_ctx) {
                    let _ = s.tx.send(GenEvent::Done(FinishReason::Length));
                    done[k] = true;
                    break;
                }
            }
        }
        let mut k = 0;
        running.retain(|s| {
            let keep = !done[k];
            k += 1;
            if !keep {
                eng.release(s.id);
                sh.running.fetch_sub(1, Ordering::SeqCst);
            }
            keep
        });
    }
}
