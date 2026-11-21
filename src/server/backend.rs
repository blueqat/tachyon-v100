//! 口（HTTP）とエンジンの境目。口は `Backend` だけを知る。
//!   - 口: 鍵・回数・依頼の確かめ・テンプレート・語に切る・文脈の長さ（400）・文字に戻す・止めの文字列・usage
//!   - Backend: 待ち行列（一杯なら QueueFull → 429）・順番決め・止める語・max_tokens・文頭の使い回し・文書の先計算
//! 実装: `fake::FakeBackend`（決まった語を流す。試し用）・`engine::SimpleScheduler`（Engine の 1 歩を回す。M1 用）。
//! 本物の順番決め（頁つき KV・基数木・追い出し）は、この trait を直に実装して差し替える。
use super::sampler::SamplingParams;
use tokio::sync::mpsc;

#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    /// 重みを読んでいる（progress は 0〜1）
    Loading { progress: f32, what: String },
    Ready,
    /// 起動に失敗した（/health は 500）
    Failed(String),
}

/// 文頭の使い回しの持ち主（docs/API.md 3.6）。namespace が同じ依頼どうしだけ KV・状態の写しを共有してよい:
/// 社内の鍵は全部 "internal"、お客さまはアカウントごと（"acct:<名前>"）。
/// ⚠️ お客さまどうしで共有すると、当たりの速さから他の人の文を推し量れてしまう（KV 共有の時間差による漏れ）
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CacheScope {
    pub namespace: String,
    /// 使った頁・当たりの数をアカウントごとに数えるため
    pub account: String,
    /// 頭の使い回しの控えの持ち主（M1 の GPU の控え・RAM の段が見る）。既定はアカウントの名前と同じ。
    /// 前段の中継（proxy）のように 1 つの鍵で大勢を代表するアカウントは、--prefix-owner-accounts に載せると
    /// 依頼の x-tachyon-prefix-owner（利用者の名前を前段で HMAC した 16 進）で "<アカウント>#<16 進>" に分ける（2026-11-15）。
    /// ⛔ 違う持ち主の控えには当てない（public の札だけは別。hostcache.rs）
    pub owner: String,
}

/// 持ち主の札（x-tachyon-prefix-owner）の形: 16〜64 字の小文字の 16 進
pub fn valid_owner_tag(t: &str) -> bool {
    (16..=64).contains(&t.len()) && t.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl CacheScope {
    pub fn for_account(account: &str, internal: bool) -> Self {
        let namespace = if internal { "internal".to_string() } else { format!("acct:{account}") };
        CacheScope { namespace, account: account.to_string(), owner: account.to_string() }
    }

    /// 決まった名前の持ち主（_health・_prewarm）
    pub fn fixed(name: &str) -> Self {
        CacheScope { namespace: name.to_string(), account: name.to_string(), owner: name.to_string() }
    }

    /// アカウントの中をさらに利用者で分ける（tag は valid_owner_tag を通った物。アカウントの名前に # は入らない）
    pub fn with_owner(mut self, tag: &str) -> Self {
        self.namespace = format!("{}#{tag}", self.namespace);
        self.owner = format!("{}#{tag}", self.account);
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Priority {
    Normal,
    /// 夜の一括（文書の先計算など）。普通の依頼が待っている間は回さない
    Low,
}

/// 先に計算した文書（その長さまでの KV と状態の写しがあるはず）
#[derive(Clone, Debug, PartialEq)]
pub struct DocRef {
    pub id: String,
    pub tokens: usize,
}

/// 状態の写しを取るとよい所（語の位置）。エンジンは使わなくてもよい（design-27b 5 章）
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PrefixHints {
    /// system（と tools）の回の終わり（次の `<|im_start|>` の位置）
    pub system_end: Option<usize>,
    /// 文書を付けた依頼: prompt の先頭 tokens 語はその文書と同じ
    pub document: Option<DocRef>,
    /// 最後の回（`<|im_start|>assistant` …）の頭の位置。次の手番はここまでが同じになりやすい（頭の使い回しの控えを取る所）
    pub turn_start: Option<usize>,
    /// 最後の発言の回（turn_start の 1 つ前の `<|im_start|>`）の頭。次の手番はここまでが同じになりやすい
    /// （最後の発言には手番ごとの前置き〔記憶など〕が入ることがあり、次の手番の履歴では消える。2026-11-14）
    pub user_start: Option<usize>,
    /// 先計算の札（prewarm）の名前。Some なら RAM の段に pinned（追い出さない）・public（どのアカウントでも当たる）で置く
    pub pin: Option<String>,
}

/// 1 つの生成の依頼（中身は語の番号だけ。文字は渡さない）
/// 画像 1 枚（口が読み・縮めた物）。エンジンが符号化して、prompt の start から n = nx·ny 語（<|image_pad|>）の所に置く。
/// rgb は w × h（32 の倍数）の RGB（行ごと）
pub struct MediaInput {
    pub start: usize,
    pub nx: usize,
    pub ny: usize,
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<u8>,
}

impl std::fmt::Debug for MediaInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MediaInput {{ start: {}, {}×{} 語, {}×{} 画素 }}", self.start, self.nx, self.ny, self.w, self.h)
    }
}

#[derive(Clone, Debug)]
pub struct GenRequest {
    /// 依頼の番号（chatcmpl-…。ログと数の突き合わせ用）
    pub id: String,
    pub scope: CacheScope,
    pub prompt: Vec<u32>,
    /// 出す語の上限（考え＋答え）。口が文脈の残りで切ってある
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    /// これが出たら止める（その語は Tokens に入れず Done(Stop(id)) で知らせる）
    pub stop_ids: Vec<u32>,
    pub hints: PrefixHints,
    pub priority: Priority,
    /// 考えの上限: 考え（<think> の中）の語がこの数に達したら close の語を差し込む（end = </think> の語。出たら考えは終わり）
    pub think_budget: Option<ThinkBudget>,
    /// 画像（無ければ空）。画像のある依頼は頭の使い回しの控えに入れない
    pub media: Vec<std::sync::Arc<MediaInput>>,
    /// response_format の JSON の形（json_object・json_schema）。出す語をこの形に絞る
    pub constraint: Option<super::jsonc::Matcher>,
}

#[derive(Clone, Debug)]
pub struct ThinkBudget {
    pub budget: usize,
    pub close: Vec<u32>,
    pub end: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// 止める語が出た（その語）
    Stop(u32),
    /// max_tokens か、エンジンの文脈の上限
    Length,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GenEvent {
    /// 待ち行列から出て計算を始めた。cached = 文頭が当たって計算しなかった語の数
    Started { cached: usize },
    /// 新しい語（下書きを受け取ったときは何語も）
    Tokens(Vec<u32>),
    /// 終わり
    Done(FinishReason),
    /// 失敗（ここで終わり。文はエンジンが作った物だけ＝依頼の中身を入れない）
    Error(String),
}

/// 語の流れ。⭐ 受け取り側を drop すると取り消し（Backend は送れなくなったら止めて頁を空ける）
pub type GenStream = mpsc::UnboundedReceiver<GenEvent>;

#[derive(Clone, Debug, PartialEq)]
pub enum SubmitError {
    /// 待ち行列が一杯（429）
    QueueFull { retry_after: u64 },
    /// 読み込み中・止めている（503）
    NotReady,
    /// エンジンの文脈に入らない（400）
    TooLong { max: usize },
    /// 依頼が変（400）
    Invalid(String),
    /// この Backend ではできない（501）
    Unsupported,
    Internal(String),
}

/// /metrics と /health に出す数
#[derive(Clone, Debug, Default)]
pub struct Stats {
    /// 待ち行列の長さ（まだ計算を始めていない依頼）
    pub waiting: usize,
    /// 走っている依頼
    pub running: usize,
    pub max_running: usize,
    pub max_waiting: usize,
    /// 計算した語の数（プレフィル＋デコード、増えるだけ）。見張りが「長いプレフィル中」と「止まっている」を見分ける
    pub progress: u64,
    /// エンジンごとの数（名前は [a-z0-9_]。tachyon_backend_<名前> で出す）
    pub extra: Vec<(String, f64)>,
}

/// 文書の先計算の依頼（docs/API.md 3.7）。tokens はテンプレートで描いた文書の先頭（system・tools・文書の回まで）
#[derive(Clone, Debug)]
pub struct DocJob {
    pub id: String,
    pub scope: CacheScope,
    pub tokens: Vec<u32>,
    pub priority: Priority,
    pub hints: PrefixHints,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DocState {
    Queued,
    Computing { done: usize, total: usize },
    /// tier: 置き場（gpu・ram・ssd）
    Ready { tier: String },
    Failed(String),
}

pub trait Backend: Send + Sync + 'static {
    fn status(&self) -> Status;

    /// 依頼を入れる（待たずに返る）。受け付けたら語の流れを返す
    fn submit(&self, req: GenRequest) -> Result<GenStream, SubmitError>;

    fn stats(&self) -> Stats;

    /// エンジンの 1 列の語の上限（あれば口の 400 の判断に使う）
    fn max_context(&self) -> Option<usize> {
        None
    }

    /// 止める（SIGTERM）: 走っている依頼と待ち行列の依頼を全部 Error で終わらせ、新しい依頼を受けない。待たずに返る
    fn shutdown(&self) {}

    /// 文書を先に計算する（待たずに返る）
    fn doc_create(&self, job: DocJob) -> Result<DocState, SubmitError> {
        let _ = job;
        Err(SubmitError::Unsupported)
    }

    fn doc_state(&self, id: &str) -> Option<DocState> {
        let _ = id;
        None
    }

    fn doc_delete(&self, id: &str) {
        let _ = id;
    }
}
