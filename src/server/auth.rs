//! API の鍵（アカウントごと）と、鍵ごとの上限（1 分あたりの依頼の数・文脈の長さ・同時の数）。
//! 設定のファイル（JSON）には鍵の sha256 を書く（平文の "keys" は試しだけ。起動時に警告）。
//! ⛔ 鍵はログにも /metrics にも出さない（数えるのはアカウントの名前だけ）。
//!
//! ```json
//! {"accounts": {
//!   "internal-dev": {"keys_sha256": ["<64 桁>"], "internal": true, "admin": true, "rpm": 600},
//!   "customer-a":   {"keys_sha256": ["<64 桁>"], "rpm": 60, "max_context": 32768, "max_concurrent": 4}
//! }}
//! ```
use super::util::{parse_hex32, sha256};
use axum::http::{header::AUTHORIZATION, HeaderMap};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Deserialize, Clone, Debug, Default)]
#[serde(deny_unknown_fields)]
pub struct KeysFile {
    pub accounts: BTreeMap<String, AccountConf>,
}

#[derive(Deserialize, Clone, Debug, Default)]
#[serde(deny_unknown_fields)]
pub struct AccountConf {
    /// 鍵の sha256（16 進 64 文字）。`tachyon-serve hash-key` で作る
    #[serde(default)]
    pub keys_sha256: Vec<String>,
    /// 平文の鍵（試しだけ）
    #[serde(default)]
    pub keys: Vec<String>,
    /// 社内の鍵（文頭の使い回し・文書を社内の鍵どうしで共有してよい。docs/API.md 3.6）
    #[serde(default)]
    pub internal: bool,
    /// /admin/* を使える
    #[serde(default)]
    pub admin: bool,
    /// 1 分あたりの依頼の数（無ければ無制限）
    #[serde(default)]
    pub rpm: Option<u32>,
    /// 1 依頼の語の上限（入力＋出力。全体の上限 256K より小さくだけできる）
    #[serde(default)]
    pub max_context: Option<usize>,
    /// 1 依頼の出力の上限
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// 同時に走らせる依頼の数
    #[serde(default)]
    pub max_concurrent: Option<usize>,
    /// 置いておける文書の数
    #[serde(default)]
    pub max_documents: Option<usize>,
}

pub struct Account {
    pub name: String,
    pub internal: bool,
    pub admin: bool,
    pub rpm: Option<u32>,
    pub max_context: Option<usize>,
    pub max_tokens: Option<usize>,
    pub max_concurrent: Option<usize>,
    pub max_documents: Option<usize>,
    inflight: AtomicUsize,
    /// 直近 60 秒の依頼の時刻（rpm の窓）
    window: Mutex<VecDeque<Instant>>,
}

impl Account {
    fn new(name: &str, c: &AccountConf) -> Self {
        Account {
            name: name.to_string(),
            internal: c.internal,
            admin: c.admin,
            rpm: c.rpm,
            max_context: c.max_context,
            max_tokens: c.max_tokens,
            max_concurrent: c.max_concurrent,
            max_documents: c.max_documents,
            inflight: AtomicUsize::new(0),
            window: Mutex::new(VecDeque::new()),
        }
    }

    /// 1 分の窓で数える。超えたら Err(待つ秒)
    pub fn take_rate(&self) -> Result<(), u64> {
        let Some(rpm) = self.rpm else { return Ok(()) };
        if rpm == 0 {
            return Err(60);
        }
        let now = Instant::now();
        let mut w = self.window.lock().unwrap();
        while w.front().is_some_and(|&t| now.duration_since(t) >= Duration::from_secs(60)) {
            w.pop_front();
        }
        if w.len() >= rpm as usize {
            let waited = now.duration_since(w[0]).as_secs();
            return Err(60u64.saturating_sub(waited).max(1));
        }
        w.push_back(now);
        Ok(())
    }

    /// 同時の数を 1 つ取る（返す物を drop すると戻る）。上限なら None
    pub fn enter(self: &Arc<Self>) -> Option<InflightGuard> {
        let n = self.inflight.fetch_add(1, Ordering::SeqCst);
        if self.max_concurrent.is_some_and(|m| n >= m) {
            self.inflight.fetch_sub(1, Ordering::SeqCst);
            return None;
        }
        Some(InflightGuard(self.clone()))
    }

    pub fn inflight(&self) -> usize {
        self.inflight.load(Ordering::SeqCst)
    }
}

pub struct InflightGuard(Arc<Account>);

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.0.inflight.fetch_sub(1, Ordering::SeqCst);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthError {
    Missing,
    Invalid,
}

pub struct Auth {
    by_hash: HashMap<[u8; 32], Arc<Account>>,
    /// 鍵を見ない（--no-auth。全部この 1 つ・社内扱い）
    open: Option<Arc<Account>>,
    pub accounts: Vec<Arc<Account>>,
    /// 平文の鍵を書いたアカウント（起動時の警告用）
    pub plain_key_accounts: Vec<String>,
}

fn valid_name(n: &str) -> bool {
    !n.is_empty() && n.len() <= 64 && n.chars().all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
}

impl Auth {
    /// 鍵なし（127.0.0.1 で試すとき）。全部 "local"（社内・admin）
    pub fn open() -> Self {
        let c = AccountConf { internal: true, admin: true, ..Default::default() };
        let a = Arc::new(Account::new("local", &c));
        Auth { by_hash: HashMap::new(), open: Some(a.clone()), accounts: vec![a], plain_key_accounts: vec![] }
    }

    pub fn from_file(path: &Path) -> Result<Self, String> {
        let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let f: KeysFile = serde_json::from_str(&s).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_keys(f)
    }

    pub fn from_keys(f: KeysFile) -> Result<Self, String> {
        let mut by_hash = HashMap::new();
        let mut accounts = vec![];
        let mut plain = vec![];
        for (name, c) in &f.accounts {
            if !valid_name(name) {
                return Err(format!("アカウントの名前 {name:?} は英数字と ._- の 64 文字まで"));
            }
            let a = Arc::new(Account::new(name, c));
            let mut hashes = vec![];
            for h in &c.keys_sha256 {
                hashes.push(parse_hex32(h).ok_or_else(|| format!("{name}: keys_sha256 は 16 進 64 文字"))?);
            }
            for k in &c.keys {
                if k.len() < 16 || k.chars().any(char::is_whitespace) {
                    return Err(format!("{name}: 平文の鍵は空白なしの 16 文字以上"));
                }
                hashes.push(sha256(k.as_bytes()));
            }
            if !c.keys.is_empty() {
                plain.push(name.clone());
            }
            if hashes.is_empty() {
                return Err(format!("{name}: 鍵が無い（keys_sha256）"));
            }
            for h in hashes {
                if by_hash.insert(h, a.clone()).is_some() {
                    return Err(format!("{name}: 同じ鍵が 2 回書かれている"));
                }
            }
            accounts.push(a);
        }
        if accounts.is_empty() {
            return Err("accounts が空".into());
        }
        Ok(Auth { by_hash, open: None, accounts, plain_key_accounts: plain })
    }

    pub fn is_open(&self) -> bool {
        self.open.is_some()
    }

    /// Authorization: Bearer <鍵>（か x-api-key: <鍵>）→ アカウント
    pub fn check(&self, h: &HeaderMap) -> Result<Arc<Account>, AuthError> {
        if let Some(a) = &self.open {
            return Ok(a.clone());
        }
        let key = presented_key(h)?;
        self.by_hash.get(&sha256(key.as_bytes())).cloned().ok_or(AuthError::Invalid)
    }
}

/// 出された鍵。ヘッダが無ければ Missing、あるのに読めなければ（Bearer でない・空・UTF-8 でない）Invalid
fn presented_key(h: &HeaderMap) -> Result<&str, AuthError> {
    if let Some(v) = h.get(AUTHORIZATION) {
        let v = v.to_str().map_err(|_| AuthError::Invalid)?.trim();
        let (scheme, rest) = v.split_once(' ').ok_or(AuthError::Invalid)?;
        let k = rest.trim();
        return if scheme.eq_ignore_ascii_case("bearer") && !k.is_empty() { Ok(k) } else { Err(AuthError::Invalid) };
    }
    match h.get("x-api-key") {
        None => Err(AuthError::Missing),
        Some(v) => v.to_str().ok().map(str::trim).filter(|k| !k.is_empty()).ok_or(AuthError::Invalid),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::util::sha256_hex;
    use axum::http::HeaderValue;

    fn keys() -> KeysFile {
        serde_json::from_value(serde_json::json!({"accounts": {
            "a": {"keys": ["sk-aaaaaaaaaaaaaaaa"], "internal": true, "rpm": 2},
            "b": {"keys_sha256": [sha256_hex(b"sk-bbbbbbbbbbbbbbbb")], "max_concurrent": 1}
        }}))
        .unwrap()
    }

    fn hdr(k: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {k}")).unwrap());
        h
    }

    #[test]
    fn check_keys_and_limits() {
        let a = Auth::from_keys(keys()).unwrap();
        assert_eq!(a.plain_key_accounts, vec!["a".to_string()]);
        assert_eq!(a.check(&hdr("sk-aaaaaaaaaaaaaaaa")).unwrap().name, "a");
        assert_eq!(a.check(&hdr("sk-bbbbbbbbbbbbbbbb")).unwrap().name, "b");
        assert_eq!(a.check(&hdr("sk-cccccccccccccccc")).err(), Some(AuthError::Invalid));
        assert_eq!(a.check(&HeaderMap::new()).err(), Some(AuthError::Missing));
        let mut h = HeaderMap::new();
        h.insert(AUTHORIZATION, HeaderValue::from_static("Basic abc"));
        assert_eq!(a.check(&h).err(), Some(AuthError::Invalid));
        let mut h = HeaderMap::new();
        h.insert("x-api-key", HeaderValue::from_static("sk-bbbbbbbbbbbbbbbb"));
        assert_eq!(a.check(&h).unwrap().name, "b");
        let acc = a.check(&hdr("sk-aaaaaaaaaaaaaaaa")).unwrap();
        assert!(acc.take_rate().is_ok() && acc.take_rate().is_ok());
        let wait = acc.take_rate().unwrap_err();
        assert!((1..=60).contains(&wait));
        let b = a.check(&hdr("sk-bbbbbbbbbbbbbbbb")).unwrap();
        let g = b.enter().unwrap();
        assert!(b.enter().is_none());
        drop(g);
        assert!(b.enter().is_some());
        assert_eq!(b.inflight(), 0);
    }

    #[test]
    fn bad_files() {
        let bad = |v: serde_json::Value| serde_json::from_value::<KeysFile>(v).map_err(|e| e.to_string()).and_then(Auth::from_keys).err().unwrap();
        bad(serde_json::json!({"accounts": {}}));
        bad(serde_json::json!({"accounts": {"x y": {"keys": ["sk-aaaaaaaaaaaaaaaa"]}}}));
        bad(serde_json::json!({"accounts": {"x": {"keys": ["short"]}}}));
        bad(serde_json::json!({"accounts": {"x": {"keys_sha256": ["abc"]}}}));
        bad(serde_json::json!({"accounts": {"x": {}}}));
        bad(serde_json::json!({"accounts": {"x": {"keys": ["sk-aaaaaaaaaaaaaaaa"]}, "y": {"keys": ["sk-aaaaaaaaaaaaaaaa"]}}}));
        bad(serde_json::json!({"accounts": {"x": {"keys": ["sk-aaaaaaaaaaaaaaaa"], "rmp": 3}}}));
        assert!(Auth::open().check(&HeaderMap::new()).is_ok());
    }
}
