//! /metrics（Prometheus の文字の形）。⛔ 数だけ（依頼の中身・鍵は入れない。ラベルはアカウントの名前・口・結果だけ）。
use super::backend::Stats;
use std::collections::BTreeMap;
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

const TTFT_B: &[f64] = &[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0];
const E2E_B: &[f64] = &[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0];
const TPS_B: &[f64] = &[1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 150.0, 200.0, 400.0];

struct Hist {
    bounds: &'static [f64],
    counts: Vec<u64>,
    sum: f64,
    n: u64,
}

impl Hist {
    fn new(bounds: &'static [f64]) -> Self {
        Hist { bounds, counts: vec![0; bounds.len()], sum: 0.0, n: 0 }
    }

    fn add(&mut self, x: f64) {
        for (i, b) in self.bounds.iter().enumerate() {
            if x <= *b {
                self.counts[i] += 1;
            }
        }
        self.sum += x;
        self.n += 1;
    }

    fn render(&self, out: &mut String, name: &str, help: &str) {
        let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} histogram");
        for (b, c) in self.bounds.iter().zip(&self.counts) {
            let _ = writeln!(out, "{name}_bucket{{le=\"{b}\"}} {c}");
        }
        let _ = writeln!(out, "{name}_bucket{{le=\"+Inf\"}} {}\n{name}_sum {}\n{name}_count {}", self.n, self.sum, self.n);
    }
}

#[derive(Clone, Default)]
struct Tokens {
    prompt: u64,
    cached: u64,
    completion: u64,
    reasoning: u64,
}

struct Inner {
    /// (アカウント, 口, 状態の番号) → 数
    requests: BTreeMap<(String, &'static str, u16), u64>,
    /// (アカウント, 口, 理由) → 断った数（400・401・429・503 など）
    rejected: BTreeMap<(String, &'static str, &'static str), u64>,
    tokens: BTreeMap<String, Tokens>,
    finish: BTreeMap<&'static str, u64>,
    ttft: Hist,
    queue: Hist,
    e2e: Hist,
    tps: Hist,
    /// 直近の秒ごとの出した語の数（秒, 数）
    recent: [(u64, u64); 16],
}

/// 終わった依頼 1 つ（数だけ）
pub struct RequestRecord<'a> {
    pub account: &'a str,
    pub endpoint: &'static str,
    pub status: u16,
    pub finish: &'static str,
    pub prompt: usize,
    pub cached: usize,
    pub completion: usize,
    pub reasoning: usize,
    pub ttft: Option<f64>,
    pub queue: Option<f64>,
    pub total: f64,
    pub decode_tps: Option<f64>,
}

/// その時の数（/metrics を書くときに口が集める）
pub struct Gauges {
    pub ready: bool,
    pub draining: bool,
    pub inflight: usize,
    pub stats: Stats,
    /// (アカウント, 走っている依頼)
    pub accounts: Vec<(String, usize)>,
    /// (アカウント, 文書の数, 文書の語の数)
    pub documents: Vec<(String, usize, usize)>,
}

pub struct Metrics {
    inner: Mutex<Inner>,
    generated: AtomicU64,
    started: Instant,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// ラベルの値の書き方（\ " 改行を逃がす）
fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

impl Metrics {
    pub fn new() -> Self {
        Metrics {
            inner: Mutex::new(Inner {
                requests: BTreeMap::new(),
                rejected: BTreeMap::new(),
                tokens: BTreeMap::new(),
                finish: BTreeMap::new(),
                ttft: Hist::new(TTFT_B),
                queue: Hist::new(TTFT_B),
                e2e: Hist::new(E2E_B),
                tps: Hist::new(TPS_B),
                recent: [(u64::MAX, 0); 16],
            }),
            generated: AtomicU64::new(0),
            started: Instant::now(),
        }
    }

    /// 出した語（stream の途中でも。全体の tok/s 用）
    pub fn generated(&self, n: usize) {
        self.generated.fetch_add(n as u64, Ordering::Relaxed);
        let sec = self.started.elapsed().as_secs();
        let mut g = self.inner.lock().unwrap();
        let slot = &mut g.recent[(sec % 16) as usize];
        if slot.0 != sec {
            *slot = (sec, 0);
        }
        slot.1 += n as u64;
    }

    /// 直近 10 秒（今の秒も含む）に出した語の数。見張りが「混んでいるが動いている」を見分けるのに使う
    pub fn recent_tokens(&self) -> u64 {
        let now = self.started.elapsed().as_secs();
        let g = self.inner.lock().unwrap();
        g.recent.iter().filter(|(s, _)| *s != u64::MAX && *s <= now && now - *s <= 10).map(|(_, n)| n).sum()
    }

    pub fn request_done(&self, r: &RequestRecord) {
        let mut g = self.inner.lock().unwrap();
        *g.requests.entry((r.account.to_string(), r.endpoint, r.status)).or_default() += 1;
        *g.finish.entry(r.finish).or_default() += 1;
        let t = g.tokens.entry(r.account.to_string()).or_default();
        t.prompt += r.prompt as u64;
        t.cached += r.cached as u64;
        t.completion += r.completion as u64;
        t.reasoning += r.reasoning as u64;
        if let Some(x) = r.ttft {
            g.ttft.add(x);
        }
        if let Some(x) = r.queue {
            g.queue.add(x);
        }
        if let Some(x) = r.decode_tps {
            g.tps.add(x);
        }
        if r.status < 400 || r.status == 499 {
            g.e2e.add(r.total);
        }
    }

    /// 受け付けて返した依頼（chat 以外の口）
    pub fn counted(&self, account: &str, endpoint: &'static str, status: u16) {
        *self.inner.lock().unwrap().requests.entry((account.to_string(), endpoint, status)).or_default() += 1;
    }

    /// 受け付けなかった依頼（鍵が違う・上限・形の誤り）。アカウントが分からなければ "-"
    pub fn rejected(&self, account: &str, endpoint: &'static str, status: u16, code: &'static str) {
        let mut g = self.inner.lock().unwrap();
        *g.requests.entry((account.to_string(), endpoint, status)).or_default() += 1;
        *g.rejected.entry((account.to_string(), endpoint, code)).or_default() += 1;
    }

    pub fn render(&self, gs: &Gauges) -> String {
        let g = self.inner.lock().unwrap();
        let mut o = String::with_capacity(8192);
        let w = &mut o;
        let gauge = |w: &mut String, name: &str, help: &str, v: f64| {
            let _ = writeln!(w, "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {v}");
        };
        gauge(w, "tachyon_ready", "1 when the engine is loaded and accepting requests", f64::from(u8::from(gs.ready && !gs.draining)));
        gauge(w, "tachyon_draining", "1 while draining (new requests get 503)", f64::from(u8::from(gs.draining)));
        gauge(w, "tachyon_uptime_seconds", "seconds since the server started", self.started.elapsed().as_secs_f64());
        gauge(w, "tachyon_inflight_requests", "generation requests in flight (queued + running)", gs.inflight as f64);
        gauge(w, "tachyon_queue_length", "requests waiting in the engine queue", gs.stats.waiting as f64);
        gauge(w, "tachyon_running_requests", "requests being computed by the engine", gs.stats.running as f64);
        gauge(w, "tachyon_max_running", "engine slots", gs.stats.max_running as f64);
        gauge(w, "tachyon_max_queue_length", "queue limit (above it requests get 429)", gs.stats.max_waiting as f64);
        // 直近 10 秒（今の秒は除く）の tok/s
        let now = self.started.elapsed().as_secs();
        let recent: u64 = g.recent.iter().filter(|(s, _)| *s != u64::MAX && *s < now && now - *s <= 10).map(|(_, n)| n).sum();
        gauge(w, "tachyon_tokens_per_second", "generated tokens per second over the last 10 s (all accounts)", recent as f64 / 10.0);
        let _ = writeln!(w, "# HELP tachyon_generated_tokens_total generated tokens (all accounts, counted while streaming)\n# TYPE tachyon_generated_tokens_total counter\ntachyon_generated_tokens_total {}", self.generated.load(Ordering::Relaxed));
        for (k, v) in &gs.stats.extra {
            let name: String = k.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' }).collect();
            gauge(w, &format!("tachyon_backend_{name}"), "engine-specific value", *v);
        }

        let _ = writeln!(w, "# HELP tachyon_requests_total finished requests by account, endpoint and HTTP status (499 = client closed)\n# TYPE tachyon_requests_total counter");
        for ((a, e, s), n) in &g.requests {
            let _ = writeln!(w, "tachyon_requests_total{{account=\"{}\",endpoint=\"{e}\",code=\"{s}\"}} {n}", esc(a));
        }
        let _ = writeln!(w, "# HELP tachyon_rejected_total rejected requests by reason\n# TYPE tachyon_rejected_total counter");
        for ((a, e, r), n) in &g.rejected {
            let _ = writeln!(w, "tachyon_rejected_total{{account=\"{}\",endpoint=\"{e}\",reason=\"{r}\"}} {n}", esc(a));
        }
        let _ = writeln!(w, "# HELP tachyon_finish_total finished generations by finish reason\n# TYPE tachyon_finish_total counter");
        for (f, n) in &g.finish {
            let _ = writeln!(w, "tachyon_finish_total{{reason=\"{f}\"}} {n}");
        }
        let per = |w: &mut String, name: &str, help: &str, f: &dyn Fn(&Tokens) -> u64| {
            let _ = writeln!(w, "# HELP {name} {help}\n# TYPE {name} counter");
            for (a, t) in &g.tokens {
                let _ = writeln!(w, "{name}{{account=\"{}\"}} {}", esc(a), f(t));
            }
        };
        per(w, "tachyon_prompt_tokens_total", "prompt tokens by account", &|t| t.prompt);
        per(w, "tachyon_cached_prompt_tokens_total", "prompt tokens served from the prefix cache by account", &|t| t.cached);
        per(w, "tachyon_completion_tokens_total", "generated tokens (reasoning + answer) by account", &|t| t.completion);
        per(w, "tachyon_reasoning_tokens_total", "generated reasoning tokens by account", &|t| t.reasoning);
        let _ = writeln!(w, "# HELP tachyon_prefix_cache_hit_ratio cached / prompt tokens since start, by account\n# TYPE tachyon_prefix_cache_hit_ratio gauge");
        for (a, t) in &g.tokens {
            let r = if t.prompt > 0 { t.cached as f64 / t.prompt as f64 } else { 0.0 };
            let _ = writeln!(w, "tachyon_prefix_cache_hit_ratio{{account=\"{}\"}} {r}", esc(a));
        }
        let _ = writeln!(w, "# HELP tachyon_account_inflight_requests requests in flight by account\n# TYPE tachyon_account_inflight_requests gauge");
        for (a, n) in &gs.accounts {
            let _ = writeln!(w, "tachyon_account_inflight_requests{{account=\"{}\"}} {n}", esc(a));
        }
        let _ = writeln!(w, "# HELP tachyon_documents precomputed documents by account\n# TYPE tachyon_documents gauge");
        for (a, n, _) in &gs.documents {
            let _ = writeln!(w, "tachyon_documents{{account=\"{}\"}} {n}", esc(a));
        }
        let _ = writeln!(w, "# HELP tachyon_document_tokens tokens of precomputed documents by account\n# TYPE tachyon_document_tokens gauge");
        for (a, _, t) in &gs.documents {
            let _ = writeln!(w, "tachyon_document_tokens{{account=\"{}\"}} {t}", esc(a));
        }
        g.ttft.render(w, "tachyon_ttft_seconds", "time from request to first generated token");
        g.queue.render(w, "tachyon_queue_wait_seconds", "time from request to the engine starting it");
        g.e2e.render(w, "tachyon_request_duration_seconds", "time from request to the last token");
        g.tps.render(w, "tachyon_decode_tokens_per_second", "per-request decode speed after the first token");
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_counts() {
        let m = Metrics::new();
        m.generated(5);
        m.request_done(&RequestRecord {
            account: "a\"b",
            endpoint: "chat",
            status: 200,
            finish: "stop",
            prompt: 100,
            cached: 64,
            completion: 10,
            reasoning: 4,
            ttft: Some(0.3),
            queue: Some(0.01),
            total: 1.0,
            decode_tps: Some(42.0),
        });
        m.rejected("-", "chat", 401, "invalid_api_key");
        let g = Gauges { ready: true, draining: false, inflight: 0, stats: Stats { extra: vec![("kv pages".into(), 3.0)], ..Default::default() }, accounts: vec![], documents: vec![] };
        let s = m.render(&g);
        assert!(s.contains("tachyon_requests_total{account=\"a\\\"b\",endpoint=\"chat\",code=\"200\"} 1"), "{s}");
        assert!(s.contains("tachyon_rejected_total{account=\"-\",endpoint=\"chat\",reason=\"invalid_api_key\"} 1"));
        assert!(s.contains("tachyon_prefix_cache_hit_ratio{account=\"a\\\"b\"} 0.64"));
        assert!(s.contains("tachyon_ttft_seconds_bucket{le=\"0.5\"} 1") && s.contains("tachyon_ttft_seconds_bucket{le=\"0.25\"} 0"));
        assert!(s.contains("tachyon_backend_kv_pages 3"));
        assert!(s.contains("tachyon_generated_tokens_total 5"));
    }
}
