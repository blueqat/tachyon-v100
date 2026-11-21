// SPDX-License-Identifier: Apache-2.0
//! Inkling-2-Mini（src/inkling.rs）を口の `Engine` に載せる。
//!
//! - 列（依頼）は lane に 1 つずつ（max_seqs 本）。デコードの 1 語の列はまとめて 1 歩（step_lanes・CUDA Graph）、
//!   プレフィルの塊は列ごとに（prefill_lane）。logits は CPU の sampler が選ぶ（ItemOut::Logits）
//! - 頭の使い回し: 依頼の turn_start（最後の回の頭）まで読んだ所で lane の状態（全体の層の KV はその長さまで・
//!   窓の層の輪・畳み込みの状態）を写して控える。同じ持ち主（scope）の次の依頼がその語の列で始まれば戻して続きから。
//!   控えは持ち主ごとに per_account まで・全部で cap_bytes まで（古い物から捨てる）
use super::engine::{Engine, EngineInfo, EngineLoader, ItemOut, SeqId, StepItem};
use crate::gguf;
use crate::inkling::{Ink, InkSeq, LaneSnap};
use serde_json::json;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct InkConfig {
    pub gguf: PathBuf,
    pub ctx: usize,
    pub max_seqs: usize,
    pub kv8: bool,
    pub prefix_cache: bool,
    pub prefix_cache_bytes: usize,
    pub prefix_per_account: usize,
    /// 控えを取る最短の語の数（短い頭は写すより読み直す方が速い）
    pub prefix_min_tokens: usize,
    /// 読み込んだ後の GPU の使用（cudaMemGetInfo の 全部 − 空き）＋ 頭の控えの上限 がこれを越えたら読み込みを失敗にする（0 で見ない）。
    /// ⚠️ 32GB の V100 なら 30 以下（1.5GB 以上の余白。それより詰めると走っている間の確保で OOM になった）
    pub max_vram_gib: f64,
}

struct Lane {
    id: Option<SeqId>,
    /// 控えを取る位置と、その頭の語・持ち主
    snap_at: Option<usize>,
    snap_toks: Vec<u32>,
    scope: String,
}

struct Snap {
    scope: String,
    toks: Vec<u32>,
    snap: LaneSnap,
    used: u64,
}

pub struct InkEngine {
    m: Ink,
    seq: InkSeq,
    lanes: Vec<Lane>,
    vocab: usize,
    cfg: InkConfig,
    snaps: Vec<Snap>,
    tick: u64,
    poisoned: Option<String>,
    hits: u64,
    hit_tokens: u64,
}

// 列の状態・重みは GPU の上（生のポインタ）。順番決めの 1 本の糸だけが触る
unsafe impl Send for InkEngine {}

fn log(v: serde_json::Value) {
    eprintln!("{v}");
}

pub fn loader(cfg: InkConfig) -> EngineLoader {
    Box::new(move |progress: &dyn Fn(f32, &str)| {
        let r = catch_unwind(AssertUnwindSafe(|| -> Result<InkEngine, String> {
            progress(0.01, "cuda");
            crate::cuda::check(unsafe { crate::cuda::ffi::cudaSetDevice(0) }, "cudaSetDevice");
            crate::inkling::set_kv8(cfg.kv8);
            progress(0.05, "gguf");
            let g = gguf::Model::open(&cfg.gguf).map_err(|e| format!("{}: {e}", cfg.gguf.display()))?;
            progress(0.1, "weights");
            let t0 = std::time::Instant::now();
            let m = Ink::load(&g, cfg.ctx)?;
            let n = cfg.max_seqs.clamp(1, m.max_lanes);
            let seq = m.new_batch(cfg.ctx, n);
            let vocab = m.cfg.vocab;
            let (free, total) = crate::model::mem_info();
            let used = (total - free) as f64 / (1u64 << 30) as f64;
            let projected = used + cfg.prefix_cache_bytes as f64 / (1u64 << 30) as f64;
            log(json!({"ev": "ink_loaded", "secs": t0.elapsed().as_secs_f64(), "weights_gb": m.weight_bytes as f64 / 1e9, "state_gb": seq.bytes() as f64 / 1e9,
                       "lanes": n, "ctx": cfg.ctx, "kv8": seq.kv8, "gpu_used_gib": (used * 100.0).round() / 100.0, "gpu_projected_gib": (projected * 100.0).round() / 100.0,
                       "max_vram_gib": cfg.max_vram_gib}));
            if cfg.max_vram_gib > 0.0 && projected > cfg.max_vram_gib {
                return Err(format!("VRAM: 読み込み後 {used:.2} GiB ＋ 頭の控え {:.2} GiB = {projected:.2} GiB が --max-vram-gib {} を越える",
                                   cfg.prefix_cache_bytes as f64 / (1u64 << 30) as f64, cfg.max_vram_gib));
            }
            Ok(InkEngine {
                m,
                seq,
                lanes: (0..n).map(|_| Lane { id: None, snap_at: None, snap_toks: vec![], scope: String::new() }).collect(),
                vocab,
                cfg: cfg.clone(),
                snaps: vec![],
                tick: 0,
                poisoned: None,
                hits: 0,
                hit_tokens: 0,
            })
        }));
        match r {
            Ok(Ok(e)) => Ok(Box::new(e) as Box<dyn Engine>),
            Ok(Err(e)) => Err(e),
            Err(p) => Err(format!("engine load panic: {}", panic_msg(p))),
        }
    })
}

fn panic_msg(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "?".into())
}

impl InkEngine {
    fn lane_of(&self, seq: SeqId) -> Option<usize> {
        self.lanes.iter().position(|l| l.id == Some(seq))
    }

    fn fail(&mut self, p: Box<dyn std::any::Any + Send>) -> String {
        let m = format!("engine failure: {}", panic_msg(p));
        self.poisoned = Some(m.clone());
        log(json!({"ev": "engine_panic", "exit_in_secs": 2}));
        std::thread::spawn(|| {
            std::thread::sleep(std::time::Duration::from_secs(2));
            super::util::hard_exit(70);
        });
        m
    }

    /// 控えを足す（同じ持ち主の同じ頭は入れ替え・持ち主ごとの数と全体の大きさを守る）
    fn put_snap(&mut self, scope: String, toks: Vec<u32>, snap: LaneSnap) {
        self.tick += 1;
        self.snaps.retain(|s| !(s.scope == scope && s.toks == toks));
        self.snaps.push(Snap { scope: scope.clone(), toks, snap, used: self.tick });
        loop {
            let mine: Vec<usize> = (0..self.snaps.len()).filter(|&i| self.snaps[i].scope == scope).collect();
            let total: usize = self.snaps.iter().map(|s| s.snap.bytes()).sum();
            let over_acct = mine.len() > self.cfg.prefix_per_account;
            if !over_acct && total <= self.cfg.prefix_cache_bytes {
                break;
            }
            let pool: Vec<usize> = if over_acct { mine } else { (0..self.snaps.len()).collect() };
            let Some(&old) = pool.iter().min_by_key(|&&i| self.snaps[i].used) else { break };
            self.snaps.remove(old);
        }
    }

    fn step_inner(&mut self, items: &[StepItem]) -> Result<Vec<ItemOut>, String> {
        let mut out: Vec<ItemOut> = vec![ItemOut::None; items.len()];
        let mut dec: Vec<(usize, usize, u32)> = vec![]; // (item, lane, 語)
        for (k, it) in items.iter().enumerate() {
            let Some(lane) = self.lane_of(it.seq) else {
                out[k] = ItemOut::Error("unknown sequence".into());
                continue;
            };
            if self.seq.pos[lane] != it.pos {
                out[k] = ItemOut::Error(format!("position mismatch: engine {} scheduler {}", self.seq.pos[lane], it.pos));
                continue;
            }
            if it.tokens.len() == 1 && it.want {
                dec.push((k, lane, it.tokens[0]));
                continue;
            }
            // プレフィルの塊: 控えの位置を跨ぐなら、そこで切って控えを取る
            let mut toks: &[u32] = &it.tokens;
            if let Some(at) = self.lanes[lane].snap_at {
                if it.pos < at && at <= it.pos + toks.len() {
                    let cut = at - it.pos;
                    self.m.prefill_lane(&mut self.seq, lane, &toks[..cut]);
                    toks = &toks[cut..];
                    let (scope, st) = (self.lanes[lane].scope.clone(), std::mem::take(&mut self.lanes[lane].snap_toks));
                    self.lanes[lane].snap_at = None;
                    // 写す前に場所を空ける（写した後に捨てると、一時に上限を 1 つぶん越える）。上限より大きい控えは取らない
                    let need = self.seq.snap_bytes(at);
                    if need <= self.cfg.prefix_cache_bytes {
                        self.snaps.retain(|s| !(s.scope == scope && s.toks == st));
                        loop {
                            let total: usize = self.snaps.iter().map(|s| s.snap.bytes()).sum();
                            let mine = self.snaps.iter().filter(|s| s.scope == scope).count();
                            if total + need <= self.cfg.prefix_cache_bytes && mine < self.cfg.prefix_per_account.max(1) {
                                break;
                            }
                            let pool: Vec<usize> = if mine >= self.cfg.prefix_per_account.max(1) {
                                (0..self.snaps.len()).filter(|&i| self.snaps[i].scope == scope).collect()
                            } else {
                                (0..self.snaps.len()).collect()
                            };
                            let Some(&old) = pool.iter().min_by_key(|&&i| self.snaps[i].used) else { break };
                            self.snaps.remove(old);
                        }
                        let snap = self.seq.save_lane(lane, at);
                        log(json!({"ev": "ink_snap", "tokens": at, "mb": snap.bytes() / (1 << 20)}));
                        self.put_snap(scope, st, snap);
                    } else {
                        log(json!({"ev": "ink_snap_skip", "tokens": at, "mb": need / (1 << 20)}));
                    }
                }
            }
            if !toks.is_empty() {
                self.m.prefill_lane(&mut self.seq, lane, toks);
            }
            if it.want {
                out[k] = ItemOut::Logits(self.m.logits_row(0));
            }
        }
        if !dec.is_empty() {
            let lanes: Vec<usize> = dec.iter().map(|d| d.1).collect();
            let toks: Vec<u32> = dec.iter().map(|d| d.2).collect();
            for ch in (0..dec.len()).collect::<Vec<_>>().chunks(self.m.max_lanes) {
                let l: Vec<usize> = ch.iter().map(|&i| lanes[i]).collect();
                let t: Vec<u32> = ch.iter().map(|&i| toks[i]).collect();
                self.m.step_lanes(&mut self.seq, &l, &t);
                let all = self.m.logits_rows(t.len());
                for (r, &i) in ch.iter().enumerate() {
                    out[dec[i].0] = ItemOut::Logits(all[r * self.vocab..(r + 1) * self.vocab].to_vec());
                }
            }
        }
        Ok(out)
    }
}

impl Engine for InkEngine {
    fn info(&self) -> EngineInfo {
        EngineInfo { max_seqs: self.lanes.len(), max_ctx: self.cfg.ctx, max_step_tokens: 4096 }
    }

    fn step(&mut self, items: &[StepItem]) -> Result<Vec<ItemOut>, String> {
        if let Some(p) = &self.poisoned {
            return Err(p.clone());
        }
        match catch_unwind(AssertUnwindSafe(|| self.step_inner(items))) {
            Ok(r) => r,
            Err(p) => Err(self.fail(p)),
        }
    }

    fn admit(&mut self, seq: SeqId, _prompt_len: usize, max_total: usize) -> bool {
        if self.poisoned.is_some() {
            return true;
        }
        let _ = max_total;
        let Some(i) = self.lanes.iter().position(|l| l.id.is_none()) else { return false };
        if catch_unwind(AssertUnwindSafe(|| self.seq.reset_lane(i))).is_err() {
            return true;
        }
        let l = &mut self.lanes[i];
        l.id = Some(seq);
        l.snap_at = None;
        l.snap_toks.clear();
        l.scope.clear();
        true
    }

    fn reuse(&mut self, seq: SeqId, scope: &str, prompt: &[u32], snap: Option<usize>) -> usize {
        if !self.cfg.prefix_cache || self.poisoned.is_some() {
            return 0;
        }
        let Some(lane) = self.lane_of(seq) else { return 0 };
        // 一番長く当たる控え（プロンプトより短い物）
        let best = (0..self.snaps.len())
            .filter(|&i| self.snaps[i].scope == scope && self.snaps[i].toks.len() < prompt.len() && prompt.starts_with(&self.snaps[i].toks))
            .max_by_key(|&i| self.snaps[i].toks.len());
        let mut got = 0;
        if let Some(i) = best {
            self.tick += 1;
            self.snaps[i].used = self.tick;
            let r = catch_unwind(AssertUnwindSafe(|| self.seq.restore_lane(lane, &self.snaps[i].snap)));
            if let Err(p) = r {
                self.fail(p);
                return 0;
            }
            got = self.snaps[i].toks.len();
            self.hits += 1;
            self.hit_tokens += got as u64;
            log(json!({"ev": "ink_prefix_hit", "tokens": got, "prompt": prompt.len()}));
        }
        // この依頼の控え: 最後の回の頭（当たった所より先・短すぎない所）
        if let Some(at) = snap.filter(|&a| a > got && a >= self.cfg.prefix_min_tokens && a < prompt.len()) {
            let l = &mut self.lanes[lane];
            l.snap_at = Some(at);
            l.snap_toks = prompt[..at].to_vec();
            l.scope = scope.to_string();
        }
        got
    }

    fn release(&mut self, seq: SeqId) {
        if let Some(i) = self.lane_of(seq) {
            let l = &mut self.lanes[i];
            l.id = None;
            l.snap_at = None;
            l.snap_toks.clear();
        }
    }
}
