//! M1 のモデル（src/model.rs。Qwen3.8-27B を GPU 1 枚で）を口の `Engine` に載せる。
//!
//!   - M3（11-11）: 列は max_seqs 本まで（升）。KV は頁を配る置き場（Seq::new_paged）を全部の列で分け合い、受け入れの時に
//!     プロンプト＋出力の上限ぶんの頁を予約する（admit。足りなければ待つ）。デコードの人が 1 人なら MTP の下書き、2 人以上なら
//!     まとめて 1 歩（Model::decode_h・CUDA Graph）。MTP の KV は本体と同じ頁の番号で持ち、まとめた歩でも毎歩追いつかせる
//!   - CUDA の文脈・重み・列の状態（KV 16 層・GDN 48 層）は順番決めの糸だけが持つ（loader もその糸で動く）
//!   - 列の状態は読み込みのときに 1 つだけ取り、依頼ごとに reset（KV は文脈の長さ分を先に取る。64K 語で 4 GiB）
//!   - 1 歩: プレフィルの塊（opt.chunk 語まで）かデコードの 1 語を forward → want なら head。
//!     罰の無い貪欲（温度 0）は GPU の argmax の語を返し、ほかは logits を返して CPU の sampler が選ぶ
//!   - 取り消し: SimpleScheduler が歩の間に受け取り側を見る（プレフィルの塊 1 つ＝数秒以内に止まる）
//!   - ⛔ 失敗（CUDA の誤りは panic になる）は捕まえて、その依頼を Error で終え、2 秒後にプロセスを終える（exit 70）。
//!     CUDA の文脈は壊れると戻らないので、systemd（Restart=on-failure）に起こし直させる
//!   - ⛔ 依頼の中身（語・文）はログに出さない。出すのは大きさ・時間・VRAM だけ
//!   - MTP（mtp_k > 0。M2）: デコードの 1 歩 = MTP の下書き k 語 → 本体の確かめ [前の語, 下書き…]（forward_verify）→ 受け取り → commit。
//!     受け取りは貪欲なら 1 位と同じ所まで、温度 > 0 は speculative sampling（sampler::spec_row。出る語の分布は下書きなしと同じ）。
//!     返すのは ItemOut::Tokens（受け取った下書き＋新しい 1 語）。MTP の KV はプレフィルの塊ごと・確かめの後に本体の h で追いつかせる
use super::engine::{Engine, EngineInfo, EngineLoader, ItemOut, SeqId, StepItem};
use super::sampler::{self, History};
use crate::cuda::DevBuf;
use crate::gguf;
use crate::model::{self, Model, Opt, Seq};
use crate::mtp::{DevRow, MtpHead, MtpOpts};
use crate::spec::{DraftAsk, Drafter, Observe};
use serde_json::json;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct M1Config {
    pub gguf: PathBuf,
    /// 1 列の語の上限（KV の大きさ。64 の倍数に切り上がる）
    pub ctx: usize,
    pub opt: Opt,
    /// 読み込んだ後の GPU の使用がこれを超えたら起動を失敗にする（GiB。0 なら見ない）
    pub max_vram_gib: f64,
    /// プレフィルの塊がこの語数以下なら、GEMM の道（塊ごとに重みを fp16 に戻す固定の手間 約 0.3 秒）ではなく
    /// GEMV の道（gemv_max 語ずつ）で回す。短い依頼（訳・分類）の最初の 1 語を早く。0 なら使わない
    pub gemv_split_max: usize,
    /// MTP の下書きの語数（0 なら使わない）。MTP は --gguf の blk.64、埋め込みと出力は本体から借りる
    pub mtp_k: usize,
    /// 前の依頼のプロンプトの頭を使い回す（アカウントごと）
    pub prefix_cache: bool,
    /// 控えの数（全部のアカウントで。1 つ = GDN の状態ぶん約 150MB の GPU。頁は KV の置き場と分け合い、足りないと古い順に捨てる）
    pub prefix_slots: usize,
    /// 1 つのアカウントが持てる控えの数の上限（ほかのアカウントの控えを追い出し続けないように）
    pub prefix_per_account: usize,
    /// 画像の符号化器（mmproj の GGUF）。None なら画像を受けない
    pub mmproj: Option<PathBuf>,
    /// MTP の層を別の GGUF から（3.6: mtp-Qwen3.6-35B-A3B-Q8_0.gguf の blk.40）。None なら --gguf の物
    pub mtp_gguf: Option<PathBuf>,
    /// MTP の lm_head を絞る語の一覧（1 行に 1 つ。None なら全部の語）
    pub draft_vocab: Option<PathBuf>,
    /// 確かめを CUDA Graph の道で（Model::verify_step）
    pub verify_graph: bool,
    /// MTP の追いつき（プロンプトの行）を GEMM の道で
    pub gemm_catchup: bool,
    /// 下書きを打ち切る確からしさ（MtpHead::draft_min_p）
    pub draft_min_p: f32,
    pub draft_min_p_deep: f32,
    /// 同時に持つ列の数（M3）
    pub max_seqs: usize,
    /// KV の頁の置き場の大きさ（語。全部の列で分け合う。0 なら ctx）
    pub kv_tokens: usize,
    /// まとめた確かめの行の予算
    pub verify_rows: usize,
    /// 頭の控えの RAM の段（hostcache。None なら使わない）
    pub host_cache: Option<super::hostcache::HostCacheConfig>,
}

/// MTP（全部の列で 1 つ。升 = 列の升の番号）
struct Mtp {
    head: MtpHead,
    k: usize,
    graph: bool,
}

/// 列の升（M3）
struct Slot {
    seq: Seq,
    id: Option<SeqId>,
    /// 受け入れで予約した語の数（頁の数え）
    reserved: usize,
    /// プロンプトを流している間（控えの位置・流した語）
    in_prompt: bool,
    scope: String,
    snap_at: Option<usize>,
    /// RAM の段の控えを取る位置（system の終わり・最後の発言の頭・手番の頭・札の終わり。昇順）
    host_cuts: Vec<usize>,
    /// 最後に RAM の段の控えを取った（または戻した）位置（every 語ごとの控えの数え）
    last_host: usize,
    /// 先計算の札の名前（Some なら pinned・public）
    pin: Option<String>,
    /// 画像のある列（RAM の段に入れない・戻さない）
    has_media: bool,
    /// 画像の符号化の終わりの印と、それまで持っておく入力（attach_media が流し、プレフィルの流れはこの印を待つ）
    media_wait: Option<(crate::vision::Mark, Vec<crate::vision::Staged>)>,
    fed: Vec<u32>,
    /// MTP にまだ見せていない本体の行（位置・語）。h は pbuf の頭から（確かめの後の受け取った行）
    pend: Option<(usize, Vec<u32>)>,
    pbuf: DevBuf,
    steps: u64,
    drafted: u64,
    accepted: u64,
    /// HCAP の影の測り（TACHYON_HCAP_SHADOW）: 外れた段で本体の語が下書きの 2・3 番手だった回数
    /// [外れの歩, 2 番手を全部の段, 2・3 番手を全部の段, 2 番手を自信の最も低い 1 段, 2 番手を低い 2 段, 2・3 番手を最も低い 1 段]
    hcap: [u64; 6],
}

/// 頭の使い回し（M2d）: 前の依頼のプロンプトの最後の回の頭の状態の控え。M7 から何個も持つ（prefix_slots）。
/// 使えるのは同じアカウント（scope）の依頼だけ。頁は、控えを取った列が終わったら控えに移す（owner → pages）。
/// 受け入れの頁が足りないときは、使っていない控えを古い順に捨てる。新しい控えは、同じアカウントの「頭が同じで短い」控え →
/// そのアカウントが上限（prefix_per_account）なら、その中のいちばん古い物 → 空き → 全体でいちばん古い物、の順に上書きする
struct PrefixCache {
    scope: String,
    toks: Vec<u32>,
    state: DevBuf,
    conv: DevBuf,
    /// 位置 len-1 の本体の h_nextn（MTP の続きに要る）
    h: DevBuf,
    valid: bool,
    /// 控えの頁を今も使っている列（終わるまで使い回せない）
    owner: Option<SeqId>,
    pages: Vec<u32>,
    hits: u64,
    reused_tokens: u64,
    /// 最後に取った・使った時（engine の中の数え）
    used: u64,
}

pub struct M1Engine {
    m: Model,
    slots: Vec<Slot>,
    pcs: Vec<PrefixCache>,
    /// 頭の控えの RAM の段（hostcache）
    host: Option<super::hostcache::HostCache>,
    /// RAM の段から戻した h（MTP の続き。d 個の f32）
    hrest: DevBuf,
    pc_tick: u64,
    pc_per_account: usize,
    /// 本体の h_nextn: プレフィル・確かめの行 [chunk][d]、まとめたデコードの行 [32][d]
    hbuf: DevBuf,
    /// プレフィルの行の h_nextn [chunk][d]（プレフィルの流れが書く。M3 で確かめの hbuf と分けた）
    hpre: DevBuf,
    /// 流れている（まだ MTP に見せていない）プレフィルの塊: (升, 位置, 語)。終わったら poll_pre が MTP に見せる
    pre: Option<(usize, usize, Vec<u32>)>,
    /// プレフィルを別の流れで非同期に（max_seqs > 1）
    async_pre: bool,
    hdec: DevBuf,
    gemv_split_max: usize,
    /// 選んでよい語の番号の上限（GGUF の [PAD…] は選ばない）
    valid_vocab: usize,
    poisoned: Option<String>,
    mtp: Option<Mtp>,
    /// 画像の符号化器（--mmproj）と、その流れ
    vision: Option<crate::vision::Vision>,
    vis_s: crate::cuda::ffi::Stream,
    max_ctx: usize,
    /// まとめた確かめの行の予算（列の数 × (下書き＋1)。GEMV の組が 4→8 本で 1 歩が跳ぶので既定 8）
    verify_rows: usize,
    /// GPU の候補で決められず logits 全部を手元へ写した行の数（確かめ用）
    cand_fallback: u64,
    /// そのうち JSON の形の絞りを語彙全部にかけた行の数（起動からの累計。mtp の記録に出す）
    mask_full: u64,
}

// DevBuf（GPU の生のポインタ）は Send でないが、M1Engine は順番決めの糸の中で作られ、その糸の外へ出ない
// （SimpleScheduler の loader と step・release は同じ糸）。CUDA の primary context はプロセスに 1 つなので、どの糸からでも同じポインタが使える
unsafe impl Send for M1Engine {}

fn panic_msg(p: Box<dyn std::any::Any + Send>) -> String {
    match p.downcast::<String>() {
        Ok(s) => *s,
        Err(p) => p.downcast::<&str>().map(|s| s.to_string()).unwrap_or_else(|_| "panic".into()),
    }
}

fn gib(b: usize) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// ログ 1 行（口のログと同じ JSON。中身は入れない）
fn log(v: serde_json::Value) {
    let mut v = v;
    v["t"] = super::util::utc_now().into();
    eprintln!("{v}");
}

/// GGUF の語の型（tokenizer.ggml.token_type）で、最後の本物の語（UNUSED = 5 でない）の次の番号
fn valid_vocab_of(g: &gguf::Model, n_vocab: usize) -> usize {
    match g.meta_arr_i64("tokenizer.ggml.token_type") {
        Some(t) => t.iter().rposition(|&x| x != 5).map(|i| i + 1).unwrap_or(n_vocab).min(n_vocab),
        None => n_vocab,
    }
}

/// 重みを読んで M1Engine を作る関数（SimpleScheduler の糸の中で呼ばれる）
pub fn loader(cfg: M1Config) -> EngineLoader {
    Box::new(move |progress: &dyn Fn(f32, &str)| {
        let t0 = Instant::now();
        let r = catch_unwind(AssertUnwindSafe(|| -> Result<M1Engine, String> {
            progress(0.01, "cuda");
            crate::cuda::check(unsafe { crate::cuda::ffi::cudaSetDevice(0) }, "cudaSetDevice");
            let (f0, tot) = model::mem_info();
            progress(0.05, "gguf");
            let g = gguf::Model::open(&cfg.gguf).map_err(|e| format!("{}: {e}", cfg.gguf.display()))?;
            progress(0.1, "weights");
            let mut m = Model::load(&g, cfg.opt, None)?;
            let n = cfg.max_seqs.clamp(1, model::GEMV_B_MAX);
            let ctx = cfg.ctx.div_ceil(64) * 64;
            let kv_tokens = if cfg.kv_tokens > 0 { cfg.kv_tokens.div_ceil(64) * 64 } else { ctx };
            progress(0.85, "mtp");
            let vision = match &cfg.mmproj {
                Some(p) => {
                    let v = crate::vision::Vision::load(&p.to_string_lossy())?;
                    log(json!({"ev": "mmproj_loaded", "gib": gib(v.bytes)}));
                    Some(v)
                }
                None => None,
            };
            let mtp = if cfg.mtp_k > 0 {
                let k = cfg.mtp_k.min(model::GEMV_B_MAX - 1).min(m.opt.gemv_max.saturating_sub(1)).max(1);
                m.enable_spec((k + 1).max(cfg.verify_rows.min(model::GEMV_B_MAX)));
                let share = m.mtp_shared(&g);
                let mg = match &cfg.mtp_gguf {
                    Some(p) => Some(gguf::Model::open(p).map_err(|e| format!("{}: {e}", p.display()))?),
                    None => None,
                };
                let mut head = MtpHead::load(mg.as_ref().unwrap_or(&g), MtpOpts { slots: n, max_ctx: ctx, topk: 1, kv_pages: kv_tokens / 64 }, share)?;
                if let Some(p) = &cfg.draft_vocab {
                    let txt = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
                    let ids: Vec<u32> = txt
                        .lines()
                        .map(|l| l.split('#').next().unwrap().trim())
                        .filter(|l| !l.is_empty())
                        .map(|l| l.parse::<u32>().map_err(|_| format!("{}: 数でない行", p.display())))
                        .collect::<Result<_, _>>()?;
                    head.set_draft_vocab(&g, &ids)?;
                }
                head.gemm_catchup = cfg.gemm_catchup;
                head.draft_min_p = cfg.draft_min_p;
                head.draft_min_p_deep = cfg.draft_min_p_deep;
                Some(Mtp { head, k, graph: cfg.verify_graph })
            } else {
                None
            };
            progress(0.9, "kv");
            let layers = 0..m.cfg.n_layer;
            let seqs = Seq::new_paged(&m.cfg, layers, ctx, n, mtp.is_some(), kv_tokens);
            let d = m.cfg.d;
            let pr = mtp.as_ref().map_or(1, |x| x.k + 1);
            let slots: Vec<Slot> = seqs
                .into_iter()
                .map(|seq| Slot {
                    seq,
                    id: None,
                    reserved: 0,
                    in_prompt: false,
                    scope: String::new(),
                    snap_at: None,
                    host_cuts: vec![],
                    last_host: 0,
                    pin: None,
                    has_media: false,
                    media_wait: None,
                    fed: vec![],
                    pend: None,
                    pbuf: DevBuf::new(pr * d * 4),
                    steps: 0,
                    drafted: 0,
                    accepted: 0,
                    hcap: [0; 6],
                })
                .collect();
            let hbuf = DevBuf::new(m.opt.chunk.max(pr).max(model::GEMV_B_MAX) * d * 4);
            let hpre = DevBuf::new(m.opt.chunk.max(model::GEMV_B_MAX) * d * 4);
            let async_pre = n > 1 && std::env::var("TACHYON_NO_ASYNC").map_or(true, |v| v != "1");
            if async_pre {
                m.enable_async_prefill();
            }
            let hdec = DevBuf::new(model::GEMV_B_MAX * d * 4);
            let n_pc = if cfg.prefix_cache { cfg.prefix_slots.max(1) } else { 0 };
            let (sa, sb) = slots[0].seq.state_bytes();
            let pcs: Vec<PrefixCache> = (0..n_pc)
                .map(|_| PrefixCache {
                    scope: String::new(),
                    toks: vec![],
                    state: DevBuf::new(sa),
                    conv: DevBuf::new(sb),
                    h: DevBuf::new(d * 4),
                    valid: false,
                    owner: None,
                    pages: vec![],
                    hits: 0,
                    reused_tokens: 0,
                    used: 0,
                })
                .collect();
            // 頭の控えの RAM の段: 1 頁 = 本体の Attention の層の K・V ＋ MTP の K・V
            let host = cfg.host_cache.clone().filter(|h| h.cap_bytes > 0).map(|mut hc| {
                let mut fb: usize = slots[0].seq.kv_page_spans(0).iter().map(|x| x.1).sum();
                if let Some(x) = mtp.as_ref() {
                    fb += x.head.kv_page_spans(0).iter().map(|x| x.1).sum::<usize>();
                }
                if hc.node.is_none() {
                    hc.node = crate::kv::gpu_numa_node(0);
                }
                super::hostcache::HostCache::new(hc, fb, (sa, sb, d * 4))
            });
            let hrest = DevBuf::new(d * 4);
            crate::cuda::sync();
            let (f1, _) = model::mem_info();
            let used = gib(tot - f1);
            let valid_vocab = valid_vocab_of(&g, m.cfg.vocab);
            log(json!({
                "ev": "engine_loaded", "secs": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0, "ctx": ctx, "chunk": m.opt.chunk,
                "max_seqs": n, "kv_tokens": kv_tokens,
                "weights_gib": gib(m.info.bytes_gpu), "scratch_gib": gib(m.info.scratch_bytes), "seq_gib": gib(slots[0].seq.bytes() * n),
                "ours_gib": gib(f0 - f1), "gpu_used_gib": used, "gpu_total_gib": gib(tot), "vocab": m.cfg.vocab, "valid_vocab": valid_vocab,
                "mtp_k": mtp.as_ref().map_or(0, |x| x.k), "mtp_gib": mtp.as_ref().map_or(0.0, |x| gib(x.head.vram_bytes())),
                "mtp_draft_vocab": mtp.as_ref().and_then(|x| x.head.draft_vocab.as_ref()).map_or(0, |(_, ids)| ids.len()),
                "verify_graph": mtp.as_ref().is_some_and(|x| x.graph),
                "prefix_slots": n_pc, "prefix_per_account": cfg.prefix_per_account.max(1), "prefix_gib": gib(n_pc * (sa + sb + d * 4)),
                "host_cache_cap_gib": host.as_ref().map_or(0.0, |_| gib(cfg.host_cache.as_ref().map_or(0, |h| h.cap_bytes))),
            }));
            if cfg.max_vram_gib > 0.0 && used > cfg.max_vram_gib {
                return Err(format!("GPU の使用 {used:.2} GiB が上限 {:.1} GiB を超える（--ctx・--kv-tokens・--max-seqs を小さく）", cfg.max_vram_gib));
            }
            Ok(M1Engine { m, slots, pcs, host, hrest, pc_tick: 0, pc_per_account: cfg.prefix_per_account.max(1), hbuf, hpre, pre: None, async_pre, hdec, gemv_split_max: cfg.gemv_split_max, valid_vocab, poisoned: None, mtp, vis_s: if vision.is_some() { crate::vision::stream_new() } else { std::ptr::null_mut() }, vision, max_ctx: ctx, verify_rows: cfg.verify_rows.max(2), cand_fallback: 0, mask_full: 0 })
        }));
        match r {
            Ok(Ok(e)) => Ok(Box::new(e) as Box<dyn Engine>),
            Ok(Err(m)) => Err(m),
            Err(p) => Err(format!("engine load panicked: {}", panic_msg(p))),
        }
    })
}

/// 語の数 → 頁の数
fn pages_for(tokens: usize) -> usize {
    tokens.div_ceil(64)
}

impl M1Engine {
    fn slot_of(&self, id: SeqId) -> Option<usize> {
        self.slots.iter().position(|s| s.id == Some(id))
    }

    /// まだ使える頁（空き＋控えの頁 − 走っている列の予約の残り）
    fn pages_available(&self) -> isize {
        let free = self.slots[0].seq.free_page_count() as isize;
        let pc = self.pcs.iter().map(|p| p.pages.len()).sum::<usize>() as isize;
        let owed: isize = self.slots.iter().filter(|s| s.id.is_some()).map(|s| pages_for(s.reserved) as isize - s.seq.n_pages_held() as isize).filter(|&x| x > 0).sum();
        free + pc - owed
    }

    /// 控え j の頁を返す（控えは使えなくなる。列が使っている間の控えは触らない）
    fn evict_at(&mut self, j: usize) {
        let pc = &mut self.pcs[j];
        if pc.owner.is_some() {
            return;
        }
        let pg = std::mem::take(&mut pc.pages);
        pc.valid = false;
        pc.toks.clear();
        if !pg.is_empty() {
            self.slots[0].seq.return_pages(pg);
        }
    }

    /// 頁を持っている控えのうち、いちばん古い物を捨てる（捨てる物が無ければ false）
    fn evict_oldest(&mut self) -> bool {
        let j = (0..self.pcs.len()).filter(|&j| self.pcs[j].owner.is_none() && !self.pcs[j].pages.is_empty()).min_by_key(|&j| self.pcs[j].used);
        match j {
            Some(j) => {
                self.evict_at(j);
                true
            }
            None => false,
        }
    }

    /// 新しい控えを置く場所（列 i の scope・toks）。どれも列が使っている間なら None
    fn pc_target(&self, scope: &str, toks: &[u32]) -> Option<usize> {
        let free = |j: &usize| self.pcs[*j].owner.is_none();
        let mine: Vec<usize> = (0..self.pcs.len()).filter(|&j| self.pcs[j].scope == scope && (self.pcs[j].valid || self.pcs[j].owner.is_some())).collect();
        // 同じ会話の前の回（頭が同じで短い）は、新しい控えで用が済む
        if let Some(&j) = mine.iter().filter(|j| free(j) && self.pcs[**j].valid && toks.starts_with(&self.pcs[**j].toks)).max_by_key(|&&j| self.pcs[j].toks.len()) {
            return Some(j);
        }
        if mine.len() >= self.pc_per_account {
            return mine.iter().copied().filter(free).min_by_key(|&j| self.pcs[j].used);
        }
        if let Some(j) = (0..self.pcs.len()).find(|&j| free(&j) && !self.pcs[j].valid) {
            return Some(j);
        }
        (0..self.pcs.len()).filter(free).min_by_key(|&j| self.pcs[j].used)
    }

    /// 列 i の MTP の頁表を本体と同じに
    fn mtp_pages(&mut self, i: usize) -> Result<(), String> {
        if let Some(x) = self.mtp.as_mut() {
            let pages: Vec<u32> = self.slots[i].seq.pages_vec();
            x.head.set_pages(i as u32, &pages)?;
        }
        Ok(())
    }

    /// 列 i を空ける（release・失敗）
    fn clear_slot(&mut self, i: usize) {
        // 確定の流れがこの列の状態を書いている間に頁・状態を渡さない
        self.m.sync_commit();
        // この列のプレフィルがまだ流れていれば待つ（頁を返す前に）。MTP には見せない
        if self.pre.as_ref().is_some_and(|p| p.0 == i) {
            self.m.sync_prefill();
            self.pre = None;
        }
        let s = &mut self.slots[i];
        if s.steps > 0 {
            log(json!({"ev": "mtp", "steps": s.steps, "drafted": s.drafted, "accepted": s.accepted,
                       "tokens_per_step": (s.steps + s.accepted) as f64 / s.steps as f64, "hcap": s.hcap,
                       "cand_fallback_total": self.cand_fallback, "mask_full_total": self.mask_full}));
        }
        let id = s.id.take();
        s.pend = None;
        // 画像: 符号化が終わるのを待ってから入力と結果を捨てる（Mark の drop が待つ）
        s.media_wait = None;
        s.seq.media.clear();
        (s.steps, s.drafted, s.accepted) = (0, 0, 0);
        s.hcap = [0; 6];
        s.reserved = 0;
        if let Some(x) = self.mtp.as_mut() {
            x.head.free(i as u32);
        }
        // 控えの頁をこの列が持っていれば、控えへ移す（控えの位置までの頁だけ）
        let mut to_pc = None;
        let in_prompt = self.slots[i].in_prompt;
        for (j, pc) in self.pcs.iter_mut().enumerate() {
            if pc.owner.is_some() && pc.owner == id {
                pc.owner = None;
                if in_prompt {
                    pc.valid = false; // 控えを取る前に終わった（取り消し）
                    pc.toks.clear();
                } else {
                    to_pc = Some(j);
                }
            }
        }
        let s = &mut self.slots[i];
        s.in_prompt = false;
        let mut pages = s.seq.take_pages_any();
        if let Some(j) = to_pc {
            let pc = &mut self.pcs[j];
            let keep = pages_for(pc.toks.len()).min(pages.len());
            pc.pages = pages.drain(..keep).collect();
        }
        self.slots[0].seq.return_pages(pages);
    }

    /// 控えから戻す（reuse の中身）
    fn reuse_inner(&mut self, i: usize, j: usize) -> Result<usize, String> {
        self.m.sync_commit();
        let pc = &mut self.pcs[j];
        let l = pc.toks.len();
        let pages = std::mem::take(&mut pc.pages);
        let s = &mut self.slots[i];
        s.seq.reset();
        s.seq.give_pages(pages);
        s.seq.load_state(&pc.state, &pc.conv, l);
        s.fed = pc.toks.clone();
        // 控えを取る位置（手番の頭）が使い回した所の中なら、この列は新しい控えを取らない → 同じ控えを生かしておき、
        // 列が終わったら頁を返してもらう（頭の l 語の KV・状態は書き換わらない）。そうでなければ控えは使い切り
        let keep = s.snap_at.is_some_and(|a| a <= l);
        if keep {
            pc.owner = s.id;
            s.snap_at = Some(usize::MAX);
        } else {
            pc.valid = false;
            pc.toks.clear();
        }
        pc.used = self.pc_tick + 1;
        pc.hits += 1;
        pc.reused_tokens += l as u64;
        s.in_prompt = true;
        let h = pc.h.f32p();
        if let Some(x) = self.mtp.as_mut() {
            x.head.free(i as u32);
            let pg = self.slots[i].seq.pages_vec();
            x.head.set_pages(i as u32, &pg)?;
            x.head.set_prefix(i as u32, l, h)?;
        }
        Ok(l)
    }

    /// 列 i のプロンプトの控えを取る（row = hbuf の中のその位置の行）
    fn snapshot(&mut self, i: usize, row: usize) {
        self.m.sync_commit();
        let Some(j) = self.pc_target(&self.slots[i].scope, &self.slots[i].fed) else {
            return;
        };
        // 置く場所の前の控えの頁は返す
        self.evict_at(j);
        self.pc_tick += 1;
        let tick = self.pc_tick;
        let d = self.m.cfg.d;
        let has_mtp = self.mtp.is_some();
        self.m.sync_prefill();
        let hb = self.hpre.f32p();
        let s = &self.slots[i];
        let pc = &mut self.pcs[j];
        pc.used = tick;
        s.seq.save_state(&pc.state, &pc.conv);
        if has_mtp {
            crate::cuda::check(unsafe { crate::cuda::ffi::cudaMemcpy(pc.h.ptr, hb.add(row * d).cast(), d * 4, 3) }, "h の控え");
        }
        pc.scope = s.scope.clone();
        pc.toks = s.fed.clone();
        pc.valid = true;
        pc.owner = s.id;
    }

    /// 直前の forward（プレフィル）の n 行の h_nextn を hpre の off 行目から（プレフィルの流れの上で）
    /// 抜き取りの候補（GPU の上位 K 語）。TACHYON_GPU_SAMPLE=0 か扱えない形なら None（手元で logits 全部から）
    fn row_cands(&self, dec: bool, n: usize, temps: &[f32]) -> Option<Vec<sampler::RowCands>> {
        if !sampler::gpu_sample_on() {
            return None;
        }
        self.m.cands(n, dec, self.valid_vocab, temps)
    }

    fn keep_h(&mut self, off: usize, n: usize) {
        if self.mtp.is_some() {
            let d = self.m.cfg.d;
            self.m.out_norm_rows(n, unsafe { self.hpre.f32p().add(off * d) });
        }
    }

    /// 流れているプレフィルの塊を待って、MTP に見せる
    fn complete_pre(&mut self) -> Result<(), String> {
        if let Some((i, p0, toks)) = self.pre.take() {
            self.m.sync_prefill();
            if self.mtp.is_some() {
                self.mtp_pages(i)?;
                let d = self.m.cfg.d;
                let h = Self::rows_at(self.hpre.f32p(), toks.len(), d);
                self.mtp.as_mut().unwrap().head.run(&[Observe { seq: i as u32, pos0: p0, tokens: &toks, h: &h }], &[])?;
            }
        }
        Ok(())
    }

    /// プレフィルの塊が終わっていれば仕上げる（待たない）
    fn poll_pre(&mut self) -> Result<(), String> {
        if self.pre.is_some() && self.m.prefill_idle() {
            self.complete_pre()?;
        }
        Ok(())
    }

    fn rows_at(p: *mut f32, n: usize, d: usize) -> Vec<DevRow> {
        (0..n).map(|i| DevRow(unsafe { p.add(i * d) as *const f32 })).collect()
    }

    /// 列 i の語を流す（短い塊は GEMV の道で gemv_max 語ずつ）。h_nextn を hbuf へ、MTP にすぐ見せる。
    /// 返り値は作業場に残った最後の forward の行の数（head はその最後の行）
    fn forward_part(&mut self, i: usize, toks: &[u32], defer: bool) -> Result<usize, String> {
        // hpre を使い回すので、前の塊を先に仕上げる
        self.complete_pre()?;
        // 画像の符号化がまだなら、プレフィルの流れにその終わりを待たせる（終わっていれば入力を捨てる）
        if let Some((mk, _)) = &self.slots[i].media_wait {
            if mk.done() {
                self.slots[i].media_wait = None;
            } else {
                self.m.prefill_wait(mk.raw());
            }
        }
        let t = toks.len();
        let g = self.m.opt.gemv_max.max(1);
        let pos0 = self.slots[i].seq.pos;
        let last = if t > g && t <= self.gemv_split_max {
            let mut n = 0;
            let mut off = 0;
            for ch in toks.chunks(g) {
                self.m.forward(&mut self.slots[i].seq, ch, None);
                self.keep_h(off, ch.len());
                off += ch.len();
                n = ch.len();
            }
            n
        } else {
            self.m.forward(&mut self.slots[i].seq, toks, None);
            self.keep_h(0, t);
            t
        };
        // 非同期なら流しっぱなしで返る（ほかの人のデコードと重ねる）。MTP には終わってから見せる
        self.pre = Some((i, pos0, toks.to_vec()));
        if !(defer && self.async_pre) {
            self.complete_pre()?;
        }
        Ok(last)
    }

    /// 列 i のまだ見せていない行を MTP に（他の列の歩の前に）
    fn flush_pend(&mut self, i: usize) -> Result<(), String> {
        if self.mtp.is_none() {
            return Ok(());
        }
        if let Some((p0, toks)) = self.slots[i].pend.take() {
            self.mtp_pages(i)?;
            let d = self.m.cfg.d;
            let h = Self::rows_at(self.slots[i].pbuf.f32p(), toks.len(), d);
            self.mtp.as_mut().unwrap().head.run(&[Observe { seq: i as u32, pos0: p0, tokens: &toks, h: &h }], &[])?;
        }
        Ok(())
    }

    /// 列 i のプレフィルの塊（またはデコードの 1 語を下書きなしで）
    fn prefill_item(&mut self, i: usize, it: &StepItem) -> Result<ItemOut, String> {
        self.flush_pend(i)?;
        let t = it.tokens.len();
        let pos0 = self.slots[i].seq.pos;
        // 切る所: GPU の控え（snap_at）と RAM の段の区切り（host_cuts）。塊の中にある物だけ
        let mut bounds: Vec<usize> = vec![];
        if self.slots[i].in_prompt {
            if let Some(a) = self.slots[i].snap_at {
                if a > pos0 && a < pos0 + t {
                    bounds.push(a - pos0);
                }
            }
            if !self.slots[i].has_media {
                bounds.extend(self.slots[i].host_cuts.iter().filter(|&&c| c > pos0 && c < pos0 + t).map(|&c| c - pos0));
            }
        }
        bounds.sort_unstable();
        bounds.dedup();
        bounds.push(t);
        let every = self.host.as_ref().map_or(0, |h| h.every());
        let host_on = self.host.is_some() && !self.slots[i].has_media;
        let mut last = 0;
        let mut beg = 0;
        let nb = bounds.len();
        for (pi, &end) in bounds.iter().enumerate() {
            let part = &it.tokens[beg..end];
            beg = end;
            if part.is_empty() {
                continue;
            }
            let want_part = it.want && pi == nb - 1;
            let at = self.slots[i].seq.pos + part.len();
            let sa = self.slots[i].snap_at;
            let snap = self.slots[i].in_prompt && (sa == Some(at) || (want_part && sa.is_none()));
            // RAM の段: 区切りの所・前の控えから every 語を超えた塊の切れ目（⛔ 答えを出す塊の終わりは札のときだけ）
            let s = &self.slots[i];
            let hsnap = host_on
                && s.in_prompt
                && (s.host_cuts.contains(&at) || (every > 0 && !want_part && at >= s.last_host + every));
            // 待たずに返してよいのは、答えも控えも要らない塊だけ
            last = self.forward_part(i, part, !want_part && !snap && !hsnap)?;
            if self.slots[i].in_prompt {
                self.slots[i].fed.extend_from_slice(part);
                if snap {
                    self.snapshot(i, part.len() - 1);
                }
                if hsnap {
                    self.host_ckpt(i, part.len() - 1);
                }
            }
        }
        if self.slots[i].in_prompt && it.want {
            self.slots[i].in_prompt = false;
        }
        if !it.want {
            return Ok(ItemOut::None);
        }
        self.m.head(last - 1, 1);
        if it.sampling.is_greedy() && !it.sampling.has_penalty() && it.constraint.is_none() {
            let id = self.m.argmax(1)[0];
            if (id as usize) < self.valid_vocab {
                return Ok(ItemOut::Token(id));
            }
        }
        Ok(ItemOut::Logits(self.m.logits(1)))
    }

    /// 列 i のデコードの 1 歩を MTP の下書きで: 下書き k 語 → 確かめ → 受け取り → commit
    fn spec_step(&mut self, i: usize, it: &StepItem) -> Result<ItemOut, String> {
        let d = self.m.cfg.d;
        let v = self.m.cfg.vocab;
        let valid = self.valid_vocab;
        let p0 = self.slots[i].seq.pos;
        let last = it.tokens[0];
        // 下書きは予約と文脈の中で
        let mut room = self.slots[i].reserved.max(p0 + 1).min(self.max_ctx).saturating_sub(p0).max(1);
        // 下書きの行（位置 p0..p0+k）の頁を先に取る（MTP の KV も同じ頁に書く。取らずに書くと頁 0 に落ちて、ほかの列の KV を壊し下書きも外れる）
        let kmax = self.mtp.as_ref().map_or(0, |x| x.k);
        while room > 1 && !self.slots[i].seq.ensure(p0 + kmax.min(room - 1) + 1) {
            room -= 1;
        }
        self.mtp_pages(i)?;
        let pend = self.slots[i].pend.take();
        let pb = self.slots[i].pbuf.f32p();
        let x = self.mtp.as_mut().unwrap();
        let k = x.k.min(room - 1);
        // 1. MTP: 見せていない本体の行＋下書きの頭を 1 回で、続けて k 段
        let mut alts: Vec<[u32; 2]> = vec![];
        let mut confs: Vec<f32> = vec![];
        let mut drafts = if k == 0 {
            if let Some((pp, ptoks)) = pend {
                let h = Self::rows_at(pb, ptoks.len(), d);
                x.head.run(&[Observe { seq: i as u32, pos0: pp, tokens: &ptoks, h: &h }], &[])?;
            }
            vec![]
        } else {
            let ask = DraftAsk { seq: i as u32, id_last: last, pos: p0, k, sample: None, gumbel: hash_draft(&it.sampling) };
            let out = match pend {
                Some((pp, ptoks)) => {
                    let h = Self::rows_at(pb, ptoks.len(), d);
                    x.head.run(&[Observe { seq: i as u32, pos0: pp, tokens: &ptoks, h: &h }], &[ask])?
                }
                None => x.head.run(&[], &[ask])?,
            };
            let o = out.into_iter().next().ok_or("MTP の下書きが無い")?;
            alts = o.alts;
            confs = o.conf;
            o.tokens
        };
        if let Some(j) = drafts.iter().position(|&t| t as usize >= valid) {
            drafts.truncate(j);
        }
        let k = drafts.len();
        let mut toks = Vec::with_capacity(k + 1);
        toks.push(last);
        toks.extend_from_slice(&drafts);
        let t = toks.len();
        // 2. 本体の確かめ（t 語。GDN の新しい状態は仮の置き場へ）
        let graph = x.graph;
        let hb = self.hbuf.f32p();
        let am_graph = if t >= 2 && graph {
            Some(self.m.verify_step(&mut self.slots[i].seq, &toks, hb))
        } else {
            if t >= 2 {
                self.m.forward_verify(&mut self.slots[i].seq, &toks);
            } else {
                self.m.forward(&mut self.slots[i].seq, &toks, None);
            }
            self.keep_h(0, t);
            self.m.head(0, t);
            None
        };
        // 3. 受け取り
        let sp = &*it.sampling;
        let mut n_acc = 0;
        let new;
        // グラフの道はデコードの作業場に、そうでなければプレフィルの作業場にロジットがある
        let gr = am_graph.is_some();
        let logits_of = |m: &Model, n: usize| if gr { m.logits_dec(n) } else { m.logits(n) };
        if sp.is_greedy() && !sp.has_penalty() && it.constraint.is_none() {
            let am = am_graph.unwrap_or_else(|| self.m.argmax(t));
            while n_acc < k && am[n_acc] == drafts[n_acc] {
                n_acc += 1;
            }
            new = if (am[n_acc] as usize) < valid {
                am[n_acc]
            } else {
                let lg = logits_of(&self.m, n_acc + 1);
                sampler::sample(&lg[n_acc * v..(n_acc + 1) * v], valid, sp, &History::default(), (p0 + n_acc + 1) as u64)
            };
        } else {
            let cands = self.row_cands(gr, t, &vec![sp.temperature.max(1e-3); t]);
            let mut full: Option<Vec<f32>> = None;
            let mut hist = it.hist.clone().unwrap_or_default();
            // JSON の形: 行 j の絞りは、受け取った下書き 0..j を流した後の状態（外れた下書きは確率 0 で外れる）
            let mut cm: Option<crate::server::jsonc::Matcher> = it.constraint.as_deref().cloned();
            let mut y = None;
            for j in 0..t {
                let pc = match &cm {
                    Some(m) => cands.as_ref().and_then(|c| sampler::prepare_cands_masked(&c[j], sp, &hist, m)),
                    None => cands.as_ref().and_then(|c| sampler::prepare_cands(&c[j], sp, &hist)),
                };
                let pr = match pc {
                    Some(pr) => pr,
                    None => {
                        self.cand_fallback += 1;
                        let lg = full.get_or_insert_with(|| logits_of(&self.m, t));
                        match &cm {
                            Some(m) => {
                                self.mask_full += 1;
                                sampler::prepare_masked(&lg[j * v..(j + 1) * v], valid, sp, &hist, m)
                            }
                            None => sampler::prepare(&lg[j * v..(j + 1) * v], valid, sp, &hist),
                        }
                    }
                };
                match sampler::spec_row_prepared(pr, sp, (p0 + j + 1) as u64, drafts.get(j).copied()) {
                    Ok(()) => {
                        hist.push(drafts[j]);
                        if let Some(m) = &mut cm {
                            m.advance(drafts[j]);
                        }
                        n_acc += 1;
                    }
                    Err(tok) => {
                        y = Some(tok);
                        break;
                    }
                }
            }
            new = y.expect("最後の行は必ず語を出す");
        }
        // 4. 受け取った分だけ確定
        if t >= 2 {
            if gr && commit_stream() {
                self.m.commit_multi(&mut [&mut self.slots[i].seq], t, &[n_acc + 1]);
            } else {
                self.m.commit_verify(&mut self.slots[i].seq, t, n_acc + 1);
            }
        }
        // 受け取った行の h を列の置き場へ（次の歩の下書きと 1 回で MTP に見せる）。グラフでない道は hpre に書いている
        let n = n_acc + 1;
        let src = if gr { hb } else { self.m.sync_prefill(); self.hpre.f32p() };
        crate::cuda::check(unsafe { crate::cuda::ffi::cudaMemcpy(pb.cast(), src.cast(), n * d * 4, 3) }, "h の写し");
        let s = &mut self.slots[i];
        s.pend = Some((p0, toks[..n].to_vec()));
        s.steps += 1;
        s.drafted += k as u64;
        s.accepted += n_acc as u64;
        if n_acc < k && alts.len() >= k && confs.len() >= k {
            // HCAP の影: 段 n_acc で外れた。そこに 2・3 番手を並べていたら、本体の語 new がそれに当たり、1 語多く確定した
            let [c2, c3] = alts[n_acc];
            let mut ord: Vec<usize> = (0..k).collect();
            ord.sort_by(|&a, &b| confs[a].partial_cmp(&confs[b]).unwrap_or(std::cmp::Ordering::Equal));
            let (h2, h3) = (new == c2, new == c3);
            s.hcap[0] += 1;
            s.hcap[1] += h2 as u64;
            s.hcap[2] += (h2 || h3) as u64;
            s.hcap[3] += (h2 && ord[0] == n_acc) as u64;
            s.hcap[4] += (h2 && ord[..2.min(k)].contains(&n_acc)) as u64;
            s.hcap[5] += ((h2 || h3) && ord[0] == n_acc) as u64;
        }
        let mut outv = drafts[..n_acc].to_vec();
        outv.push(new);
        Ok(ItemOut::Tokens(outv))
    }

    /// 何人ものデコードの 1 語をまとめて 1 歩（下書きなし）。MTP は毎歩追いつかせる
    fn batch_decode(&mut self, idx: &[(usize, usize)], items: &[StepItem], out: &mut [ItemOut]) -> Result<(), String> {
        let b = idx.len();
        let d = self.m.cfg.d;
        let toks: Vec<u32> = idx.iter().map(|&(k, _)| items[k].tokens[0]).collect();
        let p0s: Vec<usize> = idx.iter().map(|&(_, i)| self.slots[i].seq.pos).collect();
        let hd = self.hdec.f32p();
        let am = {
            // 列は升ごとに別（idx の升は重ならない）
            let mut seen = vec![false; self.slots.len()];
            for &(_, i) in idx {
                assert!(!std::mem::replace(&mut seen[i], true), "同じ列が 2 回");
            }
            let base = self.slots.as_mut_ptr();
            let mut refs: Vec<&mut Seq> = idx.iter().map(|&(_, i)| unsafe { &mut (*base.add(i)).seq }).collect();
            let hout = if self.mtp.is_some() { hd } else { std::ptr::null_mut() };
            self.m.decode_h(&mut refs, &toks, hout)
        };
        let need_logits = idx.iter().any(|&(k, _)| !(items[k].sampling.is_greedy() && !items[k].sampling.has_penalty()) || items[k].constraint.is_some())
            || am.iter().any(|&a| a as usize >= self.valid_vocab);
        // 抜き取り: GPU の上位 K 語の候補で（候補で決められない行だけ logits 全部を手元へ）
        let temps: Vec<f32> = idx.iter().map(|&(k, _)| items[k].sampling.temperature.max(1e-3)).collect();
        let cands = if need_logits { self.row_cands(true, b, &temps) } else { None };
        let mut full: Option<Vec<f32>> = None;
        let v = self.m.cfg.vocab;
        let none = History::default();
        for (r, &(k, _)) in idx.iter().enumerate() {
            let it = &items[k];
            out[k] = if it.sampling.is_greedy() && !it.sampling.has_penalty() && (am[r] as usize) < self.valid_vocab && it.constraint.is_none() {
                ItemOut::Token(am[r])
            } else {
                let sp = &*it.sampling;
                let hist = it.hist.as_ref().unwrap_or(&none);
                let pos = (p0s[r] + 1) as u64;
                let cm = it.constraint.as_deref();
                let pc = match cm {
                    Some(m) => cands.as_ref().and_then(|c| sampler::prepare_cands_masked(&c[r], sp, hist, m)).map(|pr| pr.pick(&sp, pos)),
                    None => cands.as_ref().and_then(|c| sampler::sample_cands(&c[r], sp, hist, pos)),
                };
                match pc {
                    Some(t) => ItemOut::Token(t),
                    None => {
                        self.cand_fallback += 1;
                        let lg = full.get_or_insert_with(|| self.m.logits_dec(b));
                        let row = &lg[r * v..(r + 1) * v];
                        ItemOut::Token(match cm {
                            Some(m) => {
                                self.mask_full += 1;
                                sampler::prepare_masked(row, self.valid_vocab, sp, hist, m).pick(&sp, pos)
                            }
                            None => sampler::sample(row, self.valid_vocab, sp, hist, pos),
                        })
                    }
                }
            };
        }
        // MTP: 列ごとに、見せていない行＋この歩の行
        if self.mtp.is_some() {
            let mut obs_data: Vec<(u32, usize, Vec<u32>, Vec<DevRow>)> = vec![];
            for (r, &(_, i)) in idx.iter().enumerate() {
                self.mtp_pages(i)?;
                let (mut pos0, mut tk, mut hs) = (p0s[r], vec![], vec![]);
                if let Some((pp, pt)) = self.slots[i].pend.take() {
                    pos0 = pp;
                    hs = Self::rows_at(self.slots[i].pbuf.f32p(), pt.len(), d);
                    tk = pt;
                }
                tk.push(toks[r]);
                hs.push(DevRow(unsafe { hd.add(r * d) as *const f32 }));
                obs_data.push((i as u32, pos0, tk, hs));
            }
            let obs: Vec<Observe<DevRow>> = obs_data.iter().map(|(s, p, t, h)| Observe { seq: *s, pos0: *p, tokens: t, h }).collect();
            self.mtp.as_mut().unwrap().head.run(&obs, &[])?;
        }
        Ok(())
    }

    /// 何人ものデコードを MTP の下書きでまとめて: 下書き（列ごとに k_b 語・1 回の run）→ 確かめ（verify_multi・行は [列][T]）→ 列ごとに受け取り → commit
    fn spec_multi(&mut self, idx: &[(usize, usize)], items: &[StepItem], kb: usize, out: &mut [ItemOut]) -> Result<bool, String> {
        let d = self.m.cfg.d;
        let v = self.m.cfg.vocab;
        let valid = self.valid_vocab;
        let nb = idx.len();
        // 1. MTP: 列ごとの見せていない行と下書きの頭を 1 回で
        let mut obs_data: Vec<(u32, usize, Vec<u32>, Vec<DevRow>)> = vec![];
        let mut asks = vec![];
        for &(k, i) in idx {
            self.mtp_pages(i)?;
            let p0 = self.slots[i].seq.pos;
            let mut room = self.slots[i].reserved.max(p0 + 1).min(self.max_ctx).saturating_sub(p0).max(1);
            while room > 1 && !self.slots[i].seq.ensure(p0 + kb.min(room - 1) + 1) {
                room -= 1;
            }
            self.mtp_pages(i)?;
            if let Some((pp, pt)) = self.slots[i].pend.take() {
                let hs = Self::rows_at(self.slots[i].pbuf.f32p(), pt.len(), d);
                obs_data.push((i as u32, pp, pt, hs));
            }
            let kk = kb.min(room - 1);
            if kk > 0 {
                asks.push(DraftAsk { seq: i as u32, id_last: items[k].tokens[0], pos: p0, k: kk, sample: None, gumbel: hash_draft(&items[k].sampling) });
            }
        }
        let obs: Vec<Observe<DevRow>> = obs_data.iter().map(|(s, p, t, h)| Observe { seq: *s, pos0: *p, tokens: t, h }).collect();
        let x = self.mtp.as_mut().unwrap();
        let dr = x.head.run(&obs, &asks)?;
        let mut drafts: Vec<Vec<u32>> = vec![vec![]; nb];
        for (a, o) in asks.iter().zip(dr) {
            let r = idx.iter().position(|&(_, i)| i as u32 == a.seq).unwrap();
            let mut t = o.tokens;
            if let Some(j) = t.iter().position(|&x| x as usize >= valid) {
                t.truncate(j);
            }
            drafts[r] = t;
        }
        let tq = 1 + drafts.iter().map(|x| x.len()).max().unwrap_or(0);
        if tq < 2 {
            return Ok(false); // 下書きが無い: まとめたデコードで
        }
        // 2. 確かめ（短い列は最後の語で埋める。埋めた行は受け取らない）
        let toks: Vec<Vec<u32>> = idx
            .iter()
            .enumerate()
            .map(|(r, &(k, _))| {
                let mut t = vec![items[k].tokens[0]];
                t.extend_from_slice(&drafts[r]);
                let last = *t.last().unwrap();
                t.resize(tq, last);
                t
            })
            .collect();
        let p0s: Vec<usize> = idx.iter().map(|&(_, i)| self.slots[i].seq.pos).collect();
        let hb = self.hbuf.f32p();
        let am = {
            let base = self.slots.as_mut_ptr();
            let mut refs: Vec<&mut Seq> = idx.iter().map(|&(_, i)| unsafe { &mut (*base.add(i)).seq }).collect();
            self.m.verify_multi(&mut refs, &toks, hb)
        };
        let need_logits = idx.iter().any(|&(k, _)| !(items[k].sampling.is_greedy() && !items[k].sampling.has_penalty()) || items[k].constraint.is_some());
        let temps: Vec<f32> = idx.iter().flat_map(|&(k, _)| std::iter::repeat(items[k].sampling.temperature.max(1e-3)).take(tq)).collect();
        let cands = if need_logits { self.row_cands(true, nb * tq, &temps) } else { None };
        let mut full: Option<Vec<f32>> = None;
        // 3. 受け取り（4. commit は全部の列をまとめて後で）
        let mut keeps: Vec<usize> = Vec::with_capacity(nb);
        for (r, &(k, i)) in idx.iter().enumerate() {
            let it = &items[k];
            let sp = &*it.sampling;
            let kd = drafts[r].len();
            let mut n_acc = 0;
            let new;
            if sp.is_greedy() && !sp.has_penalty() && it.constraint.is_none() {
                while n_acc < kd && am[r][n_acc] == drafts[r][n_acc] {
                    n_acc += 1;
                }
                new = if (am[r][n_acc] as usize) < valid {
                    am[r][n_acc]
                } else {
                    let row = self.m.logits_dec(nb * tq);
                    let o = (r * tq + n_acc) * v;
                    sampler::sample(&row[o..o + v], valid, sp, &History::default(), (p0s[r] + n_acc + 1) as u64)
                };
            } else {
                let mut hist = it.hist.clone().unwrap_or_default();
                let mut cm: Option<crate::server::jsonc::Matcher> = it.constraint.as_deref().cloned();
                let mut y = None;
                for j in 0..=kd {
                    let o = (r * tq + j) * v;
                    let pc = match &cm {
                        Some(m) => cands.as_ref().and_then(|c| sampler::prepare_cands_masked(&c[r * tq + j], sp, &hist, m)),
                        None => cands.as_ref().and_then(|c| sampler::prepare_cands(&c[r * tq + j], sp, &hist)),
                    };
                    let pr = match pc {
                        Some(pr) => pr,
                        None => {
                            self.cand_fallback += 1;
                            let lg = full.get_or_insert_with(|| self.m.logits_dec(nb * tq));
                            match &cm {
                                Some(m) => {
                                    self.mask_full += 1;
                                    sampler::prepare_masked(&lg[o..o + v], valid, sp, &hist, m)
                                }
                                None => sampler::prepare(&lg[o..o + v], valid, sp, &hist),
                            }
                        }
                    };
                    match sampler::spec_row_prepared(pr, sp, (p0s[r] + j + 1) as u64, drafts[r].get(j).copied()) {
                        Ok(()) => {
                            hist.push(drafts[r][j]);
                            if let Some(m) = &mut cm {
                                m.advance(drafts[r][j]);
                            }
                            n_acc += 1;
                        }
                        Err(tok) => {
                            y = Some(tok);
                            break;
                        }
                    }
                }
                new = y.expect("最後の行は必ず語を出す");
            }
            keeps.push(n_acc + 1);
            let n = n_acc + 1;
            let pb = self.slots[i].pbuf.f32p();
            crate::cuda::check(unsafe { crate::cuda::ffi::cudaMemcpy(pb.cast(), hb.add(r * tq * d).cast(), n * d * 4, 3) }, "h の写し");
            let s = &mut self.slots[i];
            s.pend = Some((p0s[r], toks[r][..n].to_vec()));
            s.steps += 1;
            s.drafted += kd as u64;
            s.accepted += n_acc as u64;
            let mut outv = drafts[r][..n_acc].to_vec();
            outv.push(new);
            out[k] = ItemOut::Tokens(outv);
        }
        {
            let base = self.slots.as_mut_ptr();
            let mut refs: Vec<&mut Seq> = idx.iter().map(|&(_, i)| unsafe { &mut (*base.add(i)).seq }).collect();
            if commit_stream() {
                self.m.commit_multi(&mut refs, tq, &keeps);
            } else {
                for (r, sq) in refs.into_iter().enumerate() {
                    self.m.commit_verify_rows(sq, tq, keeps[r], r * tq);
                }
            }
        }
        Ok(true)
    }

    fn step_inner(&mut self, items: &[StepItem]) -> Result<Vec<ItemOut>, String> {
        self.poll_pre()?;
        let mut out: Vec<ItemOut> = items.iter().map(|_| ItemOut::None).collect();
        let mut dec: Vec<(usize, usize)> = vec![];
        for (k, it) in items.iter().enumerate() {
            let Some(i) = self.slot_of(it.seq) else {
                out[k] = ItemOut::Error(format!("sequence {} is not on this engine", it.seq));
                continue;
            };
            let s = &self.slots[i];
            let t = it.tokens.len();
            if it.pos != s.seq.pos {
                out[k] = ItemOut::Error(format!("position mismatch ({} vs {})", it.pos, s.seq.pos));
                continue;
            }
            if t == 0 || t > self.m.opt.chunk {
                out[k] = ItemOut::Error(format!("a step must have 1..{} tokens (got {t})", self.m.opt.chunk));
                continue;
            }
            if s.seq.pos + t > self.max_ctx {
                out[k] = ItemOut::Error(format!("the context is full ({} tokens)", self.max_ctx));
                continue;
            }
            if it.tokens.iter().any(|&x| x as usize >= self.m.cfg.vocab) {
                out[k] = ItemOut::Error("token id out of range".into());
                continue;
            }
            // 頁（予約の中で取る。予約を越えた分も空きがあれば取る）
            if !self.slots[i].seq.ensure(it.pos + t) {
                while !self.slots[i].seq.ensure(it.pos + t) && self.evict_oldest() {}
                if !self.slots[i].seq.ensure(it.pos + t) {
                    out[k] = ItemOut::Error("out of KV pages".into());
                    continue;
                }
            }
            if t == 1 && it.want && !self.slots[i].in_prompt && it.pos > 0 {
                dec.push((k, i));
            } else {
                out[k] = self.prefill_item(i, it)?;
            }
        }
        match dec.len() {
            0 => {}
            1 if self.mtp.is_some() && self.slots[dec[0].1].seq.pos + 2 <= self.max_ctx => {
                let (k, i) = dec[0];
                out[k] = self.spec_step(i, &items[k])?;
            }
            n => {
                // まとめた確かめ: 行の予算（verify_rows）を列で割る。1 列あたり 2 語に満たなければ下書きなしでまとめる
                let kb = (self.verify_rows / n).saturating_sub(1).min(self.mtp.as_ref().map_or(0, |x| x.k));
                if kb >= 1 && dec.iter().all(|&(_, i)| self.slots[i].seq.pos + 2 <= self.max_ctx) {
                    if self.spec_multi(&dec, items, kb, &mut out)? {
                        return Ok(out);
                    }
                }
                for &(_, i) in &dec {
                    let p = self.slots[i].seq.pos + 1;
                    if !self.slots[i].seq.ensure(p) {
                        return Err("out of KV pages".into());
                    }
                }
                self.batch_decode(&dec, items, &mut out)?;
            }
        }
        Ok(out)
    }

    fn fail(&mut self, p: Box<dyn std::any::Any + Send>) -> String {
        let m = format!("engine failure: {}", panic_msg(p));
        self.poisoned = Some(m.clone());
        log(json!({"ev": "engine_panic", "exit_in_secs": 2}));
        // CUDA の文脈は壊れると戻らない: 走っていた依頼に誤りを返す間だけ待って終わる（systemd が起こし直す）
        std::thread::spawn(|| {
            std::thread::sleep(std::time::Duration::from_secs(2));
            super::util::hard_exit(70);
        });
        m
    }

    /// 頁 pg の KV の置き場（本体の Attention の層の K・V → MTP の K・V。RAM の段の 1 頁と同じ並び）
    fn page_spans(&self, pg: u32) -> Vec<(*mut u8, usize)> {
        let mut v = self.slots[0].seq.kv_page_spans(pg);
        if let Some(x) = self.mtp.as_ref() {
            v.extend_from_slice(&x.head.kv_page_spans(pg));
        }
        v
    }

    /// RAM の段の控えを取る位置（区切りの所。every 語ごとの物はプレフィルの塊の切れ目で取る＝余分に切らない）
    fn set_host_cuts(&mut self, i: usize, n: usize, done: usize, hints: &super::backend::PrefixHints) {
        let s = &mut self.slots[i];
        s.last_host = done;
        s.host_cuts.clear();
        let Some(h) = self.host.as_ref() else { return };
        let pin = hints.pin.is_some();
        let mut c: Vec<usize> = [hints.system_end, hints.user_start, hints.turn_start].into_iter().flatten().collect();
        if pin {
            c.push(n);
        }
        // 札（pin）は短くても取る（札の大きさは書いた人が決める）
        c.retain(|&x| x > done && (pin || x >= h.min_tokens()) && (x < n || (pin && x == n)));
        c.sort_unstable();
        c.dedup();
        s.host_cuts = c;
    }

    /// RAM の段の控え idx を列 i に戻す（KV の頁・GDN の状態・MTP の h）。戻した語の数
    fn host_restore(&mut self, i: usize, idx: usize) -> Result<usize, String> {
        self.m.sync_commit();
        self.m.sync_prefill();
        let l = self.host.as_ref().ok_or("RAM の段が無い")?.toks(idx).len();
        self.slots[i].seq.reset();
        while !self.slots[i].seq.ensure(l) {
            if !self.evict_oldest() {
                return Err("restore: KV の頁が足りない".into());
            }
        }
        let pages = self.slots[i].seq.pages_vec();
        let np = l.div_ceil(super::hostcache::PAGE);
        let gpu: Vec<Vec<(*mut u8, usize)>> = pages[..np].iter().map(|&pg| self.page_spans(pg)).collect();
        let st = self.slots[i].seq.state_spans();
        let h = (self.hrest.ptr.cast::<u8>(), self.m.cfg.d * 4);
        // ⛔ reset の 0 埋め（NULL の流れ）が写しの後に走ると、戻した GDN の状態が 0 になる。写しは別の流れなので先に待つ
        self.m.sync_for_copy();
        let host = self.host.as_mut().unwrap();
        let n = host.restore(idx, &gpu, st, h)?;
        let toks = host.toks(idx).to_vec();
        let s = &mut self.slots[i];
        s.seq.set_pos(n);
        s.fed = toks;
        s.in_prompt = true;
        s.last_host = n;
        // 手番の頭の GPU の控えは、戻した所より後ろにあるときだけ取る
        if s.snap_at.is_some_and(|a| a <= n) {
            s.snap_at = Some(usize::MAX);
        }
        let hp = self.hrest.f32p();
        if let Some(x) = self.mtp.as_mut() {
            x.head.free(i as u32);
            x.head.set_pages(i as u32, &pages)?;
            x.head.set_prefix(i as u32, n, hp)?;
        }
        Ok(n)
    }

    /// 列 i の今の位置（= fed の長さ）の控えを RAM の段へ（row = hpre の中のその位置の行）
    fn host_ckpt(&mut self, i: usize, row: usize) {
        let l = self.slots[i].fed.len();
        let Some(h) = self.host.as_ref() else { return };
        if (l < h.min_tokens() && self.slots[i].pin.is_none()) || self.slots[i].seq.pos != l {
            return;
        }
        self.m.sync_commit();
        // ⛔ 直前の塊の forward が NULL の流れにまだ積まれていることがある（max-seqs 1）。写しは別の流れなので両方を待つ
        self.m.sync_for_copy();
        let pages = self.slots[i].seq.pages_vec();
        let np = l.div_ceil(super::hostcache::PAGE);
        if pages.len() < np {
            return;
        }
        let gpu: Vec<Vec<(*mut u8, usize)>> = pages[..np].iter().map(|&pg| self.page_spans(pg)).collect();
        let st = self.slots[i].seq.state_spans();
        let d = self.m.cfg.d;
        let hrow = (unsafe { self.hpre.f32p().add(row * d) }.cast::<u8>(), d * 4);
        let s = &self.slots[i];
        let (public, pinned, name) = match &s.pin {
            Some(n) => (true, true, Some(n.clone())),
            None => (false, false, None),
        };
        let scope = s.scope.clone();
        let toks = s.fed.clone();
        let r = catch_unwind(AssertUnwindSafe(|| self.host.as_mut().unwrap().save(&scope, public, pinned, name, &toks, &gpu, st, hrow)));
        match r {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => log(json!({"ev": "prefix_host_save_failed", "error": e})),
            Err(p) => {
                self.fail(p);
                return;
            }
        }
        self.slots[i].last_host = l;
    }

    /// GPU の控え j を列 i に（reuse の中身・失敗したら頭から）
    fn reuse_gpu(&mut self, i: usize, j: usize, prompt_len: usize) -> usize {
        match catch_unwind(AssertUnwindSafe(|| self.reuse_inner(i, j))) {
            Ok(Ok(l)) => {
                log(json!({"ev": "prefix_reuse", "tokens": l, "prompt": prompt_len}));
                l
            }
            Ok(Err(e)) => {
                log(json!({"ev": "prefix_reuse_failed", "error": e}));
                // 頭から（頁は列に移っているので、状態を頭に戻すだけ）
                let s = &mut self.slots[i];
                s.seq.reset();
                s.fed.clear();
                if let Some(x) = self.mtp.as_mut() {
                    x.head.free(i as u32);
                }
                0
            }
            Err(p) => {
                self.fail(p);
                0
            }
        }
    }
}

impl Engine for M1Engine {
    fn info(&self) -> EngineInfo {
        EngineInfo { max_seqs: self.slots.len(), max_ctx: self.max_ctx, max_step_tokens: self.m.opt.chunk }
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

    fn busy(&mut self, seq: SeqId) -> bool {
        if self.poisoned.is_some() {
            return false;
        }
        if let Err(p) = catch_unwind(AssertUnwindSafe(|| self.poll_pre())).map(|r| r.map_err(|e| log(json!({"ev": "poll_failed", "error": e})))) {
            self.fail(p);
            return false;
        }
        match &self.pre {
            Some((i, _, _)) => self.slots[*i].id == Some(seq),
            None => false,
        }
    }

    fn admit(&mut self, seq: SeqId, prompt_len: usize, max_total: usize) -> bool {
        if self.poisoned.is_none() {
            if let Err(p) = catch_unwind(AssertUnwindSafe(|| self.m.sync_commit())) {
                self.fail(p);
                return false;
            }
        }
        if self.poisoned.is_some() {
            return true; // step が誤りを返す
        }
        let Some(i) = self.slots.iter().position(|s| s.id.is_none()) else { return false };
        // 確かめの下書きの分（k＋1 語）も見込む
        let extra = self.mtp.as_ref().map_or(0, |x| x.k + 1);
        let want = (max_total + extra).min(self.max_ctx).max(prompt_len + 1);
        if pages_for(want) as isize > self.pages_available() {
            return false;
        }
        let s = &mut self.slots[i];
        s.id = Some(seq);
        s.reserved = want;
        s.seq.reset();
        s.in_prompt = true;
        s.fed.clear();
        s.pend = None;
        s.snap_at = None;
        s.host_cuts.clear();
        s.last_host = 0;
        s.pin = None;
        s.has_media = false;
        if let Some(x) = self.mtp.as_mut() {
            x.head.free(i as u32);
        }
        true
    }

    fn reuse(&mut self, seq: SeqId, scope: &str, prompt: &[u32], snap: Option<usize>) -> usize {
        self.reuse_hints(seq, scope, prompt, &super::backend::PrefixHints { turn_start: snap, ..Default::default() })
    }

    fn reuse_hints(&mut self, seq: SeqId, scope: &str, prompt: &[u32], hints: &super::backend::PrefixHints) -> usize {
        let Some(i) = self.slot_of(seq) else { return 0 };
        self.slots[i].scope = scope.to_string();
        self.slots[i].snap_at = hints.turn_start.filter(|&a| a > 0 && a < prompt.len());
        self.slots[i].pin = hints.pin.clone();
        if hints.pin.is_some() {
            // 先計算の札は GPU の控えを取らない（その札の持ち主の名前では、ほかの誰にも当たらない）
            self.slots[i].snap_at = Some(usize::MAX);
        }
        if self.poisoned.is_some() {
            return 0;
        }
        // 同じアカウントの控えのうち、頭が合ういちばん長い物（別のアカウントの控えは見ない）
        let hit = (0..self.pcs.len())
            .filter(|&j| {
                let pc = &self.pcs[j];
                pc.valid && pc.owner.is_none() && !pc.pages.is_empty() && pc.scope == scope && !pc.toks.is_empty() && pc.toks.len() < prompt.len()
                    && prompt.starts_with(&pc.toks)
            })
            .max_by_key(|&j| self.pcs[j].toks.len());
        let gpu_len = hit.map_or(0, |j| self.pcs[j].toks.len());
        // RAM の段: GPU の控えより 1 頁以上長く当たるなら、そちらを戻す
        // 札との一致の長さ（2048 語以上の依頼だけ・位置の数だけ）。札が当たらない理由を見る
        if prompt.len() >= 2048 {
            if let Some(h) = self.host.as_ref() {
                let v: Vec<_> = h.pin_lcp(prompt).into_iter().map(|(n, l, t)| json!({"name": n, "same": l, "of": t})).collect();
                if !v.is_empty() {
                    log(json!({"ev": "prefix_pin_lcp", "prompt": prompt.len(), "pins": v}));
                }
            }
        }
        // ⛔ 遅い写しの後の休み中は当てない（ふつうのプレフィル）
        let host_hit = self.host.as_mut().and_then(|h| if h.paused() { None } else { h.find(scope, prompt) }).filter(|&(_, l)| l > gpu_len + super::hostcache::PAGE);
        if let Some((idx, _)) = host_hit {
            match catch_unwind(AssertUnwindSafe(|| self.host_restore(i, idx))) {
                Ok(Ok(l)) => {
                    self.set_host_cuts(i, prompt.len(), l, hints);
                    return l;
                }
                Ok(Err(e)) => {
                    log(json!({"ev": "prefix_host_restore_failed", "error": e}));
                    // 頭から（取った頁は置き場へ返す。GPU の控えの道が頁を渡せるように）
                    let s = &mut self.slots[i];
                    s.seq.reset();
                    s.fed.clear();
                    let pg = s.seq.take_pages_any();
                    self.slots[0].seq.return_pages(pg);
                    if let Some(x) = self.mtp.as_mut() {
                        x.head.free(i as u32);
                    }
                }
                Err(p) => {
                    self.fail(p);
                    return 0;
                }
            }
        }
        let Some(j) = hit else {
            self.set_host_cuts(i, prompt.len(), 0, hints);
            return 0;
        };
        let _ = seq;
        let l = self.reuse_gpu(i, j, prompt.len());
        self.set_host_cuts(i, prompt.len(), l, hints);
        l
    }
    fn attach_media(&mut self, seq: SeqId, media: &[std::sync::Arc<super::backend::MediaInput>]) -> Result<(), String> {
        if media.is_empty() {
            return Ok(());
        }
        let Some(i) = self.slot_of(seq) else { return Err("unknown sequence".into()) };
        self.slots[i].has_media = true;
        if self.poisoned.is_some() {
            return Ok(()); // step が誤りを返す
        }
        let Some(v) = self.vision.as_ref() else { return Err("this server was started without --mmproj".into()) };
        // 区画に切って送るのは手元で（同期の写し）。符号化は自分の流れに積むだけで待たない（ほかの人のデコードを止めない）。
        // この列のプレフィルは、その流れの終わりの印を待つ
        let vs = self.vis_s;
        let r = catch_unwind(AssertUnwindSafe(|| -> Result<(Vec<model::MediaSpan>, crate::vision::Mark, Vec<crate::vision::Staged>), String> {
            let t0 = std::time::Instant::now();
            let mut staged = vec![];
            for m in media {
                let img = crate::vision::Rgb { w: m.w, h: m.h, px: m.rgb.clone() };
                let st = v.stage_resized(&img)?;
                if st.nx != m.nx || st.ny != m.ny {
                    return Err(format!("image grid {}x{} != {}x{}", st.nx, st.ny, m.nx, m.ny));
                }
                staged.push(st);
            }
            let mut spans = vec![];
            for (m, st) in media.iter().zip(&staged) {
                // 誤りなら、積んだ分が終わってから入力を捨てる（GPU が読んでいる間に放さない）
                let emb = v.run(st, vs).map_err(|e| {
                    crate::vision::stream_sync(vs);
                    e
                })?;
                spans.push(model::MediaSpan { start: m.start, nx: m.nx, ny: m.ny, emb: std::rc::Rc::new(emb) });
            }
            let mark = crate::vision::Mark::record(vs);
            log(json!({"ev": "image_queued", "images": media.len(), "tokens": spans.iter().map(|s| s.n()).sum::<usize>(),
                       "ms": (t0.elapsed().as_secs_f64() * 1e3).round()}));
            Ok((spans, mark, staged))
        }));
        let (spans, mark, staged) = match r {
            Ok(Ok(x)) => x,
            Ok(Err(e)) => return Err(e),
            Err(p) => return Err(self.fail(p)),
        };
        let tuples: Vec<(usize, usize, usize)> = spans.iter().map(|s| (s.start, s.nx, s.ny)).collect();
        let s = &mut self.slots[i];
        s.seq.media = spans;
        s.media_wait = Some((mark, staged));
        // 画像のある列は控えを取らない（画像が違っても文字が同じだと当たってしまう）
        s.snap_at = Some(usize::MAX);
        if let Some(x) = self.mtp.as_mut() {
            x.head.set_media(i as u32, tuples)?;
        }
        Ok(())
    }

    fn release(&mut self, seq: SeqId) {
        if let Some(i) = self.slot_of(seq) {
            if self.poisoned.is_some() {
                self.slots[i].id = None;
                return;
            }
            if let Err(p) = catch_unwind(AssertUnwindSafe(|| self.clear_slot(i))) {
                self.fail(p);
                self.slots[i].id = None;
            }
        }
    }
}

/// 確定を別の流れで次の確かめに重ねる（既定 true。TACHYON_NO_COMMIT_STREAM=1 で前の道）
/// 結び付けの下書き（TACHYON_HASH_RNG=1・温度 > 0）: MTP の語を本体と同じ雑音の Gumbel-max で選ぶ
fn hash_draft(sp: &sampler::SamplingParams) -> Option<(u64, f32)> {
    // TACHYON_HASH_DRAFT_TSCALE（既定 1）: 下書きの温度 = 本体の温度 × これ（小さいほど argmax に近い。本体の分布は変わらない）
    static SC: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    let sc = *SC.get_or_init(|| std::env::var("TACHYON_HASH_DRAFT_TSCALE").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0));
    (sampler::hash_rng_on() && !sp.is_greedy()).then(|| (sp.seed, sp.temperature * sc))
}

fn commit_stream() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("TACHYON_NO_COMMIT_STREAM").map_or(true, |v| v != "1"))
}
