//! 下書き（MTP・DSpine 共通）の受け取りと、1 歩の回し方。GPU を使わない（本体と下書きは trait の向こう）。
//!
//! 位置の決まり: 確かめの行 i（入力は位置 pos0+i の語）は位置 pos0+i+1 の語を当てる。乱数の鍵はその位置。
//!
//! 受け取りの方式（`Verify`）。どれも本体の分布 P を変えない（下の統計の試験で確かめる）:
//! - `Greedy`: 温度 0。本体の argmax（同点は小さい id）が下書きと同じ間だけ受け取る
//! - `Match`: 行 i で P_i から y を Gumbel-max で取り（雑音 G(seed, 位置, 語)）、下書きと同じなら受け取って次の行へ。
//!   y は (P, その位置の雑音) だけで決まる → 同じ seed なら下書きのあり・なし・k によらず同じ出力。下書きは argmax でも、
//!   同じ雑音で Q から取った物（結び付け。当たりやすい）でもよい（llama.cpp の common_sampler_sample_and_accept_n と同じ考え）
//! - `Rs`: Leviathan 型の棄却。下書き d ~ Q（別の雑音で取る）。min(1, P(d)/Q(d)) で受け取り、外れたら norm(max(P−Q, 0)) から
//!   取って終わり。全部受け取ったら、おまけを P_k から（llama.cpp の common_sampler_sample_and_accept_n_rejection と同じ）。
//!   分布は保つが、下書きのあり・なしで同じ出力にはならない
//!
//! 本体（`Target`。M1 の model.rs が実装する予定）と下書き（`Drafter`。`mtp::MtpHead`）をつなぐ回し方は `start`（プロンプト）と
//! `step`（走っている全員を 1 歩）。MTP の追いつきは llama.cpp の draft-mtp と同じ: 本体が通した語ごとに MTP の行 (x_p, h_nextn[p-1])。
//! 確かめの後は、受け取った分だけの行（id_last と受け取った下書き）を本体の h で通し直す（`Observe`）。
use std::fmt;

pub type SeqId = u32;

// ---------------------------------------------------------------- 乱数（Philox4x32-10）

const PH_M0: u32 = 0xD251_1F53;
const PH_M1: u32 = 0xCD9E_8D57;
const PH_W0: u32 = 0x9E37_79B9;
const PH_W1: u32 = 0xBB67_AE85;

/// Philox4x32-10（Random123・curand と同じ）。GPU の試料の取り方も同じ式にする
pub fn philox4x32_10(mut c: [u32; 4], mut k: [u32; 2]) -> [u32; 4] {
    for r in 0..10 {
        if r > 0 {
            k[0] = k[0].wrapping_add(PH_W0);
            k[1] = k[1].wrapping_add(PH_W1);
        }
        let p0 = PH_M0 as u64 * c[0] as u64;
        let p1 = PH_M1 as u64 * c[2] as u64;
        c = [((p1 >> 32) as u32) ^ c[1] ^ k[0], p1 as u32, ((p0 >> 32) as u32) ^ c[3] ^ k[1], p0 as u32];
    }
    c
}

/// 乱数の使い道（鍵の 4 つ目）。同じ (seed, 位置, 語) でも使い道が違えば独立
pub const TAG_SAMPLE: u32 = 0; // 本体の y（Match・Rs のおまけ）。結び付けの下書きもこれ
pub const TAG_DRAFT: u32 = 1; // Rs の下書き d ~ Q
pub const TAG_ACCEPT: u32 = 2; // Rs の受け取りの一様乱数
pub const TAG_RESID: u32 = 3; // Rs の外れの後の norm(max(P−Q, 0))

/// (0,1) の一様乱数（53 ビット。0 と 1 は出ない）。鍵 (seed, 位置, 使い道, 語)
pub fn uniform(seed: u64, pos: u64, tag: u32, x: u32) -> f64 {
    let r = philox4x32_10([pos as u32, (pos >> 32) as u32, x, tag], [seed as u32, (seed >> 32) as u32]);
    let bits = ((r[0] as u64) << 32 | r[1] as u64) >> 11;
    (bits as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
}

/// Gumbel(0,1) = −ln(−ln U)。U は 53 ビットなので上限は約 36.7（24 ビットの f32 だと 16.6 で頭打ちになる）
pub fn gumbel(seed: u64, pos: u64, tag: u32, x: u32) -> f64 {
    -(-uniform(seed, pos, tag, x).ln()).ln()
}

// ---------------------------------------------------------------- 分布

/// 試料の取り方の加工。順は 温度 → top_k → softmax → top_p → min_p → 足して 1（vLLM・HF と同じ。llama.cpp は温度を最後に掛ける）。
/// 温度 0 は argmax の one-hot
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    /// 0 なら切らない
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
}

impl Sampling {
    pub const GREEDY: Sampling = Sampling { temperature: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0 };
    /// Qwen3.8 の Thinking の推奨（temp 1.0・top_p 0.95・top_k 20）
    pub const THINKING: Sampling = Sampling { temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0 };
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

/// 疎な分布（台の上の確率。和は 1。ids は確率の大きい順・同点は小さい id）
#[derive(Clone, Debug, PartialEq)]
pub struct Dist {
    pub ids: Vec<u32>,
    pub p: Vec<f64>,
}

impl Dist {
    pub fn one_hot(id: u32) -> Dist {
        Dist { ids: vec![id], p: vec![1.0] }
    }

    /// (id, 重み) から（重みは 0 以上。和で割る）。重みが全部 0 なら None
    pub fn from_weights(mut v: Vec<(u32, f64)>) -> Option<Dist> {
        v.retain(|&(_, w)| w > 0.0);
        let s: f64 = v.iter().map(|&(_, w)| w).sum();
        if !(s > 0.0) {
            return None;
        }
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        Some(Dist { ids: v.iter().map(|&(i, _)| i).collect(), p: v.iter().map(|&(_, w)| w / s).collect() })
    }

    /// logits（語彙全部）を加工して分布に
    pub fn from_logits(logits: &[f32], s: &Sampling) -> Dist {
        if s.is_greedy() {
            return Dist::one_hot(argmax(logits));
        }
        Self::from_pairs(logits.iter().enumerate().map(|(i, &l)| (i as u32, l)).collect(), s)
    }

    /// 候補だけの logits（上位 k など）を加工して分布に（候補の外は 0 とみなす）
    pub fn from_candidates(ids: &[u32], logits: &[f32], s: &Sampling) -> Dist {
        let v: Vec<(u32, f32)> = ids.iter().copied().zip(logits.iter().copied()).collect();
        if s.is_greedy() {
            let b = v.iter().fold(v[0], |b, &x| if x.1 > b.1 || (x.1 == b.1 && x.0 < b.0) { x } else { b });
            return Dist::one_hot(b.0);
        }
        Self::from_pairs(v, s)
    }

    fn from_pairs(v: Vec<(u32, f32)>, s: &Sampling) -> Dist {
        let t = s.temperature as f64;
        let mut v: Vec<(u32, f64)> = v.into_iter().map(|(i, l)| (i, l as f64 / t)).collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        if s.top_k > 0 && s.top_k < v.len() {
            v.truncate(s.top_k);
        }
        let m = v[0].1;
        let mut w: Vec<(u32, f64)> = v.iter().map(|&(i, l)| (i, (l - m).exp())).collect();
        let z: f64 = w.iter().map(|x| x.1).sum();
        for x in w.iter_mut() {
            x.1 /= z;
        }
        if s.top_p < 1.0 {
            // 大きい順に足して top_p に届いた所まで（少なくとも 1 つ）
            let mut acc = 0.0;
            let mut n = 0;
            for x in &w {
                acc += x.1;
                n += 1;
                if acc >= s.top_p as f64 {
                    break;
                }
            }
            w.truncate(n);
        }
        if s.min_p > 0.0 {
            let lim = w[0].1 * s.min_p as f64;
            w.retain(|x| x.1 >= lim);
        }
        Dist::from_weights(w).expect("分布が空")
    }

    pub fn prob(&self, id: u32) -> f64 {
        self.ids.iter().position(|&i| i == id).map_or(0.0, |k| self.p[k])
    }

    pub fn argmax(&self) -> u32 {
        self.ids[0]
    }
}

/// 同点は小さい id（GPU の argmax も同じ決まり）
pub fn argmax(x: &[f32]) -> u32 {
    let mut b = 0;
    for i in 1..x.len() {
        if x[i] > x[b] {
            b = i;
        }
    }
    b as u32
}

/// Gumbel-max で 1 つ取る: argmax_x [ln P(x) + G(seed, pos, tag, x)]（同点は小さい id）
pub fn sample(d: &Dist, seed: u64, pos: u64, tag: u32) -> u32 {
    let mut best = (f64::NEG_INFINITY, u32::MAX);
    for (&id, &p) in d.ids.iter().zip(&d.p) {
        if p <= 0.0 {
            continue;
        }
        let v = p.ln() + gumbel(seed, pos, tag, id);
        if v > best.0 || (v == best.0 && id < best.1) {
            best = (v, id);
        }
    }
    best.1
}

// ---------------------------------------------------------------- 確かめ

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verify {
    Greedy,
    Match,
    Rs,
}

/// 本体の確かめの 1 行の出力
#[derive(Clone, Debug)]
pub struct RowOut {
    pub argmax: u32,
    /// 加工した分布（温度 > 0 のとき。Greedy なら要らない）
    pub dist: Option<Dist>,
}

/// 確かめの結果。tokens = 受け取った下書き（n_acc 個）＋新しい 1 語。新しい語はまだ本体を通っていない（次の id_last）。
/// 止めの語は下書きと同じでも「新しい 1 語」として数える（n_acc に入れない）
#[derive(Clone, Debug, PartialEq)]
pub struct Verdict {
    pub tokens: Vec<u32>,
    pub n_acc: usize,
    /// 止めの語が出た（そこで終わり。後ろは出さない）
    pub stopped: bool,
}

fn row_dist(r: &RowOut) -> &Dist {
    r.dist.as_ref().expect("Match・Rs には本体の分布が要る")
}

/// 1 列の確かめ。rows は確かめの行（drafts.len() + 1 行）、pos0 は行 0 の入力（id_last）の位置。
/// q は Rs のときの下書きの分布（drafts と同じ数）。stop は止めの語か
pub fn verify(mode: Verify, drafts: &[u32], q: Option<&[Dist]>, rows: &[RowOut], pos0: u64, seed: u64, stop: &dyn Fn(u32) -> bool) -> Verdict {
    let k = drafts.len();
    assert_eq!(rows.len(), k + 1, "確かめの行は下書き + 1");
    let mut tokens = Vec::with_capacity(k + 1);
    let mut stopped = false;
    match mode {
        Verify::Greedy | Verify::Match => {
            for i in 0..=k {
                let y = if mode == Verify::Greedy { rows[i].argmax } else { sample(row_dist(&rows[i]), seed, pos0 + i as u64 + 1, TAG_SAMPLE) };
                tokens.push(y);
                if stop(y) {
                    stopped = true;
                    break;
                }
                if i == k || y != drafts[i] {
                    break;
                }
            }
        }
        Verify::Rs => {
            let q = if k == 0 { &[][..] } else { q.expect("Rs には下書きの分布 Q が要る") };
            assert_eq!(q.len(), k);
            let mut all = true;
            for i in 0..k {
                let pos = pos0 + i as u64 + 1;
                let (p, d) = (row_dist(&rows[i]), drafts[i]);
                let (pd, qd) = (p.prob(d), q[i].prob(d));
                if qd > 0.0 && (pd >= qd || uniform(seed, pos, TAG_ACCEPT, 0) * qd < pd) {
                    tokens.push(d);
                    if stop(d) {
                        stopped = true;
                        all = false;
                        break;
                    }
                    continue;
                }
                // 外れ: norm(max(P − Q, 0))。Q の外の語は P のまま
                let res = Dist::from_weights(p.ids.iter().zip(&p.p).map(|(&id, &pp)| (id, (pp - q[i].prob(id)).max(0.0))).collect());
                let y = match res {
                    Some(r) => sample(&r, seed, pos, TAG_RESID),
                    None => sample(p, seed, pos, TAG_SAMPLE),
                };
                tokens.push(y);
                stopped = stop(y);
                all = false;
                break;
            }
            if all {
                let y = sample(row_dist(&rows[k]), seed, pos0 + k as u64 + 1, TAG_SAMPLE);
                tokens.push(y);
                stopped = stop(y);
            }
        }
    }
    let n_acc = tokens.len() - 1;
    Verdict { tokens, n_acc, stopped }
}

/// 本体の 1 行から 1 語（下書きなしの普通のデコードと同じ取り方。pos は当てる語の位置）
pub fn sample_row(mode: Verify, r: &RowOut, pos: u64, seed: u64) -> u32 {
    match mode {
        Verify::Greedy => r.argmax,
        Verify::Match | Verify::Rs => sample(row_dist(r), seed, pos, TAG_SAMPLE),
    }
}

// ---------------------------------------------------------------- 本体と下書きの口

/// 本体の 1 歩に渡す 1 列
#[derive(Clone, Debug)]
pub struct SeqStep<'a> {
    pub seq: SeqId,
    /// tokens[0] の位置（= 本体がそれまでに通した語の数）
    pub pos0: usize,
    pub tokens: &'a [u32],
    /// 行の分布が要るときの加工（None なら argmax だけ）
    pub sampling: Option<Sampling>,
    /// この行から後ろだけ argmax・分布が要る（プレフィルの塊は最後の 1 行・確かめは 0）。h は全部の行で要る
    pub out_from: usize,
}

/// 本体の 1 歩の出力（列ごと・行ごと。行の数は tokens と同じ。out_from より前の行の argmax・dist は使わない）
pub struct StepOut<H> {
    pub rows: Vec<Vec<RowOut>>,
    /// 本体の h_nextn（output_norm の後）。MTP に渡す。次の step まで有効
    pub h: Vec<Vec<H>>,
}

/// 本体（M1 の model.rs が実装する予定）
pub trait Target {
    /// MTP に渡す h の 1 行（GPU の番地など）
    type H: Copy;
    /// 列ごとに pos0 から tokens を流す（プレフィルの塊・デコード・確かめを 1 回で）。流した語は仮に全部 KV・状態に入る
    fn step(&mut self, batch: &[SeqStep]) -> Result<StepOut<Self::H>, String>;
    /// 確かめの後: 列 seq を「pos0 から n_keep 語だけ通した」状態にする（KV の長さを戻す・GDN をやり直す）
    fn commit(&mut self, seq: SeqId, pos0: usize, n_keep: usize) -> Result<(), String>;
    /// 列を捨てる
    fn free(&mut self, seq: SeqId);
}

/// 本体が通した行を下書き役に渡す（MTP の KV を埋める。llama.cpp の process）。
/// h[l] は本体が tokens[l]（位置 pos0+l）を通して出した h_nextn。MTP の行 j（tokens[j]）が使うのは 1 つ前の位置の h:
/// j = 0 なら下書き役が覚えている前の h（列の頭は 0）、j > 0 なら h[j-1]。終わったら下書き役は h の最後の行を覚える（次の 1 行目の h）。
/// 1 回の run で同じ列の Observe は 1 つまで。h の番地は run の間だけ有効であればよい
pub struct Observe<'a, H> {
    pub seq: SeqId,
    pub pos0: usize,
    pub tokens: &'a [u32],
    pub h: &'a [H],
}

/// 下書きを頼む: 位置 pos の語 id_last（本体はまだ通していない）から k 語
#[derive(Clone, Debug)]
pub struct DraftAsk {
    pub seq: SeqId,
    pub id_last: u32,
    pub pos: usize,
    pub k: usize,
    /// None は argmax。Some((加工, seed, 結び付け)) は Q から取る（結び付け = 本体と同じ雑音 TAG_SAMPLE、でなければ TAG_DRAFT）
    pub sample: Option<(Sampling, u64, bool)>,
}

#[derive(Clone, Debug, Default)]
pub struct DraftOut {
    pub tokens: Vec<u32>,
    /// 下書きの分布（Rs で要る。argmax の下書きなら None）
    pub q: Option<Vec<Dist>>,
}

/// 下書き役（MTP・DSpine）
pub trait Drafter<H> {
    /// 追いつき（obs）と下書き（asks）を 1 回で。返り値は asks と同じ順
    fn run(&mut self, obs: &[Observe<H>], asks: &[DraftAsk]) -> Result<Vec<DraftOut>, String>;
    fn free(&mut self, seq: SeqId);
}

// ---------------------------------------------------------------- 回し方

/// 1 列の状態
#[derive(Clone, Debug)]
pub struct SpecSeq {
    pub seq: SeqId,
    /// id_last の位置（= 本体が通した語の数）
    pub pos: usize,
    /// 出したが本体を通していない最後の語
    pub id_last: u32,
    pub drafts: Vec<u32>,
    pub q: Option<Vec<Dist>>,
    /// 出した語（プロンプトの後ろ）
    pub out: Vec<u32>,
    pub done: bool,
    pub max_new: usize,
    pub sampling: Sampling,
    pub seed: u64,
    /// 下書きの数（0 なら下書きなし。追いつきは続ける）
    pub k: usize,
    /// 下書きを Q から取るか（None は argmax）。Some(true) は結び付け（Match 向け）、Some(false) は独立（Rs 向け）
    pub draft_sample: Option<bool>,
    pub n_steps: usize,
    pub n_drafted: usize,
    pub n_accepted: usize,
}

impl SpecSeq {
    pub fn new(seq: SeqId, sampling: Sampling, seed: u64, k: usize, max_new: usize) -> Self {
        SpecSeq {
            seq,
            pos: 0,
            id_last: 0,
            drafts: vec![],
            q: None,
            out: vec![],
            done: false,
            max_new,
            sampling,
            seed,
            k,
            draft_sample: None,
            n_steps: 0,
            n_drafted: 0,
            n_accepted: 0,
        }
    }
    fn ask(&self) -> Option<DraftAsk> {
        let k = self.k.min(self.max_new.saturating_sub(self.out.len()).saturating_sub(1));
        if self.done || k == 0 {
            return None;
        }
        Some(DraftAsk {
            seq: self.seq,
            id_last: self.id_last,
            pos: self.pos,
            k,
            sample: self.draft_sample.map(|c| (self.sampling, self.seed, c)),
        })
    }
    fn sampling_opt(&self) -> Option<Sampling> {
        if self.sampling.is_greedy() {
            None
        } else {
            Some(self.sampling)
        }
    }
    /// 出した語を足す（max_new で切る）。止めの語・上限で done
    fn emit(&mut self, toks: &[u32], stopped: bool) {
        let room = self.max_new - self.out.len();
        let n = toks.len().min(room);
        self.out.extend_from_slice(&toks[..n]);
        if stopped || self.out.len() >= self.max_new {
            self.done = true;
        }
    }
}

/// プロンプトを本体に塊ごとに通し、MTP を追いつかせ、最初の語と下書きを作る
pub fn start<T: Target, D: Drafter<T::H>>(
    t: &mut T, d: &mut D, s: &mut SpecSeq, prompt: &[u32], chunk: usize, mode: Verify, stop: &dyn Fn(u32) -> bool,
) -> Result<(), String> {
    assert!(!prompt.is_empty() && chunk > 0);
    let mut p0 = 0;
    while p0 < prompt.len() {
        let n = chunk.min(prompt.len() - p0);
        let last = p0 + n == prompt.len();
        let toks = &prompt[p0..p0 + n];
        let st = SeqStep { seq: s.seq, pos0: p0, tokens: toks, sampling: if last { s.sampling_opt() } else { None }, out_from: if last { n - 1 } else { n } };
        let out = t.step(&[st])?;
        let obs = [Observe { seq: s.seq, pos0: p0, tokens: toks, h: &out.h[0] }];
        if !last {
            d.run(&obs, &[])?;
            p0 += n;
            continue;
        }
        let y = sample_row(mode, &out.rows[0][n - 1], prompt.len() as u64, s.seed);
        s.pos = prompt.len();
        s.id_last = y;
        s.emit(&[y], stop(y));
        let ask = s.ask();
        let dr = d.run(&obs, ask.as_slice())?;
        if ask.is_some() {
            let o = dr.into_iter().next().ok_or("下書きが返らない")?;
            s.drafts = o.tokens;
            s.q = o.q;
        }
        p0 += n;
    }
    Ok(())
}

/// 走っている全員を 1 歩: 確かめ（[id_last, 下書き…]）→ 受け取り → 本体の確定 → MTP の追いつきと次の下書き
pub fn step<T: Target, D: Drafter<T::H>>(t: &mut T, d: &mut D, seqs: &mut [SpecSeq], mode: Verify, stop: &dyn Fn(u32) -> bool) -> Result<usize, String> {
    let live: Vec<usize> = (0..seqs.len()).filter(|&i| !seqs[i].done).collect();
    if live.is_empty() {
        return Ok(0);
    }
    let inputs: Vec<Vec<u32>> = live.iter().map(|&i| std::iter::once(seqs[i].id_last).chain(seqs[i].drafts.iter().copied()).collect()).collect();
    let batch: Vec<SeqStep> = live
        .iter()
        .zip(&inputs)
        .map(|(&i, tk)| SeqStep { seq: seqs[i].seq, pos0: seqs[i].pos, tokens: tk, sampling: seqs[i].sampling_opt(), out_from: 0 })
        .collect();
    let out = t.step(&batch)?;
    let mut keep = Vec::with_capacity(live.len());
    for (j, &i) in live.iter().enumerate() {
        let s = &mut seqs[i];
        let v = verify(mode, &s.drafts, s.q.as_deref(), &out.rows[j], s.pos as u64, s.seed, stop);
        t.commit(s.seq, s.pos, 1 + v.n_acc)?;
        s.n_steps += 1;
        s.n_drafted += s.drafts.len();
        s.n_accepted += v.n_acc;
        keep.push((s.pos, v.n_acc));
        s.pos += 1 + v.n_acc;
        s.id_last = *v.tokens.last().unwrap();
        s.emit(&v.tokens, v.stopped);
        s.drafts.clear();
        s.q = None;
    }
    // 追いつき: 本体が確かめで通した行のうち受け取った分（id_last と受け取った下書き）を本体の h で
    let obs: Vec<Observe<T::H>> = live
        .iter()
        .enumerate()
        .map(|(j, &i)| {
            let (pos0, n) = keep[j];
            Observe { seq: seqs[i].seq, pos0, tokens: &inputs[j][..=n], h: &out.h[j][..=n] }
        })
        .collect();
    let asks: Vec<(usize, DraftAsk)> = live.iter().filter_map(|&i| seqs[i].ask().map(|a| (i, a))).collect();
    let a2: Vec<DraftAsk> = asks.iter().map(|(_, a)| a.clone()).collect();
    let dr = d.run(&obs, &a2)?;
    if dr.len() != asks.len() {
        return Err(format!("下書きの数が合わない {} != {}", dr.len(), asks.len()));
    }
    for ((i, _), o) in asks.into_iter().zip(dr) {
        seqs[i].drafts = o.tokens;
        seqs[i].q = o.q;
    }
    Ok(live.len())
}

impl fmt::Display for SpecSeq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "seq {}: {} 語・{} 歩・受け取り {}/{}（1 歩 {:.2} 語）",
            self.seq,
            self.out.len(),
            self.n_steps,
            self.n_accepted,
            self.n_drafted,
            if self.n_steps > 0 { (self.n_accepted + self.n_steps) as f64 / self.n_steps as f64 } else { 0.0 }
        )
    }
}

// ---------------------------------------------------------------- 試しの道具（GPU なし）

/// 試し用の小さな言語モデル: 次の語の logits は直前の 2 語と種だけで決まる（ハッシュ）
#[derive(Clone, Copy, Debug)]
pub struct ToyLm {
    pub vocab: usize,
    pub seed: u64,
    /// logits の幅（大きいほど尖る）
    pub sharp: f32,
}

impl ToyLm {
    pub fn logits(&self, ctx: &[u32]) -> Vec<f32> {
        let a = ctx.last().copied().unwrap_or(u32::MAX);
        let b = if ctx.len() >= 2 { ctx[ctx.len() - 2] } else { u32::MAX };
        (0..self.vocab as u32)
            .map(|x| {
                let r = philox4x32_10([a, b, x, 7], [self.seed as u32, (self.seed >> 32) as u32]);
                (r[0] as f32 / u32::MAX as f32 - 0.5) * self.sharp
            })
            .collect()
    }
}

/// 偽の h: 本体がどの位置のどの文脈で出したか（下書き役が位置のずれを確かめる）
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToyH {
    pub pos: usize,
    pub ctx: u64,
}

fn ctx_hash(ctx: &[u32]) -> u64 {
    ctx.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x1000_0000_01b3))
}

/// 偽の本体（ToyLm）。列ごとに通した語を持つ。step は仮に全部足し、commit で長さを戻す
pub struct ToyTarget {
    pub lm: ToyLm,
    pub hist: std::collections::HashMap<SeqId, Vec<u32>>,
    pub n_rows: usize,
}

impl ToyTarget {
    pub fn new(lm: ToyLm) -> Self {
        ToyTarget { lm, hist: Default::default(), n_rows: 0 }
    }
}

impl Target for ToyTarget {
    type H = ToyH;
    fn step(&mut self, batch: &[SeqStep]) -> Result<StepOut<ToyH>, String> {
        let mut rows = vec![];
        let mut hs = vec![];
        for st in batch {
            let h = self.hist.entry(st.seq).or_default();
            if h.len() != st.pos0 {
                return Err(format!("本体: 列 {} の位置が合わない（通した {}・pos0 {}）", st.seq, h.len(), st.pos0));
            }
            let mut r = vec![];
            let mut hh = vec![];
            for (i, &tk) in st.tokens.iter().enumerate() {
                h.push(tk);
                let lg = self.lm.logits(h);
                let want = i >= st.out_from;
                r.push(RowOut { argmax: if want { argmax(&lg) } else { 0 }, dist: if want { st.sampling.map(|s| Dist::from_logits(&lg, &s)) } else { None } });
                hh.push(ToyH { pos: st.pos0 + i, ctx: ctx_hash(h) });
                self.n_rows += 1;
            }
            rows.push(r);
            hs.push(hh);
        }
        Ok(StepOut { rows, h: hs })
    }
    fn commit(&mut self, seq: SeqId, pos0: usize, n_keep: usize) -> Result<(), String> {
        let h = self.hist.get_mut(&seq).ok_or("本体: 知らない列")?;
        if h.len() < pos0 + n_keep {
            return Err("本体: 通していない所までは確定できない".into());
        }
        h.truncate(pos0 + n_keep);
        Ok(())
    }
    fn free(&mut self, seq: SeqId) {
        self.hist.remove(&seq);
    }
}

/// 偽の下書き役（本体と少し違う ToyLm）。MTP と同じ形で「KV」（通した語）と前の h を持ち、渡された h の位置を確かめる
pub struct ToyDrafter {
    pub lm: ToyLm,
    pub kv: std::collections::HashMap<SeqId, Vec<u32>>,
    pub last_h: std::collections::HashMap<SeqId, ToyH>,
}

impl ToyDrafter {
    pub fn new(lm: ToyLm) -> Self {
        ToyDrafter { lm, kv: Default::default(), last_h: Default::default() }
    }
}

impl Drafter<ToyH> for ToyDrafter {
    fn run(&mut self, obs: &[Observe<ToyH>], asks: &[DraftAsk]) -> Result<Vec<DraftOut>, String> {
        for o in obs {
            let kv = self.kv.entry(o.seq).or_default();
            if o.pos0 > kv.len() || o.h.len() != o.tokens.len() {
                return Err(format!("下書き: 追いつきに穴（KV {}・pos0 {}）", kv.len(), o.pos0));
            }
            kv.truncate(o.pos0);
            for (j, &tk) in o.tokens.iter().enumerate() {
                let p = o.pos0 + j;
                // 行 p の h は本体の位置 p-1 の h
                let h = if j == 0 { self.last_h.get(&o.seq).copied() } else { Some(o.h[j - 1]) };
                match h {
                    None if p == 0 => {}
                    Some(h) if h.pos + 1 == p => {}
                    _ => return Err(format!("下書き: 位置 {p} の行の h がずれている（{h:?}）")),
                }
                kv.push(tk);
            }
            self.last_h.insert(o.seq, *o.h.last().unwrap());
        }
        let mut res = vec![];
        for a in asks {
            let kv = self.kv.entry(a.seq).or_default();
            if a.pos != kv.len() {
                return Err(format!("下書き: 頼まれた位置 {} と KV の長さ {} が違う", a.pos, kv.len()));
            }
            match self.last_h.get(&a.seq) {
                Some(h) if h.pos + 1 == a.pos => {}
                h => return Err(format!("下書き: 1 段目の h がずれている（{h:?}・pos {}）", a.pos)),
            }
            let mut ctx = kv.clone();
            ctx.push(a.id_last);
            let mut toks = vec![];
            let mut qs = vec![];
            for j in 0..a.k {
                let lg = self.lm.logits(&ctx);
                let pos = (a.pos + j + 1) as u64;
                let tk = match a.sample {
                    None => argmax(&lg),
                    Some((s, seed, coupled)) => {
                        let q = Dist::from_logits(&lg, &s);
                        let tk = sample(&q, seed, pos, if coupled { TAG_SAMPLE } else { TAG_DRAFT });
                        qs.push(q);
                        tk
                    }
                };
                toks.push(tk);
                ctx.push(tk);
            }
            // MTP と同じく、下書きの行も KV に書く（次の追いつきが上書きする）
            kv.push(a.id_last);
            kv.extend_from_slice(&toks[..a.k.saturating_sub(1)]);
            res.push(DraftOut { tokens: toks, q: if a.sample.is_some() { Some(qs) } else { None } });
        }
        Ok(res)
    }
    fn free(&mut self, seq: SeqId) {
        self.kv.remove(&seq);
        self.last_h.remove(&seq);
    }
}

/// 下書きなしの普通のデコード（同じ取り方・同じ種）。比べる相手
pub fn decode_plain<T: Target>(t: &mut T, seq: SeqId, prompt: &[u32], s: Sampling, mode: Verify, seed: u64, n_new: usize, stop: &dyn Fn(u32) -> bool) -> Result<Vec<u32>, String> {
    let samp = if s.is_greedy() { None } else { Some(s) };
    let o = t.step(&[SeqStep { seq, pos0: 0, tokens: prompt, sampling: samp, out_from: prompt.len() - 1 }])?;
    let mut y = sample_row(mode, &o.rows[0][prompt.len() - 1], prompt.len() as u64, seed);
    let mut out = vec![y];
    let mut pos = prompt.len();
    while out.len() < n_new && !stop(y) {
        let o = t.step(&[SeqStep { seq, pos0: pos, tokens: &[y], sampling: samp, out_from: 0 }])?;
        pos += 1;
        y = sample_row(mode, &o.rows[0][0], pos as u64, seed);
        out.push(y);
    }
    Ok(out)
}

/// χ² の上側の点（Wilson–Hilferty。z は標準正規の上側の点。z = 3.72 で p ≈ 1e-4）
pub fn chi2_crit(df: usize, z: f64) -> f64 {
    let k = df as f64;
    let a = 2.0 / (9.0 * k);
    k * (1.0 - a + z * a.sqrt()).powi(3)
}

/// 数えた数と期待の確率で χ²（期待 5 未満の升はまとめる）。返り値 (χ², 自由度)
pub fn chi2(counts: &[u64], probs: &[f64]) -> (f64, usize) {
    let n: u64 = counts.iter().sum();
    let mut cells: Vec<(f64, f64)> = vec![];
    let (mut ro, mut re) = (0.0, 0.0);
    for (&c, &p) in counts.iter().zip(probs) {
        let e = p * n as f64;
        if e < 5.0 {
            ro += c as f64;
            re += e;
        } else {
            cells.push((c as f64, e));
        }
    }
    if re > 0.0 {
        cells.push((ro, re));
    } else if ro > 0.0 {
        // 期待 0 の升に出た（起きてはいけない語が出た）
        return (f64::INFINITY, cells.len().max(1));
    }
    let x: f64 = cells.iter().map(|&(o, e)| if e > 0.0 { (o - e) * (o - e) / e } else if o > 0.0 { f64::INFINITY } else { 0.0 }).sum();
    (x, cells.len().saturating_sub(1).max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    const Z: f64 = 3.72; // p ≈ 1e-4（種は決め打ちなので結果は毎回同じ）

    #[test]
    fn philox_kat() {
        // Random123 の kat_vectors（philox4x32 10）
        assert_eq!(philox4x32_10([0; 4], [0; 2]), [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]);
        assert_eq!(philox4x32_10([u32::MAX; 4], [u32::MAX; 2]), [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd]);
        assert_eq!(
            philox4x32_10([0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344], [0xa4093822, 0x299f31d0]),
            [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1]
        );
    }

    #[test]
    fn uniform_and_gumbel_moments() {
        let n = 200_000;
        let (mut s, mut s2, mut g) = (0.0, 0.0, 0.0);
        for i in 0..n {
            let u = uniform(42, i, TAG_SAMPLE, 3);
            assert!(u > 0.0 && u < 1.0);
            s += u;
            s2 += u * u;
            g += gumbel(42, i, TAG_SAMPLE, 3);
        }
        let m = s / n as f64;
        assert!((m - 0.5).abs() < 0.003, "平均 {m}");
        assert!((s2 / n as f64 - m * m - 1.0 / 12.0).abs() < 0.002);
        // Gumbel の平均はオイラー定数 0.5772（f64::consts::EULER_GAMMA はまだ安定版に無いので、目安の 4 桁を書く）
        #[allow(clippy::approx_constant)]
        const GUMBEL_MEAN: f64 = 0.5772;
        assert!((g / n as f64 - GUMBEL_MEAN).abs() < 0.01, "Gumbel の平均 {}", g / n as f64);
    }

    fn dists() -> Vec<Dist> {
        vec![
            Dist::from_weights(vec![(0, 0.7), (1, 0.2), (2, 0.1)]).unwrap(),
            Dist::from_weights((0..16).map(|i| (i, 1.0)).collect()).unwrap(),
            Dist::from_weights((0..12).map(|i| (i * 7 % 13, 0.6f64.powi(i as i32))).collect()).unwrap(),
            Dist::from_weights(vec![(5, 0.5), (9, 0.5)]).unwrap(),
            Dist::from_weights(vec![(3, 0.999), (4, 0.001)]).unwrap(),
        ]
    }

    #[test]
    fn gumbel_sampler_matches_p() {
        for (di, d) in dists().iter().enumerate() {
            let n = 200_000u64;
            let mut c = vec![0u64; d.ids.len()];
            for i in 0..n {
                let y = sample(d, 1000 + di as u64, i, TAG_SAMPLE);
                c[d.ids.iter().position(|&x| x == y).unwrap()] += 1;
            }
            let (x, df) = chi2(&c, &d.p);
            assert!(x < chi2_crit(df, Z), "分布 {di}: χ² {x:.1}（自由度 {df}・線 {:.1}）", chi2_crit(df, Z));
        }
    }

    #[test]
    fn from_logits_top_k_top_p() {
        let lg = [1.0f32, 3.0, 2.0, 3.0, -1.0];
        assert_eq!(Dist::from_logits(&lg, &Sampling::GREEDY), Dist::one_hot(1));
        let d = Dist::from_logits(&lg, &Sampling { temperature: 1.0, top_k: 3, top_p: 1.0, min_p: 0.0 });
        assert_eq!(d.ids, vec![1, 3, 2]);
        let z = 2.0 * 3f64.exp() + 2f64.exp();
        assert!((d.p[0] - 3f64.exp() / z).abs() < 1e-12);
        let d = Dist::from_logits(&lg, &Sampling { temperature: 1.0, top_k: 0, top_p: 0.5, min_p: 0.0 });
        assert_eq!(d.ids, vec![1, 3]);
        // 温度 0.5: 確率は 1・3 が 0.464、2 が 0.063、0 が 0.0085 → min_p 0.1（0.046 より下を切る）で 0 だけ落ちる
        let d = Dist::from_logits(&lg, &Sampling { temperature: 0.5, top_k: 0, top_p: 1.0, min_p: 0.1 });
        assert_eq!(d.ids, vec![1, 3, 2]);
    }

    /// 確かめ単体: 本体の分布 P_0, P_1(·|y0)、下書きの分布 Q。最初の 2 語の同時分布が P と同じか（Q によらず）
    #[test]
    fn verify_preserves_joint() {
        let v = 6u32;
        let p0 = Dist::from_weights((0..v).map(|i| (i, 1.0 + i as f64)).collect()).unwrap();
        let p1 = |y0: u32| Dist::from_weights((0..v).map(|i| (i, ((i * 3 + y0 * 5) % 7 + 1) as f64)).collect()).unwrap();
        let p2 = |y1: u32| Dist::from_weights((0..v).map(|i| (i, ((i + y1) % 4 + 1) as f64)).collect()).unwrap();
        // 下書きの分布（P と違う形・台も一部違う）
        let q0 = Dist::from_weights((0..v).map(|i| (i, if i == 2 { 0.0 } else { 6.0 - i as f64 })).collect()).unwrap();
        let q1 = |d0: u32| Dist::from_weights((0..v).map(|i| (i, ((i + d0) % 3 + 1) as f64)).collect()).unwrap();
        let exact = |a: u32, b: u32| p0.prob(a) * p1(a).prob(b);
        let stop = |_| false;
        for (mode, how) in [(Verify::Match, None), (Verify::Match, Some(true)), (Verify::Rs, Some(false))] {
            let n = 120_000u64;
            let mut c = vec![0u64; (v * v) as usize];
            let mut acc = 0usize;
            for i in 0..n {
                let seed = 77 + i * 1_000_003;
                let pos0 = i % 1000;
                // 下書き 2 語
                let (d0, d1) = match how {
                    None => (q0.argmax(), q1(q0.argmax()).argmax()),
                    Some(cp) => {
                        let tag = if cp { TAG_SAMPLE } else { TAG_DRAFT };
                        let d0 = sample(&q0, seed, pos0 + 1, tag);
                        (d0, sample(&q1(d0), seed, pos0 + 2, tag))
                    }
                };
                let rows = vec![
                    RowOut { argmax: p0.argmax(), dist: Some(p0.clone()) },
                    RowOut { argmax: 0, dist: Some(p1(d0)) },
                    RowOut { argmax: 0, dist: Some(p2(d1)) },
                ];
                let q = [q0.clone(), q1(d0)];
                let r = verify(mode, &[d0, d1], Some(&q), &rows, pos0, seed, &stop);
                acc += r.n_acc;
                let (a, b) = (r.tokens[0], if r.tokens.len() > 1 { r.tokens[1] } else { u32::MAX });
                if b == u32::MAX {
                    // 1 語目で外れた: 2 語目は P_1(·|a) から（下書きなしの普通のデコードと同じ取り方）
                    let b = sample(&p1(a), seed, pos0 + 2, TAG_SAMPLE);
                    c[(a * v + b) as usize] += 1;
                } else {
                    c[(a * v + b) as usize] += 1;
                }
            }
            let probs: Vec<f64> = (0..v * v).map(|k| exact(k / v, k % v)).collect();
            let (x, df) = chi2(&c, &probs);
            assert!(x < chi2_crit(df, Z), "{mode:?}/{how:?}: χ² {x:.1}（自由度 {df}・線 {:.1}）", chi2_crit(df, Z));
            assert!(acc > 0);
        }
    }

    /// Rs の 1 語目の受け取りの割合は Σ min(P, Q)、Match の one-hot の下書きは P(d)
    #[test]
    fn acceptance_rates() {
        let p = Dist::from_weights(vec![(0, 0.5), (1, 0.3), (2, 0.2)]).unwrap();
        let q = Dist::from_weights(vec![(0, 0.2), (1, 0.3), (2, 0.1), (3, 0.4)]).unwrap();
        let p1 = Dist::one_hot(0);
        let n = 100_000u64;
        let (mut a_rs, mut a_m) = (0u64, 0u64);
        for i in 0..n {
            let seed = 5 + i;
            let d = sample(&q, seed, 1, TAG_DRAFT);
            let rows = vec![RowOut { argmax: 0, dist: Some(p.clone()) }, RowOut { argmax: 0, dist: Some(p1.clone()) }];
            a_rs += verify(Verify::Rs, &[d], Some(&[q.clone()]), &rows, 0, seed, &|_| false).n_acc as u64;
            a_m += verify(Verify::Match, &[1], None, &rows, 0, seed, &|_| false).n_acc as u64;
        }
        let want_rs: f64 = (0..4).map(|i| p.prob(i).min(q.prob(i))).sum();
        let (r1, r2) = (a_rs as f64 / n as f64, a_m as f64 / n as f64);
        assert!((r1 - want_rs).abs() < 0.006, "Rs の受け取り {r1} ≠ {want_rs}");
        assert!((r2 - 0.3).abs() < 0.006, "Match の受け取り {r2} ≠ 0.3");
    }

    #[test]
    fn greedy_and_stop() {
        let rows: Vec<RowOut> = [4u32, 7, 9, 2].iter().map(|&a| RowOut { argmax: a, dist: None }).collect();
        let r = verify(Verify::Greedy, &[4, 7, 1], None, &rows, 10, 0, &|_| false);
        assert_eq!(r, Verdict { tokens: vec![4, 7, 9], n_acc: 2, stopped: false });
        let r = verify(Verify::Greedy, &[4, 7, 9], None, &rows, 10, 0, &|_| false);
        assert_eq!(r, Verdict { tokens: vec![4, 7, 9, 2], n_acc: 3, stopped: false });
        let r = verify(Verify::Greedy, &[5, 7, 9], None, &rows, 10, 0, &|_| false);
        assert_eq!(r.tokens, vec![4]);
        // 止めの語が下書きと同じでも、そこで終わる（後ろの下書きは出さない）
        let r = verify(Verify::Greedy, &[4, 7, 9], None, &rows, 10, 0, &|t| t == 7);
        assert_eq!(r, Verdict { tokens: vec![4, 7], n_acc: 1, stopped: true });
    }

    fn toy_pair(sharp_t: f32, same: bool) -> (ToyTarget, ToyDrafter) {
        let lm = ToyLm { vocab: 8, seed: 11, sharp: sharp_t };
        let dl = if same { lm } else { ToyLm { vocab: 8, seed: 12, sharp: sharp_t } };
        (ToyTarget::new(lm), ToyDrafter::new(dl))
    }

    /// 回し方: 下書きの数・人数・塊によらず、温度 0 と Match（同じ種）は下書きなしと語ごとに同じ
    #[test]
    fn loop_equals_plain_decode() {
        let stop_tok = 5u32;
        let stop = |t: u32| t == stop_tok;
        let t1 = Sampling { temperature: 1.0, top_k: 0, top_p: 1.0, min_p: 0.0 };
        // 下書きの取り方: argmax・結び付け（本体と同じ雑音）・独立（別の雑音）。Match はどれでも下書きなしと同じ出力になるはず
        let cases = [
            (Sampling::GREEDY, Verify::Greedy, None),
            (t1, Verify::Match, None),
            (t1, Verify::Match, Some(true)),
            (t1, Verify::Match, Some(false)),
            (Sampling::THINKING, Verify::Match, None),
            (Sampling::THINKING, Verify::Match, Some(true)),
        ];
        for (samp, mode, ds) in cases {
            for same in [true, false] {
                for k in [0usize, 1, 2, 3] {
                    let (mut t, mut d) = toy_pair(6.0, same);
                    let mut plain = ToyTarget::new(t.lm);
                    let prompts: Vec<Vec<u32>> = (0..6u32).map(|i| (0..(3 + i * 5)).map(|j| (j * 7 + i) % 8).collect()).collect();
                    let mut seqs: Vec<SpecSeq> = (0..6).map(|i| SpecSeq::new(i as SeqId, samp, 1234 + i as u64, k, 40)).collect();
                    for s in seqs.iter_mut() {
                        s.draft_sample = ds;
                    }
                    for (s, p) in seqs.iter_mut().zip(&prompts) {
                        start(&mut t, &mut d, s, p, 4, mode, &stop).unwrap();
                    }
                    while step(&mut t, &mut d, &mut seqs, mode, &stop).unwrap() > 0 {
                        // 本体の KV と下書きの KV（通した語）が合っている
                        for s in &seqs {
                            let th = &t.hist[&s.seq];
                            assert_eq!(th.len(), s.pos);
                            assert_eq!(&d.kv[&s.seq][..s.pos], &th[..]);
                        }
                    }
                    for (s, p) in seqs.iter().zip(&prompts) {
                        let want = decode_plain(&mut plain, s.seq, p, samp, mode, s.seed, 40, &stop).unwrap();
                        assert_eq!(s.out, want, "{mode:?}/{ds:?} k={k} same={same} seq {}", s.seq);
                        if same && k > 0 && (samp.is_greedy() || ds == Some(true)) {
                            // 同じモデルの下書きは、温度 0 か結び付けなら全部当たる。止めの語とその後ろの下書きは受け取りに数えないので、
                            // 止めの語で終わった列だけ最後の 1 歩で k まで減りうる
                            if s.out.last() == Some(&stop_tok) {
                                assert!(s.n_drafted - s.n_accepted <= k, "{s}");
                            } else {
                                assert_eq!(s.n_drafted, s.n_accepted, "{s}");
                            }
                        }
                    }
                }
            }
        }
    }

    /// 回し方全体（Rs と結び付けの Match）: 3 語の同時分布が本体の分布どおり
    #[test]
    fn loop_joint_distribution() {
        let samp = Sampling { temperature: 1.0, top_k: 0, top_p: 1.0, min_p: 0.0 };
        let lm = ToyLm { vocab: 6, seed: 21, sharp: 3.0 };
        let dl = ToyLm { vocab: 6, seed: 22, sharp: 3.0 };
        let prompt = [1u32, 4];
        // 正確な同時分布
        let pd = |ctx: &[u32]| Dist::from_logits(&lm.logits(ctx), &samp);
        let v = 6u32;
        let mut exact = vec![0f64; 216];
        for a in 0..v {
            for b in 0..v {
                for c in 0..v {
                    let p = pd(&[1, 4]).prob(a) * pd(&[1, 4, a]).prob(b) * pd(&[1, 4, a, b]).prob(c);
                    exact[(a * 36 + b * 6 + c) as usize] = p;
                }
            }
        }
        for (mode, ds) in [(Verify::Rs, Some(false)), (Verify::Match, Some(true)), (Verify::Match, None)] {
            for k in [1usize, 2, 3] {
                let mut t = ToyTarget::new(lm);
                let mut d = ToyDrafter::new(dl);
                let n = 30_000u64;
                let mut cnt = vec![0u64; 216];
                let (mut dr, mut ac) = (0, 0);
                for i in 0..n {
                    // 6 語まで出して頭の 3 語を見る（上限で下書きの数が削られないように）
                    let mut s = SpecSeq::new(i as SeqId, samp, 99 + i * 7919, k, 6);
                    s.draft_sample = ds;
                    start(&mut t, &mut d, &mut s, &prompt, 2, mode, &|_| false).unwrap();
                    let mut one = [s];
                    while step(&mut t, &mut d, &mut one, mode, &|_| false).unwrap() > 0 {}
                    let o = &one[0].out;
                    assert_eq!(o.len(), 6);
                    cnt[(o[0] * 36 + o[1] * 6 + o[2]) as usize] += 1;
                    dr += one[0].n_drafted;
                    ac += one[0].n_accepted;
                    t.free(i as SeqId);
                    d.free(i as SeqId);
                }
                let (x, df) = chi2(&cnt, &exact);
                assert!(x < chi2_crit(df, Z), "{mode:?}/{ds:?} k={k}: χ² {x:.1}（自由度 {df}・線 {:.1}）", chi2_crit(df, Z));
                assert!(ac > 0 && ac <= dr);
            }
        }
    }
}
