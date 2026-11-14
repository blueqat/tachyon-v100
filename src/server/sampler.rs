//! 次の語を CPU で選ぶ（エンジンが logits を返したとき。GPU で選ぶエンジンは使わない）。
//! 順番は HF transformers と同じ: 罰 → 温度 → top_k → top_p → min_p → 引く。
//! 乱数は（種・位置）から決める → まとめ方や歩の順番によらず同じ結果（GPU の Philox と同じ考え）。
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
pub use crate::model::RowCands;

#[derive(Clone, Debug, PartialEq)]
pub struct SamplingParams {
    /// 0（1e-5 未満）なら貪欲（argmax）
    pub temperature: f32,
    pub top_p: f32,
    /// 0 は使わない
    pub top_k: u32,
    pub min_p: f32,
    /// 出した語に 1 回でも出たら引く（OpenAI・vLLM と同じく出力の語だけ）
    pub presence_penalty: f32,
    /// 出た回数 × これを引く（出力の語だけ）
    pub frequency_penalty: f32,
    /// 1.0 はなし。入力と出力の語の logit を、正なら割り・負なら掛ける（vLLM と同じ）
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self::greedy()
    }
}

impl SamplingParams {
    pub fn greedy() -> Self {
        SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            seed: 0,
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature < 1e-5 || self.top_k == 1
    }

    pub fn has_penalty(&self) -> bool {
        self.presence_penalty != 0.0 || self.frequency_penalty != 0.0 || self.repetition_penalty != 1.0
    }
}

/// 1 列の語の数え（罰の計算用）
#[derive(Clone, Debug, Default)]
pub struct History {
    counts: HashMap<u32, u32>,
    /// 入力の語（repetition_penalty のときだけ持つ）
    prompt: HashSet<u32>,
}

impl History {
    pub fn new(prompt: &[u32], p: &SamplingParams) -> Self {
        let prompt = if p.repetition_penalty != 1.0 { prompt.iter().copied().collect() } else { HashSet::new() };
        History { counts: HashMap::new(), prompt }
    }

    pub fn push(&mut self, id: u32) {
        *self.counts.entry(id).or_default() += 1;
    }
}

/// NaN は -inf 扱い（エンジンの不具合でも止まらない）
fn clean(x: f32) -> f32 {
    if x.is_nan() {
        f32::NEG_INFINITY
    } else {
        x
    }
}

/// 値の大きい順・同じなら番号の小さい順
#[derive(Clone, Copy, PartialEq)]
struct Key(f32, u32);
impl Eq for Key {}
impl PartialOrd for Key {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Key {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.total_cmp(&o.0).then(o.1.cmp(&self.1))
    }
}

pub fn argmax(l: &[f32]) -> u32 {
    let mut best = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in l.iter().enumerate() {
        if clean(x) > bv {
            bv = x;
            best = i;
        }
    }
    best as u32
}

/// 大きい順に k 個（値, 番号）
fn top_k(l: &[f32], k: usize) -> Vec<(f32, u32)> {
    let k = k.min(l.len());
    let mut h: BinaryHeap<Reverse<Key>> = BinaryHeap::with_capacity(k + 1);
    for (i, &x) in l.iter().enumerate() {
        let key = Key(clean(x), i as u32);
        if h.len() < k {
            h.push(Reverse(key));
        } else if let Some(Reverse(min)) = h.peek() {
            if key > *min {
                h.pop();
                h.push(Reverse(key));
            }
        }
    }
    let mut v: Vec<Key> = h.into_iter().map(|Reverse(k)| k).collect();
    v.sort_by(|a, b| b.cmp(a));
    v.into_iter().map(|Key(x, i)| (x, i)).collect()
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// [0, 1) の一様な数（種と位置だけで決まる）
pub fn uniform(seed: u64, pos: u64) -> f64 {
    let x = splitmix64(seed ^ splitmix64(pos.wrapping_add(0x632B_E59B_D9B4_E019)));
    (x >> 11) as f64 / (1u64 << 53) as f64
}

fn apply_penalties(l: &mut [f32], p: &SamplingParams, h: &History) {
    if p.repetition_penalty != 1.0 {
        let r = p.repetition_penalty;
        let mut pen = |id: u32| {
            if let Some(x) = l.get_mut(id as usize) {
                *x = if *x > 0.0 { *x / r } else { *x * r };
            }
        };
        for &id in &h.prompt {
            pen(id);
        }
        for &id in h.counts.keys() {
            if !h.prompt.contains(&id) {
                pen(id);
            }
        }
    }
    if p.presence_penalty != 0.0 || p.frequency_penalty != 0.0 {
        for (&id, &c) in &h.counts {
            if let Some(x) = l.get_mut(id as usize) {
                *x -= p.presence_penalty + p.frequency_penalty * c as f32;
            }
        }
    }
}

/// 罰・温度・top_k・top_p・min_p をかけた後の分布（sample と下書きの受け取りが同じ物を使う）
pub enum Prepared {
    /// 1 語に決まる（貪欲・候補が無い・値が壊れている）
    One(u32),
    /// 候補（大きい順）と重み（和で割ると確率）
    Weighted(Vec<u32>, Vec<f64>),
}

impl Prepared {
    /// 語 id の確率
    pub fn prob(&self, id: u32) -> f64 {
        match self {
            Prepared::One(x) => (*x == id) as u8 as f64,
            Prepared::Weighted(c, w) => {
                let tot: f64 = w.iter().sum();
                c.iter().position(|&x| x == id).map_or(0.0, |i| w[i] / tot)
            }
        }
    }
    /// 一様な数 u（[0, 1)）で 1 語。skip の語は除いて残りで割り直す（下書きが外れたときの残りの分布）
    pub fn draw(&self, u: f64, skip: Option<u32>) -> u32 {
        match self {
            Prepared::One(x) => *x,
            Prepared::Weighted(c, w) => {
                let total: f64 = c.iter().zip(w).filter(|(&x, _)| Some(x) != skip).map(|(_, &wi)| wi).sum();
                if !(total > 0.0) {
                    return c[0];
                }
                let u = u * total;
                let mut acc = 0.0;
                let mut last = c[0];
                for (&x, &wi) in c.iter().zip(w) {
                    if Some(x) == skip {
                        continue;
                    }
                    acc += wi;
                    last = x;
                    if u < acc {
                        return x;
                    }
                }
                last
            }
        }
    }
}

/// logits から分布を作る。valid: 選んでよい番号の上限（tokenizer の語の数。GGUF の余りの [PAD…] は選ばない）
pub fn prepare(logits: &[f32], valid: usize, p: &SamplingParams, hist: &History) -> Prepared {
    let n = if valid == 0 { logits.len() } else { valid.min(logits.len()) };
    if n == 0 {
        return Prepared::One(0);
    }
    let buf;
    let l: &[f32] = if p.has_penalty() && (!hist.counts.is_empty() || !hist.prompt.is_empty()) {
        let mut v = logits[..n].to_vec();
        apply_penalties(&mut v, p, hist);
        buf = v;
        &buf
    } else {
        &logits[..n]
    };
    if p.is_greedy() {
        return Prepared::One(argmax(l));
    }
    let t = p.temperature;
    let m = l.iter().map(|&x| clean(x)).fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        return Prepared::One(argmax(l));
    }
    let w = |x: f32| (((x - m) / t) as f64).exp();
    // 候補（大きい順）と、top_p の分母（top_k のときは候補の和、無いときは全部の和）
    let (mut cand, norm): (Vec<(f32, u32)>, Option<f64>) = if p.top_k > 0 {
        (top_k(l, p.top_k as usize), None)
    } else if p.top_p < 1.0 {
        // 確率の和が top_p に届くまで候補を広げる（全部を並べ替えない）
        let z: f64 = l.iter().map(|&x| w(clean(x))).sum();
        let mut k = 64;
        let c = loop {
            let c = top_k(l, k);
            let mass: f64 = c.iter().map(|&(x, _)| w(x)).sum::<f64>() / z;
            if mass >= p.top_p as f64 || k >= n {
                break c;
            }
            k *= 4;
        };
        (c, Some(z))
    } else {
        // 全部（min_p があれば先に閾値で絞る）。並べ替えは要らない
        let floor = if p.min_p > 0.0 { m + t * p.min_p.ln() } else { f32::NEG_INFINITY };
        let c = l.iter().enumerate().filter(|&(_, &x)| clean(x) >= floor && clean(x) > f32::NEG_INFINITY).map(|(i, &x)| (x, i as u32)).collect();
        (c, None)
    };
    let mut ws: Vec<f64> = cand.iter().map(|&(x, _)| w(x)).collect();
    if p.top_p < 1.0 {
        let z = norm.unwrap_or_else(|| ws.iter().sum());
        let mut acc = 0.0;
        let mut keep = ws.len();
        for (i, wi) in ws.iter().enumerate() {
            acc += wi / z;
            if acc >= p.top_p as f64 {
                keep = i + 1;
                break;
            }
        }
        cand.truncate(keep);
        ws.truncate(keep);
    }
    if p.min_p > 0.0 {
        let thr = ws.iter().cloned().fold(0.0, f64::max) * p.min_p as f64;
        let mut i = 0;
        cand.retain(|_| {
            let k = ws[i] >= thr;
            i += 1;
            k
        });
        ws.retain(|&x| x >= thr);
    }
    let total: f64 = ws.iter().sum();
    if cand.is_empty() || !(total > 0.0) {
        return Prepared::One(argmax(l));
    }
    Prepared::Weighted(cand.into_iter().map(|(_, i)| i).collect(), ws)
}

/// logits から 1 語選ぶ。valid: 選んでよい番号の上限（tokenizer の語の数。GGUF の余りの [PAD…] は選ばない）。
/// pos: 乱数の位置（列の中の位置）
pub fn sample(logits: &[f32], valid: usize, p: &SamplingParams, hist: &History, pos: u64) -> u32 {
    prepare(logits, valid, p, hist).draw(uniform(p.seed, pos), None)
}

/// GPU の上位 K 語の候補（model::Model::cands）から、prepare と同じ分布を作る。候補だけでは同じにできない時は None
/// （手元で logits 全部から prepare する）。同じにできる条件:
///   罰は下げる向きだけ（presence・frequency ≥ 0、repetition ≥ 1。上げる罰は候補の外の語を上に出しうる）、
///   選ぶ語（貪欲の 1 位・top_k の k 個・top_p で残る語）が全部、候補の K 位の元の値（候補の外の語の上限）より真に大きい、
///   top_k 無しの top_p は罰が無い時だけ（分母 z は GPU が全部の語で数えた物）、top_k 無し・top_p 無しは全部の語が要る
pub fn prepare_cands(c: &RowCands, p: &SamplingParams, hist: &History) -> Option<Prepared> {
    let kk = c.ids.len();
    if kk == 0 {
        return None;
    }
    prepare_cands_b(c, c.vals[kk - 1], p, hist, false)
}

/// 絞り（response_format の JSON の形）つきの prepare_cands: 候補のうち出せる語だけで分布を作る。
/// 候補の外の語は bound（K 位の元の値）以下なので、貪欲は「出せる 1 位 > bound」なら全部の語に絞りをかけたのと同じ（同じでなければ None）。
/// 温度 > 0 は top_k・top_p を出せる候補の中で取る（候補の外＝256 位より下の語は確率に入れない近似。出せる 1 位 > bound の時だけ）
pub fn prepare_cands_masked(c: &RowCands, p: &SamplingParams, hist: &History, m: &super::jsonc::Matcher) -> Option<Prepared> {
    let kk = c.ids.len();
    if kk == 0 {
        return None;
    }
    let bound = c.vals[kk - 1];
    let (mut ids, mut vals) = (Vec::new(), Vec::new());
    for (&i, &x) in c.ids.iter().zip(&c.vals) {
        if m.allows(i) {
            ids.push(i);
            vals.push(x);
        }
    }
    if ids.is_empty() {
        return None;
    }
    let f = RowCands { ids, vals, m: f32::NAN, z: f64::NAN };
    prepare_cands_b(&f, bound, p, hist, true)
}

/// logits 全部に絞りをかけて prepare（候補で決められない時）
pub fn prepare_masked(logits: &[f32], valid: usize, p: &SamplingParams, hist: &History, m: &super::jsonc::Matcher) -> Prepared {
    let n = if valid == 0 { logits.len() } else { valid.min(logits.len()) };
    let mut v = logits[..n].to_vec();
    m.mask_logits(&mut v);
    prepare(&v, n, p, hist)
}

fn prepare_cands_b(c: &RowCands, bound: f32, p: &SamplingParams, hist: &History, masked: bool) -> Option<Prepared> {
    let kk = c.ids.len();
    if kk == 0 || p.presence_penalty < 0.0 || p.frequency_penalty < 0.0 || p.repetition_penalty < 1.0 {
        return None;
    }
    let pen = p.has_penalty() && (!hist.counts.is_empty() || !hist.prompt.is_empty());
    // 罰をかけた値（apply_penalties と同じ式・同じ順）
    let mut cand: Vec<(f32, u32)> = c
        .ids
        .iter()
        .zip(&c.vals)
        .map(|(&i, &x)| {
            let mut x = clean(x);
            if pen {
                if p.repetition_penalty != 1.0 && (hist.prompt.contains(&i) || hist.counts.contains_key(&i)) {
                    let r = p.repetition_penalty;
                    x = if x > 0.0 { x / r } else { x * r };
                }
                if p.presence_penalty != 0.0 || p.frequency_penalty != 0.0 {
                    if let Some(&n) = hist.counts.get(&i) {
                        x -= p.presence_penalty + p.frequency_penalty * n as f32;
                    }
                }
            }
            (x, i)
        })
        .collect();
    if pen {
        cand.sort_by(|a, b| Key(b.0, b.1).cmp(&Key(a.0, a.1)));
    }
    let top = cand[0];
    if !(top.0 > bound) || !top.0.is_finite() {
        return None;
    }
    if p.is_greedy() {
        return Some(Prepared::One(top.1));
    }
    let t = p.temperature;
    let m = top.0;
    let w = |x: f32| (((x - m) / t) as f64).exp();
    let norm: Option<f64> = if p.top_k > 0 {
        let k = p.top_k as usize;
        if masked {
            cand.truncate(k.min(kk));
        } else {
            if k >= kk || !(cand[k - 1].0 > bound) {
                return None;
            }
            cand.truncate(k);
        }
        None
    } else if masked {
        None // 出せる候補の中で（近似）
    } else if p.top_p < 1.0 && !pen {
        // z は GPU が行の最大 c.m で数えた物（罰が無いので m と同じ）
        if c.m != m || !(c.z > 0.0) {
            return None;
        }
        Some(c.z)
    } else {
        return None;
    };
    let mut ws: Vec<f64> = cand.iter().map(|&(x, _)| w(x)).collect();
    if p.top_p < 1.0 {
        let z = norm.unwrap_or_else(|| ws.iter().sum());
        let mut acc = 0.0;
        let mut keep = None;
        for (i, wi) in ws.iter().enumerate() {
            acc += wi / z;
            if acc >= p.top_p as f64 {
                keep = Some(i + 1);
                break;
            }
        }
        let keep = match keep {
            Some(k) => k,
            // top_k の候補の中なら全部残る（prepare と同じ）。top_k 無しで届かなければ候補が足りない
            None if norm.is_none() => ws.len(),
            None => return None,
        };
        if norm.is_some() && !(cand[keep - 1].0 > bound) {
            return None;
        }
        cand.truncate(keep);
        ws.truncate(keep);
    }
    if p.min_p > 0.0 {
        let thr = ws.iter().cloned().fold(0.0, f64::max) * p.min_p as f64;
        let mut i = 0;
        cand.retain(|_| {
            let k = ws[i] >= thr;
            i += 1;
            k
        });
        ws.retain(|&x| x >= thr);
    }
    let total: f64 = ws.iter().sum();
    if cand.is_empty() || !(total > 0.0) {
        return None;
    }
    Some(Prepared::Weighted(cand.into_iter().map(|(_, i)| i).collect(), ws))
}

/// sample の候補版（None なら手元で logits 全部から）
pub fn sample_cands(c: &RowCands, p: &SamplingParams, hist: &History, pos: u64) -> Option<u32> {
    prepare_cands(c, p, hist).map(|pr| pr.draw(uniform(p.seed, pos), None))
}

/// spec_row の分布を作る所だけ分けた物（prepare か prepare_cands の結果で受け取りを決める）
pub fn spec_row_prepared(pr: Prepared, p: &SamplingParams, pos: u64, draft: Option<u32>) -> Result<(), u32> {
    match draft {
        None => Err(pr.draw(uniform(p.seed, pos), None)),
        Some(d) => {
            if let Prepared::One(x) = pr {
                return if x == d { Ok(()) } else { Err(x) };
            }
            let pd = pr.prob(d);
            if pd > 0.0 && uniform(p.seed ^ ACCEPT_TAG, pos) < pd {
                Ok(())
            } else {
                Err(pr.draw(uniform(p.seed, pos), Some(d)))
            }
        }
    }
}

/// GPU の候補で抜き取るか（既定 1。TACHYON_GPU_SAMPLE=0 で前の道＝logits 全部を手元へ）
pub fn gpu_sample_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TACHYON_GPU_SAMPLE").map_or(true, |v| v != "0"))
}

/// 下書きの受け取りの乱数の種を本体の取り方と分ける印
const ACCEPT_TAG: u64 = 0xA5A5_5A5A_C3C3_3C3C;

/// 下書きの確かめの 1 行（speculative sampling。下書きは 1 語に決めた物＝Q は one-hot）:
/// 分布 P（この行の logits に sample と同じ加工）で、下書き d を確率 P(d) で受け取る。外れたら P から d を除いて割り直した分布で 1 語。
/// 返り値 Ok(()) は受け取り、Err(y) は外れて y を出す。どちらでも出る語の分布は P と同じ（Leviathan et al. 2023）。
/// draft が None（下書きの後ろの行）は P から普通に 1 語（Err で返す）。pos は出す語の位置（sample と同じ）
pub fn spec_row(logits: &[f32], valid: usize, p: &SamplingParams, hist: &History, pos: u64, draft: Option<u32>) -> Result<(), u32> {
    spec_row_prepared(prepare(logits, valid, p, hist), p, pos, draft)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(t: f32) -> SamplingParams {
        SamplingParams { temperature: t, seed: 7, ..SamplingParams::greedy() }
    }

    #[test]
    fn greedy_and_ties() {
        let l = [0.1, 3.0, 3.0, -1.0, f32::NAN];
        assert_eq!(sample(&l, 0, &p(0.0), &History::default(), 0), 1);
        assert_eq!(sample(&l, 0, &SamplingParams { top_k: 1, ..p(1.0) }, &History::default(), 0), 1);
        // valid で後ろを選ばない
        let l = [0.0, 1.0, 9.0];
        assert_eq!(sample(&l, 2, &p(0.0), &History::default(), 0), 1);
    }

    #[test]
    fn distribution_matches() {
        // 3 語 [ln 1, ln 2, ln 7] → 確率 0.1・0.2・0.7
        let l = [1f32.ln(), 2f32.ln(), 7f32.ln()];
        let mut c = [0usize; 3];
        let n = 40_000;
        for pos in 0..n {
            c[sample(&l, 0, &p(1.0), &History::default(), pos) as usize] += 1;
        }
        for (i, want) in [0.1, 0.2, 0.7].iter().enumerate() {
            let got = c[i] as f64 / n as f64;
            assert!((got - want).abs() < 0.01, "{i}: {got} vs {want}");
        }
        // 同じ種・同じ位置は同じ語
        let a: Vec<u32> = (0..50).map(|pos| sample(&l, 0, &p(1.0), &History::default(), pos)).collect();
        let b: Vec<u32> = (0..50).map(|pos| sample(&l, 0, &p(1.0), &History::default(), pos)).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn filters() {
        let l = [1f32.ln(), 2f32.ln(), 7f32.ln()];
        // top_p 0.6 → 0.7 の語だけ
        let q = SamplingParams { top_p: 0.6, ..p(1.0) };
        assert!((0..200).all(|pos| sample(&l, 0, &q, &History::default(), pos) == 2));
        // top_k 2 → 0 番は出ない
        let q = SamplingParams { top_k: 2, ..p(1.0) };
        assert!((0..500).all(|pos| sample(&l, 0, &q, &History::default(), pos) != 0));
        // min_p 0.25（0.7×0.25 = 0.175 未満を捨てる）→ 0 番は出ない
        let q = SamplingParams { min_p: 0.25, ..p(1.0) };
        assert!((0..500).all(|pos| sample(&l, 0, &q, &History::default(), pos) != 0));
        // top_k なし・top_p で広げる道（語が多い）
        let mut big = vec![0f32; 10_000];
        big[1234] = 20.0;
        let q = SamplingParams { top_p: 0.9, ..p(1.0) };
        assert!((0..100).all(|pos| sample(&big, 0, &q, &History::default(), pos) == 1234));
    }

    #[test]
    fn spec_row_keeps_the_distribution() {
        // 下書き d の受け取り／外れの後の語の分布が P と同じ（d がよく出る語・あまり出ない語・候補の外の語）
        let l = [1f32.ln(), 2f32.ln(), 7f32.ln(), 0.0, -50.0];
        let q = SamplingParams { top_k: 4, ..p(1.0) };
        let h = History::default();
        let pr = prepare(&l, 0, &q, &h);
        let want: Vec<f64> = (0..5).map(|i| pr.prob(i)).collect();
        assert!((want.iter().sum::<f64>() - 1.0).abs() < 1e-9 && want[4] == 0.0);
        for d in [2u32, 0, 4] {
            let mut c = [0usize; 5];
            let n = 60_000;
            for pos in 0..n {
                let y = match spec_row(&l, 0, &q, &h, pos, Some(d)) {
                    Ok(()) => d,
                    Err(y) => y,
                };
                c[y as usize] += 1;
            }
            for i in 0..5 {
                let got = c[i] as f64 / n as f64;
                assert!((got - want[i]).abs() < 0.008, "下書き {d}・語 {i}: {got} vs {}", want[i]);
            }
        }
        // 貪欲は argmax と同じときだけ受け取る
        let g = p(0.0);
        assert_eq!(spec_row(&l, 0, &g, &h, 0, Some(2)), Ok(()));
        assert_eq!(spec_row(&l, 0, &g, &h, 0, Some(1)), Err(2));
        // 罰をかけた後の分布で
        let mut h2 = History::new(&[], &g);
        h2.push(2);
        let pg = SamplingParams { presence_penalty: 5.0, ..p(0.0) };
        assert_eq!(spec_row(&l, 0, &pg, &h2, 0, Some(2)), Err(1));
    }

    #[test]
    fn penalties() {
        let l = [2.0, 1.9, 0.0];
        let mut h = History::new(&[], &p(0.0));
        h.push(0);
        let q = SamplingParams { presence_penalty: 0.5, ..p(0.0) };
        assert_eq!(sample(&l, 0, &q, &h, 0), 1);
        let q = SamplingParams { frequency_penalty: 0.2, ..p(0.0) };
        assert_eq!(sample(&l, 0, &q, &h, 0), 1);
        let q = SamplingParams { repetition_penalty: 1.1, ..p(0.0) };
        let h2 = History::new(&[0], &q);
        assert_eq!(sample(&l, 0, &q, &h2, 0), 1);
    }
}
