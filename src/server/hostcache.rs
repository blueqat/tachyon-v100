//! 頭の控えの RAM の段（前置きの計算を RAM に控え、その都度読み込む）。
//!
//! GPU の控え（m1.rs の PrefixCache・数個）の下に、固定した RAM（HostArena・zero-copy）の大きな段を置く。
//!   - 頁（64 語）ごとの KV（Attention の層の K・V ＋ MTP の K・V）を、頭からの語の鎖の指紋で 1 つずつ持つ（同じ頭は 1 回だけ）
//!   - 控え（checkpoint）= 位置 p までの語・その頁の並び・位置 p の GDN の状態（3.6 は GDN の層が状態を持つので、
//!     ちょうど p の所の写しが要る）・位置 p-1 の h（MTP の続き）。取るのは system の終わり・最後の回の頭・
//!     every 語ごと・手番の頭（turn_start）・先計算の札の終わり
//!   - 新しい依頼: プロンプトの頭と語が全部同じ控えのうち、いちばん長い物を RAM から GPU へ写し、残りだけ計算する
//!   - 先計算（prewarm）の控えは pinned（追い出さない）・public（どのアカウントでも当たる）
//! ⛔ 当たりは「頭から語が 1 つ残らず同じ」ときだけ（語を全部比べる。指紋だけで決めない）→ 別の人の文は、同じ文でない限り当たらない。
//! ⛔ public でない控えは、同じアカウント（scope）の依頼だけが使う（GPU の控えと同じ決まり）。
//! ⛔ 中身（語）はログに出さない。出すのは数・語の数・バイト・時間だけ。
use crate::kv::{ffi as kf, HostArena};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

pub const PAGE: usize = 64;

/// /health に出す数（エンジンの糸が書き、口の糸が読む）
pub static STATS: Mutex<Option<Value>> = Mutex::new(None);

#[derive(Clone, Debug)]
pub struct HostCacheConfig {
    /// RAM の段の上限（バイト）。0 なら使わない
    pub cap_bytes: usize,
    /// 控えを取る間隔（語。0 なら区切りの所だけ）
    pub every: usize,
    /// これより短い控えは取らない（語）
    pub min_tokens: usize,
    /// RAM の空き（MemAvailable）がこれを切ったら段を広げない（バイト）
    pub min_avail: usize,
    /// 固定する RAM を置く NUMA の node（GPU の近く）
    pub node: Option<i32>,
}

/// 同じ大きさの物を並べる置き場（固定した RAM。足りなくなったら塊ごとに広げる）
struct Pool {
    item: usize,
    per_chunk: usize,
    chunks: Vec<HostArena>,
    free: Vec<u32>,
    used: usize,
}

impl Pool {
    fn new(item: usize, per_chunk: usize) -> Self {
        let item = item.div_ceil(256) * 256;
        Pool { item, per_chunk: per_chunk.max(1), chunks: vec![], free: vec![], used: 0 }
    }
    fn chunk_bytes(&self) -> usize {
        self.item * self.per_chunk
    }
    fn bytes(&self) -> usize {
        self.chunks.iter().map(|c| c.bytes).sum()
    }
    /// (手元の番地, GPU から見た番地)
    fn ptr(&self, id: u32) -> (*mut u8, *mut u8) {
        let (c, k) = (id as usize / self.per_chunk, id as usize % self.per_chunk);
        let a = &self.chunks[c];
        unsafe { (a.ptr.add(k * self.item), a.dev.add(k * self.item)) }
    }
    fn take(&mut self) -> Option<u32> {
        let id = self.free.pop()?;
        self.used += 1;
        Some(id)
    }
    fn grow(&mut self, node: Option<i32>) {
        let c = self.chunks.len();
        self.chunks.push(HostArena::new(self.chunk_bytes(), node));
        // 番号の小さい物から使う（pop は後ろから）
        self.free.extend(((c * self.per_chunk) as u32..((c + 1) * self.per_chunk) as u32).rev());
    }
    fn give(&mut self, id: u32) {
        self.used -= 1;
        self.free.push(id);
    }
}

struct PageEnt {
    frame: u32,
    parent: u64,
    toks: [u32; PAGE],
    refs: u32,
}

struct Ckpt {
    scope: String,
    public: bool,
    pinned: bool,
    name: Option<String>,
    toks: Vec<u32>,
    /// 満ちた頁の鍵（頭から順）
    pages: Vec<u64>,
    /// 終わりの満ちていない頁（この控えだけの物）
    tail: Option<u32>,
    /// GDN の状態・conv・h（state プールの 1 つ）
    state: u32,
    used: u64,
    hits: u64,
}

pub struct HostCache {
    cfg: HostCacheConfig,
    frame_bytes: usize,
    /// GDN の状態・conv・h の大きさ
    st: (usize, usize, usize),
    frames: Pool,
    states: Pool,
    pages: HashMap<u64, PageEnt>,
    ckpts: Vec<Ckpt>,
    tick: u64,
    // 数え
    n_saved: u64,
    n_restored: u64,
    restored_tokens: u64,
    n_evicted: u64,
    n_skipped: u64,
    last_restore_ms: f64,
    last_save_ms: f64,
}

fn log(v: Value) {
    let mut v = v;
    v["t"] = super::util::utc_now().into();
    eprintln!("{v}");
}

const SEED: u64 = 0x9e37_79b9_7f4a_7c15;

fn mix(mut h: u64, toks: &[u32]) -> u64 {
    for &t in toks {
        h ^= t as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        h ^= h >> 29;
    }
    h
}

/// 頭から k 番目までの満ちた頁の鍵（鎖）
fn chain(toks: &[u32], n_full: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(n_full);
    let mut h = SEED;
    for k in 0..n_full {
        h = mix(h.rotate_left(17) ^ (k as u64), &toks[k * PAGE..(k + 1) * PAGE]);
        out.push(h);
    }
    out
}

fn mem_available() -> Option<usize> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let l = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: usize = l.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// GPU と RAM の間のまとめた写し（kv_copy・1 回の起動で何百でも。終わるまで待つ）
fn copy_all(d: &[kf::CopyDesc]) -> Result<(), String> {
    if d.is_empty() {
        return Ok(());
    }
    let r = unsafe { kf::kv_copy(d.as_ptr(), d.len() as i32, 0, std::ptr::null_mut()) };
    if r != 0 {
        return Err(format!("kv_copy {r}"));
    }
    crate::cuda::sync();
    Ok(())
}

impl HostCache {
    /// frame_bytes = 1 頁ぶんの KV（全部の Attention の層の K・V ＋ MTP）。st = (GDN の状態, conv, h) のバイト
    pub fn new(cfg: HostCacheConfig, frame_bytes: usize, st: (usize, usize, usize)) -> Self {
        // 塊は 256MB 前後（広げる手間と、上限の細かさの間）
        let fpc = ((256usize << 20) / frame_bytes.max(1)).max(16);
        let st_item = st.0 + st.1 + st.2;
        let spc = ((512usize << 20) / st_item.max(1)).max(2);
        let hc = HostCache {
            frames: Pool::new(frame_bytes, fpc),
            states: Pool::new(st_item, spc),
            cfg,
            frame_bytes,
            st,
            pages: HashMap::new(),
            ckpts: vec![],
            tick: 0,
            n_saved: 0,
            n_restored: 0,
            restored_tokens: 0,
            n_evicted: 0,
            n_skipped: 0,
            last_restore_ms: 0.0,
            last_save_ms: 0.0,
        };
        hc.publish();
        hc
    }

    pub fn every(&self) -> usize {
        self.cfg.every
    }
    pub fn min_tokens(&self) -> usize {
        self.cfg.min_tokens
    }

    fn bytes(&self) -> usize {
        self.frames.bytes() + self.states.bytes()
    }

    /// プロンプトの頭と語が全部同じ控えのうち、いちばん長い物（prompt より短い物だけ）
    pub fn find(&self, scope: &str, prompt: &[u32]) -> Option<(usize, usize)> {
        self.ckpts
            .iter()
            .enumerate()
            .filter(|(_, c)| (c.public || c.scope == scope) && c.toks.len() < prompt.len() && prompt.starts_with(&c.toks))
            .max_by_key(|(_, c)| c.toks.len())
            .map(|(i, c)| (i, c.toks.len()))
    }

    pub fn toks(&self, idx: usize) -> &[u32] {
        &self.ckpts[idx].toks
    }

    /// 控え idx を GPU へ: gpu[k] = 頁 k の置き場（frame の並びと同じ順）、state = GDN の状態・conv の置き場、h = MTP の h の置き場
    pub fn restore(&mut self, idx: usize, gpu: &[Vec<(*mut u8, usize)>], state: [(*mut u8, usize); 2], h: (*mut u8, usize)) -> Result<usize, String> {
        let t0 = Instant::now();
        let c = &self.ckpts[idx];
        let l = c.toks.len();
        let n_full = l / PAGE;
        let n_pages = l.div_ceil(PAGE);
        if gpu.len() < n_pages {
            return Err("restore: 頁が足りない".into());
        }
        if state[0].1 != self.st.0 || state[1].1 != self.st.1 || h.1 != self.st.2 {
            return Err("restore: 状態の大きさが違う".into());
        }
        // 頁の並び（層・K・V・MTP）が控えを取ったときと同じ大きさか（違えば別の層へ書いてしまう）
        if gpu[..n_pages].iter().any(|v| v.iter().map(|x| x.1).sum::<usize>() != self.frame_bytes) {
            return Err("restore: 頁の大きさが違う".into());
        }
        let mut d = Vec::with_capacity(n_pages * gpu.first().map_or(0, |v| v.len()) + 3);
        let mut push_frame = |frame: u32, spans: &[(*mut u8, usize)]| {
            let (_, dev) = self.frames.ptr(frame);
            let mut off = 0;
            for &(dst, n) in spans {
                d.push(kf::CopyDesc { src: unsafe { dev.add(off) }, dst, bytes: n as u64 });
                off += n;
            }
        };
        for (k, key) in c.pages.iter().enumerate() {
            let e = self.pages.get(key).ok_or("restore: 頁が無い")?;
            push_frame(e.frame, &gpu[k]);
        }
        if let Some(t) = c.tail {
            push_frame(t, &gpu[n_full]);
        }
        let (_, sdev) = self.states.ptr(c.state);
        d.push(kf::CopyDesc { src: sdev, dst: state[0].0, bytes: self.st.0 as u64 });
        d.push(kf::CopyDesc { src: unsafe { sdev.add(self.st.0) }, dst: state[1].0, bytes: self.st.1 as u64 });
        d.push(kf::CopyDesc { src: unsafe { sdev.add(self.st.0 + self.st.1) }, dst: h.0, bytes: self.st.2 as u64 });
        copy_all(&d)?;
        self.tick += 1;
        let c = &mut self.ckpts[idx];
        c.used = self.tick;
        c.hits += 1;
        self.n_restored += 1;
        self.restored_tokens += l as u64;
        self.last_restore_ms = t0.elapsed().as_secs_f64() * 1000.0;
        log(json!({"ev": "prefix_host_restore", "tokens": l, "pages": n_pages, "public": c.public, "pinned": c.pinned,
                                         "ms": (self.last_restore_ms * 10.0).round() / 10.0}));
        self.publish();
        Ok(l)
    }

    /// 同じ控え（同じ語・同じ持ち主）が既にあるか（あれば使った印だけ付ける）
    pub fn touch_same(&mut self, scope: &str, public: bool, toks: &[u32]) -> bool {
        self.tick += 1;
        let t = self.tick;
        for c in self.ckpts.iter_mut() {
            if c.toks.len() == toks.len() && (c.public == public) && (public || c.scope == scope) && c.toks == toks {
                c.used = t;
                return true;
            }
        }
        false
    }

    /// いちばん古い（pinned でない）控えを捨てる。捨てる物が無ければ false
    fn evict_one(&mut self) -> bool {
        let Some(i) = (0..self.ckpts.len()).filter(|&i| !self.ckpts[i].pinned).min_by_key(|&i| self.ckpts[i].used) else {
            return false;
        };
        let c = self.ckpts.swap_remove(i);
        self.drop_ckpt(c);
        self.n_evicted += 1;
        true
    }

    fn drop_ckpt(&mut self, c: Ckpt) {
        for key in &c.pages {
            let gone = match self.pages.get_mut(key) {
                Some(e) => {
                    e.refs -= 1;
                    e.refs == 0
                }
                None => false,
            };
            if gone {
                let e = self.pages.remove(key).unwrap();
                self.frames.give(e.frame);
            }
        }
        if let Some(t) = c.tail {
            self.frames.give(t);
        }
        self.states.give(c.state);
    }

    /// 置き場から 1 つ取る（無ければ上限・RAM の空きの中で広げる → それも駄目なら古い控えを捨てる）。pinned の控えは捨てない
    fn alloc(&mut self, frames: bool) -> Option<u32> {
        loop {
            let got = if frames { self.frames.take() } else { self.states.take() };
            if got.is_some() {
                return got;
            }
            let more = if frames { self.frames.chunk_bytes() } else { self.states.chunk_bytes() };
            let room = self.bytes() + more <= self.cfg.cap_bytes && mem_available().map_or(true, |a| a >= self.cfg.min_avail + more);
            if room {
                if frames {
                    self.frames.grow(self.cfg.node);
                } else {
                    self.states.grow(self.cfg.node);
                }
                continue;
            }
            if !self.evict_one() {
                return None;
            }
        }
    }

    /// 控えを取る（GPU → RAM）。gpu[k] = 頁 k の置き場、state・h = GDN の状態・conv・MTP の h の今の置き場
    #[allow(clippy::too_many_arguments)]
    pub fn save(&mut self, scope: &str, public: bool, pinned: bool, name: Option<String>, toks: &[u32], gpu: &[Vec<(*mut u8, usize)>],
                state: [(*mut u8, usize); 2], h: (*mut u8, usize)) -> Result<bool, String> {
        let t0 = Instant::now();
        let l = toks.len();
        if l == 0 || self.cfg.cap_bytes == 0 {
            return Ok(false);
        }
        if self.touch_same(scope, public, toks) {
            return Ok(false);
        }
        // 同じ名前の札の前の版（文が変わった）は、固定を外す（ふつうの控えとして古い順に捨てられる）
        if pinned && name.is_some() {
            for c in self.ckpts.iter_mut().filter(|c| c.pinned && c.name == name) {
                c.pinned = false;
            }
        }
        let n_full = l / PAGE;
        if gpu.len() < l.div_ceil(PAGE) || state[0].1 != self.st.0 || state[1].1 != self.st.1 || h.1 != self.st.2 {
            return Err("save: 置き場の形が違う".into());
        }
        if gpu.first().map_or(0, |v| v.iter().map(|x| x.1).sum::<usize>()) != self.frame_bytes {
            return Err("save: 頁の大きさが違う".into());
        }
        let keys = chain(toks, n_full);
        // 先に置き場を全部取る（途中で足りなくなったら返して取りやめ）。新しい頁だけ写す
        let mut new_pages: Vec<(usize, u32)> = vec![];
        let mut ok = true;
        for k in 0..n_full {
            let parent = if k == 0 { SEED } else { keys[k - 1] };
            match self.pages.get(&keys[k]) {
                Some(e) if e.parent == parent && e.toks[..] == toks[k * PAGE..(k + 1) * PAGE] => {}
                Some(_) => {
                    // 指紋がぶつかった（語が違う）: この控えは取らない
                    ok = false;
                    break;
                }
                None => match self.alloc(true) {
                    Some(f) => new_pages.push((k, f)),
                    None => {
                        ok = false;
                        break;
                    }
                },
            }
        }
        let tail = if ok && l % PAGE != 0 {
            let t = self.alloc(true);
            ok = t.is_some();
            t
        } else {
            None
        };
        let st_id = if ok { self.alloc(false) } else { None };
        // alloc が古い控えを捨てたとき、その頁が new_pages の鍵と同じなら（同じ頭の古い控え）、もう一度確かめる
        // （取り直しの alloc がまた捨てることがあるので、欠けが無くなるまで）
        while ok && st_id.is_some() {
            let Some(k) = (0..n_full).find(|&k| !self.pages.contains_key(&keys[k]) && !new_pages.iter().any(|&(kk, _)| kk == k)) else { break };
            match self.alloc(true) {
                Some(f) => new_pages.push((k, f)),
                None => ok = false,
            }
        }
        if !ok || st_id.is_none() {
            for (_, f) in new_pages {
                self.frames.give(f);
            }
            if let Some(t) = tail {
                self.frames.give(t);
            }
            if let Some(s) = st_id {
                self.states.give(s);
            }
            self.n_skipped += 1;
            self.publish();
            return Ok(false);
        }
        let st_id = st_id.unwrap();
        let mut d = vec![];
        let mut push_frame = |frame: u32, spans: &[(*mut u8, usize)], frames: &Pool| {
            let (_, dev) = frames.ptr(frame);
            let mut off = 0;
            for &(src, n) in spans {
                d.push(kf::CopyDesc { src, dst: unsafe { dev.add(off) }, bytes: n as u64 });
                off += n;
            }
        };
        for &(k, f) in &new_pages {
            push_frame(f, &gpu[k], &self.frames);
        }
        if let Some(t) = tail {
            push_frame(t, &gpu[n_full], &self.frames);
        }
        let (_, sdev) = self.states.ptr(st_id);
        d.push(kf::CopyDesc { src: state[0].0, dst: sdev, bytes: self.st.0 as u64 });
        d.push(kf::CopyDesc { src: state[1].0, dst: unsafe { sdev.add(self.st.0) }, bytes: self.st.1 as u64 });
        d.push(kf::CopyDesc { src: h.0, dst: unsafe { sdev.add(self.st.0 + self.st.1) }, bytes: self.st.2 as u64 });
        if let Err(e) = copy_all(&d) {
            for (_, f) in new_pages {
                self.frames.give(f);
            }
            if let Some(t) = tail {
                self.frames.give(t);
            }
            self.states.give(st_id);
            return Err(e);
        }
        for &(k, f) in &new_pages {
            let parent = if k == 0 { SEED } else { keys[k - 1] };
            let mut tk = [0u32; PAGE];
            tk.copy_from_slice(&toks[k * PAGE..(k + 1) * PAGE]);
            self.pages.insert(keys[k], PageEnt { frame: f, parent, toks: tk, refs: 0 });
        }
        for key in &keys {
            self.pages.get_mut(key).unwrap().refs += 1;
        }
        self.tick += 1;
        self.ckpts.push(Ckpt {
            scope: scope.to_string(),
            public,
            pinned,
            name,
            toks: toks.to_vec(),
            pages: keys,
            tail,
            state: st_id,
            used: self.tick,
            hits: 0,
        });
        self.n_saved += 1;
        self.last_save_ms = t0.elapsed().as_secs_f64() * 1000.0;
        log(json!({"ev": "prefix_host_save", "tokens": l, "new_pages": new_pages.len(), "public": public, "pinned": pinned,
                                         "ms": (self.last_save_ms * 10.0).round() / 10.0, "ckpts": self.ckpts.len()}));
        self.publish();
        Ok(true)
    }

    pub fn stats(&self) -> Value {
        let gib = |b: usize| (b as f64 / (1u64 << 30) as f64 * 1000.0).round() / 1000.0;
        let pinned: Vec<Value> = self
            .ckpts
            .iter()
            .filter(|c| c.pinned)
            .map(|c| json!({"name": c.name, "tokens": c.toks.len(), "hits": c.hits, "public": c.public}))
            .collect();
        let st_item = self.states.item;
        let pin_bytes: usize = self
            .ckpts
            .iter()
            .filter(|c| c.pinned)
            .map(|c| c.pages.len() * self.frames.item + c.tail.map_or(0, |_| self.frames.item) + st_item)
            .sum();
        json!({
            "on": self.cfg.cap_bytes > 0,
            "cap_gib": gib(self.cfg.cap_bytes),
            "ram_gib": gib(self.bytes()),
            "used_gib": gib(self.frames.used * self.frames.item + self.states.used * st_item),
            "checkpoints": self.ckpts.len(),
            "pages": self.pages.len(),
            "tokens_held": self.ckpts.iter().map(|c| c.toks.len()).sum::<usize>(),
            "every": self.cfg.every,
            "pinned": {"count": pinned.len(), "tokens": self.ckpts.iter().filter(|c| c.pinned).map(|c| c.toks.len()).sum::<usize>(),
                       "ram_gib": gib(pin_bytes), "vram_gib": 0.0, "list": pinned},
            "saved": self.n_saved, "restored": self.n_restored, "restored_tokens": self.restored_tokens,
            "evicted": self.n_evicted, "skipped": self.n_skipped,
            "last_restore_ms": (self.last_restore_ms * 10.0).round() / 10.0, "last_save_ms": (self.last_save_ms * 10.0).round() / 10.0,
            "frame_kib": self.frames.item / 1024, "state_mib": (st_item as f64 / (1u64 << 20) as f64 * 10.0).round() / 10.0,
        })
    }

    fn publish(&self) {
        if let Ok(mut g) = STATS.lock() {
            *g = Some(self.stats());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_depends_on_whole_prefix() {
        let a: Vec<u32> = (0..256).collect();
        let mut b = a.clone();
        b[3] = 999;
        let ka = chain(&a, 4);
        let kb = chain(&b, 4);
        // 頭の 1 語が違えば、後ろの頁の鍵も全部違う（同じ 2 頁目の語でも、頭が違えば別の物）
        assert!(ka.iter().zip(&kb).all(|(x, y)| x != y));
        let mut c = a.clone();
        c[200] = 7;
        let kc = chain(&c, 4);
        assert_eq!(ka[..3], kc[..3]);
        assert_ne!(ka[3], kc[3]);
    }
}
