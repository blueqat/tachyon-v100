//! KV キャッシュの頁と、GPU ⇄ RAM の段分け（kv-tier。docs/API.md 3.6・4.55）。
//!
//! - 頁 = 64 トークン × 普通の Attention の全層（Qwen3.8-27B は 16 層 × KV 4 ヘッド × 256）。
//!   GPU の置き場は頁の「枠」を並べた 1 本の塊。枠 s・層 l の層頁は `base + s*block_bytes + l*layer_page_bytes`。
//!   層頁の中は `[K: H×T×D][V: H×T×D]`（int8 なら後ろに係数 f16 `[2×H×T]`。行 = 1 トークン 1 ヘッド）。
//!   1 頁の全層が続いているので、RAM との出し入れは 1 頁 = 1 本の転送（fp16 4MB・int8 2MB）で PCIe を使い切れる。
//! - RAM 側は固定した（pinned）RAM の塊。GPU と同じ NUMA（GPU4〜7 は node1）に mbind で置き、cudaHostRegister で固定する。
//! - `PageManager` は帳簿だけ（CUDA を呼ばない。単体の試しはこちら）。頁の置き場・参照の数（文頭の共有）・
//!   書き足すときの写し（COW）・追い出しの順（優先度 → 最後に使った時 → 後ろの頁から）・転送の予定（Plan）を決める。
//! - `KvTier` が Plan を専用の 2 本の流れ（D2H・H2D）で走らせる。終わりは event で知る（`poll`）。
//!   計算の流れとは event でつなぐ（転送は「出した時までの計算」を待つ・計算は `stream_wait` で読み込みを待つ）。
//! - 線形 Attention の状態（1 人 約 151MB）のような決まった大きさの塊は `blob_*` で RAM に置ける（区切りの写し・休ませる会話）。
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ffi::c_void;

pub mod ffi {
    use std::ffi::{c_char, c_void};
    pub type Stream = *mut c_void;
    pub type Event = *mut c_void;
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct CopyDesc {
        pub src: *const u8,
        pub dst: *mut u8,
        pub bytes: u64,
    }
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct ConvDesc {
        pub src: *const u8,
        pub dst: *mut u8,
    }
    pub const H2D: i32 = 1;
    pub const D2H: i32 = 2;
    pub const D2D: i32 = 3;
    pub const NOT_READY: i32 = 600;
    extern "C" {
        pub fn cudaMalloc(p: *mut *mut c_void, size: usize) -> i32;
        pub fn cudaFree(p: *mut c_void) -> i32;
        pub fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, n: usize, kind: i32, s: Stream) -> i32;
        pub fn cudaMemsetAsync(p: *mut c_void, v: i32, n: usize, s: Stream) -> i32;
        pub fn cudaStreamCreateWithFlags(s: *mut Stream, flags: u32) -> i32;
        pub fn cudaStreamCreateWithPriority(s: *mut Stream, flags: u32, prio: i32) -> i32;
        pub fn cudaDeviceGetStreamPriorityRange(least: *mut i32, greatest: *mut i32) -> i32;
        pub fn cudaStreamDestroy(s: Stream) -> i32;
        pub fn cudaStreamSynchronize(s: Stream) -> i32;
        pub fn cudaStreamWaitEvent(s: Stream, e: Event, flags: u32) -> i32;
        pub fn cudaEventCreateWithFlags(e: *mut Event, flags: u32) -> i32;
        pub fn cudaEventDestroy(e: Event) -> i32;
        pub fn cudaEventRecord(e: Event, s: Stream) -> i32;
        pub fn cudaEventQuery(e: Event) -> i32;
        pub fn cudaEventSynchronize(e: Event) -> i32;
        pub fn cudaEventElapsedTime(ms: *mut f32, a: Event, b: Event) -> i32;
        pub fn cudaHostRegister(p: *mut c_void, n: usize, flags: u32) -> i32;
        pub fn cudaHostUnregister(p: *mut c_void) -> i32;
        pub fn cudaHostGetDevicePointer(d: *mut *mut c_void, h: *mut c_void, flags: u32) -> i32;
        pub fn cudaHostAlloc(p: *mut *mut c_void, n: usize, flags: u32) -> i32;
        pub fn cudaFreeHost(p: *mut c_void) -> i32;
        pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
        pub fn cudaGetDevice(d: *mut i32) -> i32;
        pub fn cudaDeviceGetPCIBusId(buf: *mut c_char, len: i32, dev: i32) -> i32;
        pub fn cudaDeviceSynchronize() -> i32;

        /// blocks = 1 回の起動のブロック数の上限（0 = 仕事の数だけ。zero-copy は少なくても PCIe を埋められる）
        pub fn kv_copy(d: *const CopyDesc, n: i32, blocks: i32, s: Stream) -> i32;
        pub fn kv_quant(d: *const ConvDesc, n: i32, rows: i32, dim: i32, blocks: i32, s: Stream) -> i32;
        pub fn kv_dequant(d: *const ConvDesc, n: i32, rows: i32, dim: i32, blocks: i32, s: Stream) -> i32;
        pub fn kv_fill(p: *const *mut u8, n: i32, bytes: u64, seed: u64, mode: i32, s: Stream) -> i32;
        pub fn kv_hash(p: *const *mut u8, n: i32, bytes: u64, out: *mut u64, s: Stream) -> i32;
        pub fn kv_spin(cycles: i64, s: Stream) -> i32;

        // libc（NUMA の置き場）
        pub fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> i32;
        pub fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
        pub fn syscall(n: i64, ...) -> i64;
    }
}

pub fn ck(code: i32, what: &str) {
    if code != 0 {
        crate::cuda::check(code, what);
    }
}

// ───────────────────────── 形 ─────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvDtype {
    F16,
    /// int8 ＋ 1 行（1 トークン 1 ヘッド）ごとの係数 f16
    Int8,
}

#[derive(Clone, Copy, Debug)]
pub struct KvShape {
    pub layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub page_tokens: usize,
    pub dtype: KvDtype,
}

impl KvShape {
    /// Qwen3.8-27B: 普通の Attention 16 層・KV 4 ヘッド・256
    pub fn qwen38_27b(dtype: KvDtype) -> Self {
        KvShape { layers: 16, kv_heads: 4, head_dim: 256, page_tokens: 64, dtype }
    }
    pub fn elem_bytes(&self) -> usize {
        match self.dtype {
            KvDtype::F16 => 2,
            KvDtype::Int8 => 1,
        }
    }
    /// 1 層頁の行の数（K と V の 1 トークン 1 ヘッド）
    pub fn rows(&self) -> usize {
        2 * self.kv_heads * self.page_tokens
    }
    /// 層頁の K（または V）のデータ部
    pub fn kv_part_bytes(&self) -> usize {
        self.kv_heads * self.page_tokens * self.head_dim * self.elem_bytes()
    }
    pub fn scale_bytes(&self) -> usize {
        match self.dtype {
            KvDtype::F16 => 0,
            KvDtype::Int8 => self.rows() * 2,
        }
    }
    /// 1 層頁（256B 揃え）
    pub fn layer_page_bytes(&self) -> usize {
        (2 * self.kv_part_bytes() + self.scale_bytes()).div_ceil(256) * 256
    }
    /// 1 頁（全層）= 転送の単位
    pub fn block_bytes(&self) -> usize {
        self.layers * self.layer_page_bytes()
    }
    pub fn bytes_per_token(&self) -> f64 {
        self.block_bytes() as f64 / self.page_tokens as f64
    }
    pub fn pages_for(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.page_tokens)
    }
    /// 頁の中の場所: (層, K=0/V=1, ヘッド, トークン) の行の頭
    pub fn row_off(&self, layer: usize, kv: usize, head: usize, tok: usize) -> usize {
        layer * self.layer_page_bytes() + kv * self.kv_part_bytes() + (head * self.page_tokens + tok) * self.head_dim * self.elem_bytes()
    }
    /// int8 の係数（f16）の場所
    pub fn scale_off(&self, layer: usize, kv: usize, head: usize, tok: usize) -> usize {
        layer * self.layer_page_bytes() + 2 * self.kv_part_bytes() + ((kv * self.kv_heads + head) * self.page_tokens + tok) * 2
    }
}

// ───────────────────────── 帳簿 ─────────────────────────

pub type PageId = u32;
pub type SeqId = u32;
pub type XferId = u64;
pub type BlobId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dir {
    D2H = 0,
    H2D = 1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvError {
    NoGpuPages { need: usize, can: usize },
    NoHostPages { need: usize, can: usize },
    NotResident(SeqId),
    NoSeq(SeqId),
    TooLong,
}
impl std::fmt::Display for KvError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for KvError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuLoc {
    /// 頁の枠
    Slot(u32),
    /// 置き場の外の GPU の番地（blob 用）
    Ptr(usize),
}

/// 1 本の転送の部品（RAM の枠 1 つ分まで）
#[derive(Clone, Copy, Debug)]
pub struct Seg {
    pub gpu: GpuLoc,
    pub host: u32,
    pub bytes: usize,
}

/// 1 回の転送（1 つの event で終わりを知る）
#[derive(Clone, Debug)]
pub struct Plan {
    pub id: XferId,
    pub dir: Dir,
    pub segs: Vec<Seg>,
    /// 先に終わっていなければならない転送（同じ枠を使い回すとき）
    pub deps: Vec<XferId>,
}

/// 帳簿が出す仕事: 走らせる転送と、もう走っていて待つべき転送
#[derive(Clone, Debug, Default)]
pub struct Job {
    pub plans: Vec<Plan>,
    pub waits: Vec<XferId>,
}
impl Job {
    fn add(&mut self, o: Job) {
        self.plans.extend(o.plans);
        self.waits.extend(o.waits);
    }
}

/// 書き足しの結果: 写す頁（COW、GPU の中で）と、計算の流れが先に待つ転送
#[derive(Clone, Debug, Default)]
pub struct AppendOut {
    pub cow: Vec<(u32, u32)>,
    pub deps: Vec<XferId>,
    pub new_pages: usize,
}

#[derive(Clone, Debug, Default)]
struct Page {
    live: bool,
    /// 頁の番号を使い回した回数（終わった転送が別の頁を触らないように）
    inc: u32,
    refs: u32,
    filled: u32,
    gen: u32,
    gpu: Option<u32>,
    gpu_ok: bool,
    host: Option<u32>,
    /// RAM の写しが持っている gen（None = 無い・写し中）
    host_gen: Option<u32>,
    xfer: Option<XferId>,
}

#[derive(Clone, Debug)]
struct Seq {
    pages: Vec<PageId>,
    tokens: usize,
    prio: u8,
    last: u64,
    pins: u32,
}

#[derive(Clone, Debug)]
struct Xfer {
    dir: Dir,
    /// (頁, 出した時の gen, 写し先の枠, 頁の inc)
    pages: Vec<(PageId, u32, u32, u32)>,
    /// D2H が終われば空く GPU の枠（追い出し）
    gpu_release: Vec<u32>,
    blob: Option<BlobId>,
}

#[derive(Clone, Debug)]
struct Blob {
    slots: Vec<u32>,
    bytes: usize,
    xfer: Option<XferId>,
    dead: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub d2h_pages: u64,
    pub h2d_pages: u64,
    /// RAM に同じ写しがあったので転送なしで追い出した頁
    pub clean_evicts: u64,
    pub cow: u64,
    /// RAM が足りず、GPU にもある写しを捨てた数
    pub host_drops: u64,
}

pub struct PageManager {
    pub shape: KvShape,
    /// RAM の枠 1 つの大きさ（RAM 側だけ int8 に詰めるときは GPU の頁より小さい）
    pub host_bb: usize,
    pub n_gpu: u32,
    pub n_host: u32,
    /// 1 回の転送に入れる頁の数（大きいほど event が減る・小さいほど追い出しと読み込みが重なる）
    pub chunk_pages: usize,
    pages: Vec<Page>,
    free_pages: Vec<PageId>,
    seqs: HashMap<SeqId, Seq>,
    next_seq: SeqId,
    gpu_free: BTreeSet<u32>,
    /// 追い出し中の枠（D2H が終われば空く。先に使うなら、その転送を待つ）
    gpu_pending: BTreeMap<u32, XferId>,
    host_free: BTreeSet<u32>,
    xfers: HashMap<XferId, Xfer>,
    next_xfer: XferId,
    blobs: HashMap<BlobId, Blob>,
    next_blob: BlobId,
    clock: u64,
    pub stats: Stats,
}

/// 追い出しの順を決める鍵（頁ごと。持っている列の中で一番強いもの）
#[derive(Clone, Copy, Default)]
struct Key {
    used: bool,
    prio: u8,
    last: u64,
    pos: usize,
    pinned: bool,
}

impl PageManager {
    pub fn new(shape: KvShape, n_gpu: u32, n_host: u32) -> Self {
        PageManager {
            shape,
            host_bb: shape.block_bytes(),
            n_gpu,
            n_host,
            chunk_pages: 8,
            pages: vec![],
            free_pages: vec![],
            seqs: HashMap::new(),
            next_seq: 1,
            gpu_free: (0..n_gpu).collect(),
            gpu_pending: BTreeMap::new(),
            host_free: (0..n_host).collect(),
            xfers: HashMap::new(),
            next_xfer: 1,
            blobs: HashMap::new(),
            next_blob: 1,
            clock: 0,
            stats: Stats::default(),
        }
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }
    fn seq(&self, s: SeqId) -> Result<&Seq, KvError> {
        self.seqs.get(&s).ok_or(KvError::NoSeq(s))
    }
    fn clean(&self, p: PageId) -> bool {
        let pg = &self.pages[p as usize];
        pg.host.is_some() && pg.host_gen == Some(pg.gen)
    }
    fn new_page(&mut self) -> PageId {
        let p = self.free_pages.pop().unwrap_or_else(|| {
            self.pages.push(Page::default());
            (self.pages.len() - 1) as PageId
        });
        let inc = self.pages[p as usize].inc.wrapping_add(1);
        self.pages[p as usize] = Page { live: true, inc, refs: 1, gpu_ok: true, ..Default::default() };
        p
    }
    fn release(&mut self, p: PageId) {
        let pg = std::mem::take(&mut self.pages[p as usize]);
        self.pages[p as usize].inc = pg.inc;
        debug_assert!(pg.live && pg.refs == 0 && pg.xfer.is_none());
        if let Some(g) = pg.gpu {
            self.gpu_free.insert(g);
        }
        if let Some(h) = pg.host {
            self.host_free.insert(h);
        }
        self.free_pages.push(p);
    }
    fn decref(&mut self, p: PageId) {
        let pg = &mut self.pages[p as usize];
        pg.refs -= 1;
        if pg.refs == 0 && pg.xfer.is_none() {
            self.release(p);
        }
    }
    fn alloc_gpu(&mut self) -> Option<(u32, Option<XferId>)> {
        if let Some(g) = self.gpu_free.pop_first() {
            return Some((g, None));
        }
        self.gpu_pending.pop_first().map(|(g, x)| (g, Some(x)))
    }
    fn new_xfer(&mut self, dir: Dir) -> XferId {
        let id = self.next_xfer;
        self.next_xfer += 1;
        self.xfers.insert(id, Xfer { dir, pages: vec![], gpu_release: vec![], blob: None });
        id
    }

    // ── 列 ──

    pub fn new_seq(&mut self) -> SeqId {
        let id = self.next_seq;
        self.next_seq += 1;
        let last = self.tick();
        self.seqs.insert(id, Seq { pages: vec![], tokens: 0, prio: 0, last, pins: 0 });
        id
    }
    pub fn seq_ids(&self) -> Vec<SeqId> {
        let mut v: Vec<SeqId> = self.seqs.keys().copied().collect();
        v.sort();
        v
    }
    pub fn tokens(&self, s: SeqId) -> Option<usize> {
        self.seqs.get(&s).map(|q| q.tokens)
    }
    pub fn pages_of(&self, s: SeqId) -> Option<Vec<PageId>> {
        self.seqs.get(&s).map(|q| q.pages.clone())
    }
    /// 使った印（追い出しの順は「優先度が低い → 長く使っていない」から）
    pub fn touch(&mut self, s: SeqId) {
        let t = self.tick();
        if let Some(q) = self.seqs.get_mut(&s) {
            q.last = t;
        }
    }
    /// 優先度（大きいほど残す）
    pub fn set_prio(&mut self, s: SeqId, prio: u8) {
        if let Some(q) = self.seqs.get_mut(&s) {
            q.prio = prio;
        }
    }
    /// 動いている間は追い出さない
    pub fn pin(&mut self, s: SeqId) {
        if let Some(q) = self.seqs.get_mut(&s) {
            q.pins += 1;
        }
    }
    pub fn unpin(&mut self, s: SeqId) {
        if let Some(q) = self.seqs.get_mut(&s) {
            q.pins = q.pins.saturating_sub(1);
        }
    }

    /// 文頭 n トークンを共有した新しい列（頁は参照を数えるだけ。書き足すときに写す）
    pub fn fork(&mut self, src: SeqId, n: usize) -> Result<SeqId, KvError> {
        let q = self.seq(src)?;
        if n > q.tokens {
            return Err(KvError::TooLong);
        }
        let pages: Vec<PageId> = q.pages[..self.shape.pages_for(n)].to_vec();
        let prio = q.prio;
        for &p in &pages {
            self.pages[p as usize].refs += 1;
        }
        let id = self.new_seq();
        let s = self.seqs.get_mut(&id).unwrap();
        s.pages = pages;
        s.tokens = n;
        s.prio = prio;
        Ok(id)
    }

    pub fn free_seq(&mut self, s: SeqId) {
        if let Some(q) = self.seqs.remove(&s) {
            for p in q.pages {
                self.decref(p);
            }
        }
    }

    /// 全頁が GPU にあり使えるなら、その枠の並び（Attention の頁表）
    pub fn table(&self, s: SeqId) -> Option<Vec<u32>> {
        let q = self.seqs.get(&s)?;
        q.pages
            .iter()
            .map(|&p| {
                let pg = &self.pages[p as usize];
                if pg.gpu_ok {
                    pg.gpu
                } else {
                    None
                }
            })
            .collect()
    }
    /// 全頁に GPU の枠が割り当て済みなら、その並び（読み込み中を含む。使う前に計算の流れで `stream_wait` する）
    pub fn slots(&self, s: SeqId) -> Option<Vec<u32>> {
        let q = self.seqs.get(&s)?;
        q.pages.iter().map(|&p| self.pages[p as usize].gpu).collect()
    }
    /// GPU に枠を持っていない頁の数
    pub fn missing(&self, s: SeqId) -> usize {
        self.seqs.get(&s).map_or(0, |q| q.pages.iter().filter(|&&p| self.pages[p as usize].gpu.is_none()).count())
    }
    pub fn gpu_free(&self) -> usize {
        self.gpu_free.len() + self.gpu_pending.len()
    }
    pub fn host_free(&self) -> usize {
        self.host_free.len()
    }
    pub fn host_slot(&self, p: PageId) -> Option<u32> {
        self.pages[p as usize].host
    }
    pub fn gpu_slot(&self, p: PageId) -> Option<u32> {
        self.pages[p as usize].gpu
    }
    pub fn refs(&self, p: PageId) -> u32 {
        self.pages[p as usize].refs
    }
    pub fn in_flight(&self) -> usize {
        self.xfers.len()
    }

    fn keys(&self, keep: Option<SeqId>) -> Vec<Key> {
        let mut k = vec![Key::default(); self.pages.len()];
        for (&sid, q) in &self.seqs {
            let pinned = q.pins > 0 || Some(sid) == keep;
            for (i, &p) in q.pages.iter().enumerate() {
                let e = &mut k[p as usize];
                if !e.used {
                    *e = Key { used: true, prio: q.prio, last: q.last, pos: i, pinned };
                } else {
                    e.prio = e.prio.max(q.prio);
                    e.last = e.last.max(q.last);
                    e.pos = e.pos.max(i);
                    e.pinned |= pinned;
                }
            }
        }
        k
    }

    /// 追い出してよい頁を、追い出す順に
    fn victims(&self, keep: Option<SeqId>) -> Vec<PageId> {
        let k = self.keys(keep);
        let mut v: Vec<PageId> = (0..self.pages.len() as PageId)
            .filter(|&p| {
                let pg = &self.pages[p as usize];
                let e = &k[p as usize];
                pg.live && e.used && !e.pinned && pg.gpu.is_some() && pg.gpu_ok && pg.xfer.is_none()
            })
            .collect();
        // 優先度が低い → 長く使っていない → RAM に同じ写しがある（転送なし）→ 後ろの頁（文頭は共有されやすい）
        v.sort_by_key(|&p| {
            let e = &k[p as usize];
            (e.prio, e.last, !self.clean(p), Reverse(e.pos), p)
        });
        v
    }

    /// RAM の枠を n 個空ける。足りなければ、GPU にもある写し（捨てても困らない）を新しく使った物から捨てる
    fn ensure_host(&mut self, n: usize, protect: &[PageId]) -> Result<(), KvError> {
        if self.host_free.len() >= n {
            return Ok(());
        }
        let k = self.keys(None);
        let prot: BTreeSet<PageId> = protect.iter().copied().collect();
        let mut cand: Vec<PageId> = (0..self.pages.len() as PageId)
            .filter(|&p| {
                let pg = &self.pages[p as usize];
                pg.live && pg.host.is_some() && pg.gpu.is_some() && pg.gpu_ok && pg.xfer.is_none() && !prot.contains(&p)
            })
            .collect();
        let can = self.host_free.len() + cand.len();
        if can < n {
            return Err(KvError::NoHostPages { need: n, can });
        }
        cand.sort_by_key(|&p| Reverse((k[p as usize].prio, k[p as usize].last)));
        for p in cand {
            if self.host_free.len() >= n {
                break;
            }
            let pg = &mut self.pages[p as usize];
            let h = pg.host.take().unwrap();
            pg.host_gen = None;
            self.host_free.insert(h);
            self.stats.host_drops += 1;
        }
        Ok(())
    }

    /// 頁を RAM へ追い出す（GPU の枠は D2H が終わったら空く。同じ写しが RAM にあればすぐ空く）
    fn evict_pages(&mut self, ps: &[PageId]) -> Result<Job, KvError> {
        // 呼ぶ側（victims・evict_seq）で「GPU にあり使える・転送中でない」頁だけに絞っている
        debug_assert!(ps.iter().all(|&p| {
            let pg = &self.pages[p as usize];
            pg.live && pg.gpu.is_some() && pg.gpu_ok && pg.xfer.is_none()
        }));
        let dirty: Vec<PageId> = ps.iter().copied().filter(|&p| !self.clean(p)).collect();
        let need = dirty.iter().filter(|&&p| self.pages[p as usize].host.is_none()).count();
        self.ensure_host(need, ps)?;
        for &p in ps {
            if self.clean(p) {
                let pg = &mut self.pages[p as usize];
                let g = pg.gpu.take().unwrap();
                pg.gpu_ok = false;
                self.gpu_free.insert(g);
                self.stats.clean_evicts += 1;
            }
        }
        let mut dirty = dirty;
        dirty.sort_by_key(|&p| self.pages[p as usize].gpu);
        let mut job = Job::default();
        for ch in dirty.chunks(self.chunk_pages.max(1)) {
            let id = self.new_xfer(Dir::D2H);
            let mut segs = vec![];
            for &p in ch {
                let h = match self.pages[p as usize].host {
                    Some(h) => h,
                    None => self.host_free.pop_first().unwrap(),
                };
                let pg = &mut self.pages[p as usize];
                let g = pg.gpu.take().unwrap();
                pg.gpu_ok = false;
                pg.host = Some(h);
                pg.host_gen = None;
                pg.xfer = Some(id);
                let (gen, inc) = (pg.gen, pg.inc);
                self.gpu_pending.insert(g, id);
                let x = self.xfers.get_mut(&id).unwrap();
                x.pages.push((p, gen, h, inc));
                x.gpu_release.push(g);
                segs.push(Seg { gpu: GpuLoc::Slot(g), host: h, bytes: self.shape.block_bytes() });
            }
            self.stats.d2h_pages += ch.len() as u64;
            job.plans.push(Plan { id, dir: Dir::D2H, segs, deps: vec![] });
        }
        Ok(job)
    }

    /// GPU の枠を need 個使えるようにする（足りなければ追い出す。keep の列には触らない）
    pub fn make_room(&mut self, need: usize, keep: Option<SeqId>) -> Result<Job, KvError> {
        let have = self.gpu_free();
        if have >= need {
            return Ok(Job::default());
        }
        let v = self.victims(keep);
        if have + v.len() < need {
            return Err(KvError::NoGpuPages { need, can: have + v.len() });
        }
        self.evict_pages(&v[..need - have])
    }

    /// 列を丸ごと RAM へ（休ませる会話）。共有の頁も、他に動いている列が無ければ出す
    pub fn evict_seq(&mut self, s: SeqId) -> Result<Job, KvError> {
        let k = self.keys(None);
        let ps: Vec<PageId> = self
            .seq(s)?
            .pages
            .iter()
            .copied()
            .filter(|&p| {
                let pg = &self.pages[p as usize];
                // 他の列が動いている（pin）共有の頁は残す
                pg.gpu.is_some() && pg.gpu_ok && pg.xfer.is_none() && !k[p as usize].pinned
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        self.evict_pages(&ps)
    }

    /// 列の頁を全部 GPU へ（足りなければ他の列を追い出す）。終われば `table` が使える
    pub fn prefetch(&mut self, s: SeqId) -> Result<Job, KvError> {
        let q = self.seq(s)?;
        let mut miss: Vec<PageId> = vec![];
        let mut job = Job::default();
        let mut seen = BTreeSet::new();
        for &p in &q.pages {
            if !seen.insert(p) {
                continue;
            }
            let pg = &self.pages[p as usize];
            match (pg.gpu, pg.gpu_ok) {
                (Some(_), true) => {}
                (Some(_), false) => job.waits.extend(pg.xfer),
                (None, _) => miss.push(p),
            }
        }
        if miss.is_empty() {
            return Ok(job);
        }
        job.add(self.make_room(miss.len(), Some(s))?);
        miss.sort_by_key(|&p| self.pages[p as usize].host);
        for ch in miss.chunks(self.chunk_pages.max(1)) {
            let id = self.new_xfer(Dir::H2D);
            let mut segs = vec![];
            let mut deps = BTreeSet::new();
            for &p in ch {
                let (g, dep) = self.alloc_gpu().expect("make_room の後なのに枠が無い");
                deps.extend(dep);
                let pg = &mut self.pages[p as usize];
                let h = pg.host.expect("GPU にも RAM にも無い頁");
                // RAM へ写している途中なら、それを待ってから読む
                deps.extend(pg.xfer);
                pg.gpu = Some(g);
                pg.gpu_ok = false;
                pg.xfer = Some(id);
                let inc = pg.inc;
                self.xfers.get_mut(&id).unwrap().pages.push((p, 0, g, inc));
                segs.push(Seg { gpu: GpuLoc::Slot(g), host: h, bytes: self.shape.block_bytes() });
            }
            self.stats.h2d_pages += ch.len() as u64;
            job.plans.push(Plan { id, dir: Dir::H2D, segs, deps: deps.into_iter().collect() });
        }
        Ok(job)
    }

    /// GPU に置いたまま RAM にも写しを取る（手すきのうちに。後の追い出しが転送なしになる）
    pub fn backup(&mut self, s: SeqId) -> Result<Job, KvError> {
        let ps: Vec<PageId> = self
            .seq(s)?
            .pages
            .iter()
            .copied()
            .filter(|&p| {
                let pg = &self.pages[p as usize];
                pg.gpu.is_some() && pg.gpu_ok && pg.xfer.is_none() && !self.clean(p)
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let need = ps.iter().filter(|&&p| self.pages[p as usize].host.is_none()).count();
        self.ensure_host(need, &ps)?;
        let mut ps = ps;
        ps.sort_by_key(|&p| self.pages[p as usize].gpu);
        let mut job = Job::default();
        for ch in ps.chunks(self.chunk_pages.max(1)) {
            let id = self.new_xfer(Dir::D2H);
            let mut segs = vec![];
            for &p in ch {
                let h = match self.pages[p as usize].host {
                    Some(h) => h,
                    None => self.host_free.pop_first().unwrap(),
                };
                let pg = &mut self.pages[p as usize];
                pg.host = Some(h);
                pg.host_gen = None;
                pg.xfer = Some(id);
                let (g, gen, inc) = (pg.gpu.unwrap(), pg.gen, pg.inc);
                self.xfers.get_mut(&id).unwrap().pages.push((p, gen, h, inc));
                segs.push(Seg { gpu: GpuLoc::Slot(g), host: h, bytes: self.shape.block_bytes() });
            }
            self.stats.d2h_pages += ch.len() as u64;
            job.plans.push(Plan { id, dir: Dir::D2H, segs, deps: vec![] });
        }
        Ok(job)
    }

    /// n トークン書き足すのに要る新しい GPU の枠（COW を含む）
    pub fn new_pages_for_append(&self, s: SeqId, n: usize) -> Result<usize, KvError> {
        let q = self.seq(s)?;
        let t = self.shape.page_tokens;
        let grow = (q.tokens + n).div_ceil(t) - q.tokens.div_ceil(t);
        let cow = (q.tokens % t != 0 && n > 0 && self.pages[*q.pages.last().unwrap() as usize].refs > 1) as usize;
        Ok(grow + cow)
    }

    /// n トークン分の場所を取る（列は GPU に揃っていること）。共有の途中の頁に書くなら写す（COW）。
    /// 書き込み自体は計算の側（返した cow を先に GPU の中で写し、deps を待ってから）
    pub fn append(&mut self, s: SeqId, n: usize) -> Result<AppendOut, KvError> {
        if self.table(s).is_none() {
            return Err(KvError::NotResident(s));
        }
        let need = self.new_pages_for_append(s, n)?;
        if self.gpu_free() < need {
            return Err(KvError::NoGpuPages { need, can: self.gpu_free() });
        }
        let t = self.shape.page_tokens;
        let mut out = AppendOut { new_pages: need, ..Default::default() };
        let mut left = n;
        while left > 0 {
            let (tokens, idx_page) = {
                let q = &self.seqs[&s];
                (q.tokens, q.pages.get(q.tokens / t).copied())
            };
            let off = tokens % t;
            let take = (t - off).min(left);
            let p = if off == 0 {
                let p = self.new_page();
                let (g, dep) = self.alloc_gpu().unwrap();
                out.deps.extend(dep);
                self.pages[p as usize].gpu = Some(g);
                self.seqs.get_mut(&s).unwrap().pages.push(p);
                p
            } else {
                let old = idx_page.unwrap();
                if self.pages[old as usize].refs > 1 {
                    let p = self.new_page();
                    let (g, dep) = self.alloc_gpu().unwrap();
                    out.deps.extend(dep);
                    out.cow.push((self.pages[old as usize].gpu.unwrap(), g));
                    self.pages[p as usize].gpu = Some(g);
                    self.pages[p as usize].filled = off as u32;
                    self.seqs.get_mut(&s).unwrap().pages[tokens / t] = p;
                    self.decref(old);
                    self.stats.cow += 1;
                    p
                } else {
                    old
                }
            };
            let pg = &mut self.pages[p as usize];
            pg.filled = (off + take) as u32;
            pg.gen += 1;
            // RAM の写しは古くなった（転送中でなければ枠を返す）
            if pg.xfer.is_none() {
                if let Some(h) = pg.host.take() {
                    self.host_free.insert(h);
                }
                pg.host_gen = None;
            }
            self.seqs.get_mut(&s).unwrap().tokens += take;
            left -= take;
        }
        self.touch(s);
        Ok(out)
    }

    /// 転送が終わった（KvTier::poll から）
    pub fn complete(&mut self, id: XferId) {
        let Some(x) = self.xfers.remove(&id) else { return };
        for &(p, gen, slot, inc) in &x.pages {
            let pg = &mut self.pages[p as usize];
            if !pg.live || pg.inc != inc {
                continue;
            }
            match x.dir {
                Dir::D2H => {
                    if pg.host == Some(slot) {
                        pg.host_gen = Some(gen);
                    }
                }
                Dir::H2D => {
                    if pg.gpu == Some(slot) {
                        pg.gpu_ok = true;
                    }
                }
            }
            if pg.xfer == Some(id) {
                pg.xfer = None;
                if pg.refs == 0 {
                    self.release(p);
                } else if pg.host.is_some() && pg.host_gen != Some(pg.gen) && x.dir == Dir::D2H {
                    // 写している間に書き足された → この写しは古い
                    let h = pg.host.take().unwrap();
                    pg.host_gen = None;
                    self.host_free.insert(h);
                }
            }
        }
        for g in x.gpu_release {
            if self.gpu_pending.get(&g) == Some(&id) {
                self.gpu_pending.remove(&g);
                self.gpu_free.insert(g);
            }
        }
        if let Some(b) = x.blob {
            if let Some(bl) = self.blobs.get_mut(&b) {
                if bl.xfer == Some(id) {
                    bl.xfer = None;
                    if bl.dead {
                        let bl = self.blobs.remove(&b).unwrap();
                        self.host_free.extend(bl.slots);
                    }
                }
            }
        }
    }

    /// 帳簿だけの転送（測り用。走らせると complete で消える）
    pub fn raw_plan(&mut self, dir: Dir, segs: Vec<Seg>) -> Plan {
        let id = self.new_xfer(dir);
        Plan { id, dir, segs, deps: vec![] }
    }

    // ── blob（線形 Attention の状態など、決まった大きさの塊を RAM に）──

    pub fn blob_new(&mut self, bytes: usize) -> Result<BlobId, KvError> {
        let n = bytes.div_ceil(self.host_bb);
        self.ensure_host(n, &[])?;
        let slots: Vec<u32> = (0..n).map(|_| self.host_free.pop_first().unwrap()).collect();
        let id = self.next_blob;
        self.next_blob += 1;
        self.blobs.insert(id, Blob { slots, bytes, xfer: None, dead: false });
        Ok(id)
    }
    fn blob_plan(&mut self, b: BlobId, dev: usize, dir: Dir) -> Plan {
        let bb = self.host_bb;
        let id = self.new_xfer(dir);
        let bl = self.blobs.get_mut(&b).expect("無い blob");
        let deps: Vec<XferId> = bl.xfer.into_iter().collect();
        bl.xfer = Some(id);
        let segs = bl
            .slots
            .iter()
            .enumerate()
            .map(|(i, &h)| Seg { gpu: GpuLoc::Ptr(dev + i * bb), host: h, bytes: bb.min(bl.bytes - i * bb) })
            .collect();
        self.xfers.get_mut(&id).unwrap().blob = Some(b);
        Plan { id, dir, segs, deps }
    }
    /// GPU の dev から blob へ写す
    pub fn blob_save(&mut self, b: BlobId, dev: usize) -> Job {
        Job { plans: vec![self.blob_plan(b, dev, Dir::D2H)], waits: vec![] }
    }
    /// blob を GPU の dev へ戻す
    pub fn blob_load(&mut self, b: BlobId, dev: usize) -> Job {
        Job { plans: vec![self.blob_plan(b, dev, Dir::H2D)], waits: vec![] }
    }
    pub fn blob_free(&mut self, b: BlobId) {
        if let Some(bl) = self.blobs.get_mut(&b) {
            if bl.xfer.is_some() {
                bl.dead = true;
            } else {
                let bl = self.blobs.remove(&b).unwrap();
                self.host_free.extend(bl.slots);
            }
        }
    }

    /// 帳簿のつじつま（試し用）: 枠は「空き・追い出し中・頁・blob」のちょうど 1 つ、参照の数は列の数と同じ
    pub fn check(&self) -> Result<(), String> {
        let mut refs = vec![0u32; self.pages.len()];
        for q in self.seqs.values() {
            if q.pages.len() != self.shape.pages_for(q.tokens) {
                return Err(format!("列の頁の数 {} とトークン {} が合わない", q.pages.len(), q.tokens));
            }
            for &p in &q.pages {
                refs[p as usize] += 1;
            }
        }
        let mut gpu_owner = vec![0u32; self.n_gpu as usize];
        let mut host_owner = vec![0u32; self.n_host as usize];
        for g in &self.gpu_free {
            gpu_owner[*g as usize] += 1;
        }
        for g in self.gpu_pending.keys() {
            gpu_owner[*g as usize] += 1;
        }
        for h in &self.host_free {
            host_owner[*h as usize] += 1;
        }
        for (i, pg) in self.pages.iter().enumerate() {
            if !pg.live {
                if pg.gpu.is_some() || pg.host.is_some() {
                    return Err(format!("死んだ頁 {i} が枠を持つ"));
                }
                continue;
            }
            if pg.refs != refs[i] {
                return Err(format!("頁 {i} の参照 {} ≠ 列の数 {}", pg.refs, refs[i]));
            }
            if pg.refs == 0 && pg.xfer.is_none() {
                return Err(format!("頁 {i} は誰も持たないのに残っている"));
            }
            if pg.gpu.is_none() && pg.host.is_none() && pg.refs > 0 {
                return Err(format!("頁 {i} がどこにも無い"));
            }
            if let Some(g) = pg.gpu {
                gpu_owner[g as usize] += 1;
            }
            if let Some(h) = pg.host {
                host_owner[h as usize] += 1;
            }
        }
        for bl in self.blobs.values() {
            for &h in &bl.slots {
                host_owner[h as usize] += 1;
            }
        }
        if let Some(g) = gpu_owner.iter().position(|&c| c != 1) {
            return Err(format!("GPU の枠 {g} の持ち主が {} 人", gpu_owner[g]));
        }
        if let Some(h) = host_owner.iter().position(|&c| c != 1) {
            return Err(format!("RAM の枠 {h} の持ち主が {} 人", host_owner[h]));
        }
        Ok(())
    }
}

// ───────────────────────── 固定した RAM（NUMA） ─────────────────────────

/// GPU が載っている NUMA の番号（/sys/bus/pci/devices/<bus>/numa_node）
pub fn gpu_numa_node(dev: i32) -> Option<i32> {
    let mut buf = [0 as std::ffi::c_char; 32];
    if unsafe { ffi::cudaDeviceGetPCIBusId(buf.as_mut_ptr(), 32, dev) } != 0 {
        return None;
    }
    let id = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().to_lowercase();
    let s = std::fs::read_to_string(format!("/sys/bus/pci/devices/{id}/numa_node")).ok()?;
    s.trim().parse().ok().filter(|&n: &i32| n >= 0)
}

pub fn current_device() -> i32 {
    let mut d = 0;
    unsafe { ffi::cudaGetDevice(&mut d) };
    d
}

/// 固定した RAM の塊。node を指せば mmap → mbind(MPOL_PREFERRED) → cudaHostRegister（その node の RAM から取る）。
/// MPOL_BIND にしないのは、その node が埋まったときに OOM で他人のプロセスが落ちないように（足りなければ隣の node から取る）
pub struct HostArena {
    pub ptr: *mut u8,
    /// GPU から見た番地（zero-copy のカーネル用。UVA なら ptr と同じ）
    pub dev: *mut u8,
    pub bytes: usize,
    pub node: Option<i32>,
    mapped: bool,
}
unsafe impl Send for HostArena {}

const SYS_MBIND: i64 = 237;
const SYS_GET_MEMPOLICY: i64 = 239;

impl HostArena {
    pub fn new(bytes: usize, node: Option<i32>) -> Self {
        let bytes = bytes.max(4096).div_ceil(2 << 20) * (2 << 20);
        unsafe {
            // PROT_READ|WRITE, MAP_PRIVATE|ANONYMOUS|NORESERVE
            let p = ffi::mmap(std::ptr::null_mut(), bytes, 3, 0x02 | 0x20 | 0x4000, -1, 0);
            assert!(p as isize != -1, "mmap {bytes} に失敗");
            ffi::madvise(p, bytes, 14); // MADV_HUGEPAGE（固定するのも TLB も軽くなる）
            if let Some(n) = node.filter(|n| (0..64).contains(n)) {
                let mask: u64 = 1 << n;
                // MPOL_PREFERRED = 1、flags 0
                let r = ffi::syscall(SYS_MBIND, p, bytes, 1i64, &mask as *const u64, 64u64, 0u64);
                if r != 0 {
                    eprintln!("mbind が効かない（{r}）。node の指定なしで続ける");
                }
            }
            // 先に何本かのスレッドで触っておく（ページを取る・0 で埋めるのが固定の時間の大半。1 本だと 1.3GB/s）
            prefault(p.cast(), bytes);
            // cudaHostRegisterPortable | Mapped
            ck(ffi::cudaHostRegister(p, bytes, 1 | 2), "cudaHostRegister");
            let mut d = std::ptr::null_mut();
            ck(ffi::cudaHostGetDevicePointer(&mut d, p, 0), "cudaHostGetDevicePointer");
            HostArena { ptr: p.cast(), dev: d.cast(), bytes, node, mapped: true }
        }
    }
    /// cudaHostAlloc（NUMA を指さない。比べ用）
    pub fn cuda_alloc(bytes: usize) -> Self {
        let mut p = std::ptr::null_mut();
        ck(unsafe { ffi::cudaHostAlloc(&mut p, bytes, 1 | 2) }, "cudaHostAlloc");
        let mut d = std::ptr::null_mut();
        ck(unsafe { ffi::cudaHostGetDevicePointer(&mut d, p, 0) }, "cudaHostGetDevicePointer");
        HostArena { ptr: p.cast(), dev: d.cast(), bytes, node: None, mapped: false }
    }
    /// 実際にどの node に置かれたか（samples か所を get_mempolicy で）→ node ごとの数
    pub fn node_histogram(&self, samples: usize) -> BTreeMap<i32, usize> {
        let mut h = BTreeMap::new();
        let step = (self.bytes / samples.max(1)).max(4096) & !4095;
        let mut off = 0;
        while off < self.bytes {
            let mut node: i32 = -1;
            // MPOL_F_NODE | MPOL_F_ADDR
            let r = unsafe {
                ffi::syscall(SYS_GET_MEMPOLICY, &mut node as *mut i32, std::ptr::null_mut::<u64>(), 0u64, self.ptr.add(off), 3u64)
            };
            *h.entry(if r == 0 { node } else { -1 }).or_insert(0) += 1;
            off += step;
        }
        h
    }
}
/// 2MB ごとに 1 バイト書いて、ページを先に取る（PREFAULT_THREADS 本で分ける）
fn prefault(p: *mut u8, bytes: usize) {
    let th = std::env::var("KV_PREFAULT_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(PREFAULT_THREADS).max(1);
    let per = bytes.div_ceil(th).div_ceil(2 << 20) * (2 << 20);
    let base = p as usize;
    std::thread::scope(|sc| {
        for i in 0..th {
            let (a, b) = (i * per, ((i + 1) * per).min(bytes));
            sc.spawn(move || {
                let mut o = a;
                while o < b {
                    unsafe { std::ptr::write_volatile((base + o) as *mut u8, 0) };
                    o += 4096;
                }
            });
        }
    });
}
/// 固定する前にページを取るスレッドの数（作った機械の 1 つの NUMA は 20 スレッド）
pub const PREFAULT_THREADS: usize = 8;

impl Drop for HostArena {
    fn drop(&mut self) {
        unsafe {
            if self.mapped {
                ffi::cudaHostUnregister(self.ptr.cast());
                ffi::munmap(self.ptr.cast(), self.bytes);
            } else {
                ffi::cudaFreeHost(self.ptr.cast());
            }
        }
    }
}

// ───────────────────────── 転送を走らせる側 ─────────────────────────

/// zero-copy のカーネルの既定のブロック数（kvbench pcie --blocks で測って決めた）
pub const KERNEL_BLOCKS: i32 = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyMode {
    /// 隣り合う頁をまとめて cudaMemcpyAsync（コピーエンジン。SM を使わない）
    Memcpy,
    /// kv_copy カーネルで GPU から RAM を直接読み書き（zero-copy。ばらばらの小さな頁を 1 回の起動で）
    Kernel,
}

/// 転送の待ち札: 向きごとに「この番号までの転送」
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ticket {
    pub upto: [XferId; 2],
}
impl Ticket {
    pub fn merge(self, o: Ticket) -> Ticket {
        Ticket { upto: [self.upto[0].max(o.upto[0]), self.upto[1].max(o.upto[1])] }
    }
}

pub struct TierConfig {
    pub shape: KvShape,
    pub gpu_pages: u32,
    pub host_pages: u32,
    /// None = GPU と同じ node。Some(-1) = 指さない（cudaHostAlloc）
    pub host_node: Option<i32>,
    pub mode: CopyMode,
    /// RAM 側の形式。None = GPU と同じ。GPU が F16 で Some(Int8) なら、追い出すときに int8 に詰め、戻すときに fp16 に戻す
    /// （RAM も PCIe も半分。値は int8 の丸めが入る＝戻した KV は int8 KV と同じ質）
    pub host_dtype: Option<KvDtype>,
}

pub struct KvTier {
    pub mgr: PageManager,
    pub shape: KvShape,
    /// RAM 側の形
    pub host_shape: KvShape,
    pub gpu_base: *mut u8,
    pub host: HostArena,
    pub mode: CopyMode,
    /// zero-copy のカーネル（Kernel のときと、RAM 側 int8 を zero-copy で詰めるとき）のブロック数の上限
    pub kernel_blocks: i32,
    /// RAM 側 int8 のとき、詰める・戻すのを zero-copy のカーネルでやる（false = GPU の置き場で詰めてからコピーエンジンで運ぶ）。
    /// ⚠️ zero-copy は同時に走る計算（HBM を読む仕事）を 5 倍ほど遅くする（kvbench interfere）。既定は false
    pub conv_zero_copy: bool,
    /// RAM 側 int8 のときの GPU の置き場（向きごと。stage_pages 頁分）
    stage: [*mut u8; 2],
    stage_pages: usize,
    streams: [ffi::Stream; 2],
    /// 計算の流れ（既定は null = 古い既定の流れ）。転送はこの流れの「出した時まで」を待つ
    pub compute: ffi::Stream,
    compute_ev: ffi::Event,
    inflight: [VecDeque<(XferId, ffi::Event)>; 2],
    ev_pool: Vec<ffi::Event>,
    own_gpu: bool,
}

impl KvTier {
    pub fn new(cfg: TierConfig) -> Self {
        let mut g = std::ptr::null_mut();
        ck(unsafe { ffi::cudaMalloc(&mut g, cfg.shape.block_bytes() * cfg.gpu_pages as usize) }, "cudaMalloc（KV の置き場）");
        let hdt = cfg.host_dtype.unwrap_or(cfg.shape.dtype);
        let bb = KvShape { dtype: hdt, ..cfg.shape }.block_bytes();
        let node = match cfg.host_node {
            None => gpu_numa_node(current_device()),
            Some(n) => Some(n),
        };
        let host = match node {
            Some(n) if n >= 0 => HostArena::new(bb * cfg.host_pages as usize, Some(n)),
            _ => HostArena::cuda_alloc(bb * cfg.host_pages as usize),
        };
        Self::with_buffers(cfg.shape, g.cast(), cfg.gpu_pages, host, cfg.mode, hdt, true)
    }

    /// GPU の置き場をよそ（エンジンのメモリの割り振り）から渡す
    pub fn with_buffers(shape: KvShape, gpu_base: *mut u8, gpu_pages: u32, host: HostArena, mode: CopyMode, host_dtype: KvDtype, own_gpu: bool) -> Self {
        let host_shape = KvShape { dtype: host_dtype, ..shape };
        assert!(
            host_dtype == shape.dtype || (shape.dtype == KvDtype::F16 && host_dtype == KvDtype::Int8 && shape.head_dim % 256 == 0),
            "RAM 側の形式は GPU と同じか、GPU fp16 → RAM int8（head_dim は 256 の倍数）だけ"
        );
        let host_pages = (host.bytes / host_shape.block_bytes()) as u32;
        // 転送の流れは優先度を上げる: Memcpy ではこの流れにカーネルは無く、RAM 側 int8 の詰め・戻しは短い（32MB で約 50µs）ので、
        // 計算に SM を取られて PCIe が止まらないように。Kernel（zero-copy で写す）を使うなら計算を押しのけるので注意
        let (mut least, mut greatest) = (0, 0);
        unsafe { ffi::cudaDeviceGetStreamPriorityRange(&mut least, &mut greatest) };
        let mut streams = [std::ptr::null_mut(); 2];
        for s in &mut streams {
            ck(unsafe { ffi::cudaStreamCreateWithPriority(s, 1, greatest) }, "cudaStreamCreateWithPriority"); // NonBlocking
        }
        let mut ce = std::ptr::null_mut();
        ck(unsafe { ffi::cudaEventCreateWithFlags(&mut ce, 2) }, "cudaEventCreate");
        let mut mgr = PageManager::new(shape, gpu_pages, host_pages);
        mgr.host_bb = host_shape.block_bytes();
        let stage_pages = 16;
        let mut stage = [std::ptr::null_mut(); 2];
        if host_dtype != shape.dtype {
            for p in &mut stage {
                let mut d = std::ptr::null_mut();
                ck(unsafe { ffi::cudaMalloc(&mut d, stage_pages * host_shape.block_bytes()) }, "cudaMalloc（詰める置き場）");
                *p = d.cast();
            }
        }
        KvTier {
            mgr,
            shape,
            host_shape,
            gpu_base,
            host,
            mode,
            kernel_blocks: KERNEL_BLOCKS,
            conv_zero_copy: false,
            stage,
            stage_pages,
            streams,
            compute: std::ptr::null_mut(),
            compute_ev: ce,
            inflight: [VecDeque::new(), VecDeque::new()],
            ev_pool: vec![],
            own_gpu,
        }
    }

    pub fn slot_ptr(&self, slot: u32) -> *mut u8 {
        unsafe { self.gpu_base.add(slot as usize * self.shape.block_bytes()) }
    }
    /// 枠 slot・層 layer の層頁の頭
    pub fn layer_ptr(&self, slot: u32, layer: usize) -> *mut u8 {
        unsafe { self.slot_ptr(slot).add(layer * self.shape.layer_page_bytes()) }
    }
    pub fn host_ptr(&self, slot: u32) -> *mut u8 {
        unsafe { self.host.ptr.add(slot as usize * self.mgr.host_bb) }
    }
    fn host_dev_ptr(&self, slot: u32) -> *mut u8 {
        unsafe { self.host.dev.add(slot as usize * self.mgr.host_bb) }
    }
    pub fn stream(&self, d: Dir) -> ffi::Stream {
        self.streams[d as usize]
    }

    fn event(&mut self) -> ffi::Event {
        self.ev_pool.pop().unwrap_or_else(|| {
            let mut e = std::ptr::null_mut();
            ck(unsafe { ffi::cudaEventCreateWithFlags(&mut e, 2) }, "cudaEventCreate"); // DisableTiming
            e
        })
    }
    fn ev_of(&self, id: XferId) -> Option<ffi::Event> {
        for q in &self.inflight {
            if let Some(&(_, e)) = q.iter().find(|(i, _)| *i == id) {
                return Some(e);
            }
        }
        None
    }

    /// 転送を出す（CPU は待たない）
    pub fn submit(&mut self, job: Job) -> Ticket {
        let mut t = Ticket::default();
        for &w in &job.waits {
            if let Some(d) = self.inflight.iter().position(|q| q.iter().any(|(i, _)| *i == w)) {
                t.upto[d] = t.upto[d].max(w);
            }
        }
        if job.plans.is_empty() {
            return t;
        }
        ck(unsafe { ffi::cudaEventRecord(self.compute_ev, self.compute) }, "cudaEventRecord");
        for plan in job.plans {
            let d = plan.dir as usize;
            let st = self.streams[d];
            unsafe {
                ck(ffi::cudaStreamWaitEvent(st, self.compute_ev, 0), "cudaStreamWaitEvent");
                for &dep in &plan.deps {
                    if let Some(e) = self.ev_of(dep) {
                        ck(ffi::cudaStreamWaitEvent(st, e, 0), "cudaStreamWaitEvent");
                    }
                }
            }
            self.issue(&plan);
            let e = self.event();
            ck(unsafe { ffi::cudaEventRecord(e, st) }, "cudaEventRecord");
            self.inflight[d].push_back((plan.id, e));
            t.upto[d] = t.upto[d].max(plan.id);
        }
        t
    }

    fn gpu_addr(&self, g: GpuLoc) -> *mut u8 {
        match g {
            GpuLoc::Slot(s) => self.slot_ptr(s),
            GpuLoc::Ptr(p) => p as *mut u8,
        }
    }

    /// Plan の部品を (GPU, RAM, 長さ) にし、両側とも続いているものはつなぐ
    fn coalesce(&self, segs: &[Seg], host_dev: bool) -> Vec<(*mut u8, *mut u8, usize)> {
        let mut v: Vec<(*mut u8, *mut u8, usize)> = vec![];
        for s in segs {
            let g = self.gpu_addr(s.gpu);
            let h = if host_dev { self.host_dev_ptr(s.host) } else { self.host_ptr(s.host) };
            if let Some(l) = v.last_mut() {
                if unsafe { l.0.add(l.2) } == g && unsafe { l.1.add(l.2) } == h {
                    l.2 += s.bytes;
                    continue;
                }
            }
            v.push((g, h, s.bytes));
        }
        v
    }

    fn issue(&self, plan: &Plan) {
        // RAM 側だけ int8 なら、頁は詰めながら（zero-copy のカーネルで）。blob はそのまま
        let conv = self.host_shape.dtype != self.shape.dtype;
        let (pages, raw): (Vec<Seg>, Vec<Seg>) = plan.segs.iter().partition(|s| conv && matches!(s.gpu, GpuLoc::Slot(_)));
        if !pages.is_empty() {
            self.issue_conv(plan.dir, &pages);
        }
        if !raw.is_empty() {
            self.issue_raw(plan.dir, &raw);
        }
    }

    fn issue_conv(&self, dir: Dir, segs: &[Seg]) {
        if !self.conv_zero_copy {
            return self.issue_conv_staged(dir, segs);
        }
        let st = self.streams[dir as usize];
        let hl = self.host_shape.layer_page_bytes();
        let mut d = Vec::with_capacity(segs.len() * self.shape.layers);
        for s in segs {
            let GpuLoc::Slot(g) = s.gpu else { unreachable!() };
            for l in 0..self.shape.layers {
                let (gp, hp) = (self.layer_ptr(g, l), unsafe { self.host_dev_ptr(s.host).add(l * hl) });
                d.push(match dir {
                    Dir::D2H => ffi::ConvDesc { src: gp, dst: hp },
                    Dir::H2D => ffi::ConvDesc { src: hp, dst: gp },
                });
            }
        }
        let (rows, dim) = (self.shape.rows() as i32, self.shape.head_dim as i32);
        let r = unsafe {
            match dir {
                Dir::D2H => ffi::kv_quant(d.as_ptr(), d.len() as i32, rows, dim, self.kernel_blocks, st),
                Dir::H2D => ffi::kv_dequant(d.as_ptr(), d.len() as i32, rows, dim, self.kernel_blocks, st),
            }
        };
        ck(r, "kv_quant/kv_dequant");
    }

    /// GPU の置き場で詰めて（HBM だけ・短い）、RAM とはコピーエンジンで 1 本ずつ。同じ流れの中なので置き場は 1 つで足りる
    fn issue_conv_staged(&self, dir: Dir, segs: &[Seg]) {
        let st = self.streams[dir as usize];
        let (hl, hb) = (self.host_shape.layer_page_bytes(), self.mgr.host_bb);
        let (rows, dim) = (self.shape.rows() as i32, self.shape.head_dim as i32);
        let stage = self.stage[dir as usize];
        for grp in segs.chunks(self.stage_pages) {
            let mut d = Vec::with_capacity(grp.len() * self.shape.layers);
            for (i, s) in grp.iter().enumerate() {
                let GpuLoc::Slot(g) = s.gpu else { unreachable!() };
                for l in 0..self.shape.layers {
                    let sp = unsafe { stage.add(i * hb + l * hl) };
                    d.push(match dir {
                        Dir::D2H => ffi::ConvDesc { src: self.layer_ptr(g, l), dst: sp },
                        Dir::H2D => ffi::ConvDesc { src: sp, dst: self.layer_ptr(g, l) },
                    });
                }
            }
            // RAM の枠が続いている所は 1 本に
            let mut runs: Vec<(usize, u32, usize)> = vec![];
            for (i, s) in grp.iter().enumerate() {
                match runs.last_mut() {
                    Some(r) if r.1 + r.2 as u32 == s.host => r.2 += 1,
                    _ => runs.push((i, s.host, 1)),
                }
            }
            let copy = |r: &(usize, u32, usize)| {
                let (sp, hp, n) = (unsafe { stage.add(r.0 * hb) }, self.host_ptr(r.1), r.2 * hb);
                let c = unsafe {
                    match dir {
                        Dir::D2H => ffi::cudaMemcpyAsync(hp.cast(), sp as *const c_void, n, ffi::D2H, st),
                        Dir::H2D => ffi::cudaMemcpyAsync(sp.cast(), hp as *const c_void, n, ffi::H2D, st),
                    }
                };
                ck(c, "cudaMemcpyAsync（詰めた頁）");
            };
            unsafe {
                match dir {
                    Dir::D2H => {
                        ck(ffi::kv_quant(d.as_ptr(), d.len() as i32, rows, dim, 0, st), "kv_quant");
                        runs.iter().for_each(copy);
                    }
                    Dir::H2D => {
                        runs.iter().for_each(copy);
                        ck(ffi::kv_dequant(d.as_ptr(), d.len() as i32, rows, dim, 0, st), "kv_dequant");
                    }
                }
            }
        }
    }

    fn issue_raw(&self, dir: Dir, segs: &[Seg]) {
        let st = self.streams[dir as usize];
        match self.mode {
            CopyMode::Memcpy => {
                for (g, h, n) in self.coalesce(segs, false) {
                    let r = unsafe {
                        match dir {
                            Dir::D2H => ffi::cudaMemcpyAsync(h.cast(), g as *const c_void, n, ffi::D2H, st),
                            Dir::H2D => ffi::cudaMemcpyAsync(g.cast(), h as *const c_void, n, ffi::H2D, st),
                        }
                    };
                    ck(r, "cudaMemcpyAsync");
                }
            }
            CopyMode::Kernel => {
                let d: Vec<ffi::CopyDesc> = self
                    .coalesce(segs, true)
                    .into_iter()
                    .map(|(g, h, n)| match dir {
                        Dir::D2H => ffi::CopyDesc { src: g, dst: h, bytes: n as u64 },
                        Dir::H2D => ffi::CopyDesc { src: h, dst: g, bytes: n as u64 },
                    })
                    .collect();
                ck(unsafe { ffi::kv_copy(d.as_ptr(), d.len() as i32, self.kernel_blocks, st) }, "kv_copy");
            }
        }
    }

    /// 終わった転送を帳簿に返す（CPU は待たない）
    pub fn poll(&mut self) {
        let mut done = vec![];
        for d in 0..2 {
            while let Some(&(id, e)) = self.inflight[d].front() {
                match unsafe { ffi::cudaEventQuery(e) } {
                    0 => {
                        self.inflight[d].pop_front();
                        self.ev_pool.push(e);
                        done.push(id);
                    }
                    ffi::NOT_READY => break,
                    c => ck(c, "cudaEventQuery"),
                }
            }
        }
        done.sort();
        for id in done {
            self.mgr.complete(id);
        }
    }

    pub fn is_done(&self, t: Ticket) -> bool {
        (0..2).all(|d| self.inflight[d].front().map_or(true, |&(id, _)| id > t.upto[d]))
    }

    /// 札の転送が終わるまで CPU で待つ
    pub fn wait(&mut self, t: Ticket) {
        for d in 0..2 {
            if let Some(&(_, e)) = self.inflight[d].iter().rev().find(|(id, _)| *id <= t.upto[d]) {
                ck(unsafe { ffi::cudaEventSynchronize(e) }, "cudaEventSynchronize");
            }
        }
        self.poll();
    }

    /// 札の転送が終わるまで、流れ s を GPU の上で待たせる（CPU は待たない。prefetch の後の Attention の前に）
    // s（Stream）は CUDA の不透明な handle をそのまま CUDA の API に渡すだけで、Rust の側では参照外ししない（不正な値は CUDA が誤りで返す）。
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn stream_wait(&self, t: Ticket, s: ffi::Stream) {
        for d in 0..2 {
            if let Some(&(_, e)) = self.inflight[d].iter().rev().find(|(id, _)| *id <= t.upto[d]) {
                ck(unsafe { ffi::cudaStreamWaitEvent(s, e, 0) }, "cudaStreamWaitEvent");
            }
        }
    }

    pub fn sync_all(&mut self) {
        for s in self.streams {
            ck(unsafe { ffi::cudaStreamSynchronize(s) }, "cudaStreamSynchronize");
        }
        self.poll();
    }

    // ── 順番決め（scheduler）から呼ぶ口 ──

    pub fn new_seq(&mut self) -> SeqId {
        self.mgr.new_seq()
    }
    pub fn fork(&mut self, src: SeqId, n: usize) -> Result<SeqId, KvError> {
        self.mgr.fork(src, n)
    }
    pub fn free_seq(&mut self, s: SeqId) {
        self.mgr.free_seq(s);
        self.poll();
    }
    pub fn table(&self, s: SeqId) -> Option<Vec<u32>> {
        self.mgr.table(s)
    }
    /// 読み込み中も含めた頁表（prefetch の札を `stream_wait` してから使う）
    pub fn slots(&self, s: SeqId) -> Option<Vec<u32>> {
        self.mgr.slots(s)
    }

    /// 列を RAM へ（休ませる）
    pub fn evict_seq(&mut self, s: SeqId) -> Result<Ticket, KvError> {
        self.poll();
        let j = self.mgr.evict_seq(s)?;
        Ok(self.submit(j))
    }
    /// 列を GPU へ（足りなければ長く使っていない列を追い出す）。計算の前に `stream_wait` か `wait`
    pub fn prefetch(&mut self, s: SeqId) -> Result<Ticket, KvError> {
        self.poll();
        let j = self.mgr.prefetch(s)?;
        Ok(self.submit(j))
    }
    /// 手すきのうちに RAM にも写しを取る
    pub fn backup(&mut self, s: SeqId) -> Result<Ticket, KvError> {
        self.poll();
        let j = self.mgr.backup(s)?;
        Ok(self.submit(j))
    }
    /// GPU の枠を n 個空ける（足りなければ追い出す）
    pub fn make_room(&mut self, n: usize, keep: Option<SeqId>) -> Result<Ticket, KvError> {
        self.poll();
        let j = self.mgr.make_room(n, keep)?;
        Ok(self.submit(j))
    }

    /// n トークン分の場所を取る。COW の写しは計算の流れに積む（書き込みはその後に計算の流れで）
    pub fn append(&mut self, s: SeqId, n: usize) -> Result<AppendOut, KvError> {
        self.poll();
        // 揃っていない列のために他の列を追い出さない
        if self.mgr.table(s).is_none() {
            return Err(KvError::NotResident(s));
        }
        let need = self.mgr.new_pages_for_append(s, n)?;
        let j = self.mgr.make_room(need, Some(s))?;
        self.submit(j);
        let out = self.mgr.append(s, n)?;
        for &dep in &out.deps {
            if let Some(e) = self.ev_of(dep) {
                ck(unsafe { ffi::cudaStreamWaitEvent(self.compute, e, 0) }, "cudaStreamWaitEvent");
            }
        }
        for &(src, dst) in &out.cow {
            let r = unsafe {
                ffi::cudaMemcpyAsync(self.slot_ptr(dst).cast(), self.slot_ptr(src) as *const c_void, self.shape.block_bytes(), ffi::D2D, self.compute)
            };
            ck(r, "cudaMemcpyAsync（COW）");
        }
        Ok(out)
    }

    /// GPU の dev（bytes）を RAM の blob へ（線形 Attention の状態など）。
    /// 転送は「出した時までの計算の流れ」を待つ。dev を書き換えるのは札が終わってから（`stream_wait` か `wait`）
    pub fn blob_save(&mut self, dev: *const u8, bytes: usize) -> Result<(BlobId, Ticket), KvError> {
        self.poll();
        let b = self.mgr.blob_new(bytes)?;
        let j = self.mgr.blob_save(b, dev as usize);
        Ok((b, self.submit(j)))
    }
    pub fn blob_load(&mut self, b: BlobId, dev: *mut u8) -> Ticket {
        let j = self.mgr.blob_load(b, dev as usize);
        self.submit(j)
    }
    pub fn blob_free(&mut self, b: BlobId) {
        self.mgr.blob_free(b);
    }
}

// 順番決めのスレッドへ渡せるように（CUDA の流れ・event はどのスレッドからでも使える。渡した先で cudaSetDevice を同じ GPU に）
unsafe impl Send for KvTier {}

impl Drop for KvTier {
    fn drop(&mut self) {
        unsafe {
            for s in self.streams {
                ffi::cudaStreamSynchronize(s);
                ffi::cudaStreamDestroy(s);
            }
            for q in &self.inflight {
                for &(_, e) in q {
                    ffi::cudaEventDestroy(e);
                }
            }
            for &e in &self.ev_pool {
                ffi::cudaEventDestroy(e);
            }
            ffi::cudaEventDestroy(self.compute_ev);
            for p in self.stage {
                if !p.is_null() {
                    ffi::cudaFree(p.cast());
                }
            }
            if self.own_gpu {
                ffi::cudaFree(self.gpu_base.cast());
            }
        }
    }
}

// ───────────────────────── 確かめ（GPU を使う） ─────────────────────────

/// 頁（全層）を決まった模様で埋める（計算の流れで）
// s（Stream）は CUDA の不透明な handle をそのまま CUDA の API に渡すだけで、Rust の側では参照外ししない（不正な値は CUDA が誤りで返す）。
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn fill_pages(ptrs: &[*mut u8], bytes: usize, seed: u64, mode: i32, s: ffi::Stream) {
    if ptrs.is_empty() {
        return;
    }
    ck(unsafe { ffi::kv_fill(ptrs.as_ptr(), ptrs.len() as i32, bytes as u64, seed, mode, s) }, "kv_fill");
}

/// 頁ごとの指紋（流れ s で計算して、CPU で受け取る）。GPU の置き場は使い回す（cudaFree は GPU 全体を待つので）
pub struct Hasher {
    d: *mut u64,
    cap: usize,
}
impl Hasher {
    pub fn new(cap: usize) -> Self {
        let mut d = std::ptr::null_mut();
        ck(unsafe { ffi::cudaMalloc(&mut d, cap * 8) }, "cudaMalloc");
        Hasher { d: d.cast(), cap }
    }
    // s（Stream）は CUDA の不透明な handle をそのまま CUDA の API に渡すだけで、Rust の側では参照外ししない（不正な値は CUDA が誤りで返す）。
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn hash(&self, ptrs: &[*mut u8], bytes: usize, s: ffi::Stream) -> Vec<u64> {
        let mut out = vec![0u64; ptrs.len()];
        for (i, ch) in ptrs.chunks(self.cap).enumerate() {
            unsafe {
                ck(ffi::kv_hash(ch.as_ptr(), ch.len() as i32, bytes as u64, self.d, s), "kv_hash");
                let o = out[i * self.cap..].as_mut_ptr();
                ck(ffi::cudaMemcpyAsync(o.cast(), self.d as *const c_void, ch.len() * 8, ffi::D2H, s), "cudaMemcpyAsync");
                ck(ffi::cudaStreamSynchronize(s), "cudaStreamSynchronize");
            }
        }
        out
    }
}
impl Drop for Hasher {
    fn drop(&mut self) {
        unsafe { ffi::cudaFree(self.d.cast()) };
    }
}

/// 小さな乱数（試しの再現用）
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            0
        } else {
            self.next() % n
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct SelfTest {
    pub ops: usize,
    pub verified_pages: usize,
    pub stats: Stats,
}

/// でたらめな操作（書き足し・分岐・追い出し・読み込み・写し・blob・解放）を重ね、
/// 読み込むたびに全頁の指紋が書いた時と同じかを GPU で確かめる。帳簿のつじつまも毎回見る
/// gpu_pages・host_pages を小さくすると、追い出しと RAM の写しの捨て方が多く通る
pub fn selftest(iters: usize, dtype: KvDtype, mode: CopyMode, seed: u64, gpu_pages: u32, host_pages: u32) -> Result<SelfTest, String> {
    let shape = KvShape::qwen38_27b(dtype);
    let bb = shape.block_bytes();
    let mut t = KvTier::new(TierConfig { shape, gpu_pages, host_pages, host_node: None, mode, host_dtype: None });
    let host_min = t.mgr.n_host as usize / 16;
    t.mgr.chunk_pages = 3;
    let mut r = Rng(seed);
    // 列ごと・頁ごとの、書いたときの指紋
    let mut want: HashMap<SeqId, Vec<u64>> = HashMap::new();
    let mut tickets: HashMap<SeqId, Ticket> = HashMap::new();
    let mut st = SelfTest::default();
    let mut blobs: Vec<(BlobId, u64, Ticket)> = vec![];
    let (mut blob_dev, mut blob_tmp) = (std::ptr::null_mut(), std::ptr::null_mut());
    let blob_bytes = bb * 2 + 12_345 * 16;
    ck(unsafe { ffi::cudaMalloc(&mut blob_dev, blob_bytes) }, "cudaMalloc");
    ck(unsafe { ffi::cudaMalloc(&mut blob_tmp, blob_bytes) }, "cudaMalloc");
    let hs = Hasher::new(256);
    // 指紋は別の流れ・別の置き場に同じ模様を書いて取る（計算の流れは待たない → 転送が計算を待つかを試せる）
    let mut side = std::ptr::null_mut();
    ck(unsafe { ffi::cudaStreamCreateWithFlags(&mut side, 1) }, "cudaStreamCreate");
    let mut scratch = std::ptr::null_mut();
    ck(unsafe { ffi::cudaMalloc(&mut scratch, bb * 8) }, "cudaMalloc");
    let scratch: *mut u8 = scratch.cast();
    let spin = |s: ffi::Stream| ck(unsafe { ffi::kv_spin(400_000, s) }, "kv_spin"); // 約 0.3ms

    // 列の全頁を GPU に揃えて指紋を比べる
    let verify = |t: &mut KvTier, s: SeqId, want: &HashMap<SeqId, Vec<u64>>, tk: Option<Ticket>, hs: &Hasher| -> Result<usize, String> {
        // GPU の上で待つ（CPU は待たない）→ 読み込み中の枠のまま指紋を取る
        if let Some(tk) = tk {
            t.stream_wait(tk, t.compute);
        }
        let tab = t.slots(s).ok_or("GPU に枠が無い頁がある")?;
        let ptrs: Vec<*mut u8> = tab.iter().map(|&g| t.slot_ptr(g)).collect();
        let got = hs.hash(&ptrs, bb, t.compute);
        if got != want[&s] {
            let bad = got.iter().zip(&want[&s]).position(|(a, b)| a != b);
            return Err(format!("列 {s} の頁 {bad:?} の指紋が違う"));
        }
        Ok(got.len())
    };

    for it in 0..iters {
        let ids = t.mgr.seq_ids();
        let pick = |r: &mut Rng| ids.get(r.below(ids.len() as u64) as usize).copied();
        let op = r.below(100);
        st.ops += 1;
        match op {
            // 新しい列 or 書き足し
            0..=29 => {
                // RAM に余裕を残す（終わりに全部を戻せるように）
                if t.mgr.host_free() < host_min {
                    continue;
                }
                let s = if ids.is_empty() || r.below(4) == 0 {
                    let s = t.new_seq();
                    want.insert(s, vec![]);
                    s
                } else {
                    pick(&mut r).unwrap()
                };
                // 書き足すには揃っていること
                match t.prefetch(s) {
                    Ok(tk) => t.wait(tk),
                    Err(_) => continue,
                }
                let n = 1 + r.below(200) as usize;
                t.mgr.pin(s);
                let before = t.mgr.tokens(s).unwrap();
                let out = match t.append(s, n) {
                    Ok(o) => o,
                    Err(_) => {
                        t.mgr.unpin(s);
                        continue;
                    }
                };
                let tab = t.table(s).ok_or("append の後に揃っていない")?;
                // COW した頁は、写した直後は元と同じ指紋
                let first = before / shape.page_tokens;
                if !out.cow.is_empty() {
                    let h = hs.hash(&[t.slot_ptr(tab[first])], bb, t.compute)[0];
                    if h != want[&s][first] {
                        return Err(format!("it {it}: COW の写しが元と違う"));
                    }
                }
                // 書いた頁（最後の古い頁から後ろ）を新しい模様で埋める。計算の流れは止めてから書き、待たずに進む
                let w = want.get_mut(&s).unwrap();
                w.resize(tab.len(), 0);
                let touched: Vec<usize> = (first..tab.len()).collect();
                let ptrs: Vec<*mut u8> = touched.iter().map(|&i| t.slot_ptr(tab[i])).collect();
                let sd = r.next();
                let sp: Vec<*mut u8> = (0..touched.len()).map(|i| unsafe { scratch.add(i * bb) }).collect();
                fill_pages(&sp, bb, sd, 0, side);
                let got = hs.hash(&sp, bb, side);
                for (k, &i) in touched.iter().enumerate() {
                    w[i] = got[k];
                }
                spin(t.compute);
                fill_pages(&ptrs, bb, sd, 0, t.compute);
                t.mgr.unpin(s);
                // 書いた直後に追い出す・写す（D2H は書き込みを待たなければならない）
                match r.below(3) {
                    0 => {
                        let _ = t.evict_seq(s);
                    }
                    1 => {
                        let _ = t.backup(s);
                    }
                    _ => {}
                }
            }
            // 文頭を共有した分岐
            30..=39 => {
                if let Some(s) = pick(&mut r) {
                    let n = r.below(t.mgr.tokens(s).unwrap() as u64 + 1) as usize;
                    let c = t.fork(s, n).map_err(|e| e.to_string())?;
                    let np = shape.pages_for(n);
                    want.insert(c, want[&s][..np].to_vec());
                }
            }
            // 追い出し
            40..=54 => {
                if let Some(s) = pick(&mut r) {
                    if let Ok(tk) = t.evict_seq(s) {
                        tickets.insert(s, tk);
                    }
                }
            }
            // 読み込み → 確かめ
            55..=74 => {
                if let Some(s) = pick(&mut r) {
                    if let Ok(tk) = t.prefetch(s) {
                        let n = verify(&mut t, s, &want, Some(tk), &hs)?;
                        st.verified_pages += n;
                    }
                }
            }
            // RAM にも写す
            75..=81 => {
                if let Some(s) = pick(&mut r) {
                    let _ = t.backup(s);
                }
            }
            // 解放
            82..=89 => {
                if ids.len() > 3 {
                    if let Some(s) = pick(&mut r) {
                        t.free_seq(s);
                        want.remove(&s);
                    }
                }
            }
            // blob（線形 Attention の状態の写しの代わり）
            90..=95 => {
                if blobs.len() < 4 && r.below(2) == 0 {
                    let sd = r.next();
                    fill_pages(&[blob_tmp.cast()], blob_bytes, sd, 0, side);
                    let h = hs.hash(&[blob_tmp.cast()], blob_bytes, side)[0];
                    spin(t.compute);
                    fill_pages(&[blob_dev.cast()], blob_bytes, sd, 0, t.compute);
                    if let Ok((b, tk)) = t.blob_save(blob_dev.cast(), blob_bytes) {
                        blobs.push((b, h, tk));
                        // 元を書き換えるのは D2H が読み終わってから（計算の流れを札で待たせる）
                        t.stream_wait(tk, t.compute);
                        fill_pages(&[blob_dev.cast()], blob_bytes, sd ^ 1, 0, t.compute);
                    }
                } else if !blobs.is_empty() {
                    let (b, h, _) = blobs.remove(r.below(blobs.len() as u64) as usize);
                    let tk = t.blob_load(b, blob_tmp.cast());
                    t.wait(tk);
                    if hs.hash(&[blob_tmp.cast()], blob_bytes, side)[0] != h {
                        return Err(format!("it {it}: blob {b} が戻らない"));
                    }
                    t.blob_free(b);
                }
            }
            // ときどき全部待つ
            _ => t.sync_all(),
        }
        t.poll();
        t.mgr.check().map_err(|e| format!("it {it} 操作 {op}: {e}"))?;
    }
    // 最後に全部を戻して確かめる
    t.sync_all();
    // 他の列を追い出してでも戻す。確かめたら解放（RAM が詰まっていれば、戻せる列から先に）
    while !t.mgr.seq_ids().is_empty() {
        let mut progress = false;
        let mut last_err = String::new();
        for s in t.mgr.seq_ids() {
            match t.prefetch(s) {
                Ok(tk) => {
                    st.verified_pages += verify(&mut t, s, &want, Some(tk), &hs)?;
                    t.free_seq(s);
                    progress = true;
                }
                Err(e) => last_err = format!("終わりの読み込み {s}: {e}"),
            }
        }
        if !progress {
            return Err(last_err);
        }
    }
    for (b, _, _) in blobs {
        t.blob_free(b);
    }
    t.sync_all();
    t.mgr.check()?;
    if t.mgr.gpu_free() != t.mgr.n_gpu as usize || t.mgr.host_free() != t.mgr.n_host as usize {
        return Err(format!("枠が戻らない: GPU {}/{} RAM {}/{}", t.mgr.gpu_free(), t.mgr.n_gpu, t.mgr.host_free(), t.mgr.n_host));
    }
    st.stats = t.mgr.stats.clone();
    unsafe {
        ffi::cudaFree(blob_dev);
        ffi::cudaFree(blob_tmp);
        ffi::cudaFree(scratch.cast());
        ffi::cudaStreamDestroy(side);
    }
    Ok(st)
}

// ───────────────────────── 単体の試し（帳簿だけ。GPU は要らない） ─────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mgr(g: u32, h: u32) -> PageManager {
        let mut m = PageManager::new(KvShape::qwen38_27b(KvDtype::F16), g, h);
        m.chunk_pages = 2;
        m
    }
    /// 走らせたことにする（帳簿だけ）
    fn run(m: &mut PageManager, j: &Job) {
        for p in &j.plans {
            m.complete(p.id);
        }
    }
    fn grow(m: &mut PageManager, s: SeqId, n: usize) -> AppendOut {
        let need = m.new_pages_for_append(s, n).unwrap();
        let j = m.make_room(need, Some(s)).unwrap();
        run(m, &j);
        m.append(s, n).unwrap()
    }

    #[test]
    fn shape_sizes() {
        let f = KvShape::qwen38_27b(KvDtype::F16);
        assert_eq!(f.layer_page_bytes(), 256 * 1024);
        assert_eq!(f.block_bytes(), 4 << 20);
        assert_eq!(f.bytes_per_token(), 65536.0);
        let q = KvShape::qwen38_27b(KvDtype::Int8);
        assert_eq!(q.layer_page_bytes(), 128 * 1024 + 1024);
        assert_eq!(q.block_bytes(), 16 * 132096);
        assert_eq!(q.row_off(1, 1, 2, 3), 132096 + 65536 + (2 * 64 + 3) * 256);
        assert_eq!(q.scale_off(0, 1, 0, 0), 131072 + 4 * 64 * 2);
    }

    #[test]
    fn alloc_free_refcount() {
        let mut m = mgr(8, 8);
        let a = m.new_seq();
        grow(&mut m, a, 130); // 3 頁
        assert_eq!(m.table(a).unwrap().len(), 3);
        assert_eq!(m.gpu_free(), 5);
        let b = m.fork(a, 100).unwrap(); // 2 頁を共有
        let pa = m.pages_of(a).unwrap();
        assert_eq!(m.refs(pa[0]), 2);
        assert_eq!(m.refs(pa[1]), 2);
        assert_eq!(m.refs(pa[2]), 1);
        m.check().unwrap();
        m.free_seq(a);
        assert_eq!(m.refs(pa[0]), 1);
        assert_eq!(m.gpu_free(), 6); // a だけの頁 1 つが空いた
        m.free_seq(b);
        assert_eq!(m.gpu_free(), 8);
        m.check().unwrap();
    }

    #[test]
    fn cow_on_shared_partial_page() {
        let mut m = mgr(8, 8);
        let a = m.new_seq();
        grow(&mut m, a, 100); // 頁 0 満杯・頁 1 に 36
        let b = m.fork(a, 90); // 頁 1 の 26 トークン目まで共有
        let b = b.unwrap();
        let pa = m.pages_of(a).unwrap();
        // b が書き足す → 頁 1 は共有なので写す。頁 0（満杯）はそのまま
        assert_eq!(m.new_pages_for_append(b, 10).unwrap(), 1);
        let out = grow(&mut m, b, 10);
        assert_eq!(out.cow.len(), 1);
        assert_eq!(out.cow[0].0, m.gpu_slot(pa[1]).unwrap());
        let pb = m.pages_of(b).unwrap();
        assert_eq!(pb[0], pa[0]);
        assert_ne!(pb[1], pa[1]);
        assert_eq!(m.refs(pa[1]), 1);
        assert_eq!(m.tokens(b), Some(100));
        // a が書き足しても写さない（もう 1 人だけ）
        let out = grow(&mut m, a, 5);
        assert!(out.cow.is_empty());
        // 頁の境目で分岐したら写さない
        let c = m.fork(a, 64).unwrap();
        let out = grow(&mut m, c, 3);
        assert!(out.cow.is_empty());
        assert_eq!(out.new_pages, 1);
        m.check().unwrap();
    }

    #[test]
    fn eviction_order_lru_prio_pin() {
        let mut m = mgr(6, 32);
        let s: Vec<SeqId> = (0..3).map(|_| m.new_seq()).collect();
        for &q in &s {
            grow(&mut m, q, 128); // 2 頁ずつ → 満杯
        }
        assert_eq!(m.gpu_free(), 0);
        m.touch(s[0]); // s1 が一番古い
        let d = m.new_seq();
        let p1 = m.pages_of(s[1]).unwrap();
        let g1 = m.gpu_slot(p1[1]).unwrap();
        let j = m.make_room(2, Some(d)).unwrap();
        assert_eq!(j.plans.len(), 1);
        assert_eq!(j.plans[0].segs.len(), 2);
        // s1 の 2 頁が出る
        assert_eq!(m.gpu_slot(p1[0]), None);
        assert_eq!(m.gpu_slot(p1[1]), None);
        assert!(matches!(j.plans[0].segs[1].gpu, GpuLoc::Slot(g) if g == g1));
        run(&mut m, &j);
        assert!(m.table(s[1]).is_none());
        assert!(m.table(s[0]).is_some() && m.table(s[2]).is_some());
        // 優先度: s2 を上げ、s0 を pin → 次は… s0 は pin、s2 は優先度が高い → 出せるのは無い（s1 は既に RAM）
        m.set_prio(s[2], 5);
        m.pin(s[0]);
        grow(&mut m, d, 128);
        assert_eq!(m.gpu_free(), 0);
        let e = m.new_seq();
        // d（優先度 0）が先に出る
        let j = m.make_room(2, Some(e)).unwrap();
        run(&mut m, &j);
        assert!(m.table(d).is_none());
        assert!(m.table(s[2]).is_some());
        // 次は s2（優先度 5）しか無い（空き 2 ＋ 2）
        let j = m.make_room(4, Some(e)).unwrap();
        run(&mut m, &j);
        assert!(m.table(s[2]).is_none());
        // s0 は pin なので出せない
        assert!(matches!(m.make_room(5, Some(e)), Err(KvError::NoGpuPages { .. })));
        m.check().unwrap();
    }

    #[test]
    fn evict_prefetch_roundtrip_and_clean() {
        let mut m = mgr(4, 8);
        let a = m.new_seq();
        grow(&mut m, a, 200); // 4 頁
        let j = m.evict_seq(a).unwrap();
        assert_eq!(j.plans.iter().map(|p| p.segs.len()).sum::<usize>(), 4);
        assert_eq!(j.plans.len(), 2); // chunk_pages = 2
        // 追い出し中の枠は、待てば使える
        assert_eq!(m.gpu_free(), 4);
        let b = m.new_seq();
        let out = grow(&mut m, b, 10);
        assert_eq!(out.deps.len(), 1); // まだ終わっていない D2H を待つ
        run(&mut m, &j);
        m.free_seq(b);
        // 戻す
        let j = m.prefetch(a).unwrap();
        assert!(m.table(a).is_none()); // 転送中
        run(&mut m, &j);
        assert_eq!(m.table(a).unwrap().len(), 4);
        // RAM に同じ写しがあるので、2 回目の追い出しは転送なし
        let j = m.evict_seq(a).unwrap();
        assert!(j.plans.is_empty());
        assert_eq!(m.stats.clean_evicts, 4);
        assert!(m.table(a).is_none());
        // 書き足すと RAM の写しは古くなる
        let j = m.prefetch(a).unwrap();
        run(&mut m, &j);
        grow(&mut m, a, 1); // 最後の頁に書く
        let j = m.evict_seq(a).unwrap();
        assert_eq!(j.plans.iter().map(|p| p.segs.len()).sum::<usize>(), 1); // 書いた頁だけ
        run(&mut m, &j);
        m.check().unwrap();
    }

    #[test]
    fn prefetch_waits_for_inflight_eviction() {
        let mut m = mgr(4, 8);
        let a = m.new_seq();
        grow(&mut m, a, 64);
        let j1 = m.evict_seq(a).unwrap();
        let id1 = j1.plans[0].id;
        // D2H が終わる前に読み戻す → H2D は D2H を待つ（同じ RAM の枠）
        let j2 = m.prefetch(a).unwrap();
        assert!(j2.plans[0].deps.contains(&id1));
        // 終わった順に返す
        run(&mut m, &j1);
        run(&mut m, &j2);
        assert!(m.table(a).is_some());
        m.check().unwrap();
        // 転送中に解放しても、終わった後に枠が戻る
        let j3 = m.evict_seq(a).unwrap();
        assert!(j3.plans.is_empty()); // clean
        let j4 = m.prefetch(a).unwrap();
        m.free_seq(a);
        m.check().unwrap();
        run(&mut m, &j4);
        assert_eq!(m.gpu_free(), 4);
        assert_eq!(m.host_free(), 8);
        m.check().unwrap();
    }

    #[test]
    fn host_full_drops_redundant_copies() {
        let mut m = mgr(4, 2);
        let a = m.new_seq();
        grow(&mut m, a, 128);
        let j = m.backup(a).unwrap();
        run(&mut m, &j);
        assert_eq!(m.host_free(), 0);
        let b = m.new_seq();
        grow(&mut m, b, 64);
        // RAM は満杯だが、a の写しは GPU にもあるので捨てて b を出せる
        let j = m.evict_seq(b).unwrap();
        run(&mut m, &j);
        assert_eq!(m.stats.host_drops, 1);
        assert!(m.table(b).is_none());
        m.check().unwrap();
        // a の 1 頁は写しを失った。a を出すには 1 枠要るが、捨ててよい写し（a 以外で GPU にもある物）が無い → 断る
        assert!(matches!(m.evict_seq(a), Err(KvError::NoHostPages { .. })));
        // 何も変わっていない
        assert!(m.table(a).is_some());
        assert_eq!(m.stats.host_drops, 1);
        m.check().unwrap();
    }

    #[test]
    fn append_during_backup_marks_stale() {
        let mut m = mgr(4, 4);
        let a = m.new_seq();
        grow(&mut m, a, 10);
        let j = m.backup(a).unwrap();
        grow(&mut m, a, 5); // 写している最中に書き足す
        run(&mut m, &j);
        // 写しは古いので捨てられ、次の追い出しはもう一度写す
        assert_eq!(m.host_free(), 4);
        let j = m.evict_seq(a).unwrap();
        assert_eq!(j.plans.len(), 1);
        run(&mut m, &j);
        m.check().unwrap();
    }

    #[test]
    fn blob_alloc_and_deferred_free() {
        let mut m = mgr(2, 8);
        let bb = m.shape.block_bytes();
        let b = m.blob_new(bb * 2 + 5).unwrap();
        assert_eq!(m.host_free(), 5);
        let j = m.blob_save(b, 0x1000);
        assert_eq!(j.plans[0].segs.len(), 3);
        assert_eq!(j.plans[0].segs[2].bytes, 5);
        let j2 = m.blob_load(b, 0x2000);
        assert_eq!(j2.plans[0].deps, vec![j.plans[0].id]);
        m.blob_free(b); // 転送中 → 後で
        assert_eq!(m.host_free(), 5);
        run(&mut m, &j);
        run(&mut m, &j2);
        assert_eq!(m.host_free(), 8);
        m.check().unwrap();
    }

    #[test]
    fn random_ops_keep_books_consistent() {
        // 転送は出した順にでたらめな遅れで終わらせる
        let mut m = mgr(12, 30);
        let mut r = Rng(7);
        let mut pending: Vec<Plan> = vec![];
        for _ in 0..5000 {
            let ids = m.seq_ids();
            let pick = |r: &mut Rng| ids.get(r.below(ids.len() as u64) as usize).copied();
            match r.below(10) {
                0..=2 => {
                    let s = if ids.is_empty() || r.below(3) == 0 { m.new_seq() } else { pick(&mut r).unwrap() };
                    if let Ok(j) = m.prefetch(s) {
                        pending.extend(j.plans);
                        // 書き足すには揃っていること
                        while m.table(s).is_none() && !pending.is_empty() {
                            let p = pending.remove(0);
                            m.complete(p.id);
                        }
                        let n = 1 + r.below(150) as usize;
                        if let Ok(need) = m.new_pages_for_append(s, n) {
                            if let Ok(j) = m.make_room(need, Some(s)) {
                                pending.extend(j.plans);
                                let _ = m.append(s, n);
                            }
                        }
                    }
                }
                3 => {
                    if let Some(s) = pick(&mut r) {
                        let n = r.below(m.tokens(s).unwrap() as u64 + 1) as usize;
                        m.fork(s, n).unwrap();
                    }
                }
                4 | 5 => {
                    if let Some(s) = pick(&mut r) {
                        if let Ok(j) = m.evict_seq(s) {
                            pending.extend(j.plans);
                        }
                    }
                }
                6 => {
                    if let Some(s) = pick(&mut r) {
                        if let Ok(j) = m.backup(s) {
                            pending.extend(j.plans);
                        }
                    }
                }
                7 => {
                    if ids.len() > 4 {
                        if let Some(s) = pick(&mut r) {
                            m.free_seq(s);
                        }
                    }
                }
                _ => {
                    let k = r.below(pending.len() as u64 + 1) as usize;
                    for p in pending.drain(..k.min(pending.len())) {
                        m.complete(p.id);
                    }
                }
            }
            m.check().unwrap();
        }
        for p in pending.drain(..) {
            m.complete(p.id);
        }
        for s in m.seq_ids() {
            m.free_seq(s);
        }
        m.check().unwrap();
        assert_eq!(m.gpu_free(), 12);
        assert_eq!(m.host_free(), 30);
        assert_eq!(m.in_flight(), 0);
    }

    /// GPU で: でたらめな操作の後も中身が壊れていないか（CUDA_VISIBLE_DEVICES=7 cargo test --release --lib -- --ignored）
    #[test]
    #[ignore]
    fn gpu_selftest() {
        for (dt, mode) in [(KvDtype::F16, CopyMode::Memcpy), (KvDtype::Int8, CopyMode::Kernel)] {
            let r = selftest(1500, dt, mode, 11, 40, 160).unwrap();
            eprintln!("{dt:?} {mode:?}: {r:?}");
            assert!(r.verified_pages > 100);
        }
    }
}
