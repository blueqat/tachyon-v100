//! 下書きの確かめ（M2・MTP）: `spec::Target` を M1 の Model に（1 列ずつ・温度 0）。
//!
//!   確かめ（[id_last, d1..dk] の T = 1 + k 語）は forward の GEMV の道で 1 回に流す。GDN の層は 1 語ずつの道（gdn::decode を T で）にし、
//!   新しい GDN の状態・conv の状態は仮の置き場（new_seq_spec が 2 倍取った後ろ半分）へ書く（元の状態は残る）。
//!   qkv（conv の前）・conv の後・α・β は層ごとの控え [n_gdn][tmax][..] に書く。
//!   commit(n_keep): 層ごとに conv の状態と GDN の状態を、元の状態から控えの前 n_keep 語だけ進め直す（n_valid。重みは読まない）。
//!   KV は位置で書くので戻さない（外れた位置は次に上書きされ、Attention は kv_len までしか読まない）。
//!   MTP に渡す h_nextn（output_norm の後）は全部の行を hbuf に写す（次の step まで有効）
use super::*;
use crate::mtp::{self, DevRow};
use crate::spec::{RowOut, SeqId, SeqStep, StepOut, Target};

pub(crate) struct SpecBufs {
    /// forward が確かめの形で動いている間だけ true
    pub(crate) active: bool,
    pub(crate) tmax: usize,
    cc: usize,
    hv: usize,
    qkv: DevBuf,
    c: DevBuf,
    a: DevBuf,
    b: DevBuf,
    /// i32 [1]: 仮の置き場の番号（state_at(gi) から見て）
    pub(crate) slots_out: DevBuf,
    pub(crate) slots_out_val: i32,
    /// i32 [1]: commit で進める語の数
    n_valid: DevBuf,
}

impl SpecBufs {
    fn new(cfg: &Config, n_gdn: usize, tmax: usize) -> SpecBufs {
        let (cc, hv) = (cfg.gdn.conv_ch(), cfg.gdn.hv);
        let n = n_gdn.max(1) * tmax;
        SpecBufs {
            active: false,
            tmax,
            cc,
            hv,
            qkv: DevBuf::new(n * cc * 4),
            c: DevBuf::new(n * cc * 4),
            a: DevBuf::new(n * hv * 4),
            b: DevBuf::new(n * hv * 4),
            slots_out: DevBuf::new(4),
            slots_out_val: -1,
            n_valid: DevBuf::new(4),
        }
    }
    /// GDN の層 gi の控え（qkv・α・β・conv の後）
    pub(crate) fn hist(&self, gi: usize) -> (*mut f32, *mut f32, *mut f32, *mut f32) {
        let (t, cc, hv) = (self.tmax, self.cc, self.hv);
        (at(&self.qkv, gi * t * cc * 4), at(&self.a, gi * t * hv * 4), at(&self.b, gi * t * hv * 4), at(&self.c, gi * t * cc * 4))
    }
    /// 仮の置き場の番号: 列 s の層 gi は全体で (n + s)·n_gdn + gi、state_at(gi) は s·n_gdn + gi → 差は n·n_gdn（列によらない）
    pub(crate) fn slots_out_for(&self, seq: &Seq) -> *const i32 {
        let want = (seq.pool.n_slots * seq.pool.n_gdn) as i32;
        assert_eq!(self.slots_out_val, want, "仮の置き場の番号を先に置く（forward_verify）");
        self.slots_out.ptr.cast()
    }
}

impl Model {
    /// 下書きの確かめを使えるようにする（tmax = 1 回の確かめの語の上限 = 1 + 下書きの数）
    pub fn enable_spec(&mut self, tmax: usize) {
        assert!(tmax >= 2 && tmax <= GEMV_B_MAX && tmax <= self.opt.gemv_max, "確かめの語数 {tmax} は 2..={}", self.opt.gemv_max);
        let n_gdn = self.layers.iter().filter(|l| matches!(l.mix, Mix::Gdn(_))).count();
        if self.spec.as_ref().map_or(true, |b| b.tmax < tmax) {
            self.spec = Some(SpecBufs::new(&self.cfg, n_gdn, tmax));
        }
    }

    /// 確かめに使える列（GDN の状態に仮の置き場つき）
    pub fn new_seq_spec(&self, max_ctx: usize) -> Seq {
        Seq::new_n_ex(&self.cfg, self.layer0..self.layer0 + self.layers.len(), max_ctx, 1, true).pop().unwrap()
    }

    /// 確かめの forward: tokens（2..=tmax 語）を seq の続きとして流す。GDN の新しい状態は仮の置き場へ（元は残る）。
    /// seq.pos は tokens の数だけ進む（commit_verify で受け取った分に戻す）
    pub fn forward_verify(&mut self, seq: &mut Seq, tokens: &[u32]) {
        self.forward_verify_dump(seq, tokens, None)
    }

    /// forward_verify の途中の値を写す形（確かめ用）
    pub fn forward_verify_dump(&mut self, seq: &mut Seq, tokens: &[u32], dump: Option<&mut Dump>) {
        let want = (seq.pool.n_slots * seq.pool.n_gdn) as i32;
        let b = self.spec.as_mut().expect("enable_spec を先に");
        assert!(tokens.len() >= 1 && tokens.len() <= b.tmax);
        if b.slots_out_val != want {
            upload_i32(&b.slots_out, &[want]);
            b.slots_out_val = want;
        }
        b.active = true;
        self.forward(seq, tokens, dump);
        self.spec.as_mut().unwrap().active = false;
    }

    /// 確かめの後: 直前の forward_verify（t 語）のうち前の n_keep 語だけ通した状態にする（GDN・conv の状態を元から進め直す）
    pub fn commit_verify(&mut self, seq: &mut Seq, t: usize, n_keep: usize) {
        self.commit_verify_rows(seq, t, n_keep, 0)
    }

    /// commit_verify の、控えの行 row0.. を使う形（verify_multi の列 b は row0 = b·T）
    pub fn commit_verify_rows(&mut self, seq: &mut Seq, t: usize, n_keep: usize, row0: usize) {
        assert!(n_keep >= 1 && n_keep <= t && seq.pos >= t);
        let sb = self.spec.as_ref().expect("enable_spec を先に");
        upload_i32(&sb.n_valid, &[n_keep as i32]);
        let sh = self.cfg.gdn;
        let mut gi = 0;
        for l in &self.layers {
            if let Mix::Gdn(w) = &l.mix {
                let (qkv, a, b, c) = sb.hist(gi);
                let (cc, hv) = (sh.conv_ch(), sh.hv);
                let (qkv, a, b, c) = unsafe { (qkv.add(row0 * cc), a.add(row0 * hv), b.add(row0 * hv), c.add(row0 * cc)) };
                let layer = gdn::Layer { a: w.a.f32p(), dt: w.dt.f32p(), norm: w.norm.f32p(), conv: w.conv.f32p() };
                let mut io = gdn::Io::empty();
                io.qkv = qkv;
                io.qkv_ld = sh.conv_ch();
                io.c = c;
                io.z = c; // y が null なので読まない（揃えの確かめだけ）
                io.z_ld = sh.out_dim();
                io.alpha = a;
                io.beta = b;
                io.ab_ld = sh.hv;
                io.y = null_mut();
                io.y_ld = sh.out_dim();
                io.state = seq.state_at(gi);
                io.conv_state = seq.conv_at(gi);
                io.n_valid = sb.n_valid.ptr.cast();
                gdn::conv_state_only(&sh, &layer, &io, 1, t, S);
                gdn::decode(&sh, &layer, &io, 1, t, S);
                gi += 1;
            }
        }
        seq.pos = seq.pos - t + n_keep;
    }

    /// 作業場の h の行 0..n を output_norm して y（f32 [n][d]）へ（MTP の h_nextn）
    pub(crate) fn out_norm_rows(&self, n: usize, y: *mut f32) {
        assert!(n >= 1 && n <= self.sc.tc);
        let d = self.cfg.d;
        check(
            unsafe {
                kffi::infer_m_rmsnorm(self.sc.h.f32p(), d as i64, self.out_norm.f32p(), d as i32, n as i32, self.cfg.eps, y, d as i64,
                                      null_mut(), 0, self.st)
            },
            "output_norm（h_nextn）",
        );
    }

    /// MTP に本体の埋め込み・出力を貸す（本体が先に消えないこと）
    pub fn mtp_shared(&self, g: &gguf::Model) -> mtp::Shared {
        let tv = g.tensor("token_embd.weight");
        let out = self.output.as_ref().expect("出力の層が要る");
        mtp::Shared {
            embd: Some(mtp::Embd::borrow(self.embd.ptr, self.embd_qt as i32, tv.row_bytes(), tv.k(), tv.n_rows())),
            output: Some(mtp::G3Mat::borrow(out.buf.ptr, out.qt, out.k, out.n)),
        }
    }
}

/// spec::Target の本体（1 列・温度 0）
pub struct ModelTarget<'a> {
    pub m: &'a mut Model,
    pub seq: Seq,
    hbuf: DevBuf,
    hrows: usize,
    tmax: usize,
    /// 直前の確かめ（pos0, t）。commit が受け取る
    pending: Option<(usize, usize)>,
    /// 計った時間（ms）: 確かめの forward＋出口、commit
    pub verify_ms: f64,
    pub commit_ms: f64,
    pub n_verify: usize,
    /// 確かめを CUDA Graph の道（Model::verify_step）で
    pub graph: bool,
    /// 受け取った下書きの数ごとの歩の数（[0] = 1 つも受け取らなかった）
    pub acc_hist: Vec<usize>,
}

impl<'a> ModelTarget<'a> {
    pub fn new(m: &'a mut Model, max_ctx: usize, tmax: usize) -> ModelTarget<'a> {
        m.enable_spec(tmax);
        let seq = m.new_seq_spec(max_ctx);
        let hrows = m.opt.chunk.max(tmax).min(m.sc.tc);
        let hbuf = DevBuf::new(hrows * m.cfg.d * 4);
        ModelTarget { m, seq, hbuf, hrows, tmax, pending: None, verify_ms: 0.0, commit_ms: 0.0, n_verify: 0, graph: false, acc_hist: vec![] }
    }
    /// 列の頭に戻す（次の文の前）
    pub fn reset(&mut self) {
        self.seq.reset();
        self.pending = None;
    }
}

impl<'a> Target for ModelTarget<'a> {
    type H = DevRow;

    fn step(&mut self, batch: &[SeqStep]) -> Result<StepOut<DevRow>, String> {
        if batch.len() != 1 {
            return Err(format!("ModelTarget は 1 列ずつ（{}）", batch.len()));
        }
        let st = &batch[0];
        if st.sampling.is_some() {
            return Err("ModelTarget は温度 0 だけ（試料の分布はまだ）".into());
        }
        if st.pos0 != self.seq.pos {
            return Err(format!("位置が合わない: pos0 {} ≠ 列 {}", st.pos0, self.seq.pos));
        }
        if self.pending.is_some() {
            return Err("前の確かめが commit されていない".into());
        }
        let t = st.tokens.len();
        if t == 0 || t > self.hrows {
            return Err(format!("1 回の語数 {t} は 1..={}", self.hrows));
        }
        let t0 = std::time::Instant::now();
        let verify = st.out_from == 0 && t >= 2 && t <= self.tmax;
        if verify && self.graph {
            let am = self.m.verify_step(&mut self.seq, st.tokens, self.hbuf.f32p());
            self.pending = Some((st.pos0, t));
            self.verify_ms += t0.elapsed().as_secs_f64() * 1e3;
            self.n_verify += 1;
            let d = self.m.cfg.d;
            let rows = am.into_iter().map(|a| RowOut { argmax: a, dist: None }).collect();
            let h: Vec<DevRow> = (0..t).map(|i| DevRow(unsafe { self.hbuf.f32p().add(i * d) as *const f32 })).collect();
            return Ok(StepOut { rows: vec![rows], h: vec![h] });
        }
        if verify {
            self.m.forward_verify(&mut self.seq, st.tokens);
            self.pending = Some((st.pos0, t));
        } else {
            self.m.forward(&mut self.seq, st.tokens, None);
        }
        self.m.out_norm_rows(t, self.hbuf.f32p());
        let d = self.m.cfg.d;
        let mut rows: Vec<RowOut> = (0..t).map(|_| RowOut { argmax: 0, dist: None }).collect();
        let mut r = st.out_from;
        while r < t {
            let n = (t - r).min(GEMV_B_MAX);
            self.m.head(r, n);
            for (i, a) in self.m.argmax(n).into_iter().enumerate() {
                rows[r + i].argmax = a;
            }
            r += n;
        }
        if verify {
            self.verify_ms += t0.elapsed().as_secs_f64() * 1e3;
            self.n_verify += 1;
        }
        let h: Vec<DevRow> = (0..t).map(|i| DevRow(unsafe { self.hbuf.f32p().add(i * d) as *const f32 })).collect();
        Ok(StepOut { rows: vec![rows], h: vec![h] })
    }

    fn commit(&mut self, _seq: SeqId, pos0: usize, n_keep: usize) -> Result<(), String> {
        match self.pending.take() {
            Some((p, t)) => {
                if p != pos0 || n_keep < 1 || n_keep > t {
                    return Err(format!("commit が確かめと合わない（{p}/{t} と {pos0}/{n_keep}）"));
                }
                let t0 = std::time::Instant::now();
                self.m.commit_verify(&mut self.seq, t, n_keep);
                self.commit_ms += t0.elapsed().as_secs_f64() * 1e3;
                if self.acc_hist.len() < n_keep {
                    self.acc_hist.resize(n_keep, 0);
                }
                self.acc_hist[n_keep - 1] += 1;
                Ok(())
            }
            None => {
                if self.seq.pos != pos0 + n_keep {
                    return Err(format!("commit: 確かめなしの step の後は全部受け取るはず（{pos0}+{n_keep} ≠ {}）", self.seq.pos));
                }
                Ok(())
            }
        }
    }

    fn free(&mut self, _seq: SeqId) {
        self.reset();
    }
}
