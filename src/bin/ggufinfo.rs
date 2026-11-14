//! GGUF（分割された束はまとめて）の概要を出し、目次を確かめる。重みの本体は読まない（目次だけ）。
//!
//!   ggufinfo <gguf> [-v] [--list] [--index <目次.json>]... [--meta <目次.json> [--prefix <p>]]
//!
//! いつも確かめる: テンソルごとに「型と形からの大きさ」＝「次のテンソルまでの幅」（最後は束の終わりまで）、
//!   自前のカーネルの行の大きさ（qtype_row_bytes）＝ 型の表。
//! -v       メタデータを全部と、層ごとの数・大きさ
//! --list   テンソルを全部（束・位置・型・形・大きさ）
//! --index  tools/gguf_remote.py の目次（JSON）と、url の名前が同じ束を突き合わせる
//!          （版・データの頭・テンソル全部の名前・形・型・位置・大きさ・メタデータ）
//! --meta   目次のメタデータのうち --prefix で始まるもの（既定は全部）が、このモデル（最初の束）と同じか
//! NG が 1 つでもあれば終わりの値は 1。
use tachyon_v100::gguf::{ggml_type_name, Model, Value};
use std::collections::BTreeMap;

fn gb(b: usize) -> f64 {
    b as f64 / 1e9
}

// ---------------------------------------------------------------- 小さな JSON の読み（目次を読むだけ）

#[derive(Debug)]
enum J {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    fn get(&self, k: &str) -> Option<&J> {
        match self {
            J::Obj(o) => o.iter().find(|(a, _)| a == k).map(|(_, v)| v),
            _ => None,
        }
    }
    fn int(&self) -> Option<i128> {
        match self {
            J::Num(n) => n.parse().ok(),
            _ => None,
        }
    }
    fn str(&self) -> Option<&str> {
        match self {
            J::Str(s) => Some(s),
            _ => None,
        }
    }
}

struct P<'a> {
    s: &'a [u8],
    i: usize,
}

impl P<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn eat(&mut self, c: u8) {
        self.ws();
        assert_eq!(self.s.get(self.i), Some(&c), "JSON: {} バイト目に {:?} が要る", self.i, c as char);
        self.i += 1;
    }
    fn hex4(&mut self) -> u32 {
        let h = std::str::from_utf8(&self.s[self.i..self.i + 4]).unwrap();
        self.i += 4;
        u32::from_str_radix(h, 16).expect("JSON: \\u の後が 16 進でない")
    }
    fn string(&mut self) -> String {
        self.eat(b'"');
        let mut out = Vec::new();
        loop {
            let c = self.s[self.i];
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = self.s[self.i];
                    self.i += 1;
                    let ch = match e {
                        b'n' => '\n',
                        b't' => '\t',
                        b'r' => '\r',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'u' => {
                            let mut u = self.hex4();
                            if (0xd800..0xdc00).contains(&u) && self.s[self.i..].starts_with(b"\\u") {
                                self.i += 2;
                                let lo = self.hex4();
                                u = if (0xdc00..0xe000).contains(&lo) { 0x10000 + ((u - 0xd800) << 10) + (lo - 0xdc00) } else { 0xfffd };
                            }
                            char::from_u32(u).unwrap_or('\u{fffd}')
                        }
                        c => c as char,
                    };
                    let mut b = [0u8; 4];
                    out.extend(ch.encode_utf8(&mut b).as_bytes());
                }
                c => out.push(c),
            }
        }
        String::from_utf8(out).expect("JSON: UTF-8 でない")
    }
    fn value(&mut self) -> J {
        self.ws();
        match self.s[self.i] {
            b'{' => {
                self.i += 1;
                let mut o = vec![];
                self.ws();
                if self.s[self.i] == b'}' {
                    self.i += 1;
                    return J::Obj(o);
                }
                loop {
                    let k = self.string();
                    self.eat(b':');
                    o.push((k, self.value()));
                    self.ws();
                    self.i += 1;
                    match self.s[self.i - 1] {
                        b'}' => return J::Obj(o),
                        b',' => {}
                        c => panic!("JSON: {} バイト目に , か }} が要るのに {:?}", self.i - 1, c as char),
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut a = vec![];
                self.ws();
                if self.s[self.i] == b']' {
                    self.i += 1;
                    return J::Arr(a);
                }
                loop {
                    a.push(self.value());
                    self.ws();
                    self.i += 1;
                    match self.s[self.i - 1] {
                        b']' => return J::Arr(a),
                        b',' => {}
                        c => panic!("JSON: {} バイト目に , か ] が要るのに {:?}", self.i - 1, c as char),
                    }
                }
            }
            b'"' => J::Str(self.string()),
            _ => {
                let st = self.i;
                while self.i < self.s.len() && !b",]} \n\r\t".contains(&self.s[self.i]) {
                    self.i += 1;
                }
                match std::str::from_utf8(&self.s[st..self.i]).unwrap() {
                    "null" => J::Null,
                    "true" => J::Bool(true),
                    "false" => J::Bool(false),
                    n => J::Num(n.to_owned()),
                }
            }
        }
    }
}

fn read_json(path: &str) -> J {
    let s = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    P { s: &s, i: 0 }.value()
}

/// GGUF の値と目次の値が同じか。長い配列は目次では {"len": n, "head": [先頭 8 つ]}
fn same(v: &Value, j: &J) -> bool {
    match (v, j) {
        (Value::Arr(a), J::Arr(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same(x, y)),
        (Value::Arr(a), J::Obj(_)) => {
            let (Some(n), Some(J::Arr(h))) = (j.get("len").and_then(J::int), j.get("head")) else { return false };
            n == a.len() as i128 && h.len() <= a.len() && a.iter().zip(h).all(|(x, y)| same(x, y))
        }
        (Value::Str(s), J::Str(t)) => s == t,
        (Value::Bool(b), J::Bool(c)) => b == c,
        // f32 は python の float（f64）で書かれている → f32 に戻して同じ値か
        (Value::F32(x), J::Num(n)) => n.parse::<f64>().map_or(false, |y| y as f32 == *x),
        (Value::F64(x), J::Num(n)) => n.parse::<f64>().map_or(false, |y| y == *x),
        (_, J::Num(n)) => v.as_int().is_some() && n.parse::<i128>().ok() == v.as_int(),
        _ => false,
    }
}

fn jshow(j: &J) -> String {
    let s = format!("{j:?}");
    if s.chars().count() > 120 {
        s.chars().take(120).collect::<String>() + "…"
    } else {
        s
    }
}

// ---------------------------------------------------------------- 確かめ

/// テンソルごとに 大きさ ＝ 次までの幅。(一致, 詰め物つき, NG)
fn check_spans(m: &Model) -> (usize, usize, usize) {
    let (mut ok, mut pad, mut ng) = (0, 0, 0);
    for sh in &m.shards {
        let mut ts: Vec<_> = m.tensors[sh.tensors.clone()].iter().collect();
        ts.sort_by_key(|t| t.offset);
        let end = sh.data().len();
        for (i, t) in ts.iter().enumerate() {
            let next = ts.get(i + 1).map_or(end, |n| n.offset);
            let span = next - t.offset;
            let padded = (t.size + sh.alignment - 1) / sh.alignment * sh.alignment;
            if span == t.size {
                ok += 1;
            } else if span > t.size && span <= padded {
                pad += 1;
            } else {
                ng += 1;
                eprintln!(
                    "NG 大きさ {} {} {:?}: 型と形から {} バイト、次まで {} バイト（束 {}）",
                    t.name,
                    ggml_type_name(t.ggml_type),
                    t.shape,
                    t.size,
                    span,
                    t.shard + 1
                );
            }
        }
    }
    (ok, pad, ng)
}

/// 自前のカーネルの行の大きさ（kernels/quant.cuh）と型の表が同じか。(型の数, NG)
fn check_kernel_rows(m: &Model) -> (usize, usize) {
    let mut seen = BTreeMap::new();
    for t in &m.tensors {
        if let Some(q) = t.qtype {
            seen.entry(q as i32).or_insert((q, t));
        }
    }
    let mut ng = 0;
    for (q, t) in seen.values() {
        let v = m.view(t);
        let kern = q.row_bytes(v.k());
        if kern != v.row_bytes() {
            ng += 1;
            eprintln!("NG 行の大きさ {:?} k={}: カーネル {kern}、表 {}", q, v.k(), v.row_bytes());
        }
    }
    (seen.len(), ng)
}

/// --index: 目次 1 つと、同じ名前の束を突き合わせる。NG の数
fn check_index(m: &Model, path: &str) -> usize {
    let j = read_json(path);
    let url = j.get("url").and_then(J::str).unwrap_or("");
    let base = url.rsplit('/').next().unwrap_or("");
    let Some((si, sh)) = m.shards.iter().enumerate().find(|(_, s)| s.path.file_name().and_then(|f| f.to_str()) == Some(base)) else {
        eprintln!("NG {path}: url の名前 {base:?} の束が無い");
        return 1;
    };
    let mut ng = 0;
    let mut bad = |what: String| {
        ng += 1;
        if ng <= 10 {
            eprintln!("NG {path}: {what}");
        }
    };
    if j.get("version").and_then(J::int) != Some(sh.version as i128) {
        bad(format!("版 {:?} ≠ {}", j.get("version"), sh.version));
    }
    if j.get("data_start").and_then(J::int) != Some(sh.data_start as i128) {
        bad(format!("データの頭 {:?} ≠ {}", j.get("data_start"), sh.data_start));
    }
    let empty = vec![];
    let jt = match j.get("tensors") {
        Some(J::Arr(a)) => a,
        _ => &empty,
    };
    let ours = &m.tensors[sh.tensors.clone()];
    if jt.len() != ours.len() {
        bad(format!("テンソルの数 {} ≠ {}", jt.len(), ours.len()));
    }
    let (mut t_ok, mut jsum) = (0, 0i128);
    for (a, t) in jt.iter().zip(ours) {
        let b = a.get("bytes").and_then(J::int).unwrap_or(-1);
        jsum += b.max(0);
        let shape: Option<Vec<i128>> = match a.get("shape") {
            Some(J::Arr(s)) => s.iter().map(J::int).collect(),
            _ => None,
        };
        let same_t = a.get("name").and_then(J::str) == Some(t.name.as_str())
            && shape == Some(t.shape.iter().map(|&d| d as i128).collect())
            && a.get("type").and_then(J::str) == Some(ggml_type_name(t.ggml_type).as_str())
            && a.get("offset").and_then(J::int) == Some(t.offset as i128)
            && b == t.size as i128;
        if same_t {
            t_ok += 1;
        } else {
            bad(format!("テンソル {}: 目次 {} ≠ ここ {:?} {} 位置 {} 大きさ {}", t.name, jshow(a), t.shape, ggml_type_name(t.ggml_type), t.offset, t.size));
        }
    }
    let ours_sum: usize = ours.iter().map(|t| t.size).sum();
    if jsum != ours_sum as i128 {
        bad(format!("合計 {jsum} ≠ {ours_sum}"));
    }
    let (mut m_ok, mut m_n) = (0, 0);
    if let Some(J::Obj(meta)) = j.get("meta") {
        for (k, jv) in meta {
            m_n += 1;
            match sh.meta(k) {
                Some(v) if same(v, jv) => m_ok += 1,
                Some(v) => bad(format!("メタ {k}: 目次 {} ≠ ここ {v}", jshow(jv))),
                None => bad(format!("メタ {k}: ここに無い")),
            }
        }
        for (k, _) in &sh.meta {
            if !meta.iter().any(|(a, _)| a == k) {
                bad(format!("メタ {k}: 目次に無い"));
            }
        }
    }
    println!(
        "目次 {path} ↔ 束 {}/{}: テンソル {t_ok}/{} 一致、合計 {jsum} ＝ {ours_sum}{}、メタ {m_ok}/{m_n} 一致{}",
        si + 1,
        m.shards.len(),
        jt.len(),
        if jsum == ours_sum as i128 { "" } else { " ⛔" },
        if ng == 0 { "" } else { "  ⛔ NG あり" }
    );
    ng
}

/// --meta: 目次のメタデータ（prefix で始まるもの）がこのモデルと同じか。NG の数
fn check_meta(m: &Model, path: &str, prefix: &str) -> usize {
    let j = read_json(path);
    let Some(J::Obj(meta)) = j.get("meta") else {
        eprintln!("NG {path}: meta が無い");
        return 1;
    };
    let (mut ok, mut n, mut ng) = (0, 0, 0);
    for (k, jv) in meta.iter().filter(|(k, _)| k.starts_with(prefix)) {
        n += 1;
        match m.meta(k) {
            Some(v) if same(v, jv) => ok += 1,
            Some(v) => {
                ng += 1;
                eprintln!("NG メタ {k}: 目次 {} ≠ ここ {v}", jshow(jv));
            }
            None => {
                ng += 1;
                eprintln!("NG メタ {k}: ここに無い");
            }
        }
    }
    if n == 0 {
        ng += 1;
        eprintln!("NG {path}: {prefix:?} で始まるメタが 1 つも無い");
    }
    println!("メタ {path} の {prefix}*: {n} 個のうち {ok} 個一致{}", if ng == 0 { "" } else { "  ⛔ NG あり" });
    ng
}

// ---------------------------------------------------------------- 出す

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut path, mut verbose, mut list) = (None, false, false);
    let (mut indexes, mut meta_json, mut prefix) = (vec![], None, String::new());
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-v" => verbose = true,
            "--list" => list = true,
            "--index" => indexes.push(it.next().expect("--index の後に目次").clone()),
            "--meta" => meta_json = Some(it.next().expect("--meta の後に目次").clone()),
            "--prefix" => prefix = it.next().expect("--prefix の後に頭").clone(),
            p if path.is_none() && !p.starts_with('-') => path = Some(p.to_owned()),
            p => panic!("知らない引数 {p}"),
        }
    }
    let path = path.expect("使い方: ggufinfo <gguf> [-v] [--list] [--index 目次.json]... [--meta 目次.json [--prefix p]]");
    let t0 = std::time::Instant::now();
    let m = Model::open(&path).unwrap_or_else(|e| {
        eprintln!("⛔ {e}");
        std::process::exit(1)
    });
    let open_ms = t0.elapsed().as_secs_f64() * 1e3;

    println!("{path}（開くのに {open_ms:.0} ms）");
    println!("束 {} つ:", m.shards.len());
    for (i, sh) in m.shards.iter().enumerate() {
        let n = sh.tensors.len();
        let b: usize = m.tensors[sh.tensors.clone()].iter().map(|t| t.size).sum();
        println!(
            "  {}/{} 版 {} データの頭 {:>9} テンソル {:>5} 個 {:>12} バイト（{:.3} GB） ファイル {} バイト  {}",
            i + 1,
            m.shards.len(),
            sh.version,
            sh.data_start,
            n,
            b,
            gb(b),
            sh.file_len(),
            sh.path.file_name().unwrap().to_string_lossy()
        );
    }
    println!(
        "arch {}  名前 {:?}  層 {}  メタデータ {} 個",
        m.arch(),
        m.meta_str("general.name").unwrap_or(""),
        m.n_layers(),
        m.meta_all().len()
    );
    println!("テンソル {} 個  合計 {} バイト（{:.3} GB）", m.tensors.len(), m.total_bytes(), gb(m.total_bytes()));
    let mut by_type: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for t in &m.tensors {
        let e = by_type.entry(ggml_type_name(t.ggml_type)).or_default();
        e.0 += 1;
        e.1 += t.size;
    }
    let mut by_type: Vec<_> = by_type.into_iter().collect();
    by_type.sort_by_key(|(_, (_, b))| std::cmp::Reverse(*b));
    println!("型ごと:");
    for (name, (n, b)) in &by_type {
        let known = if tachyon_v100::qtype::QType::from_name(name).is_some() { "" } else { "（カーネルなし）" };
        println!("  {name:8} {n:>5} 個 {:>9.3} GB{known}", gb(*b));
    }

    if verbose {
        println!("メタデータ:");
        for (k, v) in m.meta_all() {
            println!("  {k} [{}] = {v}", v.type_name());
        }
        for (i, sh) in m.shards.iter().enumerate().skip(1) {
            let kv: Vec<String> = sh.meta.iter().map(|(k, v)| format!("{k}={v}")).collect();
            println!("  （束 {} のメタ: {}）", i + 1, kv.join(" "));
        }
        println!("層ごと:");
        for l in 0..m.n_layers() {
            let ly = m.layer(l);
            let n = ly.tensors().count();
            if n > 0 {
                let kind = if ly.has("attn_q.weight") { "普通の Attention" } else if ly.has("ssm_out.weight") { "線形" } else { "" };
                println!("  層 {l:>2}: {n:>3} 個 {:>7.3} GB {kind}", gb(ly.bytes()));
            }
        }
        let other: usize = m.tensors.iter().filter(|t| !t.name.starts_with("blk.")).map(|t| t.size).sum();
        println!("  層の外: {:.3} GB", gb(other));
    }
    if list {
        println!("テンソル（束 位置 型 形 大きさ 名前）:");
        for t in &m.tensors {
            println!("  {} {:>12} {:8} {:<22} {:>12} {}", t.shard + 1, t.offset, ggml_type_name(t.ggml_type), format!("{:?}", t.shape), t.size, t.name);
        }
    }

    let mut ng = 0;
    let (ok, pad, bad) = check_spans(&m);
    ng += bad;
    println!("大きさの確かめ（型と形 ＝ 次までの幅）: {} 個のうち 一致 {ok}・詰め物つき {pad}・NG {bad}", m.tensors.len());
    let (nt, bad) = check_kernel_rows(&m);
    ng += bad;
    println!("カーネルの行の大きさ（qtype_row_bytes ＝ 表）: {nt} 型のうち NG {bad}");
    for p in &indexes {
        ng += check_index(&m, p);
    }
    if let Some(p) = &meta_json {
        ng += check_meta(&m, p, &prefix);
    }
    println!("{}", if ng == 0 { "全部 OK" } else { "⛔ NG あり" });
    std::process::exit(if ng == 0 { 0 } else { 1 });
}
