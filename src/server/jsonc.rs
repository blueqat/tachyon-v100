//! response_format の JSON の形で語を絞る（json_object と json_schema の一部）。
//!
//! 形（Schema）は依頼ごとに 1 回だけ作り、列ごとに Matcher（バイトで進む押し下げの機械）を持つ。
//!   - 語（token）を足せるか: その語のバイト列を写しの機械に流して、途中で外れないか（`allows`）。特別な語は外す。
//!     止める語（<|im_end|> など）は JSON が閉じた後だけ。考えの中（<think> … </think>）は何でも通す（`</think>` で絞り始める）
//!   - 空白は値の前後に 24 バイトまで続けてよい（llama.cpp の json の文法の `space` と同じ程度。空白だけで流れ続けない）
//!   - 対応する JSON Schema の語: type（文字・配列）・properties・required・additionalProperties（bool か形）・items・enum・const、
//!     飾りの description・title・default・examples・$schema・$comment・$id。⛔ それ以外（anyOf・$ref・pattern・format・minimum …）は 400
//!   - 物の鍵の順は llama.cpp の json-schema-to-grammar と同じ: required を properties の順に全部 → 残りの properties を順に（飛ばしてよい）。
//!     additionalProperties を書いていない物は、余分な鍵を許さない（llama.cpp と同じ）。true・形なら鍵は自由（順も自由・required は全部出るまで閉じない）
//!   - 数: JSON の数の形（integer は小数・指数なし）。文字: 制御文字なし・\ の後は決まった字と \uXXXX
use serde_json::Value as Json;
use std::sync::Arc;

const S: u8 = 1; // string
const I: u8 = 2; // integer
const N: u8 = 4; // number
const B: u8 = 8; // boolean
const Z: u8 = 16; // null
const O: u8 = 32; // object
const A: u8 = 64; // array
const ALL: u8 = S | I | N | B | Z | O | A;
const WS_MAX: u8 = 24;

/// 語の表（id → バイト列・特別な語・止める語）。口の起動で 1 回作る
pub struct TokTable {
    pub bytes: Vec<Box<[u8]>>,
    pub special: Vec<bool>,
    pub stops: Vec<u32>,
    pub think_end: u32,
    /// 選んでよい語の番号の上限（GGUF の余りの [PAD…] は外す）
    pub valid: usize,
}

impl TokTable {
    pub fn new(tok: &crate::tokenizer::Tokenizer, stops: &[u32]) -> Self {
        let n = tok.vocab_size();
        let mut bytes = Vec::with_capacity(n);
        let mut special = Vec::with_capacity(n);
        for id in 0..n as u32 {
            bytes.push(tok.token_bytes(id).to_vec().into_boxed_slice());
            special.push(tok.is_special(id));
        }
        TokTable { bytes, special, stops: stops.to_vec(), think_end: tok.ids.think_end, valid: n }
    }
}

#[derive(Debug, Clone)]
struct Obj {
    /// properties の順（鍵の JSON の字＝引用符込み・名前・形）
    keys: Vec<Vec<u8>>,
    names: Vec<String>,
    nodes: Vec<u32>,
    required: Vec<bool>,
    /// 余分な鍵の形（None = 許さない）
    additional: Option<u32>,
    /// 鍵の順（順の決まった物）: required を properties の順に → 残りを properties の順に。n_req = required の数
    order: Vec<u16>,
    n_req: usize,
}

#[derive(Debug, Clone)]
struct Node {
    types: u8,
    obj: Option<Obj>,
    items: Option<u32>,
    /// enum・const（JSON の字）。あればこのどれか
    lits: Option<Vec<Vec<u8>>>,
}

/// 依頼の形（作った後は変えない）
#[derive(Debug)]
pub struct Schema {
    nodes: Vec<Node>,
    root: u32,
}

const DECOR: &[&str] = &["description", "title", "default", "examples", "$schema", "$comment", "$id", "deprecated", "readOnly", "writeOnly"];
const KNOWN: &[&str] = &["type", "properties", "required", "additionalProperties", "items", "enum", "const"];

impl Schema {
    /// json_object（中身は何でもよい物）
    pub fn any_object() -> Self {
        let mut s = Schema { nodes: vec![], root: 0 };
        let any = s.any();
        s.root = s.push(Node { types: O, obj: Some(Obj { keys: vec![], names: vec![], nodes: vec![], required: vec![], additional: Some(any), order: vec![], n_req: 0 }), items: None, lits: None });
        s
    }

    fn push(&mut self, n: Node) -> u32 {
        self.nodes.push(n);
        (self.nodes.len() - 1) as u32
    }

    /// 何でもよい値（物の鍵は自由・配列の中も何でも）
    fn any(&mut self) -> u32 {
        let i = self.nodes.len() as u32;
        self.nodes.push(Node { types: ALL, obj: Some(Obj { keys: vec![], names: vec![], nodes: vec![], required: vec![], additional: Some(i), order: vec![], n_req: 0 }), items: Some(i), lits: None });
        i
    }

    /// JSON Schema（対応する語だけ）から。誤りは利用者に見せる文（param の場所つき）
    pub fn compile(schema: &Json) -> Result<Self, String> {
        Self::compile_at(schema, "schema")
    }

    /// at: 誤りの文に出す場所の名前
    pub fn compile_at(schema: &Json, at: &str) -> Result<Self, String> {
        let mut s = Schema { nodes: vec![], root: 0 };
        s.root = s.node(schema, at, 0)?;
        Ok(s)
    }

    fn node(&mut self, v: &Json, at: &str, depth: usize) -> Result<u32, String> {
        if depth > 32 {
            return Err(format!("{at}: the schema is nested too deeply (32 levels at most)"));
        }
        let o = match v {
            Json::Bool(true) => return Ok(self.any()),
            Json::Object(o) => o,
            _ => return Err(format!("{at}: a schema must be an object (or true)")),
        };
        for k in o.keys() {
            if !KNOWN.contains(&k.as_str()) && !DECOR.contains(&k.as_str()) {
                return Err(format!(
                    "{at}: the JSON Schema keyword {k:?} is not supported (supported: type, properties, required, additionalProperties, items, enum, const)"
                ));
            }
        }
        let lits = match (o.get("enum"), o.get("const")) {
            (Some(_), Some(_)) => return Err(format!("{at}: use either enum or const, not both")),
            (Some(Json::Array(a)), None) if !a.is_empty() => Some(a.iter().map(|x| serde_json::to_vec(x).unwrap()).collect::<Vec<_>>()),
            (Some(_), None) => return Err(format!("{at}.enum: must be a non-empty array")),
            (None, Some(c)) => Some(vec![serde_json::to_vec(c).unwrap()]),
            (None, None) => None,
        };
        let mut types = match o.get("type") {
            None => 0,
            Some(Json::String(t)) => type_bit(t).ok_or_else(|| format!("{at}.type: unknown type {t:?}"))?,
            Some(Json::Array(a)) if !a.is_empty() => {
                let mut m = 0;
                for t in a {
                    let t = t.as_str().ok_or_else(|| format!("{at}.type: must be strings"))?;
                    m |= type_bit(t).ok_or_else(|| format!("{at}.type: unknown type {t:?}"))?;
                }
                m
            }
            Some(_) => return Err(format!("{at}.type: must be a string or a non-empty array of strings")),
        };
        if types == 0 {
            types = if o.contains_key("properties") || o.contains_key("additionalProperties") || o.contains_key("required") {
                O
            } else if o.contains_key("items") {
                A
            } else {
                ALL
            };
        }
        if types & I != 0 && types & N != 0 {
            types &= !I; // number が integer を含む
        }
        let me = self.push(Node { types, obj: None, items: None, lits: None });
        if let Some(l) = lits {
            self.nodes[me as usize].lits = Some(l);
            return Ok(me);
        }
        if types & O != 0 {
            let mut ob = Obj { keys: vec![], names: vec![], nodes: vec![], required: vec![], additional: None, order: vec![], n_req: 0 };
            match o.get("properties") {
                None => {}
                Some(Json::Object(p)) => {
                    for (k, sv) in p {
                        let n = self.node(sv, &format!("{at}.properties.{k}"), depth + 1)?;
                        ob.keys.push(serde_json::to_vec(&Json::String(k.clone())).unwrap());
                        ob.names.push(k.clone());
                        ob.nodes.push(n);
                        ob.required.push(false);
                    }
                }
                Some(_) => return Err(format!("{at}.properties: must be an object")),
            }
            match o.get("required") {
                None => {}
                Some(Json::Array(r)) => {
                    for x in r {
                        let x = x.as_str().ok_or_else(|| format!("{at}.required: must be an array of strings"))?;
                        match ob.names.iter().position(|n| n == x) {
                            Some(i) => ob.required[i] = true,
                            None => return Err(format!("{at}.required: {x:?} is not in properties (not supported)")),
                        }
                    }
                }
                Some(_) => return Err(format!("{at}.required: must be an array of strings")),
            }
            ob.additional = match o.get("additionalProperties") {
                None | Some(Json::Bool(false)) => {
                    if ob.names.is_empty() && o.get("additionalProperties").is_none() {
                        Some(self.any()) // {"type":"object"} だけ: 中身は自由
                    } else {
                        None
                    }
                }
                Some(Json::Bool(true)) => Some(self.any()),
                Some(x @ Json::Object(_)) => Some(self.node(x, &format!("{at}.additionalProperties"), depth + 1)?),
                Some(_) => return Err(format!("{at}.additionalProperties: must be a boolean or a schema")),
            };
            ob.order = (0..ob.keys.len()).filter(|&i| ob.required[i]).chain((0..ob.keys.len()).filter(|&i| !ob.required[i])).map(|i| i as u16).collect();
            ob.n_req = ob.required.iter().filter(|&&r| r).count();
            self.nodes[me as usize].obj = Some(ob);
        }
        if types & A != 0 {
            let it = match o.get("items") {
                None => self.any(),
                Some(x @ (Json::Object(_) | Json::Bool(true))) => self.node(x, &format!("{at}.items"), depth + 1)?,
                Some(_) => return Err(format!("{at}.items: must be a schema (tuple items are not supported)")),
            };
            self.nodes[me as usize].items = Some(it);
        }
        Ok(me)
    }
}

fn type_bit(t: &str) -> Option<u8> {
    Some(match t {
        "string" => S,
        "integer" => I,
        "number" => N,
        "boolean" => B,
        "null" => Z,
        "object" => O,
        "array" => A,
        _ => return None,
    })
}

/// response_format から形を作る。None は普通の文（"text"・無し）。Err は 400 の文
pub fn from_response_format(rf: Option<&Json>) -> Result<Option<Arc<Schema>>, (String, String)> {
    let rf = match rf {
        None | Some(Json::Null) => return Ok(None),
        Some(x) => x,
    };
    let bad = |p: &str, m: String| (p.to_string(), m);
    let t = rf.get("type").and_then(Json::as_str).ok_or_else(|| bad("response_format.type", "response_format.type must be \"text\", \"json_object\" or \"json_schema\"".into()))?;
    match t {
        "text" => Ok(None),
        "json_object" => Ok(Some(Arc::new(Schema::any_object()))),
        "json_schema" => {
            let js = rf.get("json_schema").ok_or_else(|| bad("response_format.json_schema", "response_format.json_schema is required".into()))?;
            let sc = match js.get("schema") {
                None | Some(Json::Null) => return Ok(Some(Arc::new(Schema::any_object()))),
                Some(x) => x,
            };
            Schema::compile_at(sc, "response_format.json_schema.schema").map(|s| Some(Arc::new(s))).map_err(|m| bad("response_format.json_schema.schema", m))
        }
        other => Err(bad("response_format.type", format!("response_format.type {other:?} is not supported (\"text\", \"json_object\" or \"json_schema\")"))),
    }
}

// ---------------------------------------------------------------- 機械

#[derive(Clone, Debug)]
enum LitSrc {
    /// node の lits
    Enum(u32),
    /// 物 node の鍵（引用符込み）
    Key(u32),
    True,
    False,
    Null,
}

#[derive(Clone, Debug)]
enum F {
    /// 値を待つ（前の空白は可）
    Val(u32),
    /// 決まった字のどれか（cands = 候補の番号・off = 何バイト目まで合ったか）
    Lit { src: LitSrc, cands: Vec<u16>, off: u16 },
    /// 自由な文字列（開きの " の後）。esc: 0 なし・1 \ の後・2..=5 \u の残りの 16 進の数。key: 物の鍵（バイトを親に渡す）
    Str { esc: u8, key: bool },
    /// 数。st: 0 頭（- の後か最初）・1 0 だけ・2 整数の桁・3 . の後・4 小数の桁・5 e の後・6 e の符号の後・7 指数の桁
    Num { st: u8, int: bool },
    /// 物。ph: 0 { の後・2 鍵の後（: を待つ）・4 値の後・5 , の後。req: 次に出す required の番号（順の物）・opt: 次に出せる鍵の番号の下限
    Obj { n: u32, ph: u8, next: u16, cur: u32, keybuf: Vec<u8>, seen: Vec<bool> },
    /// 配列。ph: 0 [ の後・4 値の後・5 , の後
    Arr { n: u32, ph: u8 },
}

impl std::fmt::Debug for Matcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Matcher {{ depth: {}, think: {}, done: {} }}", self.stack.len(), self.pre, self.done)
    }
}

/// 列ごとの状態（安く写せる）
#[derive(Clone)]
pub struct Matcher {
    schema: Arc<Schema>,
    tab: Arc<TokTable>,
    stack: Vec<F>,
    /// 考えの中（</think> が出るまで絞らない）
    pre: bool,
    /// 根の値が閉じた
    done: bool,
    ws: u8,
}

enum Act {
    Ok,
    Reject,
    /// この字は使った・frame は閉じた（親へ知らせる）
    DoneConsumed,
    /// frame は閉じた・この字は使っていない（親でもう一度）
    DoneReprocess,
    Push(F),
    /// この字を使い、frame を差し替える
    Replace(F),
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\n' | b'\t' | b'\r')
}

impl Matcher {
    pub fn new(schema: Arc<Schema>, tab: Arc<TokTable>, in_think: bool) -> Self {
        let root = schema.root;
        Matcher { schema, tab, stack: vec![F::Val(root)], pre: in_think, done: false, ws: 0 }
    }

    pub fn in_think(&self) -> bool {
        self.pre
    }

    /// 根の値が閉じたか（止める語を出してよいか）
    pub fn complete(&self) -> bool {
        if self.pre {
            return true;
        }
        self.done || (self.stack.len() == 1 && matches!(self.stack[0], F::Num { st, .. } if num_complete(st)))
    }

    /// この語を次に出してよいか
    pub fn allows(&self, t: u32) -> bool {
        if self.pre {
            return true;
        }
        let tab = &*self.tab;
        if tab.stops.contains(&t) {
            return self.complete();
        }
        let i = t as usize;
        if i >= tab.valid || tab.special[i] {
            return false;
        }
        let bs = &tab.bytes[i];
        if bs.is_empty() || self.done {
            return false;
        }
        if self.in_plain_string() && plain(bs) {
            return true;
        }
        let mut m = self.clone_state();
        bs.iter().all(|&b| m.feed(b))
    }

    /// 文字列の中（\ の後でない）: 引用符・\・制御文字を含まない語はそのまま通る（写しを作らない近道）
    fn in_plain_string(&self) -> bool {
        matches!(self.stack.last(), Some(F::Str { esc: 0, .. }))
    }

    /// 語を出した（allows が true の語だけ）。外れなら false
    pub fn advance(&mut self, t: u32) -> bool {
        if self.pre {
            if t == self.tab.think_end {
                self.pre = false;
            }
            return true;
        }
        if self.tab.stops.contains(&t) {
            return self.complete();
        }
        let i = t as usize;
        if i >= self.tab.valid || self.tab.special[i] || self.done {
            return false;
        }
        let tab = self.tab.clone();
        tab.bytes[i].iter().all(|&b| self.feed(b))
    }

    /// logits の全部の語に絞りをかける（出せない語は −∞）。候補で決められない時の道（遅い・まれ）
    pub fn mask_logits(&self, l: &mut [f32]) {
        if self.pre {
            return;
        }
        // 最初のバイトで先に絞る（多くの語は 1 バイト目で外れる）
        let mut first = [false; 256];
        if !self.done {
            for b in 0..=255u8 {
                let mut m = self.clone_state();
                first[b as usize] = m.feed(b);
            }
        }
        let tab = &*self.tab;
        let in_str = !self.done && self.in_plain_string();
        let mut two: std::collections::HashMap<usize, bool> = std::collections::HashMap::new();
        for (i, x) in l.iter_mut().enumerate() {
            let ok = if i >= tab.valid {
                false
            } else if tab.stops.contains(&(i as u32)) {
                self.complete()
            } else if tab.special[i] || tab.bytes[i].is_empty() || !first[tab.bytes[i][0] as usize] {
                false
            } else if in_str && plain(&tab.bytes[i]) {
                true
            } else {
                let bs = &tab.bytes[i];
                // 2 バイト目までで外れる語は写しを作らない（頭 2 バイトの答えを覚える）
                let ok2 = if bs.len() >= 2 {
                    let key = (bs[0] as usize) << 8 | bs[1] as usize;
                    *two.entry(key).or_insert_with(|| {
                        let mut m = self.clone_state();
                        m.feed(bs[0]) && m.feed(bs[1])
                    })
                } else {
                    true
                };
                ok2 && {
                    let mut m = self.clone_state();
                    bs.iter().all(|&b| m.feed(b))
                }
            };
            if !ok {
                *x = f32::NEG_INFINITY;
            }
        }
    }

    fn clone_state(&self) -> Matcher {
        self.clone()
    }

    fn node(&self, n: u32) -> &Node {
        &self.schema.nodes[n as usize]
    }

    /// 1 バイト進める。外れなら false（状態は壊れてよい＝写しで試す）
    fn feed(&mut self, b: u8) -> bool {
        if self.done {
            return false;
        }
        loop {
            let Some(top) = self.stack.last() else { return false };
            let act = self.step(top.clone(), b);
            match act {
                Act::Reject => return false,
                Act::Ok => return true,
                Act::Push(f) => {
                    self.stack.push(f);
                    return true;
                }
                Act::Replace(f) => {
                    *self.stack.last_mut().unwrap() = f;
                    return true;
                }
                Act::DoneConsumed => {
                    let f = self.stack.pop().unwrap();
                    if !self.child_done(f) {
                        return false;
                    }
                    return true;
                }
                Act::DoneReprocess => {
                    let f = self.stack.pop().unwrap();
                    if !self.child_done(f) {
                        return false;
                    }
                    if self.done {
                        return false; // 根の後に字は来ない
                    }
                    // 親でもう一度
                }
            }
        }
    }

    /// 子の frame が閉じた。親の段を進める
    fn child_done(&mut self, child: F) -> bool {
        let Some(parent) = self.stack.last_mut() else {
            self.done = true;
            return true;
        };
        match parent {
            F::Val(_) => {
                // Val は子（値の中身）に差し替わるので、ここには来ない
                false
            }
            F::Obj { n, ph, next, cur, keybuf, seen } => {
                let n = *n;
                match *ph {
                    0 | 5 => {
                        // 鍵が閉じた
                        let ob = self.schema.nodes[n as usize].obj.as_ref().unwrap();
                        let (k_idx, node) = match &child {
                            F::Lit { cands, .. } => {
                                let k = cands[0] as usize;
                                (Some(k), ob.nodes[k])
                            }
                            _ => {
                                // 自由な鍵: properties の名前なら その形
                                let kb = std::mem::take(keybuf);
                                match ob.names.iter().position(|x| x.as_bytes() == kb.as_slice()) {
                                    Some(k) => (Some(k), ob.nodes[k]),
                                    None => (None, ob.additional.unwrap()),
                                }
                            }
                        };
                        if let Some(k) = k_idx {
                            if !seen.is_empty() {
                                if seen[k] {
                                    return false; // 同じ鍵を 2 回
                                }
                                seen[k] = true;
                            } else {
                                *next = ob.order.iter().position(|&x| x as usize == k).unwrap() as u16 + 1;
                            }
                        }
                        *cur = node;
                        *ph = 2;
                        true
                    }
                    3 => {
                        *ph = 4;
                        true
                    }
                    _ => false,
                }
            }
            F::Arr { ph, .. } => {
                if *ph == 3 {
                    *ph = 4;
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn ws_ok(&mut self) -> bool {
        self.ws += 1;
        self.ws <= WS_MAX
    }

    fn step(&mut self, top: F, b: u8) -> Act {
        match top {
            F::Val(n) => {
                if is_ws(b) {
                    return if self.ws_ok() { Act::Ok } else { Act::Reject };
                }
                self.ws = 0;
                match self.start_value(n, b) {
                    Some(f) => {
                        // Val をこの値の frame に差し替える（1 バイト目は使った）。値が 1 バイトで閉じる物は無い（"" と {} も 2 バイト）
                        Act::Replace(f)
                    }
                    None => Act::Reject,
                }
            }
            F::Lit { src, cands, off } => {
                let off = off as usize;
                let keep: Vec<u16> = cands.iter().copied().filter(|&c| self.lit(&src, c).get(off) == Some(&b)).collect();
                if keep.is_empty() {
                    // 区切りの字: 閉じた候補（enum の数 1 と 10 の 1 など）があれば閉じて、この字を親で
                    let done: Vec<u16> = cands.iter().copied().filter(|&c| self.lit(&src, c).len() == off).collect();
                    if done.is_empty() {
                        return Act::Reject;
                    }
                    *self.stack.last_mut().unwrap() = F::Lit { src, cands: done, off: off as u16 };
                    return Act::DoneReprocess;
                }
                if keep.iter().all(|&c| self.lit(&src, c).len() == off + 1) {
                    *self.stack.last_mut().unwrap() = F::Lit { src, cands: keep, off: (off + 1) as u16 };
                    return Act::DoneConsumed;
                }
                Act::Replace(F::Lit { src, cands: keep, off: (off + 1) as u16 })
            }
            F::Str { esc, key } => {
                let act = match esc {
                    0 => match b {
                        b'"' => return Act::DoneConsumed,
                        b'\\' => Act::Replace(F::Str { esc: 1, key }),
                        0..=0x1f => return Act::Reject,
                        _ => Act::Ok,
                    },
                    1 => match b {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => Act::Replace(F::Str { esc: 0, key }),
                        b'u' => Act::Replace(F::Str { esc: 5, key }),
                        _ => return Act::Reject,
                    },
                    k => {
                        if b.is_ascii_hexdigit() {
                            Act::Replace(F::Str { esc: if k == 2 { 0 } else { k - 1 }, key })
                        } else {
                            return Act::Reject;
                        }
                    }
                };
                if key {
                    // 親の物の鍵の字を控える（properties の名前と比べる）
                    let l = self.stack.len();
                    if l >= 2 {
                        if let F::Obj { keybuf, .. } = &mut self.stack[l - 2] {
                            if keybuf.len() < 4096 {
                                keybuf.push(b);
                            }
                        }
                    }
                }
                act
            }
            F::Num { st, int } => {
                let nst = match (st, b) {
                    (0, b'0') => 1,
                    (0, b'1'..=b'9') => 2,
                    (2, b'0'..=b'9') => 2,
                    (1 | 2, b'.') if !int => 3,
                    (1 | 2 | 4, b'e' | b'E') if !int => 5,
                    (3 | 4, b'0'..=b'9') => 4,
                    (5, b'+' | b'-') => 6,
                    (5 | 6 | 7, b'0'..=b'9') => 7,
                    _ => {
                        return if num_complete(st) { Act::DoneReprocess } else { Act::Reject };
                    }
                };
                // 桁が長すぎる数は外す（数字だけで流れ続けない）
                Act::Replace(F::Num { st: nst, int })
            }
            F::Obj { n, ph, next, cur, keybuf, seen } => {
                let sch = self.schema.clone();
                let ob = sch.nodes[n as usize].obj.as_ref().unwrap();
                let free = ob.additional.is_some();
                if is_ws(b) {
                    return if ph != 3 && self.ws_ok() { Act::Ok } else { Act::Reject };
                }
                self.ws = 0;
                match (ph, b) {
                    (0 | 5, b'"') => {
                        if free {
                            Act::Push(F::Str { esc: 0, key: true })
                        } else {
                            let cands = self.key_cands(ob, next);
                            if cands.is_empty() {
                                return Act::Reject;
                            }
                            Act::Push(F::Lit { src: LitSrc::Key(n), cands, off: 1 })
                        }
                    }
                    (0, b'}') | (4, b'}') => {
                        if self.obj_can_close(ob, next, &seen) {
                            Act::DoneConsumed
                        } else {
                            Act::Reject
                        }
                    }
                    (2, b':') => {
                        *self.stack.last_mut().unwrap() = F::Obj { n, ph: 3, next, cur, keybuf, seen };
                        Act::Push(F::Val(cur))
                    }
                    (4, b',') => {
                        if free || !self.key_cands(ob, next).is_empty() {
                            Act::Replace(F::Obj { n, ph: 5, next, cur, keybuf, seen })
                        } else {
                            Act::Reject
                        }
                    }
                    _ => Act::Reject,
                }
            }
            F::Arr { n, ph } => {
                if is_ws(b) {
                    return if ph != 3 && self.ws_ok() { Act::Ok } else { Act::Reject };
                }
                let item = self.node(n).items.unwrap();
                match (ph, b) {
                    (0 | 4, b']') => {
                        self.ws = 0;
                        Act::DoneConsumed
                    }
                    (4, b',') => {
                        self.ws = 0;
                        Act::Replace(F::Arr { n, ph: 5 })
                    }
                    (0 | 5, _) => {
                        // 値の頭: Val を積んでこのバイトをそこで
                        *self.stack.last_mut().unwrap() = F::Arr { n, ph: 3 };
                        self.stack.push(F::Val(item));
                        let top = self.stack.last().unwrap().clone();
                        match self.step(top, b) {
                            Act::Replace(f) => {
                                *self.stack.last_mut().unwrap() = f;
                                Act::Ok
                            }
                            Act::Reject => Act::Reject,
                            _ => Act::Reject,
                        }
                    }
                    _ => Act::Reject,
                }
            }
        }
    }

    fn lit<'a>(&'a self, src: &LitSrc, c: u16) -> &'a [u8] {
        match src {
            LitSrc::Enum(n) => &self.node(*n).lits.as_ref().unwrap()[c as usize],
            LitSrc::Key(n) => &self.node(*n).obj.as_ref().unwrap().keys[c as usize],
            LitSrc::True => b"true",
            LitSrc::False => b"false",
            LitSrc::Null => b"null",
        }
    }

    /// 順の決まった物で、次に出せる鍵（next = order の何番目まで出したか）: required が残っていればその 1 つ、無ければ残りの optional
    fn key_cands(&self, ob: &Obj, next: u16) -> Vec<u16> {
        let next = next as usize;
        if next < ob.n_req {
            return vec![ob.order[next]];
        }
        ob.order[next.min(ob.order.len())..].to_vec()
    }

    fn obj_can_close(&self, ob: &Obj, next: u16, seen: &[bool]) -> bool {
        if ob.additional.is_some() {
            // 自由な鍵: required が全部出たか
            (0..ob.keys.len()).all(|i| !ob.required[i] || seen.get(i).copied().unwrap_or(false))
        } else {
            next as usize >= ob.n_req
        }
    }

    /// 値の 1 バイト目から frame を作る（使ったバイトの後の状態）
    fn start_value(&self, n: u32, b: u8) -> Option<F> {
        let nd = self.node(n);
        if let Some(l) = &nd.lits {
            let cands: Vec<u16> = (0..l.len() as u16).filter(|&i| l[i as usize].first() == Some(&b)).collect();
            if cands.is_empty() {
                return None;
            }
            // 1 バイトの字（数 0〜9）はすぐ閉じない: 区切りを待つ Lit として 1 バイト目から
            return Some(F::Lit { src: LitSrc::Enum(n), cands, off: 1 });
        }
        let t = nd.types;
        match b {
            b'"' if t & S != 0 => Some(F::Str { esc: 0, key: false }),
            b'{' if t & O != 0 => {
                let ob = nd.obj.as_ref().unwrap();
                let seen = if ob.additional.is_some() && !ob.keys.is_empty() { vec![false; ob.keys.len()] } else { vec![] };
                Some(F::Obj { n, ph: 0, next: 0, cur: 0, keybuf: vec![], seen })
            }
            b'[' if t & A != 0 => Some(F::Arr { n, ph: 0 }),
            b't' if t & B != 0 => Some(F::Lit { src: LitSrc::True, cands: vec![0], off: 1 }),
            b'f' if t & B != 0 => Some(F::Lit { src: LitSrc::False, cands: vec![0], off: 1 }),
            b'n' if t & Z != 0 => Some(F::Lit { src: LitSrc::Null, cands: vec![0], off: 1 }),
            b'-' if t & (I | N) != 0 => Some(F::Num { st: 0, int: t & N == 0 }),
            b'0' if t & (I | N) != 0 => Some(F::Num { st: 1, int: t & N == 0 }),
            b'1'..=b'9' if t & (I | N) != 0 => Some(F::Num { st: 2, int: t & N == 0 }),
            _ => None,
        }
    }
}

fn plain(bs: &[u8]) -> bool {
    bs.iter().all(|&b| b >= 0x20 && b != b'"' && b != b'\\')
}

fn num_complete(st: u8) -> bool {
    matches!(st, 1 | 2 | 4 | 7)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tab_bytes(words: &[&str]) -> Arc<TokTable> {
        // 1 語 = 1 バイト（0..256）＋ 足した語。止める語 = 1000
        let mut bytes: Vec<Box<[u8]>> = (0..256u32).map(|b| vec![b as u8].into_boxed_slice()).collect();
        for w in words {
            bytes.push(w.as_bytes().to_vec().into_boxed_slice());
        }
        while bytes.len() < 1001 {
            bytes.push(Box::new([]));
        }
        let n = bytes.len();
        let mut special = vec![false; n];
        special[1000] = true;
        special[999] = true; // </think>
        Arc::new(TokTable { bytes, special, stops: vec![1000], think_end: 999, valid: n })
    }

    fn run(schema: &Arc<Schema>, s: &str) -> (bool, bool) {
        let mut m = Matcher::new(schema.clone(), tab_bytes(&[]), false);
        for &b in s.as_bytes() {
            if !m.allows(b as u32) {
                return (false, false);
            }
            assert!(m.advance(b as u32));
        }
        (true, m.allows(1000))
    }

    fn ok(schema: &Json, s: &str) -> bool {
        let sc = Arc::new(Schema::compile(schema).unwrap());
        run(&sc, s) == (true, true)
    }

    #[test]
    fn json_object_any() {
        let sc = Arc::new(Schema::any_object());
        assert_eq!(run(&sc, r#"{"a": [1, 2.5e-3, "x\né", true, null, {"b": {}}]}"#), (true, true));
        assert_eq!(run(&sc, r#"{"a": 1"#), (true, false));
        assert_eq!(run(&sc, r#"[1]"#).0, false);
        assert_eq!(run(&sc, r#"{"a": 01}"#).0, false);
        assert_eq!(run(&sc, "{\"a\": \"x\ty\"}").0, false); // 文字列の中の制御文字
        assert_eq!(run(&sc, r#"{} x"#).0, false);
        assert_eq!(run(&sc, "  {\n  \"k\" : -0.5 }").0, true);
    }

    #[test]
    fn schema_order_required_enum() {
        let s = json!({"type": "object", "properties": {
            "area": {"type": "string", "enum": ["税務", "法務"]},
            "page": {"type": "integer"},
            "note": {"type": "string"},
            "v": {"type": ["integer", "null"]}},
            "required": ["area", "page"], "additionalProperties": false});
        assert!(ok(&s, r#"{"area": "税務", "page": 3}"#));
        assert!(ok(&s, r#"{"area":"法務","page":-12,"note":"a","v":null}"#));
        assert!(ok(&s, r#"{"area":"法務","page":0,"v":7}"#));
        let sc = Arc::new(Schema::compile(&s).unwrap());
        assert_eq!(run(&sc, r#"{"page": 3"#).0, false); // required の順
        assert_eq!(run(&sc, r#"{"area": "その他""#).0, false); // enum の外
        assert_eq!(run(&sc, r#"{"area": "税務"}"#).0, false); // page が無い
        assert_eq!(run(&sc, r#"{"area": "税務", "page": 1.5"#).0, false); // integer
        assert_eq!(run(&sc, r#"{"area": "税務", "page": 1, "x""#).0, false); // 余分な鍵
        assert_eq!(run(&sc, r#"{"area": "税務", "page": 1, "v": 1, "note""#).0, false); // optional も順
    }

    #[test]
    fn arrays_nested() {
        let s = json!({"type": "object", "properties": {"periods": {"type": "array", "items": {"type": "object",
            "properties": {"label": {"type": "string"}, "sales": {"type": "object", "properties": {"value": {"type": ["integer", "null"]}, "page": {"type": ["integer", "null"]}}, "required": ["value", "page"], "additionalProperties": false}},
            "required": ["label", "sales"], "additionalProperties": false}}, "employees": {"type": ["integer", "null"]}},
            "required": ["periods", "employees"], "additionalProperties": false});
        assert!(ok(&s, r#"{"periods": [{"label": "2024", "sales": {"value": 100, "page": null}}, {"label":"2025","sales":{"value":null,"page":2}}], "employees": 12}"#));
        assert!(ok(&s, r#"{"periods": [], "employees": null}"#));
        let sc = Arc::new(Schema::compile(&s).unwrap());
        assert_eq!(run(&sc, r#"{"periods": [1"#).0, false);
        assert_eq!(run(&sc, r#"{"periods": [], "employees": "x"#).0, false);
    }

    #[test]
    fn free_keys_with_properties() {
        let s = json!({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": {"type": "string"}});
        assert!(ok(&s, r#"{"x": "s", "a": 1}"#));
        let sc = Arc::new(Schema::compile(&s).unwrap());
        assert_eq!(run(&sc, r#"{"a": "s""#).0, false);
        assert_eq!(run(&sc, r#"{"x": 1"#).0, false);
        assert_eq!(run(&sc, r#"{"x": "s"}"#).0, false); // a が無い
    }

    #[test]
    fn unsupported_is_an_error() {
        for k in ["anyOf", "$ref", "pattern", "format", "minimum", "minItems", "oneOf", "allOf"] {
            let e = Schema::compile(&json!({"type": "object", "properties": {"a": {"type": "string", k: 1}}})).unwrap_err();
            assert!(e.contains(k), "{e}");
        }
        assert!(Schema::compile(&json!({"type": "object", "properties": {}, "required": ["x"], "additionalProperties": false})).is_err());
        assert!(Schema::compile(&json!({"type": "obj"})).is_err());
        assert!(Schema::compile(&json!({"type": "string", "description": "x", "title": "y"})).is_ok());
    }

    #[test]
    fn tokens_multi_byte_think_and_ws_limit() {
        let tab = tab_bytes(&["{\"", "a\":", " 1}", "}\n", "</think>"]);
        let sc = Arc::new(Schema::compile(&json!({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": false})).unwrap());
        let mut m = Matcher::new(sc.clone(), tab.clone(), true);
        assert!(m.allows(b'x' as u32) && m.allows(1000));
        assert!(m.advance(b'x' as u32));
        assert!(m.advance(999));
        assert!(!m.allows(b'x' as u32) && !m.allows(1000));
        assert!(m.allows(256) && m.advance(256));
        assert!(m.allows(257) && m.advance(257));
        assert!(!m.allows(259)); // "}\n" はまだ
        assert!(m.allows(258) && m.advance(258));
        assert!(m.allows(1000));
        assert!(!m.allows(b' ' as u32)); // 根の後は止める語だけ
        let mut m = Matcher::new(sc, tab, false);
        for _ in 0..WS_MAX {
            assert!(m.advance(b' ' as u32));
        }
        assert!(!m.allows(b' ' as u32));
        assert!(m.allows(b'{' as u32));
    }

    #[test]
    fn enum_numbers_and_prefixes() {
        let s = json!({"type": "object", "properties": {"n": {"enum": [1, 10, "a"]}}, "required": ["n"], "additionalProperties": false});
        assert!(ok(&s, r#"{"n": 1}"#));
        assert!(ok(&s, r#"{"n": 10}"#));
        assert!(ok(&s, r#"{"n": "a"}"#));
        let sc = Arc::new(Schema::compile(&s).unwrap());
        assert_eq!(run(&sc, r#"{"n": 2"#).0, false);
        assert_eq!(run(&sc, r#"{"n": 100"#).0, false);
    }

    /// 本物の語彙（248K 語）で語彙全部への絞りの時間: JSONC_TOK=weights/q36-hf cargo test --release --lib jsonc::tests::mask_time -- --ignored --nocapture
    #[test]
    #[ignore]
    fn mask_time() {
        let dir = std::env::var("JSONC_TOK").unwrap();
        let tok = crate::tokenizer::Tokenizer::from_dir(&dir).unwrap();
        let stops: Vec<u32> = tok.eos.clone();
        let tab = Arc::new(TokTable::new(&tok, &stops));
        let sc = Arc::new(Schema::compile(&json!({"type": "object", "properties": {"concerns": {"type": "array", "items": {"type": "object", "properties": {
            "area": {"enum": ["財務", "税務", "法務"]}, "point": {"type": "string"}, "page": {"type": "integer"}}, "required": ["area", "point", "page"], "additionalProperties": false}}},
            "required": ["concerns"], "additionalProperties": false})).unwrap());
        for (name, pre) in [("最初", ""), ("鍵の途中", "{\"conc"), ("文字列の中", "{\"concerns\": [{\"area\": \"税務\", \"point\": \"売上の"), ("数の後", "{\"concerns\": [{\"area\": \"税務\", \"point\": \"x\", \"page\": 12")] {
            let mut m = Matcher::new(sc.clone(), tab.clone(), false);
            for id in tok.encode(pre).unwrap() {
                assert!(m.advance(id), "{name}");
            }
            let mut l = vec![0.0f32; tab.valid];
            let t0 = std::time::Instant::now();
            for _ in 0..5 {
                l.iter_mut().for_each(|x| *x = 0.0);
                m.mask_logits(&mut l);
            }
            let ok = l.iter().filter(|x| x.is_finite()).count();
            println!("{name}: {:.1} ms・出せる語 {ok}", t0.elapsed().as_secs_f64() * 1000.0 / 5.0);
        }
    }

    #[test]
    fn mask_matches_allows() {
        let tab = tab_bytes(&["{\"", "a\":", " 1}", "}\n"]);
        let sc = Arc::new(Schema::any_object());
        let m = Matcher::new(sc, tab.clone(), false);
        let mut l = vec![0.0f32; tab.valid];
        m.mask_logits(&mut l);
        for i in 0..tab.valid {
            assert_eq!(l[i].is_finite(), m.allows(i as u32), "{i}");
        }
    }
}
