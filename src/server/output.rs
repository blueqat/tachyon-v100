//! 出た語 → 文字（`DecodeStream`）→ 止めの文字列 → 考え／答え／道具の呼び出し（`OutputParser`）。
//! 止める語（eos など）は語の番号で、止めの文字列（OpenAI の stop）は文字で見る。
//! 止めの文字列は考えも含めた出力全体に当てる（vLLM・llama.cpp と同じ）。
use crate::template::{OutEvent, OutputParser, ToolCall};
use crate::tokenizer::{DecodeStream, InkIds, Tokenizer};
use serde_json::Value as Json;

/// 描いたプロンプトが `<think>` を開いたまま終わっているか（考えありの生成。OutputParser::for_prompt と同じ見方）
pub fn prompt_opens_think(prompt: &str) -> bool {
    let tail = &prompt[prompt.rfind("<|im_start|>").unwrap_or(0)..];
    tail.rfind("<think>") > tail.rfind("</think>")
}

/// プロンプトの最後の回（`<|im_start|>` から。Inkling は `<|message_` から）。出力の分け方に要るのはここだけ
pub fn prompt_tail(prompt: &str) -> &str {
    let i = prompt.rfind("<|im_start|>").or_else(|| prompt.rfind("<|message_")).unwrap_or(0);
    &prompt[i..]
}

/// Inkling の出力の区切り（tokenizer_config の response_template と同じ見方）:
///   [<|message_model|>名前?] <|content_thinking|> 考え <|end_message|>
///   [<|message_model|>名前?] <|content_text|>（・content_tool_error・content_invoke_tool_text）答え <|end_message|>
///   [<|message_model|>名前?] <|content_invoke_tool_json|> {"name":…,"args":{…}} <|end_message|>
///   …（何回でも）… <|content_model_end_sampling|>（止める語）
#[derive(Clone, Copy, Debug, PartialEq)]
enum InkSeg {
    /// 区切りの外（<|message_model|> の後の名前もここ）
    Between,
    Think,
    Text,
    ToolJson,
}

struct InkState {
    ids: InkIds,
    seg: InkSeg,
    tool_buf: String,
}

/// 止めの文字列。どれかの頭かもしれない末尾は次まで持つ
pub struct StopStrings {
    stops: Vec<String>,
    pending: String,
}

impl StopStrings {
    pub fn new(stops: Vec<String>) -> Self {
        StopStrings { stops: stops.into_iter().filter(|s| !s.is_empty()).collect(), pending: String::new() }
    }

    /// 文字を足す → (出してよい文字, 止めの文字列に当たったか)。当たったら、その手前までを返す
    pub fn push(&mut self, s: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (s.to_string(), false);
        }
        self.pending.push_str(s);
        if let Some(i) = self.stops.iter().filter_map(|t| self.pending.find(t.as_str())).min() {
            let out = self.pending[..i].to_string();
            self.pending.clear();
            return (out, true);
        }
        let cut = self.pending.len() - self.hold_len();
        let out: String = self.pending.drain(..cut).collect();
        (out, false)
    }

    /// 末尾のうち、どれかの止めの文字列の頭になっている一番長い長さ
    fn hold_len(&self) -> usize {
        let p = &self.pending;
        let mut best = 0;
        for t in &self.stops {
            for n in (best + 1..=(t.len() - 1).min(p.len())).rev() {
                if t.is_char_boundary(n) && p.ends_with(&t[..n]) {
                    best = n;
                    break;
                }
            }
        }
        best
    }

    /// 終わり（持っている文字を全部出す）
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stopped {
    /// 止めの文字列
    String,
    /// 止める語
    Token(u32),
}

/// 1 依頼の出力の流れ（語を 1 つずつ入れる）
pub struct OutputPipe<'t> {
    ds: DecodeStream<'t>,
    parser: OutputParser,
    stops: StopStrings,
    stop_ids: Vec<u32>,
    think: u32,
    think_end: u32,
    tool_call: u32,
    in_reasoning: bool,
    ink: Option<InkState>,
    /// 出した語の数（止める語も数える＝usage.completion_tokens）
    pub n_tokens: usize,
    /// そのうち考えの語（`</think>` まで）
    pub n_reasoning: usize,
    pub stopped: Option<Stopped>,
    finished: bool,
}

impl<'t> OutputPipe<'t> {
    /// prompt: 描いたプロンプト（最後の回だけでよい）、tools: 引数の型（文字列以外を JSON として読む）
    pub fn new(tok: &'t Tokenizer, prompt: &str, tools: Option<&Json>, stops: Vec<String>, stop_ids: Vec<u32>) -> Self {
        OutputPipe {
            // special な語（<|im_end|> など）は文字にしない。<think>・<tool_call> は special でないので残る
            ds: tok.decode_stream(true),
            parser: OutputParser::for_prompt(prompt).with_tools(tools),
            stops: StopStrings::new(stops),
            stop_ids,
            think: tok.ids.think,
            think_end: tok.ids.think_end,
            tool_call: tok.ids.tool_call,
            in_reasoning: prompt_opens_think(prompt),
            // Inkling: 考えを飛ばす前置き（…<|message_model|><|content_text|>）で終わるなら答えから
            ink: tok.ink.map(|ids| InkState { ids, seg: if prompt.ends_with("<|content_text|>") { InkSeg::Text } else { InkSeg::Between }, tool_buf: String::new() }),
            n_tokens: 0,
            n_reasoning: 0,
            stopped: None,
            finished: false,
        }
    }

    pub fn is_done(&self) -> bool {
        self.stopped.is_some() || self.finished
    }

    /// Backend が止める語で終えた（Done(Stop(id))）。その語も出した数に入れる
    pub fn stop_token(&mut self, id: u32) {
        if !self.is_done() {
            self.n_tokens += 1;
            self.stopped = Some(Stopped::Token(id));
        }
    }

    /// 1 語足す。止まったら（止めの文字列・止める語）stopped が入り、以後は何もしない
    pub fn push(&mut self, id: u32) -> Vec<OutEvent> {
        if self.is_done() {
            return vec![];
        }
        if self.stop_ids.contains(&id) {
            // Backend が止める語を流してきた（本来は Done で来る）。ここでも止める
            self.stop_token(id);
            return vec![];
        }
        self.n_tokens += 1;
        if self.ink.is_some() {
            return self.push_ink(id);
        }
        if self.n_tokens == 1 && !self.in_reasoning && id == self.think {
            self.in_reasoning = true; // 答えの頭で <think> を開いた
        }
        if self.in_reasoning {
            if id == self.tool_call {
                self.in_reasoning = false;
            } else {
                self.n_reasoning += 1;
                if id == self.think_end {
                    self.in_reasoning = false;
                }
            }
        }
        let text = self.ds.step(id);
        let (out, hit) = self.stops.push(&text);
        let mut ev = self.parser.push(&out);
        if hit {
            self.stopped = Some(Stopped::String);
            ev.extend(self.parser.finish());
        }
        ev
    }

    /// Inkling の 1 語（役の語で区切りを変え、文字は今の区切りの種類で出す）
    fn push_ink(&mut self, id: u32) -> Vec<OutEvent> {
        let st = self.ink.as_ref().unwrap();
        let k = st.ids;
        let seg = st.seg;
        let mut ev = vec![];
        let is_mark = id == k.message_model
            || id == k.content_thinking
            || id == k.content_text
            || id == k.content_tool_error
            || id == k.content_invoke_tool_text
            || id == k.content_invoke_tool_json
            || id == k.end_message;
        if is_mark {
            // 区切りの前の文字（UTF-8 の途中など）を出し切る
            let rest = self.ds.flush();
            ev.extend(self.ink_text(seg, &rest));
            if self.is_done() {
                return ev;
            }
            let st = self.ink.as_mut().unwrap();
            if id == k.end_message && seg == InkSeg::ToolJson {
                let buf = std::mem::take(&mut st.tool_buf);
                match serde_json::from_str::<Json>(buf.trim()) {
                    Ok(Json::Object(o)) => {
                        let name = o.get("name").and_then(Json::as_str).unwrap_or("").to_string();
                        let arguments = match o.get("args").or_else(|| o.get("arguments")) {
                            Some(Json::Object(a)) => a.clone(),
                            _ => serde_json::Map::new(),
                        };
                        ev.push(OutEvent::ToolCall(ToolCall { name, arguments }));
                    }
                    // 読めない呼び出しは答えの文字として返す（黙って捨てない）
                    _ => ev.push(OutEvent::Content(buf)),
                }
            }
            st.seg = if id == k.content_thinking {
                InkSeg::Think
            } else if id == k.content_text || id == k.content_tool_error || id == k.content_invoke_tool_text {
                InkSeg::Text
            } else if id == k.content_invoke_tool_json {
                InkSeg::ToolJson
            } else {
                InkSeg::Between
            };
            if seg == InkSeg::Think || st.seg == InkSeg::Think {
                self.n_reasoning += 1;
            }
            return ev;
        }
        if seg == InkSeg::Think {
            self.n_reasoning += 1;
        }
        let text = self.ds.step(id);
        self.ink_text(seg, &text)
    }

    fn ink_text(&mut self, seg: InkSeg, text: &str) -> Vec<OutEvent> {
        if text.is_empty() {
            return vec![];
        }
        match seg {
            // 呼び出しの名前（<|message_model|>名前）は見ない
            InkSeg::Between => vec![],
            InkSeg::ToolJson => {
                self.ink.as_mut().unwrap().tool_buf.push_str(text);
                vec![]
            }
            InkSeg::Think | InkSeg::Text => {
                let (out, hit) = self.stops.push(text);
                if hit {
                    self.stopped = Some(Stopped::String);
                }
                if out.is_empty() {
                    return vec![];
                }
                vec![if seg == InkSeg::Think { OutEvent::Reasoning(out) } else { OutEvent::Content(out) }]
            }
        }
    }

    /// 終わり: 持っている文字（UTF-8 の途中・止めの文字列の頭かもしれない所）を全部出す
    pub fn finish(&mut self) -> Vec<OutEvent> {
        if self.finished {
            return vec![];
        }
        self.finished = true;
        if self.stopped == Some(Stopped::String) {
            return vec![]; // 当たった時に出し切っている
        }
        if let Some(st) = &self.ink {
            let seg = st.seg;
            let rest = self.ds.flush();
            let mut ev = self.ink_text(seg, &rest);
            let tail = self.stops.flush();
            if !tail.is_empty() && matches!(seg, InkSeg::Think | InkSeg::Text) {
                ev.push(if seg == InkSeg::Think { OutEvent::Reasoning(tail) } else { OutEvent::Content(tail) });
            }
            // 長さで切れた呼び出しは答えの文字として返す
            let buf = std::mem::take(&mut self.ink.as_mut().unwrap().tool_buf);
            if !buf.is_empty() {
                ev.push(OutEvent::Content(buf));
            }
            return ev;
        }
        let mut text = self.ds.flush();
        let (out, hit) = self.stops.push(&text);
        text = out;
        if !hit {
            text.push_str(&self.stops.flush());
        } else {
            self.stopped = Some(Stopped::String);
        }
        let mut ev = self.parser.push(&text);
        ev.extend(self.parser.finish());
        ev
    }
}

/// 道具の呼び出しを Qwen3.8 の形の文字に戻す（tool_choice "none" なのに呼んだとき、答えの文字として返す）
pub fn tool_call_text(c: &ToolCall) -> String {
    let mut s = format!("<tool_call>\n<function={}>\n", c.name);
    for (k, v) in &c.arguments {
        let v = match v {
            Json::String(s) => s.clone(),
            other => other.to_string(),
        };
        s.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n"));
    }
    s.push_str("</function>\n</tool_call>");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_strings_hold_and_hit() {
        let mut s = StopStrings::new(vec!["END".into(), "</x>".into()]);
        assert_eq!(s.push("abc E"), ("abc ".to_string(), false));
        assert_eq!(s.push("N"), (String::new(), false));
        assert_eq!(s.push("x"), ("ENx".to_string(), false));
        assert_eq!(s.push("12</"), ("12".to_string(), false));
        assert_eq!(s.push("x>tail"), (String::new(), true));
        let mut s = StopStrings::new(vec!["。".into()]);
        assert_eq!(s.push("今日は"), ("今日は".to_string(), false));
        assert_eq!(s.push("晴れ。明日"), ("晴れ".to_string(), true));
        let mut s = StopStrings::new(vec![]);
        assert_eq!(s.push("x"), ("x".to_string(), false));
        let mut s = StopStrings::new(vec!["abc".into()]);
        assert_eq!(s.push("xxab"), ("xx".to_string(), false));
        assert_eq!(s.flush(), "ab");
    }

    #[test]
    fn think_detection() {
        assert!(prompt_opens_think("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"));
        assert!(!prompt_opens_think("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        assert!(!prompt_opens_think("<|im_start|>assistant\n"));
    }
}
