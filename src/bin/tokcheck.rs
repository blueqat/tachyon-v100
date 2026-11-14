//! 語の切り分けと chat_template の答え合わせ。手本は tools/tok_ref.py（HF の tokenizers と jinja2）が書いた JSON。
//!
//!   tokcheck <hf の置き場> <手本.json> [chat_template.jinja か .gguf] [--gguf <.gguf>]…
//!
//! 見ること: 描いた文が同じ・語の番号が同じ・decode が同じ・1 語ずつ戻しても同じ・
//! 描いた答えを分け直すと元の考え／答え／道具の呼び出しに戻る（stream でも同じ）。速さも測る。
//! テンプレートは手本を作った物（手本.json の template。.gguf なら中の tokenizer.chat_template をじかに読む）。
//! --gguf: その GGUF の語の表（tokenizer.ggml.*）が tokenizer.json と同じか（Tokenizer::check_gguf）
use tachyon_v100::gguf::Model;
use tachyon_v100::template::{collect_media, expand_pads, normalize_messages, parse_output, ChatOptions, ChatTemplate, MediaKind, OutputParser, ParsedOutput, TemplateError, ToolCall};
use tachyon_v100::tokenizer::Tokenizer;
use serde_json::{json, Value as Json};
use std::time::Instant;

struct Check {
    ok: usize,
    ng: Vec<String>,
}

impl Check {
    fn eq<T: PartialEq + std::fmt::Debug>(&mut self, what: &str, got: T, want: T) -> bool {
        if got == want {
            self.ok += 1;
            true
        } else {
            let g = format!("{got:?}");
            let w = format!("{want:?}");
            let i = g.chars().zip(w.chars()).take_while(|(a, b)| a == b).count();
            let cut = |s: &str| s.chars().skip(i.saturating_sub(60)).take(160).collect::<String>();
            self.ng.push(format!("{what}: 違う（{i} 文字目から）\n  得た: …{}\n  手本: …{}", cut(&g), cut(&w)));
            false
        }
    }
}

/// 1 語ずつ戻してつなげる。途中で文字が切れた回数も返す
fn stream_decode(tok: &Tokenizer, ids: &[u32], skip: bool) -> (String, usize, Vec<String>) {
    let mut ds = tok.decode_stream(skip);
    let mut s = String::new();
    let mut held = 0;
    let mut pieces = vec![];
    for &id in ids {
        let p = ds.step(id);
        held += (ds.pending_bytes() > 0) as usize;
        s.push_str(&p);
        pieces.push(p);
    }
    let f = ds.flush();
    s.push_str(&f);
    pieces.push(f);
    (s, held, pieces)
}

fn collect(p: &mut OutputParser, pieces: &[&str]) -> ParsedOutput {
    let mut out = ParsedOutput::default();
    for s in pieces {
        p.push(s).into_iter().for_each(|e| out.add(e));
    }
    p.finish().into_iter().for_each(|e| out.add(e));
    out
}

/// 一度に・1 文字ずつ・でたらめな切れ目の 3 通りで分けて、同じになるか
fn parse_all_ways(c: &mut Check, what: &str, text: &str, in_reasoning: bool, tools: Option<&Json>, seed: u64) -> ParsedOutput {
    let one = parse_output(text, in_reasoning, tools);
    let chars: Vec<String> = text.chars().map(String::from).collect();
    let by_char = collect(&mut OutputParser::new(in_reasoning).with_tools(tools), &chars.iter().map(|s| s.as_str()).collect::<Vec<_>>());
    c.eq(&format!("{what} 1 文字ずつ"), &by_char, &one);
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut cuts = vec![];
    let mut i = 0;
    let bounds: Vec<usize> = text.char_indices().map(|(i, _)| i).chain([text.len()]).collect();
    while i < bounds.len() - 1 {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (i + 1 + (x >> 33) as usize % 12).min(bounds.len() - 1);
        cuts.push(&text[bounds[i]..bounds[j]]);
        i = j;
    }
    let rnd = collect(&mut OutputParser::new(in_reasoning).with_tools(tools), &cuts);
    c.eq(&format!("{what} でたらめな切れ目"), &rnd, &one);
    one
}

fn py_strip(s: &str) -> String {
    s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)).to_string()
}

/// 描いた assistant の回を分け直して、元の message に戻るか（OutputParser の確かめ）
fn roundtrip(c: &mut Check, tok: &Tokenizer, tpl: &ChatTemplate, name: &str, msgs: &Json, tools: Option<&Json>, opt: &ChatOptions, full: &str) {
    let arr = msgs.as_array().unwrap();
    let last = arr.last().unwrap();
    let mut o = opt.clone();
    o.add_generation_prompt = true;
    let prefix = tpl.render(&Json::Array(arr[..arr.len() - 1].to_vec()), tools, &o).unwrap();
    if !c.eq(&format!("{name} 往復: 生成の頭が一致"), full.starts_with(&prefix), true) {
        return;
    }
    let completion = &full[prefix.len()..];
    let got = parse_all_ways(c, &format!("{name} 往復"), completion, opt.thinking(), tools, 1);
    // プロンプトの終わりから「考えの中か」を決めても同じ
    let fp = collect(&mut OutputParser::for_prompt(&prefix).with_tools(tools), &[completion]);
    c.eq(&format!("{name} 往復: for_prompt"), &fp, &got);
    // 1 語ずつ戻しながら分けても同じ
    let ids = tok.encode(completion).unwrap();
    let (_, _, pieces) = stream_decode(tok, &ids, false);
    let st = collect(&mut OutputParser::new(opt.thinking()).with_tools(tools), &pieces.iter().map(|s| s.as_str()).collect::<Vec<_>>());
    c.eq(&format!("{name} 往復: 1 語ずつ"), &st, &got);
    let want_r = py_strip(last["reasoning_content"].as_str().unwrap_or(""));
    let want_c = py_strip(last["content"].as_str().unwrap_or(""));
    let want_t: Vec<ToolCall> = last["tool_calls"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|t| {
            let f = t.get("function").unwrap_or(t);
            ToolCall { name: f["name"].as_str().unwrap().into(), arguments: f["arguments"].as_object().cloned().unwrap_or_default() }
        })
        .collect();
    c.eq(&format!("{name} 往復: 考え"), got.reasoning_content.as_str(), want_r.as_str());
    c.eq(&format!("{name} 往復: 答え"), got.content.as_str(), want_c.as_str());
    c.eq(&format!("{name} 往復: 道具"), &got.tool_calls, &want_t);
}

/// 手で書いた出力の分け方（手本は Python に無いので、期待する値をここに書く）
fn parser_cases(c: &mut Check) {
    let tools = json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "n": {"type": "integer"}, "s": {"type": "string"}, "o": {"type": "object"}, "b": {"type": "boolean"}, "u": {"anyOf": [{"type": "integer"}]}}}}}]);
    let tc = |name: &str, a: Json| ToolCall { name: name.into(), arguments: a.as_object().cloned().unwrap() };
    let po = |r: &str, ct: &str, t: Vec<ToolCall>| ParsedOutput { reasoning_content: r.into(), content: ct.into(), tool_calls: t };
    let cases: Vec<(&str, &str, bool, ParsedOutput)> = vec![
        ("基本", "考える。\n\n二段落目\n</think>\n\n答え。", true, po("考える。\n\n二段落目", "答え。", vec![])),
        ("考えなし", "Hello\n\nworld  \n", false, po("", "Hello\n\nworld", vec![])),
        ("答えが <think> で始まる", "\n<think>\nabc\n</think>\n\ndef", false, po("abc", "def", vec![])),
        ("途中の <think> は文字", "x <think> y", false, po("", "x <think> y", vec![])),
        ("考えの途中で道具", "調べる<tool_call>\n<function=f>\n<parameter=n>\n42\n</parameter>\n</function>\n</tool_call>", true, po("調べる", "", vec![tc("f", json!({"n": 42}))])),
        (
            "型いろいろ",
            "</think>\n\n<tool_call>\n<function=f>\n<parameter=s>\n123\n</parameter>\n<parameter=o>\n{\"a\": [1, 2]}\n</parameter>\n<parameter=b>\nTrue\n</parameter>\n<parameter=u>\n7\n</parameter>\n<parameter=extra>\n8\n</parameter>\n</function>\n</tool_call>",
            true,
            po("", "", vec![tc("f", json!({"s": "123", "o": {"a": [1, 2]}, "b": true, "u": 7, "extra": "8"}))]),
        ),
        (
            "2 つ呼ぶ・閉じ忘れ",
            "</think>\n\n前置き\n\n<tool_call>\n<function=f>\n<parameter=n>1<parameter=s>a\nb\n</function>\n<tool_call>\n<function=g>\n</function>\n</tool_call>\n後ろ",
            true,
            po("", "前置き後ろ", vec![tc("f", json!({"n": 1, "s": "a\nb"})), tc("g", json!({}))]),
        ),
        ("壊れた呼び出しは答え", "</think>\n\n<tool_call>oops</tool_call> rest", true, po("", "<tool_call>oops</tool_call> rest", vec![])),
        ("max_tokens で切れた", "a</think>\n\n<tool_call>\n<function=f>\n<parameter=s>\n東京", true, po("a", "", vec![tc("f", json!({"s": "東京"}))])),
        ("考えの途中で切れた", "まだ考えて", true, po("まだ考えて", "", vec![])),
        ("考えの途中で im_end", "考え途中\n<|im_end|>後ろ", true, po("考え途中", "", vec![])),
        ("im_end で止まる", "x</think>y<|im_end|>garbage<tool_call>", true, po("x", "y", vec![])),
        ("印の頭だけ残る", "答え </thi", false, po("", "答え </thi", vec![])),
        ("空白だけ", " \n </think>\n\n \n", true, po("", "", vec![])),
        ("壊れた呼び出しの前後の空白", "abc \n<tool_call>notcall </tool_call> x\n", false, po("", "abc \n<tool_call>notcall </tool_call> x", vec![])),
        ("閉じない壊れた呼び出し", "abc\n\n<tool_call>junk", false, po("", "abc\n\n<tool_call>junk", vec![])),
    ];
    for (i, (name, text, in_r, want)) in cases.iter().enumerate() {
        let got = parse_all_ways(c, &format!("分け方 {name}"), text, *in_r, Some(&tools), i as u64);
        c.eq(&format!("分け方 {name}"), &got, want);
    }
    // OpenAI 形への変換
    let t = tc("f", json!({"n": 1, "s": "日本"}));
    c.eq("to_openai", t.to_openai("call_0"), json!({"id": "call_0", "type": "function", "function": {"name": "f", "arguments": "{\"n\":1,\"s\":\"日本\"}"}}));
}

/// kwargs・messages の直し・画像の pad の扱い
fn misc_cases(c: &mut Check, tok: &Tokenizer, tpl: &ChatTemplate) {
    let k = |v: Json| ChatOptions::from_kwargs(&v, true);
    c.eq("kwargs: 文字列の enable_thinking は誤り", matches!(k(json!({"enable_thinking": "false"})), Err(TemplateError::BadRequest(_))), true);
    c.eq("kwargs: null は渡さない", k(json!({"reasoning_effort": null, "enable_thinking": null})).unwrap().reasoning_effort, None);
    c.eq("kwargs: messages は渡せない", matches!(k(json!({"messages": []})), Err(TemplateError::BadRequest(_))), true);
    let o = k(json!({"enable_thinking": false, "reasoning_effort": "low", "preserve_thinking": true, "foo": 1})).unwrap();
    c.eq("kwargs: 読める", (o.enable_thinking, o.reasoning_effort.clone(), o.preserve_thinking, o.extra.get("foo").cloned()), (Some(false), Some("low".into()), Some(true), Some(json!(1))));
    c.eq("kwargs: 考えない", o.thinking(), false);
    let bad = json!([{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": "{oops"}}]}]);
    c.eq("messages: 壊れた arguments は誤り", matches!(normalize_messages(&bad), Err(TemplateError::BadRequest(_))), true);
    let ok = json!([{"role": "developer", "content": "短く"}, {"role": "user", "content": "x"}, {"role": "assistant", "content": null, "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": ""}}, {"function": {"name": "g", "arguments": "{\"a\": [1, \"日本\"]}"}}]}]);
    let n = normalize_messages(&ok).unwrap();
    c.eq("messages: developer は system", n[0]["role"].clone(), json!("system"));
    c.eq("messages: 空の arguments は {}", n[2]["tool_calls"][0]["function"]["arguments"].clone(), json!({}));
    c.eq("messages: arguments を object に", n[2]["tool_calls"][1]["function"]["arguments"].clone(), json!({"a": [1, "日本"]}));
    let p = tpl.render(&n, None, &ChatOptions { add_generation_prompt: true, ..Default::default() });
    c.eq("messages: 直した物を描ける", p.is_ok(), true);
    c.eq("pad: 数が合わないと誤り", expand_pads(&[1, tok.ids.image_pad, 2], tok.ids.image_pad, &[]).is_err(), true);
    c.eq("pad: 広げる", expand_pads(&[1, 9, 2, 9], 9, &[2, 0]).unwrap(), vec![1, 9, 9, 2]);
}

fn main() {
    let (mut a, mut ggufs) = (vec![], vec![]);
    let mut it = std::env::args().skip(1);
    while let Some(x) = it.next() {
        if x == "--gguf" {
            ggufs.push(it.next().expect("--gguf の後に GGUF"));
        } else {
            a.push(x);
        }
    }
    let hf = &a[0];
    let r: Json = serde_json::from_str(&std::fs::read_to_string(&a[1]).unwrap()).unwrap();
    let tpl_path = a.get(2).cloned().unwrap_or_else(|| r["template"].as_str().unwrap().to_string());
    let t0 = Instant::now();
    let tok = Tokenizer::from_dir(hf).unwrap();
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    let tpl = if tpl_path.ends_with(".gguf") {
        ChatTemplate::from_gguf(&Model::open(&tpl_path).unwrap()).unwrap()
    } else {
        ChatTemplate::from_file(&tpl_path).unwrap()
    };
    println!("テンプレート: {tpl_path}");
    let mut c = Check { ok: 0, ng: vec![] };

    // ── GGUF の語の表 ──
    for g in &ggufs {
        let tt = Instant::now();
        match tok.check_gguf(&Model::open(g).unwrap()) {
            Ok(v) => {
                c.ok += 1;
                println!(
                    "GGUF の語の表が tokenizer.json と同じ（{:.0}ms）: 語 {}（tokenizer.json {}、余り {} は [PAD…]）・merges {}・pre {}・bos {:?} eos {:?} pad {:?}・add_bos {:?} ← {g}",
                    tt.elapsed().as_secs_f64() * 1e3,
                    v.n_vocab,
                    tok.vocab_size(),
                    v.n_vocab - tok.vocab_size(),
                    v.n_merges,
                    v.pre,
                    v.bos,
                    v.eos,
                    v.pad,
                    v.add_bos
                );
            }
            Err(e) => c.ng.push(format!("{g}: {e}")),
        }
    }
    println!("止める語 {:?}、決まった語 {:?}", tok.eos, tok.ids);
    if let Some(l) = r.get("llama").filter(|l| !l.is_null()) {
        let d: Vec<&str> = l["diffs"].as_array().into_iter().flatten().filter_map(Json::as_str).collect();
        println!("（手本を作るときの llama.cpp の切り分け: {}/{} 件が HF と同じ。違う {:?}…）", l["same"], l["n"], &d[..d.len().min(5)]);
    }

    // ── chat_template ──
    let (mut n_chat, mut n_same, mut n_err, mut n_rt, mut n_stricter) = (0, 0, 0, 0, 0);
    let mut held_total = 0;
    for cs in r["chat"].as_array().unwrap() {
        let name = cs["name"].as_str().unwrap();
        let ng0 = c.ng.len();
        let mut msgs = cs["messages"].clone();
        if cs["openai_args"].as_bool() == Some(true) {
            // OpenAI の形（arguments は JSON の文字列）にしてから直す
            for m in msgs.as_array_mut().unwrap() {
                for t in m.get_mut("tool_calls").and_then(Json::as_array_mut).into_iter().flatten() {
                    let s = t["function"]["arguments"].to_string();
                    t["function"]["arguments"] = Json::String(s);
                }
            }
            msgs = normalize_messages(&msgs).unwrap();
        }
        let tools = if cs["tools"].is_null() { None } else { Some(&cs["tools"]) };
        let opt = ChatOptions::from_kwargs(&cs["kwargs"], cs["add_generation_prompt"].as_bool().unwrap()).unwrap();
        // テンプレートそのままの振る舞い（render_raw）が手本と同じか。render は reasoning_effort を先に確かめる
        let got = tpl.render_raw(&msgs, tools, &opt);
        let strict = tpl.render(&msgs, tools, &opt);
        let ex = &cs["expect"];
        n_chat += 1;
        let bad_effort = opt.reasoning_effort.as_deref().is_some_and(|e| !tachyon_v100::template::EFFORTS.contains(&e));
        c.eq(&format!("{name}: reasoning_effort の確かめ"), matches!(strict, Err(TemplateError::BadEffort(_))), bad_effort);
        if !bad_effort {
            c.eq(&format!("{name}: render と render_raw"), &strict, &got);
        } else if ex["error"].is_null() {
            n_stricter += 1;
        }
        if let Some(e) = ex["error"].as_str() {
            if ex["raised"].as_bool() != Some(false) {
                c.eq(&format!("{name}: 誤り"), got.clone().err(), Some(TemplateError::Raised(e.to_string())));
            } else {
                // jinja2 の中の誤り（未定義の足し算・items に文字列 など）。文は jinja2 と違うので止まることだけ
                c.eq(&format!("{name}: 誤り（テンプレートの中）"), matches!(got, Err(TemplateError::Template(_))), true);
            }
            n_err += 1;
        } else {
            let want = ex["prompt"].as_str().unwrap();
            match got {
                Err(e) => c.ng.push(format!("{name}: 描けない {e}")),
                Ok(p) => {
                    c.eq(&format!("{name}: 描いた文"), p.as_str(), want);
                    let ids = tok.encode(&p).unwrap();
                    let want_ids: Vec<u32> = serde_json::from_value(ex["ids"].clone()).unwrap();
                    c.eq(&format!("{name}: 語の番号"), &ids, &want_ids);
                    c.eq(&format!("{name}: 語の番号（区切らず）"), &tok.encode_plain(&p).unwrap(), &want_ids);
                    c.eq(&format!("{name}: 語の番号（使い回し）"), &tok.encode_cached("tokcheck", &p).unwrap(), &want_ids);
                    c.eq(&format!("{name}: 語の番号（切れ目ごと）"), &tok.encode_every_cut(&p).unwrap(), &want_ids);
                    c.eq(&format!("{name}: decode"), tok.decode(&want_ids, false), ex["decoded"].as_str().unwrap().to_string());
                    c.eq(&format!("{name}: decode(skip)"), tok.decode(&want_ids, true), ex["decoded_skip"].as_str().unwrap().to_string());
                    let (s, held, _) = stream_decode(&tok, &want_ids, false);
                    held_total += held;
                    c.eq(&format!("{name}: 1 語ずつ decode"), s.as_str(), ex["decoded"].as_str().unwrap());
                    // 画像・動画: テンプレートが置いた pad の数と、集めた数が合う
                    let media = collect_media(&msgs);
                    let n_img = media.iter().filter(|m| m.0 == MediaKind::Image).count();
                    let n_vid = media.len() - n_img;
                    // （special_tokens_in_text は文の中に <|image_pad|> と書いてあるので外す）
                    if !name.starts_with("special_tokens") {
                        c.eq(&format!("{name}: 画像の数"), ids.iter().filter(|&&i| i == tok.ids.image_pad).count(), n_img);
                        c.eq(&format!("{name}: 動画の数"), ids.iter().filter(|&&i| i == tok.ids.video_pad).count(), n_vid);
                        let counts: Vec<usize> = (0..n_img).map(|k| 4 + k * 3).collect();
                        let ex_ids = expand_pads(&ids, tok.ids.image_pad, &counts).unwrap();
                        c.eq(&format!("{name}: 画像を広げた長さ"), ex_ids.len(), ids.len() + counts.iter().sum::<usize>() - n_img);
                    }
                    if cs["roundtrip"].as_bool() == Some(true) {
                        roundtrip(&mut c, &tok, &tpl, name, &msgs, tools, &opt, &p);
                        n_rt += 1;
                    }
                }
            }
        }
        if c.ng.len() == ng0 {
            n_same += 1;
        }
    }
    println!("chat_template: {n_same}/{n_chat} 件が手本と同じ（うち誤りで止まる {n_err} 件、分け直し {n_rt} 件、テンプレートは通すが render が reasoning_effort で止める {n_stricter} 件）");
    misc_cases(&mut c, &tok, &tpl);

    // ── 文の切り分け ──
    let texts = r["texts"].as_array().unwrap();
    let mut n_txt_ok = 0;
    let (mut nbytes, mut ntok) = (0usize, 0usize);
    for t in texts {
        let name = t["name"].as_str().unwrap();
        let ng0 = c.ng.len();
        let s = t["text"].as_str().unwrap();
        let want: Vec<u32> = serde_json::from_value(t["ids"].clone()).unwrap();
        let ids = tok.encode(s).unwrap();
        c.eq(&format!("文 {name}: 語の番号"), &ids, &want);
        c.eq(&format!("文 {name}: 語の番号（区切らず）"), &tok.encode_plain(s).unwrap(), &want);
        c.eq(&format!("文 {name}: 語の番号（使い回し）"), &tok.encode_cached("tokcheck", s).unwrap(), &want);
        c.eq(&format!("文 {name}: 語の番号（切れ目ごと）"), &tok.encode_every_cut(s).unwrap(), &want);
        c.eq(&format!("文 {name}: decode"), tok.decode(&ids, false), t["decoded"].as_str().unwrap().to_string());
        c.eq(&format!("文 {name}: decode(skip)"), tok.decode(&ids, true), t["decoded_skip"].as_str().unwrap().to_string());
        let (sd, held, _) = stream_decode(&tok, &ids, false);
        held_total += held;
        c.eq(&format!("文 {name}: 1 語ずつ decode"), sd.as_str(), t["decoded"].as_str().unwrap());
        let (sk, _, _) = stream_decode(&tok, &ids, true);
        c.eq(&format!("文 {name}: 1 語ずつ decode(skip)"), sk.as_str(), t["decoded_skip"].as_str().unwrap());
        nbytes += s.len();
        ntok += ids.len();
        if c.ng.len() == ng0 {
            n_txt_ok += 1;
        }
    }
    println!("切り分け: {n_txt_ok}/{} 件が手本と同じ（{:.2}MB・{ntok} 語）", texts.len(), nbytes as f64 / 1e6);

    // ── でたらめな番号の decode（壊れた UTF-8 の扱いが同じか） ──
    let rnd = r["random_ids"].as_array().unwrap();
    let mut n_rnd_ok = 0;
    for (i, x) in rnd.iter().enumerate() {
        let ng0 = c.ng.len();
        let ids: Vec<u32> = serde_json::from_value(x["ids"].clone()).unwrap();
        c.eq(&format!("でたらめ {i}: decode"), tok.decode(&ids, false), x["decoded"].as_str().unwrap().to_string());
        c.eq(&format!("でたらめ {i}: decode(skip)"), tok.decode(&ids, true), x["decoded_skip"].as_str().unwrap().to_string());
        let (sd, _, _) = stream_decode(&tok, &ids, false);
        c.eq(&format!("でたらめ {i}: 1 語ずつ"), sd.as_str(), x["decoded"].as_str().unwrap());
        if c.ng.len() == ng0 {
            n_rnd_ok += 1;
        }
    }
    println!("でたらめな番号: {n_rnd_ok}/{} 件の decode が手本と同じ。1 語ずつ戻すとき文字の途中で切れた語 {held_total} 個", rnd.len());

    // ── 出力の分け方 ──
    let ng0 = c.ng.len();
    let ok0 = c.ok;
    parser_cases(&mut c);
    println!("出力の分け方（手書き 16 件 × 3 通りの切れ目）: {} 個のうち {} 個 OK", c.ok - ok0 + c.ng.len() - ng0, c.ok - ok0);

    // ── 大きな会話（並べて切る道と使い回しの道を、区切らない切り分けと比べる） ──
    let files: Vec<&str> = texts.iter().filter(|t| t["name"].as_str().unwrap().starts_with("file:")).map(|t| t["text"].as_str().unwrap()).collect();
    let mut msgs = vec![json!({"role": "system", "content": "資料を読んで答える。"})];
    for (i, f) in files.iter().enumerate() {
        msgs.push(json!({"role": "user", "content": format!("資料 {i}:\n{f}")}));
        msgs.push(json!({"role": "assistant", "content": format!("読みました（{i}）。"), "reasoning_content": "要点を拾う。"}));
    }
    msgs.push(json!({"role": "user", "content": "全部をまとめて。"}));
    let big_opt = ChatOptions { add_generation_prompt: true, reasoning_effort: Some("medium".into()), ..Default::default() };
    let big = tpl.render(&Json::Array(msgs.clone()), None, &big_opt).unwrap();
    tok.clear_cache();
    let tt = Instant::now();
    let plain = tok.encode_plain(&big).unwrap();
    let t_plain = tt.elapsed().as_secs_f64();
    let tt = Instant::now();
    let cold = tok.encode_cached("tokcheck", &big).unwrap();
    let t_cold = tt.elapsed().as_secs_f64();
    c.eq("大きな会話: 並べて切る", &cold, &plain);
    // 次の回: 答えと新しい問いが 1 つずつ増えた会話（前の回は使い回せる）
    msgs.push(json!({"role": "assistant", "content": "まとめ: GPU は V100。", "reasoning_content": "短く。"}));
    msgs.push(json!({"role": "user", "content": "もう少し詳しく。"}));
    let big2 = tpl.render(&Json::Array(msgs), None, &big_opt).unwrap();
    let s0 = tok.cache_stats("tokcheck");
    let (h0, m0) = (s0.hits, s0.misses);
    let tt = Instant::now();
    let warm = tok.encode_cached("tokcheck", &big2).unwrap();
    let t_warm = tt.elapsed().as_secs_f64();
    let s1 = tok.cache_stats("tokcheck");
    let (h1, m1) = (s1.hits, s1.misses);
    let tt = Instant::now();
    let plain2 = tok.encode_plain(&big2).unwrap();
    let t_plain2 = tt.elapsed().as_secs_f64();
    c.eq("大きな会話: 次の回（使い回し）", &warm, &plain2);
    println!(
        "大きな会話 {:.2}MB・{} 語: 区切らず {:.0}ms／並べて {:.0}ms／次の回（使い回し 当たり {} 外れ {}）{:.1}ms（区切らずなら {:.0}ms）",
        big.len() as f64 / 1e6,
        plain.len(),
        t_plain * 1e3,
        t_cold * 1e3,
        h1 - h0,
        m1 - m0,
        t_warm * 1e3,
        t_plain2 * 1e3
    );

    // ── 速さ ──
    for t in texts.iter().filter(|t| t["name"].as_str().unwrap().starts_with("file:")) {
        let s = t["text"].as_str().unwrap();
        tok.encode_plain(s).unwrap(); // BPE の語の覚えを温める
        let tt = Instant::now();
        let n = tok.encode_plain(s).unwrap().len();
        let dt = tt.elapsed().as_secs_f64();
        tok.clear_cache();
        let tt = Instant::now();
        tok.encode(s).unwrap();
        let dp = tt.elapsed().as_secs_f64();
        println!(
            "  encode {:<34} {:>7}B {:>6} 語 区切らず {:>6.1}ms {:>5.2}MB/s・並べて {:>6.1}ms",
            t["name"].as_str().unwrap(),
            s.len(),
            n,
            dt * 1e3,
            s.len() as f64 / dt / 1e6,
            dp * 1e3
        );
    }
    let ids = tok.encode(&big).unwrap();
    let tt = Instant::now();
    let mut ds = tok.decode_stream(false);
    let mut n = 0;
    for &i in &ids {
        n += ds.step(i).len();
    }
    let dec = tt.elapsed().as_secs_f64();
    let case = r["chat"].as_array().unwrap().iter().find(|x| x["name"] == "tools_call_content_before").unwrap();
    let opt = ChatOptions::from_kwargs(&case["kwargs"], true).unwrap();
    let tt = Instant::now();
    let reps = 200;
    for _ in 0..reps {
        std::hint::black_box(tpl.render(&case["messages"], Some(&case["tools"]), &opt).unwrap());
    }
    let rend = tt.elapsed().as_secs_f64() / reps as f64;
    let tt = Instant::now();
    std::hint::black_box(tpl.render(&Json::Array(vec![json!({"role": "user", "content": big.clone()})]), None, &big_opt).unwrap());
    let rend_big = tt.elapsed().as_secs_f64();
    println!(
        "速さ: 読み込み {load_ms:.0}ms、1 語ずつ decode {:.0}ns/語（{n}B）、描く（道具 2 つ・5 回の会話）{:.0}µs、描く（{:.2}MB の 1 回）{:.1}ms",
        dec / ids.len() as f64 * 1e9,
        rend * 1e6,
        big.len() as f64 / 1e6,
        rend_big * 1e3
    );

    if c.ng.is_empty() {
        println!("全部 OK（{} 個の確かめ）", c.ok);
    } else {
        for e in c.ng.iter().take(30) {
            println!("NG {e}");
        }
        println!("NG {} 個（OK {} 個）", c.ng.len(), c.ok);
        std::process::exit(1);
    }
}
