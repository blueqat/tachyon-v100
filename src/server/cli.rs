//! `serve` の引数と起動（tachyon-serve と、M1 の tachyon の serve で同じ物を使う）。
//!
//!   serve --fake [--hf DIR | --synthetic --template FILE] [--keys FILE | --no-auth] [--host 127.0.0.1] [--port 8080]
//!         [--model-name qwen3.8-27b] [--weights GGUF | --weights-sha HEX] [--max-context 262144] [--max-output 131072]
//!         [--default-max-tokens 32768] [--max-queue 64] [--max-running 32] [--no-fim-stop]
//!         [--fake-delay-ms 1] [--fake-first-ms 0] [--step-tokens 512] [--prefill-chunk 256]
//!         [--prefix-owner-accounts a,b]（そのアカウントの頭の使い回しを x-tachyon-prefix-owner の利用者ごとに分ける）
//!         [--ink-nothink-floor-tokens 4096] [--ink-nothink-floor-effort low] [--ink-nothink-floor-budget 2048]（Inkling: 長い・文書つきの考えなしの依頼に考えを入れる）
//!         [--metrics-key]（/metrics にも admin の鍵）[--allow-plain-keys]（平文の鍵の設定で起こす。試しだけ）
//!   hash-key   標準入力の鍵 1 行 → sha256（keys.json の keys_sha256 に書く値。鍵はシェルの履歴にも ps にも残らない）
//!   gen-key    鍵を作る（鍵は標準出力に 1 行・sha256 は標準エラーに）
//!   sha FILE   重みの sha256（<FILE>.sha256 に置けば起動のたびに計算しない）
use super::engine::{EngineLoader, SchedConfig, SimpleScheduler};
use super::fake::{write_synthetic_tokenizer, FakeBackend, FakeConfig};
use super::util::{self, sha256_hex};
use super::{serve, shutdown_signal, AppState, Auth, Backend, ServerConfig};
use crate::template::ChatTemplate;
use crate::tokenizer::Tokenizer;
use std::collections::HashMap;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// 既定の tokenizer・chat_template の置き場（crate::weights_path の下）
pub fn default_hf() -> String {
    crate::weights_path("q38-27b/hf")
}

/// `--key value` と `--flag` の並び
pub struct ServeArgs {
    kv: HashMap<String, String>,
}

impl ServeArgs {
    pub fn parse(args: &[String]) -> Result<Self, String> {
        let mut kv = HashMap::new();
        let mut i = 0;
        while i < args.len() {
            let k = args[i].strip_prefix("--").ok_or_else(|| format!("引数が変: {}", args[i]))?.to_string();
            if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                kv.insert(k, args[i + 1].clone());
                i += 2;
            } else {
                kv.insert(k, "1".into());
                i += 1;
            }
        }
        Ok(ServeArgs { kv })
    }

    pub fn s(&self, k: &str) -> Option<&str> {
        self.kv.get(k).map(String::as_str)
    }

    pub fn flag(&self, k: &str) -> bool {
        self.kv.contains_key(k)
    }

    pub fn n(&self, k: &str, d: usize) -> Result<usize, String> {
        self.s(k).map(|s| s.parse().map_err(|_| format!("--{k} は数"))).unwrap_or(Ok(d))
    }
}

/// tokenizer とテンプレート（--hf DIR、または試し用の --synthetic --template FILE）
pub fn load_tokenizer(a: &ServeArgs) -> Result<(Arc<Tokenizer>, Arc<ChatTemplate>), String> {
    if a.flag("synthetic") {
        let dir = std::env::temp_dir().join(format!("tachyon-synth-{}", std::process::id()));
        write_synthetic_tokenizer(&dir).map_err(|e| e.to_string())?;
        let tok = Tokenizer::from_dir(&dir).map_err(|e| e.to_string())?;
        let tpl = a.s("template").ok_or("--synthetic には --template <chat_template.jinja> が要る")?;
        return Ok((Arc::new(tok), Arc::new(ChatTemplate::from_file(tpl).map_err(|e| e.to_string())?)));
    }
    let hf = PathBuf::from(a.s("hf").map(String::from).unwrap_or_else(default_hf));
    let mut tok = Tokenizer::from_dir(&hf).map_err(|e| format!("{}: {e}", hf.display()))?;
    // ⛔ アカウントをまたぐ切り分けの状態を残さない: BPE の語の覚え（糸ごと・全員で共有）は切る（--word-cache で戻す）。
    // 会話の回ごとの覚えはアカウントごと（Tokenizer::encode_cached）
    tok.set_word_cache(a.flag("word-cache"));
    let tpl_path = a.s("template").map(PathBuf::from).unwrap_or_else(|| hf.join("chat_template.jinja"));
    let tpl = ChatTemplate::from_file(&tpl_path).map_err(|e| e.to_string())?;
    Ok((Arc::new(tok), Arc::new(tpl)))
}

/// 重みの版: --weights-sha ＞ <weights>.sha256 ＞ 計算（17.5GB で十数秒）。どれも無ければ 0
pub fn weights_sha(a: &ServeArgs) -> Result<String, String> {
    if let Some(s) = a.s("weights-sha") {
        return Ok(s.to_ascii_lowercase());
    }
    let Some(w) = a.s("weights") else { return Ok("0".repeat(64)) };
    let side = PathBuf::from(format!("{w}.sha256"));
    if let Ok(s) = std::fs::read_to_string(&side) {
        if let Some(h) = s.split_whitespace().next().filter(|h| h.len() == 64) {
            return Ok(h.to_ascii_lowercase());
        }
    }
    eprintln!("重みの sha256 を計算する（{w}。{} に置けば次から速い）", side.display());
    util::sha256_file(Path::new(w)).map_err(|e| format!("{w}: {e}"))
}

pub fn server_config(a: &ServeArgs, sha: String) -> Result<ServerConfig, String> {
    let d = ServerConfig::default();
    let max_context = a.n("max-context", d.max_context)?;
    let thinking = match a.s("thinking-default").unwrap_or("off") {
        "on" | "true" | "1" => true,
        "off" | "false" | "0" => false,
        x => return Err(format!("--thinking-default は on か off（{x}）")),
    };
    let secs = |k: &str, d: Duration| -> Result<Duration, String> { Ok(Duration::from_secs_f64(a.s(k).map(|s| s.parse::<f64>().map_err(|_| format!("--{k} は秒"))).transpose()?.unwrap_or(d.as_secs_f64()))) };
    let (grace, deadline) = (secs("shutdown-grace-secs", d.shutdown_grace)?, secs("exit-deadline-secs", d.exit_deadline)?);
    if deadline < grace + Duration::from_secs(2) {
        return Err("--exit-deadline-secs は --shutdown-grace-secs より 2 秒以上長く（取り消しの返事を送る間）".into());
    }
    Ok(ServerConfig {
        model_name: a.s("model-name").unwrap_or(d.model_name.as_str()).to_string(),
        weights_sha: sha,
        max_context,
        max_prompt_tokens: a.n("max-prompt", max_context)?,
        max_message_bytes: a.n("max-message-bytes", d.max_message_bytes)?,
        // 画像を受けるときは本文の上限を 24MiB に（base64 で 4/3 倍）
        max_body_bytes: a.n("max-body-bytes", if a.s("mmproj").is_some() { 24 << 20 } else { d.max_body_bytes })?,
        images: a.s("mmproj").is_some(),
        max_images: a.n("max-images", d.max_images)?,
        max_image_bytes: a.n("max-image-bytes", d.max_image_bytes)?,
        max_messages: a.n("max-messages", d.max_messages)?,
        default_thinking: thinking,
        shutdown_grace: grace,
        exit_deadline: deadline,
        default_max_tokens: a.n("default-max-tokens", d.default_max_tokens)?,
        max_output: a.n("max-output", d.max_output)?,
        min_room: a.n("min-room", d.min_room)?,
        max_inflight: a.n("max-inflight", d.max_inflight)?,
        fim_stops: !a.flag("no-fim-stop"),
        metrics_key: a.flag("metrics-key"),
        prefix_owner_accounts: a.s("prefix-owner-accounts").map(|v| v.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()).unwrap_or_default(),
        ink_nothink_floor_tokens: a.n("ink-nothink-floor-tokens", d.ink_nothink_floor_tokens)?,
        ink_nothink_floor_effort: a.s("ink-nothink-floor-effort").unwrap_or(d.ink_nothink_floor_effort.as_str()).to_string(),
        ink_nothink_floor_budget: a.n("ink-nothink-floor-budget", d.ink_nothink_floor_budget)?,
        stall_timeout: Duration::from_secs(a.n("stall-secs", d.stall_timeout.as_secs() as usize)? as u64),
        ..d
    })
}

/// エンジンを作る関数（M1 の tachyon が渡す。tachyon-serve は None で --fake だけ）
pub type MakeEngine = Box<dyn FnOnce(&ServeArgs) -> Result<EngineLoader, String>>;

/// serve の本体（返るのは止められたとき）
pub fn serve_main(args: &[String], make_engine: Option<MakeEngine>) -> Result<(), String> {
    let a = ServeArgs::parse(args)?;
    let (tok, tpl) = load_tokenizer(&a)?;
    let auth = match (a.s("keys"), a.flag("no-auth")) {
        (Some(p), _) => Auth::from_file(Path::new(p))?,
        (None, true) => Auth::open(),
        (None, false) => return Err("--keys <keys.json> か --no-auth（127.0.0.1 で試すときだけ）を付ける".into()),
    };
    if !auth.plain_key_accounts.is_empty() {
        if !a.flag("allow-plain-keys") {
            return Err(format!("平文の鍵が設定にある（{}）。keys_sha256 にする（試しだけなら --allow-plain-keys）", auth.plain_key_accounts.join(", ")));
        }
        eprintln!("⚠️ 平文の鍵が設定にある（{}）。運用では keys_sha256 にする", auth.plain_key_accounts.join(", "));
    }
    let sha = weights_sha(&a)?;
    if !a.flag("fake") && sha.bytes().all(|c| c == b'0') {
        return Err("本物のエンジンには重みの版が要る: --weights <GGUF>（<GGUF>.sha256 か計算）か --weights-sha <16 進>".into());
    }
    let cfg = server_config(&a, sha)?;
    // ⛔ 名前の打ち違いで黙って分けないことがないように、載せたアカウントは鍵の設定に要る（--no-auth のときは "local"）
    if let Some(x) = cfg.prefix_owner_accounts.iter().find(|x| !auth.accounts.iter().any(|ac| ac.name == **x)) {
        return Err(format!("--prefix-owner-accounts の {x:?} は鍵の設定に無いアカウント"));
    }
    if !cfg.prefix_owner_accounts.is_empty() {
        eprintln!("頭の使い回しを利用者ごとに分けるアカウント（x-tachyon-prefix-owner）: {}", cfg.prefix_owner_accounts.join(", "));
    }
    let backend: Arc<dyn Backend> = if a.flag("fake") {
        let fc = FakeConfig {
            delay: Duration::from_millis(a.n("fake-delay-ms", 1)? as u64),
            first_delay: Duration::from_millis(a.n("fake-first-ms", 0)? as u64),
            max_running: a.n("max-running", 32)?,
            max_waiting: a.n("max-queue", 64)?,
            ..Default::default()
        };
        FakeBackend::new(tok.clone(), fc)
    } else if let Some(mk) = make_engine {
        let sc = SchedConfig {
            max_running: a.n("max-running", 32)?,
            max_waiting: a.n("max-queue", 64)?,
            step_tokens: a.n("step-tokens", 512)?,
            prefill_chunk: a.n("prefill-chunk", 256)?,
            prefill_chunk_alone: a.n("prefill-chunk-alone", 2048)?,
            valid_vocab: tok.vocab_size(),
        };
        SimpleScheduler::spawn(mk(&a)?, sc)
    } else {
        return Err("エンジンがまだつながっていない: --fake で試すか、M1 の tachyon serve（tachyon_v100::server::Engine を実装）を使う".into());
    };
    let host = a.s("host").unwrap_or("127.0.0.1").to_string();
    let port = a.n("port", 8080)? as u16;
    let threads = a.n("threads", 8)?;
    let st = AppState::new(cfg, backend, tok, tpl, auth, util::stderr_log())?;
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(threads.max(2)).enable_all().build().map_err(|e| e.to_string())?;
    let st2 = st.clone();
    if let Some(d) = a.s("prewarm-dir") {
        super::prewarm::set_dir(d);
    }
    let r = rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await.map_err(|e| format!("{host}:{port}: {e}"))?;
        if super::prewarm::dir().is_some() {
            tokio::spawn(super::prewarm::run(st2.clone()));
        }
        serve(listener, st2.clone(), shutdown_signal(st2)).await.map_err(|e| e.to_string())
    });
    // 切り分けの糸などを待ちすぎない（止めは 10 秒以内に）
    rt.shutdown_timeout(Duration::from_millis(500));
    st.log_json(serde_json::json!({"ev": "exit", "ok": r.is_ok()}));
    r
}

/// 標準入力の鍵 1 行 → sha256
pub fn hash_key_stdin() -> Result<(), String> {
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line).map_err(|e| e.to_string())?;
    let k = line.trim();
    if k.len() < 16 {
        return Err("鍵は 16 文字以上".into());
    }
    println!("{}", sha256_hex(k.as_bytes()));
    Ok(())
}

/// 鍵を作る: 鍵は標準出力（そのままファイルへ: `tachyon-serve gen-key > key.txt`）、sha256 は標準エラー
pub fn gen_key() -> Result<(), String> {
    let k = util::random_key().map_err(|e| e.to_string())?;
    eprintln!("keys_sha256: {}", sha256_hex(k.as_bytes()));
    println!("{k}");
    Ok(())
}

/// 入口（tachyon-serve の main と、M1 の tachyon の "serve"）
pub fn main_with(argv: &[String], make_engine: Option<MakeEngine>) -> i32 {
    let r = match argv.first().map(String::as_str) {
        Some("hash-key") => hash_key_stdin(),
        Some("gen-key") => gen_key(),
        Some("sha") => match argv.get(1) {
            Some(p) => util::sha256_file(Path::new(p)).map(|h| println!("{h}  {}", Path::new(p).file_name().and_then(|f| f.to_str()).unwrap_or(""))).map_err(|e| e.to_string()),
            None => Err("sha <ファイル>".into()),
        },
        Some("serve") => serve_main(&argv[1..], make_engine),
        Some("-h" | "--help" | "help") => {
            eprintln!("{}", include_str!("cli.rs").lines().take_while(|l| l.starts_with("//!")).map(|l| l.trim_start_matches("//!")).collect::<Vec<_>>().join("\n"));
            Ok(())
        }
        _ => serve_main(argv, make_engine),
    };
    match r {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("tachyon-serve: {e}");
            1
        }
    }
}
