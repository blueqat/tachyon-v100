// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 blueqat Inc.
//! V100（sm_70）向けの LLM 推論エンジン。
pub mod attn;
pub mod cuda;
pub mod gdn;
pub mod gguf;
pub mod inkling;
pub mod kv;
pub mod model;
pub mod mtp;
pub mod npz;
pub mod qtype;
pub mod server;
pub mod spec;
pub mod template;
pub mod tokenizer;
pub mod vision;

/// 重みの置き場の既定（環境変数 `TACHYON_WEIGHTS`、無ければ今の場所の `weights`）の下の `rel`
pub fn weights_path(rel: &str) -> String {
    let d = std::env::var("TACHYON_WEIGHTS").unwrap_or_else(|_| "weights".into());
    format!("{d}/{rel}")
}
