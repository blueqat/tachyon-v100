//! OpenAI 互換の口だけを起こす（中身は --fake の決まった語。M1 のモデルは tachyon の serve で同じ口に載る）。
//!   tachyon-serve --fake --hf weights/q38-27b/hf --keys keys.json --port 8960
//!   tachyon-serve hash-key < key.txt     # 鍵 → sha256（keys.json に書く値）
//!   tachyon-serve gen-key > key.txt      # 鍵を作る（sha256 は標準エラー）
//!   tachyon-serve sha <重み.gguf>        # 重みの版（<重み>.sha256 に置く）
fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(tachyon_v100::server::cli::main_with(&argv, None));
}
