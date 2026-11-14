//! 小物: 時刻・番号・16 進・sha256・ログの出し先。
use sha2::{Digest, Sha256};
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// ログ 1 行の出し先（既定は標準エラー）。⛔ 鍵・依頼の中身は渡さない（誰が・いつ・何語だけ）
pub type LogSink = Arc<dyn Fn(&str) + Send + Sync>;

pub fn stderr_log() -> LogSink {
    Arc::new(|l: &str| eprintln!("{l}"))
}

extern "C" {
    fn _exit(code: i32) -> !;
}

/// すぐ終わる（atexit・CUDA の後始末を走らせない。壊れた CUDA の文脈の後始末で止まらないように）。標準エラーは先に流す
pub fn hard_exit(code: i32) -> ! {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    unsafe { _exit(code) }
}

pub fn unix_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// 今の時刻（UTC・RFC 3339・ミリ秒まで）
pub fn utc_now() -> String {
    let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs() as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // 日数 → 年月日（Howard Hinnant の civil_from_days）
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60,
        d.subsec_millis()
    )
}

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// 番号づけ用の 64 ビット（重ならず推し量りにくい。⛔ 鍵には使わない → `random_key`）
pub fn rand_u64() -> u64 {
    let mut h = RandomState::new().build_hasher();
    h.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
    h.write_u128(SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos());
    h.finish()
}

const ALNUM: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// 英数字の番号（chatcmpl-…・call_…・doc-…）
pub fn rand_id(prefix: &str, n: usize) -> String {
    let mut s = String::with_capacity(prefix.len() + n);
    s.push_str(prefix);
    let mut x = 0;
    for i in 0..n {
        if i % 10 == 0 {
            x = rand_u64();
        }
        s.push(ALNUM[(x % 62) as usize] as char);
        x /= 62;
    }
    s
}

/// 鍵を作る（/dev/urandom から 32 バイト → "sk-tachyon-" ＋英数字 43 文字）
pub fn random_key() -> std::io::Result<String> {
    let mut b = [0u8; 64];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut b)?;
    // 62 で割った余りの偏りを避けるため 248 以上は捨てる（64 バイトあれば 43 文字は足りる）
    let s: String = b.iter().filter(|&&x| x < 248).take(43).map(|&x| ALNUM[(x % 62) as usize] as char).collect();
    if s.len() < 43 {
        return Err(std::io::Error::other("乱数が足りない"));
    }
    Ok(format!("sk-tachyon-{s}"))
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

pub fn sha256(b: &[u8]) -> [u8; 32] {
    let d = Sha256::digest(b);
    let mut out = [0u8; 32];
    out.copy_from_slice(&d);
    out
}

pub fn sha256_hex(b: &[u8]) -> String {
    hex(&sha256(b))
}

/// 16 進 64 文字 → 32 バイト
pub fn parse_hex32(s: &str) -> Option<[u8; 32]> {
    let s = s.trim();
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(s.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(out)
}

/// ファイルの sha256（重みの版。17.5GB で十数秒）
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_and_hex() {
        let a = rand_id("call_", 24);
        let b = rand_id("call_", 24);
        assert_eq!(a.len(), 29);
        assert_ne!(a, b);
        assert!(a[5..].bytes().all(|c| c.is_ascii_alphanumeric()));
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(parse_hex32(&sha256_hex(b"abc")), Some(sha256(b"abc")));
        assert_eq!(parse_hex32("zz"), None);
        let t = utc_now();
        assert!(t.len() == 24 && t.ends_with('Z') && t.as_bytes()[10] == b'T', "{t}");
        let k = random_key().unwrap();
        assert!(k.starts_with("sk-tachyon-") && k.len() == 54);
    }
}
