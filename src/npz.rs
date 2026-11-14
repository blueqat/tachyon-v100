//! numpy の .npz を書く（圧縮なしの zip に .npy を並べる。np.load で読める）。層ごとの途中の値を tools/ref_qwen35.py の
//! 出力と同じ名前で残すため（src/bin/tachyon.rs の layers）。1 つの配列・ファイル全体とも 4GB まで（zip64 は使わない）
use std::io::{self, Write};

/// 配列の中身
pub enum Data<'a> {
    F32(&'a [f32]),
    I64(&'a [i64]),
}

impl Data<'_> {
    fn descr(&self) -> &'static str {
        match self {
            Data::F32(_) => "<f4",
            Data::I64(_) => "<i8",
        }
    }
    fn len(&self) -> usize {
        match self {
            Data::F32(v) => v.len(),
            Data::I64(v) => v.len(),
        }
    }
    fn bytes(&self) -> &[u8] {
        // 安全: f32・i64 の並びをそのままバイトとして読む（このエンジンが動くのは little endian だけ）
        unsafe {
            match self {
                Data::F32(v) => std::slice::from_raw_parts(v.as_ptr().cast(), v.len() * 4),
                Data::I64(v) => std::slice::from_raw_parts(v.as_ptr().cast(), v.len() * 8),
            }
        }
    }
}

fn crc_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    for (i, e) in t.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *e = c;
    }
    t
}

fn crc32(t: &[u32; 256], mut c: u32, b: &[u8]) -> u32 {
    c = !c;
    for &x in b {
        c = t[((c ^ x as u32) & 0xff) as usize] ^ (c >> 8);
    }
    !c
}

/// .npy の頭（版 1.0。全体の長さを 64 の倍数に揃え、最後は改行）
fn npy_header(descr: &str, shape: &[usize]) -> Vec<u8> {
    let sh = match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => format!("({})", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut d = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {sh}, }}");
    let total = 10 + d.len() + 1;
    d.push_str(&" ".repeat((64 - total % 64) % 64));
    d.push('\n');
    let mut h = b"\x93NUMPY\x01\x00".to_vec();
    h.extend((d.len() as u16).to_le_bytes());
    h.extend(d.as_bytes());
    h
}

/// arrays: (名前（.npy は付けない）, 形, 中身)
pub fn write(path: &str, arrays: &[(&str, Vec<usize>, Data)]) -> io::Result<()> {
    let t = crc_table();
    let mut f = io::BufWriter::new(std::fs::File::create(path)?);
    let mut cd = vec![];
    let mut off: u64 = 0;
    let n_entries = arrays.len();
    for (name, shape, data) in arrays {
        assert_eq!(shape.iter().product::<usize>(), data.len(), "{name}: 形と中身の数が合わない");
        let fname = format!("{name}.npy");
        let head = npy_header(data.descr(), shape);
        let body = data.bytes();
        let size = (head.len() + body.len()) as u64;
        assert!(size < u32::MAX as u64 && off + size < u32::MAX as u64, "{name}: 4GB を超える（zip64 は無い）");
        let crc = crc32(&t, crc32(&t, 0, &head), body);
        // 局所の頭: 版 20・旗 0・圧縮なし・時刻 0
        let mut lh = vec![];
        lh.extend(0x0403_4b50u32.to_le_bytes());
        lh.extend(20u16.to_le_bytes());
        lh.extend(0u16.to_le_bytes());
        lh.extend(0u16.to_le_bytes());
        lh.extend(0u16.to_le_bytes());
        lh.extend(0x21u16.to_le_bytes()); // 1980-01-01
        lh.extend(crc.to_le_bytes());
        lh.extend((size as u32).to_le_bytes());
        lh.extend((size as u32).to_le_bytes());
        lh.extend((fname.len() as u16).to_le_bytes());
        lh.extend(0u16.to_le_bytes());
        lh.extend(fname.as_bytes());
        f.write_all(&lh)?;
        f.write_all(&head)?;
        f.write_all(body)?;
        // 目次の 1 行
        cd.extend(0x0201_4b50u32.to_le_bytes());
        cd.extend(20u16.to_le_bytes());
        cd.extend(20u16.to_le_bytes());
        cd.extend(0u16.to_le_bytes());
        cd.extend(0u16.to_le_bytes());
        cd.extend(0u16.to_le_bytes());
        cd.extend(0x21u16.to_le_bytes());
        cd.extend(crc.to_le_bytes());
        cd.extend((size as u32).to_le_bytes());
        cd.extend((size as u32).to_le_bytes());
        cd.extend((fname.len() as u16).to_le_bytes());
        cd.extend(0u16.to_le_bytes());
        cd.extend(0u16.to_le_bytes());
        cd.extend(0u16.to_le_bytes());
        cd.extend(0u16.to_le_bytes());
        cd.extend(0u32.to_le_bytes());
        cd.extend((off as u32).to_le_bytes());
        cd.extend(fname.as_bytes());
        off += lh.len() as u64 + size;
    }
    assert!(off + cd.len() as u64 <= u32::MAX as u64, "npz が 4GB を超える");
    f.write_all(&cd)?;
    let mut e = vec![];
    e.extend(0x0605_4b50u32.to_le_bytes());
    e.extend(0u16.to_le_bytes());
    e.extend(0u16.to_le_bytes());
    e.extend((n_entries as u16).to_le_bytes());
    e.extend((n_entries as u16).to_le_bytes());
    e.extend((cd.len() as u32).to_le_bytes());
    e.extend((off as u32).to_le_bytes());
    e.extend(0u16.to_le_bytes());
    f.write_all(&e)?;
    f.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_known() {
        // "123456789" の CRC-32 は 0xCBF43926
        assert_eq!(crc32(&crc_table(), 0, b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn header_aligned() {
        for sh in [vec![], vec![3], vec![64, 5, 5120]] {
            assert_eq!(npy_header("<f4", &sh).len() % 64, 0);
        }
    }
}
