//! ggml の量子化の型番号（kernels/quant.cuh の QType と揃える）。
//! Q3_K・IQ3_S は GEMV 2 版（kernels/gemv.cu）では扱わない。g3 が詰め直すときに Q6_K に直す。

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum QType {
    F32 = 0,
    F16 = 1,
    Q5_1 = 7,
    Q8_0 = 8,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    IQ3XXS = 18,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    BF16 = 30,
    Q2_0 = 42,
}

impl QType {
    pub fn from_name(s: &str) -> Option<Self> {
        use QType::*;
        Some(match s {
            "F32" => F32,
            "F16" => F16,
            "Q5_1" => Q5_1,
            "Q8_0" => Q8_0,
            "Q3_K" => Q3K,
            "Q4_K" => Q4K,
            "Q5_K" => Q5K,
            "Q6_K" => Q6K,
            "IQ3_XXS" => IQ3XXS,
            "IQ4_NL" => IQ4NL,
            "IQ3_S" => IQ3S,
            "IQ2_S" => IQ2S,
            "IQ4_XS" => IQ4XS,
            "BF16" => BF16,
            "Q2_0" => Q2_0,
            _ => return None,
        })
    }

    pub fn from_id(id: u32) -> Option<Self> {
        use QType::*;
        [F32, F16, Q5_1, Q8_0, Q3K, Q4K, Q5K, Q6K, IQ3XXS, IQ4NL, IQ3S, IQ2S, IQ4XS, BF16, Q2_0]
            .into_iter()
            .find(|t| *t as u32 == id)
    }

    /// k 要素の 1 行のバイト数
    pub fn row_bytes(self, k: usize) -> usize {
        unsafe { crate::cuda::ffi::infer_row_bytes(self as i32, k as i64) as usize }
    }
}
