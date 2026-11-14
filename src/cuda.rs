//! CUDA ランタイムと自前のカーネルを呼ぶ薄い層。
use std::ffi::c_void;

pub mod ffi {
    use std::ffi::c_void;
    pub type Stream = *mut c_void;
    extern "C" {
        pub fn cudaSetDevice(dev: i32) -> i32;
        pub fn cudaMalloc(p: *mut *mut c_void, size: usize) -> i32;
        pub fn cudaFree(p: *mut c_void) -> i32;
        pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, n: usize, kind: i32) -> i32;
        pub fn cudaMemsetAsync(p: *mut c_void, v: i32, n: usize, s: Stream) -> i32;
        pub fn cudaDeviceSynchronize() -> i32;
        pub fn cudaGetErrorString(e: i32) -> *const std::ffi::c_char;
        pub fn cudaEventCreate(e: *mut *mut c_void) -> i32;
        pub fn cudaEventRecord(e: *mut c_void, s: Stream) -> i32;
        pub fn cudaEventSynchronize(e: *mut c_void) -> i32;
        pub fn cudaEventElapsedTime(ms: *mut f32, a: *mut c_void, b: *mut c_void) -> i32;

        pub fn infer_gemv(qtype: i32, w: *const c_void, k: i32, n: i32, x: *const f32, y: *mut f32, b: i32, s: Stream) -> i32;
        pub fn infer_moe_gemv(
            qtype: i32, w: *const c_void, k: i32, n: i32, ids: *const i32, n_jobs: i32, x_div: i32,
            x: *const f32, y: *mut f32, s: Stream,
        ) -> i32;
        pub fn infer_dequant(qtype: i32, w: *const c_void, k: i32, n_rows: i64, y: *mut f32, s: Stream) -> i32;
        pub fn infer_row_bytes(qtype: i32, k: i64) -> i64;

        // プレフィルの GEMM（kernels/gemm.cu）。X は fp16。mode: 0 = Y を f32、1 = f16、2 = f32 の Y に足す（残差）
        // cfg は -1 で自動、splits は 0 で自動。ws は split-K の作業場（infer_gemm_ws_bytes の大きさ。null なら分けない）
        pub fn infer_gemm_ws_bytes(k: i32, n: i32, m: i32) -> i64;
        pub fn infer_gemm(
            qtype: i32, w: *const c_void, k: i32, n: i32, m: i32, x: *const c_void, y: *mut c_void, mode: i32, ws: *mut c_void,
            ws_bytes: i64, s: Stream,
        ) -> i32;
        pub fn infer_gemm_ex(
            qtype: i32, w: *const c_void, k: i32, n: i32, m: i32, x: *const c_void, y: *mut c_void, mode: i32, cfg: i32,
            splits: i32, ws: *mut c_void, ws_bytes: i64, s: Stream,
        ) -> i32;
        // gate と up をまとめて Y[m, n] = silu(X·Wg^T) * (X·Wu^T)（fp16）
        pub fn infer_gemm_swiglu(
            tg: i32, wg: *const c_void, tu: i32, wu: *const c_void, k: i32, n: i32, m: i32, x: *const c_void, y: *mut c_void,
            cfg: i32, s: Stream,
        ) -> i32;
        // 同じ X に何本か（qkv と gate、q・k・v）。行列 i は Y_i[m, ns[i]]
        pub fn infer_gemm_multi(
            nmat: i32, qtypes: *const i32, ws: *const *const c_void, ns: *const i32, ys: *const *mut c_void, k: i32, m: i32,
            x: *const c_void, mode: i32, cfg: i32, s: Stream,
        ) -> i32;
        // g3（kernels/gemv3.cu）: 自前の並びに詰め直した重みの GEMV（密なモデル向け。B = 1..32）
        pub fn infer_g3_type(qtype: i32) -> i32;
        pub fn infer_g3_bytes(qtype: i32, k: i32, n: i32) -> i64;
        pub fn infer_g3_repack(qtype: i32, src: *const c_void, dst: *mut c_void, k: i32, n: i32, s: Stream) -> i32;
        pub fn infer_g3_dequant(qtype: i32, w: *const c_void, k: i32, n: i32, y: *mut f32, s: Stream) -> i32;
        pub fn infer_g3_xprep_bytes(k: i32, b: i32) -> i64;
        pub fn infer_g3_xprep(x: *const f32, k: i32, b: i32, xp: *mut c_void, s: Stream) -> i32;
        pub fn infer_g3_gemv(
            qtype: i32, w: *const c_void, k: i32, n: i32, xp: *const c_void, y: *mut f32, b: i32, wk: i32, add: i32, s: Stream,
        ) -> i32;
        pub fn infer_g3_gemv_multi(
            nm: i32, qtype: *const i32, w: *const *const c_void, n: *const i32, y: *const *mut f32, k: i32, xp: *const c_void, b: i32, wk: i32,
            add: i32, s: Stream,
        ) -> i32;
        // 足し込み（add）を決まった順で: ws は infer_g3_ws_bytes() 以上・最初に 0 で埋める・1 つの stream に 1 つ
        // （null なら k をブロックで分けない）。順を決めるかは infer_g3_set_det（既定 1、環境変数 INFER_G3_DET）
        pub fn infer_g3_ws_bytes() -> i64;
        pub fn infer_g3_set_det(on: i32) -> i32;
        pub fn infer_g3_gemv_ws(
            qtype: i32, w: *const c_void, k: i32, n: i32, xp: *const c_void, y: *mut f32, b: i32, wk: i32, add: i32, ws: *mut c_void,
            ws_bytes: i64, s: Stream,
        ) -> i32;
        pub fn infer_g3_gemv_multi_ws(
            nm: i32, qtype: *const i32, w: *const *const c_void, n: *const i32, y: *const *mut f32, k: i32, xp: *const c_void, b: i32, wk: i32,
            add: i32, ws: *mut c_void, ws_bytes: i64, s: Stream,
        ) -> i32;
        // 確かめ用: 起動の形 out = [wk, ksp, ns, cmax, gx, smem, det]
        pub fn infer_g3_plan(k: i32, n: i32, b: i32, wk: i32, add: i32, ws_bytes: i64, out: *mut i32) -> i32;
        pub fn infer_g3_swiglu(
            tg: i32, wg: *const c_void, tu: i32, wu: *const c_void, k: i32, n: i32, xp: *const c_void, h: *mut f32, hp: *mut c_void, b: i32,
            s: Stream,
        ) -> i32;
    }
}

pub fn check(code: i32, what: &str) {
    if code != 0 {
        let msg = if code > 0 {
            unsafe { std::ffi::CStr::from_ptr(ffi::cudaGetErrorString(code)).to_string_lossy().into_owned() }
        } else {
            format!("自前のカーネルのエラー {code}")
        };
        panic!("{what}: {msg}");
    }
}

/// GPU 上の生のバッファ
pub struct DevBuf {
    pub ptr: *mut c_void,
    pub len: usize,
}

impl DevBuf {
    pub fn new(len: usize) -> Self {
        let mut ptr = std::ptr::null_mut();
        check(unsafe { ffi::cudaMalloc(&mut ptr, len.max(1)) }, "cudaMalloc");
        DevBuf { ptr, len }
    }
    pub fn from_bytes(b: &[u8]) -> Self {
        let d = Self::new(b.len());
        check(unsafe { ffi::cudaMemcpy(d.ptr, b.as_ptr().cast(), b.len(), 1) }, "cudaMemcpy H2D");
        d
    }
    pub fn from_f32(v: &[f32]) -> Self {
        Self::from_bytes(unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), v.len() * 4) })
    }
    pub fn from_i32(v: &[i32]) -> Self {
        Self::from_bytes(unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), v.len() * 4) })
    }
    pub fn to_f32(&self) -> Vec<f32> {
        let mut v = vec![0f32; self.len / 4];
        check(unsafe { ffi::cudaMemcpy(v.as_mut_ptr().cast(), self.ptr, self.len, 2) }, "cudaMemcpy D2H");
        v
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut v = vec![0u8; self.len];
        check(unsafe { ffi::cudaMemcpy(v.as_mut_ptr().cast(), self.ptr, self.len, 2) }, "cudaMemcpy D2H");
        v
    }
    pub fn f32p(&self) -> *mut f32 {
        self.ptr.cast()
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        unsafe { ffi::cudaFree(self.ptr) };
    }
}

pub fn sync() {
    check(unsafe { ffi::cudaDeviceSynchronize() }, "cudaDeviceSynchronize");
}

/// GPU 上で f を走らせた時間（ミリ秒）
pub fn time_ms(mut f: impl FnMut()) -> f32 {
    unsafe {
        let (mut a, mut b) = (std::ptr::null_mut(), std::ptr::null_mut());
        ffi::cudaEventCreate(&mut a);
        ffi::cudaEventCreate(&mut b);
        ffi::cudaEventRecord(a, std::ptr::null_mut());
        f();
        ffi::cudaEventRecord(b, std::ptr::null_mut());
        ffi::cudaEventSynchronize(b);
        let mut ms = 0f32;
        ffi::cudaEventElapsedTime(&mut ms, a, b);
        ms
    }
}
