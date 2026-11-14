// kernels/*.cu を nvcc で sm_70 向けにまとめ、静的ライブラリにしてつなぐ。
use std::{env, path::PathBuf, process::Command};

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cuda = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let nvcc = format!("{cuda}/bin/nvcc");
    // 試しに nvcc へ足す指定（例: INFER_NVCC_FLAGS="-DG3_XPF=1"）
    println!("cargo:rerun-if-env-changed=INFER_NVCC_FLAGS");
    let extra: Vec<String> = env::var("INFER_NVCC_FLAGS").unwrap_or_default().split_whitespace().map(String::from).collect();
    let mut objs = vec![];
    for src in ["kernels/gemv.cu", "kernels/gdn.cu", "kernels/attn.cu", "kernels/gemm.cu", "kernels/gemmg3.cu", "kernels/gemv3.cu", "kernels/kvcopy.cu",
                "kernels/model.cu"] {
        println!("cargo:rerun-if-changed={src}");
        let obj = out.join(PathBuf::from(src).file_stem().unwrap()).with_extension("o");
        let st = Command::new(&nvcc)
            .args(["-O3", "-std=c++17", "-arch=sm_70", "-Xcompiler", "-fPIC"])
            .args(&extra)
            .args(["-c", src, "-o"])
            .arg(&obj)
            .status()
            .expect("nvcc が動かない");
        assert!(st.success(), "nvcc 失敗: {src}");
        objs.push(obj);
    }
    // MTP の小さなカーネル（mtp-head）。上の表とは別に置く（ほかの作業が上の表に足しても、ぶつからないように）
    for src in ["kernels/mtp.cu", "kernels/vision.cu", "kernels/moe.cu", "kernels/sample.cu"] {
        println!("cargo:rerun-if-changed={src}");
        let obj = out.join(PathBuf::from(src).file_stem().unwrap()).with_extension("o");
        let st = Command::new(&nvcc).args(["-O3", "-std=c++17", "-arch=sm_70", "-Xcompiler", "-fPIC"]).args(&extra).args(["-c", src, "-o"]).arg(&obj).status().expect("nvcc が動かない");
        assert!(st.success(), "nvcc 失敗: {src}");
        objs.push(obj);
    }
    for h in ["kernels/quant.cuh", "kernels/tables.cuh", "kernels/g3.cuh"] {
        println!("cargo:rerun-if-changed={h}");
    }
    let lib = out.join("libinferk.a");
    let st = Command::new("ar").arg("crs").arg(&lib).args(&objs).status().unwrap();
    assert!(st.success());
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=inferk");
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
