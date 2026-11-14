// llama.cpp（CPU だけ）で Qwen3.8-27B の MTP を llama-server の draft-mtp と同じ使い方で通し、
// MTP のグラフの節（mtp_*・h_nextn）と logits・h を書き出す。tools/ref_mtp.py lcpp の手本。
// 作り方（llama.cpp を CPU だけでビルドした物。tools/lcpp_dump.cpp と同じ）:
//   L=third_party/lcpp-src
//   g++ -O2 -std=c++17 tools/lcpp_mtp.cpp -I$L/include -I$L/ggml/include -I$L/src -L$L/build-cpu/bin \
//       -lllama -lggml -lggml-base -Wl,-rpath,$L/build-cpu/bin -o target/lcpp_mtp
// 使い方: lcpp_mtp <本体.gguf> <main | MTP の GGUF> <トークン（カンマ区切り）> <出力 dir> <k> [threads]
//   main: 本体の GGUF の blk.64（MTP の ctx は本体の model から。llama-server で -md なし）
//   MTP の GGUF: 別の model として読む（埋め込み・出力もそのファイルの物。llama-server の -md と同じ）
//   1) 本体に全トークン → 本体の argmax と h_nextn（全行）
//   2) MTP の teacher: 行 p = (S[p], pos p, h = h_nextn[p-1]、p = 0 は 0) を 1 回で（llama.cpp の process と同じ入力）。全行の logits と h
//   3) つなぎ: id_last = 本体の最後の行の argmax、位置 N、h = h_nextn[N-1] → d1、(d1, N+1, h'_1) → d2 …（k 段）
//   節は call0（teacher）・call1..k（つなぎ）に .npy で
#include "llama.h"
#include "llama-ext.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <string>
#include <sys/stat.h>
#include <vector>

struct dump_ctx {
    std::regex re;
    std::string dir;
    int call = 0;
    int n = 0;
};

static void write_npy(const std::string & path, const std::vector<int64_t> & shape, const float * data) {
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    size_t nel = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        hdr += std::to_string(shape[i]) + (shape.size() == 1 ? "," : (i + 1 < shape.size() ? ", " : ""));
        nel *= (size_t) shape[i];
    }
    hdr += "), }";
    while ((10 + hdr.size() + 1) % 64 != 0) {
        hdr += ' ';
    }
    hdr += '\n';
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "書けない: %s\n", path.c_str());
        return;
    }
    const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    fwrite(magic, 1, 8, f);
    const uint16_t hl = (uint16_t) hdr.size();
    fwrite(&hl, 2, 1, f);
    fwrite(hdr.data(), 1, hdr.size(), f);
    fwrite(data, 4, nel, f);
    fclose(f);
}

static void write_raw(const std::string & path, const void * p, size_t n) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "書けない: %s\n", path.c_str());
        exit(1);
    }
    fwrite(p, 1, n, f);
    fclose(f);
}

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * d = (dump_ctx *) ud;
    const bool want = std::regex_search(t->name, d->re);
    if (ask) {
        return want;
    }
    if (!want || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16)) {
        return true;
    }
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    const int64_t n = ggml_nelements(t);
    std::vector<float> out(n);
    int64_t k = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0, ++k) {
                    const size_t off = i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1] + i0 * t->nb[0];
                    if (t->type == GGML_TYPE_F32) {
                        memcpy(&out[k], &raw[off], 4);
                    } else {
                        ggml_fp16_t h;
                        memcpy(&h, &raw[off], 2);
                        out[k] = ggml_fp16_to_fp32(h);
                    }
                }
            }
        }
    }
    std::vector<int64_t> shape;
    for (int i = ggml_n_dims(t) - 1; i >= 0; --i) {
        shape.push_back(t->ne[i]);
    }
    std::string name = t->name;
    for (auto & c : name) {
        if (c == '/' || c == ' ' || c == '(' || c == ')') {
            c = '_';
        }
    }
    const std::string dir = d->dir + "/call" + std::to_string(d->call);
    mkdir(dir.c_str(), 0755);
    write_npy(dir + "/" + name + ".npy", shape, out.data());
    d->n++;
    return true;
}

static int argmax(const float * x, int n) {
    int b = 0;
    for (int i = 1; i < n; ++i) {
        if (x[i] > x[b]) {
            b = i;
        }
    }
    return b;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "使い方: %s <本体.gguf> <main|MTP.gguf> <トークン> <出力 dir> <k> [threads]\n", argv[0]);
        return 1;
    }
    const std::string src = argv[2];
    const bool same = src == "main";
    std::vector<llama_token> S;
    for (const char * p = argv[3]; *p;) {
        S.push_back((llama_token) strtol(p, (char **) &p, 10));
        if (*p == ',') {
            ++p;
        }
    }
    const int N = (int) S.size();
    const std::string out = argv[4];
    const int K = atoi(argv[5]);
    const int nth = argc > 6 ? atoi(argv[6]) : 16;
    mkdir(out.c_str(), 0755);
    dump_ctx d{std::regex("^(mtp_|h_nextn|result_output)"), out};

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mtp = same;
    llama_model * mt = llama_model_load_from_file(argv[1], mp);
    if (!mt) {
        return 1;
    }
    llama_model * md = mt;
    if (!same) {
        auto mp2 = mp;
        mp2.load_mtp = true;
        md = llama_model_load_from_file(src.c_str(), mp2);
        if (!md) {
            return 1;
        }
    }
    const int n_embd = llama_model_n_embd(mt);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mt));

    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 4096;
    cp.n_ubatch = 4096;
    cp.n_seq_max = 1;
    cp.n_threads = nth;
    cp.n_threads_batch = nth;
    llama_context * ct = llama_init_from_model(mt, cp);
    if (!ct) {
        return 1;
    }
    llama_set_embeddings_nextn(ct, true, /*masked*/ false);

    auto cd = cp;
    cd.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cd.n_rs_seq = 0;
    cd.ctx_other = ct;
    cd.cb_eval = cb;
    cd.cb_eval_user_data = &d;
    llama_context * cx = llama_init_from_model(md, cd);
    if (!cx) {
        return 1;
    }
    llama_set_embeddings_nextn(cx, true, /*masked*/ true);

    // 1) 本体
    llama_batch_ext * b = llama_batch_ext_init(ct);
    for (int p = 0; p < N; ++p) {
        const int idx = llama_batch_ext_add_token(b, 0, S[p]);
        const llama_pos pos = p;
        llama_batch_ext_set_pos(b, idx, &pos);
        llama_batch_ext_set_output_logits(b, idx, true);
    }
    if (llama_process(ct, LLAMA_PROCESS_TYPE_DECODE, b) != 0) {
        fprintf(stderr, "本体の process に失敗\n");
        return 1;
    }
    std::vector<float> hT((size_t) N * n_embd);
    std::vector<int32_t> aT(N);
    for (int p = 0; p < N; ++p) {
        memcpy(&hT[(size_t) p * n_embd], llama_get_embeddings_nextn_ith(ct, p), n_embd * 4);
        aT[p] = argmax(llama_get_logits_ith(ct, p), n_vocab);
    }
    write_raw(out + "/tokens.i32", S.data(), N * 4);
    write_raw(out + "/tgt_h.f32", hT.data(), hT.size() * 4);
    write_raw(out + "/tgt_argmax.i32", aT.data(), N * 4);
    fprintf(stderr, "本体: %d 語、最後の argmax %d\n", N, aT[N - 1]);

    // 2) MTP の teacher（process と同じ入力。行ごとの logits も出す）
    std::vector<float> zero(n_embd, 0.0f);
    d.call = 0;
    llama_batch_ext * bm = llama_batch_ext_init(cx);
    for (int p = 0; p < N; ++p) {
        const int idx = llama_batch_ext_add_token(bm, 0, S[p]);
        const llama_pos pos = p;
        llama_batch_ext_set_pos(bm, idx, &pos);
        const float * h = p == 0 ? zero.data() : &hT[(size_t) (p - 1) * n_embd];
        if (!llama_batch_ext_set_embd_state(bm, idx, {h, 1, (size_t) n_embd})) {
            fprintf(stderr, "embd_state を置けない\n");
            return 1;
        }
        llama_batch_ext_set_output_logits(bm, idx, true);
    }
    if (llama_process(cx, LLAMA_PROCESS_TYPE_DECODE, bm) != 0) {
        fprintf(stderr, "MTP の process に失敗\n");
        return 1;
    }
    std::vector<float> lM((size_t) N * n_vocab), hM((size_t) N * n_embd);
    for (int p = 0; p < N; ++p) {
        memcpy(&lM[(size_t) p * n_vocab], llama_get_logits_ith(cx, p), (size_t) n_vocab * 4);
        memcpy(&hM[(size_t) p * n_embd], llama_get_embeddings_nextn_ith(cx, p), n_embd * 4);
    }
    write_raw(out + "/mtp_logits.f32", lM.data(), lM.size() * 4);
    write_raw(out + "/mtp_h.f32", hM.data(), hM.size() * 4);

    // 3) つなぎ（draft と同じ: 1 段 1 行、h は前の段の MTP の h_nextn）
    std::vector<float> h(hT.end() - n_embd, hT.end());
    llama_token t = aT[N - 1];
    std::vector<int> drafts;
    std::vector<float> lC, hC;
    for (int j = 0; j < K; ++j) {
        d.call = j + 1;
        llama_batch_ext_clear(bm);
        const int idx = llama_batch_ext_add_token(bm, 0, t);
        const llama_pos pos = N + j;
        llama_batch_ext_set_pos(bm, idx, &pos);
        llama_batch_ext_set_embd_state(bm, idx, {h.data(), 1, (size_t) n_embd});
        llama_batch_ext_set_output_logits(bm, idx, true);
        if (llama_process(cx, LLAMA_PROCESS_TYPE_DECODE, bm) != 0) {
            fprintf(stderr, "つなぎ %d に失敗\n", j);
            return 1;
        }
        const float * lg = llama_get_logits_ith(cx, idx);
        lC.insert(lC.end(), lg, lg + n_vocab);
        const float * hn = llama_get_embeddings_nextn_ith(cx, idx);
        h.assign(hn, hn + n_embd);
        hC.insert(hC.end(), h.begin(), h.end());
        t = argmax(lg, n_vocab);
        drafts.push_back(t);
    }
    write_raw(out + "/chain_logits.f32", lC.data(), lC.size() * 4);
    write_raw(out + "/chain_h.f32", hC.data(), hC.size() * 4);
    FILE * f = fopen((out + "/chain.json").c_str(), "w");
    fprintf(f, "{\"id_last\": %d, \"drafts\": [", aT[N - 1]);
    for (int j = 0; j < K; ++j) {
        fprintf(f, "%s%d", j ? ", " : "", drafts[j]);
    }
    fprintf(f, "]}\n");
    fclose(f);
    f = fopen((out + "/meta.json").c_str(), "w");
    fprintf(f, "{\"src\": \"%s\", \"path\": \"%s\", \"n\": %d, \"n_embd\": %d, \"n_vocab\": %d}\n", same ? "main" : "file",
            same ? argv[1] : src.c_str(), N, n_embd, n_vocab);
    fclose(f);
    fprintf(stderr, "MTP: 節 %d 個、下書き", d.n);
    for (int x : drafts) {
        fprintf(stderr, " %d", x);
    }
    fprintf(stderr, "\n");
    llama_batch_ext_free(b);
    llama_batch_ext_free(bm);
    llama_free(cx);
    llama_free(ct);
    if (md != mt) {
        llama_model_free(md);
    }
    llama_model_free(mt);
    return 0;
}
