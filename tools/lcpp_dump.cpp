// llama.cpp の途中の値（グラフの節）をそのまま .npy に書き出す。CPU 参照・CUDA エンジンの答え合わせ用。
// 作り方（V100 の機械で。llama.cpp は unsloth b12209-mix-b84c2bf の源を CPU だけでビルドした物。
// llama.cpp を -DGGML_CUDA=OFF でビルドした物）:
//   L=third_party/lcpp-src
//   g++ -O2 -std=c++17 tools/lcpp_dump.cpp -I$L/include -I$L/ggml/include -L$L/build-cpu/bin \
//       -lllama -lggml -lggml-base -Wl,-rpath,$L/build-cpu/bin -o target/lcpp_dump
// 使い方: lcpp_dump <model.gguf> <トークン（カンマ区切り）> <出力の dir> <名前の正規表現> [threads] [n_out]
//   1 つの ubatch で通す（同じ名前の節は後の物が残る）。最後の位置の logits も logits.npy に。
//   n_out（既定 1）: 最後の n_out 位置の logits を出す（logits.npy は (n_out, 語の数)。1 のときは今まで通り (語の数,)）
//   ⚠️ 最後の層の l_out・result_norm は出力する位置（最後の n_out 語）だけに切られている
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <string>
#include <vector>

struct dump_ctx {
    std::regex re;
    std::string dir;
    int n = 0;
};

// .npy（v1.0、little endian、C 順）。形は ggml の ne を逆に
static void write_npy(const std::string & path, const char * descr, size_t esz, const std::vector<int64_t> & shape,
                      const void * data) {
    std::string hdr = "{'descr': '" + std::string(descr) + "', 'fortran_order': False, 'shape': (";
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
    fwrite(data, esz, nel, f);
    fclose(f);
}

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * d = (dump_ctx *) ud;
    const bool want = std::regex_search(t->name, d->re);
    if (ask) {
        return want;
    }
    if (!want) {
        return true;
    }
    const bool is_f = t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16;
    const bool is_i = t->type == GGML_TYPE_I32;
    if (!is_f && !is_i) {
        return true;
    }
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    const int64_t n = ggml_nelements(t);
    std::vector<float> out_f;
    std::vector<int32_t> out_i;
    if (is_f) {
        out_f.resize(n);
    } else {
        out_i.resize(n);
    }
    int64_t k = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0, ++k) {
                    const size_t off = i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1] + i0 * t->nb[0];
                    if (t->type == GGML_TYPE_F32) {
                        memcpy(&out_f[k], &raw[off], 4);
                    } else if (t->type == GGML_TYPE_F16) {
                        ggml_fp16_t h;
                        memcpy(&h, &raw[off], 2);
                        out_f[k] = ggml_fp16_to_fp32(h);
                    } else {
                        memcpy(&out_i[k], &raw[off], 4);
                    }
                }
            }
        }
    }
    int nd = ggml_n_dims(t);
    std::vector<int64_t> shape;
    for (int i = nd - 1; i >= 0; --i) {
        shape.push_back(t->ne[i]);
    }
    std::string name = t->name;
    for (auto & c : name) {
        if (c == '/' || c == ' ' || c == '(' || c == ')') {
            c = '_';
        }
    }
    const std::string path = d->dir + "/" + name + ".npy";
    if (is_f) {
        write_npy(path, "<f4", 4, shape, out_f.data());
    } else {
        write_npy(path, "<i4", 4, shape, out_i.data());
    }
    d->n++;
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "使い方: %s <model.gguf> <トークン> <出力の dir> <正規表現> [threads] [n_out]\n", argv[0]);
        return 1;
    }
    std::vector<llama_token> toks;
    for (const char * p = argv[2]; *p;) {
        toks.push_back((llama_token) strtol(p, (char **) &p, 10));
        if (*p == ',') {
            ++p;
        }
    }
    dump_ctx d{std::regex(argv[4]), argv[3]};
    const int nth = argc > 5 ? atoi(argv[5]) : 16;
    const int n_tok = (int) toks.size();
    if (n_tok == 0) {
        fprintf(stderr, "トークンが空\n");
        return 1;
    }
    const int n_out = argc > 6 ? std::max(1, std::min(atoi(argv[6]), n_tok)) : 1;
    const uint32_t n_ctx = std::max<uint32_t>(4096, (uint32_t) (n_tok + 255) / 256 * 256);  // 1 つの ubatch に入れる

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        return 1;
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.n_ubatch = n_ctx;
    cp.n_seq_max = 1;
    cp.n_threads = nth;
    cp.n_threads_batch = nth;
    cp.cb_eval = cb;
    cp.cb_eval_user_data = &d;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        return 1;
    }
    // n_out が 1 なら今まで通り get_one（出力は最後の 1 語）。それより多いときは最後の n_out 語に印を付ける
    llama_batch b = llama_batch_get_one(toks.data(), (int32_t) n_tok);
    if (n_out > 1) {
        b = llama_batch_init(n_tok, 0, 1);
        for (int i = 0; i < n_tok; ++i) {
            b.token[i] = toks[i];
            b.pos[i] = i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = i >= n_tok - n_out;
        }
        b.n_tokens = n_tok;
    }
    const int rc = llama_decode(ctx, b);
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (rc == 0 && n_out == 1) {
        write_npy(d.dir + "/logits.npy", "<f4", 4, {nv}, llama_get_logits_ith(ctx, -1));
    } else if (rc == 0) {
        std::vector<float> lg((size_t) n_out * nv);
        for (int k = 0; k < n_out; ++k) {
            memcpy(&lg[(size_t) k * nv], llama_get_logits_ith(ctx, n_tok - n_out + k), (size_t) nv * sizeof(float));
        }
        write_npy(d.dir + "/logits.npy", "<f4", 4, {n_out, nv}, lg.data());
    }
    if (n_out > 1) {
        llama_batch_free(b);
    }
    if (rc) {
        fprintf(stderr, "decode に失敗\n");
        return 1;
    }
    fprintf(stderr, "書いた節 %d 個 → %s\n", d.n, d.dir.c_str());
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
