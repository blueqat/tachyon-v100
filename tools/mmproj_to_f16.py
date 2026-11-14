#!/usr/bin/env python3
"""mmproj の GGUF の Q8_0 の重みを F16 に戻した写しを作る（Tachyon の src/vision.rs は行列を F16 で読む）。
Qwen3.6-35B-A3B の mmproj は Q8_0 しか置いていない（llama-server が Q8_0 で使うのと同じ値＝Q8_0 を戻した値。新しい丸めは F16 の分だけ）。

  .venv/bin/python tools/mmproj_to_f16.py weights/q36/mmproj-Qwen3.6-35B-A3B-Q8_0.gguf weights/q36/mmproj-Qwen3.6-35B-A3B-F16fromQ8.gguf
"""
import sys

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType
from gguf.quants import dequantize


def main(src, dst):
    r = GGUFReader(src)
    arch = r.fields["general.architecture"]
    arch = bytes(arch.parts[arch.data[0]]).decode()
    w = GGUFWriter(dst, arch)
    for name, f in r.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        t = f.types[0]
        if t == GGUFValueType.ARRAY:
            et = f.types[1]
            if et == GGUFValueType.STRING:
                v = [bytes(f.parts[i]).decode() for i in f.data]
            else:
                v = [f.parts[i].tolist()[0] for i in f.data]
            w.add_key_value(name, v, t, sub_type=et)
        elif t == GGUFValueType.STRING:
            w.add_key_value(name, bytes(f.parts[f.data[0]]).decode(), t)
        else:
            w.add_key_value(name, f.parts[f.data[0]].tolist()[0], t)
    n_conv = 0
    for t in r.tensors:
        shape = [int(x) for x in reversed(t.shape.tolist())]   # GGUF は内側が先 → numpy の並び
        if t.tensor_type == GGMLQuantizationType.Q8_0:
            a = dequantize(t.data, t.tensor_type).reshape(shape).astype(np.float16)
            w.add_tensor(t.name, a, raw_dtype=GGMLQuantizationType.F16)
            n_conv += 1
        else:
            a = np.asarray(t.data)
            w.add_tensor(t.name, a, raw_shape=shape if t.tensor_type not in (GGMLQuantizationType.F32, GGMLQuantizationType.F16) else None,
                         raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"Q8_0 → F16: {n_conv} 本・全部 {len(r.tensors)} 本 → {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
