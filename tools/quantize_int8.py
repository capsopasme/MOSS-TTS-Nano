#!/usr/bin/env python3
"""Builds the INT8 model pack used by the app's "INT8" variant.

  * Downloads the official ONNX exports from Hugging Face.
  * Graph clean-up on the FP32 LM graphs before quantizing:
      - fold `Identity(initializer)` nodes. The local graph unrolls the local transformer 17
        times per frame and torch exported every reuse of a weight as an Identity of the
        first copy; quantize_dynamic(MatMulConstBOnly) skips MatMuls whose B is not an
        initializer, so 16 of the 17 unrolled layers used to stay FP32.
      - local_fixed_sampled_frame: the text head computes all 16384 logits only to read two
        of them (assistant slot / audio end). The head weight is sliced to those two columns
        (exact: same dot products), which removes a 768x16384 GEMV from every frame.
  * Dynamic-quantizes the three LM graphs: MatMul weights -> int8 per-channel, activations
    quantized on the fly (MatMulInteger / DynamicQuantizeMatMul, which ORT runs with ARM
    dot-product / i8mm kernels). Attention QK^T / PV (no constant B) and embedding tables
    (Gather) stay fp32.
  * Copies the codec (MOSS-Audio-Tokenizer-Nano) unchanged: quantizing it would cost audio
    quality for little gain.

Output layout (flat files are uploaded as release assets):
  out/MOSS-TTS-Nano-100M-ONNX/...
  out/MOSS-Audio-Tokenizer-Nano-ONNX/...
"""
import json
import os
import shutil
import sys

import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

TTS_REPO = "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX"
CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"
LM_GRAPHS = ["moss_tts_prefill", "moss_tts_decode_step", "moss_tts_local_fixed_sampled_frame"]
LOCAL_GRAPH = "moss_tts_local_fixed_sampled_frame"
CODEC_FILES = [
    "codec_browser_onnx_meta.json",
    "moss_audio_tokenizer_decode_step.onnx",
    "moss_audio_tokenizer_decode_shared.data",
    "moss_audio_tokenizer_encode.onnx",
    "moss_audio_tokenizer_encode.data",
    # kept for the verification script / official runtimes
    "moss_audio_tokenizer_decode_full.onnx",
]


def _consumers(graph):
    out = {}
    for n in graph.node:
        for i in n.input:
            out.setdefault(i, []).append(n)
    return out


def _patch_inputs(graph, mapping):
    """Rename node inputs (also inside If/Loop subgraphs) according to mapping."""
    for n in graph.node:
        for k in range(len(n.input)):
            if n.input[k] in mapping:
                n.input[k] = mapping[n.input[k]]
        for a in n.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                _patch_inputs(a.g, mapping)
            elif a.type == onnx.AttributeProto.GRAPHS:
                for sub in a.graphs:
                    _patch_inputs(sub, mapping)


def fold_identity_initializers(model):
    """Replace uses of Identity(initializer) by the initializer itself and drop those nodes."""
    g = model.graph
    inits = {t.name for t in g.initializer}
    graph_outputs = {o.name for o in g.output}
    alias = {}
    for n in g.node:
        if n.op_type == "Identity" and len(n.input) == 1 and n.input[0] in inits and n.output[0] not in graph_outputs:
            alias[n.output[0]] = n.input[0]
    if not alias:
        return 0

    def resolve(name):
        while name in alias:
            name = alias[name]
        return name

    _patch_inputs(g, {k: resolve(k) for k in alias})
    keep = []
    for n in g.node:
        if n.op_type == "Identity" and n.output[0] in alias:
            continue
        c = onnx.NodeProto()
        c.CopyFrom(n)
        keep.append(c)
    del g.node[:]
    g.node.extend(keep)
    return len(alias)


def _const_int(graph, name, inits_by_name):
    """Scalar int value of a Constant node output or initializer, else None."""
    if name in inits_by_name:
        arr = numpy_helper.to_array(inits_by_name[name])
        return int(arr.reshape(-1)[0]) if arr.size == 1 else None
    for n in graph.node:
        if n.op_type == "Constant" and n.output[0] == name:
            for a in n.attribute:
                if a.name == "value":
                    arr = numpy_helper.to_array(a.t)
                    return int(arr.reshape(-1)[0]) if arr.size == 1 else None
                if a.name == "value_int":
                    return int(a.i)
    return None


def slice_text_head(model, min_vocab=4096):
    """local graph: text_logits = MatMul(h, W[H, V]) only feeds Gather(axis=1, const idx).

    Rewrites W to W[:, used] and the Gather indices to positions in `used`.
    Returns a description of what was changed, or None when the pattern isn't found.
    """
    g = model.graph
    inits = {t.name: t for t in g.initializer}
    cons = _consumers(g)
    graph_outputs = {o.name for o in g.output}
    candidates = []
    for n in g.node:
        if n.op_type != "MatMul" or n.input[1] not in inits:
            continue
        w = inits[n.input[1]]
        if len(w.dims) != 2 or w.dims[1] < min_vocab:
            continue
        # Follow the output through Cast nodes; every leaf must be Gather(axis=1/-1, const scalar).
        frontier = [n.output[0]]
        gathers = []  # (gather output name, vocab index)
        ok = True
        while frontier and ok:
            t = frontier.pop()
            if t in graph_outputs:
                ok = False
                break
            for c in cons.get(t, []):
                if c.op_type == "Cast":
                    frontier.append(c.output[0])
                elif c.op_type == "Gather" and c.input[0] == t:
                    axis = next((a.i for a in c.attribute if a.name == "axis"), 0)
                    idx = _const_int(g, c.input[1], inits)
                    if axis not in (1, -1) or idx is None or not (0 <= idx < w.dims[1]):
                        ok = False
                        break
                    gathers.append((c.output[0], idx))
                else:
                    ok = False
                    break
        # Other consumers of W itself would break.
        if ok and gathers and all(c.output[0] == n.output[0] for c in cons.get(n.input[1], [])):
            candidates.append((n.output[0], n.input[1], list(w.dims), gathers))
    if len(candidates) != 1:
        return None
    mm_out, w_name, w_dims, gathers = candidates[0]
    used = sorted({idx for _, idx in gathers})
    sliced = np.ascontiguousarray(numpy_helper.to_array(inits[w_name])[:, used])
    new_w = w_name + "_sliced"
    g.initializer.append(numpy_helper.from_array(sliced, new_w))
    pos_names = {}
    for k, idx in enumerate(used):
        pos_names[idx] = f"{w_name}_sliced_pos{k}"
        g.initializer.append(numpy_helper.from_array(np.asarray(k, dtype=np.int64), pos_names[idx]))
    gather_idx = dict(gathers)
    for n in g.node:
        if n.op_type == "MatMul" and n.output[0] == mm_out:
            n.input[1] = new_w
        elif n.op_type == "Gather" and n.output[0] in gather_idx:
            n.input[1] = pos_names[gather_idx[n.output[0]]]
    # Drop the now-unused full weight.
    for i, t in enumerate(g.initializer):
        if t.name == w_name:
            del g.initializer[i]
            break
    return f"W {w_dims} -> {list(sliced.shape)} (kept vocab ids {used})"


def matmul_coverage(model_path):
    """(# MatMul with a constant B left in fp32, # MatMulInteger/DynamicQuantizeMatMul)."""
    m = onnx.load(model_path, load_external_data=False)
    inits = {t.name for t in m.graph.initializer}
    fp32_const = sum(1 for n in m.graph.node if n.op_type in ("MatMul", "Gemm") and len(n.input) > 1 and n.input[1] in inits)
    quant = sum(1 for n in m.graph.node if n.op_type in ("MatMulInteger", "DynamicQuantizeMatMul"))
    return fp32_const, quant


def prepare_fp32(src, dst, name):
    m = onnx.load(src)
    folded = fold_identity_initializers(m)
    print(f"  {name}: folded {folded} Identity(initializer) aliases", flush=True)
    if name == LOCAL_GRAPH:
        desc = slice_text_head(m)
        if desc is None:
            raise SystemExit("text head pattern not found in the local graph (export changed?)")
        print(f"  {name}: text head sliced: {desc}", flush=True)
    onnx.save(m, dst)


def main(work="work"):
    src_tts = os.path.join(work, "src", "MOSS-TTS-Nano-100M-ONNX")
    src_codec = os.path.join(work, "src", "MOSS-Audio-Tokenizer-Nano-ONNX")
    opt_tts = os.path.join(work, "fp32opt", "MOSS-TTS-Nano-100M-ONNX")
    out_tts = os.path.join(work, "out", "MOSS-TTS-Nano-100M-ONNX")
    out_codec = os.path.join(work, "out", "MOSS-Audio-Tokenizer-Nano-ONNX")
    for d in (opt_tts, out_tts, out_codec):
        os.makedirs(d, exist_ok=True)

    if not os.path.isfile(os.path.join(src_tts, "browser_poc_manifest.json")):
        from huggingface_hub import snapshot_download
        snapshot_download(TTS_REPO, local_dir=src_tts, allow_patterns=["*.onnx", "*.data", "*.json", "tokenizer.model"])
        snapshot_download(CODEC_REPO, local_dir=src_codec, allow_patterns=["*.onnx", "*.data", "*.json"])

    report = {}
    for name in LM_GRAPHS:
        src = os.path.join(src_tts, name + ".onnx")
        opt = os.path.join(opt_tts, name + ".onnx")
        dst = os.path.join(out_tts, name + ".onnx")
        print(f"preparing {name} ...", flush=True)
        prepare_fp32(src, opt, name)
        print(f"quantizing {name} ...", flush=True)
        quantize_dynamic(
            model_input=opt,
            model_output=dst,
            op_types_to_quantize=["MatMul", "Gemm"],
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QInt8,
            use_external_data_format=False,
            extra_options={"MatMulConstBOnly": True},
        )
        left, quant = matmul_coverage(dst)
        report[name] = {"fp32_const_matmuls_left": left, "int8_matmuls": quant, "mb": round(os.path.getsize(dst) / 1e6, 1)}
        print(f"  -> {report[name]}", flush=True)
        if left:
            raise SystemExit(f"{name}: {left} MatMul(s) with constant weights were not quantized")

    for name in ["browser_poc_manifest.json", "tokenizer.model"]:
        shutil.copy2(os.path.join(src_tts, name), os.path.join(out_tts, name))
    meta = json.load(open(os.path.join(src_tts, "tts_browser_onnx_meta.json")))
    meta["external_data_files"] = {}
    meta["quantization"] = "dynamic int8 (MatMul, per-channel, MatMulConstBOnly; Identity-folded; local text head sliced)"
    json.dump(meta, open(os.path.join(out_tts, "tts_browser_onnx_meta.json"), "w"), indent=2)

    for name in CODEC_FILES:
        shutil.copy2(os.path.join(src_codec, name), os.path.join(out_codec, name))
    json.dump(report, open(os.path.join(work, "quantize_report.json"), "w"), indent=2)
    print("done:", os.path.join(work, "out"), json.dumps(report))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "work")
