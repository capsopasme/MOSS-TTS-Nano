#!/usr/bin/env python3
"""Builds the INT8 model pack used by the app's "INT8" variant.

  * Downloads the official ONNX exports from Hugging Face.
  * Dynamic-quantizes the three LM graphs (prefill / decode_step / local_fixed_sampled_frame):
    MatMul/Gemm weights -> int8 per-channel, activations quantized on the fly
    (MatMulInteger, which ORT runs with ARM dot-product / i8mm kernels).
    Only MatMuls with constant B are touched (MatMulConstBOnly), so attention QK^T / PV
    stay fp32. Embedding tables (Gather) stay fp32.
  * Copies the codec (MOSS-Audio-Tokenizer-Nano) unchanged: it is small and quantizing it
    would cost audio quality for little gain.

Output layout (flat files are uploaded as release assets):
  out/MOSS-TTS-Nano-100M-ONNX/...
  out/MOSS-Audio-Tokenizer-Nano-ONNX/...
"""
import json
import os
import shutil
import sys

from huggingface_hub import snapshot_download
from onnxruntime.quantization import QuantType, quantize_dynamic

TTS_REPO = "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX"
CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"
LM_GRAPHS = ["moss_tts_prefill", "moss_tts_decode_step", "moss_tts_local_fixed_sampled_frame"]
CODEC_FILES = [
    "codec_browser_onnx_meta.json",
    "moss_audio_tokenizer_decode_step.onnx",
    "moss_audio_tokenizer_decode_shared.data",
    "moss_audio_tokenizer_encode.onnx",
    "moss_audio_tokenizer_encode.data",
    # kept for the verification script / official runtimes
    "moss_audio_tokenizer_decode_full.onnx",
]


def main(work="work"):
    src_tts = os.path.join(work, "src", "MOSS-TTS-Nano-100M-ONNX")
    src_codec = os.path.join(work, "src", "MOSS-Audio-Tokenizer-Nano-ONNX")
    out_tts = os.path.join(work, "out", "MOSS-TTS-Nano-100M-ONNX")
    out_codec = os.path.join(work, "out", "MOSS-Audio-Tokenizer-Nano-ONNX")
    os.makedirs(out_tts, exist_ok=True)
    os.makedirs(out_codec, exist_ok=True)

    snapshot_download(TTS_REPO, local_dir=src_tts, allow_patterns=["*.onnx", "*.data", "*.json", "tokenizer.model"])
    snapshot_download(CODEC_REPO, local_dir=src_codec, allow_patterns=["*.onnx", "*.data", "*.json"])

    for name in LM_GRAPHS:
        src = os.path.join(src_tts, name + ".onnx")
        dst = os.path.join(out_tts, name + ".onnx")
        print(f"quantizing {name} ...", flush=True)
        quantize_dynamic(
            model_input=src,
            model_output=dst,
            op_types_to_quantize=["MatMul", "Gemm"],
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QInt8,
            use_external_data_format=False,
            extra_options={"MatMulConstBOnly": True},
        )
        print(f"  {os.path.getsize(src) / 1e6:.1f} MB graph -> {os.path.getsize(dst) / 1e6:.1f} MB (self-contained)")

    for name in ["browser_poc_manifest.json", "tokenizer.model"]:
        shutil.copy2(os.path.join(src_tts, name), os.path.join(out_tts, name))
    meta = json.load(open(os.path.join(src_tts, "tts_browser_onnx_meta.json")))
    meta["external_data_files"] = {}
    meta["quantization"] = "dynamic int8 (MatMul/Gemm, per-channel, MatMulConstBOnly)"
    json.dump(meta, open(os.path.join(out_tts, "tts_browser_onnx_meta.json"), "w"), indent=2)

    for name in CODEC_FILES:
        shutil.copy2(os.path.join(src_codec, name), os.path.join(out_codec, name))

    # Reference copy of the fp32 originals for the verification step.
    print("done:", os.path.join(work, "out"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "work")
