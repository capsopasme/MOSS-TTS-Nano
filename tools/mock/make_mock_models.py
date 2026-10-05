#!/usr/bin/env python3
"""Builds tiny mock ONNX graphs with the *exact* I/O contract of the official
MOSS-TTS-Nano ONNX export (prefill / decode_step / local_fixed_sampled_frame / codec
decode_step), plus matching manifest/meta JSON. Used to test the Android engine's
plumbing (KV-cache hand-off, pinned outputs, seen-mask, codec state, chunking) on a
desktop JVM without downloading the 700 MB model.

Self-checking behaviour baked into the graphs:
  * decode_step global_hidden = [past_valid_lengths+1, actual_kv_len+1, 0...]
  * local frame: frame[0]=hidden[0], frame[1]=hidden[1];
    should_continue = (frames_seen_in_chunk < N) AND (hidden[0] == hidden[1])
    -> a wrong past_valid_lengths or a stale seen-mask ends the chunk early.
  * codec: audio[i] = (frames_already_decoded*3840 + i) * 1e-6 and lengths = F*3840,
    so the test can check stream state continuity and per-chunk resets.

No dependency on the `onnx` package: protobuf is encoded by hand.
"""
import json
import os
import struct
import sys

# ---------------------------------------------------------------- protobuf encoding
def _varint(n):
    out = bytearray()
    n &= (1 << 64) - 1
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)

def f_varint(field, v):
    return _varint(field << 3) + _varint(v)

def f_bytes(field, b):
    if isinstance(b, str):
        b = b.encode()
    return _varint((field << 3) | 2) + _varint(len(b)) + b

def f_float(field, v):
    return _varint((field << 3) | 5) + struct.pack("<f", v)

FLOAT, INT32, INT64, BOOL = 1, 6, 7, 9

def tensor(name, dtype, dims, values):
    fmt = {FLOAT: "f", INT32: "i", INT64: "q", BOOL: "?"}[dtype]
    raw = struct.pack("<%d%s" % (len(values), fmt), *values)
    b = b"".join(f_varint(1, d) for d in dims) + f_varint(2, dtype) + f_bytes(8, name) + f_bytes(9, raw)
    return b

def value_info(name, dtype, shape):
    dims = b""
    for d in shape:
        dims += f_bytes(1, f_bytes(2, d) if isinstance(d, str) else f_varint(1, d))
    tt = f_varint(1, dtype) + f_bytes(2, dims)
    return f_bytes(1, name) + f_bytes(2, f_bytes(1, tt))

def attr_int(name, v):
    return f_bytes(1, name) + f_varint(3, v) + f_varint(20, 2)

def attr_float(name, v):
    return f_bytes(1, name) + f_float(2, v) + f_varint(20, 1)

def attr_tensor(name, t):
    return f_bytes(1, name) + f_bytes(5, t) + f_varint(20, 4)

class G:
    def __init__(self):
        self.nodes, self.inits, self.inputs, self.outputs = [], [], [], []
        self.n = 0

    def node(self, op, ins, outs=None, attrs=()):
        if outs is None:
            self.n += 1
            outs = ["t%d" % self.n]
        b = b"".join(f_bytes(1, i) for i in ins) + b"".join(f_bytes(2, o) for o in outs)
        b += f_bytes(3, "n%d_%s" % (len(self.nodes), op)) + f_bytes(4, op)
        b += b"".join(f_bytes(5, a) for a in attrs)
        self.nodes.append(b)
        return outs[0] if len(outs) == 1 else outs

    def const(self, dtype, dims, values):
        self.n += 1
        name = "c%d" % self.n
        self.inits.append(tensor(name, dtype, dims, values))
        return name

    def inp(self, name, dtype, shape):
        self.inputs.append(value_info(name, dtype, shape))
        return name

    def out(self, name, dtype, shape):
        self.outputs.append(value_info(name, dtype, shape))

    def save(self, path):
        graph = b"".join(f_bytes(1, n) for n in self.nodes) + f_bytes(2, "g")
        graph += b"".join(f_bytes(5, t) for t in self.inits)
        graph += b"".join(f_bytes(11, v) for v in self.inputs) + b"".join(f_bytes(12, v) for v in self.outputs)
        model = f_varint(1, 8) + f_bytes(2, "mock") + f_bytes(7, graph) + f_bytes(8, f_bytes(1, "") + f_varint(2, 17))
        with open(path, "wb") as fh:
            fh.write(model)

H, LAYERS, HEADS, HD, NVQ, CB = 768, 12, 12, 64, 16, 1024
KV = ["present_%s_%d" % (kind, i) for i in range(LAYERS) for kind in ("key", "value")]
PAST = [k.replace("present_", "past_") for k in KV]

def hidden_vec(g, a_f1, b_f1):
    """Concat([a],[b],zeros(H-2)) -> [H]"""
    z = g.const(FLOAT, [H - 2], [0.0] * (H - 2))
    return g.node("Concat", [a_f1, b_f1, z], attrs=[attr_int("axis", 0)])

def make_prefill(path):
    g = G()
    ids = g.inp("input_ids", INT32, [1, "S", NVQ + 1])
    g.inp("attention_mask", INT32, [1, "S"])
    shp = g.node("Shape", [ids])                                      # [1,S,17] int64
    s = g.node("Gather", [shp, g.const(INT64, [1], [1])], attrs=[attr_int("axis", 0)])  # [1]
    s_f = g.node("Cast", [s], attrs=[attr_int("to", FLOAT)])
    vec = hidden_vec(g, s_f, s_f)
    vec3 = g.node("Reshape", [vec, g.const(INT64, [3], [1, 1, H])])
    bs = g.node("Slice", [shp, g.const(INT64, [1], [0]), g.const(INT64, [1], [2])])      # [1,S]
    full = g.node("Concat", [bs, g.const(INT64, [1], [H])], attrs=[attr_int("axis", 0)])
    g.node("Expand", [vec3, full], ["global_hidden"])
    kv = g.node("Reshape", [g.node("Identity", ["global_hidden"]), g.const(INT64, [4], [1, -1, HEADS, HD])])
    for name in KV:
        g.node("Identity", [kv], [name])
    g.out("global_hidden", FLOAT, [1, "S", H])
    for name in KV:
        g.out(name, FLOAT, [1, "S", HEADS, HD])
    g.save(path)

def make_decode(path):
    g = G()
    g.inp("input_ids", INT32, [1, 1, NVQ + 1])
    plen = g.inp("past_valid_lengths", INT32, [1])
    for name in PAST:
        g.inp(name, FLOAT, [1, "P", HEADS, HD])
    kvlen = g.node("Gather", [g.node("Shape", [PAST[0]]), g.const(INT64, [1], [1])], attrs=[attr_int("axis", 0)])
    one = g.const(FLOAT, [1], [1.0])
    # position of the token just appended: past_valid_lengths + 1 vs real KV length + 1
    vec = hidden_vec(g, g.node("Add", [g.node("Cast", [plen], attrs=[attr_int("to", FLOAT)]), one]),
                     g.node("Add", [g.node("Cast", [kvlen], attrs=[attr_int("to", FLOAT)]), one]))
    g.node("Reshape", [vec, g.const(INT64, [3], [1, 1, H])], ["global_hidden"])
    new_kv = g.node("Reshape", [vec, g.const(INT64, [4], [1, 1, HEADS, HD])])
    for p, k in zip(PAST, KV):
        g.node("Concat", [p, new_kv], [k], attrs=[attr_int("axis", 1)])
    g.out("global_hidden", FLOAT, [1, 1, H])
    for name in KV:
        g.out(name, FLOAT, [1, "P1", HEADS, HD])
    g.save(path)

def make_local(path, frames_per_chunk):
    g = G()
    gh = g.inp("global_hidden", FLOAT, [1, H])
    seen = g.inp("repetition_seen_mask", INT32, [1, NVQ, CB])
    g.inp("assistant_random_u", FLOAT, [1])
    au = g.inp("audio_random_u", FLOAT, [1, NVQ])
    h2 = g.node("Slice", [gh, g.const(INT64, [1], [0]), g.const(INT64, [1], [2]), g.const(INT64, [1], [1])])  # [1,2]
    h2i = g.node("Cast", [h2], attrs=[attr_int("to", INT32)])
    rnd = g.node("Cast", [g.node("Mul", [au, g.const(FLOAT, [], [1000.0])])], attrs=[attr_int("to", INT32)])
    rnd14 = g.node("Slice", [rnd, g.const(INT64, [1], [2]), g.const(INT64, [1], [NVQ]), g.const(INT64, [1], [1])])
    g.node("Concat", [h2i, rnd14], ["frame_token_ids"], attrs=[attr_int("axis", 1)])
    ch0 = g.node("Slice", [seen, g.const(INT64, [1], [0]), g.const(INT64, [1], [1]), g.const(INT64, [1], [1])])
    count = g.node("ReduceSum", [ch0], attrs=[attr_int("keepdims", 0)])          # scalar int32
    under = g.node("Less", [count, g.const(INT32, [], [frames_per_chunk])])
    a = g.node("Slice", [h2, g.const(INT64, [1], [0]), g.const(INT64, [1], [1]), g.const(INT64, [1], [1])])
    b = g.node("Slice", [h2, g.const(INT64, [1], [1]), g.const(INT64, [1], [2]), g.const(INT64, [1], [1])])
    same = g.node("Reshape", [g.node("Equal", [a, b]), g.const(INT64, [1], [1])])
    g.node("And", [g.node("Reshape", [under, g.const(INT64, [1], [1])]), same], ["should_continue"])
    g.out("should_continue", BOOL, [1])
    g.out("frame_token_ids", INT32, [1, NVQ])
    g.save(path)

CODEC_SPEC = {
    "transformer_offsets": [{"input_name": "t_off_in_0", "output_name": "t_off_out_0", "shape": [1]}],
    "attention_caches": [{
        "offset_input_name": "a_off_in_0", "offset_output_name": "a_off_out_0", "offset_shape": [1],
        "cached_keys_input_name": "k_in_0", "cached_keys_output_name": "k_out_0",
        "cached_values_input_name": "v_in_0", "cached_values_output_name": "v_out_0",
        "cached_positions_input_name": "pos_in_0", "cached_positions_output_name": "pos_out_0",
        "cache_shape": [1, 4, 8, 64], "positions_shape": [1, 8],
    }],
}

def make_codec(path):
    g = G()
    g.inp("audio_codes", INT32, [1, "F", NVQ])
    lens = g.inp("audio_code_lengths", INT32, [1])
    t_off = g.inp("t_off_in_0", INT32, [1])
    a_off = g.inp("a_off_in_0", INT32, [1])
    k = g.inp("k_in_0", FLOAT, [1, 4, 8, 64])
    v = g.inp("v_in_0", FLOAT, [1, 4, 8, 64])
    pos = g.inp("pos_in_0", INT32, [1, 8])
    g.node("Add", [t_off, lens], ["t_off_out_0"])
    g.node("Add", [a_off, lens], ["a_off_out_0"])
    g.node("Add", [k, g.const(FLOAT, [], [1.0])], ["k_out_0"])
    g.node("Identity", [v], ["v_out_0"])
    g.node("Identity", [pos], ["pos_out_0"])
    hop = g.const(INT32, [], [3840])
    g.node("Mul", [lens, hop], ["audio_lengths"])
    start = g.node("Cast", [g.node("Squeeze", [g.node("Mul", [t_off, hop]), g.const(INT64, [1], [0])])], attrs=[attr_int("to", FLOAT)])
    n = g.node("Cast", [g.node("Squeeze", [g.node("Mul", [lens, hop]), g.const(INT64, [1], [0])])], attrs=[attr_int("to", FLOAT)])
    ramp = g.node("Range", [start, g.node("Add", [start, n]), g.const(FLOAT, [], [1.0])])
    ramp = g.node("Mul", [ramp, g.const(FLOAT, [], [1e-6])])
    r3 = g.node("Unsqueeze", [ramp, g.const(INT64, [2], [0, 1])])
    g.node("Concat", [r3, r3], ["audio"], attrs=[attr_int("axis", 1)])
    for name, dt, shp in [("audio", FLOAT, [1, 2, "N"]), ("audio_lengths", INT32, [1]),
                          ("t_off_out_0", INT32, [1]), ("a_off_out_0", INT32, [1]),
                          ("k_out_0", FLOAT, [1, 4, 8, 64]), ("v_out_0", FLOAT, [1, 4, 8, 64]),
                          ("pos_out_0", INT32, [1, 8])]:
        g.out(name, dt, shp)
    g.save(path)

def main(root, frames_per_chunk=20):
    tts_dir = os.path.join(root, "MOSS-TTS-Nano-100M-ONNX")
    codec_dir = os.path.join(root, "MOSS-Audio-Tokenizer-Nano-ONNX")
    os.makedirs(tts_dir, exist_ok=True)
    os.makedirs(codec_dir, exist_ok=True)
    make_prefill(os.path.join(tts_dir, "moss_tts_prefill.onnx"))
    make_decode(os.path.join(tts_dir, "moss_tts_decode_step.onnx"))
    make_local(os.path.join(tts_dir, "moss_tts_local_fixed_sampled_frame.onnx"), frames_per_chunk)
    make_codec(os.path.join(codec_dir, "moss_audio_tokenizer_decode_step.onnx"))
    voices = [{"voice": "MockA", "display_name": "Mock A", "group": "Chinese Female",
               "audio_file": "", "prompt_audio_codes": [[(i * 7 + q) % 1024 for q in range(NVQ)] for i in range(100)]}]
    manifest = {
        "format_version": 1,
        "model_files": {"tts_meta": "tts_browser_onnx_meta.json",
                        "codec_meta": "../MOSS-Audio-Tokenizer-Nano-ONNX/codec_browser_onnx_meta.json",
                        "tokenizer_model": "tokenizer.model"},
        "tts_config": {"n_vq": 16, "audio_pad_token_id": 1024, "pad_token_id": 3, "im_start_token_id": 4,
                       "im_end_token_id": 5, "audio_start_token_id": 6, "audio_end_token_id": 7,
                       "audio_user_slot_token_id": 8, "audio_assistant_slot_token_id": 9,
                       "audio_codebook_sizes": [1024] * 16, "vocab_size": 16384},
        "prompt_templates": {"user_prompt_prefix_token_ids": [600, 289, 10356, 13],
                             "user_prompt_after_reference_token_ids": [10356, 10425, 3965],
                             "assistant_prompt_prefix_token_ids": [10356, 14, 5, 4, 8165, 430]},
        "generation_defaults": {"max_new_frames": 375, "do_sample": True, "sample_mode": "fixed"},
        "builtin_voices": voices,
    }
    meta = {
        "files": {"prefill": "moss_tts_prefill.onnx", "decode_step": "moss_tts_decode_step.onnx",
                  "local_fixed_sampled_frame": "moss_tts_local_fixed_sampled_frame.onnx"},
        "model_config": {"hidden_size": H},
        "onnx": {"prefill_output_names": ["global_hidden"] + KV,
                 "decode_input_names": ["input_ids", "past_valid_lengths"] + PAST,
                 "decode_output_names": ["global_hidden"] + KV},
    }
    codec_meta = {
        "files": {"decode_step": "moss_audio_tokenizer_decode_step.onnx"},
        "codec_config": {"sample_rate": 48000, "channels": 2, "num_quantizers": 16, "downsample_rate": 3840},
        "streaming_decode": CODEC_SPEC,
    }
    json.dump(manifest, open(os.path.join(tts_dir, "browser_poc_manifest.json"), "w"))
    json.dump(meta, open(os.path.join(tts_dir, "tts_browser_onnx_meta.json"), "w"))
    json.dump(codec_meta, open(os.path.join(codec_dir, "codec_browser_onnx_meta.json"), "w"))
    print("mock models written to", root)

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20)
