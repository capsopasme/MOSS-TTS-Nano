#!/usr/bin/env python3
"""Micro-benchmark of the graphs the Android engine runs, for comparing model packs / thread
counts on a given CPU (CI runs it on an ARM64 runner).

usage: bench_onnx.py <model_root> <threads> [frames]
Prints one JSON line: prefill ms, decode / local ms per frame, codec ms per frame at batch 1 and 8.
"""
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort


def sess(path, threads, mem_pattern=True):
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    o.enable_mem_pattern = mem_pattern
    return ort.InferenceSession(path, sess_options=o, providers=["CPUExecutionProvider"])


def main(root, threads, max_frames=80):
    tts = os.path.join(root, "MOSS-TTS-Nano-100M-ONNX")
    man = json.load(open(os.path.join(tts, "browser_poc_manifest.json")))
    meta = json.load(open(os.path.join(tts, "tts_browser_onnx_meta.json")))
    codec_meta_path = os.path.normpath(os.path.join(tts, man["model_files"]["codec_meta"]))
    codec_dir = os.path.dirname(codec_meta_path)
    codec_meta = json.load(open(codec_meta_path))
    cfg, tpl = man["tts_config"], man["prompt_templates"]
    nvq, pad = cfg["n_vq"], cfg["audio_pad_token_id"]

    prefill = sess(os.path.join(tts, meta["files"]["prefill"]), threads, False)
    decode = sess(os.path.join(tts, meta["files"]["decode_step"]), threads, False)
    local = sess(os.path.join(tts, meta["files"]["local_fixed_sampled_frame"]), threads)
    codec = sess(os.path.join(codec_dir, codec_meta["files"]["decode_step"]), max(1, threads // 2))

    voice = man["builtin_voices"][0]
    ids = man["text_samples"][0]["text_token_ids"]
    row = lambda t: [t] + [pad] * nvq
    rows = [row(t) for t in tpl["user_prompt_prefix_token_ids"] + [cfg["audio_start_token_id"]]]
    rows += [[cfg["audio_user_slot_token_id"]] + list(r[:nvq]) for r in voice["prompt_audio_codes"]]
    rows += [row(t) for t in [cfg["audio_end_token_id"]] + tpl["user_prompt_after_reference_token_ids"] + ids
             + tpl["assistant_prompt_prefix_token_ids"] + [cfg["audio_start_token_id"]]]
    seq = len(rows)
    feeds_p = {"input_ids": np.asarray([rows], np.int32), "attention_mask": np.ones((1, seq), np.int32)}
    prefill.run(None, feeds_p)  # warm
    t0 = time.perf_counter()
    outs = prefill.run(None, feeds_p)
    t_prefill = time.perf_counter() - t0
    names = meta["onnx"]["prefill_output_names"]
    hidden = outs[0][:, -1, :].astype(np.float32)
    past = {n.replace("present_", "past_"): v for n, v in zip(names[1:], outs[1:])}

    seen = np.zeros((1, nvq, 1024), np.int32)
    rng = np.random.default_rng(1234)
    t_local = t_decode = 0.0
    frames = []
    for step in range(max_frames):
        a = time.perf_counter()
        r = local.run(None, {"global_hidden": hidden, "repetition_seen_mask": seen,
                             "assistant_random_u": np.asarray([min(0.99999994, rng.random())], np.float32),
                             "audio_random_u": np.minimum(rng.random((1, nvq)), 0.99999994).astype(np.float32)})
        t_local += time.perf_counter() - a
        frame = np.asarray(r[1]).reshape(-1).astype(np.int32)
        # keep generating even past a stop so every pack runs the same number of steps
        for q, tok in enumerate(frame):
            seen[0, q, tok] = 1
        a = time.perf_counter()
        d = decode.run(None, {"input_ids": np.asarray([[[cfg["audio_assistant_slot_token_id"]] + frame.tolist()]], np.int32),
                              "past_valid_lengths": np.asarray([seq + step], np.int32), **past})
        t_decode += time.perf_counter() - a
        hidden = d[0].reshape(1, -1).astype(np.float32)
        past = {n.replace("present_", "past_"): v for n, v in zip(meta["onnx"]["decode_output_names"][1:], d[1:])}
        frames.append(frame)

    sd = codec_meta["streaming_decode"]

    def init_state():
        st = {}
        for t in sd["transformer_offsets"]:
            st[t["input_name"]] = np.zeros(t["shape"], np.int32)
        for a in sd["attention_caches"]:
            st[a["offset_input_name"]] = np.zeros(a["offset_shape"], np.int32)
            st[a["cached_keys_input_name"]] = np.zeros(a["cache_shape"], np.float32)
            st[a["cached_values_input_name"]] = np.zeros(a["cache_shape"], np.float32)
            st[a["cached_positions_input_name"]] = np.full(a["positions_shape"], -1, np.int32)
        return st

    omap = {t["output_name"]: t["input_name"] for t in sd["transformer_offsets"]}
    for a in sd["attention_caches"]:
        for k in ("offset", "cached_keys", "cached_values", "cached_positions"):
            omap[a[f"{k}_output_name"]] = a[f"{k}_input_name"]
    cnames = [o.name for o in codec.get_outputs()]
    codec_ms = {}
    for b in (1, 8):
        st = init_state()
        n = (len(frames) // b) * b
        t = 0.0
        for i in range(0, n, b):
            c = np.asarray([frames[i:i + b]], np.int32)
            a = time.perf_counter()
            res = dict(zip(cnames, codec.run(None, {"audio_codes": c, "audio_code_lengths": np.asarray([b], np.int32), **st})))
            t += time.perf_counter() - a
            for o, i_ in omap.items():
                st[i_] = res[o]
        codec_ms[f"b{b}"] = round(t * 1000 / max(n, 1), 2)

    print(json.dumps({
        "root": os.path.basename(os.path.normpath(root)), "threads": threads, "prefill_rows": seq,
        "prefill_ms": round(t_prefill * 1000, 1),
        "decode_ms": round(t_decode * 1000 / len(frames), 2), "local_ms": round(t_local * 1000 / len(frames), 2),
        "lm_ms_per_frame": round((t_decode + t_local) * 1000 / len(frames), 2),
        "codec_ms_per_frame": codec_ms, "realtime_budget_ms": 80,
    }))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 80)
