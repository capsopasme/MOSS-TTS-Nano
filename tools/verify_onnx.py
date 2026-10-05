#!/usr/bin/env python3
"""Python mirror of the Android engine's streaming loop (prefill -> local_fixed_sampled_frame
-> decode_step, with codec decode_step streaming). Synthesizes a sentence with a model dir and
writes a WAV + timing, so the FP32 and INT8 packs can be compared (by ear and speed) in CI.

usage: verify_onnx.py <model_root> <out.wav> [text]
"""
import json
import os
import sys
import time
import wave

import numpy as np
import onnxruntime as ort
import sentencepiece as spm


def sess(path, threads):
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=o, providers=["CPUExecutionProvider"])


def main(root, out_wav, text):
    tts_dir = os.path.join(root, "MOSS-TTS-Nano-100M-ONNX")
    manifest = json.load(open(os.path.join(tts_dir, "browser_poc_manifest.json")))
    meta = json.load(open(os.path.join(tts_dir, "tts_browser_onnx_meta.json")))
    codec_meta_path = os.path.normpath(os.path.join(tts_dir, manifest["model_files"]["codec_meta"]))
    codec_dir = os.path.dirname(codec_meta_path)
    codec_meta = json.load(open(codec_meta_path))
    cfg = manifest["tts_config"]
    tpl = manifest["prompt_templates"]
    nvq = cfg["n_vq"]
    pad = cfg["audio_pad_token_id"]
    threads = os.cpu_count() or 4

    t0 = time.perf_counter()
    prefill = sess(os.path.join(tts_dir, meta["files"]["prefill"]), threads)
    decode = sess(os.path.join(tts_dir, meta["files"]["decode_step"]), threads)
    local = sess(os.path.join(tts_dir, meta["files"]["local_fixed_sampled_frame"]), threads)
    codec = sess(os.path.join(codec_dir, codec_meta["files"]["decode_step"]), max(1, threads // 2))
    load_s = time.perf_counter() - t0

    sp = spm.SentencePieceProcessor(model_file=os.path.join(tts_dir, "tokenizer.model"))
    ids = sp.encode(text, out_type=int)
    voice = manifest["builtin_voices"][0]

    def text_row(t):
        return [t] + [pad] * nvq

    rows = [text_row(t) for t in tpl["user_prompt_prefix_token_ids"] + [cfg["audio_start_token_id"]]]
    rows += [[cfg["audio_user_slot_token_id"]] + list(r[:nvq]) for r in voice["prompt_audio_codes"]]
    rows += [text_row(t) for t in [cfg["audio_end_token_id"]] + tpl["user_prompt_after_reference_token_ids"] + ids
             + tpl["assistant_prompt_prefix_token_ids"] + [cfg["audio_start_token_id"]]]
    seq = len(rows)

    # codec state
    st = codec_meta["streaming_decode"]
    state = {}
    for s in st.get("transformer_offsets", []):
        state[s["input_name"]] = np.zeros(s["shape"], np.int32)
    for a in st.get("attention_caches", []):
        state[a["offset_input_name"]] = np.zeros(a["offset_shape"], np.int32)
        state[a["cached_keys_input_name"]] = np.zeros(a["cache_shape"], np.float32)
        state[a["cached_values_input_name"]] = np.zeros(a["cache_shape"], np.float32)
        state[a["cached_positions_input_name"]] = np.full(a["positions_shape"], -1, np.int32)
    out_map = {}
    for s in st.get("transformer_offsets", []):
        out_map[s["output_name"]] = s["input_name"]
    for a in st.get("attention_caches", []):
        for k in ("offset", "cached_keys", "cached_values", "cached_positions"):
            out_map[a[f"{k}_output_name"]] = a[f"{k}_input_name"]
    codec_out_names = [o.name for o in codec.get_outputs()]

    audio_chunks = []
    rng = np.random.default_rng(1234)
    t_start = time.perf_counter()
    first_audio = None
    t_lm = 0.0
    t_codec = 0.0

    tp = time.perf_counter()
    outs = prefill.run(None, {"input_ids": np.asarray([rows], np.int32), "attention_mask": np.ones((1, seq), np.int32)})
    t_prefill = time.perf_counter() - tp
    names = meta["onnx"]["prefill_output_names"]
    hidden = outs[0].reshape(-1, outs[0].shape[-1])[-1:].astype(np.float32)
    past = dict(zip([n.replace("present_", "past_") for n in names[1:]], outs[1:]))
    past_len = seq
    seen = np.zeros((1, nvq, cfg["audio_codebook_sizes"][0]), np.int32)
    pending = []
    frames = 0

    def flush(force):
        nonlocal first_audio, t_codec, state
        if not pending or (not force and len(pending) < 4 and first_audio is not None):
            return
        tc = time.perf_counter()
        codes = np.asarray([pending], np.int32)
        feeds = {"audio_codes": codes, "audio_code_lengths": np.asarray([len(pending)], np.int32), **state}
        res = dict(zip(codec_out_names, codec.run(None, feeds)))
        for o, i in out_map.items():
            state[i] = res[o]
        n = int(res["audio_lengths"].reshape(-1)[0])
        audio_chunks.append(res["audio"][0, :, :n].T.copy())
        pending.clear()
        t_codec += time.perf_counter() - tc
        if first_audio is None:
            first_audio = time.perf_counter() - t_start

    for _ in range(manifest["generation_defaults"]["max_new_frames"]):
        tl = time.perf_counter()
        r = local.run(None, {
            "global_hidden": hidden,
            "repetition_seen_mask": seen,
            "assistant_random_u": np.asarray([min(0.99999994, rng.random())], np.float32),
            "audio_random_u": np.asarray([[min(0.99999994, rng.random()) for _ in range(nvq)]], np.float32),
        })
        cont = bool(np.asarray(r[0]).reshape(-1)[0])
        frame = np.asarray(r[1]).reshape(-1).astype(np.int32).tolist()
        if not cont:
            t_lm += time.perf_counter() - tl
            break
        for q, tok in enumerate(frame):
            seen[0, q, tok] = 1
        row = np.asarray([[[cfg["audio_assistant_slot_token_id"]] + frame]], np.int32)
        feeds = {"input_ids": row, "past_valid_lengths": np.asarray([past_len], np.int32)}
        for n in meta["onnx"]["decode_input_names"][2:]:
            feeds[n] = past[n]
        d = decode.run(None, feeds)
        dn = meta["onnx"]["decode_output_names"]
        hidden = d[0].reshape(-1, d[0].shape[-1])[-1:].astype(np.float32)
        past = dict(zip([n.replace("present_", "past_") for n in dn[1:]], d[1:]))
        past_len += 1
        frames += 1
        t_lm += time.perf_counter() - tl
        pending.append(frame)
        flush(False)
    flush(True)
    wall = time.perf_counter() - t_start

    audio = np.concatenate(audio_chunks, axis=0) if audio_chunks else np.zeros((0, 2), np.float32)
    sr = codec_meta["codec_config"]["sample_rate"]
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    dur = len(audio) / sr
    print(json.dumps({
        "model_root": root, "load_s": round(load_s, 2), "prefill_rows": seq, "prefill_ms": round(t_prefill * 1000),
        "frames": frames, "lm_ms_per_frame": round(t_lm * 1000 / max(frames, 1), 1),
        "codec_ms_total": round(t_codec * 1000), "first_audio_ms": round((first_audio or 0) * 1000),
        "audio_s": round(dur, 2), "wall_s": round(wall, 2), "rtf": round(wall / dur, 3) if dur else None,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "你好，这是端侧流式语音合成的量化验证。今天天气不错，我们出去走走吧。")
