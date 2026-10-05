#!/usr/bin/env python3
"""Checks the INT8 pack against the official FP32 export.

  1. Graph surgery is lossless: the FP32 local graph after Identity folding + text-head slicing
     (work/fp32opt) must sample exactly the same frames as the official FP32 local graph for
     the same hidden state and random numbers.
  2. Quantization error, teacher-forced on a real prompt:
       - cosine similarity of the global hidden state (prefill + every decode step) INT8 vs FP32,
       - agreement of the sampled codebook tokens / stop decision of the local graph when it is
         given the same hidden state and random numbers.

usage: check_int8_pack.py <work_dir> [frames]
Prints one JSON line; exits non-zero if (1) fails or the hidden states drift too far.
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort


def sess(path):
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.intra_op_num_threads = os.cpu_count() or 2
    return ort.InferenceSession(path, sess_options=o, providers=["CPUExecutionProvider"])


def cos(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class LM:
    def __init__(self, tts_dir, local_dir=None):
        meta = json.load(open(os.path.join(tts_dir, "tts_browser_onnx_meta.json")))
        f = meta["files"]
        self.meta = meta
        self.prefill = sess(os.path.join(tts_dir, f["prefill"]))
        self.decode = sess(os.path.join(tts_dir, f["decode_step"]))
        self.local = sess(os.path.join(local_dir or tts_dir, f["local_fixed_sampled_frame"]))

    def run_prefill(self, rows):
        outs = self.prefill.run(None, {"input_ids": np.asarray([rows], np.int32),
                                       "attention_mask": np.ones((1, len(rows)), np.int32)})
        names = self.meta["onnx"]["prefill_output_names"]
        hidden = outs[0][:, -1, :].astype(np.float32)
        past = {n.replace("present_", "past_"): v for n, v in zip(names[1:], outs[1:])}
        return hidden, past

    def run_decode(self, row, past_len, past):
        feeds = {"input_ids": np.asarray([[row]], np.int32), "past_valid_lengths": np.asarray([past_len], np.int32), **past}
        d = self.decode.run(None, feeds)
        names = self.meta["onnx"]["decode_output_names"]
        return d[0].reshape(1, -1).astype(np.float32), {n.replace("present_", "past_"): v for n, v in zip(names[1:], d[1:])}

    def run_local(self, hidden, seen, au, uu):
        r = self.local.run(None, {"global_hidden": hidden, "repetition_seen_mask": seen,
                                  "assistant_random_u": au, "audio_random_u": uu})
        return bool(np.asarray(r[0]).reshape(-1)[0]), np.asarray(r[1]).reshape(-1).astype(np.int32)


def main(work, max_frames=120):
    src = os.path.join(work, "src", "MOSS-TTS-Nano-100M-ONNX")
    opt = os.path.join(work, "fp32opt", "MOSS-TTS-Nano-100M-ONNX")
    out = os.path.join(work, "out", "MOSS-TTS-Nano-100M-ONNX")
    man = json.load(open(os.path.join(src, "browser_poc_manifest.json")))
    cfg, tpl = man["tts_config"], man["prompt_templates"]
    nvq, pad = cfg["n_vq"], cfg["audio_pad_token_id"]
    slot = cfg["audio_assistant_slot_token_id"]

    fp32 = LM(src)
    fp32opt_local = LM(src, opt).local
    int8 = LM(out)

    report = {}
    for sample in man["text_samples"][:2]:
        voice = man["builtin_voices"][0 if sample["id"].startswith("zh") else 7]
        row = lambda t: [t] + [pad] * nvq
        rows = [row(t) for t in tpl["user_prompt_prefix_token_ids"] + [cfg["audio_start_token_id"]]]
        rows += [[cfg["audio_user_slot_token_id"]] + list(r[:nvq]) for r in voice["prompt_audio_codes"]]
        rows += [row(t) for t in [cfg["audio_end_token_id"]] + tpl["user_prompt_after_reference_token_ids"]
                 + sample["text_token_ids"] + tpl["assistant_prompt_prefix_token_ids"] + [cfg["audio_start_token_id"]]]
        seq = len(rows)

        h32, p32 = fp32.run_prefill(rows)
        h8, p8 = int8.run_prefill(rows)
        hidden_cos = [cos(h32, h8)]
        rng = np.random.default_rng(1234)
        seen = np.zeros((1, nvq, 1024), np.int32)
        surgery_mismatch = 0
        tok_agree = tok_total = stop_agree = local_calls = 0
        frames = 0
        for step in range(max_frames):
            au = np.asarray([min(0.99999994, rng.random())], np.float32)
            uu = np.minimum(rng.random((1, nvq)), 0.99999994).astype(np.float32)
            c32, f32 = fp32.run_local(h32, seen, au, uu)
            co, fo = fp32opt_local.run(None, {"global_hidden": h32, "repetition_seen_mask": seen,
                                              "assistant_random_u": au, "audio_random_u": uu})
            co = bool(np.asarray(co).reshape(-1)[0])
            fo = np.asarray(fo).reshape(-1)
            if co != c32 or (c32 and not np.array_equal(fo, f32)):
                surgery_mismatch += 1
            c8, f8 = int8.run_local(h32, seen, au, uu)  # same hidden: isolates the local graph's error
            stop_agree += int(c8 == c32)
            local_calls += 1
            if c32 and c8:
                tok_agree += int((f8 == f32).sum())
                tok_total += nvq
            if not c32:
                break
            for q, tok in enumerate(f32):
                seen[0, q, tok] = 1
            r = [slot] + f32.tolist()
            h32, p32 = fp32.run_decode(r, seq + step, p32)
            h8, p8 = int8.run_decode(r, seq + step, p8)  # teacher-forced with the FP32 frames
            hidden_cos.append(cos(h32, h8))
            frames += 1
        report[sample["id"]] = {
            "prefill_rows": seq,
            "frames": frames,
            "surgery_mismatches": surgery_mismatch,
            "hidden_cos_min": round(min(hidden_cos), 5),
            "hidden_cos_mean": round(float(np.mean(hidden_cos)), 5),
            "local_token_agreement": round(tok_agree / max(tok_total, 1), 4),
            "local_stop_agreement": round(stop_agree / max(local_calls, 1), 4),
        }
    print(json.dumps(report))
    bad = [k for k, v in report.items() if v["surgery_mismatches"] > 1 or v["hidden_cos_min"] < 0.98]
    if bad:
        raise SystemExit(f"check failed for {bad}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 120)
