#!/usr/bin/env python3
"""How "FP32-plausible" are the frames a quantized pack generates?

Each pack (FP32 reference, INT8 packs) generates audio frames free-running on real prompts with
the same random stream. Every generated frame is then scored by the *FP32* model, teacher-forced
(official local_cached_step graph): the mean negative log-likelihood per audio token under the
FP32 sampling distribution (repetition penalty 1.2, temperature 0.8, no truncation).

If a pack's NLL is close to FP32's own NLL, its samples are as typical as FP32's samples; a
clearly higher NLL means the quantized model drifts to tokens FP32 finds unlikely (audible as
artifacts / mumbling).

usage: quality_int8.py <fp32_root> <name>=<pack_root> [<name>=<pack_root> ...]
Prints one JSON line.
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort

PENALTY = 1.2
TEMPERATURE = 0.8


def sess(path):
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.intra_op_num_threads = os.cpu_count() or 2
    return ort.InferenceSession(path, sess_options=o, providers=["CPUExecutionProvider"])


class Pack:
    def __init__(self, root):
        tts = os.path.join(root, "MOSS-TTS-Nano-100M-ONNX")
        self.meta = json.load(open(os.path.join(tts, "tts_browser_onnx_meta.json")))
        f = self.meta["files"]
        self.prefill = sess(os.path.join(tts, f["prefill"]))
        self.decode = sess(os.path.join(tts, f["decode_step"]))
        self.local = sess(os.path.join(tts, f["local_fixed_sampled_frame"]))
        cached = os.path.join(tts, f.get("local_cached_step", ""))
        self.cached = sess(cached) if f.get("local_cached_step") and os.path.isfile(cached) else None

    def run_prefill(self, rows):
        outs = self.prefill.run(None, {"input_ids": np.asarray([rows], np.int32), "attention_mask": np.ones((1, len(rows)), np.int32)})
        names = self.meta["onnx"]["prefill_output_names"]
        return outs[0][:, -1, :].astype(np.float32), {n.replace("present_", "past_"): v for n, v in zip(names[1:], outs[1:])}

    def run_decode(self, row, past_len, past):
        d = self.decode.run(None, {"input_ids": np.asarray([[row]], np.int32), "past_valid_lengths": np.asarray([past_len], np.int32), **past})
        names = self.meta["onnx"]["decode_output_names"]
        return d[0].reshape(1, -1).astype(np.float32), {n.replace("present_", "past_"): v for n, v in zip(names[1:], d[1:])}


def build_rows(man, voice, ids):
    cfg, tpl = man["tts_config"], man["prompt_templates"]
    nvq, pad = cfg["n_vq"], cfg["audio_pad_token_id"]
    row = lambda t: [t] + [pad] * nvq
    rows = [row(t) for t in tpl["user_prompt_prefix_token_ids"] + [cfg["audio_start_token_id"]]]
    rows += [[cfg["audio_user_slot_token_id"]] + list(r[:nvq]) for r in voice["prompt_audio_codes"]]
    rows += [row(t) for t in [cfg["audio_end_token_id"]] + tpl["user_prompt_after_reference_token_ids"] + ids
             + tpl["assistant_prompt_prefix_token_ids"] + [cfg["audio_start_token_id"]]]
    return rows


def generate(pack, man, rows, seed, max_frames):
    cfg = man["tts_config"]
    nvq = cfg["n_vq"]
    h, past = pack.run_prefill(rows)
    seen = np.zeros((1, nvq, 1024), np.int32)
    rng = np.random.default_rng(seed)
    frames = []
    for step in range(max_frames):
        r = pack.local.run(None, {"global_hidden": h, "repetition_seen_mask": seen,
                                  "assistant_random_u": np.asarray([min(0.99999994, rng.random())], np.float32),
                                  "audio_random_u": np.minimum(rng.random((1, nvq)), 0.99999994).astype(np.float32)})
        if not bool(np.asarray(r[0]).reshape(-1)[0]):
            break
        f = np.asarray(r[1]).reshape(-1).astype(np.int32)
        frames.append(f)
        for q, tok in enumerate(f):
            seen[0, q, tok] = 1
        h, past = pack.run_decode([cfg["audio_assistant_slot_token_id"]] + f.tolist(), len(rows) + step, past)
    return frames


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def score(ref, man, rows, frames):
    """Mean FP32 NLL per audio token of `frames` (teacher-forced)."""
    cfg = man["tts_config"]
    nvq = cfg["n_vq"]
    slot = cfg["audio_assistant_slot_token_id"]
    layers = int(ref.meta["model_config"]["local_layers"])
    heads = int(ref.meta["model_config"]["local_heads"])
    hd = int(ref.meta["model_config"]["local_head_dim"])
    out_names = ref.meta["onnx"]["local_cached_output_names"]
    h, past = ref.run_prefill(rows)
    seen = np.zeros((nvq, 1024), bool)
    nll = []
    for step, f in enumerate(frames):
        lp = {f"local_past_{k}_{l}": np.zeros((1, 0, heads, hd), np.float32) for l in range(layers) for k in ("key", "value")}

        def cstep(text_tok, audio_tok, ch, stype, pvl, lp):
            o = ref.cached.run(None, {"global_hidden": h, "text_token_id": np.asarray([text_tok], np.int32),
                                      "audio_token_id": np.asarray([audio_tok], np.int32), "channel_index": np.asarray([ch], np.int32),
                                      "step_type": np.asarray([stype], np.int32), "past_valid_lengths": np.asarray([pvl], np.int32), **lp})
            nxt = {n.replace("local_present_", "local_past_"): v for n, v in zip(out_names[2:], o[2:])}
            return o[1].reshape(nvq, -1), nxt

        _, lp = cstep(0, 0, 0, 0, 0, lp)
        logits, lp = cstep(slot, 0, 0, 1, 1, lp)
        for ch in range(nvq):
            if ch > 0:
                logits, lp = cstep(0, int(f[ch - 1]), ch - 1, 2, ch + 1, lp)
            x = logits[ch].astype(np.float64)
            pen = np.where(x < 0, x * PENALTY, x / PENALTY)
            x = np.where(seen[ch], pen, x) / TEMPERATURE
            nll.append(-log_softmax(x)[int(f[ch])])
        for q, tok in enumerate(f):
            seen[q, tok] = True
        h, past = ref.run_decode([slot] + f.tolist(), len(rows) + step, past)
    return float(np.mean(nll)) if nll else float("nan")


def main():
    ref_root = sys.argv[1]
    packs = dict(a.split("=", 1) for a in sys.argv[2:])
    man = json.load(open(os.path.join(ref_root, "MOSS-TTS-Nano-100M-ONNX", "browser_poc_manifest.json")))
    ref = Pack(ref_root)
    assert ref.cached is not None, "FP32 root needs moss_tts_local_cached_step.onnx"
    runs = {"fp32": ref, **{k: Pack(v) for k, v in packs.items()}}
    cases = []
    for sample in man["text_samples"][:2]:
        voice = man["builtin_voices"][0 if sample["id"].startswith("zh") else 7]
        cases.append(build_rows(man, voice, sample["text_token_ids"]))
    result = {}
    for name, pack in runs.items():
        nlls, lens = [], []
        for ci, rows in enumerate(cases):
            for seed in (1234, 4321):
                frames = generate(pack, man, rows, seed, 260)
                lens.append(len(frames))
                nlls.append(score(ref, man, rows, frames))
        result[name] = {"fp32_nll_per_token": round(float(np.mean(nlls)), 4), "frames": lens}
    base = result["fp32"]["fp32_nll_per_token"]
    for name in result:
        result[name]["delta_vs_fp32"] = round(result[name]["fp32_nll_per_token"] - base, 4)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
