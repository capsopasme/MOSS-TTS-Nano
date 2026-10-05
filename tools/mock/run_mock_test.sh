#!/usr/bin/env bash
# Runs the real engine code on a desktop JVM (linux x86_64 / aarch64) against mock ONNX graphs
# with the exact I/O contract of the MOSS export (tools/mock/EngineMockTest.kt).
set -euo pipefail
source "$(dirname "$0")/jvm_env.sh"

# toy SentencePiece model
python3 - > corpus.txt <<'PY'
import random
random.seed(0)
zh = "今天天气很好我们一起去公园散步吧这是一个流式语音合成的测试文本模型运行在手机上速度很快效果自然一二三四五六七八九十百千万亿年月日点分秒"
en = "the quick brown fox jumps over the lazy dog streaming speech synthesis on device fast natural voice hello world"
for i in range(3000):
    n = random.randint(5, 30)
    print("".join(random.choice(zh) for _ in range(n)) + "，" + " ".join(random.choice(en.split()) for _ in range(5)) + "。")
PY
"$SPM_BIN/spm_train" --input=corpus.txt --model_prefix=toy --vocab_size=400 --character_coverage=1.0 --model_type=unigram --minloglevel=2

# mock graphs
rm -rf mock
python3 "$ROOT/tools/mock/make_mock_models.py" mock 20
cp toy.model mock/MOSS-TTS-Nano-100M-ONNX/tokenizer.model
EXPECTED=$(echo "今天天气很好，hello world。" | "$SPM_BIN/spm_encode" --model=toy.model --output_format=id)
echo "spm_encode ids: $EXPECTED"

rm -rf testclasses
compile_engine testclasses "$ROOT/tools/mock/EngineMockTest.kt"
run_kt testclasses EngineMockTestKt "$W/mock" "$JNI_LIB" "$EXPECTED"
