#!/usr/bin/env bash
# Builds everything needed to run the real engine code on a desktop JVM (linux-x64) against
# mock ONNX graphs, then runs tools/mock/EngineMockTest.kt.
set -euo pipefail
ORT_VERSION=1.22.0
KOTLIN_VERSION=2.0.21
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
W="${WORK_DIR:-$ROOT/build/mocktest}"
mkdir -p "$W"
cd "$W"

# --- ONNX Runtime C library + Java API classes + JNI glue
if [ ! -d "onnxruntime-linux-x64-$ORT_VERSION" ]; then
  curl -fsSL -o ort.tgz "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/onnxruntime-linux-x64-$ORT_VERSION.tgz"
  tar xzf ort.tgz
fi
ORTD="$W/onnxruntime-linux-x64-$ORT_VERSION"
if [ ! -d ortsrc ]; then
  git clone -q --depth 1 --branch "v$ORT_VERSION" --filter=blob:none --sparse https://github.com/microsoft/onnxruntime.git ortsrc
  git -C ortsrc sparse-checkout set java/src/main include/onnxruntime/core/providers
fi
JSRC="$W/ortsrc/java/src/main"
rm -rf ortclasses jnih && mkdir -p ortclasses jnih ortinc/onnxruntime/core/session ortjni
javac -nowarn -h jnih -d ortclasses $(find "$JSRC/java" "$JSRC/jvm" -name "*.java") 2>&1 | grep -v '^Note:' || true
cp "$ORTD"/include/*.h ortinc/ && cp "$ORTD"/include/*.h ortinc/onnxruntime/core/session/
cp -r ortsrc/include/onnxruntime/core/providers ortinc/onnxruntime/core/
cp "$ORTD/include/cpu_provider_factory.h" ortinc/onnxruntime/core/providers/cpu/ 2>/dev/null || true
echo "#define ORT_VERSION \"$ORT_VERSION\"" > ortinc/onnxruntime_config.h
JH="$(dirname "$(dirname "$(readlink -f "$(which javac)")")")"
gcc -O2 -shared -fPIC -Ijnih -Iortinc -I"$JH/include" -I"$JH/include/linux" -I"$JSRC/native" \
  $(ls "$JSRC"/native/*.c | grep -v Training) -L"$ORTD/lib" -lonnxruntime -Wl,-rpath,"$ORTD/lib" \
  -o ortjni/libonnxruntime4j_jni.so
cp "$ORTD"/lib/libonnxruntime.so* ortjni/

# --- tokenizer JNI (same CMakeLists as the app) + spm tools for a toy model
cmake -S "$ROOT/engine/src/main/cpp" -B jni -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build jni --target mossnano_jni spm_train spm_encode >/dev/null
SPM_BIN="$W/jni/_deps/sentencepiece-build/src"
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

# --- mock graphs
rm -rf mock
python3 "$ROOT/tools/mock/make_mock_models.py" mock 20
cp toy.model mock/MOSS-TTS-Nano-100M-ONNX/tokenizer.model
EXPECTED=$(echo "今天天气很好，hello world。" | "$SPM_BIN/spm_encode" --model=toy.model --output_format=id)
echo "spm_encode ids: $EXPECTED"

# --- Kotlin compiler
if command -v kotlinc >/dev/null 2>&1; then
  KOTLINC=kotlinc
else
  if [ ! -d kotlinc ]; then
    curl -fsSL -o kc.zip "https://github.com/JetBrains/kotlin/releases/download/v$KOTLIN_VERSION/kotlin-compiler-$KOTLIN_VERSION.zip"
    unzip -q kc.zip
  fi
  KOTLINC="$W/kotlinc/bin/kotlinc"
fi
STDLIB="$(dirname "$(readlink -f "$(command -v "$KOTLINC")")")/../lib/kotlin-stdlib.jar"
"$KOTLINC" -jvm-target 17 -cp ortclasses -d testclasses \
  $(find "$ROOT/engine/src/main/java" -name "*.kt") "$ROOT/tools/mock/EngineMockTest.kt"

java -Dstdout.encoding=UTF-8 -Dfile.encoding=UTF-8 -Donnxruntime.native.path="$W/ortjni" \
  -cp "testclasses:ortclasses:$STDLIB" EngineMockTestKt "$W/mock" "$W/jni/libmossnano_jni.so" "$EXPECTED"
