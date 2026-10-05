#!/usr/bin/env bash
# Sourced by run_mock_test.sh / run_real_model.sh: builds what is needed to run the real engine
# code on a desktop JVM (linux x86_64 or aarch64): ONNX Runtime + its Java API and JNI glue,
# the app's tokenizer/affinity JNI library (same CMakeLists as the app) and a Kotlin compiler.
set -euo pipefail
ORT_VERSION=1.22.0
KOTLIN_VERSION=2.0.21
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
W="${WORK_DIR:-$ROOT/build/mocktest}"
mkdir -p "$W"
cd "$W"

case "$(uname -m)" in
  x86_64) ORT_ARCH=x64 ;;
  aarch64 | arm64) ORT_ARCH=aarch64 ;;
  *) echo "unsupported arch $(uname -m)" >&2; exit 1 ;;
esac

# --- ONNX Runtime C library + Java API classes + JNI glue
ORTD="$W/onnxruntime-linux-$ORT_ARCH-$ORT_VERSION"
if [ ! -d "$ORTD" ]; then
  curl -fsSL -o ort.tgz "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/onnxruntime-linux-$ORT_ARCH-$ORT_VERSION.tgz"
  tar xzf ort.tgz
fi
if [ ! -d ortsrc ]; then
  git clone -q --depth 1 --branch "v$ORT_VERSION" --filter=blob:none --sparse https://github.com/microsoft/onnxruntime.git ortsrc
  git -C ortsrc sparse-checkout set java/src/main include/onnxruntime/core/providers
fi
JSRC="$W/ortsrc/java/src/main"
if [ ! -f ortjni/.built-$ORT_ARCH ]; then
  rm -rf ortclasses jnih ortjni && mkdir -p ortclasses jnih ortinc/onnxruntime/core/session ortjni
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
  touch ortjni/.built-$ORT_ARCH
fi

# --- tokenizer/affinity JNI (same CMakeLists as the app) + spm tools
cmake -S "$ROOT/engine/src/main/cpp" -B jni -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build jni --target mossnano_jni spm_train spm_encode >/dev/null
SPM_BIN="$W/jni/_deps/sentencepiece-build/src"
JNI_LIB="$W/jni/libmossnano_jni.so"

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

# compile_engine <out_dir> <extra .kt files...>
compile_engine() {
  local out="$1"; shift
  "$KOTLINC" -jvm-target 17 -cp ortclasses -d "$out" $(find "$ROOT/engine/src/main/java" -name "*.kt") "$@"
}

# run_kt <classes_dir> <MainClass> <args...>
run_kt() {
  local classes="$1"; shift
  LC_ALL=C.UTF-8 LANG=C.UTF-8 java -Dstdout.encoding=UTF-8 -Dfile.encoding=UTF-8 -Donnxruntime.native.path="$W/ortjni" \
    -cp "$classes:$W/ortclasses:$STDLIB" "$@"
}
