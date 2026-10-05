#!/usr/bin/env bash
# Runs the real engine on a desktop JVM against a real model directory (FP32 or the INT8 pack)
# and writes the streamed audio to a WAV file.
#   tools/mock/run_real_model.sh <model_root> <out.wav> [text] [voice_id]
# LM_THREADS (default 2) sets the LM thread count.
set -euo pipefail
MODEL_ROOT="$(readlink -f "$1")"
OUT="$(readlink -f "$(dirname "$2")")/$(basename "$2")"
shift 2
source "$(dirname "$0")/jvm_env.sh"
if [ ! -d realclasses ] || [ -n "$(find "$ROOT/engine/src/main/java" "$ROOT/tools/mock/RealModelRun.kt" -newer realclasses -name '*.kt' | head -1)" ]; then
  rm -rf realclasses
  compile_engine realclasses "$ROOT/tools/mock/RealModelRun.kt"
fi
run_kt realclasses RealModelRunKt "$MODEL_ROOT" "$OUT" "$JNI_LIB" "$@"
