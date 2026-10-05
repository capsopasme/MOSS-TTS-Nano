#!/usr/bin/env bash
# Verifies that every static Regex/Pattern in the engine compiles under ICU (Android's regex
# engine). The desktop-JVM mock test cannot catch this: the JDK accepts syntax ICU rejects.
set -euo pipefail
KOTLIN_VERSION=2.0.21
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
W="${WORK_DIR:-$ROOT/build/regexcheck}"
mkdir -p "$W"
cd "$W"

if command -v kotlinc >/dev/null 2>&1; then
  KOTLINC=kotlinc
else
  if [ ! -d kotlinc ]; then
    curl -fsSL -o kc.zip "https://github.com/JetBrains/kotlin/releases/download/v$KOTLIN_VERSION/kotlin-compiler-$KOTLIN_VERSION.zip"
    unzip -q kc.zip
  fi
  KOTLINC="$W/kotlinc/bin/kotlinc"
fi
KOTLIN="$(dirname "$KOTLINC")/kotlin"
[ -x "$KOTLIN" ] || KOTLIN=kotlin

# Sources with static patterns. They must be pure Kotlin (no Android / ORT imports).
ENGINE="$ROOT/engine/src/main/java/io/github/capsopasme/mossnano/engine"
SRCS=$(grep -lE 'Regex\(|Pattern\.compile|toRegex\(' "$ENGINE"/*.kt)
rm -rf classes && mkdir classes
"$KOTLINC" -nowarn $SRCS "$ROOT/tools/regex/RegexDump.kt" -d classes > kotlinc.log 2>&1 || { cat kotlinc.log; exit 1; }
"$KOTLIN" -cp classes RegexDumpKt classes > patterns.tsv
python3 "$ROOT/tools/regex/icu_regex_check.py" patterns.tsv
