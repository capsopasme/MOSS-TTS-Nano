// Minimal JNI bridge to google/sentencepiece for MOSS-TTS-Nano's tokenizer.model,
// plus per-thread CPU affinity helpers (used to keep the inference threads off the
// little cores).
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <jni.h>
#include <errno.h>
#include <sched.h>

#include <memory>
#include <string>
#include <vector>

#include "sentencepiece_processor.h"

namespace {

std::string JStringToUtf8(JNIEnv* env, jstring s) {
  // GetStringUTFChars returns *modified* UTF-8 (breaks supplementary chars / NUL),
  // so go through UTF-16 and convert ourselves.
  const jsize len = env->GetStringLength(s);
  const jchar* chars = env->GetStringChars(s, nullptr);
  std::string out;
  out.reserve(static_cast<size_t>(len) * 3);
  for (jsize i = 0; i < len; ++i) {
    uint32_t cp = chars[i];
    if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < len) {
      const uint32_t lo = chars[i + 1];
      if (lo >= 0xDC00 && lo <= 0xDFFF) {
        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
        ++i;
      }
    }
    if (cp < 0x80) {
      out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
      out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }
  env->ReleaseStringChars(s, chars);
  return out;
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_io_github_capsopasme_mossnano_engine_SpmTokenizer_nativeLoad(JNIEnv* env, jclass, jstring path) {
  auto sp = std::make_unique<sentencepiece::SentencePieceProcessor>();
  const std::string p = JStringToUtf8(env, path);
  const auto status = sp->Load(p);
  if (!status.ok()) return 0;
  return reinterpret_cast<jlong>(sp.release());
}

JNIEXPORT jintArray JNICALL
Java_io_github_capsopasme_mossnano_engine_SpmTokenizer_nativeEncode(JNIEnv* env, jclass, jlong handle,
                                                                   jstring text) {
  auto* sp = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(handle);
  if (sp == nullptr) return nullptr;
  std::vector<int> ids;
  const auto status = sp->Encode(JStringToUtf8(env, text), &ids);
  if (!status.ok()) return nullptr;
  jintArray out = env->NewIntArray(static_cast<jsize>(ids.size()));
  if (out == nullptr) return nullptr;
  if (!ids.empty()) {
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(ids.size()), reinterpret_cast<const jint*>(ids.data()));
  }
  return out;
}

JNIEXPORT jint JNICALL
Java_io_github_capsopasme_mossnano_engine_SpmTokenizer_nativeVocabSize(JNIEnv*, jclass, jlong handle) {
  auto* sp = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(handle);
  return sp == nullptr ? 0 : sp->GetPieceSize();
}

JNIEXPORT void JNICALL
Java_io_github_capsopasme_mossnano_engine_SpmTokenizer_nativeFree(JNIEnv*, jclass, jlong handle) {
  delete reinterpret_cast<sentencepiece::SentencePieceProcessor*>(handle);
}

// Affinity of the *calling thread* (pid 0 = calling thread for sched_*affinity on Linux).
// Threads created afterwards by this thread (e.g. ONNX Runtime's pools) inherit it.
JNIEXPORT jint JNICALL
Java_io_github_capsopasme_mossnano_engine_CpuAffinity_nativeSetCurrentThreadMask(JNIEnv*, jclass, jlong mask) {
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int i = 0; i < 64; ++i) {
    if ((static_cast<unsigned long long>(mask) >> i) & 1ULL) CPU_SET(i, &set);
  }
  return sched_setaffinity(0, sizeof(set), &set) == 0 ? 0 : errno;
}

JNIEXPORT jlong JNICALL
Java_io_github_capsopasme_mossnano_engine_CpuAffinity_nativeGetCurrentThreadMask(JNIEnv*, jclass) {
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) != 0) return 0;
  unsigned long long mask = 0;
  for (int i = 0; i < 64; ++i) {
    if (CPU_ISSET(i, &set)) mask |= (1ULL << i);
  }
  return static_cast<jlong>(mask);
}

}  // extern "C"
