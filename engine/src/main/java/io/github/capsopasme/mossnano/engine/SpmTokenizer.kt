package io.github.capsopasme.mossnano.engine

import java.io.File

/**
 * SentencePiece tokenizer (google/sentencepiece, statically linked through JNI) loaded from
 * the official `tokenizer.model`, so token ids match the Python runtime exactly.
 */
class SpmTokenizer(modelFile: File) : TextTokenizer {
    private var handle: Long

    init {
        ensureLoaded()
        require(modelFile.isFile) { "tokenizer model not found: ${modelFile.absolutePath}" }
        handle = nativeLoad(modelFile.absolutePath)
        if (handle == 0L) throw IllegalStateException("Failed to load SentencePiece model ${modelFile.absolutePath}")
    }

    @Synchronized
    override fun encode(text: String): IntArray {
        check(handle != 0L) { "tokenizer closed" }
        if (text.isEmpty()) return IntArray(0)
        return nativeEncode(handle, text) ?: throw IllegalStateException("SentencePiece encode failed")
    }

    @Synchronized
    fun vocabSize(): Int = if (handle == 0L) 0 else nativeVocabSize(handle)

    @Synchronized
    override fun close() {
        if (handle != 0L) {
            nativeFree(handle)
            handle = 0L
        }
    }

    companion object {
        @Volatile private var loaded = false

        /** Optional override for tests on a desktop JVM (absolute path of the shared library). */
        @JvmStatic var libraryPathOverride: String? = null

        @Synchronized
        private fun ensureLoaded() {
            if (loaded) return
            val override = libraryPathOverride
            if (override != null) System.load(override) else System.loadLibrary("mossnano_jni")
            loaded = true
        }

        @JvmStatic private external fun nativeLoad(path: String): Long
        @JvmStatic private external fun nativeEncode(handle: Long, text: String): IntArray?
        @JvmStatic private external fun nativeVocabSize(handle: Long): Int
        @JvmStatic private external fun nativeFree(handle: Long)
    }
}
