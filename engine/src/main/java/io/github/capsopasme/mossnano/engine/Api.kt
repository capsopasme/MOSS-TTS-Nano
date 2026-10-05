package io.github.capsopasme.mossnano.engine

import ai.onnxruntime.OrtSession
import java.io.Closeable
import java.io.File
import java.util.concurrent.CopyOnWriteArrayList

class EngineOptions(
    /** Directory containing MOSS-TTS-Nano-100M-ONNX/ and MOSS-Audio-Tokenizer-Nano-ONNX/. */
    val modelRoot: File,
    /** Threads of the shared LM pool (prefill/decode/local). 4 = X4 + 3×A720 on SD 8 Gen 3. */
    val lmThreads: Int = 4,
    /** Threads of the codec pool; it runs concurrently with the LM. */
    val codecThreads: Int = 2,
    /** Let LM pool threads spin between graph calls (lower latency, a bit more power). */
    val allowSpinning: Boolean = true,
    /** Official default for voice-clone text chunking. */
    val maxTextTokensPerChunk: Int = 75,
    /**
     * How far (in codec frames, 80 ms each) the LM may run ahead of the codec/audio output.
     * Bounds memory and stops burning CPU far ahead of playback on long texts.
     */
    val maxQueuedFrames: Int = 75,
    /** Keep ORT's workers, the codec thread and the calling thread off the little cores. */
    val pinToPerformanceCores: Boolean = true,
)

fun interface TextTokenizer : Closeable {
    fun encode(text: String): IntArray
    override fun close() {}
}

/**
 * Receives streamed audio on the engine's codec thread.
 * Samples are interleaved float32 in [-1, 1]; the array is reused after the call returns.
 */
interface AudioSink {
    fun onStart(sampleRate: Int, channels: Int) {}

    /** @return false to abort synthesis. */
    fun onAudio(samples: FloatArray, frames: Int): Boolean

    fun onFinish() {}
}

class SynthRequest(
    val text: String,
    val voice: VoicePrompt,
    /** null = random each time. */
    val seed: Long? = 1234L,
    val normalizeText: Boolean = true,
    val maxFramesPerChunk: Int = 375,
)

/** Thread-safe cancellation; also aborts an ORT run that is in flight. */
class CancelSignal {
    @Volatile var isCancelled: Boolean = false
        private set
    private val runOptions = CopyOnWriteArrayList<OrtSession.RunOptions>()

    fun cancel() {
        isCancelled = true
        for (r in runOptions) runCatching { r.setTerminate(true) }
    }

    internal fun attach(r: OrtSession.RunOptions) {
        runOptions += r
        if (isCancelled) runCatching { r.setTerminate(true) }
    }

    internal fun detach(r: OrtSession.RunOptions) {
        runOptions -= r
    }
}

class SynthStats {
    @Volatile var timeToFirstAudioMs: Long = -1
    var chunks = 0
    var textTokens = 0
    var prefillMs = 0L
    var prefillRows = 0
    var frames = 0
    var decodeMs = 0L
    var localMs = 0L
    @Volatile var codecCalls = 0
    @Volatile var codecMs = 0L
    @Volatile var audioFrames = 0L
    var wallMs = 0L
    var sampleRate = 48000

    val audioSeconds: Double get() = audioFrames.toDouble() / sampleRate
    val rtf: Double get() = if (audioSeconds > 0) wallMs / 1000.0 / audioSeconds else 0.0
    val msPerFrameLm: Double get() = if (frames > 0) (decodeMs + localMs).toDouble() / frames else 0.0

    fun summary(): String = buildString {
        append("首音 ${timeToFirstAudioMs}ms · RTF ${"%.3f".format(rtf)} · 音频 ${"%.1f".format(audioSeconds)}s · 耗时 ${wallMs}ms\n")
        append("prefill ${prefillMs}ms/${prefillRows}行 · LM ${"%.1f".format(msPerFrameLm)}ms/帧")
        if (frames > 0) append(" (decode ${decodeMs / frames} + local ${localMs / frames})")
        append(" · ${frames}帧 · ${chunks}段\n")
        append("codec ${codecCalls}次 ${codecMs}ms")
        if (frames > 0) append(" (${"%.1f".format(codecMs.toDouble() / frames)}ms/帧, 与LM并行)")
    }
}

class TtsException(message: String, cause: Throwable? = null) : Exception(message, cause)
