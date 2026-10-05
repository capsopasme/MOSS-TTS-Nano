package io.github.capsopasme.mossnano.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxTensorLike
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.Closeable

/**
 * Stateful streaming decoder for MOSS-Audio-Tokenizer-Nano (`moss_audio_tokenizer_decode_step.onnx`).
 *
 * The codec is a causal transformer with sliding-window KV caches; the cache tensors are
 * carried from one call to the next exactly like the official Python
 * `CodecStreamingDecodeSession`, so audio can be produced frame-by-frame
 * (1 frame = 3840 samples = 80 ms @ 48 kHz) instead of once per utterance.
 *
 * Efficiency details:
 *  - the zero "initial state" tensors are allocated once and reused for every reset;
 *  - state outputs are fed back as-is (no copies through the JVM);
 *  - input code / length buffers are direct and reused.
 */
internal class CodecStreamDecoder(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val cfg: CodecConfig,
    private val maxFramesPerCall: Int = 16,
) : Closeable {
    private val initialState: Map<String, OnnxTensor>
    private var stateResult: OrtSession.Result? = null
    private val feeds = LinkedHashMap<String, OnnxTensorLike>()

    private val codeBuffers = HashMap<Int, Pair<java.nio.IntBuffer, OnnxTensor>>()
    private val lengthBuf = directInts(1)
    private val lengthTensor: OnnxTensor = OnnxTensor.createTensor(env, lengthBuf, longArrayOf(1))

    /** Interleaved output scratch, grown on demand. */
    private var interleaved = FloatArray(0)

    init {
        val state = LinkedHashMap<String, OnnxTensor>()
        for (spec in cfg.stateSpecs) {
            val count = spec.shape.fold(1L) { a, b -> a * b }.toInt()
            state[spec.inputName] = when (spec.kind) {
                1 -> OnnxTensor.createTensor(env, directFloats(count), spec.shape)
                2 -> {
                    val b = directInts(count)
                    for (i in 0 until count) b.put(i, -1)
                    OnnxTensor.createTensor(env, b, spec.shape)
                }
                else -> OnnxTensor.createTensor(env, directInts(count), spec.shape)
            }
        }
        initialState = state
    }

    fun reset() {
        stateResult?.close()
        stateResult = null
    }

    /**
     * Decodes [count] frames starting at [frames][offset]. Returns the number of
     * interleaved sample-frames written to [output]'s buffer (see [lastOutput]).
     */
    fun decode(frames: List<IntArray>, offset: Int, count: Int): Int {
        require(count in 1..maxFramesPerCall)
        val nq = cfg.numQuantizers
        val (codeBuf, codeTensor) = codeBuffers.getOrPut(count) {
            val b = directInts(count * nq)
            b to OnnxTensor.createTensor(env, b, longArrayOf(1, count.toLong(), nq.toLong()))
        }
        for (f in 0 until count) {
            val frame = frames[offset + f]
            val base = f * nq
            for (q in 0 until nq) codeBuf.put(base + q, if (q < frame.size) frame[q] else 0)
        }
        lengthBuf.put(0, count)

        feeds.clear()
        feeds["audio_codes"] = codeTensor
        feeds["audio_code_lengths"] = lengthTensor
        val prev = stateResult
        for (spec in cfg.stateSpecs) {
            feeds[spec.inputName] = if (prev == null) initialState.getValue(spec.inputName) else prev.tensor(spec.outputName)
        }
        val result = session.run(feeds)
        prev?.close()
        stateResult = result

        val audioLength = result.tensor("audio_lengths").readInts()[0]
        val audio = result.tensor("audio")
        // audio: [1, C, N] channel-major float
        val shape = audio.info.shape
        val channels = shape[1].toInt()
        val total = shape[2].toInt()
        val n = audioLength.coerceIn(0, total)
        if (n == 0) return 0
        val fb = audio.floatBuffer
        val need = n * channels
        if (interleaved.size < need) interleaved = FloatArray(need)
        val out = interleaved
        for (c in 0 until channels) {
            val src = c * total
            var dst = c
            for (i in 0 until n) {
                out[dst] = fb.get(src + i)
                dst += channels
            }
        }
        return n
    }

    /** Buffer holding the interleaved samples from the last [decode] call. */
    val lastOutput: FloatArray get() = interleaved

    override fun close() {
        reset()
        initialState.values.forEach { it.close() }
        codeBuffers.values.forEach { it.second.close() }
        lengthTensor.close()
    }
}
