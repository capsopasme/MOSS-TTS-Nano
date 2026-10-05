package io.github.capsopasme.mossnano.engine

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.IntBuffer

/**
 * ORT environment with ONE global intra-op thread pool shared by the three LM graphs
 * (prefill / decode_step / local_fixed_sampled_frame).
 *
 * Why: with per-session pools, the pool of the graph that just finished keeps spinning
 * while the next graph's pool starts working, so they fight over the same big cores.
 * A single pool means exactly N busy threads during generation.
 *
 * The environment is a process singleton, so the global thread count is fixed for the
 * lifetime of the process (changing it needs an app restart).
 */
object OrtEnv {
    @Volatile var globalThreads: Int = 0
        private set
    @Volatile var globalSpinning: Boolean = true
        private set
    @Volatile var hasGlobalPool: Boolean = false
        private set

    private var env: OrtEnvironment? = null

    @Synchronized
    fun get(threads: Int, allowSpinning: Boolean): OrtEnvironment {
        env?.let { return it }
        val created = try {
            OrtEnvironment.ThreadingOptions().use { opts ->
                opts.setGlobalIntraOpNumThreads(threads.coerceAtLeast(1))
                opts.setGlobalInterOpNumThreads(1)
                opts.setGlobalSpinControl(allowSpinning)
                opts.setGlobalDenormalAsZero()
                OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, "mossnano", opts)
            }.also {
                hasGlobalPool = true
                globalThreads = threads
                globalSpinning = allowSpinning
            }
        } catch (e: IllegalStateException) {
            // Someone else created the env first -> fall back to per-session pools.
            hasGlobalPool = false
            OrtEnvironment.getEnvironment()
        }
        env = created
        return created
    }
}

/**
 * @param dynamicShapes true for prefill / decode_step, whose input shapes change on every
 *   call (prompt length, growing KV cache). ORT's memory-pattern planner would record a new
 *   allocation plan for every new shape and keep all of them, which only costs time and
 *   memory there; it stays on for the fixed-shape local graph.
 */
internal fun lmSessionOptions(threadsIfNoGlobalPool: Int, spinning: Boolean, dynamicShapes: Boolean): OrtSession.SessionOptions =
    OrtSession.SessionOptions().apply {
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
        if (dynamicShapes) setMemoryPatternOptimization(false)
        if (OrtEnv.hasGlobalPool) {
            disablePerSessionThreads()
        } else {
            setIntraOpNumThreads(threadsIfNoGlobalPool)
            setInterOpNumThreads(1)
            addConfigEntry("session.intra_op.allow_spinning", if (spinning) "1" else "0")
        }
        addConfigEntry("session.inter_op.allow_spinning", "0")
    }

/** Codec runs concurrently with the LM on its own small pool, never spinning. */
internal fun codecSessionOptions(threads: Int): OrtSession.SessionOptions =
    OrtSession.SessionOptions().apply {
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
        setIntraOpNumThreads(threads.coerceAtLeast(1))
        setInterOpNumThreads(1)
        addConfigEntry("session.intra_op.allow_spinning", "0")
        addConfigEntry("session.inter_op.allow_spinning", "0")
    }

internal fun OrtEnvironment.openSession(file: File, options: OrtSession.SessionOptions): OrtSession {
    if (!file.isFile) throw MissingModelException("Missing ONNX file: ${file.absolutePath}")
    return createSession(file.absolutePath, options)
}

internal fun directInts(n: Int): IntBuffer =
    ByteBuffer.allocateDirect(n * 4).order(ByteOrder.nativeOrder()).asIntBuffer()

internal fun directFloats(n: Int): FloatBuffer =
    ByteBuffer.allocateDirect(n * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()

/** Reads an integer-like tensor (bool / int8..int64) into an IntArray. */
internal fun OnnxTensor.readInts(): IntArray {
    return when (info.type) {
        OnnxJavaType.INT32 -> intBuffer.let { b -> IntArray(b.remaining()) { b.get(it) } }
        OnnxJavaType.INT64 -> longBuffer.let { b -> IntArray(b.remaining()) { b.get(it).toInt() } }
        OnnxJavaType.INT16 -> shortBuffer.let { b -> IntArray(b.remaining()) { b.get(it).toInt() } }
        OnnxJavaType.INT8, OnnxJavaType.UINT8, OnnxJavaType.BOOL ->
            byteBuffer.let { b -> IntArray(b.remaining()) { b.get(it).toInt() and 0xFF } }
        else -> error("Unsupported integer tensor type ${info.type}")
    }
}

internal fun OrtSession.Result.tensor(name: String): OnnxTensor =
    get(name).orElseThrow { IllegalStateException("Missing ONNX output: $name") } as OnnxTensor
