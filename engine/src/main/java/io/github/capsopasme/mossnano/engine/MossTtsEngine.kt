package io.github.capsopasme.mossnano.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxTensorLike
import ai.onnxruntime.OnnxValue
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.Closeable
import java.io.File
import java.util.SplittableRandom
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Streaming MOSS-TTS-Nano inference on ONNX Runtime (CPU).
 *
 * Pipeline per text chunk:
 *
 *   [caller thread]  prefill ─► (local_fixed_sampled_frame ─► decode_step)×N ──frames──┐
 *                                                                                       ▼ bounded queue
 *   [codec thread]                        codec decode_step (stateful) ─► AudioSink (AudioTrack)
 *
 * The LM and the codec run concurrently, the codec batch size adapts to how much audio
 * is already buffered (1 frame for the very first audio → up to 8 once ahead), and every
 * per-step input tensor is a reused direct buffer, so the steady-state loop allocates
 * nothing on the JVM side.
 */
class MossTtsEngine private constructor(
    val options: EngineOptions,
    private val cfg: ModelConfig,
    private val tokenizer: TextTokenizer,
    private val log: (String) -> Unit,
) : Closeable {

    private val env: OrtEnvironment = OrtEnv.get(options.lmThreads, options.allowSpinning)

    private val lmOptions = lmSessionOptions(options.lmThreads, options.allowSpinning)
    private val codecOptions = codecSessionOptions(options.codecThreads)
    private val prefill: OrtSession
    private val decode: OrtSession
    private val local: OrtSession
    private val codecSession: OrtSession
    private val codec: CodecStreamDecoder

    val sampleRate: Int get() = cfg.codec.sampleRate
    val channels: Int get() = cfg.codec.channels
    val builtinVoices: List<VoicePrompt> get() = cfg.builtinVoices
    val loadMs: Long

    private val nVq = cfg.tts.nVq
    private val rowWidth = cfg.tts.rowWidth
    private val codebook = cfg.tts.codebookSize
    private val hiddenSize: Int

    // ---- reused per-step tensors (direct buffers, mutated in place) -------------------
    private val hidden: java.nio.FloatBuffer
    private val hiddenLocalIn: OnnxTensor      // [1, H]   (input of local frame)
    private val hiddenDecodeOut: OnnxTensor    // [1,1,H] or [1,H] (pinned decode output)
    private val seenMask = directInts(nVq * codebook)
    private val seenMaskTensor: OnnxTensor
    private val assistantU = directFloats(1)
    private val assistantUTensor: OnnxTensor
    private val audioU = directFloats(nVq)
    private val audioUTensor: OnnxTensor
    private val rowBuf = directInts(rowWidth)
    private val rowTensor: OnnxTensor
    private val pastLenBuf = directInts(1)
    private val pastLenTensor: OnnxTensor

    private val localFeeds: Map<String, OnnxTensorLike>
    private val decodeFeeds = LinkedHashMap<String, OnnxTensorLike>()
    private val decodePastInputs: List<String>        // past_key_0 …
    private val decodePresentOutputs: List<String>     // present_key_0 …
    private val prefillPresentOutputs: List<String>
    private val decodeRequested: Set<String>
    private val decodePinned: Map<String, OnnxValue>

    // ---- codec worker -----------------------------------------------------------------
    private val queue = ArrayBlockingQueue<Msg>(options.maxQueuedFrames.coerceAtLeast(8) + 4)
    private val worker: Thread
    @Volatile private var closed = false
    private val synthLock = ReentrantLock()

    init {
        val t0 = System.nanoTime()
        prefill = env.openSession(File(cfg.ttsDir, cfg.prefillFile), lmOptions)
        decode = env.openSession(File(cfg.ttsDir, cfg.decodeStepFile), lmOptions)
        local = env.openSession(File(cfg.ttsDir, cfg.localFixedFrameFile), lmOptions)
        codecSession = env.openSession(File(cfg.codecDir, cfg.codec.decodeStepFile), codecOptions)
        codec = CodecStreamDecoder(env, codecSession, cfg.codec)
        loadMs = (System.nanoTime() - t0) / 1_000_000

        val localHiddenInfo = local.inputInfo.getValue("global_hidden").info as TensorInfo
        val decodeHiddenInfo = decode.outputInfo.getValue("global_hidden").info as TensorInfo
        hiddenSize = localHiddenInfo.shape.last().toInt().takeIf { it > 0 }
            ?: decodeHiddenInfo.shape.last().toInt().takeIf { it > 0 } ?: 768
        hidden = directFloats(hiddenSize)
        hiddenLocalIn = OnnxTensor.createTensor(env, hidden, shapeOfRank(localHiddenInfo.shape.size, hiddenSize))
        // Same direct buffer, decode's shape: decode_step writes its global_hidden straight into
        // the memory the local graph reads next -> zero copies on the hot path.
        hiddenDecodeOut = OnnxTensor.createTensor(env, hidden, shapeOfRank(decodeHiddenInfo.shape.size, hiddenSize))

        seenMaskTensor = OnnxTensor.createTensor(env, seenMask, longArrayOf(1, nVq.toLong(), codebook.toLong()))
        assistantUTensor = OnnxTensor.createTensor(env, assistantU, longArrayOf(1))
        audioUTensor = OnnxTensor.createTensor(env, audioU, longArrayOf(1, nVq.toLong()))
        rowTensor = OnnxTensor.createTensor(env, rowBuf, longArrayOf(1, 1, rowWidth.toLong()))
        pastLenTensor = OnnxTensor.createTensor(env, pastLenBuf, longArrayOf(1))

        localFeeds = linkedMapOf(
            "global_hidden" to hiddenLocalIn,
            "repetition_seen_mask" to seenMaskTensor,
            "assistant_random_u" to assistantUTensor,
            "audio_random_u" to audioUTensor,
        )
        decodePastInputs = cfg.decodeInputNames.drop(2)
        decodePresentOutputs = cfg.decodeOutputNames.drop(1)
        prefillPresentOutputs = cfg.prefillOutputNames.drop(1)
        require(decodePastInputs.size == decodePresentOutputs.size && decodePastInputs.size == prefillPresentOutputs.size) {
            "KV cache name lists do not line up"
        }
        decodeRequested = LinkedHashSet(decodePresentOutputs)
        decodePinned = mapOf("global_hidden" to hiddenDecodeOut)

        worker = Thread({ codecLoop() }, "moss-codec").apply {
            isDaemon = true
            priority = Thread.MAX_PRIORITY
            start()
        }
        log("engine loaded in ${loadMs}ms; globalPool=${OrtEnv.hasGlobalPool} lmThreads=${OrtEnv.globalThreads} codecThreads=${options.codecThreads}")
    }

    private fun shapeOfRank(rank: Int, h: Int): LongArray =
        if (rank == 3) longArrayOf(1, 1, h.toLong()) else longArrayOf(1, h.toLong())

    // =================================================================================
    // Public API
    // =================================================================================

    fun warmup() {
        val voice = builtinVoices.firstOrNull() ?: return
        val sink = object : AudioSink {
            override fun onAudio(samples: FloatArray, frames: Int) = true
        }
        synthesize(SynthRequest("你好。", voice, seed = 1L, normalizeText = false, maxFramesPerChunk = 3), sink, CancelSignal())
    }

    /** Splits/normalizes [text] the same way [synthesize] will (for UI/progress). */
    fun prepareChunks(text: String, normalize: Boolean): List<String> {
        val prepared = if (normalize) TextNormalizer.normalize(text) else text.trim()
        if (prepared.isBlank()) return emptyList()
        return TextChunker.split(prepared, options.maxTextTokensPerChunk) { tokenizer.encode(it).size }
    }

    /**
     * Synthesizes [request] and streams audio to [sink]. Blocks until all audio has been
     * delivered to the sink (or cancelled). Not reentrant: concurrent calls are serialized.
     */
    fun synthesize(request: SynthRequest, sink: AudioSink, cancel: CancelSignal = CancelSignal()): SynthStats {
        check(!closed) { "engine closed" }
        synthLock.withLock {
            val stats = SynthStats().apply { sampleRate = cfg.codec.sampleRate }
            val startNs = System.nanoTime()
            val job = Job(sink, cancel, stats, startNs)
            val runOptions = OrtSession.RunOptions()
            cancel.attach(runOptions)
            putAlways(Msg.Begin(job))
            try {
                val chunks = prepareChunks(request.text, request.normalizeText)
                val rng = SplittableRandom(request.seed ?: System.nanoTime())
                for ((index, chunk) in chunks.withIndex()) {
                    if (cancel.isCancelled || job.error != null) break
                    generateChunk(chunk, request, rng, runOptions, job)
                    stats.chunks++
                    val pause = if (index < chunks.lastIndex) TextChunker.pauseSeconds(chunk) else 0.0
                    if (!put(Msg.ChunkEnd((pause * sampleRate).toInt()), cancel)) break
                }
            } catch (e: OrtException) {
                if (!cancel.isCancelled) throw TtsException("ONNX Runtime error: ${e.message}", e)
            } finally {
                putAlways(Msg.Finish)
                job.done.await()
                cancel.detach(runOptions)
                runOptions.close()
            }
            job.error?.let { throw TtsException("audio pipeline error: ${it.message}", it) }
            stats.wallMs = (System.nanoTime() - startNs) / 1_000_000
            return stats
        }
    }

    // =================================================================================
    // LM generation (caller thread)
    // =================================================================================

    private fun generateChunk(text: String, req: SynthRequest, rng: SplittableRandom, runOptions: OrtSession.RunOptions, job: Job) {
        val stats = job.stats
        val cancel = job.cancel
        val textIds = tokenizer.encode(text)
        stats.textTokens += textIds.size
        val tts = cfg.tts
        val tpl = cfg.templates

        // ---- build prompt rows: [user prefix + <audio_start>] [voice codes] [<audio_end> + after_ref + text + assistant prefix + <audio_start>]
        val prefix = tpl.userPrefix + tts.audioStartTokenId
        val suffix = intArrayOf(tts.audioEndTokenId) + tpl.userAfterReference + textIds + tpl.assistantPrefix + tts.audioStartTokenId
        val voice = req.voice.codes
        val seqLen = prefix.size + voice.size + suffix.size
        val ids = directInts(seqLen * rowWidth)
        var p = 0
        fun textRow(token: Int) {
            ids.put(p++, token)
            repeat(nVq) { ids.put(p++, tts.audioPadTokenId) }
        }
        prefix.forEach(::textRow)
        for (codeRow in voice) {
            ids.put(p++, tts.audioUserSlotTokenId)
            for (q in 0 until nVq) ids.put(p++, if (q < codeRow.size) codeRow[q] else tts.audioPadTokenId)
        }
        suffix.forEach(::textRow)
        val mask = directInts(seqLen)
        for (i in 0 until seqLen) mask.put(i, 1)

        var past: OrtSession.Result? = null
        try {
            val t0 = System.nanoTime()
            OnnxTensor.createTensor(env, ids, longArrayOf(1, seqLen.toLong(), rowWidth.toLong())).use { idsT ->
                OnnxTensor.createTensor(env, mask, longArrayOf(1, seqLen.toLong())).use { maskT ->
                    past = prefill.run(mapOf("input_ids" to idsT, "attention_mask" to maskT), runOptions)
                }
            }
            // last-position hidden state -> shared hidden buffer
            val gh = past!!.tensor("global_hidden")
            val fb = gh.floatBuffer
            val start = fb.remaining() - hiddenSize
            for (i in 0 until hiddenSize) hidden.put(i, fb.get(start + i))
            stats.prefillMs += (System.nanoTime() - t0) / 1_000_000
            stats.prefillRows += seqLen

            var pastLen = seqLen
            var presentNames = prefillPresentOutputs
            for (i in 0 until seenMask.capacity()) seenMask.put(i, 0)
            val maxFrames = minOf(req.maxFramesPerChunk, cfg.maxNewFrames)

            for (step in 0 until maxFrames) {
                if (cancel.isCancelled || job.error != null) return

                // ---- local transformer: samples all 16 codebooks of the next frame in one graph
                val tl = System.nanoTime()
                assistantU.put(0, uniform(rng))
                for (q in 0 until nVq) audioU.put(q, uniform(rng))
                val localOut = local.run(localFeeds, runOptions)
                val shouldContinue: Boolean
                val frame: IntArray
                try {
                    shouldContinue = localOut.tensor("should_continue").readInts()[0] > 0
                    frame = localOut.tensor("frame_token_ids").readInts()
                } finally {
                    localOut.close()
                }
                stats.localMs += (System.nanoTime() - tl) / 1_000_000
                if (!shouldContinue) break

                if (!put(Msg.Frame(frame), cancel)) return
                stats.frames++

                for (q in 0 until nVq) {
                    val tok = frame[q]
                    if (tok in 0 until codebook) seenMask.put(q * codebook + tok, 1)
                }

                // ---- global transformer: one step with KV cache
                val td = System.nanoTime()
                rowBuf.put(0, tts.audioAssistantSlotTokenId)
                for (q in 0 until nVq) rowBuf.put(q + 1, frame[q])
                pastLenBuf.put(0, pastLen)
                decodeFeeds.clear()
                decodeFeeds["input_ids"] = rowTensor
                decodeFeeds["past_valid_lengths"] = pastLenTensor
                val prev = past!!
                for (k in decodePastInputs.indices) {
                    decodeFeeds[decodePastInputs[k]] = prev.tensor(presentNames[k])
                }
                val next = decode.run(decodeFeeds, decodeRequested, decodePinned, runOptions)
                prev.close()
                past = next
                presentNames = decodePresentOutputs
                pastLen++
                stats.decodeMs += (System.nanoTime() - td) / 1_000_000
            }
        } finally {
            past?.close()
        }
    }

    /** Matches the official runtime: u in [0, 0.99999994]. */
    private fun uniform(rng: SplittableRandom): Float = rng.nextDouble().toFloat().coerceIn(0f, 0.99999994f)

    // =================================================================================
    // Codec worker thread
    // =================================================================================

    private class Job(val sink: AudioSink, val cancel: CancelSignal, val stats: SynthStats, val startNs: Long) {
        val done = CountDownLatch(1)
        @Volatile var error: Throwable? = null
    }

    private sealed class Msg {
        class Begin(val job: Job) : Msg()
        class Frame(val tokens: IntArray) : Msg()
        class ChunkEnd(val pauseFrames: Int) : Msg()
        object Finish : Msg()
        object Shutdown : Msg()
    }

    private fun put(msg: Msg, cancel: CancelSignal): Boolean {
        while (!queue.offer(msg, 50, TimeUnit.MILLISECONDS)) {
            if (cancel.isCancelled || closed) return false
        }
        return true
    }

    private fun putAlways(msg: Msg) {
        while (!queue.offer(msg, 50, TimeUnit.MILLISECONDS)) {
            if (closed && msg !is Msg.Shutdown) return
        }
    }

    private fun codecLoop() {
        runCatching { threadPriorityHook?.invoke() }
        val pending = ArrayList<IntArray>(32)
        var job: Job? = null
        var emittedFrames = 0L
        var firstEmitNs = 0L
        val silence = FloatArray(4800 * cfg.codec.channels)

        fun alive(j: Job) = !j.cancel.isCancelled && j.error == null

        fun emit(j: Job, buf: FloatArray, frames: Int) {
            if (frames <= 0 || !alive(j)) return
            if (firstEmitNs == 0L) {
                firstEmitNs = System.nanoTime()
                j.stats.timeToFirstAudioMs = (firstEmitNs - j.startNs) / 1_000_000
            }
            emittedFrames += frames
            j.stats.audioFrames += frames
            if (!j.sink.onAudio(buf, frames)) j.cancel.cancel()
        }

        fun decodeFrames(j: Job, count: Int) {
            var remaining = count
            while (remaining > 0 && alive(j)) {
                val n = minOf(remaining, 16)
                val t0 = System.nanoTime()
                val produced = codec.decode(pending, 0, n)
                j.stats.codecMs += (System.nanoTime() - t0) / 1_000_000
                j.stats.codecCalls++
                repeat(n) { pending.removeAt(0) }
                remaining -= n
                emit(j, codec.lastOutput, produced)
            }
            if (!alive(j)) pending.clear()
        }

        /** Official adaptive policy (ort_cpu_runtime._resolve_stream_decode_frame_budget). */
        fun budget(): Int {
            if (firstEmitNs == 0L) return 1
            val elapsed = (System.nanoTime() - firstEmitNs) / 1e9
            val lead = emittedFrames.toDouble() / sampleRate - elapsed
            return when {
                lead < 0.20 -> 1
                lead < 0.55 -> 2
                lead < 1.10 -> 4
                else -> 8
            }
        }

        while (true) {
            val msg = try {
                queue.take()
            } catch (_: InterruptedException) {
                return
            }
            try {
                when (msg) {
                    is Msg.Begin -> {
                        job = msg.job
                        pending.clear()
                        codec.reset()
                        emittedFrames = 0
                        firstEmitNs = 0
                        msg.job.sink.onStart(sampleRate, cfg.codec.channels)
                    }
                    is Msg.Frame -> {
                        val j = job ?: continue
                        if (!alive(j)) continue
                        pending.add(msg.tokens)
                        // Pull any frames that are already waiting, then decode in budget-sized batches.
                        while (true) {
                            val head = queue.peek()
                            if (head is Msg.Frame) {
                                queue.poll()
                                pending.add(head.tokens)
                            } else {
                                break
                            }
                        }
                        var b = budget()
                        while (pending.size >= b && alive(j)) {
                            decodeFrames(j, b)
                            b = budget()
                        }
                    }
                    is Msg.ChunkEnd -> {
                        val j = job ?: continue
                        if (alive(j)) {
                            decodeFrames(j, pending.size)
                            codec.reset()
                            var left = msg.pauseFrames
                            while (left > 0 && alive(j)) {
                                val n = minOf(left, 4800)
                                emit(j, silence, n)
                                left -= n
                            }
                        }
                        pending.clear()
                    }
                    is Msg.Finish -> {
                        val j = job
                        if (j != null) {
                            if (alive(j)) decodeFrames(j, pending.size)
                            pending.clear()
                            codec.reset()
                            runCatching { j.sink.onFinish() }
                            job = null
                            j.done.countDown()
                        }
                    }
                    is Msg.Shutdown -> return
                }
            } catch (t: Throwable) {
                job?.let {
                    it.error = t
                    it.cancel.cancel()
                }
                pending.clear()
                runCatching { codec.reset() }
            }
        }
    }

    // =================================================================================
    // Voice cloning (codec encoder, loaded only on demand)
    // =================================================================================

    /**
     * Encodes a reference clip into prompt codes.
     * @param channelMajor samples laid out [channel][sample] at [sampleRate] Hz, [channels] channels.
     */
    fun encodeReferenceAudio(channelMajor: FloatArray, samples: Int): Array<IntArray> {
        val encodeFile = cfg.codec.encodeFile ?: throw MissingModelException("codec meta has no encode graph")
        synthLock.withLock {
            val opts = codecSessionOptions(options.lmThreads)
            opts.use {
                env.openSession(File(cfg.codecDir, encodeFile), opts).use { enc ->
                    val ch = cfg.codec.channels
                    val wave = directFloats(ch * samples)
                    wave.put(channelMajor, 0, ch * samples)
                    wave.rewind()
                    val len = directInts(1).apply { put(0, samples) }
                    OnnxTensor.createTensor(env, wave, longArrayOf(1, ch.toLong(), samples.toLong())).use { w ->
                        OnnxTensor.createTensor(env, len, longArrayOf(1)).use { l ->
                            enc.run(mapOf("waveform" to w, "input_lengths" to l)).use { r ->
                                val codes = r.tensor("audio_codes")
                                val frames = r.tensor("audio_code_lengths").readInts()[0]
                                val flat = codes.readInts()
                                val nq = cfg.codec.numQuantizers
                                val shapeFrames = codes.info.shape[1].toInt()
                                return Array(frames.coerceAtMost(shapeFrames)) { f -> IntArray(nq) { q -> flat[f * nq + q] } }
                            }
                        }
                    }
                }
            }
        }
    }

    override fun close() {
        if (closed) return
        closed = true
        synthLock.withLock {
            queue.clear()
            queue.offer(Msg.Shutdown)
            worker.join(2000)
            codec.close()
            listOf(hiddenLocalIn, hiddenDecodeOut, seenMaskTensor, assistantUTensor, audioUTensor, rowTensor, pastLenTensor)
                .forEach { runCatching { it.close() } }
            listOf(prefill, decode, local, codecSession).forEach { runCatching { it.close() } }
            lmOptions.close()
            codecOptions.close()
            tokenizer.close()
        }
    }

    companion object {
        /** Hook so the Android layer can raise the codec thread to audio priority. */
        @JvmStatic @Volatile var threadPriorityHook: (() -> Unit)? = null

        fun isModelPresent(root: File): Boolean = ModelConfig.findManifest(root) != null

        /** Parses only the manifest (no ONNX sessions) to list the built-in voices. */
        fun readBuiltinVoices(root: File): List<VoicePrompt> = ModelConfig.load(root).builtinVoices

        fun load(options: EngineOptions, tokenizerFactory: (File) -> TextTokenizer, log: (String) -> Unit = {}): MossTtsEngine {
            val cfg = ModelConfig.load(options.modelRoot)
            val tokenizer = tokenizerFactory(cfg.tokenizerFile)
            return try {
                MossTtsEngine(options, cfg, tokenizer, log)
            } catch (t: Throwable) {
                tokenizer.close()
                throw t
            }
        }
    }
}
