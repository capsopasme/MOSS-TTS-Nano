// Runs the real engine on a desktop JVM against a real model directory (FP32 or the INT8 pack)
// and writes the streamed audio to a WAV file, as an end-to-end check of the engine with real
// graphs (also used in CI on an ARM64 runner to measure the real RTF of each model pack).
//
//   bash tools/mock/run_real_model.sh <model_root> <out.wav> [text] [voice_id]
import io.github.capsopasme.mossnano.engine.AudioSink
import io.github.capsopasme.mossnano.engine.EngineOptions
import io.github.capsopasme.mossnano.engine.MossTtsEngine
import io.github.capsopasme.mossnano.engine.SpmTokenizer
import io.github.capsopasme.mossnano.engine.SynthRequest
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.system.exitProcess

fun main(args: Array<String>) {
    val root = File(args[0])
    val out = File(args[1])
    SpmTokenizer.libraryPathOverride = args[2]
    val text = args.getOrNull(3) ?: "你好，这是端侧流式语音合成的量化验证。今天天气不错，我们出去走走吧。"
    val voiceId = args.getOrNull(4)
    val threads = (System.getenv("LM_THREADS") ?: "2").toInt()

    val engine = MossTtsEngine.load(
        EngineOptions(root, lmThreads = threads, codecThreads = 1),
        tokenizerFactory = { SpmTokenizer(it) },
        log = { println("[log] $it") },
    )
    val voice = engine.builtinVoices.firstOrNull { it.id == voiceId } ?: engine.builtinVoices.first()
    println("voice: $voice")
    val t0 = System.nanoTime()
    engine.warmup()
    println("warmup ${(System.nanoTime() - t0) / 1_000_000} ms")
    println("chunks: ${engine.prepareChunks(text, true)}")

    val samples = ArrayList<FloatArray>()
    var channels = 2
    var rate = 48000
    var nonFinite = 0
    val sink = object : AudioSink {
        override fun onStart(sampleRate: Int, channels0: Int) { rate = sampleRate; channels = channels0 }
        override fun onAudio(samples0: FloatArray, frames: Int): Boolean {
            val copy = samples0.copyOf(frames * channels)
            for (v in copy) if (!v.isFinite()) nonFinite++
            samples += copy
            return true
        }
    }
    val stats = engine.synthesize(SynthRequest(text, voice, seed = 1234L, normalizeText = true), sink)
    println(stats.summary())
    engine.close()

    val all = FloatArray(samples.sumOf { it.size })
    var p = 0
    for (s in samples) { System.arraycopy(s, 0, all, p, s.size); p += s.size }
    var peak = 0f
    var sq = 0.0
    for (v in all) { peak = maxOf(peak, abs(v)); sq += v * v }
    val rms = sqrt(sq / maxOf(1, all.size))
    println("audio: ${all.size / channels} frames, peak=$peak rms=${"%.4f".format(rms)} nonFinite=$nonFinite")

    RandomAccessFile(out, "rw").use { f ->
        f.setLength(0)
        val data = ByteBuffer.allocate(all.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (v in all) data.putShort((v.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
        val h = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        h.put("RIFF".toByteArray()).putInt(36 + all.size * 2).put("WAVE".toByteArray())
        h.put("fmt ".toByteArray()).putInt(16).putShort(1).putShort(channels.toShort())
        h.putInt(rate).putInt(rate * channels * 2).putShort((channels * 2).toShort()).putShort(16)
        h.put("data".toByteArray()).putInt(all.size * 2)
        f.write(h.array())
        f.write(data.array())
    }
    val ok = nonFinite == 0 && stats.frames > 0 && rms > 1e-3 && peak <= 1.5f
    println(if (ok) "REAL MODEL RUN OK" else "REAL MODEL RUN SUSPICIOUS")
    exitProcess(if (ok) 0 else 1)
}
