// Desktop-JVM test of the engine against mock graphs (see make_mock_models.py).
// Run: bash tools/mock/run_mock_test.sh
import io.github.capsopasme.mossnano.engine.AudioSink
import io.github.capsopasme.mossnano.engine.CancelSignal
import io.github.capsopasme.mossnano.engine.CpuAffinity
import io.github.capsopasme.mossnano.engine.TtsException
import io.github.capsopasme.mossnano.engine.EngineOptions
import io.github.capsopasme.mossnano.engine.LoudnessMeter
import io.github.capsopasme.mossnano.engine.OutputGain
import io.github.capsopasme.mossnano.engine.VoiceLoudness
import io.github.capsopasme.mossnano.engine.MossTtsEngine
import io.github.capsopasme.mossnano.engine.SpmTokenizer
import io.github.capsopasme.mossnano.engine.SynthRequest
import io.github.capsopasme.mossnano.engine.TextChunker
import io.github.capsopasme.mossnano.engine.TextNormalizer
import java.io.File
import kotlin.math.abs
import kotlin.system.exitProcess

private var failures = 0
private fun check(cond: Boolean, msg: String) {
    if (cond) println("  ok   $msg") else { println("  FAIL $msg"); failures++ }
}

fun main(args: Array<String>) {
    val modelRoot = File(args[0])
    SpmTokenizer.libraryPathOverride = args[1]
    val framesPerChunk = 20
    val hop = 3840

    println("== tokenizer")
    SpmTokenizer(File(modelRoot, "MOSS-TTS-Nano-100M-ONNX/tokenizer.model")).use { tok ->
        val ids = tok.encode("今天天气很好，hello world。").toList()
        val expected = if (args.size > 2) args[2].trim().split(Regex("\\s+")).map { it.toInt() } else listOf(112, 248, 356, 33, 3, 102, 81, 4)
        check(ids == expected, "JNI ids match spm_encode: $ids")
        check(tok.encode("𝄞 emoji 😀 ok").isNotEmpty(), "supplementary chars survive UTF-16 -> UTF-8")
    }

    println("== normalizer")
    val cases = listOf(
        "2024年5月1日下午3:05，气温-2℃，涨幅12.5%，花了¥1,234.5元。",
        "我有2个苹果，3-5天到货，电话13812345678，第10010号。",
        "# 标题\n- 列表项一\n- 列表项二\n访问 https://example.com/a_b?x=1 看看 app.js.map 吧",
        "I paid \$12.50 on May 3rd at 10:30, about 45% of 1,000,000 in 2024",
        "真的吗？？？！！！太好了……",
        "请求接入-身份判定，GPU-A100 和 v2.3.1 版本",
    )
    for (c in cases) println("  ${c.replace("\n", "\\n")}\n    -> ${TextNormalizer.normalize(c)}")
    fun tn(s: String) = TextNormalizer.normalize(s)
    check(tn("12:30") == "十二点三十分。", "digits-only text is read in Chinese like the official pipeline: ${tn("12:30")}")
    check(tn("pages 10-20 and room A-3").contains("ten to twenty") && !tn("pages 10-20 and room A-3").contains("minus"),
        "English ranges / word-number hyphens are not read as minus: ${tn("pages 10-20 and room A-3")}")
    check(tn("It costs \$1.05, at -3 degrees").contains("one dollar five cents") && tn("It costs \$1.05, at -3 degrees").contains("minus three"),
        "currency singular + real minus: ${tn("It costs \$1.05, at -3 degrees")}")
    check(tn("from 1990-2000, on 2024-05-01").contains("nineteen ninety to two thousand") && tn("from 1990-2000, on 2024-05-01").contains("May first, twenty twenty-four"),
        "year ranges and ISO dates: ${tn("from 1990-2000, on 2024-05-01")}")

    println("== loudness / output gain")
    run {
        val sr = 48000
        val a = Math.pow(10.0, -23.0 / 20).toFloat()
        val sine = FloatArray(sr * 5 * 2) { i -> (a * kotlin.math.sin(2 * Math.PI * 1000.0 * (i / 2) / sr)).toFloat() }
        val l = LoudnessMeter.measure(sine, sine.size / 2, sr, 2)
        check(abs(l + 23.0) < 0.1, "EBU Tech 3341 case 1: stereo 1 kHz sine at -23 dBFS -> ${"%.2f".format(l)} LUFS")
        val quietThenLoud = FloatArray(sr * 10 * 2) { i -> val t = i / 2; (if (t < sr * 5) a / 100 else a) * kotlin.math.sin(2 * Math.PI * 1000.0 * t / sr).toFloat() }
        val gated = LoudnessMeter.measure(quietThenLoud, quietThenLoud.size / 2, sr, 2)
        check(abs(gated + 23.0) < 0.2, "relative gate ignores the -40 dB half: ${"%.2f".format(gated)} LUFS")
        check(LoudnessMeter.measure(FloatArray(sr * 2 * 2), sr * 2, sr, 2) == Double.NEGATIVE_INFINITY, "silence -> -inf")

        val loud = FloatArray(sr * 2) { i -> (0.9 * kotlin.math.sin(2 * Math.PI * 220.0 * (i / 2) / sr)).toFloat() }
        val g = OutputGain(6f, sr, 2)
        g.process(loud, sr)
        val peak = loud.maxOf { abs(it) }
        check(peak <= OutputGain.CEILING + 1e-6f && peak > 0.9f, "limiter: +6 dB on a 0.9 sine stays under the ceiling (peak ${"%.4f".format(peak)})")
        val soft = FloatArray(4800) { i -> (0.3 * kotlin.math.sin(i * 0.05)).toFloat() }
        val ref = soft.copyOf()
        OutputGain(0f, sr, 2).process(soft, soft.size / 2)
        check(soft.contentEquals(ref), "0 dB below the ceiling is an exact pass-through")
        val v = io.github.capsopasme.mossnano.engine.VoicePrompt("x", "x", "", arrayOf(IntArray(16)), builtin = false, loudnessLufs = -21.7)
        check(VoiceLoudness.autoGainDb(v) == 5.5f, "auto gain of a -21.7 LUFS voice: ${VoiceLoudness.autoGainDb(v)} dB")
        check(VoiceLoudness.autoGainDb(io.github.capsopasme.mossnano.engine.VoicePrompt("y", "y", "", arrayOf(IntArray(16)), builtin = false, loudnessLufs = -14.0)) == 0f, "loud voices are not turned down")
        check(VoiceLoudness.autoGainDb(io.github.capsopasme.mossnano.engine.VoicePrompt("z", "z", "", arrayOf(IntArray(16)), builtin = false)) == 0f, "unknown loudness -> no gain")
    }

    println("== cpu affinity")
    val fake = File(System.getProperty("java.io.tmpdir"), "fake_cpu_${System.nanoTime()}").apply { mkdirs() }
    File(fake, "possible").writeText("0-7\n")
    val caps = listOf(325, 325, 870, 870, 870, 870, 870, 1024) // 2 little + 5 mid + 1 prime
    caps.forEachIndexed { i, c -> File(fake, "cpu$i").mkdirs(); File(fake, "cpu$i/cpu_capacity").writeText("$c\n") }
    check(CpuAffinity.computePerformanceMask(fake) == 0b11111100L, "little cluster excluded: ${java.lang.Long.toBinaryString(CpuAffinity.computePerformanceMask(fake))}")
    caps.indices.forEach { File(fake, "cpu$it/cpu_capacity").writeText("1024\n") }
    check(CpuAffinity.computePerformanceMask(fake) == 0L, "homogeneous SoC -> no pinning")
    fake.deleteRecursively()
    val before = CpuAffinity.currentThreadMask()
    check(before != 0L, "sched_getaffinity through JNI: ${java.lang.Long.toBinaryString(before)}")
    check(CpuAffinity.setCurrentThreadMask(before) && CpuAffinity.currentThreadMask() == before, "sched_setaffinity round trip")

    println("== chunker")
    val long = "今天天气很好。我们一起去公园散步吧！这是一个流式语音合成的测试，模型运行在手机上，速度很快，效果自然。"
    val chunks = TextChunker.split(long, 12) { it.length }
    println("  $chunks")
    check(chunks.size > 2 && chunks.all { it.length <= 12 + 2 }, "split respects budget")

    println("== engine (mock graphs)")
    val engine = MossTtsEngine.load(
        EngineOptions(modelRoot, lmThreads = 2, codecThreads = 1, maxTextTokensPerChunk = 12, maxQueuedFrames = 16),
        tokenizerFactory = { SpmTokenizer(it) },
        log = { println("  [log] $it") },
    )
    engine.warmup()
    val voice = engine.builtinVoices.first()

    // ---- full run
    val ch0 = ArrayList<Float>(2_000_000)
    var started = 0
    val sink = object : AudioSink {
        override fun onStart(sampleRate: Int, channels: Int) { started++; check(sampleRate == 48000 && channels == 2, "onStart 48k stereo") }
        override fun onAudio(samples: FloatArray, frames: Int): Boolean {
            for (i in 0 until frames) {
                ch0 += samples[i * 2]
                if (samples[i * 2] != samples[i * 2 + 1]) { failures++; println("  FAIL channel interleave"); return false }
            }
            return true
        }
    }
    val text = "今天天气很好。我们一起去公园散步吧！这是一个流式语音合成的测试，模型运行在手机上。"
    val expectedChunks = engine.prepareChunks(text, normalize = false)
    println("  chunks: $expectedChunks")
    val stats = engine.synthesize(SynthRequest(text, voice, normalizeText = false), sink)
    println("  " + stats.summary().replace("\n", "\n  "))
    check(stats.frames == expectedChunks.size * framesPerChunk, "every chunk ran $framesPerChunk frames (KV length / past_valid_lengths / seen-mask consistent): ${stats.frames}")

    // Verify codec state continuity + per-chunk reset + pauses.
    var pos = 0
    var ok = true
    for ((ci, chunk) in expectedChunks.withIndex()) {
        for (k in 0 until framesPerChunk * hop) {
            if (abs(ch0[pos] - k * 1e-6f) > 1e-5f) { ok = false; println("  mismatch chunk $ci sample $k: ${ch0[pos]}"); break }
            pos++
        }
        if (!ok) break
        if (ci < expectedChunks.lastIndex) {
            val pause = (TextChunker.pauseSeconds(chunk) * 48000).toInt()
            for (k in 0 until pause) {
                if (ch0[pos] != 0f) { ok = false; println("  non-silent pause in chunk $ci at $k"); break }
                pos++
            }
            if (!ok) break
        }
    }
    check(ok && pos == ch0.size, "stream state carried across batched codec calls, reset per chunk, pauses exact ($pos/${ch0.size} samples)")
    check(stats.codecCalls < stats.frames, "adaptive batching used fewer codec calls (${stats.codecCalls}) than frames (${stats.frames})")
    check(stats.timeToFirstAudioMs >= 0, "TTFA measured: ${stats.timeToFirstAudioMs} ms")

    // ---- output gain is applied in the stream (mock audio is far below the limiter ceiling)
    val gained = ArrayList<Float>(200_000)
    val gStats = engine.synthesize(SynthRequest("今天天气很好。", voice, normalizeText = false, gainDb = 6.0206f), object : AudioSink {
        override fun onAudio(samples: FloatArray, frames: Int): Boolean { for (i in 0 until frames) gained += samples[i * 2]; return true }
    })
    var gainOk = gained.size == framesPerChunk * hop
    for (k in 0 until minOf(gained.size, framesPerChunk * hop)) if (abs(gained[k] - 2 * k * 1e-6f) > 2e-5f) { gainOk = false; println("  gain mismatch at $k: ${gained[k]}"); break }
    check(gainOk && gStats.limitedFrames == 0L, "gainDb +6 dB doubles the streamed samples, limiter idle (${gained.size} samples)")

    // ---- cancellation from the sink
    val cancel = CancelSignal()
    var got = 0
    val t0 = System.nanoTime()
    engine.synthesize(SynthRequest(text, voice, normalizeText = false), object : AudioSink {
        override fun onAudio(samples: FloatArray, frames: Int): Boolean {
            got += frames
            if (got > 48000) cancel.cancel()
            return true
        }
    }, cancel)
    val cancelMs = (System.nanoTime() - t0) / 1_000_000
    check(got < ch0.size, "cancel stops early ($got of ${ch0.size} samples, ${cancelMs}ms)")

    // ---- sink abort (return false)
    var calls = 0
    engine.synthesize(SynthRequest(text, voice, normalizeText = false), object : AudioSink {
        override fun onAudio(samples: FloatArray, frames: Int): Boolean { calls++; return false }
    })
    check(calls == 1, "sink returning false aborts after first buffer ($calls)")

    // ---- engine still healthy afterwards
    ch0.clear()
    val again = engine.synthesize(SynthRequest(text, voice, normalizeText = false), sink)
    check(again.frames == expectedChunks.size * framesPerChunk, "engine reusable after cancel/abort")

    // ---- concurrent callers are serialized
    val threads = (0 until 3).map {
        Thread {
            engine.synthesize(SynthRequest("你好。世界。", voice, normalizeText = true), object : AudioSink {
                override fun onAudio(samples: FloatArray, frames: Int) = true
            })
        }.apply { start() }
    }
    threads.forEach { it.join(20_000) }
    check(threads.none { it.isAlive }, "concurrent synthesize() calls complete")

    // ---- a failing sink surfaces as an error instead of hanging the caller
    val failing = runCatching {
        engine.synthesize(SynthRequest("你好。世界。", voice, normalizeText = false), object : AudioSink {
            override fun onAudio(samples: FloatArray, frames: Int) = true
            override fun onFinish() = throw IllegalStateException("disk full")
        })
    }
    check(failing.exceptionOrNull() is TtsException, "sink.onFinish failure -> TtsException (${failing.exceptionOrNull()?.message})")
    val afterFail = engine.synthesize(SynthRequest("你好。", voice, normalizeText = false), object : AudioSink {
        override fun onAudio(samples: FloatArray, frames: Int) = true
    })
    check(afterFail.frames == framesPerChunk, "engine healthy after sink failure")

    // ---- close() while a (slow) synthesis is in flight returns quickly instead of waiting for it
    var finished = false
    val slow = Thread {
        runCatching {
            engine.synthesize(SynthRequest(text, voice, normalizeText = false), object : AudioSink {
                override fun onAudio(samples: FloatArray, frames: Int): Boolean { Thread.sleep(50); return true }
            })
        }
        finished = true
    }.apply { start() }
    Thread.sleep(300)
    val tc = System.nanoTime()
    engine.close()
    val closeMs = (System.nanoTime() - tc) / 1_000_000
    slow.join(5_000)
    check(closeMs < 1500 && finished && !slow.isAlive, "close() during synthesis cancels it (${closeMs}ms)")
    println(if (failures == 0) "ALL PASSED" else "$failures FAILURE(S)")
    exitProcess(if (failures == 0) 0 else 1)
}
