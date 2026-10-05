package io.github.capsopasme.mossnano

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import io.github.capsopasme.mossnano.engine.LoudnessMeter
import io.github.capsopasme.mossnano.engine.OutputGain
import io.github.capsopasme.mossnano.engine.Resampler
import io.github.capsopasme.mossnano.engine.VoiceLoudness
import io.github.capsopasme.mossnano.engine.VoicePrompt
import java.nio.ByteOrder

/**
 * Turns a user audio clip into a reusable voice: decode (any format MediaCodec supports)
 * -> trim -> resample to 48 kHz -> loudness-normalize -> stereo -> MOSS-Audio-Tokenizer
 * encoder -> prompt codes.
 *
 * The clip is capped at 10 s: built-in voices use ~100 frames (8 s); longer prompts make
 * every prefill slower without improving similarity much.
 */
object VoiceCloner {
    private const val MAX_SECONDS = 10.0
    private const val TARGET_RATE = 48000

    fun clone(context: Context, uri: Uri, name: String): VoicePrompt {
        val (mono, rate) = decodeToMono(context, uri)
        require(mono.isNotEmpty()) { "无法解码音频" }
        val trimmed = trimSilence(mono, rate)
        val maxSamples = (MAX_SECONDS * rate).toInt()
        val clip = if (trimmed.size > maxSamples) trimmed.copyOf(maxSamples) else trimmed
        require(clip.size > rate) { "有效音频太短（至少 1 秒，建议 5~10 秒清晰人声）" }
        val at48 = Resampler.resample(clip, rate, TARGET_RATE)
        // The model copies the reference's level: bring every clip to the same loudness (peaks
        // limited) so clones come out about as loud as the louder built-in voices.
        val refLufs = normalizeLoudness(at48)
        // channel-major stereo [L..., R...]
        val stereo = FloatArray(at48.size * 2)
        System.arraycopy(at48, 0, stereo, 0, at48.size)
        System.arraycopy(at48, 0, stereo, at48.size, at48.size)
        val codes = EngineManager.withEngine { it.encodeReferenceAudio(stereo, at48.size) }
        require(codes.isNotEmpty()) { "编码失败" }
        return VoiceStore.save(context, name, codes, VoiceLoudness.estimateFromReference(refLufs))
    }

    private fun decodeToMono(context: Context, uri: Uri): Pair<FloatArray, Int> {
        val extractor = MediaExtractor()
        extractor.setDataSource(context, uri, null)
        var track = -1
        for (i in 0 until extractor.trackCount) {
            if (extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true) {
                track = i
                break
            }
        }
        require(track >= 0) { "文件里没有音轨" }
        extractor.selectTrack(track)
        val format = extractor.getTrackFormat(track)
        val mime = format.getString(MediaFormat.KEY_MIME)!!
        var rate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        var channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()
        val out = FloatArrayBuilder()
        val info = MediaCodec.BufferInfo()
        var inputDone = false
        var outputDone = false
        var pcmFloat = false
        val limit = (MAX_SECONDS + 20) * rate // decode a bit more than needed (silence trimming)
        try {
            while (!outputDone) {
                if (!inputDone) {
                    val inIdx = codec.dequeueInputBuffer(10_000)
                    if (inIdx >= 0) {
                        val buf = codec.getInputBuffer(inIdx)!!
                        val n = extractor.readSampleData(buf, 0)
                        if (n < 0) {
                            codec.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(inIdx, 0, n, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }
                val outIdx = codec.dequeueOutputBuffer(info, 10_000)
                if (outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    val f = codec.outputFormat
                    rate = f.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                    channels = f.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                    pcmFloat = f.containsKey(MediaFormat.KEY_PCM_ENCODING) &&
                        f.getInteger(MediaFormat.KEY_PCM_ENCODING) == android.media.AudioFormat.ENCODING_PCM_FLOAT
                } else if (outIdx >= 0) {
                    val buf = codec.getOutputBuffer(outIdx)!!.order(ByteOrder.nativeOrder())
                    buf.position(info.offset)
                    buf.limit(info.offset + info.size)
                    if (pcmFloat) {
                        val fb = buf.asFloatBuffer()
                        val frames = fb.remaining() / channels
                        for (i in 0 until frames) {
                            var s = 0f
                            for (c in 0 until channels) s += fb.get(i * channels + c)
                            out.add(s / channels)
                        }
                    } else {
                        val sb = buf.asShortBuffer()
                        val frames = sb.remaining() / channels
                        for (i in 0 until frames) {
                            var s = 0f
                            for (c in 0 until channels) s += sb.get(i * channels + c) / 32768f
                            out.add(s / channels)
                        }
                    }
                    codec.releaseOutputBuffer(outIdx, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0 || out.size > limit) outputDone = true
                }
            }
        } finally {
            codec.stop()
            codec.release()
            extractor.release()
        }
        return out.toArray() to rate
    }

    private fun trimSilence(x: FloatArray, rate: Int): FloatArray {
        val win = rate / 50 // 20 ms
        if (x.size < win * 4) return x
        var peak = 0f
        for (v in x) peak = maxOf(peak, kotlin.math.abs(v))
        val thr = peak * 0.02f
        fun rms(start: Int): Float {
            var s = 0f
            val end = minOf(x.size, start + win)
            for (i in start until end) s += x[i] * x[i]
            return kotlin.math.sqrt(s / (end - start))
        }
        var a = 0
        while (a + win < x.size && rms(a) < thr) a += win
        var b = x.size - win
        while (b > a && rms(b) < thr) b -= win
        a = maxOf(0, a - win * 5)
        b = minOf(x.size, b + win * 10)
        return x.copyOfRange(a, b)
    }

    /**
     * Gain to [VoiceLoudness.REFERENCE_TARGET_LUFS] (measured as the dual-mono stereo clip the
     * encoder sees) through the output limiter, so a dynamic clip is not clipped on the way up.
     * @return the clip's loudness afterwards (LUFS), NaN if it is too short / silent to measure.
     */
    private fun normalizeLoudness(x: FloatArray): Double {
        val dualMono = 10 * kotlin.math.log10(2.0)
        val before = LoudnessMeter.measure(x, x.size, TARGET_RATE, 1) + dualMono
        if (!before.isFinite()) return Double.NaN
        val gain = (VoiceLoudness.REFERENCE_TARGET_LUFS - before).coerceIn(-20.0, 24.0)
        OutputGain(gain.toFloat(), TARGET_RATE, 1).process(x, x.size)
        return LoudnessMeter.measure(x, x.size, TARGET_RATE, 1) + dualMono
    }

    private class FloatArrayBuilder {
        private var data = FloatArray(48000 * 4)
        var size = 0
            private set

        fun add(v: Float) {
            if (size == data.size) data = data.copyOf(data.size * 2)
            data[size++] = v
        }

        fun toArray(): FloatArray = data.copyOf(size)
    }
}
