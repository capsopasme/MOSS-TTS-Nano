package io.github.capsopasme.mossnano.engine

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.tan

/**
 * Integrated loudness (ITU-R BS.1770-4 / EBU R128, LUFS): K-weighting, 400 ms blocks with 75 %
 * overlap, absolute gate at -70 LUFS and relative gate at -10 LU. Filter coefficients for any
 * sample rate as in libebur128. Matches ffmpeg's `ebur128` filter to ~0.1 LU.
 */
class LoudnessMeter(private val sampleRate: Int, private val channels: Int) {
    // K-weighting = high-shelf pre-filter + RLB high-pass, one biquad state pair per channel
    private val pb = DoubleArray(3)
    private val pa = DoubleArray(3)
    private val rb = doubleArrayOf(1.0, -2.0, 1.0)
    private val ra = DoubleArray(3)
    private val s1 = Array(channels) { DoubleArray(2) }
    private val s2 = Array(channels) { DoubleArray(2) }

    private val hop = sampleRate / 10            // 100 ms sub-blocks
    private var hopFill = 0
    private var hopEnergy = 0.0
    private val subBlocks = ArrayList<Double>()  // mean square per 100 ms sub-block
    var peak = 0f
        private set

    init {
        var f0 = 1681.974450955533
        val g = 3.999843853973347
        var q = 0.7071752369554196
        var k = tan(PI * f0 / sampleRate)
        val vh = 10.0.pow(g / 20.0)
        val vb = vh.pow(0.4996667741545416)
        val a0 = 1.0 + k / q + k * k
        pb[0] = (vh + vb * k / q + k * k) / a0
        pb[1] = 2.0 * (k * k - vh) / a0
        pb[2] = (vh - vb * k / q + k * k) / a0
        pa[0] = 1.0
        pa[1] = 2.0 * (k * k - 1.0) / a0
        pa[2] = (1.0 - k / q + k * k) / a0
        f0 = 38.13547087602444
        q = 0.5003270373238773
        k = tan(PI * f0 / sampleRate)
        ra[0] = 1.0
        ra[1] = 2.0 * (k * k - 1.0) / (1.0 + k / q + k * k)
        ra[2] = (1.0 - k / q + k * k) / (1.0 + k / q + k * k)
    }

    /** [samples]: interleaved, [frames] frames. */
    fun add(samples: FloatArray, frames: Int, offsetFrames: Int = 0) {
        for (i in offsetFrames until offsetFrames + frames) {
            var e = 0.0
            for (c in 0 until channels) {
                val x = samples[i * channels + c]
                val ax = abs(x)
                if (ax > peak) peak = ax
                // direct form II transposed, two stages
                val st1 = s1[c]
                val y1 = pb[0] * x + st1[0]
                st1[0] = pb[1] * x - pa[1] * y1 + st1[1]
                st1[1] = pb[2] * x - pa[2] * y1
                val st2 = s2[c]
                val y2 = rb[0] * y1 + st2[0]
                st2[0] = rb[1] * y1 - ra[1] * y2 + st2[1]
                st2[1] = rb[2] * y1 - ra[2] * y2
                e += y2 * y2
            }
            hopEnergy += e
            if (++hopFill == hop) {
                subBlocks += hopEnergy / hop
                hopFill = 0
                hopEnergy = 0.0
            }
        }
    }

    /** Integrated loudness in LUFS, or [Double.NEGATIVE_INFINITY] for silence / < 400 ms of audio. */
    fun integratedLufs(): Double {
        if (subBlocks.size < 4) return Double.NEGATIVE_INFINITY
        val blocks = DoubleArray(subBlocks.size - 3) { i ->
            (subBlocks[i] + subBlocks[i + 1] + subBlocks[i + 2] + subBlocks[i + 3]) / 4
        }
        val absGate = energyOf(-70.0)
        var sum = 0.0
        var n = 0
        for (b in blocks) if (b > absGate) { sum += b; n++ }
        if (n == 0) return Double.NEGATIVE_INFINITY
        val relGate = sum / n * 10.0.pow(-1.0) // -10 LU
        sum = 0.0
        n = 0
        for (b in blocks) if (b > absGate && b > relGate) { sum += b; n++ }
        return if (n == 0) Double.NEGATIVE_INFINITY else lufsOf(sum / n)
    }

    companion object {
        fun lufsOf(meanSquare: Double): Double = -0.691 + 10.0 * log10(meanSquare)
        fun energyOf(lufs: Double): Double = 10.0.pow((lufs + 0.691) / 10.0)

        /** Integrated loudness of a whole interleaved buffer. */
        fun measure(samples: FloatArray, frames: Int, sampleRate: Int, channels: Int): Double =
            LoudnessMeter(sampleRate, channels).apply { add(samples, frames) }.integratedLufs()
    }
}

/**
 * Applies a fixed gain to the streamed audio plus a zero-latency peak limiter, so boosting a
 * quiet voice never clips (and neither do the codec's own overshoots past full scale, which
 * otherwise get hard-clipped by the PCM16 conversion): the limiter's envelope follows peaks
 * instantly, so the output never exceeds [ceiling], and decays over ~[releaseMs], which keeps
 * the gain changes smooth. Stereo channels share one envelope (no image shift). Below the
 * ceiling the signal is only multiplied by the gain (exactly unchanged at 0 dB).
 */
class OutputGain(gainDb: Float, sampleRate: Int, private val channels: Int, private val ceiling: Float = CEILING, releaseMs: Float = 80f) {
    private val gain = 10f.pow(gainDb / 20f)
    private val release = exp(-1.0 / (releaseMs / 1000.0 * sampleRate))
    private val releaseF = release.toFloat()
    private var env = 0f
    /** Frames on which the limiter reduced gain (for stats). */
    var limitedFrames = 0L
        private set

    fun process(buf: FloatArray, frames: Int) {
        val g = gain
        val ceil = ceiling
        for (i in 0 until frames) {
            val base = i * channels
            var p = 0f
            for (c in 0 until channels) {
                val a = abs(buf[base + c] * g)
                if (a > p) p = a
            }
            // max(peak, decayed envelope): never below the current peak -> output <= ceiling
            val decayed = env * releaseF
            env = if (p > decayed) p else decayed
            val k = if (env > ceil) { limitedFrames++; g * ceil / env } else g
            for (c in 0 until channels) buf[base + c] *= k
        }
    }

    /** [frames] of silence passed by (pauses between chunks): let the envelope decay over them. */
    fun skip(frames: Int) {
        env = (env * release.pow(frames.toDouble())).toFloat()
    }

    companion object {
        /** -0.5 dBFS */
        const val CEILING = 0.944f
    }
}
