package io.github.capsopasme.mossnano.engine

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.roundToInt

/**
 * Streaming WSOLA time-stretch for mono audio (speed change, pitch preserved).
 * Used by the system-TTS path, where clients expect raw PCM at the requested speech rate.
 */
class TimeStretch(sampleRate: Int, private val speed: Float) {
    private val n = (sampleRate * 0.030).toInt() and 1.inv()     // frame length (30 ms, even)
    private val hs = n / 2                                         // synthesis hop
    private val ha = hs * speed.toDouble()                         // analysis hop
    private val tol = (sampleRate * 0.008).toInt()                 // ±8 ms search
    private val window = FloatArray(n) { (0.5 - 0.5 * cos(2 * PI * it / n)).toFloat() }

    private var input = FloatArray(sampleRate)
    private var inLen = 0          // valid samples in [input]
    private var base = 0L          // absolute index of input[0]
    private var ana = 0.0          // absolute analysis position
    private var prevStart = -1L    // absolute start of the previous frame
    private val overlap = FloatArray(hs)
    private var out = FloatArray(hs * 8)

    val bypass: Boolean get() = speed in 0.99f..1.01f

    /** Feeds [count] samples; returns produced sample count, output in [output]. */
    fun process(samples: FloatArray, offset: Int, count: Int): Int {
        if (bypass) {
            ensureOut(count)
            System.arraycopy(samples, offset, out, 0, count)
            return count
        }
        append(samples, offset, count)
        var produced = 0
        while (true) {
            val anaStart = ana.roundToInt().toLong()
            val needEnd = maxOf(anaStart + tol + n, if (prevStart < 0) n.toLong() else prevStart + hs + n)
            if (needEnd > base + inLen) break
            val s = if (prevStart < 0) 0L else bestStart(anaStart)
            ensureOut(produced + hs)
            val si = (s - base).toInt()
            for (i in 0 until hs) {
                out[produced + i] = overlap[i] + input[si + i] * window[i]
                overlap[i] = input[si + hs + i] * window[hs + i]
            }
            produced += hs
            prevStart = s
            ana += ha
            compact()
        }
        return produced
    }

    /** Emits the remaining tail. */
    fun flush(): Int {
        if (bypass) return 0
        ensureOut(hs)
        System.arraycopy(overlap, 0, out, 0, hs)
        overlap.fill(0f)
        return hs
    }

    val output: FloatArray get() = out

    private fun bestStart(anaStart: Long): Long {
        val natural = (prevStart + hs - base).toInt()
        var best = anaStart
        var bestScore = Double.NEGATIVE_INFINITY
        var k = -tol
        while (k <= tol) {
            val cand = anaStart + k
            if (cand >= base) {
                val ci = (cand - base).toInt()
                var score = 0.0
                var i = 0
                while (i < n) {
                    score += input[ci + i] * input[natural + i]
                    i += 4
                }
                if (score > bestScore) {
                    bestScore = score
                    best = cand
                }
            }
            k += 2
        }
        return best
    }

    private fun append(samples: FloatArray, offset: Int, count: Int) {
        if (inLen + count > input.size) input = input.copyOf(maxOf(input.size * 2, inLen + count))
        System.arraycopy(samples, offset, input, inLen, count)
        inLen += count
    }

    private fun compact() {
        val keepFrom = minOf(prevStart + hs, ana.toLong() - tol)
        val drop = (keepFrom - base).toInt()
        if (drop > input.size / 2) {
            System.arraycopy(input, drop, input, 0, inLen - drop)
            inLen -= drop
            base += drop
        }
    }

    private fun ensureOut(size: Int) {
        if (out.size < size) out = out.copyOf(maxOf(size, out.size * 2))
    }
}
