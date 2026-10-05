package io.github.capsopasme.mossnano.engine

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.sin
import kotlin.math.sqrt

/** Windowed-sinc (Kaiser) resampler used to bring reference clips to the codec's 48 kHz. */
object Resampler {
    fun resample(input: FloatArray, fromRate: Int, toRate: Int, halfWidth: Int = 16): FloatArray {
        if (fromRate == toRate || input.isEmpty()) return input.copyOf()
        val ratio = toRate.toDouble() / fromRate
        val outLen = floor(input.size * ratio).toInt()
        val out = FloatArray(outLen)
        val cutoff = minOf(1.0, ratio) * 0.97          // anti-alias when downsampling
        val beta = 8.6
        val i0Beta = besselI0(beta)
        // Kaiser window lookup table (the Bessel series per tap is the expensive part).
        val tableRes = 512
        val table = FloatArray(halfWidth * tableRes + 2) { kaiser(it.toDouble() / (halfWidth * tableRes), beta, i0Beta).toFloat() }
        for (n in 0 until outLen) {
            val center = n / ratio
            val left = floor(center).toInt() - halfWidth + 1
            var acc = 0.0
            var norm = 0.0
            for (k in left until left + 2 * halfWidth) {
                if (k < 0 || k >= input.size) continue
                val x = center - k
                val pos = abs(x) * tableRes
                val idx = pos.toInt()
                if (idx >= table.size - 1) continue
                val frac = pos - idx
                val w = table[idx] + (table[idx + 1] - table[idx]) * frac
                val s = sinc(x * cutoff) * cutoff * w
                acc += input[k] * s
                norm += s
            }
            out[n] = if (norm != 0.0) (acc / norm).toFloat() else 0f
        }
        return out
    }

    private fun sinc(x: Double): Double = if (abs(x) < 1e-9) 1.0 else sin(PI * x) / (PI * x)

    private fun kaiser(t: Double, beta: Double, i0Beta: Double): Double {
        if (abs(t) > 1.0) return 0.0
        return besselI0(beta * sqrt(1 - t * t)) / i0Beta
    }

    private fun besselI0(x: Double): Double {
        var sum = 1.0
        var term = 1.0
        val q = x * x / 4
        var k = 1
        while (k < 50) {
            term *= q / (k * k)
            sum += term
            if (term < 1e-12 * sum) break
            k++
        }
        return sum
    }
}
