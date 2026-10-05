package io.github.capsopasme.mossnano

import io.github.capsopasme.mossnano.engine.AudioSink
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Writes streamed audio to a 16-bit PCM WAV file (header patched on finish). */
class WavFileSink(private val file: File) : AudioSink {
    private var raf: RandomAccessFile? = null
    private var channels = 2
    private var sampleRate = 48000
    private var dataBytes = 0L
    private var buf = ByteBuffer.allocate(0).order(ByteOrder.LITTLE_ENDIAN)

    override fun onStart(sampleRate: Int, channels: Int) {
        this.channels = channels
        this.sampleRate = sampleRate
        file.parentFile?.mkdirs()
        raf = RandomAccessFile(file, "rw").apply {
            setLength(0)
            write(ByteArray(44))
        }
        dataBytes = 0
    }

    override fun onAudio(samples: FloatArray, frames: Int): Boolean {
        val r = raf ?: return false
        val n = frames * channels
        if (buf.capacity() < n * 2) buf = ByteBuffer.allocate(n * 2).order(ByteOrder.LITTLE_ENDIAN)
        buf.clear()
        for (i in 0 until n) {
            val s = (samples[i].coerceIn(-1f, 1f) * 32767f).toInt()
            buf.putShort(s.toShort())
        }
        r.write(buf.array(), 0, n * 2)
        dataBytes += n * 2
        return true
    }

    override fun onFinish() {
        val r = raf ?: return
        val h = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        h.put("RIFF".toByteArray()).putInt((36 + dataBytes).toInt()).put("WAVE".toByteArray())
        h.put("fmt ".toByteArray()).putInt(16).putShort(1).putShort(channels.toShort())
        h.putInt(sampleRate).putInt(sampleRate * channels * 2).putShort((channels * 2).toShort()).putShort(16)
        h.put("data".toByteArray()).putInt(dataBytes.toInt())
        r.seek(0)
        r.write(h.array())
        r.close()
        raf = null
    }
}
