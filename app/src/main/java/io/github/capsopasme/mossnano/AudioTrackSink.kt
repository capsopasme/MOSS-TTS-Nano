package io.github.capsopasme.mossnano

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.media.PlaybackParams
import android.os.Build
import io.github.capsopasme.mossnano.engine.AudioSink
import io.github.capsopasme.mossnano.engine.CancelSignal

/**
 * Streams the engine's 48 kHz float audio straight into an AudioTrack.
 *
 *  - float PCM, native channel count -> no conversion on the hot path
 *  - start threshold = one codec frame (80 ms) on Android 12+, so playback begins as soon
 *    as the first frame is decoded instead of after a whole buffer
 *  - blocking writes give natural back-pressure to the codec thread
 *  - speed via PlaybackParams (platform time-stretch, pitch preserved)
 */
class AudioTrackSink(
    private val cancel: CancelSignal,
    private val speed: Float = 1f,
) : AudioSink {
    @Volatile private var track: AudioTrack? = null
    private var channels = 2
    private var sampleRate = 48000
    private var written = 0L

    override fun onStart(sampleRate: Int, channels: Int) {
        this.channels = channels
        this.sampleRate = sampleRate
        val mask = if (channels == 1) AudioFormat.CHANNEL_OUT_MONO else AudioFormat.CHANNEL_OUT_STEREO
        val minBuf = AudioTrack.getMinBufferSize(sampleRate, mask, AudioFormat.ENCODING_PCM_FLOAT)
        val bytesPerFrame = 4 * channels
        val bufBytes = maxOf(minBuf * 2, sampleRate * bytesPerFrame / 4) // >= 250 ms
        val t = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(mask)
                    .build()
            )
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setBufferSizeInBytes(bufBytes)
            .build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            runCatching { t.setStartThresholdInFrames(minOf(3840, t.bufferSizeInFrames)) }
        }
        if (speed != 1f) {
            runCatching { t.playbackParams = PlaybackParams().setSpeed(speed).setPitch(1f) }
        }
        t.play()
        track = t
        written = 0
    }

    override fun onAudio(samples: FloatArray, frames: Int): Boolean {
        val t = track ?: return false
        if (cancel.isCancelled) return false
        val total = frames * channels
        var off = 0
        while (off < total) {
            val n = t.write(samples, off, total - off, AudioTrack.WRITE_BLOCKING)
            if (n < 0 || cancel.isCancelled) return false
            if (n == 0) Thread.sleep(2)
            off += n
        }
        written += frames
        return true
    }

    /** Lets the tail play out (unless cancelled), then releases the track. */
    override fun onFinish() {
        val t = track ?: return
        try {
            if (!cancel.isCancelled && written > 0) {
                if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S) {
                    // Pre-12 start threshold == whole buffer: push silence so short clips start.
                    val pad = FloatArray(t.bufferSizeInFrames * channels)
                    t.write(pad, 0, pad.size, AudioTrack.WRITE_BLOCKING)
                }
                val timeoutMs = (written * 1000 / sampleRate / speed).toLong() + 1500
                val deadline = System.currentTimeMillis() + timeoutMs
                while (!cancel.isCancelled && System.currentTimeMillis() < deadline) {
                    val head = t.playbackHeadPosition.toLong() and 0xFFFFFFFFL
                    if (head >= written) break
                    Thread.sleep(15)
                }
            }
        } finally {
            runCatching { t.pause(); t.flush() }
            t.release()
            track = null
        }
    }

    /** Unblocks a pending write immediately (called from another thread on stop). */
    fun abort() {
        track?.let { runCatching { it.pause(); it.flush() } }
    }
}
