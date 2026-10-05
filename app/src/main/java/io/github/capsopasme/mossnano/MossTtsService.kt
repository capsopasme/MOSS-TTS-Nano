package io.github.capsopasme.mossnano

import android.media.AudioFormat
import android.os.Process
import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeechService
import android.speech.tts.Voice
import android.util.Log
import io.github.capsopasme.mossnano.engine.AudioSink
import io.github.capsopasme.mossnano.engine.CancelSignal
import io.github.capsopasme.mossnano.engine.SynthRequest
import io.github.capsopasme.mossnano.engine.TimeStretch
import io.github.capsopasme.mossnano.engine.VoicePrompt
import java.util.Locale

/**
 * Android system TTS engine backed by the streaming MOSS-TTS-Nano engine, so any app
 * (e-book readers, navigation, the PhoneAssistant…) can use it. Audio is pushed to the
 * framework frame-by-frame as it is generated.
 */
class MossTtsService : TextToSpeechService() {
    companion object {
        private const val TAG = "MossTtsService"
        /** "auto-zh" / "auto-en" / "auto-ja": follow the voice picked in the app (see [onGetVoices]). */
        const val AUTO_PREFIX = "auto-"
        /** Languages listed by the upstream README. */
        private val LANGS = setOf(
            "zho", "eng", "deu", "spa", "fra", "jpn", "ita", "hun", "kor", "rus",
            "fas", "ara", "pol", "por", "ces", "dan", "swe", "ell", "tur",
        )
    }

    @Volatile private var cancel: CancelSignal? = null
    @Volatile private var currentLang = arrayOf("zho", "CHN", "")

    override fun onIsLanguageAvailable(lang: String?, country: String?, variant: String?): Int {
        val l = iso3(lang) ?: return TextToSpeech.LANG_NOT_SUPPORTED
        if (l !in LANGS) return TextToSpeech.LANG_NOT_SUPPORTED
        return if (country.isNullOrEmpty()) TextToSpeech.LANG_AVAILABLE else TextToSpeech.LANG_COUNTRY_AVAILABLE
    }

    override fun onGetLanguage(): Array<String> = currentLang

    override fun onLoadLanguage(lang: String?, country: String?, variant: String?): Int {
        val r = onIsLanguageAvailable(lang, country, variant)
        if (r >= TextToSpeech.LANG_AVAILABLE) currentLang = arrayOf(iso3(lang) ?: "zho", country.orEmpty(), "")
        return r
    }

    override fun onStop() {
        cancel?.cancel()
    }

    /**
     * Besides the real voices, one "auto" voice per language is listed and returned as the default
     * voice. Apps that don't pick a voice themselves (most do not) store the default voice name when
     * they connect and send it with every request; with an auto name the engine resolves the voice
     * per request ([Voices.forLanguage]), so changing the voice in the app applies at once, also to
     * apps that stay connected for a long time. Picking a real voice in an app still pins it.
     */
    private fun autoName(lang: String) = AUTO_PREFIX + lang

    private fun autoLang(name: String?): String? =
        name?.takeIf { it.startsWith(AUTO_PREFIX) }?.removePrefix(AUTO_PREFIX)?.takeIf { it in VoiceLang.ALL }

    override fun onGetVoices(): List<Voice> {
        val real = EngineManager.voices(this)
        if (real.isEmpty()) return emptyList()
        val auto = VoiceLang.ALL.map { l ->
            Voice(autoName(l), VoiceLang.locale(l), Voice.QUALITY_VERY_HIGH, Voice.LATENCY_NORMAL, false, emptySet())
        }
        return auto + real.map { v ->
            Voice(v.id, VoiceLang.locale(VoiceLang.of(v)), Voice.QUALITY_VERY_HIGH, Voice.LATENCY_NORMAL, false, emptySet())
        }
    }

    override fun onIsValidVoiceName(voiceName: String?): Int = when {
        autoLang(voiceName) != null -> if (EngineManager.voices(this).isNotEmpty()) TextToSpeech.SUCCESS else TextToSpeech.ERROR
        EngineManager.voices(this).any { it.id == voiceName } -> TextToSpeech.SUCCESS
        else -> TextToSpeech.ERROR
    }

    override fun onLoadVoice(voiceName: String?): Int = onIsValidVoiceName(voiceName)

    override fun onGetDefaultVoiceNameFor(lang: String?, country: String?, variant: String?): String? {
        if (EngineManager.voices(this).isEmpty()) return null
        return autoName(VoiceLang.fromIso3(iso3(lang)))
    }

    /** A real voice the client picked, else the current default voice for the request's language. */
    private fun voiceFor(request: SynthesisRequest): VoicePrompt? {
        val all = EngineManager.voices(this)
        val name = request.voiceName
        if (!name.isNullOrEmpty() && autoLang(name) == null) all.firstOrNull { it.id == name }?.let { return it }
        // auto voice, no voice name, or a voice that no longer exists (deleted clone)
        val lang = autoLang(name) ?: VoiceLang.fromIso3(iso3(request.language))
        return Voices.forLanguage(this, lang, all)
    }

    override fun onSynthesizeText(request: SynthesisRequest, callback: SynthesisCallback) {
        Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)
        val text = request.charSequenceText?.toString() ?: request.text.orEmpty()
        if (text.isBlank()) {
            callback.start(48000, AudioFormat.ENCODING_PCM_16BIT, 1)
            callback.done()
            return
        }
        val settings = AppSettings(this)
        val voice = voiceFor(request)
        if (voice == null || !ModelStore.isReady(this, settings.variant)) {
            callback.error(TextToSpeech.ERROR_NOT_INSTALLED_YET)
            return
        }
        val c = CancelSignal()
        cancel = c
        val speed = (request.speechRate.coerceIn(30, 400)) / 100f
        try {
            val stats = EngineManager.withEngine { engine ->
                if (c.isCancelled) return@withEngine null
                engine.synthesize(
                    SynthRequest(
                        text, voice,
                        seed = if (settings.fixedSeed) 1234L else null,
                        normalizeText = settings.normalize,
                        gainDb = Voices.gainDb(this, voice),
                    ),
                    CallbackSink(callback, c, speed),
                    c,
                )
            }
            if (stats != null) Log.i(TAG, stats.summary())
            if (c.isCancelled) callback.error(TextToSpeech.ERROR_SERVICE) else callback.done()
        } catch (t: Throwable) {
            Log.e(TAG, "synthesis failed", t)
            callback.error(TextToSpeech.ERROR_SYNTHESIS)
        } finally {
            if (cancel === c) cancel = null
        }
    }

    /** Stereo float -> mono (WSOLA speed) -> PCM16 chunks for the framework. */
    private class CallbackSink(
        private val cb: SynthesisCallback,
        private val cancel: CancelSignal,
        speed: Float,
    ) : AudioSink {
        private var channels = 2
        private var mono = FloatArray(0)
        private var bytes = ByteArray(0)
        private var stretch: TimeStretch? = null
        private val speed = speed
        private var started = false

        override fun onStart(sampleRate: Int, channels: Int) {
            this.channels = channels
            stretch = TimeStretch(sampleRate, speed)
            started = cb.start(sampleRate, AudioFormat.ENCODING_PCM_16BIT, 1) == TextToSpeech.SUCCESS
        }

        override fun onAudio(samples: FloatArray, frames: Int): Boolean {
            if (!started || cancel.isCancelled) return false
            if (mono.size < frames) mono = FloatArray(frames)
            for (i in 0 until frames) {
                var s = 0f
                for (c in 0 until channels) s += samples[i * channels + c]
                mono[i] = s / channels
            }
            val ts = stretch!!
            val n = ts.process(mono, 0, frames)
            return push(ts.output, n)
        }

        override fun onFinish() {
            val ts = stretch ?: return
            if (started && !cancel.isCancelled) push(ts.output, ts.flush())
        }

        private fun push(src: FloatArray, n: Int): Boolean {
            if (n <= 0) return true
            if (bytes.size < n * 2) bytes = ByteArray(n * 2)
            for (i in 0 until n) {
                val v = (src[i].coerceIn(-1f, 1f) * 32767f).toInt()
                bytes[2 * i] = v.toByte()
                bytes[2 * i + 1] = (v shr 8).toByte()
            }
            val max = cb.maxBufferSize.coerceAtLeast(2) and 1.inv()
            var off = 0
            val total = n * 2
            while (off < total) {
                val len = minOf(max, total - off)
                if (cb.audioAvailable(bytes, off, len) != TextToSpeech.SUCCESS) return false
                off += len
            }
            return true
        }
    }

    private fun iso3(lang: String?): String? {
        if (lang.isNullOrEmpty()) return null
        return when (lang.lowercase()) {
            "zh", "zho", "chi", "cmn" -> "zho"
            "en", "eng" -> "eng"
            "ja", "jpn" -> "jpn"
            "ko", "kor" -> "kor"
            else -> runCatching { Locale(lang).isO3Language }.getOrNull()
        }
    }
}
