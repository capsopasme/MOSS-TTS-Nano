package io.github.capsopasme.mossnano

import android.content.Context
import io.github.capsopasme.mossnano.engine.VoicePrompt
import io.github.capsopasme.mossnano.engine.VoiceLoudness
import java.util.Locale

/** Language of a voice's reference clip: the official groups are Chinese / English / Japanese; clones count as Chinese. */
object VoiceLang {
    const val ZH = "zh"
    const val EN = "en"
    const val JA = "ja"
    val ALL = listOf(ZH, EN, JA)

    fun of(v: VoicePrompt): String = when {
        v.group.contains("English", ignoreCase = true) -> EN
        v.group.contains("Japanese", ignoreCase = true) -> JA
        else -> ZH
    }

    fun locale(lang: String): Locale = when (lang) {
        EN -> Locale.US
        JA -> Locale.JAPAN
        else -> Locale.SIMPLIFIED_CHINESE
    }

    /** ISO 639-2 code from the TTS framework ("zho", "eng", …) -> voice language; other languages use the Chinese slot. */
    fun fromIso3(iso3: String?): String = when (iso3) {
        "eng" -> EN
        "jpn" -> JA
        else -> ZH
    }

    fun label(lang: String): String = when (lang) {
        EN -> "英文"
        JA -> "日文"
        else -> "中文"
    }
}

object Voices {
    /**
     * The voice used for [lang] when nobody asked for a specific one (system TTS default, read-aloud):
     * the voice last picked in the app for that language, else the app voice if it speaks it, else
     * the first built-in voice of the language. Evaluated per request, so a change in the app applies
     * to the very next sentence, also for apps that stay connected to the engine.
     */
    fun forLanguage(context: Context, lang: String, all: List<VoicePrompt> = EngineManager.voices(context)): VoicePrompt? {
        val settings = AppSettings(context)
        fun find(id: String?) = id?.let { i -> all.firstOrNull { it.id == i } }
        find(settings.lastVoiceFor(lang))?.let { return it }
        val app = find(settings.voiceId)
        if (app != null && VoiceLang.of(app) == lang) return app
        all.firstOrNull { it.builtin && VoiceLang.of(it) == lang }?.let { return it }
        return app ?: all.firstOrNull()
    }

    /** Automatic loudness compensation of [voice] in dB (quiet reference clips -> quiet speech). */
    fun autoGainDb(voice: VoicePrompt): Float = VoiceLoudness.autoGainDb(voice)

    /** Total output gain for [voice]: automatic compensation + the user's adjustment. */
    fun gainDb(context: Context, voice: VoicePrompt): Float =
        autoGainDb(voice) + AppSettings(context).volumeOffsetDb(voice.id)
}
