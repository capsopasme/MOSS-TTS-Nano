package io.github.capsopasme.mossnano.engine

/**
 * MOSS-TTS-Nano reproduces the level of the reference clip: the built-in clips span ~17 dB of
 * loudness (Soyo / Arisa / Nathan / 说书 / 深夜电台 are much quieter than 欢迎关注模思智能), so
 * some voices come out far quieter than others. Quiet voices are lifted to [TARGET_LUFS]; loud
 * ones are left alone.
 *
 * Built-in values: integrated loudness (BS.1770, power mean) of 7 sentences per voice (3 in the
 * voice's language or Chinese + 4 Chinese, seed 1234) generated with the INT8 pack by this
 * engine on a desktop JVM. Generated speech was as loud as the decoded reference clip (median
 * difference -0.1 dB, sd 1.3 dB), which is why clones are normalized at clone time instead.
 */
object VoiceLoudness {
    /** Level quiet voices are lifted to (≈ the louder built-in voices; Apple's podcast loudness). */
    const val TARGET_LUFS = -16.0

    /** Level cloned reference clips are normalized to: generated speech ends up about as loud. */
    const val REFERENCE_TARGET_LUFS = TARGET_LUFS

    /** Never boost more than this automatically (noise floor; the user can add more per voice). */
    const val MAX_AUTO_GAIN_DB = 15.0

    private class Entry(val frames: Int, val lufs: Double)

    // voice id -> (prompt frames of the official clip, loudness of generated speech)
    private val BUILTIN = mapOf(
        "Junhao" to Entry(98, -15.0),
        "Zhiming" to Entry(98, -15.9),
        "Weiguo" to Entry(140, -23.9),
        "Xiaoyu" to Entry(180, -18.9),
        "Yuewen" to Entry(102, -16.2),
        "Lingyu" to Entry(218, -21.7),
        "Trump" to Entry(97, -22.2),
        "Ava" to Entry(98, -19.2),
        "Bella" to Entry(59, -18.1),
        "Adam" to Entry(59, -16.6),
        "Nathan" to Entry(168, -25.1),
        "Soyo" to Entry(125, -32.3),
        "Saki" to Entry(32, -18.6),
        "Mortis" to Entry(60, -19.3),
        "Umiri" to Entry(77, -17.0),
        "Mei" to Entry(49, -18.3),
        "Anon" to Entry(47, -22.3),
        "Arisa" to Entry(85, -27.5),
    )

    /** Measured loudness of a built-in voice; NaN if the clip is not the one measured (other manifest). */
    fun builtinLufs(id: String, codes: Array<IntArray>): Double {
        val e = BUILTIN[id] ?: return Double.NaN
        return if (e.frames == codes.size) e.lufs else Double.NaN
    }

    /** Expected loudness of speech cloned from a reference clip measuring [referenceLufs]. */
    fun estimateFromReference(referenceLufs: Double): Double = referenceLufs

    /** Compensation for [voice] in dB: 0 for unknown / loud voices, rounded to 0.5 dB. */
    fun autoGainDb(voice: VoicePrompt): Float {
        val l = voice.loudnessLufs
        if (!l.isFinite()) return 0f
        val deficit = (TARGET_LUFS - l).coerceIn(0.0, MAX_AUTO_GAIN_DB)
        return if (deficit < 0.5) 0f else (Math.round(deficit * 2) / 2.0).toFloat()
    }
}
