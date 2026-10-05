package io.github.capsopasme.mossnano

import android.content.Context
import android.content.SharedPreferences

enum class ModelVariant(val dirName: String, val label: String) {
    FP32("fp32", "FP32（官方原版）"),
    INT8("int8", "INT8（动态量化，更快更省内存）"),
}

enum class DownloadSource(val label: String, val hfBase: String) {
    HF("Hugging Face", "https://huggingface.co"),
    HF_MIRROR("hf-mirror.com（国内镜像）", "https://hf-mirror.com"),
}

class AppSettings(context: Context) {
    private val prefs: SharedPreferences = context.getSharedPreferences("settings", Context.MODE_PRIVATE)

    var variant: ModelVariant
        get() = runCatching { ModelVariant.valueOf(prefs.getString("variant", null)!!) }.getOrDefault(ModelVariant.FP32)
        set(v) = prefs.edit().putString("variant", v.name).apply()

    /** Shared LM pool size. Fixed per process (ORT global thread pool), applied on next start. */
    var lmThreads: Int
        get() = prefs.getInt("lmThreads", 4)
        set(v) = prefs.edit().putInt("lmThreads", v.coerceIn(1, 8)).apply()

    var codecThreads: Int
        get() = prefs.getInt("codecThreads", 2)
        set(v) = prefs.edit().putInt("codecThreads", v.coerceIn(1, 4)).apply()

    var spinning: Boolean
        get() = prefs.getBoolean("spinning", true)
        set(v) = prefs.edit().putBoolean("spinning", v).apply()

    var voiceId: String?
        get() = prefs.getString("voiceId", null)
        set(v) = prefs.edit().putString("voiceId", v).apply()

    var fixedSeed: Boolean
        get() = prefs.getBoolean("fixedSeed", true)
        set(v) = prefs.edit().putBoolean("fixedSeed", v).apply()

    var normalize: Boolean
        get() = prefs.getBoolean("normalize", true)
        set(v) = prefs.edit().putBoolean("normalize", v).apply()

    /** App playback speed (AudioTrack time-stretch, pitch preserved). */
    var speed: Float
        get() = prefs.getFloat("speed", 1.0f)
        set(v) = prefs.edit().putFloat("speed", v.coerceIn(0.5f, 2.0f)).apply()

    /** Free the ~1 GB of model memory after this many idle minutes (0 = never). */
    var idleUnloadMinutes: Int
        get() = prefs.getInt("idleUnloadMinutes", 10)
        set(v) = prefs.edit().putInt("idleUnloadMinutes", v.coerceAtLeast(0)).apply()

    var downloadSource: DownloadSource
        get() = runCatching { DownloadSource.valueOf(prefs.getString("downloadSource", null)!!) }.getOrDefault(DownloadSource.HF_MIRROR)
        set(v) = prefs.edit().putString("downloadSource", v.name).apply()

    var lastText: String
        get() = prefs.getString("lastText", null) ?: DEFAULT_TEXT
        set(v) = prefs.edit().putString("lastText", v).apply()

    companion object {
        const val DEFAULT_TEXT =
            "你好，我是运行在你手机上的 MOSS-TTS-Nano。我只有一亿参数，完全离线，边生成边播放，首个音频通常在一秒内就能听到。"
    }
}
