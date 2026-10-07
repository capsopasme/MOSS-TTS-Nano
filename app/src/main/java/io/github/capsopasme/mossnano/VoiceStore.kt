package io.github.capsopasme.mossnano

import android.content.Context
import io.github.capsopasme.mossnano.engine.VoicePrompt
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** Cloned voices = codec tokens of the user's reference clip, stored as small JSON files. */
object VoiceStore {
    private fun dir(context: Context) = File(context.filesDir, "voices").apply { mkdirs() }

    /**
     * The TTS framework asks for the voice list several times per utterance (valid voice, load
     * voice, default voice, the request itself): the files are parsed once and kept until one is
     * added, removed or changed.
     */
    @Volatile private var cache: Pair<String, List<VoicePrompt>>? = null

    fun list(context: Context): List<VoicePrompt> {
        val files = dir(context).listFiles { f -> f.extension == "json" }.orEmpty().sortedBy { it.lastModified() }
        val key = files.joinToString("|") { "${it.name}:${it.lastModified()}:${it.length()}" }
        cache?.let { (k, v) -> if (k == key) return v }
        val voices = files.mapNotNull { f ->
            runCatching {
                val o = JSONObject(f.readText())
                val rows = o.getJSONArray("codes")
                val codes = Array(rows.length()) { i ->
                    val r = rows.getJSONArray(i)
                    IntArray(r.length()) { r.getInt(it) }
                }
                // clones made before v0.3.0 have no loudness estimate -> no automatic compensation
                VoicePrompt(o.getString("id"), o.getString("name"), "克隆", codes, builtin = false, loudnessLufs = o.optDouble("lufs", Double.NaN))
            }.getOrNull()
        }
        cache = key to voices
        return voices
    }

    /** @param lufs expected loudness of speech in this voice (see [VoicePrompt.loudnessLufs]), NaN if unknown. */
    fun save(context: Context, name: String, codes: Array<IntArray>, lufs: Double): VoicePrompt {
        val id = "clone_" + System.currentTimeMillis()
        val rows = JSONArray()
        for (r in codes) rows.put(JSONArray().apply { r.forEach { put(it) } })
        val o = JSONObject().put("id", id).put("name", name).put("codes", rows)
        if (lufs.isFinite()) o.put("lufs", lufs)
        // written whole, then renamed: a voice list read meanwhile never sees half a file
        val tmp = File(dir(context), "$id.json.tmp")
        tmp.writeText(o.toString())
        tmp.renameTo(File(dir(context), "$id.json"))
        cache = null
        return VoicePrompt(id, name, "克隆", codes, builtin = false, loudnessLufs = lufs)
    }

    fun delete(context: Context, id: String) {
        File(dir(context), "$id.json").delete()
        cache = null
    }
}
