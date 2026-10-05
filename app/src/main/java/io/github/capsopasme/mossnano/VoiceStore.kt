package io.github.capsopasme.mossnano

import android.content.Context
import io.github.capsopasme.mossnano.engine.VoicePrompt
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** Cloned voices = codec tokens of the user's reference clip, stored as small JSON files. */
object VoiceStore {
    private fun dir(context: Context) = File(context.filesDir, "voices").apply { mkdirs() }

    fun list(context: Context): List<VoicePrompt> =
        dir(context).listFiles { f -> f.extension == "json" }.orEmpty().sortedBy { it.lastModified() }.mapNotNull { f ->
            runCatching {
                val o = JSONObject(f.readText())
                val rows = o.getJSONArray("codes")
                val codes = Array(rows.length()) { i ->
                    val r = rows.getJSONArray(i)
                    IntArray(r.length()) { r.getInt(it) }
                }
                VoicePrompt(o.getString("id"), o.getString("name"), "克隆", codes, builtin = false)
            }.getOrNull()
        }

    fun save(context: Context, name: String, codes: Array<IntArray>): VoicePrompt {
        val id = "clone_" + System.currentTimeMillis()
        val rows = JSONArray()
        for (r in codes) rows.put(JSONArray().apply { r.forEach { put(it) } })
        val o = JSONObject().put("id", id).put("name", name).put("codes", rows)
        File(dir(context), "$id.json").writeText(o.toString())
        return VoicePrompt(id, name, "克隆", codes, builtin = false)
    }

    fun delete(context: Context, id: String) {
        File(dir(context), "$id.json").delete()
    }
}
