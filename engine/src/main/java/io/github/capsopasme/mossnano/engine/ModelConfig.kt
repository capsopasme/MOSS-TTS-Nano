package io.github.capsopasme.mossnano.engine

import java.io.File

/** A voice = precomputed codec tokens of a short reference clip (prompt_audio_codes). */
class VoicePrompt(
    val id: String,
    val displayName: String,
    val group: String,
    /** [frames][n_vq] codec tokens. */
    val codes: Array<IntArray>,
    val builtin: Boolean,
) {
    val frames: Int get() = codes.size
    override fun toString(): String = "$displayName ($id, ${codes.size} frames)"
}

internal class TtsConfig(
    val nVq: Int,
    val audioPadTokenId: Int,
    val audioStartTokenId: Int,
    val audioEndTokenId: Int,
    val audioUserSlotTokenId: Int,
    val audioAssistantSlotTokenId: Int,
    val codebookSize: Int,
) {
    val rowWidth: Int get() = nVq + 1
}

internal class PromptTemplates(
    val userPrefix: IntArray,
    val userAfterReference: IntArray,
    val assistantPrefix: IntArray,
)

internal class CodecStateSpec(
    val inputName: String,
    val outputName: String,
    val shape: LongArray,
    /** 0 = int32 zeros, 1 = float zeros, 2 = int32 filled with -1 */
    val kind: Int,
)

internal class CodecConfig(
    val sampleRate: Int,
    val channels: Int,
    val numQuantizers: Int,
    val downsampleRate: Int,
    val decodeStepFile: String,
    val encodeFile: String?,
    val stateSpecs: List<CodecStateSpec>,
)

/**
 * Parsed view of browser_poc_manifest.json + tts_browser_onnx_meta.json +
 * codec_browser_onnx_meta.json from the official MOSS-TTS-Nano ONNX release.
 */
internal class ModelConfig(
    val ttsDir: File,
    val codecDir: File,
    val tokenizerFile: File,
    val tts: TtsConfig,
    val templates: PromptTemplates,
    val maxNewFrames: Int,
    val prefillFile: String,
    val decodeStepFile: String,
    val localFixedFrameFile: String,
    val prefillOutputNames: List<String>,
    val decodeInputNames: List<String>,
    val decodeOutputNames: List<String>,
    val codec: CodecConfig,
    val builtinVoices: List<VoicePrompt>,
) {
    companion object {
        private val MANIFEST_CANDIDATES = listOf(
            "browser_poc_manifest.json",
            "MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json",
            "MOSS-TTS-Nano-ONNX-CPU/browser_poc_manifest.json",
        )

        fun findManifest(root: File): File? =
            MANIFEST_CANDIDATES.map { File(root, it) }.firstOrNull { it.isFile }

        fun load(root: File): ModelConfig {
            val manifestFile = findManifest(root)
                ?: throw MissingModelException("browser_poc_manifest.json not found under ${root.absolutePath}")
            val manifestDir = manifestFile.parentFile!!
            val manifest = MiniJson.parse(manifestFile.readText()).obj()

            val modelFiles = manifest["model_files"].obj()
            val ttsMetaFile = resolveRelative(manifestDir, modelFiles["tts_meta"].str())
            val codecMetaFile = resolveRelative(manifestDir, modelFiles["codec_meta"].str())
            val tokenizerFile = resolveRelative(manifestDir, (modelFiles["tokenizer_model"] as? String) ?: "tokenizer.model")
            for (f in listOf(ttsMetaFile, codecMetaFile, tokenizerFile)) {
                if (!f.isFile) throw MissingModelException("Missing ${f.absolutePath}")
            }
            val ttsMeta = MiniJson.parse(ttsMetaFile.readText()).obj()
            val codecMeta = MiniJson.parse(codecMetaFile.readText()).obj()

            val cfg = manifest["tts_config"].obj()
            val tts = TtsConfig(
                nVq = cfg["n_vq"].int(),
                audioPadTokenId = cfg["audio_pad_token_id"].int(),
                audioStartTokenId = cfg["audio_start_token_id"].int(),
                audioEndTokenId = cfg["audio_end_token_id"].int(),
                audioUserSlotTokenId = (cfg["audio_user_slot_token_id"] ?: 8).int(),
                audioAssistantSlotTokenId = cfg["audio_assistant_slot_token_id"].int(),
                codebookSize = cfg["audio_codebook_sizes"].intArray().firstOrNull() ?: 1024,
            )
            val tpl = manifest["prompt_templates"].obj()
            val templates = PromptTemplates(
                userPrefix = tpl["user_prompt_prefix_token_ids"].intArray(),
                userAfterReference = tpl["user_prompt_after_reference_token_ids"].intArray(),
                assistantPrefix = tpl["assistant_prompt_prefix_token_ids"].intArray(),
            )
            val genDefaults = manifest["generation_defaults"] as? Map<*, *>
            val maxNewFrames = (genDefaults?.get("max_new_frames") ?: 375).int()

            val voices = (manifest["builtin_voices"] as? List<*>).orEmpty().mapNotNull { raw ->
                val v = raw.obj()
                val codes = (v["prompt_audio_codes"] as? List<*>)?.map { row -> row.intArray() }?.toTypedArray()
                if (codes.isNullOrEmpty()) return@mapNotNull null
                val id = v["voice"].str()
                VoicePrompt(
                    id = id,
                    displayName = (v["display_name"] as? String)?.takeIf { it.isNotBlank() } ?: id,
                    group = (v["group"] as? String).orEmpty(),
                    codes = codes,
                    builtin = true,
                )
            }

            val files = ttsMeta["files"].obj()
            val onnx = ttsMeta["onnx"].obj()
            val ttsDir = ttsMetaFile.parentFile!!

            val codecFiles = codecMeta["files"].obj()
            val codecCfg = codecMeta["codec_config"].obj()
            val streaming = (codecMeta["streaming_decode"] as? Map<*, *>)
                ?: throw MissingModelException("codec meta has no streaming_decode section")
            val specs = ArrayList<CodecStateSpec>()
            for (raw in (streaming["transformer_offsets"] as? List<*>).orEmpty()) {
                val t = raw.obj()
                specs += CodecStateSpec(t["input_name"].str(), t["output_name"].str(), t["shape"].longArray(), 0)
            }
            for (raw in (streaming["attention_caches"] as? List<*>).orEmpty()) {
                val a = raw.obj()
                specs += CodecStateSpec(a["offset_input_name"].str(), a["offset_output_name"].str(), a["offset_shape"].longArray(), 0)
                specs += CodecStateSpec(a["cached_keys_input_name"].str(), a["cached_keys_output_name"].str(), a["cache_shape"].longArray(), 1)
                specs += CodecStateSpec(a["cached_values_input_name"].str(), a["cached_values_output_name"].str(), a["cache_shape"].longArray(), 1)
                specs += CodecStateSpec(a["cached_positions_input_name"].str(), a["cached_positions_output_name"].str(), a["positions_shape"].longArray(), 2)
            }
            val codec = CodecConfig(
                sampleRate = codecCfg["sample_rate"].int(),
                channels = (codecCfg["channels"] ?: 2).int(),
                numQuantizers = (codecCfg["num_quantizers"] ?: tts.nVq).int(),
                downsampleRate = (codecCfg["downsample_rate"] ?: 3840).int(),
                decodeStepFile = codecFiles["decode_step"].str(),
                encodeFile = codecFiles["encode"] as? String,
                stateSpecs = specs,
            )

            return ModelConfig(
                ttsDir = ttsDir,
                codecDir = codecMetaFile.parentFile!!,
                tokenizerFile = tokenizerFile,
                tts = tts,
                templates = templates,
                maxNewFrames = maxNewFrames,
                prefillFile = files["prefill"].str(),
                decodeStepFile = files["decode_step"].str(),
                localFixedFrameFile = (files["local_fixed_sampled_frame"] as? String)
                    ?: throw MissingModelException("This ONNX export has no local_fixed_sampled_frame graph"),
                prefillOutputNames = onnx["prefill_output_names"].arr().map { it.str() },
                decodeInputNames = onnx["decode_input_names"].arr().map { it.str() },
                decodeOutputNames = onnx["decode_output_names"].arr().map { it.str() },
                codec = codec,
                builtinVoices = voices,
            )
        }

        /** Same alias handling as the official runtimes (old "-ONNX-CPU" dir names). */
        private fun resolveRelative(base: File, relative: String): File {
            val direct = File(base, relative).canonicalFile
            if (direct.exists()) return direct
            val alias = relative
                .replace("MOSS-Audio-Tokenizer-Nano-ONNX-CPU", "MOSS-Audio-Tokenizer-Nano-ONNX")
                .replace("MOSS-TTS-Nano-ONNX-CPU", "MOSS-TTS-Nano-100M-ONNX")
            return File(base, alias).canonicalFile
        }
    }
}

class MissingModelException(message: String) : Exception(message)
