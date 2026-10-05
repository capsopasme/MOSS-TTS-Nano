package io.github.capsopasme.mossnano

import android.content.Context
import java.io.File

/**
 * Where model files live and what has to be downloaded.
 *
 * Layout (same as the official runtimes, so `adb push` of the HF folders also works):
 *   <externalFiles>/models/<variant>/MOSS-TTS-Nano-100M-ONNX/...
 *   <externalFiles>/models/<variant>/MOSS-Audio-Tokenizer-Nano-ONNX/...
 */
object ModelStore {
    const val TTS_DIR = "MOSS-TTS-Nano-100M-ONNX"
    const val CODEC_DIR = "MOSS-Audio-Tokenizer-Nano-ONNX"
    const val TTS_REPO = "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX"
    const val CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"
    /**
     * Pinned Hugging Face revisions (the ones the sizes below and the INT8 pack were built from),
     * so an upstream re-export can't break downloads or mix versions.
     */
    const val TTS_REVISION = "f52645cb467506d8e18e746ddd59482685b74e58"
    const val CODEC_REVISION = "ceff0d0749bfb3fa2d61149794ec6feef0d1e1ae"
    /**
     * Release that hosts the INT8 pack built by .github/workflows/quantize-int8.yml. A new pack
     * gets a new tag; the expected sizes below make the app re-download changed files.
     */
    const val INT8_PACK_TAG = "models-int8-v2"
    const val INT8_RELEASE_BASE = "https://github.com/capsopasme/MOSS-TTS-Nano/releases/download/$INT8_PACK_TAG"
    // Sizes of the files in the INT8_PACK_TAG release (printed by quantize-int8.yml).
    private const val INT8_META_SIZE = 4_010L
    private const val INT8_PREFILL_SIZE = 186_880_955L
    private const val INT8_DECODE_SIZE = 186_892_813L
    private const val INT8_LOCAL_SIZE = 118_117_793L

    class RemoteFile(
        val subdir: String,
        val name: String,
        /** Expected size in bytes, or -1 when unknown (then any non-empty file counts). */
        val size: Long,
        /** Only needed for voice cloning. */
        val cloneOnly: Boolean = false,
    )

    val MANIFEST = RemoteFile(TTS_DIR, "browser_poc_manifest.json", 503_354)

    /** Only the graphs the streaming engine actually uses (skips decode_full / local_decoder / cached_step). */
    val FP32_FILES = listOf(
        MANIFEST,
        RemoteFile(TTS_DIR, "tts_browser_onnx_meta.json", 4_487),
        RemoteFile(TTS_DIR, "tokenizer.model", 470_897),
        RemoteFile(TTS_DIR, "moss_tts_prefill.onnx", 283_305),
        RemoteFile(TTS_DIR, "moss_tts_decode_step.onnx", 291_483),
        RemoteFile(TTS_DIR, "moss_tts_local_fixed_sampled_frame.onnx", 471_262),
        RemoteFile(TTS_DIR, "moss_tts_global_shared.data", 440_813_568),
        RemoteFile(TTS_DIR, "moss_tts_local_shared.data", 229_678_080),
        RemoteFile(CODEC_DIR, "codec_browser_onnx_meta.json", 17_036),
        RemoteFile(CODEC_DIR, "moss_audio_tokenizer_decode_step.onnx", 351_400),
        RemoteFile(CODEC_DIR, "moss_audio_tokenizer_decode_shared.data", 44_198_912),
        RemoteFile(CODEC_DIR, "moss_audio_tokenizer_encode.onnx", 815_775, cloneOnly = true),
        RemoteFile(CODEC_DIR, "moss_audio_tokenizer_encode.data", 44_507_136, cloneOnly = true),
    )

    /**
     * Produced by .github/workflows/quantize-int8.yml (tools/quantize_int8.py); LM graphs are
     * self-contained int8 files, everything else is the official file. Exact sizes double as an
     * integrity check and make an outdated pack (older tag) show up as "needs download".
     */
    val INT8_FILES = listOf(
        MANIFEST,
        RemoteFile(TTS_DIR, "tts_browser_onnx_meta.json", INT8_META_SIZE),
        RemoteFile(TTS_DIR, "tokenizer.model", 470_897),
        RemoteFile(TTS_DIR, "moss_tts_prefill.onnx", INT8_PREFILL_SIZE),
        RemoteFile(TTS_DIR, "moss_tts_decode_step.onnx", INT8_DECODE_SIZE),
        RemoteFile(TTS_DIR, "moss_tts_local_fixed_sampled_frame.onnx", INT8_LOCAL_SIZE),
        RemoteFile(CODEC_DIR, "codec_browser_onnx_meta.json", 17_036),
        RemoteFile(CODEC_DIR, "moss_audio_tokenizer_decode_step.onnx", 351_400),
        RemoteFile(CODEC_DIR, "moss_audio_tokenizer_decode_shared.data", 44_198_912),
        RemoteFile(CODEC_DIR, "moss_audio_tokenizer_encode.onnx", 815_775, cloneOnly = true),
        RemoteFile(CODEC_DIR, "moss_audio_tokenizer_encode.data", 44_507_136, cloneOnly = true),
    )

    /** Download size in MB of what synthesis needs (clone encoder excluded). */
    fun downloadMb(variant: ModelVariant): Long = files(variant).filter { !it.cloneOnly }.sumOf { maxOf(it.size, 0L) } / 1_000_000

    fun files(variant: ModelVariant) = if (variant == ModelVariant.FP32) FP32_FILES else INT8_FILES

    fun baseDir(context: Context): File = File(context.getExternalFilesDir(null) ?: context.filesDir, "models")

    fun root(context: Context, variant: ModelVariant): File = File(baseDir(context), variant.dirName)

    fun file(context: Context, variant: ModelVariant, f: RemoteFile): File = File(File(root(context, variant), f.subdir), f.name)

    fun isPresent(context: Context, variant: ModelVariant, f: RemoteFile): Boolean {
        val local = file(context, variant, f)
        return local.isFile && (if (f.size > 0) local.length() == f.size else local.length() > 0)
    }

    /** Everything needed for synthesis is there. */
    fun isReady(context: Context, variant: ModelVariant): Boolean =
        files(variant).filter { !it.cloneOnly }.all { isPresent(context, variant, it) }

    fun canClone(context: Context, variant: ModelVariant): Boolean =
        files(variant).filter { it.cloneOnly }.all { isPresent(context, variant, it) }

    fun url(source: DownloadSource, variant: ModelVariant, f: RemoteFile): String =
        if (variant == ModelVariant.INT8) {
            "$INT8_RELEASE_BASE/${f.name}"
        } else {
            val (repo, rev) = if (f.subdir == TTS_DIR) TTS_REPO to TTS_REVISION else CODEC_REPO to CODEC_REVISION
            "${source.hfBase}/$repo/resolve/$rev/${f.name}"
        }

    fun downloadedBytes(context: Context, variant: ModelVariant): Long =
        files(variant).sumOf { file(context, variant, it).takeIf { f -> f.isFile }?.length() ?: 0L }

    fun deleteVariant(context: Context, variant: ModelVariant) {
        root(context, variant).deleteRecursively()
    }
}
