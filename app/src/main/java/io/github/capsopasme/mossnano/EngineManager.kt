package io.github.capsopasme.mossnano

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import io.github.capsopasme.mossnano.engine.EngineOptions
import io.github.capsopasme.mossnano.engine.MossTtsEngine
import io.github.capsopasme.mossnano.engine.OrtEnv
import io.github.capsopasme.mossnano.engine.SpmTokenizer
import io.github.capsopasme.mossnano.engine.VoicePrompt
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.util.concurrent.atomic.AtomicInteger
import kotlin.concurrent.thread

/**
 * Process-wide owner of the (single, ~1 GB) engine instance shared by the in-app player
 * and the system TTS service. Loads lazily, unloads after an idle timeout.
 */
object EngineManager {
    private const val TAG = "MossEngine"

    sealed interface State {
        data object NotLoaded : State
        data object Loading : State
        data class Ready(val variant: ModelVariant, val loadMs: Long, val warmupMs: Long) : State
        data class Error(val message: String) : State
    }

    private val _state = MutableStateFlow<State>(State.NotLoaded)
    val state: StateFlow<State> = _state

    private val lock = Object()
    private var engine: MossTtsEngine? = null
    private var engineVariant: ModelVariant? = null
    private val users = AtomicInteger(0)
    private val handler = Handler(Looper.getMainLooper())
    private lateinit var app: Context

    fun init(context: Context) {
        app = context.applicationContext
    }

    private val unloadRunnable = Runnable {
        if (users.get() == 0) {
            Log.i(TAG, "idle timeout -> unloading engine")
            release()
        }
    }

    /** Thread count of the process-wide ORT pool differs from settings -> restart needed. */
    fun threadsNeedRestart(settings: AppSettings): Boolean =
        OrtEnv.hasGlobalPool && (OrtEnv.globalThreads != settings.lmThreads || OrtEnv.globalSpinning != settings.spinning)

    /** Blocking. Loads (or reuses) the engine for the selected variant and runs [block] with it. */
    fun <T> withEngine(block: (MossTtsEngine) -> T): T {
        users.incrementAndGet()
        handler.removeCallbacks(unloadRunnable)
        try {
            return block(acquire())
        } finally {
            if (users.decrementAndGet() == 0) scheduleUnload()
        }
    }

    fun preload() {
        thread(name = "moss-preload") {
            runCatching { withEngine { } }
        }
    }

    private fun scheduleUnload() {
        val minutes = AppSettings(app).idleUnloadMinutes
        handler.removeCallbacks(unloadRunnable)
        if (minutes > 0) handler.postDelayed(unloadRunnable, minutes * 60_000L)
    }

    private fun acquire(): MossTtsEngine = synchronized(lock) { acquireLocked() }

    private fun acquireLocked(): MossTtsEngine {
        val settings = AppSettings(app)
        val variant = settings.variant
        engine?.let { if (engineVariant == variant) return it }
        engine?.close()
        engine = null
        if (!ModelStore.isReady(app, variant)) {
            _state.value = State.Error("模型文件不完整，请先下载 ${variant.label}")
            throw IllegalStateException("model not downloaded")
        }
        _state.value = State.Loading
        return try {
            val e = MossTtsEngine.load(
                EngineOptions(
                    modelRoot = ModelStore.root(app, variant),
                    lmThreads = settings.lmThreads,
                    codecThreads = settings.codecThreads,
                    allowSpinning = settings.spinning,
                ),
                tokenizerFactory = { SpmTokenizer(it) },
                log = { Log.i(TAG, it) },
            )
            val t0 = System.nanoTime()
            e.warmup()
            val warmMs = (System.nanoTime() - t0) / 1_000_000
            engine = e
            engineVariant = variant
            _state.value = State.Ready(variant, e.loadMs, warmMs)
            Log.i(TAG, "engine ready: load=${e.loadMs}ms warmup=${warmMs}ms")
            e
        } catch (t: Throwable) {
            Log.e(TAG, "engine load failed", t)
            _state.value = State.Error(t.message ?: t.toString())
            throw t
        }
    }

    fun release() {
        synchronized(lock) {
            engine?.close()
            engine = null
            engineVariant = null
            _state.value = State.NotLoaded
        }
    }

    /** Built-in voices of the selected variant (manifest only) + cloned voices. */
    fun voices(context: Context): List<VoicePrompt> {
        val variant = AppSettings(context).variant
        val builtin = synchronized(lock) { engine?.takeIf { engineVariant == variant }?.builtinVoices }
            ?: runCatching { MossTtsEngine.readBuiltinVoices(ModelStore.root(context, variant)) }.getOrDefault(emptyList())
        return builtin + VoiceStore.list(context)
    }

    fun resolveVoice(context: Context, id: String?): VoicePrompt? {
        val all = voices(context)
        return all.firstOrNull { it.id == id } ?: all.firstOrNull()
    }
}
