package io.github.capsopasme.mossnano

import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.net.Uri
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.os.Process
import android.util.Log
import io.github.capsopasme.mossnano.engine.CancelSignal
import io.github.capsopasme.mossnano.engine.SynthRequest
import io.github.capsopasme.mossnano.engine.SynthStats
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.File
import java.util.concurrent.Executors

/**
 * Foreground service that synthesizes + plays (or saves) text. Holds a partial wake lock
 * while working, so reading continues with the screen off.
 */
class SpeakService : Service() {

    data class UiState(
        val busy: Boolean = false,
        val status: String = "",
        val stats: SynthStats? = null,
        val error: String? = null,
    )

    companion object {
        private const val TAG = "SpeakService"
        const val ACTION_SPEAK = "speak"
        const val ACTION_SAVE = "save"
        const val ACTION_STOP = "stop"
        const val EXTRA_TEXT = "text"
        const val EXTRA_VOICE = "voice"
        const val EXTRA_URI = "uri"

        private val _state = MutableStateFlow(UiState())
        val state: StateFlow<UiState> = _state

        fun speak(context: Context, text: String, voiceId: String? = null) {
            val i = Intent(context, SpeakService::class.java).setAction(ACTION_SPEAK)
                .putExtra(EXTRA_TEXT, text).putExtra(EXTRA_VOICE, voiceId)
            context.startForegroundService(i)
        }

        fun saveWav(context: Context, text: String, voiceId: String?, uri: Uri) {
            val i = Intent(context, SpeakService::class.java).setAction(ACTION_SAVE)
                .putExtra(EXTRA_TEXT, text).putExtra(EXTRA_VOICE, voiceId).putExtra(EXTRA_URI, uri)
            context.startForegroundService(i)
        }

        fun stop(context: Context) {
            context.startService(Intent(context, SpeakService::class.java).setAction(ACTION_STOP))
        }
    }

    private val executor = Executors.newSingleThreadExecutor { r ->
        Thread({
            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)
            r.run()
        }, "moss-synth")
    }
    @Volatile private var currentCancel: CancelSignal? = null
    @Volatile private var currentSink: AudioTrackSink? = null
    @Volatile private var generation = 0
    private lateinit var wakeLock: PowerManager.WakeLock
    private lateinit var audioManager: AudioManager
    private var focusRequest: AudioFocusRequest? = null

    override fun onCreate() {
        super.onCreate()
        wakeLock = getSystemService(PowerManager::class.java)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "mossnano:speak").apply { setReferenceCounted(false) }
        audioManager = getSystemService(AudioManager::class.java)
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_SPEAK, ACTION_SAVE -> {
                val text = intent.getStringExtra(EXTRA_TEXT).orEmpty()
                goForeground(text)
                cancelCurrent()
                val myGen = ++generation
                val voiceId = intent.getStringExtra(EXTRA_VOICE)
                @Suppress("DEPRECATION")
                val uri: Uri? = if (intent.action == ACTION_SAVE) intent.getParcelableExtra(EXTRA_URI) else null
                executor.execute { runJob(myGen, text, voiceId, uri) }
            }
            ACTION_STOP -> {
                cancelCurrent()
                executor.execute { if (currentCancel == null) shutdown() }
            }
            else -> if (currentCancel == null) shutdown()
        }
        return START_NOT_STICKY
    }

    private fun goForeground(text: String) {
        val n = Notifications.speaking(this, text)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(Notifications.ID_SPEAK, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK)
        } else {
            startForeground(Notifications.ID_SPEAK, n)
        }
    }

    private fun cancelCurrent() {
        currentCancel?.cancel()
        currentSink?.abort()
    }

    private fun runJob(gen: Int, text: String, voiceId: String?, saveTo: Uri?) {
        if (gen != generation) return // superseded before it started
        val settings = AppSettings(this)
        val cancel = CancelSignal()
        currentCancel = cancel
        wakeLock.acquire(60 * 60 * 1000L)
        try {
            _state.value = UiState(busy = true, status = if (EngineManager.state.value is EngineManager.State.Ready) "合成中…" else "加载模型…")
            val voice = EngineManager.resolveVoice(this, voiceId ?: settings.voiceId)
                ?: throw IllegalStateException("没有可用音色（模型未下载？）")
            val request = SynthRequest(
                text = text,
                voice = voice,
                seed = if (settings.fixedSeed) 1234L else null,
                normalizeText = settings.normalize,
            )
            val stats = if (saveTo == null) {
                requestFocus(cancel)
                val sink = AudioTrackSink(cancel, settings.speed)
                currentSink = sink
                EngineManager.withEngine { engine ->
                    _state.value = UiState(busy = true, status = "朗读中…")
                    engine.synthesize(request, sink, cancel)
                }
            } else {
                val tmp = File(cacheDir, "export.wav")
                val st = EngineManager.withEngine { engine ->
                    _state.value = UiState(busy = true, status = "导出中…")
                    engine.synthesize(request, WavFileSink(tmp), cancel)
                }
                if (!cancel.isCancelled) {
                    contentResolver.openOutputStream(saveTo)?.use { out -> tmp.inputStream().use { it.copyTo(out) } }
                }
                tmp.delete()
                st
            }
            Log.i(TAG, stats.summary())
            _state.value = UiState(busy = false, status = if (cancel.isCancelled) "已停止" else if (saveTo != null) "已导出" else "完成", stats = stats)
        } catch (t: Throwable) {
            Log.e(TAG, "synthesis failed", t)
            _state.value = UiState(busy = false, status = "出错", error = t.message ?: t.toString())
        } finally {
            currentSink = null
            if (currentCancel === cancel) currentCancel = null
            abandonFocus()
            if (wakeLock.isHeld) wakeLock.release()
            if (gen == generation) shutdown()
        }
    }

    private fun requestFocus(cancel: CancelSignal) {
        val req = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setOnAudioFocusChangeListener { change ->
                if (change == AudioManager.AUDIOFOCUS_LOSS) {
                    cancel.cancel()
                    currentSink?.abort()
                }
            }
            .build()
        focusRequest = req
        audioManager.requestAudioFocus(req)
    }

    private fun abandonFocus() {
        focusRequest?.let { audioManager.abandonAudioFocusRequest(it) }
        focusRequest = null
    }

    private fun shutdown() {
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        cancelCurrent()
        executor.shutdown()
        super.onDestroy()
    }
}
